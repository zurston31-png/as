"""Command line entry point.

    python -m tradebot doctor                      # what's installed / configured
    python -m tradebot serve                       # dashboard + live loop
    python -m tradebot backtest    --csv data.csv  # one period
    python -m tradebot walkforward --csv data.csv  # folds + a sealed holdout
    python -m tradebot ablate      --csv data.csv  # is the AI layer worth it?
    python -m tradebot rules        --csv data.csv # why setups do or don't fire
    python -m tradebot ablate-rules --csv data.csv # which rule removes the setups
    python -m tradebot audit verify                # check the log's hash chain
    python -m tradebot gen-data --out data/demo.csv --bars 3000
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import webbrowser
from pathlib import Path

from .ai.client import availability as ai_availability
from .backtest.confirmers import NullConfirmer
from .config import Config
from .data.replay import read_csv
from .vision.capture import ScreenCapture


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ------------------------------------------------------------------ serve

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app.server import create_app

    config = Config.load(args.config)
    if args.symbol:
        config.market.symbol = args.symbol
    if args.timeframe:
        config.market.timeframe = args.timeframe
    if args.feed:
        config.feed.source = args.feed
    if args.csv:
        config.feed.source = "replay"
        config.feed.csv_path = args.csv
    if args.strategy:
        config.strategy = {"preset": args.strategy}
    if args.no_ai:
        config.ai.enabled = False
    if args.alerts_only:
        config.execution.mode = "alerts_only"
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host

    url = f"http://{config.server.host}:{config.server.port}"
    print(f"\n  Dashboard  {url}")
    print(f"  Webhook    POST {url}/webhook/tradingview")
    print(f"  Feed       {config.feed.source}   Execution  {config.execution.mode}")
    ai_line = (f"on - {config.ai.model} (on failure: {config.ai.on_failure})"
               if config.ai.enabled else "off")
    print(f"  AI         {ai_line}")
    print(f"  Audit      {config.risk.audit_path}\n")
    if args.open or config.server.open_browser:
        webbrowser.open(url)

    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    return 0


# --------------------------------------------------------------- backtest

def _load_config(args) -> "Config":
    config = Config.load(args.config)
    if getattr(args, "symbol", None):
        config.market.symbol = args.symbol
    if getattr(args, "strategy", None):
        config.strategy = {"preset": args.strategy}
    if getattr(args, "equity", None):
        config.risk.starting_equity = args.equity
    return config


def _candles(args):
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"error: no such file: {path}")
    candles = read_csv(path)
    if getattr(args, "limit", None):
        candles = candles[-args.limit:]
    return candles


def _confirmer(config, args):
    """The AI layer for a backtest arm, cached so reruns are free."""
    if not getattr(args, "ai", False):
        return NullConfirmer()
    from .ai.confirm import ConfirmationLayer
    from .backtest.confirmers import CachedConfirmer
    return CachedConfirmer(ConfirmationLayer(config), args.ai_cache)


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest.runner import Backtester

    config = _load_config(args)
    candles = _candles(args)
    result = asyncio.run(Backtester(config, _confirmer(config, args)).run_candles(candles))

    print(f"\n  {config.market.symbol} {config.market.timeframe} · "
          f"{result.notes.get('strategy')} · {result.bars} bars")
    print(f"  {result.period[0]} -> {result.period[1]}")
    print(f"  {len(result.signals)} signals, {result.taken} taken, "
          f"{result.suppressed} duplicate(s) suppressed\n")
    print(result.metrics.render())
    if result.blocked:
        print("\n  Signals blocked by:")
        for reason, count in sorted(result.blocked.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4}  {reason}")
    print()

    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"  wrote {args.json}\n")
    return 0


# ------------------------------------------------------------ walkforward

def cmd_walkforward(args: argparse.Namespace) -> int:
    from .audit import fingerprint
    from .backtest.runner import Backtester
    from .backtest.walkforward import (
        HoldoutLedger, WalkForwardReport, combine, develop, split_folds,
    )

    config = _load_config(args)
    candles = _candles(args)
    development, holdout = develop(candles, args.holdout_fraction)

    try:
        folds = split_folds(development, args.folds, args.test_fraction, not args.rolling)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    report = WalkForwardReport()
    for fold in folds:
        result = asyncio.run(
            Backtester(config, _confirmer(config, args), fold.label, f"wf_{fold.index}")
            .run_candles(fold.test)
        )
        report.folds.append(result)
    report.combined = combine(report.folds, config.risk.starting_equity)

    print(f"\n  {config.market.symbol} {config.market.timeframe} · "
          f"{len(candles)} bars, {len(holdout)} held back\n")

    if args.holdout:
        ledger = HoldoutLedger.load(args.holdout_ledger)
        ledger.record(fingerprint(config.to_dict()), Path(args.csv).name, len(holdout))
        report.holdout = asyncio.run(
            Backtester(config, _confirmer(config, args), "holdout", "wf_holdout")
            .run_candles(holdout))
        report.holdout_warning = ledger.warning()
    else:
        print(f"  Holdout of {len(holdout)} bars is sealed. "
              f"Pass --holdout to spend a look at it.\n")

    print(report.render())
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"  wrote {args.json}\n")
    return 0


# --------------------------------------------------------------- ablate

def cmd_ablate(args: argparse.Namespace) -> int:
    from .backtest.ablation import run_ablation

    config = _load_config(args)
    candles = _candles(args)
    if args.ai:
        print("\n  Running the AI arms against the live API - this costs money.")
    report = asyncio.run(run_ablation(
        config, candles, use_live_ai=args.ai, cache_path=args.ai_cache,
        control_runs=args.control_runs, seed=args.seed, veto_rate=args.veto_rate,
    ))
    print(f"\n  {config.market.symbol} {config.market.timeframe} · {len(candles)} bars\n")
    print(report.render())
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"  wrote {args.json}\n")
    return 0


# --------------------------------------------------------- ablate-rules

def cmd_ablate_rules(args: argparse.Namespace) -> int:
    from .backtest.ruleablation import run_rule_ablation

    config = _load_config(args)
    candles = _candles(args)
    report = asyncio.run(run_rule_ablation(config, candles, args.mode))

    print(f"\n  {config.market.symbol} {config.market.timeframe} · "
          f"{report.strategy} · {report.bars} bars\n")
    print(report.render())
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"  wrote {args.json}\n")
    return 0


# ---------------------------------------------------------------- rules

def cmd_rules(args: argparse.Namespace) -> int:
    from .backtest.rulestats import analyse

    config = _load_config(args)
    report = analyse(config, _candles(args), ignore_session=args.ignore_session)
    print()
    print(report.render())
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"  wrote {args.json}\n")
    return 0


# ---------------------------------------------------------------- audit

def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import AuditLog

    config = Config.load(args.config)
    log_path = args.path or config.risk.audit_path
    audit = AuditLog(log_path)

    if args.action == "verify":
        result = audit.verify()
        mark = "ok" if result.ok else "FAIL"
        print(f"\n  [{mark}] {log_path}: {result.message}\n")
        return 0 if result.ok else 1

    records = list(audit.records())
    if args.action == "tail":
        for record in records[-args.count:]:
            payload = record.get("payload", {})
            print(f"  #{record['sequence']:<6} {record['event']:<16} "
                  f"{payload.get('decision_ts', record.get('logged_at', ''))}  "
                  f"{payload.get('side', '')} "
                  f"{'TAKEN' if payload.get('actionable') else payload.get('blocked_by', '')}")
        return 0

    print(json.dumps(records[-args.count:], indent=2))
    return 0


# ---------------------------------------------------------------- doctor

def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def cmd_doctor(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    ai = ai_availability()
    capture = ScreenCapture.availability()

    def line(label: str, ok: bool, note: str = "") -> None:
        print(f"  [{'ok ' if ok else 'ns '}] {label:<30} {note}")

    print(f"\n  Config: {args.config or 'config/config.yaml'}\n")
    print("  Core")
    for module in ("fastapi", "uvicorn", "yaml"):
        try:
            __import__(module)
            line(module, True)
        except ImportError:
            line(module, False, "pip install fastapi uvicorn pyyaml")
    # Without a websocket implementation uvicorn serves the dashboard but 404s
    # /ws, so the page loads and then never updates. Worth catching here.
    has_ws = any(_importable(m) for m in ("websockets", "wsproto"))
    line("websocket support", has_ws,
         "" if has_ws else "pip install 'uvicorn[standard]' - the dashboard won't live-update without it")

    print("\n  AI layer")
    line("anthropic SDK", bool(ai["sdk_installed"]), "" if ai["sdk_installed"] else "pip install anthropic")
    line("credentials", bool(ai["credentials_present"]),
         "" if ai["credentials_present"] else "export ANTHROPIC_API_KEY=... or run `ant auth login`")
    line("confirmation enabled", config.ai.enabled, config.ai.model)

    print("\n  Vision (optional backup layer)")
    line("mss", bool(capture["mss_installed"]), "" if capture["mss_installed"] else "pip install mss")
    line("pillow", bool(capture["pillow_installed"]), "" if capture["pillow_installed"] else "pip install pillow")
    line("enabled", config.vision.enabled, config.vision.mode)

    print("\n  Trading")
    line("execution mode", config.execution.mode != "live",
         f"{config.execution.mode}" + (" - live broker is a stub" if config.execution.mode == "live" else ""))
    line("feed", True, config.feed.source)
    line("stop required", config.risk.require_stop)
    line("kill switch clear", not config.risk.kill_switch)
    print("\n  Strategy")
    try:
        from .strategy.engine import StrategyEngine
        spec = StrategyEngine(config).spec
        line("config parses", True, f"{spec.name}, {len(spec.active())} active rules")
        for rule in spec.active():
            params = ", ".join(f"{k}={v}" for k, v in sorted(rule.params.items()))
            print(f"         {rule.mode:<9} {rule.rule:<22} {params}")
        print(f"         targets   TP1 {spec.entry.tp1_r:g}R, TP2 {spec.entry.tp2_r:g}R, "
              f"min R:R {spec.entry.min_rr:g}, warmup {spec.min_bars()} bars")
    except Exception as exc:  # noqa: BLE001 - this is the diagnostic
        line("config parses", False, str(exc))

    print("\n  Audit log")
    from .audit import AuditLog
    audit = AuditLog(config.risk.audit_path)
    result = audit.verify()
    line("hash chain", result.ok, result.message)

    print(f"\n  Risk: {config.risk.risk_per_trade_pct}% per trade, "
          f"max {config.risk.max_trades_per_day} trades/day, "
          f"{config.risk.max_daily_loss_pct}% daily loss limit")
    print(f"  Costs: {config.execution.slippage_ticks:g} tick entry slippage, "
          f"{config.execution.stop_slippage_ticks:g} on stops, "
          f"{config.execution.commission_per_unit:g}/contract per side\n")
    return 0


# -------------------------------------------------------------- gen-data

def cmd_gen_data(args: argparse.Namespace) -> int:
    from .data.synthetic import SyntheticSource

    config = Config.load(args.config)
    source = SyntheticSource(config.market.symbol, config.market.timeframe, seed=args.seed)
    candles = source.history(args.bars)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c.ts.isoformat(), c.open, c.high, c.low, c.close, c.volume])
    print(f"wrote {len(candles)} synthetic bars to {out}")
    print("This is random data for exercising the pipeline - it is not a market.")
    return 0


# ------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tradebot", description="Chart-reading trade signal bot")
    parser.add_argument("-c", "--config", default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the dashboard and the live loop")
    serve.add_argument("--symbol")
    serve.add_argument("--timeframe")
    serve.add_argument("--feed", choices=["synthetic", "replay", "webhook", "poll"])
    serve.add_argument("--strategy", help="use this preset instead of the config's")
    serve.add_argument("--csv", help="replay this file instead of a live feed")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--no-ai", action="store_true", help="rules only, no model calls")
    serve.add_argument("--alerts-only", action="store_true", help="never simulate a fill")
    serve.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    serve.set_defaults(func=cmd_serve)

    def add_data_args(parser_) -> None:
        parser_.add_argument("--csv", required=True)
        parser_.add_argument("--symbol")
        parser_.add_argument("--strategy", help="use this preset instead of the config's")
        parser_.add_argument("--equity", type=float)
        parser_.add_argument("--limit", type=int, help="only the last N bars")
        parser_.add_argument("--ai", action="store_true",
                             help="run the real AI confirmation layer (slow, costs money)")
        parser_.add_argument("--ai-cache", default="data/ai_verdicts.json",
                             help="where AI verdicts are memoised between runs")
        parser_.add_argument("--json", help="write the full result to this path")

    back = sub.add_parser("backtest", help="replay a CSV through the full pipeline")
    add_data_args(back)
    back.set_defaults(func=cmd_backtest)

    wf = sub.add_parser("walkforward", help="out-of-sample folds plus a sealed holdout")
    add_data_args(wf)
    wf.add_argument("--folds", type=int, default=4)
    wf.add_argument("--test-fraction", type=float, default=0.4,
                    help="share of the development data used for testing, across all folds")
    wf.add_argument("--holdout-fraction", type=float, default=0.2,
                    help="tail of the data sealed away from development")
    wf.add_argument("--holdout", action="store_true",
                    help="spend a look at the sealed holdout (recorded in the ledger)")
    wf.add_argument("--holdout-ledger", default="data/holdout_ledger.json")
    wf.add_argument("--rolling", action="store_true",
                    help="rolling training window instead of an expanding one")
    wf.set_defaults(func=cmd_walkforward)

    ab = sub.add_parser("ablate", help="measure whether the AI veto layer earns its place")
    add_data_args(ab)
    ab.add_argument("--control-runs", type=int, default=5,
                    help="how many randomised control arms to average over")
    ab.add_argument("--veto-rate", type=float, default=None,
                    help="force the control arms' veto rate instead of taking it "
                         "from the model - e.g. 0.3 to ask what vetoing 30%% at "
                         "random would do")
    ab.add_argument("--seed", type=int, default=1234)
    ab.set_defaults(func=cmd_ablate)

    rules = sub.add_parser(
        "rules", help="per-rule pass rates and the funnel - why setups do or don't fire")
    add_data_args(rules)
    rules.add_argument("--ignore-session", action="store_true",
                       help="evaluate every bar, not just session hours")
    rules.set_defaults(func=cmd_rules)

    ra = sub.add_parser(
        "ablate-rules",
        help="which rule removes the setups, and is removing it an improvement?")
    add_data_args(ra)
    ra.add_argument("--mode", choices=["incremental", "leave-one-out", "both"],
                    default="both")
    ra.set_defaults(func=cmd_ablate_rules)

    audit = sub.add_parser("audit", help="inspect the decision log")
    audit.add_argument("action", choices=["verify", "tail", "dump"], nargs="?", default="verify")
    audit.add_argument("--path", help="audit log path (defaults to the configured one)")
    audit.add_argument("--count", type=int, default=20)
    audit.set_defaults(func=cmd_audit)

    doctor = sub.add_parser("doctor", help="check dependencies and configuration")
    doctor.set_defaults(func=cmd_doctor)

    gen = sub.add_parser("gen-data", help="write a synthetic OHLCV CSV for testing")
    gen.add_argument("--out", default="data/demo.csv")
    gen.add_argument("--bars", type=int, default=3000)
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=cmd_gen_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nstopped")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
