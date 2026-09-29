"""Command line entry point.

    python -m tradebot doctor                 # what's installed / configured
    python -m tradebot serve                  # dashboard + live loop
    python -m tradebot backtest --csv data.csv
    python -m tradebot gen-data --out data/demo.csv --bars 3000
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import webbrowser
from pathlib import Path

from .ai.client import availability as ai_availability
from .config import Config
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
    print(f"  AI         {'on - ' + config.ai.model if config.ai.enabled else 'off'}\n")
    if args.open or config.server.open_browser:
        webbrowser.open(url)

    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    return 0


# --------------------------------------------------------------- backtest

def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest.runner import Backtester

    config = Config.load(args.config)
    if args.symbol:
        config.market.symbol = args.symbol
    if args.strategy:
        config.strategy.name = args.strategy
    if args.equity:
        config.risk.starting_equity = args.equity

    path = Path(args.csv)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    result = asyncio.run(Backtester(config, use_ai=args.ai).run(path, args.limit))

    print(f"\n  {config.market.symbol} {config.market.timeframe} · "
          f"{config.strategy.name} · {result.bars} bars")
    print(f"  {len(result.signals)} setups found, "
          f"{len([s for s in result.signals if s.actionable])} taken\n")
    print(result.metrics.render())
    if result.blocked:
        print("\n  Setups blocked by:")
        for reason, count in sorted(result.blocked.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4}  {reason}")
    print()

    if args.json:
        Path(args.json).write_text(__import__("json").dumps(result.to_dict(), indent=2, default=str))
        print(f"  wrote {args.json}\n")
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
    print(f"\n  Risk: {config.risk.risk_per_trade_pct}% per trade, "
          f"max {config.risk.max_trades_per_day} trades/day, "
          f"{config.risk.max_daily_loss_pct}% daily loss limit\n")
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
    serve.add_argument("--csv", help="replay this file instead of a live feed")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--no-ai", action="store_true", help="rules only, no model calls")
    serve.add_argument("--alerts-only", action="store_true", help="never simulate a fill")
    serve.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    serve.set_defaults(func=cmd_serve)

    back = sub.add_parser("backtest", help="replay a CSV through the full pipeline")
    back.add_argument("--csv", required=True)
    back.add_argument("--symbol")
    back.add_argument("--strategy")
    back.add_argument("--equity", type=float)
    back.add_argument("--limit", type=int, help="only the last N bars")
    back.add_argument("--ai", action="store_true", help="also run the AI confirmation layer (slow, costs money)")
    back.add_argument("--json", help="write the full result to this path")
    back.set_defaults(func=cmd_backtest)

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
