"""Command-line interface.

Commands for phases that are not yet built exit with status 2 and name the
phase that will implement them. They do not print placeholder results: a
command that prints a fabricated equity curve is worse than one that
refuses to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flow_model import __version__
from flow_model.config.loader import ConfigError, load_config, save_config
from flow_model.core.enums import SplitPhase
from flow_model.utils.logging import new_run_id, setup_logging
from flow_model.utils.serialization import dumps

PHASE_STATUS: dict[str, tuple[int, str]] = {
    "config": (1, "done"),
    "splits": (1, "done"),
    "experiments": (1, "done"),
    "data": (2, "done"),
    "features": (3, "not built"),
    "orderflow": (4, "not built"),
    "optionsflow": (5, "not built"),
    "score": (6, "not built"),
    "backtest": (7, "not built"),
    "walkforward": (8, "not built"),
    "montecarlo": (9, "not built"),
    "dashboard": (10, "not built"),
    "paper": (11, "not built"),
}


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", action="append", default=[], metavar="PATH",
        help="YAML config file; repeatable, later files win.",
    )
    parser.add_argument(
        "-s", "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
        help="Override a dotted config key, e.g. -s risk.risk_per_trade_pct=0.0025",
    )
    parser.add_argument(
        "--no-defaults", action="store_true",
        help="Do not start from the packaged defaults.yaml.",
    )


def _load(args: argparse.Namespace):
    return load_config(
        paths=args.config, overrides=args.overrides, include_defaults=not args.no_defaults
    )


def _not_built(name: str) -> int:
    phase, _ = PHASE_STATUS[name]
    print(
        f"'{name}' is not built yet -- it is Phase {phase}.\n"
        f"Run 'flow-model phases' to see build status. No placeholder results "
        f"are produced, by design.",
        file=sys.stderr,
    )
    return 2


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.action == "validate":
        print(f"config valid. config_hash={config.config_hash}")
        return 0
    if args.action == "hash":
        print(config.config_hash)
        return 0
    if args.action == "save":
        path = save_config(config, args.output)
        print(f"wrote {path} (config_hash={config.config_hash})")
        return 0
    # show
    if args.section:
        node = config
        for part in args.section.split("."):
            node = getattr(node, part)
        print(dumps(node, indent=2))
        return 0
    payload = config.to_init_dict(mode="json")
    payload["_config_hash"] = config.config_hash
    print(dumps(payload, indent=2))
    return 0


def cmd_instruments(args: argparse.Namespace) -> int:
    config = _load(args)
    header = f"{'symbol':<8}{'type':<8}{'tick':>8}{'tick$':>10}{'point$':>10}{'comm/side':>11}{'spread':>8}  tradable"
    print(header)
    print("-" * len(header))
    for symbol in sorted(config.instruments):
        s = config.instruments[symbol]
        proxy = f" (via {s.execution_proxy})" if s.execution_proxy else ""
        print(
            f"{s.symbol:<8}{s.instrument_type.value:<8}{s.tick_size:>8.4g}"
            f"{s.tick_value:>10.4g}{s.point_value:>10.4g}"
            f"{s.commission_per_side:>11.4g}{s.typical_spread_ticks:>8.3g}  "
            f"{'yes' if s.tradable else 'NO (reference only)'}{proxy}"
        )
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    from flow_model.validation.splits import guard_from_config

    config = _load(args)
    guard = guard_from_config(config)
    wf = config.walk_forward
    print(f"backtest period      : {config.backtest.start} -> {config.backtest.end}")
    print(f"sealed OOS window    : {config.seal.sealed_start} -> {config.seal.sealed_end}"
          f"  (enabled={config.seal.enabled})")
    print(f"seal opens used      : {guard.opens_used()}/{config.seal.max_opens}")
    print(f"seal audit log       : {config.seal.audit_path}")
    print(f"walk-forward windows : train={wf.train_months}m validation={wf.validation_months}m "
          f"test={wf.test_months}m step={wf.step_months}m "
          f"{'anchored' if wf.anchored else 'rolling'}")
    print(f"purge / embargo      : {wf.purge_days}d / {wf.embargo_days}d")
    allowed = guard.clip_to_allowed(config.backtest.start, config.backtest.end)
    print(f"research-usable range: {allowed}")
    for entry in guard.audit.entries():
        print(f"  seal open #{entry.sequence}: {entry.ts.isoformat()} "
              f"{entry.experiment_id} by {entry.approved_by}: {entry.reason}")
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    from flow_model.validation.experiment_log import ExperimentLog

    config = _load(args)
    with ExperimentLog(config.paths.experiment_db_path) as log:
        if args.action == "summary":
            print(dumps(log.overfitting_summary(), indent=2))
            return 0
        phase = SplitPhase(args.phase) if args.phase else None
        records = log.query(phase=phase, limit=args.limit)
        if not records:
            print("no experiments logged yet.")
            return 0
        print(f"{'id':<22}{'date':<12}{'phase':<12}{'config':<18}name")
        print("-" * 90)
        for r in records:
            flag = " [DIRTY]" if r.git_dirty else ""
            print(f"{r.experiment_id:<22}{r.created_at.date().isoformat():<12}"
                  f"{r.phase.value:<12}{r.config_hash:<18}{r.name}{flag}")
        return 0



def cmd_data(args: argparse.Namespace) -> int:
    """Load a dataset and report which Flow Score components are computable.

    Run before any backtest. With bar-only data this states plainly that 45
    of the 100 nominal points have no feed, which is the difference between
    a Flow Score and a number that merely resembles one.
    """
    from flow_model.data import (
        BarCleaner,
        DataStore,
        Feed,
        QualityGrader,
        SyntheticAdapter,
        SyntheticConfig,
        TradingCalendar,
    )
    from flow_model.validation.splits import guard_from_config

    config = _load(args)
    feeds = None
    if args.feeds:
        try:
            feeds = frozenset(Feed(name) for name in args.feeds)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.provider != "synthetic":
        print(
            f"only the synthetic provider is wired into this command so far "
            f"(asked for {args.provider!r}). Real adapters exist "
            f"(CsvAdapter, ParquetAdapter) but need a data directory; point "
            f"data.root_path at one and load them from a script.",
            file=sys.stderr,
        )
        return 2

    calendar = TradingCalendar()
    adapter = SyntheticAdapter(
        config=SyntheticConfig(
            include_ticks=Feed.TICK_AGGREGATE in (feeds or {Feed.TICK_AGGREGATE}),
            include_quotes=Feed.QUOTES in (feeds or {Feed.QUOTES}),
            include_options=Feed.OPTIONS_SNAPSHOT in (feeds or {Feed.OPTIONS_SNAPSHOT}),
        ),
        seed=config.seed,
        spec_by_symbol=dict(config.instruments),
        feeds=feeds,
        interval_seconds=config.data.primary_interval_seconds,
        calendar=calendar,
    )

    store = DataStore(
        latency_seconds=config.data.data_latency_seconds,
        guard=guard_from_config(config),
    )
    grader = QualityGrader(config.data, config.flow_score)
    cleaner = BarCleaner(config.data)

    print("!! SYNTHETIC DATA. Nothing measured on it says anything about real edge. !!\n")

    for symbol in config.backtest.symbols:
        data = store.load(
            adapter,
            symbol,
            config.backtest.start,
            config.backtest.end,
            primary_interval=config.data.primary_interval_seconds,
            clip_to_allowed=True,
        )
        _, clean_report = cleaner.clean(
            data.primary_bars, spec=config.spec(symbol), calendar=calendar
        )
        print("=" * 66)
        print("\n".join(grader.availability(data).summary_lines()))
        print(
            f"\n  rows: {clean_report.rows_in} in -> {clean_report.rows_out} out "
            f"(retention {clean_report.retention:.4f}); "
            f"gaps={clean_report.gaps_detected} "
            f"outliers_quarantined={clean_report.outliers_quarantined} "
            f"zero_volume={clean_report.zero_volume_flagged}"
        )
        print(f"  range: {data.primary_bars.first_ts} -> {data.primary_bars.last_ts}")
        print(f"  data_hash: {data.fingerprint.data_hash if data.fingerprint else '-'}")
        print()

    print(f"combined data_hash: {store.combined_data_hash()}")
    return 0


def cmd_phases(args: argparse.Namespace) -> int:
    titles = {
        1: "architecture, config, core contracts, logging, experiment log, OOS seal",
        2: "historical data interface, cleaning, quality grading, PIT MarketView",
        3: "structure, volatility, liquidity, momentum features",
        4: "order-flow features",
        5: "options-flow features",
        6: "Flow Score, entry gates, setups",
        7: "backtester and execution model",
        8: "walk-forward validation, robustness, consistency",
        9: "Monte Carlo engine",
        10: "dashboard",
        11: "paper-trading signal mode",
    }
    done = {phase for phase, status in PHASE_STATUS.values() if status == "done"}
    for phase in sorted(titles):
        mark = "x" if phase in done else " "
        print(f"[{mark}] Phase {phase:>2}: {titles[phase]}")
    print("\nNo backtest has been run. No performance figures exist yet.")
    return 0


def cmd_stub(args: argparse.Namespace) -> int:
    return _not_built(args.command)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flow-model",
        description="Flow Model -- rules-based trading research and backtesting system.",
    )
    parser.add_argument("--version", action="version", version=f"flow_model {__version__}")
    parser.add_argument("--log-level", default=None, help="Override the configured log level.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="Show, validate, hash or save the resolved configuration.")
    p.add_argument("action", nargs="?", default="show", choices=["show", "validate", "hash", "save"])
    p.add_argument("--section", default=None, help="Show only this dotted section, e.g. risk")
    p.add_argument("-o", "--output", default="reports/resolved_config.yaml")
    _add_config_args(p)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("instruments", help="List configured instruments and their economics.")
    _add_config_args(p)
    p.set_defaults(func=cmd_instruments)

    p = sub.add_parser("splits", help="Show validation splits and sealed-OOS status.")
    _add_config_args(p)
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("experiments", help="List logged experiments or the overfitting summary.")
    p.add_argument("action", nargs="?", default="list", choices=["list", "summary"])
    p.add_argument("--phase", default=None, choices=[ph.value for ph in SplitPhase])
    p.add_argument("--limit", type=int, default=50)
    _add_config_args(p)
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("phases", help="Show which build phases are complete.")
    p.set_defaults(func=cmd_phases)

    p = sub.add_parser("data", help="Load a dataset and report feed availability.")
    p.add_argument("--provider", default="synthetic",
                   help="Data provider. Only 'synthetic' is wired into the CLI so far.")
    p.add_argument("--feeds", nargs="*", default=None,
                   help="Restrict feeds, e.g. --feeds bars. Omit for everything the provider has.")
    _add_config_args(p)
    p.set_defaults(func=cmd_data)

    for name in ("features", "backtest", "walkforward", "montecarlo", "dashboard", "paper"):
        phase, _ = PHASE_STATUS[name]
        p = sub.add_parser(name, help=f"(Phase {phase} -- not built yet)")
        _add_config_args(p)
        p.set_defaults(func=cmd_stub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(
        level=args.log_level or "WARNING",
        run_id=new_run_id("cli"),
        console=True,
    )

    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 1
    except (KeyError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
