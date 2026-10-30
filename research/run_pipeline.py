"""Run the pipeline end to end and WRITE artifacts under reports/.

Unlike the other scripts in this directory, which only read and print, this one
produces files: a SQLite run database, a rendered dashboard, and RESULTS.md.
It is otherwise bound by the same rules -- it moves no threshold, changes no
config default, bypasses no gate, and never touches the sealed window.

WHAT TO EXPECT. This system takes zero trades at the committed configuration
(PROVENANCE.md 4a: the population Flow Score maximum is 69.6335 against a 70.0
threshold, over 11239 scored bars). So nearly every stage below reports
"undefined, n=0". That is the honest result, not a failure of this script, and
the point of running it is to confirm the undefined path works end to end --
because it is the only path this configuration will ever take.

EVERY FIGURE IS SYNTHETIC. No real market bar has ever entered this system.

    PYTHONPATH=. python3 research/run_pipeline.py 2016 1 4 2016 6 30
"""
from __future__ import annotations

import sys
import traceback
from datetime import date
from pathlib import Path

from flow_model.backtest.loop import BacktestLoop, StoreMarketSource
from flow_model.config.loader import default_config
from flow_model.core.enums import SplitPhase
from flow_model.dashboard.build import assert_origin_is_stated, collect, render
from flow_model.data import DataStore, SyntheticAdapter, SyntheticConfig, TradingCalendar
from flow_model.database.store import Database
from flow_model.monte_carlo.engine import run_monte_carlo

REPORTS = Path("reports")
SEALED = (date(2023, 1, 1), date(2025, 1, 1))


def _refuse_sealed(start: date, end: date) -> None:
    """The seal has a one-open budget and it is unspent. Do not spend it."""
    if start < SEALED[1] and end > SEALED[0]:
        raise SystemExit(
            f"refusing: {start}..{end} overlaps the SEALED window "
            f"{SEALED[0]}..{SEALED[1]}. The research window is 2015-01-01 to "
            "2023-01-01 and the seal's one-open budget is unspent."
        )


def stage(name: str):
    """Run a stage, and on a crash record the traceback and carry on.

    One stage failing must not cost the whole run -- the point is to find out
    what every stage does, and a stage that raises is itself a finding.
    """
    def wrap(fn):
        print(f"\n--- {name} ---", flush=True)
        try:
            out = fn()
            return ("ok", out)
        except Exception:
            tb = traceback.format_exc()
            print(f"STAGE RAISED:\n{tb}", flush=True)
            return ("raised", tb)
    return wrap


def make_runner(calendar, base_config):
    """A `Runner`: (config, split) -> RunResult, over real synthetic bars.

    A FRESH BacktestLoop per call, because a second run() on one instance now
    raises -- its RiskLimitManager accumulates.
    """
    def runner(cfg, split):
        adapter = SyntheticAdapter(
            config=SyntheticConfig(), seed=cfg.seed,
            spec_by_symbol=dict(cfg.instruments),
            interval_seconds=cfg.data.primary_interval_seconds, calendar=calendar,
        )
        st = DataStore(latency_seconds=cfg.data.data_latency_seconds)
        st.load(adapter, "NQ", split.range.start, split.range.end,
                primary_interval=cfg.data.primary_interval_seconds)
        loop = BacktestLoop(
            cfg, "NQ", calendar=calendar, split_label=split.phase,
            split_name=split.name, git_commit="", git_dirty=False,
        )
        return loop.run(StoreMarketSource.from_config(st, "NQ", cfg))
    return runner


def main() -> int:
    a = [int(x) for x in sys.argv[1:7]]
    start, end = date(a[0], a[1], a[2]), date(a[3], a[4], a[5])
    _refuse_sealed(start, end)
    REPORTS.mkdir(exist_ok=True)

    config = default_config()
    calendar = TradingCalendar()
    adapter = SyntheticAdapter(
        config=SyntheticConfig(), seed=config.seed,
        spec_by_symbol=dict(config.instruments),
        interval_seconds=config.data.primary_interval_seconds, calendar=calendar,
    )
    store = DataStore(latency_seconds=config.data.data_latency_seconds)
    store.load(adapter, "NQ", start, end,
               primary_interval=config.data.primary_interval_seconds)

    results: dict[str, object] = {}

    print(f"window {start} -> {end}  (research window; seal untouched)")

    # 1. the baseline backtest
    def _backtest():
        loop = BacktestLoop(
            config, "NQ", calendar=calendar,
            split_label=SplitPhase.TRAIN, split_name="pipeline-baseline",
            git_commit="", git_dirty=False,
        )
        run = loop.run(StoreMarketSource.from_config(store, "NQ", config))
        print(f"bars evaluated {run.bars_evaluated}  trades {run.trade_count}  "
              f"orders {len(run.orders)}  equity rows {len(run.equity)}")
        print(f"is_empty {run.is_empty}  open_at_end {run.open_at_end}")
        print(f"terminal gate {run.rejection.get('terminal_gate')}")
        return run
    results["backtest"] = stage("1. BACKTEST")(_backtest)
    status, run = results["backtest"]
    if status != "ok":
        print("\nbacktest failed; nothing downstream can run")
        return 1

    # 2. persist it
    def _persist():
        path = REPORTS / "runs.sqlite"
        with Database(path) as db:
            run_id = db.write_run(run, replace=True)
            counts = db.counts()
        size = path.stat().st_size
        print(f"wrote run {run_id} to {path} ({size} bytes)")
        print(f"row counts {counts}")
        return {"path": str(path), "run_id": run_id, "counts": counts, "bytes": size}
    results["database"] = stage("2. DATABASE")(_persist)

    # 3. monte carlo, all three methods
    def _mc():
        out = {}
        from flow_model.monte_carlo.engine import MonteCarloMethod
        for method in MonteCarloMethod:
            r = run_monte_carlo(run.trades, split_label=SplitPhase.TRAIN, method=method)
            refused = getattr(r, "refusal", None)
            print(f"{method.value:16} -> "
                  f"{'REFUSED: ' + str(refused) if refused else type(r).__name__}")
            out[method.value] = str(refused) if refused else "distribution"
        return out
    results["monte_carlo"] = stage("3. MONTE CARLO")(_mc)

    # 4. dashboard, rendered from the real artifact above
    def _dash():
        mc = None
        if results["monte_carlo"][0] == "ok":
            from flow_model.monte_carlo.engine import MonteCarloMethod
            mc = run_monte_carlo(run.trades, split_label=SplitPhase.TRAIN,
                                 method=MonteCarloMethod.IID)
        inputs = collect(run, monte_carlo=mc)
        page = render_page(inputs)
        return page
    def render_page(inputs):
        from flow_model.dashboard.build import build_page
        page = build_page(inputs)
        html = render(page)
        assert_origin_is_stated(html, page)
        out = REPORTS / "dashboard.html"
        out.write_text(html)
        panels = getattr(page, "panels", ())
        undefined = [p for p in panels if getattr(p, "undefined", None) is not None]
        print(f"rendered {out} ({len(html)} chars)")
        print(f"panels {len(panels)}  undefined {len(undefined)}")
        for p in panels:
            u = getattr(p, "undefined", None)
            print(f"  {p.panel_id.value:26} n={p.n:<5} "
                  f"{'undefined: ' + str(u) if u is not None else 'populated'}")
        return {"path": str(out), "chars": len(html),
                "panels": len(panels), "undefined": len(undefined)}
    results["dashboard"] = stage("4. DASHBOARD")(_dash)

    # 5. walk-forward over the research window
    def _wf():
        from flow_model.validation.splits import (
            DataAccessGuard, DateRange, SealAudit,
        )
        from flow_model.validation.walk_forward import (
            WindowSpec, evaluate_folds, generate_folds,
        )
        audit = SealAudit(REPORTS / "seal_audit.jsonl")
        sealed = DateRange(start=SEALED[0], end=SEALED[1])
        guard = DataAccessGuard(sealed=sealed, audit=audit,
                                config_hash=config.config_hash)
        spec = WindowSpec(train_days=120, validation_days=20, test_days=20,
                          step_days=20, purge_days=2, embargo_days=3)
        folds = generate_folds(
            data_range=DateRange(start=start, end=end), spec=spec,
            guard=guard, sealed=sealed,
        )
        print(f"folds generated: {len(folds.folds)}")
        for f in folds.folds[:6]:
            for w in f.windows if hasattr(f, "windows") else ():
                print(f"  {f.fold_id} {w.phase.value:12} {w.range}")
        report = evaluate_folds(
            fold_set=folds, config=config, runner=make_runner(calendar, config),
            guard=guard, sealed=sealed,
        )
        print(f"evaluated. folds in report: {len(report.folds)}")
        for fr in report.folds:
            for w in fr.windows:
                print(f"  {fr.fold.fold_id} {w.phase.value:12} "
                      f"trades={w.summary.trade_count.value if w.summary.trade_count.value is not None else 'undef'} "
                      f"sample={w.sample.value}")
        return report
    results["walk_forward"] = stage("5. WALK-FORWARD")(_wf)

    print("\n=== SUMMARY ===")
    for name, (status, _) in results.items():
        print(f"  {name:14} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
