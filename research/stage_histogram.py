"""My own stage-histogram measurement. Committed default config, unchanged."""
from __future__ import annotations
import json
import sys
from datetime import date
from collections import Counter

from flow_model.backtest.loop import BacktestLoop, StoreMarketSource
from flow_model.config.loader import default_config
from flow_model.core.enums import SplitPhase
from flow_model.data import DataStore, SyntheticAdapter, SyntheticConfig, TradingCalendar

start = date(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
end = date(int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

config = default_config()
calendar = TradingCalendar()
adapter = SyntheticAdapter(
    config=SyntheticConfig(),
    seed=config.seed,
    spec_by_symbol=dict(config.instruments),
    interval_seconds=config.data.primary_interval_seconds,
    calendar=calendar,
)
store = DataStore(latency_seconds=config.data.data_latency_seconds)
store.load(adapter, "NQ", start, end, primary_interval=config.data.primary_interval_seconds)

loop = BacktestLoop(
    config, "NQ", calendar=calendar,
    split_label=SplitPhase.TRAIN, split_name="verify-stages",
    git_commit="", git_dirty=False,
)
result = loop.run(StoreMarketSource.from_config(store, "NQ", config))

J = json.dumps
print("window              :", start, "->", end)
print("bars evaluated      :", result.bars_evaluated)
print("trades              :", result.trade_count, " is_empty:", result.is_empty)
print("orders submitted    :", len(result.orders))
print("ending equity       :", result.final.equity if hasattr(result.final, "equity") else result.final)

print("\n--- result.rejection (the loop's own counters) ---")
print(J(result.rejection, indent=2, default=str))

# --- independent recount, straight off the persisted rows ---
rows = result.signals
print("\n--- independent recount from", len(rows), "persisted signal rows ---")
stage_hist = Counter(r.blocking_stage for r in rows if r.blocking_stage)
wait_hist = Counter(r.wait_reason.value for r in rows if r.wait_reason is not None)
print("blocking_stage:", J(dict(sorted(stage_hist.items(), key=lambda kv: -kv[1])), indent=2))
print("wait_reason  :", J(dict(sorted(wait_hist.items(), key=lambda kv: -kv[1])), indent=2))

scored = [r.flow_score for r in rows if r.flow_score is not None]
print("\nbars carrying a flow_score  :", len(scored), "of", len(rows))
if scored:
    print("max / min / mean flow_score :", max(scored), "/", min(scored), "/", sum(scored)/len(scored))
    print("out of range [0,100]        :", [s for s in scored if s < 0 or s > 100])

reached = Counter()
for r in rows:
    for s in r.stages_run:
        reached[s] += 1
print("\n--- bars that REACHED each stage, in pipeline order ---")
order = []
for r in rows:
    if len(r.stages_run) > len(order):
        order = list(r.stages_run)
for s in order:
    print(f"  {s:<28} reached {reached[s]:>6}   blocked_here {stage_hist.get(s,0):>6}")
leftover = [s for s in reached if s not in order]
for s in leftover:
    print(f"  {s:<28} reached {reached[s]:>6}   blocked_here {stage_hist.get(s,0):>6}  (not in longest path)")
