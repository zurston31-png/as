"""The population Flow Score distribution.

Today only 283 of 19542 bars carry a flow_score, because the pipeline
short-circuits at the section 14 structure gate before the scorers run. But
`SignalEngine.score(features)` takes ONLY the feature vector -- the side is
needed by the options/order-flow GATES, not by the scorers -- so the score is
well defined for every bar that has features, i.e. every bar past the regime
gate.

This scores all of them, through the engine's own public methods. No gate is
bypassed or weakened, no threshold moved, and production code is untouched:
each bar's decision is taken exactly as the backtest takes it, and the score is
then computed from the features that decision already carried.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from datetime import date

import numpy as np

from flow_model.backtest.loop import StoreMarketSource
from flow_model.config.loader import default_config
from flow_model.data import DataStore, SyntheticAdapter, SyntheticConfig, TradingCalendar
from flow_model.signals.engine import SignalEngine

start = date(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
end = date(int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

config = default_config()
calendar = TradingCalendar()
adapter = SyntheticAdapter(
    config=SyntheticConfig(), seed=config.seed,
    spec_by_symbol=dict(config.instruments),
    interval_seconds=config.data.primary_interval_seconds, calendar=calendar,
)
store = DataStore(latency_seconds=config.data.data_latency_seconds)
store.load(adapter, "NQ", start, end, primary_interval=config.data.primary_interval_seconds)

engine = SignalEngine(config, "NQ", calendar=calendar)
source = StoreMarketSource.from_config(store, "NQ", config)
clock = source.clock_ns()

gate_scores: list[float] = []     # what the pipeline publishes today
pop_scores: list[float] = []      # every bar that HAS features
blocked_at = Counter()
no_features = 0
refused = 0

for ts_ns in clock.tolist():
    source.advance(int(ts_ns))
    if source.bar() is None:
        continue
    view = source.view()
    d = engine.decide(view, equity=float(config.risk.starting_equity))
    if d.blocking is not None:
        blocked_at[d.blocking.stage.value] += 1
    if d.signal.flow_score is not None:
        gate_scores.append(float(d.signal.flow_score))
    if d.features is None:
        no_features += 1
        continue
    outcome = engine.aggregate(engine.score(d.features))
    if outcome.score is not None:
        pop_scores.append(float(outcome.score))
    else:
        refused += 1

def describe(name, xs):
    if not xs:
        print(f"{name}: n=0 -- undefined")
        return
    a = np.asarray(xs, dtype=float)
    qs = np.percentile(a, [0, 5, 25, 50, 75, 90, 95, 99, 100])
    print(f"{name}: n={a.size}")
    print(f"   min {qs[0]:.4f}  p5 {qs[1]:.4f}  p25 {qs[2]:.4f}  median {qs[3]:.4f}")
    print(f"   p75 {qs[4]:.4f}  p90 {qs[5]:.4f}  p95 {qs[6]:.4f}  p99 {qs[7]:.4f}  max {qs[8]:.4f}")
    print(f"   mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}")
    for thr in (70.0, 75.0, 80.0):
        n = int((a >= thr).sum())
        print(f"   bars >= {thr:.0f}: {n}  ({100.0*n/a.size:.4f}%)")
    print(f"   out of [0,100]: {int(((a<0)|(a>100)).sum())}")

print(f"window {start} -> {end}   bars on clock {clock.size}")
print(f"bars with NO feature vector (blocked before liquidity): {no_features}")
print(f"bars whose score the aggregate REFUSED (honest degradation): {refused}")
print()
describe("PUBLISHED today (reach the flow_score stage)", gate_scores)
print()
describe("POPULATION (every bar with features)", pop_scores)
print()
print("blocking stages:", json.dumps(dict(sorted(blocked_at.items(), key=lambda kv: -kv[1])), indent=2))
