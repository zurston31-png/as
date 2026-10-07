"""Automated lookahead audit.

ARCHITECTURE.md section 4 names three enforcement layers against lookahead
bias. Layer 1 is `MarketView`, which makes future data absent from the
object a feature is handed. This module is layer 3: it checks empirically
that a feature's output at bar *i* does not depend on anything after bar
*i*, which is the property the whole system rests on and the one that no
amount of code reading can establish.

Four checks per sampled bar:

1. **Truncation invariance.** Compute at bar *i* with the full dataset, then
   with a dataset that ends at bar *i*.

2. **Future-mutation invariance.** Compute at bar *i*, then again against a
   dataset whose bars after *i* have been replaced with wildly different
   values.

3. **Warmup honesty.** Before `warmup_bars`, the computer must report
   `warmup_complete=False`. A half-filled rolling window produces a number
   that looks like a feature and is not one.

4. **Determinism.** Two computations at the same bar must agree. Catches a
   computer reading a clock or an unseeded generator.

## What these checks can and cannot catch -- read this before trusting a PASS

Checks 1 and 2 are **no-ops by construction** for any computer that reads
market data only through its `MarketView`. Since Phase 2 a view holds
read-only prefixes sliced at its cutoff and no reference to the parent, so
the view built from a truncated or mutated dataset is *element-wise
identical* to the view built from the full one. That is the point of the
firewall, and it means these two checks now function as regression tests on
`MarketView` and as a determinism probe rather than as a test of the
computer. Verified empirically: they pass a computer that cheats and fail one
that is non-deterministic.

The leak they do NOT catch is a computer that obtains data outside the view
-- most plausibly by capturing a full-sample statistic at construction time
(`self.scale = dataset.close.mean()`). Two defenses, because this is the
only remaining hole:

* Pass `factory=` instead of an instance. The audit then rebuilds the
  computer for each altered dataset, and a captured constant changes with
  it. A construction-capturing computer fails immediately.
* `tests/unit/test_feature_contracts.py` asserts that no `FeatureComputer`
  constructor accepts market data at all, so the capture cannot be written
  in the first place.

A failure here is not a style problem. It invalidates every result the
system would ever produce, and it makes backtests look better rather than
worse, so nothing downstream would flag it.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from pydantic import Field

from flow_model.core.determinism import rng
from flow_model.core.model import FrozenModel
from flow_model.data.market_view import MarketView
from flow_model.data.series import BarSeries, ColumnSeries, from_ns
from flow_model.data.store import SymbolData
from flow_model.features.base import FeatureBundle, FeatureComputer
from flow_model.utils.logging import get_logger

logger = get_logger("validation.lookahead")

#: Features may differ by at most this, relative, between runs. Not zero:
#: NumPy reductions are not bit-identical across different array lengths
#: (pairwise summation changes the order of operations), so an exact-equality
#: check would report spurious failures on legitimate code. A real lookahead
#: changes a feature by far more than this.
DEFAULT_TOLERANCE = 1e-9


class LookaheadFinding(FrozenModel):
    """One feature whose value depended on the future."""

    computer: str
    key: str
    bar_index: int
    kind: str = Field(description="truncation | mutation | warmup")
    baseline: float
    observed: float
    detail: str = ""

    @property
    def relative_difference(self) -> float:
        scale = max(abs(self.baseline), abs(self.observed), 1e-12)
        return abs(self.baseline - self.observed) / scale


class LookaheadAuditResult(FrozenModel):
    """Audit outcome for one computer or bundle."""

    computer: str
    bars_checked: int = Field(ge=0)
    keys_checked: tuple[str, ...] = ()
    findings: tuple[LookaheadFinding, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.passed:
            return (
                f"{self.computer}: PASS ({self.bars_checked} bars x "
                f"{len(self.keys_checked)} features)"
            )
        kinds = sorted({f.kind for f in self.findings})
        return (
            f"{self.computer}: FAIL -- {len(self.findings)} finding(s) "
            f"({', '.join(kinds)}); first: {self.findings[0].key} at bar "
            f"{self.findings[0].bar_index} "
            f"({self.findings[0].baseline} vs {self.findings[0].observed})"
        )


def _cut_series(series: ColumnSeries | None, cutoff_ns: int):
    """Prefix a series at a timestamp, or None if it was None."""
    if series is None:
        return None
    return series.prefix(series.visible_count(cutoff_ns))


def truncate(data: SymbolData, cutoff_ns: int) -> SymbolData:
    """A dataset containing only observations at or before `cutoff_ns`.

    Truncation is by TIMESTAMP, not row count: the feeds have different
    cadences, and cutting each at the same row index would misalign them and
    produce a difference that looked like a lookahead.
    """
    return SymbolData(
        symbol=data.symbol,
        primary_interval=data.primary_interval,
        bars={i: _cut_series(s, cutoff_ns) for i, s in data.bars.items()},
        ticks={i: _cut_series(s, cutoff_ns) for i, s in data.ticks.items()},
        quotes=_cut_series(data.quotes, cutoff_ns),
        options=_cut_series(data.options, cutoff_ns),
        fingerprint=data.fingerprint,
    )


def _mutate_columns(
    series: ColumnSeries | None, cutoff_ns: int, factor: float
) -> ColumnSeries | None:
    """Replace every value after `cutoff_ns` with a wildly different one.

    Prices are scaled and volumes inflated rather than randomized, so the
    mutated series still satisfies OHLC ordering and the non-negativity
    constraints -- the point is to change the future's VALUES, not to make
    the series invalid and trip a constructor instead of the audit.
    """
    if series is None or len(series) == 0:
        return series
    count = series.visible_count(cutoff_ns)
    if count >= len(series):
        return series

    columns: dict[str, np.ndarray] = {}
    for name in series.columns:
        values = np.array(series.col(name), dtype=np.float64, copy=True)
        values[count:] = values[count:] * factor
        columns[name] = values

    if isinstance(series, BarSeries):
        return BarSeries(
            symbol=series.symbol,
            ts_ns=series.ts_ns,
            interval_seconds=series.interval_seconds,
            columns=columns,
            meta=series.meta,
        )
    return type(series)(
        symbol=series.symbol, ts_ns=series.ts_ns, columns=columns, meta=series.meta
    )


def mutate_after(data: SymbolData, cutoff_ns: int, factor: float = 7.0) -> SymbolData:
    """A dataset identical up to `cutoff_ns` and wildly different after it."""
    return SymbolData(
        symbol=data.symbol,
        primary_interval=data.primary_interval,
        bars={i: _mutate_columns(s, cutoff_ns, factor) for i, s in data.bars.items()},
        ticks={i: _mutate_columns(s, cutoff_ns, factor) for i, s in data.ticks.items()},
        quotes=_mutate_columns(data.quotes, cutoff_ns, factor),
        options=_mutate_columns(data.options, cutoff_ns, factor),
        fingerprint=data.fingerprint,
    )


def _sample_indices(total: int, warmup: int, sample: int, seed: int) -> list[int]:
    """Bar indices to audit: the first usable bars, the last, and a sample.

    The boundaries matter more than the middle. The first bar past warmup is
    where an off-by-one in a rolling window shows up, and the last bar is
    where a computer that reads "the whole series" looks correct by accident.
    """
    usable = [i for i in range(max(warmup, 1) - 1, total)]
    if not usable:
        return []
    chosen = {usable[0], usable[-1]}
    if len(usable) > 2:
        chosen.add(usable[1])
        chosen.add(usable[len(usable) // 2])
    # Pre-warmup bars, so check 3 is actually evaluated. Sampling only from
    # `usable` meant every sampled bar already had a full window and the
    # warmup-honesty claim was never tested -- a computer declaring
    # warmup_bars=250 while reporting ready at bar 20 passed the audit.
    if warmup > 1:
        chosen.update({0, min(warmup - 2, total - 1), max(0, warmup // 2)})
    remaining = [i for i in usable if i not in chosen]
    if remaining and sample > len(chosen):
        generator = rng(seed, "lookahead_audit")
        take = min(sample - len(chosen), len(remaining))
        picks = generator.choice(len(remaining), size=take, replace=False)
        chosen.update(int(remaining[int(p)]) for p in picks)
    return sorted(chosen)


def _differs(baseline: float, observed: float, tolerance: float) -> bool:
    if math.isnan(baseline) and math.isnan(observed):
        return False
    if not math.isfinite(baseline) or not math.isfinite(observed):
        return baseline != observed
    scale = max(abs(baseline), abs(observed), 1.0)
    return abs(baseline - observed) / scale > tolerance


def audit_computer(
    computer: FeatureComputer | FeatureBundle | None = None,
    data: SymbolData | None = None,
    *,
    factory: Callable[[SymbolData], FeatureComputer | FeatureBundle] | None = None,
    latency_seconds: float = 0.0,
    sample: int = 40,
    mutation_factor: float = 7.0,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 20240101,
) -> LookaheadAuditResult:
    """Audit one computer against a dataset. Returns findings, never raises.

    Pass `factory` rather than `computer` to also catch construction-time
    capture: the computer is then rebuilt for each altered dataset, so a
    full-sample constant captured in `__init__` changes and is reported.
    """
    if data is None:
        raise ValueError("audit_computer requires a dataset")
    if computer is None and factory is None:
        raise ValueError("audit_computer requires either a computer or a factory")
    if computer is None:
        assert factory is not None
        computer = factory(data)
    name = getattr(computer, "name", type(computer).__name__)
    bars = data.primary_bars
    total = len(bars)
    warmup = computer.warmup_bars
    findings: list[LookaheadFinding] = []
    notes: list[str] = []

    if total <= warmup:
        return LookaheadAuditResult(
            computer=name,
            bars_checked=0,
            notes=(
                f"dataset has {total} bars but {name} needs {warmup}; nothing audited. "
                "This is not a pass.",
            ),
        )

    indices = _sample_indices(total, warmup, sample, seed)
    keys_seen: set[str] = set()

    for index in indices:
        cutoff = int(bars.ts_ns[index])
        now = from_ns(cutoff)

        view = MarketView(data, now=now, latency_seconds=latency_seconds, now_ns=cutoff)
        baseline = computer.compute(view)
        keys_seen.update(baseline.values)

        # 4. determinism: the same view twice must agree.
        repeat = computer.compute(view)
        for key, value in baseline.values.items():
            other = repeat.values.get(key)
            if other is None or _differs(value, other, tolerance):
                findings.append(
                    LookaheadFinding(
                        computer=name, key=key, bar_index=index, kind="determinism",
                        baseline=value,
                        observed=float("nan") if other is None else other,
                        detail="two computations on the identical view disagreed",
                    )
                )

        # 3. warmup honesty
        if index + 1 < warmup and baseline.warmup_complete:
            findings.append(
                LookaheadFinding(
                    computer=name, key="(warmup)", bar_index=index, kind="warmup",
                    baseline=float(warmup), observed=float(index + 1),
                    detail=(
                        f"reported warmup_complete with {index + 1} bars visible but "
                        f"declares warmup_bars={warmup}"
                    ),
                )
            )

        for kind, candidate in (
            ("truncation", truncate(data, cutoff)),
            ("mutation", mutate_after(data, cutoff, mutation_factor)),
        ):
            # With a factory, the computer is rebuilt against the altered
            # dataset, so a constant captured in __init__ moves with it.
            subject = factory(candidate) if factory is not None else computer
            observed = subject.compute(
                MarketView(
                    candidate, now=now, latency_seconds=latency_seconds, now_ns=cutoff
                )
            )
            for key, value in baseline.values.items():
                other = observed.values.get(key)
                if other is None:
                    findings.append(
                        LookaheadFinding(
                            computer=name, key=key, bar_index=index, kind=kind,
                            baseline=value, observed=float("nan"),
                            detail="key absent under the altered dataset",
                        )
                    )
                elif _differs(value, other, tolerance):
                    findings.append(
                        LookaheadFinding(
                            computer=name, key=key, bar_index=index, kind=kind,
                            baseline=value, observed=other,
                            detail=(
                                f"value changed when data after bar {index} was "
                                f"{'removed' if kind == 'truncation' else 'altered'}"
                            ),
                        )
                    )

    result = LookaheadAuditResult(
        computer=name,
        bars_checked=len(indices),
        keys_checked=tuple(sorted(keys_seen)),
        findings=tuple(findings),
        notes=tuple(notes),
    )
    (logger.info if result.passed else logger.error)("%s", result.summary())
    return result


def audit_all(
    computers: Sequence[FeatureComputer | FeatureBundle],
    data: SymbolData,
    **kwargs,
) -> tuple[LookaheadAuditResult, ...]:
    return tuple(audit_computer(c, data, **kwargs) for c in computers)


def assert_no_lookahead(results: Sequence[LookaheadAuditResult]) -> None:
    """Raise with every finding listed. For use as a phase gate."""
    failed = [r for r in results if not r.passed]
    empty = [r for r in results if r.bars_checked == 0]
    if not failed and not empty:
        return
    lines = ["lookahead audit failed:"]
    for result in failed:
        lines.append(f"  {result.summary()}")
        for finding in result.findings[:5]:
            lines.append(
                f"    {finding.kind}: {finding.key} at bar {finding.bar_index}: "
                f"{finding.baseline} -> {finding.observed} ({finding.detail})"
            )
    for result in empty:
        lines.append(f"  {result.computer}: NOTHING AUDITED -- {'; '.join(result.notes)}")
    raise AssertionError("\n".join(lines))
