"""Parameter robustness: is this value a plateau or a cliff edge?

A parameter that performs brilliantly at 65 and badly at 62.5 and 67.5 has
not been optimised - it has been fitted to noise. The neighbouring values
are drawn from the same market; if they disagree violently, the difference
is sampling luck, and the "optimum" will not survive contact with new data.

So the question is never "which value scored highest?" but "is there a
broad region of values that all work?". A slightly lower peak in the middle
of a stable plateau is worth more than a higher one on a spike, and this
module is built to prefer the plateau.

Two things it reports:

    CLIFFS      neighbouring values whose results differ by more than
                CLIFF_THRESHOLD_R. Any parameter with a cliff next to the
                chosen value is flagged, however good that value looks.

    PLATEAU     the widest run of adjacent values that are all positive
                and all within PLATEAU_TOLERANCE_R of each other. The
                recommendation is the CENTRE of that run, not its peak -
                the centre is the value with the most room for the market
                to change around it.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from app.backtesting.types import BacktestConfig
from app.data.candles import CandleSeries
from app.research.harness import Experiment, run_variants

# Neighbouring values differing by more than this are a cliff, not a slope.
CLIFF_THRESHOLD_R = 0.15

# Values within this of each other count as "the same result".
PLATEAU_TOLERANCE_R = 0.10

# A plateau narrower than this is not a plateau.
MIN_PLATEAU_WIDTH = 3


@dataclass
class ParameterPoint:
    value: float
    oos_expectancy_r: float | None
    oos_trades: int
    tested: bool

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "oos_expectancy_r": (
                round(self.oos_expectancy_r, 4) if self.oos_expectancy_r is not None else None
            ),
            "oos_trades": self.oos_trades,
            "tested": self.tested,
        }


@dataclass
class RobustnessReport:
    parameter: str
    points: list[ParameterPoint] = field(default_factory=list)
    experiment: Experiment | None = None

    @property
    def tested_points(self) -> list[ParameterPoint]:
        return [p for p in self.points if p.tested and p.oos_expectancy_r is not None]

    @property
    def conclusive(self) -> bool:
        return len(self.tested_points) >= MIN_PLATEAU_WIDTH

    @property
    def cliffs(self) -> list[tuple[float, float, float]]:
        """(from_value, to_value, jump) for every adjacent pair that jumps."""
        found = []
        ordered = sorted(self.tested_points, key=lambda p: p.value)
        for a, b in zip(ordered, ordered[1:]):
            jump = abs(b.oos_expectancy_r - a.oos_expectancy_r)
            if jump > CLIFF_THRESHOLD_R:
                found.append((a.value, b.value, jump))
        return found

    @property
    def plateau(self) -> list[ParameterPoint]:
        """The widest run of adjacent values that are all positive and all
        within PLATEAU_TOLERANCE_R of each other."""
        ordered = sorted(self.tested_points, key=lambda p: p.value)
        best: list[ParameterPoint] = []
        run: list[ParameterPoint] = []

        for point in ordered:
            if point.oos_expectancy_r <= 0:
                run = []
                continue
            if run:
                spread = max(p.oos_expectancy_r for p in run + [point]) - min(
                    p.oos_expectancy_r for p in run + [point]
                )
                if spread > PLATEAU_TOLERANCE_R:
                    run = [point]
                else:
                    run.append(point)
            else:
                run = [point]
            if len(run) > len(best):
                best = list(run)
        return best

    @property
    def recommended(self) -> float | None:
        """The CENTRE of the widest plateau, not the peak.

        The centre has the most room for the market to move around it. The
        peak of a spike is a number that happened once.
        """
        plateau = self.plateau
        if len(plateau) < MIN_PLATEAU_WIDTH:
            return None
        return statistics.median(p.value for p in plateau)

    @property
    def peak(self) -> ParameterPoint | None:
        tested = self.tested_points
        return max(tested, key=lambda p: p.oos_expectancy_r) if tested else None

    def verdict(self) -> str:
        if not self.conclusive:
            return (
                f"{self.parameter}: INSUFFICIENT DATA - only {len(self.tested_points)} of "
                f"{len(self.points)} values produced enough out-of-sample trades to compare. "
                "No value should be chosen from this."
            )
        plateau = self.plateau
        cliffs = self.cliffs
        if len(plateau) < MIN_PLATEAU_WIDTH:
            peak = self.peak
            return (
                f"{self.parameter}: NO STABLE REGION. No run of {MIN_PLATEAU_WIDTH}+ adjacent "
                f"values is consistently positive and consistent with each other. The best single "
                f"value ({peak.value:g}, {peak.oos_expectancy_r:+.3f}R) is an isolated peak, which "
                "is what an overfit parameter looks like. Do not adopt it."
            )
        lo, hi = plateau[0].value, plateau[-1].value
        text = (
            f"{self.parameter}: stable region from {lo:g} to {hi:g} "
            f"({len(plateau)} adjacent values, all positive and within "
            f"{PLATEAU_TOLERANCE_R}R of each other). Recommended value {self.recommended:g} - "
            "the centre of the plateau, not its peak."
        )
        if cliffs:
            worst = max(cliffs, key=lambda c: c[2])
            text += (
                f" WARNING: a cliff between {worst[0]:g} and {worst[1]:g} "
                f"({worst[2]:.3f}R jump) - results near there are not trustworthy."
            )
        return text

    def as_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "conclusive": self.conclusive,
            "verdict": self.verdict(),
            "recommended": self.recommended,
            "peak": self.peak.value if self.peak else None,
            "plateau": [p.value for p in self.plateau],
            "cliffs": [
                {"from": a, "to": b, "jump_r": round(j, 4)} for a, b, j in self.cliffs
            ],
            "points": [p.as_dict() for p in self.points],
        }

    def table(self) -> str:
        lines = [self.verdict(), "", f"  {'value':>10}{'OOS n':>8}{'OOS R':>10}   "]
        plateau_values = {p.value for p in self.plateau}
        for point in sorted(self.points, key=lambda p: p.value):
            marker = "plateau" if point.value in plateau_values else ""
            r = f"{point.oos_expectancy_r:+.3f}" if point.oos_expectancy_r is not None else "n/a"
            star = " " if point.tested else "*"
            lines.append(f"{star} {point.value:>9g}{point.oos_trades:>8}{r:>10}   {marker}")
        lines.append("\n  * too few out-of-sample trades to count")
        return "\n".join(lines)


def sweep_parameter(
    series: CandleSeries,
    *,
    parameter: str,
    values: list[float],
    base_config: BacktestConfig | None = None,
    train_fraction: float = 0.6,
) -> RobustnessReport:
    """Run one strategy parameter across a range of values.

    `values` should be a contiguous ladder including the value currently in
    use and its neighbours on both sides - testing 65 alone tells you
    nothing about whether 65 is safe.
    """
    base_config = base_config or BacktestConfig()
    if not hasattr(base_config, parameter):
        raise AttributeError(f"BacktestConfig has no parameter {parameter!r}")
    if len(values) < 2:
        raise ValueError("a robustness sweep needs at least two values to compare")

    variants: dict[str, BacktestConfig] = {}
    for value in values:
        variants[f"{parameter}={value:g}"] = BacktestConfig(
            **{**base_config.__dict__, parameter: value}
        )

    experiment = run_variants(
        f"robustness: {parameter}", series, variants, train_fraction=train_fraction
    )
    by_label = {v.label: v for v in experiment.variants}

    points = []
    for value in values:
        variant = by_label.get(f"{parameter}={value:g}")
        points.append(
            ParameterPoint(
                value=value,
                oos_expectancy_r=variant.oos_expectancy_r if variant else None,
                oos_trades=variant.oos_trades if variant else 0,
                tested=bool(variant and variant.tested),
            )
        )

    return RobustnessReport(parameter=parameter, points=points, experiment=experiment)
