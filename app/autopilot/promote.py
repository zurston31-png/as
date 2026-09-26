"""Whether a challenger has actually earned the champion's place.

Six independent bars. A challenger clears all six or it is discarded, and
the discard is recorded so the same idea is not quietly retried until it
passes.

    1. SAMPLE       enough out-of-sample trades to measure anything
    2. EFFECT       the edge is large enough to matter after costs
    3. SIGNIFICANCE it survives a test that knows how many challengers
                    were tried
    4. OUT-OF-SAMPLE it was not chosen on the data it is judged on
    5. CONSISTENCY  it holds across market conditions, not one lucky regime
    6. RISK         it does not buy its return with drawdown

WHY BAR 3 IS THE WHOLE POINT

Run an automated search over two hundred parameter sets and about ten will
clear p < 0.05 against a champion that is exactly as good. Not because the
data is thin - because twenty independent coin flips produce a run of heads
somewhere, and a search is built to go looking for it. Reporting the winner
without saying how many were tried turns noise into a recommendation.

So the threshold is corrected for the number of challengers evaluated in
the round (Bonferroni - conservative on purpose; a loop that runs
unattended should err toward not changing anything). `attempts` is not
optional and defaults to nothing: a caller that forgets to pass it gets an
error, not a free pass.

WHY THERE IS NO t-TEST

Trade returns are not normal. They are skewed, fat-tailed, and in memecoins
dominated by a handful of outliers - which is precisely the shape that
breaks parametric tests and precisely the shape where a false positive is
most likely. So significance comes from a paired bootstrap over the actual
realised returns: resample the same candidates many times, and ask how
often the challenger loses. No distributional assumption, and it degrades
honestly when the sample is small.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- the bars -------------------------------------------------------------

# Below this many out-of-sample trades, nothing here is measurement.
MIN_OOS_TRADES = 30

# An improvement smaller than this is not worth a strategy change even when
# it is real: it is inside the noise of fee and slippage estimation, and
# every promotion carries the risk of having measured the wrong thing.
MIN_EFFECT_R = 0.05

# Uncorrected significance level, before the multiple-comparison penalty.
BASE_ALPHA = 0.05

# Resamples for the bootstrap. Enough that the p-value is stable to ~0.005.
BOOTSTRAP_ROUNDS = 2000

# A challenger must not be worse than the champion in ANY regime with a
# usable sample, even if the average is better. An edge that only exists in
# one market condition is a bet on that condition continuing.
MIN_REGIME_TRADES = 15

# How much extra drawdown a challenger may take on per unit of extra
# return before the trade-off stops being worth it.
MAX_DRAWDOWN_COST_RATIO = 2.0


@dataclass
class Arm:
    """One side of the comparison, as measured out-of-sample."""
    label: str
    returns_r: list[float] = field(default_factory=list)
    max_drawdown_pct: float = 0.0
    # regime label -> realised returns in that regime
    by_regime: dict[str, list[float]] = field(default_factory=dict)

    @property
    def trades(self) -> int:
        return len(self.returns_r)

    @property
    def expectancy_r(self) -> float | None:
        return sum(self.returns_r) / len(self.returns_r) if self.returns_r else None

    @property
    def win_rate_pct(self) -> float | None:
        if not self.returns_r:
            return None
        return sum(1 for r in self.returns_r if r > 0) / len(self.returns_r) * 100

    @property
    def profit_factor(self) -> float | None:
        """None rather than infinity when nothing lost yet.

        A sample with no losers is a small sample, not an infinite edge,
        and inf would sort above every real number.
        """
        gains = sum(r for r in self.returns_r if r > 0)
        losses = abs(sum(r for r in self.returns_r if r < 0))
        return (gains / losses) if losses > 0 else None

    def as_dict(self) -> dict:
        def r(v, n=4):
            return round(v, n) if v is not None else None
        return {
            "label": self.label,
            "trades": self.trades,
            "expectancy_r": r(self.expectancy_r),
            "win_rate_pct": r(self.win_rate_pct, 1),
            "profit_factor": r(self.profit_factor, 3),
            "max_drawdown_pct": r(self.max_drawdown_pct, 2),
            "regimes": {k: len(v) for k, v in sorted(self.by_regime.items())},
        }


# Three outcomes, not two. "We have not measured enough to say" is a
# different answer from "we measured and it is not better", and collapsing
# them is how a loop concludes that a strategy failed when it was never
# tested. FAIL means evidence against; INSUFFICIENT_DATA means no evidence
# either way, and only the first is a reason to stop exploring an idea.
PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Bars that can only ever be answered by collecting more data. A failure on
# any of these is a statement about the SAMPLE, not about the challenger.
EVIDENCE_BARS = frozenset({"sample", "significance", "out-of-sample", "consistency"})


@dataclass
class Bar:
    """One requirement, and whether it was met."""
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return {"bar": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class Verdict:
    champion: Arm
    challenger: Arm
    attempts: int
    bars: list[Bar] = field(default_factory=list)
    p_value: float | None = None
    corrected_alpha: float = BASE_ALPHA

    @property
    def promote(self) -> bool:
        return self.outcome == PASS

    @property
    def outcome(self) -> str:
        """PASS, FAIL, or INSUFFICIENT_DATA.

        The distinction that matters: a challenger that lost on effect or
        risk has been WEIGHED and found wanting. One that only fell short
        on sample size or significance has not been weighed at all, and
        recording that as a failure would retire an idea that was never
        tested - while also making the loop look like it is ruling things
        out when it is only running out of data.
        """
        if not self.bars:
            return INSUFFICIENT_DATA
        if all(b.passed for b in self.bars):
            return PASS
        judged = [b for b in self.failed if b.name not in EVIDENCE_BARS]
        return FAIL if judged else INSUFFICIENT_DATA

    @property
    def failed(self) -> list[Bar]:
        return [b for b in self.bars if not b.passed]

    def reason(self) -> str:
        if self.outcome == INSUFFICIENT_DATA:
            first = self.failed[0] if self.failed else None
            return (
                f"INSUFFICIENT_DATA on {self.challenger.label}: "
                + (f"{first.name} - {first.detail}. " if first else "")
                + "This is not evidence against the challenger. It has not been measured "
                "well enough to judge, and the answer is more paper trading, not a "
                "different challenger."
            )
        if self.promote:
            lift = (self.challenger.expectancy_r or 0) - (self.champion.expectancy_r or 0)
            return (
                f"PROMOTE {self.challenger.label}: +{lift:.3f}R over {self.champion.label} "
                f"across {self.challenger.trades} out-of-sample trades, p={self.p_value:.4f} "
                f"against a {self.corrected_alpha:.4f} bar corrected for {self.attempts} "
                f"challenger(s). Clearing every bar is the only way through."
            )
        first = self.failed[0]
        return (
            f"KEEP {self.champion.label}: {self.challenger.label} failed on "
            f"{first.name} - {first.detail}"
            + (f" (and {len(self.failed) - 1} other bar(s))" if len(self.failed) > 1 else "")
        )

    def as_dict(self) -> dict:
        return {
            "promote": self.promote,
            "outcome": self.outcome,
            "reason": self.reason(),
            "attempts": self.attempts,
            "p_value": round(self.p_value, 5) if self.p_value is not None else None,
            "corrected_alpha": round(self.corrected_alpha, 5),
            "champion": self.champion.as_dict(),
            "challenger": self.challenger.as_dict(),
            "bars": [b.as_dict() for b in self.bars],
        }

    def table(self) -> str:
        lines = [self.reason(), ""]
        for b in self.bars:
            lines.append(f"  [{'PASS' if b.passed else 'FAIL'}] {b.name:<14} {b.detail}")
        return "\n".join(lines)


def bootstrap_p_value(
    champion: list[float],
    challenger: list[float],
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 12345,
) -> float | None:
    """How often the challenger fails to beat the champion under resampling.

    Deliberately non-parametric. Trade returns are skewed and fat-tailed -
    a handful of outliers carry the mean - which is exactly where a t-test
    reports confidence it has not earned.

    Seeded, so the same comparison gives the same answer twice. An
    unseeded gate could be re-run until it agreed.
    """
    if not champion or not challenger:
        return None

    rng = random.Random(seed)
    observed = (sum(challenger) / len(challenger)) - (sum(champion) / len(champion))
    if observed <= 0:
        return 1.0                      # not better; no need to resample

    # Resample each arm independently, with replacement, at its own size.
    # Counting how often the resampled lift is <= 0 estimates the chance
    # the observed lift came from sampling variation.
    losses = 0
    for _ in range(rounds):
        a = sum(rng.choice(champion) for _ in champion) / len(champion)
        b = sum(rng.choice(challenger) for _ in challenger) / len(challenger)
        if (b - a) <= 0:
            losses += 1
    return losses / rounds


def _regime_check(champion: Arm, challenger: Arm) -> Bar:
    """Does the edge hold everywhere, or only in one market condition?"""
    shared = [
        regime for regime in challenger.by_regime
        if regime in champion.by_regime
        and len(challenger.by_regime[regime]) >= MIN_REGIME_TRADES
        and len(champion.by_regime[regime]) >= MIN_REGIME_TRADES
    ]
    if not shared:
        return Bar(
            "consistency", False,
            f"no market condition has {MIN_REGIME_TRADES}+ trades on both sides, so the "
            "edge has not been shown to survive a change of conditions. An improvement "
            "measured in one regime is a bet on that regime continuing.",
        )

    worse = []
    for regime in shared:
        a = sum(champion.by_regime[regime]) / len(champion.by_regime[regime])
        b = sum(challenger.by_regime[regime]) / len(challenger.by_regime[regime])
        if b < a:
            worse.append(f"{regime} ({b:+.3f}R vs {a:+.3f}R)")

    if worse:
        return Bar(
            "consistency", False,
            f"worse than the champion in {', '.join(worse)}. A better average that is "
            "worse somewhere is a regime bet, not an improvement.",
        )
    return Bar(
        "consistency", True,
        f"at least as good in all {len(shared)} comparable market condition(s): "
        + ", ".join(sorted(shared)),
    )


def evaluate(champion: Arm, challenger: Arm, *, attempts: int) -> Verdict:
    """Judge one challenger against the champion.

    `attempts` is the number of challengers evaluated in this round and is
    REQUIRED. Without it the significance bar cannot be corrected, and an
    uncorrected bar in an automated search is how noise gets promoted.
    """
    if attempts < 1:
        raise ValueError(
            "attempts must be at least 1 - the significance bar is corrected for how "
            "many challengers were tried, and omitting that is how a search promotes noise"
        )

    verdict = Verdict(champion=champion, challenger=challenger, attempts=attempts)
    verdict.corrected_alpha = BASE_ALPHA / attempts

    # 1. sample
    verdict.bars.append(Bar(
        "sample", challenger.trades >= MIN_OOS_TRADES,
        f"{challenger.trades} out-of-sample trades (need >={MIN_OOS_TRADES})",
    ))

    # 2. effect
    a, b = champion.expectancy_r, challenger.expectancy_r
    lift = (b - a) if a is not None and b is not None else None
    verdict.bars.append(Bar(
        "effect", lift is not None and lift >= MIN_EFFECT_R,
        (
            f"lift {lift:+.4f}R (need >=+{MIN_EFFECT_R}R - smaller than that sits inside "
            "the error on the fee and slippage estimate)"
            if lift is not None else "expectancy unmeasurable on one or both arms"
        ),
    ))

    # 3. significance, corrected for how many were tried
    verdict.p_value = bootstrap_p_value(champion.returns_r, challenger.returns_r)
    significant = verdict.p_value is not None and verdict.p_value < verdict.corrected_alpha
    verdict.bars.append(Bar(
        "significance", significant,
        (
            f"p={verdict.p_value:.4f} against {verdict.corrected_alpha:.4f} "
            f"(0.05 corrected for {attempts} challenger(s))"
            if verdict.p_value is not None else "not enough data on one arm to resample"
        ),
    ))

    # 4. out-of-sample
    both_have_oos = champion.trades > 0 and challenger.trades > 0
    verdict.bars.append(Bar(
        "out-of-sample", both_have_oos,
        "both arms measured on data neither was tuned on" if both_have_oos
        else "one arm has no out-of-sample trades",
    ))

    # 5. consistency across regimes
    verdict.bars.append(_regime_check(champion, challenger))

    # 6. risk
    extra_dd = challenger.max_drawdown_pct - champion.max_drawdown_pct
    if lift is None or lift <= 0:
        risk_ok, detail = False, "no return improvement to weigh drawdown against"
    elif extra_dd <= 0:
        risk_ok = True
        detail = f"drawdown {extra_dd:+.2f}pp - no worse than the champion"
    else:
        ratio = extra_dd / (lift * 100)
        risk_ok = ratio <= MAX_DRAWDOWN_COST_RATIO
        detail = (
            f"buys {lift:+.3f}R with {extra_dd:+.2f}pp more drawdown "
            f"(ratio {ratio:.2f}, limit {MAX_DRAWDOWN_COST_RATIO})"
        )
    verdict.bars.append(Bar("risk", risk_ok, detail))

    return verdict
