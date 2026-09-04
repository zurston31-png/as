"""The profitability validation gate.

This module answers exactly one question, and refuses to answer anything
else: HAS THIS STRATEGY BEEN SHOWN TO WORK, OR DOES IT MERELY LOOK GOOD SO
FAR?

Every strategy starts EXPERIMENTAL and stays there until it clears every
criterion below. A strategy that is profitable on eleven trades is not
validated; it is a strategy with eleven trades. The distinction is the
whole point, because the failure mode this guards against is the one that
actually costs people money: a promising early sample, read as proof,
followed by real capital.

The criteria, and why each is there:

  SAMPLE SIZE       Below ~100 closed trades a win rate has a confidence
                    interval wide enough to contain both "excellent" and
                    "losing". No amount of good performance substitutes.
  EXPECTANCY        Must be positive AFTER costs. A strategy with a 70%
                    win rate and negative expectancy loses money slowly.
  PROFIT FACTOR     Gross profit / gross loss. Above 1.0 is "made money";
                    the bar is set higher because the estimate itself is
                    noisy and real execution is worse than simulated.
  MAX DRAWDOWN      A path nobody would actually sit through is not a
                    usable strategy, whatever its endpoint.
  CONCENTRATION     If one trade produced most of the profit, the sample
                    is one trade wearing a costume.
  MONTE CARLO       The observed order was luck. The 95th-percentile
                    resampled drawdown, not the realized one, is the
                    drawdown to plan around.
  OUT-OF-SAMPLE     In-sample results are curve-fitted by construction.
  WALK-FORWARD      The only test that asks whether the strategy survives
                    being re-fit as the market changes.

Nothing here loosens on its own, and nothing here reads the live P&L to
decide whether to relax. If the answer is "not yet", the answer is "not
yet".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ValidationStatus(str, Enum):
    EXPERIMENTAL = "experimental"      # not enough evidence, whatever the numbers say
    FAILING = "failing"                # enough evidence, and the answer is no
    VALIDATED = "validated"            # cleared every criterion


# The three states are deliberately distinct, and the distinction between
# the first two is the one that matters: EXPERIMENTAL means "we don't know
# yet", FAILING means "we know, and it's no". Collapsing them would either
# let a known-bad strategy hide behind incomplete testing, or brand an
# untested one a failure.


# --- thresholds -------------------------------------------------------------
# Deliberately module constants rather than settings: these are the bar for
# calling a strategy proven, and a bar that can be lowered from .env when
# the strategy fails to clear it is not a bar. Changing them is a code
# change, reviewed as one.

MIN_CLOSED_TRADES = 100
MIN_EXPECTANCY_USD = 0.0            # must be strictly positive, after costs
MIN_PROFIT_FACTOR = 1.30
MAX_ACCEPTABLE_DRAWDOWN_PCT = 25.0
MAX_SINGLE_TRADE_PROFIT_SHARE = 0.40
# With only a handful of winners, one of them being most of the profit is
# arithmetic, not concentration - with three winners the best is at least
# a third by definition. Below this the criterion says nothing.
MIN_WINNERS_FOR_CONCENTRATION = 10
MAX_MONTE_CARLO_P95_DRAWDOWN_PCT = 35.0
MIN_OUT_OF_SAMPLE_TRADES = 30
MIN_WALK_FORWARD_WINDOWS = 3
MIN_WALK_FORWARD_PROFITABLE_SHARE = 0.60


@dataclass
class Criterion:
    name: str
    passed: bool
    detail: str
    blocking: bool = True
    # False when there isn't enough evidence to judge this yet - either the
    # input hasn't been produced (no walk-forward has been run) or the
    # sample behind it is too small to mean anything. Insufficient evidence
    # is NOT a pass, and it is NOT a failure either.
    evidence_sufficient: bool = True

    @property
    def state(self) -> str:
        if not self.evidence_sufficient:
            return "insufficient data"
        return "pass" if self.passed else "FAIL"


@dataclass
class ValidationReport:
    status: ValidationStatus
    criteria: list[Criterion] = field(default_factory=list)
    headline: str = ""

    @property
    def failures(self) -> list[Criterion]:
        """Criteria with enough evidence behind them that failed anyway."""
        return [c for c in self.criteria if c.evidence_sufficient and not c.passed]

    @property
    def insufficient_evidence(self) -> list[Criterion]:
        """Criteria that cannot be judged yet. Not passes, not failures."""
        return [c for c in self.criteria if not c.evidence_sufficient]

    @property
    def cleared_for_real_money(self) -> bool:
        """Deliberately not the same as `status == VALIDATED`.

        Passing every statistical criterion means the paper record is
        strong. It is still a paper record: fills were simulated, and no
        amount of simulated evidence is evidence about real execution. This
        property exists so no caller can mistake one for the other, and it
        is always False.
        """
        return False

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "headline": self.headline,
            "criteria": [
                {
                    "name": c.name, "state": c.state, "passed": c.passed,
                    "evidence_sufficient": c.evidence_sufficient,
                    "blocking": c.blocking, "detail": c.detail,
                }
                for c in self.criteria
            ],
            "failure_count": len(self.failures),
            "insufficient_evidence_count": len(self.insufficient_evidence),
        }

    def summary(self) -> str:
        lines = [f"Strategy status: {self.status.value.upper()}", f"  {self.headline}", ""]
        for c in self.criteria:
            marker = {"pass": "  ok  ", "FAIL": " FAIL ", "insufficient data": "  ?   "}[c.state]
            lines.append(f"{marker} {c.name:<28} {c.detail}")
        return "\n".join(lines)


@dataclass
class ValidationInputs:
    """Everything the gate needs, gathered by the caller.

    Anything genuinely unknown stays None and is reported as insufficient
    evidence - never defaulted to a passing value, and never to zero.
    """

    closed_trades: int
    expectancy_usd: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    best_trade_share_of_profit: float | None = None
    winning_trades: int | None = None
    monte_carlo_p95_drawdown_pct: float | None = None
    monte_carlo_sample_size: int | None = None
    out_of_sample_trades: int | None = None
    out_of_sample_profitable: bool | None = None
    walk_forward_windows: int | None = None
    walk_forward_profitable_windows: int | None = None


def evaluate(inputs: ValidationInputs) -> ValidationReport:
    """Judge a strategy against every criterion. No partial credit."""
    criteria: list[Criterion] = [
        _sample_size(inputs),
        _expectancy(inputs),
        _profit_factor(inputs),
        _drawdown(inputs),
        _concentration(inputs),
        _monte_carlo(inputs),
        _out_of_sample(inputs),
        _walk_forward(inputs),
    ]

    blocking = [c for c in criteria if c.blocking]
    unknown = [c for c in blocking if not c.evidence_sufficient]
    failed = [c for c in blocking if c.evidence_sufficient and not c.passed]

    if failed:
        # A measured failure is a real answer, and outranks "not enough
        # data": a strategy losing money over 300 trades is failing, not
        # experimental, whether or not a walk-forward has been run.
        status = ValidationStatus.FAILING
        headline = (
            f"{len(failed)} criterion(s) failed on the evidence available: "
            + "; ".join(c.name for c in failed)
        )
    elif unknown:
        status = ValidationStatus.EXPERIMENTAL
        headline = (
            "no criterion has failed, but there is not enough evidence to judge "
            + ", ".join(c.name for c in unknown)
            + " - nothing is proven"
        )
    else:
        status = ValidationStatus.VALIDATED
        headline = (
            "every criterion cleared on paper. This is evidence about the STRATEGY, "
            "not about real execution - fills were simulated throughout."
        )

    return ValidationReport(status=status, criteria=criteria, headline=headline)


# ---------------------------------------------------------------------------
# individual criteria
# ---------------------------------------------------------------------------

def _sample_size(i: ValidationInputs) -> Criterion:
    """Too few trades is the one criterion that can never be a FAILURE.

    A strategy with 40 trades has not failed the sample-size test; it has
    not taken it yet. Marking it failed would brand every new strategy a
    bad one, and the status this produces (EXPERIMENTAL) is the accurate
    description of "we don't know yet".
    """
    enough = i.closed_trades >= MIN_CLOSED_TRADES
    return Criterion(
        "sample size", enough,
        f"{i.closed_trades} closed trades (need >={MIN_CLOSED_TRADES})"
        + ("" if enough else " - a good result on this many would still not be evidence"),
        evidence_sufficient=enough,
    )


def _expectancy(i: ValidationInputs) -> Criterion:
    if i.expectancy_usd is None:
        return Criterion("expectancy", False, "no closed trades to compute expectancy from",
                         evidence_sufficient=False)
    passed = i.expectancy_usd > MIN_EXPECTANCY_USD
    return Criterion(
        "expectancy", passed,
        f"${i.expectancy_usd:,.2f} per trade after costs (need > ${MIN_EXPECTANCY_USD:,.2f})",
    )


def _profit_factor(i: ValidationInputs) -> Criterion:
    if i.profit_factor is None:
        return Criterion("profit factor", False, "not computable yet (no trades)",
                         evidence_sufficient=False)
    if i.profit_factor == float("inf"):
        # Wins and no losses at all. Arithmetically infinite, evidentially
        # meaningless - it means the losing trades haven't happened yet.
        # Insufficient evidence rather than a failure, for the same reason
        # as sample size: nothing has been shown to be wrong, only that
        # nothing has been shown.
        return Criterion(
            "profit factor", False,
            "infinite - no losing trades yet, which is a sign of a small sample, not of an edge",
            evidence_sufficient=False,
        )
    passed = i.profit_factor >= MIN_PROFIT_FACTOR
    return Criterion(
        "profit factor", passed,
        f"{i.profit_factor:.2f} gross profit / gross loss (need >={MIN_PROFIT_FACTOR:.2f})",
    )


def _drawdown(i: ValidationInputs) -> Criterion:
    if i.max_drawdown_pct is None:
        return Criterion("max drawdown", False, "no equity curve yet", evidence_sufficient=False)
    passed = i.max_drawdown_pct <= MAX_ACCEPTABLE_DRAWDOWN_PCT
    return Criterion(
        "max drawdown", passed,
        f"{i.max_drawdown_pct:.1f}% realized (limit {MAX_ACCEPTABLE_DRAWDOWN_PCT:.0f}%)",
    )


def _concentration(i: ValidationInputs) -> Criterion:
    if i.best_trade_share_of_profit is None:
        return Criterion("profit concentration", False, "no profitable trades yet",
                         evidence_sufficient=False)
    if (i.winning_trades or 0) < MIN_WINNERS_FOR_CONCENTRATION:
        return Criterion(
            "profit concentration", False,
            f"only {i.winning_trades or 0} winning trade(s) "
            f"(need >={MIN_WINNERS_FOR_CONCENTRATION}) - with this few, one trade dominating "
            "the profit is arithmetic rather than concentration",
            evidence_sufficient=False,
        )
    passed = i.best_trade_share_of_profit <= MAX_SINGLE_TRADE_PROFIT_SHARE
    return Criterion(
        "profit concentration", passed,
        f"best trade is {i.best_trade_share_of_profit:.0%} of gross profit "
        f"(limit {MAX_SINGLE_TRADE_PROFIT_SHARE:.0%})"
        + ("" if passed else " - the edge rests on one trade"),
    )


def _monte_carlo(i: ValidationInputs) -> Criterion:
    if i.monte_carlo_p95_drawdown_pct is None:
        return Criterion("monte carlo drawdown", False, "no resampling run yet",
                         evidence_sufficient=False)
    from app.analysis.monte_carlo import MIN_TRADES_FOR_MONTE_CARLO

    if (i.monte_carlo_sample_size or 0) < MIN_TRADES_FOR_MONTE_CARLO:
        return Criterion(
            "monte carlo drawdown", False,
            f"resampled from only {i.monte_carlo_sample_size or 0} trades "
            f"(need >={MIN_TRADES_FOR_MONTE_CARLO}) - the distribution is arithmetic, not evidence",
            evidence_sufficient=False,
        )
    passed = i.monte_carlo_p95_drawdown_pct <= MAX_MONTE_CARLO_P95_DRAWDOWN_PCT
    return Criterion(
        "monte carlo drawdown", passed,
        f"95th-percentile resampled drawdown {i.monte_carlo_p95_drawdown_pct:.1f}% "
        f"(limit {MAX_MONTE_CARLO_P95_DRAWDOWN_PCT:.0f}%) - this, not the realized figure, "
        "is the ride to plan for",
    )


def _out_of_sample(i: ValidationInputs) -> Criterion:
    if i.out_of_sample_trades is None or i.out_of_sample_profitable is None:
        return Criterion("out-of-sample", False, "no out-of-sample test run yet",
                         evidence_sufficient=False)
    enough = i.out_of_sample_trades >= MIN_OUT_OF_SAMPLE_TRADES
    if not enough:
        return Criterion(
            "out-of-sample", False,
            f"only {i.out_of_sample_trades} out-of-sample trades "
            f"(need >={MIN_OUT_OF_SAMPLE_TRADES})",
            evidence_sufficient=False,
        )
    return Criterion(
        "out-of-sample", i.out_of_sample_profitable,
        f"{i.out_of_sample_trades} trades on unseen data, "
        + ("profitable" if i.out_of_sample_profitable else "NOT profitable - the in-sample result was curve fit"),
    )


def _walk_forward(i: ValidationInputs) -> Criterion:
    if i.walk_forward_windows is None or i.walk_forward_profitable_windows is None:
        return Criterion("walk-forward", False, "no walk-forward analysis run yet",
                         evidence_sufficient=False)
    if i.walk_forward_windows < MIN_WALK_FORWARD_WINDOWS:
        return Criterion(
            "walk-forward", False,
            f"only {i.walk_forward_windows} window(s) (need >={MIN_WALK_FORWARD_WINDOWS})",
            evidence_sufficient=False,
        )
    share = i.walk_forward_profitable_windows / i.walk_forward_windows
    passed = share >= MIN_WALK_FORWARD_PROFITABLE_SHARE
    return Criterion(
        "walk-forward", passed,
        f"{i.walk_forward_profitable_windows}/{i.walk_forward_windows} windows profitable "
        f"({share:.0%}, need >={MIN_WALK_FORWARD_PROFITABLE_SHARE:.0%})",
    )
