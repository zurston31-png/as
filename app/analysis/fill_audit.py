"""Did the paper fills actually charge what they claim to charge?

app/execution/fill_model.py models impact, spread, fees, confirmation delay
and revert-on-slippage, and tests cover each piece. That is not the same as
knowing the fills a RUN produced were realistic: a misconfiguration, a
missing market snapshot on every call, or a settings change mid-run can all
leave the model intact while the recorded trades were charged nothing.

This reads the trades that actually happened and says what they were
charged. It is the check that has to pass before any performance number is
worth quoting, because every downstream statistic inherits an optimistic
fill and none of them can detect one.

WHAT COUNTS AS SUSPICIOUS

A fill priced at or better than the reference is not impossible - drift
during confirmation runs both ways, and a favourable move can outrun the
costs. It is suspicious in BULK. `total_cost = impact + spread + fee +
adverse_drift` with the first three non-negative, so most fills must cost
something; a run where they mostly do not is a run where the costs are not
being applied.

Nothing here changes a fill. It reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import models
from app.config import settings

# Share of fills priced at or better than the reference above which the
# model is not plausibly being applied. Favourable drift is real but it is
# one term against three that always cost, so it should not win often.
MAX_FAVOURABLE_SHARE_PCT = 20.0

# Below this many fills, report the numbers and refuse the verdict. Two
# profitable trades are two observations, and "the fills look fine" from
# two is not a finding.
MIN_FILLS_FOR_VERDICT = 20

# Fills needed before a mean BELOW the spread+fee floor is worth calling
# out. Deliberately far smaller than MIN_FILLS_FOR_VERDICT, because that
# gate exists for statistical claims and this is not one.
#
# A fill's cost is impact + spread + fee + drift. The first three are
# non-negative, so only drift can push a fill under the floor - and drift
# is signed noise that averages toward zero. One or two fills below the
# floor is ordinary luck; a sustained MEAN below it is arithmetic that the
# model cannot produce, and it means the recorded costs did not come from
# the fill model operating normally.
#
# Reporting that only after 20 fills would withhold the most actionable
# thing the audit knows for as long as the book is small - which is
# exactly when a broken cost pipeline is cheapest to notice.
MIN_FILLS_FOR_FLOOR_CHECK = 3

# How far under the floor the mean has to sit before it is worth calling
# out: at least this many times below.
#
# "Below the floor" alone is far too eager. A healthy book of 20 fills
# where 17 pay the floor exactly and 3 caught favourable drift averages
# about 0.33% against a 0.40% floor - genuinely fine, and the first
# version of this check flagged it. Drift routinely drags the mean a
# little under; it cannot drag it to a fraction of the floor and keep it
# there. 2x is the gap between "a few lucky fills" and "these numbers did
# not come from the fill model".
FLOOR_BREACH_RATIO = 2.0


@dataclass
class FillRecord:
    trade_id: int
    symbol: str
    side: str
    execution_cost_pct: float | None
    fee_usd: float | None
    fill_delay_seconds: float | None
    notional_usd: float | None

    @property
    def cost_recorded(self) -> bool:
        return self.execution_cost_pct is not None

    @property
    def favourable(self) -> bool:
        """Filled at or better than the reference price."""
        return self.execution_cost_pct is not None and self.execution_cost_pct <= 0

    def problems(self) -> list[str]:
        out = []
        if self.execution_cost_pct is None:
            out.append("no execution cost recorded - the fill was not costed at all")
        if self.fee_usd is None:
            out.append("no fee recorded")
        elif self.fee_usd <= 0 and (self.notional_usd or 0) > 0:
            out.append("zero fee on a non-zero notional")
        if self.fill_delay_seconds is None:
            out.append("no confirmation delay recorded")
        return out


@dataclass
class FillAudit:
    fills: list[FillRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.fills)

    @property
    def costed(self) -> list[FillRecord]:
        return [f for f in self.fills if f.cost_recorded]

    @property
    def favourable(self) -> list[FillRecord]:
        return [f for f in self.fills if f.favourable]

    @property
    def favourable_share_pct(self) -> float | None:
        costed = self.costed
        return (len(self.favourable) / len(costed) * 100) if costed else None

    @property
    def mean_cost_pct(self) -> float | None:
        """Execution cost as a share of the notional it was charged on.

        Weighted by notional, not a flat mean over fills. A cost RATE
        averaged per-fill lets a $10 leg and a $10,000 leg vote equally,
        so a handful of tiny expensive fills can dominate the number that
        is supposed to say what the book paid. Same defect this file's
        sibling in app/analysis/postmortem.py carried.

        Fills without a positive notional are excluded rather than
        weighted at zero: a rate with no size behind it cannot be
        weighted at all, and including it as zero-weight while counting
        it in the denominator would drag the answer toward nothing.
        """
        weighted = [
            f for f in self.costed if f.notional_usd and f.notional_usd > 0
        ]
        if not weighted:
            return None
        notional = sum(f.notional_usd for f in weighted)
        return sum(
            f.execution_cost_pct * f.notional_usd for f in weighted
        ) / notional * 100

    @property
    def mean_delay_seconds(self) -> float | None:
        delays = [f.fill_delay_seconds for f in self.fills if f.fill_delay_seconds is not None]
        return sum(delays) / len(delays) if delays else None

    @property
    def uncosted(self) -> list[FillRecord]:
        return [f for f in self.fills if not f.cost_recorded]

    @property
    def floor_pct(self) -> float:
        """The cost a fill pays before impact or drift: spread plus fee."""
        return (settings.PAPER_SPREAD_PCT + settings.PAPER_FEE_PCT) * 100

    def verdict(self) -> str:
        if not self.n:
            return "NO FILLS - nothing has been executed yet, so there is nothing to audit."
        if self.uncosted:
            return (
                f"BROKEN - {len(self.uncosted)} of {self.n} fills carry no execution cost. "
                f"Those trades were free, and every performance number computed from them "
                f"is overstated by whatever the fill should have cost."
            )
        share = self.favourable_share_pct or 0.0

        # Checked BEFORE the sample-size gate on purpose: this is an
        # arithmetic impossibility, not a statistical inference, so it
        # does not need a sample to be worth saying.
        # Skipped when the favourable share is already elevated: a book
        # that mostly beats the reference price is better described by
        # SUSPICIOUS below, which names the mechanism instead of the
        # symptom.
        mean = self.mean_cost_pct
        breached = (
            mean is not None
            and (mean <= 0 or self.floor_pct / mean >= FLOOR_BREACH_RATIO)
        )
        if (
            breached
            and share <= MAX_FAVOURABLE_SHARE_PCT
            and len(self.costed) >= MIN_FILLS_FOR_FLOOR_CHECK
        ):
            ratio = (self.floor_pct / mean) if mean > 0 else None
            comparison = f"{ratio:.0f}x below" if ratio else "below"
            return (
                f"BELOW FLOOR - mean fill cost {mean:+.3f}% is {comparison} the "
                f"{self.floor_pct:.3f}% spread+fee floor across {len(self.costed)} costed "
                f"fill(s). Impact, spread and fee are all non-negative, so only "
                f"confirmation drift can go under the floor, and drift averages toward "
                f"zero. A sustained mean below it means these costs did not come from "
                f"the fill model running normally - check that market snapshots are "
                f"reaching it, and that the recorded costs are not stale rows from "
                f"before cost accounting existed."
            )

        if self.n < MIN_FILLS_FOR_VERDICT:
            return (
                f"INSUFFICIENT DATA - {self.n} fills, need {MIN_FILLS_FOR_VERDICT} before the "
                f"distribution means anything. Every fill was costed (mean "
                f"{self.mean_cost_pct:+.2f}%), which is the most that can be said."
            )
        if share > MAX_FAVOURABLE_SHARE_PCT:
            return (
                f"SUSPICIOUS - {share:.0f}% of fills were priced at or better than the "
                f"reference, above the {MAX_FAVOURABLE_SHARE_PCT:.0f}% ceiling. Impact, spread "
                f"and fees are all non-negative, so most fills should cost something. Check "
                f"that market snapshots are reaching the fill model."
            )
        return (
            f"PLAUSIBLE - {self.n} fills, mean cost {self.mean_cost_pct:+.2f}% against a "
            f"{self.floor_pct:.2f}% spread+fee floor, {share:.0f}% favourable."
        )

    def as_dict(self) -> dict:
        return {
            "fills": self.n,
            "costed": len(self.costed),
            "uncosted": len(self.uncosted),
            "favourable": len(self.favourable),
            "favourable_share_pct": (
                round(self.favourable_share_pct, 1)
                if self.favourable_share_pct is not None else None
            ),
            "mean_cost_pct": round(self.mean_cost_pct, 3) if self.mean_cost_pct is not None else None,
            "spread_plus_fee_floor_pct": round(self.floor_pct, 3),
            "mean_delay_seconds": (
                round(self.mean_delay_seconds, 2)
                if self.mean_delay_seconds is not None else None
            ),
            "min_fills_for_verdict": MIN_FILLS_FOR_VERDICT,
            "verdict": self.verdict(),
        }


def build_fill_audit(db: Session, *, mode: str = "paper") -> FillAudit:
    """Every recorded fill and what it was charged."""
    rows = (
        db.query(models.Trade)
        .filter(models.Trade.status == "filled", models.Trade.mode == mode)
        .order_by(models.Trade.id)
        .all()
    )
    return FillAudit(fills=[
        FillRecord(
            trade_id=t.id,
            symbol=t.symbol,
            side=t.side,
            execution_cost_pct=t.execution_cost_pct,
            fee_usd=t.fee_usd,
            fill_delay_seconds=t.fill_delay_seconds,
            notional_usd=t.size_usd,
        )
        for t in rows
    ])
