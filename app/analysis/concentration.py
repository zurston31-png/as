"""How few trades the record's profit actually rests on.

The validation gate already asks whether one TRADE produced most of the
profit. That question has a blind spot, and this record walked straight
into it: at 51 closed trades the best single trade was 12% of gross
profit - a comfortable pass - while four partial profit-takes produced
+$43.74 and the other 47 trades produced -$12.40 between them. No single
trade dominated. Four of them did, and nothing measured that.

TWO MEASURES, AND WHY NOT A THIRD

  FRAGILITY. The smallest number of best trades whose removal takes the
  record from profitable to not. It is scale-free, it needs no opinion
  about which exit rule "should" earn the money, and it answers the
  question a reader actually has: how much of this is a handful of lucky
  draws? A record that survives losing its best 10% is standing on its
  whole sample; one that flips on three trades out of fifty is standing
  on three trades.

  LOAD-BEARING RULE. Which exit rule contributes the most P&L, how often
  it fires, and what the book looks like without it. Reported, never
  gated - see below.

The third measure, the obvious one, is deliberately absent: "no single
exit rule may exceed X% of gross profit" is useless. Gross profit sums
only the winners, so whichever rule closes winning trades is ~100% of it
by construction, in every strategy that has ever worked. The same trap
catches "is the remainder still positive after removing the best rule?" -
strip the profitable exit from any book and what is left is the stop
losses. Both would fail a healthy strategy as readily as a fragile one,
which makes them worse than no measure at all.

WHY FRAGILITY IS REPORTED BUT DOES NOT GATE

`MIN_FRAGILITY_SHARE` is a judgement, not a derived bound. Sample size
comes from a confidence interval, expectancy from arithmetic, profit
factor from estimator noise - each of those thresholds can be argued from
first principles. "Survive losing the best 10%" cannot; it is a
reasonable-sounding number, and a reasonable-sounding number that silently
decides whether a strategy reads as VALIDATED does not belong in the gate.
So the criterion is non-blocking: it shows its verdict and its arithmetic,
and leaves the decision to a person.

Setting the bar so that today's record fails it would be the same sin as
lowering a bar so a record passes, run backwards.
"""
from __future__ import annotations

from dataclasses import dataclass

from app import models
from app.analysis import trade_analytics as ta

# The share of its best trades a record should be able to lose and still
# be profitable. Advisory - see the module docstring for why this one does
# not gate.
MIN_FRAGILITY_SHARE = 0.10


@dataclass(frozen=True)
class RuleContribution:
    """One exit rule's contribution to the record."""

    rule: str
    trade_count: int
    total_pnl_usd: float

    def share_of_trades(self, total: int) -> float:
        return self.trade_count / total if total else 0.0


@dataclass(frozen=True)
class ConcentrationReport:
    closed_trades: int
    net_pnl_usd: float
    winning_trades: int

    # None when the question does not apply: an unprofitable record has no
    # "trades to flip it negative", and saying 0 would read as maximal
    # fragility rather than as not-applicable.
    trades_to_flip: int | None
    load_bearing: RuleContribution | None
    remainder_pnl_usd: float | None
    remainder_trades: int | None

    @property
    def flip_share(self) -> float | None:
        if self.trades_to_flip is None or not self.closed_trades:
            return None
        return self.trades_to_flip / self.closed_trades

    @property
    def survives_bar(self) -> bool | None:
        share = self.flip_share
        return None if share is None else share >= MIN_FRAGILITY_SHARE

    def as_dict(self) -> dict:
        return {
            "closed_trades": self.closed_trades,
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "trades_to_flip": self.trades_to_flip,
            "flip_share": round(self.flip_share, 4) if self.flip_share is not None else None,
            "survives_bar": self.survives_bar,
            "bar": MIN_FRAGILITY_SHARE,
            "load_bearing_rule": (
                {
                    "rule": self.load_bearing.rule,
                    "trade_count": self.load_bearing.trade_count,
                    "total_pnl_usd": round(self.load_bearing.total_pnl_usd, 4),
                }
                if self.load_bearing
                else None
            ),
            "remainder_pnl_usd": (
                round(self.remainder_pnl_usd, 4) if self.remainder_pnl_usd is not None else None
            ),
            "remainder_trades": self.remainder_trades,
        }


def _trades_to_flip(pnls: list[float], net: float) -> int | None:
    """Smallest K such that dropping the K best trades leaves net <= 0.

    Only meaningful for a record that is currently profitable; a losing
    record is already where this measure is pointing and returns None.
    """
    if net <= 0:
        return None
    running = net
    for k, pnl in enumerate(sorted(pnls, reverse=True), start=1):
        running -= pnl
        if running <= 0:
            return k
    # Unreachable in practice: removing every trade leaves 0, which is
    # <= 0. Kept explicit so a future change to the loop cannot return
    # None here and have it read as "not applicable".
    return len(pnls) or None


def build(trades: list[models.Trade]) -> ConcentrationReport:
    """Measure how concentrated the record's profit is.

    Works over closed legs, the same population every other figure in the
    performance report is computed from.
    """
    closed = ta.closed_trades(trades)
    pnls = [t.pnl_usd or 0.0 for t in closed]
    net = sum(pnls)
    winners = sum(1 for p in pnls if p > 0)

    load_bearing = None
    remainder_pnl = None
    remainder_trades = None

    breakdown = ta.breakdown_by_exit_reason(trades)
    if breakdown.buckets:
        best = max(breakdown.buckets, key=lambda b: b.total_pnl_usd)
        load_bearing = RuleContribution(
            rule=best.label,
            trade_count=best.trade_count,
            total_pnl_usd=best.total_pnl_usd,
        )
        # Measured against the whole closed record rather than against the
        # breakdown's own total: trades with no close_reason are counted
        # by one and not the other, and the remainder must still describe
        # "the rest of the book".
        remainder_pnl = net - best.total_pnl_usd
        remainder_trades = len(closed) - best.trade_count

    return ConcentrationReport(
        closed_trades=len(closed),
        net_pnl_usd=net,
        winning_trades=winners,
        trades_to_flip=_trades_to_flip(pnls, net),
        load_bearing=load_bearing,
        remainder_pnl_usd=remainder_pnl,
        remainder_trades=remainder_trades,
    )
