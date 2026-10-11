"""Execution costs, itemized -- never netted into one number.

ARCHITECTURE.md section 9 names four cost terms and section 12 says why they
stay separate: cost drag must be "measurable rather than buried". A single
`costs` figure cannot answer "how much of the edge did the spread eat", which
is the question that decides whether a 1R scalp is viable at all. So every
term lands in its own `CostBreakdown` field and nothing is pre-netted.

## The one convention everything here depends on

Costs are charged as DOLLARS, and the fill PRICE handed to `TradeRecord` is
the unadjusted reference price from the price path (a bar open, a stop level,
a target). Spread and slippage are not folded into the price.

That is forced by the contract, not a preference. `TradeRecord` independently
validates

    gross_pnl == side.sign * (exit_price - entry_price) * point_value * filled_size
    pnl       == gross_pnl - costs.total

If a slipped price went into `entry_price` *and* the slippage went into
`CostBreakdown.entry_slippage`, every trade would pay its spread and slippage
twice -- and the record's own validator could not see it, because both
equations would still hold. The inverse error is worse: folding costs into the
price and leaving `CostBreakdown` empty satisfies the validator too, and
quietly makes section 12's cost-drag breakdown report zero.

`effective_price` is still computed and returned (see `adverse_price`) because
it is the number a human sanity-checks a fill against. It is a DIAGNOSTIC. It
is never what goes into `TradeRecord.entry_price`.

## Direction

Every term here is non-negative and is charged against the trade. There is no
code path on which a cost becomes a credit, and two places needed explicit
care because the arithmetic offers one:

* a crossed or zero quote (`ask <= bid`) yields a non-positive spread, which
  as a "cost" would pay the trade to trade. Such a quote is rejected and the
  configured fallback is used instead.
* the volatility slippage term is `coefficient * (ATR / median ATR - 1)`,
  which is NEGATIVE in below-median volatility. It is clamped at zero: calm
  markets are cheaper than the model's base, not cheaper than free.

## Where a quantity cannot be measured

`vol_slippage_coefficient` needs `ATR / median ATR`. When that ratio was never
measured, this module does not substitute 1.0 (the value that makes the term
vanish). It REFUSES to price the fill -- `SlippageEstimate.measurable` is
False and `backtest/execution.py` turns the refusal into a missed trade with a
stated reason. `signals/engine.py` carries the scar this rule comes from: a
missing ATR there "does not degrade the cap, it REMOVES it", and a 20x-ATR
stop was accepted by a setup that caps the stop at 2x. The analogous mistake
here is a free fill in unmeasured volatility, and it is available on exactly
the bars where volatility is most likely to be extreme.

The one exception is exact and declared: when `vol_slippage_coefficient` is
0.0 the ratio is not an input to anything, so its absence cannot flatter a
result and no refusal is raised.
"""

from __future__ import annotations

import math

from pydantic import Field, model_validator

from flow_model.core.contracts import CostBreakdown, QuoteSnapshot
from flow_model.core.enums import StrEnum
from flow_model.core.instruments import InstrumentSpec
from flow_model.core.model import FrozenModel

__all__ = [
    "CostModel",
    "CostModelError",
    "CostRefusal",
    "SpreadEstimate",
    "SpreadSource",
    "SlippageEstimate",
    "VolatilityContext",
    "adverse_price",
    "combine_costs",
    "vol_slippage_ticks",
]


class CostModelError(RuntimeError):
    """Raised when a caller violates the cost model's declared contract.

    A quote from the future, a negative size, a spec/quote symbol mismatch.
    Missing or declining DATA never raises -- it produces an unmeasurable
    estimate with a reason, exactly as the signal layer produces a WAIT.
    """


class CostRefusal(StrEnum):
    """Why a fill could not be priced. Strings so the journal can group them."""

    VOL_MULTIPLE_UNMEASURABLE = "vol_multiple_unmeasurable"


class SpreadSource(StrEnum):
    """Which of section 9's three spread sources was actually used.

    Recorded per fill because "bid/ask from quotes if present, else
    `typical_spread_ticks`" means a run's spread cost is a blend of a
    measurement and an assumption, and the proportion decides how much of the
    cost total is evidence.
    """

    QUOTE = "quote"
    OVERRIDE = "spread_ticks_override"
    TYPICAL = "typical_spread_ticks"


class VolatilityContext(FrozenModel):
    """The volatility measurement the slippage term needs, or its absence.

    `ExecutionConfig.vol_slippage_coefficient` documents its own units:
    "Extra slippage ticks per 1.0 of (ATR / median ATR - 1)". That ratio is
    the only volatility input this module takes.

    A caller may supply `vol_multiple` directly or `atr` and `median_atr` and
    let this type form the ratio. `None` everywhere means NOT MEASURED, which
    is a different statement from a ratio of 1.0, and the two are not allowed
    to collapse into each other -- that collapse is the bug this type exists
    to prevent.
    """

    atr: float | None = None
    median_atr: float | None = None
    vol_multiple: float | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "VolatilityContext":
        for name in ("atr", "median_atr", "vol_multiple"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or None, got {value!r}")
        return self

    @classmethod
    def unmeasured(cls, note: str = "") -> "VolatilityContext":
        """An explicit absence. Named so a call site cannot pass it by accident."""
        return cls(notes=(note,) if note else ())

    @property
    def multiple(self) -> float | None:
        """ATR / median ATR, or None when it was never measured.

        A supplied `vol_multiple` wins over the pair, because a caller that
        holds the ratio holds it from the same window that produced it.
        """
        if self.vol_multiple is not None:
            return self.vol_multiple if self.vol_multiple > 0.0 else None
        if self.atr is None or self.median_atr is None:
            return None
        if self.atr <= 0.0 or self.median_atr <= 0.0:
            # A non-positive ATR is not a calm market, it is an unset field or
            # a flat window. `features/volatility.py` treats `median_atr <= 0`
            # the same way for the same reason.
            return None
        return self.atr / self.median_atr

    @property
    def is_measured(self) -> bool:
        return self.multiple is not None


class SpreadEstimate(FrozenModel):
    """The spread used for one side of one fill, and where it came from."""

    ticks: float = Field(ge=0.0)
    source: SpreadSource
    quote_age_seconds: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def half_ticks(self) -> float:
        """What one side pays: a market order crosses half the book.

        A round turn therefore pays one full spread, which is the standard
        convention and the conservative reading of a bar's open print -- that
        print may have been at either side of the book and we do not know
        which.
        """
        return self.ticks / 2.0


class SlippageEstimate(FrozenModel):
    """Section 9's slippage, itemized by term, or a refusal to price it.

        base_ticks + vol_ticks + size_ticks + stop_ticks

    Kept term-by-term rather than summed so a sweep of
    `base_slippage_ticks` against `vol_slippage_coefficient` can attribute a
    change in cost drag to the knob that caused it.
    """

    base_ticks: float = Field(ge=0.0, default=0.0)
    vol_ticks: float = Field(ge=0.0, default=0.0)
    size_ticks: float = Field(ge=0.0, default=0.0)
    stop_ticks: float = Field(ge=0.0, default=0.0)
    measurable: bool = True
    refusal: str = ""
    passive: bool = False
    notes: tuple[str, ...] = ()

    @property
    def ticks(self) -> float:
        return round(self.base_ticks + self.vol_ticks + self.size_ticks + self.stop_ticks, 10)

    @model_validator(mode="after")
    def _check(self) -> "SlippageEstimate":
        if self.measurable and self.refusal:
            raise ValueError("a measurable estimate must not carry a refusal reason")
        if not self.measurable:
            if not self.refusal:
                raise ValueError("an unmeasurable estimate must state its refusal reason")
            if self.ticks != 0.0:
                raise ValueError(
                    "an unmeasurable estimate must carry no ticks: a partial "
                    "slippage figure would be charged as if it were the whole cost"
                )
        return self

    @classmethod
    def refuse(cls, reason: str, *notes: str) -> "SlippageEstimate":
        return cls(measurable=False, refusal=reason, notes=tuple(n for n in notes if n))


def vol_slippage_ticks(coefficient: float, vol_multiple: float) -> float:
    """`coefficient * (vol_multiple - 1)`, clamped at zero.

    The clamp is the whole point. At `vol_multiple = 0.6` the unclamped term
    is `-0.2` ticks at the default coefficient, and a negative slippage is a
    rebate for trading in quiet markets. Quiet markets are already cheaper
    than volatile ones here, because the term is zero rather than positive;
    they are not cheaper than `base_slippage_ticks`.
    """
    if coefficient < 0.0:
        raise ValueError(f"vol_slippage_coefficient must be non-negative, got {coefficient}")
    if not math.isfinite(vol_multiple) or vol_multiple <= 0.0:
        raise ValueError(f"vol_multiple must be positive and finite, got {vol_multiple!r}")
    return max(0.0, coefficient * (vol_multiple - 1.0))


def adverse_price(
    reference_price: float, ticks: float, tick_size: float, *, against_sign: int
) -> float:
    """Move `reference_price` by `ticks` in the direction that hurts.

    `against_sign` is +1 when a HIGHER price is worse (buying) and -1 when a
    LOWER price is worse (selling). The result is a DIAGNOSTIC -- see this
    module's header on why the recorded fill price is the reference price and
    the cost is charged in dollars.
    """
    if against_sign not in (1, -1):
        raise ValueError(f"against_sign must be +1 or -1, got {against_sign!r}")
    if ticks < 0.0:
        raise ValueError(f"ticks must be non-negative, got {ticks}")
    return round(reference_price + against_sign * ticks * tick_size, 10)


def combine_costs(*breakdowns: CostBreakdown) -> CostBreakdown:
    """Field-wise sum. Itemization survives the addition."""
    return CostBreakdown(
        commission=round(sum(b.commission for b in breakdowns), 10),
        exchange_fees=round(sum(b.exchange_fees for b in breakdowns), 10),
        entry_slippage=round(sum(b.entry_slippage for b in breakdowns), 10),
        exit_slippage=round(sum(b.exit_slippage for b in breakdowns), 10),
        spread_cost=round(sum(b.spread_cost for b in breakdowns), 10),
    )


class CostModel:
    """Prices one side of one fill. Config and an `InstrumentSpec`, no market data.

    Market data arrives per call (a quote, a volatility measurement), never in
    the constructor, so one instance serves a whole run and cannot accumulate
    state that would make bar N's cost depend on bar N-1's.
    """

    __slots__ = ("execution", "spec")

    def __init__(self, execution, spec: InstrumentSpec) -> None:
        """`execution` is a `config.schema.ExecutionConfig`."""
        if spec.tick_size <= 0 or not math.isfinite(spec.tick_size):
            raise CostModelError(f"{spec.symbol}: tick_size must be positive and finite")
        self.execution = execution
        self.spec = spec

    # --- spread --------------------------------------------------------

    def spread(
        self, quote: QuoteSnapshot | None = None, *, fill_ts=None
    ) -> SpreadEstimate:
        """Section 9: "bid/ask from quotes if present, else typical_spread_ticks".

        Precedence is override, then quote, then the spec's typical. The
        override outranks a real quote because that is what the field's name
        says it does; it exists so a stress run can price a whole backtest at
        a spread it chooses, and a quote silently beating it would make the
        stress run report costs nobody asked for.

        A quote dated after `fill_ts` is a lookahead bug in the caller and
        RAISES. A quote that is crossed, zero-width, non-finite or tighter
        than the instrument's `min_spread_ticks` is a data condition, not a
        bug: it is rejected, the fallback is used, and the rejection is noted.
        """
        notes: list[str] = []
        tick = self.spec.tick_size

        override = self.execution.spread_ticks_override
        if override is not None:
            ticks = float(override)
            if ticks < self.spec.min_spread_ticks:
                notes.append(
                    f"spread_ticks_override {ticks:g} is below {self.spec.symbol}'s "
                    f"min_spread_ticks {self.spec.min_spread_ticks:g}; floored to the "
                    "minimum, because a spread the instrument cannot quote is a "
                    "discount on a price that does not exist"
                )
                ticks = float(self.spec.min_spread_ticks)
            return SpreadEstimate(
                ticks=ticks, source=SpreadSource.OVERRIDE, notes=tuple(notes)
            )

        age: float | None = None
        if quote is not None and self.execution.use_quotes_when_available:
            if quote.symbol != self.spec.symbol:
                raise CostModelError(
                    f"quote is for {quote.symbol!r} but the spec is {self.spec.symbol!r}"
                )
            if fill_ts is not None:
                age = (fill_ts - quote.ts).total_seconds()
                if age < 0:
                    raise CostModelError(
                        f"{self.spec.symbol}: quote at {quote.ts.isoformat()} postdates the "
                        f"fill at {fill_ts.isoformat()}. A fill may not be priced off a "
                        "quote that had not happened yet"
                    )
            spread_price = quote.ask - quote.bid
            if not math.isfinite(spread_price) or spread_price <= 0.0:
                notes.append(
                    f"quote at {quote.ts.isoformat()} has ask {quote.ask:g} <= bid "
                    f"{quote.bid:g}; a non-positive spread would be a rebate, so the "
                    "quote is rejected and the configured fallback is used"
                )
            else:
                ticks = spread_price / tick
                if ticks < self.spec.min_spread_ticks:
                    notes.append(
                        f"quoted spread {ticks:g} ticks is below min_spread_ticks "
                        f"{self.spec.min_spread_ticks:g}; floored to the minimum"
                    )
                    ticks = float(self.spec.min_spread_ticks)
                return SpreadEstimate(
                    ticks=ticks,
                    source=SpreadSource.QUOTE,
                    quote_age_seconds=age,
                    notes=tuple(notes),
                )
        elif quote is not None:
            notes.append(
                "a quote was supplied but use_quotes_when_available is False, "
                "so typical_spread_ticks is used instead"
            )

        return SpreadEstimate(
            ticks=float(self.spec.typical_spread_ticks),
            source=SpreadSource.TYPICAL,
            quote_age_seconds=age,
            notes=tuple(notes),
        )

    # --- slippage ------------------------------------------------------

    def slippage(
        self,
        *,
        size: float,
        vol: VolatilityContext,
        is_stop_order: bool = False,
        passive: bool = False,
    ) -> SlippageEstimate:
        """Section 9's slippage for one side of one fill.

            base_slippage_ticks
          + vol_slippage_coefficient * max(0, ATR/median ATR - 1)
          + size_impact_ticks_per_unit * size
          + stop_order_extra_slippage_ticks   (stop-market orders only)

        `passive=True` is a RESTING LIMIT that was filled. A limit order
        cannot fill worse than its price -- that is what makes it a limit --
        so charging it slippage would book a cost the market cannot impose.
        Its real cost is miss risk and adverse selection, and those are
        modelled where they belong: in `backtest/execution.py`'s
        trade-through requirement and `max_entry_wait_bars`, not here.

        Returns an UNMEASURABLE estimate rather than a cheap one when the
        volatility term is needed and was not measured.
        """
        if size < 0.0 or not math.isfinite(size):
            raise CostModelError(f"size must be non-negative and finite, got {size!r}")
        if passive:
            if is_stop_order:
                raise CostModelError(
                    "a stop order is a market order by construction and cannot be passive"
                )
            return SlippageEstimate(
                passive=True,
                notes=(
                    "resting limit fill: a limit cannot fill worse than its price, so no "
                    "slippage is charged. Its cost is miss risk, modelled by the "
                    "trade-through requirement and max_entry_wait_bars",
                ),
            )

        notes: list[str] = []
        coefficient = float(self.execution.vol_slippage_coefficient)
        vol_ticks = 0.0
        if coefficient > 0.0:
            multiple = vol.multiple
            if multiple is None:
                return SlippageEstimate.refuse(
                    CostRefusal.VOL_MULTIPLE_UNMEASURABLE,
                    f"vol_slippage_coefficient is {coefficient:g}, so the fill's cost "
                    "depends on ATR / median ATR, and that ratio was not measured. "
                    "No value is substituted: treating it as 1.0 would delete the "
                    "volatility term exactly on the bars where volatility is most "
                    "likely to be extreme, and the deletion would be invisible",
                    *vol.notes,
                )
            vol_ticks = vol_slippage_ticks(coefficient, multiple)
        else:
            notes.append(
                "vol_slippage_coefficient is 0.0, so ATR / median ATR is not an input "
                "and its absence cannot change this fill's cost"
            )

        size_ticks = float(self.execution.size_impact_ticks_per_unit) * size
        stop_ticks = (
            float(self.execution.stop_order_extra_slippage_ticks) if is_stop_order else 0.0
        )
        return SlippageEstimate(
            base_ticks=float(self.execution.base_slippage_ticks),
            vol_ticks=round(vol_ticks, 10),
            size_ticks=round(size_ticks, 10),
            stop_ticks=stop_ticks,
            notes=tuple(notes) + vol.notes,
        )

    # --- commissions and fees -----------------------------------------

    def commission_and_fees(self, size: float, *, sides: int = 1) -> tuple[float, float]:
        """`(commission, exchange_fees)` in dollars, itemized.

        `InstrumentSpec.commission_for` returns the two SUMMED, which cannot
        be split back apart, and section 9 lists "commissions + exchange fees"
        as two terms. They are computed separately here with the same
        `commission_per_unit` rule the spec applies, and the smoke check
        asserts the two agree -- the spec's docstring promises "there is no
        possibility of two modules disagreeing about what NQ is worth", and
        a second implementation of the same arithmetic is exactly how that
        promise gets broken.
        """
        if sides < 1:
            raise CostModelError(f"sides must be >= 1, got {sides}")
        if size < 0.0 or not math.isfinite(size):
            raise CostModelError(f"size must be non-negative and finite, got {size!r}")
        multiplier = size * sides if self.spec.commission_per_unit else float(sides)
        return (
            round(self.spec.commission_per_side * multiplier, 10),
            round(self.spec.exchange_fee_per_side * multiplier, 10),
        )

    # --- dollar conversion --------------------------------------------

    def ticks_to_dollars(self, ticks: float, size: float) -> float:
        """`ticks * tick_value * size`.

        Identical to `ticks * tick_size * point_value * size`; written in
        terms of `tick_value` because that is the field the spec actually
        carries and the other form divides and re-multiplies by `tick_size`.
        """
        return round(ticks * self.spec.tick_value * size, 10)

    # --- assembled breakdowns -----------------------------------------

    def entry_costs(
        self, *, size: float, spread: SpreadEstimate, slippage: SlippageEstimate
    ) -> CostBreakdown:
        """Costs of getting in, itemized. One side's commission and fees."""
        return self._side_costs(size=size, spread=spread, slippage=slippage, entry=True)

    def exit_costs(
        self, *, size: float, spread: SpreadEstimate, slippage: SlippageEstimate
    ) -> CostBreakdown:
        """Costs of getting out, itemized. One side's commission and fees."""
        return self._side_costs(size=size, spread=spread, slippage=slippage, entry=False)

    def _side_costs(
        self,
        *,
        size: float,
        spread: SpreadEstimate,
        slippage: SlippageEstimate,
        entry: bool,
    ) -> CostBreakdown:
        if not slippage.measurable:
            raise CostModelError(
                f"refusing to assemble costs from an unmeasurable slippage estimate "
                f"({slippage.refusal}). The caller must decline the fill instead; "
                "pricing it at zero slippage is the flattering reading of a "
                "measurement that was never taken"
            )
        commission, fees = self.commission_and_fees(size, sides=1)
        slip_dollars = self.ticks_to_dollars(slippage.ticks, size)
        # A passive fill crosses no spread: it IS the resting side of the book.
        spread_ticks = 0.0 if slippage.passive else spread.half_ticks
        spread_dollars = self.ticks_to_dollars(spread_ticks, size)
        return CostBreakdown(
            commission=commission,
            exchange_fees=fees,
            entry_slippage=slip_dollars if entry else 0.0,
            exit_slippage=0.0 if entry else slip_dollars,
            spread_cost=spread_dollars,
        )
