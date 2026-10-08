"""Setup selection: a MEASUREMENT read off the structure, not a parameter.

ARCHITECTURE.md section 14.6, verbatim:

    achievable_rr = |next_opposing_zone_price - entry| / |entry - stop|

    achievable_rr < min_reward_risk        -> WAIT("rr_too_low")
    1.0 <= achievable_rr < 1.8             -> SCALP_1R
    1.8 <= achievable_rr < 2.7             -> SETUP_2R
    achievable_rr >= 2.7                   -> DIRECTIONAL_3R

and the sentence that makes it binding: "A setup is no longer assigned by
configuration; it is read off the structure. When the next level is 0.8R away
the trade is simply declined, which is the mechanism that stops a '1R scalp'
from quietly becoming a 0.4R target against a full-width stop."

## What is consumed rather than recomputed

`features/levels.py` already computes all of it: `achievable_rr`,
`level_setup_class` (the band, encoded as the setup's own R multiple),
`level_stop_price` (14.6's determined stop, beyond the level) and
`level_target_price` (the next opposing major zone). This module READS those
keys. The band edges live in `levels.SETUP_RR_BANDS` and the mapping in
`levels.setup_class_for`, and both are imported rather than restated, so
there is one copy of 14.6's numbers in the codebase.

`level_setup_class` is consumed as the verdict, and `setup_class_for` is
re-evaluated on the same inputs purely as a consistency assertion. That is
not a second source of truth -- it is the same pure function on the same
arguments -- and it turns a feature-layer regression into a loud
`SetupError` instead of a silently mislabelled trade.

## The band floor and the setup's own target are different numbers

The band that selects `DIRECTIONAL_3R` starts at 2.7, while the setup's
`reward_risk` is 3.0 and `TradeRecord.r_label_is_honest()` requires a
`DIRECTIONAL_3R` to plan at least 3R. So a bar whose measured
`achievable_rr` is 2.8 selects `DIRECTIONAL_3R`, and the target that setup
plans sits slightly BEYOND the opposing major zone the measurement came
from. The same holds in [1.8, 2.0) for `SETUP_2R`. This is a real tension
between 14.6's band edges and section 0.2's label rule, it is NOT resolved
here by moving either number, and both quantities are carried on
`SetupSelection` (`achievable_rr` and the setup's `reward_risk`) so the gap
is visible in every record rather than reconciled away. `reasons` says so in
words whenever the planned R exceeds the measured achievable R.

The alternative -- planning the target at the measured `achievable_rr` --
was rejected because it relabels: a trade planned at 2.8R and recorded as
`DIRECTIONAL_3R` fails `r_label_is_honest`, which is precisely the check
section 0.2 exists to pass.

## Promotion and truncation are both refused

The brief: "Do not force every trade into the same target." The band
determines exactly ONE setup. If that setup is disabled, or fails one of its
own gates, the bar is declined -- it is never handed to the next setup down
(promotion) or up (truncation). `select` therefore evaluates the determined
setup and no other, and `rejections` records what every enabled setup would
have said so rejection analysis can see the near-misses without the selector
acting on them.

## Which `WaitReason` each failure carries

Section 3 already assigns a reason to each of these comparisons, and those
assignments are used rather than collapsed:

| failure                                   | reason                 |
|-------------------------------------------|------------------------|
| `achievable_rr` below a reward/risk floor | `RR_TOO_LOW`           |
| flow score below the setup's minimum      | `SCORE_BELOW_THRESHOLD`|
| ATR percentile outside the setup's band   | `VOLATILITY_BAND`      |
| order-flow confirmation required, absent  | `COMPONENT_DISABLED`   |
| order-flow confirmation required, refused | `ORDERFLOW_CONFLICT`   |
| structure confirmation required, refused  | `NO_STRUCTURE`         |
| the determined setup is absent/disabled   | `NO_SETUP_MATCH`       |
| the determined setup bars this regime     | `NO_SETUP_MATCH`       |

`NO_SETUP_MATCH` is what the brief asks for when "NO setup matches", and the
two rows that carry it are exactly the cases where the structure named a
setup that this configuration does not offer for this bar. The finer reasons
are not inventions: each is the reason section 3 names for that specific
comparison, and reporting `NO_SETUP_MATCH` for a score miss would make
`SCORE_BELOW_THRESHOLD` unreachable and the rejection histogram useless.

Nothing here reads `config.research_targets` or the pre-registered
hypotheses, and no statement here asserts that a setup makes money.
"""

from __future__ import annotations

import math
from typing import Mapping

from pydantic import computed_field, model_validator

from flow_model.config.schema import FlowModelConfig, SetupConfig
from flow_model.core.contracts import ComponentScore, FeatureVector, RegimeState
from flow_model.core.enums import (
    Component,
    DataQuality,
    Regime,
    SetupType,
    Side,
    WaitReason,
)
from flow_model.core.model import FrozenModel
from flow_model.features.levels import SETUP_RR_BANDS, setup_class_for

__all__ = [
    "SETUP_BY_RR_CLASS",
    "SetupError",
    "SetupRejection",
    "SetupSelection",
    "min_flow_score_required",
    "min_reward_risk_floor",
    "select",
    "summary_lines",
    "tradable_regimes",
    "vol_percentile_band",
]


#: `level_setup_class` -> `SetupType`. `features/levels.py` encodes the class
#: AS the setup's R multiple (1.0 / 2.0 / 3.0) precisely so the encoding
#: cannot drift from the thing it names, and 0.0 means "declined". The keys
#: are taken from `SETUP_RR_BANDS` rather than written out, so adding a band
#: there without a setup here is an import-time error.
SETUP_BY_RR_CLASS: Mapping[float, SetupType] = {
    1.0: SetupType.SCALP_1R,
    2.0: SetupType.SETUP_2R,
    3.0: SetupType.DIRECTIONAL_3R,
}


def _check_band_coverage() -> None:
    """Every band `features/levels.py` can emit must name a setup here.

    Checked at import. A band edge added to 14.6 without a corresponding
    `SetupType` would otherwise surface as a bar silently declined with
    `NO_SETUP_MATCH`, which reads as "the market did not offer this setup"
    rather than "the code does not know this setup".
    """
    encoded = {cls for _, cls in SETUP_RR_BANDS}
    unknown = sorted(encoded - set(SETUP_BY_RR_CLASS))
    if unknown:  # pragma: no cover - guards a future edit to levels.py
        raise SetupError(
            f"features/levels.py emits level_setup_class {unknown}, which "
            f"SETUP_BY_RR_CLASS does not map to a SetupType"
        )
    unused = sorted(set(SETUP_BY_RR_CLASS) - encoded)
    if unused:  # pragma: no cover - guards a future edit to levels.py
        raise SetupError(
            f"SETUP_BY_RR_CLASS maps {unused}, which features/levels.py never "
            "emits; a setup that cannot be selected is dead configuration"
        )


class SetupError(RuntimeError):
    """Raised when the selector's declared contract is violated.

    A feature-layer disagreement (`level_setup_class` inconsistent with
    `achievable_rr`), a band with no setup, or a caller that passed a side
    the structure did not name. Missing or declining DATA never raises -- it
    produces a `SetupSelection` with a `wait_reason`.
    """


_check_band_coverage()


# ---------------------------------------------------------------------------
# "could any setup accept this bar" -- the union helpers the gates use
# ---------------------------------------------------------------------------
#
# Section 3 runs the regime, Flow Score and volatility gates BEFORE setup
# selection, and each of those gates is written in terms of "the setup",
# which is not known yet. The resolution is that the pre-selection gates ask
# the weaker question -- could ANY enabled setup accept this bar -- and the
# selected setup's own thresholds are then applied here. A bar that clears
# the union and fails the determined setup is declined by this module, not by
# the gate, and the reasons above distinguish the two.


def tradable_regimes(config: FlowModelConfig) -> frozenset[Regime]:
    """Regimes at least one enabled setup allows.

    `Regime.UNKNOWN` is never a member: `SetupConfig` refuses it, because it
    means the warmup window is incomplete.
    """
    allowed: set[Regime] = set()
    for setup in config.enabled_setups():
        allowed |= set(setup.allowed_regimes)
    return frozenset(allowed)


def vol_percentile_band(config: FlowModelConfig) -> tuple[float, float]:
    """The union of the enabled setups' ATR-percentile bands.

    The union, not the intersection: the gate's job before selection is to
    reject a bar no setup could take, and the per-setup band is re-applied in
    `select`. Returns `(0.0, 1.0)` when nothing is enabled, which cannot
    happen -- `FlowModelConfig` refuses a config with no enabled setup.
    """
    setups = config.enabled_setups()
    if not setups:  # pragma: no cover - FlowModelConfig refuses this
        return 0.0, 1.0
    return (
        min(float(s.min_vol_percentile) for s in setups),
        max(float(s.max_vol_percentile) for s in setups),
    )


def min_flow_score_required(config: FlowModelConfig) -> float:
    """The lowest `min_flow_score` among enabled setups.

    Section 3's Flow Score gate is `score < setup.min_score`. Before
    selection the only honest version is the lowest such threshold: a score
    below it cannot satisfy any setup.
    """
    setups = config.enabled_setups()
    if not setups:  # pragma: no cover - FlowModelConfig refuses this
        return 0.0
    return min(float(s.min_flow_score) for s in setups)


def min_reward_risk_floor(config: FlowModelConfig) -> float:
    """The reward/risk floor below which no setup is reachable.

    The structure layer's own floor (`config.levels.min_reward_risk`) and the
    lowest per-setup `min_reward_risk`, whichever is HIGHER: both have to be
    cleared, so the binding one is the maximum. `StructureLevelConfig`
    defaults this to 1.0 rather than `SetupConfig`'s 0.9 for exactly this
    reason -- 14.6's bands start at 1.0, so a value in [0.9, 1.0) would clear
    a floor and then match no band.
    """
    setups = config.enabled_setups()
    per_setup = min((float(s.min_reward_risk) for s in setups), default=0.0)
    return max(float(config.levels.min_reward_risk), per_setup)


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------


class SetupRejection(FrozenModel):
    """One setup's first-failing gate, for rejection analysis.

    Recorded for every enabled setup, including the ones the band did not
    select, so a report can answer "how close was this bar to a 2R" without
    the selector ever acting on a setup the structure did not name.
    """

    setup: SetupType
    gate: str
    wait_reason: WaitReason
    detail: str = ""
    selected_by_band: bool = False


class SetupSelection(FrozenModel):
    """The setup the structure determined, or the reason there is none."""

    symbol: str
    setup: SetupType | None = None
    setup_config: SetupConfig | None = None
    achievable_rr: float = 0.0
    structural_stop_price: float | None = None
    structural_target_price: float | None = None
    wait_reason: WaitReason | None = None
    detail: str = ""
    reasons: tuple[str, ...] = ()
    rejections: tuple[SetupRejection, ...] = ()

    @property
    def matched(self) -> bool:
        return self.setup is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def planned_reward_risk(self) -> float:
        """The R multiple the selected setup plans. 0.0 when none matched."""
        return 0.0 if self.setup_config is None else float(self.setup_config.reward_risk)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def plans_beyond_measured_target(self) -> bool:
        """True when the setup's target sits past the measured opposing zone.

        See the module docstring: 14.6's band floors (1.8, 2.7) are below the
        setups' own reward/risk (2.0, 3.0), so this is True for an
        `achievable_rr` inside the lower part of a band. Reported, never
        corrected by relabelling the setup.
        """
        if self.setup_config is None:
            return False
        return self.planned_reward_risk > self.achievable_rr + 1e-9

    @model_validator(mode="after")
    def _check(self) -> "SetupSelection":
        if (self.setup is None) == (self.wait_reason is None):
            raise SetupError(
                "a SetupSelection names a setup or a wait_reason, never both "
                "and never neither"
            )
        if self.setup is not None:
            if self.setup_config is None:
                raise SetupError("a matched selection must carry its SetupConfig")
            if self.setup_config.setup is not self.setup:
                raise SetupError(
                    f"selection names {self.setup.value} but carries the config "
                    f"for {self.setup_config.setup.value}"
                )
        return self


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

#: Feature keys `select` reads. All owned by `features/levels.py` except
#: `atr_percentile` (`features/volatility.py`), which is the quantity section
#: 3's volatility band is written against.
READS: tuple[str, ...] = (
    "achievable_rr",
    "level_setup_class",
    "level_stop_price",
    "level_target_price",
    "atr_percentile",
)


def select(
    config: FlowModelConfig,
    features: FeatureVector,
    *,
    side: Side,
    regime: RegimeState,
    flow_score: float,
    components: Mapping[Component, ComponentScore],
) -> SetupSelection:
    """The setup 14.6's measurement determines, or the reason there is none.

    `side` is the structure's direction (`level_direction`), established by
    the structure gate. It is an argument rather than something re-read here
    because the pipeline has already decided it and a second derivation could
    disagree.

    `flow_score` is the aggregate from `signals/flow_score.py`. It arrives as
    a float rather than a `FlowScore` because the only question asked of it
    is the per-setup `min_flow_score` comparison; the component directions
    arrive separately in `components`, where the confirmation flags read
    them.
    """
    symbol = features.symbol
    missing = tuple(k for k in ("achievable_rr", "level_setup_class") if k not in features.values)
    if missing:
        return SetupSelection(
            symbol=symbol,
            wait_reason=WaitReason.NO_SETUP_MATCH,
            detail=(
                f"feature key(s) {', '.join(missing)} absent from the vector, so "
                "14.6's measurement was never made and no setup can be read off it"
            ),
        )
    if features.quality_of("achievable_rr") is DataQuality.MISSING:
        return SetupSelection(
            symbol=symbol,
            wait_reason=WaitReason.NO_SETUP_MATCH,
            detail=(
                "achievable_rr is graded MISSING: the structure layer made no "
                "measurement, and a setup is a measurement here, not a default"
            ),
        )

    achievable_rr = float(features.require("achievable_rr"))
    stop_price = _optional_price(features, "level_stop_price")
    target_price = _optional_price(features, "level_target_price")
    floor = min_reward_risk_floor(config)

    emitted_class = float(features.require("level_setup_class"))
    expected_class = setup_class_for(achievable_rr, config.levels.min_reward_risk)
    if not math.isclose(emitted_class, expected_class, rel_tol=0.0, abs_tol=1e-9):
        raise SetupError(
            f"features/levels.py emitted level_setup_class={emitted_class} for "
            f"achievable_rr={achievable_rr} against min_reward_risk="
            f"{config.levels.min_reward_risk}, but 14.6's own bands give "
            f"{expected_class}. The band table is the one in levels.SETUP_RR_BANDS "
            "and the two readings must agree; a disagreement means a setup label "
            "no longer matches the measurement it is read off."
        )

    base = dict(
        symbol=symbol,
        achievable_rr=achievable_rr,
        structural_stop_price=stop_price,
        structural_target_price=target_price,
    )

    if not math.isfinite(achievable_rr) or achievable_rr < floor:
        return SetupSelection(
            **base,
            wait_reason=WaitReason.RR_TOO_LOW,
            detail=(
                f"achievable R:R {achievable_rr:.3f} is below the floor {floor:.3f} "
                "(the higher of config.levels.min_reward_risk and the lowest "
                "enabled setup's min_reward_risk). 14.6 declines the trade here "
                "rather than moving the target closer, which is what keeps a '1R "
                "scalp' from becoming a 0.4R target against a full-width stop"
            ),
        )

    determined = SETUP_BY_RR_CLASS.get(emitted_class)
    if determined is None:
        return SetupSelection(
            **base,
            wait_reason=WaitReason.RR_TOO_LOW,
            detail=(
                f"achievable R:R {achievable_rr:.3f} clears the floor {floor:.3f} "
                "but falls in no band of 14.6 (the lowest band starts at "
                f"{min(f for f, _ in SETUP_RR_BANDS):.1f}), so the structure "
                "names no setup. Declined, not promoted to the nearest one"
            ),
        )

    rejections = tuple(
        rejection
        for rejection in (
            _evaluate(
                config.setups.get(setup_type),
                setup_type,
                achievable_rr=achievable_rr,
                side=side,
                regime=regime,
                flow_score=flow_score,
                components=components,
                features=features,
                selected_by_band=setup_type is determined,
            )
            for setup_type in SetupType
        )
        if rejection is not None
    )
    determined_rejection = next(
        (r for r in rejections if r.setup is determined), None
    )
    setup_config = config.setups.get(determined)

    if setup_config is None or not setup_config.enabled:
        why = "is not configured" if setup_config is None else "is disabled in config"
        return SetupSelection(
            **base,
            wait_reason=WaitReason.NO_SETUP_MATCH,
            detail=(
                f"the structure measures achievable R:R {achievable_rr:.3f}, which "
                f"14.6 reads as {determined.value}, and that setup {why}. The bar "
                "is declined: a setup is read off the structure, so it is not "
                "retargeted to a setup the structure did not name"
            ),
            rejections=rejections,
        )

    if determined_rejection is not None:
        return SetupSelection(
            **base,
            wait_reason=determined_rejection.wait_reason,
            detail=(
                f"{determined.value} is the setup 14.6's achievable R:R "
                f"{achievable_rr:.3f} determines, and it fails its own "
                f"{determined_rejection.gate} gate: {determined_rejection.detail}. "
                "Not substituted with another setup"
            ),
            rejections=rejections,
        )

    reasons = [
        f"setup {determined.value}: measured achievable R:R {achievable_rr:.3f} "
        f"falls in 14.6's band for this setup; the setup was read off the "
        f"structure, not chosen",
        f"setup {determined.value}: plans {setup_config.reward_risk:.2f}R against "
        f"the structural stop, max_hold_bars={setup_config.max_hold_bars}",
    ]
    selection = SetupSelection(
        **base,
        setup=determined,
        setup_config=setup_config,
        reasons=tuple(reasons),
        rejections=rejections,
    )
    if selection.plans_beyond_measured_target:
        reasons.append(
            f"setup {determined.value}: the planned target is "
            f"{setup_config.reward_risk:.2f}R while the measured distance to the "
            f"next opposing major zone is {achievable_rr:.3f}R, because 14.6's "
            f"band for this setup starts below its reward_risk. The planned R is "
            "honest about the stop and optimistic about reachability; both "
            "numbers are recorded and neither was adjusted to hide the gap"
        )
        selection = selection.replace(reasons=tuple(reasons))
    return selection


def _optional_price(features: FeatureVector, key: str) -> float | None:
    """A price-valued key, or None when absent, MISSING or non-finite.

    `features/levels.py` repeats the last close in these fields when there is
    no zone rather than carrying a sentinel, so a present value is always a
    real price; what this guards is an absent or ungraded key.
    """
    if key not in features.values:
        return None
    if features.quality_of(key) is DataQuality.MISSING:
        return None
    value = float(features.values[key])
    return value if math.isfinite(value) and value > 0.0 else None


def _evaluate(
    setup_config: SetupConfig | None,
    setup_type: SetupType,
    *,
    achievable_rr: float,
    side: Side,
    regime: RegimeState,
    flow_score: float,
    components: Mapping[Component, ComponentScore],
    features: FeatureVector,
    selected_by_band: bool,
) -> SetupRejection | None:
    """One setup's first-failing gate, or None when it would accept the bar.

    Gate order is fixed and runs cheapest-and-most-structural first, for the
    same reason the pipeline's own order is fixed: a bar blocked by several
    reports one, and it must be the same one every run.
    """
    if setup_config is None:
        return SetupRejection(
            setup=setup_type,
            gate="configured",
            wait_reason=WaitReason.NO_SETUP_MATCH,
            detail="not present in config.setups",
            selected_by_band=selected_by_band,
        )
    if not setup_config.enabled:
        return SetupRejection(
            setup=setup_type,
            gate="enabled",
            wait_reason=WaitReason.NO_SETUP_MATCH,
            detail="disabled in config.setups",
            selected_by_band=selected_by_band,
        )

    def reject(gate: str, reason: WaitReason, detail: str) -> SetupRejection:
        return SetupRejection(
            setup=setup_type,
            gate=gate,
            wait_reason=reason,
            detail=detail,
            selected_by_band=selected_by_band,
        )

    if achievable_rr < float(setup_config.min_reward_risk):
        return reject(
            "min_reward_risk",
            WaitReason.RR_TOO_LOW,
            f"achievable R:R {achievable_rr:.3f} below this setup's minimum "
            f"{setup_config.min_reward_risk:.3f}",
        )
    if regime.regime not in setup_config.allowed_regimes:
        return reject(
            "allowed_regimes",
            WaitReason.NO_SETUP_MATCH,
            f"regime {regime.regime.value} is not in this setup's allowed_regimes "
            f"({', '.join(r.value for r in setup_config.allowed_regimes)})",
        )
    if flow_score < float(setup_config.min_flow_score):
        return reject(
            "min_flow_score",
            WaitReason.SCORE_BELOW_THRESHOLD,
            f"flow score {flow_score:.2f} below this setup's minimum "
            f"{setup_config.min_flow_score:.2f}",
        )

    vol_percentile = _vol_percentile(features)
    if vol_percentile is None:
        return reject(
            "vol_percentile",
            WaitReason.VOLATILITY_BAND,
            "atr_percentile is absent or graded MISSING, so the setup's "
            "volatility band cannot be evaluated and the bar is declined "
            "rather than admitted on an unmeasured quantity",
        )
    if not (
        float(setup_config.min_vol_percentile)
        <= vol_percentile
        <= float(setup_config.max_vol_percentile)
    ):
        return reject(
            "vol_percentile",
            WaitReason.VOLATILITY_BAND,
            f"ATR percentile {vol_percentile:.3f} outside this setup's band "
            f"[{setup_config.min_vol_percentile:.2f}, "
            f"{setup_config.max_vol_percentile:.2f}]",
        )

    if setup_config.require_orderflow_confirmation:
        order_flow = components.get(Component.ORDER_FLOW)
        if order_flow is None or not order_flow.enabled:
            return reject(
                "require_orderflow_confirmation",
                WaitReason.COMPONENT_DISABLED,
                "this setup requires order-flow confirmation and the ORDER_FLOW "
                "component was never measured. Section 5 accepts no proxy for a "
                "tick feed, so there is nothing to confirm with and the trade is "
                "declined rather than taken on an assumed-neutral tape",
            )
        if not order_flow.agrees_with(side):
            return reject(
                "require_orderflow_confirmation",
                WaitReason.ORDERFLOW_CONFLICT,
                f"this setup requires order-flow confirmation and ORDER_FLOW "
                f"direction {order_flow.direction:+d} does not agree with a "
                f"{side.value} ({side.sign:+d})",
            )
    if setup_config.require_structure_confirmation:
        structure = components.get(Component.STRUCTURE)
        if structure is None or not structure.enabled:
            return reject(
                "require_structure_confirmation",
                WaitReason.COMPONENT_DISABLED,
                "this setup requires structure confirmation and the STRUCTURE "
                "component was never measured",
            )
        if not structure.agrees_with(side):
            return reject(
                "require_structure_confirmation",
                WaitReason.NO_STRUCTURE,
                f"this setup requires structure confirmation and STRUCTURE "
                f"direction {structure.direction:+d} does not agree with a "
                f"{side.value} ({side.sign:+d})",
            )
    return None


def _vol_percentile(features: FeatureVector) -> float | None:
    key = "atr_percentile"
    if key not in features.values:
        return None
    if features.quality_of(key) is DataQuality.MISSING:
        return None
    value = float(features.values[key])
    return value if math.isfinite(value) else None


def summary_lines(selection: SetupSelection) -> tuple[str, ...]:
    """Human-readable account of the selection, for a report or a CLI."""
    lines = [f"setup selection for {selection.symbol}"]
    lines.append(f"  achievable R:R (measured): {selection.achievable_rr:.3f}")
    if selection.matched and selection.setup is not None:
        lines.append(f"  selected: {selection.setup.value}")
        lines.append(f"  plans:    {selection.planned_reward_risk:.2f}R")
        if selection.plans_beyond_measured_target:
            lines.append(
                "  NOTE: the planned target is beyond the measured opposing zone "
                "(14.6's band floor is below this setup's reward_risk)"
            )
    else:
        lines.append(
            f"  declined: {selection.wait_reason.value if selection.wait_reason else '?'}"
        )
        lines.append(f"  {selection.detail}")
    for rejection in selection.rejections:
        marker = "<-- band" if rejection.selected_by_band else "        "
        lines.append(
            f"  {marker} {rejection.setup.value:<15} {rejection.gate:<28} "
            f"{rejection.wait_reason.value}"
        )
    return tuple(lines)
