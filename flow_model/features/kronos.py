"""Kronos: a pre-trained candlestick foundation model, wired in quarantined.

Kronos (github.com/shiyu-coder/Kronos, MIT) is a decoder-only transformer
pre-trained on K-line sequences from "over 45 global exchanges". It tokenizes
OHLCV bars and autoregressively samples forecast paths. This module turns
those paths into bounded features that satisfy the same `FeatureComputer`
contract as every other component, and refuses to produce them where they
would not mean anything.

Why this module is shaped so defensively
----------------------------------------
Every other feature in this project is an explicit function of visible bars.
This one is a 24.7M-parameter checkpoint someone else trained on data nobody
here can enumerate, and that difference has exactly one serious consequence.

**Kronos publishes no training-data cutoff.** The research window here is
2015-01-01 to 2023-01-01, with a sealed holdout to 2025-01-01. A checkpoint
released in 2025, trained on recent global-exchange data, has almost
certainly seen all of it. A forecast it makes for 2017 is informed by what
happened in 2018.

**The project's lookahead defences cannot detect this.**
`validation/lookahead.py` runs four checks -- truncation invariance,
future-mutation invariance, warmup honesty, determinism -- and a pre-trained
model passes every one of them trivially. Truncating the dataset after bar t
does not change the weights. Mutating future bars does not change the
weights. The leak is IN the weights. `MarketView` can prove no future bar was
read at bar t; it cannot prove no future bar was read last year by whoever
produced the checkpoint. `tests/unit/test_kronos.py` asserts this blindness
deliberately, so that a green audit is never read as evidence of a clean
model.

So the guard is not the audit. It is `ContaminationGuard`, which compares the
evaluation bar against the configured `pretrain_cutoff` and, under the
default REFUSE policy, declines any bar the checkpoint may have trained on.

What that leaves, which is not nothing
--------------------------------------
* **Live and paper-forward signals are clean.** A bar that has not happened
  cannot be in anyone's training set. The guard permits it automatically,
  because its timestamp is after any cutoff including an unknown one that
  cannot extend past the present.
* **The contaminated signal is measurable on purpose.** Policy FLAG computes
  it and marks the vector DEGRADED. That is worth having: performance with
  a model that has seen the answers is an UPPER BOUND no clean model can
  beat, which makes it a useful reference point rather than a result.

What this module does NOT do
----------------------------
It does not enter the Flow Score by default. Section 7 fixes the five
component weights at a sum of 100 and every setup threshold was calibrated
on that scale, so admitting a sixth component silently rescales every gate.
`include_in_flow_score` exists for Phase 8 to sweep an alternative weighting;
the baseline leaves it off.

It also makes no claim that the forecast is skilful. Nothing here has been
measured against held-out data by this project, and the one number that would
settle it -- out-of-sample forecast accuracy on bars the checkpoint provably
never saw -- cannot be computed until a cutoff is known.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from flow_model.core.enums import DataQuality, Feed, KronosContamination, KronosMode
from flow_model.features.base import (
    FeatureComputer,
    FeatureError,
    percentile_rank,
    safe_divide,
    squash,
)

__all__ = [
    "KronosUnavailable",
    "ContaminationVerdict",
    "ContaminationGuard",
    "KronosAdapter",
    "ForecastPaths",
    "KronosFeatures",
    "KRONOS_KEYS",
]

#: Output keys, in declared order.
KRONOS_KEYS: tuple[str, ...] = (
    "kronos_available",
    "kronos_contaminated",
    "kronos_up_probability",
    "kronos_direction",
    "kronos_expected_move_atr",
    "kronos_expected_move_score",
    "kronos_path_dispersion_atr",
    "kronos_agreement",
    "kronos_horizon_bars",
    "kronos_sample_count",
)

#: Width of the dead band around 0.5 in which `kronos_up_probability` is read
#: as no opinion. JUDGEMENT CALL: a 32-path sample has a standard error of
#: about 0.09 on a probability near one half, so anything inside +/- 0.05 of
#: even money is indistinguishable from a coin flip at this path count. Set
#: wider than that and the feature would discard real signal; narrower and it
#: would report sampling noise as a direction.
DIRECTION_DEAD_BAND = 0.05

#: Scale handed to `squash` for the expected move, in ATR units. JUDGEMENT
#: CALL anchored on the existing setups: SCALP_1R's stop is 1.0 ATR, so a
#: forecast move of one ATR is the natural "this matters" point and maps to
#: about 0.76.
EXPECTED_MOVE_SCALE_ATR = 1.0


class KronosUnavailable(RuntimeError):
    """Kronos could not be loaded. Carries what is missing and how to fix it."""


# ---------------------------------------------------------------------------
# the quarantine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContaminationVerdict:
    """Whether a bar may have been in the checkpoint's training data."""

    contaminated: bool
    permitted: bool
    quality: DataQuality
    reason: str

    @property
    def refused(self) -> bool:
        return not self.permitted


class ContaminationGuard:
    """Decides whether a forecast for a given bar is worth anything.

    The rule turns on the declared MODE, because the thing that makes a bar
    safe is not its timestamp relative to the view -- it is whether the bar
    had happened when the checkpoint was trained, and only the caller knows
    which situation this is:

        RESEARCH, cutoff None  ->  contaminated. A replay is entirely history.
        RESEARCH, cutoff set   ->  contaminated at or before the cutoff.
        LIVE,     cutoff None  ->  clean. The bar has not happened, so no
                                   training set can contain it.
        LIVE,     cutoff set   ->  contaminated at or before the cutoff.

    My first version of this class tried to infer the answer by comparing the
    bar against the view's cutoff, treating a bar at the frontier as clean.
    That is vacuous in exactly the case that matters: in a backtest the view's
    cutoff IS the current bar, so the test was true on every bar of a replay
    and the guard would have waved through every contaminated historical bar
    while appearing to work. `test_the_guard_catches_what_the_audit_cannot`
    caught it.

    No clock is read. A guard that consulted the wall clock would give
    different verdicts on a replay, and a backtest is a replay.
    """

    def __init__(
        self,
        pretrain_cutoff: date | None,
        policy: KronosContamination = KronosContamination.REFUSE,
        mode: KronosMode = KronosMode.RESEARCH,
    ) -> None:
        self.pretrain_cutoff = pretrain_cutoff
        self.policy = policy
        self.mode = mode

    def verdict(self, bar_ts: datetime) -> ContaminationVerdict:
        """Judge the bar at `bar_ts` under the configured mode."""
        bar_day = bar_ts.astimezone(timezone.utc).date()

        if self.pretrain_cutoff is None:
            if self.mode is KronosMode.LIVE:
                return ContaminationVerdict(
                    contaminated=False, permitted=True, quality=DataQuality.GOOD,
                    reason=(
                        "mode=LIVE, so this bar had not happened when any checkpoint "
                        "was trained and cannot be in its training data, whatever the "
                        "unpublished cutoff is"
                    ),
                )
            contaminated = True
            detail = (
                f"mode=RESEARCH and the checkpoint publishes no training-data cutoff, "
                f"so {bar_day} may be in its training set and the forecast may be "
                "informed by the outcome it is predicting"
            )
        else:
            contaminated = bar_day <= self.pretrain_cutoff
            detail = (
                f"{bar_day} is at or before the declared pretrain_cutoff "
                f"{self.pretrain_cutoff}"
                if contaminated
                else f"{bar_day} is after the declared pretrain_cutoff {self.pretrain_cutoff}"
            )

        if not contaminated:
            return ContaminationVerdict(
                contaminated=False, permitted=True, quality=DataQuality.GOOD, reason=detail
            )

        if self.policy is KronosContamination.REFUSE:
            return ContaminationVerdict(
                contaminated=True, permitted=False, quality=DataQuality.MISSING,
                reason=(
                    f"REFUSED: {detail}. The lookahead audit cannot detect this -- the "
                    "leak is in the weights, not the data access -- so the feature is "
                    "declined rather than reported as clean. Set contamination_policy "
                    "to FLAG to measure the contaminated signal as an upper bound, or "
                    "declare pretrain_cutoff if it is known."
                ),
            )
        if self.policy is KronosContamination.FLAG:
            return ContaminationVerdict(
                contaminated=True, permitted=True, quality=DataQuality.DEGRADED,
                reason=(
                    f"CONTAMINATED but computed under policy FLAG: {detail}. Any "
                    "performance figure derived from this is an upper bound, not a "
                    "result."
                ),
            )
        return ContaminationVerdict(
            contaminated=True, permitted=True, quality=DataQuality.GOOD,
            reason=f"policy ALLOW: {detail}",
        )


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForecastPaths:
    """Sampled terminal outcomes from one forecast call.

    Only the terminal close of each path is kept. The full path matrix is
    deliberately not retained: a feature built on the interior of a sampled
    path would be reading a quantity with no counterpart in the real tape,
    and the horizon that matters to a setup is where its target sits.
    """

    terminal_closes: np.ndarray
    anchor_close: float
    horizon_bars: int

    def __post_init__(self) -> None:
        if self.terminal_closes.ndim != 1 or not self.terminal_closes.size:
            raise FeatureError("ForecastPaths needs a non-empty 1-D array of closes")
        if not np.all(np.isfinite(self.terminal_closes)):
            raise FeatureError("ForecastPaths received a non-finite close")
        if not math.isfinite(self.anchor_close) or self.anchor_close <= 0.0:
            raise FeatureError(f"anchor_close must be positive and finite, got {self.anchor_close}")


class KronosAdapter:
    """Thin wrapper over Kronos's own `KronosPredictor`.

    Isolated for two reasons. It is the only place a torch dependency is
    touched, so the rest of the package imports cleanly without it; and
    Kronos's API lives in a repo rather than a published package, so the
    import is guarded and reported rather than assumed.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._predictor = None
        self._load_error: str | None = None

    # --- availability --------------------------------------------------

    def _import_kronos(self):
        if self.config.module_path:
            if self.config.module_path not in sys.path:
                sys.path.insert(0, self.config.module_path)
        try:
            return importlib.import_module("model")
        except ImportError as exc:
            raise KronosUnavailable(
                f"Kronos's `model` package is not importable ({exc}). It is not on "
                "PyPI, so clone github.com/shiyu-coder/Kronos and point "
                "config.kronos.module_path at the checkout, or install it into the "
                "environment. Nothing is substituted in its absence."
            ) from None

    def load(self):
        """Load tokenizer and model, or raise `KronosUnavailable`.

        Cached, including the failure: retrying a missing dependency once per
        bar would turn a misconfiguration into a very slow backtest.
        """
        if self._predictor is not None:
            return self._predictor
        if self._load_error is not None:
            raise KronosUnavailable(self._load_error)
        try:
            module = self._import_kronos()
            try:
                import torch
            except ImportError as exc:
                raise KronosUnavailable(
                    f"Kronos requires torch, which is not importable ({exc}). "
                    "Install the project's 'kronos' extra."
                ) from None
            tokenizer = module.KronosTokenizer.from_pretrained(self.config.tokenizer_repo)
            model = module.Kronos.from_pretrained(self.config.model_repo)
            predictor = module.KronosPredictor(
                model, tokenizer, device=self.config.device,
                max_context=int(self.config.max_context),
            )
            self._predictor = predictor
            self._torch = torch
            return predictor
        except KronosUnavailable as exc:
            self._load_error = str(exc)
            raise
        except Exception as exc:  # pragma: no cover - depends on the checkpoint
            self._load_error = (
                f"Kronos failed to load {self.config.model_repo!r} with tokenizer "
                f"{self.config.tokenizer_repo!r}: {type(exc).__name__}: {exc}"
            )
            raise KronosUnavailable(self._load_error) from None

    def available(self) -> bool:
        try:
            self.load()
        except KronosUnavailable:
            return False
        return True

    # --- the forecast --------------------------------------------------

    def forecast(
        self,
        frame,
        x_timestamp,
        y_timestamp,
        anchor_close: float,
    ) -> ForecastPaths:
        """Sample `sample_count` paths and keep their terminal closes.

        Seeded immediately before the call. Kronos samples with a temperature
        and a nucleus cutoff, so without this the same view would give a
        different answer on every evaluation -- which would fail the lookahead
        audit's determinism check, the one check of the four that a sampling
        model can genuinely fail.
        """
        predictor = self.load()
        cfg = self.config
        torch = self._torch
        torch.manual_seed(int(cfg.seed))
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(int(cfg.seed))
        np.random.seed(int(cfg.seed) % (2**32))

        closes: list[float] = []
        for index in range(int(cfg.sample_count)):
            # One path per call, each seeded from the base seed plus its index,
            # rather than one call with sample_count paths. Kronos averages
            # internally when sample_count > 1, and an average of paths is a
            # single smoothed path: it cannot express the DISPERSION these
            # features are built on.
            torch.manual_seed(int(cfg.seed) + index)
            out = predictor.predict(
                df=frame,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=int(cfg.pred_len),
                T=float(cfg.temperature),
                top_p=float(cfg.top_p),
                sample_count=1,
            )
            closes.append(float(out["close"].iloc[-1]))
        return ForecastPaths(
            terminal_closes=np.asarray(closes, dtype=np.float64),
            anchor_close=float(anchor_close),
            horizon_bars=int(cfg.pred_len),
        )


# ---------------------------------------------------------------------------
# the feature computer
# ---------------------------------------------------------------------------


def paths_to_features(paths: ForecastPaths, atr: float | None) -> dict[str, float]:
    """Bounded features from a sample of terminal closes.

    Every quantity is a statistic OF the sample, which is why the sample size
    is emitted alongside them: `kronos_up_probability` from 32 paths carries a
    standard error near 0.09, and a reader who cannot see the denominator
    cannot know that.

    `atr` scales the move so the feature is comparable across instruments and
    across a decade of price levels. Without it the ATR-denominated keys are
    reported as 0.0 AND the caller is expected to have declined the bar --
    `signals/gates.py` and `signals/engine.py` both treat an unmeasured ATR as
    a reason to WAIT, and this module does not invent one.
    """
    closes = paths.terminal_closes
    anchor = paths.anchor_close
    n = int(closes.size)

    up = float(np.count_nonzero(closes > anchor))
    up_probability = up / n

    if up_probability > 0.5 + DIRECTION_DEAD_BAND:
        direction = 1.0
    elif up_probability < 0.5 - DIRECTION_DEAD_BAND:
        direction = -1.0
    else:
        direction = 0.0

    mean_close = float(np.mean(closes))
    move = mean_close - anchor
    dispersion = float(np.std(closes, ddof=1)) if n > 1 else 0.0

    usable_atr = atr if (atr is not None and math.isfinite(atr) and atr > 0.0) else None
    move_atr = safe_divide(move, usable_atr, 0.0) if usable_atr else 0.0
    dispersion_atr = safe_divide(dispersion, usable_atr, 0.0) if usable_atr else 0.0

    # Agreement: how one-sided the sample is, in [0,1]. 0 at an even split,
    # 1 when every path agrees. Distinct from up_probability, which carries
    # the direction; this carries only the strength of consensus, so a scorer
    # can keep magnitude and direction separate as section 7 requires.
    agreement = abs(up_probability - 0.5) * 2.0

    return {
        "kronos_up_probability": up_probability,
        "kronos_direction": direction,
        "kronos_expected_move_atr": move_atr,
        # Signed squash: the magnitude is bounded, the sign is preserved, so a
        # forecast down-move does not read as a weak up-move.
        "kronos_expected_move_score": (
            math.copysign(squash(move_atr, EXPECTED_MOVE_SCALE_ATR), move_atr)
            if move_atr
            else 0.0
        ),
        "kronos_path_dispersion_atr": dispersion_atr,
        "kronos_agreement": agreement,
        "kronos_horizon_bars": float(paths.horizon_bars),
        "kronos_sample_count": float(n),
    }


class KronosFeatures(FeatureComputer):
    """Forecast-derived features, or an honest refusal.

    Construction takes configuration only -- no view, no series, no frame --
    like every other computer in this package, and for the same reason:
    `tests/unit/test_feature_contracts.py` bans market data in a constructor
    because a full-sample statistic captured there is the one leak the
    lookahead audit cannot see. Which is a sharp irony here, since this
    computer's whole problem is a different leak that same audit also cannot
    see. The ban still applies; it is simply not sufficient.
    """

    name = "kronos"

    def __init__(self, config, adapter: KronosAdapter | None = None) -> None:
        self.config = config
        self.adapter = adapter if adapter is not None else KronosAdapter(config)
        self.guard = ContaminationGuard(
            pretrain_cutoff=config.pretrain_cutoff,
            policy=config.contamination_policy,
            mode=config.mode,
        )
        self._context = int(config.max_context)

    @property
    def warmup_bars(self) -> int:
        """The context the model is fed. Honest rather than minimal: the
        forecast is conditioned on all of it, so a shorter history is a
        different model input and not merely a noisier one."""
        return self._context

    @property
    def keys(self) -> tuple[str, ...]:
        return KRONOS_KEYS

    @property
    def required_feeds(self) -> frozenset[Feed]:
        return frozenset({Feed.BARS})

    def compute(self, view):
        if not view.warmup_ok(self._context):
            return self._not_ready(
                view,
                f"kronos needs {self._context} bars of context and the view holds "
                f"{view.bar_count()}",
            )

        if not self.config.enabled:
            return self._vector(
                view,
                {k: 0.0 for k in KRONOS_KEYS},
                quality=DataQuality.MISSING,
                notes=(
                    "kronos: disabled (config.kronos.enabled is False). The baseline "
                    "system is rules-based; this is an opt-in experiment.",
                ),
            )

        last = view.last_bar()
        verdict = self.guard.verdict(last.close_ts)

        if verdict.refused:
            values = {k: 0.0 for k in KRONOS_KEYS}
            values["kronos_contaminated"] = 1.0
            return self._vector(
                view, values, quality=verdict.quality, notes=(f"kronos: {verdict.reason}",)
            )

        try:
            frame, x_ts, y_ts, anchor = self._build_inputs(view)
            paths = self.adapter.forecast(frame, x_ts, y_ts, anchor)
        except KronosUnavailable as exc:
            values = {k: 0.0 for k in KRONOS_KEYS}
            values["kronos_contaminated"] = 1.0 if verdict.contaminated else 0.0
            return self._vector(
                view, values, quality=DataQuality.MISSING, notes=(f"kronos: {exc}",)
            )

        atr = self._atr_hint(view)
        values = paths_to_features(paths, atr)
        values["kronos_available"] = 1.0
        values["kronos_contaminated"] = 1.0 if verdict.contaminated else 0.0

        notes = [f"kronos: {verdict.reason}"]
        if atr is None:
            notes.append(
                "kronos: no ATR available, so the ATR-denominated keys are 0.0 and "
                "the bar should be declined upstream rather than sized on them"
            )
        return self._vector(view, values, quality=verdict.quality, notes=tuple(notes))

    # --- helpers --------------------------------------------------------

    def _build_inputs(self, view):
        """The Kronos input frame, built from the visible prefix only.

        Every array comes from `MarketView`'s own column accessors, which are
        pre-sliced at the cutoff, so this method has no way to reach a bar the
        view does not already expose. That is the firewall doing its job -- and
        it is worth restating that it says nothing about what the CHECKPOINT
        has seen.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise KronosUnavailable(f"kronos needs pandas ({exc})") from None

        n = self._context
        ts_ns = view.bar_timestamps(n)
        stamps = [datetime.fromtimestamp(int(t) / 1e9, tz=timezone.utc) for t in ts_ns]
        frame = pd.DataFrame(
            {
                "open": view.opens(n),
                "high": view.highs(n),
                "low": view.lows(n),
                "close": view.closes(n),
                "volume": view.volumes(n),
            }
        )
        step = stamps[-1] - stamps[-2]
        future = [stamps[-1] + step * (index + 1) for index in range(int(self.config.pred_len))]
        anchor = float(frame["close"].iloc[-1])
        return frame, pd.Series(stamps), pd.Series(future), anchor

    @staticmethod
    def _atr_hint(view) -> float | None:
        """ATR from the view's own helper if it exposes one, else None.

        Deliberately not recomputed here. `features/volatility.py` owns the
        Wilder ATR and a second implementation would be a second answer.
        """
        getter = getattr(view, "atr_hint", None)
        if getter is None:
            return None
        try:
            value = float(getter())
        except Exception:  # pragma: no cover
            return None
        return value if math.isfinite(value) and value > 0.0 else None
