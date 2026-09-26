"""Tests for the champion/challenger promotion gate.

An automated search always produces a winner. The gate exists to decide
whether that winner is real, so almost every test here is a case where the
correct answer is "do not promote" - including several where the
challenger genuinely does look better.
"""
import random

import pytest

from app.autopilot.promote import (
    BASE_ALPHA, FAIL, INSUFFICIENT_DATA, MIN_EFFECT_R, MIN_OOS_TRADES,
    MIN_REGIME_TRADES, Arm, bootstrap_p_value, evaluate,
)


def _arm(label, mean, n=60, spread=0.4, regimes=("bull", "chop"), seed=7):
    """An arm whose returns average `mean` R with some dispersion."""
    rng = random.Random(seed)
    returns = [rng.gauss(mean, spread) for _ in range(n)]
    per = len(returns) // len(regimes)
    by_regime = {r: returns[i * per:(i + 1) * per] for i, r in enumerate(regimes)}
    return Arm(label=label, returns_r=returns, by_regime=by_regime, max_drawdown_pct=10.0)


# ---------------------------------------------------------------------------
# the bar that matters most
# ---------------------------------------------------------------------------

def test_the_significance_bar_tightens_with_the_number_of_challengers_tried():
    """Two hundred parameter sets produce ten p<0.05 winners on noise alone.
    A gate that does not know how many were tried turns that into a
    recommendation."""
    champion = _arm("champ", 0.10, seed=1)
    challenger = _arm("chal", 0.35, seed=2)

    alone = evaluate(champion, challenger, attempts=1)
    searched = evaluate(champion, challenger, attempts=200)

    assert alone.corrected_alpha == pytest.approx(BASE_ALPHA)
    assert searched.corrected_alpha == pytest.approx(BASE_ALPHA / 200)
    assert searched.corrected_alpha < alone.corrected_alpha


def test_omitting_the_attempt_count_is_an_error_not_a_default():
    """A caller that forgets gets an exception, never a free pass through
    the one bar that stops a search promoting noise."""
    champion, challenger = _arm("champ", 0.1), _arm("chal", 0.3)
    with pytest.raises(TypeError):
        evaluate(champion, challenger)          # attempts is keyword-required
    with pytest.raises(ValueError):
        evaluate(champion, challenger, attempts=0)


def test_two_identical_strategies_are_not_promoted():
    """The null case. If this ever passes, the gate is decorative."""
    champion = _arm("champ", 0.15, seed=11)
    challenger = _arm("chal", 0.15, seed=12)

    verdict = evaluate(champion, challenger, attempts=1)
    assert verdict.promote is False


def test_a_marginal_winner_from_a_wide_search_is_rejected():
    """Exactly the trap this gate exists for.

    A challenger with p ~= 0.01 clears an uncorrected 0.05 bar comfortably.
    The same result, arrived at by picking the best of fifty tries, is what
    you would expect from noise - and the corrected bar (0.001) rejects it.
    Same data, same statistics, different meaning, because the number of
    attempts is part of the evidence.

    The parameters are chosen so the p-value lands between the two bars;
    an effect large enough to pass both would not test anything.
    """
    champion = _arm("champ", 0.10, n=60, spread=0.6, seed=3)
    challenger = _arm("chal", 0.20, n=60, spread=0.6, seed=4)

    solo = evaluate(champion, challenger, attempts=1)
    from_search = evaluate(champion, challenger, attempts=50)

    assert 0.001 < solo.p_value < 0.05, (
        f"test data is not marginal (p={solo.p_value}) - it proves nothing either way"
    )
    assert solo.promote is True
    assert from_search.promote is False
    assert any(
        b.name == "significance" and not b.passed for b in from_search.bars
    ), "the search-corrected rejection must come from the significance bar"


# ---------------------------------------------------------------------------
# the other bars
# ---------------------------------------------------------------------------

def test_a_thin_sample_is_rejected_however_good_it_looks():
    champion = _arm("champ", 0.1, n=MIN_OOS_TRADES - 5, seed=5)
    challenger = _arm("chal", 2.0, n=MIN_OOS_TRADES - 5, seed=6)

    verdict = evaluate(champion, challenger, attempts=1)
    assert verdict.promote is False
    assert any(b.name == "sample" and not b.passed for b in verdict.bars)


def test_a_real_but_tiny_edge_is_not_worth_a_strategy_change():
    """Inside the error on the fee and slippage estimate. Being right about
    a difference this small is not the same as it being worth acting on."""
    champion = _arm("champ", 0.100, n=400, spread=0.05, seed=21)
    challenger = _arm("chal", 0.101, n=400, spread=0.05, seed=22)

    verdict = evaluate(champion, challenger, attempts=1)
    assert verdict.promote is False
    effect = next(b for b in verdict.bars if b.name == "effect")
    assert not effect.passed
    assert "fee and slippage" in effect.detail


def test_an_edge_confined_to_one_regime_is_called_a_regime_bet():
    """A better average that is worse somewhere is not an improvement, it
    is a wager on conditions holding."""
    rng = random.Random(99)
    champ_bull = [rng.gauss(0.1, 0.3) for _ in range(30)]
    champ_chop = [rng.gauss(0.1, 0.3) for _ in range(30)]
    # storms ahead in the bull regime, quietly worse in chop
    chal_bull = [rng.gauss(0.9, 0.3) for _ in range(30)]
    chal_chop = [rng.gauss(0.02, 0.3) for _ in range(30)]

    champion = Arm("champ", champ_bull + champ_chop, 10.0,
                   {"bull": champ_bull, "chop": champ_chop})
    challenger = Arm("chal", chal_bull + chal_chop, 10.0,
                     {"bull": chal_bull, "chop": chal_chop})

    verdict = evaluate(champion, challenger, attempts=1)
    consistency = next(b for b in verdict.bars if b.name == "consistency")
    assert not consistency.passed
    assert "regime bet" in consistency.detail
    assert verdict.promote is False


def test_untested_regimes_do_not_count_as_consistency():
    """"We could not check" must not read as "we checked and it held"."""
    champion = _arm("champ", 0.1, n=40, regimes=("bull",), seed=31)
    challenger = _arm("chal", 0.6, n=40, regimes=("chop",), seed=32)

    verdict = evaluate(champion, challenger, attempts=1)
    consistency = next(b for b in verdict.bars if b.name == "consistency")
    assert not consistency.passed
    assert "has not been shown to survive" in consistency.detail


def test_return_bought_with_disproportionate_drawdown_is_rejected():
    champion = _arm("champ", 0.10, seed=41)
    challenger = _arm("chal", 0.45, seed=42)
    challenger.max_drawdown_pct = 90.0        # vs the champion's 10%

    verdict = evaluate(champion, challenger, attempts=1)
    risk = next(b for b in verdict.bars if b.name == "risk")
    assert not risk.passed
    assert verdict.promote is False


def test_a_genuinely_better_strategy_does_get_promoted():
    """The gate has to be passable, or it is just an off switch."""
    rng = random.Random(4242)
    regimes = {}
    champ_all, chal_all = [], []
    for regime in ("bull", "chop", "bear"):
        c = [rng.gauss(0.05, 0.25) for _ in range(40)]
        x = [rng.gauss(0.45, 0.25) for _ in range(40)]
        regimes[regime] = (c, x)
        champ_all += c
        chal_all += x

    champion = Arm("champ", champ_all, 12.0, {k: v[0] for k, v in regimes.items()})
    challenger = Arm("chal", chal_all, 11.0, {k: v[1] for k, v in regimes.items()})

    verdict = evaluate(champion, challenger, attempts=3)
    assert verdict.promote is True, verdict.table()
    assert "PROMOTE" in verdict.reason()


# ---------------------------------------------------------------------------
# the bootstrap
# ---------------------------------------------------------------------------

def test_a_worse_challenger_short_circuits_to_certainty():
    assert bootstrap_p_value([0.5] * 40, [0.1] * 40) == 1.0


def test_the_bootstrap_is_seeded_so_the_gate_cannot_be_re_rolled():
    """An unseeded gate could be re-run until it agreed."""
    rng = random.Random(1)
    a = [rng.gauss(0.1, 0.4) for _ in range(50)]
    b = [rng.gauss(0.4, 0.4) for _ in range(50)]
    assert bootstrap_p_value(a, b) == bootstrap_p_value(a, b)


def test_the_bootstrap_reports_unknown_on_an_empty_arm():
    assert bootstrap_p_value([], [0.1, 0.2]) is None
    assert bootstrap_p_value([0.1], []) is None


def test_profit_factor_is_unknown_rather_than_infinite_without_losers():
    """inf would sort above every real strategy."""
    assert Arm("x", [0.2, 0.3, 0.4]).profit_factor is None
    assert Arm("y", [0.2, -0.1]).profit_factor == pytest.approx(2.0)


def test_a_thin_sample_reads_as_insufficient_data_not_as_failure():
    """The distinction that matters. A challenger that lost on effect has
    been WEIGHED and found wanting; one that only fell short on sample size
    has not been weighed at all. Recording the second as a failure retires
    an idea that was never tested, and makes the loop look like it is
    ruling things out when it is only running out of data.
    """
    champion = _arm("champ", 0.1, n=5, seed=51)
    challenger = _arm("chal", 0.9, n=5, seed=52)
    verdict = evaluate(champion, challenger, attempts=1)

    assert verdict.outcome == INSUFFICIENT_DATA
    assert verdict.promote is False
    assert "not evidence against" in verdict.reason()
    assert "more paper trading" in verdict.reason()


def test_a_measurably_worse_challenger_reads_as_a_real_failure():
    """The other side of the tri-state: this one WAS weighed."""
    champion = _arm("champ", 0.60, n=80, spread=0.2, seed=61)
    challenger = _arm("chal", 0.05, n=80, spread=0.2, seed=62)

    verdict = evaluate(champion, challenger, attempts=1)
    assert verdict.outcome == FAIL
    assert "KEEP champ" in verdict.reason()
