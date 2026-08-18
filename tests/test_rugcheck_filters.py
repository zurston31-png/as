import pytest

from app.rugcheck.filters import estimate_dev_holder_pct, evaluate_token_security


def _clean_solana_response(**overrides) -> dict:
    data = {
        "mint_authority": None,
        "freeze_authority": None,
        "is_honeypot": "0",
        "creator_address": "DevWallet111",
        "lp_holders": [{"address": "LP1", "percent": "0.9", "is_locked": "1", "tag": "locked"}],
        "holders": [
            {"address": "DevWallet111", "percent": "0.05"},
            {"address": "H2", "percent": "0.05"},
            {"address": "H3", "percent": "0.03"},
        ],
        "total_liquidity": "50000",
    }
    data.update(overrides)
    return data


def test_passes_when_all_checks_clean():
    report = evaluate_token_security("solana", _clean_solana_response())
    assert report.passed, report.reasons
    assert report.mint_disabled is True
    assert report.ownership_renounced is True
    assert report.liquidity_locked is True
    assert report.is_honeypot is False


def test_fails_when_mint_authority_active():
    data = _clean_solana_response(mint_authority="SomeAuthorityAddress")
    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert any("mint authority" in r for r in report.reasons)


def test_fails_on_honeypot_flag():
    data = _clean_solana_response(is_honeypot="1")
    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert report.is_honeypot is True
    assert any("honeypot" in r for r in report.reasons)


def test_fails_on_high_holder_concentration():
    data = _clean_solana_response(holders=[{"address": f"H{i}", "percent": "0.05"} for i in range(10)])
    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert report.top10_holder_pct == pytest.approx(0.5)
    assert any("top 10 holders" in r for r in report.reasons)


def test_fails_on_thin_liquidity():
    data = _clean_solana_response(total_liquidity="100")
    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert any("liquidity too thin" in r for r in report.reasons)


def test_fails_on_unlocked_liquidity():
    data = _clean_solana_response(lp_holders=[{"address": "LP1", "percent": "0.9", "is_locked": "0", "tag": ""}])
    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert report.liquidity_locked is False


def test_empty_response_fails_closed():
    report = evaluate_token_security("solana", {})
    assert not report.passed
    assert report.reasons


def test_market_liquidity_overrides_scanner_figure():
    """Pool depth from market data wins over the scanner's own number."""
    data = _clean_solana_response(total_liquidity="100")  # scanner says thin
    thin = evaluate_token_security("solana", data)
    assert not thin.passed

    deep = evaluate_token_security("solana", data, liquidity_usd=250_000.0)
    assert deep.liquidity_usd == pytest.approx(250_000.0)
    assert not any("liquidity too thin" in r for r in deep.reasons)


def test_market_liquidity_satisfies_depth_check_when_scanner_omits_it():
    """GoPlus's Solana responses carry no liquidity field. Supplying depth
    from market data must clear the depth check rather than failing for
    missing data — the bug that rejected every Solana token."""
    data = _clean_solana_response()
    data.pop("total_liquidity")

    without = evaluate_token_security("solana", data)
    assert any("no liquidity figure" in r for r in without.reasons)

    with_market = evaluate_token_security("solana", data, liquidity_usd=250_000.0)
    assert not any("liquidity" in r and "no liquidity figure" in r for r in with_market.reasons)
    assert with_market.passed, with_market.reasons


def test_thin_market_liquidity_still_rejected():
    """The override must not become a bypass: genuinely thin pools still fail."""
    data = _clean_solana_response()
    data.pop("total_liquidity")
    report = evaluate_token_security("solana", data, liquidity_usd=50.0)
    assert not report.passed
    assert any("liquidity too thin" in r for r in report.reasons)


def test_missing_mint_field_is_not_treated_as_safe():
    """Regression: an absent mint field used to read as 'mint disabled' and
    pass. Absence must be reported as unverifiable, never as safe."""
    data = _clean_solana_response()
    data.pop("mint_authority")

    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert report.mint_disabled is not True
    assert any("could not verify" in r and "mint authority" in r for r in report.reasons)


def test_missing_freeze_field_is_not_treated_as_safe():
    data = _clean_solana_response()
    data.pop("freeze_authority")

    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert any("could not verify" in r and "freeze authority" in r for r in report.reasons)


def test_active_freeze_authority_rejected_on_solana():
    """An active freeze authority lets the issuer block your sells - the
    Solana equivalent of a honeypot."""
    data = _clean_solana_response(freeze_authority="FreezeAuthorityAddr111")
    report = evaluate_token_security("solana", data)
    assert not report.passed
    assert any("freeze authority is still active" in r for r in report.reasons)


def test_solana_object_shaped_flags_are_understood():
    """GoPlus uses {"status": "1"} objects on Solana as well as scalars."""
    data = _clean_solana_response()
    data.pop("mint_authority")
    data["mintable"] = {"status": "0", "authority": []}
    data["freezable"] = {"status": "0", "authority": []}

    report = evaluate_token_security("solana", data, liquidity_usd=250_000.0)
    assert report.passed, report.reasons
    assert report.mint_disabled is True

    data["mintable"] = {"status": "1", "authority": ["SomeAuthority"]}
    bad = evaluate_token_security("solana", data, liquidity_usd=250_000.0)
    assert not bad.passed
    assert any("mint authority is still active" in r for r in bad.reasons)


def test_missing_honeypot_field_is_not_treated_as_safe_on_evm():
    report = evaluate_token_security("ethereum", {"is_mintable": "0"}, liquidity_usd=250_000.0)
    assert not report.passed
    assert any("could not verify" in r and "honeypot status" in r for r in report.reasons)


def test_read_flag_distinguishes_absent_from_false():
    from app.rugcheck.filters import read_flag

    assert read_flag({}, "missing") is None                      # absent
    assert read_flag({"a": None}, "a") is False                  # renounced
    assert read_flag({"a": "0"}, "a") is False
    assert read_flag({"a": "1"}, "a") is True
    assert read_flag({"a": "SomeAddress"}, "a") is True          # live authority
    assert read_flag({"a": {"status": "1"}}, "a") is True
    assert read_flag({"a": {"status": "0"}}, "a") is False
    assert read_flag({"b": "1"}, "a", "b") is True               # fallback key


def test_estimate_dev_holder_pct_prefers_creator_address():
    data = _clean_solana_response()
    assert estimate_dev_holder_pct(data) == pytest.approx(0.05)


def test_estimate_dev_holder_pct_falls_back_to_largest_non_lp_holder():
    data = _clean_solana_response(creator_address="")
    assert estimate_dev_holder_pct(data) == pytest.approx(0.05)
