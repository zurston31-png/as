import pytest

from app.rugcheck.filters import estimate_dev_holder_pct, evaluate_token_security


def _clean_solana_response(**overrides) -> dict:
    data = {
        "mint_authority": None,
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


def test_estimate_dev_holder_pct_prefers_creator_address():
    data = _clean_solana_response()
    assert estimate_dev_holder_pct(data) == pytest.approx(0.05)


def test_estimate_dev_holder_pct_falls_back_to_largest_non_lp_holder():
    data = _clean_solana_response(creator_address="")
    assert estimate_dev_holder_pct(data) == pytest.approx(0.05)
