"""Rug-check filter tests.

Fixtures mirror real responses captured from the live APIs, not invented
shapes. Earlier versions of these tests passed against a schema this code
made up, which is precisely why the Solana path was broken in production
while the suite stayed green:

  - GoPlus Solana returns `lp_holders` as an EMPTY list and reports LP burn
    per pool under `dex[].burn_percent`.
  - GoPlus Solana holder entries are keyed `account`, not `address`.
  - There is no `creator_address`; the creator is `creators[0].address`.
  - RugCheck.xyz uses an entirely different vocabulary again.
"""
import pytest

from app.config import settings
from app.rugcheck.filters import (
    estimate_dev_holder_pct,
    evaluate_snapshot,
    evaluate_token_security,
    normalise_pcts,
    read_flag,
    snapshot_from_goplus,
    snapshot_from_rugcheck,
)


# ---------------------------------------------------------------------------
# fixtures shaped like real responses
# ---------------------------------------------------------------------------

def goplus_solana(**overrides) -> dict:
    """Shape captured from api.gopluslabs.io/api/v1/solana/token_security."""
    data = {
        "mintable": {"status": "0", "authority": []},
        "freezable": {"status": "0", "authority": []},
        "balance_mutable_authority": {"status": "0", "authority": []},
        "creators": [{"address": "GGRwvoNPgXx3SmRNQUVEvDDxq5AE79HFJPbG7srTh9NN",
                      "malicious_address": 0}],
        "dex": [{"dex_name": "raydium", "burn_percent": "100", "lp_amount": "1000",
                 "tvl": "456714", "type": "standard"}],
        "holders": [
            {"account": "GGRwvoNPgXx3SmRNQUVEvDDxq5AE79HFJPbG7srTh9NN",
             "percent": "0.04", "is_locked": 0, "tag": ""},
            {"account": "H2", "percent": "0.03", "is_locked": 0, "tag": ""},
            {"account": "H3", "percent": "0.02", "is_locked": 0, "tag": ""},
        ],
        "lp_holders": [],          # genuinely empty on Solana
        "holder_count": "21048",
        "total_supply": "999264872.134641",
        "non_transferable": "0",
        "trusted_token": 0,
    }
    data.update(overrides)
    return data


def rugcheck_report(**overrides) -> dict:
    """Shape captured from api.rugcheck.xyz/v1/tokens/{mint}/report."""
    data = {
        "mint": "RmtMAYVTTFv2iK9muMrXEoAnSSsZPPgRPbqZCKwNDYk",
        "creator": "GGRwvoNPgXx3SmRNQUVEvDDxq5AE79HFJPbG7srTh9NN",
        "creatorBalance": 0,
        "mintAuthority": None,      # null == renounced
        "freezeAuthority": None,
        "token": {"decimals": 6, "freezeAuthority": None, "isInitialized": True,
                  "mintAuthority": None, "supply": 999264872134641},
        "topHolders": [
            {"address": "A1", "owner": "O1", "pct": 4.2, "insider": False},
            {"address": "A2", "owner": "O2", "pct": 3.1, "insider": False},
            {"address": "A3", "owner": "O3", "pct": 2.0, "insider": False},
        ],
        "markets": [{"marketType": "raydium", "lp": {"lpLockedPct": 100.0}}],
        "lockers": {},
        "lockerScanStatus": "none",
        "totalMarketLiquidity": 993315.48,
        "totalHolders": 38485,
        "totalLPProviders": 1,
        "risks": [],
        "rugged": False,
        "score": 1,
        "score_normalised": 1,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_read_flag_distinguishes_absent_from_false():
    assert read_flag({}, "missing") is None                 # absent
    assert read_flag({"a": None}, "a") is False             # renounced
    assert read_flag({"a": "0"}, "a") is False
    assert read_flag({"a": "1"}, "a") is True
    assert read_flag({"a": "SomeAddress"}, "a") is True     # live authority
    assert read_flag({"a": {"status": "1"}}, "a") is True
    assert read_flag({"a": {"status": "0"}}, "a") is False
    assert read_flag({"b": "1"}, "a", "b") is True          # fallback key


def test_normalise_pcts_detects_scale():
    # GoPlus fraction scale is left alone.
    assert normalise_pcts([0.04, 0.03]) == pytest.approx([0.04, 0.03])
    # RugCheck 0-100 scale is converted.
    assert normalise_pcts([4.2, 3.1]) == pytest.approx([0.042, 0.031])
    assert normalise_pcts([]) == []


# ---------------------------------------------------------------------------
# GoPlus Solana
# ---------------------------------------------------------------------------

def test_goplus_solana_clean_token_passes():
    report = evaluate_snapshot(snapshot_from_goplus("solana", goplus_solana()))
    assert report.passed, report.reasons
    assert report.mint_disabled is True
    assert report.liquidity_locked is True


def test_lp_burn_read_from_dex_not_lp_holders():
    """Regression: lp_holders is always empty on Solana, so reading only
    that field failed every token for 'no LP holder data'."""
    snap = snapshot_from_goplus("solana", goplus_solana())
    assert snap.lp_secured is True
    assert snap.lp_secured_pct == pytest.approx(1.0)


def test_unburned_lp_rejected():
    data = goplus_solana(dex=[{"dex_name": "raydium", "burn_percent": "0", "tvl": "456714"}])
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert any("not sufficiently locked or burned" in r for r in report.reasons)


def test_active_mint_authority_rejected():
    data = goplus_solana(mintable={"status": "1", "authority": ["SomeAuthority"]})
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert any("mint authority is still active" in r for r in report.reasons)


def test_active_freeze_authority_rejected():
    data = goplus_solana(freezable={"status": "1", "authority": ["FreezeAuth"]})
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert any("freeze authority is still active" in r for r in report.reasons)


def test_missing_mint_field_is_not_treated_as_safe():
    """Absence must read as unverifiable, never as safe."""
    data = goplus_solana()
    data.pop("mintable")
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert report.mint_disabled is not True
    assert any("could not verify" in r and "mint authority" in r for r in report.reasons)


def test_missing_freeze_field_is_not_treated_as_safe():
    data = goplus_solana()
    data.pop("freezable")
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert any("could not verify" in r and "freeze authority" in r for r in report.reasons)


def test_high_holder_concentration_rejected():
    data = goplus_solana(holders=[{"account": f"H{i}", "percent": "0.05", "tag": ""} for i in range(10)])
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert report.top10_holder_pct == pytest.approx(0.5)
    assert any("top 10 holders" in r for r in report.reasons)


def test_liquidity_read_from_dex_tvl():
    snap = snapshot_from_goplus("solana", goplus_solana())
    assert snap.liquidity_usd == pytest.approx(456714.0)


def test_thin_liquidity_rejected():
    data = goplus_solana(dex=[{"dex_name": "raydium", "burn_percent": "100", "tvl": "100"}])
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert any("liquidity too thin" in r for r in report.reasons)


def test_dev_pct_uses_creators_and_account_keys():
    """Regression: the old parser read creator_address and holders[].address,
    neither of which exists in a Solana response, so the dev-wallet monitor
    silently had nothing to watch."""
    snap = snapshot_from_goplus("solana", goplus_solana())
    assert snap.dev_pct == pytest.approx(0.04)


def test_estimate_dev_holder_pct_falls_back_to_largest_non_lp_holder():
    data = goplus_solana(creators=[])
    assert estimate_dev_holder_pct(data) == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# RugCheck.xyz
# ---------------------------------------------------------------------------

def test_rugcheck_clean_token_passes():
    report = evaluate_snapshot(snapshot_from_rugcheck(rugcheck_report()))
    assert report.passed, report.reasons
    assert report.mint_disabled is True
    assert report.liquidity_locked is True
    assert report.liquidity_usd == pytest.approx(993315.48)


def test_rugcheck_pct_scale_converted():
    snap = snapshot_from_rugcheck(rugcheck_report())
    # 4.2 + 3.1 + 2.0 percent -> 0.093 as a fraction
    assert snap.top10_pct == pytest.approx(0.093)


def test_rugcheck_rugged_flag_rejected():
    report = evaluate_snapshot(snapshot_from_rugcheck(rugcheck_report(rugged=True)))
    assert not report.passed
    assert any("already rugged" in r for r in report.reasons)


def test_rugcheck_danger_risk_rejected():
    data = rugcheck_report(risks=[
        {"name": "Large Amount of LP Unlocked", "level": "danger", "score": 8000},
    ])
    report = evaluate_snapshot(snapshot_from_rugcheck(data))
    assert not report.passed
    assert any("Large Amount of LP Unlocked" in r for r in report.reasons)


def test_rugcheck_warn_level_risk_does_not_block():
    data = rugcheck_report(risks=[{"name": "Low amount of LP Providers", "level": "warn"}])
    report = evaluate_snapshot(snapshot_from_rugcheck(data))
    assert report.passed, report.reasons


def test_rugcheck_unrecognised_severity_blocks():
    """A risk whose severity can't be read must block, not be ignored."""
    data = rugcheck_report(risks=[{"name": "Something New", "level": "spicy"}])
    report = evaluate_snapshot(snapshot_from_rugcheck(data))
    assert not report.passed
    assert any("unrecognised severity" in r for r in report.reasons)


def test_rugcheck_active_authority_rejected():
    report = evaluate_snapshot(snapshot_from_rugcheck(rugcheck_report(mintAuthority="SomeAuth")))
    assert not report.passed
    assert any("mint authority is still active" in r for r in report.reasons)


def test_rugcheck_authority_falls_back_to_token_object():
    data = rugcheck_report()
    data.pop("freezeAuthority")
    snap = snapshot_from_rugcheck(data)
    assert snap.freeze_authority_active is False  # read from token.freezeAuthority


def test_rugcheck_complete_report_with_no_risks_satisfies_lp_check():
    """markets[].lp sub-keys vary and are often unparseable. On a complete
    report RugCheck's empty risks array IS the LP verdict, since it publishes
    unsecured liquidity as a danger risk."""
    data = rugcheck_report(markets=[{"marketType": "raydium", "lp": {"unknownKey": 1}}],
                           lockers={}, risks=[])
    snap = snapshot_from_rugcheck(data)
    assert snap.lp_secured is True
    assert snap.lp_verdict_source == "rugcheck risk analysis"
    assert evaluate_snapshot(snap).passed


def test_rugcheck_lp_inference_still_blocks_on_lp_risk():
    """The inference must not swallow a real LP problem."""
    data = rugcheck_report(markets=[{"marketType": "raydium", "lp": {"unknownKey": 1}}],
                           lockers={},
                           risks=[{"name": "Large Amount of LP Unlocked", "level": "danger"}])
    report = evaluate_snapshot(snapshot_from_rugcheck(data))
    assert not report.passed
    assert any("Large Amount of LP Unlocked" in r for r in report.reasons)


def test_rugcheck_stub_response_does_not_infer_all_clear():
    """A thin or errored report must fail closed, not inherit the
    no-risks-means-fine shortcut."""
    snap = snapshot_from_rugcheck({"mintAuthority": None, "freezeAuthority": None, "risks": []})
    assert snap.lp_secured is None
    report = evaluate_snapshot(snap)
    assert not report.passed
    assert any("could not verify" in r for r in report.reasons)


def test_rugcheck_locker_counts_as_secured():
    data = rugcheck_report(markets=[{"marketType": "raydium", "lp": {}}],
                           lockers={"someLocker": {"amount": 100}})
    snap = snapshot_from_rugcheck(data)
    assert snap.lp_secured is True


# ---------------------------------------------------------------------------
# EVM path
# ---------------------------------------------------------------------------

def test_evm_missing_honeypot_field_is_not_treated_as_safe():
    report = evaluate_token_security("ethereum", {"is_mintable": "0"}, liquidity_usd=250_000.0)
    assert not report.passed
    assert any("could not verify" in r and "honeypot status" in r for r in report.reasons)


def test_evm_does_not_require_freeze_authority():
    """Freeze authority is Solana-only; demanding it on EVM would reject
    every Ethereum token."""
    data = {"is_mintable": "0", "is_honeypot": "0",
            "lp_holders": [{"address": "LP1", "percent": "0.9", "is_locked": "1", "tag": "locked"}],
            "holders": [{"address": "H1", "percent": "0.04"}]}
    report = evaluate_token_security("ethereum", data, liquidity_usd=250_000.0)
    assert report.passed, report.reasons


def test_evm_honeypot_flag_from_simulation_rejects():
    data = {"is_mintable": "0", "is_honeypot": "0",
            "lp_holders": [{"address": "LP1", "percent": "0.9", "is_locked": "1", "tag": "locked"}],
            "holders": [{"address": "H1", "percent": "0.04"}]}
    report = evaluate_token_security("ethereum", data, honeypot_flag=True, liquidity_usd=250_000.0)
    assert not report.passed
    assert any("honeypot" in r for r in report.reasons)


# ---------------------------------------------------------------------------
# shared behaviour
# ---------------------------------------------------------------------------

def test_empty_response_fails_closed():
    report = evaluate_token_security("solana", {})
    assert not report.passed
    assert report.reasons


def test_market_liquidity_overrides_scanner_figure():
    data = goplus_solana(dex=[{"dex_name": "raydium", "burn_percent": "100", "tvl": "100"}])
    thin = evaluate_token_security("solana", data)
    assert not thin.passed

    deep = evaluate_token_security("solana", data, liquidity_usd=250_000.0)
    assert deep.liquidity_usd == pytest.approx(250_000.0)
    assert deep.passed, deep.reasons


def test_market_liquidity_override_is_not_a_bypass():
    data = goplus_solana()
    report = evaluate_token_security("solana", data, liquidity_usd=50.0)
    assert not report.passed
    assert any("liquidity too thin" in r for r in report.reasons)


def test_thresholds_come_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOP10_HOLDER_PCT", 0.05)
    data = goplus_solana()  # top 3 hold 9%
    report = evaluate_snapshot(snapshot_from_goplus("solana", data))
    assert not report.passed
    assert any("top 10 holders" in r for r in report.reasons)
