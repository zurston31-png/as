"""Aggregated pre-trade rug-pull / scam filter.

`evaluate_token_security` is a pure function (no network I/O) so it can be
unit tested directly with crafted GoPlus response fixtures. `run_rug_checks`
is the async wrapper that fetches live data and calls it.

Every check here is a *reject on failure to prove safety* design: if a data
point is missing or a lookup fails, the token does NOT pass by default.
"""
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.rugcheck import goplus, honeypot
from app.services import price_feed

logger = logging.getLogger(__name__)

LP_TAGS = {"lp", "locked", "burn"}


@dataclass
class RugCheckReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    ownership_renounced: bool | None = None
    mint_disabled: bool | None = None
    liquidity_locked: bool | None = None
    is_honeypot: bool | None = None
    top10_holder_pct: float | None = None
    liquidity_usd: float | None = None
    dev_wallet_pct: float | None = None
    raw: dict = field(default_factory=dict)


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_flag(data: dict, *keys: str) -> bool | None:
    """Read a boolean-ish scanner flag across the shapes GoPlus uses.

    Returns None when none of `keys` is present, so a field this code does
    not know the name of is never silently read as "safe". Using
    data.get(key) directly is what made the Solana mint-authority and
    honeypot checks pass vacuously: an absent key looked identical to an
    explicit "no".

    Handles the EVM scalar shape ("1"/"0"), the Solana object shape
    ({"status": "1", ...}), and authority fields where null means the
    authority has been renounced.
    """
    for key in keys:
        if key not in data:
            continue
        value = data[key]

        if isinstance(value, dict):
            status = value.get("status")
            if status is not None:
                return str(status) == "1"
            authority = value.get("authority")
            if authority is not None:
                return bool(authority)
            continue

        if value is None:
            return False  # e.g. "mint_authority": null -> renounced
        if isinstance(value, bool):
            return value
        if isinstance(value, list):
            return len(value) > 0

        text = str(value).strip().lower()
        if text in ("1", "true", "yes"):
            return True
        if text in ("0", "false", "no", "", "null", "none"):
            return False
        # A non-empty authority address, or any other truthy string.
        return True
    return None


def estimate_dev_holder_pct(data: dict) -> float | None:
    """Best-effort 'dev wallet' identification: prefer the address GoPlus
    reports as the token creator; fall back to the largest holder that
    isn't tagged as an LP/locked/burn address. This is a heuristic — GoPlus
    does not universally expose a definitive 'dev wallet' field across
    chains, so treat this as an early-warning signal, not ground truth.
    """
    holders = data.get("holders") or []
    creator = (data.get("creator_address") or "").lower()

    if creator:
        for h in holders:
            if (h.get("address") or "").lower() == creator:
                return _to_float(h.get("percent"))

    non_lp = [h for h in holders if (h.get("tag") or "").lower() not in LP_TAGS]
    if non_lp:
        return _to_float(non_lp[0].get("percent"))
    return None


def evaluate_token_security(
    chain: str,
    data: dict,
    honeypot_flag: bool = False,
    liquidity_usd: float | None = None,
) -> RugCheckReport:
    """Apply every pre-trade check to a security-scanner response.

    `liquidity_usd`, when supplied, is pool depth measured by a market data
    source (DexScreener) and takes priority over anything the security
    scanner reports. GoPlus's liquidity fields are EVM-shaped and are absent
    from its Solana responses, so relying on them alone made every Solana
    token fail the depth check for lack of data.
    """
    reasons: list[str] = []
    report = RugCheckReport(passed=True, raw={"goplus": data})

    if not data:
        return RugCheckReport(
            passed=False,
            reasons=["token not found by security scanner (too new / unindexed / invalid address)"],
        )

    is_solana = chain.lower() == "solana"
    unverifiable: list[str] = []

    # --- mint authority / ownership ---
    if is_solana:
        mintable = read_flag(data, "mintable", "mint_authority")
    else:
        mintable = read_flag(data, "is_mintable")

    report.mint_disabled = mintable is False
    report.ownership_renounced = mintable is False
    if mintable is None:
        unverifiable.append("mint authority")
    elif mintable:
        reasons.append("mint authority is still active (supply can be inflated)")

    # --- honeypot / sellability ---
    # On Solana an active freeze authority is the equivalent trap: the issuer
    # can freeze your account so the position can never be sold.
    scanner_honeypot = read_flag(data, "is_honeypot")
    if is_solana:
        freezable = read_flag(data, "freezable", "freeze_authority")
        if freezable is None:
            unverifiable.append("freeze authority")
        elif freezable:
            reasons.append("freeze authority is still active (issuer can block you from selling)")
        # Solana responses need not carry is_honeypot; freeze authority covers it.
        if scanner_honeypot is None:
            scanner_honeypot = False
    elif scanner_honeypot is None:
        unverifiable.append("honeypot status")

    is_honeypot = bool(scanner_honeypot) or honeypot_flag
    report.is_honeypot = is_honeypot
    if is_honeypot:
        reasons.append("token flagged as a honeypot (may not be sellable)")

    if read_flag(data, "cannot_sell_all"):
        reasons.append("scanner reports the position cannot be fully sold")

    # --- liquidity locked ---
    lp_holders = data.get("lp_holders") or []
    locked_pct = 0.0
    for h in lp_holders:
        tag = (h.get("tag") or "").lower()
        is_locked = h.get("is_locked") in (1, "1", True) or tag in ("locked", "burn")
        if is_locked:
            locked_pct += _to_float(h.get("percent"), 0.0)
    liquidity_locked = locked_pct >= 0.5
    report.liquidity_locked = liquidity_locked
    if lp_holders and not liquidity_locked:
        reasons.append(f"liquidity not sufficiently locked/burned ({locked_pct * 100:.1f}% locked, need >=50%)")
    elif not lp_holders:
        reasons.append("no LP holder data returned - cannot confirm liquidity is locked")

    # --- holder concentration ---
    holders = data.get("holders") or []
    if holders:
        top10_pct = sum(_to_float(h.get("percent"), 0.0) for h in holders[:10])
        report.top10_holder_pct = top10_pct
        if top10_pct > settings.MAX_TOP10_HOLDER_PCT:
            reasons.append(
                f"top 10 holders own {top10_pct * 100:.1f}% of supply "
                f"(limit {settings.MAX_TOP10_HOLDER_PCT * 100:.0f}%)"
            )
    else:
        reasons.append("no holder distribution data returned - cannot confirm concentration risk")

    report.dev_wallet_pct = estimate_dev_holder_pct(data)

    # --- liquidity depth ---
    # Market-data depth wins; fall back to the scanner's own figure (EVM only).
    if liquidity_usd is None:
        liquidity_usd = _to_float(data.get("total_liquidity") or data.get("liquidity"))
    report.liquidity_usd = liquidity_usd
    if liquidity_usd is None:
        reasons.append("no liquidity figure available from market data - cannot confirm exit depth")
    elif liquidity_usd < settings.MIN_LIQUIDITY_USD:
        reasons.append(
            f"liquidity too thin: ${liquidity_usd:,.0f} (minimum ${settings.MIN_LIQUIDITY_USD:,.0f})"
        )

    if unverifiable:
        # Never treat "the scanner didn't tell us" as "it's fine". Naming the
        # specific checks also reveals which field names this scanner uses
        # for a given chain — run DIAGNOSE_TOKEN to see the raw response.
        reasons.append(
            "could not verify " + ", ".join(unverifiable)
            + " - scanner returned no recognised field for this chain"
        )

    report.reasons = reasons
    report.passed = len(reasons) == 0
    return report


async def run_rug_checks(chain: str, token_address: str | None) -> RugCheckReport:
    if not settings.RUGCHECK_ENABLED:
        logger.warning("RUGCHECK_ENABLED=false - buy signals are NOT being screened for scams/rugs")
        return RugCheckReport(passed=True, reasons=["rug check disabled in config - proceeding without screening"])

    if not token_address:
        return RugCheckReport(passed=False, reasons=["no on-chain token address supplied with signal"])

    try:
        data = await goplus.fetch_token_security(chain, token_address)
    except Exception as exc:  # noqa: BLE001 - any network/parsing failure blocks the trade
        logger.exception("GoPlus lookup failed for %s", token_address)
        return RugCheckReport(passed=False, reasons=[f"security scanner lookup failed: {exc}"])

    honeypot_flag = False
    if chain.lower() != "solana":
        try:
            hp = await honeypot.check_honeypot_evm(token_address, settings.EVM_CHAIN_ID)
            honeypot_flag = bool(hp.get("honeypotResult", {}).get("isHoneypot"))
        except Exception:
            logger.warning("honeypot.is lookup failed for %s, relying on GoPlus only", token_address, exc_info=True)

    # Pool depth from market data rather than the security scanner - see
    # evaluate_token_security's docstring.
    liquidity_usd = await price_feed.get_liquidity_usd(token_address)

    return evaluate_token_security(
        chain, data, honeypot_flag=honeypot_flag, liquidity_usd=liquidity_usd
    )
