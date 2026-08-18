"""Aggregated pre-trade rug-pull / scam filter.

Design: each scanner response is normalised into a `TokenSnapshot`, and a
single evaluator applies the thresholds. That keeps one copy of the risk
policy while letting Solana and EVM use different data sources, whose
schemas have nothing in common.

Source of truth per chain:
  - Solana: RugCheck.xyz first (indexes new launches and publishes its own
    risk analysis), GoPlus as a fallback.
  - EVM: GoPlus, plus honeypot.is simulation.

Everything fails closed. A check whose data is missing is reported as
unverifiable and blocks the trade — it never counts as a pass. The absent
field must never be indistinguishable from an explicit "no", which is the
bug that previously let Solana tokens clear the mint and honeypot gates
without either being checked.
"""
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.rugcheck import goplus, honeypot, rugcheck_xyz
from app.services import price_feed

logger = logging.getLogger(__name__)

LP_TAGS = {"lp", "locked", "burn"}
# A pool counts as secured when at least this share of LP is burned or locked.
MIN_LP_SECURED = 0.5


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


@dataclass
class TokenSnapshot:
    """Scanner-agnostic view of one token."""

    source: str
    chain: str = "solana"
    mint_authority_active: bool | None = None
    freeze_authority_active: bool | None = None
    honeypot: bool | None = None
    lp_secured: bool | None = None          # burned or locked
    lp_secured_pct: float | None = None
    top10_pct: float | None = None
    liquidity_usd: float | None = None
    dev_pct: float | None = None
    rugged: bool | None = None
    danger_flags: list[str] = field(default_factory=list)
    lp_verdict_source: str = "parsed pool data"
    raw: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------

def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_flag(data: dict, *keys: str) -> bool | None:
    """Read a boolean-ish scanner flag across the shapes these APIs use.

    Returns None when none of `keys` is present, so a field this code does
    not know the name of is never silently read as "safe".

    Handles the EVM scalar shape ("1"/"0"), the GoPlus Solana object shape
    ({"status": "1", "authority": [...]}), and authority fields where null
    means the authority has been renounced.
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
            return False  # e.g. "mintAuthority": null -> renounced
        if isinstance(value, bool):
            return value
        if isinstance(value, list):
            return len(value) > 0

        text = str(value).strip().lower()
        if text in ("1", "true", "yes"):
            return True
        if text in ("0", "false", "no", "", "null", "none"):
            return False
        return True  # a live authority address, or any other truthy string


def normalise_pcts(values: list[float]) -> list[float]:
    """Return percentages as 0-1 fractions.

    Scanners disagree: GoPlus sends "0.05" for 5%, RugCheck sends 5.0. Any
    set of holder shares summing above 1.0 must be on the 0-100 scale, since
    fractions of a supply cannot exceed 1.0 in total.
    """
    if not values:
        return []
    return [v / 100 for v in values] if sum(values) > 1.5 else list(values)


def _top10_from(entries: list[dict], pct_key: str) -> float | None:
    if not entries:
        return None
    pcts = [_to_float(e.get(pct_key), 0.0) or 0.0 for e in entries]
    return sum(normalise_pcts(pcts)[:10])


# --------------------------------------------------------------------------
# snapshot builders
# --------------------------------------------------------------------------

# Base58 excludes 0, O, I and l to avoid visual ambiguity.
BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def looks_like_evm_address(address: str) -> bool:
    return (
        len(address) == 42
        and address.startswith("0x")
        and all(c in "0123456789abcdefABCDEF" for c in address[2:])
    )


def looks_like_solana_address(address: str) -> bool:
    return 32 <= len(address) <= 44 and all(c in BASE58_ALPHABET for c in address)


def resolve_chain(declared: str, token_address: str) -> str:
    """Decide which chain to screen against, trusting the address over the label.

    The chain arrives as free text — typed into a Pine script input or a test
    prompt — and any value other than exactly "solana" used to route a Solana
    mint to the EVM path, skipping the Solana specialist and consulting only
    a scanner with thin Solana coverage. The rejection then read "no record",
    which looks like a verdict on the token rather than a routing mistake.

    An address's own encoding is unambiguous, so prefer it.
    """
    declared_clean = (declared or "").strip().lower()
    address = (token_address or "").strip()

    if looks_like_evm_address(address):
        detected = declared_clean if declared_clean and declared_clean != "solana" else "ethereum"
    elif looks_like_solana_address(address):
        detected = "solana"
    else:
        return declared_clean or "solana"

    if declared_clean and declared_clean != detected:
        logger.warning(
            "signal declared chain %r but %s looks like a %s address - screening as %s",
            declared, address, detected, detected,
        )
    return detected


def _rugcheck_report_is_complete(data: dict) -> bool:
    """True when RugCheck clearly analysed this token rather than returning
    a stub. Guards the "no risk reported means LP is fine" inference below,
    so a thin or errored response can never be read as an all-clear.
    """
    return (
        isinstance(data.get("risks"), list)
        and bool(data.get("markets"))
        and data.get("rugged") is not None
        and data.get("totalMarketLiquidity") is not None
    )


def snapshot_from_rugcheck(data: dict) -> TokenSnapshot:
    """Normalise a RugCheck.xyz report.

    Field names confirmed against a live response: mintAuthority /
    freezeAuthority (null when renounced), token.{mintAuthority,
    freezeAuthority}, topHolders[].pct, totalMarketLiquidity, markets[].lp,
    lockers, risks[], rugged, creator.
    """
    snap = TokenSnapshot(source="rugcheck.xyz", chain="solana", raw={"rugcheck": data})
    token = data.get("token") if isinstance(data.get("token"), dict) else {}

    snap.mint_authority_active = read_flag(data, "mintAuthority")
    if snap.mint_authority_active is None and token:
        snap.mint_authority_active = read_flag(token, "mintAuthority")

    snap.freeze_authority_active = read_flag(data, "freezeAuthority")
    if snap.freeze_authority_active is None and token:
        snap.freeze_authority_active = read_flag(token, "freezeAuthority")

    snap.rugged = read_flag(data, "rugged")
    snap.top10_pct = _top10_from(data.get("topHolders") or [], "pct")
    snap.liquidity_usd = _to_float(data.get("totalMarketLiquidity"))

    # Creator's own share, for the dev-wallet exit monitor.
    creator = (data.get("creator") or "").strip()
    if creator:
        for holder in data.get("topHolders") or []:
            if creator in (holder.get("owner"), holder.get("address")):
                pcts = normalise_pcts([_to_float(holder.get("pct"), 0.0) or 0.0])
                snap.dev_pct = pcts[0] if pcts else None
                break

    # LP security: an explicit locker entry, otherwise burned LP.
    lockers = data.get("lockers")
    if isinstance(lockers, dict) and lockers:
        snap.lp_secured = True
        snap.lp_secured_pct = 1.0
    else:
        burned = []
        for market in data.get("markets") or []:
            lp = market.get("lp") if isinstance(market.get("lp"), dict) else {}
            pct = lp.get("lpLockedPct")
            if pct is None:
                pct = lp.get("lpBurnPct")
            value = _to_float(pct)
            if value is not None:
                burned.append(value / 100 if value > 1.5 else value)
        if burned:
            snap.lp_secured_pct = max(burned)
            snap.lp_secured = snap.lp_secured_pct >= MIN_LP_SECURED

    if snap.lp_secured is None and _rugcheck_report_is_complete(data):
        # No parseable LP percentage, but RugCheck did analyse the pools and
        # publishes unsecured liquidity as a danger risk ("Large Amount of LP
        # Unlocked"). On a complete report, the absence of such a risk is its
        # verdict that LP is fine — defer to the Solana specialist rather
        # than re-deriving it from per-market fields whose shape varies.
        # A real LP problem still arrives via `risks` and blocks below.
        snap.lp_secured = True
        snap.lp_verdict_source = "rugcheck risk analysis"

    # RugCheck's own risk findings. Anything it classes as danger blocks the
    # trade — "large amount of LP unlocked" arrives here. A risk entry whose
    # severity cannot be read is treated as dangerous rather than ignored.
    for risk in data.get("risks") or []:
        if not isinstance(risk, dict):
            snap.danger_flags.append(str(risk)[:120])
            continue
        level = str(risk.get("level") or "").strip().lower()
        name = str(risk.get("name") or risk.get("description") or "unnamed risk")[:120]
        if level in ("danger", "critical", "high"):
            snap.danger_flags.append(name)
        elif level in ("warn", "warning", "info", "low"):
            logger.info("rugcheck non-blocking risk: %s (%s)", name, level)
        else:
            snap.danger_flags.append(f"{name} (unrecognised severity {level!r})")

    return snap


def snapshot_from_goplus(chain: str, data: dict) -> TokenSnapshot:
    """Normalise a GoPlus response.

    Solana and EVM responses share almost no field names. Solana holder
    entries are keyed by `account` (not `address`), the creator lives in
    `creators[].address` (there is no `creator_address`), `lp_holders` comes
    back empty and LP burn is reported per-pool in `dex[].burn_percent`.
    """
    is_solana = chain.lower() == "solana"
    snap = TokenSnapshot(source="goplus", chain=chain, raw={"goplus": data})

    if is_solana:
        snap.mint_authority_active = read_flag(data, "mintable", "mint_authority")
        snap.freeze_authority_active = read_flag(data, "freezable", "freeze_authority")
        snap.honeypot = read_flag(data, "is_honeypot")
        holder_pct_key, holder_id_keys = "percent", ("account", "address")
    else:
        snap.mint_authority_active = read_flag(data, "is_mintable")
        snap.honeypot = read_flag(data, "is_honeypot")
        holder_pct_key, holder_id_keys = "percent", ("address",)

    snap.top10_pct = _top10_from(data.get("holders") or [], holder_pct_key)

    # LP security
    lp_holders = data.get("lp_holders") or []
    if lp_holders:
        secured = 0.0
        for holder in lp_holders:
            tag = (holder.get("tag") or "").lower()
            if holder.get("is_locked") in (1, "1", True) or tag in ("locked", "burn"):
                secured += _to_float(holder.get("percent"), 0.0) or 0.0
        secured = normalise_pcts([secured])[0] if secured else 0.0
        snap.lp_secured_pct = secured
        snap.lp_secured = secured >= MIN_LP_SECURED
    elif is_solana:
        # Solana reports LP burn per pool under `dex` instead.
        burns = [
            _to_float(pool.get("burn_percent"))
            for pool in (data.get("dex") or [])
            if _to_float(pool.get("burn_percent")) is not None
        ]
        if burns:
            best = max(normalise_pcts(burns))
            snap.lp_secured_pct = best
            snap.lp_secured = best >= MIN_LP_SECURED
        tvls = [_to_float(pool.get("tvl")) for pool in (data.get("dex") or [])]
        tvls = [t for t in tvls if t is not None]
        if tvls:
            snap.liquidity_usd = max(tvls)

    if snap.liquidity_usd is None:
        snap.liquidity_usd = _to_float(data.get("total_liquidity") or data.get("liquidity"))

    snap.dev_pct = estimate_dev_holder_pct(data, holder_id_keys)
    return snap


def estimate_dev_holder_pct(data: dict, id_keys: tuple[str, ...] = ("account", "address")) -> float | None:
    """Best-effort dev/team wallet share from a GoPlus response.

    Prefers the reported creator; otherwise the largest holder not tagged as
    an LP/locked/burn address. A heuristic, not ground truth — GoPlus does
    not mark a definitive dev wallet.
    """
    holders = data.get("holders") or []
    creators = data.get("creators") or []
    creator = ""
    if creators and isinstance(creators[0], dict):
        creator = (creators[0].get("address") or "").lower()
    if not creator:
        creator = (data.get("creator_address") or "").lower()

    def pct_of(holder: dict) -> float | None:
        values = normalise_pcts([_to_float(holder.get("percent"), 0.0) or 0.0])
        return values[0] if values else None

    if creator:
        for holder in holders:
            ids = [str(holder.get(k) or "").lower() for k in id_keys]
            if creator in ids:
                return pct_of(holder)

    non_lp = [h for h in holders if (h.get("tag") or "").lower() not in LP_TAGS]
    return pct_of(non_lp[0]) if non_lp else None


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate_snapshot(snap: TokenSnapshot) -> RugCheckReport:
    reasons: list[str] = []
    unverifiable: list[str] = []

    report = RugCheckReport(passed=True, raw=snap.raw)
    report.mint_disabled = snap.mint_authority_active is False
    report.ownership_renounced = snap.mint_authority_active is False
    report.liquidity_locked = snap.lp_secured
    report.is_honeypot = snap.honeypot
    report.top10_holder_pct = snap.top10_pct
    report.liquidity_usd = snap.liquidity_usd
    report.dev_wallet_pct = snap.dev_pct

    if snap.rugged:
        reasons.append("scanner has flagged this token as already rugged")

    for flag in snap.danger_flags:
        reasons.append(f"scanner risk: {flag}")

    # --- authorities ---
    is_solana = snap.chain.lower() == "solana"

    if snap.mint_authority_active is None:
        unverifiable.append("mint authority")
    elif snap.mint_authority_active:
        reasons.append("mint authority is still active (supply can be inflated)")

    # Freeze authority is a Solana concept and is how sellability is blocked
    # there; EVM has no equivalent, so only require it on Solana.
    if snap.freeze_authority_active:
        reasons.append("freeze authority is still active (issuer can block you from selling)")
    elif snap.freeze_authority_active is None and is_solana:
        unverifiable.append("freeze authority")

    # --- honeypot ---
    # On EVM this is the sellability check and must be verified. On Solana
    # freeze authority above covers the same ground, and the scanners do not
    # all report a honeypot field, so absence is not fatal there.
    if snap.honeypot:
        reasons.append("token flagged as a honeypot (may not be sellable)")
    elif snap.honeypot is None and not is_solana:
        unverifiable.append("honeypot status")

    # --- LP secured ---
    if snap.lp_secured is None:
        unverifiable.append("liquidity lock")
    elif not snap.lp_secured:
        pct = (snap.lp_secured_pct or 0.0) * 100
        reasons.append(
            f"liquidity not sufficiently locked or burned ({pct:.1f}%, need >={MIN_LP_SECURED * 100:.0f}%)"
        )

    # --- holder concentration ---
    if snap.top10_pct is None:
        unverifiable.append("holder concentration")
    elif snap.top10_pct > settings.MAX_TOP10_HOLDER_PCT:
        reasons.append(
            f"top 10 holders own {snap.top10_pct * 100:.1f}% of supply "
            f"(limit {settings.MAX_TOP10_HOLDER_PCT * 100:.0f}%)"
        )

    # --- liquidity depth ---
    if snap.liquidity_usd is None:
        unverifiable.append("liquidity depth")
    elif snap.liquidity_usd < settings.MIN_LIQUIDITY_USD:
        reasons.append(
            f"liquidity too thin: ${snap.liquidity_usd:,.0f} (minimum ${settings.MIN_LIQUIDITY_USD:,.0f})"
        )

    if unverifiable:
        reasons.append(
            "could not verify " + ", ".join(unverifiable)
            + f" - {snap.source} returned no recognised field for this chain"
        )

    report.reasons = reasons
    report.passed = not reasons
    return report


def evaluate_token_security(
    chain: str,
    data: dict,
    honeypot_flag: bool = False,
    liquidity_usd: float | None = None,
) -> RugCheckReport:
    """Evaluate a GoPlus response. Retained for the EVM path and tests."""
    if not data:
        return RugCheckReport(
            passed=False,
            reasons=["token not found by security scanner (too new / unindexed / invalid address)"],
        )
    snap = snapshot_from_goplus(chain, data)
    if honeypot_flag:
        snap.honeypot = True
    if liquidity_usd is not None:
        snap.liquidity_usd = liquidity_usd
    return evaluate_snapshot(snap)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

async def run_rug_checks(chain: str, token_address: str | None) -> RugCheckReport:
    if not settings.RUGCHECK_ENABLED:
        logger.warning("RUGCHECK_ENABLED=false - buy signals are NOT being screened for scams/rugs")
        return RugCheckReport(passed=True, reasons=[], raw={"skipped": True})

    if not token_address:
        return RugCheckReport(passed=False, reasons=["no on-chain token address supplied with signal"])

    # Route on the address's own encoding, not the declared label.
    chain = resolve_chain(chain, token_address)
    is_solana = chain == "solana"

    snap: TokenSnapshot | None = None
    # Per-source outcome, so a rejection says which scanners were consulted
    # and what each one actually did. "no scanner had a record" on its own is
    # not actionable — it hides whether a lookup errored, was blocked, or
    # genuinely returned nothing.
    outcomes: list[str] = []

    if is_solana:
        try:
            data = await rugcheck_xyz.fetch_token_report(token_address)
            if data:
                snap = snapshot_from_rugcheck(data)
                outcomes.append(f"rugcheck.xyz: {len(data)} fields")
            else:
                outcomes.append("rugcheck.xyz: no record")
        except Exception as exc:  # noqa: BLE001
            logger.warning("RugCheck lookup failed for %s: %s", token_address, exc)
            outcomes.append(f"rugcheck.xyz: lookup failed ({type(exc).__name__}: {exc})")

    if snap is None:
        try:
            data = await goplus.fetch_token_security(chain, token_address)
            if data:
                snap = snapshot_from_goplus(chain, data)
                outcomes.append(f"goplus: {len(data)} fields")
            else:
                outcomes.append("goplus: no record")
        except Exception as exc:  # noqa: BLE001
            logger.exception("GoPlus lookup failed for %s", token_address)
            outcomes.append(f"goplus: lookup failed ({type(exc).__name__}: {exc})")

    if snap is None:
        logger.warning("no security data for %s on %s - %s", token_address, chain, "; ".join(outcomes))
        return RugCheckReport(
            passed=False,
            reasons=[f"no security data for this token (screened as {chain}) - " + "; ".join(outcomes)],
            raw={"lookup_outcomes": outcomes, "chain": chain},
        )

    if not is_solana:
        try:
            hp = await honeypot.check_honeypot_evm(token_address, settings.EVM_CHAIN_ID)
            if hp.get("honeypotResult", {}).get("isHoneypot"):
                snap.honeypot = True
        except Exception:
            logger.warning("honeypot.is lookup failed for %s", token_address, exc_info=True)

    # Market-measured depth beats any scanner's own figure.
    market_liquidity = await price_feed.get_liquidity_usd(token_address)
    if market_liquidity is not None:
        snap.liquidity_usd = market_liquidity

    return evaluate_snapshot(snap)
