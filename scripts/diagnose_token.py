#!/usr/bin/env python3
"""Dump exactly what every security/market source returns for one token.

Queries DexScreener first to establish which chain the token actually
trades on — guessing the chain wrong makes every scanner return "no such
token", which looks identical to "this token is suspicious". Then asks the
security scanners appropriate to that chain.

Writes everything to diagnostic_output.json.

Windows users: double-click DIAGNOSE_TOKEN.bat instead.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "diagnostic_output.json"

DEXSCREENER = "https://api.dexscreener.com/latest/dex/tokens/{addr}"
GOPLUS_SOLANA = "https://api.gopluslabs.io/api/v1/solana/token_security?contract_addresses={addr}"
GOPLUS_EVM = "https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={addr}"
RUGCHECK = "https://api.rugcheck.xyz/v1/tokens/{addr}/report"

EVM_CHAIN_IDS = {
    "ethereum": "1", "bsc": "56", "polygon": "137", "arbitrum": "42161",
    "base": "8453", "avalanche": "43114", "optimism": "10",
}

IS_WINDOWS = os.name == "nt"


def ask(prompt: str, default: str = "") -> str:
    """input() that survives a closed/piped stdin instead of showing a
    traceback to someone who just double-clicked a .bat file."""
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def pause_and_exit(code: int = 0) -> None:
    if IS_WINDOWS:
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(code)


def fetch(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "memecoin-bot-diagnostic"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return {"__error__": f"HTTP {exc.code}", "__body__": exc.read().decode("utf-8", "replace")[:1500]}
    except Exception as exc:  # noqa: BLE001
        return {"__error__": str(exc)}


def describe_fields(record: dict, indent: str = "    ") -> None:
    for key in sorted(record):
        value = record[key]
        kind = type(value).__name__
        if isinstance(value, list):
            detail = f"list[{len(value)}]"
            if value and isinstance(value[0], dict):
                detail += f" first item keys: {sorted(value[0])}"
            elif value:
                detail += f" e.g. {value[0]!r}"[:70]
        elif isinstance(value, dict):
            detail = f"object keys: {sorted(value)}"
        else:
            detail = repr(value)[:70]
        print(f"{indent}{key:28} ({kind}) {detail}")


def main() -> None:
    print("=" * 68)
    print("  TOKEN DIAGNOSTIC")
    print("=" * 68)
    print("\n  Looks up one token across every data source the bot can use.")
    print("  Nothing is traded. Public data only.\n")

    address = ask("  Paste the token contract address: ")
    if not address:
        print("\n  No address given.")
        pause_and_exit()

    report: dict = {"token_address": address}

    # ---- 1. DexScreener: establishes the chain and real market depth ----
    print("\n  [1/3] DexScreener (market data, tells us the chain)...")
    dex_raw = fetch(DEXSCREENER.format(addr=address))
    pairs = (dex_raw or {}).get("pairs") or [] if isinstance(dex_raw, dict) else []
    report["dexscreener_pair_count"] = len(pairs)

    chain = None
    if pairs:
        best = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        chain = (best.get("chainId") or "").lower()
        symbol = (best.get("baseToken") or {}).get("symbol")
        liq = float((best.get("liquidity") or {}).get("usd") or 0)
        vol = float((best.get("volume") or {}).get("h24") or 0)
        report["dexscreener"] = {
            "chain": chain, "symbol": symbol, "dex": best.get("dexId"),
            "priceUsd": best.get("priceUsd"), "liquidity_usd": liq, "volume_24h": vol,
            "pair_created_at": best.get("pairCreatedAt"),
        }
        print(f"        found: {symbol} on {chain}")
        print(f"        liquidity ${liq:,.0f} | 24h volume ${vol:,.0f}")
    else:
        print("        no trading pools found for this address")

    if not chain:
        chain = ask("\n  Chain not detected. Enter it manually [solana]: ", "solana").lower()
    report["chain"] = chain

    # ---- 2. GoPlus ----
    print(f"\n  [2/3] GoPlus Security ({chain})...")
    if chain == "solana":
        goplus_raw = fetch(GOPLUS_SOLANA.format(addr=address))
    else:
        cid = EVM_CHAIN_IDS.get(chain)
        if not cid:
            goplus_raw = {"__error__": f"GoPlus has no chain id mapped for {chain!r}"}
        else:
            goplus_raw = fetch(GOPLUS_EVM.format(chain_id=cid, addr=address))

    report["goplus_raw"] = goplus_raw
    goplus_record = {}
    if isinstance(goplus_raw, dict) and "result" in goplus_raw:
        res = goplus_raw.get("result") or {}
        goplus_record = res.get(address) or res.get(address.lower()) or {}
    report["goplus_record"] = goplus_record
    print(f"        {len(goplus_record)} fields returned" if goplus_record else "        NO RECORD")

    # ---- 3. RugCheck (Solana specialist, indexes new launches) ----
    rugcheck_raw = None
    if chain == "solana":
        print("\n  [3/3] RugCheck.xyz (Solana specialist)...")
        rugcheck_raw = fetch(RUGCHECK.format(addr=address))
        report["rugcheck_raw"] = rugcheck_raw
        if isinstance(rugcheck_raw, dict) and "__error__" not in rugcheck_raw:
            print(f"        {len(rugcheck_raw)} fields returned")
        else:
            err = (rugcheck_raw or {}).get("__error__", "no data")
            print(f"        no usable data ({err})")
    else:
        print("\n  [3/3] RugCheck.xyz - skipped (Solana only)")

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- summary ----
    print("\n" + "=" * 68)
    print("  FIELD NAMES EACH SCANNER RETURNED")
    print("=" * 68)

    print("\n  GoPlus:")
    if goplus_record:
        describe_fields(goplus_record)
    else:
        print("    (no record - GoPlus has not indexed this token)")

    if chain == "solana":
        print("\n  RugCheck.xyz:")
        if isinstance(rugcheck_raw, dict) and "__error__" not in rugcheck_raw:
            describe_fields(rugcheck_raw)
        else:
            print("    (no usable response)")

    print("\n" + "=" * 68)
    print(f"  Saved to: {OUTPUT}")
    print("  Share that file to get the filter matched to the real data.")
    print("=" * 68)

    pause_and_exit()


if __name__ == "__main__":
    main()
