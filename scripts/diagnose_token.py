#!/usr/bin/env python3
"""Dump exactly what the security scanners return for one token.

The rug-check filter decides using the raw JSON these APIs send back. When
it rejects a token for "no data returned", this tells you whether the data
is genuinely absent or whether the filter is reading the wrong field names.

Writes the full response to diagnostic_output.json so it can be shared.

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

GOPLUS_SOLANA = "https://api.gopluslabs.io/api/v1/solana/token_security?contract_addresses={addr}"
GOPLUS_EVM = "https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={addr}"
DEXSCREENER = "https://api.dexscreener.com/latest/dex/tokens/{addr}"

IS_WINDOWS = os.name == "nt"


def pause_and_exit(code: int = 0) -> None:
    if IS_WINDOWS:
        input("\nPress Enter to close this window...")
    sys.exit(code)


def fetch(url: str) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "memecoin-bot-diagnostic"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return {"__error__": f"HTTP {exc.code}", "__body__": exc.read().decode("utf-8", "replace")[:2000]}
    except Exception as exc:  # noqa: BLE001
        return {"__error__": str(exc)}


def main() -> None:
    print("=" * 68)
    print("  TOKEN DIAGNOSTIC")
    print("=" * 68)
    print()
    print("  Looks up one token and shows exactly what the security")
    print("  scanners report about it. Nothing is traded.")
    print()

    address = input("  Paste the token contract address: ").strip()
    if not address:
        print("\n  No address given.")
        pause_and_exit()

    chain = (input("  Chain [solana]: ").strip() or "solana").lower()

    report: dict = {"token_address": address, "chain": chain}

    print("\n  Asking GoPlus Security...")
    if chain == "solana":
        goplus_raw = fetch(GOPLUS_SOLANA.format(addr=address))
    else:
        chain_ids = {"ethereum": "1", "eth": "1", "bsc": "56", "polygon": "137",
                     "arbitrum": "42161", "base": "8453", "avalanche": "43114", "optimism": "10"}
        goplus_raw = fetch(GOPLUS_EVM.format(chain_id=chain_ids.get(chain, "1"), addr=address))

    report["goplus_raw"] = goplus_raw

    result = {}
    if isinstance(goplus_raw, dict) and "result" in goplus_raw:
        res = goplus_raw.get("result") or {}
        result = res.get(address) or res.get(address.lower()) or {}
    report["goplus_token_record"] = result

    print("\n  Asking DexScreener (price and pool depth)...")
    dex_raw = fetch(DEXSCREENER.format(addr=address))
    pairs = (dex_raw or {}).get("pairs") or [] if isinstance(dex_raw, dict) else []
    report["dexscreener_pair_count"] = len(pairs)
    if pairs:
        best = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))
        report["dexscreener_best_pair"] = {
            "dex": best.get("dexId"),
            "priceUsd": best.get("priceUsd"),
            "liquidity_usd": (best.get("liquidity") or {}).get("usd"),
            "volume_24h": (best.get("volume") or {}).get("h24"),
        }

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- human-readable summary ----
    print("\n" + "=" * 68)
    print("  WHAT THE SCANNER RETURNED")
    print("=" * 68)

    if not result:
        print("\n  GoPlus returned NO RECORD for this token.")
        print("  Usually means the address is wrong, or the token is too new")
        print("  to have been indexed yet.")
    else:
        print(f"\n  GoPlus returned {len(result)} fields. Field names:\n")
        for key in sorted(result):
            value = result[key]
            kind = type(value).__name__
            if isinstance(value, list):
                detail = f"list with {len(value)} item(s)"
                if value and isinstance(value[0], dict):
                    detail += f", first item keys: {sorted(value[0])}"
            elif isinstance(value, dict):
                detail = f"object with keys: {sorted(value)}"
            else:
                detail = repr(value)[:80]
            print(f"    {key:32} ({kind}) {detail}")

    print("\n  DexScreener:")
    if pairs:
        b = report["dexscreener_best_pair"]
        print(f"    pools found:   {len(pairs)}")
        print(f"    price:         ${b['priceUsd']}")
        print(f"    liquidity:     ${float(b['liquidity_usd'] or 0):,.0f}")
        print(f"    24h volume:    ${float(b['volume_24h'] or 0):,.0f}")
    else:
        print("    no trading pools found for this address")

    print("\n" + "=" * 68)
    print(f"  Full details saved to:\n    {OUTPUT}")
    print("\n  Share that file (or the field list above) to get the filter")
    print("  matched to what this scanner actually returns.")
    print("=" * 68)

    pause_and_exit()


if __name__ == "__main__":
    main()
