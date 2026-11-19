"""The validation batch: known cases chosen by hand plus a seeded random draw.

The batch is a TEST SET for the identity, ticker-history, corporate-action,
price-adjustment and membership logic. It is not a universe and nothing in it
was chosen for its performance.

* ``KNOWN_CASES`` are listed by hand because each one exercises a specific
  situation (a failure, a buyout, a rename, a reused ticker, a re-listing after
  bankruptcy). The expected outcome is written next to each, so the results
  can be checked against what really happened.
* ``draw_random`` adds tickers drawn with a fixed seed from Alpaca's historical
  corporate-action events: companies that were acquired, merged or renamed in
  2020-2024, and companies that split in 2016-2018. They come from historical
  records, not from today's list of tickers.
"""

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: Symbols of common shares and ETFs (1 to 5 letters). Excludes CUSIP-like codes.
PLAIN_TICKER = re.compile(r"^[A-Z]{1,5}$")

RANDOM_SEED = 20261119


@dataclass(frozen=True)
class KnownCase:
    symbol: str
    category: str
    declared_type: str
    expected: str


KNOWN_CASES: tuple[KnownCase, ...] = (
    # --- Overlap with the IEX Milestone 1 data (comparison of feeds) ---------------------------
    *(
        KnownCase(s, "active", "common", "active through 2024; same company as in the IEX data")
        for s in ("AAPL", "MSFT", "NVDA", "AMD", "AMZN", "TSLA", "GOOGL")
    ),
    KnownCase(
        "META",
        "renamed",
        "common",
        "Facebook as FB until 2022-06-08, META from 2022-06-09; "
        "the ticker META belonged to an ETF from 2021-06 to 2022-01",
    ),
    KnownCase("SPY", "active", "etf", "ETF; excluded from membership"),
    KnownCase("QQQ", "active", "etf", "ETF; excluded from membership"),
    # --- Failed / bankrupt ----------------------------------------------------------------------
    KnownCase("SIVB", "failed", "common", "bank failure; last exchange trade 2023-03-09"),
    KnownCase("SBNY", "failed", "common", "bank failure March 2023"),
    KnownCase("FRC", "failed", "common", "bank failure; last exchange trade late April 2023"),
    KnownCase("BBBY", "bankrupt", "common", "bankruptcy; delisted 2023-05"),
    KnownCase("JCP", "bankrupt", "common", "J.C. Penney bankruptcy 2020"),
    KnownCase(
        "FTR", "bankrupt", "common", "Frontier bankruptcy 2020; re-emerged as a new company FYBR"
    ),
    KnownCase(
        "CHK",
        "bankrupt",
        "common",
        "Chesapeake bankruptcy 2020; NEW company listed as CHK 2021-02; renamed EXE 2024-10",
    ),
    KnownCase(
        "HTZ",
        "bankrupt",
        "common",
        "Hertz bankruptcy 2020; NEW company listed as HTZ 2021-11. "
        "Old shareholders were in fact paid out: a known exception to total loss",
    ),
    KnownCase(
        "WE",
        "bankrupt",
        "common",
        "WeWork: SPAC BOWX renamed WE 2021-10; bankruptcy 2023-11 (WEWKQ)",
    ),
    KnownCase("RAD", "bankrupt", "common", "Rite Aid bankruptcy 2023"),
    KnownCase("PRTY", "bankrupt", "common", "Party City bankruptcy 2023"),
    KnownCase("YELL", "bankrupt", "common", "Yellow Corp bankruptcy 2023"),
    KnownCase("REV", "bankrupt", "common", "Revlon bankruptcy 2022"),
    KnownCase("EXPR", "bankrupt", "common", "Express bankruptcy 2024"),
    KnownCase("BIG", "bankrupt", "common", "Big Lots bankruptcy 2024"),
    KnownCase("SAVE", "bankrupt", "common", "Spirit Airlines bankruptcy 2024-11"),
    KnownCase("ENDP", "bankrupt", "common", "Endo bankruptcy 2022"),
    KnownCase(
        "DO", "bankrupt", "common", "Diamond Offshore bankruptcy 2020; new company listed 2021"
    ),
    KnownCase(
        "WLL", "bankrupt", "common", "Whiting bankruptcy 2020; new WLL 2020; merged into Chord 2022"
    ),
    KnownCase(
        "CRC", "bankrupt", "common", "California Resources bankruptcy 2020; new CRC listed 2020-10"
    ),
    KnownCase(
        "SHLD",
        "bankrupt",
        "common",
        "Sears Holdings bankruptcy 2018 (before dense corporate actions)",
    ),
    # --- Acquired ------------------------------------------------------------------------------
    KnownCase("TWTR", "acquired_cash", "common", "taken private at $54.20, 2022-10-28"),
    KnownCase("ATVI", "acquired_cash", "common", "acquired by Microsoft at $95, 2023-10-13"),
    KnownCase("CTXS", "acquired_cash", "common", "taken private 2022-09"),
    KnownCase("SPLK", "acquired_cash", "common", "acquired by Cisco 2024-03"),
    KnownCase("SGEN", "acquired_cash", "common", "acquired by Pfizer 2023-12"),
    KnownCase("HZNP", "acquired_cash", "common", "acquired by Amgen 2023-10"),
    KnownCase("CERN", "acquired_cash", "common", "acquired by Oracle 2022-06"),
    KnownCase("MLNX", "acquired_cash", "common", "acquired by NVIDIA at $125, 2020-04-27"),
    KnownCase("XLNX", "acquired_stock", "common", "acquired by AMD in stock, 2022-02-14"),
    KnownCase("MXIM", "acquired_stock", "common", "acquired by Analog Devices in stock, 2021-08"),
    KnownCase("PXD", "acquired_stock", "common", "acquired by Exxon in stock, 2024-05"),
    KnownCase(
        "ALXN", "acquired_stock_cash", "common", "acquired by AstraZeneca, cash and stock, 2021-07"
    ),
    KnownCase(
        "VMW", "acquired_stock_cash", "common", "acquired by Broadcom, cash and stock, 2023-11"
    ),
    KnownCase(
        "LNKD",
        "acquired_cash",
        "common",
        "acquired by Microsoft 2016-12 (before dense corporate actions)",
    ),
    KnownCase(
        "MON",
        "acquired_cash",
        "common",
        "acquired by Bayer 2018-06 (before dense corporate actions)",
    ),
    KnownCase(
        "TWX",
        "acquired_stock_cash",
        "common",
        "acquired by AT&T 2018-06 (before dense corporate actions)",
    ),
    KnownCase(
        "RHT", "acquired_cash", "common", "acquired by IBM 2019-07 (before dense corporate actions)"
    ),
    KnownCase(
        "CELG",
        "acquired_stock_cash",
        "common",
        "acquired by Bristol-Myers 2019-11 (before dense corporate actions)",
    ),
    # --- Renamed (and SPAC mergers) ------------------------------------------------------------
    KnownCase(
        "UTX",
        "renamed",
        "common",
        "United Technologies renamed RTX after the Raytheon merger, 2020-04",
    ),
    KnownCase("FISV", "renamed", "common", "Fiserv renamed FI, 2023"),
    KnownCase("ANTM", "renamed", "common", "Anthem renamed ELV (Elevance), 2022-06"),
    KnownCase(
        "DISCA", "renamed", "common", "Discovery merged with WarnerMedia and became WBD, 2022-04"
    ),
    KnownCase("METV", "ticker_reuse", "etf", "the ETF that traded as META until 2022-01-28"),
)


def draw_random(
    events: Iterable[dict], already: set[str], *, quotas: dict[str, int], seed: int = RANDOM_SEED
) -> list[tuple[str, str, str]]:
    """Draw tickers from historical corporate-action events with a fixed seed.

    ``events`` are dicts with ``action_type``, ``process_date`` and the symbol
    fields; ``quotas`` maps a source name to how many tickers to draw from it.
    Returns (symbol, source, reason) tuples, in a stable order.
    """
    sources: dict[str, list[tuple[str, str]]] = {name: [] for name in quotas}
    for event in events:
        kind, when = event["action_type"], str(event.get("process_date") or "")
        if kind == "cash_merger" and "2020" <= when[:4] <= "2024":
            sources.get("cash_merger_acquiree", []).append((event.get("acquiree_symbol"), when))
        elif kind == "stock_merger" and "2020" <= when[:4] <= "2024":
            sources.get("stock_merger_acquiree", []).append((event.get("acquiree_symbol"), when))
        elif kind == "name_change" and "2020" <= when[:4] <= "2024":
            sources.get("name_change_old_symbol", []).append((event.get("old_symbol"), when))
        elif kind in ("forward_split", "reverse_split") and "2016" <= when[:4] <= "2018":
            sources.get("split_2016_2018", []).append((event.get("symbol"), when))
    rng = random.Random(seed)
    picked: list[tuple[str, str, str]] = []
    taken = set(already)
    for name in sorted(quotas):
        pool = sorted({(s, d) for s, d in sources[name] if s and PLAIN_TICKER.match(s)})
        rng.shuffle(pool)
        count = 0
        for symbol, when in pool:
            if count >= quotas[name]:
                break
            if symbol in taken:
                continue
            taken.add(symbol)
            picked.append((symbol, name, f"random draw from {name} events (event date {when})"))
            count += 1
    return picked
