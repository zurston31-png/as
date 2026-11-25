"""Corrections entered by hand, each with its source and a check against prices.

Used only where no automated source has the fact. Every entry is reported, and
each is checked against market prices when the batch runs; an entry that the
prices contradict is flagged, not silently applied.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CuratedSpinOff:
    parent_symbol: str
    ex_date: date
    #: (child ticker, child shares received per parent share)
    children: tuple[tuple[str, float], ...]
    reference: str


CURATED_SPIN_OFFS: tuple[CuratedSpinOff, ...] = (
    CuratedSpinOff(
        parent_symbol="UTX",
        ex_date=date(2020, 4, 3),
        children=(("CARR", 1.0), ("OTIS", 0.5)),
        reference=(
            "United Technologies' separation of Carrier and Otis, completed 2020-04-03 with "
            "the Raytheon merger: one Carrier share for each UTC share and one Otis share for "
            "every two UTC shares, as announced by the companies at completion. Entered by hand "
            "because Alpaca has no spin-off record for it and records the day as a 1-for-2 "
            "reverse split instead. Not yet checked against the SEC filing (SEC access pending "
            "a contact address)."
        ),
    ),
)


# =============================================================================
# Facts read from SEC filings by a person, each re-verified on every run
# =============================================================================
#
# Every entry cites SEC documents and exact passages from them. On every run the
# pipeline fetches (or reads from its cache) each document and applies the entry only
# if every passage is found in it, word for word after markup and spacing are
# removed. An entry whose passages are not found is reported and not used.

SEC = "https://www.sec.gov/Archives/edgar/data"


@dataclass(frozen=True)
class SecQuote:
    url: str
    text: str


@dataclass(frozen=True)
class CuratedRename:
    """A ticker change that Alpaca has no corporate action for."""

    old_symbol: str
    new_symbol: str
    #: First session under the new ticker (approximately: links allow +/-10 sessions).
    date: date
    quotes: tuple[SecQuote, ...]
    note: str


CURATED_RENAMES: tuple[CuratedRename, ...] = (
    CuratedRename(
        "ENRJ",
        "UAVS",
        date(2018, 3, 27),
        (
            SecQuote(
                f"{SEC}/8504/000114420418016156/tv488968_ex99-1.htm",
                "name will change to AgEagle Aerial Systems Inc. and the ticker symbol will also "
                "change to",
            ),
            SecQuote(
                f"{SEC}/8504/000114420418016156/tv488968_ex99-1.htm",
                "EnerJex Resources, Inc., (NYSE American: ENRJ)",
            ),
        ),
        "EnerJex Resources merged with AgEagle in March 2018 (reverse merger) and kept its "
        "listing under the new name and ticker UAVS. Same SEC registrant (CIK 8504).",
    ),
    CuratedRename(
        "HSACW",
        "IMVTW",
        date(2019, 12, 19),
        (
            SecQuote(
                f"{SEC}/1764013/000119312519321147/d839382d8k.htm",
                "On December 19, 2019, our common stock, units and warrants began trading on the "
                "Nasdaq Stock Market",
            ),
            SecQuote(
                f"{SEC}/1764013/000119312519321147/d839382d8k.htm",
                "Warrants to receive one half of one share of Common Stock IMVTW",
            ),
        ),
        "Health Sciences Acquisitions Corp (SPAC) became Immunovant; its warrants continued "
        "as IMVTW.",
    ),
    CuratedRename(
        "ITHXW",
        "MONDW",
        date(2022, 7, 19),
        (
            SecQuote(
                f"{SEC}/1828852/000110465922080993/tm2220826d1_ex99-2.htm",
                "Mondee to begin trading on the Nasdaq on July 19, 2022 under ticker symbol MOND "
                "and MONDW",
            ),
        ),
        "ITHAX Acquisition Corp (SPAC) became Mondee Holdings (same registrant, CIK 1828852); "
        "its warrants continued as MONDW.",
    ),
)


@dataclass(frozen=True)
class CuratedIdentityLink:
    """One security that stopped trading for longer than the trading-gap rule allows and
    resumed under a new ticker: a documented exception, applied only when every passage is
    found in its SEC document.

    The two segments are named exactly (the old ticker's last trade, the new ticker's first),
    and ``cik`` is the registrant on both sides. ``audit_merges.py`` checks independently
    that both securities resolve to that issuer, that no delisting notice (Form 25) or
    deregistration (Form 15) was filed in between, and where the price resumed.
    """

    old_symbol: str
    old_last: date
    new_symbol: str
    new_first: date
    cik: str
    quotes: tuple[SecQuote, ...]
    note: str


_YNDX = f"{SEC}/1513845"

CURATED_IDENTITY_LINKS: tuple[CuratedIdentityLink, ...] = (
    CuratedIdentityLink(
        "YNDX",
        date(2022, 2, 25),
        "NBIS",
        date(2024, 10, 21),
        "0001513845",
        (
            SecQuote(
                f"{_YNDX}/000151384522000010/yndx-20220307ex9918dbc01.htm",
                "On February 28, 2022, Nasdaq and the New York Stock Exchange suspended the "
                "trading in securities of a number of companies with material operations in "
                "Russia, including Yandex N.V.",
            ),
            SecQuote(
                f"{_YNDX}/000110465924090313/tm2421897d1_ex99-1.htm",
                "(formerly known as Yandex N.V.) (NASDAQ: YNDX, to be changed to NBIS)",
            ),
            SecQuote(
                f"{_YNDX}/000110465924109545/tm2426436d2_ex99-1.htm",
                "Trading in the Company\u2019s Class A ordinary shares was suspended on February "
                "28, 2022",
            ),
            SecQuote(
                f"{_YNDX}/000110465924109545/tm2426436d2_ex99-1.htm",
                "Following the successful divestment of our Russian assets, which completed the "
                "separation from all ties with Russia, Nasdaq has determined that trading in our "
                "shares will resume.",
            ),
            SecQuote(
                f"{_YNDX}/000110465924110042/tm2426559d1_ex99-1.htm",
                "confirms that the trading of the Company\u2019s Class A ordinary shares on the "
                "Nasdaq Global Select Market is scheduled to resume",
            ),
        ),
        "Yandex N.V. (CIK 1513845): Nasdaq suspended trading in its Class A shares on "
        "2022-02-28; the company renamed itself Nebius Group N.V. and changed its ticker to "
        "NBIS in August 2024; Nasdaq resumed trading in the same Class A ordinary shares on "
        "2024-10-21. Same registrant, same shares, no delisting: one security, suspended for "
        "665 sessions (beyond the 250-session trading-gap rule, hence the exception). During "
        "the suspension the shares could not be traded on a US exchange; the security has no "
        "bars then.",
    ),
)


@dataclass(frozen=True)
class CuratedFact:
    """Type and/or exit evidence for the security that last traded as ``symbol`` on ``last_trade``."""

    symbol: str
    last_trade: date
    security_type: str | None
    #: An exit evidence category (``redeemed``, ``failure``, ...) or None.
    category: str | None
    event_date: date | None
    quotes: tuple[SecQuote, ...]
    note: str
    cash_per_share: float | None = None


_OILD_2016 = f"{SEC}/1580167/000089109216016861/e70900ex99-1.htm"
_OILD_2020 = f"{SEC}/1415311/000119312520073553/d166686dex991.htm"
_SHLD_PLAN = f"{SEC}/1547580/000168386321004879/f9575d1.htm"
_WDRW = f"{SEC}/1424958/000119312516485506/d122095d497k.htm"

CURATED_FACTS: tuple[CuratedFact, ...] = (
    CuratedFact(
        "OILD",
        date(2016, 9, 7),
        "etf",
        "redeemed",
        date(2016, 8, 24),
        (
            SecQuote(_OILD_2016, "Crude Oil Excess Return Fund (OILU and OILD)"),
            SecQuote(_OILD_2016, "has determined to liquidate and terminate each Fund"),
        ),
        "AccuShares S&P GSCI Crude Oil Excess Return Fund, Down shares (exchange-traded "
        "commodity fund), liquidated in 2016. Holders received the net asset value; the "
        "amount is not in the filing.",
    ),
    CuratedFact(
        "OILD",
        date(2020, 3, 27),
        "etf",
        "redeemed",
        date(2020, 3, 15),
        (
            SecQuote(_OILD_2020, "UltraPro 3x Short Crude Oil ETF OILD NYSE Arca"),
            SecQuote(
                _OILD_2020,
                "plans to liquidate and close six ProShares Trust ETFs and two ProShares Trust II "
                "ETFs",
            ),
        ),
        "ProShares UltraPro 3x Short Crude Oil ETF, liquidated in March 2020 at net asset value.",
    ),
    CuratedFact(
        "SHLD",
        date(2021, 8, 17),
        "etf",
        None,
        None,
        (SecQuote(_SHLD_PLAN, "VictoryShares Protect America ETF (SHLD)"),),
        "VictoryShares Protect America ETF (launched November 2020 at about $25), not Sears "
        "Holdings. This is an early stretch of the same fund (a trading gap).",
    ),
    CuratedFact(
        "SHLD",
        date(2021, 10, 8),
        "etf",
        "redeemed",
        date(2021, 10, 15),
        (
            SecQuote(_SHLD_PLAN, "VictoryShares Protect America ETF (SHLD)"),
            SecQuote(_SHLD_PLAN, "approved a Plan of Liquidation"),
            SecQuote(
                _SHLD_PLAN,
                "It is anticipated that the Funds will liquidate on or about October 15, 2021",
            ),
        ),
        "VictoryShares Protect America ETF, liquidated in October 2021 at net asset value.",
    ),
    CuratedFact(
        "WDRW",
        date(2016, 3, 11),
        "etf",
        None,
        None,
        (
            SecQuote(
                _WDRW,
                "Direxion Daily Regional Banks Bear 3X Shares Ticker: WDRW Listed on NYSE Arca",
            ),
        ),
        "Direxion Daily Regional Banks Bear 3X Shares, a leveraged ETF.",
    ),
    CuratedFact(
        "WDRW",
        date(2020, 3, 27),
        "etf",
        None,
        None,
        (
            SecQuote(
                _WDRW,
                "Direxion Daily Regional Banks Bear 3X Shares Ticker: WDRW Listed on NYSE Arca",
            ),
        ),
        "Direxion Daily Regional Banks Bear 3X Shares, a leveraged ETF. Why it stopped trading "
        "in March 2020 was not found.",
    ),
    CuratedFact(
        "IMVTW",
        date(2020, 6, 15),
        "warrant",
        "redeemed",
        date(2020, 6, 15),
        (
            SecQuote(
                f"{SEC}/1764013/000119312520142279/d930766dex992.htm",
                "for a redemption price of $0.01 per Warrant",
            ),
            SecQuote(
                f"{SEC}/1764013/000119312520142279/d930766dex992.htm", "on June 15, 2020 (the"
            ),
        ),
        "Immunovant called its public warrants; unexercised warrants were redeemed at $0.01.",
        cash_per_share=0.01,
    ),
    CuratedFact(
        "JMPNL",
        date(2021, 9, 10),
        "debt",
        "redeemed",
        date(2021, 9, 10),
        (
            SecQuote(
                f"{SEC}/1302350/000143774921019464/ex_274358.htm",
                "The redemption price per 2027 Note will be $25 plus accrued and unpaid interest",
            ),
        ),
        "JMP Group redeemed its 7.25% Senior Notes due 2027 (Nasdaq: JMPNL) at $25 plus accrued interest.",
        cash_per_share=25.0,
    ),
    CuratedFact(
        "SCPX",
        date(2024, 6, 14),
        None,
        "trading_gap",
        date(2024, 8, 2),
        (
            SecQuote(
                f"{SEC}/1476963/000107997324001071/ex99x1.htm",
                "approved a 1-for-200 reverse stock split",
            ),
            SecQuote(
                f"{SEC}/1476963/000107997324001121/ex99x1.htm",
                "has withdrawn its delisting determination and will be lifting the trading suspension",
            ),
        ),
        "Scorpius Holdings: NYSE American suspended trading in June 2024; after a 1-for-200 reverse "
        "split the delisting determination was withdrawn and trading resumed on 2024-08-02. The same "
        "shares continued (not an exit), but they resumed about 66% below the split-adjusted last "
        "price, so a holder lost that across the gap.",
    ),
    CuratedFact(
        "YGF",
        date(2024, 2, 8),
        None,
        "delisted_no_reason",
        date(2024, 4, 9),
        (
            SecQuote(
                f"{SEC}/1875496/000135445724000260/ygfdelistreason.txt",
                "no longer qualified for listing on the Exchange pursuant to Listing Rules "
                "5250(a)(1) and 5250(c)(1)",
            ),
            SecQuote(
                f"{SEC}/1875496/000135445724000260/ygfdelistreason.txt",
                "The Company securities were suspended on March 26, 2024",
            ),
        ),
        "YanGuFang: Nasdaq delisted the shares under rules 5250(a)(1) (information requested by "
        "Nasdaq) and 5250(c)(1) (periodic reports not filed); suspended 2024-03-26. Why the company "
        "stopped reporting, and what shareholders got, is not known: no outcome or value assumed.",
    ),
    CuratedFact(
        "FRC",
        date(2023, 4, 28),
        None,
        "failure",
        date(2023, 5, 1),
        (
            SecQuote(
                f"{SEC}/19617/000001961723000304/jpmc2023exhibit99.htm",
                "assumed the deposits and certain other liabilities of First Republic Bank from "
                "the Federal Deposit Insurance Corporation (FDIC)",
            ),
            SecQuote(
                f"{SEC}/1605607/000095017023017258/pgre-ex99_1.htm",
                "Federal Deposit Insurance Corporation being appointed as receiver of our tenant "
                "First Republic Bank",
            ),
        ),
        "First Republic Bank was closed and the FDIC appointed receiver on 2023-05-01; JPMorgan "
        "bought most assets from the FDIC and did not assume its preferred stock or debt. Stated "
        "in SEC filings by two unrelated companies (JPMorgan Chase, Paramount Group). First "
        "Republic itself filed with the FDIC, not the SEC.",
    ),
    CuratedFact(
        "SMMC",
        date(2019, 10, 29),
        "common",
        None,
        None,
        (
            SecQuote(
                f"{SEC}/1774155/000095014219001603/eh1900929_ex9901.htm",
                "Class A ordinary shares and warrants that are separated will trade on the Nasdaq "
                "Capital Market under the symbols",
            ),
        ),
        "South Mountain Merger Corp. Class A ordinary shares (a SPAC, later Billtrust).",
    ),
)


@dataclass(frozen=True)
class CuratedDeal:
    """Terms of a stock or cash-and-stock acquisition, per share of the target."""

    target_symbol: str
    last_trade: date
    acquirer_symbol: str  # the acquirer's ticker as traded on the effective date
    effective_date: date
    cash_per_share: float
    shares_per_share: float
    quotes: tuple[SecQuote, ...]
    note: str


CURATED_DEALS: tuple[CuratedDeal, ...] = (
    CuratedDeal(
        "ALXN",
        date(2021, 7, 20),
        "AZN",
        date(2021, 7, 21),
        60.0,
        2.1243,
        (
            SecQuote(
                f"{SEC}/899866/000110465921094624/tm2122733d25_8k.htm",
                "converted into the right to receive (1) 2.1243 American depositary shares of AstraZeneca",
            ),
            SecQuote(
                f"{SEC}/899866/000110465921094624/tm2122733d25_8k.htm",
                "and (2) $60 in cash, without interest",
            ),
        ),
        "AstraZeneca: $60 cash plus 2.1243 AstraZeneca ADSs (Nasdaq: AZN).",
    ),
    CuratedDeal(
        "CELG",
        date(2019, 11, 20),
        "BMY",
        date(2019, 11, 20),
        50.0,
        1.0,
        (
            SecQuote(
                f"{SEC}/816284/000110465919065939/tm1923405d1_8k.htm",
                "converted into the right to receive (1) $50.00 in cash, without interest, (2) one share of "
                "Bristol-Myers Squibb common stock and (3) one tradeable contingent value right",
            ),
        ),
        "Bristol-Myers Squibb: $50 cash plus one BMY share plus one CVR. The CVR (a $9 payment "
        "contingent on FDA approvals; it later expired unpaid) is not valued, so the value is a "
        "lower bound at the effective date.",
    ),
    CuratedDeal(
        "DISCA",
        date(2022, 4, 8),
        "WBD",
        date(2022, 4, 8),
        0.0,
        1.0,
        (
            SecQuote(
                f"{SEC}/1437107/000119312522103051/d328161d8k.htm",
                "each share of Discovery Series A common stock issued and outstanding or held by Discovery as "
                "treasury stock immediately prior to such effective time was automatically reclassified and "
                "converted into one share of WBD common stock",
            ),
        ),
        "Discovery acquired WarnerMedia; each DISCA share became one Warner Bros. Discovery share. "
        "WBD first traded on 2022-04-11, the first session after the effective date.",
    ),
    CuratedDeal(
        "MXIM",
        date(2021, 8, 25),
        "ADI",
        date(2021, 8, 26),
        0.0,
        0.63,
        (
            SecQuote(
                f"{SEC}/743316/000119312521257603/d203144d8k.htm",
                "was converted into the right to receive 0.6300 (the",
            ),
        ),
        "Analog Devices: 0.63 ADI share.",
    ),
    CuratedDeal(
        "PXD",
        date(2024, 5, 2),
        "XOM",
        date(2024, 5, 3),
        0.0,
        2.3234,
        (
            SecQuote(
                f"{SEC}/1038357/000119312524130052/d803313d8k.htm",
                "each then-outstanding share of Pioneer Common Stock was converted into the right to receive "
                "2.3234 shares of ExxonMobil Common Stock",
            ),
        ),
        "ExxonMobil: 2.3234 XOM shares.",
    ),
    CuratedDeal(
        "TWX",
        date(2018, 6, 14),
        "T",
        date(2018, 6, 14),
        53.75,
        1.437,
        (
            SecQuote(
                f"{SEC}/1105705/000095015718000694/form8-k.htm",
                "converted into the right to receive 1.437 shares (the",
            ),
            SecQuote(f"{SEC}/1105705/000095015718000694/form8-k.htm", "plus $53.75 in cash"),
        ),
        "AT&T: $53.75 cash plus 1.437 AT&T shares.",
    ),
    CuratedDeal(
        "VMW",
        date(2023, 11, 21),
        "AVGO",
        date(2023, 11, 22),
        142.50,
        0.0,
        (
            SecQuote(
                f"{SEC}/1124610/000119312523266257/d573017dex991.htm",
                "elected to receive the cash consideration or did not make a valid election or did not deliver a "
                "valid election form prior to the Election Deadline. Each such VMware stockholder will be entitled to "
                "receive $142.50 in cash per share of VMware common stock",
            ),
            SecQuote(
                f"{SEC}/1124610/000119312523266257/d573017dex991.htm",
                "elected to receive the stock consideration and in accordance with the proration procedures",
            ),
        ),
        "Broadcom: holders chose $142.50 cash or 0.252 AVGO share. About 96% chose stock and were prorated; "
        "shares that elected cash or made no election received $142.50 in cash. Shares bought in the market "
        "after the election deadline could no longer elect, so a holder at the last trade received $142.50 "
        "cash (the last close, 142.48, agrees). A holder who elected stock in time received about 52% stock.",
    ),
    CuratedDeal(
        "WLL",
        date(2022, 6, 30),
        "OAS",
        date(2022, 7, 1),
        6.25,
        0.5774,
        (
            SecQuote(
                f"{SEC}/1255474/000119312522189511/d306292d8k.htm",
                "was converted into the right to receive $6.25 in cash, without interest, and 0.5774 shares of "
                "Chord Common Stock",
            ),
        ),
        "Oasis Petroleum (renamed Chord Energy; traded as OAS on the effective date, as CHRD from "
        "2022-07-05): $6.25 cash plus 0.5774 shares.",
    ),
    CuratedDeal(
        "XLNX",
        date(2022, 2, 11),
        "AMD",
        date(2022, 2, 14),
        0.0,
        1.7234,
        (
            SecQuote(
                f"{SEC}/743988/000110465922021762/tm226258d1_8k.htm",
                "was converted into the right to receive 1.7234 shares (the",
            ),
        ),
        "AMD: 1.7234 AMD shares.",
    ),
    CuratedDeal(
        "DO",
        date(2024, 9, 3),
        "NE",
        date(2024, 9, 4),
        5.65,
        0.2316,
        (
            SecQuote(
                f"{SEC}/949039/000119312524212705/d886651d8k.htm",
                "(a) $5.65 in cash, without interest, payable to the holder of each share of Diamond Offshore "
                "Common Stock",
            ),
            SecQuote(f"{SEC}/949039/000119312524212705/d886651d8k.htm", "(b) 0.2316 (the"),
        ),
        "Noble: $5.65 cash plus 0.2316 Noble A ordinary shares.",
    ),
    CuratedDeal(
        "FSBC",
        date(2020, 4, 30),
        "EVBN",
        date(2020, 5, 1),
        8.90,
        0.2197,
        (
            SecQuote(
                f"{SEC}/1667939/000094337420000234/form8k.htm",
                "either $17.80 in cash or 0.4394 shares of common stock, par value $0.50, of Evans",
            ),
            SecQuote(
                f"{SEC}/1667939/000119312520071400/d868021ddefm14a.htm",
                "that 50% of the total number of shares of FSB common stock will be converted into the stock "
                "consideration and 50% of such shares of FSB common stock will be converted into the cash "
                "consideration",
            ),
        ),
        "Evans Bancorp: election of $17.80 cash or 0.4394 EVBN share, allocated 50/50. Value used: "
        "the average per share, $8.90 plus 0.2197 EVBN share.",
    ),
    CuratedDeal(
        "FLAC",
        date(2022, 11, 22),
        "NAMS",
        date(2022, 11, 22),
        0.0,
        1.0,
        (
            SecQuote(
                f"{SEC}/1828326/000119312522292061/d426932d8k.htm",
                "against the issuance of a corresponding Holdco Share",
            ),
        ),
        "Frazier Lifesciences Acquisition (SPAC) combined with NewAmsterdam Pharma: one NewAmsterdam "
        "share per FLAC share. NAMS first traded on 2022-11-23.",
    ),
    # Stage 1.1: members that ended with no exit value in Stage 1. Terms from the target's
    # own 8-K on completion of the acquisition.
    CuratedDeal(
        "AGN",
        date(2020, 5, 8),
        "ABBV",
        date(2020, 5, 8),
        120.30,
        0.8660,
        (
            SecQuote(
                f"{SEC}/1578845/000119312520137717/d927229d8k.htm",
                "was cancelled and automatically converted into the right to receive (i) $120.30 in "
                "cash (and any cash in lieu of fractions of AbbVie Shares) and (ii) 0.8660 of a share "
                "of AbbVie common stock",
            ),
            SecQuote(
                f"{SEC}/1578845/000119312520137717/d927229d8k.htm",
                "On May 8, 2020, the Court Order was registered by the Registrar of Companies, at "
                "which time the Scheme became effective.",
            ),
        ),
        "AbbVie acquired Allergan plc by scheme of arrangement, effective 2020-05-08: $120.30 "
        "cash plus 0.8660 AbbVie share (NYSE: ABBV) per Allergan share. AGN traded on 2020-05-08; "
        "trading was suspended before the open on 2020-05-11.",
    ),
    CuratedDeal(
        "WORK",
        date(2021, 7, 20),
        "CRM",
        date(2021, 7, 21),
        26.79,
        0.0776,
        (
            SecQuote(
                f"{SEC}/1764925/000119312521220229/d205267d8k.htm",
                "was converted into the right to receive 0.0776 shares of Salesforce common stock "
                "and $26.79 in cash, without interest",
            ),
            SecQuote(
                f"{SEC}/1764925/000119312521220229/d205267d8k.htm",
                "on July 21, 2021 (the \u201cClosing Date\u201d)",
            ),
        ),
        "Salesforce acquired Slack Technologies on 2021-07-21: $26.79 cash plus 0.0776 "
        "Salesforce share (NYSE: CRM) per Slack share.",
    ),
    CuratedDeal(
        "RTN",
        date(2020, 4, 2),
        "RTX",
        date(2020, 4, 3),
        0.0,
        2.3348,
        (
            SecQuote(
                f"{SEC}/1047122/000119312520097237/d827106d8k.htm",
                "was converted into the right to receive 2.3348 (the \u201c Exchange Ratio \u201d) "
                "fully paid and nonassessable shares of UTC common stock",
            ),
            SecQuote(
                f"{SEC}/1047122/000119312520097237/d827106d8k.htm",
                "UTC changed its name to Raytheon Technologies Corporation",
            ),
            SecQuote(
                f"{SEC}/1047122/000119312520097237/d827106d8k.htm",
                "requested that trading in Raytheon Common Stock be suspended prior to the opening "
                "of trading on April 3, 2020",
            ),
        ),
        "Raytheon Company merged with United Technologies on 2020-04-03 (all stock): 2.3348 "
        "UTC shares per Raytheon share. UTC renamed itself Raytheon Technologies and traded as "
        "RTX from that day, after spinning off Carrier and Otis to its own holders of record "
        "(Raytheon holders did not receive them), so RTX's close that day is the value.",
    ),
)


#: Common stocks with no SEC registrant, and why (so no SIC can be read for them).
CURATED_NO_SEC_ISSUER: dict[tuple[str, date], str] = {
    ("FRC", date(2023, 4, 28)): (
        "First Republic Bank was a bank without a holding company; its shares were registered "
        "with the FDIC, not the SEC (it filed its reports with the FDIC). A bank is not a "
        "blank-check company, so the SIC test does not apply."
    ),
    ("SBNY", date(2023, 3, 10)): (
        "Signature Bank was a New York state-chartered bank without a holding company; its "
        "shares were registered with the FDIC, not the SEC. A bank is not a blank-check "
        "company, so the SIC test does not apply."
    ),
}
