"""SIP universe research: identity, ticker history, splits, exits and membership.

SYNTHETIC TEST FIXTURES ONLY. Tickers are fake (TESTA, TESTB, ...) and every
price is invented to reproduce a situation seen in Alpaca's real data:
filler bars after a delisting, a ticker reused by another company, a rename,
a bankruptcy followed by a new listing under the same ticker.
"""

from datetime import date

import pytest

from app.data.validation import trading_days
from app.universe import segments as seg
from app.universe.batch import draw_random
from app.universe.membership import DailyRow, MembershipRule, decide, month_end_sessions
from app.universe.pipeline import classify_exit

DAYS = trading_days(date(2023, 1, 3), date(2023, 12, 29))


def bars(days, price=10.0, volume=1000, trades=10, step=0.0):
    return [
        seg.Bar(
            d,
            price + i * step,
            price + i * step,
            price + i * step,
            price + i * step,
            volume,
            price + i * step,
            trades,
        )
        for i, d in enumerate(days)
    ]


def fillers(days, price):
    return [seg.Bar(d, price, price, price, price, 0, price, 0) for d in days]


# =============================================================================
# Filler bars and segments
# =============================================================================


def test_filler_bars_are_never_prices_and_a_long_silence_ends_a_listing():
    # Company 1 trades, the ticker goes silent for 100 sessions of filler bars, company 2 starts.
    series = bars(DAYS[:50], 11.85) + fillers(DAYS[50:150], 11.85) + bars(DAYS[150:200], 42.51)

    segments, dropped = seg.split_segments("TESTA", series)

    assert dropped == 100
    assert [(s.first_session, s.last_session) for s in segments] == [
        (DAYS[0], DAYS[49]),
        (DAYS[150], DAYS[199]),
    ]
    assert all(not bar.is_filler for s in segments for bar in s.bars)


def test_a_short_trading_halt_does_not_split_a_listing():
    series = bars(DAYS[:20]) + fillers(DAYS[20:28], 10.0) + bars(DAYS[28:40])
    segments, _ = seg.split_segments("TESTA", series)
    assert len(segments) == 1


def test_eleven_silent_sessions_split_ten_do_not():
    ten = bars(DAYS[:5]) + bars(DAYS[15:20])  # ten sessions between DAYS[4] and DAYS[15]
    eleven = bars(DAYS[:5]) + bars(DAYS[16:20])
    assert len(seg.split_segments("TESTA", ten)[0]) == 1
    assert len(seg.split_segments("TESTA", eleven)[0]) == 2


# =============================================================================
# Linking renames into one security
# =============================================================================


def segs(**series):
    return {symbol: seg.split_segments(symbol, items)[0] for symbol, items in series.items()}


def test_a_rename_joins_two_tickers_into_one_security():
    by_symbol = segs(TESTA=bars(DAYS[:100], 50.0), TESTB=bars(DAYS[100:200], 51.0))
    rename = seg.NameChange("e1", "TESTA", "TESTB", DAYS[100])

    (chain,) = seg.build_chains(by_symbol, [rename])

    assert [s.symbol for s in chain.segments] == ["TESTA", "TESTB"]
    assert chain.links == ["e1"]
    assert (chain.first_session, chain.last_session, chain.current_symbol) == (
        DAYS[0],
        DAYS[199],
        "TESTB",
    )


def test_a_reused_ticker_belongs_to_two_securities():
    # Like META: an ETF trades as TESTA, renames to TESTE; later company TESTF renames to TESTA.
    by_symbol = segs(
        TESTA=bars(DAYS[:60], 15.0) + fillers(DAYS[60:120], 12.31) + bars(DAYS[120:200], 184.0),
        TESTE=bars(DAYS[60:200], 12.3),
        TESTF=bars(DAYS[:120], 330.0),
    )
    events = [
        seg.NameChange("e1", "TESTA", "TESTE", DAYS[60]),
        seg.NameChange("e2", "TESTF", "TESTA", DAYS[120]),
    ]

    chains = seg.build_chains(by_symbol, events)

    histories = sorted([[(s.symbol, s.first_session) for s in c.segments] for c in chains])
    assert histories == sorted(
        [
            [("TESTA", DAYS[0]), ("TESTE", DAYS[60])],
            [("TESTF", DAYS[0]), ("TESTA", DAYS[120])],
        ]
    )


def test_bankruptcy_and_new_listing_under_the_same_ticker_are_two_securities():
    # Like CHK: old shares stop, filler bars, a new company lists under the same ticker.
    # No event links them.
    by_symbol = segs(
        TESTA=bars(DAYS[:50], 11.85) + fillers(DAYS[50:150], 11.85) + bars(DAYS[150:], 42.51)
    )

    chains = seg.build_chains(by_symbol, [])

    assert [(c.first_session, c.last_session) for c in chains] == [
        (DAYS[0], DAYS[49]),
        (DAYS[150], DAYS[-1]),
    ]


def test_a_rename_dated_after_the_switch_still_links():
    # Like BOWX -> WE: the new ticker trades from DAYS[100], the event is dated four sessions later.
    by_symbol = segs(TESTA=bars(DAYS[:100], 10.0), TESTB=bars(DAYS[100:200], 10.4))

    (chain,) = seg.build_chains(by_symbol, [seg.NameChange("e1", "TESTA", "TESTB", DAYS[104])])

    assert [s.symbol for s in chain.segments] == ["TESTA", "TESTB"]


def test_a_rename_does_not_bridge_a_silence_that_ends_a_listing():
    by_symbol = segs(TESTA=bars(DAYS[:90]), TESTB=bars(DAYS[105:200]))
    chains = seg.build_chains(by_symbol, [seg.NameChange("e1", "TESTA", "TESTB", DAYS[98])])
    assert len(chains) == 2


def test_duplicate_rename_events_make_one_link_and_far_events_none():
    by_symbol = segs(TESTA=bars(DAYS[:100]), TESTB=bars(DAYS[100:200]), TESTC=bars(DAYS[:50]))
    events = [
        seg.NameChange("e1", "TESTA", "TESTB", DAYS[100]),
        seg.NameChange("e2", "TESTA", "TESTB", DAYS[101]),  # duplicate record
        seg.NameChange("e3", "TESTC", "TESTB", DAYS[150]),  # TESTC stopped 100 sessions earlier
    ]
    chains = seg.build_chains(by_symbol, events)
    linked = [c for c in chains if len(c.segments) > 1]
    assert len(linked) == 1
    assert linked[0].links == ["e1"]
    assert len(chains) == 2


def test_overlapping_tickers_are_not_joined():
    by_symbol = segs(TESTA=bars(DAYS[:100]), TESTB=bars(DAYS[90:200]))
    chains = seg.build_chains(by_symbol, [seg.NameChange("e1", "TESTA", "TESTB", DAYS[100])])
    assert len(chains) == 2


# =============================================================================
# Split adjustment
# =============================================================================


def test_split_factors_follow_the_security_across_its_tickers():
    by_symbol = segs(TESTA=bars(DAYS[:100], 400.0), TESTB=bars(DAYS[100:200], 100.0))
    (chain,) = seg.build_chains(by_symbol, [seg.NameChange("e1", "TESTA", "TESTB", DAYS[100])])
    splits = [
        seg.Split("TESTA", DAYS[50], 4.0),  # 4-for-1 while it was TESTA
        seg.Split("TESTB", DAYS[150], 0.5),  # 1-for-2 reverse split after the rename
        seg.Split("TESTZ", DAYS[20], 10.0),  # another security: ignored
    ]

    factors = seg.split_factors(chain, splits)

    assert factors[DAYS[0]] == pytest.approx(2.0)  # 4 x 0.5
    assert factors[DAYS[50]] == pytest.approx(0.5)  # on the ex-date the first split already applies
    assert factors[DAYS[149]] == pytest.approx(0.5)
    assert factors[DAYS[150]] == pytest.approx(1.0)


def test_a_split_recorded_under_the_old_ticker_on_the_rename_date_belongs_to_the_security():
    # Like UTX -> RTX on 2020-04-03: the split is recorded for UTX, dated the first RTX session.
    by_symbol = segs(TESTA=bars(DAYS[:100], 60.0), TESTB=bars(DAYS[100:200], 120.0))
    (chain,) = seg.build_chains(by_symbol, [seg.NameChange("e1", "TESTA", "TESTB", DAYS[100])])

    factors = seg.split_factors(chain, [seg.Split("TESTA", DAYS[100], 0.5)])

    assert factors[DAYS[99]] == 0.5
    assert factors[DAYS[100]] == 1.0


def test_a_split_recorded_under_the_new_ticker_on_its_first_day_belongs_to_the_security():
    # Like FI -> XPRO on 2021-10-04: a 1-for-6 reverse split recorded for XPRO on its first session.
    by_symbol = segs(TESTA=bars(DAYS[:100], 3.0), TESTB=bars(DAYS[100:200], 18.0))
    (chain,) = seg.build_chains(by_symbol, [seg.NameChange("e1", "TESTA", "TESTB", DAYS[100])])

    factors = seg.split_factors(chain, [seg.Split("TESTB", DAYS[100], 1 / 6)])

    assert factors[DAYS[99]] == pytest.approx(1 / 6)  # adjusted = raw / factor = 6 x raw
    assert factors[DAYS[100]] == 1.0


def test_a_split_on_the_first_day_of_a_security_is_not_applied_to_anything():
    (chain,) = seg.build_chains(segs(TESTA=bars(DAYS[:50])), [])
    assert set(seg.split_factors(chain, [seg.Split("TESTA", DAYS[0], 2.0)]).values()) == {1.0}


def test_a_split_of_a_reused_ticker_does_not_leak_into_the_other_company():
    by_symbol = segs(TESTA=bars(DAYS[:50]) + fillers(DAYS[50:150], 10) + bars(DAYS[150:]))
    first, second = seg.build_chains(by_symbol, [])
    split = [seg.Split("TESTA", DAYS[200], 2.0)]  # belongs to the second listing only
    assert set(seg.split_factors(first, split).values()) == {1.0}
    assert seg.split_factors(second, split)[DAYS[150]] == 2.0


# =============================================================================
# Exit classification
# =============================================================================


class Event:
    def __init__(self, action_type, process_date, **fields):
        self.id = f"{action_type}-{process_date}"
        self.action_type, self.process_date = action_type, process_date
        for name in (
            "acquiree_symbol",
            "acquirer_symbol",
            "old_symbol",
            "new_symbol",
            "rate",
            "cash_rate",
        ):
            setattr(self, name, fields.get(name))


def chain_ending(symbol, last_index):
    return seg.build_chains(segs(**{symbol: bars(DAYS[: last_index + 1])}), [])[0]


@pytest.mark.parametrize(
    ("events", "expected_type", "expected_value"),
    [
        (
            [Event("cash_merger", DAYS[101], acquiree_symbol="TESTA", rate=54.2)],
            "cash_merger",
            54.2,
        ),
        (
            [Event("stock_merger", DAYS[101], acquiree_symbol="TESTA", acquirer_symbol="TESTB")],
            "stock_merger",
            None,
        ),
        (
            [Event("name_change", DAYS[103], old_symbol="TESTA", new_symbol="TESTAQ")],
            "bankruptcy_otc",
            0.0,
        ),
        ([], "unexplained", 0.0),
        (
            [Event("cash_merger", DAYS[140], acquiree_symbol="TESTA", rate=9.0)],
            "unexplained",
            0.0,
        ),  # too far
        (
            [Event("cash_merger", DAYS[101], acquiree_symbol="TESTZ", rate=9.0)],
            "unexplained",
            0.0,
        ),  # other ticker
    ],
)
def test_exit_classification(events, expected_type, expected_value):
    status, exit_type, _, _, _, value, why = classify_exit(chain_ending("TESTA", 100), events, DAYS)
    assert (status, exit_type, value) == ("ended", expected_type, expected_value)
    assert why


def test_a_security_trading_at_the_end_of_the_data_has_no_exit():
    status, exit_type, *_ = classify_exit(chain_ending("TESTA", len(DAYS) - 3), [], DAYS)
    assert (status, exit_type) == ("trading", "none")


# =============================================================================
# Point-in-time membership
# =============================================================================

CAL = trading_days(date(2022, 1, 3), date(2023, 12, 29))
RULE = MembershipRule(size=2, min_price=5.0, min_history=20, window=10, min_active_sessions=8)
COMMON = dict.fromkeys(range(1, 20), "common")


def rows(days, price, volume):
    return [DailyRow(d, price, volume) for d in days]


def test_members_are_the_most_traded_eligible_securities():
    history = {
        1: rows(CAL[:300], 50.0, 1000),  # 50,000 per day
        2: rows(CAL[:300], 20.0, 1000),  # 20,000
        3: rows(CAL[:300], 10.0, 9000),  # 90,000
        4: rows(CAL[:300], 4.0, 1_000_000),  # raw price below $5
        5: rows(CAL[285:300], 100.0, 9000),  # history too short (15 < 20 sessions)
        6: rows(CAL[:300], 99.0, 99_000),  # an ETF
    }
    types = {1: "common", 2: "common", 3: "common", 4: "common", 5: "common", 6: "etf"}
    decisions = {d.security_id: d for d in decide(CAL[299], history, types, RULE, CAL[:300])}

    assert {sid for sid, d in decisions.items() if d.member} == {3, 1}
    assert [decisions[3].rank, decisions[1].rank, decisions[2].rank] == [1, 2, 3]
    assert decisions[4].reason == "raw_price_below_minimum"
    assert decisions[5].reason == "history_too_short"
    assert decisions[6].reason == "not_common_stock"


def test_securities_of_unknown_type_are_not_eligible():
    history = {1: rows(CAL[:300], 50.0, 9000), 2: rows(CAL[:300], 20.0, 1000)}

    decisions = {
        d.security_id: d for d in decide(CAL[299], history, {2: "common"}, RULE, CAL[:300])
    }

    assert (decisions[1].eligible, decisions[1].reason) == (False, "not_common_stock")
    assert decisions[2].member


def test_a_security_held_out_for_review_is_not_chosen_and_the_reason_is_recorded():
    history = {1: rows(CAL[:300], 50.0, 9000), 2: rows(CAL[:300], 20.0, 1000)}

    decisions = {
        d.security_id: d
        for d in decide(
            CAL[299], history, COMMON, RULE, CAL[:300], {1: "price_adjustment_under_review"}
        )
    }

    assert (decisions[1].member, decisions[1].reason) == (False, "price_adjustment_under_review")
    assert decisions[2].member


def test_a_delisted_security_is_a_member_until_it_stops_trading_and_never_after():
    history = {1: rows(CAL[:150], 50.0, 9000), 2: rows(CAL[:400], 20.0, 1000)}
    months = month_end_sessions(CAL[40], CAL[399])

    member_months = {
        when: {
            d.security_id
            for d in decide(when, history, COMMON, RULE, [s for s in CAL if s <= when])
            if d.member
        }
        for when in months
    }

    assert all(1 in m for when, m in member_months.items() if when <= CAL[149])
    assert all(1 not in m for when, m in member_months.items() if when > CAL[149])


def test_no_decision_depends_on_a_later_bar():
    history = {i: rows(CAL[: 200 + 30 * i], 10.0 + i, 1000 * (10 - i)) for i in range(1, 8)}
    for when in month_end_sessions(CAL[30], CAL[-1]):
        calendar = [s for s in CAL if s <= when]
        full = decide(when, history, COMMON, RULE, calendar)
        cut = decide(
            when,
            {k: [r for r in v if r.session <= when] for k, v in history.items()},
            COMMON,
            RULE,
            calendar,
        )
        changed = {
            k: [
                r if r.session <= when else DailyRow(r.session, r.close * 50, r.volume * 50)
                for r in v
            ]
            for k, v in history.items()
        }
        assert full == cut == decide(when, changed, COMMON, RULE, calendar)


def test_month_end_sessions_are_the_last_trading_day_of_each_month():
    ends = month_end_sessions(date(2024, 1, 1), date(2024, 4, 30))
    assert ends == [
        date(2024, 1, 31),
        date(2024, 2, 29),
        date(2024, 3, 28),
        date(2024, 4, 30),
    ]  # 29 March: Good Friday


def test_rule_fingerprint_changes_with_any_number():
    assert RULE.fingerprint() == MembershipRule(**RULE.to_dict()).fingerprint()
    assert RULE.fingerprint() != MembershipRule(**{**RULE.to_dict(), "size": 3}).fingerprint()


# =============================================================================
# Random draw
# =============================================================================


def test_random_draw_is_reproducible_and_skips_known_and_code_like_symbols():
    events = [
        {"action_type": "cash_merger", "process_date": f"2022-0{m}-01", "acquiree_symbol": s}
        for m, s in enumerate(["TESTA", "TESTB", "TESTC", "123456789", "TESTD"], start=1)
    ]
    first = draw_random(events, {"TESTB"}, quotas={"cash_merger_acquiree": 2}, seed=7)
    again = draw_random(events, {"TESTB"}, quotas={"cash_merger_acquiree": 2}, seed=7)

    assert first == again
    assert len(first) == 2
    assert {s for s, _, _ in first} <= {"TESTA", "TESTC", "TESTD"}
