"""Trading calendar.

The calendar is checked against dates that can be verified independently
(real exchange holidays and early closes), not against itself, because a
rule-derived calendar that is self-consistently wrong looks perfect under
round-trip tests.

Fixtures are local rather than shared: these specs pin the exact RTH
windows under test, and a later edit to a shared NQ fixture must not be
able to change what a session-boundary assertion means.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from flow_model.config.schema import SessionFilterConfig
from flow_model.core.enums import InstrumentType, Session
from flow_model.core.instruments import InstrumentSpec, SessionWindow
from flow_model.data.base import SchemaError, SessionCalendarProtocol
from flow_model.config.loader import default_config
from flow_model.data.calendar import (
    HALF_DAY_CLOSE_MINUTES,
    JUNETEENTH_FIRST_YEAR,
    MONDAY,
    THURSDAY,
    TRADABLE_REASON,
    TRADABLE_REASONS,
    TradingCalendar,
    easter_sunday,
    good_friday,
    last_weekday,
    nth_weekday,
    observed,
    us_half_days,
    us_market_holidays,
)

ET = ZoneInfo("America/New_York")
SPAN_YEARS = range(2010, 2036)

RTH_BUCKETS = (Session.RTH_OPEN, Session.RTH_MID, Session.RTH_CLOSE)


def et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """An exchange-local wall time, returned as the UTC instant it names.

    Tests are written in exchange-local terms because that is how session
    boundaries are defined; converting here means the assertions stay
    correct across DST instead of encoding one offset.
    """
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


@pytest.fixture
def cal() -> TradingCalendar:
    return TradingCalendar()


@pytest.fixture
def nq() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="NQ",
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
        eth=SessionWindow(name="ETH", start="18:00", end="17:00"),
    )


@pytest.fixture
def gc() -> InstrumentSpec:
    """Gold: a different RTH, which is why RTH boundaries are not constants."""
    return InstrumentSpec(
        symbol="GC",
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.10,
        tick_value=10.0,
        rth=SessionWindow(name="RTH", start="08:20", end="13:30"),
        eth=SessionWindow(name="ETH", start="18:00", end="17:00"),
    )


@pytest.fixture
def qqq() -> InstrumentSpec:
    """Cash-hours ETF: an RTH window but no overnight session."""
    return InstrumentSpec(
        symbol="QQQ",
        instrument_type=InstrumentType.ETF,
        tick_size=0.01,
        tick_value=0.01,
        rth=SessionWindow(name="RTH", start="09:30", end="16:00"),
    )


@pytest.fixture
def spx() -> InstrumentSpec:
    """Reference instrument with no declared session windows at all."""
    return InstrumentSpec(
        symbol="SPX",
        instrument_type=InstrumentType.INDEX,
        tick_size=0.01,
        tick_value=0.01,
        tradable=False,
    )


# --- Easter / Good Friday --------------------------------------------------


@pytest.mark.parametrize(
    "year,expected",
    [
        (2010, date(2010, 4, 4)),
        (2020, date(2020, 4, 12)),
        (2021, date(2021, 4, 4)),
        (2022, date(2022, 4, 17)),
        (2023, date(2023, 4, 9)),
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2035, date(2035, 3, 25)),
    ],
)
def test_easter_sunday_known_dates(year, expected):
    assert easter_sunday(year) == expected


def test_easter_is_always_a_sunday():
    assert all(easter_sunday(y).weekday() == 6 for y in SPAN_YEARS)


@pytest.mark.parametrize(
    "year,expected",
    [
        (2021, date(2021, 4, 2)),
        (2022, date(2022, 4, 15)),
        (2023, date(2023, 4, 7)),
        (2024, date(2024, 3, 29)),
        (2025, date(2025, 4, 18)),
    ],
)
def test_good_friday_known_dates(year, expected):
    assert good_friday(year) == expected


def test_good_friday_is_always_a_friday_two_days_before_easter():
    for year in SPAN_YEARS:
        gf = good_friday(year)
        assert gf.weekday() == 4
        assert easter_sunday(year) - gf == timedelta(days=2)


# --- weekday arithmetic ----------------------------------------------------


@pytest.mark.parametrize(
    "year,month,weekday,n,expected",
    [
        (2021, 1, MONDAY, 3, date(2021, 1, 18)),  # MLK 2021
        (2024, 2, MONDAY, 3, date(2024, 2, 19)),  # Washington's Birthday
        (2024, 11, THURSDAY, 4, date(2024, 11, 28)),  # Thanksgiving
        (2024, 9, MONDAY, 1, date(2024, 9, 2)),  # Labor Day
        (2024, 1, MONDAY, 5, date(2024, 1, 29)),  # a month with five Mondays
    ],
)
def test_nth_weekday(year, month, weekday, n, expected):
    assert nth_weekday(year, month, weekday, n) == expected


def test_nth_weekday_rejects_a_nonexistent_occurrence():
    # February 2023 has four Thursdays; clamping to the fourth would invent
    # a holiday on a date the exchange traded.
    with pytest.raises(ValueError, match="no 5th weekday"):
        nth_weekday(2023, 2, THURSDAY, 5)
    # The leap-year neighbour does have a fifth, so the guard is not just
    # rejecting every n=5.
    assert nth_weekday(2024, 2, THURSDAY, 5) == date(2024, 2, 29)


@pytest.mark.parametrize("n", [0, 6, -1])
def test_nth_weekday_rejects_out_of_range_n(n):
    with pytest.raises(ValueError, match="n must be in 1..5"):
        nth_weekday(2024, 1, MONDAY, n)


def test_nth_weekday_rejects_out_of_range_weekday():
    with pytest.raises(ValueError, match="weekday must be in 0..6"):
        nth_weekday(2024, 1, 7, 1)


@pytest.mark.parametrize(
    "year,month,weekday,expected",
    [
        (2024, 5, MONDAY, date(2024, 5, 27)),  # Memorial Day 2024
        (2021, 5, MONDAY, date(2021, 5, 31)),  # Memorial Day 2021
        (2024, 2, 4, date(2024, 2, 23)),  # last Friday of a leap February
        (2024, 12, 1, date(2024, 12, 31)),  # last Tuesday of December
    ],
)
def test_last_weekday(year, month, weekday, expected):
    assert last_weekday(year, month, weekday) == expected


# --- observance ------------------------------------------------------------


@pytest.mark.parametrize(
    "nominal,expected",
    [
        (date(2021, 7, 4), date(2021, 7, 5)),  # Sunday -> following Monday
        (date(2020, 7, 4), date(2020, 7, 3)),  # Saturday -> preceding Friday
        (date(2024, 7, 4), date(2024, 7, 4)),  # Thursday -> unchanged
    ],
)
def test_observed_shifts_weekend_holidays(nominal, expected):
    assert observed(nominal) == expected


# --- holiday sets ----------------------------------------------------------


@pytest.mark.parametrize(
    "year,day",
    [
        (2024, date(2024, 1, 1)),  # New Year's Day
        (2021, date(2021, 1, 18)),  # MLK 2021
        (2024, date(2024, 2, 19)),  # Washington's Birthday
        (2024, date(2024, 3, 29)),  # Good Friday
        (2024, date(2024, 5, 27)),  # Memorial Day
        (2024, date(2024, 6, 19)),  # Juneteenth
        (2024, date(2024, 7, 4)),  # Independence Day
        (2024, date(2024, 9, 2)),  # Labor Day
        (2024, date(2024, 11, 28)),  # Thanksgiving
        (2024, date(2024, 12, 25)),  # Christmas
    ],
)
def test_us_market_holidays_contains_known_dates(year, day):
    assert day in us_market_holidays(year)


@pytest.mark.parametrize(
    "year,day",
    [
        (2021, date(2021, 7, 5)),  # Jul 4 fell on Sunday
        (2020, date(2020, 7, 3)),  # Jul 4 fell on Saturday
        (2023, date(2023, 1, 2)),  # Jan 1 fell on Sunday
        (2021, date(2021, 12, 24)),  # Dec 25 fell on Saturday
        (2022, date(2022, 12, 26)),  # Dec 25 fell on Sunday
        (2022, date(2022, 6, 20)),  # Jun 19 fell on Sunday
    ],
)
def test_weekend_observance_shifts_the_observed_date(year, day):
    assert day in us_market_holidays(year)


@pytest.mark.parametrize("year", [2011, 2022, 2028, 2033])
def test_saturday_new_year_does_not_close_the_preceding_friday(year):
    """The one exception to uniform weekend observance.

    When 1 January falls on a Saturday the US exchanges stay OPEN the
    preceding Friday, because that Friday is the last session of the prior
    year. Applying the uniform Saturday-shifts-back rule would wrongly
    close four real trading days in the 2010-2035 span.
    """
    assert date(year, 1, 1).weekday() == 5          # precondition: a Saturday
    friday = date(year - 1, 12, 31)
    assert friday not in us_market_holidays(year)
    assert friday not in us_market_holidays(year - 1)
    spec = default_config().spec("NQ")
    assert TradingCalendar().is_trading_day(friday, spec)


def test_sunday_new_year_still_shifts_forward():
    """Only the Saturday case is exceptional; Sunday observance is normal."""
    assert date(2023, 1, 1).weekday() == 6
    assert date(2023, 1, 2) in us_market_holidays(2023)


def test_other_saturday_holidays_still_shift_back():
    """The exception is specific to New Year's Day, not to Saturdays."""
    assert date(2020, 7, 4).weekday() == 5
    assert date(2020, 7, 3) in us_market_holidays(2020)
    assert date(2021, 12, 25).weekday() == 5
    assert date(2021, 12, 24) in us_market_holidays(2021)


@pytest.mark.parametrize("year", [2010, 2015, 2021])
def test_juneteenth_absent_before_first_observed_year(year):
    assert not any(d.month == 6 for d in us_market_holidays(year))


@pytest.mark.parametrize(
    "year,day",
    [
        (2022, date(2022, 6, 20)),
        (2023, date(2023, 6, 19)),
        (2024, date(2024, 6, 19)),
        (2025, date(2025, 6, 19)),
    ],
)
def test_juneteenth_present_from_first_observed_year(year, day):
    assert JUNETEENTH_FIRST_YEAR == 2022
    assert day in us_market_holidays(year)


@pytest.mark.parametrize("year", list(SPAN_YEARS))
def test_holiday_count_is_stable_per_era(year):
    expected = 10 if year >= JUNETEENTH_FIRST_YEAR else 9
    assert len(us_market_holidays(year)) == expected


def test_observed_holidays_are_weekdays_except_a_saturday_new_year():
    """A Saturday 1 January stays on 1 January rather than shifting back.

    It lands on a weekend, which is harmless -- the market is closed anyway
    -- and it is what keeps the preceding Friday open.
    """
    for year in SPAN_YEARS:
        for day in us_market_holidays(year):
            if day.weekday() >= 5:
                assert (day.month, day.day) == (1, 1), f"{day} is an unexpected weekend holiday"
                assert day.weekday() == 5, f"{day} should only ever be a Saturday"
            else:
                assert day.weekday() < 5


# --- half days -------------------------------------------------------------


@pytest.mark.parametrize(
    "year,day",
    [
        (2024, date(2024, 11, 29)),  # day after Thanksgiving
        (2024, date(2024, 12, 24)),  # Christmas Eve on a Tuesday
        (2024, date(2024, 7, 3)),  # Jul 3 on a Wednesday
        (2023, date(2023, 7, 3)),  # Jul 3 on a Monday
        (2023, date(2023, 11, 24)),
        (2021, date(2021, 11, 26)),
    ],
)
def test_us_half_days_contains_known_early_closes(year, day):
    assert day in us_half_days(year)


@pytest.mark.parametrize(
    "year,day,why",
    [
        (2021, date(2021, 12, 24), "it is the observed Christmas holiday"),
        (2020, date(2020, 7, 3), "it is the observed Independence Day"),
        (2022, date(2022, 12, 24), "Christmas Eve fell on a Saturday"),
        (2023, date(2023, 12, 24), "Christmas Eve fell on a Sunday"),
        (2021, date(2021, 7, 3), "Jul 3 fell on a Saturday"),
    ],
)
def test_us_half_days_excludes_holidays_and_weekends(year, day, why):
    assert day not in us_half_days(year), why


def test_half_days_and_holidays_are_disjoint():
    for year in SPAN_YEARS:
        assert not us_half_days(year) & us_market_holidays(year)


def test_no_half_day_falls_on_a_weekend():
    for year in SPAN_YEARS:
        assert all(d.weekday() < 5 for d in us_half_days(year))


# --- day classification ----------------------------------------------------


@pytest.mark.parametrize(
    "day,expected",
    [
        (date(2024, 3, 5), True),  # ordinary Tuesday
        (date(2024, 3, 9), False),  # Saturday
        (date(2024, 3, 10), False),  # Sunday
        (date(2024, 3, 29), False),  # Good Friday
        (date(2024, 11, 28), False),  # Thanksgiving
        (date(2024, 11, 29), True),  # half day still trades
    ],
)
def test_is_trading_day(cal, nq, day, expected):
    assert cal.is_trading_day(day, nq) is expected


@pytest.mark.parametrize(
    "day,expected",
    [
        (date(2024, 11, 29), True),
        (date(2024, 12, 24), True),
        (date(2024, 3, 5), False),
        (date(2024, 11, 28), False),  # a full holiday is not a half day
        (date(2024, 3, 9), False),  # a weekend is not a half day
    ],
)
def test_is_half_day(cal, nq, day, expected):
    assert cal.is_half_day(day, nq) is expected


def test_queries_outside_the_precomputed_span_still_follow_the_rules(nq):
    narrow = TradingCalendar(first_year=2024, last_year=2024)
    assert narrow.is_trading_day(date(2040, 12, 25), nq) is False
    assert narrow.is_trading_day(date(2040, 12, 26), nq) is True
    assert date(2040, 12, 25) not in narrow.holidays  # outside the eager span


def test_span_properties_cover_the_configured_years():
    narrow = TradingCalendar(first_year=2024, last_year=2024)
    assert date(2024, 11, 28) in narrow.holidays
    assert date(2024, 11, 29) in narrow.half_days
    assert date(2023, 11, 23) not in narrow.holidays


def test_first_year_after_last_year_is_rejected():
    with pytest.raises(ValueError, match="exceeds last_year"):
        TradingCalendar(first_year=2030, last_year=2020)


def test_extra_holidays_and_half_days_are_honoured(nq):
    cal = TradingCalendar(
        extra_holidays={date(2024, 3, 5)},
        extra_half_days={date(2024, 3, 6)},
    )
    assert cal.is_trading_day(date(2024, 3, 5), nq) is False
    assert cal.is_half_day(date(2024, 3, 6), nq) is True
    assert cal.rth_bounds(date(2024, 3, 5), nq) is None


def test_extra_holidays_reject_datetimes():
    # datetime subclasses date, so a datetime would type-check and then
    # match nothing: the injected holiday would silently not exist.
    with pytest.raises(TypeError, match="must contain date objects"):
        TradingCalendar(extra_holidays={datetime(2024, 3, 5, tzinfo=timezone.utc)})


# --- session buckets -------------------------------------------------------


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (18, 0, Session.ASIA),
        (23, 30, Session.ASIA),
        (2, 59, Session.ASIA),
        (3, 0, Session.LONDON),
        (7, 59, Session.LONDON),
        (8, 0, Session.PRE_RTH),
        (9, 29, Session.PRE_RTH),
        (9, 30, Session.RTH_OPEN),
        (10, 29, Session.RTH_OPEN),
        (10, 30, Session.RTH_MID),
        (14, 59, Session.RTH_MID),
        (15, 0, Session.RTH_CLOSE),
        (15, 59, Session.RTH_CLOSE),
        (16, 0, Session.POST_RTH),
        (16, 59, Session.POST_RTH),
        (17, 0, Session.CLOSED),
        (17, 59, Session.CLOSED),
    ],
)
def test_session_buckets_are_half_open(cal, nq, hour, minute, expected):
    assert cal.session_of(et(2024, 3, 5, hour, minute), nq) is expected


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (8, 19, Session.PRE_RTH),
        (8, 20, Session.RTH_OPEN),
        (9, 19, Session.RTH_OPEN),
        (9, 20, Session.RTH_MID),
        (12, 29, Session.RTH_MID),
        (12, 30, Session.RTH_CLOSE),
        (13, 29, Session.RTH_CLOSE),
        (13, 30, Session.POST_RTH),
        (16, 30, Session.POST_RTH),
    ],
)
def test_gc_buckets_follow_its_own_rth(cal, gc, hour, minute, expected):
    # 09:30 is mid-session for gold; a hardcoded equity open would label
    # every gold bar from 08:20 to 09:30 as pre-market.
    assert cal.session_of(et(2024, 3, 5, hour, minute), gc) is expected


def test_dst_pair_maps_to_different_utc_but_the_same_session(cal, nq):
    winter = et(2024, 1, 16, 9, 30)
    summer = et(2024, 7, 16, 9, 30)
    assert winter.astimezone(timezone.utc).hour == 14
    assert summer.astimezone(timezone.utc).hour == 13
    assert cal.session_of(winter, nq) is Session.RTH_OPEN
    assert cal.session_of(summer, nq) is Session.RTH_OPEN


@pytest.mark.parametrize(
    "ts_args",
    [(2024, 3, 29, 10, 0), (2024, 3, 9, 10, 0), (2024, 11, 28, 10, 0)],
)
def test_session_is_closed_on_non_trading_days(cal, nq, ts_args):
    assert cal.session_of(et(*ts_args), nq) is Session.CLOSED


def test_half_day_shifts_the_close_buckets(cal, nq):
    # Early close at 13:00, so the final hour of the session is 12:00-13:00
    # and 14:00 is already after the close.
    assert cal.session_of(et(2024, 11, 29, 11, 59), nq) is Session.RTH_MID
    assert cal.session_of(et(2024, 11, 29, 12, 30), nq) is Session.RTH_CLOSE
    assert cal.session_of(et(2024, 11, 29, 13, 0), nq) is Session.POST_RTH
    assert cal.session_of(et(2024, 11, 29, 14, 0), nq) is Session.POST_RTH


def test_instrument_without_rth_is_never_bucketed(cal, spx):
    for hour in (2, 10, 15, 20):
        assert cal.session_of(et(2024, 3, 5, hour), spx) is Session.CLOSED
    assert cal.is_rth(et(2024, 3, 5, 10), spx) is False
    assert cal.rth_bounds(date(2024, 3, 5), spx) is None
    assert cal.minutes_into_rth(et(2024, 3, 5, 10), spx) is None
    assert cal.minutes_to_rth_close(et(2024, 3, 5, 10), spx) is None


def test_short_session_drops_the_middle_bucket(cal):
    short = InstrumentSpec(
        symbol="X",
        instrument_type=InstrumentType.FUTURE,
        tick_size=0.25,
        tick_value=5.0,
        rth=SessionWindow(name="RTH", start="09:30", end="11:00"),
    )
    assert cal.session_of(et(2024, 3, 5, 10, 29), short) is Session.RTH_OPEN
    assert cal.session_of(et(2024, 3, 5, 10, 30), short) is Session.RTH_CLOSE
    assert cal.session_of(et(2024, 3, 5, 10, 59), short) is Session.RTH_CLOSE
    assert cal.session_of(et(2024, 3, 5, 11, 0), short) is Session.POST_RTH


def test_is_rth_agrees_with_session_of_across_a_whole_day(cal, nq):
    ts = et(2024, 3, 5, 0, 0)
    for _ in range(24 * 60):
        in_rth = cal.is_rth(ts, nq)
        assert in_rth is (cal.session_of(ts, nq) in RTH_BUCKETS)
        ts += timedelta(minutes=1)


# --- session date ----------------------------------------------------------


@pytest.mark.parametrize(
    "ts_args,expected",
    [
        ((2024, 3, 5, 10, 0), date(2024, 3, 5)),  # cash hours
        ((2024, 3, 4, 18, 30), date(2024, 3, 5)),  # ETH open rolls the date
        ((2024, 3, 4, 17, 59), date(2024, 3, 4)),  # before the roll
        ((2024, 3, 5, 2, 0), date(2024, 3, 5)),  # after midnight, same trade date
        ((2024, 3, 10, 18, 30), date(2024, 3, 11)),  # Sunday evening -> Monday
    ],
)
def test_session_date_rolls_at_the_eth_open(cal, nq, ts_args, expected):
    assert cal.session_date(et(*ts_args), nq) == expected


def test_eth_bar_after_midnight_belongs_to_the_same_trade_date(cal, nq):
    evening = et(2024, 3, 4, 18, 30)
    overnight = et(2024, 3, 5, 2, 0)
    # The evening bar is still 2024-03-04 in UTC, the overnight bar is
    # 2024-03-05; both belong to the 2024-03-05 session.
    assert evening.astimezone(timezone.utc).date() == date(2024, 3, 4)
    assert cal.session_date(evening, nq) == cal.session_date(overnight, nq)
    assert cal.session_date(evening, nq) == date(2024, 3, 5)


def test_session_date_is_not_rolled_over_non_trading_days(cal, nq):
    friday_evening = et(2024, 3, 8, 18, 30)
    assert cal.session_date(friday_evening, nq) == date(2024, 3, 9)
    assert cal.is_trading_day(date(2024, 3, 9), nq) is False
    assert cal.session_of(friday_evening, nq) is Session.CLOSED


def test_session_date_does_not_roll_without_a_wrapping_eth_window(cal, qqq):
    assert cal.session_date(et(2024, 3, 4, 18, 30), qqq) == date(2024, 3, 4)
    assert cal.session_date(et(2024, 3, 4, 23, 30), qqq) == date(2024, 3, 4)


def test_wrapping_eth_window_spans_midnight(nq):
    assert nq.eth.wraps_midnight is True
    assert nq.eth.contains_minutes(23 * 60) is True
    assert nq.eth.contains_minutes(2 * 60) is True
    assert nq.eth.contains_minutes(17 * 60 + 30) is False


# --- RTH bounds and offsets ------------------------------------------------


def test_rth_bounds_are_utc_instants_that_track_dst(cal, nq):
    winter = cal.rth_bounds(date(2024, 1, 16), nq)
    summer = cal.rth_bounds(date(2024, 7, 16), nq)
    assert winter == (
        datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 16, 21, 0, tzinfo=timezone.utc),
    )
    assert summer == (
        datetime(2024, 7, 16, 13, 30, tzinfo=timezone.utc),
        datetime(2024, 7, 16, 20, 0, tzinfo=timezone.utc),
    )


def test_rth_bounds_use_the_early_close_on_a_half_day(cal, nq):
    open_ts, close_ts = cal.rth_bounds(date(2024, 11, 29), nq)
    assert close_ts.astimezone(ET).hour * 60 == HALF_DAY_CLOSE_MINUTES
    assert (close_ts - open_ts) == timedelta(hours=3, minutes=30)


def test_rth_bounds_none_on_non_trading_days(cal, nq):
    assert cal.rth_bounds(date(2024, 11, 28), nq) is None
    assert cal.rth_bounds(date(2024, 3, 9), nq) is None


def test_gc_rth_bounds_come_from_its_own_window(cal, gc):
    open_ts, close_ts = cal.rth_bounds(date(2024, 3, 5), gc)
    assert open_ts.astimezone(ET).strftime("%H:%M") == "08:20"
    assert close_ts.astimezone(ET).strftime("%H:%M") == "13:30"


@pytest.mark.parametrize(
    "hour,minute,expected",
    [(9, 30, 0.0), (10, 0, 30.0), (15, 59, 389.0)],
)
def test_minutes_into_rth(cal, nq, hour, minute, expected):
    assert cal.minutes_into_rth(et(2024, 3, 5, hour, minute), nq) == expected


@pytest.mark.parametrize("ts_args", [(2024, 3, 5, 9, 29), (2024, 3, 5, 16, 0), (2024, 3, 5, 20, 0)])
def test_minutes_into_rth_is_none_outside_the_session(cal, nq, ts_args):
    # None rather than a negative offset: a feature that only inspects the
    # magnitude would read a pre-open bar as a late-session bar.
    assert cal.minutes_into_rth(et(*ts_args), nq) is None


@pytest.mark.parametrize(
    "hour,minute,expected",
    [(9, 30, 390.0), (15, 50, 10.0), (15, 59, 1.0)],
)
def test_minutes_to_rth_close(cal, nq, hour, minute, expected):
    assert cal.minutes_to_rth_close(et(2024, 3, 5, hour, minute), nq) == expected


def test_minutes_to_close_counts_to_the_early_close_on_a_half_day(cal, nq):
    # 16:00 would be three hours past the real close; a flatten rule using
    # it would hold a position through a market that is not trading.
    assert cal.minutes_to_rth_close(et(2024, 11, 29, 12, 50), nq) == 10.0
    assert cal.minutes_to_rth_close(et(2024, 11, 29, 13, 0), nq) is None


def test_minutes_are_fractional_for_sub_minute_timestamps(cal, nq):
    ts = et(2024, 3, 5, 9, 30) + timedelta(seconds=30)
    assert cal.minutes_into_rth(ts, nq) == 0.5


# --- tradability -----------------------------------------------------------


def test_is_tradable_accepts_an_ordinary_rth_bar(cal, nq):
    assert cal.is_tradable(et(2024, 3, 5, 11, 0), nq, SessionFilterConfig()) == (
        True,
        TRADABLE_REASON,
    )


@pytest.mark.parametrize(
    "ts_args,expected_reason",
    [
        ((2024, 3, 9, 11, 0), "weekend"),
        ((2024, 3, 29, 11, 0), "holiday"),
        ((2024, 11, 29, 11, 0), "half_day"),
        ((2024, 3, 5, 17, 30), "maintenance_break"),
        ((2024, 3, 5, 20, 0), "not_rth"),
        ((2024, 3, 5, 8, 30), "not_rth"),
        ((2024, 3, 5, 9, 34), "within_open_skip"),
        ((2024, 3, 5, 15, 50), "within_close_skip"),
    ],
)
def test_is_tradable_rejection_reasons_are_distinguishable(cal, nq, ts_args, expected_reason):
    ok, reason = cal.is_tradable(et(*ts_args), nq, SessionFilterConfig())
    assert ok is False
    assert reason == expected_reason


def test_every_reason_is_declared(cal, nq):
    cfg = SessionFilterConfig()
    reasons = {
        cal.is_tradable(et(2024, 3, 5, h, m), nq, cfg)[1]
        for h in range(24)
        for m in (0, 30)
    }
    assert reasons <= TRADABLE_REASONS


def test_skip_window_boundaries_are_half_open(cal, nq):
    cfg = SessionFilterConfig(skip_first_minutes=5.0, skip_last_minutes=10.0)
    assert cal.is_tradable(et(2024, 3, 5, 9, 34), nq, cfg)[1] == "within_open_skip"
    assert cal.is_tradable(et(2024, 3, 5, 9, 35), nq, cfg)[0] is True
    assert cal.is_tradable(et(2024, 3, 5, 15, 49), nq, cfg)[0] is True
    assert cal.is_tradable(et(2024, 3, 5, 15, 50), nq, cfg)[1] == "within_close_skip"


def test_skip_windows_are_configurable(cal, nq):
    wide = SessionFilterConfig(skip_first_minutes=60.0, skip_last_minutes=60.0)
    assert cal.is_tradable(et(2024, 3, 5, 10, 29), nq, wide)[1] == "within_open_skip"
    assert cal.is_tradable(et(2024, 3, 5, 15, 0), nq, wide)[1] == "within_close_skip"
    assert cal.is_tradable(et(2024, 3, 5, 12, 0), nq, wide)[0] is True

    none = SessionFilterConfig(skip_first_minutes=0.0, skip_last_minutes=0.0)
    assert cal.is_tradable(et(2024, 3, 5, 9, 30), nq, none)[0] is True
    assert cal.is_tradable(et(2024, 3, 5, 15, 59), nq, none)[0] is True


def test_trade_rth_only_false_admits_the_overnight_session(cal, nq):
    cfg = SessionFilterConfig(trade_rth_only=False)
    assert cal.is_tradable(et(2024, 3, 5, 20, 0), nq, cfg)[0] is True
    # The maintenance break is still not tradable: there is no session.
    assert cal.is_tradable(et(2024, 3, 5, 17, 30), nq, cfg)[1] == "maintenance_break"


def test_half_day_close_skip_uses_the_real_close(cal, nq):
    cfg = SessionFilterConfig(skip_half_days=False, skip_last_minutes=10.0)
    assert cal.is_tradable(et(2024, 11, 29, 12, 51), nq, cfg)[1] == "within_close_skip"
    assert cal.is_tradable(et(2024, 11, 29, 12, 49), nq, cfg)[0] is True
    assert cal.is_tradable(et(2024, 11, 29, 14, 0), nq, cfg)[1] == "not_rth"


def test_skip_holidays_false_still_finds_no_regular_session(cal, nq):
    cfg = SessionFilterConfig(skip_holidays=False)
    ok, reason = cal.is_tradable(et(2024, 3, 29, 11, 0), nq, cfg)
    assert (ok, reason) == (False, "not_rth")


def test_instrument_without_rth_is_not_tradable(cal, spx):
    assert cal.is_tradable(et(2024, 3, 5, 11, 0), spx, SessionFilterConfig()) == (
        False,
        "no_rth_window",
    )


def test_weekend_rejection_precedes_the_holiday_filter(cal, nq):
    cfg = SessionFilterConfig(skip_holidays=False, skip_half_days=False)
    assert cal.is_tradable(et(2024, 3, 9, 11, 0), nq, cfg)[1] == "weekend"


# --- purity and contract ---------------------------------------------------


def test_satisfies_the_session_calendar_protocol(cal):
    assert isinstance(cal, SessionCalendarProtocol)


@pytest.mark.parametrize(
    "method",
    ["session_of", "is_rth", "session_date", "minutes_into_rth", "minutes_to_rth_close"],
)
def test_naive_timestamps_are_rejected(cal, nq, method):
    # Assuming UTC for a naive exchange timestamp shifts every session
    # boundary by the exchange offset, which is invisible downstream.
    with pytest.raises(SchemaError, match="naive datetime"):
        getattr(cal, method)(datetime(2024, 3, 5, 10, 0), nq)


def test_is_tradable_rejects_naive_timestamps(cal, nq):
    with pytest.raises(SchemaError, match="naive datetime"):
        cal.is_tradable(datetime(2024, 3, 5, 10, 0), nq, SessionFilterConfig())


def test_non_datetime_is_rejected(cal, nq):
    with pytest.raises(SchemaError, match="expected datetime"):
        cal.session_of("2024-03-05T10:00:00Z", nq)


def test_two_calendars_classify_identically(nq):
    """Determinism: construction order and instance identity cannot matter,
    or a backtest and a live run would disagree about the same instant."""
    a, b = TradingCalendar(), TradingCalendar()
    ts = et(2024, 11, 25, 0, 0)
    for _ in range(7 * 24 * 4):
        assert a.session_of(ts, nq) is b.session_of(ts, nq)
        assert a.session_date(ts, nq) == b.session_date(ts, nq)
        assert a.is_tradable(ts, nq, SessionFilterConfig()) == b.is_tradable(
            ts, nq, SessionFilterConfig()
        )
        ts += timedelta(minutes=15)


def test_repeated_calls_return_identical_results(cal, nq):
    ts = et(2024, 3, 5, 10, 0)
    first = [cal.session_of(ts, nq), cal.session_date(ts, nq), cal.minutes_into_rth(ts, nq)]
    second = [cal.session_of(ts, nq), cal.session_date(ts, nq), cal.minutes_into_rth(ts, nq)]
    assert first == second


def test_calendar_module_reads_no_clock_and_draws_no_randomness():
    """Enforced by inspection rather than by sampling.

    A calendar that consults `today` classifies a fixed historical
    timestamp differently depending on when the backtest ran, and a
    calendar that draws from an unseeded generator breaks bit-identical
    reproduction. Neither shows up as a failing assertion on any single
    timestamp, so the module itself is audited.
    """
    import ast
    import pathlib

    import flow_model.data.calendar as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    clock_names = {"today", "now", "utcnow", "time", "monotonic", "perf_counter"}
    found = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in clock_names
    }
    assert not found, f"clock access makes the calendar time-dependent: {sorted(found)}"

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if "random" in name or name == "time"}
