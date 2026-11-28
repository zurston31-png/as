#!/usr/bin/env python3
# ruff: noqa: E501
"""Completeness-proof margin at every decision, and how much a defective day could eat into it.

The completeness proof (scripts/audit_completeness.py, section 5) keeps a ticker without daily
data out of the universe because its upper bound, 2 x sum(monthly high x volume over the decision
month and the three before) / 63, is below the N-th member's median. Alpaca's monthly bars are
sums of the daily bars, so a month holding a defective day (2021-04-19, 2021-10-25, 2022-03-08)
can understate a ticker's volume by up to one day.

For every decision and every ticker without daily data, this reports:
- ratio: bound / N-th member (the proof needs < 1);
- days: the extra high x volume needed to reach the N-th member, in days of that ticker's average
  daily high x volume over the four months. A ticker whose missing day was smaller than that
  cannot cross, whatever that day was.
- a tighter bound that is still strict: the same formula over only the months the 63-session
  window touches (the window lies inside them), plus an allowance of ALLOWANCE_DAYS average days
  of that month for every month holding a defective day. The largest loss measured on the 503
  bars Alpaca repaired for 2021-04-19 was 2.74 days (missing_day_size_2021-04-19.txt).

    python research/results/phase1h/scripts/completeness_margins.py DB
"""

import sqlite3
import sys
from collections import defaultdict
from datetime import date

WINDOW, MIN_PRICE = 63, 5.0
ALLOWANCE_DAYS = 3.0
DEFECT_MONTHS = {
    date(2021, 4, 1): "2021-04-19",
    date(2021, 10, 1): "2021-10-25",
    date(2022, 3, 1): "2022-03-08",
}


def back(m: date, k: int) -> date:
    y, mo = divmod(m.year * 12 + m.month - 1 - k, 12)
    return date(y, mo + 1, 1)


def main() -> int:
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    monthly = defaultdict(dict)
    for sym, month, high, volume in db.execute(
        "select symbol, month, high, volume from monthly_bars"
    ):
        monthly[sym][date.fromisoformat(str(month)[:10])] = (high, volume)
    sessions = defaultdict(int)
    calendar = [
        date.fromisoformat(s)
        for (s,) in db.execute("select distinct session from security_bars order by 1")
    ]
    for c in calendar:
        sessions[date(c.year, c.month, 1)] += 1
    index = {c: i for i, c in enumerate(calendar)}
    threshold = {
        date.fromisoformat(str(d)[:10]): v
        for d, v in db.execute(
            "select decision_session, min(median_dollar_volume) from membership where member group by decision_session"
        )
    }
    left_out = [
        s for (s,) in db.execute("select symbol from candidates where selection = 'not_selected'")
    ]
    print(
        f"Completeness-proof margins in {sys.argv[1]} ({len(left_out)} tickers without daily data, {len(threshold)} decisions)"
    )
    print(
        "days = extra high x volume needed to reach the N-th member, in days of the ticker's average over the 4 months\n"
    )
    print(
        "decision    defect months in the 4-month span   N-th member      closest ticker  ratio   fewest days (ticker)   tickers under 1 / 3 / 5 / 10 days   | tight+allowance: max ratio (ticker), reaching"
    )
    overall = (0.0, None, None)
    affected_min = (float("inf"), None, None)
    total_reaching = 0
    for d in sorted(threshold):
        m = date(d.year, d.month, 1)
        span = [back(m, k) for k in range(4)]
        defects = [DEFECT_MONTHS[x] for x in span if x in DEFECT_MONTHS]
        nth = threshold[d]
        best_ratio, best_sym, fewest, fewest_sym = 0.0, None, float("inf"), None
        under = [0, 0, 0, 0]
        touched = sorted(
            {date(c.year, c.month, 1) for c in calendar[index[d] - WINDOW + 1 : index[d] + 1]}
        )
        tight_max, tight_sym, reaching = 0.0, None, 0
        for sym in left_out:
            bars = monthly.get(sym, {})
            now = bars.get(m)
            if now is None or now[1] <= 0 or now[0] < MIN_PRICE:
                continue  # the proof's bound is 0: no trade or below $5 in the decision month
            total = sum(bars[x][0] * bars[x][1] for x in span if x in bars and bars[x][1] > 0)
            if total <= 0:
                continue
            b = 2.0 * total / WINDOW
            ratio = b / nth
            per_day = total / sum(sessions[x] for x in span)
            days = (nth - b) * WINDOW / 2.0 / per_day
            tight = 0.0
            for x in touched:
                if x in bars and bars[x][1] > 0:
                    hv = bars[x][0] * bars[x][1]
                    tight += hv + (ALLOWANCE_DAYS * hv / sessions[x] if x in DEFECT_MONTHS else 0.0)
            tight = 2.0 * tight / WINDOW
            assert tight <= b * (1 + ALLOWANCE_DAYS) + 1e-6
            if tight / nth > tight_max:
                tight_max, tight_sym = tight / nth, sym
            reaching += tight >= nth
            if ratio > best_ratio:
                best_ratio, best_sym = ratio, sym
            if days < fewest:
                fewest, fewest_sym = days, sym
            for i, cut in enumerate((1, 3, 5, 10)):
                under[i] += days < cut
        if best_ratio > overall[0]:
            overall = (best_ratio, best_sym, d)
        if defects and fewest < affected_min[0]:
            affected_min = (fewest, fewest_sym, d)
        print(
            f"{d}  {', '.join(defects) or '-':36} {nth:>15,.0f}  {best_sym or '-':>14}  {best_ratio:.3f}   {fewest:8.1f} ({fewest_sym})   {' / '.join(map(str, under)):19} | {tight_max:.3f} ({tight_sym}), {reaching}"
        )
        total_reaching += reaching
    print(
        f"\nclosest over all decisions: {overall[1]} on {overall[2]}, bound = {overall[0]:.4f} x N-th member (N-th = {1 / overall[0]:.3f} x bound)"
    )
    print(
        f"fewest days among decisions whose span holds a defect month: {affected_min[0]:.1f} ({affected_min[1]} on {affected_min[2]})"
    )
    print(
        f"tight bound with the defect allowance: tickers reaching the N-th member, all decisions: {total_reaching}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
