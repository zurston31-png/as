"""SIP universe research (Milestone 3 groundwork).

Builds historically valid security identities from Alpaca's consolidated (SIP)
daily bars and corporate actions, and a point-in-time membership rule on top of
them. Everything lives in its own database (``data/sip_research.db`` by default)
and never touches the IEX market data or the Strategy A results.

There is no strategy, backtest or trading code in this package.
"""
