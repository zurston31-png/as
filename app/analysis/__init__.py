"""Post-trade analysis: what the record actually says, and whether it says
enough to be believed yet.

Split from app/dashboard/analytics.py (headline portfolio numbers) and
app/backtesting/stats.py (per-run backtest statistics) because this package
answers a different question: not "how did it do?" but "is this result
strong enough, and based on enough evidence, to act on?"
"""
