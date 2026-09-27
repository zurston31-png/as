"""Safety systems that can stop the bot opening new positions.

    reconcile.py   does the portfolio arithmetic still add up?
    killswitch.py  the single gate every entry passes through

Separate from app/risk/, which decides whether a PARTICULAR trade is
within limits. These decide whether the bot should be trading AT ALL right
now - and they fail closed: when integrity cannot be established, new
entries stop.
"""
