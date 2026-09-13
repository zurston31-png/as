"""Canonical token identity.

ONE RULE, and it exists because breaking it is both easy and expensive:

    THE MINT ADDRESS IS THE TOKEN. THE SYMBOL IS A LABEL.

Symbols on Solana are not unique, not reserved, and not verified. Anyone
can mint a token called BONK. Scammers do this deliberately - the whole
point of a copycat is that it shares a symbol with something people trust.
An automatic scanner that discovers arbitrary new mints will meet symbol
collisions routinely, not rarely.

Keying anything that matters on the symbol therefore produces silent, wrong
behaviour rather than a visible error:

  * a position in mint A blocks an entry into unrelated mint B because
    both are called PEPE - a lost trade with no rejection recorded
  * the per-token exposure cap sums two unrelated assets, so a "10% max per
    token" limit quietly permits 20% in one of them
  * a re-entry cooldown started by mint A silences mint B
  * a sell signal closes whichever position the query happened to return

`instrument_key` below is the single place that decision is made. Every
dedup, exposure, and cooldown lookup goes through it.

The one legitimate exception is a centralised exchange, where there is no
mint and the exchange's own ticker IS the canonical identifier - that is
the `EXECUTION_BACKEND == "cex"` branch, and it is a different namespace,
not a fallback.
"""
from __future__ import annotations

from app.config import settings


def instrument_key(symbol: str, token_address: str | None) -> str:
    """The canonical identity for dedup, exposure and cooldown lookups.

    On a chain this is the mint address. On a CEX it is the ticker, because
    there is no mint and the exchange's symbol is genuinely canonical
    there.

    A missing mint falls back to the symbol so the bot degrades to the old
    (weaker) behaviour rather than crashing - but callers should treat a
    missing mint as a data-quality problem, not a normal case. `is_weak`
    below reports exactly that.
    """
    if settings.EXECUTION_BACKEND == "cex":
        return symbol
    return token_address or symbol


def is_weak(token_address: str | None) -> bool:
    """True when identity had to fall back to the symbol.

    Exposed so the caller can log or reject rather than silently accepting
    a namespace collision it cannot detect.
    """
    return settings.EXECUTION_BACKEND != "cex" and not token_address


def describe(symbol: str, token_address: str | None) -> str:
    """Human-readable identity for logs and rejection reasons.

    Always shows enough of the mint to distinguish two same-symbol tokens,
    because "already holding PEPE" is exactly the message that hides this
    class of bug from whoever is reading the log.
    """
    if not token_address:
        return f"{symbol} (no mint address)"
    short = token_address if len(token_address) <= 12 else f"{token_address[:6]}..{token_address[-4:]}"
    return f"{symbol} [{short}]"
