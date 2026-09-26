"""The Early Signal Engine.

Tries to detect measurable conditions that occur more often BEFORE a strong
upward move than before a weak or negative one. It does not predict pumps,
and it is not a fourth opinion on whether a chart looks good - the existing
technical score already answers that.

    features.py     what is measurable, and what is honestly unavailable
    score.py        the 0-100 Early Opportunity Score
    late_entry.py   the Late Entry Risk score and the stage ladder
    classifier.py   healthy accumulation vs parabolic pump
    engine.py       one candidate in, a decision plus reasoning out
    watchlist.py    the WATCH state machine and score history

THE WEIGHTS IN score.py ARE UNVALIDATED PRIORS.

They were chosen from reasoning about market microstructure, not from
research on historical outcomes, because no historical outcome data exists
yet. That makes them a hypothesis, and the engine treats them as one:
EARLY_SIGNAL_MAY_TRADE defaults to false, so an early signal can raise a
token to WATCH and can never on its own open a position. The existing
technical strategy remains the only thing that trades.

Turning that switch on before app/analysis/early_calibration.py shows the
score actually separates outcomes would be trading on an untested guess.
"""
