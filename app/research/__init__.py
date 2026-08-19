"""Strategy research: which parts of the strategy earn their place.

Distinct from app/analysis/, which measures what the bot DID. This package
runs controlled experiments to decide what the bot SHOULD do:

    ablation.py    does each scoring factor help, hurt, or add noise?
    robustness.py  is a parameter value a broad stable region or a cliff?
    thresholds.py  what does MIN_SIGNAL_SCORE_TO_ENTER actually buy?
    experiments.py record every run so results are reproducible and ranked

Every one of them judges on OUT-OF-SAMPLE performance. Ranking by in-sample
P&L would just select whichever variant fit the training data hardest,
which is the failure mode all of this exists to prevent.
"""
