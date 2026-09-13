"""valcoach — an agent that tracks your VALORANT matches and tells you what
you are doing wrong.

Pipeline: a provider fetches match history → matches are stored (raw payload
plus normalized tables) → every death is reconstructed in context → detectors
compare your habits against benchmarks → a report is rendered, optionally with
a written review from Claude.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
