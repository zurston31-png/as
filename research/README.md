# research/

Reproducible measurement scripts. Each one answers a question that is recorded
in `PROVENANCE.md`, so a reader can regenerate a published figure rather than
taking it on trust — the gap `PROVENANCE.md` §3 names about itself.

These are **not** part of the package and nothing in `flow_model/` imports
them. They are read-only consumers of the public API: no script here moves a
threshold, changes a config default, bypasses a gate, or writes to the repo.

Run them with the repo root on the path:

    PYTHONPATH=. python3 research/population_score.py 2016 1 4 2016 3 31
    PYTHONPATH=. python3 research/stage_histogram.py  2016 1 4 2016 3 31

Arguments are `start_year start_month start_day end_year end_month end_day`.
Both refuse nothing and assert nothing; they print measurements.

**Every figure these produce is from the synthetic generator.** No real market
bar has ever entered this system. See `PROVENANCE.md` §1.

| script | answers | recorded in |
|---|---|---|
| `population_score.py` | What is the Flow Score distribution over EVERY bar with a feature vector, not just the few that survive all eight gates? | PROVENANCE §4a |
| `stage_histogram.py` | Which gate rejects each bar, and how many reach each stage? | PROVENANCE §2.4 |
