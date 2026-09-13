# valcoach

An agent that keeps track of your VALORANT matches, reconstructs **every death**
in context, and tells you what you are actually doing wrong.

It is not a stat page. For each death it works out who killed you and with what,
where you were standing, how far the nearest living teammate was, whether anyone
traded you, whether you were already outnumbered, what you were holding, and
whether you had spent any utility. Then it compares your habits against
benchmarks and reports the ones that are costing you rounds — with the specific
deaths as evidence and a drill for each one.

```
 CRITICAL  1. You do not trade your teammates
    Only 2 of 76 kills avenged a teammate who had just died (2.6%).
    2.6% vs 12% benchmark  ·  76 kills  ·  high confidence

    Why it costs rounds: Trading is most of what 'playing as a team' means
    mechanically. If you are never the one trading, your team's entries are
    dying for nothing.

    What to do: Stand where you can see the angle your entry is about to peek,
    one step behind them, crosshair already on it. When they go down, you should
    already be looking at the killer.

 HIGH      2. You keep dying in the same place on Split
    10 deaths clustered around A Main on Split, mostly on defense.

    Evidence:
      · R2 · 52s · defense · killed by Echo#2003 (Chamber) · with Stinger ·
        at A Main  [not traded, 5252c loadout]
      · R14 · 8s · attack · killed by Sable#2004 (Reyna) · with Phantom ·
        at A Main  [first blood against, not traded, no util used]
```

## Try it right now

No account, no API key, no network — runs against synthetic matches bundled
with the package. VALORANT runs on Windows, so that comes first.

**Windows (PowerShell)** — one command per line; PowerShell 5.1 does not accept
`&&` as a separator, and the command is `python`, not `python3`:

```powershell
git clone -b claude/valorant-gameplay-agent-5f9fgk https://github.com/zurston31-png/as.git
cd as
python -m valcoach demo
python -m valcoach demo --html review.html
start review.html
```

The `-b` matters: the code lives on that branch, and a plain `git clone` checks
out `main`, which has nothing in it but this README. Already cloned without it?
Run `git checkout claude/valorant-gameplay-agent-5f9fgk`.

If `python` is not recognised, use `py` instead (`py -m valcoach demo`). If
neither exists, install Python from <https://python.org/downloads> and tick
"Add python.exe to PATH".

**macOS / Linux**

```bash
git clone -b claude/valorant-gameplay-agent-5f9fgk \
    https://github.com/zurston31-png/as.git && cd as
python3 -m valcoach demo
python3 -m valcoach demo --html review.html && open review.html
```

[Here is that HTML report, published](https://claude.ai/code/artifact/9193be24-1176-411e-b569-2d1dabd7153f)
— the same page the command writes, from the same synthetic demo matches.

## Install

Running `python -m valcoach` from the repo folder needs no install at all. To
get a `valcoach` command you can run from anywhere:

```powershell
pip install -e .                  # core tool, zero dependencies
pip install -e ".[coach]"         # plus written coaching from Claude
```

The rest of this README writes commands as `valcoach ...`. Without the install,
prefix them with `python -m` (Windows) or `python3 -m` (macOS/Linux) and run
them from the repo folder.

## Point it at your matches

You need a source of match data. Pick one:

| Source | Needs | Detail |
|---|---|---|
| **HenrikDev API** (recommended) | a free API key | Full round/kill/position data. Works from any machine. |
| **Local game client** | Windows + VALORANT running | Riot's own data, no key, no third party. Only the signed-in player. |
| **Official Riot API** | a *production* Riot key | First-party, but Riot does not grant production keys for personal projects. |

### HenrikDev (easiest)

1. Join the HenrikDev Discord and request a key — the process is documented at
   <https://docs.henrikdev.xyz/valorant/api-reference>.
2. Save it:

```powershell
valcoach init --riot-id "YourName#TAG" --region eu --henrik-key YOUR_KEY
valcoach sync --count 10
valcoach analyze
```

Regions: `na`, `eu`, `ap`, `kr`, `br`, `latam`.

### Local game client (no key, Windows)

This is the native path on a machine you actually play on. Start VALORANT, sign
in, then in PowerShell:

```powershell
valcoach init --riot-id "YourName#TAG" --provider local --region na
valcoach sync
valcoach analyze
```

It reads the Riot Client `lockfile` to authenticate against the local API, then
pulls your match history from Riot's player-data endpoints as you. Nothing is
sent anywhere.

### Callout names

Death locations are game coordinates until you download Riot's map data once:

```powershell
valcoach assets     # from valorant-api.com; adds "A Main", "Heaven", ...
```

Without it, reports still cluster your deaths and report them by coordinates.

## Everyday use

```powershell
valcoach sync                       # pull new matches
valcoach analyze --last 10          # the full review
valcoach analyze --html review.html # ... and a shareable HTML page
valcoach deaths --last 5            # every death, one line each
valcoach deaths --untraded --side attack
valcoach coach --question "why do I keep losing 1v1s on attack?"
valcoach watch --coach --notify     # review each new match as you finish it
valcoach history                    # how you've changed over past reviews
valcoach status                     # config, stored data, what's available
```

`watch` is the always-on mode: it polls your match history and reviews each new
game as it lands. With `--provider local` it also notices when you are *in* a
match and checks again the moment it ends.

## What it checks

**Positioning and opening duels**
- Giving up first blood too often, and whether it is happening on attack or defense
- Losing the opening duels you choose to take
- Dying isolated, with no teammate close enough to help or trade
- Dying in the first 15 seconds, before the round has any shape
- Dying repeatedly in the same spot on a map (spatially clustered)
- One opponent who keeps winning the same duel against you
- Deaths to the Operator, and deaths at point-blank range (flanks and uncleared corners)
- A big gap between your attack and defense halves
- Your weakest map

**Team play**
- How often your deaths go untraded, and your average distance to the nearest teammate
- How often *you* trade a teammate who just died

**Economy**
- Dying in rounds that were already lost, instead of saving the gun
- Losing your save rounds and how many credits of gear that donated
- Eco rounds that end inside 25 seconds because you peeked alone

**Utility**
- Abilities per round, and rounds where you threw nothing
- Dying with your utility unspent (role-adjusted: initiators and controllers are held to a higher bar)

**Aim and impact**
- Headshot percentage, ADR, KAST
- Rounds where you did no damage at all
- Clutch conversion

**Strengths** are reported too — a review that is only bad news gets ignored.

Every finding carries the numbers, the sample size, a confidence level, and the
individual deaths that triggered it. Nothing fires on a sample too small to
mean anything. `--strict` tightens every benchmark by ~15% for higher-level play.

## Written coaching

With the `coach` extra installed and credentials available, `--coach` (or the
`coach` command) hands the structured findings to Claude and streams back a
review: verdict, the one thing to fix first, the pattern in how you die, what is
working, a practice plan, and three in-game cues.

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell; or run: ant auth login
valcoach coach --last 10
valcoach coach --focus economy
valcoach coach --question "should I stop playing Jett?"
```

On macOS/Linux that first line is `export ANTHROPIC_API_KEY=sk-ant-...`.

The model is told to use only the data it is given and never to invent a stat —
it is writing up the analysis, not doing it. The structured report is identical
with or without a key; only the prose needs one.

Set a different model with `valcoach coach --model claude-sonnet-5` or
`"model"` in the config file.

## Keeping score

Every review is stored. The next one shows what moved:

```
  Since your last review
    Headshot %                13.8 →     14.7  ▲ 0.9
    Untraded deaths           78.9 →     76.5  ▲ 2.4
    Opening duels won         43.8 →     37.5  ▼ 6.3

  Still open from last time: You do not trade your teammates; You die in the
  first 15 seconds too often
```

## Where your data lives

Everything is local, in `%USERPROFILE%\.valcoach` on Windows or `~/.valcoach`
elsewhere (override with `VALCOACH_HOME`):

- `valcoach.db` — SQLite. Stores the **raw provider payload** for every match
  plus normalized tables derived from it. Because the raw payload is kept,
  `valcoach reindex` re-analyses your whole history whenever the analysers
  improve — old games gain new insight instead of being stuck with whatever the
  tool understood the day you downloaded them.
- `config.json` — settings and API keys (written `0600`; a key in the
  environment is never written to the file).
- `assets.json` — map callouts and UUID→name tables, if downloaded.

Outbound network calls: your chosen match-data provider, `valorant-api.com` if
you run `valcoach assets`, and the Anthropic API only when you ask for written
coaching. Nothing else, and no telemetry. The HTML report is self-contained and
loads nothing at all unless you pass `--webfonts`, which links its display
faces from Google Fonts.

## Configuration

CLI flags beat environment variables, which beat `config.json`.

| Setting | Flag | Environment |
|---|---|---|
| Riot ID | `--riot-id` | `VALCOACH_RIOT_ID` |
| Provider | `--provider` | `VALCOACH_PROVIDER` |
| Region | `--region` | `VALCOACH_REGION` |
| Queue filter | `--queue` | `VALCOACH_QUEUE` |
| HenrikDev key | `--henrik-key` | `HENRIK_API_KEY` |
| Riot key | `--riot-key` | `RIOT_API_KEY` |
| Database | `--db-path` | `VALCOACH_DB` |
| Coaching model | `--model` | `VALCOACH_MODEL` |
| Anthropic key | — | `ANTHROPIC_API_KEY` |
| Home directory | — | `VALCOACH_HOME` |

`trade_window_ms` (default 4000) and `isolation_units` (default 1800) are in
`config.json`; benchmarks live in `valcoach/analysis/detectors.py` in one dict
if you want to tune them.

## How it fits together

```
providers/   fetch match data     Henrik v2+v4 · official Riot · local client · JSON files
   ↓         normalize            valcoach/models.py
store.py     SQLite               raw payload (source of truth) + derived tables
   ↓
analysis/context.py               rebuild each round: who was alive, where, holding what
analysis/metrics.py               aggregate, split by side / map / agent
analysis/detectors.py             compare habits to benchmarks → findings with evidence
analysis/report.py                assemble, compare against your last review
   ↓
coach/                            Claude writes it up (optional)
render/                           terminal · standalone HTML
watcher.py                        poll for new matches, review them as they land
```

## Development

```powershell
python -m unittest discover -s tests -t .     # 167 tests, no network needed
python tools/make_fixture.py --count 5 --out valcoach/fixtures/demo_matches.json
```

Adding a detector: write a function in `valcoach/analysis/detectors.py` that
takes a `DetectorInput` and returns a list of `Finding`, then add it to
`DETECTORS`. A detector that raises is caught and reported as a low-severity
finding rather than breaking the report.

## Honest limitations

- **Demo data is synthetic.** `valcoach demo` uses generated matches so the tool
  can be tried offline. It is realistic in shape, not a real game.
- **Provider shapes were written from documentation**, and the build environment
  had no outbound access to the VALORANT data APIs, so the live-fetch paths are
  untested against real responses. The parsers are deliberately defensive and
  tolerate missing or renamed fields, but if `sync` returns something unexpected
  on your account, save the JSON and run `valcoach import that.json` — parsing
  is fully covered by tests for the documented v2, v4 and official shapes.
- **Benchmarks are heuristics** calibrated for roughly Silver–Diamond ranked
  play, not ground truth. They are one dict, and `--strict` exists.
- **Attack/defense is inferred** from spike plants and defuses (only attackers
  plant), then filled in by half. Modes without sides (deathmatch, escalation)
  are simply reported as unknown.
- **Callout names need `valcoach assets`.** Coordinates until then.
- **No VOD or video analysis.** Everything comes from match data, which is why
  it can be exact about position, timing and economy — and why it cannot tell
  you about your crosshair movement inside a duel.

## License

MIT
