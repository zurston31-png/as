"""The mistake engine.

Each detector looks at one habit, compares it against a benchmark, and — when
it fires — returns a finding that carries four things: what the number is, why
it costs rounds, the specific deaths that prove it, and a drill to fix it.

Benchmarks are calibrated for ranked play roughly in the Silver–Diamond range,
which is where most players are. They live in one dict on purpose: raise them
for higher-level play with ``valcoach analyze --strict`` or by editing
``BENCHMARKS``.

Detectors never guess. If the window is too small for a habit to be real, the
finding is either withheld or marked low confidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..maps import MapIndex, cluster_points
from ..models import ATTACK, DEFENSE
from .context import DeathContext, MatchContext, RoundContext
from .metrics import Metrics

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "strength": 4}

BENCHMARKS: Dict[str, float] = {
    # opening duels / positioning
    "first_death_rate": 14.0,        # % of rounds where you die first
    "opening_win_rate": 50.0,        # % of opening duels you win
    "untraded_death_rate": 60.0,     # % of your deaths nobody avenges
    "isolated_death_rate": 25.0,     # % of deaths with no teammate nearby
    "opening_death_rate": 30.0,      # % of deaths in the first 15s of a round
    "deaths_when_down_rate": 20.0,   # % of deaths taken while already outnumbered
    # aim
    "hs_pct": 18.0,
    "adr": 130.0,
    # impact
    "kast_pct": 65.0,
    "trade_participation": 12.0,
    "zero_damage_round_rate": 20.0,
    "clutch_rate": 20.0,
    # economy
    "save_rate": 50.0,               # % of losable rounds where you keep the gun
    # utility
    "util_per_round": 1.6,
    "util_unused_death_rate": 35.0,
    # repetition
    "same_spot_deaths": 3,           # deaths inside one cluster before it counts
    "nemesis_deaths": 4,             # deaths to one opponent before it counts
    "operator_deaths": 4,
}

# Minimum sample before a detector is allowed to speak at all.
MIN_ROUNDS = 20
MIN_DEATHS = 12


@dataclass
class Finding:
    id: str
    title: str
    category: str                    # positioning | aim | impact | economy | utility | teamplay | discipline
    severity: str                    # critical | high | medium | low | strength
    summary: str
    why: str = ""
    fix: str = ""
    value: Optional[float] = None
    benchmark: Optional[float] = None
    unit: str = ""
    sample: str = ""
    confidence: str = "medium"
    evidence: List[str] = field(default_factory=list)

    @property
    def delta(self) -> Optional[float]:
        if self.value is None or self.benchmark is None:
            return None
        return round(self.value - self.benchmark, 1)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["delta"] = self.delta
        return data


@dataclass
class DetectorInput:
    metrics: Metrics
    contexts: List[MatchContext]
    deaths: List[DeathContext] = field(default_factory=list)
    rounds: List[RoundContext] = field(default_factory=list)
    map_index: Optional[MapIndex] = None
    benchmarks: Dict[str, float] = field(default_factory=lambda: dict(BENCHMARKS))
    strict: bool = False

    def bench(self, key: str) -> float:
        value = self.benchmarks.get(key, BENCHMARKS.get(key, 0.0))
        if not self.strict:
            return value
        # Strict mode asks more of you: tighten every rate by ~15%.
        tighten_down = {
            "first_death_rate", "untraded_death_rate", "isolated_death_rate",
            "opening_death_rate", "deaths_when_down_rate", "zero_damage_round_rate",
            "util_unused_death_rate",
        }
        tighten_up = {
            "hs_pct", "adr", "kast_pct", "trade_participation", "clutch_rate",
            "util_per_round", "opening_win_rate", "save_rate",
        }
        if key in tighten_down:
            return round(value * 0.85, 2)
        if key in tighten_up:
            return round(value * 1.15, 2)
        return value

    @property
    def role(self) -> str:
        roles = Counter(c.role for c in self.contexts if c.role)
        return roles.most_common(1)[0][0] if roles else "unknown"

    def confidence_for(self, sample: int) -> str:
        if sample >= 40:
            return "high"
        if sample >= 15:
            return "medium"
        return "low"


def _severity(value: float, benchmark: float, worse_is_higher: bool,
              high_at: float, critical_at: float) -> Optional[str]:
    """Grade how far past the benchmark a value is (as a multiple of the gap)."""
    if benchmark == 0:
        return None
    gap = (value - benchmark) if worse_is_higher else (benchmark - value)
    if gap <= 0:
        return None
    share = gap / abs(benchmark)
    if share >= critical_at:
        return "critical"
    if share >= high_at:
        return "high"
    return "medium"


def _sample(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")


# --------------------------------------------------------------------------
# positioning / opening duels
# --------------------------------------------------------------------------
def detect_first_deaths(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.rounds < MIN_ROUNDS:
        return []
    bench = data.bench("first_death_rate")
    severity = _severity(m.first_death_rate, bench, True, 0.25, 0.6)
    if not severity:
        return []
    early = [d for d in data.deaths if d.first_death_of_round]
    attack = sum(1 for d in early if d.side == ATTACK)
    defense = sum(1 for d in early if d.side == DEFENSE)
    where = ""
    if attack and attack >= 2 * max(defense, 1):
        where = " Nearly all of them are on attack, so this is about how you enter."
    elif defense and defense >= 2 * max(attack, 1):
        where = " Most are on defense, so you are peeking into the attack's timing."
    return [
        Finding(
            id="first_deaths",
            title="You give up first blood too often",
            category="positioning",
            severity=severity,
            value=m.first_death_rate,
            benchmark=bench,
            unit="% of rounds",
            summary=(
                f"You died first in {m.first_deaths} of {m.rounds} rounds "
                f"({m.first_death_rate}%). Benchmark is under {bench}%."
            ),
            why=(
                "The first death decides most rounds: your team plays 4v5 with no "
                "information and usually has to fall back." + where
            ),
            fix=(
                "Before you take the first duel, ask what happens if you lose it. "
                "Only take it when a teammate can trade you within a second or two, "
                "and let utility (a flash, a recon dart, a smoke) open the angle "
                "instead of your body."
            ),
            sample=_sample(m.rounds, "round"),
            confidence=data.confidence_for(m.rounds),
            evidence=[d.describe() for d in early[:5]],
        )
    ]


def detect_opening_duels(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.opening_duels < 10:
        return []
    bench = data.bench("opening_win_rate")
    severity = _severity(m.opening_win_rate, bench, False, 0.15, 0.35)
    if not severity:
        return []
    return [
        Finding(
            id="opening_duels",
            title="You are losing the duels you choose to take",
            category="positioning",
            severity=severity,
            value=m.opening_win_rate,
            benchmark=bench,
            unit="% of opening duels won",
            summary=(
                f"You won {m.first_bloods} of {m.opening_duels} opening duels "
                f"({m.opening_win_rate}%)."
            ),
            why=(
                "An opening duel below 50% means you are taking fights on the "
                "opponent's terms — wide swings, no utility, or into a held angle."
            ),
            fix=(
                "Pick the fights you can pre-aim: jiggle for information first, "
                "clear one angle at a time, and stop taking the first fight on "
                "rounds where you hold the weaker gun."
            ),
            sample=_sample(m.opening_duels, "opening duel"),
            confidence=data.confidence_for(m.opening_duels),
        )
    ]


def detect_untraded_deaths(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.deaths < MIN_DEATHS:
        return []
    bench = data.bench("untraded_death_rate")
    severity = _severity(m.untraded_death_rate, bench, True, 0.15, 0.35)
    if not severity:
        return []
    untraded = [d for d in data.deaths if not d.traded]
    return [
        Finding(
            id="untraded_deaths",
            title="Nobody is trading your deaths",
            category="teamplay",
            severity=severity,
            value=m.untraded_death_rate,
            benchmark=bench,
            unit="% of deaths untraded",
            summary=(
                f"{m.untraded_deaths} of {m.deaths} deaths went untraded "
                f"({m.untraded_death_rate}%); the average gap to your nearest "
                f"living teammate was {m.avg_nearest_teammate / 100:.0f}m."
            ),
            why=(
                "A traded death still wins the round — your team gets the kill and "
                "the space. An untraded death is a free man for the enemy."
            ),
            fix=(
                "Play your fights within about 10m of one teammate so they can step "
                "in on the same angle. Say 'I'm peeking' before you swing; if nobody "
                "is close enough to answer, hold instead."
            ),
            sample=_sample(m.deaths, "death"),
            confidence=data.confidence_for(m.deaths),
            evidence=[d.describe() for d in untraded[:5]],
        )
    ]


def detect_isolated_deaths(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    isolated = [d for d in data.deaths if d.isolated]
    if m.deaths < MIN_DEATHS or not isolated:
        return []
    bench = data.bench("isolated_death_rate")
    severity = _severity(m.isolated_death_rate, bench, True, 0.3, 0.8)
    if not severity:
        return []
    return [
        Finding(
            id="isolated_deaths",
            title="You die alone, away from your team",
            category="positioning",
            severity=severity,
            value=m.isolated_death_rate,
            benchmark=bench,
            unit="% of deaths isolated",
            summary=(
                f"{len(isolated)} of {m.deaths} deaths happened with no teammate "
                f"within 18m ({m.isolated_death_rate}%)."
            ),
            why=(
                "Dying alone hands over a 5v4 and tells your team nothing. It is "
                "usually a rotation that left the group, or holding an off-angle the "
                "team cannot support."
            ),
            fix=(
                "Commit to the same side of the map as at least one teammate. If you "
                "want the off-angle, take it inside the trade radius of the group."
            ),
            sample=_sample(m.deaths, "death"),
            confidence=data.confidence_for(m.deaths),
            evidence=[d.describe() for d in isolated[:5]],
        )
    ]


def detect_repeated_spots(data: DetectorInput) -> List[Finding]:
    if len(data.deaths) < 8:
        return []
    findings: List[Finding] = []
    by_map: Dict[str, List[DeathContext]] = defaultdict(list)
    for death in data.deaths:
        if death.x is not None and death.y is not None:
            by_map[death.map_name or "unknown"].append(death)

    threshold = int(data.bench("same_spot_deaths"))
    for map_name, deaths in by_map.items():
        clusters = cluster_points([(d.x, d.y) for d in deaths])
        for cluster in clusters:
            if cluster.size < threshold:
                continue
            members = [deaths[i] for i in cluster.members]
            place = members[0].place or f"({int(cluster.x)}, {int(cluster.y)})"
            sides = Counter(d.side for d in members)
            side = sides.most_common(1)[0][0]
            severity = "high" if cluster.size >= threshold + 2 else "medium"
            findings.append(
                Finding(
                    id=f"same_spot_{map_name.lower()}_{int(cluster.x)}",
                    title=f"You keep dying in the same place on {map_name}",
                    category="positioning",
                    severity=severity,
                    value=float(cluster.size),
                    benchmark=float(threshold),
                    unit="deaths in one spot",
                    summary=(
                        f"{cluster.size} deaths clustered around {place} on "
                        f"{map_name}, mostly on {side}."
                    ),
                    why=(
                        "Repeating the same position means the enemy has already "
                        "solved it — they pre-aim, pre-fire or pre-place utility "
                        "there before you arrive."
                    ),
                    fix=(
                        f"Next time you play {map_name}, take a different timing or "
                        f"angle for that spot: arrive later, clear it with utility "
                        f"first, or hand it to a teammate and cover a second angle."
                    ),
                    sample=f"{len(deaths)} deaths on {map_name}",
                    confidence=data.confidence_for(len(deaths)),
                    evidence=[d.describe() for d in members[:5]],
                )
            )
    findings.sort(key=lambda f: -(f.value or 0))
    return findings[:2]


def detect_nemesis(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if not m.deaths_by_killer or m.deaths < MIN_DEATHS:
        return []
    name, count = m.deaths_by_killer[0]
    if count < int(data.bench("nemesis_deaths")):
        return []
    theirs = [d for d in data.deaths if d.killer_name == name]
    weapons = Counter(d.weapon for d in theirs).most_common(2)
    places = Counter(d.place for d in theirs if d.place).most_common(2)
    return [
        Finding(
            id="nemesis",
            title=f"{name} has your number",
            category="positioning",
            severity="medium" if count < 6 else "high",
            value=float(count),
            benchmark=float(int(data.bench("nemesis_deaths"))),
            unit="deaths to one opponent",
            summary=(
                f"{name} killed you {count} times "
                f"({', '.join(f'{w} x{c}' for w, c in weapons)})."
            ),
            why=(
                "One opponent repeatedly winning the same duel means they know "
                "where you will be. That is a read you can break."
            ),
            fix=(
                "Change the pattern they are reading: different entry timing, a "
                "different angle, or make a teammate take that duel while you "
                "cover the trade."
                + (f" They got you around {places[0][0]} most often."
                   if places else "")
            ),
            sample=_sample(m.deaths, "death"),
            confidence=data.confidence_for(m.deaths),
            evidence=[d.describe() for d in theirs[:5]],
        )
    ]


def detect_early_deaths(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.deaths < MIN_DEATHS:
        return []
    bench = data.bench("opening_death_rate")
    severity = _severity(m.opening_death_rate, bench, True, 0.2, 0.5)
    if not severity:
        return []
    early = [d for d in data.deaths if d.time_slot == "opening"]
    return [
        Finding(
            id="early_deaths",
            title="You die in the first 15 seconds too often",
            category="positioning",
            severity=severity,
            value=m.opening_death_rate,
            benchmark=bench,
            unit="% of deaths in first 15s",
            summary=(
                f"{len(early)} of {m.deaths} deaths came inside the first 15 "
                f"seconds ({m.opening_death_rate}%). Your average death is at "
                f"{m.avg_death_time_ms / 1000:.0f}s."
            ),
            why=(
                "Dying before the round has taken shape means you fought with no "
                "information and no utility support, and your team has to play the "
                "rest of the round a man down."
            ),
            fix=(
                "Give the round 10 seconds. Let utility and your team's info come "
                "in first; the same duel is much better a few seconds later when "
                "you know where the enemy is."
            ),
            sample=_sample(m.deaths, "death"),
            confidence=data.confidence_for(m.deaths),
            evidence=[d.describe() for d in early[:5]],
        )
    ]


def detect_lost_round_deaths(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.deaths < MIN_DEATHS:
        return []
    rate = round(100.0 * m.deaths_when_down / m.deaths, 1) if m.deaths else 0.0
    bench = data.bench("deaths_when_down_rate")
    severity = _severity(rate, bench, True, 0.3, 0.75)
    if not severity:
        return []
    down = [d for d in data.deaths if d.numbers == "down"]
    return [
        Finding(
            id="deaths_when_outnumbered",
            title="You keep dying in rounds that were already lost",
            category="economy",
            severity=severity,
            value=rate,
            benchmark=bench,
            unit="% of deaths while outnumbered",
            summary=(
                f"{m.deaths_when_down} of {m.deaths} deaths happened when your team "
                f"was already down bodies ({rate}%)."
            ),
            why=(
                "Fighting a lost round donates your gun to the enemy and starves "
                "your own next buy. The round is already gone; the weapon does not "
                "have to be."
            ),
            fix=(
                "When you are down two or more, default to saving: break contact, "
                "hold a far angle you can leave, and keep the rifle for next round. "
                "Take the fight only if it can realistically win the round."
            ),
            sample=_sample(m.deaths, "death"),
            confidence=data.confidence_for(m.deaths),
            evidence=[d.describe() for d in down[:5]],
        )
    ]


# --------------------------------------------------------------------------
# economy
# --------------------------------------------------------------------------
def detect_saves(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    losable = m.saves + m.saveable_deaths
    if losable < 6:
        return []
    save_rate = round(100.0 * m.saves / losable, 1)
    bench = data.bench("save_rate")
    severity = _severity(save_rate, bench, False, 0.2, 0.5)
    if not severity:
        return []
    lost = [
        d for d in data.deaths
        if not d.round_won and d.loadout_value >= 2900
    ]
    return [
        Finding(
            id="poor_saves",
            title="You are throwing away guns in lost rounds",
            category="economy",
            severity=severity,
            value=save_rate,
            benchmark=bench,
            unit="% of losable rounds saved",
            summary=(
                f"You kept your gear in {m.saves} of {losable} lost rounds where "
                f"you were holding real value ({save_rate}%). The rounds you died "
                f"in handed over {m.gear_lost_credits:,} credits of gear, about "
                f"{m.gear_lost_credits // max(1, m.saveable_deaths):,} per round."
            ),
            why=(
                "Every saved rifle is a round your team does not have to buy for. "
                "Dying with a full buy in a lost round costs you the next round too."
            ),
            fix=(
                "Decide early: once the round is lost, your job is to get the gun "
                "out. Move away from the fight, not toward it, and group up with "
                "whoever else is alive so you both survive."
            ),
            sample=_sample(losable, "losable round"),
            confidence=data.confidence_for(losable),
            evidence=[d.describe() for d in lost[:5]],
        )
    ]


def detect_eco_overpeek(data: DetectorInput) -> List[Finding]:
    eco_rounds = [r for r in data.rounds if r.econ == "eco"]
    if len(eco_rounds) < 5:
        return []
    early_eco = [
        d for d in data.deaths if d.econ == "eco" and d.time_in_round_ms <= 25_000
    ]
    if len(early_eco) < 3:
        return []
    rate = round(100.0 * len(early_eco) / len(eco_rounds), 1)
    if rate < 45:
        return []
    return [
        Finding(
            id="eco_overpeek",
            title="Your eco rounds end before they start",
            category="economy",
            severity="high" if rate >= 65 else "medium",
            value=rate,
            benchmark=45.0,
            unit="% of eco rounds lost early",
            summary=(
                f"On {len(early_eco)} of {len(eco_rounds)} eco rounds you died "
                f"inside 25 seconds ({rate}%)."
            ),
            why=(
                "A pistol round played alone is a free kill and a free ultimate "
                "point for the enemy. Stacked ecos win guns; split ecos lose them."
            ),
            fix=(
                "Play ecos as a group of five at one close angle — same site, same "
                "timing, shotgun-range if you can. One won eco round is worth more "
                "than five solo peeks."
            ),
            sample=_sample(len(eco_rounds), "eco round"),
            confidence=data.confidence_for(len(eco_rounds)),
            evidence=[d.describe() for d in early_eco[:5]],
        )
    ]


# --------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------
def detect_utility_usage(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.rounds < MIN_ROUNDS:
        return []
    findings: List[Finding] = []
    bench = data.bench("util_per_round")
    # Initiators and controllers are expected to throw more than duelists.
    role = data.role
    if role in ("initiator", "controller"):
        bench = round(bench * 1.35, 2)
    severity = _severity(m.util_per_round, bench, False, 0.2, 0.45)
    if severity:
        findings.append(
            Finding(
                id="low_utility",
                title="Your utility stays in the bank",
                category="utility",
                severity=severity,
                value=m.util_per_round,
                benchmark=bench,
                unit="ability casts per round",
                summary=(
                    f"{m.util_casts} casts across {m.rounds} rounds "
                    f"({m.util_per_round}/round), and {m.no_util_rounds} rounds "
                    f"with none at all."
                    + (f" As {role} you should be well above that." if role != "unknown" else "")
                ),
                why=(
                    "Abilities are free damage, free information and free space. "
                    "Unused utility is the cheapest mistake in the game to fix, "
                    "because it costs nothing to throw."
                ),
                fix=(
                    "Spend your basics every round before the first contact, even "
                    "imperfectly. A wasted flash beats a saved one — you get them "
                    "back next round."
                ),
                sample=_sample(m.rounds, "round"),
                confidence=data.confidence_for(m.rounds),
            )
        )

    if m.deaths >= MIN_DEATHS:
        rate = round(100.0 * m.util_unused_deaths / m.deaths, 1)
        bench_u = data.bench("util_unused_death_rate")
        sev = _severity(rate, bench_u, True, 0.3, 0.7)
        if sev:
            unused = [d for d in data.deaths if d.util_unused]
            findings.append(
                Finding(
                    id="died_with_utility",
                    title="You die with your abilities unused",
                    category="utility",
                    severity=sev,
                    value=rate,
                    benchmark=bench_u,
                    unit="% of deaths with no util used",
                    summary=(
                        f"In {m.util_unused_deaths} of {m.deaths} deaths you had "
                        f"not cast a single ability that round ({rate}%)."
                    ),
                    why=(
                        "Dying with full utility means you fought a duel you could "
                        "have tilted in your favour and chose not to."
                    ),
                    fix=(
                        "Make the ability the thing that opens the angle: flash, "
                        "smoke or dart first, then peek. If you are about to take a "
                        "duel with full util, that is the cue to use it."
                    ),
                    sample=_sample(m.deaths, "death"),
                    confidence=data.confidence_for(m.deaths),
                    evidence=[d.describe() for d in unused[:5]],
                )
            )
    return findings


# --------------------------------------------------------------------------
# aim and impact
# --------------------------------------------------------------------------
def detect_headshots(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.shots < 100:
        return []
    bench = data.bench("hs_pct")
    severity = _severity(m.hs_pct, bench, False, 0.15, 0.35)
    if not severity:
        return []
    return [
        Finding(
            id="low_headshots",
            title="Your crosshair sits too low",
            category="aim",
            severity=severity,
            value=m.hs_pct,
            benchmark=bench,
            unit="% headshots",
            summary=(
                f"{m.headshots} of {m.shots} registered shots were headshots "
                f"({m.hs_pct}%)."
            ),
            why=(
                "Head-level crosshair placement is what turns a 3-shot kill into a "
                "1-shot kill. At the same reaction speed, higher placement simply "
                "wins more duels."
            ),
            fix=(
                "Ten minutes of Range 'Bots: Easy' with strafing before you queue, "
                "aiming to end every kill without moving the mouse down. Then hold "
                "your crosshair at head height while walking — most low HS% comes "
                "from re-aiming after the peek, not from flicks."
            ),
            sample=_sample(m.shots, "registered shot"),
            confidence=data.confidence_for(m.rounds),
        )
    ]


def detect_damage_output(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.rounds < MIN_ROUNDS:
        return []
    findings: List[Finding] = []
    bench = data.bench("adr")
    severity = _severity(m.adr, bench, False, 0.15, 0.35)
    if severity:
        findings.append(
            Finding(
                id="low_adr",
                title="Low damage per round",
                category="impact",
                severity=severity,
                value=m.adr,
                benchmark=bench,
                unit="damage per round",
                summary=f"{m.adr} damage per round across {m.rounds} rounds.",
                why=(
                    "Damage is the honest measure of contribution — it counts the "
                    "chip damage that lets a teammate finish a kill."
                ),
                fix=(
                    "Look for the second shot rather than the perfect first one: "
                    "trade damage into a group fight instead of waiting for a clean "
                    "1v1, and stop disengaging after one bullet."
                ),
                sample=_sample(m.rounds, "round"),
                confidence=data.confidence_for(m.rounds),
            )
        )

    zero_rate = round(100.0 * m.zero_damage_rounds / m.rounds, 1)
    bench_z = data.bench("zero_damage_round_rate")
    sev = _severity(zero_rate, bench_z, True, 0.25, 0.6)
    if sev:
        findings.append(
            Finding(
                id="silent_rounds",
                title="Too many rounds where you do nothing",
                category="impact",
                severity=sev,
                value=zero_rate,
                benchmark=bench_z,
                unit="% of rounds with zero damage",
                summary=(
                    f"{m.zero_damage_rounds} of {m.rounds} rounds ended with no "
                    f"damage dealt at all ({zero_rate}%)."
                ),
                why=(
                    "A round with no damage is a round your team played 4v5 in "
                    "effect. Usually it means arriving after the fight resolved."
                ),
                fix=(
                    "Pick your spot before the round starts and be there when "
                    "contact happens. If you are consistently late, rotate earlier "
                    "or hold closer to the likely fight."
                ),
                sample=_sample(m.rounds, "round"),
                confidence=data.confidence_for(m.rounds),
            )
        )
    return findings


def detect_kast(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.rounds < MIN_ROUNDS:
        return []
    bench = data.bench("kast_pct")
    severity = _severity(m.kast_pct, bench, False, 0.1, 0.25)
    if not severity:
        return []
    return [
        Finding(
            id="low_kast",
            title="You are absent from too many rounds",
            category="impact",
            severity=severity,
            value=m.kast_pct,
            benchmark=bench,
            unit="% KAST",
            summary=(
                f"You had a kill, assist, survival or traded death in "
                f"{m.kast_rounds} of {m.rounds} rounds ({m.kast_pct}%)."
            ),
            why=(
                "KAST measures how often you mattered at all. A low number with a "
                "fine K/D means you are feast-or-famine: big rounds, then nothing."
            ),
            fix=(
                "Aim for a contribution every round, even a small one: chip damage, "
                "a flash for a teammate, or simply surviving to the next round with "
                "your gun."
            ),
            sample=_sample(m.rounds, "round"),
            confidence=data.confidence_for(m.rounds),
        )
    ]


def detect_trade_participation(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.kills < 15:
        return []
    bench = data.bench("trade_participation")
    severity = _severity(m.trade_participation, bench, False, 0.25, 0.6)
    if not severity:
        return []
    return [
        Finding(
            id="low_trade_participation",
            title="You do not trade your teammates",
            category="teamplay",
            severity=severity,
            value=m.trade_participation,
            benchmark=bench,
            unit="% of kills that were trades",
            summary=(
                f"Only {m.trade_kills} of {m.kills} kills avenged a teammate who "
                f"had just died ({m.trade_participation}%)."
            ),
            why=(
                "Trading is most of what 'playing as a team' means mechanically. "
                "If you are never the one trading, your team's entries are dying "
                "for nothing."
            ),
            fix=(
                "Stand where you can see the angle your entry is about to peek, one "
                "step behind them, crosshair already on it. When they go down, you "
                "should already be looking at the killer."
            ),
            sample=_sample(m.kills, "kill"),
            confidence=data.confidence_for(m.kills),
        )
    ]


def detect_clutches(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.clutch_attempts < 6:
        return []
    bench = data.bench("clutch_rate")
    severity = _severity(m.clutch_rate, bench, False, 0.25, 0.6)
    if not severity:
        return []
    return [
        Finding(
            id="low_clutch_rate",
            title="Clutch situations are slipping away",
            category="impact",
            severity=severity,
            value=m.clutch_rate,
            benchmark=bench,
            unit="% of clutches won",
            summary=(
                f"You won {m.clutch_wins} of {m.clutch_attempts} rounds where you "
                f"were the last one alive ({m.clutch_rate}%)."
            ),
            why=(
                "Clutches are won by information and the clock, not aim. Most lost "
                "clutches are lost by fighting two players at once or by running "
                "out of time before isolating a duel."
            ),
            fix=(
                "In a 1vX, play the clock and force them to come to you one at a "
                "time: take a position where only one angle can be peeked, and use "
                "sound instead of vision to pick the moment."
            ),
            sample=_sample(m.clutch_attempts, "clutch"),
            confidence=data.confidence_for(m.clutch_attempts),
        )
    ]


# --------------------------------------------------------------------------
# matchup / map patterns
# --------------------------------------------------------------------------
def detect_side_imbalance(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    attack = m.by_side.get(ATTACK)
    defense = m.by_side.get(DEFENSE)
    if not attack or not defense or min(attack.rounds, defense.rounds) < 15:
        return []
    worse, better = (
        (attack, defense) if attack.win_rate < defense.win_rate
        else (defense, attack)
    )
    gap = round(better.win_rate - worse.win_rate, 1)
    if gap < 15:
        return []
    return [
        Finding(
            id=f"side_gap_{worse.label}",
            title=f"Your {worse.label} rounds are much weaker",
            category="positioning",
            severity="high" if gap >= 25 else "medium",
            value=worse.win_rate,
            benchmark=better.win_rate,
            unit=f"% of {worse.label} rounds won",
            summary=(
                f"{worse.label}: {worse.win_rate}% round win rate, "
                f"{worse.kd} K/D, first death in {worse.first_death_rate}% of "
                f"rounds. {better.label}: {better.win_rate}% and {better.kd} K/D."
            ),
            why=(
                f"A {gap} point gap between sides is a habit, not variance. "
                f"{'Attacking asks you to create space on a timer; ' if worse.label == ATTACK else 'Defending asks you to hold and delay; '}"
                "whatever works on the other side is not transferring."
            ),
            fix=(
                "Before each round on your weak side, pick one job and say it out "
                "loud: which angle, which utility, and who you are playing with. "
                "Review your next two games on that side only."
            ),
            sample=_sample(worse.rounds, f"{worse.label} round"),
            confidence=data.confidence_for(worse.rounds),
        )
    ]


def detect_map_weakness(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if len(m.by_map) < 2:
        return []
    eligible = [s for s in m.by_map.values() if s.rounds >= 20]
    if len(eligible) < 2:
        return []
    overall = m.round_win_rate
    worst = min(eligible, key=lambda s: s.win_rate)
    if overall - worst.win_rate < 12:
        return []
    return [
        Finding(
            id=f"map_weak_{worst.label.lower()}",
            title=f"{worst.label} is your worst map right now",
            category="positioning",
            severity="medium",
            value=worst.win_rate,
            benchmark=overall,
            unit="% of rounds won",
            summary=(
                f"On {worst.label} you win {worst.win_rate}% of rounds "
                f"({worst.kd} K/D over {worst.rounds} rounds) against "
                f"{overall}% overall."
            ),
            why=(
                "A single weak map usually comes down to not knowing the default "
                "setup — which angles are yours, and where the team expects you."
            ),
            fix=(
                f"Pick one site on {worst.label} and learn a single default for it: "
                "your position, your first utility, and your rotation. One map, one "
                "site, one job."
            ),
            sample=_sample(worst.rounds, "round"),
            confidence=data.confidence_for(worst.rounds),
        )
    ]


def detect_weapon_matchups(data: DetectorInput) -> List[Finding]:
    if len(data.deaths) < MIN_DEATHS:
        return []
    findings: List[Finding] = []
    ops = [d for d in data.deaths if d.weapon_kind == "sniper"]
    if len(ops) >= int(data.bench("operator_deaths")):
        places = Counter(d.place for d in ops if d.place).most_common(2)
        findings.append(
            Finding(
                id="sniper_deaths",
                title="The Operator keeps catching you",
                category="positioning",
                severity="medium" if len(ops) < 7 else "high",
                value=float(len(ops)),
                benchmark=float(int(data.bench("operator_deaths"))),
                unit="deaths to snipers",
                summary=(
                    f"{len(ops)} deaths to a sniper"
                    + (f", mostly around {places[0][0]}" if places else "")
                    + "."
                ),
                why=(
                    "Sniper deaths are almost always positional: you walked into a "
                    "long angle that was already held, with nothing to break it."
                ),
                fix=(
                    "Treat long angles as closed until you spend something on them: "
                    "smoke, flash, or a teammate's shot. If you must cross, cross "
                    "fast and at an unexpected time — not first, not repeatedly."
                ),
                sample=_sample(len(data.deaths), "death"),
                confidence=data.confidence_for(len(data.deaths)),
                evidence=[d.describe() for d in ops[:4]],
            )
        )

    measured = [d for d in data.deaths if d.distance_to_killer is not None]
    close = [d for d in measured if d.distance_to_killer <= 700]
    close_rate = round(100.0 * len(close) / len(measured), 1) if measured else 0.0
    if len(measured) >= 15 and len(close) >= 6 and close_rate >= 32.0:
        findings.append(
            Finding(
                id="close_range_deaths",
                title="You are getting run down at close range",
                category="positioning",
                severity="high" if close_rate >= 45 else "medium",
                value=close_rate,
                benchmark=32.0,
                unit="% of deaths inside 7m",
                summary=(
                    f"{len(close)} of {len(measured)} deaths happened inside about "
                    f"7m of your killer ({close_rate}%)."
                ),
                why=(
                    "Some close fights are normal, but when a third of your deaths "
                    "are point blank, people are reaching you from angles you never "
                    "cleared — flanks and corners rather than held duels."
                ),
                fix=(
                    "Clear close angles before you settle on a long one, and keep "
                    "your back to a wall you have already checked. Check your flank "
                    "on a timer, not on a sound cue."
                ),
                sample=_sample(len(measured), "positioned death"),
                confidence=data.confidence_for(len(measured)),
                evidence=[d.describe() for d in close[:4]],
            )
        )
    return findings


def detect_discipline(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    if m.afk_rounds < 2:
        return []
    return [
        Finding(
            id="afk_rounds",
            title="Rounds where you were not there",
            category="discipline",
            severity="medium",
            value=float(m.afk_rounds),
            benchmark=0.0,
            unit="AFK / spawn rounds",
            summary=(
                f"{m.afk_rounds} rounds are flagged as AFK or spent in spawn."
            ),
            why="Those rounds are a guaranteed 4v5 for your team.",
            fix="Buy and take your position in the first 10 seconds of the round.",
            sample=_sample(m.rounds, "round"),
            confidence="high",
        )
    ]


# --------------------------------------------------------------------------
# strengths (a report that is only bad news gets ignored)
# --------------------------------------------------------------------------
def detect_strengths(data: DetectorInput) -> List[Finding]:
    m = data.metrics
    out: List[Finding] = []
    if m.rounds < MIN_ROUNDS:
        return out

    if m.hs_pct >= data.bench("hs_pct") * 1.25 and m.shots >= 100:
        out.append(
            Finding(
                id="good_headshots", title="Your crosshair placement is a weapon",
                category="aim", severity="strength", value=m.hs_pct,
                benchmark=data.bench("hs_pct"), unit="% headshots",
                summary=f"{m.hs_pct}% headshot rate over {m.shots} shots.",
                why="You win duels you have no business winning. Keep feeding it.",
                fix="Protect it: warm up before ranked and keep your sensitivity fixed.",
                sample=_sample(m.rounds, "round"), confidence="high",
            )
        )
    if m.clutch_attempts >= 6 and m.clutch_rate >= 35:
        out.append(
            Finding(
                id="good_clutches", title="You are dangerous in a 1vX",
                category="impact", severity="strength", value=m.clutch_rate,
                benchmark=data.bench("clutch_rate"), unit="% clutches won",
                summary=f"{m.clutch_wins} of {m.clutch_attempts} clutches won.",
                why="Composure under pressure is rare and it wins close games.",
                fix="Lean into it: your team can trust you with the last-man role.",
                sample=_sample(m.clutch_attempts, "clutch"), confidence="medium",
            )
        )
    if m.multikill_rounds >= max(4, m.rounds * 0.12):
        out.append(
            Finding(
                id="good_multikills", title="You convert rounds in bunches",
                category="impact", severity="strength",
                value=float(m.multikill_rounds), benchmark=None,
                unit="multi-kill rounds",
                summary=(
                    f"{m.multikill_rounds} multi-kill rounds "
                    f"({m.triplekill_rounds} with three or more)."
                ),
                why="When you get the first kill you keep going, which snowballs rounds.",
                fix="Make those rounds more likely: fight with a teammate close.",
                sample=_sample(m.rounds, "round"), confidence="medium",
            )
        )
    attack = m.by_side.get(ATTACK)
    defense = m.by_side.get(DEFENSE)
    for split in (attack, defense):
        if split and split.rounds >= 20 and split.win_rate >= 60:
            out.append(
                Finding(
                    id=f"good_side_{split.label}",
                    title=f"Your {split.label} half is strong",
                    category="positioning", severity="strength",
                    value=split.win_rate, benchmark=50.0,
                    unit=f"% of {split.label} rounds won",
                    summary=(
                        f"{split.win_rate}% round win rate on {split.label} with "
                        f"{split.kd} K/D."
                    ),
                    why="Whatever you are doing there is working — copy it across.",
                    fix=(
                        f"Write down what your default is on {split.label} and try "
                        "the same clarity on the other side."
                    ),
                    sample=_sample(split.rounds, "round"), confidence="medium",
                )
            )
    return out


DETECTORS: Sequence[Callable[[DetectorInput], List[Finding]]] = (
    detect_first_deaths,
    detect_opening_duels,
    detect_untraded_deaths,
    detect_isolated_deaths,
    detect_repeated_spots,
    detect_nemesis,
    detect_early_deaths,
    detect_lost_round_deaths,
    detect_saves,
    detect_eco_overpeek,
    detect_utility_usage,
    detect_headshots,
    detect_damage_output,
    detect_kast,
    detect_trade_participation,
    detect_clutches,
    detect_side_imbalance,
    detect_map_weakness,
    detect_weapon_matchups,
    detect_discipline,
    detect_strengths,
)


def run_detectors(data: DetectorInput) -> List[Finding]:
    findings: List[Finding] = []
    for detector in DETECTORS:
        try:
            findings.extend(detector(data) or [])
        except Exception as exc:  # noqa: BLE001 - one bad rule must not kill a report
            findings.append(
                Finding(
                    id=f"detector_error_{detector.__name__}",
                    title=f"Detector {detector.__name__} failed",
                    category="discipline",
                    severity="low",
                    summary=f"{type(exc).__name__}: {exc}",
                    confidence="low",
                )
            )
    findings.sort(
        key=lambda f: (
            SEVERITY_ORDER.get(f.severity, 9),
            -(abs(f.delta) if f.delta is not None else 0),
        )
    )
    return findings
