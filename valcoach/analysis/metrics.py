"""Aggregate metrics over a window of matches.

Everything here is derived from ``analysis.context`` objects, so a metric is
always traceable back to the individual rounds and deaths that produced it —
which is what lets the detectors quote evidence instead of vibes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import UNKNOWN_SIDE
from .context import DeathContext, MatchContext, OPENING_WINDOW_MS


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _ratio(part: float, whole: float) -> float:
    return round(part / whole, 2) if whole else 0.0


def _mean(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else 0.0


@dataclass
class SplitMetrics:
    """A per-side / per-map / per-agent slice."""

    label: str
    rounds: int = 0
    kills: int = 0
    deaths: int = 0
    rounds_won: int = 0
    first_deaths: int = 0
    untraded_deaths: int = 0
    damage: int = 0

    @property
    def kd(self) -> float:
        return _ratio(self.kills, self.deaths)

    @property
    def win_rate(self) -> float:
        return _pct(self.rounds_won, self.rounds)

    @property
    def first_death_rate(self) -> float:
        return _pct(self.first_deaths, self.rounds)

    @property
    def adr(self) -> float:
        return _ratio(self.damage, self.rounds)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            kd=self.kd, win_rate=self.win_rate,
            first_death_rate=self.first_death_rate, adr=self.adr,
        )
        return data


@dataclass
class Metrics:
    riot_id: str = ""
    puuid: str = ""
    matches: int = 0
    match_wins: int = 0
    rounds: int = 0
    rounds_won: int = 0

    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage: int = 0
    headshots: int = 0
    bodyshots: int = 0
    legshots: int = 0

    first_bloods: int = 0
    first_deaths: int = 0
    untraded_deaths: int = 0
    isolated_deaths: int = 0
    trade_kills: int = 0
    opening_deaths: int = 0          # died in the first 15s of a round
    deaths_when_down: int = 0        # died while already outnumbered
    post_plant_deaths: int = 0
    util_unused_deaths: int = 0
    eco_deaths_early: int = 0
    saveable_deaths: int = 0         # died in a lost round holding real gear
    gear_lost_credits: int = 0
    saves: int = 0
    kast_rounds: int = 0
    zero_damage_rounds: int = 0
    no_util_rounds: int = 0
    multikill_rounds: int = 0
    triplekill_rounds: int = 0
    clutch_attempts: int = 0
    clutch_wins: int = 0
    util_casts: int = 0
    loadout_total: int = 0
    afk_rounds: int = 0

    avg_death_time_ms: float = 0.0
    avg_nearest_teammate: float = 0.0

    by_side: Dict[str, SplitMetrics] = field(default_factory=dict)
    by_map: Dict[str, SplitMetrics] = field(default_factory=dict)
    by_agent: Dict[str, SplitMetrics] = field(default_factory=dict)
    deaths_by_weapon_class: Dict[str, int] = field(default_factory=dict)
    deaths_by_time_slot: Dict[str, int] = field(default_factory=dict)
    deaths_by_killer: List[Tuple[str, int]] = field(default_factory=list)
    deaths_by_weapon: List[Tuple[str, int]] = field(default_factory=list)

    # ---- derived -------------------------------------------------------
    @property
    def kd(self) -> float:
        return _ratio(self.kills, self.deaths)

    @property
    def kda(self) -> float:
        return _ratio(self.kills + self.assists, self.deaths)

    @property
    def adr(self) -> float:
        return _ratio(self.damage, self.rounds)

    @property
    def shots(self) -> int:
        return self.headshots + self.bodyshots + self.legshots

    @property
    def hs_pct(self) -> float:
        return _pct(self.headshots, self.shots)

    @property
    def kast_pct(self) -> float:
        return _pct(self.kast_rounds, self.rounds)

    @property
    def round_win_rate(self) -> float:
        return _pct(self.rounds_won, self.rounds)

    @property
    def match_win_rate(self) -> float:
        return _pct(self.match_wins, self.matches)

    @property
    def first_blood_rate(self) -> float:
        return _pct(self.first_bloods, self.rounds)

    @property
    def first_death_rate(self) -> float:
        return _pct(self.first_deaths, self.rounds)

    @property
    def opening_duels(self) -> int:
        return self.first_bloods + self.first_deaths

    @property
    def opening_win_rate(self) -> float:
        return _pct(self.first_bloods, self.opening_duels)

    @property
    def untraded_death_rate(self) -> float:
        return _pct(self.untraded_deaths, self.deaths)

    @property
    def isolated_death_rate(self) -> float:
        return _pct(self.isolated_deaths, self.deaths)

    @property
    def trade_participation(self) -> float:
        """Share of your kills that avenged a teammate who just died."""
        return _pct(self.trade_kills, self.kills)

    @property
    def opening_death_rate(self) -> float:
        return _pct(self.opening_deaths, self.deaths)

    @property
    def util_per_round(self) -> float:
        return _ratio(self.util_casts, self.rounds)

    @property
    def avg_loadout(self) -> float:
        return _ratio(self.loadout_total, self.rounds)

    @property
    def clutch_rate(self) -> float:
        return _pct(self.clutch_wins, self.clutch_attempts)

    @property
    def deaths_per_round(self) -> float:
        return _ratio(self.deaths, self.rounds)

    @property
    def nemesis(self) -> Optional[Tuple[str, int]]:
        return self.deaths_by_killer[0] if self.deaths_by_killer else None

    def summary(self) -> Dict[str, Any]:
        """Flat, JSON-safe view — this is what the coach model is shown."""
        return {
            "riot_id": self.riot_id,
            "matches": self.matches,
            "match_win_rate": self.match_win_rate,
            "rounds": self.rounds,
            "round_win_rate": self.round_win_rate,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "kd": self.kd,
            "kda": self.kda,
            "adr": self.adr,
            "hs_pct": self.hs_pct,
            "kast_pct": self.kast_pct,
            "first_blood_rate": self.first_blood_rate,
            "first_death_rate": self.first_death_rate,
            "opening_duels": self.opening_duels,
            "opening_win_rate": self.opening_win_rate,
            "untraded_death_rate": self.untraded_death_rate,
            "isolated_death_rate": self.isolated_death_rate,
            "trade_participation": self.trade_participation,
            "opening_death_rate": self.opening_death_rate,
            "deaths_when_outnumbered": self.deaths_when_down,
            "post_plant_deaths": self.post_plant_deaths,
            "util_per_round": self.util_per_round,
            "util_unused_deaths": self.util_unused_deaths,
            "no_util_rounds": self.no_util_rounds,
            "zero_damage_rounds": self.zero_damage_rounds,
            "multikill_rounds": self.multikill_rounds,
            "clutch_attempts": self.clutch_attempts,
            "clutch_wins": self.clutch_wins,
            "clutch_rate": self.clutch_rate,
            "avg_loadout": self.avg_loadout,
            "saveable_deaths": self.saveable_deaths,
            "gear_lost_credits": self.gear_lost_credits,
            "saves": self.saves,
            "avg_death_time_s": round(self.avg_death_time_ms / 1000, 1),
            "avg_nearest_teammate_units": self.avg_nearest_teammate,
            "by_side": {k: v.to_dict() for k, v in self.by_side.items()},
            "by_map": {k: v.to_dict() for k, v in self.by_map.items()},
            "by_agent": {k: v.to_dict() for k, v in self.by_agent.items()},
            "deaths_by_weapon_class": self.deaths_by_weapon_class,
            "deaths_by_time_slot": self.deaths_by_time_slot,
            "top_killers": self.deaths_by_killer[:5],
            "top_death_weapons": self.deaths_by_weapon[:5],
            "nemesis": self.nemesis,
        }


def _bump(store: Dict[str, SplitMetrics], key: str) -> SplitMetrics:
    if key not in store:
        store[key] = SplitMetrics(label=key)
    return store[key]


def compute_metrics(
    contexts: Iterable[MatchContext],
    riot_id: str = "",
    puuid: str = "",
) -> Metrics:
    m = Metrics(riot_id=riot_id, puuid=puuid)
    death_times: List[float] = []
    nearest: List[float] = []
    killer_counter: Counter = Counter()
    weapon_counter: Counter = Counter()
    class_counter: Counter = Counter()
    slot_counter: Counter = Counter()

    for ctx in contexts:
        m.matches += 1
        if ctx.won:
            m.match_wins += 1
        agent_split = _bump(m.by_agent, ctx.agent or "unknown")
        map_split = _bump(m.by_map, ctx.map_name or "unknown")

        deaths_by_round: Dict[int, DeathContext] = {
            d.round_index: d for d in ctx.deaths
        }
        kills_by_round: Dict[int, List[Any]] = defaultdict(list)
        for kill in ctx.kills:
            kills_by_round[kill.round_index].append(kill)
            if kill.first_blood:
                m.first_bloods += 1
            if kill.trade_kill:
                m.trade_kills += 1

        for rnd in ctx.rounds:
            side_split = _bump(m.by_side, rnd.side or UNKNOWN_SIDE)
            round_kills = kills_by_round.get(rnd.round_index, [])
            death = deaths_by_round.get(rnd.round_index)

            m.rounds += 1
            m.damage += rnd.damage
            m.util_casts += rnd.util_casts
            m.loadout_total += rnd.loadout_value
            m.kills += len(round_kills)
            if rnd.won:
                m.rounds_won += 1
            if rnd.damage == 0:
                m.zero_damage_rounds += 1
            if rnd.util_casts == 0:
                m.no_util_rounds += 1
            if len(round_kills) >= 2:
                m.multikill_rounds += 1
            if len(round_kills) >= 3:
                m.triplekill_rounds += 1
            if rnd.clutch_attempt:
                m.clutch_attempts += 1
                if rnd.clutch_won:
                    m.clutch_wins += 1
            if rnd.afk or rnd.stayed_in_spawn:
                m.afk_rounds += 1
            if rnd.saved:
                m.saves += 1
            if rnd.lost_gear_value:
                m.saveable_deaths += 1
                m.gear_lost_credits += rnd.lost_gear_value
            # KAST: a kill, an assist, a survival, or a traded death.
            if round_kills or rnd.got_assist or rnd.survived or rnd.was_traded:
                m.kast_rounds += 1

            for split in (side_split, map_split, agent_split):
                split.rounds += 1
                split.kills += len(round_kills)
                split.damage += rnd.damage
                if rnd.won:
                    split.rounds_won += 1

            if death is None:
                continue

            m.deaths += 1
            for split in (side_split, map_split, agent_split):
                split.deaths += 1
                if death.first_death_of_round:
                    split.first_deaths += 1
                if not death.traded:
                    split.untraded_deaths += 1

            if death.first_death_of_round:
                m.first_deaths += 1
            if not death.traded:
                m.untraded_deaths += 1
            if death.isolated:
                m.isolated_deaths += 1
            if death.time_in_round_ms <= OPENING_WINDOW_MS:
                m.opening_deaths += 1
            if death.numbers == "down":
                m.deaths_when_down += 1
            if death.post_plant:
                m.post_plant_deaths += 1
            if death.util_unused:
                m.util_unused_deaths += 1
            if death.econ == "eco" and death.time_in_round_ms <= 25_000:
                m.eco_deaths_early += 1

            death_times.append(death.time_in_round_ms)
            if death.nearest_teammate_distance is not None:
                nearest.append(death.nearest_teammate_distance)
            if death.killer_name:
                killer_counter[death.killer_name] += 1
            if death.weapon:
                weapon_counter[death.weapon] += 1
            class_counter[death.weapon_kind or "other"] += 1
            slot_counter[death.time_slot] += 1

    m.avg_death_time_ms = _mean(death_times)
    m.avg_nearest_teammate = _mean(nearest)
    m.deaths_by_killer = killer_counter.most_common()
    m.deaths_by_weapon = weapon_counter.most_common()
    m.deaths_by_weapon_class = dict(class_counter.most_common())
    m.deaths_by_time_slot = dict(slot_counter.most_common())
    return m


def add_shot_totals(metrics: Metrics, matches: Iterable[Any], puuid: str) -> Metrics:
    """Fold in per-match shot counts (head/body/leg) and assists."""
    for match in matches:
        player = match.player(puuid)
        if not player:
            continue
        metrics.headshots += player.stats.headshots
        metrics.bodyshots += player.stats.bodyshots
        metrics.legshots += player.stats.legshots
        metrics.assists += player.stats.assists
        if not metrics.damage:
            metrics.damage += player.stats.damage_made
    return metrics
