"""Turn raw match data into per-round, per-death and per-kill context.

This is the layer that answers "how did I die?" rather than "how many times did
I die?". For every death it reconstructs the state of the round at that instant:
who was still alive on both sides, how far away the nearest living teammate was,
what you were holding, what utility you still had, whether a teammate avenged
you, and where on the map it happened.

Everything downstream (metrics, detectors, the coaching prompt) reads these
objects, so the rest of the tool never has to touch kill events again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..maps import MapIndex, agent_role
from ..models import UNKNOWN_SIDE, Kill, Match, Round

# Economy buckets, in credits of gear carried into the round.
FULL_BUY = 3900
HALF_BUY = 2000
SAVEABLE_LOADOUT = 2900

# A death inside this many ms of round start is an "opening" death.
OPENING_WINDOW_MS = 15_000
LATE_WINDOW_MS = 45_000

WEAPON_CLASSES = {
    "classic": "sidearm", "shorty": "sidearm", "frenzy": "sidearm",
    "ghost": "sidearm", "sheriff": "sidearm",
    "stinger": "smg", "spectre": "smg",
    "bucky": "shotgun", "judge": "shotgun",
    "bulldog": "rifle", "guardian": "rifle", "phantom": "rifle", "vandal": "rifle",
    "marshal": "sniper", "outlaw": "sniper", "operator": "sniper",
    "ares": "lmg", "odin": "lmg",
    "melee": "melee", "tactical knife": "melee",
}
LONG_RANGE_CLASSES = {"sniper", "rifle", "lmg"}
SHORT_RANGE_CLASSES = {"shotgun", "smg", "sidearm", "melee"}


def weapon_class(name: str, weapon_id: str = "") -> str:
    key = (name or "").strip().lower()
    if key in WEAPON_CLASSES:
        return WEAPON_CLASSES[key]
    if not key and not weapon_id:
        return "ability"
    return "other"


def econ_state(loadout_value: int) -> str:
    if loadout_value >= FULL_BUY:
        return "full"
    if loadout_value >= HALF_BUY:
        return "half"
    return "eco"


def time_bucket(ms: int) -> str:
    if ms <= OPENING_WINDOW_MS:
        return "opening"
    if ms <= LATE_WINDOW_MS:
        return "mid"
    return "late"


@dataclass
class DeathContext:
    match_id: str
    map_name: str
    queue: str
    started_at: int
    round_index: int
    time_in_round_ms: int
    side: str
    killer_name: str
    killer_agent: str
    killer_puuid: str
    weapon: str
    weapon_kind: str
    assisted_by: int = 0
    x: Optional[float] = None
    y: Optional[float] = None
    place: str = ""
    first_death_of_round: bool = False
    traded: bool = False
    trade_delay_ms: Optional[int] = None
    nearest_teammate_distance: Optional[float] = None
    nearest_teammate: str = ""
    distance_to_killer: Optional[float] = None
    teammates_alive: int = 0
    enemies_alive: int = 0
    numbers: str = "even"              # "up" | "even" | "down"
    loadout_value: int = 0
    weapon_held: str = ""
    econ: str = "eco"
    credits_remaining: int = 0
    util_casts_in_round: int = 0
    util_unused: bool = False
    post_plant: bool = False
    round_won: bool = False
    time_slot: str = "mid"
    self_inflicted: bool = False

    @property
    def isolated(self) -> bool:
        """No living teammate close enough to have helped or traded."""
        return (
            self.nearest_teammate_distance is not None
            and self.nearest_teammate_distance >= 1800.0
        )

    def describe(self) -> str:
        """One human-readable line, as shown by ``valcoach deaths``."""
        bits = [
            f"R{self.round_index + 1}",
            f"{self.time_in_round_ms / 1000:.0f}s",
            self.side if self.side != UNKNOWN_SIDE else "",
            f"killed by {self.killer_name or 'unknown'}"
            + (f" ({self.killer_agent})" if self.killer_agent else ""),
            f"with {self.weapon or 'unknown'}",
        ]
        if self.place:
            bits.append(f"at {self.place}")
        tags = []
        if self.first_death_of_round:
            tags.append("first blood against")
        if not self.traded:
            tags.append("not traded")
        if self.isolated and self.nearest_teammate_distance is not None:
            tags.append(f"nearest mate {self.nearest_teammate_distance / 100:.0f}m")
        if self.numbers == "down":
            tags.append(f"{self.teammates_alive + 1}v{self.enemies_alive} down")
        if self.util_unused:
            tags.append("no util used")
        if self.econ != "eco" and self.loadout_value:
            tags.append(f"{self.loadout_value}c loadout")
        if self.post_plant:
            tags.append("post-plant")
        line = " · ".join(b for b in bits if b)
        return f"{line}" + (f"  [{', '.join(tags)}]" if tags else "")


@dataclass
class KillContext:
    match_id: str
    round_index: int
    time_in_round_ms: int
    side: str
    victim_name: str
    victim_puuid: str
    weapon: str
    first_blood: bool = False
    trade_kill: bool = False
    headshot_round: bool = False


@dataclass
class RoundContext:
    match_id: str
    round_index: int
    side: str
    won: bool
    econ: str
    loadout_value: int
    survived: bool
    got_kill: bool
    got_assist: bool
    was_traded: bool
    damage: int
    util_casts: int
    kills: int
    clutch_attempt: bool = False
    clutch_enemies: int = 0
    clutch_won: bool = False
    afk: bool = False
    stayed_in_spawn: bool = False
    saved: bool = False              # survived a lost round holding real gear
    lost_gear_value: int = 0         # gear handed over by dying in a lost round


@dataclass
class MatchContext:
    match_id: str
    map_name: str
    queue: str
    agent: str
    role: str
    started_at: int
    won: Optional[bool]
    rounds: List[RoundContext] = field(default_factory=list)
    deaths: List[DeathContext] = field(default_factory=list)
    kills: List[KillContext] = field(default_factory=list)
    score: Tuple[int, int] = (0, 0)
    assists: int = 0


def _distance(ax: Optional[float], ay: Optional[float],
              bx: Optional[float], by: Optional[float]) -> Optional[float]:
    if None in (ax, ay, bx, by):
        return None
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def build_match_context(
    match: Match,
    puuid: str,
    trade_window_ms: int = 4000,
    map_index: Optional[MapIndex] = None,
) -> Optional[MatchContext]:
    """Build the full context for one player in one match."""
    player = match.player(puuid)
    if player is None:
        return None
    maps = map_index if map_index is not None else MapIndex()
    my_team = player.ref.team
    teammates = {p.ref.puuid for p in match.teammates(puuid)}
    enemies = {p.ref.puuid for p in match.enemies(puuid)}
    names = {p.ref.puuid: p.ref.riot_id for p in match.players}

    team_result = match.teams.get(my_team)
    other_team = next((t for t in match.teams if t != my_team), "")
    ctx = MatchContext(
        match_id=match.match_id,
        map_name=match.map_name,
        queue=match.queue or match.mode,
        agent=player.ref.agent,
        role=agent_role(player.ref.agent),
        started_at=match.started_at,
        won=team_result.won if team_result else None,
        score=(
            team_result.rounds_won if team_result else 0,
            match.teams[other_team].rounds_won if other_team in match.teams else 0,
        ),
        assists=player.stats.assists,
    )

    for rnd in match.rounds:
        round_kills = match.kills_in_round(rnd.index)
        state = rnd.states.get(puuid)
        side = rnd.side_for(my_team)
        won = bool(rnd.winning_team and rnd.winning_team == my_team)

        alive = {p.ref.puuid: True for p in match.players}
        my_death: Optional[DeathContext] = None
        clutch_attempt = False
        clutch_enemies = 0

        for kill in round_kills:
            victim, killer = kill.victim.puuid, kill.killer.puuid
            if victim == puuid and alive.get(puuid):
                my_death = _death_context(
                    match, rnd, kill, puuid, teammates, enemies, alive, names,
                    side, won, state, maps, round_kills, trade_window_ms,
                )
                ctx.deaths.append(my_death)
            elif killer == puuid and victim != puuid:
                is_first = bool(round_kills) and round_kills[0] is kill
                trade = _was_trade_kill(
                    kill, round_kills, teammates, trade_window_ms
                )
                ctx.kills.append(
                    KillContext(
                        match_id=match.match_id,
                        round_index=rnd.index,
                        time_in_round_ms=kill.time_in_round_ms,
                        side=side,
                        victim_name=names.get(victim, kill.victim.name),
                        victim_puuid=victim,
                        weapon=kill.weapon.label,
                        first_blood=is_first,
                        trade_kill=trade,
                    )
                )
            alive[victim] = False

            # Last one standing with enemies left: a clutch situation.
            if (
                not clutch_attempt
                and alive.get(puuid)
                and sum(1 for p in teammates if alive.get(p)) == 0
                and sum(1 for p in enemies if alive.get(p)) >= 1
            ):
                clutch_attempt = True
                clutch_enemies = sum(1 for p in enemies if alive.get(p))

        survived = my_death is None
        round_kill_count = sum(1 for k in ctx.kills if k.round_index == rnd.index)
        got_assist = any(
            k.round_index == rnd.index
            and any(a.puuid == puuid for a in k.assistants)
            for k in round_kills
        )
        loadout = state.loadout_value if state else 0
        lost_gear = 0
        saved = False
        if not won and loadout >= SAVEABLE_LOADOUT:
            if survived:
                saved = True
            else:
                lost_gear = loadout

        ctx.rounds.append(
            RoundContext(
                match_id=match.match_id,
                round_index=rnd.index,
                side=side,
                won=won,
                econ=econ_state(loadout),
                loadout_value=loadout,
                survived=survived,
                got_kill=round_kill_count > 0,
                got_assist=got_assist,
                was_traded=bool(my_death and my_death.traded),
                damage=state.damage if state else 0,
                util_casts=state.casts.total if state else 0,
                kills=round_kill_count,
                clutch_attempt=clutch_attempt,
                clutch_enemies=clutch_enemies,
                clutch_won=clutch_attempt and won,
                afk=bool(state.was_afk) if state else False,
                stayed_in_spawn=bool(state.stayed_in_spawn) if state else False,
                saved=saved,
                lost_gear_value=lost_gear,
            )
        )

    return ctx


def _was_trade_kill(
    kill: Kill,
    round_kills: Sequence[Kill],
    teammates: Iterable[str],
    window_ms: int,
) -> bool:
    """True when this kill avenged a teammate who died moments earlier."""
    mates = set(teammates)
    for other in round_kills:
        if other is kill:
            continue
        if other.victim.puuid not in mates:
            continue
        if other.killer.puuid != kill.victim.puuid:
            continue
        delta = kill.time_in_round_ms - other.time_in_round_ms
        if 0 <= delta <= window_ms:
            return True
    return False


def _death_context(
    match: Match,
    rnd: Round,
    kill: Kill,
    puuid: str,
    teammates: set,
    enemies: set,
    alive: Dict[str, bool],
    names: Dict[str, str],
    side: str,
    round_won: bool,
    state,
    maps: MapIndex,
    round_kills: Sequence[Kill],
    trade_window_ms: int,
) -> DeathContext:
    locations = {loc.puuid: loc for loc in kill.player_locations}
    my_loc = locations.get(puuid)
    vx = kill.victim_x if kill.victim_x is not None else (my_loc.x if my_loc else None)
    vy = kill.victim_y if kill.victim_y is not None else (my_loc.y if my_loc else None)

    nearest_dist: Optional[float] = None
    nearest_name = ""
    for mate in teammates:
        if not alive.get(mate):
            continue
        loc = locations.get(mate)
        if loc is None:
            continue
        dist = _distance(vx, vy, loc.x, loc.y)
        if dist is not None and (nearest_dist is None or dist < nearest_dist):
            nearest_dist, nearest_name = dist, names.get(mate, "")

    killer_loc = locations.get(kill.killer.puuid)
    dist_to_killer = (
        _distance(vx, vy, killer_loc.x, killer_loc.y) if killer_loc else None
    )

    traded, trade_delay = False, None
    for other in round_kills:
        if other is kill:
            continue
        if other.killer.puuid in teammates and other.victim.puuid == kill.killer.puuid:
            delta = other.time_in_round_ms - kill.time_in_round_ms
            if 0 <= delta <= trade_window_ms:
                traded, trade_delay = True, delta
                break

    mates_alive = sum(1 for p in teammates if alive.get(p))
    foes_alive = sum(1 for p in enemies if alive.get(p))
    mine = mates_alive + 1                      # I am still alive at this instant
    numbers = "even" if mine == foes_alive else ("up" if mine > foes_alive else "down")
    util = state.casts.total if state else 0
    loadout = state.loadout_value if state else 0

    return DeathContext(
        match_id=match.match_id,
        map_name=match.map_name,
        queue=match.queue or match.mode,
        started_at=match.started_at,
        round_index=rnd.index,
        time_in_round_ms=kill.time_in_round_ms,
        side=side,
        killer_name=names.get(kill.killer.puuid, kill.killer.name),
        killer_agent=kill.killer.agent,
        killer_puuid=kill.killer.puuid,
        weapon=kill.weapon.label,
        weapon_kind=weapon_class(kill.weapon.name, kill.weapon.id),
        assisted_by=len(kill.assistants),
        x=vx,
        y=vy,
        place=maps.describe(match.map_name, vx, vy),
        first_death_of_round=bool(round_kills) and round_kills[0] is kill,
        traded=traded,
        trade_delay_ms=trade_delay,
        nearest_teammate_distance=nearest_dist,
        nearest_teammate=nearest_name,
        distance_to_killer=dist_to_killer,
        teammates_alive=mates_alive,
        enemies_alive=foes_alive,
        numbers=numbers,
        loadout_value=loadout,
        weapon_held=state.weapon.label if state else "",
        econ=econ_state(loadout),
        credits_remaining=state.remaining_credits if state else 0,
        util_casts_in_round=util,
        util_unused=util == 0,
        post_plant=bool(
            rnd.plant_time_ms is not None
            and kill.time_in_round_ms > rnd.plant_time_ms
        ),
        round_won=round_won,
        time_slot=time_bucket(kill.time_in_round_ms),
        self_inflicted=kill.is_self_kill,
    )


def build_contexts(
    matches: Iterable[Match],
    puuid: str,
    trade_window_ms: int = 4000,
    map_index: Optional[MapIndex] = None,
) -> List[MatchContext]:
    maps = map_index if map_index is not None else MapIndex()
    out = []
    for match in matches:
        ctx = build_match_context(match, puuid, trade_window_ms, maps)
        if ctx:
            out.append(ctx)
    out.sort(key=lambda c: c.started_at)
    return out
