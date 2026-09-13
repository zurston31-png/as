"""Normalized data model.

Providers (HenrikDev, official Riot API, local client, JSON dumps) all parse into
these types so that everything downstream — storage, analysis, coaching — is
independent of which API the data came from.

Coordinates are raw Valorant game-space units as reported by Riot's match data
(x/y, roughly +/- 10000). They are only ever used for relative comparisons
(distance between players, clustering of death spots) and for nearest-callout
lookups, so no absolute calibration is required.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

ATTACK = "attack"
DEFENSE = "defense"
UNKNOWN_SIDE = "unknown"


@dataclass(frozen=True)
class PlayerRef:
    """Minimal identity of a player inside a match."""

    puuid: str
    name: str = ""
    tag: str = ""
    team: str = ""          # "Red" / "Blue" (deathmatch: per-player teams)
    agent: str = ""

    @property
    def riot_id(self) -> str:
        if self.name and self.tag:
            return f"{self.name}#{self.tag}"
        return self.name or self.puuid


@dataclass
class Weapon:
    id: str = ""
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or self.id or "unknown"


@dataclass
class PlayerLocation:
    puuid: str
    team: str
    x: float
    y: float
    view_radians: Optional[float] = None


@dataclass
class Kill:
    """A single kill event. `player_locations` is the position of every *living*
    player at the instant of the kill, which is what makes isolation and
    numbers-advantage analysis possible."""

    round_index: int
    time_in_round_ms: int
    time_in_match_ms: int
    killer: PlayerRef
    victim: PlayerRef
    weapon: Weapon = field(default_factory=Weapon)
    assistants: List[PlayerRef] = field(default_factory=list)
    victim_x: Optional[float] = None
    victim_y: Optional[float] = None
    player_locations: List[PlayerLocation] = field(default_factory=list)
    secondary_fire: bool = False

    @property
    def is_self_kill(self) -> bool:
        """Falls / spike / round-loss self damage show up as killer == victim."""
        return self.killer.puuid == self.victim.puuid


@dataclass
class AbilityCasts:
    c: int = 0
    q: int = 0
    e: int = 0
    x: int = 0

    @property
    def total(self) -> int:
        return self.c + self.q + self.e + self.x

    @property
    def basics(self) -> int:
        """Everything except the ultimate."""
        return self.c + self.q + self.e


@dataclass
class RoundPlayerState:
    """What one player did / held in one round."""

    puuid: str
    loadout_value: int = 0
    remaining_credits: int = 0
    spent_credits: int = 0
    weapon: Weapon = field(default_factory=Weapon)
    armor: str = ""
    damage: int = 0
    headshots: int = 0
    bodyshots: int = 0
    legshots: int = 0
    kills: int = 0
    score: int = 0
    casts: AbilityCasts = field(default_factory=AbilityCasts)
    was_afk: bool = False
    stayed_in_spawn: bool = False
    received_penalty: bool = False

    @property
    def shots(self) -> int:
        return self.headshots + self.bodyshots + self.legshots


@dataclass
class Round:
    index: int
    result: str = ""                       # "Eliminated", "Bomb detonated", ...
    winning_team: str = ""
    ceremony: str = ""
    bomb_planted: bool = False
    plant_time_ms: Optional[int] = None
    plant_site: str = ""
    planter_team: str = ""
    bomb_defused: bool = False
    defuse_time_ms: Optional[int] = None
    defuser_team: str = ""
    attacking_team: str = ""               # inferred; "" when unknown
    states: Dict[str, RoundPlayerState] = field(default_factory=dict)

    def side_for(self, team: str) -> str:
        if not self.attacking_team or not team:
            return UNKNOWN_SIDE
        return ATTACK if team == self.attacking_team else DEFENSE


@dataclass
class PlayerMatchStats:
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    score: int = 0
    headshots: int = 0
    bodyshots: int = 0
    legshots: int = 0
    damage_made: int = 0
    damage_received: int = 0


@dataclass
class MatchPlayer:
    ref: PlayerRef
    party_id: str = ""
    rank: str = ""
    level: int = 0
    stats: PlayerMatchStats = field(default_factory=PlayerMatchStats)


@dataclass
class TeamResult:
    team: str
    won: Optional[bool] = None
    rounds_won: int = 0
    rounds_lost: int = 0


@dataclass
class Match:
    match_id: str
    map_name: str = ""
    mode: str = ""
    queue: str = ""
    region: str = ""
    cluster: str = ""
    started_at: int = 0                    # unix seconds
    duration_ms: int = 0
    season: str = ""
    provider: str = ""
    players: List[MatchPlayer] = field(default_factory=list)
    teams: Dict[str, TeamResult] = field(default_factory=dict)
    rounds: List[Round] = field(default_factory=list)
    kills: List[Kill] = field(default_factory=list)

    # ---- lookups -------------------------------------------------------
    def player(self, puuid: str) -> Optional[MatchPlayer]:
        for p in self.players:
            if p.ref.puuid == puuid:
                return p
        return None

    def find_player(self, needle: str) -> Optional[MatchPlayer]:
        """Resolve a puuid or a 'Name#TAG' (case-insensitive) to a player."""
        needle = (needle or "").strip()
        if not needle:
            return None
        low = needle.lower()
        for p in self.players:
            if p.ref.puuid == needle or p.ref.riot_id.lower() == low:
                return p
        for p in self.players:
            if p.ref.name.lower() == low:
                return p
        return None

    def team_of(self, puuid: str) -> str:
        p = self.player(puuid)
        return p.ref.team if p else ""

    def teammates(self, puuid: str) -> List[MatchPlayer]:
        team = self.team_of(puuid)
        if not team:
            return []
        return [p for p in self.players if p.ref.team == team and p.ref.puuid != puuid]

    def enemies(self, puuid: str) -> List[MatchPlayer]:
        team = self.team_of(puuid)
        if not team:
            return []
        return [p for p in self.players if p.ref.team != team]

    def round_at(self, index: int) -> Optional[Round]:
        for r in self.rounds:
            if r.index == index:
                return r
        return None

    def kills_in_round(self, index: int) -> List[Kill]:
        return sorted(
            (k for k in self.kills if k.round_index == index),
            key=lambda k: k.time_in_round_ms,
        )

    @property
    def rounds_played(self) -> int:
        return len(self.rounds) or self._rounds_from_teams()

    def _rounds_from_teams(self) -> int:
        if not self.teams:
            return 0
        return max((t.rounds_won + t.rounds_lost) for t in self.teams.values())

    def won_by(self, puuid: str) -> Optional[bool]:
        team = self.teams.get(self.team_of(puuid))
        return team.won if team else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def dedupe_refs(refs: Iterable[PlayerRef]) -> List[PlayerRef]:
    seen: Dict[str, PlayerRef] = {}
    for r in refs:
        seen.setdefault(r.puuid, r)
    return list(seen.values())


def as_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
