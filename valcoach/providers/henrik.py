"""HenrikDev API provider (https://docs.henrikdev.xyz).

This is the practical way for a normal player to get full match data: Riot's own
VAL-MATCH-V1 endpoints need a production key that is not granted for personal
projects, while HenrikDev issues free keys instantly.

The parser accepts **both** the v4 and the older v2/v3 payload shapes because
both are in active use, and it reads every field defensively: a provider adding,
renaming or dropping a key should degrade one analysis, never crash a report.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import (
    AbilityCasts,
    Kill,
    Match,
    MatchPlayer,
    PlayerLocation,
    PlayerMatchStats,
    PlayerRef,
    Round,
    RoundPlayerState,
    TeamResult,
    Weapon,
)
from ..webreq import get_json
from .base import ProviderError

BASE_URL = "https://api.henrikdev.xyz"
PAYLOAD_FORMAT = "henrik"

# Rounds per half, used to infer which side you were on in rounds without a
# spike plant to anchor the inference.
HALF_LENGTH_BY_MODE = {
    "competitive": 12,
    "unrated": 12,
    "premier": 12,
    "custom": 12,
    "swiftplay": 4,
    "spikerush": 3,
    "spike rush": 3,
}
DEFAULT_HALF_LENGTH = 12
SIDELESS_MODES = {"deathmatch", "teamdeathmatch", "team deathmatch", "escalation",
                  "replication", "snowballfight", "onefa"}


# --------------------------------------------------------------------------
# small readers
# --------------------------------------------------------------------------
def _g(obj: Any, *keys: str, default: Any = None) -> Any:
    """First present key from a mapping (``_g(d, 'a', 'b')``)."""
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _name_of(obj: Any, default: str = "") -> str:
    """Handle both ``"Ascent"`` and ``{"name": "Ascent", "id": ...}``."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return str(_g(obj, "name", "displayName", "id", default=default) or default)
    return default


def _team_name(value: Any) -> str:
    raw = _name_of(value) if not isinstance(value, str) else value
    raw = (raw or "").strip()
    low = raw.lower()
    if low in ("red", "blue"):
        return low.capitalize()
    return raw


def _epoch(value: Any) -> int:
    """Accept unix seconds, unix millis, or an ISO-8601 timestamp."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        v = float(value)
        return int(v / 1000) if v > 1e11 else int(v)
    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit():
        return _epoch(int(text))
    try:
        return int(
            _dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        )
    except ValueError:
        return 0


def _weapon(obj: Any) -> Weapon:
    if isinstance(obj, dict):
        return Weapon(id=str(_g(obj, "id", "weapon_id", default="") or ""),
                      name=str(_g(obj, "name", "weapon_name", default="") or ""))
    if isinstance(obj, str):
        return Weapon(id=obj, name="")
    return Weapon()


def _player_ref(obj: Any, agents: Optional[Dict[str, str]] = None) -> PlayerRef:
    """Parse the several player-reference shapes used across API versions."""
    if not isinstance(obj, dict):
        return PlayerRef(puuid=str(obj or ""))
    puuid = str(_g(obj, "puuid", "player_puuid", default="") or "")
    name = str(_g(obj, "name", "display_name", "player_display_name",
                  default="") or "")
    if "#" in name and not _g(obj, "tag"):
        name, tag = name.split("#", 1)
    else:
        tag = str(_g(obj, "tag", "tagline", default="") or "")
    team = _team_name(_g(obj, "team", "team_id", "player_team", default=""))
    agent = _name_of(_g(obj, "agent", "character", default=""))
    if not agent and agents:
        agent = agents.get(puuid, "")
    return PlayerRef(puuid=puuid, name=name, tag=tag, team=team, agent=agent)


def _flat_ref(ev: Dict[str, Any], role: str, agents: Dict[str, str]) -> PlayerRef:
    """v2 kill events flatten both sides onto the event (``killer_puuid`` ...)."""
    puuid = str(_g(ev, f"{role}_puuid", default="") or "")
    return PlayerRef(
        puuid=puuid,
        name=str(_g(ev, f"{role}_display_name", default="") or ""),
        team=_team_name(_g(ev, f"{role}_team", default="")),
        agent=agents.get(puuid, ""),
    )


def _locations(raw: Any, agents: Optional[Dict[str, str]] = None) -> List[PlayerLocation]:
    out: List[PlayerLocation] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        who = entry.get("player") if isinstance(entry.get("player"), dict) else entry
        ref = _player_ref(who, agents)
        loc = entry.get("location") if isinstance(entry.get("location"), dict) else entry
        x, y = _as_float(_g(loc, "x")), _as_float(_g(loc, "y"))
        if x is None or y is None or not ref.puuid:
            continue
        out.append(
            PlayerLocation(
                puuid=ref.puuid,
                team=ref.team,
                x=x,
                y=y,
                view_radians=_as_float(_g(entry, "view_radians", "viewRadians")),
            )
        )
    return out


# --------------------------------------------------------------------------
# payload parsing
# --------------------------------------------------------------------------
def unwrap(payload: Any) -> Any:
    """Henrik wraps responses as {"status": 200, "data": ...}."""
    while isinstance(payload, dict) and "data" in payload and (
        "status" in payload or "results" in payload or len(payload) <= 2
    ):
        payload = payload["data"]
    return payload


def is_match_payload(payload: Any) -> bool:
    data = unwrap(payload)
    return isinstance(data, dict) and "metadata" in data and (
        "players" in data or "rounds" in data
    )


def iter_match_payloads(payload: Any) -> Iterable[Any]:
    """Yield each match in a payload that may hold one match or a list."""
    data = unwrap(payload)
    if isinstance(data, list):
        for item in data:
            if is_match_payload(item):
                yield item
    elif is_match_payload(data):
        yield data


def parse_match(payload: Any) -> Optional[Match]:
    data = unwrap(payload)
    if not isinstance(data, dict):
        return None
    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        return None

    mode = _name_of(_g(meta, "mode", "game_mode", default="")).strip()
    queue = _name_of(_g(meta, "queue", "mode", default="")).strip()
    match = Match(
        match_id=str(_g(meta, "match_id", "matchid", "matchId", default="") or ""),
        map_name=_name_of(_g(meta, "map", default="")),
        mode=mode,
        queue=queue or mode,
        region=str(_g(meta, "region", default="") or ""),
        cluster=str(_g(meta, "cluster", default="") or ""),
        started_at=_epoch(_g(meta, "started_at", "game_start", "game_start_patched")),
        duration_ms=_duration_ms(meta),
        season=_name_of(_g(meta, "season", "season_id", default="")),
        provider=PAYLOAD_FORMAT,
    )

    match.players = _parse_players(data)
    agents = {p.ref.puuid: p.ref.agent for p in match.players}
    teams_of = {p.ref.puuid: p.ref.team for p in match.players}
    match.teams = _parse_teams(data, match.players)
    match.rounds = _parse_rounds(data, agents, teams_of)
    match.kills = _parse_kills(data, agents, teams_of)
    _infer_sides(match)
    return match


def _duration_ms(meta: Dict[str, Any]) -> int:
    ms = _g(meta, "game_length_in_ms", "duration_in_ms")
    if ms is not None:
        return _as_int(ms)
    secs = _g(meta, "game_length", "duration")
    if isinstance(secs, dict):  # v4 sometimes nests {"milliseconds": ...}
        return _as_int(_g(secs, "milliseconds", "ms", default=0))
    return _as_int(secs) * 1000


def _parse_players(data: Dict[str, Any]) -> List[MatchPlayer]:
    raw = data.get("players")
    if isinstance(raw, dict):            # v2: {"all_players": [...]} or {"red": [...]}
        if isinstance(raw.get("all_players"), list):
            raw = raw["all_players"]
        else:
            merged: List[Any] = []
            for value in raw.values():
                if isinstance(value, list):
                    merged.extend(value)
            raw = merged
    if not isinstance(raw, list):
        return []

    players: List[MatchPlayer] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ref = _player_ref(entry)
        stats_raw = entry.get("stats") if isinstance(entry.get("stats"), dict) else entry
        shots = stats_raw if isinstance(stats_raw, dict) else {}
        shot_block = shots.get("shots") if isinstance(shots.get("shots"), dict) else shots
        # v4 nests damage under stats, v2 puts it on the player entry.
        dmg = stats_raw.get("damage") if isinstance(stats_raw, dict) else None
        if not isinstance(dmg, dict):
            dmg = entry.get("damage") if isinstance(entry.get("damage"), dict) else {}
        stats = PlayerMatchStats(
            kills=_as_int(_g(stats_raw, "kills")),
            deaths=_as_int(_g(stats_raw, "deaths")),
            assists=_as_int(_g(stats_raw, "assists")),
            score=_as_int(_g(stats_raw, "score")),
            headshots=_as_int(_g(shot_block, "headshots", "head")),
            bodyshots=_as_int(_g(shot_block, "bodyshots", "body")),
            legshots=_as_int(_g(shot_block, "legshots", "leg")),
            damage_made=_as_int(
                _g(dmg, "made", "dealt") if dmg else _g(entry, "damage_made", "damage")
            ),
            damage_received=_as_int(
                _g(dmg, "received") if dmg else _g(entry, "damage_received")
            ),
        )
        tier = entry.get("tier") if isinstance(entry.get("tier"), dict) else None
        players.append(
            MatchPlayer(
                ref=ref,
                party_id=str(_g(entry, "party_id", "party", default="") or ""),
                rank=_name_of(tier) if tier else str(
                    _g(entry, "currenttier_patched", "current_tier_patched", default="")
                    or ""
                ),
                level=_as_int(_g(entry, "account_level", "level")),
                stats=stats,
            )
        )
    return players


def _parse_teams(data: Dict[str, Any], players: List[MatchPlayer]) -> Dict[str, TeamResult]:
    out: Dict[str, TeamResult] = {}
    raw = data.get("teams")
    if isinstance(raw, list):            # v4
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            team = _team_name(_g(entry, "team_id", "team", default=""))
            rounds = entry.get("rounds") if isinstance(entry.get("rounds"), dict) else {}
            out[team] = TeamResult(
                team=team,
                won=_g(entry, "won", "has_won"),
                rounds_won=_as_int(_g(rounds, "won") if rounds else _g(entry, "rounds_won")),
                rounds_lost=_as_int(
                    _g(rounds, "lost") if rounds else _g(entry, "rounds_lost")
                ),
            )
    elif isinstance(raw, dict):          # v2: {"red": {...}, "blue": {...}}
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            team = _team_name(key)
            out[team] = TeamResult(
                team=team,
                won=_g(entry, "has_won", "won"),
                rounds_won=_as_int(_g(entry, "rounds_won")),
                rounds_lost=_as_int(_g(entry, "rounds_lost")),
            )
    if not out:
        for p in players:
            out.setdefault(p.ref.team, TeamResult(team=p.ref.team))
    return out


def _parse_round_state(entry: Dict[str, Any], agents: Dict[str, str]) -> RoundPlayerState:
    who = entry.get("player") if isinstance(entry.get("player"), dict) else entry
    ref = _player_ref(who, agents)
    econ = entry.get("economy") if isinstance(entry.get("economy"), dict) else {}
    stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else entry
    casts_raw = entry.get("ability_casts") or {}
    casts = AbilityCasts(
        c=_as_int(_g(casts_raw, "c_cast", "c_casts", "c")),
        q=_as_int(_g(casts_raw, "q_cast", "q_casts", "q")),
        e=_as_int(_g(casts_raw, "e_cast", "e_casts", "e")),
        x=_as_int(_g(casts_raw, "x_cast", "x_casts", "x")),
    )
    return RoundPlayerState(
        puuid=ref.puuid,
        loadout_value=_as_int(_g(econ, "loadout_value", "loadoutValue")),
        remaining_credits=_as_int(_g(econ, "remaining")),
        spent_credits=_as_int(_g(econ, "spent")),
        weapon=_weapon(_g(econ, "weapon")),
        armor=_name_of(_g(econ, "armor", default="")),
        damage=_as_int(_g(stats, "damage")),
        headshots=_as_int(_g(stats, "headshots")),
        bodyshots=_as_int(_g(stats, "bodyshots")),
        legshots=_as_int(_g(stats, "legshots")),
        kills=_as_int(_g(stats, "kills")),
        score=_as_int(_g(stats, "score")),
        casts=casts,
        was_afk=bool(_g(entry, "was_afk", default=False)),
        stayed_in_spawn=bool(_g(entry, "stayed_in_spawn", default=False)),
        received_penalty=bool(_g(entry, "received_penalty", "was_penalized",
                                 default=False)),
    )


def _parse_rounds(
    data: Dict[str, Any], agents: Dict[str, str], teams_of: Dict[str, str]
) -> List[Round]:
    raw = data.get("rounds")
    if not isinstance(raw, list):
        return []
    rounds: List[Round] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        rnd = Round(
            index=_as_int(_g(entry, "id", "round", "round_num"), default=i)
            if _g(entry, "id", "round", "round_num") is not None else i,
            result=_name_of(_g(entry, "result", "end_type", default="")),
            winning_team=_team_name(_g(entry, "winning_team", default="")),
            ceremony=_name_of(_g(entry, "ceremony", default="")),
        )

        plant = _g(entry, "plant", "plant_events")
        if isinstance(plant, dict):
            planter = _g(plant, "player", "planted_by")
            ptime = _g(plant, "round_time_in_ms", "plant_time_in_round")
            if planter or ptime is not None:
                rnd.bomb_planted = True
                rnd.plant_time_ms = _as_int(ptime) if ptime is not None else None
                rnd.plant_site = _name_of(_g(plant, "site", "plant_site", default=""))
                pref = _player_ref(planter, agents) if planter else PlayerRef("")
                rnd.planter_team = pref.team or teams_of.get(pref.puuid, "")
        if not rnd.bomb_planted:
            rnd.bomb_planted = bool(_g(entry, "bomb_planted", default=False))

        defuse = _g(entry, "defuse", "defuse_events")
        if isinstance(defuse, dict):
            defuser = _g(defuse, "player", "defused_by")
            dtime = _g(defuse, "round_time_in_ms", "defuse_time_in_round")
            if defuser or dtime is not None:
                rnd.bomb_defused = True
                rnd.defuse_time_ms = _as_int(dtime) if dtime is not None else None
                dref = _player_ref(defuser, agents) if defuser else PlayerRef("")
                rnd.defuser_team = dref.team or teams_of.get(dref.puuid, "")
        if not rnd.bomb_defused:
            rnd.bomb_defused = bool(_g(entry, "bomb_defused", default=False))

        states = _g(entry, "stats", "player_stats") or []
        if isinstance(states, list):
            for st_entry in states:
                if not isinstance(st_entry, dict):
                    continue
                state = _parse_round_state(st_entry, agents)
                if state.puuid:
                    rnd.states[state.puuid] = state
        rounds.append(rnd)
    return rounds


def _parse_kills(
    data: Dict[str, Any], agents: Dict[str, str], teams_of: Dict[str, str]
) -> List[Kill]:
    raw = data.get("kills")
    events: List[Tuple[int, Dict[str, Any]]] = []
    if isinstance(raw, list):                     # v4: flat list
        events = [(-1, e) for e in raw if isinstance(e, dict)]
    else:                                         # v2: nested in round player stats
        for i, rnd in enumerate(data.get("rounds") or []):
            if not isinstance(rnd, dict):
                continue
            for st in _g(rnd, "player_stats", "stats") or []:
                if not isinstance(st, dict):
                    continue
                for ev in st.get("kill_events") or []:
                    if isinstance(ev, dict):
                        events.append((i, ev))

    kills: List[Kill] = []
    seen = set()
    for fallback_round, ev in events:
        if isinstance(ev.get("killer"), dict):          # v4: nested refs
            killer = _player_ref(ev.get("killer"), agents)
            victim = _player_ref(ev.get("victim"), agents)
        else:                                          # v2: refs flattened onto event
            killer = _flat_ref(ev, "killer", agents)
            victim = _flat_ref(ev, "victim", agents)
        if not killer.team:
            killer = PlayerRef(killer.puuid, killer.name, killer.tag,
                               teams_of.get(killer.puuid, ""), killer.agent)
        if not victim.team:
            victim = PlayerRef(victim.puuid, victim.name, victim.tag,
                               teams_of.get(victim.puuid, ""), victim.agent)

        loc = _g(ev, "location", "victim_death_location")
        weapon = _g(ev, "weapon")
        if weapon is None:
            weapon = {
                "id": _g(ev, "damage_weapon_id", default=""),
                "name": _g(ev, "damage_weapon_name", default=""),
            }
        round_index = _g(ev, "round", "round_index")
        kill = Kill(
            round_index=_as_int(round_index, default=fallback_round)
            if round_index is not None else fallback_round,
            time_in_round_ms=_as_int(
                _g(ev, "time_in_round_in_ms", "kill_time_in_round")
            ),
            time_in_match_ms=_as_int(
                _g(ev, "time_in_match_in_ms", "kill_time_in_match")
            ),
            killer=killer,
            victim=victim,
            weapon=_weapon(weapon),
            assistants=[
                _player_ref(a if isinstance(a, dict) else {"puuid": a}, agents)
                for a in (_g(ev, "assistants", default=[]) or [])
            ],
            victim_x=_as_float(_g(loc, "x")),
            victim_y=_as_float(_g(loc, "y")),
            player_locations=_locations(
                _g(ev, "player_locations", "player_locations_on_kill", default=[]),
                agents,
            ),
            secondary_fire=bool(_g(ev, "secondary_fire_mode", default=False)),
        )
        key = (kill.round_index, kill.time_in_match_ms, kill.killer.puuid,
               kill.victim.puuid)
        if key in seen:
            continue
        seen.add(key)
        kills.append(kill)

    kills.sort(key=lambda k: (k.round_index, k.time_in_round_ms))
    return kills


def _infer_sides(match: Match) -> None:
    """Work out which team was attacking in each round.

    Only attackers can plant and only defenders can defuse, so plant/defuse
    events are hard anchors. Unanchored rounds are filled in from the half
    structure (sides swap at halftime, and every round in overtime).
    """
    mode_key = (match.mode or match.queue or "").lower().replace(" ", "")
    if mode_key in {m.replace(" ", "") for m in SIDELESS_MODES}:
        return
    teams = [t for t in match.teams if t]
    if len(teams) != 2:
        return
    other = {teams[0]: teams[1], teams[1]: teams[0]}
    half = HALF_LENGTH_BY_MODE.get(
        (match.queue or match.mode or "").lower(), DEFAULT_HALF_LENGTH
    )

    anchors: Dict[int, str] = {}
    for rnd in match.rounds:
        if rnd.planter_team in other:
            anchors[rnd.index] = rnd.planter_team
        elif rnd.defuser_team in other:
            anchors[rnd.index] = other[rnd.defuser_team]
    if not anchors:
        return

    def bucket(index: int) -> int:
        """Rounds that are guaranteed to share a side assignment."""
        if index < half:
            return 0
        if index < 2 * half:
            return 1
        return 2 + (index - 2 * half)      # overtime: one bucket per round

    by_bucket: Dict[int, str] = {}
    for index, team in anchors.items():
        by_bucket.setdefault(bucket(index), team)

    for rnd in match.rounds:
        b = bucket(rnd.index)
        if b in by_bucket:
            rnd.attacking_team = by_bucket[b]
            continue
        if b in (0, 1):                    # flip from the other regulation half
            sibling = by_bucket.get(1 - b)
            if sibling:
                rnd.attacking_team = other[sibling]
                continue
        ot = sorted(k for k in by_bucket if k >= 2)
        if b >= 2 and ot:
            nearest = min(ot, key=lambda k: abs(k - b))
            team = by_bucket[nearest]
            rnd.attacking_team = team if (b - nearest) % 2 == 0 else other[team]


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------
class HenrikProvider:
    """Client for api.henrikdev.xyz. Needs a free API key (see README)."""

    name = "henrik"
    payload_format = PAYLOAD_FORMAT

    def __init__(
        self,
        api_key: str = "",
        region: str = "na",
        platform: str = "pc",
        base_url: str = BASE_URL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or ""
        self.region = (region or "na").lower()
        self.platform = (platform or "pc").lower()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": self.api_key} if self.api_key else {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return get_json(
            f"{self.base_url}{path}",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )

    # -- Provider protocol
    def resolve_player(self, riot_id: str) -> Tuple[str, str]:
        riot_id = (riot_id or "").strip()
        if "#" not in riot_id:
            raise ProviderError(
                f"Riot ID must look like Name#TAG (got {riot_id!r})"
            )
        name, tag = riot_id.split("#", 1)
        from urllib.parse import quote

        data = unwrap(
            self._get(f"/valorant/v2/account/{quote(name.strip())}/{quote(tag.strip())}")
        )
        if not isinstance(data, dict) or not data.get("puuid"):
            raise ProviderError(f"could not resolve Riot ID {riot_id!r}")
        canonical = f"{data.get('name', name)}#{data.get('tag', tag)}"
        if data.get("region"):
            self.region = str(data["region"]).lower()
        return str(data["puuid"]), canonical

    def recent_matches(
        self, puuid: str, count: int = 10, queue: Optional[str] = None
    ) -> List[Any]:
        """Full match payloads. Henrik returns whole matches here, so one call
        replaces a list-then-fetch-each round trip."""
        data = self._get(
            f"/valorant/v4/by-puuid/matches/{self.platform}/{self.region}/{puuid}",
            params={"size": max(1, min(int(count), 10)), "mode": queue or None},
        )
        payloads = list(iter_match_payloads(data))
        if payloads:
            return payloads
        # Older deployments only expose v3 (no platform segment).
        data = self._get(
            f"/valorant/v3/by-puuid/matches/{self.region}/{puuid}",
            params={"size": max(1, min(int(count), 10)), "mode": queue or None},
        )
        return list(iter_match_payloads(data))

    def recent_match_ids(
        self, puuid: str, count: int = 10, queue: Optional[str] = None
    ) -> List[str]:
        ids: List[str] = []
        for payload in self.recent_matches(puuid, count, queue):
            meta = unwrap(payload).get("metadata") or {}
            mid = _g(meta, "match_id", "matchid", default="")
            if mid:
                ids.append(str(mid))
        return ids

    def fetch_match(self, match_id: str) -> Any:
        for path in (
            f"/valorant/v4/match/{self.region}/{match_id}",
            f"/valorant/v2/match/{match_id}",
        ):
            try:
                data = self._get(path)
            except Exception:  # noqa: BLE001 - try the next API version
                continue
            for payload in iter_match_payloads(data):
                return payload
        raise ProviderError(f"match {match_id} not found")

    def parse(self, payload: Any) -> Optional[Match]:
        return parse_match(payload)
