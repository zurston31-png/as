"""Official Riot VAL-MATCH-V1 payload support.

Two things use this module:

* ``RiotOfficialProvider`` — the public Riot API. Match endpoints need a
  *production* key, which Riot does not hand out for personal projects, so most
  people will not use this path. It is here because it costs little and anyone
  who does hold a key gets first-party data.
* ``parse_match`` — the same payload shape the **local game client** serves
  (``providers/local_client.py``), which needs no key at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
from .henrik import _as_float, _as_int, _g, _infer_sides, _team_name

PAYLOAD_FORMAT = "riot"

# Riot's internal map code names, which is what ``mapId`` contains.
MAP_CODE_NAMES = {
    "ascent": "Ascent",
    "duality": "Bind",
    "triad": "Haven",
    "bonsai": "Split",
    "port": "Icebox",
    "foxtrot": "Breeze",
    "canyon": "Fracture",
    "pitt": "Pearl",
    "jam": "Lotus",
    "juliett": "Sunset",
    "infinity": "Abyss",
    "range": "The Range",
    "piazza": "Piazza",
    "district": "District",
    "kasbah": "Kasbah",
    "drift": "Drift",
    "glitch": "Glitch",
}

ROUTING_BY_REGION = {
    "na": "americas", "br": "americas", "latam": "americas",
    "eu": "europe", "tr": "europe", "ru": "europe",
    "kr": "asia", "ap": "asia", "jp": "asia",
}


def map_name_from_id(map_id: Any) -> str:
    text = str(map_id or "")
    if not text:
        return ""
    tail = text.rstrip("/").rsplit("/", 1)[-1]
    return MAP_CODE_NAMES.get(tail.lower(), tail)


def is_match_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "matchInfo" in payload


def parse_match(payload: Any, assets: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[Match]:
    """Parse an official-format match (Riot API or local client)."""
    if not is_match_payload(payload):
        return None
    assets = assets or {}
    agent_names = assets.get("agents", {})
    weapon_names = assets.get("weapons", {})

    info = payload.get("matchInfo") or {}
    match = Match(
        match_id=str(_g(info, "matchId", default="") or ""),
        map_name=map_name_from_id(_g(info, "mapId")),
        mode=str(_g(info, "gameMode", default="") or "").rstrip("/").rsplit("/", 1)[-1],
        queue=str(_g(info, "queueId", "queueID", default="") or ""),
        region=str(_g(info, "region", default="") or ""),
        started_at=int(_as_int(_g(info, "gameStartMillis")) / 1000),
        duration_ms=_as_int(_g(info, "gameLengthMillis")),
        season=str(_g(info, "seasonId", default="") or ""),
        provider=PAYLOAD_FORMAT,
    )

    agents: Dict[str, str] = {}
    teams_of: Dict[str, str] = {}
    for entry in payload.get("players") or []:
        if not isinstance(entry, dict):
            continue
        puuid = str(_g(entry, "puuid", default="") or "")
        char_id = str(_g(entry, "characterId", default="") or "")
        agent = agent_names.get(char_id.lower(), char_id)
        team = _team_name(_g(entry, "teamId", default=""))
        agents[puuid] = agent
        teams_of[puuid] = team
        stats = entry.get("stats") or {}
        match.players.append(
            MatchPlayer(
                ref=PlayerRef(
                    puuid=puuid,
                    name=str(_g(entry, "gameName", default="") or ""),
                    tag=str(_g(entry, "tagLine", default="") or ""),
                    team=team,
                    agent=agent,
                ),
                party_id=str(_g(entry, "partyId", default="") or ""),
                rank=str(_g(entry, "competitiveTier", default="") or ""),
                level=_as_int(_g(entry, "accountLevel")),
                stats=PlayerMatchStats(
                    kills=_as_int(_g(stats, "kills")),
                    deaths=_as_int(_g(stats, "deaths")),
                    assists=_as_int(_g(stats, "assists")),
                    score=_as_int(_g(stats, "score")),
                ),
            )
        )

    for entry in payload.get("teams") or []:
        if not isinstance(entry, dict):
            continue
        team = _team_name(_g(entry, "teamId", default=""))
        played = _as_int(_g(entry, "roundsPlayed"))
        won = _as_int(_g(entry, "roundsWon"))
        match.teams[team] = TeamResult(
            team=team, won=bool(_g(entry, "won", default=False)),
            rounds_won=won, rounds_lost=max(0, played - won),
        )

    def locations(raw: Any) -> List[PlayerLocation]:
        out: List[PlayerLocation] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            puuid = str(_g(item, "puuid", default="") or "")
            loc = item.get("location") or {}
            x, y = _as_float(_g(loc, "x")), _as_float(_g(loc, "y"))
            if not puuid or x is None or y is None:
                continue
            out.append(
                PlayerLocation(
                    puuid=puuid, team=teams_of.get(puuid, ""), x=x, y=y,
                    view_radians=_as_float(_g(item, "viewRadians")),
                )
            )
        return out

    def ref(puuid: Any) -> PlayerRef:
        puuid = str(puuid or "")
        player = match.player(puuid)
        if player:
            return player.ref
        return PlayerRef(puuid=puuid, team=teams_of.get(puuid, ""),
                         agent=agents.get(puuid, ""))

    for i, rnd_raw in enumerate(payload.get("roundResults") or []):
        if not isinstance(rnd_raw, dict):
            continue
        planter = str(_g(rnd_raw, "bombPlanter", default="") or "")
        defuser = str(_g(rnd_raw, "bombDefuser", default="") or "")
        plant_time = _g(rnd_raw, "plantRoundTime")
        defuse_time = _g(rnd_raw, "defuseRoundTime")
        rnd = Round(
            index=_as_int(_g(rnd_raw, "roundNum"), default=i),
            result=str(_g(rnd_raw, "roundResult", default="") or ""),
            winning_team=_team_name(_g(rnd_raw, "winningTeam", default="")),
            ceremony=str(_g(rnd_raw, "roundCeremony", default="") or ""),
            bomb_planted=bool(planter),
            plant_time_ms=_as_int(plant_time) if planter else None,
            plant_site=str(_g(rnd_raw, "plantSite", default="") or ""),
            planter_team=teams_of.get(planter, ""),
            bomb_defused=bool(defuser),
            defuse_time_ms=_as_int(defuse_time) if defuser else None,
            defuser_team=teams_of.get(defuser, ""),
        )

        for st in rnd_raw.get("playerStats") or []:
            if not isinstance(st, dict):
                continue
            puuid = str(_g(st, "puuid", default="") or "")
            if not puuid:
                continue
            econ = st.get("economy") or {}
            ability = st.get("ability") or {}
            weapon_id = str(_g(econ, "weapon", default="") or "")
            damage = headshots = bodyshots = legshots = 0
            for dmg in st.get("damage") or []:
                if not isinstance(dmg, dict):
                    continue
                damage += _as_int(_g(dmg, "damage"))
                headshots += _as_int(_g(dmg, "headshots"))
                bodyshots += _as_int(_g(dmg, "bodyshots"))
                legshots += _as_int(_g(dmg, "legshots"))
            rnd.states[puuid] = RoundPlayerState(
                puuid=puuid,
                loadout_value=_as_int(_g(econ, "loadoutValue")),
                remaining_credits=_as_int(_g(econ, "remaining")),
                spent_credits=_as_int(_g(econ, "spent")),
                weapon=Weapon(id=weapon_id,
                              name=weapon_names.get(weapon_id.lower(), "")),
                armor=str(_g(econ, "armor", default="") or ""),
                damage=damage,
                headshots=headshots,
                bodyshots=bodyshots,
                legshots=legshots,
                kills=len(st.get("kills") or []),
                score=_as_int(_g(st, "score")),
                casts=AbilityCasts(
                    c=_as_int(_g(ability, "grenadeEffects", "c_cast")),
                    q=_as_int(_g(ability, "ability1Effects", "q_cast")),
                    e=_as_int(_g(ability, "ability2Effects", "e_cast")),
                    x=_as_int(_g(ability, "ultimateEffects", "x_cast")),
                ),
                was_afk=bool(_g(st, "wasAfk", default=False)),
                stayed_in_spawn=bool(_g(st, "stayedInSpawn", default=False)),
                received_penalty=bool(_g(st, "wasPenalized", default=False)),
            )

            for ev in st.get("kills") or []:
                if not isinstance(ev, dict):
                    continue
                finishing = ev.get("finishingDamage") or {}
                weapon_item = str(_g(finishing, "damageItem", default="") or "")
                victim_loc = ev.get("victimLocation") or {}
                match.kills.append(
                    Kill(
                        round_index=rnd.index,
                        time_in_round_ms=_as_int(_g(ev, "timeSinceRoundStartMillis")),
                        time_in_match_ms=_as_int(_g(ev, "timeSinceGameStartMillis")),
                        killer=ref(_g(ev, "killer")),
                        victim=ref(_g(ev, "victim")),
                        weapon=Weapon(
                            id=weapon_item,
                            name=weapon_names.get(weapon_item.lower(), ""),
                        ),
                        assistants=[ref(a) for a in (ev.get("assistants") or [])],
                        victim_x=_as_float(_g(victim_loc, "x")),
                        victim_y=_as_float(_g(victim_loc, "y")),
                        player_locations=locations(ev.get("playerLocations")),
                        secondary_fire=bool(
                            _g(finishing, "isSecondaryFireMode", default=False)
                        ),
                    )
                )
        match.rounds.append(rnd)

    match.kills.sort(key=lambda k: (k.round_index, k.time_in_round_ms))
    for player in match.players:
        deaths = sum(1 for k in match.kills if k.victim.puuid == player.ref.puuid)
        if not player.stats.deaths:
            player.stats.deaths = deaths
        made = hs = bs = ls = 0
        for rnd in match.rounds:
            st = rnd.states.get(player.ref.puuid)
            if st:
                made += st.damage
                hs += st.headshots
                bs += st.bodyshots
                ls += st.legshots
        player.stats.damage_made = player.stats.damage_made or made
        player.stats.headshots = player.stats.headshots or hs
        player.stats.bodyshots = player.stats.bodyshots or bs
        player.stats.legshots = player.stats.legshots or ls
    _infer_sides(match)
    return match


class RiotOfficialProvider:
    """Public Riot API client (needs a production key for match endpoints)."""

    name = "riot"
    payload_format = PAYLOAD_FORMAT

    def __init__(self, api_key: str = "", region: str = "na", timeout: float = 30.0,
                 assets: Optional[Dict[str, Dict[str, str]]] = None):
        self.api_key = api_key or ""
        self.region = (region or "na").lower()
        self.timeout = timeout
        self.assets = assets or {}

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ProviderError("the official Riot API requires an API key")
        return {"X-Riot-Token": self.api_key}

    def _host(self, routing: bool = False) -> str:
        shard = ROUTING_BY_REGION.get(self.region, "americas") if routing else self.region
        return f"https://{shard}.api.riotgames.com"

    def resolve_player(self, riot_id: str) -> Tuple[str, str]:
        if "#" not in (riot_id or ""):
            raise ProviderError(f"Riot ID must look like Name#TAG (got {riot_id!r})")
        name, tag = riot_id.split("#", 1)
        from urllib.parse import quote

        data = get_json(
            f"{self._host(routing=True)}/riot/account/v1/accounts/by-riot-id/"
            f"{quote(name.strip())}/{quote(tag.strip())}",
            headers=self._headers(), timeout=self.timeout,
        )
        if not isinstance(data, dict) or not data.get("puuid"):
            raise ProviderError(f"could not resolve Riot ID {riot_id!r}")
        return str(data["puuid"]), f"{data.get('gameName', name)}#{data.get('tagLine', tag)}"

    def recent_match_ids(self, puuid: str, count: int = 10,
                         queue: Optional[str] = None) -> List[str]:
        data = get_json(
            f"{self._host()}/val/match/v1/matchlists/by-puuid/{puuid}",
            headers=self._headers(), timeout=self.timeout,
        )
        history = (data or {}).get("history") or []
        ids = []
        for entry in history:
            if queue and str(entry.get("queueId", "")).lower() != queue.lower():
                continue
            if entry.get("matchId"):
                ids.append(str(entry["matchId"]))
        return ids[: max(1, int(count))]

    def fetch_match(self, match_id: str) -> Any:
        return get_json(
            f"{self._host()}/val/match/v1/matches/{match_id}",
            headers=self._headers(), timeout=self.timeout,
        )

    def parse(self, payload: Any) -> Optional[Match]:
        return parse_match(payload, self.assets)
