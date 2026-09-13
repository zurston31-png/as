"""Builders for hand-made matches, so each test controls exactly one thing."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from valcoach.models import (
    AbilityCasts,
    Kill,
    Match,
    MatchPlayer,
    PlayerLocation,
    PlayerRef,
    Round,
    RoundPlayerState,
    TeamResult,
    Weapon,
)

ME = "me"


def make_match(
    match_id: str = "m1",
    map_name: str = "Ascent",
    queue: str = "competitive",
    team_size: int = 5,
    agent: str = "Jett",
) -> Match:
    """A match with `me` on Blue plus mates m1.. and enemies e1.., no rounds yet."""
    match = Match(match_id=match_id, map_name=map_name, queue=queue, mode=queue,
                  started_at=1_700_000_000, provider="test")
    players: List[MatchPlayer] = [
        MatchPlayer(ref=PlayerRef(puuid=ME, name="You", tag="0000", team="Blue",
                                  agent=agent))
    ]
    for i in range(1, team_size):
        players.append(
            MatchPlayer(ref=PlayerRef(puuid=f"mate{i}", name=f"Mate{i}", tag=f"{i}",
                                      team="Blue", agent="Sova"))
        )
    for i in range(1, team_size + 1):
        players.append(
            MatchPlayer(ref=PlayerRef(puuid=f"foe{i}", name=f"Foe{i}", tag=f"{i}",
                                      team="Red", agent="Omen"))
        )
    match.players = players
    match.teams = {
        "Blue": TeamResult(team="Blue", won=True, rounds_won=13, rounds_lost=5),
        "Red": TeamResult(team="Red", won=False, rounds_won=5, rounds_lost=13),
    }
    return match


def add_round(
    match: Match,
    index: int = 0,
    winner: str = "Blue",
    attacking_team: str = "Blue",
    loadouts: Optional[Dict[str, int]] = None,
    casts: Optional[Dict[str, int]] = None,
    damage: Optional[Dict[str, int]] = None,
    plant_ms: Optional[int] = None,
    planter_team: str = "",
    result: str = "Eliminated",
) -> Round:
    rnd = Round(
        index=index, result=result, winning_team=winner,
        attacking_team=attacking_team,
        bomb_planted=plant_ms is not None,
        plant_time_ms=plant_ms,
        planter_team=planter_team or (attacking_team if plant_ms is not None else ""),
    )
    loadouts = loadouts or {}
    casts = casts or {}
    damage = damage or {}
    for player in match.players:
        puuid = player.ref.puuid
        rnd.states[puuid] = RoundPlayerState(
            puuid=puuid,
            loadout_value=loadouts.get(puuid, 3900),
            remaining_credits=1000,
            weapon=Weapon(id="vandal", name="Vandal"),
            armor="Heavy Bulletproof Vest",
            damage=damage.get(puuid, 140),
            headshots=1, bodyshots=3, legshots=0,
            casts=AbilityCasts(c=casts.get(puuid, 1), q=0, e=0, x=0),
        )
    match.rounds.append(rnd)
    return rnd


def add_kill(
    match: Match,
    round_index: int,
    time_ms: int,
    killer: str,
    victim: str,
    weapon: str = "Vandal",
    victim_pos: Tuple[float, float] = (0.0, 0.0),
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    assistants: Iterable[str] = (),
) -> Kill:
    """Add a kill. `positions` gives locations for other players at that instant."""
    refs = {p.ref.puuid: p.ref for p in match.players}
    locations = [
        PlayerLocation(puuid=puuid, team=refs[puuid].team, x=xy[0], y=xy[1],
                       view_radians=0.0)
        for puuid, xy in (positions or {}).items()
        if puuid in refs
    ]
    kill = Kill(
        round_index=round_index,
        time_in_round_ms=time_ms,
        time_in_match_ms=round_index * 100_000 + time_ms,
        killer=refs[killer],
        victim=refs[victim],
        weapon=Weapon(id=weapon.lower(), name=weapon),
        assistants=[refs[a] for a in assistants],
        victim_x=victim_pos[0],
        victim_y=victim_pos[1],
        player_locations=locations,
    )
    match.kills.append(kill)
    match.kills.sort(key=lambda k: (k.round_index, k.time_in_round_ms))
    return kill


def henrik_v2_payload() -> dict:
    """A minimal payload in the older (v2) HenrikDev shape."""
    return {
        "status": 200,
        "data": {
            "metadata": {
                "match_id": "v2-match",
                "map": "Bind",
                "game_start": 1_700_000_000,
                "game_length": 1800,
                "mode": "Competitive",
                "region": "eu",
            },
            "players": {
                "all_players": [
                    {
                        "puuid": "p1", "name": "Alpha", "tag": "1111",
                        "team": "Red", "character": "Jett",
                        "stats": {"kills": 10, "deaths": 5, "assists": 2,
                                  "score": 3000, "headshots": 8, "bodyshots": 20,
                                  "legshots": 1},
                        "damage_made": 2000, "damage_received": 1500,
                    },
                    {
                        "puuid": "p2", "name": "Beta", "tag": "2222",
                        "team": "Blue", "character": "Sage",
                        "stats": {"kills": 5, "deaths": 10, "assists": 4,
                                  "score": 1500, "headshots": 3, "bodyshots": 15,
                                  "legshots": 2},
                        "damage_made": 1200, "damage_received": 2100,
                    },
                ]
            },
            "teams": {
                "red": {"has_won": True, "rounds_won": 13, "rounds_lost": 7},
                "blue": {"has_won": False, "rounds_won": 7, "rounds_lost": 13},
            },
            "rounds": [
                {
                    "winning_team": "Red",
                    "end_type": "Eliminated",
                    "bomb_planted": True,
                    "bomb_defused": False,
                    "plant_events": {
                        "plant_location": {"x": 10, "y": 20},
                        "planted_by": {"puuid": "p1", "display_name": "Alpha#1111",
                                       "team": "Red"},
                        "plant_site": "A",
                        "plant_time_in_round": 30000,
                    },
                    "defuse_events": {},
                    "player_stats": [
                        {
                            "player_puuid": "p1",
                            "player_display_name": "Alpha#1111",
                            "player_team": "Red",
                            "ability_casts": {"c_cast": 1, "q_cast": 2, "e_cast": 0,
                                              "x_cast": 0},
                            "damage": 150, "headshots": 1, "bodyshots": 2,
                            "legshots": 0, "kills": 1, "score": 300,
                            "economy": {
                                "loadout_value": 3900, "remaining": 500,
                                "weapon": {"id": "vandal", "name": "Vandal"},
                                "armor": {"id": "heavy", "name": "Heavy Shields"},
                            },
                            "was_afk": False, "was_penalized": False,
                            "stayed_in_spawn": False,
                            "kill_events": [
                                {
                                    "kill_time_in_round": 21000,
                                    "kill_time_in_match": 21000,
                                    "killer_puuid": "p1",
                                    "killer_display_name": "Alpha#1111",
                                    "killer_team": "Red",
                                    "victim_puuid": "p2",
                                    "victim_display_name": "Beta#2222",
                                    "victim_team": "Blue",
                                    "victim_death_location": {"x": 100, "y": 200},
                                    "damage_weapon_id": "vandal",
                                    "damage_weapon_name": "Vandal",
                                    "assistants": [],
                                    "player_locations_on_kill": [
                                        {
                                            "player_puuid": "p1",
                                            "player_display_name": "Alpha#1111",
                                            "player_team": "Red",
                                            "view_radians": 1.2,
                                            "location": {"x": 150, "y": 250},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }


def riot_official_payload() -> dict:
    """A minimal payload in the official Riot / local-client shape."""
    return {
        "matchInfo": {
            "matchId": "riot-match",
            "mapId": "/Game/Maps/Duality/Duality",
            "gameLengthMillis": 1_800_000,
            "gameStartMillis": 1_700_000_000_000,
            "queueId": "competitive",
            "gameMode": "/Game/GameModes/Bomb/BombGameMode.BombGameMode_C",
            "seasonId": "season-1",
        },
        "players": [
            {
                "puuid": "r1", "gameName": "Gamma", "tagLine": "3333",
                "teamId": "Blue", "characterId": "AGENT-UUID", "partyId": "party1",
                "competitiveTier": 18, "accountLevel": 100,
                "stats": {"kills": 2, "deaths": 1, "assists": 0, "score": 500,
                          "roundsPlayed": 1},
            },
            {
                "puuid": "r2", "gameName": "Delta", "tagLine": "4444",
                "teamId": "Red", "characterId": "AGENT-UUID-2", "partyId": "party2",
                "competitiveTier": 17, "accountLevel": 90,
                "stats": {"kills": 1, "deaths": 2, "assists": 1, "score": 300,
                          "roundsPlayed": 1},
            },
        ],
        "teams": [
            {"teamId": "Blue", "won": True, "roundsPlayed": 20, "roundsWon": 13},
            {"teamId": "Red", "won": False, "roundsPlayed": 20, "roundsWon": 7},
        ],
        "roundResults": [
            {
                "roundNum": 0,
                "roundResult": "Eliminated",
                "winningTeam": "Blue",
                "bombPlanter": "r2",
                "plantRoundTime": 32000,
                "plantSite": "B",
                "playerStats": [
                    {
                        "puuid": "r1",
                        "score": 300,
                        "economy": {"loadoutValue": 4700, "remaining": 200,
                                    "weapon": "WEAPON-UUID", "armor": "ARMOR-UUID",
                                    "spent": 3900},
                        "ability": {"grenadeEffects": 1, "ability1Effects": 1,
                                    "ability2Effects": 0, "ultimateEffects": 0},
                        "damage": [
                            {"receiver": "r2", "damage": 150, "headshots": 1,
                             "bodyshots": 2, "legshots": 0}
                        ],
                        "kills": [
                            {
                                "timeSinceGameStartMillis": 40000,
                                "timeSinceRoundStartMillis": 25000,
                                "killer": "r1",
                                "victim": "r2",
                                "victimLocation": {"x": 500, "y": 600},
                                "assistants": [],
                                "playerLocations": [
                                    {"puuid": "r1", "viewRadians": 2.0,
                                     "location": {"x": 520, "y": 640}}
                                ],
                                "finishingDamage": {
                                    "damageType": "Weapon",
                                    "damageItem": "WEAPON-UUID",
                                    "isSecondaryFireMode": False,
                                },
                            }
                        ],
                        "wasAfk": False, "wasPenalized": False,
                        "stayedInSpawn": False,
                    }
                ],
            }
        ],
    }
