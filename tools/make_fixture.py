#!/usr/bin/env python3
"""Generate synthetic HenrikDev-shaped match payloads.

Used for the bundled demo and for tests, so both run with no API key and no
network. The data is *plausible*, not real: round flow, economy, ability casts,
kill positions and timings are sampled from a seeded RNG, with a few habits
deliberately baked into the subject player so the detectors have something to
find (over-peeking on attack, dying isolated at the same spot, low headshot
rate on eco rounds).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

MAPS = ["Ascent", "Bind", "Haven", "Split", "Lotus", "Sunset"]
RIFLES = [("Vandal", "Rifle"), ("Phantom", "Rifle"), ("Guardian", "Rifle")]
SMGS = [("Spectre", "SMG"), ("Stinger", "SMG")]
PISTOLS = [("Classic", "Sidearm"), ("Ghost", "Sidearm"), ("Sheriff", "Sidearm"),
           ("Frenzy", "Sidearm")]
SNIPERS = [("Operator", "Sniper"), ("Marshal", "Sniper")]
AGENTS_BY_ROLE = {
    "duelist": ["Jett", "Raze", "Reyna", "Neon"],
    "initiator": ["Sova", "Breach", "Fade", "Gekko"],
    "controller": ["Omen", "Brimstone", "Viper", "Astra"],
    "sentinel": ["Killjoy", "Cypher", "Sage", "Chamber"],
}
ENEMY_NAMES = ["Kaido", "Mercy", "Volt", "Nyx", "Riven", "Bolt", "Echo", "Sable"]
MATE_NAMES = ["Pike", "Juno", "Rook", "Lark", "Vesper", "Onyx"]


def _uuid(seed: str) -> str:
    h = hashlib.sha1(seed.encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _ref(player: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "puuid": player["puuid"],
        "name": player["name"],
        "tag": player["tag"],
        "team": player["team_id"],
    }


class MatchBuilder:
    def __init__(self, rng: random.Random, subject: str, index: int,
                 started_at: int):
        self.rng = rng
        self.index = index
        self.started_at = started_at
        self.map_name = rng.choice(MAPS)
        name, _, tag = subject.partition("#")
        self.subject_name = name or "You"
        self.subject_tag = tag or "0000"
        self.players = self._make_players()
        self.me = self.players[0]
        # A habit to find: one favourite (bad) spot on this map.
        self.bad_spot = (rng.uniform(-6000, 6000), rng.uniform(-6000, 6000))

    # -- setup
    def _make_players(self) -> List[Dict[str, Any]]:
        rng = self.rng
        players: List[Dict[str, Any]] = []
        roles = list(AGENTS_BY_ROLE)
        my_agent = rng.choice(AGENTS_BY_ROLE[rng.choice(roles)])
        players.append(
            self._player(self.subject_name, self.subject_tag, "Blue", my_agent, 1)
        )
        mates = rng.sample(MATE_NAMES, 4)
        for i, mate in enumerate(mates):
            agent = rng.choice(AGENTS_BY_ROLE[roles[i % len(roles)]])
            players.append(self._player(mate, f"{1000 + i}", "Blue", agent, 2 + i))
        foes = rng.sample(ENEMY_NAMES, 5)
        for i, foe in enumerate(foes):
            agent = rng.choice(AGENTS_BY_ROLE[roles[i % len(roles)]])
            players.append(self._player(foe, f"{2000 + i}", "Red", agent, 7 + i))
        return players

    def _player(self, name: str, tag: str, team: str, agent: str,
                slot: int) -> Dict[str, Any]:
        return {
            "puuid": _uuid(f"{name}#{tag}"),
            "name": name,
            "tag": tag,
            "team_id": team,
            "party_id": _uuid(f"party-{team}-{slot // 3}"),
            "agent": {"id": _uuid(agent), "name": agent},
            "account_level": self.rng.randint(30, 400),
            "tier": {"id": 18, "name": self.rng.choice(
                ["Gold 3", "Platinum 1", "Platinum 3", "Diamond 1"])},
            "stats": {"score": 0, "kills": 0, "deaths": 0, "assists": 0,
                      "headshots": 0, "bodyshots": 0, "legshots": 0,
                      "damage": {"dealt": 0, "received": 0}},
        }

    # -- helpers
    def _loadout(self, round_index: int, credits_state: str) -> Dict[str, Any]:
        rng = self.rng
        if credits_state == "eco":
            weapon, kind = rng.choice(PISTOLS)
            value = rng.choice([0, 400, 800, 950])
        elif credits_state == "half":
            weapon, kind = rng.choice(SMGS + PISTOLS)
            value = rng.randint(1900, 3400)
        else:
            weapon, kind = rng.choice(RIFLES + RIFLES + SNIPERS)
            value = rng.randint(3900, 5600)
        return {
            "loadout_value": value,
            "remaining": rng.randint(0, 4200),
            "spent": value,
            "weapon": {"id": _uuid(weapon), "name": weapon, "type": kind},
            "armor": {"id": _uuid("heavy"), "name": "Heavy Bulletproof Vest"
                      if value > 1500 else "Light Shield"},
        }

    def _position(self, near: Optional[tuple] = None, jitter: float = 900.0):
        rng = self.rng
        if near:
            return (near[0] + rng.uniform(-jitter, jitter),
                    near[1] + rng.uniform(-jitter, jitter))
        return (rng.uniform(-7000, 7000), rng.uniform(-7000, 7000))

    # -- build
    def build(self) -> Dict[str, Any]:
        rng = self.rng
        attackers_first_half = "Blue" if self.index % 2 == 0 else "Red"
        rounds: List[Dict[str, Any]] = []
        kills: List[Dict[str, Any]] = []
        score = {"Blue": 0, "Red": 0}
        round_index = 0
        clock_ms = 0

        while max(score.values()) < 13 and round_index < 24:
            attackers = (
                attackers_first_half if round_index < 12
                else ("Red" if attackers_first_half == "Blue" else "Blue")
            )
            state = self._economy_state(round_index, score)
            round_payload, round_kills, winner = self._round(
                round_index, attackers, state, clock_ms
            )
            score[winner] += 1
            rounds.append(round_payload)
            kills.extend(round_kills)
            clock_ms += rng.randint(55_000, 105_000)
            round_index += 1

        self._roll_up_player_stats(rounds, kills)
        winner_team = "Blue" if score["Blue"] > score["Red"] else "Red"
        return {
            "status": 200,
            "data": {
                "metadata": {
                    "match_id": _uuid(f"match-{self.index}-{self.started_at}"),
                    "map": {"id": _uuid(self.map_name), "name": self.map_name},
                    "game_version": "release-11.00-shipping-7-000000",
                    "game_length_in_ms": clock_ms,
                    "started_at": self.started_at,
                    "is_completed": True,
                    "queue": {"id": "competitive", "name": "Competitive",
                              "mode_type": "Standard"},
                    "mode": "Competitive",
                    "season": {"id": _uuid("season"), "short": "e12a3"},
                    "platform": "pc",
                    "region": "na",
                    "cluster": "US West",
                },
                "players": self.players,
                "observers": [],
                "coaches": [],
                "teams": [
                    {
                        "team_id": team,
                        "rounds": {"won": score[team],
                                   "lost": score["Red" if team == "Blue" else "Blue"]},
                        "won": team == winner_team,
                    }
                    for team in ("Blue", "Red")
                ],
                "rounds": rounds,
                "kills": kills,
            },
        }

    def _economy_state(self, round_index: int, score: Dict[str, int]) -> str:
        if round_index in (0, 12):
            return "eco"
        roll = self.rng.random()
        if roll < 0.2:
            return "eco"
        if roll < 0.35:
            return "half"
        return "full"

    def _round(self, round_index: int, attackers: str, econ_state: str,
               clock_ms: int):
        rng = self.rng
        my_team = self.me["team_id"]
        i_attack = attackers == my_team
        alive = {p["puuid"]: True for p in self.players}
        kills: List[Dict[str, Any]] = []
        time_ms = rng.randint(4_000, 12_000)

        # Baked-in habit: on attack the subject peeks early and often dies first.
        my_death_time: Optional[int] = None
        if i_attack and rng.random() < 0.55:
            my_death_time = rng.randint(6_000, 18_000)
        elif rng.random() < 0.45:
            my_death_time = rng.randint(20_000, 70_000)

        events: List[Dict[str, Any]] = []
        n_events = rng.randint(3, 9)
        for _ in range(n_events):
            time_ms += rng.randint(2_500, 9_000)
            if time_ms > 95_000:
                break
            if my_death_time and time_ms >= my_death_time and alive[self.me["puuid"]]:
                events.append(self._kill_event(
                    round_index, my_death_time, clock_ms, alive,
                    victim=self.me, near=self.bad_spot if rng.random() < 0.55 else None,
                ))
                alive[self.me["puuid"]] = False
                my_death_time = None
                time_ms = self._maybe_trade(
                    events, alive, my_team, round_index, time_ms, clock_ms
                )
                continue
            killer_team = "Blue" if rng.random() < 0.5 else "Red"
            candidates = [
                p for p in self.players
                if p["team_id"] == killer_team and alive[p["puuid"]]
            ]
            victims = [
                p for p in self.players
                if p["team_id"] != killer_team and alive[p["puuid"]]
                and p["puuid"] != self.me["puuid"]
            ]
            if not candidates or not victims:
                continue
            killer = rng.choice(candidates)
            victim = rng.choice(victims)
            events.append(self._kill_event(
                round_index, time_ms, clock_ms, alive, victim=victim, killer=killer,
            ))
            alive[victim["puuid"]] = False

        # A scheduled death that the event loop ran out of room for still happens.
        if my_death_time and alive[self.me["puuid"]]:
            events.append(self._kill_event(
                round_index, my_death_time, clock_ms, alive, victim=self.me,
                near=self.bad_spot if rng.random() < 0.55 else None,
            ))
            alive[self.me["puuid"]] = False
            self._maybe_trade(
                events, alive, my_team, round_index, my_death_time or time_ms, clock_ms
            )
        kills.extend(events)

        blue_alive = sum(
            1 for p in self.players if p["team_id"] == "Blue" and alive[p["puuid"]]
        )
        red_alive = sum(
            1 for p in self.players if p["team_id"] == "Red" and alive[p["puuid"]]
        )
        if blue_alive == 0:
            winner, result = "Red", "Eliminated"
        elif red_alive == 0:
            winner, result = "Blue", "Eliminated"
        else:
            winner = attackers if rng.random() < 0.5 else (
                "Red" if attackers == "Blue" else "Blue")
            result = "Bomb detonated" if winner == attackers else "Bomb defused"

        planted = result in ("Bomb detonated", "Bomb defused") or rng.random() < 0.45
        plant = None
        defuse = None
        if planted:
            planter = rng.choice([
                p for p in self.players if p["team_id"] == attackers
            ])
            plant_time = rng.randint(25_000, 65_000)
            plant = {
                "round_time_in_ms": plant_time,
                "site": rng.choice(["A", "B", "C"]),
                "location": {"x": self._position()[0], "y": self._position()[1]},
                "player": _ref(planter),
                "player_locations": [],
            }
            if result == "Bomb defused":
                defuser = rng.choice([
                    p for p in self.players if p["team_id"] != attackers
                ])
                defuse = {
                    "round_time_in_ms": plant_time + rng.randint(8_000, 30_000),
                    "location": {"x": self._position()[0], "y": self._position()[1]},
                    "player": _ref(defuser),
                    "player_locations": [],
                }

        return (
            {
                "id": round_index,
                "result": result,
                "ceremony": "",
                "winning_team": winner,
                "plant": plant,
                "defuse": defuse,
                "stats": [
                    self._round_stats(p, econ_state, kills, round_index, i_attack)
                    for p in self.players
                ],
            },
            kills,
            winner,
        )

    def _maybe_trade(self, events: List[Dict[str, Any]], alive: Dict[str, bool],
                     my_team: str, round_index: int, time_ms: int,
                     clock_ms: int) -> int:
        """A teammate avenges the last kill ~45% of the time — i.e. a trade."""
        rng = self.rng
        killer_puuid = events[-1]["killer"]["puuid"]
        if rng.random() >= 0.45 or not alive.get(killer_puuid):
            return time_ms
        avengers = [
            p for p in self.players
            if p["team_id"] == my_team and alive[p["puuid"]]
        ]
        killer_player = next(
            (p for p in self.players if p["puuid"] == killer_puuid), None
        )
        if not avengers or killer_player is None:
            return time_ms
        time_ms += rng.randint(600, 2_800)
        events.append(self._kill_event(
            round_index, time_ms, clock_ms, alive,
            victim=killer_player, killer=rng.choice(avengers),
        ))
        alive[killer_puuid] = False
        return time_ms

    def _kill_event(self, round_index: int, time_ms: int, clock_ms: int,
                    alive: Dict[str, bool], victim: Dict[str, Any],
                    killer: Optional[Dict[str, Any]] = None,
                    near: Optional[tuple] = None) -> Dict[str, Any]:
        rng = self.rng
        if killer is None:
            foes = [
                p for p in self.players
                if p["team_id"] != victim["team_id"] and alive[p["puuid"]]
            ]
            killer = rng.choice(foes) if foes else self.players[-1]
        vx, vy = self._position(near)
        weapon, kind = rng.choice(RIFLES + RIFLES + SMGS + PISTOLS + SNIPERS)
        # Engagement range follows the weapon, so distance-based analysis is
        # exercised with something like a real distribution.
        kill_range = {
            "Sniper": (2800.0, 7000.0), "Rifle": (700.0, 3400.0),
            "SMG": (350.0, 1300.0), "Sidearm": (300.0, 1800.0),
        }.get(kind, (500.0, 2500.0))

        locations = []
        for p in self.players:
            if not alive[p["puuid"]] or p["puuid"] == victim["puuid"]:
                continue
            # Baked-in habit: the subject's team is usually far away when they die.
            if p["team_id"] == victim["team_id"] and victim["puuid"] == self.me["puuid"]:
                px, py = self._position((vx, vy), jitter=rng.uniform(1500, 4200))
            elif p["puuid"] == killer["puuid"]:
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(*kill_range)
                px, py = vx + dist * math.cos(angle), vy + dist * math.sin(angle)
            else:
                px, py = self._position((vx, vy), jitter=rng.uniform(800, 3000))
            locations.append({
                "player": _ref(p),
                "view_radians": round(rng.uniform(0, 6.28), 3),
                "location": {"x": round(px, 1), "y": round(py, 1)},
            })

        # Roughly a quarter of kills are assisted, as in a real game.
        assistants = []
        if rng.random() < 0.28:
            helpers = [
                p for p in self.players
                if p["team_id"] == killer["team_id"] and alive[p["puuid"]]
                and p["puuid"] != killer["puuid"]
            ]
            if helpers:
                assistants = [_ref(rng.choice(helpers))]

        return {
            "round": round_index,
            "time_in_round_in_ms": time_ms,
            "time_in_match_in_ms": clock_ms + time_ms,
            "killer": _ref(killer),
            "victim": _ref(victim),
            "assistants": assistants,
            "location": {"x": round(vx, 1), "y": round(vy, 1)},
            "weapon": {"type": kind, "id": _uuid(weapon), "name": weapon},
            "secondary_fire_mode": False,
            "player_locations": locations,
        }

    def _round_stats(self, player: Dict[str, Any], econ_state: str,
                     kills: List[Dict[str, Any]], round_index: int,
                     i_attack: bool) -> Dict[str, Any]:
        rng = self.rng
        is_me = player["puuid"] == self.me["puuid"]
        state = econ_state if not is_me else (
            "eco" if econ_state == "eco" else econ_state
        )
        round_kills = [
            k for k in kills
            if k["round"] == round_index and k["killer"]["puuid"] == player["puuid"]
        ]
        damage = sum(rng.randint(110, 160) for _ in round_kills) + rng.randint(0, 130)
        # Baked-in habit: a below-average headshot rate for the subject.
        hs_rate = 0.14 if is_me else rng.uniform(0.18, 0.3)
        shots = max(1, round(damage / 45))
        headshots = sum(1 for _ in range(shots) if rng.random() < hs_rate)
        legshots = sum(1 for _ in range(shots) if rng.random() < 0.05)
        bodyshots = max(0, shots - headshots - legshots)
        casts = {
            "c_casts": rng.randint(0, 2), "q_casts": rng.randint(0, 2),
            "e_casts": rng.randint(0, 2), "x_casts": 1 if rng.random() < 0.12 else 0,
        }
        if is_me and rng.random() < 0.45:      # dies with utility in the bank
            casts = {"c_casts": 0, "q_casts": 0, "e_casts": rng.randint(0, 1),
                     "x_casts": 0}
        return {
            "ability_casts": casts,
            "player": _ref(player),
            "damage_events": [],
            "stats": {
                "bodyshots": bodyshots, "headshots": headshots, "legshots": legshots,
                "damage": damage, "score": damage * 2, "kills": len(round_kills),
            },
            "economy": self._loadout(round_index, state),
            "was_afk": False,
            "received_penalty": False,
            "stayed_in_spawn": False,
        }

    def _roll_up_player_stats(self, rounds: List[Dict[str, Any]],
                              kills: List[Dict[str, Any]]) -> None:
        for player in self.players:
            puuid = player["puuid"]
            stats = player["stats"]
            stats["kills"] = sum(1 for k in kills if k["killer"]["puuid"] == puuid)
            stats["deaths"] = sum(1 for k in kills if k["victim"]["puuid"] == puuid)
            stats["assists"] = sum(
                1 for k in kills
                for a in k.get("assistants", []) if a.get("puuid") == puuid
            )
            for rnd in rounds:
                for st in rnd["stats"]:
                    if st["player"]["puuid"] != puuid:
                        continue
                    stats["headshots"] += st["stats"]["headshots"]
                    stats["bodyshots"] += st["stats"]["bodyshots"]
                    stats["legshots"] += st["stats"]["legshots"]
                    stats["score"] += st["stats"]["score"]
                    stats["damage"]["dealt"] += st["stats"]["damage"]
            stats["damage"]["received"] = int(stats["damage"]["dealt"] * 0.95)


def generate(count: int = 3, seed: int = 7, subject: str = "You#0000",
             first_start: int = 1757000000) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        started = first_start + i * 3600 * 5
        out.append(MatchBuilder(rng, subject, i, started).build())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subject", default="You#0000")
    parser.add_argument("--out", default="-", help="output file, or - for stdout")
    args = parser.parse_args(argv)

    payloads = generate(args.count, args.seed, args.subject)
    text = json.dumps(payloads, indent=1)
    if args.out == "-":
        print(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.out} ({len(payloads)} matches)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
