"""Parsers must handle every payload shape in the wild, and fail softly."""

from __future__ import annotations

import json
import os
import unittest

from valcoach.providers import detect_format, iter_payloads, parse_payload
from valcoach.providers import henrik, riot_official
from valcoach.models import ATTACK, DEFENSE

from .helpers import henrik_v2_payload, riot_official_payload

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "valcoach", "fixtures", "demo_matches.json",
)


class TestFormatDetection(unittest.TestCase):
    def test_detects_henrik_v4(self):
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            payloads = json.load(handle)
        self.assertEqual(detect_format(payloads[0]), "henrik")

    def test_detects_henrik_v2(self):
        self.assertEqual(detect_format(henrik_v2_payload()), "henrik")

    def test_detects_riot_official(self):
        self.assertEqual(detect_format(riot_official_payload()), "riot")

    def test_unknown_payload(self):
        self.assertEqual(detect_format({"hello": "world"}), "")
        self.assertIsNone(parse_payload({"hello": "world"}))

    def test_iter_payloads_handles_lists_and_wrappers(self):
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            payloads = json.load(handle)
        self.assertEqual(len(list(iter_payloads(payloads))), len(payloads))
        self.assertEqual(len(list(iter_payloads(payloads[0]))), 1)
        self.assertEqual(len(list(iter_payloads({"status": 200, "data": []}))), 0)


class TestHenrikV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            cls.payloads = json.load(handle)
        cls.match = parse_payload(cls.payloads[0])

    def test_metadata(self):
        m = self.match
        self.assertTrue(m.match_id)
        self.assertTrue(m.map_name)
        self.assertEqual(m.queue.lower(), "competitive")
        self.assertGreater(m.started_at, 1_600_000_000)
        self.assertGreater(m.duration_ms, 0)

    def test_players_and_teams(self):
        m = self.match
        self.assertEqual(len(m.players), 10)
        self.assertEqual({p.ref.team for p in m.players}, {"Red", "Blue"})
        self.assertEqual(set(m.teams), {"Red", "Blue"})
        me = m.find_player("You#0000")
        self.assertIsNotNone(me)
        self.assertEqual(len(m.teammates(me.ref.puuid)), 4)
        self.assertEqual(len(m.enemies(me.ref.puuid)), 5)
        self.assertGreater(me.stats.kills + me.stats.deaths, 0)

    def test_rounds_and_kills(self):
        m = self.match
        self.assertGreaterEqual(len(m.rounds), 13)
        self.assertGreater(len(m.kills), 20)
        for rnd in m.rounds:
            self.assertEqual(len(rnd.states), 10)
        first = m.kills[0]
        self.assertNotEqual(first.killer.puuid, "")
        self.assertNotEqual(first.victim.puuid, "")
        self.assertNotEqual(first.killer.puuid, first.victim.puuid)
        self.assertIsNotNone(first.victim_x)
        self.assertTrue(first.player_locations)

    def test_side_inference(self):
        m = self.match
        sides = {r.side_for("Blue") for r in m.rounds}
        self.assertTrue({ATTACK, DEFENSE} <= sides)
        # Sides must swap exactly once at the half in a standard match.
        first_half = {r.attacking_team for r in m.rounds if r.index < 12}
        second_half = {r.attacking_team for r in m.rounds if 12 <= r.index < 24}
        self.assertEqual(len(first_half), 1)
        if second_half:
            self.assertEqual(len(second_half), 1)
            self.assertNotEqual(first_half, second_half)

    def test_kills_sorted_and_deduped(self):
        keys = [
            (k.round_index, k.time_in_match_ms, k.killer.puuid, k.victim.puuid)
            for k in self.match.kills
        ]
        self.assertEqual(len(keys), len(set(keys)))
        rounds = [k.round_index for k in self.match.kills]
        self.assertEqual(rounds, sorted(rounds))


class TestHenrikV2(unittest.TestCase):
    def setUp(self):
        self.match = parse_payload(henrik_v2_payload())

    def test_parses_flat_player_list(self):
        m = self.match
        self.assertEqual(m.match_id, "v2-match")
        self.assertEqual(m.map_name, "Bind")
        self.assertEqual(m.duration_ms, 1_800_000)
        self.assertEqual(len(m.players), 2)
        alpha = m.find_player("Alpha#1111")
        self.assertEqual(alpha.ref.team, "Red")
        self.assertEqual(alpha.stats.kills, 10)
        self.assertEqual(alpha.stats.headshots, 8)
        self.assertEqual(alpha.stats.damage_made, 2000)

    def test_kill_events_nested_in_rounds(self):
        m = self.match
        self.assertEqual(len(m.kills), 1)
        kill = m.kills[0]
        # The critical bit: killer and victim must not be confused, since a v2
        # event carries both on the same object.
        self.assertEqual(kill.killer.puuid, "p1")
        self.assertEqual(kill.victim.puuid, "p2")
        self.assertEqual(kill.killer.team, "Red")
        self.assertEqual(kill.victim.team, "Blue")
        self.assertEqual(kill.weapon.name, "Vandal")
        self.assertEqual(kill.victim_x, 100)
        self.assertEqual(len(kill.player_locations), 1)

    def test_round_state_and_plant(self):
        rnd = self.match.rounds[0]
        self.assertTrue(rnd.bomb_planted)
        self.assertEqual(rnd.plant_time_ms, 30000)
        self.assertEqual(rnd.planter_team, "Red")
        state = rnd.states["p1"]
        self.assertEqual(state.loadout_value, 3900)
        self.assertEqual(state.casts.total, 3)
        self.assertEqual(state.weapon.name, "Vandal")

    def test_teams_dict_shape(self):
        self.assertTrue(self.match.teams["Red"].won)
        self.assertEqual(self.match.teams["Red"].rounds_won, 13)


class TestRiotOfficial(unittest.TestCase):
    def setUp(self):
        self.match = parse_payload(riot_official_payload())

    def test_map_code_name_resolved(self):
        self.assertEqual(self.match.map_name, "Bind")
        self.assertEqual(riot_official.map_name_from_id("/Game/Maps/Jam/Jam"), "Lotus")
        self.assertEqual(riot_official.map_name_from_id("/Game/Maps/Weird/Weird"),
                         "Weird")

    def test_players_rounds_kills(self):
        m = self.match
        self.assertEqual(m.started_at, 1_700_000_000)
        self.assertEqual(len(m.players), 2)
        self.assertEqual(len(m.kills), 1)
        kill = m.kills[0]
        self.assertEqual(kill.killer.puuid, "r1")
        self.assertEqual(kill.victim.puuid, "r2")
        self.assertEqual(kill.time_in_round_ms, 25000)
        state = m.rounds[0].states["r1"]
        self.assertEqual(state.loadout_value, 4700)
        self.assertEqual(state.damage, 150)
        self.assertEqual(state.casts.total, 2)

    def test_side_inferred_from_planter(self):
        # Red planted, so Red attacked and Blue defended.
        self.assertEqual(self.match.rounds[0].attacking_team, "Red")
        self.assertEqual(self.match.rounds[0].side_for("Blue"), DEFENSE)

    def test_assets_translate_uuids(self):
        assets = {
            "agents": {"agent-uuid": "Jett"},
            "weapons": {"weapon-uuid": "Phantom"},
        }
        match = riot_official.parse_match(riot_official_payload(), assets)
        self.assertEqual(match.player("r1").ref.agent, "Jett")
        self.assertEqual(match.kills[0].weapon.name, "Phantom")


class TestDefensiveParsing(unittest.TestCase):
    def test_missing_sections_do_not_raise(self):
        payload = {"status": 200, "data": {"metadata": {"match_id": "bare"},
                                           "players": []}}
        match = parse_payload(payload)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_id, "bare")
        self.assertEqual(match.players, [])
        self.assertEqual(match.rounds, [])

    def test_nulls_in_kill_events(self):
        payload = henrik_v2_payload()
        rnd = payload["data"]["rounds"][0]
        rnd["player_stats"][0]["kill_events"][0]["victim_death_location"] = None
        rnd["player_stats"][0]["economy"] = None
        match = parse_payload(payload)
        self.assertIsNotNone(match)
        self.assertIsNone(match.kills[0].victim_x)
        self.assertEqual(match.rounds[0].states["p1"].loadout_value, 0)

    def test_timestamp_formats(self):
        self.assertEqual(henrik._epoch(1_700_000_000), 1_700_000_000)
        self.assertEqual(henrik._epoch(1_700_000_000_000), 1_700_000_000)
        self.assertEqual(henrik._epoch("2023-11-14T22:13:20Z"), 1_700_000_000)
        self.assertEqual(henrik._epoch(None), 0)
        self.assertEqual(henrik._epoch("nonsense"), 0)

    def test_map_and_team_name_shapes(self):
        self.assertEqual(henrik._name_of({"name": "Ascent"}), "Ascent")
        self.assertEqual(henrik._name_of("Ascent"), "Ascent")
        self.assertEqual(henrik._team_name("red"), "Red")
        self.assertEqual(henrik._team_name({"id": "Blue"}), "Blue")


if __name__ == "__main__":
    unittest.main()
