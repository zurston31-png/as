"""Callout lookup, death clustering, and configuration precedence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from valcoach.config import Config
from valcoach.maps import CLUSTER_RADIUS, MapIndex, agent_role, cluster_points

ASSETS = {
    "maps": {
        "Ascent": {
            "callouts": [
                {"region": "Main", "super_region": "A",
                 "location": {"x": 1000, "y": 1000}},
                {"region": "Site", "super_region": "B",
                 "location": {"x": -4000, "y": -4000}},
            ]
        }
    },
    "agents": {"uuid-1": "Jett"},
    "weapons": {"uuid-2": "Vandal"},
}


class TestCallouts(unittest.TestCase):
    def setUp(self):
        self.index = MapIndex(ASSETS)

    def test_nearest_callout_named(self):
        self.assertTrue(self.index.has_callouts)
        self.assertEqual(self.index.callout("Ascent", 1100, 900), "A Main")
        self.assertEqual(self.index.callout("Ascent", -3900, -4100), "B Site")

    def test_map_name_is_case_insensitive(self):
        self.assertEqual(self.index.callout("ascent", 1000, 1000), "A Main")

    def test_far_from_every_callout_is_unnamed(self):
        self.assertEqual(self.index.callout("Ascent", 50_000, 50_000), "")

    def test_unknown_map_has_no_callouts(self):
        self.assertEqual(self.index.callout("Haven", 0, 0), "")

    def test_describe_falls_back_to_coordinates(self):
        self.assertEqual(MapIndex({}).describe("Ascent", 12.6, -8.2), "(12, -8)")
        self.assertEqual(MapIndex({}).describe("Ascent", None, None),
                         "unknown position")
        self.assertEqual(self.index.describe("Ascent", 1000, 1000), "A Main")

    def test_malformed_callouts_are_skipped(self):
        index = MapIndex({"maps": {"Ascent": {"callouts": [
            {"region": "Broken", "location": {"x": None, "y": 1}},
            {"region": "Good", "super_region": "A", "location": {"x": 5, "y": 5}},
        ]}}})
        self.assertEqual(index.callout("Ascent", 6, 6), "A Good")


class TestClustering(unittest.TestCase):
    def test_nearby_points_group_together(self):
        points = [(0, 0), (200, 100), (-150, 50), (9_000, 9_000)]
        clusters = cluster_points(points)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].size, 3)
        self.assertEqual(clusters[1].size, 1)
        self.assertLess(clusters[0].radius, CLUSTER_RADIUS)

    def test_clusters_are_ordered_by_size(self):
        points = [(0, 0), (9_000, 9_000), (9_100, 9_050), (9_050, 8_950)]
        clusters = cluster_points(points)
        self.assertEqual([c.size for c in clusters], [3, 1])

    def test_missing_coordinates_are_ignored(self):
        clusters = cluster_points([(None, None), (0, 0), (10, 10)])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 2)

    def test_members_point_back_at_input_indices(self):
        points = [(0, 0), (8_000, 0), (100, 100)]
        clusters = cluster_points(points)
        self.assertEqual(clusters[0].members, [0, 2])
        self.assertEqual(clusters[1].members, [1])

    def test_empty_input(self):
        self.assertEqual(cluster_points([]), [])


class TestAgentRoles(unittest.TestCase):
    def test_known_agents(self):
        self.assertEqual(agent_role("Jett"), "duelist")
        self.assertEqual(agent_role("sova"), "initiator")
        self.assertEqual(agent_role("Viper"), "controller")
        self.assertEqual(agent_role("KAY/O"), "initiator")
        self.assertEqual(agent_role("Killjoy"), "sentinel")

    def test_unknown_agent_is_not_an_error(self):
        self.assertEqual(agent_role("Some Future Agent"), "unknown")
        self.assertEqual(agent_role(""), "unknown")


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._env = dict(os.environ)
        os.environ["VALCOACH_HOME"] = self.tmp.name
        for key in ("VALCOACH_RIOT_ID", "VALCOACH_REGION", "HENRIK_API_KEY",
                    "VALCOACH_PROVIDER", "VALCOACH_QUEUE"):
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self.tmp.cleanup()

    def test_defaults(self):
        config = Config.load()
        self.assertEqual(config.provider, "henrik")
        self.assertEqual(config.region, "na")
        self.assertEqual(config.model, "claude-opus-5")
        self.assertEqual(config.trade_window_ms, 4000)
        self.assertTrue(config.db_path.endswith(".db"))

    def test_file_then_env_then_flags(self):
        config = Config.load()
        config.riot_id = "FromFile#1111"
        config.region = "eu"
        config.save()

        self.assertEqual(Config.load().riot_id, "FromFile#1111")

        os.environ["VALCOACH_RIOT_ID"] = "FromEnv#2222"
        self.assertEqual(Config.load().riot_id, "FromEnv#2222")
        self.assertEqual(Config.load().region, "eu")      # still from the file

        flagged = Config.load(overrides={"riot_id": "FromFlag#3333"})
        self.assertEqual(flagged.riot_id, "FromFlag#3333")

    def test_blank_overrides_are_ignored(self):
        config = Config.load()
        config.riot_id = "Keep#1111"
        config.save()
        self.assertEqual(Config.load(overrides={"riot_id": ""}).riot_id, "Keep#1111")

    def test_types_are_coerced(self):
        config = Config.load(overrides={"trade_window_ms": "5000",
                                        "file_paths": "a.json,b.json"})
        self.assertEqual(config.trade_window_ms, 5000)
        self.assertEqual(config.file_paths, ["a.json", "b.json"])

    def test_shard_defaults_to_region(self):
        self.assertEqual(Config.load(overrides={"region": "eu"}).shard, "eu")

    def test_api_keys_are_redacted_for_display(self):
        config = Config.load(overrides={"henrik_api_key": "secret-value"})
        shown = config.redacted()
        self.assertNotIn("secret-value", json.dumps(shown))
        self.assertIn("set (", shown["henrik_api_key"])

    def test_environment_key_is_not_written_to_the_config_file(self):
        os.environ["ANTHROPIC_API_KEY"] = "env-secret"
        config = Config.load()
        path = config.save()
        with open(path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["anthropic_api_key"], "")

    def test_assets_load_when_present(self):
        with open(os.path.join(self.tmp.name, "assets.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(ASSETS, handle)
        assets = Config.load().load_assets()
        self.assertEqual(assets["agents"]["uuid-1"], "Jett")

    def test_missing_or_broken_assets_are_empty(self):
        self.assertEqual(Config.load().load_assets(), {})
        with open(os.path.join(self.tmp.name, "assets.json"), "w",
                  encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertEqual(Config.load().load_assets(), {})

    def test_broken_config_file_falls_back_to_defaults(self):
        with open(os.path.join(self.tmp.name, "config.json"), "w",
                  encoding="utf-8") as handle:
            handle.write("}}broken")
        self.assertEqual(Config.load().provider, "henrik")


if __name__ == "__main__":
    unittest.main()
