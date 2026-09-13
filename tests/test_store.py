"""Storage must be idempotent, re-parseable, and queryable by player."""

from __future__ import annotations

import os
import tempfile
import unittest

from valcoach.store import Store

from .helpers import ME, add_kill, add_round, henrik_v2_payload, make_match


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "test.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()


class TestSaveAndLoad(StoreTestCase):
    def _sample(self):
        match = make_match(match_id="abc")
        add_round(match, 0)
        add_kill(match, 0, 9_000, "foe1", ME, positions={"mate1": (100.0, 0.0)})
        return match

    def test_save_reports_new_then_existing(self):
        match = self._sample()
        self.assertTrue(self.store.save_match(match, {"raw": 1}, "test"))
        self.assertFalse(self.store.save_match(match, {"raw": 1}, "test"))
        self.assertEqual(self.store.counts()["matches"], 1)

    def test_derived_tables_populated(self):
        self.store.save_match(self._sample(), {"raw": 1}, "test")
        counts = self.store.counts()
        self.assertEqual(counts["rounds"], 1)
        self.assertEqual(counts["kills"], 1)
        self.assertEqual(counts["players"], 10)

    def test_resaving_does_not_duplicate_children(self):
        match = self._sample()
        self.store.save_match(match, {"raw": 1}, "test")
        self.store.save_match(match, {"raw": 1}, "test")
        self.assertEqual(self.store.counts()["kills"], 1)
        self.assertEqual(self.store.counts()["rounds"], 1)

    def test_raw_payload_round_trip(self):
        self.store.save_match(self._sample(), {"hello": "world"}, "test")
        payload, fmt = self.store.raw_payload("abc")
        self.assertEqual(payload, {"hello": "world"})
        self.assertEqual(fmt, "test")

    def test_load_matches_reparses_payload(self):
        payload = henrik_v2_payload()
        from valcoach.providers import parse_payload

        match = parse_payload(payload)
        self.store.save_match(match, payload, "henrik")
        loaded = self.store.load_matches(puuid="p1")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].match_id, "v2-match")
        self.assertEqual(len(loaded[0].kills), 1)

    def test_load_skips_unparseable_payloads(self):
        match = self._sample()
        self.store.save_match(match, {"not": "a match"}, "mystery")
        self.assertEqual(self.store.load_matches(), [])

    def test_reindex_rebuilds_derived_rows(self):
        payload = henrik_v2_payload()
        from valcoach.providers import parse_payload

        self.store.save_match(parse_payload(payload), payload, "henrik")
        self.store.conn.execute("DELETE FROM kills")
        self.store.conn.commit()
        self.assertEqual(self.store.counts()["kills"], 0)
        self.assertEqual(self.store.reindex(), 1)
        self.assertEqual(self.store.counts()["kills"], 1)


class TestQueries(StoreTestCase):
    def setUp(self):
        super().setUp()
        for i, (map_name, queue, started) in enumerate(
            [("Ascent", "competitive", 1_700_000_000),
             ("Bind", "competitive", 1_700_100_000),
             ("Split", "unrated", 1_700_200_000)]
        ):
            match = make_match(match_id=f"m{i}", map_name=map_name, queue=queue)
            match.started_at = started
            add_round(match, 0)
            self.store.save_match(match, {"n": i}, "test")

    def test_filter_by_queue_and_map(self):
        self.assertEqual(len(self.store.match_rows(queue="competitive")), 2)
        self.assertEqual(len(self.store.match_rows(map_name="split")), 1)

    def test_newest_first_and_limit(self):
        rows = self.store.match_rows(limit=2)
        self.assertEqual([r["match_id"] for r in rows], ["m2", "m1"])

    def test_filter_by_player(self):
        self.assertEqual(len(self.store.match_rows(puuid=ME)), 3)
        self.assertEqual(len(self.store.match_rows(puuid="nobody")), 0)

    def test_since_filter(self):
        self.assertEqual(len(self.store.match_rows(since=1_700_150_000)), 1)

    def test_resolve_puuid_by_riot_id(self):
        self.assertEqual(self.store.resolve_puuid("You#0000"), ME)
        self.assertEqual(self.store.resolve_puuid("you#0000"), ME)
        self.assertEqual(self.store.resolve_puuid("You"), ME)
        self.assertEqual(self.store.resolve_puuid(ME), ME)
        self.assertIsNone(self.store.resolve_puuid("Someone#9999"))
        self.assertIsNone(self.store.resolve_puuid(""))

    def test_known_match_ids(self):
        self.assertEqual(self.store.known_match_ids(), {"m0", "m1", "m2"})


class TestReportMemory(StoreTestCase):
    def test_reports_round_trip_newest_first(self):
        first = self.store.save_report(
            ME, "You#0000", ["m1"], {"kd": 1.0},
            [{"id": "a", "title": "A", "severity": "high"}], "narrative one",
        )
        second = self.store.save_report(
            ME, "You#0000", ["m2"], {"kd": 1.2},
            [{"id": "b", "title": "B", "severity": "medium"}], "narrative two",
        )
        self.assertNotEqual(first, second)
        reports = self.store.recent_reports(ME)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["metrics"]["kd"], 1.2)
        self.assertEqual(reports[0]["findings"][0]["id"], "b")
        self.assertEqual(reports[0]["narrative"], "narrative two")
        self.assertEqual(reports[0]["match_ids"], ["m2"])

    def test_reports_are_scoped_to_a_player(self):
        self.store.save_report(ME, "You#0000", [], {}, [])
        self.assertEqual(len(self.store.recent_reports("someone-else")), 0)

    def test_meta_values(self):
        self.store.set_meta("last_sync", "123")
        self.store.conn.commit()
        self.assertEqual(self.store.get_meta("last_sync"), "123")
        self.store.set_meta("last_sync", "456")
        self.assertEqual(self.store.get_meta("last_sync"), "456")
        self.assertEqual(self.store.get_meta("missing", "fallback"), "fallback")


class TestMemoryStore(unittest.TestCase):
    def test_in_memory_database_works(self):
        store = Store(":memory:")
        match = make_match()
        add_round(match, 0)
        self.assertTrue(store.save_match(match, {"x": 1}, "test"))
        self.assertEqual(store.counts()["matches"], 1)
        store.close()


if __name__ == "__main__":
    unittest.main()
