"""Detectors must fire on real habits, stay quiet otherwise, and never crash."""

from __future__ import annotations

import unittest

from valcoach.analysis.context import build_contexts
from valcoach.analysis.detectors import (
    BENCHMARKS,
    DetectorInput,
    Finding,
    detect_first_deaths,
    detect_headshots,
    detect_isolated_deaths,
    detect_nemesis,
    detect_repeated_spots,
    detect_untraded_deaths,
    detect_utility_usage,
    run_detectors,
)
from valcoach.analysis.metrics import Metrics, compute_metrics
from valcoach.maps import MapIndex

from .helpers import ME, add_kill, add_round, make_match

NO_MAPS = MapIndex({})


def build_input(match, strict: bool = False) -> DetectorInput:
    contexts = build_contexts([match], ME, 4000, NO_MAPS)
    metrics = compute_metrics(contexts, riot_id="You#0000", puuid=ME)
    return DetectorInput(
        metrics=metrics,
        contexts=contexts,
        deaths=[d for c in contexts for d in c.deaths],
        rounds=[r for c in contexts for r in c.rounds],
        map_index=NO_MAPS,
        strict=strict,
    )


def ids(findings) -> set:
    return {f.id for f in findings}


class TestFirstDeaths(unittest.TestCase):
    def test_fires_when_first_death_rate_is_high(self):
        match = make_match()
        for i in range(24):
            add_round(match, i, attacking_team="Blue")
            if i % 3 == 0:                      # first death in a third of rounds
                add_kill(match, i, 8_000, "foe1", ME)
            else:
                add_kill(match, i, 8_000, "mate1", "foe1")
        findings = detect_first_deaths(build_input(match))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].id, "first_deaths")
        self.assertGreater(findings[0].value, BENCHMARKS["first_death_rate"])
        self.assertTrue(findings[0].evidence)
        self.assertIn("attack", findings[0].why)

    def test_quiet_when_first_deaths_are_rare(self):
        match = make_match()
        for i in range(24):
            add_round(match, i)
            if i == 0:
                add_kill(match, i, 8_000, "foe1", ME)
            else:
                add_kill(match, i, 8_000, "mate1", "foe1")
        self.assertEqual(detect_first_deaths(build_input(match)), [])

    def test_quiet_on_a_tiny_sample(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 5_000, "foe1", ME)
        self.assertEqual(detect_first_deaths(build_input(match)), [])


class TestTradesAndIsolation(unittest.TestCase):
    def _match_with_untraded_deaths(self, traded: int, untraded: int):
        match = make_match()
        index = 0
        for _ in range(traded):
            add_round(match, index)
            add_kill(match, index, 10_000, "foe1", ME)
            add_kill(match, index, 11_000, "mate1", "foe1")
            index += 1
        for _ in range(untraded):
            add_round(match, index)
            add_kill(match, index, 10_000, "foe1", ME)
            index += 1
        return match

    def test_untraded_rate_fires(self):
        findings = detect_untraded_deaths(
            build_input(self._match_with_untraded_deaths(2, 14))
        )
        self.assertEqual(ids(findings), {"untraded_deaths"})
        self.assertGreater(findings[0].value, BENCHMARKS["untraded_death_rate"])

    def test_untraded_rate_quiet_when_team_trades(self):
        findings = detect_untraded_deaths(
            build_input(self._match_with_untraded_deaths(12, 4))
        )
        self.assertEqual(findings, [])

    def test_isolation_fires_with_distant_teammates(self):
        match = make_match()
        for i in range(16):
            add_round(match, i)
            add_kill(
                match, i, 12_000, "foe1", ME, victim_pos=(0.0, 0.0),
                positions={"mate1": (5_000.0, 0.0), "foe1": (500.0, 0.0)},
            )
        findings = detect_isolated_deaths(build_input(match))
        self.assertEqual(ids(findings), {"isolated_deaths"})
        self.assertEqual(findings[0].value, 100.0)

    def test_isolation_quiet_when_playing_together(self):
        match = make_match()
        for i in range(16):
            add_round(match, i)
            add_kill(
                match, i, 12_000, "foe1", ME, victim_pos=(0.0, 0.0),
                positions={"mate1": (500.0, 0.0), "foe1": (400.0, 0.0)},
            )
        self.assertEqual(detect_isolated_deaths(build_input(match)), [])


class TestRepeatedSpots(unittest.TestCase):
    def test_cluster_of_deaths_is_reported(self):
        match = make_match()
        for i in range(10):
            add_round(match, i)
            # Five deaths in one small area, the rest spread out.
            pos = (100.0 + i * 50, 100.0) if i < 5 else (9_000.0 * i, 7_000.0)
            add_kill(match, i, 12_000, "foe1", ME, victim_pos=pos)
        findings = detect_repeated_spots(build_input(match))
        self.assertTrue(findings)
        top = findings[0]
        self.assertIn("Ascent", top.title)
        self.assertGreaterEqual(top.value, 5)
        self.assertTrue(top.evidence)

    def test_scattered_deaths_are_not_a_pattern(self):
        match = make_match()
        for i in range(10):
            add_round(match, i)
            add_kill(match, i, 12_000, "foe1", ME,
                     victim_pos=(i * 6_000.0, i * 5_000.0))
        self.assertEqual(detect_repeated_spots(build_input(match)), [])


class TestNemesis(unittest.TestCase):
    def test_one_opponent_dominating(self):
        match = make_match()
        for i in range(14):
            add_round(match, i)
            killer = "foe1" if i < 6 else f"foe{(i % 4) + 2}"
            add_kill(match, i, 12_000, killer, ME)
        findings = detect_nemesis(build_input(match))
        self.assertEqual(ids(findings), {"nemesis"})
        self.assertIn("Foe1", findings[0].title)
        self.assertEqual(findings[0].value, 6)

    def test_spread_out_killers(self):
        match = make_match()
        for i in range(14):
            add_round(match, i)
            add_kill(match, i, 12_000, f"foe{(i % 5) + 1}", ME)
        self.assertEqual(detect_nemesis(build_input(match)), [])


class TestUtility(unittest.TestCase):
    def test_low_utility_fires(self):
        match = make_match()
        for i in range(24):
            add_round(match, i, casts={ME: 0})
            add_kill(match, i, 12_000, "foe1", ME)
        findings = detect_utility_usage(build_input(match))
        self.assertIn("low_utility", ids(findings))
        self.assertIn("died_with_utility", ids(findings))

    def test_healthy_utility_is_quiet(self):
        match = make_match()
        for i in range(24):
            add_round(match, i, casts={ME: 3})
            add_kill(match, i, 12_000, "foe1", ME)
        self.assertEqual(detect_utility_usage(build_input(match)), [])

    def test_support_roles_are_held_to_a_higher_bar(self):
        duelist = make_match(agent="Jett")
        support = make_match(agent="Sova")
        for match in (duelist, support):
            for i in range(24):
                add_round(match, i, casts={ME: 2})
                add_kill(match, i, 12_000, "mate1", "foe1")
        self.assertEqual(detect_utility_usage(build_input(duelist)), [])
        self.assertIn("low_utility", ids(detect_utility_usage(build_input(support))))


class TestAimAndStrict(unittest.TestCase):
    def _metrics(self, hs_pct_shots=(15, 200)) -> DetectorInput:
        heads, shots = hs_pct_shots
        metrics = Metrics(riot_id="x", puuid=ME, rounds=100, deaths=40, kills=40)
        metrics.headshots = heads
        metrics.bodyshots = shots - heads
        return DetectorInput(metrics=metrics, contexts=[], map_index=NO_MAPS)

    def test_low_headshots_fires(self):
        findings = detect_headshots(self._metrics((20, 200)))   # 10%
        self.assertEqual(ids(findings), {"low_headshots"})
        self.assertEqual(findings[0].severity, "critical")

    def test_good_headshots_quiet(self):
        self.assertEqual(detect_headshots(self._metrics((60, 200))), [])   # 30%

    def test_strict_mode_tightens_benchmarks(self):
        data = self._metrics((40, 200))                          # 20%, normally fine
        self.assertEqual(detect_headshots(data), [])
        data.strict = True
        self.assertEqual(ids(detect_headshots(data)), {"low_headshots"})

    def test_small_sample_is_ignored(self):
        data = self._metrics((2, 40))
        self.assertEqual(detect_headshots(data), [])


class TestRunDetectors(unittest.TestCase):
    def test_findings_are_sorted_by_severity(self):
        match = make_match()
        for i in range(24):
            add_round(match, i, casts={ME: 0}, attacking_team="Blue")
            add_kill(match, i, 8_000, "foe1", ME, victim_pos=(100.0, 100.0),
                     positions={"mate1": (6_000.0, 0.0), "foe1": (400.0, 0.0)})
        findings = run_detectors(build_input(match))
        self.assertTrue(findings)
        order = ["critical", "high", "medium", "low", "strength"]
        positions = [order.index(f.severity) for f in findings]
        self.assertEqual(positions, sorted(positions))

    def test_a_broken_detector_does_not_break_the_report(self):
        import valcoach.analysis.detectors as module

        def exploding(_data):
            raise ValueError("boom")

        original = module.DETECTORS
        module.DETECTORS = (exploding,) + tuple(original)
        try:
            match = make_match()
            add_round(match, 0)
            add_kill(match, 0, 5_000, "foe1", ME)
            findings = run_detectors(build_input(match))
        finally:
            module.DETECTORS = original
        errors = [f for f in findings if f.id.startswith("detector_error")]
        self.assertEqual(len(errors), 1)
        self.assertIn("boom", errors[0].summary)

    def test_finding_serialises_with_delta(self):
        finding = Finding(
            id="x", title="t", category="aim", severity="high", summary="s",
            value=10.0, benchmark=18.0,
        )
        data = finding.to_dict()
        self.assertEqual(data["delta"], -8.0)
        self.assertEqual(data["severity"], "high")

    def test_clean_player_gets_no_problems(self):
        """A player who trades, uses utility and survives should pass cleanly."""
        match = make_match()
        for i in range(24):
            add_round(match, i, casts={ME: 3}, winner="Blue")
            foe = f"foe{(i % 5) + 1}"
            if i % 3 == 0:
                # A teammate dies and I avenge them: that is a trade kill.
                add_kill(match, i, 18_000, foe, "mate2",
                         victim_pos=(float(i) * 2_000, 400.0))
                add_kill(match, i, 19_500, ME, foe,
                         victim_pos=(float(i) * 2_000, 500.0))
            else:
                add_kill(match, i, 20_000, ME, foe,
                         victim_pos=(float(i) * 2_000, 500.0))
            if i % 4 == 0:                       # occasional traded death
                add_kill(match, i, 30_000, "foe5", ME,
                         victim_pos=(float(i) * 2_000, 900.0),
                         positions={"mate1": (float(i) * 2_000 + 300, 900.0),
                                    "foe5": (float(i) * 2_000 + 600, 900.0)})
                add_kill(match, i, 31_000, "mate1", "foe5")
        data = build_input(match)
        problems = [f for f in run_detectors(data) if f.severity != "strength"]
        self.assertEqual(
            problems, [], f"unexpected findings: {[f.id for f in problems]}"
        )


if __name__ == "__main__":
    unittest.main()


class TestLocationClaims(unittest.TestCase):
    """Never assert "mostly around X" unless X actually repeats in the data."""

    def _sniper_match(self, same_place: bool):
        match = make_match()
        for i in range(14):
            add_round(match, i)
            # Same spot every time, or scattered.
            pos = (100.0, 100.0) if same_place else (i * 9_000.0, i * 7_000.0)
            add_kill(match, i, 12_000, "foe1", ME, weapon="Operator",
                     victim_pos=pos)
        return match

    def test_scattered_sniper_deaths_name_no_place(self):
        from valcoach.analysis.detectors import detect_weapon_matchups

        findings = detect_weapon_matchups(build_input(self._sniper_match(False)))
        sniper = next(f for f in findings if f.id == "sniper_deaths")
        self.assertNotIn("mostly around", sniper.summary)

    def test_repeated_sniper_deaths_do_name_the_place(self):
        from valcoach.analysis.detectors import detect_weapon_matchups
        from valcoach.maps import MapIndex

        data = build_input(self._sniper_match(True))
        # With callouts loaded the repeated position resolves to a name.
        data.map_index = MapIndex({"maps": {"Ascent": {"callouts": [
            {"region": "Main", "super_region": "A",
             "location": {"x": 100, "y": 100}}]}}})
        contexts = build_contexts(
            [self._sniper_match(True)], ME, 4000, data.map_index
        )
        data.deaths = [d for c in contexts for d in c.deaths]
        sniper = next(
            f for f in detect_weapon_matchups(data) if f.id == "sniper_deaths"
        )
        self.assertIn("mostly around A Main", sniper.summary)

    def test_nemesis_place_claim_needs_a_repeat(self):
        from valcoach.analysis.detectors import detect_nemesis

        match = make_match()
        for i in range(14):
            add_round(match, i)
            add_kill(match, i, 12_000, "foe1", ME,
                     victim_pos=(i * 9_000.0, i * 7_000.0))
        finding = detect_nemesis(build_input(match))[0]
        self.assertNotIn("keep getting you around", finding.fix)

    def test_repeated_spot_reports_how_tight_the_cluster_is(self):
        from valcoach.analysis.detectors import detect_repeated_spots

        match = make_match()
        for i in range(10):
            add_round(match, i)
            add_kill(match, i, 12_000, "foe1", ME,
                     victim_pos=(100.0 + i * 60, 100.0))
        finding = detect_repeated_spots(build_input(match))[0]
        self.assertIn("circle at", finding.summary)
        self.assertRegex(finding.summary, r"inside a \d+m circle")
