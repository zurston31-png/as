"""Death context is the foundation of every finding, so it gets tested hard."""

from __future__ import annotations

import unittest

from valcoach.analysis.context import (
    OPENING_WINDOW_MS,
    build_match_context,
    econ_state,
    time_bucket,
    weapon_class,
)
from valcoach.maps import MapIndex
from valcoach.models import ATTACK, DEFENSE

from .helpers import ME, add_kill, add_round, make_match

NO_MAPS = MapIndex({})


def context_for(match, trade_window_ms: int = 4000):
    return build_match_context(match, ME, trade_window_ms, NO_MAPS)


class TestClassifiers(unittest.TestCase):
    def test_weapon_classes(self):
        self.assertEqual(weapon_class("Vandal"), "rifle")
        self.assertEqual(weapon_class("Operator"), "sniper")
        self.assertEqual(weapon_class("Judge"), "shotgun")
        self.assertEqual(weapon_class("Classic"), "sidearm")
        self.assertEqual(weapon_class("", ""), "ability")
        self.assertEqual(weapon_class("Some New Gun"), "other")

    def test_econ_buckets(self):
        self.assertEqual(econ_state(4700), "full")
        self.assertEqual(econ_state(2500), "half")
        self.assertEqual(econ_state(800), "eco")

    def test_time_buckets(self):
        self.assertEqual(time_bucket(5_000), "opening")
        self.assertEqual(time_bucket(30_000), "mid")
        self.assertEqual(time_bucket(60_000), "late")


class TestTrades(unittest.TestCase):
    def test_death_is_traded_inside_window(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 10_000, "foe1", ME)
        add_kill(match, 0, 12_000, "mate1", "foe1")
        death = context_for(match).deaths[0]
        self.assertTrue(death.traded)
        self.assertEqual(death.trade_delay_ms, 2_000)

    def test_death_is_not_traded_outside_window(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 10_000, "foe1", ME)
        add_kill(match, 0, 20_000, "mate1", "foe1")
        death = context_for(match).deaths[0]
        self.assertFalse(death.traded)
        self.assertIsNone(death.trade_delay_ms)

    def test_killing_a_different_enemy_is_not_a_trade(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 10_000, "foe1", ME)
        add_kill(match, 0, 11_000, "mate1", "foe2")
        self.assertFalse(context_for(match).deaths[0].traded)

    def test_my_kill_counts_as_a_trade_kill(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 8_000, "foe1", "mate1")
        add_kill(match, 0, 9_500, ME, "foe1")
        kill = context_for(match).kills[0]
        self.assertTrue(kill.trade_kill)

    def test_unrelated_kill_is_not_a_trade_kill(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 8_000, "foe1", "mate1")
        add_kill(match, 0, 9_500, ME, "foe2")
        self.assertFalse(context_for(match).kills[0].trade_kill)

    def test_trade_window_is_configurable(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 10_000, "foe1", ME)
        add_kill(match, 0, 16_000, "mate1", "foe1")
        self.assertFalse(context_for(match, 4_000).deaths[0].traded)
        self.assertTrue(context_for(match, 8_000).deaths[0].traded)


class TestIsolation(unittest.TestCase):
    def test_nearest_living_teammate_measured(self):
        match = make_match()
        add_round(match, 0)
        add_kill(
            match, 0, 10_000, "foe1", ME, victim_pos=(0.0, 0.0),
            positions={"mate1": (600.0, 0.0), "mate2": (5_000.0, 0.0),
                       "foe1": (400.0, 0.0)},
        )
        death = context_for(match).deaths[0]
        self.assertAlmostEqual(death.nearest_teammate_distance, 600.0, places=1)
        self.assertEqual(death.nearest_teammate, "Mate1#1")
        self.assertFalse(death.isolated)
        self.assertAlmostEqual(death.distance_to_killer, 400.0, places=1)

    def test_far_from_everyone_is_isolated(self):
        match = make_match()
        add_round(match, 0)
        add_kill(
            match, 0, 10_000, "foe1", ME, victim_pos=(0.0, 0.0),
            positions={"mate1": (4_000.0, 0.0), "foe1": (300.0, 0.0)},
        )
        self.assertTrue(context_for(match).deaths[0].isolated)

    def test_dead_teammates_do_not_count_as_nearby(self):
        match = make_match()
        add_round(match, 0)
        # Mate1 dies first; only the far Mate2 is alive when I die.
        add_kill(match, 0, 5_000, "foe2", "mate1")
        add_kill(
            match, 0, 10_000, "foe1", ME, victim_pos=(0.0, 0.0),
            positions={"mate1": (100.0, 0.0), "mate2": (6_000.0, 0.0)},
        )
        death = context_for(match).deaths[0]
        self.assertAlmostEqual(death.nearest_teammate_distance, 6_000.0, places=1)
        self.assertTrue(death.isolated)

    def test_missing_positions_leave_distance_unknown(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 10_000, "foe1", ME)
        death = context_for(match).deaths[0]
        self.assertIsNone(death.nearest_teammate_distance)
        self.assertFalse(death.isolated)


class TestNumbersAndTiming(unittest.TestCase):
    def test_even_numbers_at_first_death(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 10_000, "foe1", ME)
        death = context_for(match).deaths[0]
        self.assertEqual(death.numbers, "even")
        self.assertEqual(death.teammates_alive, 4)
        self.assertEqual(death.enemies_alive, 5)
        self.assertTrue(death.first_death_of_round)

    def test_outnumbered_when_team_already_lost_players(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 5_000, "foe1", "mate1")
        add_kill(match, 0, 6_000, "foe1", "mate2")
        add_kill(match, 0, 20_000, "foe1", ME)
        death = context_for(match).deaths[-1]
        self.assertEqual(death.numbers, "down")
        self.assertEqual(death.teammates_alive, 2)
        self.assertEqual(death.enemies_alive, 5)
        self.assertFalse(death.first_death_of_round)

    def test_up_a_man(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 4_000, ME, "foe1")
        add_kill(match, 0, 5_000, "mate1", "foe2")
        add_kill(match, 0, 9_000, "foe3", ME)
        death = context_for(match).deaths[0]
        self.assertEqual(death.numbers, "up")

    def test_opening_and_late_deaths(self):
        match = make_match()
        add_round(match, 0)
        add_round(match, 1)
        add_kill(match, 0, OPENING_WINDOW_MS - 1_000, "foe1", ME)
        add_kill(match, 1, 70_000, "foe1", ME)
        deaths = context_for(match).deaths
        self.assertEqual(deaths[0].time_slot, "opening")
        self.assertEqual(deaths[1].time_slot, "late")

    def test_post_plant_death(self):
        match = make_match()
        add_round(match, 0, attacking_team="Red", plant_ms=30_000)
        add_kill(match, 0, 45_000, "foe1", ME)
        death = context_for(match).deaths[0]
        self.assertTrue(death.post_plant)
        self.assertEqual(death.side, DEFENSE)

    def test_pre_plant_death_is_not_post_plant(self):
        match = make_match()
        add_round(match, 0, attacking_team="Blue", plant_ms=30_000)
        add_kill(match, 0, 20_000, "foe1", ME)
        death = context_for(match).deaths[0]
        self.assertFalse(death.post_plant)
        self.assertEqual(death.side, ATTACK)


class TestEconomyAndUtility(unittest.TestCase):
    def test_loadout_and_econ_recorded(self):
        match = make_match()
        add_round(match, 0, loadouts={ME: 900})
        add_kill(match, 0, 10_000, "foe1", ME)
        death = context_for(match).deaths[0]
        self.assertEqual(death.loadout_value, 900)
        self.assertEqual(death.econ, "eco")
        self.assertEqual(death.weapon_held, "Vandal")

    def test_util_unused_flagged(self):
        match = make_match()
        add_round(match, 0, casts={ME: 0})
        add_kill(match, 0, 10_000, "foe1", ME)
        self.assertTrue(context_for(match).deaths[0].util_unused)

    def test_util_used_not_flagged(self):
        match = make_match()
        add_round(match, 0, casts={ME: 2})
        add_kill(match, 0, 10_000, "foe1", ME)
        death = context_for(match).deaths[0]
        self.assertFalse(death.util_unused)
        self.assertEqual(death.util_casts_in_round, 2)

    def test_saving_a_lost_round(self):
        match = make_match()
        add_round(match, 0, winner="Red", loadouts={ME: 4700})
        rnd = context_for(match).rounds[0]
        self.assertTrue(rnd.saved)
        self.assertEqual(rnd.lost_gear_value, 0)

    def test_dying_with_gear_in_a_lost_round(self):
        match = make_match()
        add_round(match, 0, winner="Red", loadouts={ME: 4700})
        add_kill(match, 0, 10_000, "foe1", ME)
        rnd = context_for(match).rounds[0]
        self.assertFalse(rnd.saved)
        self.assertEqual(rnd.lost_gear_value, 4700)

    def test_eco_round_death_is_not_counted_as_lost_gear(self):
        match = make_match()
        add_round(match, 0, winner="Red", loadouts={ME: 800})
        add_kill(match, 0, 10_000, "foe1", ME)
        rnd = context_for(match).rounds[0]
        self.assertEqual(rnd.lost_gear_value, 0)
        self.assertFalse(rnd.saved)


class TestRoundOutcomes(unittest.TestCase):
    def test_clutch_attempt_and_win(self):
        match = make_match()
        add_round(match, 0, winner="Blue")
        for i, mate in enumerate(("mate1", "mate2", "mate3", "mate4"), start=1):
            add_kill(match, 0, 5_000 * i, "foe1", mate)
        add_kill(match, 0, 40_000, ME, "foe1")
        rnd = context_for(match).rounds[0]
        self.assertTrue(rnd.clutch_attempt)
        self.assertTrue(rnd.clutch_won)
        self.assertEqual(rnd.clutch_enemies, 5)

    def test_clutch_lost(self):
        match = make_match()
        add_round(match, 0, winner="Red")
        for i, mate in enumerate(("mate1", "mate2", "mate3", "mate4"), start=1):
            add_kill(match, 0, 5_000 * i, "foe1", mate)
        add_kill(match, 0, 40_000, "foe1", ME)
        rnd = context_for(match).rounds[0]
        self.assertTrue(rnd.clutch_attempt)
        self.assertFalse(rnd.clutch_won)

    def test_no_clutch_when_teammates_alive(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 5_000, "foe1", "mate1")
        self.assertFalse(context_for(match).rounds[0].clutch_attempt)

    def test_kast_inputs(self):
        match = make_match()
        add_round(match, 0)
        add_round(match, 1)
        add_kill(match, 0, 5_000, ME, "foe1")          # kill
        add_kill(match, 1, 5_000, "foe1", ME)          # death, untraded
        ctx = context_for(match)
        self.assertTrue(ctx.rounds[0].got_kill)
        self.assertTrue(ctx.rounds[0].survived)
        self.assertFalse(ctx.rounds[1].survived)
        self.assertFalse(ctx.rounds[1].was_traded)

    def test_assist_recorded(self):
        match = make_match()
        add_round(match, 0)
        add_kill(match, 0, 5_000, "mate1", "foe1", assistants=[ME])
        self.assertTrue(context_for(match).rounds[0].got_assist)

    def test_unknown_player_returns_no_context(self):
        match = make_match()
        add_round(match, 0)
        self.assertIsNone(build_match_context(match, "nobody", 4000, NO_MAPS))

    def test_describe_mentions_the_essentials(self):
        match = make_match()
        add_round(match, 0, attacking_team="Blue", casts={ME: 0})
        add_kill(match, 0, 9_000, "foe1", ME, weapon="Operator",
                 positions={"mate1": (9_000.0, 0.0), "foe1": (5_000.0, 0.0)})
        text = context_for(match).deaths[0].describe()
        self.assertIn("R1", text)
        self.assertIn("attack", text)
        self.assertIn("Foe1", text)
        self.assertIn("Operator", text)
        self.assertIn("not traded", text)
        self.assertIn("no util used", text)


if __name__ == "__main__":
    unittest.main()
