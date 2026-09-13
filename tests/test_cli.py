"""End-to-end CLI behaviour, entirely offline."""

from __future__ import annotations

import io
import json
import os
import contextlib
import tempfile
import unittest

from valcoach.cli import main
from valcoach.render.html import render_html
from valcoach.render.text import render_report
from valcoach.analysis.report import build_report
from valcoach.providers import iter_payloads, parse_payload

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "valcoach", "fixtures", "demo_matches.json",
)


def run(*argv: str):
    """Run the CLI, returning (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self.tmp.name, "home")
        self.db = os.path.join(self.tmp.name, "valcoach.db")
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("VALCOACH_HOME", "VALCOACH_DB", "NO_COLOR",
                        "ANTHROPIC_API_KEY", "VALCOACH_RIOT_ID")
        }
        os.environ["VALCOACH_HOME"] = self.home
        os.environ["NO_COLOR"] = "1"
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("VALCOACH_RIOT_ID", None)

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def seed(self):
        code, _, err = run("import", FIXTURE, "--db-path", self.db)
        self.assertEqual(code, 0, err)


class TestDemo(CliTestCase):
    def test_demo_runs_with_no_config_or_network(self):
        code, out, err = run("demo", "--db", self.db, "--no-color")
        self.assertEqual(code, 0, err)
        self.assertIn("synthetic matches", out)
        self.assertIn("THE NUMBERS", out)
        self.assertIn("WHAT'S GOING WRONG", out)
        self.assertIn("HOW YOU DIED", out)
        self.assertIn("FOCUS FOR YOUR NEXT SESSION", out)

    def test_demo_json_is_valid_and_complete(self):
        code, out, _ = run("demo", "--db", self.db, "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out[out.index("{"):])
        self.assertEqual(payload["riot_id"], "You#0000")
        self.assertGreater(len(payload["findings"]), 3)
        self.assertGreater(len(payload["deaths"]), 10)
        self.assertGreater(payload["metrics"]["rounds"], 50)
        for finding in payload["findings"]:
            self.assertIn(finding["severity"],
                          {"critical", "high", "medium", "low", "strength"})
            self.assertTrue(finding["title"])
            self.assertTrue(finding["summary"])

    def test_demo_writes_html(self):
        path = os.path.join(self.tmp.name, "out", "review.html")
        code, out, err = run("demo", "--db", self.db, "--html", path, "--no-color")
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as handle:
            html = handle.read()
        self.assertIn("<title>", html)
        self.assertIn("what to fix", html)
        self.assertNotIn("<!doctype", html.lower())
        self.assertNotIn("<html", html.lower())
        self.assertIn("prefers-color-scheme", html)

    def test_strict_mode_finds_more(self):
        _, normal, _ = run("demo", "--db", self.db, "--json")
        _, strict, _ = run("demo", "--db", self.db + "2", "--json", "--strict")
        normal_count = len(json.loads(normal[normal.index("{"):])["findings"])
        strict_count = len(json.loads(strict[strict.index("{"):])["findings"])
        self.assertGreaterEqual(strict_count, normal_count)


class TestImportAndAnalyze(CliTestCase):
    def test_import_then_analyze(self):
        self.seed()
        code, out, err = run(
            "analyze", "--db-path", self.db, "--player", "You#0000", "--no-color",
        )
        self.assertEqual(code, 0, err)
        self.assertIn("VALORANT review", out)
        self.assertIn("You#0000", out)

    def test_import_is_idempotent(self):
        self.seed()
        code, out, _ = run("import", FIXTURE, "--db-path", self.db)
        self.assertEqual(code, 0)
        self.assertIn("(0 new)", out)

    def test_analyze_without_data_explains_itself(self):
        code, _, err = run(
            "analyze", "--db-path", self.db, "--player", "Nobody#0000",
        )
        self.assertEqual(code, 1)
        self.assertIn("sync", err)

    def test_analyze_saves_a_review_and_history_shows_it(self):
        self.seed()
        run("analyze", "--db-path", self.db, "--player", "You#0000", "--no-color")
        code, out, err = run(
            "history", "--db-path", self.db, "--player", "You#0000"
        )
        self.assertEqual(code, 0, err)
        self.assertIn("review #1", out)
        self.assertIn("K/D", out)

    def test_second_review_reports_progress(self):
        self.seed()
        run("analyze", "--db-path", self.db, "--player", "You#0000", "--last", "5")
        code, out, _ = run(
            "analyze", "--db-path", self.db, "--player", "You#0000", "--last", "3",
            "--no-color",
        )
        self.assertEqual(code, 0)
        self.assertIn("Since your last review", out)

    def test_no_save_skips_the_review_record(self):
        self.seed()
        run("analyze", "--db-path", self.db, "--player", "You#0000", "--no-save")
        code, out, _ = run("history", "--db-path", self.db, "--player", "You#0000")
        self.assertIn("no saved reviews", out)


class TestDeaths(CliTestCase):
    def test_death_log_and_filters(self):
        self.seed()
        code, out, err = run(
            "deaths", "--db-path", self.db, "--player", "You#0000", "--last", "5",
            "--no-color",
        )
        self.assertEqual(code, 0, err)
        self.assertIn("deaths ·", out)
        self.assertIn("killed most by:", out)

        _, untraded, _ = run(
            "deaths", "--db-path", self.db, "--player", "You#0000", "--last", "5",
            "--untraded", "--json",
        )
        rows = json.loads(untraded)
        self.assertTrue(rows)
        self.assertTrue(all(row["traded"] is False for row in rows))

        _, first, _ = run(
            "deaths", "--db-path", self.db, "--player", "You#0000", "--last", "5",
            "--first", "--json",
        )
        self.assertTrue(
            all(row["first_death_of_round"] for row in json.loads(first))
        )

    def test_side_filter(self):
        self.seed()
        _, out, _ = run(
            "deaths", "--db-path", self.db, "--player", "You#0000", "--last", "5",
            "--side", "attack", "--json",
        )
        rows = json.loads(out)
        self.assertTrue(rows)
        self.assertTrue(all(row["side"] == "attack" for row in rows))

    def test_map_filter_on_unknown_map_is_empty(self):
        self.seed()
        _, out, _ = run(
            "deaths", "--db-path", self.db, "--player", "You#0000",
            "--map", "Nowhere", "--json",
        )
        self.assertEqual(json.loads(out), [])


class TestOtherCommands(CliTestCase):
    def test_status_reports_configuration(self):
        self.seed()
        code, out, err = run("status", "--db-path", self.db)
        self.assertEqual(code, 0, err)
        self.assertIn("valcoach", out)
        self.assertIn("stored:", out)
        self.assertIn("5 matches", out)
        self.assertIn("coaching:", out)

    def test_init_writes_config(self):
        code, out, err = run(
            "init", "--riot-id", "You#0000", "--region", "eu", "--no-resolve",
            "--db-path", self.db,
        )
        self.assertEqual(code, 0, err)
        self.assertIn("You#0000", out)
        config_file = os.path.join(self.home, "config.json")
        self.assertTrue(os.path.exists(config_file))
        with open(config_file, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["riot_id"], "You#0000")
        self.assertEqual(saved["region"], "eu")

    def test_init_requires_a_riot_id(self):
        code, _, err = run("init", "--no-resolve")
        self.assertEqual(code, 2)
        self.assertIn("Riot ID is required", err)

    def test_matches_lists_stored_games(self):
        self.seed()
        code, out, err = run("matches", "--db-path", self.db, "--limit", "3")
        self.assertEqual(code, 0, err)
        self.assertIn("match id", out)
        self.assertEqual(out.strip().count("\n"), 3)

    def test_matches_with_empty_database(self):
        code, out, _ = run("matches", "--db-path", self.db)
        self.assertEqual(code, 0)
        self.assertIn("no matches stored", out)

    def test_reindex(self):
        self.seed()
        code, out, err = run("reindex", "--db-path", self.db)
        self.assertEqual(code, 0, err)
        self.assertIn("re-parsed 5", out)

    def test_watch_stops_after_the_requested_cycles(self):
        self.seed()
        code, out, err = run(
            "watch", "--db-path", self.db, "--cycles", "1", "--interval", "30",
            "--provider", "file", "--files", FIXTURE,
        )
        self.assertEqual(code, 0, err)
        self.assertIn("watching for new matches", out)

    def test_unknown_command_exits_with_usage(self):
        with self.assertRaises(SystemExit):
            run("nonsense")


class TestRenderers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            payloads = json.load(handle)
        matches = [
            parse_payload(single)
            for payload in payloads for single in iter_payloads(payload)
        ]
        me = matches[0].find_player("You#0000")
        cls.report = build_report(matches, me.ref.puuid, me.ref.riot_id)

    def test_text_report_is_plain_without_color(self):
        text = render_report(self.report, color=False)
        self.assertNotIn("\033[", text)
        self.assertIn("THE NUMBERS", text)

    def test_text_report_can_hide_details(self):
        brief = render_report(self.report, color=False, detail=False)
        full = render_report(self.report, color=False, detail=True)
        self.assertLess(len(brief), len(full))
        self.assertNotIn("Why it costs rounds", brief)

    def test_empty_report_is_handled(self):
        empty = build_report([], "nobody", "Nobody#0000")
        text = render_report(empty, color=False)
        self.assertIn("No matches", text)
        html = render_html(empty)
        self.assertIn("<title>", html)

    def test_html_escapes_player_names(self):
        report = self.report
        report.findings[0].title = "<script>alert(1)</script>"
        html = render_html(report)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_html_narrative_markdown(self):
        report = self.report
        report.narrative = "## Verdict\nYou peek too early.\n\n- One\n- Two\n"
        html = render_html(report)
        self.assertIn("<h3>Verdict</h3>", html)
        self.assertIn("<li>One</li>", html)
        report.narrative = ""

    def test_html_has_one_scatter_panel_per_map(self):
        html = render_html(self.report)
        maps = {d.map_name for d in self.report.deaths}
        for name in maps:
            self.assertIn(f">{name} ", html)
        self.assertGreater(html.count("<circle"), 10)


if __name__ == "__main__":
    unittest.main()


class TestFindingCopy(unittest.TestCase):
    """A strength is not a mistake, so it must not be labelled like one."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, "r", encoding="utf-8") as handle:
            payloads = json.load(handle)
        matches = [
            parse_payload(single)
            for payload in payloads for single in iter_payloads(payload)
        ]
        me = matches[0].find_player("You#0000")
        cls.report = build_report(matches, me.ref.puuid, me.ref.riot_id)

    def test_strengths_use_a_positive_why_label(self):
        self.assertTrue(self.report.strengths, "fixture should surface a strength")
        from valcoach.render.html import _finding_card

        card = _finding_card(self.report.strengths[0])
        self.assertIn("Why it matters", card)
        self.assertNotIn("Why it costs rounds", card)

    def test_problems_keep_the_cost_framing(self):
        from valcoach.render.html import _finding_card

        card = _finding_card(self.report.problems[0])
        self.assertIn("Why it costs rounds", card)

    def test_text_renderer_matches(self):
        from valcoach.render.text import Painter, render_findings

        paint = Painter(False)
        self.assertIn("Why it matters",
                      render_findings(self.report.strengths, paint))
        self.assertIn("Why it costs rounds",
                      render_findings(self.report.problems[:1], paint))

    def test_demo_data_now_records_assists(self):
        self.assertGreater(self.report.metrics.assists, 0)
