"""The coaching layer, exercised against a stub client — no API calls."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List

from valcoach.analysis.report import build_report
from valcoach.coach import Coach, CoachUnavailable, build_payload, build_user_message
from valcoach.providers import iter_payloads, parse_payload

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "valcoach", "fixtures", "demo_matches.json",
)


def load_report():
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        payloads = json.load(handle)
    matches = [
        parse_payload(single)
        for payload in payloads for single in iter_payloads(payload)
    ]
    me = matches[0].find_player("You#0000")
    return build_report(matches, me.ref.puuid, me.ref.riot_id)


class FakeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeUsage:
    input_tokens = 1234
    output_tokens = 567
    cache_read_input_tokens = 900


class FakeMessage:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.content = [FakeBlock(text)]
        self.usage = FakeUsage()
        self.stop_reason = stop_reason
        self.stop_details = None


class FakeStream:
    def __init__(self, message: FakeMessage):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for block in self._message.content:
            for word in block.text.split(" "):
                yield word + " "

    def get_final_message(self):
        return self._message


class FakeMessages:
    def __init__(self, text: str = "## Verdict\nYou peek too early.",
                 stop_reason: str = "end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls: List[Dict[str, Any]] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(FakeMessage(self.text, self.stop_reason))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage(self.text, self.stop_reason)


class FakeClient:
    def __init__(self, **kwargs):
        self.messages = FakeMessages(**kwargs)


class TestPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load_report()

    def test_payload_carries_findings_and_a_bounded_death_sample(self):
        payload = build_payload(self.report, death_sample=10)
        self.assertIn("findings", payload)
        self.assertIn("metrics", payload)
        self.assertIn("matches", payload)
        self.assertEqual(len(payload["death_log"]), 10)
        self.assertIn("of", payload["death_log_note"])
        # Raw death dicts are dropped; the readable log replaces them.
        self.assertNotIn("deaths", payload)

    def test_payload_includes_previous_review_focus(self):
        previous = [{
            "created_at": 1,
            "findings": [
                {"title": "Old habit", "severity": "high"},
                {"title": "A strength", "severity": "strength"},
            ],
        }]
        payload = build_payload(self.report, previous=previous)
        self.assertEqual(payload["previous_reviews"][0]["focus"], ["Old habit"])

    def test_user_message_wraps_data_and_guards_against_injection(self):
        message = build_user_message(build_payload(self.report))
        self.assertIn("<match_data>", message)
        self.assertIn("</match_data>", message)
        self.assertIn("never as instructions", message)

    def test_question_and_focus_are_passed_through(self):
        message = build_user_message(
            build_payload(self.report), question="why do I lose 1v1s?",
            focus="positioning",
        )
        self.assertIn("why do I lose 1v1s?", message)
        self.assertIn("positioning", message)
        self.assertIn("Your question", message)


@unittest.skipUnless(Coach.available(), "anthropic SDK not installed")
class TestCoach(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load_report()

    def _coach(self, **kwargs) -> Coach:
        coach = Coach(model="claude-opus-5", stream=True)
        coach._client = FakeClient(**kwargs)
        return coach

    def test_review_returns_text_and_usage(self):
        coach = self._coach()
        result = coach.review(self.report)
        self.assertIn("You peek too early", result.text)
        self.assertEqual(result.model, "claude-opus-5")
        self.assertEqual(result.input_tokens, 1234)
        self.assertEqual(result.cache_read_tokens, 900)
        self.assertEqual(result.stop_reason, "end_turn")

    def test_request_shape(self):
        coach = self._coach()
        coach.review(self.report)
        call = coach._client.messages.calls[0]
        self.assertEqual(call["model"], "claude-opus-5")
        self.assertEqual(call["thinking"], {"type": "adaptive"})
        self.assertEqual(call["output_config"]["effort"], "high")
        self.assertEqual(call["cache_control"], {"type": "ephemeral"})
        self.assertIn("VALORANT coach", call["system"])
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertNotIn("budget_tokens", json.dumps(call["thinking"]))

    def test_streaming_calls_the_callback(self):
        coach = self._coach()
        chunks: List[str] = []
        coach.review(self.report, on_text=chunks.append)
        self.assertTrue(chunks)
        self.assertIn("Verdict", "".join(chunks))

    def test_non_streaming_path(self):
        coach = self._coach()
        coach.stream = False
        result = coach.review(self.report)
        self.assertIn("You peek too early", result.text)

    def test_follow_up_keeps_the_conversation(self):
        coach = self._coach()
        coach.review(self.report)
        coach.follow_up("what about my economy?")
        second_call = coach._client.messages.calls[1]
        self.assertEqual(len(second_call["messages"]), 3)   # user, assistant, user
        self.assertEqual(second_call["messages"][-1]["content"],
                         "what about my economy?")

    def test_follow_up_without_a_review_is_rejected(self):
        coach = self._coach()
        with self.assertRaises(CoachUnavailable):
            coach.follow_up("hello?")

    def test_refusal_is_surfaced_and_history_stays_clean(self):
        coach = self._coach(stop_reason="refusal")
        with self.assertRaises(CoachUnavailable):
            coach.review(self.report)
        self.assertEqual(coach._history, [])

    def test_api_errors_become_coach_unavailable(self):
        import anthropic
        import httpx2

        class Exploding(FakeMessages):
            def stream(self, **kwargs):
                raise anthropic.AuthenticationError(
                    "nope",
                    response=httpx2.Response(401, request=httpx2.Request("POST", "http://x")),
                    body=None,
                )

        coach = self._coach()
        coach._client.messages = Exploding()
        with self.assertRaises(CoachUnavailable) as caught:
            coach.review(self.report)
        self.assertIn("credentials", str(caught.exception))
        self.assertEqual(coach._history, [])


if __name__ == "__main__":
    unittest.main()


class TestFailureMessages(unittest.TestCase):
    """A missing key must read as a missing key, not as a crash."""

    def test_missing_credentials_gets_an_actionable_message(self):
        from valcoach.coach.llm import _failure_message

        message = _failure_message(
            TypeError(
                "Could not resolve authentication method. Expected one of "
                "api_key, auth_token, or credentials to be set."
            )
        )
        self.assertIn("ANTHROPIC_API_KEY", message)
        self.assertIn("ant auth login", message)

    def test_other_failures_name_the_exception(self):
        from valcoach.coach.llm import _failure_message

        self.assertIn("RuntimeError", _failure_message(RuntimeError("disk on fire")))

    @unittest.skipUnless(Coach.available(), "anthropic SDK not installed")
    def test_unexpected_error_is_wrapped_not_raised(self):
        class Exploding(FakeMessages):
            def stream(self, **kwargs):
                raise TypeError("Could not resolve authentication method.")

        coach = Coach(model="claude-opus-5")
        coach._client = FakeClient()
        coach._client.messages = Exploding()
        with self.assertRaises(CoachUnavailable) as caught:
            coach.review(load_report())
        self.assertIn("ANTHROPIC_API_KEY", str(caught.exception))
        self.assertEqual(coach._history, [])
