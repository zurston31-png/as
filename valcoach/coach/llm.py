"""The coaching voice: structured findings in, plain-language review out.

Uses the official Anthropic SDK. The SDK is an optional dependency — without it
(or without credentials) ``valcoach`` still produces the full structured report,
it just cannot write the narrative.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .prompts import SYSTEM, build_payload, build_user_message

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16_000            # a review is ~1-2 pages; this is ample headroom


class CoachUnavailable(RuntimeError):
    """Raised when the SDK or credentials are missing."""


@dataclass
class CoachResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: str = ""


def _failure_message(exc: BaseException) -> str:
    """Turn a non-API failure into something actionable.

    The SDK raises a bare TypeError at request time when it cannot resolve any
    credential — which reads as a crash rather than "you have no API key".
    """
    text = str(exc)
    if "authentication method" in text or "api_key" in text:
        return (
            "no usable Anthropic credentials — set ANTHROPIC_API_KEY or run "
            "`ant auth login`"
        )
    return f"the coaching request failed ({type(exc).__name__}): {exc}"


@dataclass
class Coach:
    """Wraps one Anthropic client and keeps the conversation for follow-ups.

    Keeping the history means a follow-up question re-sends the same match data
    prefix, which the prompt cache can serve instead of reprocessing it.
    """

    model: str = DEFAULT_MODEL
    api_key: str = ""
    effort: str = "high"
    stream: bool = True
    _client: Any = field(default=None, repr=False)
    _history: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    # ---- setup ---------------------------------------------------------
    @staticmethod
    def available() -> bool:
        """Is the optional SDK installed? (Does not check credentials.)"""
        return importlib.util.find_spec("anthropic") is not None

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise CoachUnavailable(
                "the anthropic package is not installed — run "
                "`pip install anthropic` to enable written coaching"
            ) from exc
        try:
            # With no api_key the SDK resolves ANTHROPIC_API_KEY, then
            # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile.
            self._client = (
                anthropic.Anthropic(api_key=self.api_key)
                if self.api_key
                else anthropic.Anthropic()
            )
        except Exception as exc:  # noqa: BLE001 - surfaces as a clean CLI message
            raise CoachUnavailable(f"could not create an Anthropic client: {exc}") from exc
        return self._client

    # ---- requests ------------------------------------------------------
    def review(
        self,
        report: Any,
        question: Optional[str] = None,
        focus: Optional[str] = None,
        previous: Optional[List[Dict[str, Any]]] = None,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> CoachResult:
        payload = build_payload(report, previous=previous)
        message = build_user_message(payload, question=question, focus=focus)
        return self._send(message, on_text=on_text)

    def follow_up(
        self, question: str, on_text: Optional[Callable[[str], None]] = None
    ) -> CoachResult:
        if not self._history:
            raise CoachUnavailable("ask for a review before a follow-up question")
        return self._send(question, on_text=on_text)

    def _send(
        self, text: str, on_text: Optional[Callable[[str], None]] = None
    ) -> CoachResult:
        import anthropic

        client = self.client()
        self._history.append({"role": "user", "content": text})
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            # A copy: the reply is appended to _history below, and the SDK must
            # not see a list that changes under it.
            "messages": list(self._history),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            # Auto-caches the last cacheable block, so follow-up questions reuse
            # the match-data prefix instead of paying for it again.
            "cache_control": {"type": "ephemeral"},
        }
        try:
            if self.stream:
                with client.messages.stream(**kwargs) as stream:
                    for chunk in stream.text_stream:
                        if on_text:
                            on_text(chunk)
                    message = stream.get_final_message()
            else:
                message = client.messages.create(**kwargs)
        except anthropic.NotFoundError as exc:
            self._history.pop()
            raise CoachUnavailable(
                f"model {self.model!r} is not available to this account: {exc}"
            ) from exc
        except anthropic.AuthenticationError as exc:
            self._history.pop()
            raise CoachUnavailable(
                "no usable Anthropic credentials — set ANTHROPIC_API_KEY or run "
                f"`ant auth login` ({exc})"
            ) from exc
        except anthropic.RateLimitError as exc:
            self._history.pop()
            raise CoachUnavailable(f"rate limited by the API, try again shortly: {exc}") from exc
        except anthropic.APIStatusError as exc:
            self._history.pop()
            raise CoachUnavailable(f"API error ({exc.status_code}): {exc}") from exc
        except anthropic.APIConnectionError as exc:
            self._history.pop()
            raise CoachUnavailable(f"could not reach the API: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - see _credential_hint
            self._history.pop()
            raise CoachUnavailable(_failure_message(exc)) from exc

        if getattr(message, "stop_reason", "") == "refusal":
            self._history.pop()
            raise CoachUnavailable(
                "the model declined to answer this request "
                f"({getattr(message, 'stop_details', None)})"
            )

        answer = "".join(
            block.text for block in message.content
            if getattr(block, "type", "") == "text"
        ).strip()
        self._history.append({"role": "assistant", "content": message.content})
        usage = getattr(message, "usage", None)
        return CoachResult(
            text=answer,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=getattr(message, "stop_reason", "") or "",
        )
