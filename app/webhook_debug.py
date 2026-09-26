"""Diagnostics for the inbound TradingView webhook.

FastAPI rejects a body that fails schema validation *before* the route
handler runs, so a 422 leaves nothing in the log to explain it: TradingView
reports a failed alert and the terminal stays silent. These helpers make
the rejection legible - which field, what arrived, what was expected -
without ever printing the shared secret.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.schemas import TradingViewAlert

REDACTED = "<redacted>"

# Matches the secret's value in a raw body, including escaped quotes inside
# it, so the raw text can be logged verbatim minus the one thing that must
# never be logged.
_SECRET_IN_TEXT = re.compile(r'("secret"\s*:\s*)"(?:[^"\\]|\\.)*"')


def secret_presence(payload: Any) -> str:
    """Whether a secret arrived at all, and how long it was - never its value.

    The two failure modes look identical from the outside (TradingView sends
    nothing vs. sends the wrong thing) and are fixed differently, so the
    length is worth knowing and the value is not worth the risk.
    """
    if not isinstance(payload, dict) or "secret" not in payload:
        return "missing"
    value = payload["secret"]
    if not isinstance(value, str):
        return f"present but {type(value).__name__}, expected string"
    if not value:
        return "present but empty"
    return f"present, {len(value)} chars"


def redact_text(raw: str) -> str:
    return _SECRET_IN_TEXT.sub(rf'\1"{REDACTED}"', raw)


def redact(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if "secret" in out:
        out["secret"] = REDACTED
    return out


def format_errors(exc: ValidationError) -> list[str]:
    """Pydantic's errors as flat strings, with any secret value stripped.

    A "field required" error carries the *whole* body as its input, not the
    absent field, so redacting only errors located at `secret` would still
    print the secret whenever some other field was missing.
    """
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<body>"
        received = error.get("input")
        if "secret" in error["loc"]:
            shown = REDACTED
        else:
            shown = repr(redact(received))
        problems.append(f"{location}: {error['msg']} (received {shown})")
    return problems


def describe_field_mismatches(payload: Any) -> list[str]:
    """Names that differ between the payload and the schema.

    Complements the pydantic errors, which describe wrong *types* but say
    nothing about a field TradingView sent under a different name.
    """
    if not isinstance(payload, dict):
        return [f"body is {type(payload).__name__}, expected a JSON object"]

    fields = TradingViewAlert.model_fields
    problems = [
        f"missing required field '{name}'"
        for name, field in fields.items()
        if field.is_required() and name not in payload
    ]
    problems += [f"unexpected field '{key}' (ignored)" for key in payload if key not in fields]
    return problems


def expected_shape() -> dict[str, str]:
    """The schema as a name -> "type (required|optional)" map, for logging."""
    shape = {}
    for name, field in TradingViewAlert.model_fields.items():
        annotation = getattr(field.annotation, "__name__", str(field.annotation))
        shape[name] = f"{annotation} ({'required' if field.is_required() else 'optional'})"
    return shape


def parse_body(raw_text: str) -> tuple[Any, str | None]:
    """Parse a raw body, returning (payload, error).

    Uses the stdlib decoder rather than a strict one because Pine's
    `str.tostring()` emits a bare `NaN` for an `na` series value, which
    stdlib json accepts and a strict JSON parser would reject - rejecting
    the alert over an indicator that has not warmed up yet. The resulting
    non-finite floats are turned into absent values by
    `TradingViewAlert.non_finite_is_absent`, so none reaches the database;
    the current Pine script sends `null` outright.
    """
    if not raw_text.strip():
        return None, "body was empty"
    try:
        return json.loads(raw_text), None
    except ValueError as exc:
        return None, f"body is not valid JSON: {exc}"
