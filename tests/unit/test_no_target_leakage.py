"""Architectural guards.

The brief says: "Never optimize for a desired win rate" and "Never fabricate
historical performance". Those are process rules, and a comment cannot
enforce a process rule. These tests do.

Guard 1: the brief's target win rates exist in exactly one place
(`config.schema.ResearchTargets`) and may only be *read* by reporting code
(`analytics`, `validation`) -- never by anything that generates signals,
sizes positions, or selects parameters.

Guard 2: the target numbers themselves (0.88, 0.92, 0.72) do not appear as
literals in model code, where they could become a thresholded objective.

Guard 3: no module hard-codes a performance figure to report.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "flow_model"

# Modules allowed to read ResearchTargets: they report on it, they do not fit to it.
TARGET_READERS_ALLOWED = {"analytics", "validation", "config", "dashboard"}

# Modules that must never consult a target, because they decide what to trade.
DECISION_MODULES = {"features", "signals", "regime", "risk", "backtest", "monte_carlo", "data"}

#: Every straightforward encoding of the brief's target win rates. The first
#: version of this guard checked floats only, so the integer percentage form
#: (`88`), the `88 / 100` expression and the string `"0.88"` all slipped
#: through -- three ways to write the same number the test was meant to ban.
TARGET_LITERALS = {0.88, 0.92, 0.72}
TARGET_PERCENTAGES = {88, 92, 72}
TARGET_TOLERANCE = 1e-9


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _top_package(path: Path) -> str:
    rel = path.relative_to(PACKAGE)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def test_research_targets_defined_in_exactly_one_place():
    definitions = [
        p for p in _python_files()
        if re.search(r"^class ResearchTargets", p.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert len(definitions) == 1, f"ResearchTargets defined in multiple places: {definitions}"
    assert definitions[0].name == "schema.py"


#: Tokens a decision module must never mention. `ResearchTargets` holds the
#: brief's desired win rates; `hypotheses` holds the pre-registered
#: refutation criteria. Both are scoring yardsticks, and the reason is the
#: same for each: a rule that can see the number it will be judged against
#: will, given enough iterations, reproduce that number and demonstrate
#: nothing. Pre-registration only has force while the thing being tested
#: cannot read the test.
FORBIDDEN_IN_DECISION_CODE = (
    "research_targets",
    "ResearchTargets",
    "hypotheses",
    "HYPOTHESES",
    "Hypothesis",
)


def test_decision_modules_never_reference_research_targets():
    """A module that decides what to trade must not know the desired win rate,
    nor the criteria it will be scored against."""
    offenders: list[str] = []
    for path in _python_files():
        if _top_package(path) not in DECISION_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if any(token in code for token in FORBIDDEN_IN_DECISION_CODE):
                offenders.append(f"{path.relative_to(PACKAGE)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "signal/risk/backtest code must not read the brief's target win rates "
        "or the pre-registered falsification criteria:\n"
        + "\n".join(offenders)
    )


def _is_target(value: object) -> bool:
    """True for any numeric encoding of a target win rate."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value in TARGET_PERCENTAGES
    if isinstance(value, float):
        if any(abs(value - t) < TARGET_TOLERANCE for t in TARGET_LITERALS):
            return True
        return any(abs(value - p) < TARGET_TOLERANCE for p in TARGET_PERCENTAGES)
    if isinstance(value, str):
        try:
            return _is_target(float(value))
        except ValueError:
            return False
    return False


def _folded_value(node: ast.AST) -> object:
    """Constant-fold the `n / 100` shape, so `88 / 100` is caught too."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult)):
        left, right = _folded_value(node.left), _folded_value(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Div):
                return left / right if right else None
            return left * right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _folded_value(node.operand)
        return -inner if isinstance(inner, (int, float)) else None
    return None


def test_target_literals_do_not_appear_in_decision_modules():
    """Guards against the target leaking in as a bare number.

    Covers the fraction (0.88), the percentage (88 and 88.0), the `88 / 100`
    expression and the string form. Verified against the current tree: no
    innocent constant in a decision module collides with 72, 88 or 92.
    """
    offenders: list[str] = []
    for path in _python_files():
        if _top_package(path) not in DECISION_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            value = _folded_value(node)
            if value is not None and _is_target(value):
                offenders.append(
                    f"{path.relative_to(PACKAGE)}:{getattr(node, 'lineno', '?')}: "
                    f"target value {value!r}"
                )
    assert not offenders, (
        "the brief's target win rates appear as literals in decision code:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source,should_trip",
    [
        ("x = 0.88", True),
        ("x = 88", True),
        ("x = 88.0", True),
        ("x = 88 / 100", True),
        ('x = "0.88"', True),
        ("x = 0.90", False),
        ("x = 100", False),
        ("x = 0.72", True),
    ],
)
def test_the_literal_guard_actually_trips(source, should_trip):
    """A guard that cannot fail is worth nothing, so its detector is tested
    directly against each encoding."""
    tripped = any(
        (lambda v: v is not None and _is_target(v))(_folded_value(node))
        for node in ast.walk(ast.parse(source))
    )
    assert tripped is should_trip


def test_research_targets_are_excluded_from_the_config_hash(config):
    """A target cannot change the model's identity, because it is not an input."""
    altered = config.replace(
        research_targets=config.research_targets.replace(combined_10y_win_rate=0.99)
    )
    assert altered.config_hash == config.config_hash


def test_research_targets_document_themselves_as_hypotheses(config):
    note = config.research_targets.note.lower()
    assert "hypothes" in note or "not objectives" in note
    assert "report measured values" in note


def test_no_module_claims_profitability():
    """Guard 3: no fabricated performance claims in the source."""
    banned = re.compile(
        r"(?:is|are)\s+profitable|proven\s+edge|guaranteed\s+(?:profit|return)"
        r"|has\s+an?\s+edge",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    for path in _python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = banned.search(line)
            if match and "no claim" not in line.lower() and "never" not in line.lower():
                offenders.append(f"{path.relative_to(PACKAGE)}:{lineno}: {line.strip()}")
    assert not offenders, "unsupported performance claims found:\n" + "\n".join(offenders)


def test_package_docstring_disclaims_an_edge():
    import flow_model

    assert "no claim" in (flow_model.__doc__ or "").lower()


@pytest.mark.parametrize("module_dir", sorted(DECISION_MODULES))
def test_decision_module_directories_exist(module_dir):
    """The architecture's module boundaries are real directories, so the
    guards above actually cover the code that will be written there."""
    assert (PACKAGE / module_dir).is_dir()
