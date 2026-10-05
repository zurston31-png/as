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

TARGET_LITERALS = {0.88, 0.92, 0.72, 88.0, 92.0, 72.0}


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


def test_decision_modules_never_reference_research_targets():
    """A module that decides what to trade must not know the desired win rate."""
    offenders: list[str] = []
    for path in _python_files():
        if _top_package(path) not in DECISION_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "research_targets" in code or "ResearchTargets" in code:
                offenders.append(f"{path.relative_to(PACKAGE)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "signal/risk/backtest code must not read the brief's target win rates:\n"
        + "\n".join(offenders)
    )


def test_target_literals_do_not_appear_in_decision_modules():
    """Guards against the target leaking in as a bare number."""
    offenders: list[str] = []
    for path in _python_files():
        if _top_package(path) not in DECISION_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if node.value in TARGET_LITERALS:
                    offenders.append(
                        f"{path.relative_to(PACKAGE)}:{node.lineno}: literal {node.value}"
                    )
    assert not offenders, (
        "the brief's target win rates appear as literals in decision code:\n"
        + "\n".join(offenders)
    )


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
