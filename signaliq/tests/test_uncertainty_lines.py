import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from universe_validation_batch import UNCERTAIN, uncertainty_lines


def _rows():
    with UNCERTAIN.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_labels_shown_when_they_match_the_run():
    rows = _rows()
    stored = {"2021-11-30": 605191825.84, "2021-12-31": 628197190.6, "2022-04-29": 687334895.5}
    lines = uncertainty_lines(stored, rows)
    assert lines[0].endswith(": 3")
    assert sum("UNCERTAIN." in line for line in lines) == len(rows) == 4
    assert any("FDX kept; DASH enters" in line for line in lines)


def test_stale_label_is_not_shown():
    lines = uncertainty_lines({"2021-11-30": 600_000_000.0}, _rows())
    assert "LABEL DOES NOT MATCH THIS RUN" in lines[1]
    assert not any("UNCERTAIN." in line for line in lines)


def test_missing_label_file_is_reported():
    assert "missing" in uncertainty_lines({}, None)[0]
