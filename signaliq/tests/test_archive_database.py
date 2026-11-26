"""Archive round trip: export, verify, restore (SEC answers compressed), same fingerprints."""

import gzip
import json
import subprocess
import sys
from pathlib import Path

from app.database import create_session_factory, session_scope
from app.universe import database as udb
from app.universe import sec, sec_cache
from app.universe.models import ExternalResponse

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / args[0]), *map(str, args[1:])],
        capture_output=True,
        text=True,
    )


def _fingerprints(db, *extra):
    out = run("table_fingerprints.py", db, *extra)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _source(tmp_path, monkeypatch):
    monkeypatch.delenv(sec_cache.PARENT_VARIABLE, raising=False)
    path = tmp_path / "source.db"
    engine = udb.open_research_database(f"sqlite:///{path.as_posix()}")
    with session_scope(create_session_factory(engine)) as s:
        s.add(ExternalResponse(source="sec", request_key="big", response="<p>10-K</p>" * 20_000))
        s.add(ExternalResponse(source="sec", request_key="small", response="tiny"))
        s.add(ExternalResponse(source="openfigi", request_key="AAPL", response=[{"x": 1.5}]))
    engine.dispose()
    return path


def test_export_verify_restore_gives_back_the_same_data(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    archive = tmp_path / "archive"
    assert run("archive_database.py", "export", source, "--out", archive).returncode == 0
    verified = run("archive_database.py", "verify", archive, "--against", source)
    assert verified.returncode == 0, verified.stdout
    assert "RESULT: PASS" in verified.stdout

    restored = tmp_path / "restored.db"
    out = run("archive_database.py", "restore", archive, "--out", restored, "--compress-sec")
    assert out.returncode == 0, out.stdout + out.stderr
    # The big answer is stored compressed, and still reads back as the original text.
    engine = udb.open_research_database(f"sqlite:///{restored.as_posix()}", read_only=False)
    with session_scope(create_session_factory(engine)) as s:
        raw = s.get(ExternalResponse, ("sec", "big")).response
        assert isinstance(raw, dict)
        assert sec.CachedSec(s, network=False).get("big") == "<p>10-K</p>" * 20_000
    engine.dispose()
    assert _fingerprints(restored, "--logical") == _fingerprints(source)
    assert _fingerprints(restored) != _fingerprints(source)  # stored differently, same data


def test_a_damaged_archive_file_fails_verification(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    archive = tmp_path / "archive"
    run("archive_database.py", "export", source, "--out", archive)
    manifest = json.loads((archive / "manifest.json").read_text())
    target = next(e for e in manifest["files"] if e["table"] == "external_responses")
    path = archive / target["file"]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(lines[:-1])  # one row lost
    result = run("archive_database.py", "verify", archive)
    assert result.returncode == 1
    assert "FAIL sha256" in result.stdout
    assert "RESULT: FAIL" in result.stdout


def test_restore_never_overwrites(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    archive = tmp_path / "archive"
    run("archive_database.py", "export", source, "--out", archive)
    existing = tmp_path / "existing.db"
    existing.write_bytes(b"keep me")
    assert run("archive_database.py", "restore", archive, "--out", existing).returncode == 1
    assert existing.read_bytes() == b"keep me"
