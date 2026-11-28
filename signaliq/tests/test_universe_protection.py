"""Protected databases, the read-only parent cache, offline mode, compression, disk guard."""

import hashlib
import sqlite3

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

from app.database import create_session_factory, session_scope
from app.universe import database as udb
from app.universe import evidence, sec, sec_cache
from app.universe.models import ExternalResponse


def _url(path):
    return f"sqlite:///{path.as_posix()}"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _no_parent(monkeypatch):
    monkeypatch.delenv(sec_cache.PARENT_VARIABLE, raising=False)
    monkeypatch.delenv(sec_cache.OFFLINE_VARIABLE, raising=False)
    monkeypatch.delenv(udb.READ_ONLY_VARIABLE, raising=False)
    yield
    sec_cache.configure_parent(None)


def _make_db(path, rows):
    engine = udb.open_research_database(_url(path))
    with session_scope(create_session_factory(engine)) as s:
        for key, value in rows.items():
            s.add(ExternalResponse(source="sec", request_key=key, response=value))
    engine.dispose()


@pytest.fixture
def protected(tmp_path, monkeypatch):
    """A research database registered as protected, like the Stage 1.1 database."""
    path = tmp_path / "archived.db"
    _make_db(path, {"https://www.sec.gov/a": "archived answer"})
    monkeypatch.setitem(udb.PROTECTED_DATABASES, path.resolve(), "a test archive")
    return path


def test_a_protected_database_cannot_be_opened_for_writing(protected):
    before = _sha(protected)
    with pytest.raises(udb.ProtectedDatabase):
        udb.open_research_database(_url(protected))
    assert _sha(protected) == before


def test_the_stage_1_1_databases_are_registered_as_protected():
    names = {p.name for p in udb.PROTECTED_DATABASES}
    assert {"universe_stage.db", "universe_stage_v1_0_tables.db", "signaliq.db"} <= names
    assert {"sip_research.db", "alpaca_responses.db"} <= names


def test_there_is_no_default_research_database(monkeypatch):
    monkeypatch.delenv(udb.URL_VARIABLE, raising=False)
    with pytest.raises(udb.NoResearchDatabase):
        udb.open_research_database()


def test_market_data_writers_refuse_a_protected_database(protected, settings):
    from app.data.ingestion import run_ingestion
    from app.database import create_db_engine, init_db

    before = _sha(protected)
    with pytest.raises(udb.ProtectedDatabase):
        run_ingestion(settings.model_copy(update={"database_url": _url(protected)}), ["TESTA"])
    engine = create_db_engine(_url(protected))  # a migration would write: refused
    with pytest.raises(udb.ProtectedDatabase):
        init_db(engine)
    engine.dispose()
    assert _sha(protected) == before


def test_read_only_mode_reads_and_sqlite_refuses_every_write(protected):
    before = _sha(protected)
    engine = udb.open_research_database(_url(protected), read_only=True)
    with session_scope(create_session_factory(engine)) as s:
        assert (
            s.get(ExternalResponse, ("sec", "https://www.sec.gov/a")).response == "archived answer"
        )
    with (
        pytest.raises(OperationalError, match="readonly"),
        session_scope(create_session_factory(engine)) as s,
    ):
        s.add(ExternalResponse(source="sec", request_key="new", response="x"))
    with pytest.raises(OperationalError, match="readonly"), engine.begin() as connection:
        connection.execute(text("delete from external_responses"))
    engine.dispose()
    assert _sha(protected) == before


def test_read_only_mode_through_the_environment(protected, monkeypatch):
    monkeypatch.setenv(udb.READ_ONLY_VARIABLE, "1")
    engine = udb.open_research_database(_url(protected))
    with pytest.raises(OperationalError, match="readonly"), engine.begin() as connection:
        connection.execute(text("delete from external_responses"))
    engine.dispose()


def test_the_parent_connection_itself_refuses_writes(protected):
    connection = sec_cache.open_parent(protected)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("delete from external_responses")
    connection.close()


def test_a_new_database_reads_the_parent_and_writes_only_to_itself(
    protected, tmp_path, monkeypatch
):
    before = _sha(protected)
    monkeypatch.setenv(sec_cache.PARENT_VARIABLE, str(protected))
    child = tmp_path / "stage2.db"
    engine = udb.open_research_database(_url(child))
    with session_scope(create_session_factory(engine)) as s:
        client = sec.CachedSec(s, network=False)
        assert client.get("https://www.sec.gov/a") == "archived answer"
        assert client.get("https://www.sec.gov/b") is None
        sec_cache.store(s, "https://www.sec.gov/b", "new answer")
        sec_cache.store(s, "https://www.sec.gov/a", "overriding answer")
    with session_scope(create_session_factory(engine)) as s:
        assert sec.CachedSec(s, network=False).get("https://www.sec.gov/b") == "new answer"
        # Own answers come first; distinct keys are counted once.
        assert sec.CachedSec(s, network=False).get("https://www.sec.gov/a") == "overriding answer"
        assert sec_cache.count(s) == 2
        own = s.scalar(select(func.count()).select_from(ExternalResponse))
        assert own == 2
    engine.dispose()
    assert _sha(protected) == before


def test_the_parent_must_differ_from_the_database_opened(tmp_path, monkeypatch):
    path = tmp_path / "one.db"
    _make_db(path, {})
    monkeypatch.setenv(sec_cache.PARENT_VARIABLE, str(path))
    with pytest.raises(RuntimeError, match="parent cache"):
        udb.open_research_database(_url(path))


def test_strict_cache_still_records_an_answer_missing_from_both(protected, tmp_path, monkeypatch):
    monkeypatch.setenv(sec_cache.PARENT_VARIABLE, str(protected))
    engine = udb.open_research_database(_url(tmp_path / "stage2.db"))
    with session_scope(create_session_factory(engine)) as s:
        strict = evidence.StrictCache(s)
        assert strict.get("https://www.sec.gov/a") == "archived answer"
        assert strict.get("https://www.sec.gov/missing") is None
        assert strict.missing == {"https://www.sec.gov/missing"}
    engine.dispose()


def test_cover_text_falls_back_to_a_whole_document_in_the_parent(tmp_path, monkeypatch):
    parent = tmp_path / "parent.db"
    doc = "<html>Title of each class Trading Symbol(s) ABC Common Stock</html>"
    _make_db(parent, {sec.cache_key("https://www.sec.gov/doc.htm"): doc})
    monkeypatch.setenv(sec_cache.PARENT_VARIABLE, str(parent))
    engine = udb.open_research_database(_url(tmp_path / "child.db"))
    with session_scope(create_session_factory(engine)) as s:
        assert sec.CachedSec(s, network=False).cover_text("https://www.sec.gov/doc.htm") == (
            sec.cover_snippet(doc)
        )
    engine.dispose()


def test_offline_mode_stops_instead_of_requesting(tmp_path, monkeypatch):
    monkeypatch.setenv(sec_cache.OFFLINE_VARIABLE, "1")
    engine = udb.open_research_database(_url(tmp_path / "r.db"))
    with session_scope(create_session_factory(engine)) as s:
        client = sec.CachedSec(s, network=True, sleep=lambda _: None)

        def no_network(*_args, **_kwargs):
            raise AssertionError("an offline run made an HTTP request")

        monkeypatch.setattr(client._http, "get", no_network)
        with pytest.raises(sec.SecUnavailable, match="offline"):
            client.get("https://www.sec.gov/not-cached")
        assert client.requests == 0
        assert client.stored == 0
        client.close()
    engine.dispose()


def test_long_answers_are_stored_compressed_and_read_back_unchanged(tmp_path):
    engine = udb.open_research_database(_url(tmp_path / "r.db"))
    long_text = "<p>Trading Symbol(s) XYZ</p>" * 10_000
    with session_scope(create_session_factory(engine)) as s:
        sec_cache.store(s, "long", long_text)
        sec_cache.store(s, "short", "small answer")
    with session_scope(create_session_factory(engine)) as s:
        stored = s.get(ExternalResponse, ("sec", "long")).response
        assert isinstance(stored, dict)
        assert len(stored["zlib_b64"]) < len(long_text) / 10
        assert s.get(ExternalResponse, ("sec", "short")).response == "small answer"
        assert sec_cache.lookup(s, "long") == long_text
        assert sec.CachedSec(s, network=False).get("long") == long_text
    engine.dispose()


def test_a_write_stops_cleanly_when_the_disk_is_nearly_full(tmp_path, monkeypatch):
    engine = udb.open_research_database(_url(tmp_path / "r.db"))
    monkeypatch.setattr(sec_cache, "MIN_FREE_BYTES", 10**18)
    with session_scope(create_session_factory(engine)) as s:
        with pytest.raises(sec_cache.DiskLow):
            sec_cache.store(s, "key", "answer")
        assert s.get(ExternalResponse, ("sec", "key")) is None
    engine.dispose()
