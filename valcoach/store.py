"""SQLite persistence.

Two layers live side by side:

* ``matches.payload`` keeps the **raw provider response** for every match. That
  is the source of truth — when the analysers improve, ``reindex()`` replays
  every stored match through the parser again, so old games gain new insight
  instead of being stuck with whatever we understood the day we downloaded them.
* The normalized tables (``match_players``, ``rounds``, ``round_states``,
  ``kills``) are derived from that payload and exist for fast querying.

Reports are stored too, which is what gives the coach a memory: it can compare
this session against what it told you last week.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Match

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_id       TEXT PRIMARY KEY,
    provider       TEXT,
    payload_format TEXT,
    payload        TEXT,
    map_name       TEXT,
    mode           TEXT,
    queue          TEXT,
    region         TEXT,
    season         TEXT,
    started_at     INTEGER,
    duration_ms    INTEGER,
    ingested_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_matches_started ON matches(started_at DESC);

CREATE TABLE IF NOT EXISTS match_players (
    match_id        TEXT,
    puuid           TEXT,
    name            TEXT,
    tag             TEXT,
    team            TEXT,
    agent           TEXT,
    party_id        TEXT,
    rank            TEXT,
    level           INTEGER,
    kills           INTEGER,
    deaths          INTEGER,
    assists         INTEGER,
    score           INTEGER,
    headshots       INTEGER,
    bodyshots       INTEGER,
    legshots        INTEGER,
    damage_made     INTEGER,
    damage_received INTEGER,
    PRIMARY KEY (match_id, puuid)
);
CREATE INDEX IF NOT EXISTS idx_mp_puuid ON match_players(puuid);

CREATE TABLE IF NOT EXISTS rounds (
    match_id       TEXT,
    idx            INTEGER,
    result         TEXT,
    winning_team   TEXT,
    ceremony       TEXT,
    bomb_planted   INTEGER,
    plant_time_ms  INTEGER,
    plant_site     TEXT,
    planter_team   TEXT,
    bomb_defused   INTEGER,
    defuse_time_ms INTEGER,
    attacking_team TEXT,
    PRIMARY KEY (match_id, idx)
);

CREATE TABLE IF NOT EXISTS round_states (
    match_id         TEXT,
    idx              INTEGER,
    puuid            TEXT,
    loadout_value    INTEGER,
    remaining_credits INTEGER,
    weapon_id        TEXT,
    weapon_name      TEXT,
    armor            TEXT,
    damage           INTEGER,
    headshots        INTEGER,
    bodyshots        INTEGER,
    legshots         INTEGER,
    kills            INTEGER,
    casts_c          INTEGER,
    casts_q          INTEGER,
    casts_e          INTEGER,
    casts_x          INTEGER,
    was_afk          INTEGER,
    stayed_in_spawn  INTEGER,
    PRIMARY KEY (match_id, idx, puuid)
);
CREATE INDEX IF NOT EXISTS idx_rs_puuid ON round_states(puuid);

CREATE TABLE IF NOT EXISTS kills (
    match_id        TEXT,
    round_index     INTEGER,
    time_in_round_ms INTEGER,
    time_in_match_ms INTEGER,
    killer_puuid    TEXT,
    killer_name     TEXT,
    killer_team     TEXT,
    killer_agent    TEXT,
    victim_puuid    TEXT,
    victim_name     TEXT,
    victim_team     TEXT,
    victim_agent    TEXT,
    weapon_id       TEXT,
    weapon_name     TEXT,
    victim_x        REAL,
    victim_y        REAL,
    secondary_fire  INTEGER,
    locations       TEXT,
    PRIMARY KEY (match_id, round_index, time_in_match_ms, killer_puuid, victim_puuid)
);
CREATE INDEX IF NOT EXISTS idx_kills_victim ON kills(victim_puuid);
CREATE INDEX IF NOT EXISTS idx_kills_killer ON kills(killer_puuid);

CREATE TABLE IF NOT EXISTS identities (
    puuid     TEXT PRIMARY KEY,
    name      TEXT,
    tag       TEXT,
    last_seen INTEGER
);

CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER,
    puuid      TEXT,
    riot_id    TEXT,
    match_ids  TEXT,
    metrics    TEXT,
    findings   TEXT,
    narrative  TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_puuid ON reports(puuid, created_at DESC);
"""


def default_db_path() -> str:
    base = os.environ.get("VALCOACH_HOME") or os.path.join(
        os.path.expanduser("~"), ".valcoach"
    )
    return os.path.join(base, "valcoach.db")


class Store:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_db_path()
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    # ---- lifecycle -----------------------------------------------------
    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- meta ----------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    # ---- writes --------------------------------------------------------
    def has_match(self, match_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        return row is not None

    def known_match_ids(self) -> set:
        return {
            r["match_id"] for r in self.conn.execute("SELECT match_id FROM matches")
        }

    def save_match(
        self,
        match: Match,
        raw_payload: Optional[Any] = None,
        payload_format: str = "",
    ) -> bool:
        """Insert or replace a match and everything derived from it.

        Returns True when the match was not previously stored.
        """
        is_new = not self.has_match(match.match_id)
        payload = (
            json.dumps(raw_payload, separators=(",", ":"))
            if raw_payload is not None
            else None
        )
        cur = self.conn.cursor()
        if payload is None and not is_new:
            # Keep the payload we already have (e.g. reindex path).
            cur.execute(
                """UPDATE matches SET provider=?, map_name=?, mode=?, queue=?, region=?,
                       season=?, started_at=?, duration_ms=? WHERE match_id=?""",
                (
                    match.provider, match.map_name, match.mode, match.queue,
                    match.region, match.season, match.started_at, match.duration_ms,
                    match.match_id,
                ),
            )
        else:
            cur.execute(
                """INSERT INTO matches(match_id, provider, payload_format, payload,
                        map_name, mode, queue, region, season, started_at, duration_ms,
                        ingested_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(match_id) DO UPDATE SET
                        provider=excluded.provider,
                        payload_format=excluded.payload_format,
                        payload=excluded.payload,
                        map_name=excluded.map_name,
                        mode=excluded.mode,
                        queue=excluded.queue,
                        region=excluded.region,
                        season=excluded.season,
                        started_at=excluded.started_at,
                        duration_ms=excluded.duration_ms""",
                (
                    match.match_id, match.provider,
                    payload_format or match.provider, payload,
                    match.map_name, match.mode, match.queue, match.region,
                    match.season, match.started_at, match.duration_ms,
                    int(time.time()),
                ),
            )

        cur.execute("DELETE FROM match_players WHERE match_id=?", (match.match_id,))
        cur.execute("DELETE FROM rounds WHERE match_id=?", (match.match_id,))
        cur.execute("DELETE FROM round_states WHERE match_id=?", (match.match_id,))
        cur.execute("DELETE FROM kills WHERE match_id=?", (match.match_id,))

        now = int(time.time())
        for p in match.players:
            s = p.stats
            cur.execute(
                """INSERT INTO match_players VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    match.match_id, p.ref.puuid, p.ref.name, p.ref.tag, p.ref.team,
                    p.ref.agent, p.party_id, p.rank, p.level, s.kills, s.deaths,
                    s.assists, s.score, s.headshots, s.bodyshots, s.legshots,
                    s.damage_made, s.damage_received,
                ),
            )
            cur.execute(
                """INSERT INTO identities(puuid, name, tag, last_seen) VALUES(?,?,?,?)
                   ON CONFLICT(puuid) DO UPDATE SET name=excluded.name,
                        tag=excluded.tag, last_seen=excluded.last_seen""",
                (p.ref.puuid, p.ref.name, p.ref.tag, now),
            )

        for r in match.rounds:
            cur.execute(
                "INSERT INTO rounds VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    match.match_id, r.index, r.result, r.winning_team, r.ceremony,
                    int(r.bomb_planted), r.plant_time_ms, r.plant_site, r.planter_team,
                    int(r.bomb_defused), r.defuse_time_ms, r.attacking_team,
                ),
            )
            for st in r.states.values():
                cur.execute(
                    "INSERT INTO round_states VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        match.match_id, r.index, st.puuid, st.loadout_value,
                        st.remaining_credits, st.weapon.id, st.weapon.name, st.armor,
                        st.damage, st.headshots, st.bodyshots, st.legshots, st.kills,
                        st.casts.c, st.casts.q, st.casts.e, st.casts.x,
                        int(st.was_afk), int(st.stayed_in_spawn),
                    ),
                )

        for k in match.kills:
            cur.execute(
                """INSERT OR REPLACE INTO kills VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    match.match_id, k.round_index, k.time_in_round_ms,
                    k.time_in_match_ms, k.killer.puuid, k.killer.name, k.killer.team,
                    k.killer.agent, k.victim.puuid, k.victim.name, k.victim.team,
                    k.victim.agent, k.weapon.id, k.weapon.name, k.victim_x, k.victim_y,
                    int(k.secondary_fire),
                    json.dumps([asdict(pl) for pl in k.player_locations],
                               separators=(",", ":")),
                ),
            )
        self.conn.commit()
        return is_new

    # ---- reads ---------------------------------------------------------
    def raw_payload(self, match_id: str) -> Tuple[Optional[Any], str]:
        row = self.conn.execute(
            "SELECT payload, payload_format FROM matches WHERE match_id=?", (match_id,)
        ).fetchone()
        if not row or not row["payload"]:
            return None, ""
        return json.loads(row["payload"]), row["payload_format"] or ""

    def match_rows(
        self,
        puuid: Optional[str] = None,
        limit: Optional[int] = None,
        queue: Optional[str] = None,
        map_name: Optional[str] = None,
        since: Optional[int] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT m.* FROM matches m"]
        args: List[Any] = []
        where = []
        if puuid:
            sql.append("JOIN match_players p ON p.match_id = m.match_id")
            where.append("p.puuid = ?")
            args.append(puuid)
        if queue:
            where.append("LOWER(m.queue) = LOWER(?)")
            args.append(queue)
        if map_name:
            where.append("LOWER(m.map_name) = LOWER(?)")
            args.append(map_name)
        if since:
            where.append("m.started_at >= ?")
            args.append(int(since))
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY m.started_at DESC")
        if limit:
            sql.append("LIMIT ?")
            args.append(int(limit))
        return list(self.conn.execute(" ".join(sql), args))

    def load_matches(self, **kwargs: Any) -> List[Match]:
        """Rehydrate Match objects (re-parsed from the stored raw payload)."""
        from .providers import parse_payload  # local import: avoids a cycle

        out: List[Match] = []
        for row in self.match_rows(**kwargs):
            payload, fmt = self.raw_payload(row["match_id"])
            if payload is None:
                continue
            try:
                match = parse_payload(payload, fmt)
            except Exception:  # noqa: BLE001 - a single bad payload must not kill a report
                continue
            if match:
                out.append(match)
        return out

    def resolve_puuid(self, needle: str) -> Optional[str]:
        """Accept a puuid or 'Name#TAG' and return the puuid we have stored."""
        needle = (needle or "").strip()
        if not needle:
            return None
        row = self.conn.execute(
            "SELECT puuid FROM identities WHERE puuid=?", (needle,)
        ).fetchone()
        if row:
            return row["puuid"]
        if "#" in needle:
            name, tag = needle.split("#", 1)
            row = self.conn.execute(
                "SELECT puuid FROM identities WHERE LOWER(name)=LOWER(?) "
                "AND LOWER(tag)=LOWER(?) ORDER BY last_seen DESC",
                (name.strip(), tag.strip()),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT puuid FROM identities WHERE LOWER(name)=LOWER(?) "
                "ORDER BY last_seen DESC",
                (needle,),
            ).fetchone()
        return row["puuid"] if row else None

    def counts(self) -> Dict[str, int]:
        q = lambda t: self.conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        return {
            "matches": q("matches"),
            "rounds": q("rounds"),
            "kills": q("kills"),
            "players": q("identities"),
            "reports": q("reports"),
        }

    def reindex(self) -> int:
        """Re-parse every stored payload through the current parser."""
        from .providers import parse_payload

        done = 0
        for row in self.conn.execute("SELECT match_id FROM matches").fetchall():
            payload, fmt = self.raw_payload(row["match_id"])
            if payload is None:
                continue
            try:
                match = parse_payload(payload, fmt)
            except Exception:  # noqa: BLE001
                continue
            if match:
                self.save_match(match)
                done += 1
        return done

    # ---- report memory -------------------------------------------------
    def save_report(
        self,
        puuid: str,
        riot_id: str,
        match_ids: Sequence[str],
        metrics: Dict[str, Any],
        findings: Iterable[Dict[str, Any]],
        narrative: str = "",
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO reports(created_at, puuid, riot_id, match_ids, metrics,
                                   findings, narrative)
               VALUES(?,?,?,?,?,?,?)""",
            (
                int(time.time()), puuid, riot_id, json.dumps(list(match_ids)),
                json.dumps(metrics, default=str), json.dumps(list(findings), default=str),
                narrative,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def recent_reports(self, puuid: str, limit: int = 5) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            # id breaks ties: two reviews can land in the same second.
            "SELECT * FROM reports WHERE puuid=? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (puuid, int(limit)),
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "riot_id": r["riot_id"],
                    "match_ids": json.loads(r["match_ids"] or "[]"),
                    "metrics": json.loads(r["metrics"] or "{}"),
                    "findings": json.loads(r["findings"] or "[]"),
                    "narrative": r["narrative"] or "",
                }
            )
        return out
