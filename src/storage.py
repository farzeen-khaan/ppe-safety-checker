"""F5 — Incident log: every analysis saved to Postgres (Neon).

Usage:
    python -m src.storage [--limit 10]      # print the most recent analyses

The connection string comes from DATABASE_URL. Every environment uses Postgres: the Neon
`main` branch in production, `dev` for the local app, `test` for the automated tests
(DATABASE_URL_TEST). All SQL in the project lives in this file. Design notes are in
decisions.md; in short:

- analyses:          one row per photo. Write-once, except the alert status.
                     Holds the counts too, so History and the digest read one table.
- person_verdicts:   one row per detected person, linked to its analysis.
- digest_runs:       one row per daily digest run (F8): window, outcome, error.
- schema_migrations: which numbered schema steps have been applied.

CHECK constraints make the database reject impossible data (for example counts that
don't add up), even if the Python code has a bug.

Neon scales an idle branch to zero after 5 minutes, so the first connection after a
pause can be slow. Each connection attempt may take 10 s and is retried once; after
that, StorageUnavailable is raised with a clear message (never the password).

This module imports no detection code (only src.verdicts), so the digest stays light.
"""

import argparse
import atexit
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from src.report import Report  # light: no detection imports
from src.verdicts import MIN_PERSON_HEIGHT, Detection, PersonVerdict, count_verdicts

load_dotenv()  # reads .env into environment variables; real environment variables win

SEVERITIES = ("High", "Medium", "None")
REPORT_SOURCES = ("gemini", "template")
ALERT_STATUSES = ("not_needed", "pending", "sent", "failed", "disabled")
DIGEST_STATUSES = ("sent", "failed")

# Cold starts: a sleeping Neon branch usually wakes in about a second. One attempt may take
# CONNECT_TIMEOUT_S; the pool retries once after about 1 s (its first backoff step), so a
# caller waits at most POOL_TIMEOUT_S before StorageUnavailable.
CONNECT_TIMEOUT_S = 10
POOL_TIMEOUT_S = 2 * CONNECT_TIMEOUT_S + 2
# Idle connections are closed after a minute, so the pool never keeps Neon awake.
MAX_IDLE_S = 60
MAX_CONNECTIONS = 4  # Gradio handlers, the alert thread and the History tab, at most a few at once

# Any fixed number: lets only one process apply migrations at a time (app and digest may start together).
MIGRATION_LOCK_ID = 727_000_008


class StorageUnavailable(RuntimeError):
    """The incident log can't be reached or isn't configured. The message never contains the password."""


# Numbered schema steps, applied in order and never edited once released.
# To change the schema later, append a new step (3, 4, ...).
# Postgres port of the Phase 4 and Phase 7 SQLite steps (see decisions.md, Phase 8a).
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "Create analyses and person_verdicts",
        [
            f"""
            CREATE TABLE analyses (
                id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                created_at         TIMESTAMPTZ NOT NULL,      -- an instant; shown in UTC text by this module

                -- The photo (never stored itself: name and fingerprint only)
                image_name         TEXT    NOT NULL,
                image_sha256       TEXT    NOT NULL,
                image_width        INTEGER NOT NULL,
                image_height       INTEGER NOT NULL,

                -- What produced the verdicts, so old rows stay interpretable after changes
                model_version      TEXT    NOT NULL,
                conf_threshold     DOUBLE PRECISION NOT NULL,
                min_person_height  DOUBLE PRECISION NOT NULL,

                -- Counts (written once, from the same verdicts as person_verdicts)
                people_count       INTEGER NOT NULL,
                compliant_count    INTEGER NOT NULL,
                violation_count    INTEGER NOT NULL,
                too_far_count      INTEGER NOT NULL,
                missing_helmet_only_count INTEGER NOT NULL,
                missing_vest_only_count   INTEGER NOT NULL,
                missing_both_count        INTEGER NOT NULL,

                -- Report (F4) and its provenance
                severity           TEXT    NOT NULL,
                report_summary     TEXT    NOT NULL,
                report_action      TEXT    NOT NULL,
                report_source      TEXT    NOT NULL,
                report_model       TEXT,               -- which Gemini model wrote it
                report_error       TEXT,               -- why the template was used

                -- Alert (F7): the only part that changes after the insert
                alert_status       TEXT    NOT NULL,
                alert_error        TEXT,
                alert_updated_at   TIMESTAMPTZ,

                processing_ms      INTEGER NOT NULL,   -- speed target: under 5 s

                CHECK (image_width > 0 AND image_height > 0),
                CHECK (conf_threshold > 0 AND conf_threshold < 1),
                CHECK (min_person_height >= 0 AND min_person_height < 1),
                CHECK (people_count >= 0 AND compliant_count >= 0 AND violation_count >= 0
                       AND too_far_count >= 0 AND missing_helmet_only_count >= 0
                       AND missing_vest_only_count >= 0 AND missing_both_count >= 0),
                CHECK (people_count = compliant_count + violation_count + too_far_count),
                CHECK (violation_count = missing_helmet_only_count + missing_vest_only_count
                                         + missing_both_count),
                CHECK (severity IN {SEVERITIES}),
                CHECK (report_source IN {REPORT_SOURCES}),
                CHECK ((report_source = 'gemini'   AND report_model IS NOT NULL AND report_error IS NULL)
                    OR (report_source = 'template' AND report_model IS NULL     AND report_error IS NOT NULL)),
                CHECK (alert_status IN {ALERT_STATUSES}),
                CHECK (processing_ms >= 0)
            )
            """,
            # History ("newest first") and the digest ("last 24 hours") both filter or sort by time.
            "CREATE INDEX idx_analyses_created_at ON analyses (created_at)",
            """
            CREATE TABLE person_verdicts (
                analysis_id   BIGINT  NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
                person_index  INTEGER NOT NULL,        -- 1, 2, 3 ... within the photo
                has_helmet    BOOLEAN,                 -- NULL: unknown (too far to judge)
                has_vest      BOOLEAN,
                confidence    DOUBLE PRECISION NOT NULL,
                x1 DOUBLE PRECISION NOT NULL, y1 DOUBLE PRECISION NOT NULL,
                x2 DOUBLE PRECISION NOT NULL, y2 DOUBLE PRECISION NOT NULL,

                PRIMARY KEY (analysis_id, person_index),
                CHECK (person_index >= 1),
                CHECK ((has_helmet IS NULL) = (has_vest IS NULL)),  -- too far means both unknown
                CHECK (confidence >= 0 AND confidence <= 1),
                CHECK (x2 > x1 AND y2 > y1)
            )
            """,
        ],
    ),
    (
        2,
        "Create digest_runs",
        [
            f"""
            CREATE TABLE digest_runs (
                id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                ran_at          TIMESTAMPTZ NOT NULL,   -- when the run finished
                window_start    TIMESTAMPTZ NOT NULL,   -- inclusive
                window_end      TIMESTAMPTZ NOT NULL,   -- exclusive
                analyses_count  INTEGER,                -- NULL if the log could not be read
                status          TEXT    NOT NULL,
                error           TEXT,                   -- webhook URL already removed

                CHECK (window_start < window_end),
                CHECK (analyses_count IS NULL OR analyses_count >= 0),
                CHECK (status IN {DIGEST_STATUSES}),
                CHECK ((status = 'sent') = (error IS NULL))
            )
            """,
        ],
    ),
]


# --- Connections and transactions ---

def database_url() -> str:
    """DATABASE_URL, or StorageUnavailable if it is not set (there is no local fallback)."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise StorageUnavailable("DATABASE_URL is not set: the incident log needs a Postgres connection string")
    return url


def describe_database(dsn: str | None = None) -> str:
    """'host/dbname' for messages: never the user or password."""
    parts = urlsplit(dsn or database_url())
    return f"{parts.hostname}{parts.path}"


def _scrub(text: str, dsn: str) -> str:
    """Remove the connection string and its password from an error message."""
    password = urlsplit(dsn).password
    text = text.replace(dsn, "<DATABASE_URL>")
    return text.replace(password, "<password>") if password else text


_pools: dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def _pool(dsn: str) -> ConnectionPool:
    """One small pool per connection string, created on first use."""
    with _pools_lock:
        pool = _pools.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                dsn,
                min_size=0,  # no connection is kept open when idle, so Neon can scale to zero
                max_size=MAX_CONNECTIONS,
                max_idle=MAX_IDLE_S,
                timeout=POOL_TIMEOUT_S,
                check=ConnectionPool.check_connection,  # a connection dropped while idle is replaced, not used
                kwargs={
                    "autocommit": True,  # no hidden transactions; transaction() starts them explicitly
                    "row_factory": dict_row,  # rows behave like dicts: row["severity"]
                    "connect_timeout": CONNECT_TIMEOUT_S,
                    "prepare_threshold": None,  # the Neon pooler (PgBouncer) shares server connections
                    "application_name": "ppe-safety-checker",
                },
                name="incident-log",
                open=True,
            )
            _pools[dsn] = pool
        return pool


def close_pools() -> None:
    """Close every pool (at exit, and between tests that change DATABASE_URL)."""
    with _pools_lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()
    _initialised.clear()


atexit.register(close_pools)


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Borrow a connection for one unit of work, then give it back.

    The pool hands each connection to one thread at a time, because Gradio runs requests
    in threads and the alerts run on their own thread.
    """
    dsn = dsn or database_url()
    pool = _pool(dsn)
    try:
        conn = pool.getconn()
    except PoolTimeout as exc:
        raise StorageUnavailable(
            f"The incident log ({describe_database(dsn)}) could not be reached within {POOL_TIMEOUT_S} s, "
            f"retries included ({_scrub(str(exc), dsn)}). Neon may be waking up or unreachable; try again shortly."
        ) from None
    try:
        yield conn
    finally:
        pool.putconn(conn)  # a broken connection, or one left inside a transaction, is discarded


@contextmanager
def transaction(conn: psycopg.Connection) -> Iterator[None]:
    """All statements inside succeed together or not at all (COMMIT, or ROLLBACK on any error)."""
    with conn.transaction():
        yield


_initialised: set[str] = set()


def init_db(dsn: str | None = None, migrations: list | None = None) -> None:
    """Create the tables if needed and apply pending schema steps. Safe to call often, from many processes."""
    dsn = dsn or database_url()
    with connect(dsn) as conn, transaction(conn):
        # Held until COMMIT: a second process starting at the same time waits here, then finds nothing to do.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL)"
        )
        applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
        # DDL is transactional in Postgres: if any step fails, every step in this call is rolled back.
        for version, description, statements in migrations or MIGRATIONS:
            if version in applied:
                continue
            for sql in statements:
                conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) VALUES (%s, %s, %s)",
                (version, description, now_utc()),
            )
    if migrations is None:
        _initialised.add(dsn)


def _ensure_db(dsn: str | None) -> str:
    dsn = dsn or database_url()
    if dsn not in _initialised:
        init_db(dsn)
    return dsn


# --- Time helpers ---

# Times are stored as TIMESTAMPTZ and shown in Pakistan time. Pakistan has no daylight saving,
# so a fixed offset is exact (and needs no tzdata package).
PKT = timezone(timedelta(hours=5), "PKT")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _whole_seconds(moment: datetime) -> datetime:
    """Stored times are kept to the second, as readers see them (see to_db_time)."""
    if moment.tzinfo is None:
        raise ValueError("use timezone-aware datetimes, e.g. datetime.now(timezone.utc)")
    return moment.astimezone(timezone.utc).replace(microsecond=0)


def to_db_time(moment: datetime) -> str:
    """Fixed-width UTC text ('2026-11-05T13:00:00Z'): how every time leaves this module."""
    return _whole_seconds(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(row: dict) -> dict:
    """A database row as callers have always seen it: times as UTC text."""
    return {k: to_db_time(v) if isinstance(v, datetime) else v for k, v in row.items()}


# --- Writes ---

def analysis_row(
    *,
    image_name: str,
    image_sha256: str,
    image_width: int,
    image_height: int,
    verdicts: list[PersonVerdict],
    report: Report,
    model_version: str,
    conf_threshold: float,
    processing_ms: int,
    created_at: datetime | None = None,
) -> dict:
    """The analyses columns for one photo, before it is saved (also used by the digest tests)."""
    counts = count_verdicts(verdicts)
    return {
        "created_at": _whole_seconds(created_at or now_utc()),
        "image_name": image_name,
        "image_sha256": image_sha256,
        "image_width": image_width,
        "image_height": image_height,
        "model_version": model_version,
        "conf_threshold": conf_threshold,
        "min_person_height": MIN_PERSON_HEIGHT,
        "people_count": counts["people"],
        "compliant_count": counts["compliant"],
        "violation_count": counts["violations"],
        "too_far_count": counts["too_far"],
        "missing_helmet_only_count": sum(v.missing == ["helmet"] for v in verdicts),
        "missing_vest_only_count": sum(v.missing == ["vest"] for v in verdicts),
        "missing_both_count": sum(len(v.missing) == 2 for v in verdicts),
        "severity": report.severity,
        "report_summary": report.summary,
        "report_action": report.recommended_action,
        "report_source": report.source,
        "report_model": report.model,
        "report_error": report.error if report.source == "template" else None,
        # alerts.py changes 'pending' to 'sent', 'failed' or 'disabled'.
        "alert_status": "pending" if counts["violations"] else "not_needed",
        "processing_ms": processing_ms,
    }


def save_analysis(
    *,
    image_name: str,
    image_sha256: str,
    image_width: int,
    image_height: int,
    verdicts: list[PersonVerdict],
    report: Report,
    model_version: str,
    conf_threshold: float,
    processing_ms: int,
    created_at: datetime | None = None,
    dsn: str | None = None,
) -> int:
    """Save one analysis and its people in a single transaction. Returns the new analysis id."""
    row = analysis_row(
        image_name=image_name, image_sha256=image_sha256, image_width=image_width,
        image_height=image_height, verdicts=verdicts, report=report, model_version=model_version,
        conf_threshold=conf_threshold, processing_ms=processing_ms, created_at=created_at,
    )
    columns = ", ".join(row)
    placeholders = ", ".join(f"%({name})s" for name in row)  # named parameters: never build SQL from values

    people = [
        (i, v.has_helmet, v.has_vest, v.person.confidence, v.person.x1, v.person.y1, v.person.x2, v.person.y2)
        for i, v in enumerate(verdicts, 1)
    ]

    with connect(_ensure_db(dsn)) as conn, transaction(conn):
        analysis_id = conn.execute(
            f"INSERT INTO analyses ({columns}) VALUES ({placeholders}) RETURNING id", row
        ).fetchone()["id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO person_verdicts"
                " (analysis_id, person_index, has_helmet, has_vest, confidence, x1, y1, x2, y2)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(analysis_id, *p) for p in people],
            )
    return analysis_id


def update_alert_status(analysis_id: int, status: str, error: str | None = None, dsn: str | None = None) -> None:
    """Record what happened to the alert for one analysis (F7)."""
    with connect(_ensure_db(dsn)) as conn, transaction(conn):
        cursor = conn.execute(
            "UPDATE analyses SET alert_status = %s, alert_error = %s, alert_updated_at = %s WHERE id = %s",
            (status, error, _whole_seconds(now_utc()), analysis_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no analysis with id {analysis_id}")


def record_digest_run(
    *,
    window_start: datetime,
    window_end: datetime,
    status: str,
    analyses_count: int | None = None,
    error: str | None = None,
    dsn: str | None = None,
) -> int:
    """Record one digest run (F8). The caller removes the webhook URL from `error` first."""
    with connect(_ensure_db(dsn)) as conn, transaction(conn):
        return conn.execute(
            "INSERT INTO digest_runs (ran_at, window_start, window_end, analyses_count, status, error)"
            " VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (_whole_seconds(now_utc()), _whole_seconds(window_start), _whole_seconds(window_end),
             analyses_count, status, error),
        ).fetchone()["id"]


# --- Reads ---

def list_analyses(limit: int = 50, offset: int = 0, dsn: str | None = None) -> list[dict]:
    """Analyses, newest first (History tab). id breaks ties within the same second."""
    with connect(_ensure_db(dsn)) as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s", (limit, offset)
        ).fetchall()
    return [_row(r) for r in rows]


def get_analysis(analysis_id: int, dsn: str | None = None) -> dict | None:
    """One analysis with its people, or None if the id does not exist."""
    with connect(_ensure_db(dsn)) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = %s", (analysis_id,)).fetchone()
        if row is None:
            return None
        people = conn.execute(
            "SELECT * FROM person_verdicts WHERE analysis_id = %s ORDER BY person_index", (analysis_id,)
        ).fetchall()
    result = _row(row)
    result["people"] = [_person_from_row(p) for p in people]
    return result


def get_since(since: datetime, until: datetime | None = None, dsn: str | None = None) -> list[dict]:
    """Analyses with since <= created_at < until, oldest first (digest input)."""
    until = until or now_utc() + timedelta(seconds=1)
    with connect(_ensure_db(dsn)) as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE created_at >= %s AND created_at < %s ORDER BY created_at, id",
            (_whole_seconds(since), _whole_seconds(until)),
        ).fetchall()
    return [_row(r) for r in rows]


def get_last_24h(now: datetime | None = None, dsn: str | None = None) -> list[dict]:
    now = now or now_utc()
    return get_since(now - timedelta(hours=24), now + timedelta(seconds=1), dsn=dsn)


def list_digest_runs(limit: int = 20, dsn: str | None = None) -> list[dict]:
    """Digest runs, newest first."""
    with connect(_ensure_db(dsn)) as conn:
        rows = conn.execute("SELECT * FROM digest_runs ORDER BY id DESC LIMIT %s", (limit,)).fetchall()
    return [_row(r) for r in rows]


def _person_from_row(row: dict) -> dict:
    """Turn a stored person back into a dict, with status computed by the same rules code."""
    verdict = PersonVerdict(
        person=Detection(5, "Person", row["confidence"], row["x1"], row["y1"], row["x2"], row["y2"]),
        has_helmet=row["has_helmet"],
        has_vest=row["has_vest"],
    )
    return {**row, "status": verdict.status}


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the most recent analyses in the incident log.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = list_analyses(limit=args.limit)
    print(f"Database: {describe_database()}  ({len(rows)} shown, newest first)")
    for r in rows:
        source = r["report_model"] or "template"
        print(
            f"#{r['id']:<4} {r['created_at']}  {r['severity']:<6} people {r['people_count']}"
            f" (ok {r['compliant_count']}, violation {r['violation_count']}, too far {r['too_far_count']})"
            f"  report: {source:<24} alert: {r['alert_status']:<10} {r['processing_ms']} ms  {r['image_name']}"
        )


if __name__ == "__main__":
    main()
