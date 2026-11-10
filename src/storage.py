"""F5 — Incident log: every analysis saved to SQLite.

Usage:
    python -m src.storage [--limit 10]      # print the most recent analyses

All SQL in the project lives in this file, so moving to a hosted database (Phase 8)
changes one file. Design notes are in decisions.md; in short:

- analyses:        one row per photo. Write-once, except the alert status.
                   Holds the counts too, so History and the digest read one table.
- person_verdicts: one row per detected person, linked to its analysis.
- digest_runs:     one row per daily digest run (F8): window, outcome, error.
- schema_migrations: which numbered schema steps have been applied.

CHECK constraints make the database reject impossible data (for example counts that
don't add up), even if the Python code has a bug.
"""

import argparse
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.detect import Detection
from src.report import Report
from src.violations import MIN_PERSON_HEIGHT, PersonVerdict, count_verdicts

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "incidents.db"

SEVERITIES = ("High", "Medium", "None")
REPORT_SOURCES = ("gemini", "template")
ALERT_STATUSES = ("not_needed", "pending", "sent", "failed", "disabled")
DIGEST_STATUSES = ("sent", "failed")

# Numbered schema steps, applied in order and never edited once released.
# To change the schema later, append a new step (2, 3, ...).
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "Create analyses and person_verdicts",
        [
            f"""
            CREATE TABLE analyses (
                id                 INTEGER PRIMARY KEY,
                created_at         TEXT    NOT NULL,   -- UTC, e.g. 2026-11-05T13:00:00Z

                -- The photo (never stored itself: name and fingerprint only)
                image_name         TEXT    NOT NULL,
                image_sha256       TEXT    NOT NULL,
                image_width        INTEGER NOT NULL,
                image_height       INTEGER NOT NULL,

                -- What produced the verdicts, so old rows stay interpretable after changes
                model_version      TEXT    NOT NULL,
                conf_threshold     REAL    NOT NULL,
                min_person_height  REAL    NOT NULL,

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
                alert_updated_at   TEXT,

                processing_ms      INTEGER NOT NULL,   -- speed target: under 5 s

                CHECK (created_at LIKE '____-__-__T__:__:__Z'),
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
                analysis_id   INTEGER NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
                person_index  INTEGER NOT NULL,        -- 1, 2, 3 ... within the photo
                has_helmet    INTEGER,                 -- 1 yes, 0 no, NULL unknown (too far)
                has_vest      INTEGER,
                confidence    REAL    NOT NULL,
                x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,

                PRIMARY KEY (analysis_id, person_index),
                CHECK (person_index >= 1),
                CHECK (has_helmet IN (0, 1) OR has_helmet IS NULL),
                CHECK (has_vest   IN (0, 1) OR has_vest   IS NULL),
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
                id              INTEGER PRIMARY KEY,
                ran_at          TEXT    NOT NULL,   -- UTC, when the run finished
                window_start    TEXT    NOT NULL,   -- UTC, inclusive
                window_end      TEXT    NOT NULL,   -- UTC, exclusive
                analyses_count  INTEGER,            -- NULL if the log could not be read
                status          TEXT    NOT NULL,
                error           TEXT,               -- webhook URL already removed

                CHECK (ran_at LIKE '____-__-__T__:__:__Z'),
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

def db_path() -> Path:
    """INCIDENT_DB_PATH from .env if set, else db/incidents.db in the project."""
    return Path(os.environ.get("INCIDENT_DB_PATH") or DEFAULT_DB_PATH)


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection for one unit of work, then close it.

    One connection per call, because Gradio runs requests in threads and a SQLite
    connection must not be shared between threads.
    """
    path = Path(path or db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: no hidden transactions; we start them explicitly in transaction().
    conn = sqlite3.connect(path, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["severity"]
    conn.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite, per connection
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """All statements inside succeed together or not at all."""
    conn.execute("BEGIN IMMEDIATE")  # take the write lock now, not halfway through
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


_initialised: set[str] = set()


def init_db(path: Path | None = None) -> None:
    """Create the database if needed and apply any pending schema steps. Safe to call often."""
    path = Path(path or db_path())
    with connect(path) as conn:
        # WAL lets readers (History tab) work while a write is in progress. Stored in the file.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, description, statements in MIGRATIONS:
            if version in applied:
                continue
            with transaction(conn):
                for sql in statements:
                    conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                    (version, description, to_db_time(now_utc())),
                )
    _initialised.add(str(path.resolve()))


def _ensure_db(path: Path | None) -> Path:
    path = Path(path or db_path())
    if str(path.resolve()) not in _initialised:
        init_db(path)
    return path


# --- Time helpers ---

# Times are stored in UTC and shown in Pakistan time. Pakistan has no daylight saving,
# so a fixed offset is exact (and needs no tzdata package).
PKT = timezone(timedelta(hours=5), "PKT")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_db_time(moment: datetime) -> str:
    """Fixed-width UTC text, so sorting the text sorts by time."""
    if moment.tzinfo is None:
        raise ValueError("use timezone-aware datetimes, e.g. datetime.now(timezone.utc)")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Writes ---

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
    path: Path | None = None,
) -> int:
    """Save one analysis and its people in a single transaction. Returns the new analysis id."""
    counts = count_verdicts(verdicts)
    row = {
        "created_at": to_db_time(created_at or now_utc()),
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
        # Phase 6 changes 'pending' to 'sent', 'failed' or 'disabled'.
        "alert_status": "pending" if counts["violations"] else "not_needed",
        "processing_ms": processing_ms,
    }
    columns = ", ".join(row)
    placeholders = ", ".join(f":{name}" for name in row)  # named parameters: never build SQL from values

    people = [
        (
            i,
            None if v.has_helmet is None else int(v.has_helmet),
            None if v.has_vest is None else int(v.has_vest),
            v.person.confidence,
            v.person.x1, v.person.y1, v.person.x2, v.person.y2,
        )
        for i, v in enumerate(verdicts, 1)
    ]

    with connect(_ensure_db(path)) as conn, transaction(conn):
        analysis_id = conn.execute(f"INSERT INTO analyses ({columns}) VALUES ({placeholders})", row).lastrowid
        conn.executemany(
            "INSERT INTO person_verdicts"
            " (analysis_id, person_index, has_helmet, has_vest, confidence, x1, y1, x2, y2)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(analysis_id, *p) for p in people],
        )
    return analysis_id


def update_alert_status(
    analysis_id: int, status: str, error: str | None = None, path: Path | None = None
) -> None:
    """Record what happened to the alert for one analysis (Phase 6)."""
    with connect(_ensure_db(path)) as conn, transaction(conn):
        cursor = conn.execute(
            "UPDATE analyses SET alert_status = ?, alert_error = ?, alert_updated_at = ? WHERE id = ?",
            (status, error, to_db_time(now_utc()), analysis_id),
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
    path: Path | None = None,
) -> int:
    """Record one digest run (F8). The caller removes the webhook URL from `error` first."""
    with connect(_ensure_db(path)) as conn, transaction(conn):
        return conn.execute(
            "INSERT INTO digest_runs (ran_at, window_start, window_end, analyses_count, status, error)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (to_db_time(now_utc()), to_db_time(window_start), to_db_time(window_end), analyses_count, status, error),
        ).lastrowid


# --- Reads ---

def list_analyses(limit: int = 50, offset: int = 0, path: Path | None = None) -> list[dict]:
    """Analyses, newest first (History tab). id breaks ties within the same second."""
    with connect(_ensure_db(path)) as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis(analysis_id: int, path: Path | None = None) -> dict | None:
    """One analysis with its people, or None if the id does not exist."""
    with connect(_ensure_db(path)) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if row is None:
            return None
        people = conn.execute(
            "SELECT * FROM person_verdicts WHERE analysis_id = ? ORDER BY person_index",
            (analysis_id,),
        ).fetchall()
    result = dict(row)
    result["people"] = [_person_from_row(p) for p in people]
    return result


def get_since(since: datetime, until: datetime | None = None, path: Path | None = None) -> list[dict]:
    """Analyses with since <= created_at < until, oldest first (digest input)."""
    with connect(_ensure_db(path)) as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE created_at >= ? AND created_at < ? ORDER BY created_at, id",
            (to_db_time(since), to_db_time(until or now_utc() + timedelta(seconds=1))),
        ).fetchall()
    return [dict(r) for r in rows]


def list_digest_runs(limit: int = 20, path: Path | None = None) -> list[dict]:
    """Digest runs, newest first."""
    with connect(_ensure_db(path)) as conn:
        rows = conn.execute("SELECT * FROM digest_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_last_24h(now: datetime | None = None, path: Path | None = None) -> list[dict]:
    now = now or now_utc()
    return get_since(now - timedelta(hours=24), now + timedelta(seconds=1), path=path)


def _person_from_row(row: sqlite3.Row) -> dict:
    """Turn a stored person back into a dict, with status computed by the same rules code."""
    verdict = PersonVerdict(
        person=Detection(5, "Person", row["confidence"], row["x1"], row["y1"], row["x2"], row["y2"]),
        has_helmet=None if row["has_helmet"] is None else bool(row["has_helmet"]),
        has_vest=None if row["has_vest"] is None else bool(row["has_vest"]),
    )
    return {**dict(row), "status": verdict.status}


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the most recent analyses in the incident log.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = list_analyses(limit=args.limit)
    print(f"Database: {db_path()}  ({len(rows)} shown, newest first)")
    for r in rows:
        source = r["report_model"] or "template"
        print(
            f"#{r['id']:<4} {r['created_at']}  {r['severity']:<6} people {r['people_count']}"
            f" (ok {r['compliant_count']}, violation {r['violation_count']}, too far {r['too_far_count']})"
            f"  report: {source:<24} alert: {r['alert_status']:<10} {r['processing_ms']} ms  {r['image_name']}"
        )


if __name__ == "__main__":
    main()
