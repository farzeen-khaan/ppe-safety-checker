"""Tests for F5 storage on Postgres.

Tests that take the `db` fixture run against the Neon test branch (DATABASE_URL_TEST) and
leave its tables empty; without that variable they are skipped. The rest run offline.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from psycopg import errors

from src import storage
from src.detect import Detection
from src.report import Report
from src.violations import PersonVerdict

T0 = datetime(2026, 11, 5, 13, 0, 0, tzinfo=timezone.utc)
ALL_TABLES = "digest_runs, person_verdicts, analyses, schema_migrations"


def person(has_helmet, has_vest, x1=0.0) -> PersonVerdict:
    box = Detection(5, "Person", 0.9, x1, 0.0, x1 + 100, 300.0)
    return PersonVerdict(person=box, has_helmet=has_helmet, has_vest=has_vest)


GEMINI_REPORT = Report("High", "Summary.", "Action.", source="gemini", model="gemini-test")
TEMPLATE_REPORT = Report("None", "Summary.", "Action.", source="template", error="LLM disabled")


def save(verdicts=None, report=GEMINI_REPORT, created_at=T0, name="photo.jpg") -> int:
    return storage.save_analysis(
        image_name=name,
        image_sha256="ab" * 32,
        image_width=2000,
        image_height=1000,
        verdicts=verdicts if verdicts is not None else [person(True, True), person(False, False), person(None, None)],
        report=report,
        model_version="best.pt:test",
        conf_threshold=0.25,
        processing_ms=1234,
        created_at=created_at,
    )


def versions(db) -> list[int]:
    with storage.connect(db) as conn:
        return [r["version"] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]


def drop_everything(db) -> None:
    with storage.connect(db) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {ALL_TABLES} CASCADE")
    storage._initialised.discard(db)


@pytest.fixture
def fresh(db):
    """The test branch with no tables at all; fully migrated again afterwards."""
    drop_everything(db)
    yield db
    storage._initialised.discard(db)
    storage.init_db(db)


# --- Schema and migrations ---

def test_init_is_idempotent_and_records_migrations(db):
    storage.init_db(db)  # a second call must not fail or re-apply anything
    with storage.connect(db) as conn:
        tables = {r["table_name"] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        )}
    assert versions(db) == [1, 2]
    assert {"analyses", "person_verdicts", "digest_runs", "schema_migrations"} <= tables


def test_existing_version_1_log_is_upgraded(fresh):
    """A log created before Phase 7 gains digest_runs and keeps its analyses."""
    storage.init_db(fresh, migrations=storage.MIGRATIONS[:1])
    assert versions(fresh) == [1]
    # Raw SQL, because save_analysis would apply the pending migration first.
    with storage.connect(fresh) as conn:
        old_id = conn.execute(
            "INSERT INTO analyses (created_at, image_name, image_sha256, image_width, image_height, model_version,"
            " conf_threshold, min_person_height, people_count, compliant_count, violation_count, too_far_count,"
            " missing_helmet_only_count, missing_vest_only_count, missing_both_count, severity, report_summary,"
            " report_action, report_source, report_model, report_error, alert_status, processing_ms)"
            " VALUES (%s, 'old.jpg', 'ab', 10, 10, 'm', 0.25, 0.15, 0, 0, 0, 0, 0, 0, 0, 'None', 's', 'a',"
            " 'template', NULL, 'LLM disabled', 'not_needed', 1) RETURNING id",
            (T0,),
        ).fetchone()["id"]
    storage.init_db(fresh)
    assert versions(fresh) == [1, 2]
    assert storage.get_analysis(old_id)["image_name"] == "old.jpg"
    assert storage.list_digest_runs() == []


def test_a_failed_migration_changes_nothing(fresh):
    """DDL is transactional in Postgres: a broken step rolls back every step of that run."""
    broken = storage.MIGRATIONS + [(3, "Broken step", ["CREATE TABLE half_done (x INTEGER)", "NOT VALID SQL"])]
    with pytest.raises(errors.SyntaxError):
        storage.init_db(fresh, migrations=broken)
    with storage.connect(fresh) as conn:
        left = conn.execute(
            "SELECT count(*) AS n FROM information_schema.tables WHERE table_schema = current_schema()"
        ).fetchone()["n"]
    assert left == 0  # not even schema_migrations, steps 1-2 or half_done


def test_processes_starting_together_migrate_once(fresh):
    """The app and the digest may start at the same moment; the advisory lock serialises them."""
    failures = []

    def start():
        try:
            storage.init_db(fresh)
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            failures.append(exc)

    threads = [threading.Thread(target=start) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == [] and versions(fresh) == [1, 2]


# --- Round trip ---

def test_save_and_read_back(db):
    a = storage.get_analysis(save())

    assert a["created_at"] == "2026-11-05T13:00:00Z"  # TIMESTAMPTZ, returned as the same UTC text as before
    assert (a["people_count"], a["compliant_count"], a["violation_count"], a["too_far_count"]) == (3, 1, 1, 1)
    assert a["missing_both_count"] == 1
    assert (a["report_source"], a["report_model"], a["report_error"]) == ("gemini", "gemini-test", None)
    assert a["min_person_height"] == 0.15 and a["conf_threshold"] == 0.25  # DOUBLE PRECISION: exact
    assert a["model_version"] == "best.pt:test"

    statuses = [p["status"] for p in a["people"]]
    assert statuses == ["compliant", "missing helmet and vest", "too far to judge"]
    assert [(p["has_helmet"], p["has_vest"]) for p in a["people"]] == [(True, True), (False, False), (None, None)]
    assert a["people"][0]["confidence"] == 0.9


def test_times_are_kept_to_the_second(db):
    a = storage.get_analysis(save(created_at=T0 + timedelta(microseconds=999_999)))
    assert a["created_at"] == "2026-11-05T13:00:00Z"
    assert [r["id"] for r in storage.get_since(T0, T0 + timedelta(seconds=1))] == [a["id"]]


def test_template_report_keeps_its_reason(db):
    a = storage.get_analysis(save(verdicts=[person(True, True)], report=TEMPLATE_REPORT))
    assert (a["report_source"], a["report_model"], a["report_error"]) == ("template", None, "LLM disabled")


def test_photo_with_no_people(db):
    a = storage.get_analysis(save(verdicts=[], report=TEMPLATE_REPORT))
    assert a["people_count"] == 0 and a["people"] == []


def test_unknown_id_returns_none(db):
    assert storage.get_analysis(999) is None


# --- Alert status ---

def test_alert_starts_pending_only_when_there_are_violations(db):
    with_violation = storage.get_analysis(save())
    without = storage.get_analysis(save(verdicts=[person(True, True)], report=TEMPLATE_REPORT))
    assert with_violation["alert_status"] == "pending"
    assert without["alert_status"] == "not_needed"


def test_update_alert_status(db):
    analysis_id = save()
    storage.update_alert_status(analysis_id, "failed", error="timeout")
    a = storage.get_analysis(analysis_id)
    assert (a["alert_status"], a["alert_error"]) == ("failed", "timeout")
    assert a["alert_updated_at"].endswith("Z")


def test_update_alert_status_rejects_unknown_id_and_status(db):
    with pytest.raises(KeyError):
        storage.update_alert_status(999, "sent")
    with pytest.raises(psycopg.IntegrityError):
        storage.update_alert_status(save(), "delivered")  # not an allowed status


# --- The database rejects impossible data (CHECK constraints) ---

def insert_raw(db, **changes):
    """Insert a valid analyses row directly with SQL, with some columns changed."""
    row = {
        "created_at": T0, "image_name": "x.jpg", "image_sha256": "ab",
        "image_width": 10, "image_height": 10, "model_version": "m", "conf_threshold": 0.25,
        "min_person_height": 0.15, "people_count": 1, "compliant_count": 1, "violation_count": 0,
        "too_far_count": 0, "missing_helmet_only_count": 0, "missing_vest_only_count": 0,
        "missing_both_count": 0, "severity": "None", "report_summary": "s", "report_action": "a",
        "report_source": "gemini", "report_model": "m", "report_error": None,
        "alert_status": "not_needed", "processing_ms": 1,
    } | changes
    with storage.connect(db) as conn:
        conn.execute(
            f"INSERT INTO analyses ({', '.join(row)}) VALUES ({', '.join(f'%({k})s' for k in row)})", row
        )


def test_valid_raw_row_is_accepted(db):
    insert_raw(db)  # proves the rejections below are caused by the one changed column


@pytest.mark.parametrize(
    "changes",
    [
        {"severity": "high"},                                    # not an allowed value
        {"people_count": 2},                                     # counts don't add up
        {"violation_count": 1, "compliant_count": 0},            # violations without a breakdown
        {"report_source": "template"},                           # template needs a reason, no model
        {"report_model": None},                                  # gemini needs a model
        {"conf_threshold": 1.5},
        {"processing_ms": -1},
        {"alert_status": "maybe"},
        {"image_width": 0},
    ],
)
def test_impossible_rows_are_rejected(db, changes):
    with pytest.raises(errors.CheckViolation):
        insert_raw(db, **changes)


@pytest.mark.parametrize("value", ["not a time", "2026-13-45T99:00:00Z"])
def test_created_at_must_be_a_real_time(db, value):
    # SQLite needed CHECK (created_at LIKE '____-__-__T__:__:__Z'); TIMESTAMPTZ refuses non-times itself.
    with pytest.raises(psycopg.DataError):
        insert_raw(db, created_at=value)


def test_missing_required_value_is_rejected(db):
    with pytest.raises(errors.NotNullViolation):
        insert_raw(db, image_name=None)


def test_person_must_be_both_known_or_both_unknown(db):
    analysis_id = save()
    with storage.connect(db) as conn, pytest.raises(errors.CheckViolation):
        conn.execute(
            "INSERT INTO person_verdicts VALUES (%s, 9, NULL, TRUE, 0.9, 0, 0, 10, 10)", (analysis_id,)
        )


def test_person_index_is_unique_within_an_analysis(db):
    analysis_id = save()
    with storage.connect(db) as conn, pytest.raises(errors.UniqueViolation):
        conn.execute("INSERT INTO person_verdicts VALUES (%s, 1, TRUE, TRUE, 0.9, 0, 0, 10, 10)", (analysis_id,))


def test_ids_cannot_be_chosen_by_the_caller(db):
    # GENERATED ALWAYS AS IDENTITY: only the database hands out ids.
    with storage.connect(db) as conn, pytest.raises(errors.GeneratedAlways):
        conn.execute("INSERT INTO digest_runs (id, ran_at, window_start, window_end, status)"
                     " VALUES (1, now(), now() - interval '1 hour', now(), 'sent')")


# --- Transactions and foreign keys ---

def test_failed_save_leaves_nothing_behind(db):
    broken = person(True, True)
    broken.person.x2 = broken.person.x1 - 1  # invalid box: the person insert will fail
    with pytest.raises(errors.CheckViolation):
        save(verdicts=[person(True, True), broken], report=TEMPLATE_REPORT)
    assert storage.list_analyses() == []  # the analysis row was rolled back too
    with storage.connect(db) as conn:
        assert conn.execute("SELECT count(*) AS n FROM person_verdicts").fetchone()["n"] == 0


def test_connection_is_usable_after_a_failed_save(db):
    with pytest.raises(errors.CheckViolation):
        save(verdicts=[person(True, True)], report=Report("Bad", "s", "a", source="gemini", model="m"))
    assert storage.get_analysis(save()) is not None  # the pool discarded nothing it shouldn't have


def test_person_needs_an_existing_analysis(db):
    with storage.connect(db) as conn, pytest.raises(errors.ForeignKeyViolation):
        conn.execute("INSERT INTO person_verdicts VALUES (999, 1, TRUE, TRUE, 0.9, 0, 0, 10, 10)")


def test_deleting_an_analysis_deletes_its_people(db):
    analysis_id = save()
    with storage.connect(db) as conn:
        conn.execute("DELETE FROM analyses WHERE id = %s", (analysis_id,))
        left = conn.execute("SELECT count(*) AS n FROM person_verdicts").fetchone()["n"]
    assert left == 0


# --- Reads: order and time windows ---

def test_list_is_newest_first_with_id_breaking_ties(db):
    a = save(created_at=T0, name="a")
    b = save(created_at=T0 + timedelta(hours=1), name="b")
    c = save(created_at=T0 + timedelta(hours=1), name="c")  # same second as b
    assert [r["id"] for r in storage.list_analyses()] == [c, b, a]
    assert [r["id"] for r in storage.list_analyses(limit=1, offset=1)] == [b]


def test_last_24h_window_edges(db):
    now = T0
    save(created_at=now - timedelta(hours=24, seconds=1), name="too old")
    save(created_at=now - timedelta(hours=24), name="exactly 24h ago")
    save(created_at=now, name="now")
    names = [r["image_name"] for r in storage.get_last_24h(now=now)]
    assert names == ["exactly 24h ago", "now"]


def test_get_since_is_half_open_and_oldest_first(db):
    save(created_at=T0 - timedelta(seconds=1), name="before")
    save(created_at=T0, name="start")
    save(created_at=T0 + timedelta(hours=23, minutes=59, seconds=59), name="last second")
    save(created_at=T0 + timedelta(hours=24), name="end")
    names = [r["image_name"] for r in storage.get_since(T0, T0 + timedelta(hours=24))]
    assert names == ["start", "last second"]


def test_times_in_other_zones_are_the_same_instant(db):
    pkt_time = T0.astimezone(storage.PKT)  # 18:00 PKT is 13:00 UTC
    save(created_at=pkt_time)
    assert storage.get_since(T0, T0 + timedelta(seconds=1))[0]["created_at"] == "2026-11-05T13:00:00Z"


def test_naive_datetimes_are_refused():
    with pytest.raises(ValueError):
        storage.to_db_time(datetime(2026, 11, 5, 13, 0))  # no timezone: ambiguous


# --- Digest runs ---

def test_record_digest_run(db):
    storage.record_digest_run(window_start=T0 - timedelta(hours=24), window_end=T0, status="sent", analyses_count=4)
    storage.record_digest_run(window_start=T0, window_end=T0 + timedelta(hours=24), status="failed",
                              error="HTTP 404: Unknown Webhook")
    newest, oldest = storage.list_digest_runs()
    assert (newest["status"], newest["analyses_count"], newest["error"]) == ("failed", None, "HTTP 404: Unknown Webhook")
    assert (oldest["window_start"], oldest["window_end"], oldest["analyses_count"]) == (
        "2026-11-04T13:00:00Z", "2026-11-05T13:00:00Z", 4
    )


@pytest.mark.parametrize("status, error", [("sent", "x"), ("failed", None), ("maybe", None)])
def test_impossible_digest_runs_are_rejected(db, status, error):
    with pytest.raises(errors.CheckViolation):
        storage.record_digest_run(window_start=T0 - timedelta(hours=1), window_end=T0, status=status, error=error)


def test_digest_window_must_not_be_empty(db):
    with pytest.raises(errors.CheckViolation):
        storage.record_digest_run(window_start=T0, window_end=T0, status="sent")


# --- Configuration, cold starts and an unreachable database ---

def test_no_database_url_is_a_clear_error():
    with pytest.raises(storage.StorageUnavailable, match="DATABASE_URL is not set"):
        storage.list_analyses()


def test_unreachable_database_gives_a_clear_error_without_the_password(monkeypatch):
    monkeypatch.setattr(storage, "CONNECT_TIMEOUT_S", 1)
    monkeypatch.setattr(storage, "POOL_TIMEOUT_S", 2)
    dsn = "postgresql://someone:S3CRET-pw@127.0.0.1:1/incidents"  # nothing listens on port 1
    started = time.perf_counter()
    try:
        with pytest.raises(storage.StorageUnavailable) as caught:
            storage.list_analyses(dsn=dsn)
    finally:
        storage.close_pools()
    message = str(caught.value)
    assert "incident log (127.0.0.1/incidents) could not be reached within 2 s" in message
    assert "S3CRET-pw" not in message and "someone" not in message
    assert time.perf_counter() - started < 10  # bounded: the app never hangs on the log


def test_describe_database_never_shows_credentials():
    assert storage.describe_database("postgresql://u:pw@ep-x.neon.tech/neondb?sslmode=require") == "ep-x.neon.tech/neondb"


def test_first_connect_failing_is_retried(db, monkeypatch):
    """A Neon branch waking up may refuse or time out the first connection; one retry follows."""
    storage.close_pools()  # no warm connections: the next call must connect
    real_connect = psycopg.Connection.connect
    attempts = []

    def flaky(cls, *args, **kwargs):
        attempts.append(time.perf_counter())
        if len(attempts) == 1:
            raise psycopg.OperationalError("timeout expired")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(psycopg.Connection, "connect", classmethod(flaky))
    assert storage.list_analyses() == []
    assert len(attempts) == 2
    assert 0.5 < attempts[1] - attempts[0] < 3  # the pool's first backoff step is about 1 s
