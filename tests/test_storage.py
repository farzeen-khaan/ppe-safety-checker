"""Tests for F5 storage. Each test uses its own temporary database file."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src import storage
from src.detect import Detection
from src.report import Report
from src.violations import PersonVerdict

T0 = datetime(2026, 11, 5, 13, 0, 0, tzinfo=timezone.utc)


def person(has_helmet, has_vest, x1=0.0) -> PersonVerdict:
    box = Detection(5, "Person", 0.9, x1, 0.0, x1 + 100, 300.0)
    return PersonVerdict(person=box, has_helmet=has_helmet, has_vest=has_vest)


GEMINI_REPORT = Report("High", "Summary.", "Action.", source="gemini", model="gemini-test")
TEMPLATE_REPORT = Report("None", "Summary.", "Action.", source="template", error="LLM disabled")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    storage.init_db(path)
    return path


def save(db, verdicts=None, report=GEMINI_REPORT, created_at=T0, name="photo.jpg") -> int:
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
        path=db,
    )


# --- Schema and migrations ---

def test_init_is_idempotent_and_records_migrations(db):
    storage.init_db(db)  # second call must not fail or re-apply anything
    with storage.connect(db) as conn:
        versions = [r["version"] for r in conn.execute("SELECT version FROM schema_migrations")]
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert versions == [1]
    assert {"analyses", "person_verdicts", "schema_migrations"} <= tables


def test_wal_mode_is_on(db):
    with storage.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


# --- Round trip ---

def test_save_and_read_back(db):
    analysis_id = save(db)
    a = storage.get_analysis(analysis_id, path=db)

    assert a["created_at"] == "2026-11-05T13:00:00Z"
    assert (a["people_count"], a["compliant_count"], a["violation_count"], a["too_far_count"]) == (3, 1, 1, 1)
    assert a["missing_both_count"] == 1
    assert (a["report_source"], a["report_model"], a["report_error"]) == ("gemini", "gemini-test", None)
    assert a["min_person_height"] == 0.15
    assert a["model_version"] == "best.pt:test"

    statuses = [p["status"] for p in a["people"]]
    assert statuses == ["compliant", "missing helmet and vest", "too far to judge"]
    too_far = a["people"][2]
    assert too_far["has_helmet"] is None and too_far["has_vest"] is None  # unknown stays unknown


def test_template_report_keeps_its_reason(db):
    a = storage.get_analysis(save(db, verdicts=[person(True, True)], report=TEMPLATE_REPORT), path=db)
    assert (a["report_source"], a["report_model"], a["report_error"]) == ("template", None, "LLM disabled")


def test_photo_with_no_people(db):
    a = storage.get_analysis(save(db, verdicts=[], report=TEMPLATE_REPORT), path=db)
    assert a["people_count"] == 0 and a["people"] == []


def test_unknown_id_returns_none(db):
    assert storage.get_analysis(999, path=db) is None


# --- Alert status ---

def test_alert_starts_pending_only_when_there_are_violations(db):
    with_violation = storage.get_analysis(save(db), path=db)
    without = storage.get_analysis(save(db, verdicts=[person(True, True)], report=TEMPLATE_REPORT), path=db)
    assert with_violation["alert_status"] == "pending"
    assert without["alert_status"] == "not_needed"


def test_update_alert_status(db):
    analysis_id = save(db)
    storage.update_alert_status(analysis_id, "failed", error="timeout", path=db)
    a = storage.get_analysis(analysis_id, path=db)
    assert (a["alert_status"], a["alert_error"]) == ("failed", "timeout")
    assert a["alert_updated_at"] is not None


def test_update_alert_status_rejects_unknown_id_and_status(db):
    with pytest.raises(KeyError):
        storage.update_alert_status(999, "sent", path=db)
    with pytest.raises(sqlite3.IntegrityError):
        storage.update_alert_status(save(db), "delivered", path=db)  # not an allowed status


# --- The database rejects impossible data (CHECK constraints) ---

def insert_raw(db, **changes):
    """Insert a valid analyses row directly with SQL, with some columns changed."""
    row = {
        "created_at": "2026-11-05T13:00:00Z", "image_name": "x.jpg", "image_sha256": "ab",
        "image_width": 10, "image_height": 10, "model_version": "m", "conf_threshold": 0.25,
        "min_person_height": 0.15, "people_count": 1, "compliant_count": 1, "violation_count": 0,
        "too_far_count": 0, "missing_helmet_only_count": 0, "missing_vest_only_count": 0,
        "missing_both_count": 0, "severity": "None", "report_summary": "s", "report_action": "a",
        "report_source": "gemini", "report_model": "m", "report_error": None,
        "alert_status": "not_needed", "processing_ms": 1,
    } | changes
    with storage.connect(db) as conn:
        conn.execute(
            f"INSERT INTO analyses ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})", row
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
        {"created_at": "2026-11-05 13:00:00"},                   # not the fixed UTC format
        {"conf_threshold": 1.5},
        {"processing_ms": -1},
        {"alert_status": "maybe"},
    ],
)
def test_impossible_rows_are_rejected(db, changes):
    with pytest.raises(sqlite3.IntegrityError):
        insert_raw(db, **changes)


def test_person_must_be_both_known_or_both_unknown(db):
    analysis_id = save(db)
    with storage.connect(db) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO person_verdicts VALUES (?, 9, NULL, 1, 0.9, 0, 0, 10, 10)", (analysis_id,)
        )


# --- Transactions and foreign keys ---

def test_failed_save_leaves_nothing_behind(db):
    broken = person(True, True)
    broken.person.x2 = broken.person.x1 - 1  # invalid box: the person insert will fail
    with pytest.raises(sqlite3.IntegrityError):
        save(db, verdicts=[person(True, True), broken], report=TEMPLATE_REPORT)
    assert storage.list_analyses(path=db) == []  # the analysis row was rolled back too


def test_person_needs_an_existing_analysis(db):
    with storage.connect(db) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO person_verdicts VALUES (999, 1, 1, 1, 0.9, 0, 0, 10, 10)")


def test_deleting_an_analysis_deletes_its_people(db):
    analysis_id = save(db)
    with storage.connect(db) as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        left = conn.execute("SELECT COUNT(*) FROM person_verdicts").fetchone()[0]
    assert left == 0


# --- Reads: order and time windows ---

def test_list_is_newest_first_with_id_breaking_ties(db):
    a = save(db, created_at=T0, name="a")
    b = save(db, created_at=T0 + timedelta(hours=1), name="b")
    c = save(db, created_at=T0 + timedelta(hours=1), name="c")  # same second as b
    assert [r["id"] for r in storage.list_analyses(path=db)] == [c, b, a]
    assert [r["id"] for r in storage.list_analyses(limit=1, offset=1, path=db)] == [b]


def test_last_24h_window_edges(db):
    now = T0
    save(db, created_at=now - timedelta(hours=24, seconds=1), name="too old")
    save(db, created_at=now - timedelta(hours=24), name="exactly 24h ago")
    save(db, created_at=now, name="now")
    names = [r["image_name"] for r in storage.get_last_24h(now=now, path=db)]
    assert names == ["exactly 24h ago", "now"]


def test_naive_datetimes_are_refused():
    with pytest.raises(ValueError):
        storage.to_db_time(datetime(2026, 11, 5, 13, 0))  # no timezone: ambiguous
