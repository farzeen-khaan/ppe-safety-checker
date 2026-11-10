"""Tests for F7 alerts. The HTTP call is replaced with a fake: nothing reaches Discord.

The log is an in-memory fake (AlertLog) unless a test takes the `db` fixture, which uses
the real log on the Neon test branch.
"""

import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import requests

from src import alerts, discord_webhook, pipeline, storage
from src.detect import Detection
from src.report import Report
from src.violations import PersonVerdict

FAKE_URL = "https://discord.com/api/webhooks/123456789/SECRET-token_abc"
T0 = datetime(2026, 11, 5, 13, 4, 0, tzinfo=timezone.utc)  # 18:04 PKT
IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)
REPORT = Report("High", "Two workers lack PPE.", "Stop work.", source="gemini", model="gemini-test")


def person(has_helmet, has_vest) -> PersonVerdict:
    return PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 100, 300), has_helmet, has_vest)


VIOLATION = [person(False, True), person(False, False), person(True, True), person(None, None)]
COMPLIANT = [person(True, True)]
TOO_FAR_ONLY = [person(True, True), person(None, None), person(None, None)]


def response(status: int, body: dict | None = None, headers: dict | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body or {}).encode()
    r.headers.update(headers or {})
    return r


class FakeDiscord:
    """Stands in for requests.post: returns the queued answers in order and records each call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", FAKE_URL)


@pytest.fixture
def waits(monkeypatch):
    """Record retry waits instead of really sleeping."""
    slept = []
    monkeypatch.setattr(discord_webhook, "sleep", slept.append)
    return slept


def fake_discord(monkeypatch, *answers) -> FakeDiscord:
    fake = FakeDiscord(*answers)
    monkeypatch.setattr(discord_webhook.requests, "post", fake)
    return fake


class AlertLog:
    """In-memory stand-in for save_analysis and update_alert_status (rows built by storage.analysis_row)."""

    def __init__(self):
        self.rows: dict[int, dict] = {}

    def save_analysis(self, **fields) -> int:
        analysis_id = len(self.rows) + 1
        self.rows[analysis_id] = {**storage.analysis_row(**fields), "alert_error": None}
        return analysis_id

    def update_alert_status(self, analysis_id, status, error=None):
        if analysis_id not in self.rows:
            raise KeyError(f"no analysis with id {analysis_id}")
        self.rows[analysis_id].update(alert_status=status, alert_error=error)


_log: AlertLog | None = None  # None: the test uses the real log (db fixture)


@pytest.fixture(autouse=True)
def alert_log(request, monkeypatch):
    global _log
    _log = None if "db" in request.fixturenames else AlertLog()
    if _log is not None:
        monkeypatch.setattr(alerts, "update_alert_status", _log.update_alert_status)
        monkeypatch.setattr(pipeline, "save_analysis", _log.save_analysis)
    yield _log
    _log = None


def stored(analysis_id: int) -> dict:
    return _log.rows[analysis_id] if _log is not None else storage.get_analysis(analysis_id)


def saved(verdicts) -> int:
    report = REPORT if any(v.is_violation for v in verdicts) else Report(
        "None", "All fine.", "None.", source="template", error="LLM disabled"
    )
    fields = dict(
        image_name="site.jpg", image_sha256="ab" * 32, image_width=200, image_height=100,
        verdicts=verdicts, report=report, model_version="best.pt:test", conf_threshold=0.25,
        processing_ms=100, created_at=T0,
    )
    return _log.save_analysis(**fields) if _log is not None else storage.save_analysis(**fields)


def run_alert(verdicts):
    analysis_id = saved(verdicts)
    status, future = alerts.alert_after_save(analysis_id, verdicts, REPORT, IMAGE, "site.jpg", T0)
    final = future.result(timeout=10) if future else None
    return status, final, stored(analysis_id)


# --- When to alert ---

def test_violation_sends_image_and_message(monkeypatch, enabled):
    fake = fake_discord(monkeypatch, response(200))
    status, final, row = run_alert(VIOLATION)

    assert (status, final, row["alert_status"], row["alert_error"]) == ("sending", "sent", "sent", None)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["files"]["files[0]"][1][:2] == b"\xff\xd8"  # the annotated image, as JPEG bytes
    payload = json.loads(call["data"]["payload_json"])
    assert payload["allowed_mentions"] == {"parse": []}  # never pings anyone
    assert "High severity" in payload["content"]


def test_compliant_photo_does_not_alert(monkeypatch, enabled):
    fake = fake_discord(monkeypatch)
    status, final, row = run_alert(COMPLIANT)
    assert (status, final, row["alert_status"]) == ("not_needed", None, "not_needed")
    assert fake.calls == []


def test_too_far_only_does_not_alert(monkeypatch, enabled):
    fake = fake_discord(monkeypatch)
    status, _, row = run_alert(TOO_FAR_ONLY)
    assert (status, row["alert_status"]) == ("not_needed", "not_needed")
    assert fake.calls == []


@pytest.mark.parametrize(
    "env, reason",
    [
        ({"ALERTS_ENABLED": "false", "DISCORD_WEBHOOK_URL": FAKE_URL}, "ALERTS_ENABLED is not true"),
        ({"ALERTS_ENABLED": "", "DISCORD_WEBHOOK_URL": FAKE_URL}, "ALERTS_ENABLED is not true"),
        ({"ALERTS_ENABLED": "true", "DISCORD_WEBHOOK_URL": ""}, "DISCORD_WEBHOOK_URL is not set"),
    ],
)
def test_disabled_is_skipped_and_recorded(monkeypatch, env, reason):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fake = fake_discord(monkeypatch)
    status, _, row = run_alert(VIOLATION)
    assert (status, row["alert_status"], row["alert_error"]) == ("disabled", "disabled", reason)
    assert fake.calls == []


def test_enabled_accepts_any_case(monkeypatch, enabled):
    monkeypatch.setenv("ALERTS_ENABLED", " True ")
    assert alerts.webhook_config() == (FAKE_URL, None)


# --- Retries and failures ---

def test_server_error_is_retried_once_then_recorded(monkeypatch, enabled, waits):
    fake = fake_discord(monkeypatch, response(500, {"message": "oops"}), response(503))
    _, final, row = run_alert(VIOLATION)
    assert (final, row["alert_status"]) == ("failed", "failed")
    assert row["alert_error"] == "HTTP 503"
    assert len(fake.calls) == 2 and waits == [alerts.RETRY_DELAY_S]


def test_retry_can_succeed(monkeypatch, enabled, waits):
    fake_discord(monkeypatch, requests.ConnectionError("no route"), response(200))
    assert run_alert(VIOLATION)[1] == "sent"


def test_wrong_webhook_is_not_retried(monkeypatch, enabled, waits):
    fake = fake_discord(monkeypatch, response(404, {"message": "Unknown Webhook", "code": 10015}))
    _, final, row = run_alert(VIOLATION)
    assert (final, row["alert_error"]) == ("failed", "HTTP 404: Unknown Webhook")
    assert len(fake.calls) == 1 and waits == []


def test_429_waits_for_discords_retry_after(monkeypatch, enabled, waits):
    fake = fake_discord(monkeypatch, response(429, {"retry_after": 1.5, "global": False}), response(200))
    assert run_alert(VIOLATION)[1] == "sent"
    assert waits == [1.5] and len(fake.calls) == 2


def test_429_falls_back_to_the_header(monkeypatch, enabled, waits):
    fake_discord(monkeypatch, response(429, headers={"Retry-After": "3"}), response(200))
    run_alert(VIOLATION)
    assert waits == [3.0]


def test_429_with_a_long_wait_fails_instead_of_waiting(monkeypatch, enabled, waits):
    fake = fake_discord(monkeypatch, response(429, {"retry_after": 120}))
    _, final, row = run_alert(VIOLATION)
    assert final == "failed" and "not retried" in row["alert_error"]
    assert waits == [] and len(fake.calls) == 1


def test_unexpected_error_is_recorded_not_raised(monkeypatch, enabled):
    fake_discord(monkeypatch, RuntimeError("bug"))  # not a requests error
    _, final, row = run_alert(VIOLATION)
    assert (final, row["alert_error"]) == ("failed", "RuntimeError: bug")


# --- The webhook URL is a secret ---

def test_url_never_appears_in_the_stored_error(monkeypatch, enabled, waits):
    # urllib3 errors quote the full URL or its path; both contain the token.
    leak = requests.ConnectionError(
        f"HTTPSConnectionPool(host='discord.com', port=443): Max retries exceeded with url: "
        f"/api/webhooks/123456789/SECRET-token_abc?wait=true (Caused by {FAKE_URL})"
    )
    fake_discord(monkeypatch, leak, leak)
    _, final, row = run_alert(VIOLATION)
    assert final == "failed"
    assert "SECRET-token_abc" not in row["alert_error"] and "123456789" not in row["alert_error"]
    assert "ConnectionError" in row["alert_error"]  # still useful for debugging


def test_redact_handles_other_hosts():
    text = "failed: https://discordapp.com/api/webhooks/42/abc-DEF_1"
    assert alerts.redact(text) == "failed: https://discordapp.com/api/webhooks/<redacted>"


# --- The message ---

def test_message_contents():
    text = alerts.build_message(VIOLATION, REPORT, 7, "site_1.jpg", T0)
    assert "PPE violation — High severity" in text
    assert "2 of 3 workers checked in violation:** 1 missing helmet, 1 missing helmet and vest." in text
    assert "1 person could not be checked" in text
    assert "Two workers lack PPE." in text
    assert "Analysis #7 · site\\_1.jpg · 05 Nov 2026, 18:04 PKT" in text


def test_message_without_too_far_workers_has_no_such_line():
    text = alerts.build_message([person(False, True)], REPORT, 7, "a.jpg", T0)
    assert "could not be checked" not in text
    assert "1 of 1 worker checked in violation:** 1 missing helmet." in text


def test_message_stays_within_discords_limit():
    long_report = Report("High", "x " * 3000, "Stop.", source="gemini", model="m")
    text = alerts.build_message(VIOLATION, long_report, 7, "n" * 500 + ".jpg", T0)
    assert len(text) <= alerts.MAX_CONTENT_CHARS
    assert text.endswith("PKT") and "…" in text  # summary shortened, footer kept


def test_large_images_are_downscaled():
    jpg = alerts.encode_image(np.zeros((4000, 3000, 3), dtype=np.uint8))
    decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == alerts.MAX_IMAGE_SIDE


# --- Pipeline integration ---

def inspection(verdicts) -> pipeline.Inspection:
    return pipeline.Inspection(verdicts, IMAGE, 200, 100, "ab" * 32, 0.25, started=0.0)


def test_pipeline_starts_alert_after_save(monkeypatch, enabled):
    fake_discord(monkeypatch, response(200))
    result = pipeline.finish_analysis(inspection(VIOLATION), "site.jpg", use_llm=False)
    assert result.alert_status == "sending" and result.alert.result(timeout=10) == "sent"
    assert stored(result.analysis_id)["alert_status"] == "sent"


def test_alert_outcomes_reach_the_real_log(db, monkeypatch, enabled, waits):
    """Against Postgres: pending after the save, then sent or failed from the alert thread."""
    fake_discord(monkeypatch, response(200), response(404, {"message": "Unknown Webhook"}))
    sent = pipeline.finish_analysis(inspection(VIOLATION), "site.jpg", use_llm=False)
    assert sent.alert.result(timeout=30) == "sent"
    failed = pipeline.finish_analysis(inspection(VIOLATION), "site.jpg", use_llm=False)
    assert failed.alert.result(timeout=30) == "failed"
    rows = {r["id"]: r for r in storage.list_analyses()}
    assert rows[sent.analysis_id]["alert_status"] == "sent" and rows[sent.analysis_id]["alert_updated_at"]
    assert (rows[failed.analysis_id]["alert_status"], rows[failed.analysis_id]["alert_error"]) == (
        "failed", "HTTP 404: Unknown Webhook"
    )


def test_pipeline_still_alerts_when_the_save_fails(monkeypatch, enabled):
    fake = fake_discord(monkeypatch, response(200))

    def broken_save(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "save_analysis", broken_save)
    result = pipeline.finish_analysis(inspection(VIOLATION), "site.jpg", use_llm=False)
    assert result.save_error and result.alert.result(timeout=10) == "sent"
    assert "Analysis not saved to the log" in json.loads(fake.calls[0]["data"]["payload_json"])["content"]


def test_dry_run_never_alerts(monkeypatch, enabled):
    fake = fake_discord(monkeypatch)
    result = pipeline.finish_analysis(inspection(VIOLATION), "site.jpg", use_llm=False, save=False)
    assert (result.alert_status, result.alert) == ("not_needed", None) and fake.calls == []
