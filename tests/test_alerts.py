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
from src.verdicts import Detection
from src.report import Report
from src.violations import PersonVerdict

FAKE_URL = "https://discord.com/api/webhooks/123456789/SECRET-token_abc"
T0 = datetime(2026, 11, 5, 13, 4, 0, tzinfo=timezone.utc)  # 18:04 PKT
IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)
REPORT = Report("High", "Two workers lack PPE.", "Stop work.", source="llm", provider="acme", model="model-x")


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
    status = alerts.alert_after_save(analysis_id, verdicts, REPORT, IMAGE, "site.jpg", T0)  # synchronous
    final = status if status in ("sent", "failed") else None
    return status, final, stored(analysis_id)


# --- When to alert ---

def test_violation_sends_image_and_message(monkeypatch, enabled):
    fake = fake_discord(monkeypatch, response(200))
    status, final, row = run_alert(VIOLATION)

    assert (status, final, row["alert_status"], row["alert_error"]) == ("sent", "sent", "sent", None)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["files"]["files[0]"][1][:2] == b"\xff\xd8"  # the annotated image, as JPEG bytes
    payload = json.loads(call["data"]["payload_json"])
    assert payload["allowed_mentions"] == {"parse": []}  # never pings anyone
    assert "High severity" in payload["content"]
    embed = payload["embeds"][0]
    assert embed["color"] == alerts.COLOURS["High"]
    assert embed["image"]["url"] == "attachment://ppe_alert.jpg"  # the uploaded file, shown inside the embed


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

def fields(message) -> dict:
    return {f["name"]: f["value"] for f in message.embed["fields"]}


def test_alert_embed():
    m = alerts.build_alert(VIOLATION, REPORT, 7, "site_1.jpg", T0)
    assert m.content == "🚨 PPE violation — High severity: 2 of 3 workers checked missing PPE"
    e = m.embed
    assert (e["title"], e["color"], e["description"]) == ("PPE violation — High severity", 0xD93025, "Two workers lack PPE.")
    assert fields(m) == {
        "In violation": "2 of 3 workers checked",
        "Missing": "1 missing a helmet, 1 missing both helmet and vest",
        "Not checked": "1 too far from the camera",
        "Recommended action": "Stop work.",
        "Photo": r"site\_1.jpg",  # markdown-escaped
        "Analysis": "#7",
    }
    assert e["footer"]["text"] == "Automated summary, check against the figures above · 05 Nov 2026, 18:04 PKT"
    assert e["timestamp"] == "2026-11-05T13:04:00+00:00"


def test_no_vendor_or_model_name_reaches_the_user():
    m = alerts.build_alert(VIOLATION, REPORT, 7, "a.jpg", T0)
    text = json.dumps({"content": m.content, "embed": m.embed}).lower()
    for hidden in ("acme", "model-x", "llm", "gemini"):
        assert hidden not in text


def test_standard_summary_is_labelled_as_such():
    standard = Report("High", "S.", "A.", source="template", error="LLM disabled")
    m = alerts.build_alert(VIOLATION, standard, 7, "a.jpg", T0)
    assert m.embed["footer"]["text"].startswith("Standard summary, built from the figures above")


def test_colour_follows_severity():
    for severity in ("High", "Medium", "None"):
        r = Report(severity, "S.", "A.", source="template", error="x")
        assert alerts.build_alert(VIOLATION, r, 1, "a.jpg", T0).embed["color"] == alerts.COLOURS[severity]


def test_no_not_checked_field_without_too_far_workers():
    m = alerts.build_alert([person(False, True)], REPORT, 7, "a.jpg", T0)
    assert "Not checked" not in fields(m)
    assert fields(m)["In violation"] == "1 of 1 worker checked"


def test_long_photo_names_keep_their_start_and_extension():
    m = alerts.build_alert(VIOLATION, REPORT, 7, "source-56088ecf75821c1b93f4015a2bc0a071_extra_long.jpg", T0)
    photo = fields(m)["Photo"]
    assert photo.startswith("source-56088ecf") and photo.endswith("ra\\_long.jpg") and "…" in photo
    assert len(photo.replace("\\", "")) <= 40  # the limit applies before markdown escaping


def test_alert_stays_within_discords_embed_limits():
    long_report = Report("High", "x " * 3000, "y " * 3000, source="llm", provider="acme", model="m")
    m = alerts.build_alert(VIOLATION, long_report, 7, "n" * 500 + ".jpg", T0)
    assert alerts.embed_problems(m.content, [m.embed]) == []
    assert m.embed["description"].endswith("…") and fields(m)["Recommended action"].endswith("…")


def test_an_embed_that_breaks_the_limits_is_a_recorded_failure_not_a_send(monkeypatch, enabled):
    fake = fake_discord(monkeypatch)
    bad = alerts.AlertMessage("c", {"title": "t" * 300})
    analysis_id = saved(VIOLATION)
    assert alerts.deliver(FAKE_URL, analysis_id, bad, IMAGE) == "failed"
    assert fake.calls == [] and "embed title is 300" in stored(analysis_id)["alert_error"]


def test_large_images_are_downscaled():
    jpg = alerts.encode_image(np.zeros((4000, 3000, 3), dtype=np.uint8))
    decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == alerts.MAX_IMAGE_SIDE


# --- Pipeline integration (synchronous: the alert is done when analyze() returns) ---

@pytest.fixture
def inspected(monkeypatch):
    """analyze() without the model: inspect_image returns the given verdicts."""
    def use(verdicts):
        monkeypatch.setattr(pipeline, "inspect_image", lambda image, conf=0.25: pipeline.Inspection(
            verdicts, IMAGE, 200, 100, "ab" * 32, 0.25, started=0.0))
    return use


def test_analyze_sends_the_alert_before_returning(monkeypatch, enabled, inspected):
    fake_discord(monkeypatch, response(200))
    inspected(VIOLATION)
    result = pipeline.analyze(IMAGE, "site.jpg")
    assert result.alert_status == "sent"
    assert stored(result.analysis_id)["alert_status"] == "sent"


def test_the_alert_carries_the_rule_based_facts(monkeypatch, enabled, inspected):
    """The written summary doesn't exist yet when the alert goes out: the alert has the standard one."""
    fake = fake_discord(monkeypatch, response(200))
    inspected(VIOLATION)
    pipeline.analyze(IMAGE, "site.jpg")
    embed = json.loads(fake.calls[0]["data"]["payload_json"])["embeds"][0]
    assert embed["description"].startswith("The flagged workers are missing")
    assert embed["footer"]["text"].startswith("Standard summary, built from the figures above")


def test_alert_outcomes_reach_the_real_log(db, monkeypatch, enabled, waits, inspected):
    """Against Postgres: pending at the save, then sent or failed, all inside analyze()."""
    fake_discord(monkeypatch, response(200), response(404, {"message": "Unknown Webhook"}))
    inspected(VIOLATION)
    sent = pipeline.analyze(IMAGE, "site.jpg")
    failed = pipeline.analyze(IMAGE, "site.jpg")
    assert (sent.alert_status, failed.alert_status) == ("sent", "failed")
    rows = {r["id"]: r for r in storage.list_analyses()}
    assert rows[sent.analysis_id]["alert_status"] == "sent" and rows[sent.analysis_id]["alert_updated_at"]
    assert (rows[failed.analysis_id]["alert_status"], rows[failed.analysis_id]["alert_error"]) == (
        "failed", "HTTP 404: Unknown Webhook"
    )


def test_pipeline_still_alerts_when_the_save_fails(monkeypatch, enabled, inspected):
    fake = fake_discord(monkeypatch, response(200))

    def broken_save(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "save_analysis", broken_save)
    inspected(VIOLATION)
    result = pipeline.analyze(IMAGE, "site.jpg")
    assert result.save_error and result.alert_status == "sent"
    fields = json.loads(fake.calls[0]["data"]["payload_json"])["embeds"][0]["fields"]
    assert {"name": "Analysis", "value": "Not saved to the log", "inline": True} in fields


def test_dry_run_never_alerts(monkeypatch, enabled, inspected):
    fake = fake_discord(monkeypatch)
    inspected(VIOLATION)
    result = pipeline.analyze(IMAGE, "site.jpg", save=False)
    assert result.alert_status == "not_needed" and fake.calls == []


def test_no_background_threads():
    import src.alerts as module
    assert not hasattr(module, "_executor")
