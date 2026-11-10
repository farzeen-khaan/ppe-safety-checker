"""Tests for F8 digest. Each test uses a temporary log (conftest); Gemini and Discord are fakes."""

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from src import alerts, digest, report, storage
from src.detect import Detection
from src.report import Report
from src.violations import PersonVerdict

UNTIL = datetime(2026, 11, 10, 13, 0, 0, tzinfo=timezone.utc)  # 18:00 PKT, the scheduled time
UNTIL_ARG = ["--until", "2026-11-10T13:00:00Z"]
DIGEST_URL = "https://discord.com/api/webhooks/111/DIGEST-secret_token"
ALERTS_URL = "https://discord.com/api/webhooks/222/ALERTS-secret_token"


def person(has_helmet, has_vest) -> PersonVerdict:
    return PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 100, 300), has_helmet, has_vest)


OK, NO_HELMET, NO_VEST, NO_BOTH, FAR = (
    person(True, True), person(False, True), person(True, False), person(False, False), person(None, None)
)


def add(verdicts, before: timedelta, gemini=True, name="site.jpg", alert=None) -> int:
    """Save an analysis `before` the end of the window, with severity from the real rules."""
    severity = report.compute_severity(verdicts)
    stored = (
        Report(severity, "S.", "A.", source="gemini", model="gemini-test") if gemini
        else Report(severity, "S.", "A.", source="template", error="LLM disabled")
    )
    analysis_id = storage.save_analysis(
        image_name=name, image_sha256="ab" * 32, image_width=640, image_height=480,
        verdicts=verdicts, report=stored, model_version="best.pt:test", conf_threshold=0.25,
        processing_ms=100, created_at=UNTIL - before,
    )
    if alert:
        storage.update_alert_status(analysis_id, alert, None if alert == "sent" else "x")
    return analysis_id


def hours(n: float) -> timedelta:
    return timedelta(hours=n)


@pytest.fixture
def a_day():
    """Six analyses inside the window and one outside. Counted by hand below each line."""
    ids = {
        # people 2, compliant 1, violation 1 (helmet), Medium, Gemini, alert sent
        "A": add([OK, NO_HELMET], hours(20), name="a.jpg", alert="sent"),
        # people 3, compliant 1, violation 1 (both), too far 1, High, Gemini, alert failed
        "B": add([NO_BOTH, OK, FAR], hours(10), name="b.jpg", alert="failed"),
        # people 3, violation 3 (helmet, vest, helmet), High, Gemini, alert sent
        "C": add([NO_HELMET, NO_VEST, NO_HELMET], hours(5), name="site_c.jpg", alert="sent"),
        # people 4, compliant 2, too far 2, None, template, no alert needed
        "D": add([OK, OK, FAR, FAR], hours(3), gemini=False, name="d.jpg"),
        # people 0, None, template
        "F": add([], hours(2), gemini=False, name="empty.jpg"),
        # people 2, violation 2 (vest, helmet), Medium, Gemini, alert disabled
        "E": add([NO_VEST, NO_HELMET], hours(1), name="e.jpg", alert="disabled"),
    }
    add([NO_BOTH], hours(25), name="yesterday.jpg")  # before the window: must be ignored
    return ids


def rows(hours_back: int = 24):
    return storage.get_since(UNTIL - timedelta(hours=hours_back), UNTIL)


def no_llm(monkeypatch):
    monkeypatch.setattr(report, "ask_llm", lambda *a, **k: pytest.fail("Gemini must not be called"))


# --- Aggregation ---

def test_aggregation_matches_a_hand_count(a_day):
    s = digest.summarise(rows())
    assert s == {
        "analyses": 6,
        "people_detected": 14, "checked_for_ppe": 11, "compliant": 4, "in_violation": 7, "too_far_to_judge": 3,
        "missing_helmet_only": 4, "missing_vest_only": 2, "missing_helmet_and_vest": 1,
        "severity_high": 2, "severity_medium": 2, "severity_none": 2,
        "reports_gemini": 4, "reports_template": 2,
        "alerts_sent": 2, "alerts_failed": 1, "alerts_disabled": 1, "alerts_pending": 0,
    }


def test_full_message(a_day):
    message = digest.build_digest(UNTIL - hours(24), UNTIL, use_llm=False).message
    lines = message.split("\n")
    assert lines[:7] == [
        "📋 **Daily PPE digest — 10 Nov 2026**",
        "Window: 09 Nov 18:00 – 10 Nov 18:00 PKT",
        "",
        "**6 analyses** · 14 people detected · 11 checked · 🚨 7 in violation · ❔ 3 too far to judge",
        "Violations: 4 missing helmet · 2 missing vest · 1 missing both",
        "Severity: 2 High · 2 Medium · 2 None",
        "",
    ]
    assert lines[7:11] == [
        "**Worst analyses**",
        f"1. 10 Nov 13:00 · site\\_c.jpg · 3 in violation · High (#{a_day['C']})",  # 08:00 UTC shown in PKT
        f"2. 10 Nov 17:00 · e.jpg · 2 in violation · Medium (#{a_day['E']})",
        f"3. 10 Nov 08:00 · b.jpg · 1 in violation · High (#{a_day['B']})",
    ]
    assert "**Overview** (template: Gemini was unavailable)" in message
    assert lines[-1] == "Reliability: alerts 2 sent · 1 failed · 1 disabled | reports 4 by Gemini · 2 template"
    assert "yesterday" not in message


# --- Window ---

def test_window_boundaries():
    second = timedelta(seconds=1)
    add([NO_HELMET], hours(24) + second, name="just outside")
    add([NO_HELMET], hours(24), name="exactly 24 h before")  # start: included
    add([NO_HELMET], hours(24) - second, name="just inside")
    add([NO_HELMET], second, name="a second before the end")
    add([NO_HELMET], timedelta(0), name="exactly at the end")  # end: excluded, it belongs to the next digest
    assert [r["image_name"] for r in rows()] == ["exactly 24 h before", "just inside", "a second before the end"]
    assert digest.build_digest(UNTIL - hours(24), UNTIL, use_llm=False).analyses == 3


def test_hours_option_changes_the_window(monkeypatch, capsys, a_day):
    assert digest.main([*UNTIL_ARG, "--hours", "4", "--dry-run", "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("📋 **PPE digest, last 4 hours — 10 Nov 2026**\nWindow: 10 Nov 14:00 – 10 Nov 18:00 PKT")
    assert "**3 analyses**" in out  # D, F, E; C is 5 h before


@pytest.mark.parametrize("value", ["0", "-3", "abc"])
def test_hours_must_be_positive(value):
    with pytest.raises(SystemExit):
        digest.main(["--hours", value, "--dry-run"])


def test_until_must_have_a_time_zone():
    with pytest.raises(SystemExit):
        digest.main(["--until", "2026-11-10T13:00:00", "--dry-run"])


def test_until_in_pkt_is_converted_to_utc():
    assert digest.parse_until("2026-11-10T18:00:00+05:00") == UNTIL


# --- Top 3 ---

def test_worst_ranks_violations_before_severity(a_day):
    # E (Medium, 2 in violation) beats B (High, 1 in violation); clean analyses never appear.
    assert [r["id"] for r in digest.worst_analyses(rows())] == [a_day["C"], a_day["E"], a_day["B"]]


def test_worst_tie_breaks_by_severity_then_newest():
    medium_new = add([NO_HELMET], hours(1))  # 1 in violation, Medium
    high_old = add([NO_BOTH], hours(9))  # 1 in violation, High
    high_new = add([NO_BOTH], hours(3))  # 1 in violation, High, newer
    medium_old = add([NO_VEST], hours(8))
    assert [r["id"] for r in digest.worst_analyses(rows(), n=4)] == [high_new, high_old, medium_new, medium_old]
    assert len(digest.worst_analyses(rows())) == 3


def test_worst_same_second_uses_the_later_id():
    first = add([NO_HELMET], hours(2))
    second = add([NO_HELMET], hours(2))
    assert [r["id"] for r in digest.worst_analyses(rows())] == [second, first]


# --- Empty window ---

def test_empty_window_message(monkeypatch):
    no_llm(monkeypatch)
    result = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert result.analyses == 0
    assert result.message == (
        "📋 **Daily PPE digest — 10 Nov 2026**\n"
        "Window: 09 Nov 18:00 – 10 Nov 18:00 PKT\n\n"
        "No analyses in the last 24 hours. The digest ran normally; there was nothing to report."
    )


def test_empty_window_is_sent_and_recorded(monkeypatch):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert "No analyses in the last 24 hours" in discord.calls[0]["json"]["content"]
    run = storage.list_digest_runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("sent", 0, None)


# --- Overview ---

def test_gemini_gets_aggregates_only(monkeypatch, a_day):
    sent = {}

    def fake_llm(facts, prompt, schema):
        sent["facts"] = facts
        return schema(overview="Three sentences. About the day. Done."), "gemini-test"

    monkeypatch.setattr(report, "ask_llm", fake_llm)
    message = digest.build_digest(UNTIL - hours(24), UNTIL).message
    assert "**Overview** (Gemini, gemini-test)\nThree sentences. About the day. Done." in message
    facts = sent["facts"]
    assert set(facts) == set(digest.OVERVIEW_FACTS) | {
        "missing_helmet_total", "missing_vest_total", "violation_rate_of_checked_pct"
    }
    assert all(isinstance(v, int) for v in facts.values())  # numbers only: no names, ids or lists
    assert (facts["missing_helmet_total"], facts["missing_vest_total"], facts["violation_rate_of_checked_pct"]) == (5, 3, 64)


@pytest.mark.parametrize("answer", [RuntimeError("503 overloaded"), "", "x" * (digest.MAX_OVERVIEW_CHARS + 1)])
def test_gemini_failure_gives_template_overview(monkeypatch, a_day, answer):
    def fake_llm(facts, prompt, schema):
        if isinstance(answer, Exception):
            raise answer
        return schema(overview=answer), "gemini-test"

    monkeypatch.setattr(report, "ask_llm", fake_llm)
    result = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert result.overview_model is None
    assert "**Overview** (template: Gemini was unavailable)\n6 analyses checked 11 workers for PPE; 7 (64%)" in result.message


def test_gemini_failure_through_the_real_model_chain(monkeypatch, a_day):
    """report.ask_llm itself (not a fake): every model fails, so the digest falls back."""
    tried = []

    def failing_gemini(payload, model, prompt, schema):
        tried.append(model)
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(report, "ask_gemini", failing_gemini)
    monkeypatch.setattr(report, "DEFAULT_MODELS", ["first", "second"])
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    result = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert tried == ["first", "second"] and result.overview_model is None


def test_template_overview_wording(a_day):
    text = digest.template_overview(digest.overview_facts(digest.summarise(rows())))
    assert text == (
        "6 analyses checked 11 workers for PPE; 7 (64%) were in violation. "
        "The most common problem was a missing helmet (5 workers). "
        "3 workers could not be checked because they were too far from the camera. "
        "Tomorrow, prioritise helmet checks and closer photos so every worker can be checked."
    )


def test_template_overview_when_nobody_could_be_checked():
    add([FAR, FAR], hours(2), gemini=False)
    text = digest.template_overview(digest.overview_facts(digest.summarise(rows())))
    assert "none were close enough to check" in text
    assert "compliant" not in text and "None%" not in text


# --- Length and mentions ---

def test_many_analyses_stay_under_discords_limit(monkeypatch):
    # Worst case per part: 200 rows, names that double when markdown-escaped, a maximum-length overview.
    for i in range(200):
        add([NO_BOTH] * 5 + [FAR] * 5, timedelta(minutes=i + 1), name="_" * 300 + f"{i}.jpg", alert="failed")
    monkeypatch.setattr(
        report, "ask_llm", lambda facts, prompt, schema: (schema(overview="y" * digest.MAX_OVERVIEW_CHARS), "m" * 60)
    )
    message = digest.build_digest(UNTIL - hours(24), UNTIL).message
    assert len(message) <= digest.MAX_CONTENT_CHARS
    assert "**200 analyses**" in message
    assert sum(line[:3] in ("1. ", "2. ", "3. ") for line in message.split("\n")) == 3


def test_over_the_limit_fails_instead_of_truncating(monkeypatch, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch)
    monkeypatch.setattr(digest, "MAX_CONTENT_CHARS", 100)
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    assert discord.calls == []  # nothing partial is ever sent
    run = storage.list_digest_runs()[0]
    assert run["status"] == "failed" and "over Discord's 100" in run["error"] and run["analyses_count"] == 6


def test_mentions_never_ping(monkeypatch):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    add([NO_HELMET], hours(1), name="@everyone @here.jpg")
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    payload = discord.calls[0]["json"]
    assert "@everyone" in payload["content"]  # shown as text...
    assert payload["allowed_mentions"] == {"parse": []}  # ...but Discord is told to ping nobody


# --- Sending, exit codes and run records ---

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
        self.urls = []

    def __call__(self, url, **kwargs):
        self.urls.append(url)
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def fake_discord(monkeypatch, *answers) -> FakeDiscord:
    discord = FakeDiscord(*answers)
    monkeypatch.setattr(alerts.requests, "post", discord)
    monkeypatch.setattr(alerts, "sleep", lambda s: discord.calls.append({"slept": s}))
    return discord


def test_sends_one_message_and_records_it(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert len(discord.calls) == 1 and discord.urls == [DIGEST_URL]
    assert discord.calls[0]["json"]["content"].startswith("📋 **Daily PPE digest")
    assert "Digest sent: 6 analyses" in capsys.readouterr().out
    [run] = storage.list_digest_runs()
    assert run["window_start"] == "2026-11-09T13:00:00Z" and run["window_end"] == "2026-11-10T13:00:00Z"
    assert (run["status"], run["analyses_count"], run["error"]) == ("sent", 6, None)


def test_digest_webhook_is_preferred_over_the_alerts_webhook(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", ALERTS_URL)
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    assert digest.digest_webhook_url() == DIGEST_URL


@pytest.mark.parametrize("digest_value", [None, "", "   "])
def test_falls_back_to_the_alerts_webhook(monkeypatch, digest_value):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", ALERTS_URL)
    if digest_value is not None:
        monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", digest_value)
    assert digest.digest_webhook_url() == ALERTS_URL


def test_sends_even_when_alerts_are_disabled(monkeypatch):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)  # ALERTS_ENABLED is false (conftest)
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main(["--no-llm"]) == 0 and len(discord.calls) == 1


def test_dry_run_sends_and_records_nothing(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch)  # any call would fail: no answers queued
    assert digest.main([*UNTIL_ARG, "--dry-run", "--no-llm"]) == 0
    assert discord.calls == []
    assert capsys.readouterr().out.startswith("📋 **Daily PPE digest")
    assert storage.list_digest_runs() == []


def test_dry_run_needs_no_webhook(capsys):
    assert digest.main(["--dry-run", "--no-llm"]) == 0


def test_missing_webhook_fails_before_any_work(monkeypatch, capsys):
    monkeypatch.setattr(digest, "build_digest", lambda *a, **k: pytest.fail("must not build"))
    assert digest.main(UNTIL_ARG) == 1
    assert "DISCORD_DIGEST_WEBHOOK_URL" in capsys.readouterr().err
    run = storage.list_digest_runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("failed", None, digest.NO_WEBHOOK)


def test_send_failure_exits_1_and_is_recorded(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(404, {"message": "Unknown Webhook"}))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    assert len(discord.calls) == 1  # 404: not retried
    assert "Digest not sent: HTTP 404: Unknown Webhook" in capsys.readouterr().err
    run = storage.list_digest_runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("failed", 6, "HTTP 404: Unknown Webhook")


def test_server_error_is_retried_once(monkeypatch, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(503), response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert [c.get("slept") for c in discord.calls] == [None, 2, None]
    assert storage.list_digest_runs()[0]["status"] == "sent"


def test_rate_limit_waits_for_retry_after(monkeypatch, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(429, {"retry_after": 1.5}), response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert discord.calls[1] == {"slept": 1.5}


def test_webhook_url_never_appears_in_stored_errors(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    leak = requests.ConnectionError(f"Max retries exceeded with url: {DIGEST_URL}?wait=true (/api/webhooks/111/DIGEST-secret_token)")
    fake_discord(monkeypatch, leak, leak)
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    error = storage.list_digest_runs()[0]["error"]
    assert error.startswith("ConnectionError")
    assert "DIGEST-secret_token" not in error and DIGEST_URL not in error
    assert "DIGEST-secret_token" not in capsys.readouterr().err


def test_unexpected_send_exception_is_redacted_and_recorded(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)

    def broken(url, content):
        raise RuntimeError(f"bug while posting to {url}")

    monkeypatch.setattr(digest, "send_to_discord", broken)
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    run = storage.list_digest_runs()[0]
    assert run["error"] == "RuntimeError: bug while posting to <webhook>"
    assert "DIGEST-secret_token" not in capsys.readouterr().err


def test_unreadable_log_fails_without_sending(monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch)

    def broken(*a, **k):
        raise OSError("disk unreadable")

    monkeypatch.setattr(digest, "get_since", broken)
    assert digest.main(UNTIL_ARG) == 1
    assert discord.calls == []
    assert "OSError: disk unreadable" in capsys.readouterr().err
    run = storage.list_digest_runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("failed", None, "OSError: disk unreadable")


def test_a_recording_problem_does_not_change_the_outcome(monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    fake_discord(monkeypatch, response(200))

    def broken(**kwargs):
        raise OSError("read-only database")

    monkeypatch.setattr(digest, "record_digest_run", broken)
    assert digest.main(["--no-llm"]) == 0  # the digest was delivered; re-running would post it twice
    assert "could not record the digest run" in capsys.readouterr().err
