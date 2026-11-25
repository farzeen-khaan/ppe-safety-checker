"""Tests for F8 digest. Gemini and Discord are always fakes.

Most tests run offline against FakeLog, an in-memory stand-in for the two storage functions
the digest uses (get_since, record_digest_run). Its rows are built by storage.analysis_row,
the same code save_analysis uses, so their columns and counts match the real log.
Tests that take the `db` fixture use the real log on the Neon test branch instead.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from src import digest, discord_webhook, llm, report, storage
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


class FakeLog:
    """Stands in for storage.get_since and storage.record_digest_run, in memory."""

    def __init__(self):
        self.rows: list[dict] = []
        self.runs: list[dict] = []  # newest first, like storage.list_digest_runs

    def add(self, fields: dict, alert: str | None) -> int:
        row = {"id": len(self.rows) + 1, **fields, "alert_error": None, "alert_updated_at": None}
        row["created_at"] = storage.to_db_time(row["created_at"])
        if alert:
            row["alert_status"], row["alert_error"] = alert, None if alert == "sent" else "x"
        self.rows.append(row)
        return row["id"]

    def get_since(self, since, until):
        start, end = storage.to_db_time(since), storage.to_db_time(until)
        found = [r for r in self.rows if start <= r["created_at"] < end]
        return sorted(found, key=lambda r: (r["created_at"], r["id"]))

    def record_digest_run(self, *, window_start, window_end, status, analyses_count=None, error=None):
        self.runs.insert(0, {
            "window_start": storage.to_db_time(window_start), "window_end": storage.to_db_time(window_end),
            "status": status, "analyses_count": analyses_count, "error": error,
        })


_log: FakeLog | None = None  # None: the test uses the real log (db fixture)


@pytest.fixture(autouse=True)
def log(request, monkeypatch):
    global _log
    if "db" in request.fixturenames:
        _log = None
        yield None
        return
    _log = FakeLog()
    monkeypatch.setattr(digest, "get_since", _log.get_since)
    monkeypatch.setattr(digest, "record_digest_run", _log.record_digest_run)
    yield _log
    _log = None


def add(verdicts, before: timedelta, gemini=True, name="site.jpg", alert=None) -> int:
    """Add an analysis `before` the end of the window, with severity from the real rules."""
    severity = report.compute_severity(verdicts)
    stored = (
        Report(severity, "S.", "A.", source="llm", provider="acme", model="model-x") if gemini
        else Report(severity, "S.", "A.", source="template", error="LLM disabled")
    )
    fields = dict(
        image_name=name, image_sha256="ab" * 32, image_width=640, image_height=480,
        verdicts=verdicts, report=stored, model_version="best.pt:test", conf_threshold=0.25,
        processing_ms=100, created_at=UNTIL - before,
    )
    if _log is not None:
        return _log.add(storage.analysis_row(**fields), alert)
    analysis_id = storage.save_analysis(**fields)
    if alert:
        storage.update_alert_status(analysis_id, alert, None if alert == "sent" else "x")
    return analysis_id


def runs() -> list[dict]:
    return _log.runs if _log is not None else storage.list_digest_runs()


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
    return digest.get_since(UNTIL - timedelta(hours=hours_back), UNTIL)


def no_llm(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: pytest.fail("the model must not be called"))


def model_says(text, model="model-x"):
    return lambda prompt, facts, schema: llm.Generated(schema(summary=text), "acme", model)


def fields(d) -> dict:
    return {f["name"]: f["value"] for f in d.embed["fields"]}


# --- Aggregation ---

def test_aggregation_matches_a_hand_count(a_day):
    s = digest.summarise(rows())
    assert s == {
        "analyses": 6,
        "people_detected": 14, "checked_for_ppe": 11, "compliant": 4, "in_violation": 7, "too_far_to_judge": 3,
        "missing_helmet_only": 4, "missing_vest_only": 2, "missing_helmet_and_vest": 1,
        "severity_high": 2, "severity_medium": 2, "severity_none": 2,
        "summaries_automated": 4, "summaries_standard": 2,
        "alerts_sent": 2, "alerts_failed": 1, "alerts_disabled": 1, "alerts_pending": 0,
    }


def test_full_embed(a_day):
    d = digest.build_digest(UNTIL - hours(24), UNTIL, use_llm=False)
    assert d.content == "📋 Daily PPE digest — 10 Nov 2026: 6 analyses, 7 workers in violation"
    e = d.embed
    assert (e["title"], e["description"]) == ("Daily PPE digest — 10 Nov 2026", "Window: 09 Nov 18:00 – 10 Nov 18:00 PKT")
    assert e["color"] == digest.COLOURS["High"]  # the worst severity in the window
    assert fields(d) == {
        "Analyses": "6", "People detected": "14", "Checked": "11", "In violation": "7", "Not checked (too far)": "3",
        "Severity": "2 High · 2 Medium · 2 None",
        "Violations": "4 missing helmet · 2 missing vest · 1 missing both",
        "Worst analyses": "\n".join([
            f"1. 10 Nov 13:00 · site\\_c.jpg · 3 in violation · High (#{a_day['C']})",  # 08:00 UTC shown in PKT
            f"2. 10 Nov 17:00 · e.jpg · 2 in violation · Medium (#{a_day['E']})",
            f"3. 10 Nov 08:00 · b.jpg · 1 in violation · High (#{a_day['B']})",
        ]),
        "Summary": fields(d)["Summary"],
        "Reliability": "Alerts: 2 sent · 1 failed · 1 not sent (alerts off)\nSummaries: 4 automated, 2 standard",
    }
    assert e["footer"]["text"] == "Standard summary, built from the figures above · 10 Nov 2026, 18:00 PKT"
    assert e["timestamp"] == "2026-11-10T13:00:00+00:00"
    assert "yesterday" not in json.dumps(e)


def test_colour_is_the_worst_severity_present():
    add([NO_HELMET], hours(1))
    assert digest.build_digest(UNTIL - hours(24), UNTIL, use_llm=False).embed["color"] == digest.COLOURS["Medium"]
    add([OK], hours(2))
    add([NO_BOTH], hours(3))
    assert digest.build_digest(UNTIL - hours(24), UNTIL, use_llm=False).embed["color"] == digest.COLOURS["High"]


def test_no_vendor_or_model_name_reaches_the_user(monkeypatch, a_day):
    monkeypatch.setattr(llm, "generate", model_says("Three short sentences.", model="model-x"))
    d = digest.build_digest(UNTIL - hours(24), UNTIL)
    text = (d.content + json.dumps(d.embed)).lower()
    for hidden in ("acme", "model-x", "gemini", "llm"):
        assert hidden not in text
    assert d.embed["footer"]["text"].startswith("Automated summary, check against the figures above")


# --- Window ---

def test_window_boundaries(db):
    """Against the real log: the boundary is decided by Postgres comparing TIMESTAMPTZ values."""
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
    assert "**PPE digest, last 4 hours — 10 Nov 2026**\nWindow: 10 Nov 14:00 – 10 Nov 18:00 PKT" in out
    assert "**Analyses**\n3\n" in out  # D, F, E; C is 5 h before


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
    d = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert (d.analyses, d.content) == (0, "📋 Daily PPE digest — 10 Nov 2026: no analyses in the last 24 hours")
    assert d.embed["description"] == (
        "Window: 09 Nov 18:00 – 10 Nov 18:00 PKT\n\n"
        "No analyses in the last 24 hours. The digest ran normally; there was nothing to report."
    )
    assert d.embed["color"] == digest.GREY and "fields" not in d.embed
    assert d.embed["footer"]["text"] == "10 Nov 2026, 18:00 PKT"  # no summary, so no summary note


def test_empty_window_is_sent_and_recorded(monkeypatch):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert "No analyses in the last 24 hours" in discord.calls[0]["json"]["embeds"][0]["description"]
    run = runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("sent", 0, None)


# --- Summary ---

def test_model_gets_aggregates_only_and_the_rules(monkeypatch, a_day):
    sent = {}

    def fake(prompt, facts, schema):
        sent.update(prompt=prompt, facts=facts, schema=schema)
        return model_says("Three sentences. About the day. Done.")(prompt, facts, schema)

    monkeypatch.setattr(llm, "generate", fake)
    d = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert fields(d)["Summary"] == "Three sentences. About the day. Done." and d.automated
    facts = sent["facts"]
    assert set(facts) == set(digest.SUMMARY_FACTS) | {
        "missing_helmet_total", "missing_vest_total", "violation_rate_of_checked_pct"
    }
    assert all(isinstance(v, int) for v in facts.values())  # numbers only: no names, ids or lists
    assert (facts["missing_helmet_total"], facts["missing_vest_total"], facts["violation_rate_of_checked_pct"]) == (5, 3, 64)
    assert facts["checked_for_ppe"] == facts["compliant"] + facts["in_violation"]
    assert sent["schema"] is digest.DigestSummary
    for rule in ("Use only the facts", "it is important to", "never call them compliant", "camera",
                 "site supervisor", "do not repeat them", "Do not restate the counts", 'never say "stop work"',
                 "Verify on site, then correct:", "2 or 3 short sentences"):
        assert rule.lower() in sent["prompt"].lower(), rule


@pytest.mark.parametrize("answer", [llm.LLMUnavailable("a: 503"), "", "x" * (digest.MAX_SUMMARY_CHARS + 1)])
def test_model_failure_gives_the_standard_summary(monkeypatch, a_day, answer):
    def fake(prompt, facts, schema):
        if isinstance(answer, Exception):
            raise answer
        return model_says(answer)(prompt, facts, schema)

    monkeypatch.setattr(llm, "generate", fake)
    d = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert not d.automated
    assert fields(d)["Summary"].startswith("Most workers in violation are missing helmets.")
    assert d.embed["footer"]["text"].startswith("Standard summary, built from the figures above")


def test_model_failure_through_the_real_interface(monkeypatch, a_day):
    """llm.generate itself (not a fake): every model fails, so the digest falls back."""
    tried = []

    def failing_call(text, schema, model):
        tried.append(model)
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(llm, "_call", failing_call)
    monkeypatch.setattr(llm, "DEFAULT_MODELS", ["first", "second"])
    d = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert tried == ["first", "second"] and not d.automated


def test_standard_summary_wording(a_day):
    text = digest.template_summary(digest.summary_facts(digest.summarise(rows())))
    assert text == (
        "Most workers in violation are missing helmets. "
        "Some workers were too far from the camera to check. "
        "Verify on site, then correct: check helmets and vests at the start of tomorrow's shift, "
        "and move the camera closer so more workers can be checked."
    )
    assert not any(ch.isdigit() for ch in text)  # the fields already show every count


def test_standard_summary_when_nobody_could_be_checked():
    add([FAR, FAR], hours(2), gemini=False)
    text = digest.template_summary(digest.summary_facts(digest.summarise(rows())))
    assert text == (
        "No worker could be checked: everyone was too far from the camera. "
        "Move the camera closer so more workers can be checked tomorrow."
    )
    assert "compliant" not in text and "None%" not in text


def test_standard_summary_without_violations():
    add([OK, OK], hours(2), gemini=False)
    text = digest.template_summary(digest.summary_facts(digest.summarise(rows())))
    assert text == "No checked worker was missing PPE. No PPE action needed tomorrow."


def test_standard_summary_never_says_stop_work_and_is_short(a_day):
    add([NO_BOTH] * 2 + [FAR] * 20, hours(2))  # 23 of 36 people too far: over half
    text = digest.template_summary(digest.summary_facts(digest.summarise(rows())))
    assert "stop work" not in text.lower() and text.count(". ") + 1 <= 3
    assert "Half or more of the people detected were too far from the camera to check." in text


def test_disabled_alerts_read_as_not_sent(a_day):
    reliability = fields(digest.build_digest(UNTIL - hours(24), UNTIL, use_llm=False))["Reliability"]
    assert "1 not sent (alerts off)" in reliability and "disabled" not in reliability


# --- Length and mentions ---

def test_many_analyses_stay_within_discords_limits(monkeypatch):
    # Worst case per part: 200 rows, names that double when markdown-escaped, a maximum-length summary.
    for i in range(200):
        add([NO_BOTH] * 5 + [FAR] * 5, timedelta(minutes=i + 1), name="_" * 300 + f"{i}.jpg", alert="failed")
    monkeypatch.setattr(llm, "generate", model_says("y" * digest.MAX_SUMMARY_CHARS))
    d = digest.build_digest(UNTIL - hours(24), UNTIL)
    assert digest.embed_problems(d.content, [d.embed]) == []
    assert fields(d)["Analyses"] == "200"
    assert fields(d)["Worst analyses"].count("\n") == 2  # only the top 3
    for line in fields(d)["Worst analyses"].split("\n"):
        assert ".jpg" in line  # shortened names keep their extension


def test_over_the_limit_fails_instead_of_truncating(monkeypatch, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch)
    monkeypatch.setitem(digest_limits(), "total", 100)
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    assert discord.calls == []  # nothing partial is ever sent
    run = runs()[0]
    assert run["status"] == "failed" and "embeds total" in run["error"] and run["analyses_count"] == 6


def digest_limits() -> dict:
    return discord_webhook.EMBED_LIMITS


def test_mentions_never_ping(monkeypatch):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    add([NO_HELMET], hours(1), name="@everyone @here.jpg")
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    payload = discord.calls[0]["json"]
    assert "@everyone" in payload["embeds"][0]["fields"][7]["value"]  # shown as text...
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
    monkeypatch.setattr(discord_webhook.requests, "post", discord)
    monkeypatch.setattr(discord_webhook, "sleep", lambda s: discord.calls.append({"slept": s}))
    return discord


def test_sends_one_message_and_records_it(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert len(discord.calls) == 1 and discord.urls == [DIGEST_URL]
    assert discord.calls[0]["json"]["content"].startswith("📋 Daily PPE digest")
    assert discord.calls[0]["json"]["embeds"][0]["title"] == "Daily PPE digest — 10 Nov 2026"
    assert "Digest sent: 6 analyses" in capsys.readouterr().out
    [run] = runs()
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
    assert capsys.readouterr().out.startswith("📋 Daily PPE digest")
    assert runs() == []


def test_dry_run_needs_no_webhook(capsys):
    assert digest.main(["--dry-run", "--no-llm"]) == 0


def test_missing_webhook_fails_before_any_work(monkeypatch, capsys):
    monkeypatch.setattr(digest, "build_digest", lambda *a, **k: pytest.fail("must not build"))
    assert digest.main(UNTIL_ARG) == 1
    assert "DISCORD_DIGEST_WEBHOOK_URL" in capsys.readouterr().err
    run = runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("failed", None, digest.NO_WEBHOOK)


def test_send_failure_exits_1_and_is_recorded(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(404, {"message": "Unknown Webhook"}))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    assert len(discord.calls) == 1  # 404: not retried
    assert "Digest not sent: HTTP 404: Unknown Webhook" in capsys.readouterr().err
    run = runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("failed", 6, "HTTP 404: Unknown Webhook")


def test_server_error_is_retried_once(monkeypatch, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    discord = fake_discord(monkeypatch, response(503), response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    assert [c.get("slept") for c in discord.calls] == [None, 2, None]
    assert runs()[0]["status"] == "sent"


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
    error = runs()[0]["error"]
    assert error.startswith("ConnectionError")
    assert "DIGEST-secret_token" not in error and DIGEST_URL not in error
    assert "DIGEST-secret_token" not in capsys.readouterr().err


def test_unexpected_send_exception_is_redacted_and_recorded(monkeypatch, capsys, a_day):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)

    def broken(url, content, **kwargs):
        raise RuntimeError(f"bug while posting to {url}")

    monkeypatch.setattr(digest, "send_to_discord", broken)
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 1
    run = runs()[0]
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
    run = runs()[0]
    assert (run["status"], run["analyses_count"], run["error"]) == ("failed", None, "OSError: disk unreadable")


def test_a_recording_problem_does_not_change_the_outcome(monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    fake_discord(monkeypatch, response(200))

    def broken(**kwargs):
        raise OSError("read-only database")

    monkeypatch.setattr(digest, "record_digest_run", broken)
    assert digest.main(["--no-llm"]) == 0  # the digest was delivered; re-running would post it twice
    assert "could not record the digest run" in capsys.readouterr().err

# --- End to end on the real log ---

def test_end_to_end_on_the_test_branch(db, monkeypatch, capsys):
    """Real storage (Neon test branch), real retry code, fake Discord."""
    monkeypatch.setenv("DISCORD_DIGEST_WEBHOOK_URL", DIGEST_URL)
    worst = add([NO_HELMET, NO_VEST], hours(2), name="two.jpg", alert="failed")
    add([NO_BOTH], hours(3), name="one.jpg", alert="sent")
    add([OK, FAR], hours(4), gemini=False, name="ok.jpg")
    add([NO_BOTH], hours(30), name="old.jpg")  # outside the window
    discord = fake_discord(monkeypatch, response(200))
    assert digest.main([*UNTIL_ARG, "--no-llm"]) == 0
    embed = discord.calls[0]["json"]["embeds"][0]
    got = {f["name"]: f["value"] for f in embed["fields"]}
    assert (got["Analyses"], got["People detected"], got["Checked"], got["In violation"],
            got["Not checked (too far)"]) == ("3", "5", "4", "3", "1")
    assert got["Worst analyses"].startswith(f"1. 10 Nov 16:00 · two.jpg · 2 in violation · Medium (#{worst})")
    assert got["Reliability"].split("\n") == ["Alerts: 1 sent · 1 failed", "Summaries: 2 automated, 1 standard"]
    [run] = storage.list_digest_runs()
    assert (run["status"], run["analyses_count"], run["window_end"]) == ("sent", 3, "2026-11-10T13:00:00Z")


# --- A light install ---

def test_running_the_digest_never_imports_the_detection_stack():
    """GitHub Actions installs requirements-digest.txt only: no torch, OpenCV or Ultralytics."""
    script = "; ".join([
        "import sys",
        "import src.digest as d",
        "d.get_since = lambda since, until: []",  # no database needed: an empty window
        "code = d.main(['--dry-run', '--no-llm'])",
        "heavy = sorted(m for m in ('torch', 'ultralytics', 'cv2', 'numpy', 'src.detect', 'src.violations',"
        " 'src.alerts', 'gradio') if m in sys.modules)",
        "print('HEAVY', heavy, 'EXIT', code)",
    ])
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"} | {"PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                            cwd=Path(__file__).resolve().parent.parent, timeout=120, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert "HEAVY [] EXIT 0" in result.stdout
