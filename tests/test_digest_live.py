"""Live tests for F8: the real Gemini API and the real digest webhook. Skipped unless RUN_LIVE_TESTS=1.

    $env:RUN_LIVE_TESTS = "1"; python -m pytest -m live      # PowerShell

test_real_digest_is_posted sends one digest, built from sample data on the Neon test branch,
to the channel in DISCORD_DIGEST_WEBHOOK_URL (or DISCORD_WEBHOOK_URL). It is labelled
as a test so nobody mistakes it for a real day.
"""

import os
from datetime import timedelta

import pytest

from src import digest, report, storage
from src.detect import Detection
from src.report import Report
from src.violations import PersonVerdict

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]

ROW = {
    "people_count": 4, "too_far_count": 1, "compliant_count": 1, "violation_count": 2,
    "missing_helmet_only_count": 1, "missing_vest_only_count": 0, "missing_both_count": 1,
    "severity": "High", "report_source": "gemini", "alert_status": "sent",
}


def test_real_gemini_writes_an_overview():
    facts = digest.overview_facts(digest.summarise([ROW]))
    for model in report.gemini_models():
        try:
            answer = report.ask_gemini(facts, model, digest.OVERVIEW_PROMPT, digest.Overview)
        except Exception as exc:
            assert "400" not in str(exc), f"{model} rejected our request: {exc}"
            continue  # overloaded or rate-limited: the service's state, not our bug
        assert 0 < len(answer.overview) <= digest.MAX_OVERVIEW_CHARS


def test_real_digest_is_posted(db):
    url = digest.digest_webhook_url()
    if not url:
        pytest.skip("neither DISCORD_DIGEST_WEBHOOK_URL nor DISCORD_WEBHOOK_URL is set")
    # The db fixture points storage at the Neon test branch and empties it afterwards: never the real log.
    now = storage.now_utc()
    for has_helmet, has_vest, name in ((False, False, "live_test_1.jpg"), (True, False, "live_test_2.jpg")):
        verdicts = [PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 100, 300), has_helmet, has_vest)]
        storage.save_analysis(
            image_name=name, image_sha256="ab" * 32, image_width=640, image_height=480, verdicts=verdicts,
            report=Report(report.compute_severity(verdicts), "S.", "A.", source="template", error="LLM disabled"),
            model_version="live-test", conf_threshold=0.25, processing_ms=1, created_at=now - timedelta(minutes=5),
        )
    built = digest.build_digest(now - timedelta(hours=24), now, use_llm=False)
    message = "🧪 **Automated live test with sample data, please ignore.**\n" + built.message
    assert built.analyses == 2 and len(message) <= digest.MAX_CONTENT_CHARS
    error = digest.send_to_discord(url, message)
    assert error is None, error  # already redacted, so the URL never shows in the output
