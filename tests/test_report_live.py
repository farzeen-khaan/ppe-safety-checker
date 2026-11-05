"""Live test against the real Gemini API. Skipped unless RUN_LIVE_TESTS=1.

Unit tests replace Gemini with fakes, so they cannot catch rules of the real API
(for example: it rejects timeouts under 10 s). Run this after changing LLM settings:

    $env:RUN_LIVE_TESTS = "1"; python -m pytest -m live      # PowerShell
"""

import os

import pytest

from src import report
from src.detect import Detection
from src.violations import PersonVerdict

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]


def test_real_gemini_accepts_our_settings():
    """Every configured model must accept our request settings, even if it is busy right now."""
    payload = report.build_payload([PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 100, 300), False, True)])
    for model in report.gemini_models():
        try:
            answer = report.ask_gemini(payload, model)
        except Exception as exc:
            # Overload (503) and rate limits (429) are the service's state, not our bug.
            # A 400 means the request itself is wrong: that must fail the test.
            assert "400" not in str(exc), f"{model} rejected our request: {exc}"
            continue
        assert answer.summary and answer.recommended_action
