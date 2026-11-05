"""Live test against the real Discord webhook. Skipped unless RUN_LIVE_TESTS=1.

Posts one test message to the channel in DISCORD_WEBHOOK_URL. Unit tests use a fake
Discord, so only this proves the real webhook accepts our request format:

    $env:RUN_LIVE_TESTS = "1"; python -m pytest -m live      # PowerShell
"""

import os

import numpy as np
import pytest

from src import alerts

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]


def test_real_webhook_accepts_a_message_with_an_image():
    url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        pytest.skip("DISCORD_WEBHOOK_URL is not set")
    image = np.full((120, 320, 3), 255, dtype=np.uint8)
    error = alerts.send_alert(url, "🧪 PPE Safety Checker: automated live test, please ignore.", alerts.encode_image(image))
    assert error is None, error  # the error is already redacted, so the URL never shows in the output
