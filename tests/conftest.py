"""Shared test setup: no test may post to the real Discord channel or write to the real log."""

import pytest


@pytest.fixture(autouse=True)
def isolate_from_real_services(monkeypatch, tmp_path, request):
    # .env may switch alerts on; tests that need them on set their own fake webhook.
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    # The digest posts whenever a URL is set, so the real ones are removed too (live tests keep them).
    if request.node.get_closest_marker("live") is None:
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("DISCORD_DIGEST_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("INCIDENT_DB_PATH", str(tmp_path / "incidents.db"))
