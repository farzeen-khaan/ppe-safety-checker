"""Shared test setup: no test may post to the real Discord channel or write to the real log."""

import pytest


@pytest.fixture(autouse=True)
def isolate_from_real_services(monkeypatch, tmp_path):
    # .env may switch alerts on; tests that need them on set their own fake webhook.
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    monkeypatch.setenv("INCIDENT_DB_PATH", str(tmp_path / "incidents.db"))
