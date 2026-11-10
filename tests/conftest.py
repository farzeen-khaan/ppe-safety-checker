"""Shared test setup: no test may post to the real Discord channel or touch the real database.

- DATABASE_URL (the app's dev or main branch) is removed for every test. A test that
  reaches storage without asking for a database fails with "DATABASE_URL is not set".
- Tests that need Postgres ask for the `db` fixture. It points DATABASE_URL at
  DATABASE_URL_TEST (the Neon test branch) and empties the tables after the test.
  Without DATABASE_URL_TEST those tests are skipped with a clear message; every
  other test still runs offline.
"""

import os
from urllib.parse import urlsplit

import pytest

from src import storage  # importing it reads .env, so both URLs are known below

TEST_URL = os.environ.get("DATABASE_URL_TEST", "").strip()
_APP_URL = os.environ.get("DATABASE_URL", "").strip()  # only compared, never used
TABLES = "analyses, person_verdicts, digest_runs"


def _same_database(a: str, b: str) -> bool:
    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.hostname, pa.path) == (pb.hostname, pb.path)


@pytest.fixture(autouse=True)
def isolate_from_real_services(monkeypatch, request):
    # .env may switch alerts on; tests that need them on set their own fake webhook.
    monkeypatch.setenv("ALERTS_ENABLED", "false")
    # The digest posts whenever a URL is set, so the real ones are removed too (live tests keep them).
    if request.node.get_closest_marker("live") is None:
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("DISCORD_DIGEST_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)  # never the app's database, live tests included


_cleaned_at_start = False


def empty_tables() -> None:
    with storage.connect(TEST_URL) as conn:
        conn.execute(f"TRUNCATE {TABLES} RESTART IDENTITY")


@pytest.fixture
def db(monkeypatch) -> str:
    """The Neon test branch, migrated and empty. Returns its connection string."""
    global _cleaned_at_start
    if not TEST_URL:
        pytest.skip("DATABASE_URL_TEST is not set: database tests need the Neon test branch (see .env.example)")
    if _APP_URL and _same_database(TEST_URL, _APP_URL):
        pytest.exit("DATABASE_URL_TEST points at the same database as DATABASE_URL; refusing to run", returncode=2)
    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    storage._ensure_db(TEST_URL)  # migrations, once per run
    if not _cleaned_at_start:  # rows left by an interrupted earlier run
        empty_tables()
        _cleaned_at_start = True
    yield TEST_URL
    empty_tables()
