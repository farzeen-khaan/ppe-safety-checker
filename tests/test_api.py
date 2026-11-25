"""Tests for the FastAPI app (F6) with TestClient. The model, the language model and Discord are fakes.

Most tests replace the pipeline and storage calls the endpoints make; test_full_flow_on_the_test_branch
runs the real ONNX model and the real log (Neon test branch), with the language model and Discord faked.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as web
from src import llm, pipeline, storage
from src.report import Report
from src.storage import StorageUnavailable
from src.verdicts import Detection, PersonVerdict

DATASET = Path(__file__).resolve().parent.parent / "data" / "construction-ppe" / "images" / "val"
PHOTO = DATASET / "source-56088ecf75821c1b93f4015a2bc0a071.jpg"  # 6 people, 2 in violation, 3 too far


def person(has_helmet, has_vest, x1=0.0) -> PersonVerdict:
    return PersonVerdict(Detection(5, "Person", 0.9, x1, 0.0, x1 + 100, 300.0), has_helmet, has_vest)


def jpeg_bytes(width=320, height=240) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((height, width, 3), 200, dtype=np.uint8))
    return buffer.tobytes()


def upload(client, data=None, content_type="image/jpeg", name="site.jpg", headers=None):
    return client.post("/api/analyze", files={"photo": (name, data if data is not None else jpeg_bytes(), content_type)},
                       headers=headers or {})


ROW = {
    "id": 7, "created_at": "2026-11-25T13:00:00Z", "image_name": "site.jpg", "severity": "High",
    "people_count": 3, "compliant_count": 1, "violation_count": 1, "too_far_count": 1,
    "report_summary": "The flagged worker is missing both a helmet and a vest.", "report_action": "Verify on site, then correct: …",
    "report_source": "llm", "report_provider": "acme", "report_model": "model-x", "report_error": None,
    "alert_status": "sent", "conf_threshold": 0.25, "processing_ms": 812,
    "people": [
        {"person_index": 1, "status": "compliant", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 30},
        {"person_index": 2, "status": "missing helmet and vest", "confidence": 0.8, "x1": 20, "y1": 0, "x2": 30, "y2": 30},
        {"person_index": 3, "status": "too far to judge", "confidence": 0.6, "x1": 40, "y1": 0, "x2": 42, "y2": 3},
    ],
}


@pytest.fixture
def client(monkeypatch):
    """The app with every storage call faked (rate limit open) and a fake database URL for the IP salt."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@db.invalid/neondb")
    monkeypatch.setattr(storage, "rate_limit", lambda bucket, client, limit, window: None)
    return TestClient(web.app)


@pytest.fixture
def analyzed(monkeypatch):
    """pipeline.analyze replaced: verdicts for 3 people, saved as #7, alert sent."""
    calls = []

    def fake(image, image_name, **kwargs):
        calls.append((image.shape, image_name))
        verdicts = [person(True, True), person(False, False, 120), person(None, None, 240)]
        return pipeline.AnalysisResult(7, verdicts, pipeline.standard_report(verdicts), image, 640, None, "sent")

    monkeypatch.setattr(web, "analyze", fake)
    return calls


# --- The page and the docs ---

def test_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for text in ("PPE Safety Checker", "/api/analyze", "Summaries are automated", "first request", "10 photos per hour"):
        assert text.lower() in r.text.lower(), text


def test_docs_and_schema(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) >= {"/api/analyze", "/api/analyses/{analysis_id}/summary", "/api/history",
                          "/api/analyses/{analysis_id}", "/api/health"}


# --- POST /api/analyze ---

def test_analyze_returns_verdicts_and_the_annotated_image(client, analyzed):
    r = upload(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert (body["id"], body["severity"], body["alert_status"], body["processing_ms"]) == (7, "High", "sent", 640)
    assert body["counts"] == {"people": 3, "compliant": 1, "in_violation": 1, "too_far": 1}
    assert [p["status"] for p in body["people"]] == ["compliant", "missing helmet and vest", "too far to judge"]
    assert body["annotated_jpeg"].startswith("data:image/jpeg;base64,")
    assert analyzed == [((240, 320, 3), "site.jpg")]


def test_analyze_keeps_only_the_file_name(client, analyzed):
    upload(client, name="../../etc/passwd.jpg")
    assert analyzed[0][1] == "passwd.jpg"


@pytest.mark.parametrize("content_type", ["text/plain", "application/pdf", "image/gif"])
def test_wrong_file_type_is_415(client, analyzed, content_type):
    r = upload(client, b"hello", content_type=content_type)
    assert r.status_code == 415 and r.json()["detail"] == "Upload a JPG, PNG or WEBP photo."
    assert analyzed == []


def test_unreadable_image_is_400(client, analyzed):
    r = upload(client, b"not really a jpeg")
    assert r.status_code == 400 and "not an image this app can read" in r.json()["detail"]


def test_too_large_is_413(client, analyzed):
    r = upload(client, b"\xff" * (web.MAX_UPLOAD_BYTES + 200_000))
    assert r.status_code == 413 and "larger than 3 MB" in r.json()["detail"]
    assert analyzed == []


def test_rate_limited_is_429_with_retry_after(client, analyzed, monkeypatch):
    monkeypatch.setattr(storage, "rate_limit", lambda bucket, client, limit, window: 1800)
    r = upload(client)
    assert r.status_code == 429 and r.headers["retry-after"] == "1800"
    assert r.json()["detail"] == "Too many requests from your connection. Try again in about 30 min."
    assert analyzed == []  # refused before the model runs


def test_the_limit_is_per_bucket(client, analyzed, monkeypatch):
    seen = []
    monkeypatch.setattr(storage, "rate_limit", lambda bucket, client, limit, window: seen.append((bucket, limit)))
    upload(client)
    assert seen == [("analyze", web.ANALYZE_LIMIT)]


def test_database_waking_is_503_with_retry_hint(client, analyzed, monkeypatch):
    def asleep(*a, **k):
        raise StorageUnavailable("could not be reached")

    monkeypatch.setattr(storage, "rate_limit", asleep)
    r = upload(client)
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert "waking up" in r.json()["detail"]


def test_a_failed_save_is_503(client, monkeypatch):
    monkeypatch.setattr(web, "analyze", lambda image, name, **k: pipeline.AnalysisResult(
        None, [], pipeline.standard_report([]), image, 10, "StorageUnavailable: …", "not_needed"))
    assert upload(client).status_code == 503


# --- POST /api/analyses/{id}/summary ---

def test_summary(client, monkeypatch):
    monkeypatch.setattr(web, "write_summary", lambda analysis_id: Report(
        "High", "Most flagged workers are missing helmets.", "Verify on site, then correct: …",
        source="llm", provider="acme", model="model-x"))
    r = client.post("/api/analyses/7/summary")
    assert r.status_code == 200
    assert r.json() == {"id": 7, "summary": "Most flagged workers are missing helmets.",
                        "recommended_action": "Verify on site, then correct: …", "automated": True,
                        "note": "Automated summary, check against the figures above."}
    assert "acme" not in r.text and "model-x" not in r.text  # provenance stays in the log


def test_summary_unknown_id_is_404(client, monkeypatch):
    def missing(analysis_id):
        raise KeyError(analysis_id)

    monkeypatch.setattr(web, "write_summary", missing)
    r = client.post("/api/analyses/999/summary")
    assert r.status_code == 404 and r.json()["detail"] == "No analysis with id 999."


def test_summary_rate_limited_is_429(client, monkeypatch):
    monkeypatch.setattr(storage, "rate_limit", lambda bucket, client, limit, window: 60 if bucket == "summary" else None)
    assert client.post("/api/analyses/7/summary").status_code == 429


def test_summary_database_waking_is_503(client, monkeypatch):
    def asleep(analysis_id):
        raise StorageUnavailable("x")

    monkeypatch.setattr(web, "write_summary", asleep)
    assert client.post("/api/analyses/7/summary").status_code == 503


# --- GET /api/history and /api/analyses/{id} ---

def test_history(client, monkeypatch):
    rows = [ROW | {"id": 8, "report_source": "template", "report_error": pipeline.PENDING},
            ROW | {"id": 7}, ROW | {"id": 6, "report_source": "template", "report_error": "LLMUnavailable: …"}]
    monkeypatch.setattr(storage, "list_analyses", lambda limit: rows[:limit])
    r = client.get("/api/history?limit=3")
    assert r.status_code == 200
    assert [(h["id"], h["summary"]) for h in r.json()] == [(8, "pending"), (7, "automated"), (6, "standard")]
    assert r.json()[1]["counts"] == {"people": 3, "compliant": 1, "in_violation": 1, "too_far": 1}


@pytest.mark.parametrize("limit", [0, 101, "x"])
def test_history_limit_is_validated(client, limit):
    assert client.get(f"/api/history?limit={limit}").status_code == 422


def test_history_database_waking_is_503(client, monkeypatch):
    def asleep(limit):
        raise StorageUnavailable("x")

    monkeypatch.setattr(storage, "list_analyses", asleep)
    assert client.get("/api/history").status_code == 503


def test_one_analysis(client, monkeypatch):
    monkeypatch.setattr(storage, "get_analysis", lambda analysis_id: ROW if analysis_id == 7 else None)
    r = client.get("/api/analyses/7")
    assert r.status_code == 200
    body = r.json()
    assert (body["summary"], body["note"]) == ("automated", "Automated summary, check against the figures above.")
    assert [p["status"] for p in body["people"]] == ["compliant", "missing helmet and vest", "too far to judge"]
    assert "model-x" not in r.text
    assert client.get("/api/analyses/8").status_code == 404


# --- GET /api/health ---

def test_health_ok(client, monkeypatch):
    monkeypatch.setattr(storage, "ping", lambda: None)
    assert client.get("/api/health").json() == {"status": "ok", "database": "ok"}


def test_health_reports_an_unreachable_database_without_failing(client, monkeypatch):
    def asleep():
        raise StorageUnavailable("x")

    monkeypatch.setattr(storage, "ping", asleep)
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["database"] == "unreachable"


# --- Client identity for the rate limit ---

def test_client_key_is_a_salted_hash_of_the_real_ip(client, analyzed, monkeypatch):
    seen = []
    monkeypatch.setattr(storage, "rate_limit", lambda bucket, key, limit, window: seen.append(key))
    upload(client, headers={"x-real-ip": "203.0.113.7"})
    upload(client, headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    upload(client, headers={"x-real-ip": "198.51.100.2"})
    assert seen[0] == seen[1] != seen[2]  # same client by either header; another client differs
    assert len(seen[0]) == 64 and "203.0.113.7" not in seen[0]
    assert seen[0] != hashlib.sha256(b"203.0.113.7").hexdigest()  # salted: not reversible by hashing every IPv4


# --- End to end: real ONNX model and the real log (test branch); language model and Discord faked ---

@pytest.mark.skipif(not PHOTO.exists(), reason="the dataset (data/) is not downloaded")
def test_full_flow_on_the_test_branch(db, monkeypatch):
    sent = []
    monkeypatch.setenv("ALERTS_ENABLED", "true")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/fake")
    monkeypatch.setattr("src.alerts.send_to_discord", lambda url, content, jpg, embeds: sent.append(embeds) or None)
    monkeypatch.setattr(llm, "generate", lambda prompt, facts, schema: llm.Generated(
        schema(summary="Most flagged workers are missing helmets.", recommended_action="Verify on site, then correct: x."),
        "acme", "model-x"))
    client = TestClient(web.app)

    r = upload(client, PHOTO.read_bytes(), name=PHOTO.name, headers={"x-real-ip": "203.0.113.9"})
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["counts"] == {"people": 6, "compliant": 1, "in_violation": 2, "too_far": 3}
    assert (a["severity"], a["alert_status"]) == ("High", "sent") and len(sent) == 1

    assert client.get("/api/history").json()[0]["summary"] == "pending"
    s = client.post(f"/api/analyses/{a['id']}/summary").json()
    assert s["automated"] and s["summary"] == "Most flagged workers are missing helmets."
    again = client.post(f"/api/analyses/{a['id']}/summary").json()
    assert again == s  # saved: a second call doesn't ask the model again

    detail = client.get(f"/api/analyses/{a['id']}").json()
    assert (detail["summary"], detail["image_name"], len(detail["people"])) == ("automated", PHOTO.name, 6)
    assert storage.get_analysis(a["id"])["model_version"].startswith("best.onnx:")
