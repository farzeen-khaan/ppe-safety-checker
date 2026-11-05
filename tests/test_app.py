"""Tests for the web UI helpers (no browser, no model)."""

import cv2
import gradio as gr
import numpy as np
import pytest

import app
from src.detect import Detection
from src.report import Report
from src.violations import PersonVerdict


def test_read_image_accepts_a_real_image(tmp_path):
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), np.zeros((20, 30, 3), dtype=np.uint8))
    assert app.read_image(str(path)).shape == (20, 30, 3)


def test_read_image_handles_non_english_file_names(tmp_path):
    path = tmp_path / "سائٹ_تصویر.png"  # cv2.imread fails on such paths on Windows
    ok, data = cv2.imencode(".png", np.zeros((5, 5, 3), dtype=np.uint8))
    path.write_bytes(data.tobytes())
    assert app.read_image(str(path)).shape == (5, 5, 3)


@pytest.mark.parametrize("content, name", [(b"hello, not an image", "notes.txt"), (b"fake", "renamed.jpg")])
def test_read_image_rejects_non_images_with_a_clear_message(tmp_path, content, name):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(gr.Error, match="not an image"):
        app.read_image(str(path))


def test_read_image_without_upload():
    with pytest.raises(gr.Error, match="upload a photo"):
        app.read_image(None)


def test_no_people_message():
    assert "No workers detected" in app.summary_markdown([])


def test_verdict_rows():
    v = PersonVerdict(Detection(5, "Person", 0.912, 0, 0, 10, 30), has_helmet=False, has_vest=True)
    assert app.verdict_rows([v]) == [[1, "missing helmet", "0.91"]]


def test_to_pkt_adds_five_hours():
    assert app.to_pkt("2026-11-05T13:00:00Z") == "2026-11-05 18:00"
    assert app.to_pkt("2026-11-05T20:30:00Z") == "2026-11-06 01:30"  # crosses midnight


def test_report_markdown_names_its_source():
    gemini = Report("High", "S.", "A.", source="gemini", model="gemini-x")
    fallback = Report("High", "S.", "A.", source="template", error="503")
    assert "Written by gemini-x" in app.report_markdown(gemini)
    assert "Gemini was unavailable" in app.report_markdown(fallback)
    assert "severity: High" in app.report_markdown(gemini)


def test_show_details_for_a_clicked_history_row(monkeypatch):
    from types import SimpleNamespace

    fake = {
        "id": 7, "image_name": "site.jpg", "created_at": "2026-11-05T13:00:00Z", "severity": "Medium",
        "report_summary": "S.", "report_action": "A.", "report_model": None, "alert_status": "pending",
        "conf_threshold": 0.25, "processing_ms": 1500,
        "people": [{"person_index": 1, "status": "missing vest", "confidence": 0.88}],
    }
    monkeypatch.setattr(app, "get_analysis", lambda analysis_id: fake if analysis_id == 7 else None)

    text, people = app.show_details(SimpleNamespace(row_value=[7, "2026-11-05 18:00"]))
    assert "Analysis #7" in text and "18:00 PKT" in text and "Report by template" in text
    assert people == [[1, "missing vest", "0.88"]]
    assert app.show_details(SimpleNamespace(row_value=[8]))[0] == "That analysis no longer exists."
