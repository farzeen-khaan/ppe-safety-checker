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


def test_no_people_badge():
    html = app.badge_html([], "None")
    assert "No workers detected" in html and "badge-grey" in html


def test_verdict_rows_are_readable():
    vs = [PersonVerdict(Detection(5, "Person", 0.912, 0, 0, 10, 30), has_helmet=False, has_vest=True),
          PersonVerdict(Detection(5, "Person", 0.5, 0, 0, 10, 30), has_helmet=None, has_vest=None)]
    assert app.verdict_rows(vs) == [[1, "Missing helmet", "0.91"], [2, "Not checked (too far)", "0.50"]]


@pytest.mark.parametrize("severity", ["High", "Medium", "None"])
def test_severity_badge_colour(severity):
    assert f'class="badge sev-{severity}">Severity: {severity}<' in app.severity_badge(severity)


def test_badge_shows_not_checked_in_grey():
    far = PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 10, 30), None, None)
    no_helmet = PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 10, 30), False, True)
    html = app.badge_html([no_helmet, far, far], "Medium")
    assert "sev-Medium" in html and '<span class="badge badge-grey">2 not checked</span>' in html
    assert "3 people: 0 compliant, 1 in violation, 2 too far to judge" in html


def test_summary_card_discloses_automation_without_naming_the_model():
    gemini = Report("High", "S.", "A.", source="llm", provider="acme", model="model-x")
    card = app.report_card(gemini)
    assert "<h3>Summary</h3>" in card and "Automated summary, check against the figures above." in card
    for hidden in ("acme", "model-x", "gemini", "LLM", "Written by"):
        assert hidden not in card
    standard = app.report_card(Report("High", "S.", "A.", source="template", error="503"))
    assert "Standard summary, built from the figures above." in standard and "503" not in standard


def test_summary_card_escapes_html():
    card = app.summary_card("<script>alert(1)</script>", "a & b", automated=True)
    assert "<script>" not in card and "&lt;script&gt;" in card and "a &amp; b" in card


def test_to_pkt_adds_five_hours():
    assert app.to_pkt("2026-11-05T13:00:00Z") == "2026-11-05 18:00"
    assert app.to_pkt("2026-11-05T20:30:00Z") == "2026-11-06 01:30"  # crosses midnight


def test_history_shows_summary_kind_not_the_model():
    row = {"id": 7, "created_at": "2026-11-05T13:00:00Z", "image_name": "source-56088ecf75821c1b93f4015a2bc0a071.jpg",
           "people_count": 3, "compliant_count": 1, "violation_count": 1, "too_far_count": 1, "severity": "High",
           "report_source": "llm", "report_provider": "acme", "report_model": "model-x", "alert_status": "sent"}
    [cells] = app.history_rows([row])
    assert cells[8] == "automated" and "model-x" not in cells and "acme" not in cells
    assert cells[2].startswith("source-") and cells[2].endswith("2bc0a071.jpg") and "…" in cells[2]
    assert len(cells[2]) <= 32
    assert app.history_rows([row | {"report_source": "template"}])[0][8] == "standard"


def test_show_details_for_a_clicked_history_row(monkeypatch):
    from types import SimpleNamespace

    fake = {
        "id": 7, "image_name": "site.jpg", "created_at": "2026-11-05T13:00:00Z", "severity": "Medium",
        "report_summary": "S.", "report_action": "A.", "report_source": "llm", "report_provider": "acme",
        "report_model": "model-x", "alert_status": "pending", "conf_threshold": 0.25, "processing_ms": 1500,
        "people": [{"person_index": 1, "status": "missing vest", "confidence": 0.88, "has_helmet": True}],
    }
    monkeypatch.setattr(app, "get_analysis", lambda analysis_id: fake if analysis_id == 7 else None)

    text, people = app.show_details(SimpleNamespace(row_value=[7, "2026-11-05 18:00"]))
    assert "Analysis #7" in text and "18:00 PKT" in text and "sev-Medium" in text
    assert "Automated summary" in text and "model-x" not in text and "acme" not in text
    assert people == [[1, "Missing vest", "0.88"]]
    assert app.show_details(SimpleNamespace(row_value=[8]))[0] == "<p>That analysis no longer exists.</p>"


def test_history_says_so_when_the_log_is_unreachable(monkeypatch):
    def unreachable(**kwargs):
        raise app.StorageUnavailable("could not be reached")

    monkeypatch.setattr(app, "list_analyses", unreachable)
    rows, note = app.load_history()
    assert rows == [] and "can't be reached right now" in note


def test_details_say_so_when_the_log_is_unreachable(monkeypatch):
    from types import SimpleNamespace

    def unreachable(analysis_id):
        raise app.StorageUnavailable("could not be reached")

    monkeypatch.setattr(app, "get_analysis", unreachable)
    text, people = app.show_details(SimpleNamespace(row_value=[7]))
    assert "can't be reached right now" in text and people == []
