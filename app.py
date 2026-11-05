"""F6 — Web UI: Gradio app with Analyse and History tabs.

Run locally:
    python app.py           # then open http://127.0.0.1:7860

Analyse shows the annotated image and verdicts first (about 1 s), then the report
when it is ready, so a slow LLM never blocks the result. History is text only:
the incident log stores no photos.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.detect import load_model
from src.pipeline import finish_analysis, inspect_image
from src.report import Report
from src.storage import get_analysis, list_analyses
from src.violations import MIN_PERSON_HEIGHT, PersonVerdict, summary_text

MAX_UPLOAD_MB = 20
HISTORY_LIMIT = 100
# Pakistan has no daylight saving, so a fixed offset is exact (and needs no tzdata package).
PKT = timezone(timedelta(hours=5), "PKT")

VERDICT_HEADERS = ["Person", "Verdict", "Detection confidence"]
HISTORY_HEADERS = ["ID", "Time (PKT)", "Image", "People", "OK", "Violations", "Too far", "Severity", "Report by", "Alert"]


# --- Helpers (plain functions, so they can be tested without a browser) ---

def read_image(path: str | None) -> np.ndarray:
    """Load an uploaded file as a BGR image, or raise gr.Error with a clear message."""
    if not path:
        raise gr.Error("Please upload a photo first.")
    file = Path(path)
    if file.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise gr.Error(f"“{file.name}” is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller photo.")
    # np.fromfile + imdecode instead of cv2.imread: imread fails on Windows paths with non-English characters.
    image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise gr.Error(f"“{file.name}” is not an image this app can read. Please upload a JPG, PNG or WEBP photo.")
    return image


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV works in BGR; browsers show RGB. Without this, green boxes would appear red."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def verdict_rows(verdicts: list[PersonVerdict]) -> list[list]:
    return [[i, v.status, f"{v.person.confidence:.2f}"] for i, v in enumerate(verdicts, 1)]


def summary_markdown(verdicts: list[PersonVerdict]) -> str:
    if not verdicts:
        return (
            "**No workers detected in this photo.** Try a closer or clearer photo of the site. "
            "The report below still records the check."
        )
    return f"**{summary_text(verdicts)}**"


def report_markdown(report: Report) -> str:
    if report.source == "gemini":
        written_by = f"Written by {report.model}."
    else:
        written_by = "Template report (Gemini was unavailable)." if report.error != "LLM disabled" else "Template report."
    return (
        f"### Incident report — severity: {report.severity}\n\n"
        f"{report.summary}\n\n"
        f"**Recommended action:** {report.recommended_action}\n\n"
        f"<sub>{written_by}</sub>"
    )


def to_pkt(utc_text: str) -> str:
    """'2026-11-05T13:00:00Z' (stored UTC) → '2026-11-05 18:00' (Pakistan time, for display)."""
    moment = datetime.strptime(utc_text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return moment.astimezone(PKT).strftime("%Y-%m-%d %H:%M")


def history_rows(rows: list[dict]) -> list[list]:
    return [
        [
            r["id"],
            to_pkt(r["created_at"]),
            r["image_name"],
            r["people_count"],
            r["compliant_count"],
            r["violation_count"],
            r["too_far_count"],
            r["severity"],
            r["report_model"] or "template",
            r["alert_status"],
        ]
        for r in rows
    ]


# --- Event handlers ---

def analyse(path: str | None):
    """Generator: yields once with image and verdicts, then again with the report."""
    image = read_image(path)
    inspection = inspect_image(image)
    annotated = to_rgb(inspection.annotated)
    rows = verdict_rows(inspection.verdicts)
    summary = summary_markdown(inspection.verdicts)

    yield annotated, rows, summary, "_Writing the incident report…_", "Analysing… report in progress."

    result = finish_analysis(inspection, image_name=Path(path).name)
    if result.save_error:
        status = f"Analysed in {result.processing_ms / 1000:.1f} s, but **not saved** to the log: {result.save_error}"
    else:
        status = f"Analysed in {result.processing_ms / 1000:.1f} s. Saved as analysis #{result.analysis_id}."
    yield annotated, rows, summary, report_markdown(result.report), status


def load_history():
    rows = list_analyses(limit=HISTORY_LIMIT)
    note = f"{len(rows)} most recent analyses, newest first. Select a row to see its report." if rows else (
        "No analyses yet. Analyse a photo first."
    )
    return history_rows(rows), note


def show_details(evt: gr.SelectData):
    """When a History row is clicked: its report and per-person verdicts."""
    if not evt.row_value:
        return "", []
    analysis = get_analysis(int(evt.row_value[0]))
    if analysis is None:
        return "That analysis no longer exists.", []
    text = (
        f"### Analysis #{analysis['id']} — {analysis['image_name']} ({to_pkt(analysis['created_at'])} PKT)\n\n"
        f"**Severity: {analysis['severity']}.** {analysis['report_summary']}\n\n"
        f"**Recommended action:** {analysis['report_action']}\n\n"
        f"<sub>Report by {analysis['report_model'] or 'template'} · alert: {analysis['alert_status']} · "
        f"confidence threshold {analysis['conf_threshold']} · {analysis['processing_ms'] / 1000:.1f} s</sub>"
    )
    people = [[p["person_index"], p["status"], f"{p['confidence']:.2f}"] for p in analysis["people"]]
    return text, people


# --- Layout ---

with gr.Blocks(title="PPE Safety Checker") as demo:
    gr.Markdown(
        "# PPE Safety Checker\n"
        "Upload a construction-site photo. Each worker is checked for a **helmet** and a **vest**: "
        "green = compliant, red = violation, grey = too far to judge "
        f"(smaller than {MIN_PERSON_HEIGHT:.0%} of the photo height)."
    )

    with gr.Tab("Analyse"):
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Site photo (JPG, PNG, WEBP)", file_types=["image"], type="filepath")
                analyse_button = gr.Button("Analyse", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                result_image = gr.Image(label="Result", type="numpy", interactive=False)
                summary = gr.Markdown()
                verdict_table = gr.Dataframe(headers=VERDICT_HEADERS, interactive=False, label="Per-person verdicts")
                report = gr.Markdown()

    with gr.Tab("History") as history_tab:
        with gr.Row():
            refresh_button = gr.Button("Refresh", scale=0)
            history_note = gr.Markdown()
        history_table = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, label="Past analyses")
        details = gr.Markdown()
        details_people = gr.Dataframe(headers=VERDICT_HEADERS, interactive=False, label="Per-person verdicts")

    # Only the button runs an analysis (not the upload itself), so each saved analysis is a deliberate click.
    analyse_button.click(
        analyse, inputs=upload, outputs=[result_image, verdict_table, summary, report, status], api_name="analyse"
    )

    for trigger in (refresh_button.click, history_tab.select, demo.load):
        trigger(load_history, outputs=[history_table, history_note])
    history_table.select(show_details, outputs=[details, details_people])


if __name__ == "__main__":
    load_model()  # load YOLO once at start-up, so the first analysis isn't slower than the rest
    demo.launch()
