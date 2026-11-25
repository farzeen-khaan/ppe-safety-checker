"""F6 — Web UI: Gradio app with Analyse and History tabs.

Run locally:
    python app.py           # then open http://127.0.0.1:7860

Analyse shows the annotated image and verdicts first (about 1 s), then the summary when it
is ready, so a slow model never blocks the result. History is text only: the incident log
stores no photos.
"""

import html
from datetime import datetime, timezone
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.detect import load_model
from src.discord_webhook import COLOURS, GREY, shorten_name, summary_note
from src.pipeline import finish_analysis, inspect_image
from src.report import Report, compute_severity
from src.storage import PKT, StorageUnavailable, get_analysis, list_analyses
from src.violations import MIN_PERSON_HEIGHT, PersonVerdict, count_verdicts, summary_text

MAX_UPLOAD_MB = 20
HISTORY_LIMIT = 100

VERDICT_HEADERS = ["#", "Verdict", "Confidence"]
VERDICT_WIDTHS = ["12%", "58%", "30%"]
HISTORY_HEADERS = ["ID", "Time (PKT)", "Photo", "People", "OK", "In violation", "Not checked", "Severity",
                   "Summary", "Alert"]
# Shown after the status line. The alert may still be sending; History shows the outcome.
ALERT_NOTES = {
    "sending": " A Discord alert is being sent; History shows when it has gone.",
    "disabled": " Alerts are off, so no Discord alert was sent.",
}
UNREACHABLE = "The incident log can't be reached right now."


def _hex(colour: int) -> str:
    return f"#{colour:06x}"


CSS = f"""
.app-header {{ padding: 4px 0 12px; }}
.app-header h1 {{ font-size: 1.7rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }}
.app-header p {{ margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 1rem; }}
.legend {{ font-size: 0.85rem; color: var(--body-text-color-subdued); margin-top: 6px; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 10px; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.875rem;
          color: #fff; line-height: 1.5; }}
.sev-High {{ background: {_hex(COLOURS["High"])}; }}
.sev-Medium {{ background: {_hex(COLOURS["Medium"])}; }}
.sev-None {{ background: {_hex(COLOURS["None"])}; }}
.badge-grey {{ background: {_hex(GREY)}; }}
.counts {{ margin: 8px 0 0; font-size: 0.95rem; }}
.card {{ border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px 18px;
         background: var(--block-background-fill); }}
.card h3 {{ margin: 0 0 8px; font-size: 1.05rem; }}
.card p {{ margin: 0 0 8px; line-height: 1.55; }}
.card .note {{ margin: 10px 0 0; font-size: 0.8rem; color: var(--body-text-color-subdued); }}
.empty {{ text-align: center; padding: 56px 20px; border: 1px dashed var(--border-color-primary); border-radius: 12px;
          color: var(--body-text-color-subdued); }}
.empty strong {{ display: block; font-size: 1.05rem; color: var(--body-text-color); margin-bottom: 6px; }}
.loading {{ color: var(--body-text-color-subdued); }}
.loading .bar {{ height: 10px; border-radius: 6px; margin: 10px 0; background: linear-gradient(90deg,
    var(--border-color-primary) 25%, var(--block-background-fill) 50%, var(--border-color-primary) 75%);
    background-size: 200% 100%; animation: shimmer 1.4s infinite; }}
@keyframes shimmer {{ from {{ background-position: 200% 0; }} to {{ background-position: -200% 0; }} }}
.status p {{ margin: 0; font-size: 0.9rem; }}
.readable, .readable * {{ --font-mono: var(--font); }}  /* tables read as text, not code */
.readable td, .readable th, .readable td *, .readable th * {{ font-family: var(--font) !important; }}
@media (max-width: 640px) {{
  .app-header h1 {{ font-size: 1.35rem; }}
  .card {{ padding: 14px; }}
  .empty {{ padding: 32px 14px; }}
}}
"""

THEME = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

HEADER = f"""
<div class="app-header">
  <h1>PPE Safety Checker</h1>
  <p>Checks every worker in a construction-site photo for a helmet and a high-visibility vest.</p>
  <div class="legend">
    <span class="dot" style="background:{_hex(COLOURS["None"])};margin-left:0"></span>Compliant
    <span class="dot" style="background:{_hex(COLOURS["High"])}"></span>Violation
    <span class="dot" style="background:{_hex(GREY)}"></span>Not checked: smaller than {MIN_PERSON_HEIGHT:.0%} of the photo height
  </div>
</div>
"""

EMPTY_STATE = """
<div class="empty"><strong>No photo analysed yet</strong>
Upload a site photo and press Analyse. Verdicts appear here first, then the summary.</div>
"""

SUMMARY_LOADING = """
<div class="card loading"><h3>Summary</h3>Writing the summary…<div class="bar"></div><div class="bar" style="width:70%"></div></div>
"""


# --- Helpers (plain functions, testable without a browser) ---

def read_image(path: str | None) -> np.ndarray:
    """Load an uploaded file as a BGR image, or raise gr.Error with a clear message."""
    if not path:
        raise gr.Error("Please upload a photo first.")
    file = Path(path)
    if file.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise gr.Error(f"“{file.name}” is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller photo.")
    # cv2.imread fails on Windows paths with non-English characters; imdecode doesn't.
    image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise gr.Error(f"“{file.name}” is not an image this app can read. Please upload a JPG, PNG or WEBP photo.")
    return image


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """OpenCV works in BGR; browsers show RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def display_status(v: PersonVerdict) -> str:
    return "Not checked (too far)" if v.too_far else v.status.capitalize()


def verdict_rows(verdicts: list[PersonVerdict]) -> list[list]:
    return [[i, display_status(v), f"{v.person.confidence:.2f}"] for i, v in enumerate(verdicts, 1)]


def severity_badge(severity: str) -> str:
    return f'<span class="badge sev-{severity}">Severity: {severity}</span>'


def badge_html(verdicts: list[PersonVerdict], severity: str) -> str:
    """Severity badge, 'N not checked' in grey, and the count line."""
    if not verdicts:
        return ('<div class="badges"><span class="badge badge-grey">No workers detected</span></div>'
                '<p class="counts">Try a closer or clearer photo of the site.</p>')
    c = count_verdicts(verdicts)
    badges = [severity_badge(severity)]
    if c["too_far"]:
        badges.append(f'<span class="badge badge-grey">{c["too_far"]} not checked</span>')
    return f'<div class="badges">{"".join(badges)}</div><p class="counts">{html.escape(summary_text(verdicts))}</p>'


def summary_card(summary: str, action: str, automated: bool) -> str:
    """The written summary, with the disclosure note. Escaped: the text may come from a model."""
    return (
        '<div class="card"><h3>Summary</h3>'
        f"<p>{html.escape(summary)}</p>"
        f"<p><strong>Recommended action:</strong> {html.escape(action)}</p>"
        f'<p class="note">{summary_note(automated)}</p></div>'
    )


def report_card(report: Report) -> str:
    return summary_card(report.summary, report.recommended_action, report.automated)


def to_pkt(utc_text: str) -> str:
    """'2026-11-05T13:00:00Z' (stored UTC) → '2026-11-05 18:00' (Pakistan time)."""
    moment = datetime.strptime(utc_text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return moment.astimezone(PKT).strftime("%Y-%m-%d %H:%M")


def summary_kind(report_source: str) -> str:
    return "automated" if report_source == "llm" else "standard"


def history_rows(rows: list[dict]) -> list[list]:
    return [
        [
            r["id"], to_pkt(r["created_at"]), shorten_name(r["image_name"], 32), r["people_count"],
            r["compliant_count"], r["violation_count"], r["too_far_count"], r["severity"],
            summary_kind(r["report_source"]), r["alert_status"],
        ]
        for r in rows
    ]


# --- Event handlers ---

def analyse(path: str | None):
    """Generator: a loading state, then image and verdicts, then the summary."""
    image = read_image(path)
    hidden, shown = gr.update(visible=False), gr.update(visible=True)
    yield (hidden, gr.update(value='<p class="loading">Checking each worker…</p>', visible=True), hidden, hidden,
           hidden, "Checking each worker…")

    inspection = inspect_image(image)
    verdicts = inspection.verdicts
    badge = badge_html(verdicts, compute_severity(verdicts))
    image_update = gr.update(value=to_rgb(inspection.annotated), visible=True)
    table_update = gr.update(value=verdict_rows(verdicts), visible=bool(verdicts))
    yield (hidden, gr.update(value=badge, visible=True), image_update, table_update,
           gr.update(value=SUMMARY_LOADING, visible=True), "Verdicts ready. Writing the summary…")

    result = finish_analysis(inspection, image_name=Path(path).name)
    if result.save_error:
        status = f"Analysed in {result.processing_ms / 1000:.1f} s, but **not saved** to the log: {result.save_error}"
    else:
        status = f"Analysed in {result.processing_ms / 1000:.1f} s. Saved as analysis #{result.analysis_id}."
    status += ALERT_NOTES.get(result.alert_status, "")
    yield (hidden, shown, shown, gr.update(visible=bool(verdicts)),
           gr.update(value=report_card(result.report), visible=True), status)


def load_history():
    try:
        rows = list_analyses(limit=HISTORY_LIMIT)
    except StorageUnavailable:  # e.g. Neon waking up: say so, keep the app running
        return [], f"{UNREACHABLE} Press Refresh in a moment."
    if not rows:
        return [], "No analyses yet. Analyse a photo first."
    return history_rows(rows), f"{len(rows)} most recent analyses, newest first. Select a row to see its summary."


def show_details(evt: gr.SelectData):
    """When a History row is clicked: its summary and per-person verdicts."""
    if not evt.row_value:
        return "", []
    try:
        analysis = get_analysis(int(evt.row_value[0]))
    except StorageUnavailable:
        return f"<p>{UNREACHABLE} Try again in a moment.</p>", []
    if analysis is None:
        return "<p>That analysis no longer exists.</p>", []
    a = analysis
    heading = (
        f'<p><strong>Analysis #{a["id"]}</strong> · {html.escape(a["image_name"])} · {to_pkt(a["created_at"])} PKT</p>'
        f'<div class="badges">{severity_badge(a["severity"])}</div>'
    )
    card = summary_card(a["report_summary"], a["report_action"], a["report_source"] == "llm")
    meta = (f'<p class="legend">Alert: {a["alert_status"]} · confidence threshold {a["conf_threshold"]} · '
            f'{a["processing_ms"] / 1000:.1f} s</p>')
    people = [[p["person_index"], "Not checked (too far)" if p["has_helmet"] is None else p["status"].capitalize(),
               f"{p['confidence']:.2f}"] for p in a["people"]]
    return heading + card + meta, people


# --- Layout ---

with gr.Blocks(title="PPE Safety Checker") as demo:
    gr.HTML(HEADER)

    with gr.Tab("Analyse"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=300):
                upload = gr.File(label="Site photo (JPG, PNG, WEBP)", file_types=["image"], type="filepath")
                analyse_button = gr.Button("Analyse", variant="primary", size="lg")
                status = gr.Markdown(elem_classes="status")
            with gr.Column(scale=3, min_width=300):
                empty = gr.HTML(EMPTY_STATE)
                badge = gr.HTML(visible=False)
                result_image = gr.Image(type="numpy", interactive=False, show_label=False, visible=False)
                verdict_table = gr.Dataframe(headers=VERDICT_HEADERS, interactive=False, label="Per-person verdicts",
                                             visible=False, wrap=True, column_widths=VERDICT_WIDTHS,
                                             elem_classes="readable")
                summary = gr.HTML(visible=False)

    with gr.Tab("History") as history_tab:
        with gr.Row():
            refresh_button = gr.Button("Refresh", scale=0, size="sm")
            history_note = gr.Markdown()
        history_table = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False, label="Past analyses", wrap=True,
                                     elem_classes="readable")
        details = gr.HTML()
        details_people = gr.Dataframe(headers=VERDICT_HEADERS, interactive=False, label="Per-person verdicts",
                                      column_widths=VERDICT_WIDTHS, elem_classes="readable")

    # Only the button runs an analysis, so each saved analysis is a deliberate click.
    analyse_button.click(
        analyse, inputs=upload, outputs=[empty, badge, result_image, verdict_table, summary, status],
        api_name="analyse", show_progress="minimal",
    )
    for trigger in (refresh_button.click, history_tab.select, demo.load):
        trigger(load_history, outputs=[history_table, history_note])
    history_table.select(show_details, outputs=[details, details_people])


if __name__ == "__main__":
    load_model()  # once at start-up, so the first analysis isn't slower than the rest
    demo.launch(theme=THEME, css=CSS)
