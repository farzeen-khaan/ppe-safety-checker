"""One photo through the system, in two calls (the web app's two requests).

Usage:
    python -m src.pipeline IMAGE [IMAGE ...] [--conf 0.25] [--no-llm] [--no-alert] [--out outputs]

- analyze():       detect (ONNX) → rules → annotated image → save → alert. Fast (about a second
                   when warm); the alert is sent before it returns. The saved summary is the
                   standard one, marked PENDING, because the written summary is slow.
- write_summary(): the written summary for a saved analysis, from its stored verdicts, saved
                   over the standard one. Calling it again returns the saved automated summary.

Nothing here keeps state between calls, so each can run in a separate serverless request.
"""

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.alerts import alert_after_save
from src.drawing import draw_verdicts
from src.onnx_detect import DEFAULT_CONF, detect, model_version
from src.report import Report, build_payload, generate_report, template_report
from src.storage import get_analysis, now_utc, save_analysis, update_report
from src.verdicts import Detection, PersonVerdict
from src.violations import check_people, summary_text

PENDING = "summary not written yet"  # report_error until write_summary() runs


@dataclass
class Inspection:
    verdicts: list[PersonVerdict]
    annotated: np.ndarray  # BGR image with green/red/grey boxes
    image_width: int
    image_height: int
    image_sha256: str
    conf: float
    started: float  # time.perf_counter() at the start


@dataclass
class AnalysisResult:
    analysis_id: int | None  # None when not saved
    verdicts: list[PersonVerdict]
    report: Report  # the standard summary; the written one comes from write_summary()
    annotated: np.ndarray
    processing_ms: int  # detection to saved (the alert is not included)
    save_error: str | None = None
    alert_status: str = "not_needed"  # not_needed, disabled, sent or failed


def inspect_image(image: np.ndarray, conf: float = DEFAULT_CONF) -> Inspection:
    """Detections, per-person verdicts and the annotated image for one BGR image."""
    started = time.perf_counter()
    height, width = image.shape[:2]
    verdicts = check_people(detect(image, conf=conf), image_height=height)
    return Inspection(
        verdicts=verdicts,
        annotated=draw_verdicts(image, verdicts),
        image_width=width,
        image_height=height,
        # Fingerprint of the decoded pixels: the same photo gets the same value however it arrived.
        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        conf=conf,
        started=started,
    )


def standard_report(verdicts: list[PersonVerdict]) -> Report:
    """Rule-based severity and wording, saved until the written summary replaces it."""
    facts = build_payload(verdicts)
    summary, action = template_report(facts)
    return Report(facts["severity"], summary, action, source="template", error=PENDING)


def analyze(
    image: np.ndarray, image_name: str, conf: float = DEFAULT_CONF, save: bool = True, alert: bool = True
) -> AnalysisResult:
    """The first call: verdicts, saved, alert sent. A failed save is reported, not raised.

    The alert still goes out if the save failed: the unsafe worker is real either way.
    save=False (a dry run) never alerts.
    """
    inspection = inspect_image(image, conf=conf)
    report = standard_report(inspection.verdicts)
    created_at = now_utc()  # one timestamp for the log row and the alert

    analysis_id, save_error = None, None
    if save:
        try:
            analysis_id = save_analysis(
                image_name=image_name,
                image_sha256=inspection.image_sha256,
                image_width=inspection.image_width,
                image_height=inspection.image_height,
                verdicts=inspection.verdicts,
                report=report,
                model_version=model_version(),
                conf_threshold=inspection.conf,
                processing_ms=round((time.perf_counter() - inspection.started) * 1000),
                created_at=created_at,
            )
        except Exception as exc:  # the user still gets the verdicts
            save_error = f"{type(exc).__name__}: {exc}"
    processing_ms = round((time.perf_counter() - inspection.started) * 1000)

    alert_status = "not_needed"
    if save and alert:
        alert_status = alert_after_save(
            analysis_id, inspection.verdicts, report, inspection.annotated, image_name, created_at
        )
    return AnalysisResult(
        analysis_id, inspection.verdicts, report, inspection.annotated, processing_ms, save_error, alert_status
    )


def stored_verdicts(analysis: dict) -> list[PersonVerdict]:
    """The saved per-person verdicts, rebuilt for the rules and the report."""
    return [
        PersonVerdict(
            Detection(5, "Person", p["confidence"], p["x1"], p["y1"], p["x2"], p["y2"]), p["has_helmet"], p["has_vest"]
        )
        for p in analysis["people"]
    ]


def stored_report(analysis: dict) -> Report:
    a = analysis
    return Report(a["severity"], a["report_summary"], a["report_action"], a["report_source"],
                  a.get("report_provider"), a["report_model"], a["report_error"])


def write_summary(analysis_id: int, use_llm: bool = True) -> Report:
    """The second call: write the summary for a saved analysis and save it. KeyError if the id is unknown.

    An analysis that already has an automated summary keeps it (no second model call).
    """
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise KeyError(f"no analysis with id {analysis_id}")
    if analysis["report_source"] == "llm":
        return stored_report(analysis)
    report = generate_report(stored_verdicts(analysis), use_llm=use_llm)
    update_report(analysis_id, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse photos and save them to the incident log.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--no-llm", action="store_true", help="keep the standard summary")
    parser.add_argument("--no-alert", action="store_true", help="do not send Discord alerts")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="folder for annotated images")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping {path}: not a readable image")
            continue
        result = analyze(image, path.name, conf=args.conf, alert=not args.no_alert)
        cv2.imwrite(str(args.out / f"{path.stem}_verdicts.jpg"), result.annotated)
        if result.save_error is not None:
            print(f"NOT SAVED ({result.save_error}) {path.name}: {summary_text(result.verdicts)}")
            continue
        report = write_summary(result.analysis_id, use_llm=not args.no_llm)
        print(
            f"#{result.analysis_id} {path.name}: {summary_text(result.verdicts)}; severity {report.severity}; "
            f"summary {'automated' if report.automated else 'standard'}; {result.processing_ms} ms; "
            f"alert: {result.alert_status}"
        )


if __name__ == "__main__":
    main()
