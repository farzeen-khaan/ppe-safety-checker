"""Run the full analysis for one photo: F1 detect → F2 rules → F3 image → F4 report → F5 save → F7 alert.

Usage:
    python -m src.pipeline IMAGE [IMAGE ...] [--conf 0.25] [--no-llm] [--no-alert] [--out outputs]

The work is split in two so the web UI can show results early:
- inspect_image():   detection, rules and annotated image (fast, about 1 s)
- finish_analysis(): incident report, save, then start the alert (slow when Gemini is slow)
analyse_image() runs both. The alert is sent on a background thread and never delays the result.
"""

import argparse
import hashlib
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.alerts import alert_after_save
from src.detect import DEFAULT_CONF, detect, model_version
from src.report import Report, generate_report
from src.storage import now_utc, save_analysis
from src.violations import PersonVerdict, check_people, draw_verdicts, summary_text


@dataclass
class Inspection:
    """Result of the fast half: everything except the report and the save."""

    verdicts: list[PersonVerdict]
    annotated: np.ndarray  # BGR image with green/red/grey boxes
    image_width: int
    image_height: int
    image_sha256: str
    conf: float
    started: float  # time.perf_counter() at the start, to measure the whole analysis


@dataclass
class AnalysisResult:
    analysis_id: int | None  # None when not saved
    verdicts: list[PersonVerdict]
    report: Report
    annotated: np.ndarray
    processing_ms: int
    save_error: str | None = None  # set if the incident log could not be written
    alert_status: str = "not_needed"  # not_needed, disabled or sending (see alerts.alert_after_save)
    alert: Future | None = None  # the background send; .result() is 'sent' or 'failed'


def inspect_image(image: np.ndarray, conf: float = DEFAULT_CONF) -> Inspection:
    """F1-F3 for one BGR image: detections, per-person verdicts and the annotated image."""
    started = time.perf_counter()
    height, width = image.shape[:2]
    verdicts = check_people(detect(image, conf=conf), image_height=height)
    return Inspection(
        verdicts=verdicts,
        annotated=draw_verdicts(image, verdicts),
        image_width=width,
        image_height=height,
        # Fingerprint of the decoded pixels: the same photo gets the same value
        # whether it came from a file or from the web UI.
        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        conf=conf,
        started=started,
    )


def finish_analysis(
    inspection: Inspection, image_name: str, use_llm: bool = True, save: bool = True, alert: bool = True
) -> AnalysisResult:
    """F4, F5, F7: write the report, save the analysis, start the alert.

    A failed save is reported, not raised. The alert still goes out if the save failed:
    the unsafe worker is real either way. save=False (a dry run) never alerts.
    """
    report = generate_report(inspection.verdicts, use_llm=use_llm)
    processing_ms = round((time.perf_counter() - inspection.started) * 1000)
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
                processing_ms=processing_ms,
                created_at=created_at,
            )
        except Exception as exc:  # the user still gets the verdicts and the report
            save_error = f"{type(exc).__name__}: {exc}"

    alert_status, alert_future = "not_needed", None
    if save and alert:
        alert_status, alert_future = alert_after_save(
            analysis_id, inspection.verdicts, report, inspection.annotated, image_name, created_at
        )
    return AnalysisResult(
        analysis_id, inspection.verdicts, report, inspection.annotated, processing_ms, save_error,
        alert_status, alert_future,
    )


def analyse_image(
    image: np.ndarray,
    image_name: str,
    conf: float = DEFAULT_CONF,
    use_llm: bool = True,
    save: bool = True,
    alert: bool = True,
) -> AnalysisResult:
    """Analyse one BGR image and (by default) save it to the incident log."""
    return finish_analysis(inspect_image(image, conf=conf), image_name, use_llm=use_llm, save=save, alert=alert)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse photos and save them to the incident log.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--no-llm", action="store_true", help="use the template report only")
    parser.add_argument("--no-alert", action="store_true", help="do not send Discord alerts")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="folder for annotated images")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping {path}: not a readable image")
            continue
        result = analyse_image(image, path.name, conf=args.conf, use_llm=not args.no_llm, alert=not args.no_alert)
        cv2.imwrite(str(args.out / f"{path.stem}_verdicts.jpg"), result.annotated)
        saved = f"#{result.analysis_id}" if result.save_error is None else f"NOT SAVED ({result.save_error})"
        print(
            f"{saved} {path.name}: {summary_text(result.verdicts)}; "
            f"severity {result.report.severity}; report by {result.report.model or 'template'}; "
            f"{result.processing_ms} ms"
        )
        if result.alert is not None:  # on the command line, waiting is fine
            print(f"  alert: {result.alert.result()}")
        elif result.alert_status == "disabled":
            print("  alert: disabled")


if __name__ == "__main__":
    main()
