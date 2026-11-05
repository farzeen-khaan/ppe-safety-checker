"""Run the full analysis for one photo: F1 detect → F2 rules → F3 image → F4 report → F5 save.

Usage:
    python -m src.pipeline IMAGE [IMAGE ...] [--conf 0.25] [--no-llm] [--out outputs]

The work is split in two so the web UI can show results early:
- inspect_image():   detection, rules and annotated image (fast, about 1 s)
- finish_analysis(): incident report and save (slow when Gemini is slow)
analyse_image() runs both. Phase 6 adds F7 (alert) after the save.
"""

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detect import DEFAULT_CONF, detect, model_version
from src.report import Report, generate_report
from src.storage import save_analysis
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
    inspection: Inspection, image_name: str, use_llm: bool = True, save: bool = True
) -> AnalysisResult:
    """F4-F5: write the report and save the analysis. A failed save is reported, not raised."""
    report = generate_report(inspection.verdicts, use_llm=use_llm)
    processing_ms = round((time.perf_counter() - inspection.started) * 1000)

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
            )
        except Exception as exc:  # the user still gets the verdicts and the report
            save_error = f"{type(exc).__name__}: {exc}"
    return AnalysisResult(analysis_id, inspection.verdicts, report, inspection.annotated, processing_ms, save_error)


def analyse_image(
    image: np.ndarray,
    image_name: str,
    conf: float = DEFAULT_CONF,
    use_llm: bool = True,
    save: bool = True,
) -> AnalysisResult:
    """Analyse one BGR image and (by default) save it to the incident log."""
    return finish_analysis(inspect_image(image, conf=conf), image_name, use_llm=use_llm, save=save)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse photos and save them to the incident log.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--no-llm", action="store_true", help="use the template report only")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="folder for annotated images")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping {path}: not a readable image")
            continue
        result = analyse_image(image, path.name, conf=args.conf, use_llm=not args.no_llm)
        cv2.imwrite(str(args.out / f"{path.stem}_verdicts.jpg"), result.annotated)
        saved = f"#{result.analysis_id}" if result.save_error is None else f"NOT SAVED ({result.save_error})"
        print(
            f"{saved} {path.name}: {summary_text(result.verdicts)}; "
            f"severity {result.report.severity}; report by {result.report.model or 'template'}; "
            f"{result.processing_ms} ms"
        )


if __name__ == "__main__":
    main()
