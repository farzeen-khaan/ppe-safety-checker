"""F1 — Detection: run the fine-tuned YOLO model on an image.

Usage:
    python -m src.detect IMAGE [IMAGE ...] [--conf 0.25] [--out outputs]

For each image this saves an annotated copy and a JSON file of detections.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
# Chosen in Phase 2: with the too-far cut-off, 0.25 had the fewest silent misses and false alarms.
DEFAULT_CONF = 0.25


@dataclass
class Detection:
    """One detected object. Box is in pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    """Load the model once; later calls return the same object."""
    return YOLO(str(model_path))


@lru_cache(maxsize=1)
def model_version(model_path: Path = MODEL_PATH) -> str:
    """Short fingerprint of the weights file, saved with every analysis.

    If the model is retrained, old and new results can still be told apart.
    """
    return "best.pt:" + hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]


def detect(image: np.ndarray, conf: float = DEFAULT_CONF) -> list[Detection]:
    """Run the model on a BGR image (as loaded by cv2.imread) and return its detections."""
    model = load_model()
    result = model.predict(image, conf=conf, verbose=False)[0]

    detections = []
    for (x1, y1, x2, y2), cls, score in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        detections.append(
            Detection(
                class_id=int(cls),
                class_name=model.names[int(cls)],
                confidence=round(score, 3),
                x1=round(x1, 1),
                y1=round(y1, 1),
                x2=round(x2, 1),
                y2=round(y2, 1),
            )
        )
    return detections


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the image with every box and its label drawn on it.

    This is a raw debug view of the model output. Per-person green/red verdicts come in F3.
    """
    annotated = image.copy()
    for d in detections:
        # Red for missing-PPE classes, blue for everything else (colours are BGR).
        color = (0, 0, 255) if d.class_name.startswith("no_") else (255, 128, 0)
        top_left = (int(d.x1), int(d.y1))
        cv2.rectangle(annotated, top_left, (int(d.x2), int(d.y2)), color, 2)
        cv2.putText(
            annotated,
            f"{d.class_name} {d.confidence:.2f}",
            (top_left[0], max(top_left[1] - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPE detection on images.")
    parser.add_argument("images", nargs="+", type=Path, help="image files to analyse")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output folder")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping {path}: not a readable image")
            continue

        detections = detect(image, conf=args.conf)
        cv2.imwrite(str(args.out / f"{path.stem}_detected.jpg"), draw_detections(image, detections))
        (args.out / f"{path.stem}_detections.json").write_text(
            json.dumps([asdict(d) for d in detections], indent=2)
        )

        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        print(f"{path.name}: {len(detections)} detections {counts}")


if __name__ == "__main__":
    main()
