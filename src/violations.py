"""F2 — Violation rules and F3 — annotated image.

Usage:
    python -m src.violations IMAGE [IMAGE ...] [--conf 0.25] [--out outputs]

For each person the model finds, decide whether they wear a helmet and a vest.

People shorter than 15% of the image height are "too far to judge": below that size the
model rarely sees their PPE (verdicts were under 60% correct on the validation set), so
we report them instead of guessing.

An item (helmet, vest, no_helmet, no_vest) belongs to a person when:
- at least half of the item's box lies inside the person's box, and
- the item's centre sits at the right height: helmets in the top third of the box,
  vests between 15% and 70% of the way down.
If both a positive and a negative box match, the more confident one wins.
If nothing matches, the item counts as missing.

The zones were measured on the training labels: they accept 98% of real helmets and
99.6% of real vests, while giving under 2% to a neighbouring person.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detect import DEFAULT_CONF, Detection, detect

# Share of a PPE box that must lie inside the person's box for it to count for that person.
MIN_OVERLAP = 0.5

# Where an item's centre may sit, as a fraction of the person's height (0 = top, 1 = bottom).
ZONES = {
    "helmet": (0.0, 1 / 3),
    "vest": (0.15, 0.70),
}

# People shorter than this share of the image height are too far to judge.
# Calibrated on the 200 validation images: 15-20% is the first height band above 85% correct.
MIN_PERSON_HEIGHT = 0.15

GREEN = (0, 170, 0)  # OpenCV colours are BGR, not RGB
RED = (0, 0, 220)
GREY = (128, 128, 128)


@dataclass
class PersonVerdict:
    """The decision for one detected person.

    has_helmet / has_vest are None when the person is too far to judge: unknown, not missing.
    """

    person: Detection
    has_helmet: bool | None
    has_vest: bool | None

    @property
    def too_far(self) -> bool:
        return self.has_helmet is None

    @property
    def missing(self) -> list[str]:
        items = []
        if self.has_helmet is False:
            items.append("helmet")
        if self.has_vest is False:
            items.append("vest")
        return items

    @property
    def is_violation(self) -> bool:
        return bool(self.missing)

    @property
    def status(self) -> str:
        """One of: compliant, missing helmet, missing vest, missing helmet and vest, too far to judge."""
        if self.too_far:
            return "too far to judge"
        if not self.missing:
            return "compliant"
        return "missing " + " and ".join(self.missing)


def overlap_fraction(item: Detection, x1: float, y1: float, x2: float, y2: float) -> float:
    """Share of the item's box area that lies inside the rectangle (x1, y1, x2, y2)."""
    inter_w = min(item.x2, x2) - max(item.x1, x1)
    inter_h = min(item.y2, y2) - max(item.y1, y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    item_area = (item.x2 - item.x1) * (item.y2 - item.y1)
    return (inter_w * inter_h) / item_area if item_area > 0 else 0.0


def belongs_to(item: Detection, person: Detection, zone: tuple[float, float]) -> bool:
    """True if the item is mostly inside the person's box and its centre is in the zone."""
    if overlap_fraction(item, person.x1, person.y1, person.x2, person.y2) < MIN_OVERLAP:
        return False
    centre_y = (item.y1 + item.y2) / 2
    position = (centre_y - person.y1) / (person.y2 - person.y1)
    low, high = zone
    return low <= position <= high


def best_confidence(detections: list[Detection], class_name: str, person: Detection, zone: tuple[float, float]) -> float:
    """Highest confidence among detections of this class that belong to the person; 0 if none."""
    scores = [d.confidence for d in detections if d.class_name == class_name and belongs_to(d, person, zone)]
    return max(scores, default=0.0)


def wears(detections: list[Detection], person: Detection, item: str) -> bool:
    """True if the item (e.g. 'helmet') is seen on the person with more confidence than 'no_<item>'."""
    zone = ZONES[item]
    present = best_confidence(detections, item, person, zone)
    absent = best_confidence(detections, f"no_{item}", person, zone)
    return present > 0 and present >= absent


def check_people(detections: list[Detection], image_height: int) -> list[PersonVerdict]:
    """Return a verdict for every Person detection.

    image_height is needed because "too far" is relative to the photo, not a pixel count.
    """
    verdicts = []
    for p in detections:
        if p.class_name != "Person":
            continue
        if (p.y2 - p.y1) / image_height < MIN_PERSON_HEIGHT:
            verdicts.append(PersonVerdict(person=p, has_helmet=None, has_vest=None))
        else:
            verdicts.append(
                PersonVerdict(
                    person=p,
                    has_helmet=wears(detections, p, "helmet"),
                    has_vest=wears(detections, p, "vest"),
                )
            )
    return verdicts


def count_verdicts(verdicts: list[PersonVerdict]) -> dict[str, int]:
    """Counts used by the banner, and later by the report, log, alert and digest."""
    too_far = sum(v.too_far for v in verdicts)
    violations = sum(v.is_violation for v in verdicts)
    return {
        "people": len(verdicts),
        "compliant": len(verdicts) - too_far - violations,
        "violations": violations,
        "too_far": too_far,
    }


def summary_text(verdicts: list[PersonVerdict]) -> str:
    """Short count line, e.g. '4 people: 1 compliant, 2 in violation, 1 too far to judge'.

    The too-far count is always shown, even when 0, so distant workers are never silently dropped.
    """
    c = count_verdicts(verdicts)
    noun = "person" if c["people"] == 1 else "people"
    return (
        f"{c['people']} {noun}: {c['compliant']} compliant, "
        f"{c['violations']} in violation, {c['too_far']} too far to judge"
    )


def draw_verdicts(image: np.ndarray, verdicts: list[PersonVerdict]) -> np.ndarray:
    """F3: draw each person green (compliant), red (violation) or grey (too far), plus a summary banner."""
    annotated = image.copy()
    # Scale lines and text with the image, so labels stay readable on large photos.
    scale = max(image.shape[:2]) / 1000
    thickness = max(1, round(2 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for v in verdicts:
        p = v.person
        if v.too_far:
            color, label = GREY, "Too far"
        elif v.is_violation:
            color, label = RED, "No " + " + ".join(v.missing)
        else:
            color, label = GREEN, "OK"
        top_left = (int(p.x1), int(p.y1))
        cv2.rectangle(annotated, top_left, (int(p.x2), int(p.y2)), color, thickness)

        # Filled label background so text is readable on any photo.
        (tw, th), base = cv2.getTextSize(label, font, 0.6 * scale, thickness)
        y_top = top_left[1] - th - base
        if y_top < 0:  # no room above the box: put the label just inside it
            y_top = top_left[1]
        cv2.rectangle(annotated, (top_left[0], y_top), (top_left[0] + tw, y_top + th + base), color, -1)
        cv2.putText(annotated, label, (top_left[0], y_top + th), font, 0.6 * scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Summary banner in the bottom-left corner, where it rarely covers a person's label.
    text = summary_text(verdicts)
    (tw, th), base = cv2.getTextSize(text, font, 0.8 * scale, thickness)
    pad = int(10 * scale)
    height = image.shape[0]
    banner_top = height - (th + base + 2 * pad)
    cv2.rectangle(annotated, (0, banner_top), (tw + 2 * pad, height), (0, 0, 0), -1)
    cv2.putText(annotated, text, (pad, banner_top + pad + th), font, 0.8 * scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PPE compliance per person.")
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

        verdicts = check_people(detect(image, conf=args.conf), image_height=image.shape[0])
        cv2.imwrite(str(args.out / f"{path.stem}_verdicts.jpg"), draw_verdicts(image, verdicts))
        print(f"{path.name}: {summary_text(verdicts)}")
        for i, v in enumerate(verdicts, 1):
            print(f"  person {i} ({v.person.confidence:.2f}): {v.status}")


if __name__ == "__main__":
    main()
