"""F2 — Violation rules: pure Python, no OpenCV and no detection imports.

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

# The types and counts live in src.verdicts and are re-exported here for existing callers.
from src.verdicts import MIN_PERSON_HEIGHT, Detection, PersonVerdict, count_verdicts  # noqa: F401

# Share of a PPE box that must lie inside the person's box for it to count for that person.
MIN_OVERLAP = 0.5

# Where an item's centre may sit, as a fraction of the person's height (0 = top, 1 = bottom).
ZONES = {
    "helmet": (0.0, 1 / 3),
    "vest": (0.15, 0.70),
}

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
