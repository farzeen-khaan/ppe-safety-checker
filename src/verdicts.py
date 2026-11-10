"""Data types shared by detection, the rules, the report, the log and the digest.

Standard library only. The digest (run by GitHub Actions) imports the log and the report,
which need these types but must not pull in torch, Ultralytics or OpenCV. So the types live
here, and src.detect / src.violations re-export them for existing callers.
"""

from dataclasses import dataclass

# People shorter than this share of the image height are too far to judge.
# Calibrated on the 200 validation images: 15-20% is the first height band above 85% correct.
MIN_PERSON_HEIGHT = 0.15


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
