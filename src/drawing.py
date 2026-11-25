"""F3 — Annotated image: each person green (compliant), red (violation) or grey (too far), plus a banner.

Separate from the rules (src/violations.py) so the rules stay pure Python; this module needs OpenCV.
"""

import cv2
import numpy as np

from src.verdicts import PersonVerdict
from src.violations import summary_text

GREEN = (0, 170, 0)  # OpenCV colours are BGR, not RGB
RED = (0, 0, 220)
GREY = (128, 128, 128)


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
