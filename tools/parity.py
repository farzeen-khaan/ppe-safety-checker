"""ONNX vs PyTorch parity on the 20-image check set (dev tool; needs requirements-dev.txt).

Usage:
    python -m tools.parity            # prints per-image numbers and a summary

The check set is the 20 validation images of the Phase 2 hand check. Each ONNX box is
matched to the PyTorch box of the same class with the highest IoU (greedy, best first).
tests/test_onnx_parity.py asserts the tolerances on the same numbers.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src import detect as torch_detect
from src import onnx_detect
from src.violations import check_people

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "construction-ppe" / "images" / "val"
CHECK_SET = [
    "source-223442b1ecf2d156710f2b828bb05db2", "source-4e4870d31f0c2ff366debd49c66a0803",
    "source-6af78fcdef2424c52bcfefc5f4cf220d", "source-8497dd5065ce426e2d223085b4a773a0",
    "source-ab78be39fcf54451aa89437c75b82cb2", "source-ce9bda3a2d92dbfcd2a2256b6d8e17f4",
    "ultralytics-ppe-nov2026-046", "ultralytics-ppe-nov2026-096", "ultralytics-ppe-nov2026-146",
    "ultralytics-ppe-nov2026-196", "ultralytics-ppe-nov2026-246", "ultralytics-ppe-nov2026-296",
    "ultralytics-ppe-nov2026-346", "ultralytics-ppe-nov2026-396", "ultralytics-ppe-nov2026-446",
    "ultralytics-ppe-nov2026-496", "ultralytics-ppe-nov2026-546", "ultralytics-ppe-nov2026-596",
    "ultralytics-ppe-nov2026-646", "ultralytics-ppe-nov2026-696",
]


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda d: (d.x2 - d.x1) * (d.y2 - d.y1)  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


@dataclass
class ImageParity:
    name: str
    torch_boxes: int
    onnx_boxes: int
    matched: int
    unmatched: list  # (side, class, confidence) for boxes with no partner
    min_iou: float
    max_conf_delta: float
    verdicts_equal: bool


def compare(name: str, conf: float = onnx_detect.DEFAULT_CONF) -> ImageParity:
    path = next(VAL.glob(name + ".*"))
    image = cv2.imread(str(path))
    t = torch_detect.detect(image, conf=conf)
    o = onnx_detect.detect(image, conf=conf)
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(o) for j, b in enumerate(t)
                    if a.class_name == b.class_name), reverse=True)
    used_o, used_t, matches = set(), set(), []
    for v, i, j in pairs:
        if v > 0 and i not in used_o and j not in used_t:
            used_o.add(i), used_t.add(j), matches.append((v, o[i], t[j]))
    unmatched = [("onnx", d.class_name, d.confidence) for i, d in enumerate(o) if i not in used_o]
    unmatched += [("torch", d.class_name, d.confidence) for j, d in enumerate(t) if j not in used_t]
    h = image.shape[0]
    tv = [v.status for v in check_people(t, h)]
    ov = [v.status for v in check_people(o, h)]
    return ImageParity(
        name, len(t), len(o), len(matches), unmatched,
        min((m[0] for m in matches), default=1.0),
        max((abs(m[1].confidence - m[2].confidence) for m in matches), default=0.0),
        sorted(tv) == sorted(ov),
    )


def main() -> None:
    results = [compare(n) for n in CHECK_SET]
    for r in results:
        print(f"{r.name:42} torch {r.torch_boxes:3} onnx {r.onnx_boxes:3} matched {r.matched:3} "
              f"min IoU {r.min_iou:.4f} max Δconf {r.max_conf_delta:.4f} verdicts {'same' if r.verdicts_equal else 'DIFF'}"
              + (f"  unmatched {r.unmatched}" if r.unmatched else ""))
    all_matches = sum(r.matched for r in results)
    print(f"\n{len(results)} images, {sum(r.torch_boxes for r in results)} torch boxes, "
          f"{sum(r.onnx_boxes for r in results)} onnx boxes, {all_matches} matched; "
          f"worst IoU {min(r.min_iou for r in results):.4f}; worst Δconf {max(r.max_conf_delta for r in results):.4f}; "
          f"unmatched {sum(len(r.unmatched) for r in results)}; "
          f"verdicts identical on {sum(r.verdicts_equal for r in results)} of {len(results)} images")


if __name__ == "__main__":
    main()
