"""Tests for the deployed ONNX detector: preprocessing, postprocessing, and parity with PyTorch.

The parity tests need the PyTorch detector (requirements-dev.txt) and the dataset (data/); they
are skipped without them. Everything else runs with the deploy requirements only.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import onnx_detect
from src.verdicts import Detection

ROOT = Path(__file__).resolve().parent.parent


# --- Letterbox: Ultralytics' rounding, grey padding, RGB, 0..1 ---

def test_letterbox_landscape():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # pure blue in BGR
    blob, r, (left, top) = onnx_detect.letterbox(image, 640)
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert (r, left, top) == (1.0, 0, 140)
    assert blob[0, :, 0, 0].tolist() == pytest.approx([114 / 255] * 3)  # padding
    assert blob[0, :, 320, 320].tolist() == pytest.approx([0.0, 0.0, 1.0])  # BGR blue became RGB blue


def test_letterbox_scales_large_portrait_photos_down():
    blob, r, (left, top) = onnx_detect.letterbox(np.zeros((1280, 960, 3), dtype=np.uint8), 640)
    assert blob.shape == (1, 3, 640, 640) and r == 0.5 and (left, top) == (80, 0)


# --- Postprocessing ---

def test_nms_keeps_the_best_box_per_overlapping_group_and_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    classes = np.array([0, 0, 0, 1])  # the last box overlaps the first but is another class
    assert onnx_detect.nms(boxes, scores, classes, iou=0.5).tolist() == [0, 2, 3]


def test_nms_of_nothing():
    assert onnx_detect.nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0)).size == 0


def test_decode_raw_turns_centre_boxes_into_corners_with_the_best_class():
    output = np.zeros((1, 6, 1), dtype=np.float32)  # 4 box values + 2 classes, 1 anchor
    output[0, :, 0] = [50, 40, 20, 10, 0.2, 0.7]
    assert onnx_detect.decode_raw(output).tolist() == [pytest.approx([40, 35, 60, 45, 0.7, 1])]


# --- The model ---

def test_model_loads_once_and_has_the_training_classes():
    model = onnx_detect.load_model()
    assert onnx_detect.load_model() is model  # cached for the life of the instance
    assert model.size == 640 and not model.end2end
    assert {"Person", "helmet", "vest", "no_helmet", "no_vest"} <= set(model.names.values())
    assert onnx_detect.MODEL_PATH == ROOT / "models" / "best.onnx"  # resolved from the package, not the cwd


def test_model_version_fingerprints_the_onnx_file():
    assert onnx_detect.model_version().startswith("best.onnx:") and len(onnx_detect.model_version()) == 22


def test_a_blank_image_has_no_detections():
    assert onnx_detect.detect(np.full((480, 640, 3), 255, dtype=np.uint8)) == []


def test_the_deployed_detector_never_imports_torch():
    script = ("import sys, numpy as np; from src import onnx_detect, pipeline; "
              "onnx_detect.detect(np.zeros((64, 64, 3), dtype=np.uint8)); "
              "print(sorted(m for m in ('torch', 'ultralytics', 'gradio', 'src.detect') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# --- Parity with PyTorch on the 20-image check set (dev requirements + dataset) ---

try:
    from ultralytics import YOLO  # noqa: F401
    from tools import parity as parity_tool
    HAVE_TORCH = (ROOT / "data").exists()
except ImportError:
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs requirements-dev.txt (PyTorch) and the dataset in data/")
RULE_CLASSES = {"Person", "helmet", "vest", "no_helmet", "no_vest"}


def torch_detections(image, rect: bool) -> list[Detection]:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "best.pt"))
    r = model.predict(image, conf=onnx_detect.DEFAULT_CONF, verbose=False, rect=rect)[0]
    return [Detection(int(c), model.names[int(c)], round(float(s), 3), *[round(v, 1) for v in b])
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def matched(a: list[Detection], b: list[Detection], classes=None):
    """Greedy same-class matching by IoU. Returns (pairs as (iou, Δconf), unmatched count)."""
    a = [d for d in a if classes is None or d.class_name in classes]
    b = [d for d in b if classes is None or d.class_name in classes]
    pairs = sorted(((parity_tool.iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                    if x.class_name == y.class_name), reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if v > 0 and i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j), out.append((v, abs(a[i].confidence - b[j].confidence)))
    return out, len(a) - len(used_a) + len(b) - len(used_b)


@pytest.fixture(scope="module")
def check_set():
    return [cv2.imread(str(next(parity_tool.VAL.glob(n + ".*")))) for n in parity_tool.CHECK_SET]


@needs_torch
def test_same_input_same_boxes(check_set):
    """Same square letterbox: the ONNX model is the PyTorch model (measured: IoU ≥ 0.9995, Δconf 0.000)."""
    pairs, unmatched = [], 0
    for image in check_set:
        p, u = matched(onnx_detect.detect(image), torch_detections(image, rect=False))
        pairs += p
        unmatched += u
    assert unmatched == 0
    assert min(v for v, _ in pairs) >= 0.99
    assert max(d for _, d in pairs) <= 0.005


@needs_torch
def test_same_verdicts_as_the_pytorch_app(check_set):
    """Against PyTorch as the app used it (minimal-padding letterbox), on the classes the rules use:
    every box matched, IoU ≥ 0.90, Δconf ≤ 0.20 (measured 0.9255 and 0.177), identical verdicts on all 20."""
    from src.violations import check_people

    pairs = []
    for image in check_set:
        o, t = onnx_detect.detect(image), torch_detections(image, rect=True)
        p, unmatched = matched(o, t, RULE_CLASSES)
        assert unmatched == 0
        pairs += p
        h = image.shape[0]
        assert sorted(v.status for v in check_people(o, h)) == sorted(v.status for v in check_people(t, h))
    assert min(v for v, _ in pairs) >= 0.90
    assert max(d for _, d in pairs) <= 0.20
