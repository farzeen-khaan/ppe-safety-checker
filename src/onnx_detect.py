"""F1 — Detection with ONNX Runtime on CPU: the deployed detector. Same interface as src/detect.py.

    detections = detect(image_bgr, conf=0.25)   # -> list[Detection], boxes in original pixels

No torch or Ultralytics: preprocessing (letterbox) and postprocessing are numpy and OpenCV.
The model is loaded once per process (one serverless instance) from models/best.onnx,
resolved relative to this file, so the same path works locally and on Vercel.

The exported YOLO26 model is end-to-end (metadata end2end=True, output 1x300x6 =
x1, y1, x2, y2, score, class in letterboxed pixels): like Ultralytics, keep rows with
score > conf and apply no NMS. nms() handles exports that are not end-to-end.
Export command and parity results: decisions.md, Phase 8b.
"""

import ast
import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.verdicts import Detection

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.onnx"
DEFAULT_CONF = 0.25  # as src/detect.py: fewest silent misses and false alarms (decisions.md)
IOU = 0.7  # Ultralytics' default NMS threshold, for non-end-to-end exports
MAX_DET = 300
PAD_VALUE = 114  # Ultralytics' letterbox grey


class Model:
    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"])  # a dict literal written by the exporter
        self.end2end = meta.get("end2end") == "True"
        self.input_name = self.session.get_inputs()[0].name
        self.size = self.session.get_inputs()[0].shape[2]  # 640: the export is fixed-size


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH) -> Model:
    """Load once; later calls in the same process return the same session."""
    return Model(model_path)


@lru_cache(maxsize=1)
def model_version(model_path: Path = MODEL_PATH) -> str:
    """Fingerprint of the weights, saved with every analysis: 'best.onnx:1a2b3c4d5e6f'."""
    return f"{model_path.name}:" + hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping the aspect ratio, pad to size x size with grey (Ultralytics' rounding).

    Returns (NCHW float32 RGB in 0..1, scale, (left, top) padding).
    """
    h, w = image_bgr.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if (w, h) != (new_w, new_h):
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(image_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR -> RGB, HWC -> NCHW
    return np.ascontiguousarray(blob), r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float = IOU) -> np.ndarray:
    """Per-class non-maximum suppression. Returns kept indices, highest score first."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        while idx.size:
            best, rest = idx[0], idx[1:]
            keep.append(best)
            x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
            overlap = inter / (area(boxes[best]) + area(boxes[rest]) - inter + 1e-9)
            idx = rest[overlap <= iou]
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(-scores[keep], kind="stable")] if keep.size else keep


def decode_raw(output: np.ndarray) -> np.ndarray:
    """A non-end-to-end export (1, 4 + classes, anchors: cx, cy, w, h, class scores) -> rows of x1, y1, x2, y2, score, class."""
    pred = output[0].T
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, scores[np.arange(len(cls)), cls], cls], 1)


def detect(image: np.ndarray, conf: float = DEFAULT_CONF) -> list[Detection]:
    """Run the model on a BGR image and return its detections in the image's own pixels."""
    model = load_model()
    blob, r, (left, top) = letterbox(image, model.size)
    output = model.session.run(None, {model.input_name: blob})[0]

    rows = output[0] if model.end2end else decode_raw(output)
    rows = rows[rows[:, 4] > conf]  # strictly greater, as Ultralytics
    if not model.end2end:
        rows = rows[nms(rows[:, :4], rows[:, 4], rows[:, 5])]
    rows = rows[:MAX_DET]

    h, w = image.shape[:2]
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clip(0, w)  # undo the letterbox
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clip(0, h)
    return [
        Detection(
            class_id=int(c),
            class_name=model.names[int(c)],
            confidence=round(float(s), 3),
            x1=round(float(b[0]), 1), y1=round(float(b[1]), 1), x2=round(float(b[2]), 1), y2=round(float(b[3]), 1),
        )
        for b, s, c in zip(boxes, rows[:, 4], rows[:, 5])
    ]
