"""Export models/best.pt to models/best.onnx for the deployed app (dev tool; needs requirements-dev.txt).

Usage:
    python -m tools.export_onnx

Equivalent command line:
    yolo export model=models/best.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1

Settings, and why:
- imgsz=640: the size the model was trained and validated at.
- nms left unset (Ultralytics' default): keeps the standard head, whose raw output (1, 16, 8400)
  is decoded and NMS'd in src/onnx_detect.py. nms=False would switch YOLO26 to its NMS-free
  one-to-one head, which is a different head from the one the PyTorch app used: on the check
  set it changed verdicts on 5 of 20 images (decisions.md, Phase 8b).
- opset=17, simplify=True (onnxslim), static shape, batch 1, fp32: plain CPU inference in
  onnxruntime, nothing exotic for the serverless runtime to support.
Then check parity: python -m tools.parity
"""

from ultralytics import YOLO


def main() -> None:
    path = YOLO("models/best.pt").export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, batch=1)
    print(f"Exported {path}. Now run: python -m tools.parity")


if __name__ == "__main__":
    main()
