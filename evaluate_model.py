from __future__ import annotations

import argparse
from pathlib import Path

from local_setup import ROOT, enable_modules


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO11s or EALC-YOLO11s model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "test_only.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for path in (args.checkpoint, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    enable_modules()
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()), split=args.split,
        imgsz=640, batch=args.batch, workers=args.workers, device=args.device,
        conf=0.001, iou=0.70, max_det=300, plots=False, verbose=False,
        project=str(ROOT / "evaluation_runs")
    )
    for name, value in (
        ("Precision", metrics.box.mp), ("Recall", metrics.box.mr),
        ("AP50", metrics.box.map50), ("AP75", metrics.box.map75),
        ("mAP50-95", metrics.box.map),
    ):
        print(f"{name}: {100 * value:.3f}%")


if __name__ == "__main__":
    main()

