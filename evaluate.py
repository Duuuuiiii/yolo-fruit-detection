"""Evaluate the released model on a labeled validation or test split."""
import argparse
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the prepared dataset YAML")
    parser.add_argument("--model", default=str(BASE / "models/best.pt"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    metrics = YOLO(args.model).val(
        data=args.data, split=args.split, imgsz=640, batch=4,
        device=args.device, workers=0, project=str(BASE / "results"),
        name=f"evaluate_{args.split}",
    )
    print(f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
