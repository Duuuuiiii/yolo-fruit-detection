"""Run fruit detection on an image, directory, video, or camera."""
import argparse
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image, image directory, video, or camera index")
    parser.add_argument("--model", default=str(BASE / "models/best.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=BASE / "results/predict")
    parser.add_argument("--device", default=None, help="0 for GPU, cpu for CPU; default: automatic")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    results = model.predict(
        source=source, conf=args.conf, imgsz=640, device=args.device,
        save=True, stream=True, show=args.show,
        project=str(args.output.parent), name=args.output.name, exist_ok=True,
    )
    for result in results:
        counts = {}
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().tolist():
                name = result.names[class_id]
                counts[name] = counts.get(name, 0) + 1
        print(f"{result.path}: {counts}")


if __name__ == "__main__":
    main()
