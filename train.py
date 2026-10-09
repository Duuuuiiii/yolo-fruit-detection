"""Audited six-class fruit-detector training entry point."""
from __future__ import annotations

import argparse
import math
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input"
RAW = BASE / "raw_export"
DATASET = BASE / "dataset_6class"
RUNS = BASE / "runs"
TARGET_NAMES = ["apple", "orange", "banana", "kiwi", "lemon", "strawberry"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def die(message: str) -> None:
    raise SystemExit(f"\n停止：{message}\n")


def find_export() -> Path:
    zips = sorted(INPUT.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        die("请把 Roboflow 导出的 YOLOv11 PyTorch ZIP 放进 input 文件夹后重新运行。")
    return zips[0]


def unpack(zip_path: Path) -> Path:
    if RAW.exists():
        die("raw_export 已存在；请备份并移走它，或使用 --data 直接训练已有数据。")
    RAW.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            destination = (RAW / member.filename).resolve()
            if not destination.is_relative_to(RAW.resolve()):
                die("ZIP 包含越界路径，拒绝解压。")
        zf.extractall(RAW)
    yamls = list(RAW.rglob("data.yaml")) + list(RAW.rglob("data.yml"))
    if len(yamls) != 1:
        die("导出包中未找到唯一的 data.yaml；请在 Roboflow 选择 YOLOv11 PyTorch 格式。")
    return yamls[0]


def names_from_yaml(path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, list):
        die("data.yaml 的类别格式无法识别。")
    return [str(x).strip().lower() for x in names]


def image_for_label(label: Path, images_dir: Path) -> Path | None:
    stem = label.stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_label(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            die(f"标注格式错误：{path.name}")
        try:
            row = [float(v) for v in values]
        except ValueError:
            die(f"标注不是数值：{path.name}")
        if not all(math.isfinite(v) for v in row):
            die(f"标注含无效数值：{path.name}")
        if row[0] != int(row[0]) or row[0] < 0:
            die(f"标注类别 ID 错误：{path.name}")
        if not all(0 <= v <= 1 for v in row[1:]) or min(row[3:]) <= 0:
            die(f"标注坐标超出 YOLO 格式范围：{path.name}")
        rows.append(row)
    return rows


def build_six_class_dataset(source_yaml: Path, source_names: list[str]) -> tuple[Path, dict[str, int]]:
    missing = [name for name in TARGET_NAMES if name not in source_names]
    if missing:
        die(f"数据集缺少目标类别：{missing}；实际类别为 {source_names}")
    selected_ids = {source_names.index(name): i for i, name in enumerate(TARGET_NAMES)}
    if DATASET.exists():
        die("dataset_6class 已存在；请使用 --data dataset_6class/data.yaml，或备份并移走旧数据。")
    counts = {name: 0 for name in TARGET_NAMES}
    split_counts = {}
    for split in ("train", "valid", "val", "test"):
        source_images = source_yaml.parent / split / "images"
        source_labels = source_yaml.parent / split / "labels"
        if not source_images.exists() or not source_labels.exists():
            continue
        out_split = "val" if split in {"valid", "val"} else split
        out_images = DATASET / out_split / "images"
        out_labels = DATASET / out_split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        kept = 0
        for label in source_labels.glob("*.txt"):
            rows = parse_label(label)
            classes = {int(row[0]) for row in rows}
            # Keep only images whose every visible annotated fruit belongs to our six classes.
            if not rows or not classes.issubset(selected_ids):
                continue
            image = image_for_label(label, source_images)
            if image is None:
                continue
            remapped = []
            for row in rows:
                old_id = int(row[0])
                new_id = selected_ids[old_id]
                remapped.append(f"{new_id} " + " ".join(f"{v:.6f}" for v in row[1:]))
                counts[TARGET_NAMES[new_id]] += 1
            shutil.copy2(image, out_images / image.name)
            (out_labels / label.name).write_text("\n".join(remapped) + "\n", encoding="utf-8")
            kept += 1
        split_counts[out_split] = split_counts.get(out_split, 0) + kept
    if split_counts.get("train", 0) < 200 or split_counts.get("val", 0) < 40:
        die(f"筛选后数据不足：{split_counts}。不能可靠训练。")
    cfg = {"path": str(DATASET), "train": "train/images", "val": "val/images", "names": TARGET_NAMES}
    if (DATASET / "test/images").exists():
        cfg["test"] = "test/images"
    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (BASE / "audit_report.txt").write_text(
        "类别顺序：" + ", ".join(TARGET_NAMES) + "\n"
        + "图片数：" + str(split_counts) + "\n"
        + "框数：" + str(counts) + "\n",
        encoding="utf-8",
    )
    return yaml_path, counts


def make_audit_preview() -> None:
    pairs = []
    for split in ("train", "val", "test"):
        images = DATASET / split / "images"
        labels = DATASET / split / "labels"
        if images.exists():
            pairs.extend((p, labels / f"{p.stem}.txt") for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    random.Random(209).shuffle(pairs)
    chosen = pairs[:24]
    tiles = []
    suspicious = 0
    for image_path, label_path in chosen:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        for cls, x, y, bw, bh in parse_label(label_path):
            x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
            x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
            if bw * bh > 0.92:
                suspicious += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, TARGET_NAMES[int(cls)], (max(0, x1), max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        image = cv2.resize(image, (320, 240))
        tiles.append(image)
    if len(tiles) < 8:
        die("可用于审查的标注图片太少。")
    while len(tiles) % 4:
        tiles.append(tiles[-1] * 0)
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(BASE / "audit_preview.jpg"), cv2.vconcat(rows))
    if suspicious >= 4:
        die("抽检发现多个框覆盖超过 92% 图像面积，疑似全屏错误标注；已拒绝训练。")


def train(data_yaml: Path, device: str = "0") -> None:
    # COCO-pretrained initialization, then all layers fine-tuned on the audited fruit dataset.
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(data_yaml),
        epochs=120,
        imgsz=640,
        batch=4,
        device=device,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=30,
        close_mosaic=15,
        pretrained=True,
        seed=209,
        deterministic=True,
        amp=False,
        cache=False,
        project=str(RUNS),
        name="fruit6_yolo11s",
        exist_ok=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Use an already prepared six-class dataset YAML")
    parser.add_argument("--device", default="0", help="GPU index (0) or cpu")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and audit data without training")
    args = parser.parse_args()
    if args.data:
        if args.prepare_only:
            parser.error("--data and --prepare-only cannot be used together")
        if not args.data.is_file():
            parser.error("Dataset YAML does not exist")
        if names_from_yaml(args.data) != TARGET_NAMES:
            parser.error("Dataset class order must match the six-class model")
        train(args.data.resolve(), args.device)
        raise SystemExit(0)
    INPUT.mkdir(exist_ok=True)
    export = find_export()
    yaml_file = unpack(export)
    source_names = names_from_yaml(yaml_file)
    dataset_yaml, _ = build_six_class_dataset(yaml_file, source_names)
    make_audit_preview()
    print("已生成标注抽查预览，请检查 audit_preview.jpg 和 audit_report.txt。")
    if not args.prepare_only:
        train(dataset_yaml, args.device)
