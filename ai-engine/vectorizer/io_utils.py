from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_image(path: str | Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def write_image(path: str | Path, image) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Unable to write image: {target}")
    return target


def save_json(path: str | Path, payload) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def copy_file(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    ensure_dir(destination_path.parent)
    shutil.copy2(source_path, destination_path)
    return destination_path


def bgr_to_hex(color) -> str:
    blue, green, red = [int(round(channel)) for channel in color]
    return f"#{red:02x}{green:02x}{blue:02x}"
