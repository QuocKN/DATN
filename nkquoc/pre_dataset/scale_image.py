from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
INPUT_PATH = Path(
    r"G:\DATN_DATA\RF\DroneDetect\spectrograms_MAV60"
)
RECURSIVE = True
TOP_RATIO = 1 / 6
BOTTOM_RATIO = 5 / 6


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTS else []

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def save_in_place(image: Image.Image, output_path: Path) -> None:
    suffix = output_path.suffix or ".png"
    fd, temp_name = tempfile.mkstemp(prefix=f"{output_path.stem}_", suffix=suffix, dir=output_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def scale_image(path: Path, top_ratio: float, bottom_ratio: float) -> None:
    with Image.open(path) as img:
        w, h = img.size
        crop_top = int(h * top_ratio)
        crop_bottom = int(h * bottom_ratio)

        if crop_top < 0 or crop_bottom > h or crop_top >= crop_bottom:
            raise ValueError(
                f"Invalid crop ratios for {path}: top={top_ratio}, bottom={bottom_ratio}, height={h}"
            )

        img_crop = img.crop((0, crop_top, w, crop_bottom))
        img_out = img_crop.resize((w, h), Image.Resampling.BILINEAR)
        save_in_place(img_out, path)


def process_images(paths: Iterable[Path], top_ratio: float, bottom_ratio: float) -> int:
    count = 0
    for path in paths:
        scale_image(path, top_ratio, bottom_ratio)
        count += 1
        print(f"scaled: {path}")
    return count


def main() -> None:
    image_paths = collect_images(INPUT_PATH, RECURSIVE)
    if not image_paths:
        print(f"No images found: {INPUT_PATH}")
        return

    count = process_images(image_paths, TOP_RATIO, BOTTOM_RATIO)
    print(f"Done. Scaled {count} image(s) in place.")


if __name__ == "__main__":
    main()
