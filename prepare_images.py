"""Создать несколько размеров изображения без изменения оригинала."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 1536])
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes):
        parser.error("Размеры должны быть положительными")

    with Image.open(args.source) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        image = original.copy()
        image.thumbnail((size, size))
        target = args.output_dir / f"test-{size}.jpg"
        image.save(target)
        print(f"{target}: {image.size}")


if __name__ == "__main__":
    main()
