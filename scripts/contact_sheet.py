"""Render a grid of random sample images (with YOLO boxes drawn) per dataset for visual inspection.

Usage: python scripts/contact_sheet.py <dataset_dir> <out.jpg> [n=24] [seed=0]
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

TILE = 320


def find_label(img: Path) -> Path | None:
    lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    return lbl if lbl.exists() else None


def draw(img_path: Path) -> np.ndarray:
    im = cv2.imread(str(img_path))
    h, w = im.shape[:2]
    lbl = find_label(img_path)
    if lbl:
        for line in lbl.read_text().split("\n"):
            v = line.split()
            if len(v) < 5:
                continue
            c, coords = v[0], np.array(v[1:], dtype=float)
            if len(coords) == 4:  # bbox
                cx, cy, bw, bh = coords
                xs, ys = [cx - bw / 2, cx + bw / 2], [cy - bh / 2, cy + bh / 2]
            else:  # polygon
                xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 200))
            cv2.putText(im, c, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 600), (0, 255, 0), 2)
    s = TILE / max(h, w)
    im = cv2.resize(im, (int(w * s), int(h * s)))
    tile = np.zeros((TILE, TILE, 3), np.uint8)
    tile[: im.shape[0], : im.shape[1]] = im
    return tile


def main():
    root, out = Path(sys.argv[1]), sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    tiles = [draw(p) for p in random.sample(imgs, min(n, len(imgs)))]
    cols = 6
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{root.name}: {len(imgs)} images -> {out}")


if __name__ == "__main__":
    main()
