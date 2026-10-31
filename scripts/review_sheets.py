"""Render numbered review sheets (30 tiles each) for an auto-labeled dir: red=tuna, cyan=other_fish.

Usage: python scripts/review_sheets.py <auto_dir> <out_prefix>
Writes <out_prefix>_<k>.jpg and <auto_dir>/review_index.txt (tile number -> image stem).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

T = 360


def tile(img: Path, lbl: Path, idx: int) -> np.ndarray:
    a = cv2.imread(str(img))
    H, W = a.shape[:2]
    for line in lbl.read_text().split("\n"):
        v = line.split()
        if len(v) != 5:
            continue
        c, cx, cy, w, h = v[0], *map(float, v[1:])
        col = (0, 0, 255) if c == "0" else (255, 200, 0)
        cv2.rectangle(a, (int((cx - w / 2) * W), int((cy - h / 2) * H)), (int((cx + w / 2) * W), int((cy + h / 2) * H)),
                      col, max(3, W // 150))
    s = T / max(H, W)
    a = cv2.resize(a, (int(W * s), int(H * s)))
    t = np.full((T, T, 3), 40, np.uint8)
    t[: a.shape[0], : a.shape[1]] = a
    cv2.rectangle(t, (0, T - 34), (70, T), (0, 0, 0), -1)
    cv2.putText(t, str(idx), (4, T - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return t


def main():
    d, prefix = Path(sys.argv[1]), sys.argv[2]
    imgs = sorted((d / "images").iterdir())
    (d / "review_index.txt").write_text("\n".join(f"{i} {p.stem}" for i, p in enumerate(imgs)))
    for k in range(0, len(imgs), 30):
        ts = [tile(p, d / "labels" / f"{p.stem}.txt", k + i) for i, p in enumerate(imgs[k : k + 30])]
        ts += [np.zeros_like(ts[0])] * (-len(ts) % 6)
        cv2.imwrite(f"{prefix}_{k // 30}.jpg", np.vstack([np.hstack(ts[i : i + 6]) for i in range(0, len(ts), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(imgs)} images -> {(len(imgs) + 29) // 30} sheets")


if __name__ == "__main__":
    main()
