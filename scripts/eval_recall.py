"""Tuna recall on a YOLO split: a GT tuna counts as found if any predicted box of the given classes hits it.

Usage: python scripts/eval_recall.py <weights.pt> <split_dir> [out_dir_for_miss_images]
Reports, per confidence threshold, tuna recall when only "tuna" predictions count and when any fish
prediction counts (a tuna found but called other_fish), plus false tuna boxes per image.
"""
import sys
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

CONFS = (0.5, 0.25, 0.1, 0.05)


def load(lbl: Path, W: int, H: int):
    rows = [l.split() for l in lbl.read_text().splitlines() if len(l.split()) == 5] if lbl.exists() else []
    cls = torch.tensor([int(float(r[0])) for r in rows], dtype=torch.int64)
    b = torch.tensor([[float(v) for v in r[1:]] for r in rows]).reshape(-1, 4)
    xyxy = torch.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                        (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
    return cls, xyxy


def main():
    model = YOLO(sys.argv[1])
    split = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    stats = {c: [0, 0, 0, 0] for c in CONFS}  # found as tuna, found as any fish, false tuna boxes, total
    imgs = sorted((split / "images").iterdir())
    for img in imgs:
        r = model.predict(str(img), conf=min(CONFS), device="mps", verbose=False)[0]
        H, W = r.orig_shape
        gcls, gbox = load(split / "labels" / f"{img.stem}.txt", W, H)
        gt = gbox[gcls == 0]
        pb, pc, ps = r.boxes.xyxy.cpu(), r.boxes.cls.cpu().int(), r.boxes.conf.cpu()
        for c in CONFS:
            k = ps >= c
            tb, ab = pb[k & (pc == 0)], pb[k]
            hit_t = box_iou(gt, tb).max(1).values >= 0.5 if len(gt) and len(tb) else torch.zeros(len(gt), dtype=torch.bool)
            hit_a = box_iou(gt, ab).max(1).values >= 0.5 if len(gt) and len(ab) else torch.zeros(len(gt), dtype=torch.bool)
            fp = (box_iou(tb, gbox).max(1).values < 0.5).sum() if len(tb) and len(gbox) else len(tb)
            s = stats[c]
            s[0] += int(hit_t.sum()); s[1] += int(hit_a.sum()); s[2] += int(fp); s[3] += len(gt)
            if out and c == 0.25 and not hit_t.all():
                a = r.orig_img.copy()
                for g in gt[~hit_t].int().tolist():
                    cv2.rectangle(a, g[:2], g[2:], (0, 255, 255), max(2, W // 300))
                for b, cc in zip(tb.int().tolist(), [0] * len(tb)):
                    cv2.rectangle(a, b[:2], b[2:], (0, 0, 255), max(1, W // 400))
                cv2.imwrite(str(out / img.name), a)
    print(f"{split}: {len(imgs)} images, {stats[CONFS[0]][3]} GT tuna")
    print("conf   recall(tuna)  recall(any fish)  false tuna/img")
    for c in CONFS:
        f_t, f_a, fp, n = stats[c]
        print(f"{c:<6} {f_t / max(n, 1):.3f}         {f_a / max(n, 1):.3f}             {fp / len(imgs):.2f}")


if __name__ == "__main__":
    main()
