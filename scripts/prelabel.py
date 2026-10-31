"""Pre-label review-rejected images for human correction in Label Studio.

Usage: python scripts/prelabel.py <weights.pt> <out_dir> [conf=0.15]

For every auto-labeled source, images NOT in data/review/<name>_accept.txt are copied to
<out_dir>/images as <source>__<stem>.jpg. Their label = model boxes (conf >= conf) plus
auto-label (Grounding DINO) boxes the model did not cover (IoU < 0.5), so the human mostly
deletes and fixes rather than draws. <out_dir>/order.txt lists images most-underwater first.
"""
import csv
import shutil
import sys
from pathlib import Path

import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["brin", "inat_thunnus", "inat_katsuwonus", "inat_scomber", "inat_trachurus"]


def read_yolo(path: Path):
    rows = [l.split() for l in path.read_text().splitlines() if len(l.split()) == 5] if path.exists() else []
    cls = torch.tensor([int(r[0]) for r in rows], dtype=torch.int32)
    xywh = torch.tensor([[float(v) for v in r[1:]] for r in rows]).reshape(-1, 4)
    xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    return cls, xyxy


def main():
    weights, out = sys.argv[1], Path(sys.argv[2])
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    model = YOLO(weights)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    order, n_model, n_auto = [], 0, 0
    for name in SOURCES:
        auto = ROOT / "data/auto" / name
        accept = set((ROOT / "data/review" / f"{name}_accept.txt").read_text().split())
        uw = {}
        meta = ROOT / "data/meta" / f"uw_{name}.csv"
        for r in csv.DictReader(open(meta)):
            uw[Path(r["path"]).resolve()] = float(r["p_underwater"])
        for img in sorted((auto / "images").iterdir()):
            if img.stem in accept:
                continue
            src = img.resolve()
            dst = out / "images" / f"{name}__{img.stem}{img.suffix.lower()}"
            shutil.copyfile(src, dst)
            r = model.predict(str(src), conf=conf, device="mps", verbose=False)[0]
            H, W = r.orig_shape
            m_cls, m_box = r.boxes.cls.cpu().int(), r.boxes.xyxyn.cpu()
            a_cls, a_box = read_yolo(auto / "labels" / f"{img.stem}.txt")
            if len(a_box) and len(m_box):
                keep = box_iou(a_box, m_box).max(1).values < 0.5
                a_cls, a_box = a_cls[keep], a_box[keep]
            lines = []
            for c, (x1, y1, x2, y2) in zip(torch.cat([m_cls, a_cls]).tolist(), torch.cat([m_box, a_box]).tolist()):
                lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            (out / "labels" / f"{dst.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_model += len(m_box)
            n_auto += len(a_box)
            order.append((uw.get(src, 0.0), dst.name))
        print(f"{name}: done", flush=True)
    order.sort(key=lambda t: -t[0])
    (out / "order.txt").write_text("\n".join(f"{n} {p:.3f}" for p, n in order) + "\n")
    print(f"{len(order)} images, model boxes={n_model}, extra auto boxes={n_auto} -> {out}")


if __name__ == "__main__":
    main()
