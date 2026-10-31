"""Zero-shot CLIP scoring: is each image an underwater photo of live fish?

Usage: python scripts/filter_underwater.py <dataset_dir> <out.csv>
Writes path,p_underwater per image (sorted descending).
"""
import csv
import sys
from pathlib import Path

import open_clip
import torch
from PIL import Image

POS = [
    "an underwater photo of fish swimming in the sea",
    "an underwater video frame of a school of tuna",
    "a murky underwater camera image of fish in a net",
]
NEG = [
    "a photo of a dead fish on a boat deck",
    "a photo of a fish held by a fisherman",
    "a photo of fish at a fish market on ice",
    "an illustration of a fish on a white background",
    "a CCTV image of a fishing vessel deck",
    "a photo of a fish jumping out of the water at the surface",
]


def main():
    root, out = Path(sys.argv[1]), sys.argv[2]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok(POS + NEG).to(dev))
        t /= t.norm(dim=-1, keepdim=True)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for i in range(0, len(imgs), 64):
        batch = imgs[i : i + 64]
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in batch]).to(dev)
        with torch.no_grad():
            f = model.encode_image(x)
            f /= f.norm(dim=-1, keepdim=True)
            prob = (100 * f @ t.T).softmax(-1)
        p_uw = prob[:, : len(POS)].sum(-1).cpu().tolist()
        rows += list(zip(map(str, batch), p_uw))
    rows.sort(key=lambda r: -r[1])
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows([("path", "p_underwater"), *rows])
    n = sum(r[1] > 0.5 for r in rows)
    print(f"{root.name}: {n}/{len(rows)} images p_underwater>0.5 -> {out}")


if __name__ == "__main__":
    main()
