"""Auto-label fish boxes with Grounding DINO, then classify each crop tuna / other_fish with CLIP.

Usage:
  python scripts/autolabel.py --csv data/meta/uw_inat_thunnus.csv --min-uw 0.5 --source tuna --out data/auto/inat_thunnus
  python scripts/autolabel.py --csv data/meta/uw_brin.csv --min-uw 0.5 --source tuna --merge-gt --out data/auto/brin

--source tuna : image comes from a tuna observation; fish >=15% of the biggest fish's area are tuna,
                smaller fish are tuna only if CLIP agrees.
--source other: image comes from a non-tuna observation; every fish is other_fish.
--merge-gt    : keep existing YOLO labels next to the image as tuna and add only non-overlapping new boxes.
Output: <out>/images/*.jpg (symlinks), <out>/labels/*.txt (0=tuna, 1=other_fish), <out>/boxes.csv for review.
"""
import argparse
import csv
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torchvision.ops import box_iou, nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

TUNA = ["a photo of a tuna", "a bluefin tuna swimming underwater", "a yellowfin tuna", "a skipjack tuna"]
FISH = ["a photo of a small silver fish", "a photo of a mackerel", "a photo of a jack fish",
        "a photo of a yellowtail amberjack", "a photo of a shark", "a photo of a reef fish"]
JUNK = ["a photo of a scuba diver", "a photo of a seabird", "a photo of a dolphin", "sea foam and waves",
        "a fishing line and lure", "a human hand", "empty blue water"]


def load_gt(img: Path):
    lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            v = [float(x) for x in line.split()[1:]]
            if len(v) == 4:
                cx, cy, w, h = v
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            elif len(v) >= 6:  # polygon -> bbox
                boxes.append([min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2])])
    return torch.tensor(boxes).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min-uw", type=float, default=0.5)
    ap.add_argument("--source", choices=["tuna", "other"], required=True)
    ap.add_argument("--merge-gt", action="store_true")
    ap.add_argument("--box-thr", type=float, default=0.22)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev)
    clip, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        txt = clip.encode_text(tok(TUNA + FISH + JUNK).to(dev))
        txt /= txt.norm(dim=-1, keepdim=True)
    nt, nf = len(TUNA), len(FISH)

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.csv)) if float(r["p_underwater"]) >= a.min_uw]
    log = csv.writer(open(out / "boxes.csv", "w", newline=""))
    log.writerow(["image", "cls", "det_score", "p_tuna", "p_fish", "p_junk", "from_gt"])
    n_box = [0, 0]
    for i, r in enumerate(rows):
        src = Path(r["path"]).resolve()
        im = Image.open(src).convert("RGB")
        W, H = im.size
        inp = proc(images=im, text="fish.", return_tensors="pt").to(dev)
        with torch.no_grad():
            o = gd(**inp)
        det = proc.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=a.box_thr, text_threshold=0.25, target_sizes=[(H, W)])[0]
        boxes, scores = det["boxes"].cpu(), det["scores"].cpu()
        keep = nms(boxes, scores, 0.5)
        boxes, scores = boxes[keep], scores[keep]
        # drop boxes that are just the whole frame
        big = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (W * H) > 0.9
        boxes, scores = boxes[~big], scores[~big]

        gt = load_gt(src) * torch.tensor([W, H, W, H]) if a.merge_gt else torch.zeros(0, 4)
        if len(gt) and len(boxes):
            novel = box_iou(boxes, gt).max(1).values < 0.3
            boxes, scores = boxes[novel], scores[novel]

        labels = [(b.tolist(), 0, 1.0, 1.0, 0.0, 0.0, 1) for b in gt]
        if len(boxes):
            crops = []
            for x1, y1, x2, y2 in boxes.tolist():
                pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
                crops.append(pre(im.crop((max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)))))
            with torch.no_grad():
                f = clip.encode_image(torch.stack(crops).to(dev))
                f /= f.norm(dim=-1, keepdim=True)
                p = (100 * f @ txt.T).softmax(-1).cpu()
            pt, pf, pj = p[:, :nt].sum(1), p[:, nt:nt + nf].sum(1), p[:, nt + nf:].sum(1)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ref = max(areas.max().item(), ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).max().item() if len(gt) else 0)
            for j, b in enumerate(boxes.tolist()):
                if pj[j] > 0.5:
                    continue
                if a.source == "other":
                    c = 1
                else:
                    # tuna observation: anything comparable in size to the biggest fish is tuna;
                    # clearly smaller fish (bait / bycatch) are tuna only if CLIP agrees
                    c = 0 if (areas[j] >= 0.15 * ref or pt[j] >= pf[j]) else 1
                labels.append((b, c, float(scores[j]), float(pt[j]), float(pf[j]), float(pj[j]), 0))

        stem = f"{src.parent.parent.name}_{src.stem}" if a.merge_gt else src.stem
        dst = out / "images" / f"{stem}{src.suffix}"
        if not dst.exists():
            dst.symlink_to(src)
        with open(out / "labels" / f"{stem}.txt", "w") as fh:
            for (x1, y1, x2, y2), c, *_ in labels:
                fh.write(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        for b, c, s, ptv, pfv, pjv, g in labels:
            log.writerow([stem, c, f"{s:.3f}", f"{ptv:.3f}", f"{pfv:.3f}", f"{pjv:.3f}", g])
            n_box[c] += 1
        if i % 50 == 0:
            print(f"{i}/{len(rows)} images, tuna={n_box[0]} other={n_box[1]}", flush=True)
    print(f"done {len(rows)} images, tuna={n_box[0]} other={n_box[1]} -> {out}")


if __name__ == "__main__":
    main()
