"""Assemble reviewed sources into one YOLO dataset (classes: 0=tuna, 1=other_fish).

Usage: python scripts/build_dataset.py <out_dir>

Sources (edit SOURCES to add more):
  - Roboflow sets used as-is (all boxes -> tuna), split kept from Roboflow.
  - Auto-labeled sets filtered by data/review/<name>_accept.txt (if present);
    split from Roboflow folder prefix if any, else a stable 75/25 hash split.
Val holds only underwater images so its recall reflects the target domain.

Human-corrected images (data/relabel/fixed, from Label Studio) are added too. "val_hard" holds whole
groups the model never trains on: every frame of HARD_VIDEOS (frames of one video are near-duplicates,
so splitting them leaks) plus the corrected iNaturalist photos (their observations are not used elsewhere).
data_hard.yaml points val at val_hard for evaluation.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    # (name, kind, path)
    ("red_tuna", "roboflow_train_only", ROOT / "data/raw/red_tuna"),  # mosaics, mixed domain -> train only
    ("brin", "auto", ROOT / "data/auto/brin"),
    ("inat_thunnus", "auto", ROOT / "data/auto/inat_thunnus"),
    ("inat_katsuwonus", "auto", ROOT / "data/auto/inat_katsuwonus"),
    ("inat_scomber", "auto", ROOT / "data/auto/inat_scomber"),
    ("inat_trachurus", "auto", ROOT / "data/auto/inat_trachurus"),
    ("inat_seriola", "auto", ROOT / "data/auto/inat_seriola"),
    ("relabel", "relabel", ROOT / "data/relabel"),
    # unreviewed negatives (every fish = other_fish), train only; photos already used by the
    # reviewed iNat sets above are skipped so nothing leaks into val
    ("neg_scomber", "neg", ROOT / "data/auto/neg_scomber"),
    ("neg_trachurus", "neg", ROOT / "data/auto/neg_trachurus"),
    ("neg_seriola", "neg", ROOT / "data/auto/neg_seriola"),
    ("neg_coryphaena", "neg", ROOT / "data/auto/neg_coryphaena"),
]
REVIEWED = {p.stem for d in ("inat_scomber", "inat_trachurus", "inat_seriola") if (ROOT / "data/auto" / d / "images").exists()
            for p in (ROOT / "data/auto" / d / "images").iterdir()}
HARD_VIDEOS = ["yt-o7psTM0Jhl4", "yt-Iysln6vKyL4"]


def in_hard_video(stem: str) -> bool:
    return any(f"{v}-" in stem for v in HARD_VIDEOS)


def hash_split(stem: str) -> str:
    return "val" if int(hashlib.md5(stem.encode()).hexdigest(), 16) % 4 == 0 else "train"


def link(src_img: Path, label_text: str, out: Path, split: str, stem: str):
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    dst = out / split / "images" / f"{stem}{src_img.suffix}"
    if not dst.exists():
        dst.symlink_to(src_img.resolve())
    (out / split / "labels" / f"{stem}.txt").write_text(label_text)


def main():
    out = Path(sys.argv[1])
    counts = {}
    for name, kind, path in SOURCES:
        if not path.exists():
            print(f"skip {name} (missing)")
            continue
        n = {"train": 0, "val": 0, "val_hard": 0}
        if kind == "neg":
            for img in sorted((path / "images").iterdir()):
                if img.stem in REVIEWED:
                    continue
                link(img, (path / "labels" / f"{img.stem}.txt").read_text(), out, "train", f"{name}_{img.stem}")
                n["train"] += 1
        elif kind == "relabel":
            for lbl in sorted((path / "fixed").glob("*.txt")):
                img = next((path / "images").glob(f"{lbl.stem}.*"))
                split = "val_hard" if (lbl.stem.startswith("inat_") or in_hard_video(lbl.stem)) else "train"
                link(img, lbl.read_text(), out, split, f"{name}_{lbl.stem}")
                n[split] += 1
        elif kind == "roboflow_train_only":
            for img in path.rglob("images/*"):
                lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
                lines = [l.split() for l in lbl.read_text().split("\n") if l.strip()] if lbl.exists() else []
                text = "".join("0 " + " ".join(v[1:5]) + "\n" for v in lines if len(v) == 5)
                split = "val_hard" if in_hard_video(img.stem) else "train"
                link(img, text, out, split, f"{name}_{img.parent.parent.name}_{img.stem}")
                n[split] += 1
        else:
            acc_file = ROOT / "data/review" / f"{name}_accept.txt"
            accept = set(acc_file.read_text().split()) if acc_file.exists() else None
            for img in sorted((path / "images").iterdir()):
                if accept is not None and img.stem not in accept:
                    continue
                prefix = img.stem.split("_", 1)[0]
                split = ("val" if prefix in {"valid", "test"} else "train") if prefix in {"train", "valid", "test"} \
                    else hash_split(img.stem)
                split = "val_hard" if in_hard_video(img.stem) else split
                link(img, (path / "labels" / f"{img.stem}.txt").read_text(), out, split, f"{name}_{img.stem}")
                n[split] += 1
        counts[name] = n
        print(f"{name}: train={n['train']} val={n['val']} val_hard={n['val_hard']}")
    for yaml, val in (("data.yaml", "val"), ("data_hard.yaml", "val_hard")):
        (out / yaml).write_text(
            f"path: {out.resolve()}\ntrain: train/images\nval: {val}/images\nnames:\n  0: tuna\n  1: other_fish\n")
    print("total", {s: sum(c[s] for c in counts.values()) for s in ("train", "val", "val_hard")})


if __name__ == "__main__":
    main()
