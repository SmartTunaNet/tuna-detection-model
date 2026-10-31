"""Detect, track and count tuna in a video, and raise a quota signal.

Usage:
  python scripts/count_video.py <weights.pt> <video> <out.mp4> [--quota 20] [--min-frames 8]
                                [--line x1,y1,x2,y2] [--max-seconds N] [--show]
--show opens a window with the live result while processing (q or Esc stops early).

Counting modes:
  default : a track counts once as tuna when it has been seen for --min-frames frames and most of its
            detections (score-weighted) were "tuna". Counts unique tuna that appeared in view.
  --line  : normalized line (0..1). A tuna track counts when its centre crosses the line from the left
            side to the right side of the direction x1,y1 -> x2,y2 (e.g. the set-net entrance).
When the count reaches --quota the state flips to CLOSE (a gate signal would be sent here).
Writes the annotated video, <out>.events.csv (each count / gate event) and <out>.summary.txt.
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
RED, CYAN, GREEN, ORANGE = (230, 40, 40), (0, 200, 255), (40, 200, 90), (255, 140, 0)


def side(p, a, b):
    return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--quota", type=int, default=20)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--line", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    big, small = ImageFont.truetype(FONT, max(22, H // 24)), ImageFont.truetype(FONT, max(14, H // 48))
    line = None
    if a.line:
        x1, y1, x2, y2 = map(float, a.line.split(","))
        line = ((x1 * W, y1 * H), (x2 * W, y2 * H))

    votes = defaultdict(lambda: [0.0, 0.0])  # track id -> [tuna score, other score]
    seen = defaultdict(int)
    last_side = {}
    counted, events, count, closed_at = set(), [], 0, None
    n, t_infer = 0, 0.0
    max_frames = int(a.max_seconds * fps) if a.max_seconds else None
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        t0 = time.perf_counter()
        r = model.track(frame, persist=True, tracker=str(ROOT / "trackers/bytetrack_tuna.yaml"), conf=0.05,
                        device="mps", verbose=False)[0]
        t_infer += time.perf_counter() - t0
        boxes = []
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                     r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                votes[tid][c] += s
                seen[tid] += 1
                is_tuna = votes[tid][0] >= votes[tid][1]
                ctr = ((x1 + x2) / 2, (y1 + y2) / 2)
                hit = False
                if tid not in counted and is_tuna and seen[tid] >= a.min_frames:
                    if line is None:
                        hit = True
                    else:
                        sd = side(ctr, *line)
                        if last_side.get(tid, 0) > 0 and sd < 0:
                            hit = True
                        if sd != 0:
                            last_side[tid] = sd
                elif line is not None:
                    sd = side(ctr, *line)
                    if sd != 0:
                        last_side[tid] = sd
                if hit:
                    counted.add(tid)
                    count += 1
                    events.append((n, round(n / fps, 2), tid, count, "count"))
                    if count >= a.quota and closed_at is None:
                        closed_at = n
                        events.append((n, round(n / fps, 2), tid, count, "GATE_CLOSE"))
                # once counted, keep showing it as tuna so the boxes match the count
                boxes.append((x1, y1, x2, y2, tid, is_tuna or tid in counted, s, tid in counted))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(img)
        lw = max(2, W // 400)
        for x1, y1, x2, y2, tid, is_tuna, s, done in boxes:
            col = RED if is_tuna else CYAN
            dr.rectangle([x1, y1, x2, y2], outline=col, width=lw + (1 if done else 0))
            tag = f"{'참치' if is_tuna else '기타'} #{tid}"
            tw = dr.textlength(tag, font=small)
            dr.rectangle([x1, y1 - small.size - 6, x1 + tw + 8, y1], fill=col)
            dr.text((x1 + 4, y1 - small.size - 5), tag, font=small, fill=(255, 255, 255))
        if line:
            dr.line([line[0], line[1]], fill=ORANGE, width=lw + 1)
        state_col = ORANGE if closed_at is not None else GREEN
        panel = f"참치 {count} / 쿼터 {a.quota}"
        state = "입구 차단" if closed_at is not None else "입구 개방"
        pw = max(dr.textlength(panel, font=big), dr.textlength(state, font=big)) + 32
        dr.rectangle([12, 12, 12 + pw, 24 + 2 * big.size + 20], fill=(0, 0, 0))
        dr.text((28, 20), panel, font=big, fill=(255, 255, 255))
        dr.text((28, 30 + big.size), state, font=big, fill=state_col)
        shown = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        vw.write(shown)
        n += 1
        if a.show:
            cv2.imshow("tuna counter", shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cap.release()
    vw.release()
    if a.show:
        cv2.destroyAllWindows()
    # re-encode to H.264 so it plays in browsers and slides
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(out)], check=True)
    raw.unlink()

    with open(out.with_suffix(".events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track_id", "count", "event"])
        w.writerows(events)
    tuna_tracks = sum(1 for t, v in votes.items() if v[0] >= v[1])
    summary = (f"video={a.video}\nframes={n} ({n / fps:.1f} s at {fps:.0f} fps)\n"
               f"inference+tracking: {t_infer / max(n, 1) * 1000:.1f} ms/frame ({n / max(t_infer, 1e-9):.1f} fps)\n"
               f"tracks total={len(votes)} tuna-majority={tuna_tracks}\n"
               f"counted tuna={count} (min_frames={a.min_frames}, mode={'line' if line else 'unique-track'})\n"
               f"quota={a.quota} gate_close_at={'%.1f s' % (closed_at / fps) if closed_at is not None else 'never'}\n")
    out.with_suffix(".summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
