"""Export submitted Label Studio annotations to YOLO labels.

Usage: LS_PAT=<token> python scripts/ls_export.py <project_id> <relabel_dir>
Writes <relabel_dir>/fixed/<stem>.txt for every annotated, non-excluded task (latest annotation wins),
and <relabel_dir>/excluded.txt with the names marked "exclude".
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ls_import import NAMES, call  # noqa: E402


def main():
    pid, d = sys.argv[1], Path(sys.argv[2])
    tok = call("POST", "/api/token/refresh", None, {"refresh": os.environ["LS_PAT"]})["access"]
    out = d / "fixed"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"):
        f.unlink()
    tasks, page = [], 1
    while True:
        r = call("GET", f"/api/tasks?project={pid}&page={page}&page_size=100&fields=all", tok)
        batch = r["tasks"] if isinstance(r, dict) else r
        tasks += batch
        if len(batch) < 100:
            break
        page += 1
    done, excluded, n_box = 0, [], [0, 0]
    for t in tasks:
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ann = max(anns, key=lambda a: a["updated_at"])
        name = t["data"]["name"]
        if any(r["type"] == "choices" and "exclude" in r["value"]["choices"] for r in ann["result"]):
            excluded.append(name)
            continue
        lines = []
        for r in ann["result"]:
            if r["type"] != "rectanglelabels":
                continue
            v = r["value"]
            c = NAMES.index(v["rectanglelabels"][0])
            x, y, w, h = (max(0.0, min(100.0, v[k])) / 100 for k in ("x", "y", "width", "height"))
            w, h = min(w, 1 - x), min(h, 1 - y)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{c} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            n_box[c] += 1
        (out / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        done += 1
    (d / "excluded.txt").write_text("\n".join(excluded) + ("\n" if excluded else ""))
    print(f"annotated={done + len(excluded)} kept={done} excluded={len(excluded)} tuna={n_box[0]} other={n_box[1]}")


if __name__ == "__main__":
    main()
