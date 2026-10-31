"""Create a Label Studio project from a pre-labeled dir (scripts/prelabel.py output).

Usage: LS_PAT=<personal access token> python scripts/ls_import.py <relabel_dir> [project title]
Needs Label Studio started with
  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo>/data
Tasks are imported in <relabel_dir>/order.txt order; model boxes come in as predictions (editable pre-labels).
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image

URL = "http://localhost:8080"
ROOT = Path(__file__).resolve().parent.parent
NAMES = ["tuna", "other_fish"]
CONFIG = """<View>
  <Header value="source: $name" size="6"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="tuna" background="red" hotkey="1"/>
    <Label value="other_fish" background="#00c8ff" hotkey="2"/>
  </RectangleLabels>
  <Choices name="exclude" toName="image" showInline="true">
    <Choice value="exclude" alias="exclude" hint="Not usable (not underwater, illustration, too unclear)"/>
  </Choices>
</View>"""


def call(method, path, token, body=None):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or "null")


def main():
    d = Path(sys.argv[1]).resolve()
    title = sys.argv[2] if len(sys.argv) > 2 else "tuna relabel"
    access = call("POST", "/api/token/refresh", None, {"refresh": os.environ["LS_PAT"]})["access"]
    proj = call("POST", "/api/projects", access, {"title": title, "label_config": CONFIG,
                                                   "show_collab_predictions": True, "model_version": "v1_baseline"})
    call("POST", "/api/storages/localfiles", access, {"project": proj["id"], "title": "relabel images",
                                                       "path": str(d / "images"), "use_blob_urls": False})
    rel = (d / "images").relative_to(ROOT / "data")
    tasks = []
    for line in (d / "order.txt").read_text().split("\n"):
        if not line.strip():
            continue
        name = line.split()[0]
        W, H = Image.open(d / "images" / name).size
        result = []
        lbl = d / "labels" / f"{Path(name).stem}.txt"
        for i, row in enumerate(l.split() for l in lbl.read_text().splitlines() if l.strip()):
            c, cx, cy, w, h = int(float(row[0])), *map(float, row[1:])
            result.append({"id": f"p{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                           "original_width": W, "original_height": H, "image_rotation": 0,
                           "value": {"x": (cx - w / 2) * 100, "y": (cy - h / 2) * 100, "width": w * 100,
                                     "height": h * 100, "rotation": 0, "rectanglelabels": [NAMES[c]]}})
        tasks.append({"data": {"image": f"/data/local-files/?d={rel}/{name}", "name": name},
                      "predictions": [{"model_version": "v1_baseline", "result": result}]})
    call("POST", f"/api/projects/{proj['id']}/import", access, tasks)
    print(f"project {proj['id']} '{title}': {len(tasks)} tasks -> {URL}/projects/{proj['id']}/data")


if __name__ == "__main__":
    main()
