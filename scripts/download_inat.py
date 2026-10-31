"""Download CC-licensed research-grade iNaturalist photos for a taxon, recording license per photo.

Usage: python scripts/download_inat.py "<taxon name>" <out_dir> [max_photos]
Writes <out_dir>/images/*.jpg and <out_dir>/meta.csv (file, obs_id, photo_id, license, attribution, taxon).
"""
import csv
import sys
import time
import urllib.parse
from pathlib import Path

import requests

LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
API = "https://api.inaturalist.org/v1/observations"


def main():
    taxon, out = sys.argv[1], Path(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.csv"
    done = set()
    if meta_path.exists():
        done = {r["photo_id"] for r in csv.DictReader(open(meta_path))}
    fh = open(meta_path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["file", "obs_id", "photo_id", "license", "attribution", "taxon"])
    s = requests.Session()
    id_above, n = 0, 0
    while True:
        q = dict(taxon_name=taxon, photos="true", photo_license=LICENSES, quality_grade="research",
                 per_page=200, order_by="id", order="asc", id_above=id_above)
        r = s.get(f"{API}?{urllib.parse.urlencode(q)}", timeout=60).json()
        res = r["results"]
        if not res:
            break
        for obs in res:
            for ph in obs["photos"]:
                pid = str(ph["id"])
                if pid in done or not ph.get("license_code"):
                    continue
                url = ph["url"].replace("/square.", "/large.")
                fn = f"{obs['id']}_{pid}.jpg"
                try:
                    img = s.get(url, timeout=60)
                    img.raise_for_status()
                except requests.RequestException as e:
                    print("skip", url, e)
                    continue
                (out / "images" / fn).write_bytes(img.content)
                w.writerow([fn, obs["id"], pid, ph["license_code"], ph.get("attribution", ""),
                            obs["taxon"]["name"]])
                n += 1
                if cap and n >= cap:
                    break
        fh.flush()
        if cap and n >= cap:
            break
        id_above = res[-1]["id"]
        print(f"{taxon}: {n} photos so far")
        time.sleep(1)  # API rate limit
    print(f"done {taxon}: {n} new photos -> {out}")


if __name__ == "__main__":
    main()
