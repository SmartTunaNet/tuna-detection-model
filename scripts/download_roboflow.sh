#!/usr/bin/env bash
# Usage: scripts/download_roboflow.sh <workspace/project> <version> <format> <outname>
set -euo pipefail
: "${ROBOFLOW_API_KEY:?ROBOFLOW_API_KEY not set}"
proj=$1; ver=$2; fmt=$3; out=$4
dest="data/raw/$out"
[ -d "$dest" ] && { echo "skip $out (exists)"; exit 0; }
link=$(curl -s "https://api.roboflow.com/$proj/$ver/$fmt?api_key=$ROBOFLOW_API_KEY" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['export']['link'])")
mkdir -p "$dest"
curl -sL "$link" -o "$dest.zip"
unzip -qo "$dest.zip" -d "$dest" && rm "$dest.zip"
echo "ok $out: $(find "$dest" -name '*.jpg' -o -name '*.png' | wc -l) images"
