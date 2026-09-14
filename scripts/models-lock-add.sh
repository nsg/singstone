#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MODEL_FILE [MODELS_LOCK]" >&2
  exit 2
fi

model_file=$1
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
lock_file=${2:-"$script_dir/../docs/models.lock"}
if [[ ! -f $model_file ]]; then
  echo "not a file: $model_file" >&2
  exit 1
fi

filename=$(basename -- "$model_file")
size=$(stat -c %s -- "$model_file")
sha256=$(sha256sum -- "$model_file" | cut -d ' ' -f 1)
tmp_file="${lock_file}.tmp"

jq --arg filename "$filename" --arg sha256 "$sha256" --argjson size "$size" \
  '.models += [{name: $filename, purpose: "TODO", upstream: "TODO", url: "TODO", revision: "TODO", license: "TODO", filename: $filename, size: $size, sha256: $sha256}]' \
  "$lock_file" > "$tmp_file"
mv -- "$tmp_file" "$lock_file"
echo "added $filename to $lock_file" >&2
