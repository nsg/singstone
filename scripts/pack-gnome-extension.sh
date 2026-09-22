#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
extension_dir="$repo_root/gnome-shell-extension/singstone@nsg.github.io"
archive="$repo_root/dist/singstone-gnome-shell-extension.zip"

for file in "$extension_dir"/*.js; do
  node --input-type=module --check < "$file"
done

mkdir -p "$repo_root/dist"
rm -f "$archive"
(
  cd "$extension_dir"
  zip -X -r "$archive" .
)
zip -X -j "$archive" "$repo_root/LICENSE"

printf '%s\n' "dist/singstone-gnome-shell-extension.zip"
