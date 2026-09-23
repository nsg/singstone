#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
extension_dir="$repo_root/gnome-shell-extension/singstone@nsg.github.io"
archive="$repo_root/dist/singstone-gnome-shell-extension.zip"
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT

commit=${GITHUB_SHA:-}
if [[ -z "$commit" ]]; then
  if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    printf '%s\n' 'warning: working tree is dirty; packing without a commit stamp' >&2
  else
    commit=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
  fi
fi

for file in "$extension_dir"/*.js; do
  node --input-type=module --check < "$file"
done

cp -a "$extension_dir/." "$staging/"
cp "$repo_root/LICENSE" "$staging/LICENSE"
if [[ -n "$commit" ]]; then
  node -e '
    const fs = require("fs");
    const [path, commit] = process.argv.slice(1);
    const metadata = JSON.parse(fs.readFileSync(path, "utf8"));
    metadata.commit = commit;
    metadata["version-name"] = commit.slice(0, 7);
    fs.writeFileSync(path, JSON.stringify(metadata, null, 2) + "\n");
  ' "$staging/metadata.json" "$commit"
fi

mkdir -p "$repo_root/dist"
rm -f "$archive"
(
  cd "$staging"
  zip -X -r "$archive" .
)

printf '%s\n' "dist/singstone-gnome-shell-extension.zip"
