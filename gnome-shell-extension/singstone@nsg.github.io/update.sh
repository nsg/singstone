#!/usr/bin/env bash
# Downloads the latest Singstone snap and GNOME Shell extension and installs
# both. The extension's update menu item runs this in a terminal.
set -euo pipefail

release_url=https://github.com/nsg/singstone/releases/latest/download
snap_asset=singstone_amd64.snap
extension_asset=singstone-gnome-shell-extension.zip
bus_name=io.github.nsg.Singstone
workdir=
reopen=false

fetch() {
  local asset=$1
  printf '\nDownloading %s\n' "$asset"
  if command -v curl >/dev/null; then
    curl --fail --location --progress-bar \
      --retry 5 --retry-all-errors --continue-at - \
      --speed-limit 1024 --speed-time 60 \
      --output "$workdir/$asset" "$release_url/$asset"
  else
    wget --continue --tries=5 --quiet --show-progress \
      --output-document="$workdir/$asset" "$release_url/$asset"
  fi
}

app_running() {
  local owned
  owned=$(gdbus call --session --dest org.freedesktop.DBus \
    --object-path /org/freedesktop/DBus \
    --method org.freedesktop.DBus.NameHasOwner "$bus_name" 2>/dev/null) || return 1
  [[ $owned == '(true,)' ]]
}

recorder() {
  gdbus call --session --dest "$bus_name" \
    --object-path /io/github/nsg/Singstone/Recorder \
    --method "io.github.nsg.Singstone.Recorder.$1"
}

close_app() {
  app_running || return 0

  local status
  status=$(recorder GetStatus)
  if [[ $status == *"'recording': <true>"* ]]; then
    echo 'Singstone is recording. Stop the recording and run the update again.' >&2
    return 1
  fi

  printf '\nClosing Singstone\n'
  recorder Quit >/dev/null
  reopen=true
  for _ in {1..30}; do
    app_running || return 0
    sleep 0.5
  done
  echo 'Singstone did not close.' >&2
  return 1
}

finish() {
  local status=$?
  if [[ -n $workdir ]]; then
    rm -rf -- "$workdir"
  fi
  # Also after a failed install, so the app that was closed comes back.
  if $reopen; then
    setsid --fork snap run singstone gui </dev/null >/dev/null 2>&1 || true
  fi
  if (( status == 0 )); then
    printf '\nSingstone and the extension are updated.\n'
    printf 'Log out and back in to load the new extension.\n'
  else
    printf '\nThe update failed.\n' >&2
  fi
  read -r -p 'Press Enter to close this window. ' || true
}

main() {
  trap finish EXIT
  local cache=${XDG_CACHE_HOME:-$HOME/.cache}
  mkdir -p "$cache"
  workdir=$(mktemp -d "$cache/singstone-update.XXXXXX")

  fetch "$snap_asset"
  fetch "$extension_asset"
  close_app

  printf '\nInstalling Singstone (asks for your password)\n'
  sudo snap install --dangerous "$workdir/$snap_asset"

  printf '\nInstalling the extension\n'
  gnome-extensions install --force "$workdir/$extension_asset"
}

# Installing the extension replaces this file; bash has read all of it by the
# time main runs, and the exit keeps it from reading on in the new copy.
main "$@"; exit
