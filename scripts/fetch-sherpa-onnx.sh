#!/usr/bin/env bash
# Fetch the pinned sherpa-onnx prebuilt shared libraries, verify their SHA-256
# and install them under third_party/sherpa-onnx (where .cargo/config.toml
# points SHERPA_ONNX_LIB_DIR). This is the only network step of the build and
# must be run before `cargo build`; the build itself works offline.
#
# Shared (not static) libraries are required: whisper.cpp and the prebuilt
# ONNX Runtime are compiled with different C++ toolchains and linking both
# statically into one executable violates the ODR for libstdc++ templates
# (observed crash in std::regex inside onnxruntime device discovery).
# At runtime the binary finds the .so files via rpath ($ORIGIN and the
# build-time library directory); copy them next to the binary when installing.
#
# Alternative: build sherpa-onnx v1.13.8 from source with
#   cmake -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_C_API=ON ...
# and point SHERPA_ONNX_LIB_DIR at the resulting lib directory.
set -euo pipefail
VERSION=1.13.8
ARCHIVE="sherpa-onnx-v${VERSION}-linux-x64-shared-lib.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${VERSION}/${ARCHIVE}"
SHA256="3892d184be41027e18165e67f549cd4e4cdd8dcd73ac5579e97afd55e14e30b6"

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/third_party"
mkdir -p "$dest"
cd "$dest"
if [ ! -f "$ARCHIVE" ]; then
  echo "downloading $URL"
  curl -fL --proto '=https' --tlsv1.2 -o "$ARCHIVE.part" "$URL"
  mv "$ARCHIVE.part" "$ARCHIVE"
fi
echo "$SHA256  $ARCHIVE" | sha256sum -c -
rm -rf "sherpa-onnx-v${VERSION}-linux-x64-shared-lib" sherpa-onnx
tar xjf "$ARCHIVE"
ln -sfn "sherpa-onnx-v${VERSION}-linux-x64-shared-lib" sherpa-onnx
echo "installed $(ls sherpa-onnx/lib | wc -l) libraries into $dest/sherpa-onnx/lib"
