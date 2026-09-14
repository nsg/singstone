# Provenance

- Crate: `sherpa-onnx-sys` 1.13.8 from crates.io
  (https://static.crates.io/crates/sherpa-onnx-sys/sherpa-onnx-sys-1.13.8.crate)
- Upstream: https://github.com/k2-fsa/sherpa-onnx (`rust-api/sherpa-onnx-sys`)
- License: Apache-2.0 (LICENSE kept)

## Local modifications
- `build.rs` rewritten: only links libraries from `SHERPA_ONNX_LIB_DIR`;
  the upstream download/extract logic (ureq, zip, tar, bzip2) is removed.
- `Cargo.toml` rewritten accordingly (no build-dependencies).
- `src/` is unmodified. Verify with:
  `diff -r <extracted upstream crate>/src vendor/sherpa-onnx-sys/src`
