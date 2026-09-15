# Provenance

- Crate: `sherpa-onnx-sys` 1.13.8 from crates.io
  (https://static.crates.io/crates/sherpa-onnx-sys/sherpa-onnx-sys-1.13.8.crate)
- Crate archive SHA-256:
  `14bbeabb73f73f1c1f4a278af0ee9ae90a19f9727482027047fb9422fa14f8fe`
- Recorded upstream commit: `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`
- Upstream path: https://github.com/k2-fsa/sherpa-onnx/tree/v1.13.8/sherpa-onnx/rust/sherpa-onnx-sys
- License: Apache-2.0 (LICENSE kept)

The crate's `.cargo_vcs_info.json` records the commit above and marks the source
tree as dirty. The crates.io archive and its checksum are therefore the exact
reference for comparison; the Git commit is provenance, not a byte-for-byte
substitute.

## Local modifications

- `build.rs` rewritten: only links libraries from `SHERPA_ONNX_LIB_DIR`;
  the upstream download/extract logic (ureq, zip, tar, bzip2) is removed.
- `Cargo.toml` rewritten accordingly (no build-dependencies).
- `src/` is unmodified.

To verify the bindings after extracting the crate archive at the repository
root:

```sh
diff -r sherpa-onnx-sys-1.13.8/src vendor/sherpa-onnx-sys/src
```
