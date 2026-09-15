# `sherpa-onnx-sys` local build patch

This directory contains the published `sherpa-onnx-sys` 1.13.8 crate with a
small local build patch. Singstone keeps this copy because the upstream build
script downloads native libraries during `cargo build`. Our version performs
no network access: it links the libraries already present in
`SHERPA_ONNX_LIB_DIR`.

Only `Cargo.toml` and `build.rs` carry functional changes. The Rust bindings in
`src/` are unchanged from the published crate.

For the original project documentation and source, see:

- [sherpa-onnx README](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/README.md)
- [`sherpa-onnx-sys` source at v1.13.8](https://github.com/k2-fsa/sherpa-onnx/tree/v1.13.8/sherpa-onnx/rust/sherpa-onnx-sys)
- [Local provenance and verification](UPSTREAM.md)

Singstone obtains the native libraries separately with
[`scripts/fetch-sherpa-onnx.sh`](../../scripts/fetch-sherpa-onnx.sh), which
checks the pinned archive's SHA-256 digest before extracting it.
