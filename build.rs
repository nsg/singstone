//! Embed run-time search paths for the sherpa-onnx shared libraries: next to
//! the binary ($ORIGIN, for installed copies) and the build-time library
//! directory (for development and tests). Link-arg instructions emitted by a
//! dependency's build script do not propagate to this binary, hence this file.

fn main() {
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    if let Some(dir) = std::env::var_os("SHERPA_ONNX_LIB_DIR") {
        let dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.into());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}
