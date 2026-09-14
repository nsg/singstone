//! Link against locally supplied sherpa-onnx libraries. Never downloads.
use std::env;
use std::path::PathBuf;

const STATIC_LIBS: &[&str] = &[
    "sherpa-onnx-c-api",
    "sherpa-onnx-core",
    "kaldi-decoder-core",
    "sherpa-onnx-kaldifst-core",
    "sherpa-onnx-fstfar",
    "sherpa-onnx-fst",
    "kaldi-native-fbank-core",
    "kissfft-float",
    "piper_phonemize",
    "espeak-ng",
    "ucd",
    "onnxruntime",
    "ssentencepiece_core",
];

fn main() {
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    if env::var_os("DOCS_RS").is_some() {
        return;
    }
    let lib_dir = env::var_os("SHERPA_ONNX_LIB_DIR")
        .map(PathBuf::from)
        .expect("SHERPA_ONNX_LIB_DIR must point at a directory of verified sherpa-onnx libraries (see scripts/fetch-sherpa-onnx.sh)");
    assert!(lib_dir.is_dir(), "SHERPA_ONNX_LIB_DIR is not a directory: {}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    let shared = env::var_os("CARGO_FEATURE_SHARED").is_some();
    if shared {
        println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
        println!("cargo:rustc-link-lib=dylib=onnxruntime");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    } else {
        for lib in STATIC_LIBS {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}
