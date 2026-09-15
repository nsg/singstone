# Dependency review

Every direct dependency is listed with why it exists, what its build does, its
`unsafe` surface, and the trust assessment. Numbers are for the pinned
versions in `Cargo.lock` on 2026-09-15. The runtime closure is 56 crates
(`cargo tree --edges normal`); the full build closure is 201 crates.

Policy checks: `cargo audit` (RustSec advisories) and `cargo deny check`
(licenses, bans, and sources; `deny.toml`).

## Direct dependencies

| Crate | Purpose | build.rs | Native code | Assessment |
|---|---|---|---|---|
| `pipewire` 0.10.1 + `libspa`, `-sys` | PipeWire capture | bindgen over system headers | links system libpipewire | Official freedesktop project (pipewire-rs). FFI wrappers, ~450 `unsafe` sites, expected for bindings. Pulls `cookie-factory` (dormant, tiny) and `nom` 8 for POD serialization. **Trusted, review recommended for the stream/buffer code path we use.** |
| `whisper-rs` 0.16.0 + `-sys` 0.15.0 | whisper.cpp transcription | cmake build of vendored whisper.cpp; bindgen | whisper.cpp + ggml (C/C++) compiled at build time | Single-maintainer crate (tazz4843, Codeberg) but widely used; no downloads. The real attack surface is whisper.cpp/ggml parsing model files: treat models as untrusted input, hence `models.lock`. **Acceptable; the `-sys` build script deserves a read (cmake flags, feature gating).** |
| `sherpa-onnx` 1.13.8 + `-sys` | diarization, speaker embeddings | selects supplied libraries; otherwise downloads and extracts a release archive | prebuilt shared libs `libsherpa-onnx-c-api.so`, `libonnxruntime.so` (32 MB) | Official k2-fsa crates. `.cargo/config.toml` sets `SHERPA_ONNX_LIB_DIR`, so the build script uses the separately fetched, checksum-verified libraries and does not enter its download path. Its downloader and archive dependencies remain in Cargo's build closure. Shared linking avoids combining ONNX Runtime and whisper.cpp objects built with different C++ toolchains. **The prebuilt binary is the weakest link:** it is a GitHub CI artifact that we do not reproduce. Building sherpa-onnx and ONNX Runtime from source is the upgrade path. |
| `clap` 4.6 (`derive`, `env`) | CLI | none (clap_derive proc macro) | none | clap-rs org, ubiquitous. Adds ~15 small crates (anstream/anstyle/…). Chosen over a hand-written parser on the user's request. |
| `serde`, `serde_json` | JSON/JSONL | version-detection build scripts; `serde_derive` proc macro | none | dtolnay. `serde_json` now depends on `zmij` (float formatting, also dtolnay). Ubiquitous. |
| `crossbeam-queue` 0.3 | RT-safe bounded queue | `crossbeam-utils` build script (feature detection) | none | crossbeam-rs org. Lock-free code, `unsafe`-heavy by nature but well reviewed. |
| `rustix` 1.1 (`fs`, `event`, `time`, `process`) | inotify, poll, monotonic clock | build script (target detection) | raw syscalls via `linux-raw-sys` | Bytecode Alliance. Large `unsafe` surface (syscall wrappers) but the most reviewed such crate; also a transitive dependency of `pipewire`. |
| `sha2` 0.10 | model and speaker-DB fingerprinting | none (`cpufeatures` runtime detection) | none | RustCrypto. Pure Rust; integrity check only, not security-critical. |

Build-only tooling of note: `bindgen` 0.72 (+ `clang-sys`, `libloading` to
dlopen libclang at build time), `cmake`, `system-deps` (+ `toml`), `cc`.
All Rust-lang / gtk-rs / Alex Crichton lineage. They run arbitrary code at
build time by design; they are the reason the build must run in a trusted
environment, not why it must be online.

## Flagged items

1. **sherpa-onnx-sys download fallback.** An unverified network fetch during
   `cargo build` is unacceptable for a privacy-first tool. The checked-in Cargo
   configuration supplies `third_party/sherpa-onnx/lib`, bypassing that code.
   Trusted builds run offline after the verified fetch step.
2. **Prebuilt ONNX Runtime / sherpa shared libraries.** Opaque binaries from
   GitHub Releases, loaded at run time via rpath. Mitigation: SHA-256 pinned and verified before use, and
   the fetch is an explicit, separate step. Recommended follow-up: reproduce
   the libraries from source in CI and compare.
3. **Model files.** whisper.cpp GGML and ONNX parsers are native code fed
   with multi-hundred-MB files. `singstone process`/`enroll` refuse models
   not listed in `models.lock` unless `--allow-unverified-models` is passed.
4. **`whisper-rs` bus factor.** One maintainer; watch for staleness. The
   crate is thin; a fork would be maintainable.
5. **`cookie-factory` 0.3.3** (via `libspa`): last released 2022. Small and
   pure; low risk, but it will show up in staleness scans.
6. **Duplicate crate versions** (`nom` 7/8, `syn` 2/3, `shlex` 1/2) are
   build-vs-runtime splits from bindgen; harmless, listed by `cargo deny` as
   warnings.

Nothing in the runtime closure opens sockets. The recorder's privacy claim is
verifiable: `cargo tree --edges normal` contains no HTTP, TLS or socket crate.
