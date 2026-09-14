# Dependency review

Every direct dependency is listed with why it exists, what its build does, its
`unsafe` surface, and the trust assessment. Numbers are for the pinned
versions in `Cargo.lock` on 2026-09-14. The runtime closure is 56 crates
(`cargo tree --edges normal`); the full build closure is 98 crates after the
`sherpa-onnx-sys` patch described below (it was 201 before).

Policy checks: `cargo audit` (RustSec, no advisories), `cargo deny check`
(licenses, bans, sources; `deny.toml`), `cargo vet` (`supply-chain/`).
The `cargo vet` baseline still exempts most crates; see "Vet status".

## Direct dependencies

| Crate | Purpose | build.rs | Native code | Assessment |
|---|---|---|---|---|
| `pipewire` 0.10.1 + `libspa`, `-sys` | PipeWire capture | bindgen over system headers | links system libpipewire | Official freedesktop project (pipewire-rs). FFI wrappers, ~450 `unsafe` sites, expected for bindings. Pulls `cookie-factory` (dormant, tiny) and `nom` 8 for POD serialization. **Trusted, review recommended for the stream/buffer code path we use.** |
| `whisper-rs` 0.16.0 + `-sys` 0.15.0 | whisper.cpp transcription | cmake build of vendored whisper.cpp; bindgen | whisper.cpp + ggml (C/C++) compiled at build time | Single-maintainer crate (tazz4843, Codeberg) but widely used; no downloads. The real attack surface is whisper.cpp/ggml parsing model files: treat models as untrusted input, hence `models.lock`. **Acceptable; the `-sys` build script deserves a read (cmake flags, feature gating).** |
| `sherpa-onnx` 1.13.8 + `-sys` | diarization, speaker embeddings | **patched locally, see below** | prebuilt shared libs `libsherpa-onnx-c-api.so`, `libonnxruntime.so` (32 MB) | Official k2-fsa crate. Upstream `-sys` build script downloads GitHub release archives **without any checksum** and drags in `ureq`/`rustls`/`ring`/`zip`/`aes`/`zstd`/`tar`/`bzip2` (~100 crates) as build-dependencies. We replace the crate with `vendor/sherpa-onnx-sys` (same `src/`, 40-line build script that only links from `SHERPA_ONNX_LIB_DIR`) via `[patch.crates-io]`, and pin the archive SHA-256 in `scripts/fetch-sherpa-onnx.sh`. Shared rather than static linking is required: the prebuilt ONNX Runtime and the locally compiled whisper.cpp use different C++ toolchains, and linking both statically into one executable merges incompatible libstdc++ template instantiations (crash in `std::regex` inside ONNX Runtime device discovery). **The prebuilt binary itself is the weakest link in the whole chain**: it is a GitHub CI artifact, not reproducible by us. Building sherpa-onnx + onnxruntime from source is the documented upgrade path. |
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

1. **sherpa-onnx-sys download-at-build (fixed by patch).** Unverified network
   fetch during `cargo build` is unacceptable for a privacy-first tool. The
   patch removes it; `cargo build --offline` now succeeds once
   `third_party/sherpa-onnx` exists.
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

## Vet status

`cargo vet` is initialised with imports from Mozilla, Google, Bytecode
Alliance, ISRG and Embark. Crates not covered by those audits at the pinned
version are exempted in `supply-chain/config.toml` as the initial baseline.
Each exemption must be replaced by a project audit (`cargo vet certify`) or
an import before a release; do not add new exemptions to silence CI.
Priority order for project audits: `sherpa-onnx`, `sherpa-onnx-sys`
(vendored copy), `whisper-rs`, `whisper-rs-sys`, `pipewire`, `libspa`,
`pipewire-sys`, `libspa-sys`, then the build tooling.
