# Dependency review

Every direct dependency is listed with why it exists, what its build does, its
`unsafe` surface, and the trust assessment. Numbers are for the pinned
versions in `Cargo.lock` on 2026-09-16. The runtime closure is 120 crates
(`cargo tree --edges normal`); the all-target closure is 301 crates.

`cargo audit` checks the lockfile against RustSec advisories. Dependency changes
are reviewed when they are introduced.

## Direct dependencies

| Crate | Purpose | build.rs | Native code | Assessment |
|---|---|---|---|---|
| `pipewire` 0.10.1 + `libspa`, `-sys` | PipeWire capture | bindgen over system headers | links system libpipewire | Official freedesktop project (pipewire-rs). FFI wrappers, ~450 `unsafe` sites, expected for bindings. Pulls `cookie-factory` (dormant, tiny) and `nom` 8 for POD serialization. **Trusted, review recommended for the stream/buffer code path we use.** |
| `whisper-rs` 0.16.0 + `-sys` 0.15.0 | whisper.cpp transcription | cmake build of vendored whisper.cpp; bindgen | whisper.cpp + ggml (C/C++) compiled at build time | Single-maintainer crate (tazz4843, Codeberg) but widely used; no downloads. The real attack surface is whisper.cpp/ggml parsing model files: treat models as untrusted input, hence `models.lock`. **Acceptable; the `-sys` build script deserves a read (cmake flags, feature gating).** |
| `sherpa-onnx` 1.13.8 + `-sys` | diarization, speaker embeddings | selects supplied libraries; otherwise downloads and extracts a release archive | shared `libsherpa-onnx-c-api.so` and `libonnxruntime.so` | Official k2-fsa crates. Snapcraft builds sherpa-onnx 1.13.8 and ONNX Runtime 1.28.2 from exact upstream commits, then supplies that library directory to Cargo. `.cargo/config.toml` points direct developer builds at `target/native/lib`, preventing the crate's download fallback. Shared linking avoids combining ONNX Runtime and whisper.cpp objects built with different C++ toolchains. **Acceptable; native source pins are reviewed in `snap/snapcraft.yaml`, while model URLs and digests live in `docs/models.lock`.** |
| `gtk4` 0.11.4 + `libadwaita` 0.9.2 | native desktop interface | `system-deps` probes installed GTK, GLib, Cairo, Pango, and Libadwaita libraries | links the GNOME platform libraries supplied by the Snap's GNOME extension | Official gtk-rs project under the GNOME organization. These crates are generated, widely deployed FFI bindings. Default features are disabled and the minimum APIs are pinned to GTK 4.10 / Libadwaita 1.5, matching core24. **Accepted for the user-approved native GUI; all recording and inference remain in the existing process and confinement boundary.** |
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

## Native GPU toolchain

The AMD64 Snap builds a second Whisper executable with Intel oneAPI DPC++/C++
2026.1, oneMKL SYCL BLAS 2026.1, and `GGML_SYCL_F16=ON`. The pinned ggml source
contains one unused `syclcompat` include removed by newer upstream versions;
the checked-in one-line patch removes it so this release builds with oneAPI
2026. The build packages come from Intel's signed oneAPI APT repository; its
current signing key is vendored under `snap/keys` and selected by its full
fingerprint in the Snap recipe. The Snap copies the resulting executable's
Intel ELF dependency closure and the dynamically loaded Level Zero Unified
Runtime adapter. The packaged notices come from the same oneAPI installation.

Ubuntu Noble's `libze1` and `libze-intel-gpu1` packages provide the Level Zero
loader and Intel compute driver. The main app already receives `/dev/dri`
access through the GNOME extension's `opengl` interface. A separate SYCL probe
requires an Intel GPU with FP16 and a working queue before the launcher starts
the GPU-linked executable; failure selects the independently built CPU binary.

## Flagged items

1. **sherpa-onnx-sys download fallback.** The checked-in Cargo configuration
   always supplies a local library directory, so the crate never enters its
   prebuilt-download path. Snapcraft fills that directory with shared libraries
   built from exact ONNX Runtime and sherpa-onnx commits.
2. **Native source builds.** ONNX Runtime and sherpa-onnx fetch their own
   commit-pinned source dependencies while Snapcraft builds them. CI performs
   this in the disposable package build environment and keeps the resulting
   source-built libraries inside the Snap.
3. **Model files.** whisper.cpp GGML and ONNX parsers are native code fed
   with multi-hundred-MB files. `singstone process`/`enroll` refuse models
   not listed in `models.lock` unless `--allow-unverified-models` is passed.
4. **`whisper-rs` bus factor.** One maintainer; watch for staleness. The
   crate is thin; a fork would be maintainable.
5. **`cookie-factory` 0.3.3** (via `libspa`): last released 2022. Small and
   pure; low risk, but it will show up in staleness scans.
6. **Duplicate crate versions** (`nom` 7/8, `syn` 2/3, `shlex` 1/2) are
   harmless build-versus-runtime splits.

No Singstone recording, GUI, or inference code opens sockets. The GNOME stack
includes general-purpose GIO APIs, but the strictly confined runtime app has no
Snap network plug; only the separate model-setup service receives network
access.
