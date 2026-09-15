<div align="center">
  <h1>singstone</h1>
  <p>Local-only meeting recorder, transcriber and speaker diarizer for Linux and PipeWire.</p>

[![AI usage: mostly](https://nsg.github.io/aibadge/mostly.svg)](https://nsg.github.io/aibadge/#mostly)
</div>

## About

singstone records microphone and system audio on one meeting clock, optionally
files screenshots, then builds a timestamped, speaker-attributed transcript.
Capture, Whisper transcription, sherpa-onnx diarization, and voice recognition
all run locally. The binary contains no network code.

![Top-to-bottom overview of the Singstone workflow](docs/workflow-overview.svg)

## Features

- Captures a microphone, a PipeWire sink monitor, or both as aligned 16 kHz mono
  `f32le` audio.
- Keeps filesystem work outside the real-time callback; fills gaps with silence
  and logs capture problems.
- Copies new screenshots into the session with meeting-relative timestamps.
- Transcribes with whisper.cpp and diarizes with pyannote segmentation plus
  speaker embeddings through sherpa-onnx.
- Matches diarized clusters to enrolled voices; uncertain matches remain
  anonymous.
- Verifies model size, purpose, and SHA-256 before loading native parsers.

## Quick start

Download the `singstone-snap` artifact from a successful CI run, then install
the unsigned build and connect its PipeWire interface:

```bash
sudo snap install --dangerous ./singstone_0.1.0_amd64.snap
sudo snap connect singstone:pipewire
```

The package already contains the Whisper, pyannote, and TitaNet models pinned
by [`docs/models.lock`](docs/models.lock). Use headphones when capturing both
sources so remote speech does not leak into the microphone track.

Record a meeting, stop with Ctrl-C, then process the session:

```bash
singstone devices
singstone record --mic default --system default \
  --screenshots ~/Pictures/Screenshots --local-speaker "Me"
singstone process session-20260914-103000
```

The results are `SESSION/transcript.jsonl` for programs and
`SESSION/transcript.txt` for people.

## Configuration

Model paths and the speaker database accept flags or environment variables:

| Variable | Flag | Meaning |
|---|---|---|
| `SINGSTONE_WHISPER_MODEL` | `--whisper-model` | whisper.cpp GGML model |
| `SINGSTONE_SEGMENTATION_MODEL` | `--segmentation-model` | pyannote segmentation ONNX model |
| `SINGSTONE_EMBEDDING_MODEL` | `--embedding-model` | sherpa-onnx speaker embedding model |
| `SINGSTONE_MODELS_LOCK` | `--models-lock` | trusted model manifest; `$XDG_CONFIG_HOME/singstone/models.lock`, falling back to `~/.config/singstone/models.lock` |
| `SINGSTONE_SPEAKERS_DB` | `--speakers-db` | enrolled speakers; `$XDG_DATA_HOME/singstone/speakers.json`, falling back to `~/.local/share/singstone/speakers.json` |

The Snap sets all five variables to its bundled models, manifest, and persistent
speaker database. Flags and environment variables remain useful for source
builds or intentionally testing another model.

Model-backed commands reject models whose purpose, size, and SHA-256 do not
match `models.lock`. A missing lock file is also an error. Use
`--allow-unverified-models` only when intentionally testing an unlisted model.
Add an entry to the configured copy with
`scripts/models-lock-add.sh FILE "${XDG_CONFIG_HOME:-$HOME/.config}/singstone/models.lock"`,
then review and complete its metadata before use.

## CLI

| Command | Purpose |
|---|---|
| `devices` | List selectable PipeWire sources and sinks |
| `record` | Capture audio and optional screenshots into a new session |
| `process` | Run all configured offline processing stages |
| `transcribe` | Produce word-level text and timestamps |
| `diarize` | Produce anonymous speaker intervals |
| `recognize` | Match speaker clusters to enrolled voices |
| `render` | Build final transcript files from intermediate artifacts |
| `enroll` | Add voice samples to the speaker database |
| `speakers` | List enrolled speakers |

Run `singstone COMMAND --help` for every flag. Device arguments accept
`default`, `none`, a PipeWire node ID, or a node name printed by `devices`.
System audio uses the selected sink's monitor.

### Recording

![Top-to-bottom diagram of the Singstone recording stage](docs/recording-stage.svg)

`record` creates a private session and a common monotonic start time. PipeWire
callbacks copy samples and timing into bounded queues; writer threads align and
append each audio track. Confirmed gaps and dropped blocks become silence so the
tracks stay synchronized, with every correction logged in `*.timeline.jsonl`.
The optional inotify watcher copies completed screenshots on the same clock.

Stop with Ctrl-C, SIGTERM, or `--duration`. One failed source does not stop a
healthy source. See [Recording](docs/recording.md) for timing, failure handling,
screenshot behavior, and the complete artifact contract.

### Processing

![Top-to-bottom diagram of the Singstone processing pipeline](docs/processing-pipeline.svg)

Use `process` for the normal one-command path. Use the split stages to inspect an
intermediate artifact or tune one model without repeating unrelated work:

```bash
singstone transcribe SESSION --whisper-model /path/to/whisper.bin
singstone diarize SESSION --segmentation-model /path/to/segmentation.onnx \
  --embedding-model /path/to/embedding.onnx
singstone recognize SESSION --embedding-model /path/to/embedding.onnx \
  --speakers-db /path/to/speakers.json
singstone render SESSION
```

| Stage | Contract |
|---|---|
| `transcribe` | Enabled audio → energy VAD → Whisper → `words.jsonl` + metadata |
| `diarize` | System audio and optional mic → pyannote segmentation, embeddings, clustering → `diarization.jsonl` + metadata |
| `recognize` | Diarization, source audio, and matching model/database → cosine comparison → `speaker-assignments.json` |
| `render` | Words, diarization, and optional current assignments → `transcript.jsonl` and `transcript.txt`; no model |

`--cluster-threshold` controls cluster merging; lower values usually produce
more speakers. `--speaker-threshold` controls how confident a voice match must
be. The microphone uses `--local-speaker` unless `--diarize-mic` is enabled.

| Problem | Rerun |
|---|---|
| Incorrect words or language | `transcribe`, `render` |
| Incorrect speaker boundaries or count | `diarize`, `recognize`, `render` |
| Correct clusters but incorrect names | `recognize`, `render` |
| Manually edited words | `render` |
| Several people share the microphone | `diarize --diarize-mic`, `recognize`, `render` |

Keep these boundaries in mind:

- Stop recording before processing. `transcribe` and `diarize` may run in
  parallel; same-stage writers and downstream/upstream overlap are unsupported.
- Standalone stages preserve the previous output when a required input fails.
  `process` may skip missing enabled audio and recover from diarization or
  recognition failures, so a partial run can replace tuned artifacts.
- Metadata hashes warn when audio or intermediate files change. If assignments
  refer to different diarization, `render` discards all stored names.
- Enrollment samples must be at least two seconds of 16 kHz mono WAV or raw
  `f32le`. Recognition requires the enrolled model's exact SHA-256 and embedding
  dimension.

See [Processing stages](docs/processing.md) for the VAD and chunking algorithm,
model roles, clustering and recognition details, artifact schemas, provenance,
and render rules.

### Session artifacts

| Path | Purpose |
|---|---|
| `manifest.json` | Session state, clock, format, sources, and local speaker |
| `audio/*.f32le` | Meeting-aligned raw audio |
| `audio/*.timeline.jsonl` | Capture timing and error diagnostics |
| `screenshots/`, `screenshots.jsonl` | Timestamped screenshot copies and index |
| `words.jsonl`, `words.meta.json` | Timed transcription and provenance |
| `diarization.jsonl`, `diarization.meta.json` | Speaker intervals and provenance |
| `speaker-assignments.json` | Optional recognized names and provenance |
| `transcript.jsonl`, `transcript.txt` | Final machine-readable and text transcripts |

All meeting-relative timestamps are milliseconds from the meeting start. With
ffmpeg installed, convert raw audio when another tool needs WAV:

```bash
ffmpeg -f f32le -ar 16000 -ac 1 -i SESSION/audio/system.f32le system.wav
```

## Snap package

[`snap/snapcraft.yaml`](snap/snapcraft.yaml) is the complete distribution
recipe. It builds each native component from reviewed source instead of
fetching opaque shared libraries:

| Part | Pinned input | Packaged output |
|---|---|---|
| ONNX Runtime | upstream commit `33ca962…` (`v1.28.2`) | CPU `libonnxruntime.so` |
| sherpa-onnx | upstream commit `11afbd0…` (`v1.13.8`) | `libsherpa-onnx-c-api.so`, linked to the source-built ONNX Runtime |
| Singstone | this checkout plus `Cargo.lock`, Rust 1.98.1 | release executable and whisper.cpp compiled by `whisper-rs` |
| Models | immutable URLs and SHA-256 checksums | Whisper, pyannote segmentation, TitaNet, and `models.lock` |

Build on Ubuntu 24.04 x86-64 in Snapcraft's isolated LXD environment:

```bash
sudo snap install snapcraft --classic
snapcraft pack --use-lxd
```

The finished Snap uses `strict` confinement. `home` lets commands read and
write ordinary files in your home directory. `pipewire` lets the recorder use
the PipeWire socket and must be connected once after installation. The package
declares neither `network` nor `network-bind`, so snapd denies runtime network
access. Network access is used only while Snapcraft retrieves the pinned source
and model inputs.

## Development

The Snap is the supported complete build. For a direct Cargo workflow, first
build ONNX Runtime 1.28.2 and sherpa-onnx 1.13.8 from source and put their shared
libraries in `target/native/lib`. The checked-in Cargo configuration deliberately
sets that local path so `sherpa-onnx-sys` cannot silently download a prebuilt
archive.

```bash
sudo apt install pipewire pipewire-bin wireplumber pipewire-audio-client-libraries
cargo install --locked cargo-audit

cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
source scripts/pw-headless.sh
SINGSTONE_PW_TEST=1 cargo test --test record_pipewire -- --test-threads=1
SINGSTONE_TEST_MODELS=/path/to/models SINGSTONE_TEST_SAMPLES=/path/to/samples \
  cargo test --test process_ami
cargo audit
```

Optional Cargo features `vulkan`, `intel-sycl`, and `openblas` accelerate
whisper.cpp. Diarization runs on CPU. CI builds the complete Snap from the
pinned sources, runs the Rust tests during that build, reviews the package,
checks its interfaces, installs it, and runs CLI smoke tests.

## Documentation

- [Recording internals](docs/recording.md)
- [Processing stages](docs/processing.md)
- [Dependency review and supply-chain policy](docs/dependencies.md)
- [Original design specification](docs/design-spec.md)

## License

MIT
