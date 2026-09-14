<div align="center">
  <h1>singstone</h1>
  <p>Local-only meeting recorder, transcriber and speaker diarizer for Linux and PipeWire.</p>

[![AI usage: mostly](https://nsg.github.io/aibadge/mostly.svg)](https://nsg.github.io/aibadge/#mostly)
</div>

## About

singstone records a meeting from two PipeWire sources at once, the microphone
and the system audio (what you hear from the remote participants), and later
turns the recording into a timestamped, speaker-attributed transcript. Every
step runs on your machine: capture, whisper.cpp transcription, sherpa-onnx
diarization and optional recognition of enrolled voices. The binary contains
no network code.

Recording and processing are separate. Recording writes append-only raw PCM
plus timing metadata and survives crashes; processing is an offline,
idempotent step that can be re-run with different models or thresholds. The
outputs (`transcript.jsonl`, `screenshots.jsonl`, copied screenshots) are
meant to feed a separate summarization pipeline. Summaries, LLMs, OCR and a
GUI are deliberately out of scope.

## Features

- Records microphone, system audio (sink monitor) or both, as 16 kHz mono
  `f32le` files that share one meeting clock (`sample / 16000` = meeting time).
- Real-time safe capture: the PipeWire callback only copies into a lock-free
  queue; gaps and drops are filled with silence and logged, never hidden.
- Watches your screenshot directory (inotify) and files each new screenshot
  under its meeting timestamp.
- Word-level transcription with whisper.cpp, energy-based voice activity
  detection to keep whisper away from silence.
- Speaker diarization of the remote track with sherpa-onnx (pyannote
  segmentation + speaker embeddings); the mic track is attributed to you
  unless `--diarize-mic`.
- Enroll known voices once; later recordings name them when the match is
  confident, otherwise speakers stay `SPEAKER_NN`.
- Supply-chain policy: pinned native libraries with verified hashes,
  SHA-256-locked model files, `cargo audit`/`deny`/`vet`, offline builds.

## Quick start

Ubuntu 24.04, PipeWire, Rust stable. Headphones are strongly recommended when
recording both sources; otherwise the mic hears the speakers and remote
speech appears in both tracks.

```bash
# build dependencies
sudo apt install cmake clang libclang-dev libpipewire-0.3-dev libspa-0.2-dev pkg-config

# verified sherpa-onnx native libraries (only network step of the build)
scripts/fetch-sherpa-onnx.sh
cargo build
# the binary loads libsherpa-onnx-c-api.so / libonnxruntime.so from
# third_party/sherpa-onnx/lib or from its own directory ($ORIGIN); when
# installing, copy those two files next to the binary.

# models (not downloaded by singstone; see docs/models.lock for hashes)
mkdir -p ~/.local/share/singstone/models && cd ~/.local/share/singstone/models
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx
mkdir -p ~/.config/singstone && cp /path/to/singstone/docs/models.lock ~/.config/singstone/
```

Record, then process:

```bash
singstone devices
singstone record --mic default --system default \
    --screenshots ~/Pictures/Screenshots --local-speaker "Me"
# ... meeting ... Ctrl-C

singstone process session-20260914-103000 \
    --whisper-model ~/.local/share/singstone/models/ggml-base.en.bin \
    --segmentation-model ~/.local/share/singstone/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
    --embedding-model ~/.local/share/singstone/models/nemo_en_titanet_small.onnx
```

## Configuration

Model paths and the speaker database can be given as flags or environment
variables:

| Variable | Flag | Meaning |
|---|---|---|
| `SINGSTONE_WHISPER_MODEL` | `--whisper-model` | whisper.cpp GGML model |
| `SINGSTONE_SEGMENTATION_MODEL` | `--segmentation-model` | sherpa-onnx pyannote segmentation `model.onnx` |
| `SINGSTONE_EMBEDDING_MODEL` | `--embedding-model` | sherpa-onnx speaker embedding model |
| `SINGSTONE_MODELS_LOCK` | `--models-lock` | trusted model manifest; default `$XDG_CONFIG_HOME/singstone/models.lock` |
| `SINGSTONE_SPEAKERS_DB` | `--speakers-db` | enrolled speakers; default `$XDG_DATA_HOME/singstone/speakers.json` |

Models are treated as untrusted input to native parsers: `process` and
`enroll` refuse a model whose SHA-256 is not in `models.lock` unless
`--allow-unverified-models` is passed. Each entry also carries a `purpose`
(`transcription`, `diarization-segmentation` or `speaker-embedding`) and the
file size, both checked against the role the model is used for.
`scripts/models-lock-add.sh FILE` appends a skeleton entry for a new model.

Optional Cargo features: `vulkan`, `intel-sycl`, `openblas` (whisper.cpp
acceleration, CPU is the default). Diarization runs on CPU.

## CLI

```text
singstone devices                     list PipeWire sources and sinks
singstone record [--mic X] [--system Y] [--screenshots DIR]
                 [--local-speaker NAME] [--output-dir DIR] [--duration SECS]
singstone process SESSION [--diarize-mic] [--no-diarize] [--skip-transcription]
                 [--language en] [--threads N] [--speaker-threshold 0.6]
                 [--cluster-threshold 1.0] [--num-speakers N]
singstone enroll NAME SAMPLE... [--replace]      samples: 16 kHz mono WAV or raw f32le
singstone speakers
```

Device arguments accept `default`, `none`, a PipeWire node id or a node name
as printed by `devices`. System audio captures the selected sink's monitor.

`process` is idempotent and rewrites its outputs atomically. Use
`--skip-transcription` to reuse an existing `words.jsonl` while re-running
diarization or speaker recognition with different thresholds (the whisper
model is then not needed); whisper is by far the slowest stage. Diarization
and speaker-recognition failures degrade to anonymous or `unknown` speakers
instead of aborting. `--cluster-threshold` controls how eagerly diarized
segments are merged into speakers (lower = more speakers); 0.9 to 1.1 gave
the best results with the titanet-small embedding model on a 4-speaker AMI
meeting (the 4 real speakers came out with 75-87 % purity, split over about
a dozen clusters). `--speaker-threshold` is the minimum cosine similarity for naming a
cluster after an enrolled speaker; unmatched clusters stay `SPEAKER_NN`.

### Session layout

```text
session-20260914-103000/
├── manifest.json            state, start time, format, sources, local speaker
├── audio/mic.f32le          16 kHz mono float PCM, meeting-aligned
├── audio/mic.timeline.jsonl start/xrun/dropped/overlap/clock_jump/stop events
├── audio/system.f32le
├── audio/system.timeline.jsonl
├── screenshots/000123456-Shot.png
├── screenshots.jsonl        {"time_ms":123456,"file":"screenshots/...","original":"/home/..."}
├── words.jsonl              {"source":"system","start_ms":10120,"end_ms":10480,"text":"We"}
├── diarization.jsonl        {"source":"system","start_ms":12000,"end_ms":18300,"cluster":0}
├── transcript.jsonl         canonical output, see below
└── transcript.txt           [00:00:12.340] Alice: I think we should ship it on Friday.
```

`transcript.jsonl` lines:

```json
{"start_ms":12340,"end_ms":15780,"source":"system","speaker_id":"spk_0","speaker":"Alice","text":"I think we should ship it on Friday."}
{"start_ms":15920,"end_ms":17620,"source":"mic","speaker_id":"local","speaker":"Me","text":"That works for me."}
```

Convert a track to WAV with ffmpeg if needed:
`ffmpeg -f f32le -ar 16000 -ac 1 -i audio/system.f32le system.wav`.

### Timing model

Both audio files start at the same meeting instant (the monotonic clock
when `record` started). Each captured block is placed by PipeWire's own
stream time (`now - delay`); late stream start, xruns and queue overruns are
filled with silence and logged in the per-stream timeline, so a crashed or
interrupted recording still lines up. Gap insertion needs confirmation by
the following block, is capped at 60 s per block, and a capped gap is
logged as `clock_jump`. Session names and `started_wallclock` use the local
time zone read from `/etc/localtime`, falling back to UTC.

The headless test harness (`scripts/pw-headless.sh`) feeds the virtual mic
through a `pw-loopback`, which by itself introduces about 20 ms between the
two paths; the integration test allows 50 ms. Verify alignment on real
hardware before relying on it for cross-talk analysis.

## Development

```bash
# extra packages for the PipeWire integration tests (headless daemon)
sudo apt install pipewire pipewire-bin wireplumber pipewire-audio-client-libraries
cargo install --locked cargo-audit cargo-deny cargo-vet

cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
source scripts/pw-headless.sh                    # headless PipeWire with test-sink / test-mic
SINGSTONE_PW_TEST=1 cargo test --test record_pipewire -- --test-threads=1
SINGSTONE_TEST_MODELS=... SINGSTONE_TEST_SAMPLES=... cargo test --test process_ami
cargo audit && cargo deny check && cargo vet
```

`docs/design-spec.md` is the original design document, `docs/dependencies.md`
the dependency review and supply-chain policy.

## License

MIT OR Apache-2.0
