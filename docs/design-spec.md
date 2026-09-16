# Design Specification: `meetrec` — Local-Only Linux Meeting Recorder, Transcriber and Speaker Diarizer

## 1. Purpose

Build a small, reliable Rust application for Ubuntu/Linux that:

1. Records microphone audio.
2. Records system/output audio from PipeWire.
3. Can record either source independently or both simultaneously.
4. Transcribes recorded speech locally.
5. Diarizes speakers locally.
6. Optionally recognizes previously enrolled speakers from voice embeddings.
7. Watches a screenshot directory while recording and associates new screenshots with meeting timestamps.
8. Produces a simple, machine-readable timestamped transcript.

The application stops there.

Summarization, meeting notes, LLM processing, OCR, screenshot interpretation, RAG, synchronization to note-taking software, etc. are explicitly outside this project.

The intended downstream flow is:

```text
meetrec
    │
    ├── transcript.jsonl
    ├── screenshots.jsonl
    └── screenshots/*
             │
             ▼
    separate post-processing system
             │
             ├── LLM
             ├── vision model
             └── final Markdown meeting notes
```

The recorder/transcriber itself must require **no network access at runtime**.

---

## 2. Design priorities

In order:

1. Reliability of the original recording.
2. Privacy/local-only operation.
3. Simple architecture.
4. Reproducible and auditable dependency chain.
5. Correct timestamps and synchronization.
6. Good transcription.
7. Speaker diarization.
8. Known-speaker recognition.
9. Performance/GPU acceleration.
10. UI polish.

Do not sacrifice reliable recording to perform real-time inference.

The initial implementation should therefore be **record first, process afterward**.

---

# 3. Non-goals

Do not implement these in the initial project:

- Live meeting summaries.
- LLM integration.
- HTTP API clients.
- Cloud APIs.
- Screenshot OCR or vision analysis.
- GUI.
- Browser integration.
- Calendar integration.
- Meeting-bot functionality.
- Joining Zoom/Teams/Meet as a participant.
- Echo cancellation.
- Noise suppression.
- Source separation.
- Training neural networks.
- Real-time transcription.
- Python runtime.
- FFmpeg runtime dependency.
- PulseAudio compatibility layer if native PipeWire is available.

A later application can consume the artifacts from this project.

---

# 4. Target platform

Initial supported target:

```text
Linux x86_64
Ubuntu
PipeWire audio
Rust stable
```

Linux-specific solutions are encouraged. Portability to Windows/macOS is not a design goal.

This allows us to use Linux primitives directly rather than pulling in large cross-platform abstraction crates.

---

# 5. High-level architecture

```text
                       ┌─────────────────┐
Microphone ─PipeWire──►│                 │
                       │ Recording core  ├──► mic.f32le
System sink ─PipeWire─►│                 ├──► system.f32le
                       │                 │
                       └────────┬────────┘
                                │
                         timing metadata
                                │
                                ▼
                     ┌────────────────────┐
Screenshot dir ─────►│ Screenshot watcher│
                     └─────────┬──────────┘
                               │
                               ▼
                        screenshots.jsonl


           AFTER RECORDING:

system.f32le ──► Whisper ───────────► timed words
       │
       └────────► sherpa-onnx ──────► diarization
                                      │
                                      ▼
                              speaker clusters
                                      │
                        optional voiceprint match
                                      │
                                      ▼
mic.f32le ─────► Whisper ───────────► local-speaker words
                                      │
                                      ▼
                                timestamp merge
                                      │
                                      ▼
                              transcript.jsonl
```

Recording and inference must be separate subsystems.

If inference crashes, it must never destroy or invalidate the original recording.

---

# 6. Command-line interface

One binary.

Suggested name:

```text
meetrec
```

Minimum commands:

```text
meetrec devices

meetrec record \
    --mic default \
    --system default \
    --screenshots ~/Pictures/Screenshots \
    --local-speaker "My Name"

meetrec record --mic default --system none

meetrec record --mic none --system default

meetrec process ./session-20260914-103000 \
    --whisper-model ~/models/ggml-large-v3-turbo.bin

meetrec process SESSION --diarize-mic

meetrec enroll "Alice" alice-sample-1.f32le alice-sample-2.f32le

meetrec speakers
```

Device arguments should accept:

```text
default
none
<PipeWire node ID>
<PipeWire node name>
```

Do not introduce a large CLI framework solely for this.

A small hand-written `std::env::args()` parser is sufficient initially.

---

# 7. Session layout

Each recording gets an independent directory:

```text
session-20260914-103000/
├── manifest.json
├── audio/
│   ├── mic.f32le
│   ├── mic.timeline.jsonl
│   ├── system.f32le
│   └── system.timeline.jsonl
├── screenshots/
│   ├── 000123456-screenshot.png
│   └── 000845211-screenshot.png
├── screenshots.jsonl
├── words.jsonl
├── diarization.jsonl
└── transcript.jsonl
```

`manifest.json` should contain at minimum:

```json
{
  "format_version": 1,
  "state": "stopped",
  "started_wallclock": "2026-09-14T10:30:00+02:00",
  "sample_rate": 16000,
  "channels": 1,
  "sample_format": "f32le",
  "mic": {
    "enabled": true,
    "pipewire_node": "..."
  },
  "system": {
    "enabled": true,
    "pipewire_node": "..."
  },
  "local_speaker": "My Name"
}
```

While recording:

```json
"state": "recording"
```

Change it to `"stopped"` only after a clean stop.

A crashed session must remain processable.

---

# 8. Audio capture

## PipeWire

Use the official Rust PipeWire bindings:

```text
pipewire
```

As of this design, `pipewire` 0.10.1 is the current Rust binding and exposes safe Rust interfaces around libpipewire.

### Microphone

Capture from a normal PipeWire source.

### System audio

Capture the selected output sink's monitor.

PipeWire explicitly provides:

```text
stream.capture.sink=true
```

for capturing sink output rather than an ordinary source.

Allow an explicit target object so the user can choose which sink is being monitored.

`devices` should enumerate usable sources and sinks through the PipeWire registry and show human-readable names plus stable-enough identifiers.

---

# 9. Audio format

Request:

```text
16,000 Hz
mono
32-bit floating point
```

from PipeWire for both streams.

That is already appropriate input for Whisper and the speaker models.

Prefer allowing PipeWire's graph to perform the conversion/remixing rather than adding an independent resampling library.

Validate the negotiated format rather than assuming negotiation succeeded.

If the desired format cannot be obtained, fail with a useful message in the MVP rather than silently introducing complicated conversion paths.

---

# 10. Real-time capture rules

The PipeWire process callback is effectively real-time code.

It must do almost nothing.

Allowed:

```text
dequeue buffer
read timing
copy samples into preallocated block
push block into bounded queue
increment atomics
return
```

Forbidden inside the callback:

```text
filesystem writes
model inference
JSON serialization
malloc-heavy operations
logging
mutex waits
network activity
formatting strings
speaker recognition
transcription
```

Use:

```text
crossbeam_queue::ArrayQueue
```

between the PipeWire callback and a normal writer thread.

If the queue fills:

1. Never block the PipeWire thread.
2. Increment an atomic dropped-block counter.
3. Drop that block.
4. Record the discontinuity in timing metadata.

---

# 11. Audio timestamps and synchronization

This is important.

Do **not** independently timestamp microphone and system buffers using wall-clock calls and hope they line up.

Use PipeWire stream timing.

Maintain:

```text
session monotonic t0
PipeWire graph timestamp
sample index
```

for each stream.

The output audio files should behave as if they begin at the same meeting timestamp.

If one stream starts later, insert initial silence.

If an xrun produces a timing gap, insert the appropriate amount of silence.

That gives the useful invariant:

```text
meeting_time_seconds = sample_index / 16000
```

for both files.

Also preserve diagnostic timing information in:

```text
mic.timeline.jsonl
system.timeline.jsonl
```

Example:

```json
{"sample":320000,"time_ms":20000,"event":"xrun","missing_samples":640}
```

This greatly simplifies everything downstream.

---

# 12. Storage format

Store the primary audio as raw append-only PCM:

```text
mic.f32le
system.f32le
```

Do not introduce a WAV library just to write a 44-byte header.

The format is known from `manifest.json`:

```text
IEEE f32
little endian
mono
16000 Hz
```

Benefits:

- trivial implementation,
- append-only,
- excellent crash recovery,
- no container finalization,
- no extra dependency,
- easy memory mapping or streaming later.

A conversion tool can always generate WAV afterward if needed.

---

# 13. Microphone versus system-audio semantics

When recording both sources, keep them completely separate.

Normally:

```text
microphone = local user
system     = remote meeting participants
```

Therefore the default processing behavior should be:

### Microphone

Transcribe it, but assign every speech segment directly to:

```text
local_speaker
```

There is no reason to spend diarization compute distinguishing one known microphone user.

### System audio

Transcribe and diarize normally.

### Optional physical-room mode

Support:

```text
--diarize-mic
```

for cases where several people are physically present around the microphone.

---

# 14. Echo limitation

If the laptop speakers are playing remote participants and the microphone also hears those speakers, the same speech will occur in both tracks.

Do **not** solve this in the first release.

Document:

> Headphones are strongly recommended when simultaneously recording microphone and system audio.

A future version can support PipeWire echo-cancellation or transcript deduplication.

Keep the canonical transcript `source` field so such processing can later be added cleanly.

---

# 15. Speech recognition

Use:

```text
whisper-rs
```

which is the Rust binding to `whisper.cpp`.

For the first implementation:

```text
CPU = mandatory
Vulkan = optional feature
Intel SYCL = optional/experimental feature
```

Enable token timestamps.

The processing layer wants the smallest reasonably timed units Whisper can supply, rather than only whole paragraphs.

Canonical intermediate output:

```json
{"source":"system","start_ms":10120,"end_ms":10480,"text":"We"}
{"source":"system","start_ms":10480,"end_ms":10830,"text":"should"}
{"source":"system","start_ms":10830,"end_ms":11210,"text":"ship"}
```

The exact Whisper output granularity can be normalized internally.

---

# 16. Intel GPU strategy

Start with this order:

```text
1. Vulkan
2. CPU
3. SYCL as an optional experimental build
```

Suggested Cargo features:

```toml
[features]
default = []
vulkan = ["whisper-rs/vulkan"]
intel-sycl = ["whisper-rs/intel-sycl"]
openblas = ["whisper-rs/openblas"]
```

Keep sherpa-onnx on CPU initially.

Diarization is an offline operation and does not need to complicate the GPU story in version 1.

---

# 17. Speaker diarization

Use the **official**:

```text
sherpa-onnx
```

Rust crate.

Do not use the older third-party `sherpa-rs` binding.

Sherpa's diarization architecture is a good match because it already combines:

```text
speaker segmentation
        +
speaker embedding extraction
        +
clustering
```

into offline diarization.

A reasonable initial model pair is:

```text
sherpa-onnx-pyannote-segmentation-3-0
+
nemo_en_titanet_small.onnx
```

for English-heavy meetings.

**Do not assume model licenses are interchangeable.** The model manifest must explicitly record and review each selected model's license.

---

# 18. Speaker identification / enrollment

Do not train a model.

Use voice embeddings.

Enrollment flow:

```text
audio samples for Alice
        │
        ▼
speaker embedding model
        │
        ▼
several Alice embeddings
        │
        ▼
normalize / aggregate
        │
        ▼
local speaker database
```

Suggested database:

```text
$XDG_DATA_HOME/meetrec/speakers.json
```

Example conceptual schema:

```json
{
  "embedding_model": {
    "name": "nemo_en_titanet_small",
    "sha256": "...",
    "dimension": 192
  },
  "speakers": {
    "Alice": {
      "embeddings": [
        [...]
      ]
    },
    "Bob": {
      "embeddings": [
        [...]
      ]
    }
  }
}
```

Never compare embeddings generated by different models.

The model SHA-256 and embedding dimension are part of the database identity.

For recognition:

1. Diarize the recording.
2. Gather several clean/long segments belonging to each cluster.
3. Compute embeddings.
4. Aggregate them.
5. Search the enrolled-speaker database.
6. Only assign a person's name if similarity exceeds a conservative threshold.
7. Otherwise leave the speaker anonymous.

For example:

```text
Alice
Bob
SPEAKER_03
SPEAKER_04
```

False negatives are preferable to confidently assigning speech to the wrong person.

The matching threshold must be configurable and calibrated with actual meeting audio.

---

# 19. Combining diarization with transcription

Whisper and the diarization model operate independently.

Processing system audio:

```text
                    ┌── Whisper ───────► timed words
system.f32le ───────┤
                    └── diarizer ──────► speaker intervals
```

Suppose diarization produces:

```text
00:12.000 – 00:18.300 SPEAKER_00
00:18.600 – 00:23.100 SPEAKER_01
```

For each timed word:

1. Calculate overlap with speaker intervals.
2. Assign it to the speaker with maximum overlap.
3. If there is no overlap, allow a small configurable nearest-segment tolerance.
4. Otherwise mark it unknown.

Then group adjacent words into utterances.

Split an utterance when:

- speaker changes,
- source changes,
- sentence boundary is appropriate,
- or silence exceeds roughly one second.

Do not force overlapping speech into a fake total ordering.

If two people speak simultaneously, preserve both utterances and their actual overlapping timestamp ranges.

---

# 20. Final transcript format

Canonical format:

```text
transcript.jsonl
```

Example:

```json
{"start_ms":12340,"end_ms":15780,"source":"system","speaker_id":"spk_0","speaker":"Alice","text":"I think we should ship it on Friday."}
{"start_ms":15920,"end_ms":17620,"source":"mic","speaker_id":"local","speaker":"Me","text":"That works for me."}
{"start_ms":18010,"end_ms":21540,"source":"system","speaker_id":"spk_1","speaker":"SPEAKER_02","text":"I'll update the deployment plan."}
```

Required fields:

```text
start_ms
end_ms
source
speaker_id
speaker
text
```

`source` is:

```text
mic
system
```

Human-readable output can optionally also be generated:

```text
[00:12.340] Alice: I think we should ship it on Friday.
[00:15.920] Me: That works for me.
[00:18.010] SPEAKER_02: I'll update the deployment plan.
```

JSONL remains canonical.

---

# 21. Screenshot capture

The application does **not** take screenshots itself.

It watches an existing screenshot directory.

Because the target is explicitly Linux, use Linux `inotify` directly rather than adding a generic cross-platform watcher dependency.

Use:

```text
rustix
```

with its `fs` feature.

Watch primarily for:

```text
CLOSE_WRITE
MOVED_TO
```

so a file is not copied while the screenshot application is still writing it.

On detection:

1. Validate extension against configured types such as PNG/JPEG/WebP.
2. Record the meeting-relative monotonic timestamp.
3. Copy the file into the session's `screenshots/` directory.
4. Preserve its original filename in metadata.
5. Do not decode it.
6. Do not OCR it.
7. Do not send it anywhere.

Example:

```json
{"time_ms":123456,"file":"screenshots/000123456-Screenshot.png","original":"/home/me/Pictures/Screenshots/Screenshot.png"}
```

Renaming the session copy with its meeting timestamp is useful:

```text
000123456-Screenshot.png
```

A later vision/LLM pipeline can correlate:

```text
screenshot at 02:03
+
transcript around 02:03
```

without this recorder knowing anything about vision models.

---

# 22. Dependencies

Keep the dependency tree intentionally small.

Recommended direct runtime dependencies:

| Crate | Purpose |
|---|---|
| `pipewire` | PipeWire audio capture |
| `whisper-rs` | Whisper transcription |
| `sherpa-onnx` | diarization + speaker embeddings |
| `serde` | structured data |
| `serde_json` | JSON/JSONL |
| `crossbeam-queue` | bounded RT → writer queue |
| `rustix` | Linux inotify / filesystem primitives |

Do **not** automatically add crates such as:

```text
tokio
reqwest
clap
anyhow
notify
hound
rubato
ffmpeg wrappers
GUI frameworks
database frameworks
logging frameworks
```

unless a concrete requirement justifies them.

Writing 50–100 lines of straightforward Rust is preferable to adding another low-trust dependency for trivial functionality.

---

# 23. Unsafe-code policy

Application-owned Rust code should begin with:

```rust
#![forbid(unsafe_code)]
```

Native interaction is unavoidable because:

```text
PipeWire
whisper.cpp
ONNX Runtime / sherpa-onnx
```

ultimately contain C/C++/FFI code.

Unsafe code should therefore be isolated to upstream bindings that have been reviewed.

Do not write application-specific unsafe FFI unless there is a compelling reason.

---

# 24. Supply-chain security policy

Popularity is **not** sufficient proof that a crate is secure.

Likewise, this specification should not claim that every recommended speech/audio crate has received an independent formal security audit when that cannot be established.

Instead, make dependency review part of the build policy.

Use:

```text
cargo-audit / RustSec
```

### Release rule

A production release requires:

```text
cargo audit
```

The check must pass.

For higher-risk domain-specific dependencies:

```text
pipewire
whisper-rs
sherpa-onnx
their -sys crates
```

perform a project-specific source review of the exact pinned version when the
risk warrants it.

---

# 25. Special sherpa-onnx supply-chain warning

This deserves explicit treatment.

The official `sherpa-onnx` Rust crate may download matching prebuilt native sherpa-onnx libraries during its build if no local library path is supplied.

That behavior is unacceptable for this project's release pipeline.

Therefore:

```text
NEVER allow sherpa-onnx's build.rs to download binaries during trusted builds.
```

The Snap build therefore compiles the pinned sherpa-onnx and ONNX Runtime
revisions from reviewed source and sets:

```bash
SHERPA_ONNX_LIB_DIR=/snapcraft/staged/lib
```

Direct Cargo development uses `target/native/lib`. Keeping the variable set
even when that directory is absent makes the build fail clearly instead of
entering the crate's download fallback. CI enforces the source-built path by
building the complete Snap.

---

# 26. Dependency pinning and reproducible builds

Repository must commit:

```text
Cargo.toml
Cargo.lock
model manifest
```

Rules:

- No wildcard dependencies.
- No unpinned Git dependencies.
- Prefer crates.io releases.
- No third-party Cargo registries.
- Review build scripts and proc macros particularly carefully.
- Pin native libraries.
- Pin model files.
- Build against a supported/current Rust stable toolchain.
- CI release builds use `--locked`.
- Materialize dependencies with `cargo vendor` for offline release builds; do
  not commit those generated copies.
- Trusted release builds should succeed with network disabled.

A desirable final release procedure is approximately:

```bash
cargo vendor
cargo audit
cargo test --locked
cargo clippy --locked -- -D warnings
cargo build --release --locked --offline
```

---

# 27. Model supply-chain policy

The Snap downloads models through a separate, per-user setup service. The main
recording and processing app has no network interface. Its first model-backed
command starts the service, waits on an atomic progress file, and continues
after setup succeeds.

Maintain something such as:

```text
models.lock
```

containing:

```text
logical model name
upstream project
upstream URL
upstream revision/version
license
expected filename
byte size
SHA-256
purpose
```

Example conceptual entry:

```text
name: whisper-large-v3-turbo
purpose: transcription
sha256: ...

name: sherpa-pyannote-segmentation-3.0
purpose: diarization-segmentation
sha256: ...

name: nemo-en-titanet-small
purpose: speaker-embedding
sha256: ...
```

The service downloads only URLs in the installed manifest. It checks the
download artifact and final model size and SHA-256, then atomically publishes
the verified file under `$SNAP_USER_COMMON/models`. The cache survives Snap
updates. Processing verifies models again before passing them to native code.

Native ML parsers are a significant attack surface.

Treat arbitrary model files as untrusted executable-like inputs rather than harmless data.

---

# 28. Runtime privacy guarantees

Recording and inference should have no network interface. Only the confined
model setup service receives outbound network access.

In particular:

```text
do not depend on reqwest
do not open TCP/UDP sockets
do not implement telemetry
do not implement update checks
do not report crashes remotely
```

The Snap's per-app interfaces make the privacy claim enforceable: `singstone`
has `home` and `pipewire`, while `model-setup` has only `network`.

The user's later summarization application may communicate with another trusted machine, but that is a separate process and a separate security boundary.

---

# 29. Filesystem security

Meeting recordings contain sensitive data.

Session directories should be private to the user.

Target:

```text
directory: 0700
files:     0600
```

or equivalently operate with:

```text
umask 0077
```

Do not dump transcript contents to diagnostic logs by default.

Do not store temporary copies under globally readable `/tmp`.

Speaker embeddings should receive the same protection as meeting recordings.

---

# 30. Failure behavior

### Audio capture failure

Recording the remaining active source should continue where sensible, but the session metadata must clearly indicate failure.

### Queue overrun

Continue recording, record an xrun/drop diagnostic, and insert corresponding silence.

### Screenshot watcher failure

Warn and continue recording audio.

### Whisper failure

Leave original audio untouched.

### Diarization failure

Still produce a transcript with generic/unknown speakers if possible.

### Speaker recognition failure

Keep anonymous `SPEAKER_N` identities.

### Crash while recording

Raw PCM remains valid.

### Crash while processing

Derived files should be rewritten atomically on the next `process`.

Original capture artifacts are immutable after recording ends.

---

# 31. Idempotent processing

This should work repeatedly:

```bash
meetrec process SESSION
meetrec process SESSION
meetrec process SESSION
```

Processing should derive all outputs from original recording artifacts.

Changing a transcription model or speaker threshold should not require another meeting recording.

Use temporary files such as:

```text
transcript.jsonl.tmp
```

and atomically rename once complete.

---

# 32. Suggested module structure

```text
src/
├── main.rs
├── cli.rs
├── session.rs
├── audio/
│   ├── mod.rs
│   ├── pipewire.rs
│   ├── capture.rs
│   ├── timing.rs
│   └── writer.rs
├── screenshot/
│   ├── mod.rs
│   └── inotify.rs
├── transcription/
│   ├── mod.rs
│   └── whisper.rs
├── diarization/
│   ├── mod.rs
│   └── sherpa.rs
├── speaker/
│   ├── mod.rs
│   ├── embedding.rs
│   └── database.rs
├── merge/
│   ├── mod.rs
│   └── utterances.rs
└── format/
    ├── mod.rs
    └── jsonl.rs
```

Keep interfaces narrow enough that inference engines can later be replaced without changing recording code.

For example:

```rust
trait Transcriber {
    fn transcribe(&self, samples: &[f32]) -> Result<Vec<TimedWord>, Error>;
}

trait Diarizer {
    fn diarize(&self, samples: &[f32]) -> Result<Vec<SpeakerSegment>, Error>;
}
```

Do not over-engineer this into a generic plugin system.

Static interfaces are enough.

---

# 33. Core internal types

Conceptually:

```rust
struct TimedWord {
    start_ms: u64,
    end_ms: u64,
    text: String,
    source: AudioSource,
}

struct SpeakerSegment {
    start_ms: u64,
    end_ms: u64,
    cluster: u32,
}

struct Utterance {
    start_ms: u64,
    end_ms: u64,
    source: AudioSource,
    speaker_id: String,
    speaker: String,
    text: String,
}

enum AudioSource {
    Mic,
    System,
}
```

Use integer timestamps in milliseconds externally.

Avoid floating-point time values in persistent formats.

---

# 34. Tests

## Unit tests

Must cover:

- timestamp/sample conversion,
- insertion of silence for gaps,
- word ↔ diarization overlap matching,
- nearest-speaker fallback,
- utterance grouping,
- overlapping speech,
- speaker-model fingerprint validation,
- speaker matching thresholds,
- JSONL round trips,
- screenshot extension filters,
- screenshot timestamps.

## Integration tests

Create short deterministic fixture recordings.

Test:

```text
known audio
    ↓
transcription
    ↓
expected approximate timestamps
```

For PipeWire integration, create virtual sources/sinks where practical and inject known signals into both.

Verify mic/system synchronization remains within an explicit tolerance, initially perhaps:

```text
≤ 50 ms
```

Test discontinuities by simulating dropped capture blocks and ensure silence is inserted correctly.

---

# 35. CI requirements

At minimum:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo audit
```

Additionally, CI builds and reviews the strict Snap from exact source pins and
the pinned model manifest. It verifies that only the model setup service has a
network interface, confirms weights are absent, installs the package, and runs
CLI smoke tests. Cargo tests run inside that package build against the same
source-built native libraries that are shipped.

---

# 36. Recommended implementation sequence

### Phase 1 — Recorder

Implement only:

```text
devices
record mic
record system
record both
raw PCM
timestamps
session manifest
```

Prove synchronization and crash recovery first.

### Phase 2 — Screenshot watcher

Add:

```text
rustix inotify
timestamp
copy into session
screenshots.jsonl
```

### Phase 3 — Whisper

Add offline transcription and word/token timestamps.

### Phase 4 — Diarization

Add sherpa-onnx processing of system audio.

### Phase 5 — Merge

Produce:

```text
transcript.jsonl
```

with anonymous speakers.

At this point the application already fulfills its main purpose.

### Phase 6 — Speaker enrollment

Add embeddings and recognition of recurring participants.

### Phase 7 — Acceleration

Benchmark CPU.

Then introduce Vulkan.

Only after that investigate Intel SYCL.

Do not begin the project by debugging GPU drivers.

---

# 37. Acceptance criteria for MVP

The MVP is complete when the following works:

```bash
meetrec record \
    --mic default \
    --system default \
    --screenshots ~/Pictures/Screenshots \
    --local-speaker "Me"
```

The user can have a one-hour meeting, take screenshots with their normal screenshot tool, press Ctrl-C, and then run:

```bash
meetrec process session-...
```

and receive:

```text
transcript.jsonl
screenshots.jsonl
screenshots/*
```

with transcript entries of approximately this form:

```text
00:01:42.300 Alice:
We should probably move that task to next week.

00:01:47.100 Me:
Yeah, I agree.

00:01:49.800 SPEAKER_02:
I'll update the issue after this meeting.
```

Recording, transcription, diarization and speaker recognition happen locally.
A network connection is needed once to fill the verified model cache.

---

# 38. Dependency/trust conclusion

Use these as the initial dependency choices:

```text
PipeWire capture       → pipewire
ASR                    → whisper-rs / whisper.cpp
diarization            → official sherpa-onnx
speaker recognition    → official sherpa-onnx
RT handoff             → crossbeam-queue
serialization          → serde + serde_json
screenshot watching    → rustix/inotify
everything trivial     → std / hand-written code
```

This is intentionally not a blanket claim that every crate above has received a formal third-party security audit.

The project's chain of trust should instead be:

```text
small dependency set
        ↓
exact pinned versions
        ↓
project review for high-risk crates
        ↓
RustSec/advisory checking
        ↓
review of build.rs/proc macros/native code
        ↓
offline Cargo source snapshot
        ↓
verified native libraries
        ↓
reviewed model manifest
        ↓
SHA-256-verified per-user cache
        ↓
network isolated to the setup service
```

That gives substantially stronger assurance than choosing dependencies merely because they have many downloads.

---

# 39. Instruction to the implementation agent

Implement the system incrementally and resist expanding its scope.

Prefer readable Rust and explicit Linux APIs over frameworks.

Keep application-owned code free of `unsafe`.

Do not add a dependency without documenting:

```text
why it is necessary
why std/in-house code is insufficient
maintainer/project provenance
security/audit status
build-script behavior
unsafe-code surface
native dependencies
```

Any dependency addition that substantially increases the supply-chain surface should require explicit approval.

The most important architectural boundary is:

```text
RECORDING
   ↓
immutable local artifacts
   ↓
OFFLINE PROCESSING
   ↓
timestamped transcript + screenshot index
```

Do not put LLM functionality inside this project.

The finished application should be boring, deterministic, private, and dependable.
