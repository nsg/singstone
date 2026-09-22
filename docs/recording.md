# Recording

`singstone record` captures microphone audio, system audio, or both into a new
session directory. It performs no model inference and needs no network access.
See the [README](../README.md#quick-start) for installation and a first run.

```bash
singstone record --mic default --system default \
    --screenshots ~/Pictures/Screenshots \
    --local-speaker "Me"
```

![Top-to-bottom diagram of the Singstone recording stage](recording-stage.svg)

## Inputs and outputs

Device arguments accept `default`, `none`, a PipeWire node ID, or a node name
shown by `singstone devices`. System audio comes from the selected sink's
monitor. At least one audio source must be enabled.

The command creates a private `session-YYYYMMDD-HHMMSS/` directory containing:

| Artifact | Contents |
|---|---|
| `manifest.json` | State, start time, audio format, source selection, and local speaker |
| `audio/mic.f32le` | Meeting-aligned 16 kHz mono float PCM from the microphone |
| `audio/system.f32le` | Meeting-aligned 16 kHz mono float PCM from the sink monitor |
| `audio/*.timeline.jsonl` | Start, xrun, dropped-block, overlap, clock-jump, error, and stop events |
| `screenshots/` | Copies of screenshots created while recording |
| `screenshots.jsonl` | Meeting timestamp, copied path, and original path for each screenshot |

Disabled sources do not produce audio or timeline files.

## Audio capture

Both streams share one monotonic start time. Each PipeWire callback validates
the negotiated 16 kHz mono `f32le` format, copies samples and PipeWire timing
into 2,048-sample blocks, and pushes them into its bounded lock-free queue. It
does no filesystem I/O.

A writer thread per source places blocks using PipeWire's stream time
(`now - delay`). Late starts and confirmed gaps are filled with silence so the
files remain aligned to the meeting clock. Blocks whose timestamps overlap
continue at the current output position, and substantial overlaps are logged.
Queue overruns, xruns, clock jumps, and stream errors are also appended to the
source timeline instead of being hidden. Gap insertion is capped at 60 seconds
per block; larger requested gaps are recorded as `clock_jump` events.

Audio is flushed at least once per second. On shutdown, writers append a stop
event, flush and sync their files, and record any stream error in the manifest.
One failed source does not stop another healthy source. The command fails if all
enabled sources fail or capture no samples.

## Screenshot watcher

With `--screenshots DIR`, an inotify thread watches for completed writes and
files moved into `DIR`. Matching extensions are copied into the session using a
meeting-time prefix, synced, and appended to `screenshots.jsonl`. Existing files
are ignored. Watcher errors are warnings and do not interrupt audio capture.

## Stopping

Stop with Ctrl-C, SIGTERM, or `--duration SECONDS`. Singstone joins the watcher
and writers, preserves per-stream errors, and atomically changes the manifest
state from `recording` to `stopped`.

An interrupted session keeps its append-only capture files. Processing can read
recovered audio while the manifest still says `recording`, but finish recording
before starting any processing stage to avoid races with growing files.

## Remote control

On the `io.github.nsg.Singstone` session bus name, `/io/github/nsg/Singstone/Recorder` implements `io.github.nsg.Singstone.Recorder`.

| Methods | Signal | Status keys |
|---|---|---|
| `StartRecording`, `StopRecording`, `GetStatus` | `StatusChanged` | `recording`, `stopping`, `mic`, `system`, `mic_level`, `system_level`, `elapsed`, `screenshots` |
