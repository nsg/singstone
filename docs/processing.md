# Processing stages

This guide describes the four persistent processing stages, their models, and
their artifacts. See the [README](../README.md#quick-start) for installation and
a first run.

![Top-to-bottom diagram of the Singstone processing pipeline](processing-pipeline.svg)

`transcribe` and `diarize` are independent: neither consumes the other's
output. `recognize` needs diarization, while `render` needs transcription and
diarization and can use recognition when it is available. All timestamps remain
relative to the meeting clock established by `record`.

In the Snap, the first model-backed stage starts the confined model setup
service and waits with a progress bar until the verified per-user cache is
ready. Later stages reuse that cache.

Before a model-backed stage starts, singstone checks the model's file size,
SHA-256 digest and declared purpose against `models.lock`. `--allow-unverified-models`
can explicitly relax that check. Standalone commands generally treat a missing
enabled audio track, model or required upstream artifact as an error, retaining
the existing output. Empty enabled audio and empty diarization are valid empty
inputs; with empty diarization, `recognize` does not open the required model path
or speaker database. The aggregate `process` command keeps its recovery-oriented
behavior: it may skip missing enabled audio and warns on diarization or
recognition failures so a transcript can still be produced. Because it writes
the same artifacts, partial results can replace outputs from earlier tuned stage
runs.

## 1. Transcribe: audio to timed words

```bash
singstone transcribe SESSION \
    --whisper-model ~/.local/share/singstone/models/ggml-large-v3-turbo.bin \
    --language auto --threads 8
```

**Input:** `manifest.json`, every enabled `audio/*.f32le` track, and a
whisper.cpp GGML model. The audio must be 16 kHz mono float PCM, as produced by
`record`.

**Model:** Whisper, through `whisper-rs`. The packaged
`ggml-large-v3-turbo.bin` model is multilingual, and language detection is
enabled by default. A specific Whisper language code can be supplied with
`--language`; any compatible, correctly locked whisper.cpp GGML model can also
be supplied. The same loaded model and inference state are reused for the
microphone and system tracks. Each enabled raw track is read into memory before
its transcription pass.

The stage performs these operations:

1. It runs an energy-based voice activity detector over 20 ms frames. A frame
   above -45 dBFS is considered speech. The detector keeps 200 ms of hangover,
   rejects raw speech bursts shorter than 250 ms, merges retained regions
   separated by at most 500 ms, and pads them by 300 ms. This detector is signal
   processing and uses no model.
2. Speech regions longer than 25 minutes are split at the quietest 20 ms frame
   within 30 seconds of the target boundary. Regions shorter than one second
   are widened into the surrounding source audio. Only a final inference chunk
   that is still shorter than one second is padded with zero samples, because
   Whisper needs enough samples to process it reliably.
3. Retained audio is passed to Whisper in chunks of at most 30 seconds. The
   decoder uses greedy sampling, token timestamps, no previous-text context and
   the configured thread count.
4. Tokens are grouped at whitespace boundaries, with punctuation attached to
   its word, and timestamps are converted back to meeting-relative
   milliseconds. Empty tokens, bracketed annotations such as `[music]`,
   parenthesized annotations and music-note-only output are removed. Words that
   do not touch a retained VAD region are dropped. Timestamps are clamped to the
   source chunk and made non-overlapping so words do not run backward.
5. Words from both sources are sorted by start time, with microphone words first
   on an exact tie, and written atomically.

The main output is one JSON object per line in `words.jsonl`:

```json
{"source":"system","start_ms":10120,"end_ms":10480,"text":"We"}
```

`words.meta.json` records the format version, output hash, hashes of the audio
tracks that were used, Whisper model hash, language and effective thread count.
`render` checks the recorded hashes and warns if the audio or word file changed
after transcription. Deliberate manual edits remain usable; the warning makes
the provenance mismatch visible.

```json
{
  "format_version": 1,
  "output_file": "words.jsonl",
  "output_sha256": "...",
  "mic_audio_sha256": "...",
  "system_audio_sha256": "...",
  "whisper_model_sha256": "...",
  "language": "en",
  "threads": 8
}
```

Disabled or unavailable input tracks have a `null` audio hash. The metadata
does not include the executable version or the fixed VAD settings, so treat it
as a staleness check rather than a complete recipe for reproducing inference.

## 2. Diarize: audio to anonymous speaker intervals

```bash
singstone diarize SESSION \
    --segmentation-model ~/.local/share/singstone/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
    --embedding-model ~/.local/share/singstone/models/nemo_en_titanet_small.onnx \
    --cluster-threshold 1.0
```

**Input:** `manifest.json`, the system audio track, and two ONNX models. The
microphone track is also processed when `--diarize-mic` is supplied.
`words.jsonl` is not an input, so this stage can run before or in parallel with
transcription.

**Models:** sherpa-onnx runs a pyannote speaker-segmentation model and a speaker
embedding model. The segmentation model finds speaker-homogeneous time ranges;
the embedding model represents the voice in each range as a numeric vector.
Fast agglomerative clustering groups similar vectors under numeric cluster IDs.
Both models run on CPU.

Each source is diarized independently, so microphone cluster `0` and system
cluster `0` describe unrelated voices. `--cluster-threshold` controls how
readily clusters are combined. Lower values usually produce more speakers;
values around 0.9 to 1.1 have worked best with
the documented TitaNet model on the tested AMI sample. `--num-speakers N`
replaces automatic speaker-count estimation with that fixed count separately
for each enabled source. Cluster numbers are local implementation labels, not
persistent identities: cluster `2` from one diarization run need not represent
cluster `2` after changing a model or threshold.

The result is sorted by meeting time and written to `diarization.jsonl`:

```json
{"source":"system","start_ms":12000,"end_ms":18300,"cluster":0}
```

Segment timestamps are converted to integer milliseconds and clamped to the
available audio. `diarization.meta.json` records both model hashes, relevant
settings, effective thread count, input audio hashes and the output hash. It
also records whether microphone diarization was enabled, allowing `render` to
inherit that policy automatically.

```json
{
  "format_version": 1,
  "output_file": "diarization.jsonl",
  "output_sha256": "...",
  "mic_audio_sha256": null,
  "system_audio_sha256": "...",
  "segmentation_model_sha256": "...",
  "embedding_model_sha256": "...",
  "diarize_mic": false,
  "cluster_threshold": 1.0,
  "num_speakers": null,
  "threads": 8
}
```

## 3. Recognize: anonymous clusters to enrolled names

Speaker recognition is optional. Without it, `render` assigns anonymous display
names such as `SPEAKER_00`. These names are deterministic for one fixed set of
diarization segments and assignments, but their numbers can change after either
artifact changes. Use `speaker_id`, such as `spk_3` or `mic_1`, to identify the
underlying source and cluster in a particular diarization result.

```bash
singstone recognize SESSION \
    --embedding-model ~/.local/share/singstone/models/nemo_en_titanet_small.onnx \
    --speakers-db ~/.local/share/singstone/speakers.json \
    --speaker-threshold 0.6
```

**Input:** `diarization.jsonl`, the referenced raw audio tracks, a sherpa-onnx
speaker embedding model, and a database created by `singstone enroll`. The
model's exact SHA-256 digest and embedding dimension must match those recorded
when the database was enrolled.

**Model:** any compatible sherpa-onnx speaker embedding model that matches the
enrollment database. It will usually be the model used for diarization, and the
documented model is NeMo TitaNet Small, but diarization and recognition do not
require the same model. Recognition does not run the segmentation model and
does not transcribe speech.

For every distinct `(source, cluster)` pair, the stage:

1. Selects diarized intervals at least 1.5 seconds long, longest first, up to
   30 seconds of speech per cluster. If the last selected interval would exceed
   that budget, it is cut to the remaining duration.
2. Computes a normalized embedding for each selected interval, gives each
   interval equal weight when averaging the embeddings, and normalizes the
   result again.
3. Computes cosine similarity against every enrolled embedding. The highest
   score becomes `best_candidate`.
4. Assigns that candidate's name only when the score is at least
   `--speaker-threshold`. A lower threshold names more clusters but raises the
   chance of a false match. The default is `0.6`.

The versioned `speaker-assignments.json` contains every diarized cluster, even
when it has no usable embedding or confident match:

```json
{
  "format_version": 1,
  "provenance": {
    "diarization_file": "diarization.jsonl",
    "diarization_sha256": "...",
    "embedding_model_sha256": "...",
    "speakers_database_sha256": "...",
    "speaker_threshold": 0.6
  },
  "assignments": [
    {
      "source": "system",
      "cluster": 0,
      "best_candidate": "Alice",
      "score": 0.82,
      "speaker": "Alice"
    }
  ]
}
```

`best_candidate` and `score` remain present below the threshold, while
`speaker` is omitted. This makes threshold problems distinguishable from
embedding failures. Recognition is performed separately for the microphone and
system sources, and the same enrolled name may legitimately match more than one
cluster.

## 4. Render: intermediate artifacts to the final transcript

```bash
singstone render SESSION
```

**Input:** `manifest.json`, `words.jsonl`, `diarization.jsonl`, their metadata
sidecars when present, optionally `speaker-assignments.json`, and the raw audio
tracks when both are available. Missing assignment data is allowed, but
malformed or unsupported assignment metadata or diarization metadata is an
error. This stage uses no ML model and is fast.

When microphone and system speech overlap, `render` compares their timed word
sequences. With both audio tracks available, it confirms candidates from
correlated 10 ms audio energy patterns at a plausible speaker-to-microphone
delay. Only matching microphone words are removed, so a local interjection or
simultaneous unrelated speech is preserved. If either audio track is
unavailable, render falls back to stricter long-sequence text matching. The raw
`words.jsonl` remains unchanged.

Every decision is written to `leakage-suppressions.jsonl`, including the two
matched spans, word count, text similarity, optional audio similarity and
estimated delay. An empty file means no microphone words were suppressed.

For each word, `render` considers only diarization segments from the same audio
source. It chooses the cluster with the greatest timestamp overlap. When there
is no overlap, it can use the nearest segment within 500 ms; otherwise the word
is assigned to `unknown`.

Microphone words normally bypass diarization and use the local speaker name from
`manifest.json`, with speaker ID `local`. If the diarization stage included the
microphone, `render` reads that choice from `diarization.meta.json`; microphone
words then use microphone cluster IDs and recognized or anonymous names instead
of the manifest's local name. A word with no overlapping or nearby microphone
segment becomes `unknown`. For legacy sessions without metadata, `render`
infers the choice from the presence of microphone segments.
`--diarize-mic` and `--diarize-mic=false` provide explicit overrides; an
override that conflicts with current metadata is rejected instead of silently
changing speaker attribution.

Unrecognized clusters receive `SPEAKER_NN` display names in first-seen segment
order. `NN` is a display counter rather than the cluster number. Recognized
names replace only the display name; the source and numeric cluster remain
visible through IDs such as `spk_3` or `mic_1`.

Adjacent words are grouped into utterances. A new utterance starts when the
speaker changes, the silence between words exceeds one second, or an utterance
has reached 15 seconds and ends with sentence punctuation. Whitespace and
punctuation spacing are normalized, and utterances from both sources are sorted
on the common meeting timeline.

The stage replaces the canonical `transcript.jsonl` and readable
`transcript.txt` atomically one file at a time:

```json
{"start_ms":12340,"end_ms":15780,"source":"system","speaker_id":"spk_0","speaker":"Alice","text":"I think we should ship it on Friday."}
```

```text
[00:00:12.340] Alice: I think we should ship it on Friday.
```

Before rendering, singstone compares the available provenance hashes with the
current intermediate files and audio. Edited or replaced word and diarization
artifacts produce warnings but remain renderable. Speaker assignments are
treated more strictly: if their recorded diarization path or hash does not
match, all stored names are discarded and clusters remain anonymous. This
staleness check does not compare the current enrollment database, embedding
model or recognition threshold; rerun `recognize` yourself after changing any
of those inputs.

## Rerun recipes

| Problem or goal | Commands to rerun |
|---|---|
| Incorrect words or language | `transcribe`, then `render` |
| Too many or too few speaker clusters | `diarize`, `recognize`, then `render` |
| Correct clusters but incorrect names | `recognize`, then `render` |
| Manually edited words | `render` |
| Manually edited speaker segments | `recognize`, then `render`; or only `render` for anonymous names |
| Anonymous transcript with no enrollment database | Skip `recognize`; run `render` |
| Several people share the local microphone | `diarize --diarize-mic`, `recognize`, then `render` |

The VAD thresholds and chunk sizes described above are currently fixed. Whisper
chunks do not overlap and do not share text context, so transcription quality
can degrade at a 30-second boundary. Stop `record` before processing its session.
`transcribe` and `diarize` may run concurrently because they write disjoint
artifacts. Concurrent runs of the same stage, or a downstream stage while its
upstream artifact is still being written, are unsupported.
