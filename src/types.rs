//! Core data types shared by recording and processing. All persistent
//! timestamps are integer milliseconds of meeting time.

use serde::{Deserialize, Serialize};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FORMAT_VERSION: u32 = 1;
pub const DEFAULT_SPEAKER_THRESHOLD: f32 = 0.72;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioSource {
    Mic,
    System,
}

impl AudioSource {
    pub fn as_str(self) -> &'static str {
        match self {
            AudioSource::Mic => "mic",
            AudioSource::System => "system",
        }
    }
}

impl std::fmt::Display for AudioSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn samples_to_ms(sample: u64) -> u64 {
    sample * 1000 / SAMPLE_RATE as u64
}

/// One line of `words.jsonl`: the smallest timed unit whisper produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedWord {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// One line of `diarization.jsonl`: a speaker-homogeneous interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
    pub cluster: u32,
}

/// One line of `embeddings.jsonl`: a fixed-duration voice sample derived from
/// one diarization segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingChunk {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
    pub cluster: u32,
    pub embedding: Vec<f32>,
}

/// One cluster's recognition diagnostics in `speaker-assignments.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerAssignment {
    pub source: AudioSource,
    pub cluster: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_candidate: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// Inputs and settings used to produce `speaker-assignments.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerAssignmentProvenance {
    pub diarization_file: String,
    pub diarization_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speakers_database_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meeting_sha256: Option<String>,
    pub speaker_threshold: f32,
}

/// Versioned speaker recognition artifact consumed by the render stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerAssignments {
    pub format_version: u32,
    pub provenance: SpeakerAssignmentProvenance,
    pub assignments: Vec<SpeakerAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerCorrection {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: String,
    /// A name carried forward from a lock rather than chosen for this range.
    #[serde(default, skip_serializing_if = "is_false")]
    pub inferred: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerCorrections {
    pub format_version: u32,
    pub corrections: Vec<SpeakerCorrection>,
}

pub const HIDDEN_SOURCES_FORMAT_VERSION: u32 = 2;

/// `hidden-sources.json`: the sources left out of the transcript and the lines
/// of them that stay in it anyway.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenSources {
    pub format_version: u32,
    pub hidden: Vec<AudioSource>,
    /// Keeps the lines named after `manifest.local_speaker`.
    #[serde(default = "default_true")]
    pub keep_local_speaker: bool,
    #[serde(default)]
    pub keep_speakers: Vec<String>,
    #[serde(default)]
    pub keep: Vec<KeptRange>,
}

/// A kept stretch of a hidden source; a word is kept when its midpoint is in
/// `[start_ms, end_ms]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeptRange {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
}

impl KeptRange {
    pub fn contains(self, source: AudioSource, at_ms: u64) -> bool {
        self.source == source && (self.start_ms..=self.end_ms).contains(&at_ms)
    }
}

impl Default for HiddenSources {
    fn default() -> Self {
        Self {
            format_version: HIDDEN_SOURCES_FORMAT_VERSION,
            hidden: Vec::new(),
            keep_local_speaker: true,
            keep_speakers: Vec::new(),
            keep: Vec::new(),
        }
    }
}

impl HiddenSources {
    /// Whether the file has nothing to say beyond the defaults.
    pub fn is_default(&self) -> bool {
        self.hidden.is_empty()
            && self.keep_local_speaker
            && self.keep_speakers.is_empty()
            && self.keep.is_empty()
    }
}

fn default_true() -> bool {
    true
}

/// Where the meeting starts and ends in the recording; an open bound reaches
/// the edge of the recording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeetingSpan {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_ms: Option<u64>,
}

impl MeetingSpan {
    pub fn is_whole(self) -> bool {
        self.start_ms.is_none() && self.end_ms.is_none()
    }

    /// Whether the meeting is not empty, counting an open start as 0 ms.
    pub fn is_valid(self) -> bool {
        self.end_ms
            .is_none_or(|end| end > self.start_ms.unwrap_or(0))
    }

    /// Whether the meeting is not empty in a recording of `recorded_ms`.
    pub fn is_valid_within(self, recorded_ms: u64) -> bool {
        self.is_valid() && self.start_ms.is_none_or(|start| start < recorded_ms)
    }

    /// Whether a line starting at `start_ms` belongs to the meeting.
    pub fn contains(self, start_ms: u64) -> bool {
        self.start_ms.is_none_or(|start| start_ms >= start)
            && self.end_ms.is_none_or(|end| start_ms < end)
    }

    pub fn length_ms(self, recorded_ms: u64) -> u64 {
        self.end_ms
            .unwrap_or(recorded_ms)
            .min(recorded_ms)
            .saturating_sub(self.start_ms.unwrap_or(0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeetingSpanFile {
    pub format_version: u32,
    #[serde(flatten)]
    pub span: MeetingSpan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Utterance {
    pub start_ms: u64,
    pub end_ms: u64,
    pub source: AudioSource,
    pub speaker_id: String,
    pub speaker: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub locked: bool,
    #[serde(
        default,
        skip_serializing_if = "is_false",
        deserialize_with = "deserialize_echo"
    )]
    pub echo: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn deserialize_echo<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum EchoValue {
        Bool(bool),
        Legacy(String),
    }

    Ok(match EchoValue::deserialize(deserializer)? {
        EchoValue::Bool(echo) => echo,
        EchoValue::Legacy(_reason) => true,
    })
}

/// One line of `screenshots.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenshotEntry {
    pub time_ms: u64,
    /// Session-relative path, e.g. `screenshots/000123456-Shot.png`.
    pub file: String,
    /// Absolute path the screenshot tool wrote.
    pub original: String,
}

/// One line of `<source>.timeline.jsonl`: diagnostic capture timing events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TimelineEvent {
    /// Stream produced its first buffer; `sample` is where its audio begins
    /// in the output file (leading silence was inserted before it).
    Start { sample: u64, time_ms: u64 },
    /// A timing gap was detected and filled with silence.
    Xrun {
        sample: u64,
        time_ms: u64,
        missing_samples: u64,
    },
    /// A capture timestamp requested more than the maximum bounded gap fill.
    ClockJump {
        sample: u64,
        time_ms: u64,
        requested_samples: u64,
    },
    /// The writer queue was full and capture blocks were discarded.
    Dropped {
        sample: u64,
        time_ms: u64,
        blocks: u64,
    },
    /// A block timestamp was substantially earlier than the output position.
    Overlap {
        sample: u64,
        time_ms: u64,
        extra_samples: u64,
    },
    /// The stream failed; recording of other streams continued.
    Error {
        sample: u64,
        time_ms: u64,
        message: String,
    },
    /// Clean end of capture.
    Stop { sample: u64, time_ms: u64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamInfo {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipewire_node: Option<String>,
    /// Set when the stream failed during recording.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Recording,
    Stopped,
}

/// `manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub state: SessionState,
    /// RFC 3339 local time with offset, e.g. `2026-09-14T10:30:00+02:00`.
    pub started_wallclock: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_format: String,
    pub mic: StreamInfo,
    pub system: StreamInfo,
    pub local_speaker: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot_dir: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_echo_reads_true_and_false_echo_is_omitted() {
        let legacy = r#"{"start_ms":0,"end_ms":100,"source":"mic","speaker_id":"mic_0","speaker":"Alice","text":"hello","echo":"local_roster"}"#;
        let utterance: Utterance = serde_json::from_str(legacy).expect("read legacy utterance");
        assert!(utterance.echo);

        let plain = Utterance {
            echo: false,
            ..utterance
        };
        let serialized = serde_json::to_value(plain).expect("serialize utterance");
        assert!(serialized.get("echo").is_none());
    }
}
