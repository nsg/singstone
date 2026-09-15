//! Core data types shared by recording and processing. All persistent
//! timestamps are integer milliseconds of meeting time.

use serde::{Deserialize, Serialize};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FORMAT_VERSION: u32 = 1;

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
    pub speaker_threshold: f32,
}

/// Versioned speaker recognition artifact consumed by the render stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerAssignments {
    pub format_version: u32,
    pub provenance: SpeakerAssignmentProvenance,
    pub assignments: Vec<SpeakerAssignment>,
}

/// One line of `transcript.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Utterance {
    pub start_ms: u64,
    pub end_ms: u64,
    pub source: AudioSource,
    pub speaker_id: String,
    pub speaker: String,
    pub text: String,
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
