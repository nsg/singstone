//! Speech recognition producing timed words.

pub mod backend;
pub mod vad;
pub mod whisper;

use crate::types::{AudioSource, TimedWord};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranscriptionProgress {
    pub source: AudioSource,
    pub processed_seconds: f64,
    pub total_seconds: f64,
    pub audio_seconds: f64,
    pub elapsed_seconds: f64,
    pub decoded_tokens: u64,
    pub tokens_per_second: Option<f64>,
}

impl TranscriptionProgress {
    pub fn realtime_speed(self) -> f64 {
        if self.elapsed_seconds > 0.0 {
            self.audio_seconds / self.elapsed_seconds
        } else {
            0.0
        }
    }

    pub fn fraction(self) -> f64 {
        if self.total_seconds > 0.0 {
            (self.processed_seconds / self.total_seconds).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

pub type ProgressReporter = Arc<dyn Fn(TranscriptionProgress) + Send + Sync>;

pub trait Transcriber {
    /// Transcribe 16 kHz mono samples; words are relative to sample 0.
    fn transcribe(&mut self, samples: &[f32])
    -> Result<Vec<TimedWord>, Box<dyn std::error::Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_speed_is_audio_divided_by_elapsed_time() {
        let progress = TranscriptionProgress {
            source: AudioSource::Mic,
            processed_seconds: 15.0,
            total_seconds: 60.0,
            audio_seconds: 15.0,
            elapsed_seconds: 30.0,
            decoded_tokens: 100,
            tokens_per_second: Some(5.0),
        };
        assert_eq!(progress.realtime_speed(), 0.5);

        let mut no_elapsed = progress;
        no_elapsed.elapsed_seconds = 0.0;
        assert_eq!(no_elapsed.realtime_speed(), 0.0);
        assert_eq!(progress.fraction(), 0.25);
    }
}
