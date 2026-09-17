//! Speech recognition producing timed words.

pub mod backend;
pub mod vad;
pub mod whisper;

use crate::types::TimedWord;

pub trait Transcriber {
    /// Transcribe 16 kHz mono samples; words are relative to sample 0.
    fn transcribe(&mut self, samples: &[f32])
    -> Result<Vec<TimedWord>, Box<dyn std::error::Error>>;
}
