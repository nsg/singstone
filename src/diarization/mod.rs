//! Speaker diarization producing speaker-homogeneous intervals.

pub mod sherpa;

use crate::types::SpeakerSegment;

pub trait Diarizer {
    fn diarize(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<SpeakerSegment>, Box<dyn std::error::Error>>;
}
