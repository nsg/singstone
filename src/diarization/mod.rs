//! Speaker diarization producing speaker-homogeneous intervals.

pub mod cluster;
pub mod pipeline;
pub mod segmentation;

use crate::speaker::embedding::EmbeddingExtractor;
use crate::types::{AudioSource, SpeakerSegment};
pub use pipeline::Clustering;
use segmentation::Segmenter;
use std::path::{Path, PathBuf};

pub struct Diarizer {
    segmentation_model: PathBuf,
    /// Loaded until first use, then reloaded per source, so its working memory
    /// is released before the embedding workers need theirs.
    segmenter: Option<Segmenter>,
    extractor: EmbeddingExtractor,
    threads: usize,
}

impl Diarizer {
    /// Segmentation runs on `threads` inference threads; embeddings run on
    /// `threads` workers with one inference thread each.
    pub fn new(
        segmentation_model: &Path,
        embedding_model: &Path,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let extractor = EmbeddingExtractor::new(embedding_model, 1)?;
        let segmenter = Segmenter::new(segmentation_model, threads)?;
        Ok(Self {
            segmentation_model: segmentation_model.to_path_buf(),
            segmenter: Some(segmenter),
            extractor,
            threads: threads.max(1),
        })
    }

    pub fn diarize(
        &mut self,
        samples: &[f32],
        source: AudioSource,
        clustering: Clustering,
    ) -> Result<Vec<SpeakerSegment>, Box<dyn std::error::Error>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let mut segmenter = match self.segmenter.take() {
            Some(segmenter) => segmenter,
            None => Segmenter::new(&self.segmentation_model, self.threads)?,
        };
        let segmentation = segmenter.segment(samples)?;
        drop(segmenter);
        Ok(pipeline::diarize(
            &segmentation,
            samples,
            &self.extractor,
            clustering,
            source,
            self.threads,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarizer_smoke_test_when_models_are_configured() {
        let Some(models) = std::env::var_os("SINGSTONE_TEST_MODELS") else {
            return;
        };
        let models = PathBuf::from(models);
        let mut diarizer = Diarizer::new(
            &models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            &models.join("nemo_en_titanet_small.onnx"),
            1,
        )
        .expect("create diarizer");
        let runtimes = std::fs::read_to_string("/proc/self/maps")
            .expect("read memory maps")
            .lines()
            .filter_map(|line| line.split_whitespace().nth(5))
            .filter(|path| path.contains("libonnxruntime.so"))
            .map(str::to_owned)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            runtimes.len(),
            1,
            "one ONNX Runtime is loaded: {runtimes:?}"
        );
        let audio = std::env::var_os("SINGSTONE_TEST_SAMPLES").map_or_else(
            || vec![0.0; 16_000],
            |samples| {
                let bytes = std::fs::read(
                    PathBuf::from(samples).join("session-ami-3min/audio/system.f32le"),
                )
                .expect("read sample audio");
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .take(16_000)
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            },
        );
        let clustering = Clustering {
            threshold: 0.5,
            num_speakers: None,
        };
        let segments = diarizer
            .diarize(&audio, AudioSource::System, clustering)
            .expect("diarize");
        assert!(segments.iter().all(|segment| segment.end_ms <= 1_000));
    }
}
