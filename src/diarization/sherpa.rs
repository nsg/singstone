use super::Diarizer;
use crate::types::{AudioSource, SpeakerSegment, samples_to_ms};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};
use std::path::Path;

pub struct SherpaDiarizer {
    inner: OfflineSpeakerDiarization,
    source: AudioSource,
}

impl SherpaDiarizer {
    pub fn new(
        segmentation_model: &Path,
        embedding_model: &Path,
        source: AudioSource,
        threads: usize,
        cluster_threshold: f32,
        num_speakers: Option<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threads = threads.try_into().unwrap_or(i32::MAX);
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(segmentation_model.to_string_lossy().into_owned()),
                    ..Default::default()
                },
                num_threads: threads,
                debug: false,
                provider: Some("cpu".to_string()),
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(embedding_model.to_string_lossy().into_owned()),
                num_threads: threads,
                debug: false,
                provider: Some("cpu".to_string()),
            },
            clustering: FastClusteringConfig {
                threshold: cluster_threshold,
                num_clusters: num_speakers
                    .and_then(|count| i32::try_from(count).ok())
                    .unwrap_or(-1),
                compute_confidence: false,
            },
            ..Default::default()
        };
        let inner = OfflineSpeakerDiarization::create(&config)
            .ok_or("failed to create sherpa-onnx diarizer")?;
        if inner.sample_rate() != 16_000 {
            return Err(format!(
                "diarization model expects {} Hz, not 16000 Hz",
                inner.sample_rate()
            )
            .into());
        }
        Ok(Self { inner, source })
    }
}

impl Diarizer for SherpaDiarizer {
    fn diarize(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<SpeakerSegment>, Box<dyn std::error::Error>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let result = self
            .inner
            .process(samples)
            .ok_or("sherpa-onnx diarization failed")?;
        let audio_end_ms = samples_to_ms(samples.len() as u64);
        Ok(result
            .sort_by_start_time()
            .into_iter()
            .filter(|segment| segment.speaker >= 0 && segment.end >= segment.start)
            .map(|segment| SpeakerSegment {
                source: self.source,
                start_ms: seconds_to_ms(segment.start).min(audio_end_ms),
                end_ms: seconds_to_ms(segment.end).min(audio_end_ms),
                cluster: segment.speaker as u32,
            })
            .collect())
    }
}

fn seconds_to_ms(seconds: f32) -> u64 {
    if seconds.is_finite() && seconds > 0.0 {
        (f64::from(seconds) * 1000.0).round() as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_diarizer_smoke_test_when_models_are_configured() {
        let Some(models) = std::env::var_os("SINGSTONE_TEST_MODELS") else {
            return;
        };
        let models = std::path::PathBuf::from(models);
        let mut diarizer = SherpaDiarizer::new(
            &models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            &models.join("nemo_en_titanet_small.onnx"),
            AudioSource::System,
            1,
            0.5,
            None,
        )
        .expect("create diarizer");
        let audio = std::env::var_os("SINGSTONE_TEST_SAMPLES").map_or_else(
            || vec![0.0; 16_000],
            |samples| {
                let bytes = std::fs::read(
                    std::path::PathBuf::from(samples).join("session-ami-3min/audio/system.f32le"),
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
        let segments = diarizer.diarize(&audio).expect("diarize");
        assert!(segments.iter().all(|segment| segment.end_ms <= 1_000));
    }
}
