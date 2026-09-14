use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};
use std::collections::BTreeMap;
use std::path::Path;

pub struct EmbeddingExtractor {
    inner: SpeakerEmbeddingExtractor,
}

impl EmbeddingExtractor {
    pub fn new(model: &Path, threads: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model.to_string_lossy().into_owned()),
            num_threads: threads.try_into().unwrap_or(i32::MAX),
            debug: false,
            provider: Some("cpu".to_string()),
        };
        let inner = SpeakerEmbeddingExtractor::create(&config)
            .ok_or("failed to create speaker embedding extractor")?;
        Ok(Self { inner })
    }

    pub fn dimension(&self) -> usize {
        self.inner.dim().try_into().unwrap_or(0)
    }

    pub fn embed(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let stream = self.inner.create_stream()?;
        stream.accept_waveform(16_000, samples);
        stream.input_finished();
        if !self.inner.is_ready(&stream) {
            return None;
        }
        let mut embedding = self.inner.compute(&stream)?;
        normalize(&mut embedding).then_some(embedding)
    }
}

pub fn normalize(vector: &mut [f32]) -> bool {
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return false;
    }
    for value in vector {
        *value = (*value as f64 / norm) as f32;
    }
    true
}

pub fn cosine(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let score = left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>();
    score.is_finite().then_some(score as f32)
}

pub fn mean_normalized(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dimension = vectors.first()?.len();
    if dimension == 0 || vectors.iter().any(|vector| vector.len() != dimension) {
        return None;
    }
    let mut mean = vec![0.0f32; dimension];
    let mut used = 0usize;
    for vector in vectors {
        let mut normalized = vector.clone();
        if normalize(&mut normalized) {
            for (sum, value) in mean.iter_mut().zip(normalized) {
                *sum += value;
            }
            used += 1;
        }
    }
    if used == 0 || !normalize(&mut mean) {
        None
    } else {
        Some(mean)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterCandidate {
    pub cluster: u32,
    pub name: String,
    pub score: f32,
}

pub fn unique_matches(candidates: &[ClusterCandidate], threshold: f32) -> BTreeMap<u32, String> {
    let mut ranked = candidates
        .iter()
        .filter(|candidate| candidate.score >= threshold && candidate.score.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.cluster.cmp(&right.cluster))
    });
    let mut matches = BTreeMap::new();
    for candidate in ranked {
        if matches.contains_key(&candidate.cluster) {
            continue;
        }
        matches.insert(candidate.cluster, candidate.name.clone());
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_computes_cosine() {
        let mut vector = vec![3.0, 4.0];
        assert!(normalize(&mut vector));
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((cosine(&vector, &vector).expect("same dimension") - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&vector, &[1.0]), None);
    }

    #[test]
    fn mean_uses_normalized_vectors() {
        let mean = mean_normalized(&[vec![2.0, 0.0], vec![0.0, 3.0]]).expect("valid vectors");
        let expected = 1.0 / 2.0f32.sqrt();
        assert!((mean[0] - expected).abs() < 1e-6);
        assert!((mean[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn threshold_and_one_name_per_cluster_are_enforced() {
        let candidates = vec![
            ClusterCandidate {
                cluster: 0,
                name: "Alice".into(),
                score: 0.7,
            },
            ClusterCandidate {
                cluster: 1,
                name: "Alice".into(),
                score: 0.9,
            },
            ClusterCandidate {
                cluster: 2,
                name: "Bob".into(),
                score: 0.59,
            },
        ];
        let matches = unique_matches(&candidates, 0.6);
        assert_eq!(matches.get(&1).map(String::as_str), Some("Alice"));
        assert_eq!(matches.get(&0).map(String::as_str), Some("Alice"));
        assert!(!matches.contains_key(&2));
    }
}
