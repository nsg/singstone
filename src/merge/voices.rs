//! The diarized microphone voices, with how much each sounds like a remote
//! voice: the loudspeaker's echo still sounds like the remote speaker it
//! repeats, someone in the room does not.

use crate::speaker::embedding::{cosine, mean_normalized};
use crate::types::{AudioSource, EmbeddingChunk, SpeakerSegment};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct MicVoices {
    /// Diarized microphone turns as `(start_ms, end_ms, cluster)`, in order.
    pub turns: Vec<(u64, u64, u32)>,
    /// For each microphone cluster with a voiceprint, its highest cosine
    /// similarity to a system cluster's voiceprint.
    pub remote_similarity: BTreeMap<u32, f32>,
}

impl MicVoices {
    pub fn new(segments: &[SpeakerSegment], chunks: &[EmbeddingChunk]) -> Self {
        let mut turns = segments
            .iter()
            .filter(|segment| segment.source == AudioSource::Mic)
            .map(|segment| (segment.start_ms, segment.end_ms, segment.cluster))
            .collect::<Vec<_>>();
        turns.sort_unstable();
        let system = voiceprints(chunks, AudioSource::System);
        let remote_similarity = voiceprints(chunks, AudioSource::Mic)
            .into_iter()
            .filter_map(|(cluster, mic)| {
                let best = system
                    .values()
                    .filter_map(|remote| cosine(&mic, remote))
                    .max_by(f32::total_cmp)?;
                Some((cluster, best))
            })
            .collect();
        Self {
            turns,
            remote_similarity,
        }
    }
}

fn voiceprints(chunks: &[EmbeddingChunk], source: AudioSource) -> BTreeMap<u32, Vec<f32>> {
    let mut clusters = BTreeMap::<u32, Vec<&[f32]>>::new();
    for chunk in chunks.iter().filter(|chunk| chunk.source == source) {
        clusters
            .entry(chunk.cluster)
            .or_default()
            .push(&chunk.embedding);
    }
    clusters
        .into_iter()
        .filter_map(|(cluster, vectors)| Some((cluster, mean_normalized(&vectors)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(source: AudioSource, cluster: u32, embedding: [f32; 2]) -> EmbeddingChunk {
        EmbeddingChunk {
            source,
            start_ms: 0,
            end_ms: 2_000,
            cluster,
            embedding: embedding.to_vec(),
        }
    }

    fn segment(source: AudioSource, cluster: u32, start_ms: u64, end_ms: u64) -> SpeakerSegment {
        SpeakerSegment {
            source,
            start_ms,
            end_ms,
            cluster,
        }
    }

    #[test]
    fn each_microphone_voice_is_compared_with_its_closest_remote_voice() {
        let chunks = [
            chunk(AudioSource::System, 0, [1.0, 0.0]),
            chunk(AudioSource::System, 1, [0.6, 0.8]),
            chunk(AudioSource::Mic, 0, [1.0, 0.0]),
            chunk(AudioSource::Mic, 1, [0.0, 1.0]),
        ];
        let segments = [
            segment(AudioSource::Mic, 1, 5_000, 7_000),
            segment(AudioSource::System, 0, 900, 3_900),
            segment(AudioSource::Mic, 0, 1_000, 4_000),
        ];

        let voices = MicVoices::new(&segments, &chunks);

        assert_eq!(voices.turns, [(1_000, 4_000, 0), (5_000, 7_000, 1)]);
        assert_eq!(voices.remote_similarity.len(), 2);
        assert!((voices.remote_similarity[&0] - 1.0).abs() < 1e-6);
        assert!((voices.remote_similarity[&1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn without_remote_voiceprints_no_voice_is_compared() {
        let chunks = [chunk(AudioSource::Mic, 0, [0.0, 1.0])];
        let segments = [segment(AudioSource::Mic, 0, 0, 1_000)];

        let voices = MicVoices::new(&segments, &chunks);

        assert_eq!(voices.turns, [(0, 1_000, 0)]);
        assert!(voices.remote_similarity.is_empty());
    }
}
