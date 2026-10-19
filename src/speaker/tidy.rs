//! Keeps learned voices clean: chunks that sound like someone else are not
//! learned.

use crate::speaker::database::SpeakerDatabase;
use crate::speaker::embedding::topk_mean;

/// People with fewer samples are neither judged nor judged against.
pub const MIN_JUDGED_SAMPLES: usize = 10;
/// How much better another person must match for a sample to count as theirs.
pub const CLEARLY_BETTER: f32 = 0.05;

/// Whether `vector` may be learned as `name`: not when another known person
/// matches it clearly better and well enough to be recognized.
pub fn admits(database: &SpeakerDatabase, name: &str, vector: &[f32], threshold: f32) -> bool {
    let Some(own) = database
        .speakers
        .get(name)
        .filter(|record| record.samples.len() >= MIN_JUDGED_SAMPLES)
        .and_then(|record| topk_mean(vector, &record.samples))
    else {
        return true;
    };
    !database
        .speakers
        .iter()
        .filter(|(other, record)| {
            other.as_str() != name && record.samples.len() >= MIN_JUDGED_SAMPLES
        })
        .filter_map(|(_, record)| topk_mean(vector, &record.samples))
        .any(|score| score >= threshold && score - own >= CLEARLY_BETTER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker::database::{EmbeddingModelIdentity, SpeakerRecord};

    /// `count` unit vectors close to axis `axis` of a 4-dimensional space.
    fn voice(axis: usize, count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|n| {
                let mut vector = [0.0f32; 4];
                vector[axis] = 1.0;
                vector[(axis + 1 + n % 3) % 4] = 0.05 * (n as f32 + 1.0);
                let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                vector.iter().map(|v| v / norm).collect()
            })
            .collect()
    }

    fn database(people: &[(&str, Vec<Vec<f32>>)]) -> SpeakerDatabase {
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "model".into(),
            sha256: "hash".into(),
            dimension: 4,
        });
        for (name, vectors) in people {
            database
                .speakers
                .insert((*name).into(), SpeakerRecord::settled(vectors.clone()));
        }
        database
    }

    #[test]
    fn admission_skips_a_chunk_that_clearly_matches_another_known_person() {
        let chunk = voice(1, 11).pop().unwrap();
        let known = database(&[("Alice", voice(0, 10)), ("Bob", voice(1, 10))]);
        assert!(!admits(&known, "Alice", &chunk, 0.72));
        assert!(admits(&known, "Bob", &chunk, 0.72));
        let new_rival = database(&[("Alice", voice(0, 10)), ("Bob", voice(1, 9))]);
        assert!(admits(&new_rival, "Alice", &chunk, 0.72));
        let new_voice = database(&[("Alice", voice(0, 9)), ("Bob", voice(1, 10))]);
        assert!(admits(&new_voice, "Alice", &chunk, 0.72));
    }
}
