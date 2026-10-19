//! Judges every learned sample against its own voice and its closest rivals.
//!
//! The voice map draws these verdicts and tidying acts on them, so both read
//! them from here.
//!
//! A sample is compared with its voice's samples from other sessions, so a
//! batch learned under the wrong name in one meeting cannot vouch for itself.

use crate::speaker::database::Sample;
use crate::speaker::embedding::cosine;
use crate::speaker::tidy::{ISOLATION_MIN_GAP, MIN_OTHER_SESSION_SAMPLES, NEIGHBORHOOD};

/// A voice sample: its vector and the session it was learned in, if known.
pub trait Recorded: AsRef<[f32]> {
    fn session(&self) -> Option<&str>;
}

impl Recorded for Sample {
    fn session(&self) -> Option<&str> {
        self.session.as_deref()
    }
}

impl Recorded for Vec<f32> {
    fn session(&self) -> Option<&str> {
        None
    }
}

/// People every sample is scored against, as the recognizer's centroid
/// prefilter does for a cluster.
pub const RIVAL_CENTERS: usize = 5;
/// Extra people, nearest to the sample itself, that catch mislabeled samples.
const SAMPLE_RIVALS: usize = 2;
const OUTLIER_MIN_GAP: f32 = 0.15;
const OUTLIER_MIN_SAMPLES: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Flag {
    None,
    /// Unusually far from the center of its person's samples.
    Outlier,
    /// Unusually far from even its closest samples of the same person.
    Isolated,
    /// Scores at least as high against another person as against its own.
    Stray,
}

#[derive(Debug, Clone)]
pub struct SampleReview {
    pub center_similarity: f32,
    /// Recognizer score against the person's other samples.
    pub own_score: Option<f32>,
    /// The person's closest other samples, closest first.
    pub neighbors: Vec<(usize, f32)>,
    pub isolated: bool,
    /// The best scoring other person.
    pub rival: Option<(usize, f32)>,
    /// Recognizer scores against every person the sample was compared with.
    pub scores: Vec<(usize, f32)>,
    pub flag: Flag,
}

#[derive(Debug, Clone)]
pub struct PersonReview {
    /// People with the nearest centroids.
    pub rivals: Vec<usize>,
    /// Own score below which a sample counts as isolated.
    pub isolation_floor: Option<f32>,
    pub samples: Vec<SampleReview>,
}

#[derive(Debug, Clone)]
pub struct Review {
    pub center_distance: Vec<Vec<f32>>,
    pub people: Vec<PersonReview>,
}

pub fn review<V: Recorded>(samples: &[&[V]], centroids: &[Vec<f32>], threshold: f32) -> Review {
    let count = samples.len();
    let center_distance = (0..count)
        .map(|i| {
            (0..count)
                .map(|j| distance(&centroids[i], &centroids[j]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let people = (0..count)
        .map(|person| {
            let own = samples[person];
            let similarities = own
                .iter()
                .map(|vector| cosine(vector.as_ref(), &centroids[person]).unwrap_or(0.0))
                .collect::<Vec<_>>();
            let outlier_floor = mad_floor(&similarities, OUTLIER_MIN_GAP, OUTLIER_MIN_SAMPLES);
            let rivals = nearest(&center_distance[person], person, RIVAL_CENTERS);
            let neighborhoods = (0..own.len())
                .map(|sample| closest_samples(own, sample))
                .collect::<Vec<_>>();
            let own_scores = neighborhoods
                .iter()
                .map(|neighbors| {
                    let top = &neighbors[..neighbors.len().min(3)];
                    (!top.is_empty())
                        .then(|| top.iter().map(|n| n.1).sum::<f32>() / top.len() as f32)
                })
                .collect::<Vec<_>>();
            let isolation_floor = mad_floor(
                &own_scores.iter().flatten().copied().collect::<Vec<_>>(),
                ISOLATION_MIN_GAP,
                OUTLIER_MIN_SAMPLES,
            );
            let reviews = own
                .iter()
                .zip(neighborhoods)
                .enumerate()
                .map(|(sample, (vector, neighbors))| {
                    let vector = vector.as_ref();
                    let to_centroids = centroids
                        .iter()
                        .map(|centroid| distance(vector, centroid))
                        .collect::<Vec<_>>();
                    let own_score = own_scores[sample];
                    let isolated = own_score
                        .zip(isolation_floor)
                        .is_some_and(|(score, floor)| score < floor);
                    let mut candidates = rivals.clone();
                    for j in nearest(&to_centroids, person, SAMPLE_RIVALS) {
                        if !candidates.contains(&j) {
                            candidates.push(j);
                        }
                    }
                    let scores = candidates
                        .into_iter()
                        .filter_map(|other| {
                            topk_mean_excluding(vector, samples[other], None)
                                .map(|score| (other, score))
                        })
                        .collect::<Vec<_>>();
                    let rival =
                        scores
                            .iter()
                            .copied()
                            .fold(None, |best, (other, score)| match best {
                                Some((_, top)) if top >= score => best,
                                _ => Some((other, score)),
                            });
                    let stray = rival.is_some_and(|(_, score)| match own_score {
                        Some(own) => score >= own,
                        None => score >= threshold,
                    });
                    let flag = if stray {
                        Flag::Stray
                    } else if isolated {
                        Flag::Isolated
                    } else if outlier_floor.is_some_and(|floor| similarities[sample] < floor) {
                        Flag::Outlier
                    } else {
                        Flag::None
                    };
                    SampleReview {
                        center_similarity: similarities[sample],
                        own_score,
                        neighbors,
                        isolated,
                        rival,
                        scores,
                        flag,
                    }
                })
                .collect();
            PersonReview {
                rivals,
                isolation_floor,
                samples: reviews,
            }
        })
        .collect();
    Review {
        center_distance,
        people,
    }
}

pub(crate) fn distance(left: &[f32], right: &[f32]) -> f32 {
    (1.0 - cosine(left, right).unwrap_or(0.0)).max(0.0)
}

/// Averages the middle pair of an even count, as `speaker_score` does.
pub(crate) fn median(values: &[f32]) -> Option<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let middle = sorted.len() / 2;
    match sorted.len() {
        0 => None,
        n if n % 2 == 0 => Some((sorted[middle - 1] + sorted[middle]) / 2.0),
        _ => Some(sorted[middle]),
    }
}

/// Indexes of the `take` smallest distances in `row`, other than `skip`.
pub(crate) fn nearest(row: &[f32], skip: usize, take: usize) -> Vec<usize> {
    let mut order = (0..row.len()).filter(|&j| j != skip).collect::<Vec<_>>();
    order.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
    order.truncate(take);
    order
}

/// The `NEIGHBORHOOD` samples most similar to `own[sample]`, closest first,
/// from other sessions. A sample without a session is its own session. When
/// too few samples lie outside the session, as for a voice learned in one
/// meeting, all others are used.
fn closest_samples<V: Recorded>(own: &[V], sample: usize) -> Vec<(usize, f32)> {
    let query = own[sample].as_ref();
    let session = own[sample].session();
    let foreign = |other: &V| session.is_none() || other.session() != session;
    let others = || {
        own.iter()
            .enumerate()
            .filter(move |(index, _)| *index != sample)
    };
    let everyone = others().filter(|(_, other)| foreign(other)).count() < MIN_OTHER_SESSION_SAMPLES;
    let mut similar = others()
        .filter(|(_, other)| everyone || foreign(other))
        .filter_map(|(index, other)| cosine(query, other.as_ref()).map(|score| (index, score)))
        .collect::<Vec<_>>();
    similar.sort_by(|a, b| b.1.total_cmp(&a.1));
    similar.truncate(NEIGHBORHOOD);
    similar
}

pub(crate) fn topk_mean_excluding<V: AsRef<[f32]>>(
    query: &[f32],
    enrolled: &[V],
    skip: Option<usize>,
) -> Option<f32> {
    let mut best = [f32::NEG_INFINITY; 3];
    let mut found = 0;
    for (index, candidate) in enrolled.iter().enumerate() {
        if Some(index) == skip {
            continue;
        }
        let Some(score) = cosine(query, candidate.as_ref()) else {
            continue;
        };
        found += 1;
        if score > best[2] {
            best[2] = score;
            best.sort_by(|a, b| b.total_cmp(a));
        }
    }
    let used = found.min(3);
    (used > 0).then(|| best[..used].iter().sum::<f32>() / used as f32)
}

/// Value below which a sample stands out from its person's other samples:
/// the median less four robust deviations, but at least `min_gap`.
fn mad_floor(values: &[f32], min_gap: f32, min_samples: usize) -> Option<f32> {
    if values.len() < min_samples {
        return None;
    }
    let middle = median(values)?;
    let deviations = values
        .iter()
        .map(|s| (s - middle).abs())
        .collect::<Vec<_>>();
    let spread = 1.4826 * median(&deviations)?;
    Some(middle - (4.0 * spread).max(min_gap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors_come_from_other_sessions_while_enough_are_there() {
        let sample = |angle: f32, session: Option<&str>| {
            let mut sample = Sample::settled(vec![angle.cos(), angle.sin()]);
            sample.session = session.map(str::to_owned);
            sample
        };
        let indexes = |own: &[Sample], index: usize| {
            let mut found = closest_samples(own, index)
                .into_iter()
                .map(|n| n.0)
                .collect::<Vec<_>>();
            found.sort_unstable();
            found
        };
        let mut mixed = vec![sample(0.0, Some("a")), sample(0.01, Some("a"))];
        mixed.extend((1..4).map(|i| sample(i as f32 * 0.3, Some("b"))));
        mixed.push(sample(0.02, None));
        assert_eq!(indexes(&mixed, 0), [2, 3, 4, 5]);
        assert_eq!(indexes(&mixed, 5), [0, 1, 2, 3, 4]);
        let one_meeting = (0..5)
            .map(|i| sample(i as f32 * 0.1, Some("only")))
            .collect::<Vec<_>>();
        assert_eq!(indexes(&one_meeting, 0), [1, 2, 3, 4]);
    }

    #[test]
    fn median_matches_the_recognizer_for_even_counts() {
        assert_eq!(median(&[0.2, 0.9]), Some(0.55));
        assert_eq!(median(&[0.9, 0.2, 0.5]), Some(0.5));
        assert_eq!(median(&[]), None);
    }
}
