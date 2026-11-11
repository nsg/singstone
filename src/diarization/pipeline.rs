//! Diarization from raw segmentation: one voiceprint per chunk and local
//! speaker, clustered once, then assigned chunk by chunk so that two voices
//! the segmentation model separated inside one window stay apart. A window
//! with more voices than there are clusters reveals a voice the clustering
//! missed, such as someone who only gives short replies; those voices become
//! clusters of their own.

use super::cluster::{self, Stop};
use super::segmentation::{LOCAL_SPEAKERS, Segmentation};
use crate::speaker::embedding::{self, EmbeddingExtractor};
use crate::types::{AudioSource, SAMPLE_RATE, SpeakerSegment};

const MIN_ACTIVE_FRAMES: usize = 3;
const MIN_CLEAN_SECONDS: f64 = 0.3;
const SHORT_SPEECH_REPEATS: usize = 8;
const MIN_TRAINING_CLEAN_SECONDS: f64 = 2.0;
const MAX_TRAINING_ITEMS: usize = 4_000;
const MIN_CLUSTER_SIZE: usize = 3;
/// A cluster also needs this share of the clustered voiceprints, so a long
/// meeting does not split into fragments of a few seconds each.
const MIN_CLUSTER_SHARE: f64 = 0.02;
/// A group of unplaced voiceprints needs this share of them to become a voice.
const MIN_NEW_VOICE_SHARE: f64 = 0.05;
/// Rounds of adding clusters for voices that found no cluster in a window.
const NEW_VOICE_ROUNDS: usize = 3;
/// A single unplaced voiceprint becomes a cluster only with this much speech.
const MIN_NEW_VOICE_SECONDS: f64 = 0.6;
const MERGE_GAP_SECONDS: f64 = 0.5;
/// Longest input given to the embedding model at once: ten seconds.
const MAX_EMBEDDING_SAMPLES: usize = 160_000;

#[derive(Clone, Copy, Debug)]
pub struct Clustering {
    pub threshold: f32,
    pub num_speakers: Option<u32>,
}

/// One active local speaker of one chunk.
struct Item {
    chunk: usize,
    local: usize,
    clean_seconds: f64,
    active_seconds: f64,
    vector: Option<Vec<f32>>,
}

pub fn diarize(
    segmentation: &Segmentation,
    samples: &[f32],
    extractor: &EmbeddingExtractor,
    clustering: Clustering,
    source: AudioSource,
    threads: usize,
) -> Vec<SpeakerSegment> {
    let items = local_voiceprints(segmentation, samples, extractor, threads);
    let Some(mut centroids) = train_centroids(&items, clustering) else {
        return Vec::new();
    };
    if clustering.num_speakers.is_none() {
        add_unplaced_voices(&items, &mut centroids, clustering.threshold);
    }
    let assignment = assign(&items, &centroids, segmentation.num_chunks);
    let audio_end_ms = crate::types::samples_to_ms(samples.len() as u64);
    reconstruct(segmentation, &assignment)
        .into_iter()
        .map(|(start_ms, end_ms, cluster)| SpeakerSegment {
            source,
            start_ms: start_ms.min(audio_end_ms),
            end_ms: end_ms.min(audio_end_ms),
            cluster: cluster as u32,
        })
        .filter(|segment| segment.end_ms > segment.start_ms)
        .collect()
}

fn local_voiceprints(
    segmentation: &Segmentation,
    samples: &[f32],
    extractor: &EmbeddingExtractor,
    threads: usize,
) -> Vec<Item> {
    let frame_seconds = segmentation.frame_seconds();
    let mut requests = Vec::new();
    for chunk in 0..segmentation.num_chunks {
        let frames = segmentation.frames(chunk);
        for local in 0..LOCAL_SPEAKERS {
            let bit = 1u8 << local;
            let active = frames
                .iter()
                .map(|&mask| mask & bit != 0)
                .collect::<Vec<_>>();
            if active.iter().filter(|&&on| on).count() < MIN_ACTIVE_FRAMES {
                continue;
            }
            let clean = frames.iter().map(|&mask| mask == bit).collect::<Vec<_>>();
            let clean_seconds = clean.iter().filter(|&&on| on).count() as f64 * frame_seconds;
            let active_seconds = active.iter().filter(|&&on| on).count() as f64 * frame_seconds;
            let used = if clean_seconds >= MIN_CLEAN_SECONDS {
                &clean
            } else {
                &active
            };
            requests.push((
                chunk,
                local,
                clean_seconds,
                active_seconds,
                frame_ranges(segmentation, chunk, used),
                frame_ranges(segmentation, chunk, &active),
            ));
        }
    }
    parallel_map(
        &requests,
        threads,
        |(chunk, local, clean_seconds, active_seconds, used, active)| {
            let vector = embed_concatenated(extractor, samples, used).or_else(|| {
                let repeated = active.repeat(SHORT_SPEECH_REPEATS);
                embed_concatenated(extractor, samples, &repeated)
            });
            Item {
                chunk: *chunk,
                local: *local,
                clean_seconds: *clean_seconds,
                active_seconds: *active_seconds,
                vector,
            }
        },
    )
}

/// Maps `items` in order on up to `threads` scoped worker threads.
fn parallel_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    map: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let per_worker = items.len().div_ceil(threads.max(1)).max(1);
    let map = &map;
    std::thread::scope(|scope| {
        let workers = items
            .chunks(per_worker)
            .map(|part| scope.spawn(move || part.iter().map(map).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("diarization worker panicked"))
            .collect()
    })
}

/// Embeds the concatenation of sample ranges, cut off after ten seconds.
fn embed_concatenated(
    extractor: &EmbeddingExtractor,
    samples: &[f32],
    ranges: &[(usize, usize)],
) -> Option<Vec<f32>> {
    let mut audio = Vec::new();
    for &(start, end) in ranges {
        let room = MAX_EMBEDDING_SAMPLES - audio.len();
        let end = end.min(samples.len()).min(start.saturating_add(room));
        if end > start {
            audio.extend_from_slice(&samples[start..end]);
        }
        if audio.len() == MAX_EMBEDDING_SAMPLES {
            break;
        }
    }
    (!audio.is_empty())
        .then(|| extractor.embed(&audio))
        .flatten()
}

/// Sample ranges of the runs of `mask` in `chunk`.
fn frame_ranges(segmentation: &Segmentation, chunk: usize, mask: &[bool]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (frame, &on) in mask.iter().chain([&false]).enumerate() {
        match (on, start) {
            (true, None) => start = Some(frame),
            (false, Some(first)) => {
                ranges.push((
                    segmentation.frame_to_sample(chunk, first),
                    segmentation.frame_to_sample(chunk, frame),
                ));
                start = None;
            }
            _ => {}
        }
    }
    ranges
}

fn train_centroids(items: &[Item], clustering: Clustering) -> Option<Vec<Vec<f32>>> {
    let mut training = items
        .iter()
        .filter(|item| item.vector.is_some() && item.clean_seconds >= MIN_TRAINING_CLEAN_SECONDS)
        .collect::<Vec<_>>();
    if training.len() < 2 {
        training = items.iter().filter(|item| item.vector.is_some()).collect();
    }
    training = evenly_spaced(training, MAX_TRAINING_ITEMS);
    let vectors = training
        .iter()
        .filter_map(|item| item.vector.clone())
        .collect::<Vec<_>>();
    if vectors.is_empty() {
        return None;
    }
    let stop = clustering
        .num_speakers
        .map_or(Stop::Distance(clustering.threshold), |count| {
            Stop::Count(count as usize)
        });
    let labels = cluster::agglomerate(&vectors, stop);
    // A fixed count keeps its clusters however little some of them speak.
    let min_size = match clustering.num_speakers {
        Some(_) => MIN_CLUSTER_SIZE,
        None => MIN_CLUSTER_SIZE.max((MIN_CLUSTER_SHARE * vectors.len() as f64) as usize),
    };
    let labels = cluster::absorb_small_clusters(&vectors, &labels, min_size);
    let count = labels.iter().max().map_or(0, |max| max + 1);
    cluster::centroids(&vectors, &labels, count)
        .into_iter()
        .collect::<Option<Vec<_>>>()
}

/// At most `limit` of `items`, keeping their order and spreading the picks
/// evenly; items are in time order, so the picks spread evenly in time.
fn evenly_spaced<T>(items: Vec<T>, limit: usize) -> Vec<T> {
    let count = items.len();
    if count <= limit {
        return items;
    }
    let mut next = 0;
    items
        .into_iter()
        .enumerate()
        .filter(|(index, _)| {
            let keep = next < limit && *index == next * count / limit;
            next += usize::from(keep);
            keep
        })
        .map(|(_, item)| item)
        .collect()
}

/// Adds a cluster for each group of voiceprints that the per-window
/// matching cannot place because their window has more voices than there are
/// clusters, then matches again, a few rounds at most. A voiceprint within
/// `threshold` of an existing cluster is not new: it is more likely the
/// segmentation model splitting one voice in two.
fn add_unplaced_voices(items: &[Item], centroids: &mut Vec<Vec<f32>>, threshold: f32) {
    for _ in 0..NEW_VOICE_ROUNDS {
        let unplaced = match_chunks(items, centroids)
            .into_iter()
            .filter_map(|(index, cluster)| cluster.is_none().then_some(index))
            .filter(|&index| {
                items[index].vector.as_deref().is_some_and(|vector| {
                    centroids.iter().all(|centroid| {
                        embedding::cosine(vector, centroid)
                            .is_none_or(|similarity| 1.0 - similarity > threshold)
                    })
                })
            })
            .collect::<Vec<_>>();
        let unplaced = evenly_spaced(unplaced, MAX_TRAINING_ITEMS);
        if unplaced.is_empty() {
            return;
        }
        let vectors = unplaced
            .iter()
            .filter_map(|&index| items[index].vector.clone())
            .collect::<Vec<_>>();
        let labels = cluster::agglomerate(&vectors, Stop::Distance(threshold));
        let count = labels.iter().max().map_or(0, |max| max + 1);
        let added = cluster::centroids(&vectors, &labels, count)
            .into_iter()
            .enumerate()
            .filter_map(|(group, centroid)| {
                let members = unplaced
                    .iter()
                    .zip(&labels)
                    .filter(|&(_, label)| *label == group)
                    .map(|(&index, _)| &items[index])
                    .collect::<Vec<_>>();
                let longest = members
                    .iter()
                    .map(|item| item.active_seconds)
                    .fold(0.0, f64::max);
                let enough = members.len() as f64 >= MIN_NEW_VOICE_SHARE * unplaced.len() as f64;
                (enough && (members.len() >= 2 || longest >= MIN_NEW_VOICE_SECONDS))
                    .then_some(centroid)
                    .flatten()
            })
            .collect::<Vec<_>>();
        if added.is_empty() {
            return;
        }
        centroids.extend(added);
    }
}

/// Each item's cluster by per-window matching; `None` for an item its window
/// has no cluster left for.
fn match_chunks(items: &[Item], centroids: &[Vec<f32>]) -> Vec<(usize, Option<usize>)> {
    let mut matched = Vec::with_capacity(items.len());
    let mut start = 0;
    while start < items.len() {
        let chunk = items[start].chunk;
        let end = start
            + items[start..]
                .iter()
                .take_while(|item| item.chunk == chunk)
                .count();
        let scores = items[start..end]
            .iter()
            .map(|item| {
                centroids
                    .iter()
                    .map(|centroid| {
                        item.vector
                            .as_deref()
                            .and_then(|vector| embedding::cosine(vector, centroid))
                            .unwrap_or(0.0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        matched.extend((start..end).zip(assign_chunk(&scores)));
        start = end;
    }
    matched
}

/// Cluster of each `(chunk, local speaker)`, indexed `chunk * 3 + local`.
/// An item its window has no cluster left for takes its most similar one.
fn assign(items: &[Item], centroids: &[Vec<f32>], num_chunks: usize) -> Vec<Option<usize>> {
    let mut assignment = vec![None; num_chunks * LOCAL_SPEAKERS];
    for (index, cluster) in match_chunks(items, centroids) {
        let item = &items[index];
        let cluster = cluster.unwrap_or_else(|| {
            let vector = item.vector.as_deref();
            (0..centroids.len()).fold(0, |best, candidate| {
                let score = |cluster: usize| {
                    vector
                        .and_then(|vector| embedding::cosine(vector, &centroids[cluster]))
                        .unwrap_or(0.0)
                };
                if score(candidate) > score(best) {
                    candidate
                } else {
                    best
                }
            })
        });
        assignment[item.chunk * LOCAL_SPEAKERS + item.local] = Some(cluster);
    }
    assignment
}

/// Distinct clusters maximizing the summed score. With more rows than
/// clusters, only as many rows as there are clusters get one, and the rest
/// get `None`. With `n` rows, some optimum gives every placed row one of its
/// `n` best clusters, so only those are searched.
fn assign_chunk(scores: &[Vec<f32>]) -> Vec<Option<usize>> {
    let clusters = scores.first().map_or(0, Vec::len);
    let placed = scores.len().min(clusters);
    let candidates = scores
        .iter()
        .map(|row| {
            let mut ranked = (0..row.len()).collect::<Vec<_>>();
            ranked.sort_by(|left, right| row[*right].total_cmp(&row[*left]).then(left.cmp(right)));
            ranked.truncate(scores.len());
            ranked.sort_unstable();
            ranked
        })
        .collect::<Vec<_>>();
    struct Search<'a> {
        scores: &'a [Vec<f32>],
        candidates: &'a [Vec<usize>],
        placed: usize,
        chosen: Vec<Option<usize>>,
        best_total: f32,
        best_choice: Vec<Option<usize>>,
    }
    fn search(state: &mut Search, total: f32, used: usize) {
        let row = state.chosen.len();
        let rows = state.scores.len();
        if row == rows {
            if used == state.placed && total > state.best_total {
                state.best_total = total;
                state.best_choice = state.chosen.clone();
            }
            return;
        }
        for index in 0..state.candidates[row].len() {
            let cluster = state.candidates[row][index];
            if state.chosen.contains(&Some(cluster)) {
                continue;
            }
            state.chosen.push(Some(cluster));
            search(state, total + state.scores[row][cluster], used + 1);
            state.chosen.pop();
        }
        if state.placed - used < rows - row {
            state.chosen.push(None);
            search(state, total, used);
            state.chosen.pop();
        }
    }
    let mut state = Search {
        scores,
        candidates: &candidates,
        placed,
        chosen: Vec::with_capacity(scores.len()),
        best_total: f32::NEG_INFINITY,
        best_choice: vec![None; scores.len()],
    };
    search(&mut state, 0.0, 0);
    state.best_choice
}

/// Overlap-adds the assigned local activity on a global frame grid, keeps the
/// estimated number of speakers per frame, and returns `(start_ms, end_ms,
/// cluster)` runs. Votes are kept sparse, so memory does not grow with the
/// number of clusters.
fn reconstruct(
    segmentation: &Segmentation,
    assignment: &[Option<usize>],
) -> Vec<(u64, u64, usize)> {
    let shift = segmentation.rf_shift as f64;
    let total = (((segmentation.num_chunks - 1) * segmentation.step + segmentation.window) as f64
        / shift)
        .round_ties_even() as usize
        + 1;
    let mut speakers = vec![0u32; total];
    let mut weight = vec![0u32; total];
    // `(global frame << 32) | cluster`, one entry per vote.
    let mut votes = Vec::<u64>::new();
    let frame_step = segmentation.window as f64 / segmentation.num_frames as f64;
    for chunk in 0..segmentation.num_chunks {
        let offset = (chunk * segmentation.step) as f64;
        for (frame, &mask) in segmentation.frames(chunk).iter().enumerate() {
            let global = (((offset + frame as f64 * frame_step) / shift).round_ties_even()
                as usize)
                .min(total - 1);
            speakers[global] += mask.count_ones();
            weight[global] += 1;
            for local in (0..LOCAL_SPEAKERS).filter(|local| mask & (1 << local) != 0) {
                if let Some(cluster) = assignment[chunk * LOCAL_SPEAKERS + local] {
                    votes.push(((global as u64) << 32) | cluster as u64);
                }
            }
        }
    }
    votes.sort_unstable();
    // `(cluster << 32) | global frame` for every kept cluster and frame.
    let mut kept = Vec::<u64>::new();
    let mut tally = Vec::<(u32, u64)>::new();
    for frame_votes in votes.chunk_by(|left, right| left >> 32 == right >> 32) {
        let global = (frame_votes[0] >> 32) as usize;
        tally.clear();
        tally.extend(
            frame_votes
                .chunk_by(|left, right| left == right)
                .map(|same| (same.len() as u32, same[0] & u64::from(u32::MAX))),
        );
        tally.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
        let count = (speakers[global] as f64 / weight[global] as f64 + 0.5).floor() as usize;
        kept.extend(
            tally
                .iter()
                .take(count)
                .map(|(_, cluster)| (cluster << 32) | global as u64),
        );
    }
    kept.sort_unstable();
    let to_ms = |global: u64| {
        ((global as f64 * shift + 0.5 * segmentation.rf_size as f64) * 1_000.0 / SAMPLE_RATE as f64)
            .round() as u64
    };
    let merge_gap = MERGE_GAP_SECONDS * SAMPLE_RATE as f64 / shift;
    let mut segments = Vec::new();
    for cluster_frames in kept.chunk_by(|left, right| left >> 32 == right >> 32) {
        let cluster = (cluster_frames[0] >> 32) as usize;
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for global in cluster_frames
            .iter()
            .map(|entry| entry & u64::from(u32::MAX))
        {
            match runs.last_mut() {
                Some((_, end)) if ((global - *end) as f64) < merge_gap => *end = global + 1,
                _ => runs.push((global, global + 1)),
            }
        }
        segments.extend(
            runs.into_iter()
                .map(|(start, end)| (to_ms(start), to_ms(end), cluster)),
        );
    }
    segments.sort();
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_assignment_keeps_local_speakers_apart() {
        let scores = [vec![0.9, 0.8, 0.1], vec![0.85, 0.2, 0.1]];
        assert_eq!(assign_chunk(&scores), [Some(1), Some(0)]);
        let crowded = [vec![0.9, 0.1], vec![0.8, 0.2], vec![0.3, 0.4]];
        assert_eq!(assign_chunk(&crowded), [Some(0), None, Some(1)]);
        let lonely = [vec![0.5], vec![0.9]];
        assert_eq!(assign_chunk(&lonely), [None, Some(0)]);
        let mut many = vec![vec![0.0; 400]; 3];
        many[0][7] = 0.9;
        many[1][7] = 0.8;
        many[1][300] = 0.75;
        many[2][7] = 0.7;
        many[2][300] = 0.6;
        many[2][42] = 0.5;
        assert_eq!(assign_chunk(&many), [Some(7), Some(300), Some(42)]);
    }

    #[test]
    fn a_voice_without_a_cluster_gets_one_unless_it_matches_one() {
        let item = |chunk, local, vector: [f32; 3]| Item {
            chunk,
            local,
            clean_seconds: 1.0,
            active_seconds: 1.0,
            vector: Some(vector.to_vec()),
        };
        let host = [1.0, 0.0, 0.0];
        let guest = [0.0, 1.0, 0.0];
        let items = [
            item(0, 0, host),
            item(0, 1, guest),
            item(1, 0, host),
            item(1, 2, [0.1, 1.0, 0.0]),
        ];
        let mut centroids = vec![host.to_vec()];
        add_unplaced_voices(&items, &mut centroids, 0.5);
        assert_eq!(centroids.len(), 2);
        assert_eq!(
            assign(&items, &centroids, 2),
            [Some(0), Some(1), None, Some(0), None, Some(1)]
        );
        let split = [item(0, 0, host), item(0, 1, [1.0, 0.3, 0.0])];
        let mut centroids = vec![host.to_vec()];
        add_unplaced_voices(&split, &mut centroids, 0.5);
        assert_eq!(centroids.len(), 1);
    }

    #[test]
    fn a_fixed_count_keeps_a_speaker_who_says_little() {
        let item = |chunk, vector: [f32; 3]| Item {
            chunk,
            local: 0,
            clean_seconds: 3.0,
            active_seconds: 3.0,
            vector: Some(vector.to_vec()),
        };
        let items = (0..1_000)
            .map(|chunk| item(chunk, [1.0, 0.01 * (chunk % 7) as f32, 0.0]))
            .chain((1_000..1_010).map(|chunk| item(chunk, [0.0, 0.0, 1.0])))
            .collect::<Vec<_>>();
        let fixed = Clustering {
            threshold: 0.5,
            num_speakers: Some(2),
        };
        assert_eq!(train_centroids(&items, fixed).map(|c| c.len()), Some(2));
        let free = Clustering {
            num_speakers: None,
            ..fixed
        };
        assert_eq!(train_centroids(&items, free).map(|c| c.len()), Some(1));
    }

    #[test]
    fn training_subsample_is_bounded_and_spread() {
        assert_eq!(evenly_spaced((0..10).collect(), 4), [0, 2, 5, 7]);
        assert_eq!(evenly_spaced((0..3).collect(), 4), [0, 1, 2]);
        assert_eq!(evenly_spaced((0..4_001).collect(), 4_000).len(), 4_000);
    }

    #[test]
    fn reconstruction_keeps_the_estimated_speaker_count() {
        // Three chunks, one 100 ms frame apart: frame f of chunk c lands on
        // global frame c + f. Cluster 1 is outvoted where the chunks agree on
        // one speaker, and wins where it is the only voice left.
        let segmentation = Segmentation {
            num_chunks: 3,
            num_frames: 4,
            window: 6_400,
            step: 1_600,
            rf_shift: 1_600,
            rf_size: 1_600,
            activity: [[0b01; 4], [0b01, 0b01, 0b01, 0], [0b10; 4]].concat(),
        };
        let mut assignment = vec![None; 9];
        assignment[0] = Some(0);
        assignment[3] = Some(0);
        assignment[7] = Some(1);
        assert_eq!(
            reconstruct(&segmentation, &assignment),
            [(50, 450, 0), (450, 650, 1)]
        );
    }
}
