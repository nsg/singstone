//! Lays learned voices out on a plane for inspection.
//!
//! Distances are cosine distances (`1 - cosine`), the quantity the recognizer
//! thresholds. Person centers are placed by multidimensional scaling of their
//! centroid distances, so far-apart centers mean different voices. Each sample
//! starts in its person's own principal plane and is then pulled by every
//! nearby center toward the distances it really has, so a sample that sounds
//! like someone else drifts toward them. A person-wide offset is removed from
//! sample distances first: in high dimensions every sample lies about equally
//! far from its centroid, which would otherwise draw each cloud as a ring.

use crate::speaker::classify::{self, Flag, distance, median, nearest};
use crate::speaker::database::SpeakerDatabase;
use crate::speaker::embedding::mean_normalized;

/// Above this many flagged samples, regrouping costs more than it shows.
const MAX_REGROUP: usize = 1_000;
const DOT_ANCHORS: usize = 6;
/// Nearest samples a flagged sample is also placed against.
const NEIGHBORS: usize = 5;
const MIN_DISTANCE: f32 = 0.01;
pub const CLOSE_MARGIN: f32 = 0.1;

#[derive(Debug, Clone)]
pub struct VoiceMap {
    pub people: Vec<Person>,
    pub dots: Vec<Dot>,
    pub close_pairs: Vec<ClosePair>,
    pub threshold: f32,
    /// Normalized stress of the center layout: 0 is a perfect fit.
    pub stress: f32,
}

#[derive(Debug, Clone)]
pub struct Person {
    pub name: String,
    pub center: [f32; 2],
    pub radius: f32,
    pub samples: usize,
    pub median_similarity: f32,
    pub outliers: usize,
    pub strays: usize,
}

#[derive(Debug, Clone)]
pub struct Dot {
    pub person: usize,
    pub sample: usize,
    pub position: [f32; 2],
    pub center_similarity: f32,
    pub own_score: Option<f32>,
    pub rival: Option<(usize, f32)>,
    pub flag: Flag,
}

#[derive(Debug, Clone)]
pub struct ClosePair {
    pub a: usize,
    pub b: usize,
    /// Median recognizer score of one person's samples against the other, the
    /// higher of both directions.
    pub score: f32,
}

impl ClosePair {
    pub fn overlapping(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

impl VoiceMap {
    pub fn build(database: &SpeakerDatabase, threshold: f32) -> Self {
        let speakers = database
            .speakers
            .iter()
            .filter_map(|(name, record)| {
                let centroid = mean_normalized(&record.embeddings)?;
                Some((name.clone(), record.embeddings.as_slice(), centroid))
            })
            .collect::<Vec<_>>();
        let names = speakers.iter().map(|s| s.0.clone()).collect::<Vec<_>>();
        let samples = speakers.iter().map(|s| s.1).collect::<Vec<_>>();
        let centroids = speakers.into_iter().map(|s| s.2).collect::<Vec<_>>();
        Self::from_parts(names, &samples, &centroids, threshold)
    }

    fn from_parts(
        names: Vec<String>,
        samples: &[&[Vec<f32>]],
        centroids: &[Vec<f32>],
        threshold: f32,
    ) -> Self {
        let count = names.len();
        let review = classify::review(samples, centroids, threshold);
        let center_distance = review.center_distance;
        let (centers, stress) = layout_centers(&center_distance);

        let mut dots = Vec::new();
        let mut flagged = Vec::new();
        let mut pair_scores = vec![vec![None::<f32>; count]; count];
        let mut people = Vec::with_capacity(count);
        for (person, verdicts) in review.people.into_iter().enumerate() {
            let own = samples[person];
            let similarities = verdicts
                .samples
                .iter()
                .map(|sample| sample.center_similarity)
                .collect::<Vec<_>>();
            let shell = shell_offset(&similarities);
            let directions = principal_directions(own, &centroids[person]);
            let first_dot = dots.len();
            let mut cross = vec![Vec::new(); count];
            for (sample, (vector, verdict)) in own.iter().zip(verdicts.samples).enumerate() {
                for &(other, score) in &verdict.scores {
                    if verdicts.rivals.contains(&other) {
                        cross[other].push(score);
                    }
                }
                let to_centroids = centroids
                    .iter()
                    .map(|centroid| distance(vector, centroid))
                    .collect::<Vec<_>>();
                let anchors =
                    center_anchors(&to_centroids, person, &centers, &center_distance, shell);
                let own = (to_centroids[person] - shell).max(MIN_DISTANCE);
                let mut position = [
                    centers[person][0] + directions[sample][0] * own,
                    centers[person][1] + directions[sample][1] * own,
                ];
                for _ in 0..30 {
                    position = majorize(position, &anchors);
                }
                if verdict.flag != Flag::None {
                    flagged.push((dots.len(), anchors, shell));
                }
                dots.push(Dot {
                    person,
                    sample,
                    position,
                    center_similarity: verdict.center_similarity,
                    own_score: verdict.own_score,
                    rival: verdict.rival,
                    flag: verdict.flag,
                });
            }
            for other in verdicts.rivals {
                pair_scores[person][other] = median(&cross[other]);
            }
            let mine = &dots[first_dot..];
            people.push(Person {
                name: names[person].clone(),
                center: centers[person],
                radius: 0.0,
                samples: own.len(),
                median_similarity: median(&similarities).unwrap_or(1.0),
                outliers: mine.iter().filter(|d| d.flag == Flag::Outlier).count(),
                strays: mine.iter().filter(|d| d.flag == Flag::Stray).count(),
            });
        }
        regroup(&mut dots, samples, &flagged);
        for (index, person) in people.iter_mut().enumerate() {
            let mut radii = dots
                .iter()
                .filter(|dot| dot.person == index)
                .map(|dot| planar_distance(dot.position, person.center))
                .collect::<Vec<_>>();
            radii.sort_by(f32::total_cmp);
            person.radius = quantile(&radii, 0.8).unwrap_or(0.0).max(0.02);
        }

        let mut close_pairs = Vec::new();
        for (a, row) in pair_scores.iter().enumerate() {
            for (b, &forward) in row.iter().enumerate().skip(a + 1) {
                let score = [forward, pair_scores[b][a]]
                    .into_iter()
                    .flatten()
                    .reduce(f32::max);
                if let Some(score) = score.filter(|score| *score >= threshold - CLOSE_MARGIN) {
                    close_pairs.push(ClosePair { a, b, score });
                }
            }
        }
        close_pairs.sort_by(|l, r| r.score.total_cmp(&l.score));

        Self {
            people,
            dots,
            close_pairs,
            threshold,
            stress,
        }
    }
}

fn planar_distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn quantile(sorted: &[f32], q: f32) -> Option<f32> {
    let last = sorted.len().checked_sub(1)?;
    Some(sorted[((last as f32) * q).round() as usize])
}

/// Distance every sample of a person shares, removed before placement.
fn shell_offset(similarities: &[f32]) -> f32 {
    if similarities.len() < 3 {
        return 0.0;
    }
    let mut distances = similarities.iter().map(|s| 1.0 - s).collect::<Vec<_>>();
    distances.sort_by(f32::total_cmp);
    0.9 * quantile(&distances, 0.1).unwrap_or(0.0).max(0.0)
}

/// Unit direction of each sample in its person's two main axes of variation.
fn principal_directions(samples: &[Vec<f32>], centroid: &[f32]) -> Vec<[f32; 2]> {
    let residuals = samples
        .iter()
        .map(|v| {
            v.iter()
                .zip(centroid)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let first = principal_axis(&residuals, &[]);
    let second = principal_axis(&residuals, first.as_slice());
    residuals
        .iter()
        .enumerate()
        .map(|(index, residual)| {
            let x = first.as_deref().map_or(0.0, |axis| dot(residual, axis));
            let y = second.as_deref().map_or(0.0, |axis| dot(residual, axis));
            let length = x.hypot(y);
            if length > 1e-6 {
                [x / length, y / length]
            } else {
                let angle = index as f32 * 2.399_963;
                [angle.cos(), angle.sin()]
            }
        })
        .collect()
}

fn principal_axis(residuals: &[Vec<f32>], orthogonal_to: &[Vec<f32>]) -> Option<Vec<f32>> {
    let start = residuals
        .iter()
        .max_by(|a, b| dot(a, a).total_cmp(&dot(b, b)))?;
    let mut axis = start.clone();
    for _ in 0..40 {
        for other in orthogonal_to {
            let along = dot(&axis, other);
            axis.iter_mut()
                .zip(other)
                .for_each(|(a, o)| *a -= along * o);
        }
        if !normalize(&mut axis) {
            return None;
        }
        let mut next = vec![0.0; axis.len()];
        for residual in residuals {
            let along = dot(residual, &axis);
            next.iter_mut()
                .zip(residual)
                .for_each(|(n, r)| *n += along * r);
        }
        axis = next;
    }
    for other in orthogonal_to {
        let along = dot(&axis, other);
        axis.iter_mut()
            .zip(other)
            .for_each(|(a, o)| *a -= along * o);
    }
    normalize(&mut axis).then_some(axis)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn normalize(vector: &mut [f32]) -> bool {
    let length = dot(vector, vector).sqrt();
    if !length.is_finite() || length < 1e-9 {
        return false;
    }
    vector.iter_mut().for_each(|v| *v /= length);
    true
}

type Anchor = ([f32; 2], f32, f32);

/// Fixed centers a sample is placed against, as (position, target, weight).
fn center_anchors(
    to_centroids: &[f32],
    person: usize,
    centers: &[[f32; 2]],
    center_distance: &[Vec<f32>],
    shell: f32,
) -> Vec<Anchor> {
    let mut anchors = nearest(&center_distance[person], person, DOT_ANCHORS);
    for j in nearest(to_centroids, person, 2) {
        if !anchors.contains(&j) {
            anchors.push(j);
        }
    }
    anchors.push(person);
    anchors
        .into_iter()
        .map(|j| {
            let t = (to_centroids[j] - shell).max(MIN_DISTANCE);
            (centers[j], t, 1.0 / (t * t))
        })
        .collect()
}

/// Lets flagged samples also pull toward the samples they sound most like,
/// so a batch of similar bad samples shows as one island rather than being
/// spread out by their centroid distances alone.
fn regroup(dots: &mut [Dot], samples: &[&[Vec<f32>]], flagged: &[(usize, Vec<Anchor>, f32)]) {
    if flagged.len() > MAX_REGROUP {
        return;
    }
    let vector = |dot: &Dot| samples[dot.person][dot.sample].as_slice();
    let neighbors = flagged
        .iter()
        .map(|(index, _, shell)| {
            let own = vector(&dots[*index]);
            let mut distances = dots
                .iter()
                .enumerate()
                .filter(|(j, _)| j != index)
                .map(|(j, dot)| (j, distance(own, vector(dot))))
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.1.total_cmp(&b.1));
            distances.truncate(NEIGHBORS);
            distances
                .into_iter()
                .map(|(j, d)| {
                    let t = (d - shell).max(MIN_DISTANCE);
                    (j, t, 1.0 / (t * t))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for _ in 0..20 {
        for ((index, centers, _), neighbors) in flagged.iter().zip(&neighbors) {
            // Neighbors share the pull equally with the centers, so a batch
            // keeps its distance from everyone instead of collapsing inward.
            let center_weight = centers.iter().map(|a| a.2).sum::<f32>();
            let neighbor_weight = neighbors.iter().map(|n| n.2).sum::<f32>().max(1e-6);
            let scale = center_weight / neighbor_weight;
            let mut anchors = centers.clone();
            anchors.extend(
                neighbors
                    .iter()
                    .map(|&(j, t, w)| (dots[j].position, t, w * scale)),
            );
            dots[*index].position = majorize(dots[*index].position, &anchors);
        }
    }
}

fn majorize(position: [f32; 2], anchors: &[([f32; 2], f32, f32)]) -> [f32; 2] {
    let mut sum = [0.0f32; 2];
    let mut weights = 0.0f32;
    for &(anchor, target, weight) in anchors {
        let dx = position[0] - anchor[0];
        let dy = position[1] - anchor[1];
        let length = dx.hypot(dy).max(1e-6);
        sum[0] += weight * (anchor[0] + target * dx / length);
        sum[1] += weight * (anchor[1] + target * dy / length);
        weights += weight;
    }
    if weights > 0.0 {
        [sum[0] / weights, sum[1] / weights]
    } else {
        position
    }
}

/// Classical scaling followed by weighted stress majorization.
fn layout_centers(distances: &[Vec<f32>]) -> (Vec<[f32; 2]>, f32) {
    let count = distances.len();
    if count == 0 {
        return (Vec::new(), 0.0);
    }
    if count == 1 {
        return (vec![[0.0, 0.0]], 0.0);
    }
    let mut points = classical_scaling(distances);
    let weight = |d: f32| 1.0 / d.max(MIN_DISTANCE).powi(2);
    for _ in 0..300 {
        for i in 0..count {
            let anchors = (0..count)
                .filter(|&j| j != i)
                .map(|j| {
                    (
                        points[j],
                        distances[i][j].max(MIN_DISTANCE),
                        weight(distances[i][j]),
                    )
                })
                .collect::<Vec<_>>();
            points[i] = majorize(points[i], &anchors);
        }
    }
    let mut misfit = 0.0f32;
    let mut total = 0.0f32;
    for i in 0..count {
        for j in i + 1..count {
            let w = weight(distances[i][j]);
            misfit += w * (planar_distance(points[i], points[j]) - distances[i][j]).powi(2);
            total += w * distances[i][j].powi(2);
        }
    }
    let stress = if total > 0.0 {
        (misfit / total).sqrt()
    } else {
        0.0
    };
    (points, stress)
}

fn classical_scaling(distances: &[Vec<f32>]) -> Vec<[f32; 2]> {
    let count = distances.len();
    let squared = distances
        .iter()
        .map(|row| {
            row.iter()
                .map(|d| f64::from(*d).powi(2))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let row_mean = squared
        .iter()
        .map(|row| row.iter().sum::<f64>() / count as f64)
        .collect::<Vec<_>>();
    let total_mean = row_mean.iter().sum::<f64>() / count as f64;
    let mut gram = vec![vec![0.0f64; count]; count];
    for i in 0..count {
        for j in 0..count {
            gram[i][j] = -0.5 * (squared[i][j] - row_mean[i] - row_mean[j] + total_mean);
        }
    }
    let mut axes: Vec<(Vec<f64>, f64)> = Vec::new();
    for axis in 0..2 {
        let mut vector = (0..count)
            .map(|i| ((i * 7 + axis * 3 + 1) % 11) as f64 - 5.0)
            .collect::<Vec<_>>();
        let mut value = 0.0;
        for _ in 0..200 {
            let mut next = (0..count)
                .map(|i| (0..count).map(|j| gram[i][j] * vector[j]).sum::<f64>())
                .collect::<Vec<_>>();
            for (previous, _) in &axes {
                let along = next.iter().zip(previous).map(|(a, b)| a * b).sum::<f64>();
                next.iter_mut()
                    .zip(previous)
                    .for_each(|(n, p)| *n -= along * p);
            }
            let length = next.iter().map(|v| v * v).sum::<f64>().sqrt();
            if length < 1e-12 {
                break;
            }
            value = length;
            vector = next.into_iter().map(|v| v / length).collect();
        }
        axes.push((vector, value));
    }
    (0..count)
        .map(|i| {
            let coordinate = |axis: usize| (axes[axis].0[i] * axes[axis].1.sqrt()) as f32;
            let x = coordinate(0);
            let y = coordinate(1);
            // Break exact ties so majorization has a direction to move in.
            let jitter = (i as f32 * 2.399_963).sin() * 1e-3;
            [x + jitter, y - jitter]
        })
        .collect()
}

/// Gives each person a palette slot, keeping close neighbors visually apart.
/// `palette` holds OKLab coordinates.
pub fn assign_colors(centers: &[[f32; 2]], palette: &[[f32; 3]]) -> Vec<usize> {
    let count = centers.len();
    if palette.is_empty() {
        return vec![0; count];
    }
    let mut nearest = centers
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            centers
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, b)| planar_distance(*a, *b))
                .reduce(f32::min)
        })
        .collect::<Vec<_>>();
    nearest.sort_by(f32::total_cmp);
    let scale = (median(&nearest).unwrap_or(1.0) * 2.5).max(1e-3);
    let proximity =
        |i: usize, j: usize| (-(planar_distance(centers[i], centers[j]) / scale).powi(2)).exp();
    let likeness = |a: usize, b: usize| {
        // Lightness alone separates colors poorly on a dark canvas.
        let d = ((0.5 * (palette[a][0] - palette[b][0])).powi(2)
            + (palette[a][1] - palette[b][1]).powi(2)
            + (palette[a][2] - palette[b][2]).powi(2))
        .sqrt();
        (-(d / 0.15).powi(2)).exp()
    };
    let mut order = (0..count).collect::<Vec<_>>();
    let crowding = (0..count)
        .map(|i| {
            (0..count)
                .filter(|&j| j != i)
                .map(|j| proximity(i, j))
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    order.sort_by(|&a, &b| crowding[b].total_cmp(&crowding[a]));
    let mut colors: Vec<Option<usize>> = vec![None; count];
    for _ in 0..4 {
        for &i in &order {
            let mut used = vec![0usize; palette.len()];
            colors.iter().flatten().for_each(|&c| used[c] += 1);
            let best = (0..palette.len()).min_by(|&a, &b| {
                let cost = |c: usize| {
                    (0..count)
                        .filter(|&j| j != i)
                        .filter_map(|j| colors[j].map(|cj| proximity(i, j) * likeness(c, cj)))
                        .sum::<f32>()
                };
                cost(a)
                    .total_cmp(&cost(b))
                    .then_with(|| used[a].cmp(&used[b]))
            });
            colors[i] = best;
        }
    }
    colors.into_iter().map(|c| c.unwrap_or(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(mut vector: Vec<f32>) -> Vec<f32> {
        assert!(normalize(&mut vector));
        vector
    }

    fn axis(dimension: usize, index: usize) -> Vec<f32> {
        let mut vector = vec![0.0; dimension];
        vector[index] = 1.0;
        vector
    }

    fn cloud(center: &[f32], count: usize, spread: f32, seed: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|n| {
                unit(
                    center
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let wobble = (((n + seed) * 31 + i * 17) as f32 * 0.618).sin();
                            c + spread * wobble
                        })
                        .collect(),
                )
            })
            .collect()
    }

    fn build(clouds: Vec<Vec<Vec<f32>>>) -> VoiceMap {
        let names = (0..clouds.len()).map(|i| format!("P{i}")).collect();
        let centroids = clouds
            .iter()
            .map(|c| mean_normalized(c).unwrap())
            .collect::<Vec<_>>();
        let refs = clouds.iter().map(Vec::as_slice).collect::<Vec<_>>();
        VoiceMap::from_parts(names, &refs, &centroids, 0.72)
    }

    #[test]
    fn similar_voices_sit_closer_than_different_ones() {
        let a = axis(16, 0);
        let b = unit(
            a.iter()
                .zip(axis(16, 1))
                .map(|(x, y)| x + 0.6 * y)
                .collect(),
        );
        let c = axis(16, 5);
        let map = build(vec![
            cloud(&a, 12, 0.05, 0),
            cloud(&b, 12, 0.05, 3),
            cloud(&c, 12, 0.05, 7),
        ]);
        let near = planar_distance(map.people[0].center, map.people[1].center);
        let far = planar_distance(map.people[0].center, map.people[2].center);
        assert!(near * 2.0 < far, "near {near} far {far}");
        assert!(map.stress < 0.1, "stress {}", map.stress);
    }

    #[test]
    fn flags_a_sample_that_sounds_like_someone_else() {
        let a = axis(16, 0);
        let b = axis(16, 4);
        let mut first = cloud(&a, 12, 0.05, 0);
        let second = cloud(&b, 12, 0.05, 5);
        first.push(second[0].clone());
        let map = build(vec![first, second]);
        let stray = map
            .dots
            .iter()
            .find(|d| d.person == 0 && d.sample == 12)
            .unwrap();
        assert_eq!(stray.flag, Flag::Stray);
        assert_eq!(stray.rival.map(|r| r.0), Some(1));
        let to_rival = planar_distance(stray.position, map.people[1].center);
        let to_own = planar_distance(stray.position, map.people[0].center);
        assert!(to_rival < to_own, "rival {to_rival} own {to_own}");
        assert_eq!(map.people[0].strays, 1);
    }

    #[test]
    fn flags_a_sample_far_from_its_own_cloud() {
        let a = axis(16, 0);
        let mut first = cloud(&a, 20, 0.03, 0);
        first.push(unit(
            a.iter()
                .zip(axis(16, 9))
                .map(|(x, y)| x + 1.2 * y)
                .collect(),
        ));
        let map = build(vec![first, cloud(&axis(16, 4), 10, 0.03, 2)]);
        let flagged = map
            .dots
            .iter()
            .filter(|d| d.flag == Flag::Outlier)
            .map(|d| (d.person, d.sample))
            .collect::<Vec<_>>();
        assert_eq!(flagged, vec![(0, 20)]);
    }

    #[test]
    fn regroups_similar_flagged_samples_into_one_island() {
        let a = axis(16, 0);
        let room = unit(
            a.iter()
                .zip(axis(16, 9))
                .map(|(x, y)| x + 1.0 * y)
                .collect(),
        );
        let mut vectors = cloud(&a, 10, 0.03, 0);
        vectors.extend(cloud(&room, 6, 0.005, 11));
        let target = distance(&vectors[12], &a);
        let mut dots = (0..vectors.len())
            .map(|sample| {
                let angle = sample as f32 * 1.05;
                Dot {
                    person: 0,
                    sample,
                    position: [target * angle.cos(), target * angle.sin()],
                    center_similarity: 0.0,
                    own_score: None,
                    rival: None,
                    flag: Flag::None,
                }
            })
            .collect::<Vec<_>>();
        let ring = |dots: &[Dot], sample: usize| {
            let p = dots[sample].position;
            p[0].hypot(p[1])
        };
        let spread = |dots: &[Dot]| {
            let island = &dots[10..];
            let middle = [
                island.iter().map(|d| d.position[0]).sum::<f32>() / 6.0,
                island.iter().map(|d| d.position[1]).sum::<f32>() / 6.0,
            ];
            island
                .iter()
                .map(|d| planar_distance(d.position, middle))
                .fold(0.0, f32::max)
        };
        assert!(spread(&dots) > target * 0.5);
        let flagged = (10..16)
            .map(|index| {
                let t = ring(&dots, index);
                (index, vec![([0.0, 0.0], t, 1.0 / (t * t))], 0.0)
            })
            .collect::<Vec<_>>();
        regroup(&mut dots, &[vectors.as_slice()], &flagged);
        assert!(spread(&dots) < target * 0.1, "spread {}", spread(&dots));
        for index in 10..16 {
            assert!((ring(&dots, index) - target).abs() < target * 0.2);
        }
    }

    #[test]
    fn reports_overlapping_people() {
        let a = axis(16, 0);
        let twin = unit(
            a.iter()
                .zip(axis(16, 1))
                .map(|(x, y)| x + 0.15 * y)
                .collect(),
        );
        let map = build(vec![
            cloud(&a, 10, 0.05, 0),
            cloud(&twin, 10, 0.05, 4),
            cloud(&axis(16, 8), 10, 0.05, 8),
        ]);
        assert_eq!(map.close_pairs.len(), 1);
        let pair = &map.close_pairs[0];
        assert_eq!((pair.a, pair.b), (0, 1));
        assert!(pair.overlapping(map.threshold));
    }

    #[test]
    fn scores_a_sample_against_a_person_whose_centroid_lies_far_away() {
        let mix = |a: usize, wa: f32, b: usize, wb: f32| {
            axis(16, a)
                .iter()
                .zip(axis(16, b))
                .map(|(x, y)| wa * x + wb * y)
                .collect::<Vec<_>>()
        };
        let mut first = vec![axis(16, 0)];
        first.extend((1..4).map(|i| mix(0, 0.8, i, 0.6)));
        let near_copy = (1.0f32 - 0.98 * 0.98).sqrt();
        let mut second = (5..8)
            .map(|i| mix(0, 0.98, i, near_copy))
            .collect::<Vec<_>>();
        second.extend(cloud(&axis(16, 4), 40, 0.02, 3));
        let map = build(vec![first, second]);
        let query = &map.dots[0];
        assert_eq!(query.flag, Flag::Stray);
        assert_eq!(query.rival.map(|r| r.0), Some(1));
        assert!(query.rival.unwrap().1 > 0.97);
    }

    #[test]
    fn handles_tiny_databases() {
        assert!(build(Vec::new()).people.is_empty());
        let single = build(vec![vec![axis(8, 0)]]);
        assert_eq!(single.dots.len(), 1);
        assert!(single.dots[0].position.iter().all(|v| v.is_finite()));
        let pair = build(vec![
            vec![axis(8, 0)],
            vec![axis(8, 0)],
            vec![axis(8, 3), axis(8, 3)],
        ]);
        assert!(
            pair.dots
                .iter()
                .all(|d| d.position.iter().all(|v| v.is_finite()))
        );
        assert!(pair.stress.is_finite());
        assert_eq!(pair.dots[0].flag, Flag::Stray);
        assert_eq!(assign_colors(&[[0.0, 0.0]], &[[0.7, 0.0, 0.0]]), vec![0]);
    }

    #[test]
    fn neighbors_get_different_colors() {
        let centers = [[0.0, 0.0], [0.1, 0.0], [0.05, 0.08], [5.0, 5.0]];
        let palette = [
            [0.7, 0.1, 0.0],
            [0.7, -0.1, 0.0],
            [0.7, 0.0, 0.1],
            [0.7, 0.0, -0.1],
        ];
        let colors = assign_colors(&centers, &palette);
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[0], colors[2]);
        assert_ne!(colors[1], colors[2]);
    }
}
