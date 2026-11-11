//! Average-linkage agglomerative clustering of unit vectors on cosine
//! distance, by the nearest-neighbor chain algorithm in O(N²) time and
//! O(N²/2) memory.

use crate::speaker::embedding;

#[derive(Clone, Copy, Debug)]
pub enum Stop {
    /// Merge while the closest clusters are at most this distance apart.
    Distance(f32),
    /// Merge until this many clusters remain.
    Count(usize),
}

/// Cluster label per vector, numbered 0.. in order of first appearance.
pub fn agglomerate(vectors: &[Vec<f32>], stop: Stop) -> Vec<usize> {
    let n = vectors.len();
    if n < 2 {
        return vec![0; n];
    }
    let mut distances = Condensed::new(n, |i, j| {
        1.0 - embedding::cosine(&vectors[i], &vectors[j]).unwrap_or(0.0)
    });
    let mut merges = nn_chain(&mut distances, n);
    merges.sort_by(|left, right| left.2.total_cmp(&right.2));
    let applied = match stop {
        Stop::Distance(threshold) => merges
            .iter()
            .take_while(|(_, _, distance)| *distance <= threshold)
            .count(),
        // Exactly `count` clusters, even when several merges tie at the cut
        // height. SciPy's `fcluster(..., "maxclust")` cuts by height and can
        // then return fewer clusters; the exact count is intended here.
        Stop::Count(count) => n.saturating_sub(count.max(1)),
    };
    let mut parent = (0..n).collect::<Vec<_>>();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    for &(a, b, _) in &merges[..applied] {
        let (a, b) = (root(&mut parent, a), root(&mut parent, b));
        parent[a] = b;
    }
    let roots = (0..n)
        .map(|node| root(&mut parent, node))
        .collect::<Vec<_>>();
    renumber(&roots)
}

/// Moves members of clusters smaller than `min_size` to the most similar
/// centroid of a cluster that is at least that large, when one exists.
pub fn absorb_small_clusters(
    vectors: &[Vec<f32>],
    labels: &[usize],
    min_size: usize,
) -> Vec<usize> {
    let count = labels.iter().max().map_or(0, |max| max + 1);
    let mut sizes = vec![0; count];
    for &label in labels {
        sizes[label] += 1;
    }
    let large = (0..count)
        .filter(|&label| sizes[label] >= min_size)
        .collect::<Vec<_>>();
    if large.is_empty() {
        return labels.to_vec();
    }
    let centroids = centroids(vectors, labels, count);
    let relabeled = labels
        .iter()
        .zip(vectors)
        .map(|(&label, vector)| {
            if sizes[label] >= min_size {
                return label;
            }
            large
                .iter()
                .copied()
                .map(|candidate| {
                    let score = centroids[candidate]
                        .as_deref()
                        .and_then(|centroid| embedding::cosine(centroid, vector))
                        .unwrap_or(f32::NEG_INFINITY);
                    (candidate, score)
                })
                .fold((large[0], f32::NEG_INFINITY), |best, current| {
                    if current.1 > best.1 { current } else { best }
                })
                .0
        })
        .collect::<Vec<_>>();
    renumber(&relabeled)
}

/// Normalized mean vector of each label, or `None` when it cannot be formed.
pub fn centroids(vectors: &[Vec<f32>], labels: &[usize], count: usize) -> Vec<Option<Vec<f32>>> {
    let mut members = vec![Vec::new(); count];
    for (&label, vector) in labels.iter().zip(vectors) {
        members[label].push(vector.clone());
    }
    members
        .iter()
        .map(|members| {
            let dimension = members.first()?.len();
            let mut sum = vec![0.0f32; dimension];
            for vector in members {
                for (total, value) in sum.iter_mut().zip(vector) {
                    *total += value;
                }
            }
            embedding::normalize(&mut sum).then_some(sum)
        })
        .collect()
}

fn renumber(labels: &[usize]) -> Vec<usize> {
    let mut map = std::collections::HashMap::new();
    labels
        .iter()
        .map(|label| {
            let next = map.len();
            *map.entry(*label).or_insert(next)
        })
        .collect()
}

struct Condensed {
    n: usize,
    values: Vec<f32>,
}

impl Condensed {
    fn new(n: usize, distance: impl Fn(usize, usize) -> f32) -> Self {
        let mut values = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in i + 1..n {
                values.push(distance(i, j));
            }
        }
        Self { n, values }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.n * i - i * (i + 1) / 2 + j - i - 1
    }

    fn get(&self, i: usize, j: usize) -> f32 {
        self.values[self.index(i, j)]
    }

    fn set(&mut self, i: usize, j: usize, value: f32) {
        let index = self.index(i, j);
        self.values[index] = value;
    }
}

/// Merges as `(removed, kept, distance)`, not yet in distance order.
fn nn_chain(distances: &mut Condensed, n: usize) -> Vec<(usize, usize, f32)> {
    let mut size = vec![1usize; n];
    let mut active = vec![true; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut merges = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&on| on).expect("an active cluster"));
        }
        let (x, y, distance) = loop {
            let x = chain[chain.len() - 1];
            let previous = (chain.len() > 1).then(|| chain[chain.len() - 2]);
            let (mut y, mut best) = match previous {
                Some(previous) => (previous, distances.get(x, previous)),
                None => (usize::MAX, f32::INFINITY),
            };
            for i in (0..n).filter(|&i| active[i] && i != x) {
                let distance = distances.get(x, i);
                if distance < best {
                    best = distance;
                    y = i;
                }
            }
            if previous == Some(y) {
                chain.truncate(chain.len() - 2);
                break (x, y, best);
            }
            chain.push(y);
        };
        let (removed, kept) = if x < y { (x, y) } else { (y, x) };
        let total = (size[removed] + size[kept]) as f32;
        for i in (0..n).filter(|&i| active[i] && i != removed && i != kept) {
            let merged = (size[removed] as f32 * distances.get(i, removed)
                + size[kept] as f32 * distances.get(i, kept))
                / total;
            distances.set(i, kept, merged);
        }
        size[kept] += size[removed];
        active[removed] = false;
        merges.push((removed, kept, distance));
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin()]
    }

    #[test]
    fn average_linkage_cuts_at_distance_or_count() {
        let vectors = [0.0, 5.0, 10.0, 90.0, 95.0, 200.0].map(unit);
        assert_eq!(
            agglomerate(&vectors, Stop::Distance(0.1)),
            [0, 0, 0, 1, 1, 2]
        );
        assert_eq!(
            agglomerate(&vectors, Stop::Distance(0.0)),
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(agglomerate(&vectors, Stop::Count(2)), [0, 0, 0, 0, 0, 1]);
        assert_eq!(agglomerate(&vectors, Stop::Count(1)), [0; 6]);
    }

    #[test]
    fn small_clusters_join_the_nearest_large_centroid() {
        let vectors = [0.0, 5.0, 10.0, 80.0, 85.0, 90.0, 30.0, 70.0].map(unit);
        let labels = [0, 0, 0, 1, 1, 1, 2, 3];
        assert_eq!(
            absorb_small_clusters(&vectors, &labels, 3),
            [0, 0, 0, 1, 1, 1, 0, 1]
        );
        assert_eq!(absorb_small_clusters(&vectors[6..], &[0, 1], 3), [0, 1]);
    }
}
