//! Slowing speech down without changing its pitch: WSOLA, which overlaps
//! windowed frames of the input at a slower pace, each shifted a little so
//! that its waveform lines up with what the frame before would have played.

use std::f32::consts::PI;

/// 32 ms frames, half overlapping.
const FRAME: usize = 512;
const HOP: usize = FRAME / 2;
/// How far a frame may shift to line up: about half the period of a low voice.
const TOLERANCE: isize = 128;
/// Every this many samples are compared when lining frames up.
const STRIDE: usize = 4;

/// `samples` played at `rate` (0.5 is half speed) at the same pitch, made a
/// little at a time so that playback can start at once.
pub struct Slowed {
    samples: Vec<f32>,
    rate: f64,
    window: Vec<f32>,
    frame: usize,
    previous: isize,
    pending: Vec<f32>,
    weight: Vec<f32>,
    remaining: usize,
}

impl Slowed {
    pub fn new(samples: Vec<f32>, rate: f64) -> Self {
        Self {
            remaining: (samples.len() as f64 / rate).round() as usize,
            samples,
            rate,
            window: (0..FRAME)
                .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / FRAME as f32).cos())
                .collect(),
            frame: 0,
            previous: 0,
            pending: vec![0.0; FRAME],
            weight: vec![0.0; FRAME],
        }
    }

    fn at(&self, position: isize) -> f32 {
        usize::try_from(position)
            .ok()
            .and_then(|index| self.samples.get(index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Appends the next few milliseconds to `out`; false at the end.
    pub fn fill(&mut self, out: &mut Vec<f32>) -> bool {
        if self.remaining == 0 {
            return false;
        }
        let nominal = (self.frame as f64 * HOP as f64 * self.rate).round() as isize;
        let position = if self.frame == 0 {
            0
        } else {
            let natural = self.previous + HOP as isize;
            (nominal - TOLERANCE..=nominal + TOLERANCE)
                .map(|candidate| {
                    let similarity = (0..FRAME as isize)
                        .step_by(STRIDE)
                        .map(|index| self.at(natural + index) * self.at(candidate + index))
                        .sum::<f32>();
                    (candidate, similarity)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(nominal, |(candidate, _)| candidate)
        };
        for index in 0..FRAME {
            let gain = self.window[index];
            self.pending[index] += gain * self.at(position + index as isize);
            self.weight[index] += gain;
        }
        let count = HOP.min(self.remaining);
        out.extend(
            self.pending[..count]
                .iter()
                .zip(&self.weight)
                .map(|(sample, weight)| {
                    if *weight > 1e-3 {
                        sample / weight
                    } else {
                        *sample
                    }
                }),
        );
        for buffer in [&mut self.pending, &mut self.weight] {
            buffer.copy_within(HOP.., 0);
            buffer[FRAME - HOP..].fill(0.0);
        }
        self.remaining -= count;
        self.previous = position;
        self.frame += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_speed_is_twice_as_long_at_the_same_pitch() {
        let tone = (0..16_000)
            .map(|index| (2.0 * PI * 200.0 * index as f32 / 16_000.0).sin() * 0.5)
            .collect::<Vec<_>>();
        let mut slowed = Slowed::new(tone, 0.5);
        let mut slow = Vec::new();
        while slowed.fill(&mut slow) {}
        assert_eq!(slow.len(), 32_000);
        let middle = &slow[4_000..28_000];
        let crossings = middle
            .windows(2)
            .filter(|pair| (pair[0] < 0.0) != (pair[1] < 0.0))
            .count();
        let hertz = crossings as f32 / 2.0 / (middle.len() as f32 / 16_000.0);
        assert!((hertz - 200.0).abs() < 5.0, "{hertz} Hz");
        let rms =
            (middle.iter().map(|sample| sample * sample).sum::<f32>() / middle.len() as f32).sqrt();
        assert!((rms - 0.5 / 2f32.sqrt()).abs() < 0.04, "rms {rms}");
    }

    #[test]
    fn input_shorter_than_a_frame_still_comes_out_twice_as_long() {
        let mut slowed = Slowed::new(vec![0.25; 100], 0.5);
        let mut slow = Vec::new();
        while slowed.fill(&mut slow) {}
        assert_eq!(slow.len(), 200);
        assert!(
            slow.iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 0.25 + 1e-6)
        );
    }
}
