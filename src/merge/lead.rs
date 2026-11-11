//! Which track carries the speech, 10 ms frame by 10 ms frame.
//!
//! In a call, usually one side talks at a time. The microphone hears the
//! people in the room plus the loudspeaker's echo of the remote side; the
//! system audio holds only the remote side. So the microphone leads where it
//! is clearly louder than the echo the system audio predicts, and the system
//! audio leads where it is active and the microphone is not.
//!
//! The predicted echo is the system envelope, delayed, held over a decaying
//! room tail, and scaled by the leak gain. The gain is a low quantile of the
//! microphone-to-prediction ratio over the system's active frames within a
//! minute or so: local speech only ever raises that ratio, so its low end is
//! the leak, and following it in time copes with volume changes.

use crate::types::Lead;

/// Frames of room reverberation kept in the prediction, and their decay.
const TAIL_FRAMES: usize = 15;
const TAIL_DECAY: f32 = 0.7;
const GAIN_HALF_WINDOW_FRAMES: usize = 6_000;
const GAIN_STEP_FRAMES: usize = 500;
const GAIN_QUANTILE: f32 = 0.2;
const MIN_GAIN_FRAMES: usize = 200;
const NOISE_QUANTILE: f32 = 0.1;
/// Frames quieter than this are digital silence (a muted or missing
/// microphone), not room noise, and the noise floor is never lower.
const NOISE_FLOOR: f32 = 3e-4;
/// Microphone speech is at least this many times the noise floor.
const MIC_SPEECH_OVER_NOISE: f32 = 3.0;
/// The microphone leads when it is this much louder than the predicted echo.
pub const MIC_EXCESS_DB: f32 = 8.0;
const MIN_DELAY_FRAMES: i32 = -5;
const MAX_DELAY_FRAMES: i32 = 30;
const MIN_DELAY_SEARCH_FRAMES: usize = 100;
/// Delays scoring within this of the best are equally good; the earliest
/// wins, since speech rhythm repeats the true peak a syllable later.
const DELAY_SCORE_TOLERANCE: f64 = 0.02;
/// Microphone frames this far above the typical echo ratio (natural log,
/// about 13 dB) are local speech and left out of the delay estimate.
const LOCAL_SPEECH_LOG_RATIO: f64 = 1.5;
/// Without calibration, envelopes this correlated show a loudspeaker leak.
const MIN_ECHO_PATH_CORRELATION: f64 = 0.3;
/// A calibrated delay applies this far from where it was measured.
const DELAY_HORIZON_FRAMES: usize = 18_000;

/// The lead of every microphone-time frame and the delays it assumed.
#[derive(Default)]
pub struct LeadTrack {
    pub frames: Vec<Option<Lead>>,
    /// Whether the microphone hears the loudspeaker at all. Without such a
    /// path (headphones), quiet microphone sound is not echo.
    pub echo_path: bool,
    /// `(frame, delay)` where calibration measured the loudspeaker delay.
    calibrated: Vec<(usize, i32)>,
    fallback_delay: i32,
}

impl LeadTrack {
    /// `calibrated` holds `(frame, delay)` points in frame order; without
    /// any, one delay is estimated for the whole recording.
    pub fn new(
        mic: &[f32],
        system: &[f32],
        activity_threshold: f32,
        calibrated: Vec<(usize, i32)>,
    ) -> Self {
        let (fallback_delay, echo_path) = if calibrated.is_empty() {
            let (delay, correlation) = estimate_delay(mic, system, activity_threshold);
            (delay, correlation >= MIN_ECHO_PATH_CORRELATION)
        } else {
            let mut delays = calibrated
                .iter()
                .map(|(_, delay)| *delay)
                .collect::<Vec<_>>();
            delays.sort_unstable();
            (delays[delays.len() / 2], true)
        };
        let mut track = Self {
            frames: Vec::new(),
            echo_path,
            calibrated,
            fallback_delay,
        };
        let n = mic.len();
        let delayed = (0..n)
            .map(|frame| {
                usize::try_from(frame as i64 - i64::from(track.delay_at(frame)))
                    .ok()
                    .and_then(|index| system.get(index).copied())
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let predicted = room_tail(&delayed);
        let active = predicted
            .iter()
            .map(|&level| level >= activity_threshold)
            .collect::<Vec<_>>();
        let gains = if echo_path {
            leak_gains(mic, &predicted, &active)
        } else {
            vec![0.0; n]
        };
        let noise = quantile(
            mic.iter()
                .copied()
                .filter(|&level| level >= NOISE_FLOOR)
                .collect(),
            NOISE_QUANTILE,
        )
        .unwrap_or(NOISE_FLOOR)
        .max(NOISE_FLOOR);
        let excess = 10f32.powf(MIC_EXCESS_DB / 20.0);
        track.frames = (0..n)
            .map(|frame| {
                let echo = gains[frame] * predicted[frame] + noise;
                let mic_leads =
                    mic[frame] > MIC_SPEECH_OVER_NOISE * noise && mic[frame] >= excess * echo;
                match (mic_leads, active[frame]) {
                    (true, true) => Some(Lead::Both),
                    (true, false) => Some(Lead::Mic),
                    (false, true) => Some(Lead::System),
                    (false, false) => None,
                }
            })
            .collect();
        track
    }

    /// The loudspeaker delay in frames at a microphone frame: from the
    /// nearest calibration point within three minutes, else the overall one.
    pub fn delay_at(&self, frame: usize) -> i32 {
        let next = self.calibrated.partition_point(|(at, _)| *at < frame);
        [next.checked_sub(1), Some(next)]
            .into_iter()
            .flatten()
            .filter_map(|index| self.calibrated.get(index))
            .map(|(at, delay)| (at.abs_diff(frame), *delay))
            .filter(|(distance, _)| *distance <= DELAY_HORIZON_FRAMES)
            .min_by_key(|(distance, _)| *distance)
            .map_or(self.fallback_delay, |(_, delay)| delay)
    }
}

/// The delay at which the microphone's log envelope follows the system's
/// over the system's active frames, and that correlation.
fn estimate_delay(mic: &[f32], system: &[f32], threshold: f32) -> (i32, f64) {
    let log = |level: f32| f64::from(level + 1e-6).ln();
    let scores = (MIN_DELAY_FRAMES..=MAX_DELAY_FRAMES)
        .filter_map(|delay| {
            let pairs = system
                .iter()
                .enumerate()
                .filter(|(_, level)| **level >= threshold)
                .filter_map(|(index, &level)| {
                    let mic_index = usize::try_from(index as i64 + i64::from(delay)).ok()?;
                    Some((log(level), log(*mic.get(mic_index)?)))
                })
                .collect::<Vec<_>>();
            let pairs = without_local_speech(pairs);
            (pairs.len() >= MIN_DELAY_SEARCH_FRAMES).then(|| (delay, correlation(&pairs)))
        })
        .collect::<Vec<_>>();
    let best = scores
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    scores
        .into_iter()
        .find(|(_, score)| *score >= best - DELAY_SCORE_TOLERANCE)
        .unwrap_or((0, f64::NEG_INFINITY))
}

/// The `(system, mic)` log-level pairs whose microphone level is not far
/// above the typical ratio: those frames are someone in the room talking
/// over the remote side, which says nothing about the echo.
fn without_local_speech(pairs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut offsets = pairs.iter().map(|(x, y)| y - x).collect::<Vec<_>>();
    offsets.sort_unstable_by(f64::total_cmp);
    let Some(&median) = offsets.get(offsets.len() / 2) else {
        return pairs;
    };
    pairs
        .into_iter()
        .filter(|(x, y)| y - x <= median + LOCAL_SPEECH_LOG_RATIO)
        .collect()
}

fn correlation(pairs: &[(f64, f64)]) -> f64 {
    let count = pairs.len() as f64;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / count;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / count;
    let (mut covariance, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (x, y) in pairs {
        covariance += (x - mean_x) * (y - mean_y);
        var_x += (x - mean_x) * (x - mean_x);
        var_y += (y - mean_y) * (y - mean_y);
    }
    let denominator = (var_x * var_y).sqrt();
    if denominator > 1e-12 {
        covariance / denominator
    } else {
        f64::NEG_INFINITY
    }
}

/// Each frame's level, or a decaying earlier one where the room still rings.
fn room_tail(levels: &[f32]) -> Vec<f32> {
    let mut held = levels.to_vec();
    let mut decay = 1.0;
    for lag in 1..=TAIL_FRAMES {
        decay *= TAIL_DECAY;
        for frame in lag..levels.len() {
            held[frame] = held[frame].max(levels[frame - lag] * decay);
        }
    }
    held
}

/// Leak gain per frame, as a ratio of levels.
fn leak_gains(mic: &[f32], predicted: &[f32], active: &[bool]) -> Vec<f32> {
    let ratios = (0..mic.len())
        .filter(|&frame| active[frame])
        .map(|frame| (frame, mic[frame] / predicted[frame].max(1e-9)))
        .collect::<Vec<_>>();
    let overall = quantile(
        ratios.iter().map(|(_, ratio)| *ratio).collect(),
        GAIN_QUANTILE,
    )
    .unwrap_or(0.0);
    let mut gains = vec![overall; mic.len()];
    for start in (0..mic.len()).step_by(GAIN_STEP_FRAMES) {
        let low = start.saturating_sub(GAIN_HALF_WINDOW_FRAMES);
        let high = start + GAIN_HALF_WINDOW_FRAMES;
        let first = ratios.partition_point(|(frame, _)| *frame < low);
        let last = ratios.partition_point(|(frame, _)| *frame < high);
        if last - first >= MIN_GAIN_FRAMES {
            let local = quantile(
                ratios[first..last]
                    .iter()
                    .map(|(_, ratio)| *ratio)
                    .collect(),
                GAIN_QUANTILE,
            )
            .unwrap_or(overall);
            let end = (start + GAIN_STEP_FRAMES).min(mic.len());
            gains[start..end].fill(local);
        }
    }
    gains
}

fn quantile(mut values: Vec<f32>, fraction: f32) -> Option<f32> {
    values.retain(|value| value.is_finite());
    values.sort_unstable_by(f32::total_cmp);
    let index = ((values.len() as f32 * fraction) as usize).min(values.len().checked_sub(1)?);
    values.get(index).copied()
}

/// The lead of a stretch of frames: both sides when double talk is the most
/// common, otherwise the side with more frames; `None` without any.
pub fn summarize(frames: &[Option<Lead>]) -> Option<Lead> {
    let count = |lead| frames.iter().filter(|frame| **frame == Some(lead)).count();
    let (mic, system, both) = (count(Lead::Mic), count(Lead::System), count(Lead::Both));
    if mic + system + both == 0 {
        None
    } else if both >= mic && both >= system {
        Some(Lead::Both)
    } else if mic > system {
        Some(Lead::Mic)
    } else {
        Some(Lead::System)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_louder_side_leads_and_echo_follows_the_system() {
        // System talks for 4 s, then the room talks alone, then both.
        let frames = 1_200;
        let mut system = vec![0.0f32; frames];
        let mut mic = vec![0.001f32; frames];
        for (frame, level) in system.iter_mut().enumerate().take(400) {
            *level = 0.2 + 0.1 * ((frame / 7) % 3) as f32;
        }
        system[800..1_000].fill(0.2);
        for frame in 5..frames {
            mic[frame] += system[frame - 5] * 0.3;
        }
        mic[450..750].iter_mut().for_each(|level| *level += 0.3);
        mic[850..950].iter_mut().for_each(|level| *level += 0.6);
        let track = LeadTrack::new(&mic, &system, 0.02, Vec::new());
        assert!(track.echo_path);
        assert_eq!(track.delay_at(0), 5);
        let lead = track.frames;
        assert_eq!(summarize(&lead[50..380]), Some(Lead::System));
        assert_eq!(summarize(&lead[460..740]), Some(Lead::Mic));
        assert_eq!(summarize(&lead[860..940]), Some(Lead::Both));
        assert_eq!(summarize(&lead[1_100..1_200]), None);
    }
}
