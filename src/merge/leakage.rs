use crate::audio::archive;
use crate::types::{AudioSource, SAMPLE_RATE, TimedWord};
use serde::Serialize;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const PHRASE_GAP_MS: u64 = 900;
const MAX_PHRASE_MS: u64 = 12_000;
const CANDIDATE_EARLY_MS: u64 = 1_500;
const CANDIDATE_LATE_MS: u64 = 500;
const TEXT_ONLY_MIN_MATCHES: usize = 4;
const AUDIO_MIN_MATCHES: usize = 2;
const TEXT_ONLY_MIN_SIMILARITY: f32 = 0.85;
const AUDIO_MIN_SIMILARITY: f32 = 0.65;
const AUDIO_MIN_CORRELATION: f32 = 0.75;
const MIN_MATCHED_SPAN_MS: u64 = 600;
const TEXT_MAX_DELAY_MS: i64 = 750;
const MAX_WORD_DELAY_DEVIATION_MS: u64 = 350;
const MAX_AUDIO_TEXT_DELAY_DIFFERENCE_MS: i64 = 500;
const AUDIO_FRAME_MS: u64 = 10;
const AUDIO_FRAME_SAMPLES: usize = SAMPLE_RATE as usize * AUDIO_FRAME_MS as usize / 1_000;
const MIN_DELAY_FRAMES: i32 = -25;
const MAX_DELAY_FRAMES: i32 = 100;
const CALIBRATION_WINDOW_FRAMES: usize = 2_000 / AUDIO_FRAME_MS as usize;
const CALIBRATION_MIN_ACTIVE_FRACTION: f32 = 0.5;
const CALIBRATION_MIN_CORRELATION: f32 = 0.8;
const CALIBRATION_DELAY_TOLERANCE_FRAMES: i32 = 3;
const REFERENCE_HORIZON_MS: u64 = 180_000;
const AUDIO_WORD_MIN_EVALUATED_MS: u64 = 200;
const AUDIO_WORD_MIN_CONTEXT_MS: u64 = 1_500;
const AUDIO_WORD_DELAY_SEARCH_FRAMES: i32 = 3;
const AUDIO_WORD_MIN_CORRELATION: f32 = 0.75;
const AUDIO_WORD_MAX_EXCESS_LN: f32 = 0.7;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EchoDetection {
    pub mic_start_ms: u64,
    pub mic_end_ms: u64,
    pub system_start_ms: u64,
    pub system_end_ms: u64,
    pub words: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_delay_ms: Option<i64>,
    pub evidence: &'static str,
}

#[derive(Debug)]
pub struct DetectionResult {
    pub echo: Vec<bool>,
    pub detections: Vec<EchoDetection>,
}

pub fn detect_leaked_mic_words(
    words: &[TimedWord],
    audio: Option<&AudioEnvelopes>,
) -> DetectionResult {
    let mut echo = vec![false; words.len()];
    let mut detections = audio
        .map(|audio| detect_with_audio(words, audio, &mut echo))
        .unwrap_or_default();

    let system_tokens = source_tokens(words, AudioSource::System);
    let mut maximum_system_ends = Vec::with_capacity(system_tokens.len());
    let mut maximum_end = 0;
    for (index, _) in &system_tokens {
        maximum_end = maximum_end.max(words[*index].end_ms);
        maximum_system_ends.push(maximum_end);
    }

    for phrase in mic_phrases(words, &echo) {
        let mic_tokens = phrase
            .iter()
            .filter_map(|&index| normalize_token(&words[index].text).map(|text| (index, text)))
            .collect::<Vec<_>>();
        if mic_tokens.len() < AUDIO_MIN_MATCHES {
            continue;
        }
        let mic_start_ms = phrase
            .iter()
            .map(|&index| words[index].start_ms)
            .min()
            .unwrap_or(0);
        let mic_end_ms = phrase
            .iter()
            .map(|&index| words[index].end_ms)
            .max()
            .unwrap_or(mic_start_ms);
        let candidate_start = mic_start_ms.saturating_sub(CANDIDATE_EARLY_MS);
        let candidate_end = mic_end_ms.saturating_add(CANDIDATE_LATE_MS);
        let first = maximum_system_ends.partition_point(|end| *end < candidate_start);
        let last =
            system_tokens.partition_point(|(index, _)| words[*index].start_ms <= candidate_end);
        let candidates = system_tokens[first.min(last)..last]
            .iter()
            .filter(|(index, _)| words[*index].end_ms >= candidate_start)
            .cloned()
            .collect::<Vec<_>>();
        if candidates.len() < AUDIO_MIN_MATCHES {
            continue;
        }

        let matches = align_tokens(&mic_tokens, &candidates);
        if matches.len() < AUDIO_MIN_MATCHES {
            continue;
        }
        let similarity = matches.len() as f32 / mic_tokens.len().min(candidates.len()) as f32;
        let matched_system_start = matches
            .iter()
            .map(|(_, system)| words[*system].start_ms)
            .min()
            .unwrap_or(candidate_start);
        let matched_system_end = matches
            .iter()
            .map(|(_, system)| words[*system].end_ms)
            .max()
            .unwrap_or(candidate_end);
        let matched_span_ms = matched_system_end.saturating_sub(matched_system_start);
        let Some((text_delay_ms, maximum_delay_deviation_ms)) = timing_alignment(&matches, words)
        else {
            continue;
        };
        let timing_consistent = maximum_delay_deviation_ms <= MAX_WORD_DELAY_DEVIATION_MS;
        let audio_match =
            audio.and_then(|audio| audio.similarity(matched_system_start, matched_system_end));
        let text_only = audio.is_none()
            && matches.len() >= TEXT_ONLY_MIN_MATCHES
            && similarity >= TEXT_ONLY_MIN_SIMILARITY
            && matched_span_ms >= MIN_MATCHED_SPAN_MS
            && timing_consistent
            && (-250..=TEXT_MAX_DELAY_MS).contains(&text_delay_ms);
        let audio_supported = matches.len() >= AUDIO_MIN_MATCHES
            && similarity >= AUDIO_MIN_SIMILARITY
            && matched_span_ms >= MIN_MATCHED_SPAN_MS
            && timing_consistent
            && audio_match.is_some_and(|(correlation, delay)| {
                correlation >= AUDIO_MIN_CORRELATION
                    && (text_delay_ms - delay).abs() <= MAX_AUDIO_TEXT_DELAY_DIFFERENCE_MS
            });
        if !text_only && !audio_supported {
            continue;
        }

        for (mic, _) in &matches {
            echo[*mic] = true;
        }
        let mic_match_start = matches
            .iter()
            .map(|(mic, _)| words[*mic].start_ms)
            .min()
            .unwrap_or(mic_start_ms);
        let mic_match_end = matches
            .iter()
            .map(|(mic, _)| words[*mic].end_ms)
            .max()
            .unwrap_or(mic_end_ms);
        detections.push(EchoDetection {
            mic_start_ms: mic_match_start,
            mic_end_ms: mic_match_end,
            system_start_ms: matched_system_start,
            system_end_ms: matched_system_end,
            words: matches.len(),
            text_similarity: Some(similarity),
            audio_similarity: audio_match.map(|(correlation, _)| correlation),
            audio_delay_ms: audio_match.map(|(_, delay)| delay),
            evidence: if audio_supported {
                "text_audio"
            } else {
                "text"
            },
        });
    }

    detections.sort_by_key(|detection| detection.mic_start_ms);

    DetectionResult { echo, detections }
}

fn detect_with_audio(
    words: &[TimedWord],
    audio: &AudioEnvelopes,
    echo: &mut [bool],
) -> Vec<EchoDetection> {
    let mut detections = Vec::new();
    for phrase in mic_phrases(words, echo) {
        let evidence = phrase
            .iter()
            .map(|&index| audio.word_evidence(&words[index]))
            .collect::<Vec<_>>();
        let mut phrase_echo = evidence
            .iter()
            .map(AudioWordEvidence::passes)
            .collect::<Vec<_>>();
        let smoothed = (1..phrase_echo.len().saturating_sub(1))
            .filter(|&position| {
                !phrase_echo[position]
                    && phrase_echo[position - 1]
                    && phrase_echo[position + 1]
                    && evidence[position].is_explained()
            })
            .collect::<Vec<_>>();
        for position in smoothed {
            phrase_echo[position] = true;
        }
        for (&index, &is_echo) in phrase.iter().zip(&phrase_echo) {
            echo[index] = is_echo;
        }

        let mut run_start = 0;
        while run_start < phrase.len() {
            if !phrase_echo[run_start] {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start + 1;
            while run_end < phrase.len() && phrase_echo[run_end] {
                run_end += 1;
            }
            let run = &phrase[run_start..run_end];
            let mic_start_ms = run
                .iter()
                .map(|&index| words[index].start_ms)
                .min()
                .unwrap_or(0);
            let mic_end_ms = run
                .iter()
                .map(|&index| words[index].end_ms)
                .max()
                .unwrap_or(mic_start_ms);
            let correlations = evidence[run_start..run_end]
                .iter()
                .filter_map(|evidence| evidence.correlation)
                .collect::<Vec<_>>();
            let audio_similarity =
                correlations.iter().sum::<f32>() / correlations.len().max(1) as f32;
            let delay_ms = i64::from(
                evidence[run_start]
                    .delay_frames
                    .expect("an audio-flagged run starts with an estimate"),
            ) * AUDIO_FRAME_MS as i64;
            detections.push(EchoDetection {
                mic_start_ms,
                mic_end_ms,
                system_start_ms: shifted_to_system_time(mic_start_ms, delay_ms),
                system_end_ms: shifted_to_system_time(mic_end_ms, delay_ms),
                words: run.len(),
                text_similarity: None,
                audio_similarity: Some(audio_similarity),
                audio_delay_ms: Some(delay_ms),
                evidence: "audio",
            });
            run_start = run_end;
        }
    }
    detections
}

fn shifted_to_system_time(mic_time_ms: u64, delay_ms: i64) -> u64 {
    if delay_ms >= 0 {
        mic_time_ms.saturating_sub(delay_ms as u64)
    } else {
        mic_time_ms.saturating_add(delay_ms.unsigned_abs())
    }
}

fn mic_phrases(words: &[TimedWord], suppressed: &[bool]) -> Vec<Vec<usize>> {
    let mut indices = words
        .iter()
        .enumerate()
        .filter_map(|(index, word)| {
            (word.source == AudioSource::Mic && !suppressed[index]).then_some(index)
        })
        .collect::<Vec<_>>();
    indices.sort_by_key(|&index| (words[index].start_ms, words[index].end_ms));
    let mut phrases = Vec::<Vec<usize>>::new();
    for index in indices {
        let split = phrases.last().is_some_and(|phrase| {
            let first = &words[phrase[0]];
            let previous = &words[*phrase.last().expect("nonempty phrase")];
            words[index].start_ms.saturating_sub(previous.end_ms) > PHRASE_GAP_MS
                || words[index].end_ms.saturating_sub(first.start_ms) > MAX_PHRASE_MS
        });
        if split || phrases.is_empty() {
            phrases.push(Vec::new());
        }
        phrases.last_mut().expect("phrase exists").push(index);
    }
    phrases
}

fn source_tokens(words: &[TimedWord], source: AudioSource) -> Vec<(usize, String)> {
    let mut tokens = words
        .iter()
        .enumerate()
        .filter_map(|(index, word)| {
            (word.source == source)
                .then(|| normalize_token(&word.text).map(|text| (index, text)))
                .flatten()
        })
        .collect::<Vec<_>>();
    tokens.sort_by_key(|(index, _)| (words[*index].start_ms, words[*index].end_ms));
    tokens
}

fn normalize_token(text: &str) -> Option<String> {
    let normalized = text
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

fn align_tokens(left: &[(usize, String)], right: &[(usize, String)]) -> Vec<(usize, usize)> {
    let width = right.len() + 1;
    let mut lengths = vec![0usize; (left.len() + 1) * width];
    for row in 0..left.len() {
        for column in 0..right.len() {
            let value = if tokens_match(&left[row].1, &right[column].1) {
                lengths[row * width + column] + 1
            } else {
                lengths[row * width + column + 1].max(lengths[(row + 1) * width + column])
            };
            lengths[(row + 1) * width + column + 1] = value;
        }
    }
    let mut row = left.len();
    let mut column = right.len();
    let mut matches = Vec::new();
    while row > 0 && column > 0 {
        if tokens_match(&left[row - 1].1, &right[column - 1].1)
            && lengths[row * width + column] == lengths[(row - 1) * width + column - 1] + 1
        {
            matches.push((left[row - 1].0, right[column - 1].0));
            row -= 1;
            column -= 1;
        } else if lengths[(row - 1) * width + column] >= lengths[row * width + column - 1] {
            row -= 1;
        } else {
            column -= 1;
        }
    }
    matches.reverse();
    matches
}

fn tokens_match(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let maximum = left.chars().count().max(right.chars().count());
    let allowed = match maximum {
        0..=4 => 0,
        5..=7 => 1,
        _ => 2,
    };
    allowed > 0 && edit_distance_with_limit(left, right, allowed).is_some()
}

fn timing_alignment(matches: &[(usize, usize)], words: &[TimedWord]) -> Option<(i64, u64)> {
    let mut delays = matches
        .iter()
        .map(|(mic, system)| words[*mic].start_ms as i64 - words[*system].start_ms as i64)
        .collect::<Vec<_>>();
    delays.sort_unstable();
    let median = *delays.get(delays.len() / 2)?;
    let maximum_deviation = delays
        .iter()
        .map(|delay| delay.abs_diff(median))
        .max()
        .unwrap_or(0);
    Some((median, maximum_deviation))
}

fn edit_distance_with_limit(left: &str, right: &str, limit: usize) -> Option<usize> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > limit {
        return None;
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_character) in left.iter().enumerate() {
        current[0] = left_index + 1;
        let mut row_minimum = current[0];
        for (right_index, right_character) in right.iter().enumerate() {
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + usize::from(left_character != right_character));
            row_minimum = row_minimum.min(current[right_index + 1]);
        }
        if row_minimum > limit {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= limit).then_some(previous[right.len()])
}

pub struct AudioEnvelopes {
    mic: Vec<f32>,
    system: Vec<f32>,
    activity_threshold: Option<f32>,
    reference_points: Vec<AudioReferencePoint>,
}

#[derive(Clone, Copy)]
struct AudioReferencePoint {
    center_frame: usize,
    delay_frames: i32,
    gain_ln: f32,
}

#[derive(Clone, Copy)]
struct AudioEstimate {
    delay_frames: i32,
    gain_ln: f32,
}

struct AudioWordEvidence {
    excess_ln: Option<f32>,
    correlation: Option<f32>,
    delay_frames: Option<i32>,
}

impl AudioWordEvidence {
    fn is_explained(&self) -> bool {
        self.excess_ln
            .is_some_and(|excess| excess <= AUDIO_WORD_MAX_EXCESS_LN)
    }

    fn passes(&self) -> bool {
        self.is_explained()
            && self
                .correlation
                .is_some_and(|correlation| correlation >= AUDIO_WORD_MIN_CORRELATION)
    }
}

impl AudioEnvelopes {
    pub fn read(mic_path: &Path, system_path: &Path) -> io::Result<Self> {
        let mic = read_rms_envelope(mic_path)?;
        let system = read_rms_envelope(system_path)?;
        Ok(Self::from_envelopes(mic, system))
    }

    #[cfg(test)]
    fn from_samples(mic: &[f32], system: &[f32]) -> Self {
        Self::from_envelopes(rms_envelope(mic), rms_envelope(system))
    }

    fn from_envelopes(mic: Vec<f32>, system: Vec<f32>) -> Self {
        let activity_threshold = activity_threshold(&system);
        let reference_points = activity_threshold
            .map(|threshold| calibrate_audio(&mic, &system, threshold))
            .unwrap_or_default();
        Self {
            mic,
            system,
            activity_threshold,
            reference_points,
        }
    }

    fn similarity(&self, start_ms: u64, end_ms: u64) -> Option<(f32, i64)> {
        let system_start = usize::try_from(start_ms.saturating_sub(200) / AUDIO_FRAME_MS).ok()?;
        let system_end = usize::try_from(end_ms.saturating_add(200) / AUDIO_FRAME_MS).ok()?;
        if system_end.saturating_sub(system_start) < 30 {
            return None;
        }
        let system_start = system_start.min(self.system.len());
        let system_end = system_end.min(self.system.len());
        let system = self.system.get(system_start..system_end)?;
        if system.len() < 30 {
            return None;
        }
        let maximum = system.iter().copied().fold(0.0f32, f32::max);
        if maximum <= 1e-5 {
            return None;
        }
        let activity_threshold = (maximum * 0.1).max(1e-4);
        (MIN_DELAY_FRAMES..=MAX_DELAY_FRAMES)
            .filter_map(|delay| {
                envelope_correlation(
                    system,
                    system_start,
                    &self.mic,
                    0,
                    delay,
                    activity_threshold,
                )
                .map(|correlation| (correlation, i64::from(delay) * AUDIO_FRAME_MS as i64))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
    }

    fn word_evidence(&self, word: &TimedWord) -> AudioWordEvidence {
        self.estimates_for_word(word)
            .into_iter()
            .map(|estimate| self.word_evidence_for_estimate(word, estimate))
            .max_by(|left, right| {
                left.passes()
                    .cmp(&right.passes())
                    .then_with(|| left.is_explained().cmp(&right.is_explained()))
                    .then_with(|| {
                        left.correlation
                            .unwrap_or(f32::NEG_INFINITY)
                            .total_cmp(&right.correlation.unwrap_or(f32::NEG_INFINITY))
                    })
            })
            .unwrap_or(AudioWordEvidence {
                excess_ln: None,
                correlation: None,
                delay_frames: None,
            })
    }

    fn estimates_for_word(&self, word: &TimedWord) -> Vec<AudioEstimate> {
        let midpoint_ms = word
            .start_ms
            .saturating_add(word.end_ms.saturating_sub(word.start_ms) / 2);
        let midpoint_frame = usize::try_from(midpoint_ms / AUDIO_FRAME_MS).unwrap_or(usize::MAX);
        let horizon_frames = usize::try_from(REFERENCE_HORIZON_MS / AUDIO_FRAME_MS)
            .expect("reference horizon fits usize");
        let first = self.reference_points.partition_point(|point| {
            point.center_frame < midpoint_frame.saturating_sub(horizon_frames)
        });
        let split = self
            .reference_points
            .partition_point(|point| point.center_frame <= midpoint_frame);
        let last = self.reference_points.partition_point(|point| {
            point.center_frame <= midpoint_frame.saturating_add(horizon_frames)
        });
        let before = self.reference_points[first..split].iter().rev().take(3);
        let after = self.reference_points[split..last].iter().take(3);
        [estimate_from_points(before), estimate_from_points(after)]
            .into_iter()
            .flatten()
            .collect()
    }

    fn word_evidence_for_estimate(
        &self,
        word: &TimedWord,
        estimate: AudioEstimate,
    ) -> AudioWordEvidence {
        let threshold = self
            .activity_threshold
            .expect("reference points require an activity threshold");
        let (evaluated_start_ms, evaluated_end_ms) =
            padded_interval(word, AUDIO_WORD_MIN_EVALUATED_MS);
        let mic_start = usize::try_from(evaluated_start_ms / AUDIO_FRAME_MS)
            .unwrap_or(usize::MAX)
            .min(self.mic.len());
        let mic_end =
            usize::try_from(evaluated_end_ms.saturating_add(AUDIO_FRAME_MS - 1) / AUDIO_FRAME_MS)
                .unwrap_or(usize::MAX)
                .min(self.mic.len());
        let (mic_energy, predicted_energy) =
            (mic_start..mic_end).fold((0.0f64, 0.0f64), |(mic, predicted), mic_index| {
                let system_value = signed_frame_index(mic_index, -estimate.delay_frames)
                    .and_then(|index| self.system.get(index).copied())
                    .unwrap_or(0.0)
                    .max(threshold);
                (
                    mic + f64::from(self.mic[mic_index]).powi(2),
                    predicted + f64::from(system_value).powi(2),
                )
            });
        let excess_ln = (mic_end > mic_start).then(|| {
            0.5 * ((mic_energy + 1e-12) / (predicted_energy + 1e-12)).ln() as f32 - estimate.gain_ln
        });

        let (context_start_ms, context_end_ms) = padded_interval(word, AUDIO_WORD_MIN_CONTEXT_MS);
        let context_mic_start = usize::try_from(context_start_ms / AUDIO_FRAME_MS)
            .unwrap_or(usize::MAX)
            .min(self.mic.len());
        let context_mic_end =
            usize::try_from(context_end_ms.saturating_add(AUDIO_FRAME_MS - 1) / AUDIO_FRAME_MS)
                .unwrap_or(usize::MAX)
                .min(self.mic.len());
        let system_start = signed_frame_index(context_mic_start, -estimate.delay_frames)
            .unwrap_or(0)
            .min(self.system.len());
        let system_end = signed_frame_index(context_mic_end, -estimate.delay_frames)
            .unwrap_or(0)
            .min(self.system.len());
        let correlation = self
            .system
            .get(system_start..system_end)
            .and_then(|system| {
                (estimate.delay_frames - AUDIO_WORD_DELAY_SEARCH_FRAMES
                    ..=estimate.delay_frames + AUDIO_WORD_DELAY_SEARCH_FRAMES)
                    .filter_map(|delay| {
                        envelope_correlation(system, system_start, &self.mic, 0, delay, threshold)
                    })
                    .max_by(f32::total_cmp)
            });

        AudioWordEvidence {
            excess_ln,
            correlation,
            delay_frames: Some(estimate.delay_frames),
        }
    }
}

fn estimate_from_points<'a>(
    points: impl Iterator<Item = &'a AudioReferencePoint>,
) -> Option<AudioEstimate> {
    let points = points.copied().collect::<Vec<_>>();
    let mut delays = points
        .iter()
        .map(|point| point.delay_frames)
        .collect::<Vec<_>>();
    let mut gains = points.iter().map(|point| point.gain_ln).collect::<Vec<_>>();
    delays.sort_unstable();
    Some(AudioEstimate {
        delay_frames: *delays.get(delays.len() / 2)?,
        gain_ln: median_f32(&mut gains)?,
    })
}

fn padded_interval(word: &TimedWord, minimum_ms: u64) -> (u64, u64) {
    let padding_ms = minimum_ms.saturating_sub(word.end_ms.saturating_sub(word.start_ms));
    (
        word.start_ms.saturating_sub(padding_ms / 2),
        word.end_ms
            .saturating_add(padding_ms.saturating_sub(padding_ms / 2)),
    )
}

fn signed_frame_index(index: usize, offset: i32) -> Option<usize> {
    i64::try_from(index)
        .ok()?
        .checked_add(i64::from(offset))?
        .try_into()
        .ok()
}

#[cfg(test)]
fn rms_envelope(samples: &[f32]) -> Vec<f32> {
    samples
        .as_chunks::<AUDIO_FRAME_SAMPLES>()
        .0
        .iter()
        .map(|frame| {
            let mean_square = frame
                .iter()
                .map(|sample| {
                    if sample.is_finite() {
                        sample * sample
                    } else {
                        0.0
                    }
                })
                .sum::<f32>()
                / frame.len() as f32;
            mean_square.sqrt()
        })
        .collect()
}

fn read_rms_envelope(path: &Path) -> io::Result<Vec<f32>> {
    if archive::is_archived(path) {
        let mut envelope = Vec::new();
        archive::decode_each(path, AUDIO_FRAME_SAMPLES, |frame| {
            if frame.len() == AUDIO_FRAME_SAMPLES {
                let mean_square = frame.iter().map(|sample| sample * sample).sum::<f32>()
                    / AUDIO_FRAME_SAMPLES as f32;
                envelope.push(mean_square.sqrt());
            }
        })?;
        return Ok(envelope);
    }
    let mut reader = BufReader::new(File::open(path)?);
    let frame_bytes = AUDIO_FRAME_SAMPLES * 4;
    let mut bytes = vec![0u8; frame_bytes];
    let mut envelope = Vec::new();
    loop {
        match reader.read_exact(&mut bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let mean_square = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|sample| {
                let sample = f32::from_le_bytes(*sample);
                if sample.is_finite() {
                    sample * sample
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            / AUDIO_FRAME_SAMPLES as f32;
        envelope.push(mean_square.sqrt());
    }
    Ok(envelope)
}

fn activity_threshold(system: &[f32]) -> Option<f32> {
    let mut levels = system.to_vec();
    levels.sort_unstable_by(f32::total_cmp);
    let rank = levels.len().saturating_mul(95).div_ceil(100);
    let percentile = *levels.get(rank.saturating_sub(1))?;
    (percentile > 1e-5).then_some((0.1 * percentile).max(1e-4))
}

fn calibrate_audio(mic: &[f32], system: &[f32], threshold: f32) -> Vec<AudioReferencePoint> {
    if system.len() < CALIBRATION_WINDOW_FRAMES {
        return Vec::new();
    }
    let minimum_active =
        (CALIBRATION_WINDOW_FRAMES as f32 * CALIBRATION_MIN_ACTIVE_FRACTION) as usize;
    let mut active_prefix = Vec::with_capacity(system.len() + 1);
    active_prefix.push(0usize);
    for &level in system {
        active_prefix
            .push(active_prefix.last().copied().unwrap_or(0) + usize::from(level >= threshold));
    }
    let system_log = system
        .iter()
        .map(|level| (level + 1e-6).ln())
        .collect::<Vec<_>>();
    let mic_log = mic
        .iter()
        .map(|level| (level + 1e-6).ln())
        .collect::<Vec<_>>();
    let candidates = (0..=system.len().saturating_sub(CALIBRATION_WINDOW_FRAMES))
        .step_by(CALIBRATION_WINDOW_FRAMES)
        .filter(|&start| {
            active_prefix[start + CALIBRATION_WINDOW_FRAMES] - active_prefix[start]
                >= minimum_active
        })
        .filter_map(|start| {
            let active_frames = (start..start + CALIBRATION_WINDOW_FRAMES)
                .filter(|&index| system[index] >= threshold)
                .collect::<Vec<_>>();
            let (correlation, delay) = (MIN_DELAY_FRAMES..=MAX_DELAY_FRAMES)
                .filter_map(|delay| {
                    calibration_correlation(&active_frames, &system_log, start, &mic_log, delay)
                        .map(|correlation| (correlation, delay))
                })
                .max_by(|left, right| left.0.total_cmp(&right.0))?;
            if correlation < CALIBRATION_MIN_CORRELATION {
                return None;
            }
            let gain_ln = mean_gain_ln(&active_frames, &system_log, start, &mic_log, delay)?;
            Some(AudioReferencePoint {
                center_frame: start + CALIBRATION_WINDOW_FRAMES / 2,
                delay_frames: delay,
                gain_ln,
            })
        })
        .collect::<Vec<_>>();
    let horizon_frames = usize::try_from(REFERENCE_HORIZON_MS / AUDIO_FRAME_MS)
        .expect("reference horizon fits usize");
    candidates
        .iter()
        .copied()
        .filter(|candidate| {
            let first = candidates.partition_point(|point| {
                point.center_frame < candidate.center_frame.saturating_sub(horizon_frames)
            });
            let last = candidates.partition_point(|point| {
                point.center_frame <= candidate.center_frame.saturating_add(horizon_frames)
            });
            candidates[first..last]
                .iter()
                .filter(|other| {
                    other.center_frame != candidate.center_frame
                        && (other.delay_frames - candidate.delay_frames).abs()
                            <= CALIBRATION_DELAY_TOLERANCE_FRAMES
                })
                .take(2)
                .count()
                >= 2
        })
        .collect()
}

fn mean_gain_ln(
    active_frames: &[usize],
    system_log: &[f32],
    system_start: usize,
    mic_log: &[f32],
    delay: i32,
) -> Option<f32> {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    let (system_range, mic_start) = calibration_aligned_range(system_start, mic_log.len(), delay)?;
    let first = active_frames.partition_point(|&index| index < system_range.start);
    let last = active_frames.partition_point(|&index| index < system_range.end);
    for &system_index in &active_frames[first..last] {
        let mic_index = mic_start + system_index - system_range.start;
        sum += mic_log[mic_index] - system_log[system_index];
        count += 1;
    }
    (count > 0).then_some(sum / count as f32)
}

fn calibration_correlation(
    active_frames: &[usize],
    system_log: &[f32],
    system_start: usize,
    mic_log: &[f32],
    delay: i32,
) -> Option<f32> {
    let mut count = 0usize;
    let mut system_sum = 0.0f32;
    let mut mic_sum = 0.0f32;
    let mut system_square_sum = 0.0f32;
    let mut mic_square_sum = 0.0f32;
    let mut product_sum = 0.0f32;
    let (system_range, mic_start) = calibration_aligned_range(system_start, mic_log.len(), delay)?;
    let first = active_frames.partition_point(|&index| index < system_range.start);
    let last = active_frames.partition_point(|&index| index < system_range.end);
    for &system_index in &active_frames[first..last] {
        let mic_index = mic_start + system_index - system_range.start;
        let system_value = system_log[system_index];
        let mic_value = mic_log[mic_index];
        count += 1;
        system_sum += system_value;
        mic_sum += mic_value;
        system_square_sum += system_value * system_value;
        mic_square_sum += mic_value * mic_value;
        product_sum += system_value * mic_value;
    }
    if count < 20 {
        return None;
    }
    let count = count as f32;
    let covariance = product_sum - system_sum * mic_sum / count;
    let system_variance = system_square_sum - system_sum * system_sum / count;
    let mic_variance = mic_square_sum - mic_sum * mic_sum / count;
    let denominator = (system_variance * mic_variance).sqrt();
    (denominator > 1e-6).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn calibration_aligned_range(
    system_start: usize,
    mic_len: usize,
    delay: i32,
) -> Option<(std::ops::Range<usize>, usize)> {
    let system_end = system_start.saturating_add(CALIBRATION_WINDOW_FRAMES);
    if delay >= 0 {
        let delay = delay as usize;
        let end = system_end.min(mic_len.saturating_sub(delay));
        (system_start < end).then_some((system_start..end, system_start + delay))
    } else {
        let delay = delay.unsigned_abs() as usize;
        let start = system_start.max(delay);
        let end = system_end.min(mic_len.saturating_add(delay));
        (start < end).then_some((start..end, start - delay))
    }
}

fn median_f32(values: &mut [f32]) -> Option<f32> {
    values.sort_unstable_by(f32::total_cmp);
    values.get(values.len() / 2).copied()
}

fn envelope_correlation(
    system: &[f32],
    system_start: usize,
    mic: &[f32],
    mic_start: usize,
    delay: i32,
    activity_threshold: f32,
) -> Option<f32> {
    let pairs = system
        .iter()
        .enumerate()
        .filter_map(|(local_system_index, &system_value)| {
            let system_index = i64::try_from(system_start + local_system_index).ok()?;
            let mic_index = system_index + i64::from(delay) - i64::try_from(mic_start).ok()?;
            let mic_value = *mic.get(usize::try_from(mic_index).ok()?)?;
            (system_value >= activity_threshold)
                .then_some(((system_value + 1e-6).ln(), (mic_value + 1e-6).ln()))
        })
        .collect::<Vec<_>>();
    if pairs.len() < 20 {
        return None;
    }
    let count = pairs.len() as f32;
    let system_mean = pairs.iter().map(|(system, _)| system).sum::<f32>() / count;
    let mic_mean = pairs.iter().map(|(_, mic)| mic).sum::<f32>() / count;
    let mut covariance = 0.0;
    let mut system_variance = 0.0;
    let mut mic_variance = 0.0;
    for (system, mic) in pairs {
        let system = system - system_mean;
        let mic = mic - mic_mean;
        covariance += system * mic;
        system_variance += system * system;
        mic_variance += mic * mic;
    }
    let denominator = (system_variance * mic_variance).sqrt();
    (denominator > 1e-6).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn word(source: AudioSource, start_ms: u64, text: &str) -> TimedWord {
        TimedWord {
            source,
            start_ms,
            end_ms: start_ms + 300,
            text: text.into(),
        }
    }

    fn words(text: &[&str], source: AudioSource, start_ms: u64) -> Vec<TimedWord> {
        text.iter()
            .enumerate()
            .map(|(index, text)| word(source, start_ms + index as u64 * 350, text))
            .collect()
    }

    fn duplicate_words(system_start: u64, mic_start: u64) -> Vec<TimedWord> {
        let mut input = words(
            &["we", "should", "release", "Friday"],
            AudioSource::System,
            system_start,
        );
        input.extend(words(
            &["we", "should", "release", "Friday"],
            AudioSource::Mic,
            mic_start,
        ));
        input
    }

    #[test]
    fn detects_clear_text_duplicate_without_audio() {
        let mut input = words(
            &["We", "should", "release", "Friday."],
            AudioSource::System,
            1_000,
        );
        input.extend(words(
            &["we", "should", "releaze", "friday"],
            AudioSource::Mic,
            1_120,
        ));
        let result = detect_leaked_mic_words(&input, None);
        assert_eq!(
            result.echo,
            [false, false, false, false, true, true, true, true]
        );
        assert_eq!(result.detections[0].evidence, "text");
    }

    #[test]
    fn preserves_unmatched_local_words_during_double_talk() {
        let mut input = words(
            &["we", "should", "release", "Friday"],
            AudioSource::System,
            1_000,
        );
        input.extend(words(
            &["yes", "we", "should", "release", "Friday"],
            AudioSource::Mic,
            770,
        ));
        let result = detect_leaked_mic_words(&input, None);
        assert_eq!(
            result.echo,
            [false, false, false, false, false, true, true, true, true]
        );
    }

    #[test]
    fn preserves_unrelated_simultaneous_speech() {
        let mut input = words(
            &["we", "should", "release", "Friday"],
            AudioSource::System,
            1_000,
        );
        input.extend(words(
            &["I", "have", "another", "question"],
            AudioSource::Mic,
            1_050,
        ));
        let result = detect_leaked_mic_words(&input, None);
        assert!(result.echo.iter().all(|echo| !echo));
        assert!(result.detections.is_empty());
    }

    #[test]
    fn short_duplicate_needs_audio_evidence() {
        let mut input = words(&["thank", "you"], AudioSource::System, 1_000);
        input.extend(words(&["Thank", "you!"], AudioSource::Mic, 1_120));
        assert!(
            detect_leaked_mic_words(&input, None)
                .echo
                .iter()
                .all(|echo| !echo)
        );

        let (system, mic) = correlated_audio(4_000, 120);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let result = detect_leaked_mic_words(&input, Some(&audio));
        assert_eq!(result.echo, [false, false, true, true]);
        assert_eq!(result.detections[0].evidence, "text_audio");
        assert!(result.detections[0].audio_similarity.unwrap() > 0.9);
        assert!((result.detections[0].audio_delay_ms.unwrap() - 120).abs() <= 10);
    }

    #[test]
    fn audio_detects_garbled_leak_without_text_matches() {
        let mut input = words(
            &["we", "should", "release", "Friday"],
            AudioSource::System,
            1_000,
        );
        input.extend(words(
            &["he", "shirt", "really", "fried day"],
            AudioSource::Mic,
            1_120,
        ));
        let (system, mic) = calibrated_audio(8_000, 120);
        let audio = AudioEnvelopes::from_samples(&mic, &system);

        let result = detect_leaked_mic_words(&input, Some(&audio));

        assert_eq!(
            result.echo,
            [false, false, false, false, true, true, true, true]
        );
        assert_eq!(result.detections.len(), 1);
        assert_eq!(result.detections[0].evidence, "audio");
        assert_eq!(result.detections[0].words, 4);
        assert_eq!(result.detections[0].text_similarity, None);
        let serialized = serde_json::to_value(&result.detections[0]).expect("serialize");
        assert!(serialized.get("text_similarity").is_none());
    }

    #[test]
    fn audio_keeps_local_speech_while_system_is_silent() {
        let (mut system, mut mic) = calibrated_audio(8_000, 120);
        clear_audio_range(&mut system, 6_500, 8_000);
        clear_audio_range(&mut mic, 6_620, 8_000);
        add_local_signal(&mut mic, 7_120, 7_770, 1.0);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        assert!(!audio.reference_points.is_empty());
        let input = words(&["local", "speech"], AudioSource::Mic, 7_120);

        let result = detect_leaked_mic_words(&input, Some(&audio));

        assert!(result.echo.iter().all(|echo| !echo));
        assert!(result.detections.is_empty());
    }

    #[test]
    fn audio_keeps_double_talk_and_detects_leak_only_words() {
        let (system, mut mic) = calibrated_audio(8_000, 120);
        add_local_signal(&mut mic, 5_120, 5_770, 1.5);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        assert!(!audio.reference_points.is_empty());
        let mut input = words(
            &["remote", "audio", "leak", "only"],
            AudioSource::System,
            1_000,
        );
        input.extend(words(
            &["wrong", "words", "from", "mic"],
            AudioSource::Mic,
            1_120,
        ));
        input.extend(words(&["my", "question"], AudioSource::Mic, 5_120));

        let result = detect_leaked_mic_words(&input, Some(&audio));

        assert_eq!(
            result.echo,
            [
                false, false, false, false, true, true, true, true, false, false
            ]
        );
        assert_eq!(result.detections.len(), 1);
        assert_eq!(result.detections[0].evidence, "audio");
        assert_eq!(result.detections[0].words, 4);
    }

    #[test]
    fn uncorrelated_audio_preserves_matching_speech() {
        let input = duplicate_words(1_000, 1_000);
        let (system, _) = calibrated_audio(8_000, 120);
        let mic = synthetic_audio(8_000, |frame| {
            if frame % 50 < 40 {
                0.25 + ((frame * 17 + 11) % 29) as f32 / 38.0
            } else {
                0.0
            }
        });
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        assert!(audio.reference_points.is_empty());
        let result = detect_leaked_mic_words(&input, Some(&audio));
        assert!(result.echo.iter().all(|echo| !echo));
        assert!(result.detections.is_empty());
    }

    #[test]
    fn preserves_a_later_repetition_without_audio() {
        let input = duplicate_words(1_000, 2_200);
        let result = detect_leaked_mic_words(&input, None);
        assert!(result.echo.iter().all(|echo| !echo));
        assert!(result.detections.is_empty());
    }

    #[test]
    fn audio_delay_must_agree_with_word_timing() {
        let input = duplicate_words(1_000, 2_200);
        let (system, mic) = correlated_audio(4_000, 120);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let result = detect_leaked_mic_words(&input, Some(&audio));
        assert!(result.echo.iter().all(|echo| !echo));
        assert!(result.detections.is_empty());
    }

    #[test]
    fn one_timing_outlier_prevents_detection() {
        let mut input = words(&["one", "two", "three", "four"], AudioSource::System, 1_000);
        input.extend(words(&["one", "two", "three"], AudioSource::Mic, 1_120));
        for (start, text) in [
            (2_400, "local"),
            (3_000, "speech"),
            (3_600, "keeps"),
            (4_200, "going"),
            (4_800, "until"),
        ] {
            input.push(word(AudioSource::Mic, start, text));
        }
        input.push(word(AudioSource::Mic, 5_050, "four"));
        let result = detect_leaked_mic_words(&input, None);
        assert!(result.echo.iter().all(|echo| !echo));
        assert!(result.detections.is_empty());
    }

    #[test]
    fn follows_gain_changes_in_both_directions() {
        for (before_gain, after_gain) in [(0.2, 0.6), (0.6, 0.2)] {
            let (system, mut mic) = variable_gain_audio(24_000, 120, |frame| {
                if frame < 1_200 {
                    before_gain
                } else {
                    after_gain
                }
            });
            add_local_signal(&mut mic, 7_120, 7_770, 1.5);
            add_local_signal(&mut mic, 19_120, 19_770, 1.5);
            let audio = AudioEnvelopes::from_samples(&mic, &system);
            let input = [
                word(AudioSource::Mic, 5_120, "early leak"),
                word(AudioSource::Mic, 7_120, "early local"),
                word(AudioSource::Mic, 17_120, "late leak"),
                word(AudioSource::Mic, 19_120, "late local"),
            ];

            let result = detect_leaked_mic_words(&input, Some(&audio));

            assert_eq!(result.echo, [true, false, true, false]);
        }
    }

    #[test]
    fn leak_stopping_halfway_does_not_mark_later_microphone_words() {
        let (system, mut mic) = calibrated_audio(24_000, 120);
        clear_audio_range(&mut mic, 12_120, 24_000);
        add_local_signal(&mut mic, 17_120, 17_770, 1.0);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let input = [
            word(AudioSource::Mic, 5_120, "first-half leak"),
            word(AudioSource::Mic, 17_120, "headphones local"),
        ];

        let result = detect_leaked_mic_words(&input, Some(&audio));

        assert_eq!(result.echo, [true, false]);
    }

    #[test]
    fn isolated_disagreeing_candidate_is_not_a_reference_point() {
        let (system, mut mic) = calibrated_audio(14_000, 120);
        clear_audio_range(&mut mic, 6_600, 8_600);
        copy_leak_range(&system, &mut mic, 6_000, 8_000, 600, 0.2);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let threshold = audio.activity_threshold.expect("active system audio");
        let system_log = audio
            .system
            .iter()
            .map(|level| (level + 1e-6).ln())
            .collect::<Vec<_>>();
        let mic_log = audio
            .mic
            .iter()
            .map(|level| (level + 1e-6).ln())
            .collect::<Vec<_>>();
        let active_frames = (600..800)
            .filter(|&index| audio.system[index] >= threshold)
            .collect::<Vec<_>>();
        let (_, delay) = (MIN_DELAY_FRAMES..=MAX_DELAY_FRAMES)
            .filter_map(|delay| {
                calibration_correlation(&active_frames, &system_log, 600, &mic_log, delay)
                    .map(|correlation| (correlation, delay))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .expect("isolated candidate");

        assert!((delay - 60).abs() <= CALIBRATION_DELAY_TOLERANCE_FRAMES);
        assert!(
            !audio
                .reference_points
                .iter()
                .any(|point| point.center_frame == 700)
        );
        assert!(audio.reference_points.len() >= 3);
    }

    #[test]
    fn file_audio_reader_handles_bounded_and_partial_input() {
        let root = std::env::temp_dir().join(format!(
            "singstone-leakage-audio-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create fixture");
        let (system, mic) = correlated_audio(4_000, 120);
        let system_path = root.join("system.f32le");
        let mic_path = root.join("mic.f32le");
        write_audio_with_partial_tail(&system_path, &system);
        write_audio_with_partial_tail(&mic_path, &mic);
        let audio = AudioEnvelopes::read(&mic_path, &system_path).expect("open audio");
        let mut input = words(&["thank", "you"], AudioSource::System, 1_000);
        input.extend(words(&["thank", "you"], AudioSource::Mic, 1_120));
        let result = detect_leaked_mic_words(&input, Some(&audio));
        assert_eq!(result.echo, [false, false, true, true]);
        assert!((result.detections[0].audio_delay_ms.unwrap() - 120).abs() <= 10);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    fn write_audio_with_partial_tail(path: &Path, samples: &[f32]) {
        let mut bytes = Vec::with_capacity(samples.len() * 4 + 2);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes.extend_from_slice(&[1, 2]);
        fs::write(path, bytes).expect("write audio");
    }

    fn correlated_audio(duration_ms: usize, delay_ms: usize) -> (Vec<f32>, Vec<f32>) {
        let system = synthetic_audio(duration_ms, |frame| {
            0.1 + ((frame * 7 + frame * frame * 3) % 31) as f32 / 31.0
        });
        let delay = delay_ms * SAMPLE_RATE as usize / 1_000;
        let mut mic = vec![0.0; system.len()];
        for (index, sample) in system.iter().copied().enumerate() {
            if index + delay < mic.len() {
                mic[index + delay] = sample * 0.2;
            }
        }
        (system, mic)
    }

    fn calibrated_audio(duration_ms: usize, delay_ms: usize) -> (Vec<f32>, Vec<f32>) {
        let system = synthetic_audio(duration_ms, |frame| {
            if frame % 50 < 40 {
                0.25 + ((frame * 7 + frame * frame * 3) % 31) as f32 / 41.0
            } else {
                0.0
            }
        });
        let delay = delay_ms * SAMPLE_RATE as usize / 1_000;
        let mut mic = vec![0.0; system.len()];
        for (index, sample) in system.iter().copied().enumerate() {
            if index + delay < mic.len() {
                mic[index + delay] = sample * 0.2;
            }
        }
        (system, mic)
    }

    fn variable_gain_audio(
        duration_ms: usize,
        delay_ms: usize,
        gain: impl Fn(usize) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let (system, _) = calibrated_audio(duration_ms, delay_ms);
        let delay = delay_ms * SAMPLE_RATE as usize / 1_000;
        let mut mic = vec![0.0; system.len()];
        for (index, sample) in system.iter().copied().enumerate() {
            if index + delay < mic.len() {
                mic[index + delay] = sample * gain(index / AUDIO_FRAME_SAMPLES);
            }
        }
        (system, mic)
    }

    fn copy_leak_range(
        system: &[f32],
        mic: &mut [f32],
        start_ms: usize,
        end_ms: usize,
        delay_ms: usize,
        gain: f32,
    ) {
        let start = start_ms * SAMPLE_RATE as usize / 1_000;
        let end = (end_ms * SAMPLE_RATE as usize / 1_000).min(system.len());
        let delay = delay_ms * SAMPLE_RATE as usize / 1_000;
        for index in start..end {
            if index + delay < mic.len() {
                mic[index + delay] = system[index] * gain;
            }
        }
    }

    fn clear_audio_range(audio: &mut [f32], start_ms: usize, end_ms: usize) {
        let start = start_ms * SAMPLE_RATE as usize / 1_000;
        let end = (end_ms * SAMPLE_RATE as usize / 1_000).min(audio.len());
        audio[start.min(end)..end].fill(0.0);
    }

    fn add_local_signal(audio: &mut [f32], start_ms: usize, end_ms: usize, amplitude: f32) {
        let start = start_ms * SAMPLE_RATE as usize / 1_000;
        let end = (end_ms * SAMPLE_RATE as usize / 1_000).min(audio.len());
        for (index, sample) in audio[start.min(end)..end].iter_mut().enumerate() {
            *sample += (index as f32 * 0.113).sin() * amplitude;
        }
    }

    fn synthetic_audio(duration_ms: usize, amplitude: impl Fn(usize) -> f32) -> Vec<f32> {
        let samples = duration_ms * SAMPLE_RATE as usize / 1_000;
        (0..samples)
            .map(|index| {
                let frame = index / AUDIO_FRAME_SAMPLES;
                let phase = index as f32 * 0.071;
                phase.sin() * amplitude(frame)
            })
            .collect()
    }
}
