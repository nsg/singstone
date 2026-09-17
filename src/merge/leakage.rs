use crate::types::{AudioSource, SAMPLE_RATE, TimedWord};
use serde::Serialize;
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LeakageSuppression {
    pub mic_start_ms: u64,
    pub mic_end_ms: u64,
    pub system_start_ms: u64,
    pub system_end_ms: u64,
    pub suppressed_words: usize,
    pub text_similarity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_delay_ms: Option<i64>,
    pub evidence: &'static str,
}

#[derive(Debug)]
pub struct SuppressionResult {
    pub words: Vec<TimedWord>,
    pub suppressions: Vec<LeakageSuppression>,
}

pub fn suppress_leaked_mic_words(
    words: &[TimedWord],
    audio: Option<&AudioEnvelopes>,
) -> SuppressionResult {
    let mic_phrases = mic_phrases(words);
    let system_tokens = source_tokens(words, AudioSource::System);
    let mut maximum_system_ends = Vec::with_capacity(system_tokens.len());
    let mut maximum_end = 0;
    for (index, _) in &system_tokens {
        maximum_end = maximum_end.max(words[*index].end_ms);
        maximum_system_ends.push(maximum_end);
    }
    let mut suppressed = vec![false; words.len()];
    let mut suppressions = Vec::new();

    for phrase in mic_phrases {
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
            suppressed[*mic] = true;
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
        suppressions.push(LeakageSuppression {
            mic_start_ms: mic_match_start,
            mic_end_ms: mic_match_end,
            system_start_ms: matched_system_start,
            system_end_ms: matched_system_end,
            suppressed_words: matches.len(),
            text_similarity: similarity,
            audio_similarity: audio_match.map(|(correlation, _)| correlation),
            audio_delay_ms: audio_match.map(|(_, delay)| delay),
            evidence: if audio_supported {
                "text_audio"
            } else {
                "text"
            },
        });
    }

    let words = words
        .iter()
        .enumerate()
        .filter(|(index, _)| !suppressed[*index])
        .map(|(_, word)| word.clone())
        .collect();
    SuppressionResult {
        words,
        suppressions,
    }
}

fn mic_phrases(words: &[TimedWord]) -> Vec<Vec<usize>> {
    let mut indices = words
        .iter()
        .enumerate()
        .filter_map(|(index, word)| (word.source == AudioSource::Mic).then_some(index))
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

enum AudioStorage {
    Files {
        mic: RefCell<BufReader<File>>,
        system: RefCell<BufReader<File>>,
    },
    #[cfg(test)]
    Memory { mic: Vec<f32>, system: Vec<f32> },
}

pub struct AudioEnvelopes {
    storage: AudioStorage,
}

impl AudioEnvelopes {
    pub fn read(mic_path: &Path, system_path: &Path) -> io::Result<Self> {
        Ok(Self {
            storage: AudioStorage::Files {
                mic: RefCell::new(BufReader::new(File::open(mic_path)?)),
                system: RefCell::new(BufReader::new(File::open(system_path)?)),
            },
        })
    }

    #[cfg(test)]
    fn from_samples(mic: &[f32], system: &[f32]) -> Self {
        Self {
            storage: AudioStorage::Memory {
                mic: rms_envelope(mic),
                system: rms_envelope(system),
            },
        }
    }

    fn similarity(&self, start_ms: u64, end_ms: u64) -> Option<(f32, i64)> {
        let system_start = usize::try_from(start_ms.saturating_sub(200) / AUDIO_FRAME_MS).ok()?;
        let system_end = usize::try_from(end_ms.saturating_add(200) / AUDIO_FRAME_MS).ok()?;
        if system_end.saturating_sub(system_start) < 30 {
            return None;
        }
        let mic_start = system_start.saturating_sub(MIN_DELAY_FRAMES.unsigned_abs() as usize);
        let mic_end = system_end.saturating_add(MAX_DELAY_FRAMES as usize);
        let (system, mic) = match &self.storage {
            AudioStorage::Files { mic, system } => {
                let system =
                    read_rms_envelope_range(&mut *system.borrow_mut(), system_start, system_end)
                        .ok()?;
                let mic =
                    read_rms_envelope_range(&mut *mic.borrow_mut(), mic_start, mic_end).ok()?;
                (system, mic)
            }
            #[cfg(test)]
            AudioStorage::Memory { mic, system } => (
                envelope_range(system, system_start, system_end),
                envelope_range(mic, mic_start, mic_end),
            ),
        };
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
                    &system,
                    system_start,
                    &mic,
                    mic_start,
                    delay,
                    activity_threshold,
                )
                .map(|correlation| (correlation, i64::from(delay) * AUDIO_FRAME_MS as i64))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
    }
}

#[cfg(test)]
fn envelope_range(envelope: &[f32], start: usize, end: usize) -> Vec<f32> {
    envelope
        .get(start.min(envelope.len())..end.min(envelope.len()))
        .unwrap_or_default()
        .to_vec()
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

fn read_rms_envelope_range(
    reader: &mut (impl Read + Seek),
    start: usize,
    end: usize,
) -> io::Result<Vec<f32>> {
    let frame_bytes = AUDIO_FRAME_SAMPLES * 4;
    let offset = start
        .checked_mul(frame_bytes)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "audio offset overflow"))?;
    reader.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; frame_bytes];
    let mut envelope = Vec::with_capacity(end.saturating_sub(start));
    for _ in start..end {
        let mut filled = 0;
        while filled < bytes.len() {
            match reader.read(&mut bytes[filled..])? {
                0 => break,
                count => filled += count,
            }
        }
        if filled < bytes.len() {
            break;
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
    fn suppresses_clear_text_duplicate_without_audio() {
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
        let result = suppress_leaked_mic_words(&input, None);
        assert_eq!(result.words.len(), 4);
        assert!(
            result
                .words
                .iter()
                .all(|word| word.source == AudioSource::System)
        );
        assert_eq!(result.suppressions[0].evidence, "text");
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
        let result = suppress_leaked_mic_words(&input, None);
        let mic = result
            .words
            .iter()
            .filter(|word| word.source == AudioSource::Mic)
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(mic, ["yes"]);
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
        let result = suppress_leaked_mic_words(&input, None);
        assert_eq!(result.words.len(), input.len());
        assert!(result.suppressions.is_empty());
    }

    #[test]
    fn short_duplicate_needs_audio_evidence() {
        let mut input = words(&["thank", "you"], AudioSource::System, 1_000);
        input.extend(words(&["Thank", "you!"], AudioSource::Mic, 1_120));
        assert_eq!(suppress_leaked_mic_words(&input, None).words.len(), 4);

        let (system, mic) = correlated_audio(4_000, 120);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let result = suppress_leaked_mic_words(&input, Some(&audio));
        assert_eq!(result.words.len(), 2);
        assert_eq!(result.suppressions[0].evidence, "text_audio");
        assert!(result.suppressions[0].audio_similarity.unwrap() > 0.9);
        assert!((result.suppressions[0].audio_delay_ms.unwrap() - 120).abs() <= 10);
    }

    #[test]
    fn uncorrelated_audio_preserves_matching_speech() {
        let input = duplicate_words(1_000, 1_000);
        let (system, _) = correlated_audio(4_000, 120);
        let mic = synthetic_audio(4_000, |frame| 0.1 + ((frame * 17 + 11) % 29) as f32 / 29.0);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let result = suppress_leaked_mic_words(&input, Some(&audio));
        assert_eq!(result.words.len(), input.len());
        assert!(result.suppressions.is_empty());
    }

    #[test]
    fn preserves_a_later_repetition_without_audio() {
        let input = duplicate_words(1_000, 2_200);
        let result = suppress_leaked_mic_words(&input, None);
        assert_eq!(result.words.len(), input.len());
        assert!(result.suppressions.is_empty());
    }

    #[test]
    fn audio_delay_must_agree_with_word_timing() {
        let input = duplicate_words(1_000, 2_200);
        let (system, mic) = correlated_audio(4_000, 120);
        let audio = AudioEnvelopes::from_samples(&mic, &system);
        let result = suppress_leaked_mic_words(&input, Some(&audio));
        assert_eq!(result.words.len(), input.len());
        assert!(result.suppressions.is_empty());
    }

    #[test]
    fn one_timing_outlier_prevents_suppression() {
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
        let result = suppress_leaked_mic_words(&input, None);
        assert_eq!(result.words.len(), input.len());
        assert!(result.suppressions.is_empty());
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
        let result = suppress_leaked_mic_words(&input, Some(&audio));
        assert_eq!(result.words.len(), 2);
        assert!((result.suppressions[0].audio_delay_ms.unwrap() - 120).abs() <= 10);
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
