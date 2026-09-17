use super::Transcriber;
use super::backend;
use super::vad::{self, SpeechRegion, VadConfig};
use crate::types::{AudioSource, SAMPLE_RATE, TimedWord, samples_to_ms};
use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

pub struct WhisperTranscriber {
    context: WhisperContext,
    state: WhisperState,
    source: AudioSource,
    language: String,
    threads: usize,
    vad: VadConfig,
}

impl WhisperTranscriber {
    pub fn new(
        model: &Path,
        source: AudioSource,
        language: String,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        whisper_rs::install_logging_hooks();
        eprintln!(
            "transcribe: Whisper backend: {}",
            backend::current().description
        );
        let context = WhisperContext::new_with_params(model, WhisperContextParameters::default())?;
        let state = context.create_state()?;
        Ok(Self {
            context,
            state,
            source,
            language,
            threads,
            vad: VadConfig::default(),
        })
    }

    pub fn set_source(&mut self, source: AudioSource) {
        self.source = source;
    }

    fn params<'a>(language: &'a str, threads: usize) -> FullParams<'a, 'static> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(threads.try_into().unwrap_or(i32::MAX));
        // A null language detects and then transcribes; detect_language stops after detection.
        params.set_language((language != "auto").then_some(language));
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_nst(true);
        params.set_single_segment(false);
        params.set_no_context(true);
        params
    }
}

impl Transcriber for WhisperTranscriber {
    fn transcribe(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<TimedWord>, Box<dyn std::error::Error>> {
        let mut regions = vad::detect(samples, self.vad);
        pad_short_regions(&mut regions, samples.len());
        let speech_samples = regions
            .iter()
            .map(|region| region.end - region.start)
            .sum::<usize>();
        eprintln!(
            "transcribe {}: {} speech regions, {:.1} s speech",
            self.source,
            regions.len(),
            speech_samples as f64 / SAMPLE_RATE as f64
        );
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let mut words = Vec::new();
        let chunk_samples = 30 * SAMPLE_RATE as usize;
        for region in &regions {
            for (chunk_number, chunk) in samples[region.start..region.end]
                .chunks(chunk_samples)
                .enumerate()
            {
                let mut padded = Vec::new();
                let audio = if chunk.len() < SAMPLE_RATE as usize {
                    padded.extend_from_slice(chunk);
                    padded.resize(SAMPLE_RATE as usize, 0.0);
                    padded.as_slice()
                } else {
                    chunk
                };
                let params = Self::params(&self.language, self.threads);
                self.state.full(params, audio)?;
                let offset_samples = region.start + chunk_number * chunk_samples;
                let offset_ms = samples_to_ms(offset_samples as u64);
                let chunk_end_ms = offset_ms + samples_to_ms(chunk.len() as u64);
                let mut chunk_words = Vec::new();
                for segment in self.state.as_iter() {
                    let segment_text = segment.to_str_lossy()?.trim().to_string();
                    if segment_text.is_empty() || is_annotation(&segment_text) {
                        continue;
                    }
                    let segment_start = segment.start_timestamp().max(0) as u64 * 10;
                    let segment_end = segment.end_timestamp().max(0) as u64 * 10;
                    let mut current: Option<TimedWord> = None;
                    for index in 0..segment.n_tokens() {
                        let Some(token) = segment.get_token(index) else {
                            continue;
                        };
                        if token.token_id() >= self.context.token_eot() {
                            continue;
                        }
                        let raw = token.to_str_lossy()?;
                        let token_text = raw.trim();
                        if token_text.is_empty() {
                            continue;
                        }
                        let starts_word = raw.chars().next().is_some_and(char::is_whitespace);
                        if starts_word && let Some(word) = current.take() {
                            push_word(&mut chunk_words, word);
                        }
                        let data = token.token_data();
                        let start = if data.t0 >= 0 {
                            data.t0 as u64 * 10
                        } else {
                            segment_start
                        };
                        let end = if data.t1 >= 0 {
                            data.t1 as u64 * 10
                        } else {
                            segment_end
                        };
                        if let Some(word) = current.as_mut() {
                            word.text.push_str(token_text);
                            word.end_ms = end.saturating_add(offset_ms);
                        } else {
                            current = Some(TimedWord {
                                source: self.source,
                                start_ms: start.saturating_add(offset_ms),
                                end_ms: end.saturating_add(offset_ms),
                                text: token_text.to_string(),
                            });
                        }
                    }
                    if let Some(word) = current {
                        push_word(&mut chunk_words, word);
                    }
                }
                normalize_chunk_words(&mut chunk_words, offset_ms, chunk_end_ms);
                words.extend(chunk_words);
            }
        }
        words.sort_by_key(|word| (word.start_ms, word.end_ms));
        words.retain(|word| {
            !word.text.trim().is_empty()
                && word.end_ms >= word.start_ms
                && regions.iter().any(|region| {
                    let start_ms = samples_to_ms(region.start as u64);
                    let end_ms = samples_to_ms(region.end as u64);
                    word.end_ms.min(end_ms) >= word.start_ms.max(start_ms)
                })
        });
        Ok(words)
    }
}

fn normalize_chunk_words(words: &mut [TimedWord], start_ms: u64, end_ms: u64) {
    for word in words.iter_mut() {
        word.start_ms = word.start_ms.clamp(start_ms, end_ms);
        word.end_ms = word.end_ms.clamp(start_ms, end_ms);
    }
    words.sort_by_key(|word| (word.start_ms, word.end_ms));
    let mut previous_end = start_ms;
    for word in words {
        word.start_ms = word.start_ms.max(previous_end).min(end_ms);
        word.end_ms = word.end_ms.max(word.start_ms).min(end_ms);
        previous_end = word.end_ms;
    }
}

fn pad_short_regions(regions: &mut Vec<SpeechRegion>, audio_len: usize) {
    for region in regions.iter_mut() {
        let missing = (SAMPLE_RATE as usize).saturating_sub(region.end - region.start);
        let before = (missing / 2).min(region.start);
        region.start -= before;
        region.end = (region.end + missing - before).min(audio_len);
        if region.end - region.start < SAMPLE_RATE as usize {
            region.start = region.end.saturating_sub(SAMPLE_RATE as usize);
        }
    }
    regions.sort_by_key(|region| (region.start, region.end));
    let mut merged: Vec<SpeechRegion> = Vec::with_capacity(regions.len());
    for region in regions.drain(..) {
        if let Some(previous) = merged.last_mut()
            && region.start <= previous.end
        {
            previous.end = previous.end.max(region.end);
        } else {
            merged.push(region);
        }
    }
    *regions = merged;
}

fn push_word(words: &mut Vec<TimedWord>, word: TimedWord) {
    if !word.text.trim().is_empty() && !is_annotation(&word.text) {
        words.push(word);
    }
}

fn is_annotation(text: &str) -> bool {
    let text = text.trim();
    (text.starts_with('[') && text.ends_with(']'))
        || (text.starts_with('(') && text.ends_with(')'))
        || (!text.is_empty()
            && text
                .chars()
                .all(|character| character.is_whitespace() || matches!(character, '♪' | '♫')))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_bracketed_annotations() {
        assert!(is_annotation("[BLANK_AUDIO]"));
        assert!(is_annotation(" (music) "));
        assert!(is_annotation("♪"));
        assert!(is_annotation(" ♫ ♪ "));
        assert!(!is_annotation("[aside] hello"));
    }

    #[test]
    fn chunk_bounds_do_not_constrain_following_region() {
        let mut first = vec![TimedWord {
            source: AudioSource::System,
            start_ms: 1_100,
            end_ms: 29_500,
            text: "hallucination".into(),
        }];
        normalize_chunk_words(&mut first, 1_000, 2_500);
        assert_eq!((first[0].start_ms, first[0].end_ms), (1_100, 2_500));

        let mut following = vec![TimedWord {
            source: AudioSource::System,
            start_ms: 10_100,
            end_ms: 10_300,
            text: "next".into(),
        }];
        normalize_chunk_words(&mut following, 10_000, 11_500);
        assert_eq!(
            (following[0].start_ms, following[0].end_ms),
            (10_100, 10_300)
        );
    }

    #[test]
    fn padding_remerges_adjacent_regions() {
        let mut regions = vec![
            SpeechRegion {
                start: 0,
                end: SAMPLE_RATE as usize / 2,
            },
            SpeechRegion {
                start: SAMPLE_RATE as usize,
                end: SAMPLE_RATE as usize * 3 / 2,
            },
        ];
        pad_short_regions(&mut regions, SAMPLE_RATE as usize * 2);
        assert_eq!(
            regions,
            vec![SpeechRegion {
                start: 0,
                end: 28_000
            }]
        );
    }

    #[test]
    fn whitespace_only_words_are_not_emitted() {
        let mut words = Vec::new();
        push_word(
            &mut words,
            TimedWord {
                source: AudioSource::Mic,
                start_ms: 0,
                end_ms: 0,
                text: " \t ".into(),
            },
        );
        assert!(words.is_empty());
    }
}
