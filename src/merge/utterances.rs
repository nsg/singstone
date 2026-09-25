use crate::meeting::MeetingDetails;
use crate::types::{AudioSource, EchoEvidence, SpeakerSegment, TimedWord, Utterance};
use std::collections::HashMap;

pub const DEFAULT_NEAREST_TOLERANCE_MS: u64 = 500;

pub fn assign_cluster(
    word: &TimedWord,
    segments: &[SpeakerSegment],
    tolerance_ms: u64,
) -> Option<u32> {
    let same_source = segments
        .iter()
        .filter(|segment| segment.source == word.source);
    let mut best_overlap = None;
    for segment in same_source.clone() {
        let overlap = word
            .end_ms
            .min(segment.end_ms)
            .saturating_sub(word.start_ms.max(segment.start_ms));
        if overlap > 0 && best_overlap.is_none_or(|(best, _)| overlap > best) {
            best_overlap = Some((overlap, segment.cluster));
        }
    }
    if let Some((_, cluster)) = best_overlap {
        return Some(cluster);
    }
    same_source
        .filter_map(|segment| {
            let distance = if word.end_ms <= segment.start_ms {
                segment.start_ms - word.end_ms
            } else if segment.end_ms <= word.start_ms {
                word.start_ms.saturating_sub(segment.end_ms)
            } else {
                0
            };
            (distance <= tolerance_ms).then_some((distance, segment.start_ms, segment.cluster))
        })
        .min_by_key(|item| (item.0, item.1, item.2))
        .map(|item| item.2)
}

pub fn build_utterances(
    words: &[TimedWord],
    segments: &[SpeakerSegment],
    recognized: &HashMap<(AudioSource, u32), String>,
    local_speaker: &str,
    diarize_mic: bool,
    tolerance_ms: u64,
    meeting: Option<&MeetingDetails>,
) -> Vec<Utterance> {
    let mut anonymous = HashMap::new();
    let mut next_anonymous = 0usize;
    let mut ordered_segments = segments.iter().collect::<Vec<_>>();
    ordered_segments.sort_by_key(|segment| {
        (
            segment.start_ms,
            source_order(segment.source),
            segment.cluster,
        )
    });
    for segment in ordered_segments {
        let key = (segment.source, segment.cluster);
        if !recognized.contains_key(&key) {
            anonymous.entry(key).or_insert_with(|| {
                let name = format!("SPEAKER_{next_anonymous:02}");
                next_anonymous += 1;
                name
            });
        }
    }

    let mut output = Vec::new();
    for source in [AudioSource::Mic, AudioSource::System] {
        let mut source_words = words
            .iter()
            .filter(|word| word.source == source)
            .collect::<Vec<_>>();
        source_words.sort_by_key(|word| (word.start_ms, word.end_ms));
        let mut current: Option<Utterance> = None;
        for word in source_words {
            let (speaker_id, speaker) = word_speaker(
                word,
                segments,
                recognized,
                &anonymous,
                local_speaker,
                diarize_mic,
                tolerance_ms,
            );
            let split = current.as_ref().is_some_and(|utterance| {
                utterance.speaker_id != speaker_id
                    || word.start_ms.saturating_sub(utterance.end_ms) > 1_000
                    || (utterance.end_ms.saturating_sub(utterance.start_ms) >= 15_000
                        && ends_sentence(&utterance.text))
            });
            if split && let Some(utterance) = current.take() {
                output.push(utterance);
            }
            if let Some(utterance) = current.as_mut() {
                utterance.end_ms = utterance.end_ms.max(word.end_ms);
                append_word(&mut utterance.text, &word.text);
            } else {
                current = Some(Utterance {
                    start_ms: word.start_ms,
                    end_ms: word.end_ms,
                    source,
                    speaker_id,
                    speaker,
                    text: clean_text(&word.text),
                    echo: None,
                });
            }
        }
        if let Some(utterance) = current {
            output.push(utterance);
        }
    }
    output.retain(|utterance| !utterance.text.is_empty());
    output.sort_by_key(|utterance| (utterance.start_ms, source_order(utterance.source)));
    mark_echoes(&mut output, recognized, diarize_mic, meeting);
    output
}

fn mark_echoes(
    utterances: &mut [Utterance],
    recognized: &HashMap<(AudioSource, u32), String>,
    diarize_mic: bool,
    meeting: Option<&MeetingDetails>,
) {
    let Some(meeting) = meeting.filter(|_| diarize_mic) else {
        return;
    };
    let complete_local_roster = meeting.local.unknown == 0
        && !meeting.local.known.is_empty()
        && meeting.local.known.iter().all(|name| {
            recognized.iter().any(|((source, _), recognized)| {
                *source == AudioSource::Mic
                    && recognized == name
                    && meeting.is_local_attendee(recognized)
            })
        });
    for utterance in utterances
        .iter_mut()
        .filter(|utterance| utterance.source == AudioSource::Mic)
    {
        if meeting.is_remote_attendee(&utterance.speaker) {
            utterance.echo = Some(EchoEvidence::RemoteAttendee);
        } else if complete_local_roster && is_anonymous(&utterance.speaker) {
            utterance.echo = Some(EchoEvidence::LocalRoster);
        }
    }
}

fn is_anonymous(speaker: &str) -> bool {
    speaker.starts_with("SPEAKER_") || speaker == "unknown"
}

fn word_speaker(
    word: &TimedWord,
    segments: &[SpeakerSegment],
    recognized: &HashMap<(AudioSource, u32), String>,
    anonymous: &HashMap<(AudioSource, u32), String>,
    local_speaker: &str,
    diarize_mic: bool,
    tolerance_ms: u64,
) -> (String, String) {
    if word.source == AudioSource::Mic && !diarize_mic {
        return ("local".to_string(), local_speaker.to_string());
    }
    match assign_cluster(word, segments, tolerance_ms) {
        Some(cluster) => {
            let id_prefix = if word.source == AudioSource::Mic {
                "mic"
            } else {
                "spk"
            };
            let speaker = recognized
                .get(&(word.source, cluster))
                .cloned()
                .or_else(|| anonymous.get(&(word.source, cluster)).cloned())
                .unwrap_or_else(|| format!("SPEAKER_{cluster:02}"));
            (format!("{id_prefix}_{cluster}"), speaker)
        }
        None => ("unknown".to_string(), "unknown".to_string()),
    }
}

fn source_order(source: AudioSource) -> u8 {
    match source {
        AudioSource::Mic => 0,
        AudioSource::System => 1,
    }
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end().ends_with(['.', '?', '!'])
}

fn clean_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn append_word(text: &mut String, word: &str) {
    let word = clean_text(word);
    if word.is_empty() {
        return;
    }
    let attaches = word.starts_with(|character: char| ".,?!:;%)]}".contains(character));
    if !text.is_empty() && !attaches {
        text.push(' ');
    }
    text.push_str(&word);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(source: AudioSource, start: u64, end: u64, text: &str) -> TimedWord {
        TimedWord {
            source,
            start_ms: start,
            end_ms: end,
            text: text.into(),
        }
    }

    fn segment(source: AudioSource, start: u64, end: u64, cluster: u32) -> SpeakerSegment {
        SpeakerSegment {
            source,
            start_ms: start,
            end_ms: end,
            cluster,
        }
    }

    #[test]
    fn assigns_maximum_overlap() {
        let target = word(AudioSource::System, 100, 300, "hello");
        let segments = [
            segment(AudioSource::System, 50, 180, 1),
            segment(AudioSource::System, 150, 350, 2),
            segment(AudioSource::Mic, 100, 300, 3),
        ];
        assert_eq!(assign_cluster(&target, &segments, 500), Some(2));
    }

    #[test]
    fn nearest_fallback_and_unknown_work() {
        let segments = [segment(AudioSource::System, 1_000, 2_000, 4)];
        assert_eq!(
            assign_cluster(&word(AudioSource::System, 500, 700, "near"), &segments, 500),
            Some(4)
        );
        assert_eq!(
            assign_cluster(&word(AudioSource::System, 0, 100, "far"), &segments, 500),
            None
        );
    }

    #[test]
    fn groups_and_formats_punctuation() {
        let words = vec![
            word(AudioSource::System, 0, 100, " Hello "),
            word(AudioSource::System, 100, 200, ","),
            word(AudioSource::System, 200, 300, " world"),
            word(AudioSource::System, 300, 400, "!"),
        ];
        let segments = [segment(AudioSource::System, 0, 500, 0)];
        let utterances =
            build_utterances(&words, &segments, &HashMap::new(), "Me", false, 500, None);
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "Hello, world!");
    }

    #[test]
    fn splits_on_speaker_gap_and_long_sentence_boundary() {
        let mut words = vec![
            word(AudioSource::System, 0, 15_000, "Long sentence."),
            word(AudioSource::System, 15_100, 15_300, "Next"),
            word(AudioSource::System, 17_000, 17_100, "Gap"),
        ];
        let segments = vec![segment(AudioSource::System, 0, 20_000, 0)];
        let utterances =
            build_utterances(&words, &segments, &HashMap::new(), "Me", false, 500, None);
        assert_eq!(utterances.len(), 3);
        words[2].start_ms = 15_400;
        words[2].end_ms = 15_500;
        let segments = vec![
            segment(AudioSource::System, 0, 15_350, 0),
            segment(AudioSource::System, 15_350, 20_000, 1),
        ];
        assert_eq!(
            build_utterances(&words, &segments, &HashMap::new(), "Me", false, 0, None,).len(),
            3
        );
    }

    #[test]
    fn overlapping_sources_are_preserved() {
        let words = vec![
            word(AudioSource::Mic, 100, 500, "local"),
            word(AudioSource::System, 200, 600, "remote"),
        ];
        let segments = [segment(AudioSource::System, 0, 1_000, 0)];
        let utterances =
            build_utterances(&words, &segments, &HashMap::new(), "Me", false, 500, None);
        assert_eq!(utterances.len(), 2);
        assert!(utterances[0].end_ms > utterances[1].start_ms);
    }

    #[test]
    fn recognized_clusters_do_not_consume_anonymous_numbers() {
        let words = vec![
            word(AudioSource::System, 0, 100, "known"),
            word(AudioSource::System, 200, 300, "anonymous"),
        ];
        let segments = [
            segment(AudioSource::System, 0, 100, 4),
            segment(AudioSource::System, 200, 300, 9),
        ];
        let recognized = HashMap::from([((AudioSource::System, 4), "Alice".to_string())]);
        let utterances = build_utterances(&words, &segments, &recognized, "Me", false, 0, None);
        assert_eq!(utterances[0].speaker, "Alice");
        assert_eq!(utterances[1].speaker, "SPEAKER_00");
    }

    fn details(local: (&[&str], u32), remote: (&[&str], u32)) -> MeetingDetails {
        MeetingDetails::new(
            String::new(),
            crate::meeting::Attendees {
                known: local.0.iter().map(|name| (*name).to_owned()).collect(),
                unknown: local.1,
            },
            crate::meeting::Attendees {
                known: remote.0.iter().map(|name| (*name).to_owned()).collect(),
                unknown: remote.1,
            },
        )
    }

    fn one_utterance_echo(
        source: AudioSource,
        cluster: u32,
        recognized: HashMap<(AudioSource, u32), String>,
        meeting: &MeetingDetails,
    ) -> Option<EchoEvidence> {
        build_utterances(
            &[word(source, 0, 100, "hello")],
            &[segment(source, 0, 100, cluster)],
            &recognized,
            "Me",
            true,
            0,
            Some(meeting),
        )[0]
        .echo
    }

    #[test]
    fn remote_attendee_on_microphone_is_echo() {
        let meeting = details((&["Laura"], 0), (&["Andrew"], 0));
        let recognized = HashMap::from([((AudioSource::Mic, 1), "Andrew".into())]);
        assert_eq!(
            one_utterance_echo(AudioSource::Mic, 1, recognized, &meeting),
            Some(EchoEvidence::RemoteAttendee)
        );
    }

    #[test]
    fn complete_local_roster_marks_anonymous_microphone_echo() {
        let meeting = details((&["Laura"], 0), (&[], 0));
        let recognized = HashMap::from([((AudioSource::Mic, 1), "Laura".into())]);
        assert_eq!(
            one_utterance_echo(AudioSource::Mic, 2, recognized, &meeting),
            Some(EchoEvidence::LocalRoster)
        );
    }

    #[test]
    fn incomplete_or_unidentified_local_roster_does_not_mark_echo() {
        let unknown_local = details((&["Laura"], 1), (&[], 0));
        let recognized = HashMap::from([((AudioSource::Mic, 1), "Laura".into())]);
        assert_eq!(
            one_utterance_echo(AudioSource::Mic, 2, recognized, &unknown_local),
            None
        );

        let missing_local = details((&["Laura", "Pat"], 0), (&[], 0));
        let recognized = HashMap::from([((AudioSource::Mic, 1), "Laura".into())]);
        assert_eq!(
            one_utterance_echo(AudioSource::Mic, 2, recognized, &missing_local),
            None
        );
    }

    #[test]
    fn system_utterances_are_never_echoes() {
        let meeting = details((&["Laura"], 0), (&["Andrew"], 0));
        let recognized = HashMap::from([((AudioSource::System, 1), "Andrew".into())]);
        assert_eq!(
            one_utterance_echo(AudioSource::System, 1, recognized, &meeting),
            None
        );
    }
}
