//! Lines cut into speaker turns, kept in `speaker-turns.json`.

use crate::speaker::database::canonical_name_key;
use crate::types::{AudioSource, Turn, TurnGroup};

/// What is left of a line once its cuts or turn names change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Settled {
    /// Still cut into turns.
    Turns(TurnGroup),
    /// One line again, locked to the name when it has one.
    Line(Option<String>),
}

/// The line `[start_ms, end_ms]` cut before each of `cuts`. A turn that keeps
/// its start keeps its speaker, and a new turn starts unnamed; a removed cut
/// joins its turn to the one before. The first turn keeps its speaker, or
/// takes the one of the turn that starts at `first_from`, which held the
/// line's first words.
pub fn with_cuts(
    previous: Option<&TurnGroup>,
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
    cuts: &[u64],
    first_from: Option<u64>,
) -> TurnGroup {
    let mut starts = cuts
        .iter()
        .copied()
        .filter(|cut| *cut > start_ms && *cut <= end_ms)
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    let speaker_at = |at_ms: u64| {
        previous?
            .turns
            .iter()
            .find(|turn| turn.start_ms == at_ms)?
            .speaker
            .clone()
    };
    let first = Turn {
        start_ms,
        speaker: match first_from {
            Some(at_ms) => speaker_at(at_ms),
            None => previous.and_then(|group| group.turns.first()?.speaker.clone()),
        },
    };
    let turns = std::iter::once(first)
        .chain(starts.into_iter().map(|start_ms| Turn {
            start_ms,
            speaker: speaker_at(start_ms),
        }))
        .collect();
    TurnGroup {
        source,
        start_ms,
        end_ms,
        turns,
    }
}

/// Joins a group whose turns all have one speaker, and one left with a single
/// turn. `first_line_speaker` is what the unnamed first turn is called in the
/// transcript; the other turns count only once named.
pub fn settle(group: TurnGroup, first_line_speaker: Option<&str>) -> Settled {
    if group.turns.len() < 2 {
        return Settled::Line(group.turns.into_iter().next().and_then(|turn| turn.speaker));
    }
    let Some(name) = group.turns[1].speaker.clone() else {
        return Settled::Turns(group);
    };
    let key = canonical_name_key(&name);
    let first = group.turns[0].speaker.as_deref().or(first_line_speaker);
    let same = first.is_some_and(|first| canonical_name_key(first) == key)
        && group.turns[2..].iter().all(|turn| {
            turn.speaker
                .as_deref()
                .is_some_and(|speaker| canonical_name_key(speaker) == key)
        });
    if same {
        Settled::Line(Some(group.turns[0].speaker.clone().unwrap_or(name)))
    } else {
        Settled::Turns(group)
    }
}

/// Why `groups` cannot be rendered: a group must hold its turns in order,
/// starting with it, and groups of one source may touch but not overlap.
pub fn check_groups(groups: &[TurnGroup]) -> Result<(), String> {
    for group in groups {
        let in_order = group.turns.first().map(|turn| turn.start_ms) == Some(group.start_ms)
            && group.start_ms <= group.end_ms
            && group
                .turns
                .windows(2)
                .all(|pair| pair[0].start_ms < pair[1].start_ms)
            && group.turns.iter().all(|turn| turn.start_ms <= group.end_ms);
        if !in_order {
            return Err(format!(
                "the cut line {} needs turns in order inside it, the first at its start",
                group.id()
            ));
        }
    }
    let mut ordered = groups.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|group| (group.source.as_str(), group.start_ms, group.end_ms));
    for pair in ordered.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        if left.source == right.source
            && (left.start_ms == right.start_ms || right.start_ms < left.end_ms)
        {
            return Err(format!(
                "the cut lines {} and {} overlap",
                left.id(),
                right.id()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(turns: &[(u64, Option<&str>)]) -> TurnGroup {
        TurnGroup {
            source: AudioSource::System,
            start_ms: 1_000,
            end_ms: 9_000,
            turns: turns
                .iter()
                .map(|&(start_ms, speaker)| Turn {
                    start_ms,
                    speaker: speaker.map(str::to_owned),
                })
                .collect(),
        }
    }

    fn speakers(group: &TurnGroup) -> Vec<(u64, Option<&str>)> {
        group
            .turns
            .iter()
            .map(|turn| (turn.start_ms, turn.speaker.as_deref()))
            .collect()
    }

    #[test]
    fn cuts_keep_speakers_by_start_and_new_turns_start_unnamed() {
        let before = group(&[
            (1_000, Some("Laura")),
            (3_000, Some("Andrew")),
            (5_000, None),
        ]);
        let after = with_cuts(
            Some(&before),
            AudioSource::System,
            1_000,
            9_000,
            &[7_000, 5_000, 1_000, 9_500, 7_000],
            None,
        );
        // The cut at 3 s is gone, so Andrew's turn joins Laura's.
        assert_eq!(
            speakers(&after),
            [(1_000, Some("Laura")), (5_000, None), (7_000, None)]
        );
    }

    #[test]
    fn an_emptied_first_turn_gives_way_to_the_turn_holding_the_first_words() {
        let before = group(&[
            (1_000, None),
            (2_000, Some("Andrew")),
            (3_000, Some("Craig")),
        ]);
        let after = with_cuts(
            Some(&before),
            AudioSource::System,
            1_000,
            9_000,
            &[3_000],
            Some(2_000),
        );
        assert_eq!(
            speakers(&after),
            [(1_000, Some("Andrew")), (3_000, Some("Craig"))]
        );
        // With every cut gone the line is Andrew's.
        let line = with_cuts(
            Some(&before),
            AudioSource::System,
            1_000,
            9_000,
            &[],
            Some(2_000),
        );
        assert_eq!(settle(line, None), Settled::Line(Some("Andrew".into())));
    }

    #[test]
    fn same_speaker_everywhere_joins_into_a_named_line() {
        let named = group(&[(1_000, Some("Laura")), (3_000, Some(" laura "))]);
        assert_eq!(settle(named, None), Settled::Line(Some("Laura".into())));

        let unnamed_first = group(&[
            (1_000, None),
            (3_000, Some("Laura")),
            (5_000, Some("Laura")),
        ]);
        assert_eq!(
            settle(unnamed_first.clone(), Some("LAURA")),
            Settled::Line(Some("Laura".into()))
        );
        assert!(matches!(
            settle(unnamed_first, Some("SPEAKER_01")),
            Settled::Turns(_)
        ));

        let who = group(&[
            (1_000, Some("Laura")),
            (3_000, Some("Laura")),
            (5_000, None),
        ]);
        assert!(matches!(settle(who, Some("Laura")), Settled::Turns(_)));
    }

    #[test]
    fn groups_may_touch_but_not_overlap_repeat_or_hold_stray_turns() {
        let at = |start_ms, end_ms, turns: &[u64]| TurnGroup {
            source: AudioSource::System,
            start_ms,
            end_ms,
            turns: turns
                .iter()
                .map(|&start_ms| Turn {
                    start_ms,
                    speaker: None,
                })
                .collect(),
        };
        let mic = TurnGroup {
            source: AudioSource::Mic,
            ..at(1_500, 2_500, &[1_500])
        };
        assert!(
            check_groups(&[at(1_000, 2_000, &[1_000]), at(2_000, 3_000, &[2_000]), mic]).is_ok()
        );
        for bad in [
            vec![at(1_000, 2_000, &[1_000]), at(1_999, 3_000, &[1_999])],
            vec![
                at(1_000, 2_000, &[1_000]),
                at(1_000, 2_000, &[1_000, 1_500]),
            ],
            vec![at(1_000, 2_000, &[1_000, 2_500])],
            vec![at(1_000, 2_000, &[1_200])],
        ] {
            assert!(check_groups(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_single_turn_is_a_line_with_its_name() {
        assert_eq!(
            settle(group(&[(1_000, Some("Craig"))]), None),
            Settled::Line(Some("Craig".into()))
        );
        assert_eq!(
            settle(group(&[(1_000, None)]), Some("Laura")),
            Settled::Line(None)
        );
    }
}
