//! Lines a hidden source keeps: their state on the timeline, their menu items
//! and the bar under the Hide toggles that lists what is kept.

use crate::merge::process::KeepChange;
use crate::speaker::database::canonical_name_key;
use crate::types::{AudioSource, HiddenSources, KeptRange, Utterance};
use gtk::prelude::*;
use gtk4 as gtk;

/// How a hidden source treats a line on the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LineKeep {
    Shown,
    Hidden,
    /// Kept by a speaker rule, or else by a kept range.
    Kept {
        by_speaker: bool,
    },
}

/// The state of a rendered line; `hidden_line` when it came from
/// `hidden-lines.jsonl`.
pub(super) fn line_keep(
    rules: &HiddenSources,
    local_speaker: &str,
    line: &Utterance,
    hidden_line: bool,
) -> LineKeep {
    if hidden_line {
        return LineKeep::Hidden;
    }
    if !rules.hidden.contains(&line.source) {
        return LineKeep::Shown;
    }
    LineKeep::Kept {
        by_speaker: rules.keeps_speaker_of(local_speaker, line),
    }
}

fn line_range(line: &Utterance) -> KeptRange {
    KeptRange {
        source: line.source,
        start_ms: line.start_ms,
        end_ms: line.end_ms,
    }
}

/// The menu items that change what keeps this line.
pub(super) fn line_actions(state: LineKeep, line: &Utterance) -> Vec<(String, KeepChange)> {
    let speaker = &line.speaker;
    match state {
        LineKeep::Shown => Vec::new(),
        LineKeep::Hidden => {
            let mut actions = vec![(
                "Keep this line".to_owned(),
                KeepChange::Line(line_range(line)),
            )];
            if !line.echo {
                actions.push((
                    format!("Keep all of {speaker}’s lines"),
                    KeepChange::Speaker(speaker.clone()),
                ));
            }
            actions
        }
        // A speaker rule keeps the line whatever its range, so it is the one
        // to remove.
        LineKeep::Kept { by_speaker: true } => vec![(
            format!("Stop keeping {speaker}’s lines"),
            KeepChange::HideSpeaker(speaker.clone()),
        )],
        LineKeep::Kept { by_speaker: false } => vec![(
            "Hide this line again".to_owned(),
            KeepChange::HideLine(line_range(line)),
        )],
    }
}

/// "Kept" next to the speaker, saying why.
pub(super) fn kept_marker(by_speaker: bool, speaker: &str, source: AudioSource) -> gtk::Label {
    let marker = gtk::Label::new(Some("Kept"));
    marker.add_css_class("caption");
    marker.add_css_class("dim-label");
    marker.add_css_class("kept-marker");
    marker.set_valign(gtk::Align::Center);
    let why = if by_speaker {
        format!(
            "Kept from the hidden {}: {speaker}’s lines are kept",
            source_name(source)
        )
    } else {
        format!("Kept from the hidden {}", source_name(source))
    };
    marker.set_tooltip_text(Some(&why));
    marker.update_property(&[gtk::accessible::Property::Label(&why)]);
    marker
}

fn source_name(source: AudioSource) -> &'static str {
    match source {
        AudioSource::Mic => "microphone",
        AudioSource::System => "system audio",
    }
}

pub(super) fn hidden_title(hidden: &[AudioSource]) -> &'static str {
    match hidden {
        [AudioSource::Mic] => "Microphone hidden",
        [AudioSource::System] => "System audio hidden",
        _ => "Microphone and system audio hidden",
    }
}

/// One chip per kept name, with what it keeps. The local speaker's chip
/// keeps only the identified voice until that name is also kept by hand.
pub(super) fn speaker_chips(rules: &HiddenSources, local_speaker: &str) -> Vec<(String, String)> {
    let key = canonical_name_key(local_speaker);
    let by_hand = rules
        .keep_speakers
        .iter()
        .any(|name| canonical_name_key(name) == key);
    rules
        .kept_speakers(local_speaker)
        .into_iter()
        .map(|name| {
            let what = if !by_hand && canonical_name_key(&name) == key {
                format!("Keeping the lines recognized as {name}’s voice or named {name} by you")
            } else {
                format!("Keeping every line named {name}")
            };
            (name, what)
        })
        .collect()
}

pub(super) fn lines(count: usize) -> String {
    format!("{count} line{}", if count == 1 { "" } else { "s" })
}

/// What the bar shows besides the chips.
pub(super) struct KeepSummary {
    /// Names and what each keeps.
    pub kept_speakers: Vec<(String, String)>,
    pub kept_lines: usize,
    /// None when the session was rendered before hidden lines were kept apart.
    pub hidden_lines: Option<usize>,
    pub hidden_shown: bool,
}

/// "Microphone hidden · keeping [Stefan ✕] · 3 lines kept · Show 40 hidden lines".
pub(super) fn keep_bar(
    bar: &gtk::Box,
    hidden: &[AudioSource],
    summary: KeepSummary,
    forget_speaker: impl Fn(String) + Clone + 'static,
    show_hidden: impl Fn(bool) + 'static,
) {
    while let Some(child) = bar.first_child() {
        bar.remove(&child);
    }
    let text = |text: &str| {
        let label = gtk::Label::new(Some(text));
        label.add_css_class("dim-label");
        label.set_valign(gtk::Align::Center);
        bar.append(&label);
        label
    };
    text(hidden_title(hidden));
    if !summary.kept_speakers.is_empty() || summary.kept_lines > 0 {
        text("· keeping");
    }
    for (name, what) in summary.kept_speakers {
        let chip = gtk::Box::new(gtk::Orientation::Horizontal, 2);
        chip.add_css_class("keep-chip");
        chip.set_valign(gtk::Align::Center);
        chip.set_tooltip_text(Some(&what));
        chip.update_property(&[gtk::accessible::Property::Description(&what)]);
        chip.append(&gtk::Label::new(Some(&name)));
        let remove = gtk::Button::from_icon_name("window-close-symbolic");
        remove.add_css_class("flat");
        remove.add_css_class("circular");
        let label = format!("Stop keeping {name}’s lines");
        remove.set_tooltip_text(Some(&label));
        remove.update_property(&[gtk::accessible::Property::Label(&label)]);
        let forget = forget_speaker.clone();
        remove.connect_clicked(move |_| forget(name.clone()));
        chip.append(&remove);
        bar.append(&chip);
    }
    if summary.kept_lines > 0 {
        text(&format!("· {} kept", lines(summary.kept_lines)));
    }
    if summary.hidden_lines != Some(0) {
        let toggle = gtk::Button::new();
        toggle.add_css_class("flat");
        toggle.add_css_class("meeting-fold-toggle");
        let shown = summary.hidden_shown;
        let count = summary
            .hidden_lines
            .map(|count| format!("{count} "))
            .unwrap_or_default();
        let label = format!(
            "{} {count}hidden line{}",
            if shown { "Fold away" } else { "Show" },
            if summary.hidden_lines == Some(1) {
                ""
            } else {
                "s"
            }
        );
        toggle.set_label(&label);
        toggle.set_tooltip_text(Some(if shown {
            "Fold the hidden lines away again"
        } else {
            "Show the hidden lines greyed, to keep some of them"
        }));
        toggle.connect_clicked(move |_| show_hidden(!shown));
        bar.append(&toggle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(source: AudioSource, speaker: &str, echo: bool) -> Utterance {
        Utterance {
            start_ms: 1_000,
            end_ms: 2_000,
            source,
            speaker_id: "mic_0".into(),
            speaker: speaker.into(),
            text: "text".into(),
            locked: false,
            echo,
        }
    }

    fn rules() -> HiddenSources {
        HiddenSources {
            hidden: vec![AudioSource::Mic],
            keep_speakers: vec!["Anna".into()],
            keep: vec![KeptRange {
                source: AudioSource::Mic,
                start_ms: 1_000,
                end_ms: 2_000,
            }],
            ..HiddenSources::default()
        }
    }

    #[test]
    fn line_state_tells_why_a_line_is_kept() {
        let mic = |speaker| line(AudioSource::Mic, speaker, false);
        assert_eq!(
            line_keep(&rules(), "Stefan", &mic("Anna"), true),
            LineKeep::Hidden
        );
        assert_eq!(
            line_keep(
                &rules(),
                "Stefan",
                &line(AudioSource::System, "Anna", false),
                false
            ),
            LineKeep::Shown
        );
        assert_eq!(
            line_keep(&rules(), "Stefan", &mic("stefan"), false),
            LineKeep::Kept { by_speaker: true }
        );
        // An echo line is kept only by its range, whatever its name.
        assert_eq!(
            line_keep(
                &rules(),
                "Stefan",
                &line(AudioSource::Mic, "Anna", true),
                false
            ),
            LineKeep::Kept { by_speaker: false }
        );
    }

    #[test]
    fn local_chip_says_whether_it_keeps_by_name() {
        let automatic = HiddenSources {
            hidden: vec![AudioSource::Mic],
            keep_speakers: vec!["Anna".into()],
            ..HiddenSources::default()
        };
        let names = |rules: &HiddenSources| {
            speaker_chips(rules, "Stefan ")
                .into_iter()
                .map(|(name, what)| (name, what.contains("every line")))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(&automatic),
            [("Stefan".into(), false), ("Anna".into(), true)]
        );
        let by_hand = HiddenSources {
            keep_speakers: vec!["Anna".into(), "stefan".into()],
            ..automatic
        };
        assert_eq!(
            names(&by_hand),
            [("Stefan".into(), true), ("Anna".into(), true)]
        );
    }

    #[test]
    fn menu_offers_keeping_or_hiding_again() {
        let anna = line(AudioSource::Mic, "Anna", false);
        let range = line_range(&anna);
        assert_eq!(
            line_actions(LineKeep::Hidden, &anna),
            [
                ("Keep this line".to_owned(), KeepChange::Line(range)),
                (
                    "Keep all of Anna’s lines".to_owned(),
                    KeepChange::Speaker("Anna".into())
                ),
            ]
        );
        assert_eq!(
            line_actions(LineKeep::Hidden, &line(AudioSource::Mic, "Anna", true)).len(),
            1
        );
        assert_eq!(
            line_actions(LineKeep::Kept { by_speaker: true }, &anna),
            [(
                "Stop keeping Anna’s lines".to_owned(),
                KeepChange::HideSpeaker("Anna".into())
            )]
        );
        assert_eq!(
            line_actions(LineKeep::Kept { by_speaker: false }, &anna),
            [(
                "Hide this line again".to_owned(),
                KeepChange::HideLine(range)
            )]
        );
        assert!(line_actions(LineKeep::Shown, &anna).is_empty());
    }
}
