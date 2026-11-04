//! Cutting a transcript line into speaker turns: the cut mode of a card, and
//! the card of a line cut into turns.

use super::keep::{self, LineKeep};
use super::wrap_layout::WrapLayout;
use super::{
    AssignTarget, SessionDetail, avatar_color, format_timeline_timestamp, set_playback_button,
    show_assignment_dialog, show_error, status_pill, update_transcript,
};
use crate::cli::ProcessArgs;
use crate::format::jsonl;
use crate::merge::process::{self, CutRequest, KeepChange};
use crate::merge::utterances::midpoint;
use crate::speaker::database::canonical_name_key;
use crate::types::{AudioSource, TimedWord, TurnGroup, Utterance};
use gtk::{gdk, glib, prelude::*};
use gtk4 as gtk;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

/// A word as cut mode shows it: punctuation, and a word that starts with the
/// one before, stay with the word before, since no cut fits between them,
/// unless one of `cuts` puts them in different turns the way rendering does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Token {
    pub text: String,
    pub start_ms: u64,
}

pub(super) fn tokens(words: &[TimedWord], cuts: &[u64]) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    for word in words {
        let text = word.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            continue;
        }
        let attaches = text.starts_with(|character: char| ".,?!:;%)]}".contains(character));
        let same_turn = |last: &Token| {
            !cuts
                .iter()
                .any(|cut| *cut > last.start_ms && *cut <= word.start_ms)
        };
        match tokens.last_mut() {
            Some(last) if attaches && same_turn(last) => last.text.push_str(&text),
            Some(last) if word.start_ms <= last.start_ms => {
                last.text.push(' ');
                last.text.push_str(&text);
            }
            _ => tokens.push(Token {
                text,
                start_ms: word.start_ms,
            }),
        }
    }
    tokens
}

/// The words of the line `[start_ms, end_ms]`, each by its midpoint, as the
/// transcript groups them.
pub(super) fn line_words(
    words: Vec<TimedWord>,
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
) -> Vec<TimedWord> {
    let mut words = words
        .into_iter()
        .filter(|word| word.source == source && (start_ms..=end_ms).contains(&midpoint(word)))
        .collect::<Vec<_>>();
    words.sort_by_key(|word| (word.start_ms, word.end_ms));
    words
}

/// The transcript lines with the turns of each cut line folded into one line
/// that spans the whole cut line, and the turns of each line, empty for an
/// uncut one. `ranges` holds each cut line's range by group id: the line
/// starts exactly where the group does, shown or not, as rendering places it
/// in the meeting, and ends at the later of the group's and its words' ends.
pub(super) type Folded = (Vec<Utterance>, Vec<LineKeep>, Vec<Vec<Line>>);

type Line = (Utterance, LineKeep);

pub(super) fn fold_groups(lines: Vec<Line>, ranges: &HashMap<String, (u64, u64)>) -> Folded {
    let mut folded: Vec<(Utterance, LineKeep, Vec<Line>)> = Vec::new();
    for (line, state) in lines {
        let Some(group) = line.group.clone() else {
            folded.push((line, state, Vec::new()));
            continue;
        };
        match folded
            .iter_mut()
            .find(|(folded, _, turns)| !turns.is_empty() && folded.group.as_ref() == Some(&group))
        {
            Some((_, _, turns)) => turns.push((line, state)),
            None => folded.push((line.clone(), state, vec![(line, state)])),
        }
    }
    for (line, state, group_turns) in &mut folded {
        if group_turns.is_empty() {
            continue;
        }
        group_turns.sort_by_key(|(turn, _)| turn.start_ms);
        // The line is hidden only when every turn is, and is kept as its
        // first kept turn is.
        *state = group_turns
            .iter()
            .map(|(_, state)| *state)
            .find(|state| *state != LineKeep::Hidden)
            .unwrap_or(LineKeep::Hidden);
        let first = group_turns
            .iter()
            .find(|(_, turn_state)| turn_state == state)
            .map_or(&group_turns[0].0, |(turn, _)| turn);
        let shown_start = group_turns[0].0.start_ms;
        let shown_end = group_turns
            .iter()
            .map(|(turn, _)| turn.end_ms)
            .max()
            .unwrap_or(first.end_ms);
        let (start_ms, end_ms) = line
            .group
            .as_ref()
            .and_then(|group| ranges.get(group))
            .map_or((shown_start, shown_end), |&(start, end)| {
                (start, end.max(shown_end))
            });
        *line = Utterance {
            start_ms,
            end_ms,
            text: group_turns
                .iter()
                .map(|(turn, _)| turn.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            locked: group_turns.iter().all(|(turn, _)| turn.locked),
            echo: group_turns.iter().all(|(turn, _)| turn.echo),
            ..first.clone()
        };
    }
    folded.sort_by_key(|(line, _, _)| (line.start_ms, line.source == AudioSource::System));
    let mut lines = Vec::new();
    let mut states = Vec::new();
    let mut turns = Vec::new();
    for (line, state, group_turns) in folded {
        lines.push(line);
        states.push(state);
        turns.push(group_turns);
    }
    (lines, states, turns)
}

/// Each person's colour on the transcript: the colour most of their uncut
/// lines have, by canonical name.
pub(super) fn person_colors(lines: &[Utterance]) -> HashMap<String, usize> {
    let mut counts: HashMap<String, [usize; 3]> = HashMap::new();
    for line in lines
        .iter()
        .filter(|line| line.group.is_none() && !is_anonymous(&line.speaker))
    {
        counts.entry(canonical_name_key(&line.speaker)).or_default()[avatar_color(line)] += 1;
    }
    counts
        .into_iter()
        .map(|(name, counts)| {
            let color = (0..3)
                .max_by_key(|&color| (counts[color], std::cmp::Reverse(color)))
                .unwrap_or(0);
            (name, color)
        })
        .collect()
}

/// The turn a word of each token start belongs to, by the rule rendering
/// uses: the latest cut at or before it, none for the first turn.
fn token_turns(starts: &[u64], cuts: &[u64]) -> Vec<Option<u64>> {
    starts
        .iter()
        .map(|&start| cuts.iter().copied().filter(|cut| *cut <= start).max())
        .collect()
}

/// Removes the turn `key` by dropping every cut after the turn before it, so
/// that no cut left without words keeps the turns apart.
fn remove_turn(cuts: &mut Vec<u64>, before: Option<u64>, key: u64) {
    cuts.retain(|cut| before.is_some_and(|before| *cut <= before) || *cut > key);
}

/// How a turn is shown: by its own name, by the line's name for an unnamed
/// first turn, or as "Who?".
#[derive(Debug, Clone, PartialEq, Eq)]
enum TurnName {
    Named(String),
    Line { name: String, anonymous: bool },
    Unassigned,
}

impl TurnName {
    fn label(&self) -> Option<&str> {
        match self {
            TurnName::Named(name) | TurnName::Line { name, .. } => Some(name),
            TurnName::Unassigned => None,
        }
    }
}

/// The colour class of each turn: the colour of the person's own lines,
/// green on the microphone, and for someone without lines of their own a
/// colour the card does not use yet where one is left; amber for an anonymous
/// voice and grey for "Who?".
fn turn_colors(
    names: &[TurnName],
    source: AudioSource,
    known: &HashMap<String, usize>,
) -> Vec<String> {
    let person = |name: &TurnName| match name {
        TurnName::Named(name)
        | TurnName::Line {
            name,
            anonymous: false,
        } => Some(canonical_name_key(name)),
        _ => None,
    };
    let known_color = |key: &str| match source {
        AudioSource::Mic => Some(2),
        AudioSource::System => known.get(key).copied(),
    };
    let mut used = names
        .iter()
        .filter_map(person)
        .filter_map(|key| known_color(&key))
        .collect::<Vec<_>>();
    let mut chosen: HashMap<String, usize> = HashMap::new();
    let mut new_people = 0;
    names
        .iter()
        .map(|name| match (name, person(name)) {
            (TurnName::Unassigned, _) => "turn-none".to_owned(),
            (_, None) => "turn-unknown".to_owned(),
            (_, Some(key)) => {
                let color = known_color(&key).unwrap_or_else(|| {
                    *chosen.entry(key).or_insert_with(|| {
                        let color = (0..3)
                            .find(|color| !used.contains(color))
                            .unwrap_or(new_people % 3);
                        new_people += 1;
                        used.push(color);
                        color
                    })
                });
                format!("turn-{color}")
            }
        })
        .collect()
}

fn turns_label(count: usize) -> String {
    format!("{count} turn{}", if count == 1 { "" } else { "s" })
}

pub(super) fn is_anonymous(name: &str) -> bool {
    name.starts_with("SPEAKER_") || name == "unknown"
}

/// A line to cut: an uncut one, or the group it was cut into.
pub(super) struct CutTarget {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
    pub group: Option<TurnGroup>,
    pub line_speaker: String,
    /// Each person's colour on the transcript, see [`person_colors`].
    pub colors: Rc<HashMap<String, usize>>,
}

impl CutTarget {
    fn cuts(&self) -> Vec<u64> {
        self.group
            .iter()
            .flat_map(|group| group.turns.iter().skip(1).map(|turn| turn.start_ms))
            .collect()
    }

    /// The name of the turn that starts at cut `key`, or of the first turn.
    fn name_of(&self, key: Option<u64>) -> TurnName {
        let stored = self.group.as_ref().and_then(|group| {
            group
                .turns
                .iter()
                .find(|turn| {
                    key.map_or(turn.start_ms == group.start_ms, |key| turn.start_ms == key)
                })
                .and_then(|turn| turn.speaker.clone())
        });
        match stored {
            Some(name) => TurnName::Named(name),
            None if key.is_none() => TurnName::Line {
                anonymous: is_anonymous(&self.line_speaker),
                name: self.line_speaker.clone(),
            },
            None => TurnName::Unassigned,
        }
    }
}

pub(super) fn scissors_button(tooltip: &str) -> gtk::Button {
    let button = gtk::Button::from_icon_name("edit-cut-symbolic");
    button.add_css_class("flat");
    button.add_css_class("transcript-icon");
    button.set_valign(gtk::Align::Center);
    button.set_tooltip_text(Some(tooltip));
    button.update_property(&[gtk::accessible::Property::Label(tooltip)]);
    button
}

/// Opens cut mode in `card`, over its own content, which comes back on
/// Cancel or Escape.
pub(super) fn enter_cut_mode(card: &gtk::Box, target: CutTarget, detail: &SessionDetail) {
    let Some(session_dir) = detail.selected.borrow().clone() else {
        return;
    };
    let words = crate::session::Session::open(&session_dir)
        .and_then(|session| jsonl::read_all::<TimedWord>(&session.words_path()))
        .map(|words| line_words(words, target.source, target.start_ms, target.end_ms));
    let words = match words {
        Ok(words) if tokens(&words, &target.cuts()).len() > 1 => words,
        Ok(_) => {
            show_error(
                &detail.window,
                "Nothing to cut",
                "This line has only one word.",
            );
            return;
        }
        Err(error) => {
            show_error(
                &detail.window,
                "Could not read the words of this line",
                &error.to_string(),
            );
            return;
        }
    };
    let content = children(card);
    for child in &content {
        child.set_visible(false);
    }
    card.add_css_class("cut-editing");
    let editor = gtk::Box::new(gtk::Orientation::Vertical, 6);
    card.append(&editor);

    let state = Rc::new(CutState {
        cuts: RefCell::new(target.cuts()),
        target,
        words,
        badge: gtk::Label::new(None),
        body: gtk::Box::new(gtk::Orientation::Vertical, 2),
    });
    let original = state.saved_cuts();

    let head = gtk::Box::new(gtk::Orientation::Horizontal, 7);
    let first = state
        .target
        .name_of(state.token_turns().first().copied().flatten());
    let colors = turn_colors(
        std::slice::from_ref(&first),
        state.target.source,
        &state.target.colors,
    );
    head.append(&avatar(&colors[0], 18));
    let name = gtk::Label::new(first.label());
    name.add_css_class("heading");
    if matches!(
        first,
        TurnName::Line {
            anonymous: true,
            ..
        }
    ) {
        name.add_css_class("warning-text");
    }
    head.append(&name);
    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    head.append(&spacer);
    state.badge.add_css_class("turn-count");
    state.badge.set_valign(gtk::Align::Center);
    head.append(&state.badge);
    editor.append(&head);
    editor.append(&state.body);

    let bar = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    bar.add_css_class("cut-bar");
    let hint = gtk::Label::new(Some(
        "Click between words to cut, on a cut to join · Esc cancels",
    ));
    hint.set_tooltip_text(Some(
        "With the keyboard, Tab to the gap after a word and press Enter to cut there, or to a cut to remove it",
    ));
    hint.add_css_class("dim-label");
    hint.add_css_class("caption");
    hint.set_wrap(true);
    hint.set_xalign(0.0);
    hint.set_hexpand(true);
    bar.append(&hint);
    let cancel = gtk::Button::with_label("Cancel");
    let done = gtk::Button::with_label("Done");
    done.add_css_class("suggested-action");
    for button in [&cancel, &done] {
        button.set_valign(gtk::Align::Center);
        bar.append(button);
    }
    editor.append(&bar);

    let close: Rc<dyn Fn()> = {
        let card = card.downgrade();
        let editor = editor.downgrade();
        Rc::new(move || {
            if let (Some(card), Some(editor)) = (card.upgrade(), editor.upgrade()) {
                card.remove(&editor);
                card.remove_css_class("cut-editing");
                for child in &content {
                    child.set_visible(true);
                }
            }
        })
    };
    let close_for_cancel = close.clone();
    cancel.connect_clicked(move |_| close_for_cancel());
    let keys = gtk::EventControllerKey::new();
    let close_for_keys = close.clone();
    keys.connect_key_pressed(move |_, key, _, _| {
        if key == gdk::Key::Escape {
            close_for_keys();
            return glib::Propagation::Stop;
        }
        glib::Propagation::Proceed
    });
    editor.add_controller(keys);

    let detail = detail.clone();
    let state_for_done = state.clone();
    done.connect_clicked(move |_| {
        let saved = state_for_done.saved_cuts();
        if saved == original {
            close();
            return;
        }
        let (first_from, cuts) = saved;
        let request = CutRequest {
            source: state_for_done.target.source,
            start_ms: state_for_done.target.start_ms,
            end_ms: state_for_done.target.end_ms,
            cuts,
            first_from,
        };
        update_transcript(
            &detail,
            "Cut lines once processing has finished.",
            "Could not cut the line",
            move |session| {
                process::cut_line(&ProcessArgs::for_session(session.to_owned()), &request)
            },
        );
    });

    render_cuts(&state, None);
    done.grab_focus();
}

struct CutState {
    target: CutTarget,
    words: Vec<TimedWord>,
    /// Where turns start, as stored: a cut moved by reprocessing keeps its
    /// time, and with it its speaker.
    cuts: RefCell<Vec<u64>>,
    badge: gtk::Label,
    body: gtk::Box,
}

impl CutState {
    fn tokens(&self) -> Vec<Token> {
        tokens(&self.words, &self.cuts.borrow())
    }

    fn token_turns(&self) -> Vec<Option<u64>> {
        let starts = self
            .tokens()
            .iter()
            .map(|token| token.start_ms)
            .collect::<Vec<_>>();
        token_turns(&starts, &self.cuts.borrow())
    }

    fn saved_cuts(&self) -> (Option<u64>, Vec<u64>) {
        saved_cuts(&self.token_turns())
    }
}

/// The cuts to save for turns that hold words, by the turn of each token:
/// the turn whose speaker the first turn takes when that is not the first
/// turn itself, and the cuts that start the others.
fn saved_cuts(turns: &[Option<u64>]) -> (Option<u64>, Vec<u64>) {
    let mut keys = turns.to_vec();
    keys.dedup();
    let first = keys.first().copied().flatten();
    (first, keys.into_iter().skip(1).flatten().collect())
}

/// Lays out the turns of cut mode again; `focus` is the word start whose gap,
/// or the cut whose divider, takes the keyboard focus.
fn render_cuts(state: &Rc<CutState>, focus: Option<u64>) {
    while let Some(child) = state.body.first_child() {
        state.body.remove(&child);
    }
    let tokens = state.tokens();
    let mut turns: Vec<(Option<u64>, Vec<&Token>)> = Vec::new();
    for (token, key) in tokens.iter().zip(state.token_turns()) {
        match turns.last_mut() {
            Some((last, tokens)) if *last == key => tokens.push(token),
            _ => turns.push((key, vec![token])),
        }
    }
    state.badge.set_label(&turns_label(turns.len()));
    let names = turns
        .iter()
        .map(|(key, _)| state.target.name_of(*key))
        .collect::<Vec<_>>();
    let colors = turn_colors(&names, state.target.source, &state.target.colors);
    let mut focus_widget: Option<gtk::Widget> = None;
    for (index, (key, tokens)) in turns.iter().enumerate() {
        if let Some(key) = *key
            && index > 0
        {
            let divider = divider(tokens[0]);
            let state_for_click = state.clone();
            let before = turns[index - 1].0;
            let first_word = tokens[0].start_ms;
            divider.connect_clicked(move |_| {
                remove_turn(&mut state_for_click.cuts.borrow_mut(), before, key);
                render_cuts(&state_for_click, Some(first_word));
            });
            if focus == Some(key) {
                focus_widget = Some(divider.clone().upcast());
            }
            state.body.append(&divider);
        }
        let block = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        block.add_css_class("cut-turn");
        block.add_css_class(&colors[index]);
        block.set_layout_manager(Some(WrapLayout::new(0, 2)));
        for (position, token) in tokens.iter().enumerate() {
            let unit = gtk::Box::new(gtk::Orientation::Horizontal, 0);
            let word = gtk::Label::new(Some(&token.text));
            unit.append(&word);
            if let Some(next) = tokens.get(position + 1) {
                let gap = gtk::Button::new();
                gap.add_css_class("cut-gap");
                gap.set_tooltip_text(Some("Cut here"));
                let label = format!("Cut before “{}”", next.text);
                gap.update_property(&[gtk::accessible::Property::Label(&label)]);
                let state_for_click = state.clone();
                let at = next.start_ms;
                gap.connect_clicked(move |_| {
                    state_for_click.cuts.borrow_mut().push(at);
                    render_cuts(&state_for_click, Some(at));
                });
                if focus == Some(next.start_ms) {
                    focus_widget = Some(gap.clone().upcast());
                }
                unit.append(&gap);
            }
            block.append(&unit);
        }
        state.body.append(&block);
    }
    if let Some(widget) = focus_widget {
        widget.grab_focus();
    }
}

fn divider(next: &Token) -> gtk::Button {
    let button = gtk::Button::new();
    button.add_css_class("flat");
    button.add_css_class("cut-divider");
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let dash = || {
        let line = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        line.add_css_class("cut-dash");
        line.set_hexpand(true);
        line.set_valign(gtk::Align::Center);
        line
    };
    let label = gtk::Label::new(Some("✂"));
    row.append(&dash());
    row.append(&label);
    row.append(&dash());
    button.set_child(Some(&row));
    let name = format!("Remove the cut before “{}”", next.text);
    button.update_property(&[gtk::accessible::Property::Label(&name)]);
    let hover = gtk::EventControllerMotion::new();
    let (entered, left) = (label.clone(), label);
    let (entered_button, left_button) = (button.downgrade(), button.downgrade());
    hover.connect_enter(move |_, _, _| {
        entered.set_label("✕ remove cut");
        if let Some(button) = entered_button.upgrade() {
            button.add_css_class("removing");
        }
    });
    hover.connect_leave(move |_| {
        left.set_label("✂");
        if let Some(button) = left_button.upgrade() {
            button.remove_css_class("removing");
        }
    });
    button.add_controller(hover);
    button
}

fn children(widget: &gtk::Box) -> Vec<gtk::Widget> {
    let mut children = Vec::new();
    let mut child = widget.first_child();
    while let Some(current) = child {
        child = current.next_sibling();
        children.push(current);
    }
    children
}

fn avatar(color: &str, size: i32) -> gtk::Image {
    let avatar = gtk::Image::from_icon_name("avatar-default-symbolic");
    avatar.set_pixel_size(size);
    avatar.add_css_class("speaker-avatar");
    avatar.add_css_class("turn-avatar");
    avatar.add_css_class(color);
    avatar.set_valign(gtk::Align::Center);
    avatar
}

/// Gives a widget the line menu with these keep items.
pub(super) type LineMenu<'a> = &'a dyn Fn(&gtk::Widget, Vec<(String, KeepChange)>);

/// The card of a line cut into turns: one play button for the whole line,
/// and every turn with its own speaker.
/// `menu` gives a widget the line menu with these keep items; each turn gets
/// its own speaker's.
#[allow(clippy::too_many_arguments)]
pub(super) fn group_card(
    line: &Utterance,
    turns: &[(Utterance, LineKeep)],
    group: Option<&TurnGroup>,
    detail: &SessionDetail,
    audio_path: PathBuf,
    frequent: &Rc<Vec<String>>,
    colors: &Rc<HashMap<String, usize>>,
    menu: LineMenu,
) -> gtk::Box {
    let group = group.cloned().unwrap_or_else(|| TurnGroup {
        source: line.source,
        start_ms: line.start_ms,
        end_ms: line.end_ms,
        turns: turns
            .iter()
            .map(|(turn, _)| crate::types::Turn {
                start_ms: turn.start_ms,
                speaker: None,
            })
            .collect(),
    });
    // A turn may come out as several lines, as when part of it is kept.
    let mut shown: Vec<(usize, Vec<&(Utterance, LineKeep)>)> = Vec::new();
    for turn in turns {
        let index = group.turn_at(turn.0.start_ms);
        match shown.last_mut() {
            Some((last, parts)) if *last == index => parts.push(turn),
            _ => shown.push((index, vec![turn])),
        }
    }
    let names = shown
        .iter()
        .map(|(index, parts)| match group.turns[*index].speaker.clone() {
            Some(name) => TurnName::Named(name),
            None if *index == 0 => TurnName::Line {
                anonymous: is_anonymous(&parts[0].0.speaker),
                name: parts[0].0.speaker.clone(),
            },
            None => TurnName::Unassigned,
        })
        .collect::<Vec<_>>();
    let turn_colors = turn_colors(&names, line.source, colors);

    let card = gtk::Box::new(gtk::Orientation::Vertical, 4);
    card.add_css_class("transcript-card");
    card.add_css_class("turns-card");
    let head = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    let play = gtk::Button::new();
    play.add_css_class("flat");
    play.add_css_class("transcript-icon");
    play.set_valign(gtk::Align::Center);
    set_playback_button(&play, false);
    let playback = detail.playback.clone();
    let row_id = playback.allocate_id();
    let window = detail.window.clone();
    let (start_ms, end_ms) = (group.start_ms, group.end_ms.max(line.end_ms));
    play.connect_clicked(move |button| {
        playback.toggle(row_id, button, &window, &audio_path, start_ms, end_ms);
    });
    head.append(&play);
    // Turns of a hidden source may be left out.
    let count = if shown.len() < group.turns.len() {
        format!("{} of {}", shown.len(), turns_label(group.turns.len()))
    } else {
        turns_label(shown.len())
    };
    let badge = gtk::Label::new(Some(&format!(
        "{count} · {}–{}",
        format_timeline_timestamp(line.start_ms),
        format_timeline_timestamp(line.end_ms)
    )));
    badge.add_css_class("turn-count");
    badge.set_valign(gtk::Align::Center);
    head.append(&badge);
    let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    head.append(&spacer);
    let scissors = scissors_button("Change the cuts…");
    let card_for_cut = card.downgrade();
    let detail_for_cut = detail.clone();
    let line_speaker = names
        .first()
        .and_then(TurnName::label)
        .unwrap_or_default()
        .to_owned();
    let group_for_cut = group.clone();
    let colors_for_cut = colors.clone();
    scissors.connect_clicked(move |_| {
        if let Some(card) = card_for_cut.upgrade() {
            enter_cut_mode(
                &card,
                CutTarget {
                    source: group_for_cut.source,
                    start_ms: group_for_cut.start_ms,
                    end_ms: group_for_cut.end_ms,
                    group: Some(group_for_cut.clone()),
                    line_speaker: line_speaker.clone(),
                    colors: colors_for_cut.clone(),
                },
                &detail_for_cut,
            );
        }
    });
    head.append(&scissors);
    menu(head.upcast_ref(), Vec::new());
    card.append(&head);
    let whole_line = |turn: &Utterance| Utterance {
        start_ms: line.start_ms,
        end_ms: line.end_ms,
        ..turn.clone()
    };

    for (position, ((index, parts), name)) in shown.iter().zip(&names).enumerate() {
        let row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        row.add_css_class("turn");
        if position > 0 {
            row.add_css_class("turn-follows");
        }
        if parts.iter().all(|(_, state)| *state == LineKeep::Hidden) {
            row.add_css_class("outside-line");
        }
        if parts.iter().all(|(part, _)| part.echo) {
            row.add_css_class("echo-row");
        }
        let stripe = gtk::Box::new(gtk::Orientation::Vertical, 0);
        stripe.add_css_class("turn-stripe");
        stripe.add_css_class(&turn_colors[position]);
        row.append(&stripe);
        let body = gtk::Box::new(gtk::Orientation::Vertical, 2);
        body.set_hexpand(true);
        let turn_head = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        let who = who_button(name, &turn_colors[position]);
        let current = match name {
            TurnName::Named(name) => Some(name.clone()),
            _ => None,
        };
        let detail_for_who = detail.clone();
        let frequent = frequent.clone();
        let target = AssignTarget::Turn {
            source: group.source,
            group_start_ms: group.start_ms,
            turn_start_ms: group.turns[*index].start_ms,
        };
        who.connect_clicked(move |_| {
            show_assignment_dialog(&detail_for_who, target, current.as_deref(), &frequent);
        });
        turn_head.append(&who);
        if matches!(name, TurnName::Named(_)) || parts[0].0.locked && *index == 0 {
            let lock = gtk::Image::from_icon_name("changes-prevent-symbolic");
            lock.set_pixel_size(12);
            let why = if matches!(name, TurnName::Named(_)) {
                "Named by you for this turn only"
            } else {
                "Confirmed by you"
            };
            lock.set_tooltip_text(Some(why));
            lock.update_property(&[gtk::accessible::Property::Label(why)]);
            lock.add_css_class("dim-label");
            lock.set_valign(gtk::Align::Center);
            turn_head.append(&lock);
        }
        let time = gtk::Label::new(Some(&format_timeline_timestamp(parts[0].0.start_ms)));
        time.add_css_class("dim-label");
        time.add_css_class("caption");
        time.add_css_class("turn-time");
        time.set_valign(gtk::Align::Center);
        turn_head.append(&time);
        if let Some(LineKeep::Kept { by_speaker, .. }) = parts
            .iter()
            .map(|(_, state)| *state)
            .find(|state| matches!(state, LineKeep::Kept { .. }))
        {
            turn_head.append(&keep::kept_marker(
                by_speaker,
                &parts[0].0.speaker,
                parts[0].0.source,
            ));
        }
        if parts.iter().any(|(part, _)| part.echo) {
            let echo = status_pill("Echo", "idle");
            echo.add_css_class("caption");
            echo.set_tooltip_text(Some("Loudspeaker sound picked up by the microphone"));
            turn_head.append(&echo);
        }
        body.append(&turn_head);
        let text = gtk::Label::new(Some(
            &parts
                .iter()
                .map(|(part, _)| part.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        ));
        text.set_xalign(0.0);
        text.set_wrap(true);
        text.set_wrap_mode(gtk::pango::WrapMode::WordChar);
        text.set_selectable(true);
        body.append(&text);
        row.append(&body);
        // Keeping a line keeps all of the cut line; keeping a speaker, this
        // turn's.
        menu(
            row.upcast_ref(),
            keep::line_actions(parts[0].1, &whole_line(&parts[0].0)),
        );
        card.append(&row);
    }
    card
}

fn who_button(name: &TurnName, color: &str) -> gtk::Button {
    let button = gtk::Button::new();
    button.add_css_class("flat");
    let accessible = match name.label() {
        Some(label) => {
            button.add_css_class("turn-who");
            let content = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            content.append(&avatar(color, 12));
            let text = gtk::Label::new(Some(label));
            text.set_ellipsize(gtk::pango::EllipsizeMode::End);
            text.set_max_width_chars(18);
            if matches!(
                name,
                TurnName::Line {
                    anonymous: true,
                    ..
                }
            ) {
                text.add_css_class("warning-text");
            }
            content.append(&text);
            button.set_child(Some(&content));
            format!("Name this turn, now {label}")
        }
        None => {
            button.add_css_class("turn-who-none");
            button.set_child(Some(&gtk::Label::new(Some("Who?"))));
            "Name this turn, now unassigned".to_owned()
        }
    };
    button.set_valign(gtk::Align::Center);
    button.set_tooltip_text(Some("Who says this?"));
    button.update_property(&[gtk::accessible::Property::Label(&accessible)]);
    button
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(start_ms: u64, text: &str) -> TimedWord {
        TimedWord {
            source: AudioSource::System,
            start_ms,
            end_ms: start_ms + 100,
            text: text.into(),
        }
    }

    #[test]
    fn tokens_join_punctuation_and_words_that_start_together() {
        let words = [
            word(0, " Yes"),
            word(100, ","),
            word(200, " they"),
            word(200, " did"),
            word(300, " "),
            word(400, " great"),
        ];
        assert_eq!(
            tokens(&words, &[]),
            [
                Token {
                    text: "Yes,".into(),
                    start_ms: 0
                },
                Token {
                    text: "they did".into(),
                    start_ms: 200
                },
                Token {
                    text: "great".into(),
                    start_ms: 400
                },
            ]
        );
    }

    fn line(start_ms: u64, speaker: &str, group: Option<&str>) -> Utterance {
        Utterance {
            start_ms,
            end_ms: start_ms + 500,
            source: AudioSource::System,
            speaker_id: "spk_1".into(),
            speaker: speaker.into(),
            text: speaker.to_lowercase(),
            locked: true,
            echo: false,
            group: group.map(str::to_owned),
        }
    }

    #[test]
    fn folded_cut_line_spans_its_turns_and_is_hidden_only_when_all_are() {
        let kept = LineKeep::Kept { by_speaker: true };
        let (lines, states, turns) = fold_groups(
            vec![
                (line(0, "Laura", Some("system-0")), LineKeep::Hidden),
                (line(500, "Me", None), LineKeep::Shown),
                (line(600, "Andrew", Some("system-0")), kept),
            ],
            &HashMap::new(),
        );
        assert_eq!(lines.len(), 2);
        assert_eq!(
            (lines[0].start_ms, lines[0].end_ms, lines[0].text.as_str()),
            (0, 1_100, "laura andrew")
        );
        assert_eq!(lines[0].speaker, "Andrew");
        assert_eq!(states, [kept, LineKeep::Shown]);
        assert_eq!(turns[0].len(), 2);
        assert!(turns[1].is_empty());
    }

    #[test]
    fn punctuation_rendered_in_a_turn_of_its_own_is_a_token_of_its_own() {
        let words = [
            word(1_000, "Hello"),
            word(1_500, "my"),
            word(2_100, "friend"),
            word(3_100, "."),
        ];
        assert_eq!(tokens(&words, &[]).len(), 3);
        let mut cuts = vec![2_000, 3_000];
        let starts = |cuts: &[u64]| {
            tokens(&words, cuts)
                .iter()
                .map(|token| token.start_ms)
                .collect::<Vec<_>>()
        };
        assert_eq!(starts(&cuts), [1_000, 1_500, 2_100, 3_100]);
        // Cutting elsewhere keeps the turn that holds only the full stop.
        cuts.push(1_500);
        assert_eq!(
            saved_cuts(&token_turns(&starts(&cuts), &cuts)),
            (None, vec![1_500, 2_000, 3_000])
        );
    }

    #[test]
    fn a_first_turn_left_without_words_is_not_saved() {
        let starts = [2_100, 3_100];
        let mut cuts = vec![2_000, 3_000];
        let turns = token_turns(&starts, &cuts);
        assert_eq!(saved_cuts(&turns), (Some(2_000), vec![3_000]));
        // Removing the only cut shown leaves one plain line.
        remove_turn(&mut cuts, turns[0], 3_000);
        assert_eq!(
            saved_cuts(&token_turns(&starts, &cuts)),
            (Some(2_000), vec![])
        );
    }

    #[test]
    fn folded_cut_line_starts_at_its_first_turn_when_that_is_not_shown() {
        let ranges = HashMap::from([("system-1000".to_owned(), (1_000, 3_500))]);
        let (lines, _, _) = fold_groups(
            vec![
                (line(2_000, "Me", None), LineKeep::Shown),
                (
                    line(3_000, "Andrew", Some("system-1000")),
                    LineKeep::Kept { by_speaker: true },
                ),
            ],
            &ranges,
        );
        assert_eq!(
            lines
                .iter()
                .map(|line| (line.start_ms, line.end_ms))
                .collect::<Vec<_>>(),
            [(1_000, 3_500), (2_000, 2_500)]
        );
        // A reprocessed word that starts before the group does not move it.
        let (lines, _, _) = fold_groups(
            vec![(line(900, "Laura", Some("system-1000")), LineKeep::Shown)],
            &ranges,
        );
        assert_eq!((lines[0].start_ms, lines[0].end_ms), (1_000, 3_500));
    }

    #[test]
    fn cut_mode_shows_moved_cuts_where_rendering_puts_them_and_removes_them_whole() {
        let starts = [1_000, 2_100, 3_000];
        // A cut reprocessing left before a word still starts that word's turn.
        assert_eq!(
            token_turns(&starts, &[2_000]),
            [None, Some(2_000), Some(2_000)]
        );
        // Two cuts before one word: the later one holds it.
        let mut cuts = vec![2_000, 2_050];
        assert_eq!(
            token_turns(&starts, &cuts),
            [None, Some(2_050), Some(2_050)]
        );
        remove_turn(&mut cuts, None, 2_050);
        assert_eq!(token_turns(&starts, &cuts), [None, None, None]);
        let mut cuts = vec![2_000, 2_900, 3_000];
        remove_turn(&mut cuts, Some(2_000), 3_000);
        assert_eq!(cuts, [2_000]);
    }

    #[test]
    fn person_colors_follow_most_of_their_uncut_lines() {
        let at = |speaker_id: &str, speaker: &str, group: Option<&str>| Utterance {
            speaker_id: speaker_id.into(),
            ..line(0, speaker, group)
        };
        // spk_0 and spk_1 give colours 0 and 1.
        let colors = person_colors(&[
            at("spk_0", "Andrew", None),
            at("spk_0", "andrew", None),
            at("spk_1", "Andrew", None),
            at("spk_1", "Laura", Some("system-0")),
            at("spk_1", "SPEAKER_01", None),
        ]);
        assert_eq!(colors, HashMap::from([("andrew".to_owned(), 0)]));
    }

    #[test]
    fn turn_colors_reuse_each_persons_line_colour() {
        let named = |name: &str| TurnName::Named(name.into());
        let known = HashMap::from([("laura".to_owned(), 0), ("andrew".to_owned(), 2)]);
        let names = [
            TurnName::Line {
                name: "SPEAKER_01".into(),
                anonymous: true,
            },
            named("Laura"),
            TurnName::Unassigned,
            named("Andrew"),
            named("Anna"),
            named("anna"),
        ];
        assert_eq!(
            turn_colors(&names, AudioSource::System, &known),
            [
                "turn-unknown",
                "turn-0",
                "turn-none",
                "turn-2",
                "turn-1",
                "turn-1"
            ]
        );
        assert_eq!(
            turn_colors(&names, AudioSource::Mic, &known),
            [
                "turn-unknown",
                "turn-2",
                "turn-none",
                "turn-2",
                "turn-2",
                "turn-2"
            ]
        );
    }
}
