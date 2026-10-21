use super::timeline_layout::TimelineRow;
use super::{format_duration, format_timeline_timestamp};
use crate::types::{MeetingSpan, Utterance};
use gtk::{gdk, gio, glib, prelude::*};
use gtk4 as gtk;
use std::cell::Cell;
use std::rc::Rc;

pub(super) type SpanSetter = Rc<dyn Fn(MeetingSpan)>;

/// The session's marks, the recording's length (0 when unknown), and how to
/// change the marks.
#[derive(Clone)]
pub(super) struct Marks {
    pub span: MeetingSpan,
    pub recorded_ms: u64,
    pub set: SpanSetter,
}

/// Whether choosing `marked` changes the marks to a meeting that is not empty.
pub(super) fn can_mark(span: MeetingSpan, recorded_ms: u64, marked: MeetingSpan) -> bool {
    marked != span && marked.is_valid() && (recorded_ms == 0 || marked.is_valid_within(recorded_ms))
}

fn row_lines<'a>(row: &TimelineRow, utterances: &'a [Utterance]) -> Vec<&'a Utterance> {
    row.mic
        .iter()
        .chain(&row.system)
        .map(|&index| &utterances[index])
        .collect()
}

/// The first row inside the meeting and the first row after it; the rows
/// before the first fold away as before the meeting, those from the second on
/// as after it.
pub(super) fn fold_bounds(
    rows: &[TimelineRow],
    utterances: &[Utterance],
    span: MeetingSpan,
) -> (usize, usize) {
    let before = |row: &TimelineRow| {
        row_lines(row, utterances)
            .iter()
            .all(|line| span.start_ms.is_some_and(|start| line.start_ms < start))
    };
    let after = |row: &TimelineRow| {
        row_lines(row, utterances)
            .iter()
            .all(|line| span.end_ms.is_some_and(|end| line.start_ms >= end))
    };
    let first_inside = rows.iter().take_while(|row| before(row)).count();
    let after_count = rows[first_inside..]
        .iter()
        .rev()
        .take_while(|row| after(row))
        .count();
    (first_inside, rows.len() - after_count)
}

/// Where the meeting ends when it ends after this row: the row's end, cut
/// back so that a next row starting earlier stays outside.
pub(super) fn row_end_ms(rows: &[TimelineRow], utterances: &[Utterance], index: usize) -> u64 {
    let lines = row_lines(&rows[index], utterances);
    let last_start = lines.iter().map(|line| line.start_ms).max().unwrap_or(0);
    let end = lines.iter().map(|line| line.end_ms).max().unwrap_or(0);
    rows.get(index + 1)
        .map_or(end, |next| end.min(next.start_ms))
        .max(last_start + 1)
}

pub(super) fn fold_lines(rows: &[TimelineRow]) -> usize {
    rows.iter()
        .map(|row| row.mic.len() + row.system.len())
        .sum()
}

pub(super) fn fold_summary(before: bool, duration_ms: u64, lines: usize) -> String {
    format!(
        "{} the meeting · {} · {lines} line{}",
        if before { "Before" } else { "After" },
        format_duration(duration_ms),
        if lines == 1 { "" } else { "s" }
    )
}

/// "1:12 – 46:30 · 45 min 18 s" for the Session panel.
pub(super) fn span_summary(span: MeetingSpan, recorded_ms: u64) -> String {
    format!(
        "{} – {} · {}",
        format_timeline_timestamp(span.start_ms.unwrap_or(0)),
        format_timeline_timestamp(span.end_ms.unwrap_or(recorded_ms)),
        format_duration(span.length_ms(recorded_ms))
    )
}

/// Opens the mark menu on a right-click, Shift+F10 or the Menu key. A
/// right-click on selected text keeps the text's own menu.
pub(super) fn attach_mark_menu(widget: &gtk::Widget, start_ms: u64, end_ms: u64, marks: &Marks) {
    let click = gtk::GestureClick::builder()
        .button(gdk::BUTTON_SECONDARY)
        .propagation_phase(gtk::PropagationPhase::Capture)
        .build();
    let marks_for_click = marks.clone();
    click.connect_pressed(move |gesture, _, x, y| {
        let Some(widget) = gesture.widget() else {
            return;
        };
        if widget
            .pick(x, y, gtk::PickFlags::DEFAULT)
            .and_downcast::<gtk::Label>()
            .is_some_and(|label| label.selection_bounds().is_some())
        {
            return;
        }
        gesture.set_state(gtk::EventSequenceState::Claimed);
        open_mark_menu(&widget, x, y, start_ms, end_ms, &marks_for_click);
    });
    widget.add_controller(click);

    let keys = gtk::ShortcutController::new();
    let marks = marks.clone();
    keys.add_shortcut(gtk::Shortcut::new(
        gtk::ShortcutTrigger::parse_string("<Shift>F10|Menu"),
        Some(gtk::CallbackAction::new(move |widget, _| {
            let x = f64::from(widget.width()) / 2.0;
            let y = f64::from(widget.height()) / 2.0;
            open_mark_menu(widget, x, y, start_ms, end_ms, &marks);
            glib::Propagation::Stop
        })),
    ));
    widget.add_controller(keys);
}

fn open_mark_menu(widget: &gtk::Widget, x: f64, y: f64, start_ms: u64, end_ms: u64, marks: &Marks) {
    let span = marks.span;
    let actions = gio::SimpleActionGroup::new();
    for (name, marked) in [
        (
            "start",
            MeetingSpan {
                start_ms: Some(start_ms),
                ..span
            },
        ),
        (
            "end",
            MeetingSpan {
                end_ms: Some(end_ms),
                ..span
            },
        ),
    ] {
        let action = gio::SimpleAction::new(name, None);
        action.set_enabled(can_mark(span, marks.recorded_ms, marked));
        let set = marks.set.clone();
        action.connect_activate(move |_, _| set(marked));
        actions.add_action(&action);
    }
    widget.insert_action_group("span", Some(&actions));
    let items = gio::Menu::new();
    items.append(Some("Meeting starts here"), Some("span.start"));
    items.append(Some("Meeting ends here"), Some("span.end"));
    let model = gio::Menu::new();
    model.append_section(
        Some(&format!("At {}", format_timeline_timestamp(start_ms))),
        &items,
    );
    let menu = gtk::PopoverMenu::from_model(Some(&model));
    menu.set_has_arrow(false);
    menu.set_halign(gtk::Align::Start);
    menu.set_parent(widget);
    menu.set_pointing_to(Some(&gdk::Rectangle::new(x as i32, y as i32, 1, 1)));
    menu.connect_closed(|menu| {
        let menu = menu.clone();
        glib::idle_add_local_once(move || menu.unparent());
    });
    menu.popup();
}

/// The blue line across the timeline where the meeting starts or ends, with a
/// button that moves that end back to the edge of the recording.
pub(super) fn meeting_flag(starts: bool, at_ms: u64, remove: impl Fn() + 'static) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    row.set_margin_top(6);
    row.set_margin_bottom(6);
    let line = || {
        let line = gtk::Separator::new(gtk::Orientation::Horizontal);
        line.add_css_class("meeting-flag-line");
        line.set_hexpand(true);
        line.set_valign(gtk::Align::Center);
        line
    };
    row.append(&line());
    let flag = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    flag.add_css_class("meeting-flag");
    let icon = gtk::Image::from_icon_name(if starts {
        "go-first-symbolic"
    } else {
        "go-last-symbolic"
    });
    icon.set_pixel_size(14);
    flag.append(&icon);
    let text = format!(
        "Meeting {} {}",
        if starts { "starts" } else { "ends" },
        format_timeline_timestamp(at_ms)
    );
    flag.append(&gtk::Label::new(Some(&text)));
    let button = gtk::Button::from_icon_name("window-close-symbolic");
    button.add_css_class("flat");
    button.add_css_class("circular");
    let label = if starts {
        "Remove the start mark: the meeting starts with the recording"
    } else {
        "Remove the end mark: the meeting runs to the end of the recording"
    };
    button.set_tooltip_text(Some(label));
    button.update_property(&[gtk::accessible::Property::Label(label)]);
    button.connect_clicked(move |_| remove());
    flag.append(&button);
    row.append(&flag);
    row.append(&line());
    row
}

/// The lines outside the meeting, folded into a pill that shows them greyed
/// on a shaded band.
pub(super) fn outside_fold(
    before: bool,
    duration_ms: u64,
    lines: usize,
    rows: gtk::Box,
    expanded: bool,
    toggled: impl Fn(bool) + 'static,
) -> gtk::Box {
    let band = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let pill = gtk::Box::new(gtk::Orientation::Horizontal, 4);
    pill.add_css_class("meeting-fold");
    pill.set_halign(gtk::Align::Center);
    pill.set_margin_top(6);
    pill.set_margin_bottom(6);
    let summary = gtk::Label::new(Some(&fold_summary(before, duration_ms, lines)));
    summary.add_css_class("dim-label");
    pill.append(&summary);
    let toggle = gtk::Button::new();
    toggle.add_css_class("flat");
    toggle.add_css_class("meeting-fold-toggle");
    pill.append(&toggle);
    band.append(&pill);
    band.append(&rows);
    let where_ = if before { "before" } else { "after" };
    let show = {
        let (band, toggle) = (band.clone(), toggle.clone());
        move |expanded: bool| {
            rows.set_visible(expanded);
            if expanded {
                band.add_css_class("outside-meeting");
            } else {
                band.remove_css_class("outside-meeting");
            }
            let action = if expanded { "Hide" } else { "Show" };
            toggle.set_label(action);
            let label = format!("{action} the lines {where_} the meeting");
            toggle.set_tooltip_text(Some(&label));
            toggle.update_property(&[gtk::accessible::Property::Label(&label)]);
        }
    };
    show(expanded);
    let state = Cell::new(expanded);
    toggle.connect_clicked(move |_| {
        state.set(!state.get());
        show(state.get());
        toggled(state.get());
    });
    band
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioSource;

    fn line(source: AudioSource, start_ms: u64, end_ms: u64) -> Utterance {
        Utterance {
            start_ms,
            end_ms,
            source,
            speaker_id: "spk_0".into(),
            speaker: "Laura".into(),
            text: "text".into(),
            locked: false,
            echo: false,
        }
    }

    fn row(start_ms: u64, mic: Vec<usize>, system: Vec<usize>) -> TimelineRow {
        TimelineRow {
            start_ms,
            mic,
            system,
        }
    }

    #[test]
    fn folds_keep_only_whole_rows_outside_the_meeting() {
        let lines = [
            line(AudioSource::Mic, 0, 1_000),
            line(AudioSource::System, 2_000, 3_000),
            line(AudioSource::Mic, 4_000, 5_000),
            line(AudioSource::System, 4_200, 5_000),
            line(AudioSource::Mic, 6_000, 7_000),
        ];
        let rows = [
            row(0, vec![0], vec![]),
            row(2_000, vec![], vec![1]),
            row(4_000, vec![2], vec![3]),
            row(6_000, vec![4], vec![]),
        ];
        let span = |start_ms, end_ms| MeetingSpan { start_ms, end_ms };
        assert_eq!(fold_bounds(&rows, &lines, span(None, None)), (0, 4));
        assert_eq!(fold_bounds(&rows, &lines, span(Some(2_000), None)), (1, 4));
        assert_eq!(fold_bounds(&rows, &lines, span(None, Some(5_000))), (0, 3));
        // A row the start mark cuts through stays unfolded.
        assert_eq!(
            fold_bounds(&rows, &lines, span(Some(4_100), Some(6_000))),
            (2, 3)
        );
    }

    #[test]
    fn end_mark_keeps_an_early_next_row_outside() {
        let lines = [
            line(AudioSource::Mic, 0, 3_000),
            line(AudioSource::System, 2_500, 9_000),
            line(AudioSource::Mic, 10_000, 11_000),
        ];
        let rows = [
            row(0, vec![0], vec![]),
            row(2_500, vec![], vec![1]),
            row(10_000, vec![2], vec![]),
        ];
        assert_eq!(row_end_ms(&rows, &lines, 0), 2_500);
        assert_eq!(row_end_ms(&rows, &lines, 1), 9_000);
        assert_eq!(row_end_ms(&rows, &lines, 2), 11_000);
    }

    #[test]
    fn marks_that_leave_no_meeting_or_change_nothing_are_refused() {
        let span = |start_ms, end_ms| MeetingSpan { start_ms, end_ms };
        let current = span(Some(1_000), None);
        assert!(can_mark(current, 60_000, span(Some(2_000), None)));
        assert!(!can_mark(current, 60_000, current));
        assert!(!can_mark(current, 60_000, span(Some(1_000), Some(1_000))));
        assert!(!can_mark(
            MeetingSpan::default(),
            60_000,
            span(None, Some(0))
        ));
        assert!(!can_mark(
            MeetingSpan::default(),
            60_000,
            span(Some(60_000), None)
        ));
        assert!(can_mark(
            MeetingSpan::default(),
            0,
            span(Some(60_000), None)
        ));
    }

    #[test]
    fn summaries_name_the_span_and_its_length() {
        assert_eq!(
            fold_summary(true, 72_000, 5),
            "Before the meeting · 1 min 12 s · 5 lines"
        );
        assert_eq!(
            fold_summary(false, 32_000, 1),
            "After the meeting · 32 s · 1 line"
        );
        assert_eq!(
            span_summary(
                MeetingSpan {
                    start_ms: Some(72_000),
                    end_ms: Some(2_790_000)
                },
                2_822_000
            ),
            "1:12 – 46:30 · 45 min 18 s"
        );
        assert_eq!(
            span_summary(
                MeetingSpan {
                    start_ms: None,
                    end_ms: Some(60_000)
                },
                2_822_000
            ),
            "0:00 – 1:00 · 1 min 00 s"
        );
    }
}
