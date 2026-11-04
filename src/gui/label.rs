//! Labeling who speaks when in a transcript line's audio, saved as a
//! diarization test case: the button on every bubble and the labeling view.
//! It never changes the transcript.

use super::player::{display_norm, rgb, rounded};
use super::{SessionDetail, format_timeline_timestamp, open_path, show_error};
use crate::audio::stretch::Slowed;
use crate::audio::track;
use crate::merge::process;
use crate::session::{self, Session};
use crate::testcase::{
    self, Audio, Bar, CONTEXT_STEP_MS, Draft, Inputs, Labels, LineRef, MAX_CONTEXT_MS,
    MAX_SPEAKERS, MIN_BAR_MS, NOT_SURE, Track,
};
use crate::types::{AudioSource, SAMPLE_RATE};
use adw::prelude::*;
use gtk::{cairo, gdk, glib};
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, Read};
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// The playback controller's id for the labeling view.
const LABEL_ID: u64 = u64::MAX - 1;
const NUDGE_MS: u64 = 50;
const SLOW_RATE: f64 = 0.5;
const EDGE_GRAB: f64 = 6.0;
const CLOSE_ZONE: f64 = 16.0;
const LEVEL_MS: u64 = 10;
const LANE_HEIGHT: i32 = 28;
const ROW_SPACING: i32 = 4;
const SAMPLES_PER_MS: usize = SAMPLE_RATE as usize / 1000;

type Rgb = (f64, f64, f64);
const COLORS: [Rgb; 9] = [
    (0.102, 0.373, 0.706),
    (0.380, 0.208, 0.514),
    (0.149, 0.635, 0.412),
    (0.776, 0.275, 0.0),
    (0.647, 0.114, 0.176),
    (0.110, 0.498, 0.569),
    (0.596, 0.416, 0.267),
    (0.525, 0.369, 0.235),
    (0.369, 0.361, 0.392),
];
const WAVE_COLOR: Rgb = (0.208, 0.518, 0.894);
const PLAYHEAD_COLOR: Rgb = (0.878, 0.106, 0.141);
const ACCENT: Rgb = (0.208, 0.518, 0.894);

/// The session's saved cases by line, for the buttons.
pub(super) type SavedCases = Rc<HashMap<LineRef, testcase::Saved>>;

pub(super) fn saved_cases(session: &Session) -> SavedCases {
    Rc::new(
        testcase::saved_cases(session)
            .into_iter()
            .map(|saved| (saved.line, saved))
            .collect(),
    )
}

/// The label button of a bubble covering `line`, said by `speaker`.
pub(super) fn label_button(
    detail: &SessionDetail,
    line: LineRef,
    speaker: String,
    saved: &SavedCases,
) -> gtk::Button {
    let button = gtk::Button::new();
    button.add_css_class("flat");
    button.add_css_class("transcript-icon");
    button.set_valign(gtk::Align::Center);
    show_saved(&button, saved.get(&line).map(|saved| saved.speakers));
    let detail = detail.clone();
    button.connect_clicked(move |button| open(&detail, line, &speaker, button));
    button
}

fn show_saved(button: &gtk::Button, speakers: Option<usize>) {
    let tooltip = match speakers {
        None => {
            button.set_icon_name("bookmark-new-symbolic");
            button.remove_css_class("accent");
            "Label speakers for a test case".to_owned()
        }
        Some(count) => {
            button.set_icon_name("user-bookmarks-symbolic");
            button.add_css_class("accent");
            format!(
                "Labeled with {count} speaker{} for a test case. Edit the labels",
                if count == 1 { "" } else { "s" }
            )
        }
    };
    button.set_tooltip_text(Some(&tooltip));
    button.update_property(&[gtk::accessible::Property::Label(&tooltip)]);
}

#[derive(Clone, Copy)]
struct Clock {
    started: Instant,
    from_ms: u64,
    until_ms: u64,
    rate: f64,
}

impl Clock {
    fn position(&self) -> u64 {
        let elapsed = self.started.elapsed().as_secs_f64() * 1000.0 * self.rate;
        self.from_ms
            .saturating_add(elapsed as u64)
            .min(self.until_ms)
    }
}

#[derive(Clone, Copy)]
enum Drag {
    Seek { resume: bool },
    New { id: u64, anchor_ms: u64 },
    Move { id: u64, grab_ms: u64, bar: Bar },
    Resize { id: u64, left: bool, bar: Bar },
}

/// What a point on a lane is over.
#[derive(Clone, Copy, PartialEq)]
enum Hit {
    Remove(u64),
    Edge(u64, bool),
    Body(u64),
    Empty,
}

/// The clip as shown: its samples and level per `LEVEL_MS`.
struct View {
    key: (Track, u64),
    samples: Rc<Vec<f32>>,
    levels: Vec<track::Level>,
    norm: f32,
}

struct State {
    session: Session,
    title: String,
    line: LineRef,
    speaker: String,
    recorded_ms: u64,
    labels: Labels,
    next_id: u64,
    audio: Option<Rc<Audio>>,
    error: Option<String>,
    inputs: Rc<Inputs>,
    view: Option<View>,
    position_ms: u64,
    playing: Option<Clock>,
    slow: bool,
    /// Number keys down, to tell a held key from its repeats.
    down: [bool; 10],
    /// The bar each held number key grows.
    held: [Option<u64>; 10],
    selected: Option<u64>,
    hover: Option<(u8, f64)>,
    drag: Option<(Drag, f64)>,
    dirty: bool,
    tick: u64,
}

impl State {
    fn clip(&self) -> (u64, u64) {
        testcase::clip_range(self.line, self.labels.context_ms, self.recorded_ms)
    }

    fn position(&self) -> u64 {
        self.playing
            .map_or(self.position_ms, |clock| clock.position())
    }

    fn bar(&self, id: u64) -> Option<Bar> {
        self.labels.bars.iter().find(|bar| bar.id == id).copied()
    }

    fn set_bar(&mut self, id: u64, change: impl FnOnce(&mut Bar)) {
        if let Some(bar) = self.labels.bars.iter_mut().find(|bar| bar.id == id) {
            change(bar);
        }
    }

    fn add_bar(&mut self, lane: u8, start_ms: u64) -> u64 {
        self.next_id += 1;
        self.labels.bars.push(Bar {
            lane,
            start_ms,
            end_ms: start_ms,
            id: self.next_id,
        });
        self.next_id
    }

    fn remove_bar(&mut self, id: u64) {
        self.labels.bars.retain(|bar| bar.id != id);
        if self.selected == Some(id) {
            self.selected = None;
        }
        for held in &mut self.held {
            if *held == Some(id) {
                *held = None;
            }
        }
    }

    /// A drawn or tapped bar is done: too short ones go, the rest join their lane.
    fn finish_bar(&mut self, id: u64) {
        match self.bar(id) {
            Some(bar) if bar.end_ms.saturating_sub(bar.start_ms) < MIN_BAR_MS => {
                self.remove_bar(id)
            }
            Some(_) => {
                testcase::merge_lane(&mut self.labels.bars, id);
                self.dirty = true;
            }
            None => {}
        }
    }

    fn view(&mut self) -> Option<&View> {
        let key = (self.labels.track, self.labels.context_ms);
        if self.view.as_ref().is_none_or(|view| view.key != key) {
            let audio = self.audio.as_ref()?;
            let samples = audio.clip(self.labels.track, self.clip());
            let levels = samples
                .chunks(LEVEL_MS as usize * SAMPLES_PER_MS)
                .map(|chunk| track::Level {
                    peak: chunk
                        .iter()
                        .fold(0.0f32, |peak, sample| peak.max(sample.abs())),
                    rms: (chunk.iter().map(|sample| sample * sample).sum::<f32>()
                        / chunk.len() as f32)
                        .sqrt(),
                })
                .collect::<Vec<_>>();
            let norm = display_norm(&levels);
            self.view = Some(View {
                key,
                samples: Rc::new(samples),
                levels,
                norm,
            });
        }
        self.view.as_ref()
    }

    fn hit(&self, lane: u8, x: f64, width: f64) -> Hit {
        let clip = self.clip();
        let mut found = Hit::Empty;
        for bar in self.labels.bars.iter().filter(|bar| bar.lane == lane) {
            let from = x_of(bar.start_ms.max(clip.0), clip, width);
            let to = x_of(bar.end_ms.min(clip.1), clip, width);
            if to <= from || x < from - EDGE_GRAB || x > to + EDGE_GRAB {
                continue;
            }
            found = if to - from >= 2.0 * CLOSE_ZONE && x >= to - CLOSE_ZONE && x < to - EDGE_GRAB {
                Hit::Remove(bar.id)
            } else if (x - from).abs() <= EDGE_GRAB && (x - from).abs() <= (x - to).abs() {
                Hit::Edge(bar.id, true)
            } else if (x - to).abs() <= EDGE_GRAB {
                Hit::Edge(bar.id, false)
            } else {
                Hit::Body(bar.id)
            };
        }
        found
    }
}

/// The view's widgets and state.
struct Inner {
    state: RefCell<State>,
    dialog: adw::Dialog,
    title: adw::WindowTitle,
    /// The rows that draw the clip; `bars` are the lanes and the overlap row.
    rows: Vec<gtk::DrawingArea>,
    bars: Vec<gtk::DrawingArea>,
    playhead: gtk::DrawingArea,
    lane_rows: Vec<[gtk::Widget; 2]>,
    add_lane: gtk::Button,
    play: gtk::Button,
    position: gtk::Label,
    length: gtk::Label,
    slow: gtk::ToggleButton,
    context: gtk::Label,
    less_context: gtk::Button,
    more_context: gtk::Button,
    tracks: [gtk::ToggleButton; 3],
    save: gtk::Button,
    note: gtk::Entry,
    detail: SessionDetail,
    button: glib::WeakRef<gtk::Button>,
}

/// The open view. Callbacks on its own widgets hold a [`WeakEditor`], so a
/// closed view, with its audio, is freed.
#[derive(Clone)]
struct Editor(Rc<Inner>);

impl std::ops::Deref for Editor {
    type Target = Inner;

    fn deref(&self) -> &Inner {
        &self.0
    }
}

#[derive(Clone)]
struct WeakEditor(Weak<Inner>);

impl WeakEditor {
    fn upgrade(&self) -> Option<Editor> {
        self.0.upgrade().map(Editor)
    }
}

/// Takes the open view out of a [`WeakEditor`], or returns (`$default`).
macro_rules! open_or {
    ($editor:ident) => {
        let Some($editor) = $editor.upgrade() else {
            return;
        };
    };
    ($editor:ident, $default:expr) => {
        let Some($editor) = $editor.upgrade() else {
            return $default;
        };
    };
}

fn open(detail: &SessionDetail, line: LineRef, speaker: &str, button: &gtk::Button) {
    let Some(dir) = detail.selected.borrow().clone() else {
        return;
    };
    let opened = Session::open(&dir).and_then(|session| {
        let manifest = session.read_manifest()?;
        Ok((process::recorded_duration_ms(&session, &manifest), session))
    });
    let (recorded_ms, session) = match opened {
        Ok(opened) => opened,
        Err(error) => {
            show_error(
                &detail.window,
                "Could not open the line",
                &error.to_string(),
            );
            return;
        }
    };
    let saved = testcase::saved_cases(&session)
        .into_iter()
        .find(|saved| saved.line == line);
    let labels = match saved
        .as_ref()
        .map(|saved| testcase::reopen(&saved.folder, line, recorded_ms))
    {
        Some(Ok(labels)) => labels,
        Some(Err(error)) => {
            show_error(
                &detail.window,
                "Could not read the saved labels",
                &format!("{error}. Labeling starts over; saving replaces the saved case."),
            );
            Labels::new(line.source)
        }
        None => Labels::new(line.source),
    };
    detail.playback.stop();
    let editor = Editor::new(detail, session, line, speaker, recorded_ms, labels, button);
    editor.load();
    editor.dialog.present(Some(&detail.window));
}

fn key_hint(text: &str) -> gtk::Label {
    let hint = gtk::Label::new(Some(text));
    hint.add_css_class("label-key");
    hint.set_valign(gtk::Align::Center);
    hint
}

/// A row's name; a speaker's (`key` hint and `dot` colour) stands out.
fn row_label(text: &str, dot: Option<&str>, key: Option<&str>) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    row.set_size_request(118, -1);
    if let Some(class) = dot {
        let dot = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        dot.add_css_class("lane-dot");
        dot.add_css_class(class);
        dot.set_valign(gtk::Align::Center);
        row.append(&dot);
    }
    let label = gtk::Label::new(Some(text));
    label.set_xalign(0.0);
    if dot.is_some_and(|class| class != "lane-unsure") {
        label.add_css_class("heading");
    } else {
        label.add_css_class("dim-label");
    }
    row.append(&label);
    if let Some(key) = key {
        row.append(&key_hint(key));
    }
    row
}

fn area(height: i32, label: &str) -> gtk::DrawingArea {
    let area = gtk::DrawingArea::builder()
        .hexpand(true)
        .content_height(height)
        .build();
    area.update_property(&[gtk::accessible::Property::Label(label)]);
    area
}

impl Editor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        detail: &SessionDetail,
        session: Session,
        line: LineRef,
        speaker: &str,
        recorded_ms: u64,
        labels: Labels,
        button: &gtk::Button,
    ) -> Self {
        let dialog = adw::Dialog::builder()
            .title("Label speakers")
            .content_width(1180)
            .can_close(false)
            .build();
        let header = adw::HeaderBar::builder()
            .show_start_title_buttons(false)
            .show_end_title_buttons(false)
            .build();
        let cancel = gtk::Button::with_label("Cancel");
        header.pack_start(&cancel);
        let save = gtk::Button::with_label("Save test case");
        save.add_css_class("suggested-action");
        save.set_sensitive(false);
        header.pack_end(&save);
        let title = adw::WindowTitle::new("Label speakers", "");
        header.set_title_widget(Some(&title));

        let body = gtk::Box::new(gtk::Orientation::Vertical, 10);
        body.set_margin_top(12);
        body.set_margin_bottom(10);
        body.set_margin_start(14);
        body.set_margin_end(14);
        let tools = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        let play = gtk::Button::from_icon_name("media-playback-start-symbolic");
        play.add_css_class("circular");
        play.add_css_class("suggested-action");
        tools.append(&play);
        let position = gtk::Label::new(Some("0:00.0"));
        position.add_css_class("heading");
        position.add_css_class("player-clock");
        tools.append(&position);
        let length = gtk::Label::new(None);
        length.add_css_class("dim-label");
        length.add_css_class("player-clock");
        tools.append(&length);
        let slow = gtk::ToggleButton::with_label("Speed 1×");
        slow.set_tooltip_text(Some("Play at half speed, at the same pitch"));
        slow.set_margin_start(6);
        tools.append(&slow);
        let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        spacer.set_hexpand(true);
        tools.append(&spacer);
        let context_heading = gtk::Label::new(Some("Context"));
        context_heading.add_css_class("dim-label");
        context_heading.add_css_class("caption");
        tools.append(&context_heading);
        let less_context = gtk::Button::with_label("−");
        less_context.set_tooltip_text(Some("Less audio around the line"));
        less_context.update_property(&[gtk::accessible::Property::Label("Less context")]);
        let context = gtk::Label::new(None);
        context.add_css_class("player-clock");
        let more_context = gtk::Button::with_label("+");
        more_context.set_tooltip_text(Some("More audio around the line"));
        more_context.update_property(&[gtk::accessible::Property::Label("More context")]);
        tools.append(&less_context);
        tools.append(&context);
        tools.append(&more_context);
        let track_box = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        track_box.add_css_class("linked");
        track_box.set_margin_start(6);
        let tracks = [
            ("System", "Label the system audio"),
            ("Mic", "Label the microphone audio"),
            ("Both", "Label both tracks mixed"),
        ]
        .map(|(name, tooltip)| {
            let toggle = gtk::ToggleButton::with_label(name);
            toggle.set_tooltip_text(Some(tooltip));
            track_box.append(&toggle);
            toggle
        });
        tracks[1].set_group(Some(&tracks[0]));
        tracks[2].set_group(Some(&tracks[0]));
        tools.append(&track_box);
        body.append(&tools);

        let grid = gtk::Grid::builder()
            .column_spacing(8)
            .row_spacing(ROW_SPACING)
            .build();
        let ruler = area(18, "Time ruler. Click or drag to move the playhead.");
        ruler.set_cursor_from_name(Some("pointer"));
        let wave = area(64, "Waveform. Click or drag to move the playhead.");
        wave.set_cursor_from_name(Some("pointer"));
        let words = area(46, "Words of the transcript");
        grid.attach(&row_label("", None, None), 0, 0, 1, 1);
        grid.attach(&ruler, 1, 0, 1, 1);
        grid.attach(&row_label("Audio", None, None), 0, 1, 1, 1);
        grid.attach(&wave, 1, 1, 1, 1);
        grid.attach(&row_label("Words", None, None), 0, 2, 1, 1);
        grid.attach(&words, 1, 2, 1, 1);
        let rows = vec![ruler.clone(), wave.clone(), words.clone()];
        let mut lane_areas = Vec::new();
        let mut lane_rows = Vec::new();
        for lane in 1..=MAX_SPEAKERS {
            let label = row_label(
                &format!("Speaker {lane}"),
                Some(&format!("lane-{lane}")),
                Some(&lane.to_string()),
            );
            let lane_area = area(
                LANE_HEIGHT,
                &format!(
                    "Speaker {lane}. Drag to mark when they speak, or hold {lane} while playing."
                ),
            );
            grid.attach(&label, 0, 2 + i32::from(lane), 1, 1);
            grid.attach(&lane_area, 1, 2 + i32::from(lane), 1, 1);
            lane_rows.push([label.upcast(), lane_area.clone().upcast()]);
            lane_areas.push((lane, lane_area));
        }
        let unsure_row = 3 + i32::from(MAX_SPEAKERS);
        let unsure = area(
            LANE_HEIGHT,
            "Not sure. Drag to mark a stretch left out of scoring, or hold 0 while playing.",
        );
        grid.attach(
            &row_label("Not sure", Some("lane-unsure"), Some("0")),
            0,
            unsure_row,
            1,
            1,
        );
        grid.attach(&unsure, 1, unsure_row, 1, 1);
        lane_areas.push((NOT_SURE, unsure));
        let add_lane = gtk::Button::with_label("+ Add speaker");
        add_lane.add_css_class("add-speaker");
        add_lane.set_halign(gtk::Align::Start);
        grid.attach(&add_lane, 1, unsure_row + 1, 1, 1);
        let overlap = area(12, "Where two or more speakers talk at once");
        grid.attach(&row_label("Overlap", None, None), 0, unsure_row + 2, 1, 1);
        grid.attach(&overlap, 1, unsure_row + 2, 1, 1);
        let bars = lane_areas
            .iter()
            .map(|(_, area)| area.clone())
            .chain([overlap.clone()])
            .collect::<Vec<_>>();
        let playhead = gtk::DrawingArea::builder()
            .can_target(false)
            .can_focus(false)
            .build();
        let layers = gtk::Overlay::builder().child(&grid).build();
        layers.add_overlay(&playhead);
        body.append(
            &gtk::ScrolledWindow::builder()
                .hscrollbar_policy(gtk::PolicyType::Never)
                .vexpand(true)
                .propagate_natural_height(true)
                .child(&layers)
                .build(),
        );
        let help = gtk::Label::new(None);
        help.set_markup(
            "<b>Space</b> play/pause · hold <b>1</b>–<b>9</b> while playing to mark a speaker, hold <b>0</b> for <i>Not sure</i> · \
             a number key starts playback · drag on a lane to mark, drag a bar's edge or middle to change it · \
             click or drag the waveform to seek · <b>Delete</b> removes the selected bar · \
             <b>←</b> <b>→</b> nudge the selected bar's edge nearest the playhead by 50 ms · <b>Ctrl+S</b> saves",
        );
        help.set_wrap(true);
        help.set_xalign(0.0);
        help.add_css_class("dim-label");
        help.add_css_class("caption");
        body.append(&help);

        let footer = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        footer.add_css_class("label-footer");
        let note_label = gtk::Label::new(Some("Note"));
        note_label.add_css_class("dim-label");
        footer.append(&note_label);
        let note = gtk::Entry::builder()
            .hexpand(true)
            .placeholder_text("Optional, e.g. P2 and P3 laugh over each other at the end")
            .text(&labels.note)
            .build();
        note.update_property(&[gtk::accessible::Property::Label("Note")]);
        footer.append(&note);

        let toolbar = adw::ToolbarView::new();
        toolbar.add_top_bar(&header);
        toolbar.set_content(Some(&body));
        toolbar.add_bottom_bar(&footer);
        dialog.set_child(Some(&toolbar));

        let title_text = session.title();
        let next_id = labels.bars.iter().map(|bar| bar.id).max().unwrap_or(0);
        let editor = Self(Rc::new(Inner {
            state: RefCell::new(State {
                session,
                title: title_text,
                line,
                speaker: speaker.to_owned(),
                recorded_ms,
                position_ms: 0,
                labels,
                next_id,
                audio: None,
                error: None,
                inputs: Rc::default(),
                view: None,
                playing: None,
                slow: false,
                down: [false; 10],
                held: [None; 10],
                selected: None,
                hover: None,
                drag: None,
                dirty: false,
                tick: 0,
            }),
            dialog,
            title,
            rows,
            bars,
            playhead,
            lane_rows,
            add_lane,
            play,
            position,
            length,
            slow,
            context,
            less_context,
            more_context,
            tracks,
            save,
            note,
            detail: detail.clone(),
            button: button.downgrade(),
        }));
        {
            let mut state = editor.state.borrow_mut();
            state.position_ms = state.clip().0;
        }
        editor.wire(&cancel, &ruler, &wave, &words, &lane_areas, &overlap);
        editor.refresh();
        editor
    }

    fn downgrade(&self) -> WeakEditor {
        WeakEditor(Rc::downgrade(&self.0))
    }

    /// Wires every widget to the editor. Only the `closed` handler holds the
    /// editor itself, until the dialog closes.
    #[allow(clippy::too_many_arguments)]
    fn wire(
        &self,
        cancel: &gtk::Button,
        ruler: &gtk::DrawingArea,
        wave: &gtk::DrawingArea,
        words: &gtk::DrawingArea,
        lanes: &[(u8, gtk::DrawingArea)],
        overlap: &gtk::DrawingArea,
    ) {
        type Draw =
            fn(&Editor, &gtk::DrawingArea, &cairo::Context, f64, f64) -> Result<(), cairo::Error>;
        let rows: [(&gtk::DrawingArea, Draw); 4] = [
            (ruler, Editor::draw_ruler),
            (wave, Editor::draw_wave),
            (words, Editor::draw_words),
            (overlap, Editor::draw_overlap),
        ];
        for (area, draw) in rows {
            let editor = self.downgrade();
            area.set_draw_func(move |area, cr, width, height| {
                open_or!(editor);
                let _ = draw(&editor, area, cr, f64::from(width), f64::from(height));
            });
        }
        let editor = self.downgrade();
        let (ruler_ref, overlap_ref) = (ruler.downgrade(), overlap.downgrade());
        self.playhead.set_draw_func(move |layer, cr, _, _| {
            open_or!(editor);
            let (Some(ruler), Some(overlap)) = (ruler_ref.upgrade(), overlap_ref.upgrade()) else {
                return;
            };
            let _ = editor.draw_playhead(layer, cr, &ruler, &overlap);
        });
        for area in [ruler, wave] {
            let drag = gtk::GestureDrag::new();
            let editor = self.downgrade();
            drag.connect_drag_begin(move |gesture, x, _| {
                open_or!(editor);
                editor.seek_begin(x, widget_width(gesture));
            });
            let editor = self.downgrade();
            drag.connect_drag_update(move |gesture, dx, _| {
                open_or!(editor);
                editor.drag_update(dx, widget_width(gesture));
            });
            let editor = self.downgrade();
            drag.connect_drag_end(move |_, _, _| {
                open_or!(editor);
                editor.drag_end();
            });
            area.add_controller(drag);
        }
        for (lane, area) in lanes {
            let lane = *lane;
            let editor = self.downgrade();
            area.set_draw_func(move |area, cr, width, height| {
                open_or!(editor);
                let _ = editor.draw_lane(lane, area, cr, f64::from(width), f64::from(height));
            });
            let drag = gtk::GestureDrag::new();
            let editor = self.downgrade();
            drag.connect_drag_begin(move |gesture, x, _| {
                open_or!(editor);
                editor.lane_begin(lane, x, widget_width(gesture));
            });
            let editor = self.downgrade();
            drag.connect_drag_update(move |gesture, dx, _| {
                open_or!(editor);
                editor.drag_update(dx, widget_width(gesture));
            });
            let editor = self.downgrade();
            drag.connect_drag_end(move |_, _, _| {
                open_or!(editor);
                editor.drag_end();
            });
            area.add_controller(drag);
            let motion = gtk::EventControllerMotion::new();
            let editor = self.downgrade();
            motion.connect_motion(move |controller, x, _| {
                open_or!(editor);
                let Some(area) = controller.widget() else {
                    return;
                };
                let hit = {
                    let mut state = editor.state.borrow_mut();
                    state.hover = Some((lane, x));
                    state.hit(lane, x, f64::from(area.width()))
                };
                area.set_cursor_from_name(Some(match hit {
                    Hit::Remove(_) => "pointer",
                    Hit::Edge(..) => "ew-resize",
                    Hit::Body(_) => "grab",
                    Hit::Empty => "crosshair",
                }));
                area.queue_draw();
            });
            let editor = self.downgrade();
            motion.connect_leave(move |controller| {
                open_or!(editor);
                editor.state.borrow_mut().hover = None;
                if let Some(area) = controller.widget() {
                    area.queue_draw();
                }
            });
            area.add_controller(motion);
        }

        let keys = gtk::EventControllerKey::new();
        keys.set_propagation_phase(gtk::PropagationPhase::Capture);
        let editor = self.downgrade();
        keys.connect_key_pressed(move |_, key, _, modifiers| {
            open_or!(editor, glib::Propagation::Proceed);
            editor.key_pressed(key, modifiers)
        });
        let editor = self.downgrade();
        keys.connect_key_released(move |_, key, _, _| {
            open_or!(editor);
            editor.key_released(key);
        });
        self.dialog.add_controller(keys);

        let editor = self.downgrade();
        self.play.connect_clicked(move |_| {
            open_or!(editor);
            editor.toggle_play();
        });
        let editor = self.downgrade();
        self.slow.connect_toggled(move |toggle| {
            open_or!(editor);
            let restart = {
                let mut state = editor.state.borrow_mut();
                state.slow = toggle.is_active();
                state.playing.is_some()
            };
            toggle.set_label(if toggle.is_active() {
                "Speed ½×"
            } else {
                "Speed 1×"
            });
            if restart {
                editor.start_playing();
            }
        });
        for (step, button) in [
            (-(CONTEXT_STEP_MS as i64), &self.less_context),
            (CONTEXT_STEP_MS as i64, &self.more_context),
        ] {
            let editor = self.downgrade();
            button.connect_clicked(move |_| {
                open_or!(editor);
                let context = editor.state.borrow().labels.context_ms;
                editor.change_view(|labels| {
                    labels.context_ms = context.saturating_add_signed(step).min(MAX_CONTEXT_MS);
                });
            });
        }
        for (toggle, track) in self
            .tracks
            .iter()
            .zip([Track::System, Track::Mic, Track::Both])
        {
            let editor = self.downgrade();
            toggle.connect_toggled(move |toggle| {
                open_or!(editor);
                if toggle.is_active() && editor.state.borrow().labels.track != track {
                    editor.change_view(|labels| labels.track = track);
                }
            });
        }
        let editor = self.downgrade();
        self.add_lane.connect_clicked(move |_| {
            open_or!(editor);
            {
                let mut state = editor.state.borrow_mut();
                state.labels.lanes = (state.labels.lanes + 1).min(MAX_SPEAKERS);
                state.dirty = true;
            }
            // The dialog keeps the height it opened with; grow it by the new
            // lane, as far as the window allows.
            if let Some(content) = editor.dialog.child() {
                editor
                    .dialog
                    .set_content_height(content.height() + LANE_HEIGHT + ROW_SPACING);
            }
            editor.refresh();
        });
        let editor = self.downgrade();
        self.note.connect_changed(move |_| {
            open_or!(editor);
            editor.state.borrow_mut().dirty = true;
        });
        let editor = self.downgrade();
        self.save.connect_clicked(move |_| {
            open_or!(editor);
            editor.save();
        });
        cancel.connect_clicked(|cancel| {
            if let Some(dialog) = cancel.ancestor(adw::Dialog::static_type()) {
                dialog.downcast_ref::<adw::Dialog>().map(adw::Dialog::close);
            }
        });
        let editor = self.downgrade();
        self.dialog.connect_close_attempt(move |dialog| {
            open_or!(editor);
            if !editor.state.borrow().dirty {
                dialog.force_close();
                return;
            }
            let confirm = adw::AlertDialog::new(
                Some("Discard the labels?"),
                Some("The changes to this test case are not saved."),
            );
            confirm.add_response("keep", "Keep Editing");
            confirm.add_response("discard", "Discard");
            confirm.set_response_appearance("discard", adw::ResponseAppearance::Destructive);
            confirm.set_default_response(Some("keep"));
            confirm.set_close_response("keep");
            let dialog = dialog.downgrade();
            confirm.connect_response(Some("discard"), move |_, _| {
                if let Some(dialog) = dialog.upgrade() {
                    dialog.force_close();
                }
            });
            confirm.present(Some(&editor.detail.window));
        });
        let editor = self.downgrade();
        let focus_lost = self.detail.window.connect_is_active_notify(move |window| {
            open_or!(editor);
            if !window.is_active() {
                editor.release_all();
            }
        });
        // The one strong hold on the open view, let go when it closes.
        let open = RefCell::new(Some((self.clone(), focus_lost)));
        self.dialog.connect_closed(move |_| {
            let Some((editor, focus_lost)) = open.take() else {
                return;
            };
            editor.state.borrow_mut().tick += 1;
            if editor.detail.playback.is_playing(LABEL_ID) {
                editor.detail.playback.stop();
            }
            editor.detail.window.disconnect(focus_lost);
        });
    }

    /// Reads the line's audio with the most context there may be, and what
    /// the session says about it, away from the GTK thread.
    fn load(&self) {
        let (session, span) = {
            let state = self.state.borrow();
            (
                state.session.clone(),
                testcase::clip_range(state.line, MAX_CONTEXT_MS, state.recorded_ms),
            )
        };
        let result = Arc::new(Mutex::new(None));
        let thread_result = result.clone();
        std::thread::spawn(move || {
            let audio = Audio::read(&session, span.0, span.1).map_err(|error| error.to_string());
            let inputs = Inputs::read(&session, span);
            *thread_result.lock().expect("label audio mutex") = Some((audio, inputs));
        });
        let editor = self.downgrade();
        glib::timeout_add_local(Duration::from_millis(50), move || {
            open_or!(editor, glib::ControlFlow::Break);
            let Some((audio, inputs)) = result.lock().expect("label audio mutex").take() else {
                return glib::ControlFlow::Continue;
            };
            {
                let mut state = editor.state.borrow_mut();
                state.inputs = Rc::new(inputs);
                match audio {
                    Ok(audio) => state.audio = Some(Rc::new(audio)),
                    Err(error) => state.error = Some(error),
                }
            }
            editor.refresh();
            glib::ControlFlow::Break
        });
    }

    fn redraw(&self) {
        for area in self.rows.iter().chain(&self.bars) {
            area.queue_draw();
        }
        self.playhead.queue_draw();
    }

    fn redraw_bars(&self) {
        for area in &self.bars {
            area.queue_draw();
        }
    }

    fn refresh_clock(&self) {
        let (position, clip) = {
            let state = self.state.borrow();
            (state.position(), state.clip())
        };
        self.position
            .set_label(&tenths(position.saturating_sub(clip.0)));
        self.length
            .set_label(&format!("/ {}", tenths(clip.1 - clip.0)));
    }

    /// Bring every label and button in line with the state, and redraw. The
    /// state is read first: changing a widget may call back into the editor.
    fn refresh(&self) {
        self.refresh_clock();
        let (subtitle, lanes, context, track, loaded, playing, audible) = {
            let state = self.state.borrow();
            let track = match state.labels.track {
                Track::System => "system audio",
                Track::Mic => "microphone audio",
                Track::Both => "both tracks",
            };
            (
                format!(
                    "{} · {track} · {} – {}",
                    state.title,
                    tenths(state.line.start_ms),
                    tenths(state.line.end_ms)
                ),
                state.labels.lanes,
                state.labels.context_ms,
                state.labels.track,
                state.audio.is_some(),
                state.playing.is_some(),
                state.line.start_ms < state.recorded_ms,
            )
        };
        self.title.set_subtitle(&subtitle);
        for (index, row) in self.lane_rows.iter().enumerate() {
            for widget in row {
                widget.set_visible(index < usize::from(lanes));
            }
        }
        self.add_lane.set_sensitive(lanes < MAX_SPEAKERS);
        self.context.set_label(&format!("{} s", context / 1000));
        self.less_context.set_sensitive(context > 0);
        self.more_context.set_sensitive(context < MAX_CONTEXT_MS);
        let stored = {
            let state = self.state.borrow();
            [AudioSource::System, AudioSource::Mic]
                .map(|source| state.session.stored_audio_path(source).is_file())
        };
        for (toggle, (candidate, available)) in self.tracks.iter().zip([
            (Track::System, stored[0]),
            (Track::Mic, stored[1]),
            (Track::Both, stored[0] && stored[1]),
        ]) {
            toggle.set_sensitive(available || candidate == track);
            if candidate == track && !toggle.is_active() {
                toggle.set_active(true);
            }
        }
        self.save.set_sensitive(loaded && audible);
        self.save.set_tooltip_text(Some(if audible {
            "Save the clip and the labels as a test case (Ctrl+S)"
        } else {
            "The recording ends before this line, so there is no audio to save"
        }));
        self.play.set_sensitive(loaded && audible);
        let (icon, tooltip) = if playing {
            ("media-playback-pause-symbolic", "Pause (Space)")
        } else {
            ("media-playback-start-symbolic", "Play (Space)")
        };
        self.play.set_icon_name(icon);
        self.play.set_tooltip_text(Some(tooltip));
        self.play
            .update_property(&[gtk::accessible::Property::Label(tooltip)]);
        self.redraw();
    }

    /// Changes the track or context; the labels stay where they are in the
    /// recording.
    fn change_view(&self, change: impl FnOnce(&mut Labels)) {
        if self.detail.playback.is_playing(LABEL_ID) {
            self.detail.playback.stop();
        }
        {
            let mut state = self.state.borrow_mut();
            change(&mut state.labels);
            state.dirty = true;
            let clip = state.clip();
            state.position_ms = state.position_ms.clamp(clip.0, clip.1);
        }
        self.refresh();
    }

    fn toggle_play(&self) {
        if self.detail.playback.is_playing(LABEL_ID) {
            self.detail.playback.stop();
        } else {
            self.start_playing();
        }
    }

    /// Plays the clip from the playhead, or from its start when the playhead
    /// is at the end.
    fn start_playing(&self) {
        let prepared = {
            let mut state = self.state.borrow_mut();
            let clip = state.clip();
            let mut from = state.position().clamp(clip.0, clip.1);
            if from + 100 >= clip.1 {
                from = clip.0;
            }
            let rate = if state.slow { SLOW_RATE } else { 1.0 };
            state.view().map(|view| {
                let offset = ((from - clip.0) as usize * SAMPLES_PER_MS).min(view.samples.len());
                (view.samples[offset..].to_vec(), from, clip.1, rate)
            })
        };
        let Some((samples, from_ms, until_ms, rate)) = prepared else {
            return;
        };
        if until_ms <= from_ms {
            return;
        }
        let stream: Box<dyn Read + Send> = if rate < 1.0 {
            Box::new(SlowStream {
                slowed: Slowed::new(samples, rate),
                chunk: Vec::new(),
                bytes: Vec::new(),
                offset: 0,
            })
        } else {
            Box::new(io::Cursor::new(
                samples
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<u8>>(),
            ))
        };
        let editor = self.downgrade();
        if let Err(error) = self.detail.playback.play(
            LABEL_ID,
            stream,
            &self.detail.window,
            Box::new(move || {
                if let Some(editor) = editor.upgrade() {
                    editor.playback_stopped();
                }
            }),
        ) {
            show_error(
                &self.detail.window,
                "Could not play audio",
                &error.to_string(),
            );
            return;
        }
        let tick = {
            let mut state = self.state.borrow_mut();
            state.playing = Some(Clock {
                started: Instant::now(),
                from_ms,
                until_ms,
                rate,
            });
            state.position_ms = from_ms;
            state.tick += 1;
            state.tick
        };
        let editor = self.downgrade();
        glib::timeout_add_local(Duration::from_millis(30), move || {
            open_or!(editor, glib::ControlFlow::Break);
            let growing = {
                let mut state = editor.state.borrow_mut();
                if state.tick != tick || state.playing.is_none() {
                    return glib::ControlFlow::Break;
                }
                let position = state.position();
                let held = state.held.into_iter().flatten().collect::<Vec<_>>();
                for id in &held {
                    state.set_bar(*id, |bar| bar.end_ms = position.max(bar.start_ms));
                }
                !held.is_empty()
            };
            editor.refresh_clock();
            editor.playhead.queue_draw();
            if growing {
                editor.redraw_bars();
            }
            glib::ControlFlow::Continue
        });
        self.refresh();
    }

    /// Playback ended or was stopped: held keys stop marking there.
    fn playback_stopped(&self) {
        {
            let mut state = self.state.borrow_mut();
            if let Some(clock) = state.playing.take() {
                let position = clock.position();
                state.position_ms = position;
                for lane in 0..state.held.len() {
                    if let Some(id) = state.held[lane].take() {
                        state.set_bar(id, |bar| bar.end_ms = position.max(bar.start_ms));
                        state.finish_bar(id);
                    }
                }
            }
        }
        self.refresh();
    }

    fn typing(&self) -> bool {
        self.dialog.focus().is_some_and(|focus| {
            focus == self.note.clone().upcast::<gtk::Widget>() || focus.is_ancestor(&self.note)
        })
    }

    fn key_pressed(&self, key: gdk::Key, modifiers: gdk::ModifierType) -> glib::Propagation {
        let chorded = modifiers.intersects(
            gdk::ModifierType::CONTROL_MASK
                | gdk::ModifierType::ALT_MASK
                | gdk::ModifierType::SUPER_MASK,
        );
        if modifiers.contains(gdk::ModifierType::CONTROL_MASK)
            && matches!(key, gdk::Key::s | gdk::Key::S)
        {
            if self.save.is_sensitive() {
                self.save();
            }
            return glib::Propagation::Stop;
        }
        if chorded || self.typing() {
            return glib::Propagation::Proceed;
        }
        if let Some(lane) = digit(key) {
            self.press_lane(lane);
            return glib::Propagation::Stop;
        }
        match key {
            gdk::Key::space | gdk::Key::KP_Space => self.toggle_play(),
            gdk::Key::Delete | gdk::Key::KP_Delete | gdk::Key::BackSpace => {
                let removed = {
                    let mut state = self.state.borrow_mut();
                    let selected = state.selected;
                    if let Some(id) = selected {
                        state.remove_bar(id);
                        state.dirty = true;
                    }
                    selected.is_some()
                };
                if !removed {
                    return glib::Propagation::Proceed;
                }
                self.redraw();
            }
            gdk::Key::Left | gdk::Key::Right => {
                if !self.nudge(if key == gdk::Key::Left { -1 } else { 1 }) {
                    return glib::Propagation::Proceed;
                }
            }
            _ => return glib::Propagation::Proceed,
        }
        glib::Propagation::Stop
    }

    fn key_released(&self, key: gdk::Key) {
        if let Some(lane) = digit(key) {
            self.release_lane(lane);
        }
    }

    /// A number key went down: mark that speaker from now on, starting
    /// playback first when paused. Repeats of a held key do nothing.
    fn press_lane(&self, lane: u8) {
        {
            let mut state = self.state.borrow_mut();
            let index = usize::from(lane);
            if state.down[index] || (lane != NOT_SURE && lane > state.labels.lanes) {
                return;
            }
            state.down[index] = true;
        }
        if self.state.borrow().playing.is_none() {
            self.start_playing();
        }
        let mut state = self.state.borrow_mut();
        if state.playing.is_none() {
            return;
        }
        let position = state.position();
        let id = state.add_bar(lane, position);
        state.held[usize::from(lane)] = Some(id);
        state.selected = Some(id);
        state.dirty = true;
    }

    fn release_lane(&self, lane: u8) {
        {
            let mut state = self.state.borrow_mut();
            let index = usize::from(lane);
            state.down[index] = false;
            let Some(id) = state.held[index].take() else {
                return;
            };
            if state.playing.is_some() {
                let position = state.position();
                state.set_bar(id, |bar| bar.end_ms = position.max(bar.start_ms));
            }
            state.finish_bar(id);
        }
        self.redraw();
    }

    /// The window lost the keyboard: a key let go elsewhere is never heard.
    fn release_all(&self) {
        for lane in 0..=9 {
            self.release_lane(lane);
        }
    }

    /// Moves the selected bar's edge nearest the playhead by 50 ms.
    fn nudge(&self, direction: i64) -> bool {
        {
            let mut state = self.state.borrow_mut();
            let Some(id) = state.selected else {
                return false;
            };
            let Some(bar) = state.bar(id) else {
                return false;
            };
            let (position, clip) = (state.position(), state.clip());
            let step = direction * NUDGE_MS as i64;
            let moved = if position.abs_diff(bar.start_ms) < position.abs_diff(bar.end_ms) {
                Bar {
                    start_ms: within(
                        bar.start_ms.saturating_add_signed(step),
                        clip.0.min(bar.start_ms),
                        bar.end_ms.saturating_sub(MIN_BAR_MS),
                    ),
                    ..bar
                }
            } else {
                Bar {
                    end_ms: within(
                        bar.end_ms.saturating_add_signed(step),
                        bar.start_ms + MIN_BAR_MS,
                        clip.1.max(bar.end_ms),
                    ),
                    ..bar
                }
            };
            state.set_bar(id, |bar| *bar = moved);
            testcase::merge_lane(&mut state.labels.bars, id);
            state.dirty = true;
        }
        self.redraw();
        true
    }

    fn seek_begin(&self, x: f64, width: f64) {
        let playing = self.detail.playback.is_playing(LABEL_ID);
        if playing {
            self.detail.playback.stop();
        }
        {
            let mut state = self.state.borrow_mut();
            let clip = state.clip();
            state.position_ms = ms_at(x, clip, width);
            state.drag = Some((Drag::Seek { resume: playing }, x));
        }
        self.refresh();
    }

    fn lane_begin(&self, lane: u8, x: f64, width: f64) {
        {
            let mut state = self.state.borrow_mut();
            let clip = state.clip();
            let at = ms_at(x, clip, width);
            let drag = match state.hit(lane, x, width) {
                Hit::Remove(id) => {
                    state.remove_bar(id);
                    state.dirty = true;
                    None
                }
                Hit::Edge(id, left) => state.bar(id).map(|bar| Drag::Resize { id, left, bar }),
                Hit::Body(id) => state.bar(id).map(|bar| Drag::Move {
                    id,
                    grab_ms: at,
                    bar,
                }),
                Hit::Empty if lane == NOT_SURE || lane <= state.labels.lanes => {
                    let id = state.add_bar(lane, at);
                    Some(Drag::New { id, anchor_ms: at })
                }
                Hit::Empty => None,
            };
            state.selected = match drag {
                Some(Drag::Resize { id, .. } | Drag::Move { id, .. }) => Some(id),
                _ => None,
            };
            state.drag = drag.map(|drag| (drag, x));
        }
        self.redraw();
    }

    fn drag_update(&self, dx: f64, width: f64) {
        {
            let mut state = self.state.borrow_mut();
            let Some((drag, start_x)) = state.drag else {
                return;
            };
            let clip = state.clip();
            let at = ms_at(start_x + dx, clip, width);
            match drag {
                Drag::Seek { .. } => state.position_ms = at,
                Drag::New { id, anchor_ms } => state.set_bar(id, |bar| {
                    bar.start_ms = anchor_ms.min(at);
                    bar.end_ms = anchor_ms.max(at);
                }),
                Drag::Resize { id, left, .. } => state.set_bar(id, |bar| {
                    if left {
                        bar.start_ms = at.min(bar.end_ms.saturating_sub(MIN_BAR_MS));
                    } else {
                        bar.end_ms = at.max(bar.start_ms + MIN_BAR_MS);
                    }
                }),
                Drag::Move { id, grab_ms, bar } => {
                    let earliest = clip.0 as i64 - bar.start_ms as i64;
                    let latest = clip.1 as i64 - bar.end_ms as i64;
                    let shift = (at as i64 - grab_ms as i64)
                        .max(earliest.min(0))
                        .min(latest.max(0));
                    state.set_bar(id, |moved| {
                        moved.start_ms = bar.start_ms.saturating_add_signed(shift);
                        moved.end_ms = bar.end_ms.saturating_add_signed(shift);
                    });
                }
            }
        }
        self.refresh_clock();
        self.redraw();
    }

    fn drag_end(&self) {
        let resume = {
            let mut state = self.state.borrow_mut();
            match state.drag.take() {
                Some((Drag::Seek { resume }, _)) => resume,
                Some((Drag::New { id, .. }, _)) => {
                    state.finish_bar(id);
                    if state.bar(id).is_some() {
                        state.selected = Some(id);
                    }
                    false
                }
                Some((Drag::Resize { id, bar, .. } | Drag::Move { id, bar, .. }, _)) => {
                    if state.bar(id) != Some(bar) {
                        state.finish_bar(id);
                    }
                    false
                }
                None => false,
            }
        };
        if resume {
            self.start_playing();
        }
        self.redraw();
    }

    fn save(&self) {
        let root = self.detail.config.borrow().test_cases_dir.clone();
        let labeled_at = session::local_time_from_unix(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
        )
        .rfc3339();
        let result = {
            let state = self.state.borrow();
            let Some(audio) = state.audio.clone() else {
                return;
            };
            let labels = Labels {
                note: self.note.text().to_string(),
                ..state.labels.clone()
            };
            testcase::save(
                &root,
                &Draft {
                    session: &state.session,
                    line: state.line,
                    line_speaker: &state.speaker,
                    labels: &labels,
                    recorded_ms: state.recorded_ms,
                    audio: &audio,
                    inputs: &state.inputs,
                    labeled_at: &labeled_at,
                },
            )
        };
        let saved = match result {
            Ok(saved) => saved,
            Err(error) => {
                show_error(
                    &self.detail.window,
                    "Could not save the test case",
                    &error.to_string(),
                );
                return;
            }
        };
        self.state.borrow_mut().dirty = false;
        self.dialog.force_close();
        if let Some(button) = self.button.upgrade() {
            show_saved(&button, Some(saved.speakers));
        }
        let toast = adw::Toast::builder()
            .title("Saved test case")
            .button_label("Open folder")
            .timeout(6)
            .build();
        let folder = saved.folder;
        toast.connect_button_clicked(move |_| open_path(&folder));
        self.detail.toasts.add_toast(toast);
    }

    /// The playhead across every row, drawn over them so that playing
    /// redraws only this line.
    fn draw_playhead(
        &self,
        layer: &gtk::DrawingArea,
        cr: &cairo::Context,
        ruler: &gtk::DrawingArea,
        overlap: &gtk::DrawingArea,
    ) -> Result<(), cairo::Error> {
        let (Some(top), Some(bottom)) = (
            ruler.compute_point(layer, &gtk::graphene::Point::new(0.0, 0.0)),
            overlap.compute_point(
                layer,
                &gtk::graphene::Point::new(0.0, overlap.height() as f32),
            ),
        ) else {
            return Ok(());
        };
        let state = self.state.borrow();
        let width = f64::from(ruler.width());
        let x = f64::from(top.x()) + x_of(state.position(), state.clip(), width);
        rgb(cr, PLAYHEAD_COLOR, 1.0);
        cr.rectangle(
            (x - 1.0).clamp(f64::from(top.x()), f64::from(top.x()) + width - 2.0),
            f64::from(top.y()),
            2.0,
            f64::from(bottom.y() - top.y()),
        );
        cr.fill()
    }

    fn draw_ruler(
        &self,
        area: &gtk::DrawingArea,
        cr: &cairo::Context,
        width: f64,
        height: f64,
    ) -> Result<(), cairo::Error> {
        let state = self.state.borrow();
        let clip = state.clip();
        let ink = ink(area);
        let span = (clip.1 - clip.0).max(1);
        let per_second = width * 1000.0 / span as f64;
        let label_step = [1u64, 2, 5, 10, 15, 30, 60]
            .into_iter()
            .find(|step| *step as f64 * per_second >= 48.0)
            .unwrap_or(120);
        cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
        cr.set_font_size(10.0);
        let mut second = clip.0.div_ceil(1000);
        while second * 1000 <= clip.1 {
            let x = x_of(second * 1000, clip, width).round();
            let labeled = second.is_multiple_of(label_step);
            if labeled || per_second >= 6.0 {
                ink(cr, 0.3);
                let tick = if labeled { 6.0 } else { 4.0 };
                cr.rectangle(x, height - tick, 1.0, tick);
                cr.fill()?;
            }
            if labeled {
                ink(cr, 0.6);
                let text = format_timeline_timestamp(second * 1000);
                let extents = cr.text_extents(&text)?;
                let left = (x - extents.width() / 2.0).clamp(0.0, width - extents.width());
                cr.move_to(left, 10.0);
                cr.show_text(&text)?;
            }
            second += 1;
        }
        Ok(())
    }

    fn draw_wave(
        &self,
        area: &gtk::DrawingArea,
        cr: &cairo::Context,
        width: f64,
        height: f64,
    ) -> Result<(), cairo::Error> {
        let mut state = self.state.borrow_mut();
        let clip = state.clip();
        let line = state.line;
        let ink = ink(area);
        ink(cr, 0.05);
        rounded(cr, 0.0, 0.0, width, height, 6.0);
        cr.fill()?;
        let message = state.error.clone();
        let Some(view) = state.view() else {
            ink(cr, 0.55);
            cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
            cr.set_font_size(11.0);
            let text = message.map_or_else(
                || "Reading the audio…".to_owned(),
                |error| format!("Could not read the audio: {error}"),
            );
            cr.move_to(10.0, height / 2.0 + 4.0);
            cr.show_text(&text)?;
            return Ok(());
        };
        let middle = height / 2.0;
        let half = middle - 3.0;
        let count = view.levels.len();
        let columns = width.max(1.0) as usize;
        for column in 0..columns {
            let from = column * count / columns;
            let to = ((column + 1) * count / columns).max(from + 1).min(count);
            let Some(slice) = view.levels.get(from..to) else {
                continue;
            };
            let peak = slice.iter().map(|level| level.peak).fold(0.0, f32::max);
            let rms = slice.iter().map(|level| level.rms).fold(0.0, f32::max);
            let at_ms = ms_at(column as f64 + 0.5, clip, width);
            let alpha = if (line.start_ms..line.end_ms).contains(&at_ms) {
                1.0
            } else {
                0.45
            };
            let peak_height = (f64::from(peak / view.norm).min(1.0) * half).max(0.5);
            let rms_height = (f64::from(rms / view.norm).min(1.0) * half).max(0.5);
            rgb(cr, WAVE_COLOR, 0.45 * alpha);
            cr.rectangle(column as f64, middle - peak_height, 1.0, peak_height * 2.0);
            cr.fill()?;
            rgb(cr, WAVE_COLOR, alpha);
            cr.rectangle(column as f64, middle - rms_height, 1.0, rms_height * 2.0);
            cr.fill()?;
        }
        for (from, to) in [(clip.0, line.start_ms), (line.end_ms, clip.1)] {
            if to > from {
                let (left, right) = (x_of(from, clip, width), x_of(to, clip, width));
                hatch(cr, left, 0.0, right - left, height, &ink, 0.16)?;
            }
        }
        Ok(())
    }

    fn draw_words(
        &self,
        area: &gtk::DrawingArea,
        cr: &cairo::Context,
        width: f64,
        _height: f64,
    ) -> Result<(), cairo::Error> {
        let state = self.state.borrow();
        let clip = state.clip();
        let ink = ink(area);
        // The words the case saves, on the clip's axis.
        let words = testcase::clip_words(&state.inputs.words, state.labels.track, clip);
        cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
        cr.set_font_size(10.5);
        // Each word goes on the first of three rows where it fits after the
        // word before; when none has room, on the one that frees up first.
        let mut row_ends = [f64::MIN; 3];
        for word in &words {
            let x = x_of(clip.0 + (word.start * 1000.0).round() as u64, clip, width).round();
            let text = word.text.as_str();
            let row = row_ends
                .iter()
                .position(|end| *end <= x)
                .unwrap_or_else(|| {
                    (0..row_ends.len())
                        .min_by(|a, b| row_ends[*a].total_cmp(&row_ends[*b]))
                        .unwrap_or(0)
                });
            row_ends[row] = x + 6.0 + cr.text_extents(text)?.x_advance();
            let top = row as f64 * 15.0 + 1.0;
            ink(cr, 0.18);
            cr.rectangle(x, top, 1.0, 14.0);
            cr.fill()?;
            ink(cr, 0.65);
            cr.move_to(x + 2.5, top + 11.0);
            cr.show_text(text)?;
        }
        Ok(())
    }

    fn draw_lane(
        &self,
        lane: u8,
        area: &gtk::DrawingArea,
        cr: &cairo::Context,
        width: f64,
        height: f64,
    ) -> Result<(), cairo::Error> {
        let state = self.state.borrow();
        let clip = state.clip();
        let ink = ink(area);
        ink(cr, 0.05);
        rounded(cr, 0.0, 0.0, width, height, 6.0);
        cr.fill()?;
        if state.held[usize::from(lane)].is_some() {
            rgb(cr, ACCENT, 1.0);
            cr.set_line_width(2.0);
            rounded(cr, 1.0, 1.0, width - 2.0, height - 2.0, 6.0);
            cr.stroke()?;
        }
        let hover = state
            .hover
            .filter(|(hovered, _)| *hovered == lane)
            .map(|(_, x)| x);
        for bar in state.labels.bars.iter().filter(|bar| bar.lane == lane) {
            let left = x_of(bar.start_ms.max(clip.0), clip, width);
            let right = x_of(bar.end_ms.min(clip.1), clip, width);
            if right <= left {
                continue;
            }
            let bar_width = (right - left).max(2.0);
            rounded(
                cr,
                left,
                3.0,
                bar_width,
                height - 6.0,
                5.0_f64.min(bar_width / 2.0),
            );
            if lane == NOT_SURE {
                ink(cr, 0.08);
                cr.fill_preserve()?;
                ink(cr, 0.25);
                cr.set_line_width(1.0);
                cr.stroke()?;
                hatch(cr, left, 3.0, bar_width, height - 6.0, &ink, 0.35)?;
            } else {
                rgb(cr, lane_color(lane), 1.0);
                cr.fill()?;
            }
            if state.selected == Some(bar.id) {
                ink(cr, 1.0);
                cr.set_line_width(2.0);
                rounded(cr, left - 1.0, 2.0, bar_width + 2.0, height - 4.0, 6.0);
                cr.stroke()?;
            }
            if hover.is_some_and(|x| x >= left - EDGE_GRAB && x <= right + EDGE_GRAB)
                && right - left >= 2.0 * CLOSE_ZONE
            {
                if lane == NOT_SURE {
                    ink(cr, 0.9);
                } else {
                    cr.set_source_rgb(1.0, 1.0, 1.0);
                }
                cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Bold);
                cr.set_font_size(12.0);
                cr.move_to(right - CLOSE_ZONE + 3.0, height / 2.0 + 4.0);
                cr.show_text("×")?;
            }
        }
        Ok(())
    }

    fn draw_overlap(
        &self,
        area: &gtk::DrawingArea,
        cr: &cairo::Context,
        width: f64,
        height: f64,
    ) -> Result<(), cairo::Error> {
        let state = self.state.borrow();
        let clip = state.clip();
        let ink = ink(area);
        ink(cr, 0.05);
        rounded(cr, 0.0, 0.0, width, height, 4.0);
        cr.fill()?;
        for (from, to, first, second) in testcase::overlaps(&state.labels.bars) {
            let left = x_of(from.max(clip.0), clip, width);
            let right = x_of(to.min(clip.1), clip, width);
            if right <= left {
                continue;
            }
            cr.save()?;
            rounded(cr, left, 0.0, (right - left).max(2.0), height, 3.0);
            cr.clip();
            let mut x = left;
            let mut stripe = 0;
            while x < right {
                let lane = if stripe % 2 == 0 { first } else { second };
                rgb(cr, lane_color(lane), 1.0);
                cr.rectangle(x, 0.0, 4.0, height);
                cr.fill()?;
                x += 4.0;
                stripe += 1;
            }
            cr.restore()?;
        }
        Ok(())
    }
}

/// Playback of the clip at a slower rate, as f32le bytes.
struct SlowStream {
    slowed: Slowed,
    chunk: Vec<f32>,
    bytes: Vec<u8>,
    offset: usize,
}

impl Read for SlowStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.offset == self.bytes.len() {
            self.chunk.clear();
            while self.chunk.len() < 1_600 && self.slowed.fill(&mut self.chunk) {}
            if self.chunk.is_empty() {
                return Ok(0);
            }
            self.bytes = self
                .chunk
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            self.offset = 0;
        }
        let count = buffer.len().min(self.bytes.len() - self.offset);
        buffer[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
        self.offset += count;
        Ok(count)
    }
}

fn lane_color(lane: u8) -> Rgb {
    COLORS[usize::from(lane.clamp(1, MAX_SPEAKERS)) - 1]
}

/// `value` within `[low, high]`, or `low` when the range is empty.
fn within(value: u64, low: u64, high: u64) -> u64 {
    value.min(high).max(low)
}

fn widget_width(gesture: &gtk::GestureDrag) -> f64 {
    gesture
        .widget()
        .map_or(1.0, |widget| f64::from(widget.width()))
}

fn digit(key: gdk::Key) -> Option<u8> {
    key.to_unicode()
        .and_then(|character| character.to_digit(10))
        .map(|digit| digit as u8)
}

/// "1:02.5"
fn tenths(ms: u64) -> String {
    format!("{}:{:02}.{}", ms / 60_000, ms / 1000 % 60, ms / 100 % 10)
}

fn x_of(ms: u64, (start, end): (u64, u64), width: f64) -> f64 {
    width * ms.saturating_sub(start) as f64 / end.saturating_sub(start).max(1) as f64
}

fn ms_at(x: f64, (start, end): (u64, u64), width: f64) -> u64 {
    let fraction = (x / width.max(1.0)).clamp(0.0, 1.0);
    start + (fraction * end.saturating_sub(start) as f64).round() as u64
}

type Ink = Box<dyn Fn(&cairo::Context, f64)>;

fn ink(area: &gtk::DrawingArea) -> Ink {
    let color = area.color();
    Box::new(move |cr, alpha| {
        cr.set_source_rgba(
            f64::from(color.red()),
            f64::from(color.green()),
            f64::from(color.blue()),
            f64::from(color.alpha()) * alpha,
        );
    })
}

fn hatch(
    cr: &cairo::Context,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    ink: &Ink,
    alpha: f64,
) -> Result<(), cairo::Error> {
    cr.save()?;
    cr.rectangle(x, y, width, height);
    cr.clip();
    ink(cr, alpha);
    cr.set_line_width(1.5);
    let mut offset = x - height;
    while offset < x + width {
        cr.move_to(offset, y + height);
        cr.line_to(offset + height, y);
        offset += 6.0;
    }
    cr.stroke()?;
    cr.restore()
}
