//! The recording player: both tracks as waveforms on one time axis, with
//! scrubbing, playback through the shared playback controller, and cutting
//! the recording in two.

use super::wrap_layout::WrapLayout;
use super::{PlaybackController, format_duration, format_timeline_timestamp, show_error};
use crate::audio::track::{self, BUCKET_MS, Envelope, TrackReader};
use crate::session::{self, Session};
use crate::split::{self, MIN_PART_MS};
use crate::types::{AudioSource, SAMPLE_RATE};
use adw::prelude::*;
use gtk::{cairo, gdk, glib};
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::{self, Read};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// The playback controller's id for the player; transcript rows count up from 1.
const PLAYER_ID: u64 = u64::MAX;
const JUMP_MS: u64 = 5_000;
const NUDGE_MS: u64 = 1_000;
const HEAR_MS: u64 = 4_000;
const SAMPLES_PER_MS: u64 = SAMPLE_RATE as u64 / 1000;
const MIX_SAMPLES: u64 = 1_600;

const GUTTER: f64 = 58.0;
const RIGHT: f64 = 10.0;
const RULER: f64 = 22.0;
const LANE: f64 = 52.0;
const MARKERS: f64 = 24.0;
const CUT_GRAB: f64 = 8.0;

type Rgb = (f64, f64, f64);
const MIC_COLOR: Rgb = (0.208, 0.518, 0.894);
const SYSTEM_COLOR: Rgb = (0.149, 0.635, 0.412);
const FIRST_COLOR: Rgb = (0.208, 0.518, 0.894);
const SECOND_COLOR: Rgb = (0.902, 0.380, 0.0);
const SHOT_COLOR: Rgb = (0.208, 0.518, 0.894);

pub(super) type SplitHandler = Rc<dyn Fn(PathBuf, u64, [String; 2])>;

#[derive(Clone, Copy)]
struct Clock {
    started: Instant,
    from_ms: u64,
    until_ms: u64,
}

impl Clock {
    fn position(&self) -> u64 {
        let elapsed = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.from_ms.saturating_add(elapsed).min(self.until_ms)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Scrub { resume: bool },
    Cut,
}

#[derive(Default)]
struct State {
    session: Option<PathBuf>,
    started: String,
    length_ms: u64,
    tracks: Vec<AudioSource>,
    muted: Vec<AudioSource>,
    levels: Option<Rc<Levels>>,
    envelope_error: Option<String>,
    screenshots: Vec<u64>,
    position_ms: u64,
    hover_x: Option<f64>,
    playing: Option<Clock>,
    cut_ms: Option<u64>,
    drag: Option<(Drag, f64)>,
    split_blocked: Option<String>,
    load_generation: u64,
    cancel_load: Option<Arc<AtomicBool>>,
    tick_generation: u64,
}

impl State {
    fn quiet(&self) -> &[(u64, u64)] {
        self.levels.as_ref().map_or(&[], |levels| &levels.quiet)
    }

    fn position(&self) -> u64 {
        self.playing
            .map_or(self.position_ms, |clock| clock.position())
    }

    fn can_split(&self) -> bool {
        self.split_blocked.is_none() && self.length_ms >= 2 * MIN_PART_MS
    }

    fn clamp_cut(&self, at_ms: u64) -> u64 {
        at_ms.clamp(MIN_PART_MS, self.length_ms.saturating_sub(MIN_PART_MS))
    }
}

/// A recording's envelope, and what the player derives from it, all worked
/// out away from the GTK thread.
struct Levels {
    envelope: Envelope,
    /// Per track, the level drawn full height.
    norms: Vec<f32>,
    quiet: Vec<(u64, u64)>,
}

impl Levels {
    fn new(envelope: Envelope) -> Self {
        Self {
            norms: envelope
                .tracks
                .iter()
                .map(|(_, levels)| display_norm(levels))
                .collect(),
            quiet: envelope.quiet_stretches(),
            envelope,
        }
    }
}

/// Stored audio sizes and times, to notice a changed recording.
type EnvelopeKey = Vec<(u64, Option<SystemTime>)>;
type EnvelopeCache = Rc<RefCell<HashMap<PathBuf, (EnvelopeKey, Rc<Levels>)>>>;

#[derive(Clone)]
pub(super) struct Player {
    pub root: gtk::Box,
    area: gtk::DrawingArea,
    chips: [gtk::ToggleButton; 2],
    play_button: gtk::Button,
    position_label: gtk::Label,
    rest_label: gtk::Label,
    split_button: gtk::Button,
    panel: gtk::Box,
    cut_label: gtk::Label,
    snap_button: gtk::Button,
    confirm_button: gtk::Button,
    titles: [gtk::Entry; 2],
    part_info: [gtk::Label; 2],
    cut_note: gtk::Label,
    playback: PlaybackController,
    window: adw::ApplicationWindow,
    state: Rc<RefCell<State>>,
    on_split: Rc<RefCell<Option<SplitHandler>>>,
    envelopes: EnvelopeCache,
}

fn chip(label: &str, icon: &str, tooltip: &str) -> gtk::ToggleButton {
    let content = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    content.append(&gtk::Image::from_icon_name(icon));
    content.append(&gtk::Label::new(Some(label)));
    let chip = gtk::ToggleButton::builder()
        .child(&content)
        .active(true)
        .tooltip_text(tooltip)
        .build();
    chip.add_css_class("player-chip");
    chip
}

fn small_button(label: &str, tooltip: &str) -> gtk::Button {
    let button = gtk::Button::with_label(label);
    button.set_tooltip_text(Some(tooltip));
    button
}

impl Player {
    pub fn new(playback: &PlaybackController, window: &adw::ApplicationWindow) -> Self {
        let root = gtk::Box::new(gtk::Orientation::Vertical, 8);
        root.set_margin_start(16);
        root.set_margin_end(16);
        root.set_margin_bottom(6);
        let header = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        let heading = gtk::Label::new(Some("Recording"));
        heading.add_css_class("title-4");
        heading.set_xalign(0.0);
        heading.set_hexpand(true);
        header.append(&heading);
        let chips = [
            chip(
                "Microphone",
                "audio-input-microphone-symbolic",
                "Hear the microphone track while listening",
            ),
            chip(
                "System",
                "audio-speakers-symbolic",
                "Hear the system audio track while listening",
            ),
        ];
        for chip in &chips {
            header.append(chip);
        }
        root.append(&header);

        let card = gtk::Box::new(gtk::Orientation::Vertical, 8);
        card.add_css_class("player-card");
        let area = gtk::DrawingArea::builder()
            .hexpand(true)
            .focusable(true)
            .build();
        area.set_cursor_from_name(Some("pointer"));
        area.update_property(&[gtk::accessible::Property::Label(
            "Recording timeline. Space plays or pauses, the arrow keys jump 5 seconds.",
        )]);
        card.append(&area);
        let transport = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        let play_button = gtk::Button::from_icon_name("media-playback-start-symbolic");
        play_button.add_css_class("circular");
        play_button.add_css_class("suggested-action");
        play_button.set_tooltip_text(Some("Play (Space)"));
        transport.append(&play_button);
        let back = small_button("−5 s", "Back 5 seconds (←)");
        let forward = small_button("+5 s", "Forward 5 seconds (→)");
        transport.append(&back);
        transport.append(&forward);
        let position_label = gtk::Label::new(Some("0:00"));
        position_label.add_css_class("player-clock");
        position_label.add_css_class("heading");
        position_label.set_margin_start(6);
        transport.append(&position_label);
        let rest_label = gtk::Label::new(None);
        rest_label.add_css_class("player-clock");
        rest_label.add_css_class("dim-label");
        rest_label.set_hexpand(true);
        rest_label.set_xalign(0.0);
        transport.append(&rest_label);
        let split_button = gtk::Button::with_label("Split here…");
        split_button.add_css_class("destructive-action");
        transport.append(&split_button);
        card.append(&transport);
        root.append(&card);

        let panel = gtk::Box::new(gtk::Orientation::Vertical, 10);
        panel.add_css_class("split-panel");
        panel.set_visible(false);
        let cut_row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        let cut_tools = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        cut_tools.set_layout_manager(Some(WrapLayout::new(6, 6)));
        cut_tools.set_hexpand(true);
        let cut_at = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        cut_at.append(&gtk::Image::from_icon_name("edit-cut-symbolic"));
        cut_at.append(&gtk::Label::new(Some("Cut at")));
        let cut_label = gtk::Label::new(None);
        cut_label.add_css_class("heading");
        cut_label.add_css_class("player-clock");
        cut_label.set_margin_end(6);
        cut_at.append(&cut_label);
        cut_tools.append(&cut_at);
        cut_row.append(&cut_tools);
        let earlier = small_button("−1 s", "Move the cut 1 second earlier");
        let later = small_button("+1 s", "Move the cut 1 second later");
        let snap_button = small_button(
            "Snap to quiet",
            "Move the cut into the nearest stretch where both tracks are near silence",
        );
        let hear = small_button("Hear the cut", "Play from 4 s before to 4 s after the cut");
        for button in [&earlier, &later, &snap_button, &hear] {
            cut_tools.append(button);
        }
        let cancel = gtk::Button::with_label("Cancel");
        cancel.set_valign(gtk::Align::Start);
        cancel.add_css_class("flat");
        cut_row.append(&cancel);
        let confirm_button = gtk::Button::with_label("Split…");
        confirm_button.add_css_class("destructive-action");
        confirm_button.set_valign(gtk::Align::Start);
        cut_row.append(&confirm_button);
        panel.append(&cut_row);
        let parts = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        parts.set_homogeneous(true);
        let mut titles = Vec::new();
        let mut part_info = Vec::new();
        for index in 0..2 {
            let part = gtk::Box::new(gtk::Orientation::Vertical, 6);
            part.add_css_class("split-part");
            part.add_css_class(&format!("split-part-{}", index + 1));
            let name = gtk::Label::new(Some(&format!("Part {}", index + 1)));
            name.add_css_class("caption");
            name.add_css_class("dim-label");
            name.set_xalign(0.0);
            part.append(&name);
            let title = gtk::Entry::new();
            title.set_tooltip_text(Some("Title of the new session"));
            part.append(&title);
            let info = gtk::Label::new(None);
            info.add_css_class("caption");
            info.add_css_class("dim-label");
            info.set_xalign(0.0);
            info.set_wrap(true);
            part.append(&info);
            parts.append(&part);
            titles.push(title);
            part_info.push(info);
        }
        panel.append(&parts);
        let cut_note = gtk::Label::new(None);
        cut_note.set_xalign(0.0);
        cut_note.set_wrap(true);
        cut_note.add_css_class("caption");
        panel.append(&cut_note);
        root.append(&panel);

        let player = Self {
            root,
            area,
            chips,
            play_button,
            position_label,
            rest_label,
            split_button,
            panel,
            cut_label,
            snap_button,
            confirm_button,
            titles: titles.try_into().expect("two titles"),
            part_info: part_info.try_into().expect("two parts"),
            cut_note,
            playback: playback.clone(),
            window: window.clone(),
            state: Rc::new(RefCell::new(State::default())),
            on_split: Rc::new(RefCell::new(None)),
            envelopes: Rc::new(RefCell::new(HashMap::new())),
        };
        player.wire(&card, &back, &forward, &earlier, &later, &hear, &cancel);
        player.layout_area();
        player
    }

    pub fn connect_split(&self, handler: SplitHandler) {
        *self.on_split.borrow_mut() = Some(handler);
    }

    /// Show `session`. Reloading the session on screen keeps the playhead and
    /// the cut.
    pub fn load(&self, session: &Session, started: &str, length_ms: u64, screenshots: Vec<u64>) {
        let tracks = [AudioSource::Mic, AudioSource::System]
            .into_iter()
            .filter(|source| session.stored_audio_path(*source).is_file())
            .collect::<Vec<_>>();
        let same = self.state.borrow().session.as_deref() == Some(session.dir.as_path());
        {
            let mut state = self.state.borrow_mut();
            state.started = started.to_owned();
            state.length_ms = length_ms;
            state.screenshots = screenshots;
            state.playing = None;
            state.drag = None;
            if !same {
                state.session = Some(session.dir.clone());
                state.position_ms = 0;
                state.cut_ms = None;
                state.muted.clear();
            }
            state.position_ms = state.position_ms.min(length_ms);
            if state.cut_ms.is_some() && length_ms < 2 * MIN_PART_MS {
                state.cut_ms = None;
            }
            state.cut_ms = state.cut_ms.map(|cut| state.clamp_cut(cut));
            state.tracks = tracks.clone();
        }
        if !same {
            for (entry, title) in self.titles.iter().zip(split::default_titles(session)) {
                entry.set_text(&title);
            }
        }
        for (chip, source) in self
            .chips
            .iter()
            .zip([AudioSource::Mic, AudioSource::System])
        {
            chip.set_visible(tracks.contains(&source));
            let audible = !self.state.borrow().muted.contains(&source);
            chip.set_active(audible);
        }
        self.play_button
            .set_icon_name("media-playback-start-symbolic");
        self.layout_area();
        self.load_envelope(session);
        self.refresh();
    }

    /// Forget the session, for example when nothing is selected.
    pub fn clear(&self) {
        let mut state = self.state.borrow_mut();
        if let Some(cancel) = state.cancel_load.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        state.load_generation += 1;
        state.session = None;
        state.cut_ms = None;
        state.playing = None;
    }

    /// Why splitting is not possible now, or `None` when it is.
    pub fn set_split_blocked(&self, reason: Option<&str>) {
        self.state.borrow_mut().split_blocked = reason.map(str::to_owned);
        self.refresh();
    }

    fn layout_area(&self) {
        let lanes = self.state.borrow().tracks.len().max(1) as f64;
        self.area
            .set_content_height((RULER + lanes * LANE + MARKERS) as i32);
    }

    #[allow(clippy::too_many_arguments)]
    fn wire(
        &self,
        card: &gtk::Box,
        back: &gtk::Button,
        forward: &gtk::Button,
        earlier: &gtk::Button,
        later: &gtk::Button,
        hear: &gtk::Button,
        cancel: &gtk::Button,
    ) {
        let player = self.clone();
        self.area.set_draw_func(move |area, cr, width, height| {
            let _ = player.draw(area, cr, f64::from(width), f64::from(height));
        });

        let drag = gtk::GestureDrag::new();
        let player = self.clone();
        drag.connect_drag_begin(move |_, x, _| player.drag_begin(x));
        let player = self.clone();
        drag.connect_drag_update(move |_, dx, _| player.drag_update(dx));
        let player = self.clone();
        drag.connect_drag_end(move |_, _, _| player.drag_end());
        self.area.add_controller(drag);

        let motion = gtk::EventControllerMotion::new();
        let player = self.clone();
        motion.connect_motion(move |_, x, _| {
            let near_cut = {
                let mut state = player.state.borrow_mut();
                state.hover_x = Some(x);
                let width = f64::from(player.area.width());
                state
                    .cut_ms
                    .is_some_and(|cut| (x_of(cut, state.length_ms, width) - x).abs() <= CUT_GRAB)
            };
            player
                .area
                .set_cursor_from_name(Some(if near_cut { "col-resize" } else { "pointer" }));
            player.area.queue_draw();
        });
        let player = self.clone();
        motion.connect_leave(move |_| {
            player.state.borrow_mut().hover_x = None;
            player.area.queue_draw();
        });
        self.area.add_controller(motion);

        let keys = gtk::EventControllerKey::new();
        keys.set_propagation_phase(gtk::PropagationPhase::Capture);
        let player = self.clone();
        keys.connect_key_pressed(move |_, key, _, modifiers| {
            if !modifiers.is_empty() {
                return glib::Propagation::Proceed;
            }
            match key {
                gdk::Key::space => player.toggle_play(),
                gdk::Key::Left => player.jump(-(JUMP_MS as i64)),
                gdk::Key::Right => player.jump(JUMP_MS as i64),
                _ => return glib::Propagation::Proceed,
            }
            glib::Propagation::Stop
        });
        card.add_controller(keys);

        let player = self.clone();
        self.play_button
            .connect_clicked(move |_| player.toggle_play());
        let player = self.clone();
        back.connect_clicked(move |_| player.jump(-(JUMP_MS as i64)));
        let player = self.clone();
        forward.connect_clicked(move |_| player.jump(JUMP_MS as i64));
        for (chip, source) in self
            .chips
            .iter()
            .zip([AudioSource::Mic, AudioSource::System])
        {
            let player = self.clone();
            chip.connect_toggled(move |chip| player.set_muted(source, !chip.is_active()));
        }

        let player = self.clone();
        self.split_button.connect_clicked(move |_| {
            {
                let mut state = player.state.borrow_mut();
                if !state.can_split() {
                    return;
                }
                let position = state.position();
                state.cut_ms = Some(state.clamp_cut(position));
            }
            player.refresh();
        });
        let player = self.clone();
        earlier.connect_clicked(move |_| player.nudge(-(NUDGE_MS as i64)));
        let player = self.clone();
        later.connect_clicked(move |_| player.nudge(NUDGE_MS as i64));
        let player = self.clone();
        self.snap_button.connect_clicked(move |_| {
            {
                let mut state = player.state.borrow_mut();
                let Some(cut) = state.cut_ms else { return };
                if let Some(snapped) = track::snap_to_quiet(state.quiet(), cut) {
                    state.cut_ms = Some(state.clamp_cut(snapped));
                }
            }
            player.refresh();
        });
        let player = self.clone();
        hear.connect_clicked(move |_| {
            let (cut, length) = {
                let state = player.state.borrow();
                (state.cut_ms, state.length_ms)
            };
            if let Some(cut) = cut {
                player.start_playing(cut.saturating_sub(HEAR_MS), (cut + HEAR_MS).min(length));
            }
        });
        let player = self.clone();
        cancel.connect_clicked(move |_| {
            player.state.borrow_mut().cut_ms = None;
            player.refresh();
        });
        let player = self.clone();
        self.confirm_button
            .connect_clicked(move |_| player.confirm_split());
    }

    fn drag_begin(&self, x: f64) {
        self.area.grab_focus();
        let width = f64::from(self.area.width());
        let (near_cut, playing) = {
            let state = self.state.borrow();
            (
                state
                    .cut_ms
                    .is_some_and(|cut| (x_of(cut, state.length_ms, width) - x).abs() <= CUT_GRAB),
                state.playing.is_some(),
            )
        };
        if near_cut {
            self.state.borrow_mut().drag = Some((Drag::Cut, x));
            return;
        }
        if playing {
            self.playback.stop();
        }
        {
            let mut state = self.state.borrow_mut();
            state.drag = Some((Drag::Scrub { resume: playing }, x));
            state.position_ms = ms_of(x, state.length_ms, width);
        }
        self.refresh();
    }

    fn drag_update(&self, dx: f64) {
        let width = f64::from(self.area.width());
        {
            let mut state = self.state.borrow_mut();
            let Some((drag, start)) = state.drag else {
                return;
            };
            let at = ms_of(start + dx, state.length_ms, width);
            match drag {
                Drag::Cut => state.cut_ms = Some(state.clamp_cut(at)),
                Drag::Scrub { .. } => state.position_ms = at,
            }
        }
        self.refresh();
    }

    fn drag_end(&self) {
        let (drag, position, length) = {
            let mut state = self.state.borrow_mut();
            (state.drag.take(), state.position_ms, state.length_ms)
        };
        if let Some((Drag::Scrub { resume: true }, _)) = drag {
            self.start_playing(position, length);
        }
    }

    fn toggle_play(&self) {
        if self.playback.is_playing(PLAYER_ID) {
            self.playback.stop();
            return;
        }
        let (mut position, length) = {
            let state = self.state.borrow();
            (state.position_ms, state.length_ms)
        };
        if position + 100 >= length {
            position = 0;
        }
        self.start_playing(position, length);
    }

    fn jump(&self, delta_ms: i64) {
        let (position, until, playing) = {
            let state = self.state.borrow();
            let until = state
                .playing
                .map_or(state.length_ms, |clock| clock.until_ms);
            (state.position(), until, state.playing.is_some())
        };
        let target = position.saturating_add_signed(delta_ms).min(until);
        if playing && target < until {
            self.start_playing(target, until);
            return;
        }
        if playing {
            self.playback.stop();
        }
        self.state.borrow_mut().position_ms = target;
        self.refresh();
    }

    fn nudge(&self, delta_ms: i64) {
        {
            let mut state = self.state.borrow_mut();
            let Some(cut) = state.cut_ms else { return };
            state.cut_ms = Some(state.clamp_cut(cut.saturating_add_signed(delta_ms)));
        }
        self.refresh();
    }

    fn set_muted(&self, source: AudioSource, muted: bool) {
        let restart = {
            let mut state = self.state.borrow_mut();
            if muted == state.muted.contains(&source) {
                return;
            }
            state.muted.retain(|candidate| *candidate != source);
            if muted {
                state.muted.push(source);
            }
            state
                .playing
                .map(|clock| (clock.position(), clock.until_ms))
        };
        if let Some((position, until)) = restart {
            self.start_playing(position, until);
        }
        self.area.queue_draw();
    }

    /// Play from `from_ms` to `until_ms`, the unmuted tracks mixed.
    fn start_playing(&self, from_ms: u64, until_ms: u64) {
        if until_ms <= from_ms {
            return;
        }
        let (dir, sources) = {
            let state = self.state.borrow();
            let Some(dir) = state.session.clone() else {
                return;
            };
            let sources = state
                .tracks
                .iter()
                .copied()
                .filter(|source| !state.muted.contains(source))
                .collect::<Vec<_>>();
            (dir, sources)
        };
        let stream = Session::open(&dir)
            .and_then(|session| MixedStream::open(&session, &sources, from_ms, until_ms));
        let player = self.clone();
        let started = stream.and_then(|stream| {
            self.playback.play(
                PLAYER_ID,
                Box::new(stream),
                &self.window,
                Box::new(move || player.playback_stopped()),
            )
        });
        if let Err(error) = started {
            show_error(&self.window, "Could not play audio", &error.to_string());
            return;
        }
        let tick = {
            let mut state = self.state.borrow_mut();
            state.playing = Some(Clock {
                started: Instant::now(),
                from_ms,
                until_ms,
            });
            state.position_ms = from_ms;
            state.tick_generation += 1;
            state.tick_generation
        };
        self.play_button
            .set_icon_name("media-playback-pause-symbolic");
        self.play_button.set_tooltip_text(Some("Pause (Space)"));
        let player = self.clone();
        glib::timeout_add_local(Duration::from_millis(50), move || {
            {
                let state = player.state.borrow();
                if state.tick_generation != tick || state.playing.is_none() {
                    return glib::ControlFlow::Break;
                }
            }
            player.refresh_clock();
            player.area.queue_draw();
            glib::ControlFlow::Continue
        });
        self.refresh();
    }

    fn playback_stopped(&self) {
        {
            let mut state = self.state.borrow_mut();
            if let Some(clock) = state.playing.take() {
                state.position_ms = clock.position();
            }
        }
        self.play_button
            .set_icon_name("media-playback-start-symbolic");
        self.play_button.set_tooltip_text(Some("Play (Space)"));
        self.refresh();
    }

    fn load_envelope(&self, session: &Session) {
        let key = envelope_key(session);
        let cached = self
            .envelopes
            .borrow()
            .get(&session.dir)
            .filter(|(cached, _)| *cached == key)
            .map(|(_, envelope)| envelope.clone());
        let generation = {
            let mut state = self.state.borrow_mut();
            if let Some(cancel) = state.cancel_load.take() {
                cancel.store(true, Ordering::Relaxed);
            }
            state.load_generation += 1;
            state.envelope_error = None;
            state.load_generation
        };
        if let Some(levels) = cached {
            self.set_levels(levels);
            return;
        }
        self.state.borrow_mut().levels = None;
        let cancel = Arc::new(AtomicBool::new(false));
        self.state.borrow_mut().cancel_load = Some(cancel.clone());
        let result = Arc::new(Mutex::new(None));
        let thread_result = result.clone();
        let thread_session = session.clone();
        std::thread::spawn(move || {
            let value = Envelope::compute(&thread_session, &cancel)
                .map(|envelope| envelope.map(Levels::new))
                .map_err(|error| error.to_string());
            *thread_result.lock().expect("envelope result mutex") = Some(value);
        });
        let player = self.clone();
        let dir = session.dir.clone();
        glib::timeout_add_local(Duration::from_millis(100), move || {
            if player.state.borrow().load_generation != generation {
                return glib::ControlFlow::Break;
            }
            let Some(value) = result.lock().expect("envelope result mutex").take() else {
                return glib::ControlFlow::Continue;
            };
            player.state.borrow_mut().cancel_load = None;
            match value {
                Ok(Some(levels)) => {
                    let levels = Rc::new(levels);
                    let mut cache = player.envelopes.borrow_mut();
                    if cache.len() >= 8 {
                        cache.clear();
                    }
                    cache.insert(dir.clone(), (key.clone(), levels.clone()));
                    drop(cache);
                    player.set_levels(levels);
                }
                Ok(None) => {}
                Err(error) => {
                    eprintln!("error: could not read the recording's levels: {error}");
                    player.state.borrow_mut().envelope_error = Some(error);
                    player.refresh();
                }
            }
            glib::ControlFlow::Break
        });
    }

    fn set_levels(&self, levels: Rc<Levels>) {
        self.state.borrow_mut().levels = Some(levels);
        self.refresh();
    }

    fn refresh_clock(&self) {
        let (position, length, wall) = {
            let state = self.state.borrow();
            let position = state.position();
            (
                position,
                state.length_ms,
                wall_clock(&state.started, position),
            )
        };
        self.position_label
            .set_label(&format_timeline_timestamp(position));
        let wall = wall.map(|time| format!(" · {time}")).unwrap_or_default();
        self.rest_label
            .set_label(&format!("/ {}{wall}", format_timeline_timestamp(length)));
    }

    /// Bring every label and button in line with the state, and redraw. The
    /// state is read first: changing a widget may call back into the player.
    fn refresh(&self) {
        self.refresh_clock();
        let (can_split, cut, blocked, tooltip, cut_text, parts, snappable, note) = {
            let state = self.state.borrow();
            let tooltip = state.split_blocked.clone().unwrap_or_else(|| {
                if state.length_ms < 2 * MIN_PART_MS {
                    "The recording is too short to split"
                } else {
                    "Cut the recording in two at the playhead"
                }
                .to_owned()
            });
            let cut = state.cut_ms.unwrap_or(0);
            let wall = wall_clock(&state.started, cut)
                .map(|time| format!(" ({time})"))
                .unwrap_or_default();
            let before = state.screenshots.iter().filter(|time| **time < cut).count();
            let parts = [
                (0, cut, before),
                (
                    cut,
                    state.length_ms.saturating_sub(cut),
                    state.screenshots.len() - before,
                ),
            ]
            .map(|(start, length, shots)| part_summary(&state.started, start, length, shots));
            let note = if state.envelope_error.is_some() {
                (
                    "The recording's levels could not be read, so quiet stretches are unknown.",
                    true,
                )
            } else if state.levels.is_none() {
                ("Looking for quiet stretches…", false)
            } else if track::is_quiet(state.quiet(), cut) {
                ("✓ The cut sits in a quiet stretch on both tracks.", false)
            } else if state.quiet().is_empty() {
                (
                    "⚠ Someone is talking at the cut, and the recording has no quiet stretch to move it to.",
                    true,
                )
            } else {
                (
                    "⚠ Someone is talking at the cut. Snap to quiet moves it to the nearest pause.",
                    true,
                )
            };
            (
                state.can_split(),
                state.cut_ms,
                state.split_blocked.clone(),
                tooltip,
                format!("{}{wall}", format_timeline_timestamp(cut)),
                parts,
                state.levels.is_some() && !state.quiet().is_empty(),
                note,
            )
        };
        self.split_button.set_sensitive(can_split && cut.is_none());
        self.split_button.set_tooltip_text(Some(&tooltip));
        self.panel.set_visible(cut.is_some());
        if cut.is_some() {
            self.cut_label.set_label(&cut_text);
            for (label, text) in self.part_info.iter().zip(&parts) {
                label.set_label(text);
            }
            self.confirm_button.set_sensitive(can_split);
            self.confirm_button.set_tooltip_text(blocked.as_deref());
            self.snap_button.set_sensitive(snappable);
            let (note, warning) = note;
            self.cut_note.set_label(note);
            if warning {
                self.cut_note.add_css_class("warning-text");
                self.cut_note.remove_css_class("dim-label");
            } else {
                self.cut_note.remove_css_class("warning-text");
                self.cut_note.add_css_class("dim-label");
            }
        }
        self.area.queue_draw();
    }

    fn confirm_split(&self) {
        let (dir, cut, started, length, before, after) = {
            let state = self.state.borrow();
            let (Some(dir), Some(cut)) = (state.session.clone(), state.cut_ms) else {
                return;
            };
            if !state.can_split() {
                return;
            }
            let before = state.screenshots.iter().filter(|time| **time < cut).count();
            (
                dir,
                cut,
                state.started.clone(),
                state.length_ms,
                before,
                state.screenshots.len() - before,
            )
        };
        let session = Session::open(&dir).map(|session| session.title());
        let title = session.unwrap_or_default();
        let titles = self
            .titles
            .clone()
            .map(|entry| entry.text().trim().to_owned());
        let shown = |index: usize| {
            if titles[index].is_empty() {
                format!("Part {}", index + 1)
            } else {
                titles[index].clone()
            }
        };
        let line = |index: usize, start, length, shots| {
            let start = wall_clock(&started, start)
                .map(|time| format!("{time}, "))
                .unwrap_or_default();
            format!(
                "• <b>{}</b>: {start}{}, {shots} screenshot{}",
                glib::markup_escape_text(&shown(index)),
                format_duration(length),
                if shots == 1 { "" } else { "s" }
            )
        };
        let dialog = adw::AlertDialog::new(
            Some(&format!(
                "Split “{title}” at {}?",
                format_timeline_timestamp(cut)
            )),
            None,
        );
        dialog.set_body("The recording becomes two new sessions:");
        let extra = gtk::Box::new(gtk::Orientation::Vertical, 12);
        let parts = gtk::Label::new(None);
        parts.set_markup(&format!(
            "{}\n{}",
            line(0, 0, cut, before),
            line(1, cut, length - cut, after)
        ));
        parts.set_xalign(0.0);
        parts.set_wrap(true);
        parts.set_halign(gtk::Align::Center);
        extra.append(&parts);
        let note = gtk::Label::new(Some(
            "Both tracks are cut at the same sample. Screenshots go with the part they were taken in. Meeting details are copied to both. The original session is kept as a backup: its audio is archived as FLAC and it stays in the sidebar as Archived.",
        ));
        note.set_wrap(true);
        note.set_justify(gtk::Justification::Center);
        note.add_css_class("caption");
        note.add_css_class("dim-label");
        extra.append(&note);
        dialog.set_extra_child(Some(&extra));
        dialog.add_responses(&[("cancel", "Cancel"), ("split", "Split")]);
        dialog.set_default_response(Some("split"));
        dialog.set_close_response("cancel");
        dialog.set_response_appearance("split", adw::ResponseAppearance::Destructive);
        let player = self.clone();
        dialog.connect_response(Some("split"), move |_, _| {
            let handler = player.on_split.borrow().clone();
            if let Some(handler) = handler {
                player.playback.stop();
                handler(dir.clone(), cut, titles.clone());
            }
        });
        dialog.present(Some(&self.window));
    }

    fn draw(
        &self,
        area: &gtk::DrawingArea,
        cr: &cairo::Context,
        width: f64,
        height: f64,
    ) -> Result<(), cairo::Error> {
        let state = self.state.borrow();
        let fg = area.color();
        let ink = |cr: &cairo::Context, alpha: f64| {
            cr.set_source_rgba(
                f64::from(fg.red()),
                f64::from(fg.green()),
                f64::from(fg.blue()),
                f64::from(fg.alpha()) * alpha,
            );
        };
        let length = state.length_ms.max(1);
        let x = |ms: u64| x_of(ms, length, width);
        let lanes_top = RULER;
        let lanes_bottom = height - MARKERS;
        let lane_height = (lanes_bottom - lanes_top) / state.tracks.len().max(1) as f64;
        cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);

        if let Some(cut) = state.cut_ms {
            for (from, to, color, alpha) in [
                (GUTTER, x(cut), FIRST_COLOR, 0.07),
                (x(cut), width - RIGHT, SECOND_COLOR, 0.09),
            ] {
                rgb(cr, color, alpha);
                cr.rectangle(from, lanes_top, to - from, lanes_bottom - lanes_top);
                cr.fill()?;
            }
            cr.save()?;
            for &(start, end) in state.quiet() {
                let (from, to) = (x(start), x(end));
                if to - from >= 3.0 {
                    cr.rectangle(from, lanes_top, to - from, lanes_bottom - lanes_top);
                }
            }
            cr.clip();
            ink(cr, 0.16);
            cr.set_line_width(1.0);
            let mut offset = GUTTER - (lanes_bottom - lanes_top);
            while offset < width {
                cr.move_to(offset, lanes_bottom);
                cr.line_to(offset + (lanes_bottom - lanes_top), lanes_top);
                offset += 6.0;
            }
            cr.stroke()?;
            cr.restore()?;
        }

        // Ruler.
        cr.set_font_size(10.0);
        let step = ruler_step(length, width - GUTTER - RIGHT);
        let mut tick = 0;
        while tick <= length {
            let tick_x = x(tick);
            ink(cr, 0.12);
            cr.rectangle(tick_x.round(), lanes_top, 1.0, lanes_bottom - lanes_top);
            cr.fill()?;
            ink(cr, 0.55);
            let label = format_timeline_timestamp(tick);
            let extents = cr.text_extents(&label)?;
            let label_x = (tick_x - extents.width() / 2.0)
                .clamp(GUTTER - 4.0, width - RIGHT - extents.width());
            cr.move_to(label_x, RULER - 8.0);
            cr.show_text(&label)?;
            tick += step;
        }

        // Lanes.
        for (index, source) in state.tracks.iter().enumerate() {
            let top = lanes_top + index as f64 * lane_height;
            let middle = top + lane_height / 2.0;
            if index > 0 {
                ink(cr, 0.1);
                cr.rectangle(GUTTER, top, width - GUTTER - RIGHT, 1.0);
                cr.fill()?;
            }
            let (name, role) = match source {
                AudioSource::Mic => ("Mic", "you"),
                AudioSource::System => ("System", "remote"),
            };
            cr.set_font_size(11.0);
            ink(cr, 0.75);
            let extents = cr.text_extents(name)?;
            cr.move_to(GUTTER - 8.0 - extents.width(), middle - 1.0);
            cr.show_text(name)?;
            cr.set_font_size(9.5);
            ink(cr, 0.45);
            let extents = cr.text_extents(role)?;
            cr.move_to(GUTTER - 8.0 - extents.width(), middle + 11.0);
            cr.show_text(role)?;

            let muted = state.muted.contains(source);
            let levels = state.levels.as_ref().and_then(|levels| {
                levels
                    .envelope
                    .tracks
                    .iter()
                    .position(|(candidate, _)| candidate == source)
                    .map(|position| (&levels.envelope.tracks[position].1, levels.norms[position]))
            });
            let Some((levels, norm)) = levels else {
                ink(cr, 0.2);
                cr.rectangle(GUTTER, middle, width - GUTTER - RIGHT, 1.0);
                cr.fill()?;
                continue;
            };
            let half = lane_height / 2.0 - 5.0;
            let lane_color = match source {
                AudioSource::Mic => MIC_COLOR,
                AudioSource::System => SYSTEM_COLOR,
            };
            let alpha = if muted { 0.3 } else { 1.0 };
            let columns = (width - GUTTER - RIGHT).max(1.0) as usize;
            for column in 0..columns {
                let column_x = GUTTER + column as f64;
                let from = ms_of(column_x, length, width) / BUCKET_MS;
                let to = (ms_of(column_x + 1.0, length, width) / BUCKET_MS).max(from + 1);
                let Some(slice) = levels.get(from as usize..(to as usize).min(levels.len())) else {
                    continue;
                };
                if slice.is_empty() {
                    continue;
                }
                let peak = slice.iter().map(|level| level.peak).fold(0.0, f32::max);
                let rms = slice.iter().map(|level| level.rms).fold(0.0, f32::max);
                let color = match state.cut_ms {
                    Some(cut) if ms_of(column_x, length, width) >= cut => SECOND_COLOR,
                    Some(_) => FIRST_COLOR,
                    None => lane_color,
                };
                let peak_height = (f64::from(peak / norm).min(1.0) * half).max(0.5);
                let rms_height = (f64::from(rms / norm).min(1.0) * half).max(0.5);
                rgb(cr, color, 0.45 * alpha);
                cr.rectangle(column_x, middle - peak_height, 1.0, peak_height * 2.0);
                cr.fill()?;
                rgb(cr, color, alpha);
                cr.rectangle(column_x, middle - rms_height, 1.0, rms_height * 2.0);
                cr.fill()?;
            }
        }
        if state.levels.is_none() {
            cr.set_font_size(11.0);
            ink(cr, 0.55);
            let text = if state.envelope_error.is_some() {
                "Could not read the waveform"
            } else {
                "Reading the recording…"
            };
            let extents = cr.text_extents(text)?;
            cr.move_to(
                GUTTER + (width - GUTTER - RIGHT - extents.width()) / 2.0,
                lanes_top + 14.0,
            );
            cr.show_text(text)?;
        }

        // Screenshot markers.
        let marker_middle = lanes_bottom + MARKERS / 2.0;
        for time in &state.screenshots {
            let marker_x = x(*time);
            rgb(cr, SHOT_COLOR, 0.85);
            rounded(cr, marker_x - 4.5, marker_middle - 4.5, 9.0, 9.0, 2.0);
            cr.fill()?;
        }

        if let Some(cut) = state.cut_ms {
            let cut_x = x(cut);
            rgb(cr, SECOND_COLOR, 1.0);
            cr.rectangle(cut_x - 1.0, lanes_top, 2.0, lanes_bottom - lanes_top + 4.0);
            cr.fill()?;
            cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Bold);
            cr.set_font_size(10.0);
            let label = format!("⇔ {}", format_timeline_timestamp(cut));
            let extents = cr.text_extents(&label)?;
            let pill_width = extents.width() + 14.0;
            let pill_x = (cut_x - pill_width / 2.0).clamp(GUTTER, width - RIGHT - pill_width);
            rounded(cr, pill_x, marker_middle - 8.0, pill_width, 16.0, 8.0);
            cr.fill()?;
            cr.set_source_rgb(1.0, 1.0, 1.0);
            cr.move_to(pill_x + 7.0, marker_middle + 3.5);
            cr.show_text(&label)?;
            for (from, to, color, part) in
                [(0, cut, FIRST_COLOR, 1), (cut, length, SECOND_COLOR, 2)]
            {
                let label = format!("Part {part} · {}", format_duration(to - from));
                let extents = cr.text_extents(&label)?;
                let center = (x(from) + x(to)) / 2.0;
                let pill_width = extents.width() + 12.0;
                if pill_width > x(to) - x(from) {
                    continue;
                }
                let pill_x = center - pill_width / 2.0;
                rgb(cr, color, 1.0);
                rounded(cr, pill_x, 2.0, pill_width, 15.0, 7.5);
                cr.fill()?;
                cr.set_source_rgb(1.0, 1.0, 1.0);
                cr.move_to(pill_x + 6.0, 13.0);
                cr.show_text(&label)?;
            }
            cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
        }

        // Playhead.
        let head = x(state.position());
        ink(cr, 0.9);
        cr.rectangle(
            head - 0.75,
            lanes_top - 2.0,
            1.5,
            lanes_bottom - lanes_top + 2.0,
        );
        cr.fill()?;
        cr.move_to(head - 5.0, lanes_top - 7.0);
        cr.line_to(head + 5.0, lanes_top - 7.0);
        cr.line_to(head, lanes_top - 1.0);
        cr.close_path();
        cr.fill()?;

        // Hover time.
        if let Some(hover) = state
            .hover_x
            .filter(|hover| *hover >= GUTTER && state.drag.is_none())
        {
            ink(cr, 0.35);
            cr.rectangle(hover.round(), lanes_top, 1.0, lanes_bottom - lanes_top);
            cr.fill()?;
            cr.set_font_size(10.0);
            let label = format_timeline_timestamp(ms_of(hover, length, width));
            let extents = cr.text_extents(&label)?;
            let pill_width = extents.width() + 10.0;
            let pill_x = (hover + 6.0).min(width - RIGHT - pill_width);
            ink(cr, 0.85);
            rounded(cr, pill_x, lanes_top + 3.0, pill_width, 15.0, 4.0);
            cr.fill()?;
            let background = if fg.red() + fg.green() + fg.blue() > 1.5 {
                0.12
            } else {
                1.0
            };
            cr.set_source_rgb(background, background, background);
            cr.move_to(pill_x + 5.0, lanes_top + 14.0);
            cr.show_text(&label)?;
        }
        Ok(())
    }
}

fn rgb(cr: &cairo::Context, (red, green, blue): Rgb, alpha: f64) {
    cr.set_source_rgba(red, green, blue, alpha);
}

fn rounded(cr: &cairo::Context, x: f64, y: f64, width: f64, height: f64, radius: f64) {
    cr.new_sub_path();
    cr.arc(x + width - radius, y + radius, radius, -PI / 2.0, 0.0);
    cr.arc(
        x + width - radius,
        y + height - radius,
        radius,
        0.0,
        PI / 2.0,
    );
    cr.arc(x + radius, y + height - radius, radius, PI / 2.0, PI);
    cr.arc(x + radius, y + radius, radius, PI, 1.5 * PI);
    cr.close_path();
}

fn x_of(ms: u64, length_ms: u64, width: f64) -> f64 {
    let span = (width - GUTTER - RIGHT).max(1.0);
    GUTTER + span * ms as f64 / length_ms.max(1) as f64
}

fn ms_of(x: f64, length_ms: u64, width: f64) -> u64 {
    let span = (width - GUTTER - RIGHT).max(1.0);
    let fraction = ((x - GUTTER) / span).clamp(0.0, 1.0);
    (fraction * length_ms as f64).round() as u64
}

/// The tick spacing that keeps ruler labels about 90 px apart.
fn ruler_step(length_ms: u64, span: f64) -> u64 {
    const STEPS: [u64; 14] = [
        1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1_800, 3_600, 7_200,
    ];
    let wanted = length_ms as f64 * 90.0 / span.max(1.0);
    STEPS
        .iter()
        .map(|seconds| seconds * 1000)
        .find(|step| *step as f64 >= wanted)
        .unwrap_or(14_400_000)
}

/// The level drawn full height: nearly the loudest, so that one click does
/// not flatten everything else.
fn display_norm(levels: &[track::Level]) -> f32 {
    let mut peaks = levels.iter().map(|level| level.peak).collect::<Vec<_>>();
    if peaks.is_empty() {
        return 1.0;
    }
    let index = (peaks.len() * 995 / 1000).min(peaks.len() - 1);
    let (_, norm, _) = peaks.select_nth_unstable_by(index, f32::total_cmp);
    norm.max(0.02)
}

fn wall_clock(started: &str, offset_ms: u64) -> Option<String> {
    session::wallclock_after(started, offset_ms)
        .map(|clock| format!("{:02}:{:02}", clock.time.hour, clock.time.minute))
}

/// "Starts 09:30 · 40 min 31 s · 5 screenshots".
fn part_summary(started: &str, start_ms: u64, length_ms: u64, shots: usize) -> String {
    let start = wall_clock(started, start_ms)
        .map(|time| format!("Starts {time} · "))
        .unwrap_or_default();
    format!(
        "{start}{} · {shots} screenshot{}",
        format_duration(length_ms),
        if shots == 1 { "" } else { "s" }
    )
}

fn envelope_key(session: &Session) -> EnvelopeKey {
    [AudioSource::Mic, AudioSource::System]
        .into_iter()
        .map(|source| {
            std::fs::metadata(session.stored_audio_path(source)).map_or((0, None), |metadata| {
                (metadata.len(), metadata.modified().ok())
            })
        })
        .collect()
}

/// The chosen tracks from one point on, mixed into one stream of f32le
/// samples; a track that ends early falls silent.
struct MixedStream {
    tracks: Vec<TrackReader>,
    remaining: u64,
    chunk: Vec<f32>,
    mix: Vec<f32>,
    bytes: Vec<u8>,
    offset: usize,
}

impl MixedStream {
    fn open(
        session: &Session,
        sources: &[AudioSource],
        from_ms: u64,
        until_ms: u64,
    ) -> io::Result<Self> {
        let start = from_ms * SAMPLES_PER_MS;
        let mut tracks = Vec::new();
        for source in sources {
            tracks.extend(TrackReader::open(session, *source, start)?);
        }
        Ok(Self {
            tracks,
            remaining: until_ms.saturating_sub(from_ms) * SAMPLES_PER_MS,
            chunk: Vec::new(),
            mix: Vec::new(),
            bytes: Vec::new(),
            offset: 0,
        })
    }
}

impl Read for MixedStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.offset == self.bytes.len() {
            if self.remaining == 0 {
                return Ok(0);
            }
            let wanted = MIX_SAMPLES.min(self.remaining) as usize;
            self.mix.clear();
            self.mix.resize(wanted, 0.0);
            for track in &mut self.tracks {
                track.read(&mut self.chunk, wanted)?;
                for (mixed, sample) in self.mix.iter_mut().zip(&self.chunk) {
                    *mixed += if sample.is_finite() { *sample } else { 0.0 };
                }
            }
            self.bytes.clear();
            self.bytes.extend(
                self.mix
                    .iter()
                    .flat_map(|sample| sample.clamp(-1.0, 1.0).to_le_bytes()),
            );
            self.offset = 0;
            self.remaining -= wanted as u64;
        }
        let count = buffer.len().min(self.bytes.len() - self.offset);
        buffer[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
        self.offset += count;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruler_labels_stay_apart_on_round_steps() {
        assert_eq!(ruler_step(67 * 60_000, 700.0), 600_000);
        assert_eq!(ruler_step(20_000, 700.0), 5_000);
    }

    #[test]
    fn part_summary_reads_like_the_mockup() {
        assert_eq!(
            part_summary("2026-10-23T09:30:00+02:00", 2_431_000, 1_609_000, 3),
            "Starts 10:10 · 26 min 49 s · 3 screenshots"
        );
    }
}
