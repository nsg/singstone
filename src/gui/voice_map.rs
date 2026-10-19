use crate::speaker::classify::Flag as DotFlag;
use crate::speaker::database::{PruneReason, SpeakerDatabase, unix_now};
use crate::speaker::tidy::{RESTORE_PERIOD, Verdict};
use crate::speaker::voice_map::{Dot, VoiceMap, assign_colors};
use crate::types::DEFAULT_SPEAKER_THRESHOLD;
use adw::prelude::*;
use gtk::{cairo, glib};
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::RefCell;
use std::f64::consts::TAU;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

const BACKGROUND: [f64; 3] = [0.035, 0.047, 0.075];
const INK: [f64; 3] = [0.93, 0.94, 0.96];
const MUTED: [f64; 3] = [0.58, 0.62, 0.70];
const OVERLAP: [f64; 3] = [1.0, 0.42, 0.42];
const CAUTION: [f64; 3] = [1.0, 0.72, 0.30];
const PADDING: f64 = 70.0;
const HIT_RADIUS: f64 = 9.0;
const PILL_FONT: f64 = 11.0;
const PILL_HEIGHT: f64 = 18.0;
const MAX_ZOOM: f64 = 40.0;
/// Fewer outliers than this show on the map and in hover cards only.
const MIN_LISTED_OUTLIERS: usize = 3;

/// Hues avoid the red and amber kept for problem marks.
const HUES: [f64; 8] = [145.0, 175.0, 200.0, 235.0, 265.0, 295.0, 325.0, 350.0];

#[derive(Clone, Copy, PartialEq)]
enum Hover {
    Dot(usize),
    Removed(usize),
    Person(usize),
}

/// A change to one sample, offered in the popover of a clicked dot.
#[derive(Clone)]
enum SampleAction {
    Remove { person: String, embedding: Vec<f32> },
    Restore { person: String, embedding: Vec<f32> },
}

#[derive(Clone, PartialEq)]
struct SceneKey {
    width: i32,
    height: i32,
    scale: i32,
    zoom: f64,
    pan: (f64, f64),
    focus: Vec<usize>,
    problems: bool,
    show_removed: bool,
    generation: u64,
}

#[derive(Default)]
struct State {
    map: Option<VoiceMap>,
    colors: Vec<[f64; 3]>,
    world_center: [f64; 2],
    world_span: [f64; 2],
    zoom: f64,
    pan: (f64, f64),
    drag_origin: Option<(f64, f64)>,
    pointer: (f64, f64),
    hover: Option<Hover>,
    /// People shown at full strength; everyone when empty.
    focus: Vec<usize>,
    problems: bool,
    show_removed: bool,
    generation: u64,
    scene: Option<(SceneKey, cairo::ImageSurface)>,
    loaded: Option<(PathBuf, Option<SystemTime>)>,
    loading: bool,
    action: Option<SampleAction>,
}

impl State {
    fn base_scale(&self, width: f64, height: f64) -> f64 {
        let x = (width - 2.0 * PADDING).max(40.0) / self.world_span[0].max(1e-3);
        let y = (height - 2.0 * PADDING).max(40.0) / self.world_span[1].max(1e-3);
        x.min(y)
    }

    fn to_screen(&self, point: [f32; 2], width: f64, height: f64) -> (f64, f64) {
        let scale = self.base_scale(width, height) * self.zoom;
        (
            width / 2.0 + (f64::from(point[0]) - self.world_center[0]) * scale + self.pan.0,
            height / 2.0 + (f64::from(point[1]) - self.world_center[1]) * scale + self.pan.1,
        )
    }

    fn to_world(&self, x: f64, y: f64, width: f64, height: f64) -> [f64; 2] {
        let scale = self.base_scale(width, height) * self.zoom;
        [
            (x - width / 2.0 - self.pan.0) / scale + self.world_center[0],
            (y - height / 2.0 - self.pan.1) / scale + self.world_center[1],
        ]
    }

    fn zoom_at(&mut self, factor: f64, x: f64, y: f64, width: f64, height: f64) {
        let world = self.to_world(x, y, width, height);
        self.zoom = (self.zoom * factor).clamp(0.5, MAX_ZOOM);
        let scale = self.base_scale(width, height) * self.zoom;
        self.pan = (
            x - width / 2.0 - (world[0] - self.world_center[0]) * scale,
            y - height / 2.0 - (world[1] - self.world_center[1]) * scale,
        );
    }

    fn frame_people(&mut self, people: &[usize], width: f64, height: f64) {
        let Some(map) = &self.map else { return };
        let points = map
            .dots
            .iter()
            .filter(|dot| people.contains(&dot.person))
            .map(|dot| dot.position)
            .chain(people.iter().map(|&p| map.people[p].center))
            .collect::<Vec<_>>();
        let Some((min, max)) = bounds(&points) else {
            return;
        };
        let span = [(max[0] - min[0]).max(0.05), (max[1] - min[1]).max(0.05)];
        let base = self.base_scale(width, height);
        let fit = ((width - 2.0 * PADDING) / span[0]).min((height - 2.0 * PADDING) / span[1]);
        self.zoom = (fit / base * 0.75).clamp(1.0, MAX_ZOOM);
        let scale = base * self.zoom;
        let middle = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
        self.pan = (
            -(middle[0] - self.world_center[0]) * scale,
            -(middle[1] - self.world_center[1]) * scale,
        );
    }

    fn hit(&self, x: f64, y: f64, width: f64, height: f64) -> Option<Hover> {
        let map = self.map.as_ref()?;
        let distance = |p: [f32; 2]| {
            let (sx, sy) = self.to_screen(p, width, height);
            (sx - x).hypot(sy - y)
        };
        let centers = map
            .people
            .iter()
            .enumerate()
            .map(|(i, p)| (distance(p.center), i))
            .filter(|(d, _)| *d <= HIT_RADIUS)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, person)) = centers {
            return Some(Hover::Person(person));
        }
        let visible = |dot: &&Dot| self.focus.is_empty() || self.focus.contains(&dot.person);
        let dots = map
            .dots
            .iter()
            .enumerate()
            .filter(|(_, dot)| visible(dot))
            .map(|(i, dot)| (distance(dot.position), Hover::Dot(i)));
        let removed = map
            .removed
            .iter()
            .enumerate()
            .filter(|(_, dot)| self.show_removed && visible(dot))
            .map(|(i, dot)| (distance(dot.position), Hover::Removed(i)));
        dots.chain(removed)
            .filter(|(d, _)| *d <= HIT_RADIUS)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, hover)| hover)
    }
}

fn bounds(points: &[[f32; 2]]) -> Option<([f64; 2], [f64; 2])> {
    let first = points.first()?;
    let mut min = [f64::from(first[0]), f64::from(first[1])];
    let mut max = min;
    for p in points {
        for axis in 0..2 {
            min[axis] = min[axis].min(f64::from(p[axis]));
            max[axis] = max[axis].max(f64::from(p[axis]));
        }
    }
    Some((min, max))
}

type Callback = Box<dyn Fn()>;

#[derive(Clone)]
pub struct VoiceMapView {
    pub root: gtk::Box,
    area: gtk::DrawingArea,
    canvas_stack: gtk::Stack,
    status: adw::StatusPage,
    side: gtk::Box,
    findings: gtk::ListBox,
    voices: gtk::ListBox,
    summary: gtk::Label,
    popover: gtk::Popover,
    action_button: gtk::Button,
    changed: Rc<RefCell<Option<Callback>>>,
    state: Rc<RefCell<State>>,
}

impl VoiceMapView {
    pub fn new() -> Self {
        let state = Rc::new(RefCell::new(State {
            zoom: 1.0,
            problems: true,
            ..State::default()
        }));
        let area = gtk::DrawingArea::builder()
            .hexpand(true)
            .vexpand(true)
            .build();
        area.set_cursor_from_name(Some("crosshair"));

        let overlay = gtk::Overlay::new();
        overlay.set_child(Some(&area));
        let tools = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        tools.add_css_class("voice-map-tools");
        tools.set_halign(gtk::Align::End);
        tools.set_valign(gtk::Align::Start);
        tools.set_margin_top(12);
        tools.set_margin_end(12);
        let problems = gtk::ToggleButton::builder()
            .icon_name("dialog-warning-symbolic")
            .tooltip_text("Mark outliers, stray samples and close voices")
            .active(true)
            .build();
        let removed = gtk::ToggleButton::builder()
            .icon_name("user-trash-symbolic")
            .tooltip_text("Show removed samples")
            .build();
        let fit = gtk::Button::from_icon_name("zoom-fit-best-symbolic");
        fit.set_tooltip_text(Some("Show everyone"));
        tools.append(&problems);
        tools.append(&removed);
        tools.append(&fit);
        overlay.add_overlay(&tools);

        let status = adw::StatusPage::builder()
            .icon_name("system-users-symbolic")
            .title("No learned voices yet")
            .description("Name someone in a transcript to teach Singstone their voice")
            .build();
        let loading = gtk::Box::new(gtk::Orientation::Vertical, 12);
        loading.set_valign(gtk::Align::Center);
        let spinner = gtk::Spinner::builder()
            .spinning(true)
            .width_request(32)
            .height_request(32)
            .build();
        loading.append(&spinner);
        let loading_label = gtk::Label::new(Some("Laying out voices…"));
        loading_label.add_css_class("dim-label");
        loading.append(&loading_label);
        let canvas_stack = gtk::Stack::new();
        canvas_stack.add_named(&overlay, Some("map"));
        canvas_stack.add_named(&status, Some("empty"));
        canvas_stack.add_named(&loading, Some("loading"));
        canvas_stack.set_hexpand(true);

        let side = gtk::Box::new(gtk::Orientation::Vertical, 12);
        side.set_margin_top(18);
        side.set_margin_bottom(18);
        side.set_margin_start(14);
        side.set_margin_end(14);
        let summary = gtk::Label::new(None);
        summary.set_xalign(0.0);
        summary.set_wrap(true);
        summary.add_css_class("dim-label");
        summary.add_css_class("caption");
        let findings_title = gtk::Label::new(Some("Findings"));
        findings_title.set_xalign(0.0);
        findings_title.add_css_class("heading");
        let findings = gtk::ListBox::new();
        findings.add_css_class("boxed-list");
        findings.set_selection_mode(gtk::SelectionMode::None);
        let voices_title = gtk::Label::new(Some("Voices"));
        voices_title.set_xalign(0.0);
        voices_title.add_css_class("heading");
        let voices = gtk::ListBox::new();
        voices.add_css_class("boxed-list");
        voices.set_selection_mode(gtk::SelectionMode::None);
        side.append(&findings_title);
        side.append(&findings);
        side.append(&voices_title);
        side.append(&voices);
        side.append(&summary);
        let side_scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .child(&side)
            .width_request(320)
            .build();

        let side_panel = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        side_panel.append(&gtk::Separator::new(gtk::Orientation::Vertical));
        side_panel.append(&side_scroll);
        let root = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        root.append(&canvas_stack);
        root.append(&side_panel);

        let action_button = gtk::Button::new();
        action_button.add_css_class("flat");
        let popover = gtk::Popover::builder()
            .child(&action_button)
            .position(gtk::PositionType::Top)
            .build();
        popover.set_parent(&area);

        let view = Self {
            root,
            area,
            canvas_stack,
            status,
            side: side_panel,
            findings,
            voices,
            summary,
            popover,
            action_button,
            changed: Rc::new(RefCell::new(None)),
            state,
        };
        view.wire(&problems, &removed, &fit);
        view
    }

    /// Calls `callback` after a sample was removed or restored here.
    pub fn connect_changed(&self, callback: impl Fn() + 'static) {
        *self.changed.borrow_mut() = Some(Box::new(callback));
    }

    fn wire(&self, problems: &gtk::ToggleButton, removed: &gtk::ToggleButton, fit: &gtk::Button) {
        let state = self.state.clone();
        self.area.set_draw_func(move |area, cr, width, height| {
            let _ = draw(
                &mut state.borrow_mut(),
                area.scale_factor(),
                cr,
                width,
                height,
            );
        });

        let motion = gtk::EventControllerMotion::new();
        let view = self.clone();
        motion.connect_motion(move |_, x, y| {
            let (width, height) = view.size();
            let mut state = view.state.borrow_mut();
            state.pointer = (x, y);
            if state.drag_origin.is_some() {
                return;
            }
            let hover = state.hit(x, y, width, height);
            if hover != state.hover {
                state.hover = hover;
                drop(state);
                view.area.queue_draw();
            } else if hover.is_some() {
                drop(state);
                view.area.queue_draw();
            }
        });
        let view = self.clone();
        motion.connect_leave(move |_| {
            view.state.borrow_mut().hover = None;
            view.area.queue_draw();
        });
        self.area.add_controller(motion);

        let scroll = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::VERTICAL);
        let view = self.clone();
        scroll.connect_scroll(move |_, _, dy| {
            let (width, height) = view.size();
            let mut state = view.state.borrow_mut();
            let (x, y) = state.pointer;
            state.zoom_at((-dy * 0.18).exp(), x, y, width, height);
            state.hover = None;
            drop(state);
            view.area.queue_draw();
            glib::Propagation::Stop
        });
        self.area.add_controller(scroll);

        let drag = gtk::GestureDrag::new();
        let view = self.clone();
        drag.connect_drag_begin(move |_, _, _| {
            let mut state = view.state.borrow_mut();
            state.drag_origin = Some(state.pan);
        });
        let view = self.clone();
        drag.connect_drag_update(move |_, dx, dy| {
            let mut state = view.state.borrow_mut();
            if let Some(origin) = state.drag_origin
                && dx.hypot(dy) > 3.0
            {
                state.pan = (origin.0 + dx, origin.1 + dy);
                state.hover = None;
                drop(state);
                view.area.set_cursor_from_name(Some("grabbing"));
                view.area.queue_draw();
            }
        });
        let view = self.clone();
        drag.connect_drag_end(move |gesture, dx, dy| {
            view.area.set_cursor_from_name(Some("crosshair"));
            view.state.borrow_mut().drag_origin = None;
            if dx.hypot(dy) > 3.0 {
                return;
            }
            let Some((x, y)) = gesture.start_point() else {
                return;
            };
            let (width, height) = view.size();
            let hit = view.state.borrow().hit(x, y, width, height);
            let person = match (hit, view.state.borrow().map.as_ref()) {
                (Some(Hover::Person(p)), _) => Some(p),
                (Some(Hover::Dot(d)), Some(map)) => Some(map.dots[d].person),
                (Some(Hover::Removed(d)), Some(map)) => Some(map.removed[d].person),
                _ => None,
            };
            view.set_focus(person.into_iter().collect(), false);
            if let Some(hit) = hit {
                view.offer_action(hit, x, y);
            }
        });
        self.area.add_controller(drag);

        let keys = gtk::EventControllerKey::new();
        let view = self.clone();
        keys.connect_key_pressed(move |_, key, _, _| {
            if key == gtk::gdk::Key::Escape {
                view.set_focus(Vec::new(), false);
                return glib::Propagation::Stop;
            }
            glib::Propagation::Proceed
        });
        self.root.add_controller(keys);

        let view = self.clone();
        problems.connect_toggled(move |button| {
            view.state.borrow_mut().problems = button.is_active();
            view.area.queue_draw();
        });
        let view = self.clone();
        removed.connect_toggled(move |button| {
            view.state.borrow_mut().show_removed = button.is_active();
            view.area.queue_draw();
        });
        let view = self.clone();
        self.action_button.connect_clicked(move |_| {
            view.popover.popdown();
            let action = view.state.borrow_mut().action.take();
            if let Some(action) = action {
                view.apply(action);
            }
        });
        let view = self.clone();
        fit.connect_clicked(move |_| {
            let mut state = view.state.borrow_mut();
            state.zoom = 1.0;
            state.pan = (0.0, 0.0);
            state.focus.clear();
            drop(state);
            view.area.queue_draw();
        });
    }

    /// Pops up the remove or restore button for a clicked sample.
    fn offer_action(&self, hit: Hover, x: f64, y: f64) {
        let mut state = self.state.borrow_mut();
        let Some(map) = &state.map else { return };
        let (action, label) = match hit {
            Hover::Dot(index) => {
                let dot = &map.dots[index];
                (
                    SampleAction::Remove {
                        person: map.people[dot.person].name.clone(),
                        embedding: dot.embedding.clone(),
                    },
                    "Remove sample",
                )
            }
            Hover::Removed(index) => {
                let dot = &map.removed[index];
                (
                    SampleAction::Restore {
                        person: map.people[dot.person].name.clone(),
                        embedding: dot.embedding.clone(),
                    },
                    "Restore",
                )
            }
            Hover::Person(_) => return,
        };
        state.action = Some(action);
        drop(state);
        self.action_button.set_label(label);
        self.popover
            .set_pointing_to(Some(&gtk::gdk::Rectangle::new(x as i32, y as i32, 1, 1)));
        self.popover.popup();
    }

    fn apply(&self, action: SampleAction) {
        let Some(path) = self.state.borrow().loaded.as_ref().map(|(p, _)| p.clone()) else {
            return;
        };
        let result = SpeakerDatabase::update(&path, |database| {
            let changed = match &action {
                SampleAction::Remove { person, embedding } => {
                    database.speakers.get_mut(person).is_some_and(|record| {
                        record.set_aside(embedding, PruneReason::User, unix_now())
                    })
                }
                SampleAction::Restore { person, embedding } => database
                    .speakers
                    .get_mut(person)
                    .is_some_and(|record| record.restore(embedding)),
            };
            database.refresh_centroids();
            changed
        });
        match result {
            Ok(_) => {
                self.load(&path);
                if let Some(changed) = self.changed.borrow().as_ref() {
                    changed();
                }
            }
            Err(error) => {
                let dialog = adw::AlertDialog::new(
                    Some("Could not change the sample"),
                    Some(&error.to_string()),
                );
                dialog.add_response("close", "Close");
                dialog.present(Some(&self.root));
            }
        }
    }

    fn size(&self) -> (f64, f64) {
        (f64::from(self.area.width()), f64::from(self.area.height()))
    }

    fn set_focus(&self, people: Vec<usize>, frame: bool) {
        let mut state = self.state.borrow_mut();
        if frame && !people.is_empty() {
            let (width, height) = self.size();
            let mut framed = people.clone();
            if let [person] = people[..]
                && let Some(map) = &state.map
            {
                framed.extend(map.close_pairs.iter().filter_map(|pair| {
                    (pair.a == person)
                        .then_some(pair.b)
                        .or((pair.b == person).then_some(pair.a))
                }));
            }
            state.frame_people(&framed, width, height);
        }
        state.focus = people;
        drop(state);
        self.area.queue_draw();
    }

    /// Recomputes the map when the database changed since the last load.
    pub fn load(&self, path: &Path) {
        let stamp = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        {
            let state = self.state.borrow();
            if state.loading
                || state
                    .loaded
                    .as_ref()
                    .is_some_and(|(p, s)| p == path && *s == stamp)
            {
                return;
            }
        }
        self.state.borrow_mut().loading = true;
        self.canvas_stack.set_visible_child_name("loading");
        let result = Arc::new(Mutex::new(None));
        let thread_result = result.clone();
        let db_path = path.to_path_buf();
        std::thread::spawn(move || {
            let map = match SpeakerDatabase::load(&db_path) {
                Ok(database) => Ok(Some(VoiceMap::build(
                    &database,
                    DEFAULT_SPEAKER_THRESHOLD,
                    unix_now(),
                ))),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(error.to_string()),
            };
            *thread_result.lock().expect("voice map mutex") = Some(map);
        });
        let view = self.clone();
        let path = path.to_path_buf();
        glib::timeout_add_local(Duration::from_millis(100), move || {
            let Some(result) = result.lock().expect("voice map mutex").take() else {
                return glib::ControlFlow::Continue;
            };
            let loaded = result.is_ok();
            view.show(result, path.clone(), stamp);
            if loaded {
                // The database may have changed while the map was computed.
                view.load(&path);
            }
            glib::ControlFlow::Break
        });
    }

    fn show(
        &self,
        result: Result<Option<VoiceMap>, String>,
        path: PathBuf,
        stamp: Option<SystemTime>,
    ) {
        let mut state = self.state.borrow_mut();
        state.loading = false;
        state.loaded = result.is_ok().then_some((path, stamp));
        match &result {
            Ok(_) => {
                self.status.set_title("No learned voices yet");
                self.status.set_description(Some(
                    "Name someone in a transcript to teach Singstone their voice",
                ));
            }
            Err(error) => {
                self.status.set_title("Could not read the speaker database");
                self.status
                    .set_description(Some(&glib::markup_escape_text(error)));
            }
        }
        let map = result.ok().flatten().filter(|map| !map.people.is_empty());
        let Some(map) = map else {
            state.map = None;
            drop(state);
            self.canvas_stack.set_visible_child_name("empty");
            self.fill_side(None, &[]);
            return;
        };
        // After a sample is removed or restored, keep the view where it was.
        let same_people = state.map.as_ref().is_some_and(|shown| {
            shown
                .people
                .iter()
                .map(|p| &p.name)
                .eq(map.people.iter().map(|p| &p.name))
        });
        if !same_people {
            let palette = palette();
            let lab = palette.iter().map(|c| c.1).collect::<Vec<_>>();
            let centers = map.people.iter().map(|p| p.center).collect::<Vec<_>>();
            state.colors = assign_colors(&centers, &lab)
                .into_iter()
                .map(|slot| palette[slot].0)
                .collect();
            let points = map
                .dots
                .iter()
                .chain(&map.removed)
                .map(|d| d.position)
                .chain(centers)
                .collect::<Vec<_>>();
            if let Some((min, max)) = bounds(&points) {
                state.world_center = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
                state.world_span = [(max[0] - min[0]).max(0.1), (max[1] - min[1]).max(0.1)];
            }
            state.zoom = 1.0;
            state.pan = (0.0, 0.0);
            state.focus.clear();
        }
        state.hover = None;
        state.generation += 1;
        let colors = state.colors.clone();
        state.map = Some(map);
        drop(state);
        self.canvas_stack.set_visible_child_name("map");
        let state = self.state.borrow();
        self.fill_side(state.map.as_ref(), &colors);
        drop(state);
        self.area.queue_draw();
    }

    fn fill_side(&self, map: Option<&VoiceMap>, colors: &[[f64; 3]]) {
        while let Some(child) = self.findings.first_child() {
            self.findings.remove(&child);
        }
        while let Some(child) = self.voices.first_child() {
            self.voices.remove(&child);
        }
        self.side.set_visible(map.is_some());
        let Some(map) = map else {
            self.summary.set_text("");
            return;
        };
        let total: usize = map.people.iter().map(|p| p.samples).sum();
        self.summary.set_text(&format!(
            "{} voices, {} samples. The distance between two centers is the cosine \
             distance between those voices; the scale bar spans it for two voices at the \
             match threshold ({:.2}). Samples sit closer to their center than their true \
             distance, and two dimensions cannot keep every distance exact (layout stress \
             {:.2}), so trust the findings and hover scores over the eye for close calls.",
            map.people.len(),
            total,
            map.threshold,
            map.stress
        ));

        // (rank, weight) orders findings by severity, then by size.
        let mut findings = Vec::new();
        for pair in &map.close_pairs {
            let (a, b) = (&map.people[pair.a].name, &map.people[pair.b].name);
            let overlapping = pair.overlapping(map.threshold);
            findings.push((
                (if overlapping { 0 } else { 2 }, 0),
                if overlapping {
                    "dialog-error-symbolic"
                } else {
                    "dialog-warning-symbolic"
                },
                format!("{a} and {b}"),
                format!(
                    "{} voices, score {:.2}",
                    if overlapping { "Overlapping" } else { "Close" },
                    pair.score
                ),
                vec![pair.a, pair.b],
            ));
        }
        let mut medians = map
            .people
            .iter()
            .filter(|p| p.samples >= 5)
            .map(|p| p.median_similarity)
            .collect::<Vec<_>>();
        medians.sort_by(f32::total_cmp);
        let typical = medians.get(medians.len() / 2).copied().unwrap_or(1.0);
        for (index, person) in map.people.iter().enumerate() {
            if person.strays > 0 {
                let mut rivals = map
                    .dots
                    .iter()
                    .filter(|d| d.person == index && d.flag == DotFlag::Stray)
                    .filter_map(|d| d.rival.map(|r| r.0))
                    .collect::<Vec<_>>();
                rivals.sort_unstable();
                rivals.dedup();
                let names = rivals
                    .iter()
                    .map(|&r| map.people[r].name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                findings.push((
                    (1, usize::MAX - person.strays),
                    "dialog-error-symbolic",
                    person.name.clone(),
                    format!(
                        "{} like {names}",
                        plural(person.strays, "sample sounds", "samples sound")
                    ),
                    vec![index],
                ));
            }
            if person.due > 0 || person.settling > 0 {
                let mut parts = Vec::new();
                if person.due > 0 {
                    parts.push(format!(
                        "{} due for removal",
                        plural(person.due, "sample", "samples")
                    ));
                }
                if person.settling > 0 {
                    parts.push(format!(
                        "{} settling",
                        plural(person.settling, "sample", "samples")
                    ));
                }
                findings.push((
                    (1, usize::MAX - person.due),
                    "user-trash-symbolic",
                    person.name.clone(),
                    parts.join(", "),
                    vec![index],
                ));
            }
            if person.samples >= 5 && person.median_similarity < typical - 0.1 {
                findings.push((
                    (3, 0),
                    "dialog-warning-symbolic",
                    person.name.clone(),
                    format!(
                        "Wide spread, median similarity {:.2}",
                        person.median_similarity
                    ),
                    vec![index],
                ));
            }
            if person.outliers >= MIN_LISTED_OUTLIERS {
                findings.push((
                    (4, usize::MAX - person.outliers),
                    "dialog-warning-symbolic",
                    person.name.clone(),
                    format!(
                        "{} far from the rest",
                        plural(person.outliers, "sample", "samples")
                    ),
                    vec![index],
                ));
            }
            if person.samples < 3 {
                findings.push((
                    (5, 0),
                    "dialog-information-symbolic",
                    person.name.clone(),
                    format!("Only {}", plural(person.samples, "sample", "samples")),
                    vec![index],
                ));
            }
        }
        findings.sort_by_key(|finding| finding.0);
        let mut findings = findings
            .into_iter()
            .map(|(_, icon, title, subtitle, person)| (icon, title, subtitle, person))
            .collect::<Vec<_>>();
        if findings.is_empty() {
            findings.push((
                "emblem-ok-symbolic",
                "Nothing stands out".into(),
                String::new(),
                Vec::new(),
            ));
        }
        for (icon, title, subtitle, people) in findings {
            let row = adw::ActionRow::builder()
                .title(glib::markup_escape_text(&title))
                .build();
            if !subtitle.is_empty() {
                row.set_subtitle(&glib::markup_escape_text(&subtitle));
            }
            let image = gtk::Image::from_icon_name(icon);
            match icon {
                "dialog-error-symbolic" => image.add_css_class("error"),
                "dialog-warning-symbolic" => image.add_css_class("warning"),
                "emblem-ok-symbolic" => image.add_css_class("success"),
                _ => image.add_css_class("dim-label"),
            }
            row.add_prefix(&image);
            if !people.is_empty() {
                row.set_activatable(true);
                let view = self.clone();
                row.connect_activated(move |_| view.set_focus(people.clone(), true));
            }
            self.findings.append(&row);
        }

        let mut order = (0..map.people.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| map.people[i].name.to_lowercase());
        for index in order {
            let person = &map.people[index];
            let row = adw::ActionRow::builder()
                .title(glib::markup_escape_text(&person.name))
                .subtitle(format!(
                    "{} · similarity {:.2}",
                    plural(person.samples, "sample", "samples"),
                    person.median_similarity
                ))
                .activatable(true)
                .build();
            row.add_prefix(&swatch(colors[index]));
            let view = self.clone();
            row.connect_activated(move |_| view.set_focus(vec![index], true));
            self.voices.append(&row);
        }
    }
}

fn plural(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

fn swatch(color: [f64; 3]) -> gtk::DrawingArea {
    let area = gtk::DrawingArea::builder()
        .content_width(14)
        .content_height(14)
        .valign(gtk::Align::Center)
        .build();
    area.set_draw_func(move |_, cr, width, height| {
        cr.arc(
            f64::from(width) / 2.0,
            f64::from(height) / 2.0,
            6.0,
            0.0,
            TAU,
        );
        cr.set_source_rgb(color[0], color[1], color[2]);
        let _ = cr.fill();
    });
    area
}

/// Sixteen sRGB colors with their OKLab coordinates.
fn palette() -> Vec<([f64; 3], [f32; 3])> {
    [(0.76, 0.15), (0.88, 0.10)]
        .into_iter()
        .flat_map(|(lightness, chroma)| {
            HUES.iter().map(move |hue| {
                let h = hue.to_radians();
                let lab = [lightness, chroma * h.cos(), chroma * h.sin()];
                (oklab_to_srgb(lab), lab.map(|v| v as f32))
            })
        })
        .collect()
}

fn oklab_to_srgb([l, a, b]: [f64; 3]) -> [f64; 3] {
    let l_ = (l + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m_ = (l - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s_ = (l - 0.089_484_177_5 * a - 1.291_485_548 * b).powi(3);
    let linear = [
        4.076_741_662_1 * l_ - 3.307_711_591_3 * m_ + 0.230_969_929_2 * s_,
        -1.268_438_004_6 * l_ + 2.609_757_401_1 * m_ - 0.341_319_396_5 * s_,
        -0.004_196_086_3 * l_ - 0.703_418_614_7 * m_ + 1.707_614_701 * s_,
    ];
    linear.map(|c| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    })
}

fn draw(
    state: &mut State,
    scale: i32,
    cr: &cairo::Context,
    width: i32,
    height: i32,
) -> Result<(), cairo::Error> {
    if state.map.is_none() || width <= 0 || height <= 0 {
        return Ok(());
    }
    let key = SceneKey {
        width,
        height,
        scale,
        zoom: state.zoom,
        pan: state.pan,
        focus: state.focus.clone(),
        problems: state.problems,
        show_removed: state.show_removed,
        generation: state.generation,
    };
    if state
        .scene
        .as_ref()
        .is_none_or(|(cached, _)| *cached != key)
    {
        let surface =
            cairo::ImageSurface::create(cairo::Format::ARgb32, width * scale, height * scale)?;
        surface.set_device_scale(f64::from(scale), f64::from(scale));
        {
            let scene = cairo::Context::new(&surface)?;
            draw_scene(state, &scene, f64::from(width), f64::from(height))?;
        }
        state.scene = Some((key, surface));
    }
    if let Some((_, surface)) = &state.scene {
        cr.set_source_surface(surface, 0.0, 0.0)?;
        cr.paint()?;
    }
    draw_hover(state, cr, f64::from(width), f64::from(height))
}

fn rgba(cr: &cairo::Context, color: [f64; 3], alpha: f64) {
    cr.set_source_rgba(color[0], color[1], color[2], alpha);
}

fn draw_scene(
    state: &State,
    cr: &cairo::Context,
    width: f64,
    height: f64,
) -> Result<(), cairo::Error> {
    let map = state.map.as_ref().expect("map is loaded");
    let glow = cairo::RadialGradient::new(
        width * 0.5,
        height * 0.45,
        0.0,
        width * 0.5,
        height * 0.45,
        width.max(height) * 0.75,
    );
    glow.add_color_stop_rgb(0.0, 0.075, 0.09, 0.14);
    glow.add_color_stop_rgb(1.0, BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]);
    cr.set_source(&glow)?;
    cr.paint()?;

    let unit = state.base_scale(width, height) * state.zoom;
    draw_grid(state, cr, width, height, unit)?;

    let dimmed = |person: usize| !state.focus.is_empty() && !state.focus.contains(&person);
    let point_radius = (1.6 + unit * 0.006).clamp(2.0, 4.5);

    cr.set_operator(cairo::Operator::Add);
    for (index, person) in map.people.iter().enumerate() {
        let (x, y) = state.to_screen(person.center, width, height);
        let radius = (f64::from(person.radius) * unit * 2.2).max(26.0);
        let color = state.colors[index];
        let alpha = if dimmed(index) { 0.04 } else { 0.20 };
        let halo = cairo::RadialGradient::new(x, y, 0.0, x, y, radius);
        halo.add_color_stop_rgba(0.0, color[0], color[1], color[2], alpha);
        halo.add_color_stop_rgba(0.55, color[0], color[1], color[2], alpha * 0.45);
        halo.add_color_stop_rgba(1.0, color[0], color[1], color[2], 0.0);
        cr.set_source(&halo)?;
        cr.arc(x, y, radius, 0.0, TAU);
        cr.fill()?;
    }
    for dot in &map.dots {
        let (x, y) = state.to_screen(dot.position, width, height);
        let alpha = if dimmed(dot.person) { 0.03 } else { 0.16 };
        rgba(cr, state.colors[dot.person], alpha);
        cr.arc(x, y, point_radius * 2.6, 0.0, TAU);
        cr.fill()?;
    }
    cr.set_operator(cairo::Operator::Over);

    if state.problems {
        for pair in &map.close_pairs {
            if dimmed(pair.a) && dimmed(pair.b) {
                continue;
            }
            let a = state.to_screen(map.people[pair.a].center, width, height);
            let b = state.to_screen(map.people[pair.b].center, width, height);
            let color = if pair.overlapping(map.threshold) {
                OVERLAP
            } else {
                CAUTION
            };
            rgba(cr, color, 0.85);
            cr.set_line_width(1.6);
            cr.set_dash(&[5.0, 4.0], 0.0);
            cr.move_to(a.0, a.1);
            cr.line_to(b.0, b.1);
            cr.stroke()?;
            cr.set_dash(&[], 0.0);
        }
    }

    if state.show_removed {
        for dot in &map.removed {
            let (x, y) = state.to_screen(dot.position, width, height);
            let alpha = if dimmed(dot.person) { 0.08 } else { 0.32 };
            rgba(cr, state.colors[dot.person], alpha);
            cr.arc(x, y, point_radius, 0.0, TAU);
            cr.fill()?;
            rgba(cr, MUTED, alpha);
            cr.set_line_width(1.0);
            cr.arc(x, y, point_radius + 2.5, 0.0, TAU);
            cr.stroke()?;
        }
    }

    for dot in &map.dots {
        let faded = dimmed(dot.person);
        let (x, y) = state.to_screen(dot.position, width, height);
        rgba(
            cr,
            state.colors[dot.person],
            if faded { 0.18 } else { 0.92 },
        );
        cr.arc(x, y, point_radius, 0.0, TAU);
        cr.fill()?;
        if !state.problems || faded {
            continue;
        }
        if let Some((reason, _)) = &dot.tidy
            && (dot.due() || dot.settling())
        {
            let color = reason_color(reason);
            let ring = point_radius + 3.5;
            rgba(cr, color, 0.95);
            cr.set_line_width(1.6);
            if dot.settling() {
                cr.set_dash(&[2.5, 2.5], 0.0);
            }
            cr.arc(x, y, ring, 0.0, TAU);
            cr.stroke()?;
            cr.set_dash(&[], 0.0);
            if dot.due() {
                cross(cr, x, y, ring + 2.5)?;
            }
            continue;
        }
        match dot.flag {
            DotFlag::None => {}
            DotFlag::Outlier | DotFlag::Isolated => {
                rgba(cr, CAUTION, 0.9);
                cr.set_line_width(1.3);
                cr.arc(x, y, point_radius + 3.5, 0.0, TAU);
                cr.stroke()?;
            }
            DotFlag::Stray => {
                if state.focus.contains(&dot.person)
                    && let Some((rival, _)) = dot.rival
                {
                    let (rx, ry) = state.to_screen(map.people[rival].center, width, height);
                    rgba(cr, OVERLAP, 0.35);
                    cr.set_line_width(1.0);
                    cr.move_to(x, y);
                    cr.line_to(rx, ry);
                    cr.stroke()?;
                }
                rgba(cr, OVERLAP, 0.95);
                cr.set_line_width(1.6);
                cr.arc(x, y, point_radius + 3.5, 0.0, TAU);
                cr.stroke()?;
            }
        }
    }

    for (index, person) in map.people.iter().enumerate() {
        let (x, y) = state.to_screen(person.center, width, height);
        let faded = dimmed(index);
        rgba(cr, BACKGROUND, if faded { 0.4 } else { 0.85 });
        cr.arc(x, y, 6.0, 0.0, TAU);
        cr.fill()?;
        rgba(cr, state.colors[index], if faded { 0.3 } else { 1.0 });
        cr.set_line_width(2.0);
        cr.arc(x, y, 4.5, 0.0, TAU);
        cr.stroke()?;
        rgba(cr, INK, if faded { 0.3 } else { 1.0 });
        cr.arc(x, y, 1.6, 0.0, TAU);
        cr.fill()?;
    }
    let mut pills = Vec::new();
    if state.problems {
        for pair in &map.close_pairs {
            if dimmed(pair.a) && dimmed(pair.b) {
                continue;
            }
            let a = state.to_screen(map.people[pair.a].center, width, height);
            let b = state.to_screen(map.people[pair.b].center, width, height);
            let color = if pair.overlapping(map.threshold) {
                OVERLAP
            } else {
                CAUTION
            };
            let (mut x, mut y) = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            let length = (b.0 - a.0).hypot(b.1 - a.1);
            if length < 80.0 {
                let (nx, ny) = if length > 1.0 {
                    (-(b.1 - a.1) / length, (b.0 - a.0) / length)
                } else {
                    (0.0, -1.0)
                };
                let side = if nx < 0.0 { -1.0 } else { 1.0 };
                x += side * nx * 40.0;
                y += side * ny * 40.0;
            }
            pills.push((format!("{:.2}", pair.score), x, y, color));
        }
    }
    set_font(cr, PILL_FONT, true);
    let mut obstacles = Vec::new();
    for (text, x, y, _) in &pills {
        let w = cr.text_extents(text)?.x_advance() + 12.0;
        obstacles.push((x - w / 2.0, y + PILL_HEIGHT / 2.0, w, PILL_HEIGHT));
    }
    draw_labels(state, cr, width, height, obstacles)?;
    for (text, x, y, color) in &pills {
        pill(cr, text, *x, *y, *color)?;
    }

    draw_scale(state, cr, width, height, unit)?;
    draw_legend(state, cr, height)
}

fn draw_grid(
    state: &State,
    cr: &cairo::Context,
    width: f64,
    height: f64,
    unit: f64,
) -> Result<(), cairo::Error> {
    let step = [0.05, 0.1, 0.2, 0.5]
        .into_iter()
        .find(|step| step * unit >= 60.0)
        .unwrap_or(1.0);
    let top_left = state.to_world(0.0, 0.0, width, height);
    let bottom_right = state.to_world(width, height, width, height);
    rgba(cr, INK, 0.035);
    cr.set_line_width(1.0);
    let mut x = (top_left[0] / step).floor() * step;
    while x <= bottom_right[0] {
        let (sx, _) = state.to_screen([x as f32, 0.0], width, height);
        cr.move_to(sx.round() + 0.5, 0.0);
        cr.line_to(sx.round() + 0.5, height);
        x += step;
    }
    let mut y = (top_left[1] / step).floor() * step;
    while y <= bottom_right[1] {
        let (_, sy) = state.to_screen([0.0, y as f32], width, height);
        cr.move_to(0.0, sy.round() + 0.5);
        cr.line_to(width, sy.round() + 0.5);
        y += step;
    }
    cr.stroke()
}

fn set_font(cr: &cairo::Context, size: f64, bold: bool) {
    cr.select_font_face(
        "Sans",
        cairo::FontSlant::Normal,
        if bold {
            cairo::FontWeight::Bold
        } else {
            cairo::FontWeight::Normal
        },
    );
    cr.set_font_size(size);
}

fn shadowed_text(
    cr: &cairo::Context,
    text: &str,
    x: f64,
    y: f64,
    color: [f64; 3],
    alpha: f64,
) -> Result<(), cairo::Error> {
    rgba(cr, BACKGROUND, 0.85 * alpha);
    for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 1.0)] {
        cr.move_to(x + dx, y + dy);
        cr.show_text(text)?;
    }
    rgba(cr, color, alpha);
    cr.move_to(x, y);
    cr.show_text(text)?;
    cr.new_path();
    Ok(())
}

fn pill(
    cr: &cairo::Context,
    text: &str,
    x: f64,
    y: f64,
    color: [f64; 3],
) -> Result<(), cairo::Error> {
    set_font(cr, PILL_FONT, true);
    let extents = cr.text_extents(text)?;
    let (w, h) = (extents.x_advance() + 12.0, PILL_HEIGHT);
    rounded(cr, x - w / 2.0, y - h / 2.0, w, h, h / 2.0);
    rgba(cr, BACKGROUND, 0.92);
    cr.fill_preserve()?;
    rgba(cr, color, 0.9);
    cr.set_line_width(1.2);
    cr.stroke()?;
    rgba(cr, color, 1.0);
    cr.move_to(x - extents.x_advance() / 2.0, y + 4.0);
    cr.show_text(text)?;
    cr.new_path();
    Ok(())
}

fn rounded(cr: &cairo::Context, x: f64, y: f64, w: f64, h: f64, r: f64) {
    cr.new_sub_path();
    cr.arc(x + w - r, y + r, r, -TAU / 4.0, 0.0);
    cr.arc(x + w - r, y + h - r, r, 0.0, TAU / 4.0);
    cr.arc(x + r, y + h - r, r, TAU / 4.0, TAU / 2.0);
    cr.arc(x + r, y + r, r, TAU / 2.0, TAU * 0.75);
    cr.close_path();
}

/// Obstacles and labels are (left, baseline, width, height) boxes.
fn draw_labels(
    state: &State,
    cr: &cairo::Context,
    width: f64,
    height: f64,
    obstacles: Vec<(f64, f64, f64, f64)>,
) -> Result<(), cairo::Error> {
    let map = state.map.as_ref().expect("map is loaded");
    set_font(cr, 12.5, true);
    let mut order = (0..map.people.len()).collect::<Vec<_>>();
    order.sort_by_key(|&i| {
        (
            !state.focus.contains(&i),
            std::cmp::Reverse(map.people[i].samples),
        )
    });
    let mut placed = obstacles;
    for index in order {
        let person = &map.people[index];
        let (x, y) = state.to_screen(person.center, width, height);
        let extents = cr.text_extents(&person.name)?;
        let (w, h) = (extents.x_advance(), 13.0);
        let candidates = [
            (x + 9.0, y - 7.0),
            (x + 9.0, y + 15.0),
            (x - 9.0 - w, y - 7.0),
            (x - 9.0 - w, y + 15.0),
            (x - w / 2.0, y - 13.0),
            (x - w / 2.0, y + 21.0),
        ];
        let free = |(lx, ly): (f64, f64)| {
            !placed.iter().any(|&(px, py, pw, ph)| {
                lx < px + pw + 4.0 && lx + w + 4.0 > px && ly - h < py + 2.0 && ly + 2.0 > py - ph
            })
        };
        let (spot, crowded) = match candidates.into_iter().find(|c| free(*c)) {
            Some(spot) => (spot, false),
            None => (candidates[0], true),
        };
        let faded = !state.focus.is_empty() && !state.focus.contains(&index);
        let alpha = if faded {
            0.3
        } else if crowded {
            0.55
        } else {
            1.0
        };
        shadowed_text(cr, &person.name, spot.0, spot.1, INK, alpha)?;
        placed.push((spot.0, spot.1, w, h));
    }
    Ok(())
}

fn draw_scale(
    state: &State,
    cr: &cairo::Context,
    width: f64,
    height: f64,
    unit: f64,
) -> Result<(), cairo::Error> {
    let Some(map) = &state.map else { return Ok(()) };
    let length = f64::from(1.0 - map.threshold) * unit;
    if !(8.0..=width * 0.6).contains(&length) {
        return Ok(());
    }
    let (x, y) = (width - 24.0 - length, height - 28.0);
    rgba(cr, INK, 0.7);
    cr.set_line_width(1.5);
    cr.move_to(x, y - 4.0);
    cr.line_to(x, y);
    cr.line_to(x + length, y);
    cr.line_to(x + length, y - 4.0);
    cr.stroke()?;
    set_font(cr, 11.0, false);
    let text = format!("centers at similarity {:.2}", map.threshold);
    let extents = cr.text_extents(&text)?;
    shadowed_text(
        cr,
        &text,
        x + length - extents.x_advance(),
        y - 9.0,
        MUTED,
        1.0,
    )
}

/// Sounding like someone else is marked red, being far from the rest amber.
fn reason_color(reason: &PruneReason) -> [f64; 3] {
    match reason {
        PruneReason::SoundsLike { .. } => OVERLAP,
        _ => CAUTION,
    }
}

/// The "due for removal" mark across a dot's ring.
fn cross(cr: &cairo::Context, x: f64, y: f64, reach: f64) -> Result<(), cairo::Error> {
    let d = reach * std::f64::consts::FRAC_1_SQRT_2;
    cr.set_line_width(1.6);
    cr.move_to(x - d, y - d);
    cr.line_to(x + d, y + d);
    cr.move_to(x - d, y + d);
    cr.line_to(x + d, y - d);
    cr.stroke()
}

#[derive(Clone, Copy, PartialEq)]
enum Mark {
    Ring,
    Settling,
    Due,
    Line,
}

fn draw_legend(state: &State, cr: &cairo::Context, height: f64) -> Result<(), cairo::Error> {
    if !state.problems {
        return Ok(());
    }
    let entries: [(&str, [f64; 3], Mark); 6] = [
        ("Sample far from its voice", CAUTION, Mark::Ring),
        ("Sample sounds like someone else", OVERLAP, Mark::Ring),
        (
            "Settling: judged after more sessions",
            MUTED,
            Mark::Settling,
        ),
        ("Due for removal at the next tidy", MUTED, Mark::Due),
        ("Close voices", CAUTION, Mark::Line),
        ("Overlapping voices", OVERLAP, Mark::Line),
    ];
    let (x, mut y) = (22.0, height - 22.0 - 18.0 * (entries.len() - 1) as f64);
    set_font(cr, 11.0, false);
    for (label, color, mark) in entries {
        rgba(cr, color, 0.95);
        if mark == Mark::Line {
            cr.set_line_width(1.6);
            cr.set_dash(&[5.0, 4.0], 0.0);
            cr.move_to(x - 6.0, y - 4.0);
            cr.line_to(x + 12.0, y - 4.0);
            cr.stroke()?;
            cr.set_dash(&[], 0.0);
        } else {
            cr.set_line_width(1.5);
            if mark == Mark::Settling {
                cr.set_dash(&[2.5, 2.5], 0.0);
            }
            cr.arc(x + 3.0, y - 4.0, 6.0, 0.0, TAU);
            cr.stroke()?;
            cr.set_dash(&[], 0.0);
            if mark == Mark::Due {
                cross(cr, x + 3.0, y - 4.0, 8.5)?;
            }
            rgba(cr, MUTED, 1.0);
            cr.arc(x + 3.0, y - 4.0, 2.5, 0.0, TAU);
            cr.fill()?;
        }
        shadowed_text(cr, label, x + 22.0, y, MUTED, 1.0)?;
        y += 18.0;
    }
    Ok(())
}

fn draw_hover(
    state: &State,
    cr: &cairo::Context,
    width: f64,
    height: f64,
) -> Result<(), cairo::Error> {
    let (Some(map), Some(hover)) = (&state.map, state.hover) else {
        return Ok(());
    };
    let (person, anchor, lines) = match hover {
        Hover::Dot(index) => {
            let dot = &map.dots[index];
            let owner = &map.people[dot.person];
            let mut lines = vec![
                (
                    format!("Sample {} of {}", dot.sample + 1, owner.samples),
                    MUTED,
                ),
                (
                    format!("Similarity to own voice  {:.2}", dot.center_similarity),
                    INK,
                ),
            ];
            if let Some(score) = dot.own_score {
                lines.push((format!("Match score, own samples  {score:.2}"), INK));
            }
            if let Some((rival, score)) = dot.rival {
                lines.push((
                    format!("Best other: {}  {score:.2}", map.people[rival].name),
                    if dot.flag == DotFlag::Stray {
                        OVERLAP
                    } else {
                        MUTED
                    },
                ));
            }
            match dot.flag {
                DotFlag::Stray => lines.push(("Sounds more like someone else".into(), OVERLAP)),
                DotFlag::Outlier => {
                    lines.push(("Far from this voice's other samples".into(), CAUTION))
                }
                DotFlag::Isolated => {
                    lines.push(("Isolated: no close samples of this voice".into(), CAUTION))
                }
                DotFlag::None => {}
            }
            if let Some((reason, verdict)) = &dot.tidy {
                let color = reason_color(reason);
                lines.push(match verdict {
                    Verdict::Remove => ("Due for removal at the next tidy".into(), color),
                    Verdict::Deferred => ("Due for removal, after worse samples go".into(), color),
                    Verdict::Settling(sessions) => (
                        format!(
                            "Settling: judged after {}",
                            plural(*sessions, "more session", "more sessions")
                        ),
                        MUTED,
                    ),
                    Verdict::Recurring => ("Kept: recurs across sessions".into(), MUTED),
                    Verdict::Kept => ("Kept: you restored it".into(), MUTED),
                });
            }
            lines.push(("Click to remove this sample".into(), MUTED));
            (dot.person, dot.position, lines)
        }
        Hover::Removed(index) => {
            let dot = &map.removed[index];
            let mut lines = vec![(
                match dot.removed.as_ref().map(|r| &r.0) {
                    Some(PruneReason::SoundsLike { rival }) => {
                        format!("Removed: sounded like {rival}")
                    }
                    Some(PruneReason::Isolated) => "Removed: isolated".into(),
                    _ => "Removed by you".into(),
                },
                CAUTION,
            )];
            if let Some(score) = dot.own_score {
                lines.push((format!("Match score, own samples  {score:.2}"), INK));
            }
            if let Some((_, at)) = dot.removed {
                let days = unix_now().saturating_sub(at) / 86_400;
                let left = (RESTORE_PERIOD / 86_400).saturating_sub(days);
                lines.push((
                    format!(
                        "Deleted for good in {}",
                        plural(left as usize, "day", "days")
                    ),
                    MUTED,
                ));
            }
            lines.push(("Click to restore".into(), MUTED));
            (dot.person, dot.position, lines)
        }
        Hover::Person(index) => {
            let person = &map.people[index];
            let mut lines = vec![
                (
                    plural(person.samples, "voice sample", "voice samples"),
                    MUTED,
                ),
                (
                    format!(
                        "Median similarity to voice  {:.2}",
                        person.median_similarity
                    ),
                    INK,
                ),
            ];
            if person.outliers > 0 {
                lines.push((plural(person.outliers, "outlier", "outliers"), CAUTION));
            }
            if person.strays > 0 {
                lines.push((
                    plural(person.strays, "stray sample", "stray samples"),
                    OVERLAP,
                ));
            }
            if person.due > 0 {
                lines.push((
                    format!(
                        "{} due for removal",
                        plural(person.due, "sample", "samples")
                    ),
                    OVERLAP,
                ));
            }
            if person.settling > 0 {
                lines.push((
                    format!("{} settling", plural(person.settling, "sample", "samples")),
                    MUTED,
                ));
            }
            for pair in map
                .close_pairs
                .iter()
                .filter(|p| p.a == index || p.b == index)
            {
                let other = if pair.a == index { pair.b } else { pair.a };
                let overlapping = pair.overlapping(map.threshold);
                lines.push((
                    format!(
                        "{} {}  {:.2}",
                        if overlapping { "Overlaps" } else { "Close to" },
                        map.people[other].name,
                        pair.score
                    ),
                    if overlapping { OVERLAP } else { CAUTION },
                ));
            }
            (index, person.center, lines)
        }
    };
    let color = state.colors[person];
    let (x, y) = state.to_screen(anchor, width, height);
    if let Hover::Dot(index) = hover
        && let Some((rival, _)) = map.dots[index].rival
        && map.dots[index].flag == DotFlag::Stray
    {
        let (rx, ry) = state.to_screen(map.people[rival].center, width, height);
        rgba(cr, OVERLAP, 0.8);
        cr.set_line_width(1.2);
        cr.move_to(x, y);
        cr.line_to(rx, ry);
        cr.stroke()?;
    }
    rgba(cr, INK, 0.95);
    cr.set_line_width(2.0);
    cr.arc(x, y, 7.0, 0.0, TAU);
    cr.stroke()?;

    let title = &map.people[person].name;
    set_font(cr, 13.0, true);
    let mut card_width = cr.text_extents(title)?.x_advance() + 22.0;
    set_font(cr, 11.5, false);
    for (line, _) in &lines {
        card_width = card_width.max(cr.text_extents(line)?.x_advance());
    }
    card_width += 28.0;
    let card_height = 36.0 + 17.0 * lines.len() as f64;
    let mut cx = x + 16.0;
    let mut cy = y + 16.0;
    if cx + card_width > width - 8.0 {
        cx = x - 16.0 - card_width;
    }
    if cy + card_height > height - 8.0 {
        cy = y - 16.0 - card_height;
    }
    rounded(cr, cx, cy, card_width, card_height, 10.0);
    cr.set_source_rgba(0.06, 0.075, 0.11, 0.94);
    cr.fill_preserve()?;
    rgba(cr, color, 0.7);
    cr.set_line_width(1.2);
    cr.stroke()?;
    rgba(cr, color, 1.0);
    cr.arc(cx + 18.0, cy + 19.0, 5.0, 0.0, TAU);
    cr.fill()?;
    set_font(cr, 13.0, true);
    rgba(cr, INK, 1.0);
    cr.move_to(cx + 30.0, cy + 24.0);
    cr.show_text(title)?;
    set_font(cr, 11.5, false);
    for (row, (line, ink)) in lines.iter().enumerate() {
        rgba(cr, *ink, 1.0);
        cr.move_to(cx + 14.0, cy + 46.0 + 17.0 * row as f64);
        cr.show_text(line)?;
    }
    cr.new_path();
    Ok(())
}
