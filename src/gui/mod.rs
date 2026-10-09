mod config;
mod queue;
mod remote;
mod timeline_layout;
mod wrap_layout;

use crate::audio::{archive, devices, record};
use crate::cli::{ProcessArgs, RecordArgs, RenderArgs};
use crate::format::jsonl;
use crate::meeting::{self, Attendees, MeetingDetails};
use crate::merge::events::{self, ProcessingEvent};
use crate::merge::process::{self, ProcessingProgress, ProcessingStage};
use crate::session::Session;
use crate::speaker::database::{self, SpeakerDatabase, canonical_name_key};
use crate::transcription::{TranscriptionProgress, backend};
use crate::types::{AudioSource, Manifest, SAMPLE_RATE, ScreenshotEntry, SessionState, Utterance};
use adw::prelude::*;
use gtk::{gdk, gio, glib};
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, ExitCode, ExitStatus, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rustix::process::{Pid, Signal};

use self::config::GuiConfig;
use self::queue::{
    JobState, Outcome as QueueOutcome, ProcessingQueue, finished_banner_text, ordinal,
};
use self::timeline_layout::{TimelineLayout, TimelineRow, timeline_rows};
use self::wrap_layout::WrapLayout;

const APP_ID: &str = "io.github.nsg.Singstone";

const CSS: &str = r#"
.speaker-avatar { border-radius: 999px; padding: 5px; color: white; }
.speaker-0 { background: #1a5fb4; }
.speaker-1 { background: #613583; }
.speaker-2 { background: #26a269; }
.speaker-unknown { color: #63460a; background: #f6d32d; }
.warning-text { color: @warning_color; }
.pill { border-radius: 999px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
.pill-ok { color: @success_fg_color; background: @success_bg_color; }
.pill-idle { color: @accent_fg_color; background: @accent_bg_color; }
.pill-busy { color: @warning_fg_color; background: @warning_bg_color; }
.pill-queued { color: @window_fg_color; background: alpha(currentColor, 0.15); }
.pill-failed { color: @error_fg_color; background: @error_bg_color; }
.pill-done { color: white; background: #1c7f91; }
.pill-archived { color: white; background: @purple_3; }
.recording-dot { color: #e01b24; }
.record-button { border-radius: 999px; padding-left: 12px; padding-right: 12px; font-weight: 600; }
.live-bar { background: alpha(#e01b24, 0.10); padding: 7px 12px; }
.pill-button { border-radius: 999px; padding: 8px 26px; font-weight: 600; }
.navigation-sidebar row { border-radius: 10px; margin: 2px 6px; }
.shot-card { padding: 8px; border-radius: 12px; background: alpha(currentColor, 0.05); }
.mono-button { font-family: monospace; font-size: 12px; }
.timeline-pane-headers { padding: 2px 6px 6px; }
.timeline-pane-header { padding: 0 6px; }
.timeline-cell { padding: 8px 2px; }
.transcript-card { padding: 10px; border: 1px solid @borders; border-radius: 12px; background: @card_bg_color; }
.timeline-marker { min-width: 54px; }
.timeline-time { padding: 2px 5px; border: 1px solid @borders; border-radius: 999px; background: @window_bg_color; font-feature-settings: "tnum"; }
.transcript-icon { min-width: 18px; min-height: 18px; padding: 0; }
.transcript-card .pill { padding-left: 3px; padding-right: 3px; }
.echo-row { opacity: 0.55; }
.speaker-shortcut { border: 1px solid alpha(@accent_color, 0.28); border-radius: 999px; padding: 1px 7px; color: @accent_color; background: alpha(@accent_bg_color, 0.12); }
.queue-panel { border-top: 1px solid @borders; background: alpha(currentColor, 0.035); padding: 8px 10px 10px; }
.queue-card { border: 1px solid @borders; border-radius: 10px; background: @card_bg_color; padding: 8px; }
.queue-progress-hint { border-radius: 10px; color: @accent_color; background: alpha(@accent_bg_color, 0.16); padding: 10px; }
.queue-compact-progress { min-height: 3px; }
.queue-entry-button { padding: 0; font-weight: normal; }
.queue-entry-button.heading { font-weight: bold; }
"#;

#[derive(Clone)]
struct SessionSummary {
    path: PathBuf,
    title: String,
    started: String,
    duration_ms: u64,
    status: &'static str,
    status_class: &'static str,
}

struct RecordingJob {
    stop: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<PathBuf, String>>>>,
    started: Instant,
    telemetry: record::RecordingTelemetry,
    mic: bool,
    system: bool,
}

#[derive(Clone, Default)]
struct PlaybackController {
    active: Rc<RefCell<Option<ActivePlayback>>>,
    next_id: Rc<Cell<u64>>,
}

struct ActivePlayback {
    row_id: u64,
    generation: u64,
    child: Child,
    button: gtk::Button,
    writer_error: Arc<Mutex<Option<String>>>,
    stderr: Arc<Mutex<String>>,
}

struct SpawnedPlayback {
    child: Child,
    writer_error: Arc<Mutex<Option<String>>>,
    stderr: Arc<Mutex<String>>,
}

#[derive(Clone)]
struct SessionDetail {
    root: gtk::Box,
    title: gtk::Label,
    subtitle: gtk::Label,
    status: gtk::Label,
    process_button: gtk::Button,
    done_button: gtk::Button,
    archive_button: gtk::Button,
    meeting_button: gtk::Button,
    rename_button: gtk::Button,
    delete_button: gtk::Button,
    hide_mic: gtk::ToggleButton,
    hide_system: gtk::ToggleButton,
    loading_hidden_sources: Rc<Cell<bool>>,
    body_stack: gtk::Stack,
    transcript_panes: gtk::Box,
    transcript: gtk::Box,
    empty_state: Rc<RefCell<Option<gtk::Label>>>,
    screenshots: gtk::FlowBox,
    metadata: gtk::Box,
    files: gtk::Box,
    selected: Rc<RefCell<Option<PathBuf>>>,
    playback: PlaybackController,
    window: adw::ApplicationWindow,
    queue: Rc<RefCell<ProcessingQueue>>,
    cancel_requested: Rc<Cell<bool>>,
    busy: Rc<Cell<bool>>,
    processing: ProcessingView,
}

#[derive(Clone)]
struct ProcessingView {
    root: gtk::Box,
    spinner: gtk::Spinner,
    label: gtk::Label,
    progress: gtk::ProgressBar,
    stages: Vec<gtk::Label>,
    metrics: gtk::Box,
    throughput: gtk::Label,
    throughput_detail: gtk::Label,
}

#[derive(Clone)]
enum QueuePanelAction {
    Select(PathBuf),
    Cancel,
    Remove(PathBuf),
    Retry(PathBuf),
    Details(String),
    Clear,
}

#[derive(Clone)]
struct QueuePanelProgress {
    spinner: gtk::Spinner,
    compact_spinner: gtk::Spinner,
    stage: gtk::Label,
    bar: gtk::ProgressBar,
    step: gtk::Label,
    percent: gtk::Label,
    cancel: gtk::Button,
    compact_bar: gtk::ProgressBar,
    compact_status: gtk::Label,
}

type QueuePanelHandler = Rc<dyn Fn(QueuePanelAction)>;

#[derive(Clone)]
struct QueuePanel {
    root: gtk::Box,
    expanded: gtk::Box,
    collapsed: gtk::Box,
    is_collapsed: Rc<Cell<bool>>,
    progress: Rc<RefCell<Option<QueuePanelProgress>>>,
    handler: Rc<RefCell<Option<QueuePanelHandler>>>,
}

impl QueuePanel {
    fn new() -> Self {
        let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
        root.add_css_class("queue-panel");
        root.set_visible(false);
        let expanded = gtk::Box::new(gtk::Orientation::Vertical, 7);
        let collapsed = gtk::Box::new(gtk::Orientation::Vertical, 4);
        collapsed.set_visible(false);
        root.append(&expanded);
        root.append(&collapsed);
        Self {
            root,
            expanded,
            collapsed,
            is_collapsed: Rc::new(Cell::new(false)),
            progress: Rc::new(RefCell::new(None)),
            handler: Rc::new(RefCell::new(None)),
        }
    }

    fn set_handler(&self, handler: QueuePanelHandler) {
        *self.handler.borrow_mut() = Some(handler);
    }

    fn emit(&self, action: QueuePanelAction) {
        if let Some(handler) = self.handler.borrow().as_ref() {
            handler(action);
        }
    }

    fn toggle(&self) {
        self.is_collapsed.set(!self.is_collapsed.get());
        self.expanded.set_visible(!self.is_collapsed.get());
        self.collapsed.set_visible(self.is_collapsed.get());
    }

    fn render(&self, queue: &ProcessingQueue, recording_held: bool, cancelled: bool) {
        clear_box(&self.expanded);
        clear_box(&self.collapsed);
        *self.progress.borrow_mut() = None;
        let visible = !queue.is_idle() || !queue.failed().is_empty();
        self.root.set_visible(visible);
        if queue.is_idle() {
            self.is_collapsed.set(false);
        }
        if !visible {
            return;
        }

        if queue.is_idle() {
            self.expanded.set_visible(true);
            self.collapsed.set_visible(false);
            let header = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let titles = gtk::Box::new(gtk::Orientation::Vertical, 1);
            titles.set_hexpand(true);
            let title = gtk::Label::new(Some("Queue finished"));
            title.set_xalign(0.0);
            title.add_css_class("heading");
            titles.append(&title);
            let summary = gtk::Label::new(Some(&format!(
                "{} processed · {} failed",
                queue.processed_count(),
                queue.failed().len()
            )));
            summary.set_xalign(0.0);
            summary.add_css_class("dim-label");
            summary.add_css_class("caption");
            titles.append(&summary);
            header.append(&titles);
            let clear = gtk::Button::with_label("Clear");
            clear.add_css_class("flat");
            let panel = self.clone();
            clear.connect_clicked(move |_| panel.emit(QueuePanelAction::Clear));
            header.append(&clear);
            self.expanded.append(&header);
        } else {
            let header = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let title = gtk::Label::new(Some("Processing queue"));
            title.set_xalign(0.0);
            title.set_hexpand(true);
            title.add_css_class("heading");
            header.append(&title);
            if let Some((position, total)) = queue.batch_position_total() {
                let count = gtk::Label::new(Some(&format!("{position} of {total}")));
                count.add_css_class("dim-label");
                count.add_css_class("caption");
                header.append(&count);
            }
            let collapse = gtk::Button::from_icon_name("pan-down-symbolic");
            collapse.add_css_class("flat");
            collapse.set_tooltip_text(Some("Collapse processing queue"));
            let panel = self.clone();
            collapse.connect_clicked(move |_| panel.toggle());
            header.append(&collapse);
            self.expanded.append(&header);
        }

        let mut dynamic = None;
        if let Some(path) = queue.running() {
            let (title, _) = session_panel_info(path);
            let card = gtk::Box::new(gtk::Orientation::Vertical, 5);
            card.add_css_class("queue-card");
            let top = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let spinner = gtk::Spinner::new();
            spinner.set_spinning(true);
            top.append(&spinner);
            let select = queue_entry_button(&title, true);
            select.add_css_class("flat");
            select.add_css_class("queue-entry-button");
            select.add_css_class("heading");
            select.set_hexpand(true);
            select.set_halign(gtk::Align::Fill);
            let panel = self.clone();
            let select_path = path.to_owned();
            select.connect_clicked(move |_| {
                panel.emit(QueuePanelAction::Select(select_path.clone()))
            });
            top.append(&select);
            let cancel = gtk::Button::from_icon_name("window-close-symbolic");
            cancel.add_css_class("flat");
            cancel.set_tooltip_text(Some("Cancel processing"));
            cancel.set_sensitive(!cancelled);
            let panel = self.clone();
            cancel.connect_clicked(move |_| panel.emit(QueuePanelAction::Cancel));
            top.append(&cancel);
            card.append(&top);
            let stage = gtk::Label::new(Some(if cancelled {
                "Cancelling…"
            } else {
                ProcessingStage::Preparing.label()
            }));
            stage.set_xalign(0.0);
            stage.set_ellipsize(gtk::pango::EllipsizeMode::End);
            card.append(&stage);
            let bar = gtk::ProgressBar::new();
            bar.set_pulse_step(0.04);
            card.append(&bar);
            let caption = gtk::Box::new(gtk::Orientation::Horizontal, 4);
            let step = gtk::Label::new(Some("Step 1 of 7"));
            step.set_xalign(0.0);
            step.set_hexpand(true);
            step.add_css_class("dim-label");
            step.add_css_class("caption");
            caption.append(&step);
            let percent = gtk::Label::new(None);
            percent.add_css_class("dim-label");
            percent.add_css_class("caption");
            caption.append(&percent);
            card.append(&caption);
            self.expanded.append(&card);

            let compact_bar = gtk::ProgressBar::new();
            compact_bar.add_css_class("queue-compact-progress");
            compact_bar.add_css_class("osd");
            compact_bar.set_pulse_step(0.04);
            self.collapsed.append(&compact_bar);
            let compact = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let compact_spinner = gtk::Spinner::new();
            compact_spinner.set_spinning(true);
            compact.append(&compact_spinner);
            let select = queue_entry_button(&title, true);
            select.add_css_class("flat");
            select.add_css_class("queue-entry-button");
            select.add_css_class("heading");
            select.set_hexpand(true);
            let panel = self.clone();
            let select_path = path.to_owned();
            select.connect_clicked(move |_| {
                panel.emit(QueuePanelAction::Select(select_path.clone()))
            });
            compact.append(&select);
            let compact_status = gtk::Label::new(None);
            compact_status.add_css_class("dim-label");
            compact_status.add_css_class("caption");
            compact.append(&compact_status);
            let expand = gtk::Button::from_icon_name("pan-up-symbolic");
            expand.add_css_class("flat");
            expand.set_tooltip_text(Some("Expand processing queue"));
            let panel = self.clone();
            expand.connect_clicked(move |_| panel.toggle());
            compact.append(&expand);
            self.collapsed.append(&compact);
            dynamic = Some(QueuePanelProgress {
                spinner,
                compact_spinner,
                stage,
                bar,
                step,
                percent,
                cancel,
                compact_bar,
                compact_status,
            });
        } else if recording_held && !queue.waiting().is_empty() {
            let paused = gtk::Label::new(Some(
                "Paused while recording. The next session starts when the recording stops.",
            ));
            paused.set_wrap(true);
            paused.set_xalign(0.0);
            paused.add_css_class("dim-label");
            self.expanded.append(&paused);
        }

        if queue.running().is_none() && !queue.waiting().is_empty() {
            let compact = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let status = gtk::Label::new(Some(if recording_held {
                "Paused while recording"
            } else {
                "Starting next session…"
            }));
            status.set_xalign(0.0);
            status.set_hexpand(true);
            status.set_ellipsize(gtk::pango::EllipsizeMode::End);
            status.add_css_class("heading");
            compact.append(&status);
            if let Some((position, total)) = queue.batch_position_total() {
                let count = gtk::Label::new(Some(&format!("{position} of {total}")));
                count.add_css_class("dim-label");
                count.add_css_class("caption");
                compact.append(&count);
            }
            let expand = gtk::Button::from_icon_name("pan-up-symbolic");
            expand.add_css_class("flat");
            expand.set_tooltip_text(Some("Expand processing queue"));
            let panel = self.clone();
            expand.connect_clicked(move |_| panel.toggle());
            compact.append(&expand);
            self.collapsed.append(&compact);
        }

        let entries = gtk::Box::new(gtk::Orientation::Vertical, 7);
        let first_position = if queue.running().is_some() { 2 } else { 1 };
        for (index, path) in queue.waiting().iter().enumerate() {
            let (title, duration) = session_panel_info(path);
            let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let position = gtk::Label::new(Some(&(first_position + index).to_string()));
            position.add_css_class("dim-label");
            position.add_css_class("caption");
            row.append(&position);
            let select = queue_entry_button(&title, false);
            select.add_css_class("flat");
            select.add_css_class("queue-entry-button");
            select.set_hexpand(true);
            select.set_halign(gtk::Align::Fill);
            let panel = self.clone();
            let select_path = path.clone();
            select.connect_clicked(move |_| {
                panel.emit(QueuePanelAction::Select(select_path.clone()))
            });
            row.append(&select);
            let duration = gtk::Label::new(Some(&duration));
            duration.add_css_class("dim-label");
            duration.add_css_class("caption");
            row.append(&duration);
            let remove = gtk::Button::from_icon_name("window-close-symbolic");
            remove.add_css_class("flat");
            remove.set_tooltip_text(Some("Remove from queue"));
            let panel = self.clone();
            let remove_path = path.clone();
            remove.connect_clicked(move |_| {
                panel.emit(QueuePanelAction::Remove(remove_path.clone()))
            });
            row.append(&remove);
            entries.append(&row);
        }

        for failed in queue.failed() {
            let (title, _) = session_panel_info(&failed.path);
            let card = gtk::Box::new(gtk::Orientation::Vertical, 4);
            card.add_css_class("queue-card");
            let top = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            top.append(&gtk::Image::from_icon_name("dialog-warning-symbolic"));
            let select = queue_entry_button(&title, true);
            select.add_css_class("flat");
            select.add_css_class("queue-entry-button");
            select.add_css_class("heading");
            select.set_hexpand(true);
            let panel = self.clone();
            let select_path = failed.path.clone();
            select.connect_clicked(move |_| {
                panel.emit(QueuePanelAction::Select(select_path.clone()))
            });
            top.append(&select);
            let retry = gtk::Button::with_label("Retry");
            let panel = self.clone();
            let retry_path = failed.path.clone();
            retry.connect_clicked(move |_| panel.emit(QueuePanelAction::Retry(retry_path.clone())));
            top.append(&retry);
            card.append(&top);
            let bottom = gtk::Box::new(gtk::Orientation::Horizontal, 4);
            let message = gtk::Label::new(Some("Processing failed."));
            message.add_css_class("dim-label");
            message.set_hexpand(true);
            message.set_xalign(0.0);
            bottom.append(&message);
            let details = gtk::Button::with_label("Details…");
            details.add_css_class("flat");
            details.add_css_class("link");
            let panel = self.clone();
            let error = failed.error.clone();
            details.connect_clicked(move |_| panel.emit(QueuePanelAction::Details(error.clone())));
            bottom.append(&details);
            card.append(&bottom);
            entries.append(&card);
        }
        if entries.first_child().is_some() {
            // A long queue scrolls instead of growing the window.
            let scroll = gtk::ScrolledWindow::builder()
                .hscrollbar_policy(gtk::PolicyType::Never)
                .propagate_natural_height(true)
                .max_content_height(170)
                .child(&entries)
                .build();
            self.expanded.append(&scroll);
        }

        *self.progress.borrow_mut() = dynamic;
        self.expanded.set_visible(!self.is_collapsed.get());
        self.collapsed
            .set_visible(self.is_collapsed.get() && !queue.is_idle());
    }

    fn update_progress(
        &self,
        update: ProcessingProgress,
        metrics: Option<TranscriptionProgress>,
        cancelled: bool,
        paused: bool,
        position_total: Option<(usize, usize)>,
    ) {
        let Some(widgets) = self.progress.borrow().as_ref().cloned() else {
            return;
        };
        widgets.cancel.set_sensitive(!cancelled);
        widgets.spinner.set_spinning(!paused);
        widgets.compact_spinner.set_spinning(!paused);
        widgets.stage.set_label(if cancelled {
            "Cancelling…"
        } else if paused {
            "Paused while recording"
        } else {
            update.stage.label()
        });
        let step = (update.stage.index() + 1).min(7);
        let speed = metrics.map(|metrics| format!(" · {:.2}× realtime", metrics.realtime_speed()));
        widgets.step.set_label(&format!(
            "Step {step} of 7{}",
            speed.as_deref().unwrap_or("")
        ));
        let fraction = metrics
            .map(TranscriptionProgress::fraction)
            .or(update.fraction);
        if let Some(fraction) = fraction {
            widgets.bar.set_fraction(fraction);
            widgets.compact_bar.set_fraction(fraction);
            widgets
                .percent
                .set_label(&format!("{:.0}%", fraction * 100.0));
        } else if !paused {
            widgets.bar.pulse();
            widgets.compact_bar.pulse();
            widgets.percent.set_label("");
        }
        let mut status = if paused && !cancelled {
            "Paused while recording".into()
        } else {
            position_total
                .map(|(position, total)| format!("{position} of {total}"))
                .unwrap_or_default()
        };
        if !paused && let Some(fraction) = fraction {
            status.push_str(&format!(" · {:.0}%", fraction * 100.0));
        }
        widgets.compact_status.set_label(&status);
    }
}

fn session_panel_info(path: &Path) -> (String, String) {
    let Ok(session) = Session::open(path) else {
        return (session_title(path), String::new());
    };
    let title = session_display_title(&session, path);
    let duration = session
        .read_manifest()
        .map(|manifest| format_duration(session_duration_ms(&session, &manifest)))
        .unwrap_or_default();
    (title, duration)
}

fn queue_entry_button(title: &str, heading: bool) -> gtk::Button {
    let label = gtk::Label::new(Some(title));
    label.set_xalign(0.0);
    label.set_hexpand(true);
    label.set_ellipsize(gtk::pango::EllipsizeMode::End);
    if heading {
        label.add_css_class("heading");
    }
    gtk::Button::builder().child(&label).build()
}

pub fn run() -> ExitCode {
    let app = adw::Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_window);
    let _ = app.run_with_args(&["singstone"]);
    ExitCode::SUCCESS
}

fn build_window(app: &adw::Application) {
    if let Some(window) = app.active_window() {
        window.present();
        return;
    }

    install_css();

    let config = Rc::new(RefCell::new(GuiConfig::load().unwrap_or_default()));
    let style_manager = adw::StyleManager::default();
    if let Some(dark_mode) = config.borrow().dark_mode {
        set_dark_mode(&style_manager, dark_mode);
    }

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Singstone")
        .default_width(1280)
        .default_height(700)
        .build();

    let toolbar = adw::ToolbarView::new();
    let header = adw::HeaderBar::new();
    let record_content = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    record_content.append(&gtk::Image::from_icon_name("media-record-symbolic"));
    record_content.append(&gtk::Label::new(Some("Record")));
    let quick_record = gtk::Button::builder().child(&record_content).build();
    quick_record.add_css_class("destructive-action");
    quick_record.add_css_class("record-button");
    quick_record.set_tooltip_text(Some("Start recording now with the current settings"));
    header.pack_start(&quick_record);
    let new_recording = gtk::Button::from_icon_name("document-new-symbolic");
    new_recording.add_css_class("flat");
    new_recording.set_tooltip_text(Some("Set up a new recording"));
    header.pack_start(&new_recording);
    let speakers = header_action_button("system-users-symbolic", "Speakers");
    speakers.set_tooltip_text(Some("Manage learned speakers"));
    header.pack_start(&speakers);
    let settings = header_action_button("preferences-system-symbolic", "Settings");
    settings.set_tooltip_text(Some("Open application settings"));
    header.pack_start(&settings);
    let header_live = gtk::Box::new(gtk::Orientation::Horizontal, 7);
    let live_icon = gtk::Image::from_icon_name("media-record-symbolic");
    live_icon.add_css_class("recording-dot");
    header_live.append(&live_icon);
    let live_heading = gtk::Label::new(Some("Recording"));
    live_heading.add_css_class("heading");
    header_live.append(&live_heading);
    header_live.set_visible(false);
    header.pack_start(&header_live);

    let about = gtk::Button::from_icon_name("help-about-symbolic");
    about.set_tooltip_text(Some("About Singstone"));
    header.pack_end(&about);
    let theme = gtk::ToggleButton::new();
    theme.add_css_class("flat");
    theme.set_active(style_manager.is_dark());
    update_theme_button(&theme);
    header.pack_end(&theme);
    let whisper_backend = backend::current();
    let backend_badge = status_pill(
        whisper_backend.badge,
        if whisper_backend.accelerated {
            "ok"
        } else {
            "idle"
        },
    );
    backend_badge.set_tooltip_text(Some(&format!(
        "Whisper compute: {}",
        whisper_backend.description
    )));
    header.pack_end(&backend_badge);
    toolbar.add_top_bar(&header);

    let banner = adw::Banner::new("");
    banner.set_revealed(false);
    banner.set_button_label(Some("Dismiss"));
    let banner_for_click = banner.clone();
    banner.connect_button_clicked(move |_| banner_for_click.set_revealed(false));
    toolbar.add_top_bar(&banner);

    let session_paths = Rc::new(RefCell::new(Vec::<PathBuf>::new()));
    let processing_queue = Rc::new(RefCell::new(ProcessingQueue::default()));
    let cancel_requested = Rc::new(Cell::new(false));
    let foreground_busy = Rc::new(Cell::new(false));
    let session_list = gtk::ListBox::new();
    session_list.add_css_class("navigation-sidebar");
    session_list.set_selection_mode(gtk::SelectionMode::Single);
    let search = gtk::SearchEntry::builder()
        .placeholder_text("Filter sessions…")
        .margin_top(8)
        .margin_bottom(8)
        .margin_start(10)
        .margin_end(10)
        .build();
    let queue_panel = QueuePanel::new();
    let sidebar = build_sidebar(&search, &session_list, &queue_panel);

    let detail = build_session_detail(
        &window,
        &processing_queue,
        &cancel_requested,
        &foreground_busy,
    );
    let recording_page = build_recording_page(&config.borrow());
    let speakers_page = build_speakers_page(&window);
    let settings_page = build_settings_page(&config.borrow());
    let close_when_recording_stops = Rc::new(Cell::new(false));

    let content_toolbar = adw::ToolbarView::new();
    let live_bar = build_live_bar();
    live_bar.root.set_visible(false);
    content_toolbar.add_top_bar(&live_bar.root);

    let stack = adw::ViewStack::new();
    stack.add_named(&recording_page.root, Some("new"));
    stack.add_named(&detail.root, Some("session"));
    stack.add_named(&speakers_page.root, Some("speakers"));
    stack.add_named(&settings_page.root, Some("settings"));
    let speakers_for_switch = speakers_page.clone();
    let window_for_speakers = window.clone();
    let playback_for_switch = detail.playback.clone();
    stack.connect_visible_child_name_notify(move |stack| {
        let visible = stack.visible_child_name();
        if visible.as_deref() != Some("session") {
            playback_for_switch.stop();
        }
        if visible.as_deref() == Some("speakers") {
            speakers_for_switch.refresh(&window_for_speakers);
        }
    });
    stack.set_visible_child_name("session");
    content_toolbar.set_content(Some(&stack));

    for (button, page_name) in [
        (&new_recording, "new"),
        (&speakers, "speakers"),
        (&settings, "settings"),
    ] {
        let stack = stack.clone();
        let session_list = session_list.clone();
        button.connect_clicked(move |_| {
            session_list.unselect_all();
            stack.set_visible_child_name(page_name);
        });
    }

    let split = adw::NavigationSplitView::new();
    split.set_min_sidebar_width(250.0);
    split.set_max_sidebar_width(340.0);
    split.set_sidebar_width_fraction(0.31);
    split.set_sidebar(Some(&adw::NavigationPage::new(&sidebar, "Sessions")));
    split.set_content(Some(&adw::NavigationPage::new(&content_toolbar, "Session")));
    toolbar.set_content(Some(&split));
    window.set_content(Some(&toolbar));

    wire_session_browser(
        &window,
        &config,
        &search,
        &session_list,
        &session_paths,
        &detail,
        &stack,
        &processing_queue,
    );
    let recorder_connection = app.dbus_connection();
    let recorder_handlers = wire_recording(
        &window,
        &recording_page,
        &live_bar,
        &header_live,
        &quick_record,
        &banner,
        &config,
        &session_list,
        &session_paths,
        &search,
        &detail,
        &stack,
        &close_when_recording_stops,
        recorder_connection.clone(),
        &processing_queue,
    );
    let queue_controller = wire_processing(
        app,
        &window,
        &detail,
        &banner,
        &config,
        &session_list,
        &session_paths,
        &search,
        &processing_queue,
        &queue_panel,
        &stack,
        &recording_page.job,
        &foreground_busy,
    );
    wire_session_actions(
        &window,
        &detail,
        &config,
        &session_list,
        &session_paths,
        &search,
        &recording_page.job,
        &processing_queue,
        &queue_controller,
    );
    wire_settings(
        &window,
        &settings_page,
        &recording_page,
        &config,
        &session_list,
        &session_paths,
        &search,
        &processing_queue,
    );

    let window_for_about = window.clone();
    about.connect_clicked(move |_| show_about(&window_for_about));

    let changing_theme = Rc::new(Cell::new(false));
    let changing_theme_for_toggle = changing_theme.clone();
    let config_for_theme = config.clone();
    let window_for_theme = window.clone();
    let style_for_theme = style_manager.clone();
    let current_dark = Rc::new(Cell::new(style_manager.is_dark()));
    let current_dark_for_toggle = current_dark.clone();
    theme.connect_toggled(move |button| {
        if changing_theme_for_toggle.replace(true) {
            return;
        }
        let dark_mode = button.is_active();
        let mut updated = config_for_theme.borrow().clone();
        updated.dark_mode = Some(dark_mode);
        if let Err(error) = updated.save() {
            let previous = current_dark_for_toggle.get();
            button.set_active(previous);
            show_error(
                &window_for_theme,
                "Could not save appearance",
                &error.to_string(),
            );
        } else {
            set_dark_mode(&style_for_theme, dark_mode);
            current_dark_for_toggle.set(dark_mode);
            *config_for_theme.borrow_mut() = updated;
        }
        update_theme_button(button);
        changing_theme_for_toggle.set(false);
    });

    let theme_for_system = theme.clone();
    let config_for_system = config.clone();
    let changing_theme_for_system = changing_theme.clone();
    let current_dark_for_system = current_dark.clone();
    style_manager.connect_dark_notify(move |style| {
        if config_for_system.borrow().dark_mode.is_some() || changing_theme_for_system.replace(true)
        {
            return;
        }
        let dark_mode = style.is_dark();
        theme_for_system.set_active(dark_mode);
        current_dark_for_system.set(dark_mode);
        update_theme_button(&theme_for_system);
        changing_theme_for_system.set(false);
    });

    let playback_on_close = detail.playback.clone();
    let stop_on_close = recording_page.job.clone();
    let close_after_stop = close_when_recording_stops.clone();
    let quitting_confirmed = Rc::new(Cell::new(false));
    let quitting_confirmed_for_close = quitting_confirmed.clone();
    let quit_dialog_open = Rc::new(Cell::new(false));
    let quit_dialog_open_for_close = quit_dialog_open.clone();
    let controller_on_close = queue_controller.clone();
    let window_on_close = window.clone();
    window.connect_close_request(move |_| {
        let outstanding = controller_on_close.queue.borrow().outstanding_count();
        if outstanding > 0 && !quitting_confirmed_for_close.get() {
            if quit_dialog_open_for_close.get() {
                return glib::Propagation::Stop;
            }
            quit_dialog_open_for_close.set(true);
            let verb = if outstanding == 1 { "is" } else { "are" };
            let noun = if outstanding == 1 {
                "session"
            } else {
                "sessions"
            };
            let dialog = adw::AlertDialog::new(
                Some("Quit while processing?"),
                Some(&format!(
                    "{outstanding} {noun} {verb} still in the processing queue. Quitting stops the one in progress. Recorded audio is kept, and unfinished sessions can be processed again later."
                )),
            );
            dialog.add_responses(&[("keep", "Keep running"), ("quit", "Quit")]);
            dialog.set_default_response(Some("keep"));
            dialog.set_close_response("keep");
            dialog.set_response_appearance("quit", adw::ResponseAppearance::Destructive);
            let quitting_confirmed = quitting_confirmed_for_close.clone();
            let quit_dialog_open = quit_dialog_open_for_close.clone();
            let controller = controller_on_close.clone();
            let window = window_on_close.clone();
            dialog.connect_response(None, move |_, response| {
                quit_dialog_open.set(false);
                if response == "quit" {
                    quitting_confirmed.set(true);
                    controller.stop_for_quit();
                    window.close();
                }
            });
            dialog.present(Some(&window_on_close));
            return glib::Propagation::Stop;
        }
        playback_on_close.stop();
        if let Some(job) = stop_on_close.borrow().as_ref() {
            job.stop.store(true, Ordering::Release);
            close_after_stop.set(true);
            return glib::Propagation::Stop;
        }
        glib::Propagation::Proceed
    });

    if let Some(connection) = recorder_connection.as_ref()
        && let Err(error) = remote::register(connection, recorder_handlers)
    {
        eprintln!("Failed to register recorder D-Bus interface: {error}");
    }

    window.present();
}

fn set_dark_mode(style_manager: &adw::StyleManager, dark_mode: bool) {
    style_manager.set_color_scheme(if dark_mode {
        adw::ColorScheme::ForceDark
    } else {
        adw::ColorScheme::ForceLight
    });
}

fn update_theme_button(button: &gtk::ToggleButton) {
    if button.is_active() {
        button.set_icon_name("weather-clear-night-symbolic");
        button.set_tooltip_text(Some("Use light mode"));
    } else {
        button.set_icon_name("weather-clear-symbolic");
        button.set_tooltip_text(Some("Use dark mode"));
    }
}

fn install_css() {
    let provider = gtk::CssProvider::new();
    provider.load_from_data(CSS);
    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

fn build_sidebar(
    search: &gtk::SearchEntry,
    list: &gtk::ListBox,
    queue_panel: &QueuePanel,
) -> gtk::Box {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let title = gtk::Label::new(Some("Sessions"));
    title.add_css_class("title-4");
    title.set_margin_top(16);
    title.set_margin_bottom(8);
    root.append(&title);
    root.append(search);
    let scroll = gtk::ScrolledWindow::builder()
        .vexpand(true)
        .child(list)
        .build();
    root.append(&scroll);
    root.append(&queue_panel.root);
    root
}

struct RecordingPage {
    root: gtk::Box,
    mic: adw::ComboRow,
    system: adw::ComboRow,
    screenshots: adw::SwitchRow,
    name: adw::EntryRow,
    button: gtk::Button,
    hint: gtk::Label,
    mic_targets: Rc<RefCell<Vec<String>>>,
    system_targets: Rc<RefCell<Vec<String>>>,
    job: Rc<RefCell<Option<RecordingJob>>>,
}

fn build_recording_page(config: &GuiConfig) -> RecordingPage {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let scroll = gtk::ScrolledWindow::builder().vexpand(true).build();
    let clamp = adw::Clamp::builder()
        .maximum_size(580)
        .margin_top(24)
        .margin_bottom(24)
        .margin_start(12)
        .margin_end(12)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 18);
    clamp.set_child(Some(&body));
    scroll.set_child(Some(&clamp));
    root.append(&scroll);

    let audio = adw::PreferencesGroup::builder()
        .title("Audio sources")
        .description("All audio stays on this device")
        .build();
    let mic = adw::ComboRow::builder()
        .title("Microphone")
        .subtitle("Voices in the room")
        .build();
    mic.add_prefix(&gtk::Image::from_icon_name(
        "audio-input-microphone-symbolic",
    ));
    let system = adw::ComboRow::builder()
        .title("System audio")
        .subtitle("Remote participants from the selected output")
        .build();
    system.add_prefix(&gtk::Image::from_icon_name("audio-speakers-symbolic"));
    mic.set_model(Some(&gtk::StringList::new(&[
        "Default (WirePlumber)",
        "None",
    ])));
    system.set_model(Some(&gtk::StringList::new(&[
        "Default (WirePlumber)",
        "None",
    ])));
    audio.add(&mic);
    audio.add(&system);
    body.append(&audio);

    let screenshot_group = adw::PreferencesGroup::builder()
        .title("Screenshots")
        .build();
    let screenshots = adw::SwitchRow::builder()
        .title("File new screenshots")
        .subtitle(config.screenshots_dir.display().to_string())
        .active(true)
        .build();
    screenshots.add_prefix(&gtk::Image::from_icon_name("image-x-generic-symbolic"));
    screenshot_group.add(&screenshots);
    body.append(&screenshot_group);

    let identity = adw::PreferencesGroup::builder()
        .title("Speaker label")
        .description("Default name in the meeting details dialog")
        .build();
    let name = adw::EntryRow::builder()
        .title("Your name in the transcript")
        .text(&config.local_speaker)
        .build();
    identity.add(&name);
    body.append(&identity);

    let note = gtk::Label::new(Some(
        "Headphones are recommended when recording microphone and system audio together.",
    ));
    note.set_wrap(true);
    note.set_xalign(0.0);
    note.add_css_class("dim-label");
    note.add_css_class("caption");
    body.append(&note);

    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    let actions = gtk::Box::new(gtk::Orientation::Vertical, 8);
    actions.set_margin_top(14);
    actions.set_margin_bottom(18);
    let button = gtk::Button::with_label("Start recording");
    button.add_css_class("suggested-action");
    button.add_css_class("pill-button");
    button.set_halign(gtk::Align::Center);
    actions.append(&button);
    let hint = gtk::Label::new(Some(&format!(
        "Creates a new session in {}",
        config.meetings_dir.display()
    )));
    hint.add_css_class("dim-label");
    hint.add_css_class("caption");
    actions.append(&hint);
    root.append(&actions);

    let page = RecordingPage {
        root,
        mic,
        system,
        screenshots,
        name,
        button,
        hint,
        mic_targets: Rc::new(RefCell::new(vec!["default".into(), "none".into()])),
        system_targets: Rc::new(RefCell::new(vec!["default".into(), "none".into()])),
        job: Rc::new(RefCell::new(None)),
    };
    discover_devices(&page);
    page
}

fn discover_devices(page: &RecordingPage) {
    let result = Arc::new(Mutex::new(None));
    let thread_result = result.clone();
    std::thread::spawn(move || {
        let value = devices::enumerate().map_err(|error| error.to_string());
        *thread_result.lock().expect("device result mutex") = Some(value);
    });
    let mic = page.mic.clone();
    let system = page.system.clone();
    let mic_targets = page.mic_targets.clone();
    let system_targets = page.system_targets.clone();
    glib::timeout_add_local(Duration::from_millis(100), move || {
        let Some(result) = result.lock().expect("device result mutex").take() else {
            return glib::ControlFlow::Continue;
        };
        if let Ok(devices) = result {
            let mut mic_labels = vec!["Default (WirePlumber)".to_owned()];
            let mut mic_values = vec!["default".to_owned()];
            let mut system_labels = vec!["Default (WirePlumber)".to_owned()];
            let mut system_values = vec!["default".to_owned()];
            for device in devices {
                let label = device.label();
                if matches!(
                    device.media_class.as_str(),
                    "Audio/Source" | "Audio/Source/Virtual" | "Audio/Duplex"
                ) {
                    mic_labels.push(label.clone());
                    mic_values.push(device.name.clone());
                }
                if matches!(device.media_class.as_str(), "Audio/Sink" | "Audio/Duplex") {
                    system_labels.push(label);
                    system_values.push(device.name);
                }
            }
            mic_labels.push("None".into());
            mic_values.push("none".into());
            system_labels.push("None".into());
            system_values.push("none".into());
            let mic_refs: Vec<_> = mic_labels.iter().map(String::as_str).collect();
            let system_refs: Vec<_> = system_labels.iter().map(String::as_str).collect();
            mic.set_model(Some(&gtk::StringList::new(&mic_refs)));
            system.set_model(Some(&gtk::StringList::new(&system_refs)));
            *mic_targets.borrow_mut() = mic_values;
            *system_targets.borrow_mut() = system_values;
        }
        glib::ControlFlow::Break
    });
}

struct LiveBar {
    root: gtk::Box,
    time: gtk::Label,
    mic_group: gtk::Box,
    mic_level: gtk::LevelBar,
    system_group: gtk::Box,
    system_level: gtk::LevelBar,
    screenshot_count: gtk::Label,
    stop: gtk::Button,
}

fn build_live_bar() -> LiveBar {
    let root = gtk::Box::new(gtk::Orientation::Horizontal, 12);
    root.add_css_class("live-bar");
    let icon = gtk::Image::from_icon_name("media-record-symbolic");
    icon.add_css_class("recording-dot");
    root.append(&icon);
    let time = gtk::Label::new(Some("00:00"));
    time.add_css_class("title-3");
    root.append(&time);
    root.append(&gtk::Separator::new(gtk::Orientation::Vertical));
    let levels = gtk::Box::new(gtk::Orientation::Vertical, 4);
    levels.set_hexpand(true);
    levels.set_valign(gtk::Align::Center);
    let mic_group = meter_group("Mic");
    let mic_level = mic_group
        .last_child()
        .and_downcast::<gtk::LevelBar>()
        .expect("meter group level bar");
    levels.append(&mic_group);
    let system_group = meter_group("System");
    let system_level = system_group
        .last_child()
        .and_downcast::<gtk::LevelBar>()
        .expect("meter group level bar");
    levels.append(&system_group);
    root.append(&levels);
    let screenshot_count = gtk::Label::new(Some("0 screenshots"));
    screenshot_count.add_css_class("dim-label");
    screenshot_count.add_css_class("caption");
    screenshot_count.set_tooltip_text(Some("Screenshots filed"));
    root.append(&screenshot_count);
    let stop = gtk::Button::with_label("Stop");
    stop.add_css_class("destructive-action");
    stop.add_css_class("pill-button");
    root.append(&stop);
    LiveBar {
        root,
        time,
        mic_group,
        mic_level,
        system_group,
        system_level,
        screenshot_count,
        stop,
    }
}

fn meter_group(name: &str) -> gtk::Box {
    let group = gtk::Box::new(gtk::Orientation::Horizontal, 5);
    let label = gtk::Label::new(Some(name));
    label.set_size_request(52, -1);
    label.set_xalign(0.0);
    label.add_css_class("caption");
    group.append(&label);
    let meter = gtk::LevelBar::for_interval(0.0, 1.0);
    meter.set_hexpand(true);
    meter.set_value(0.0);
    group.append(&meter);
    group
}

fn build_session_detail(
    window: &adw::ApplicationWindow,
    queue: &Rc<RefCell<ProcessingQueue>>,
    cancel_requested: &Rc<Cell<bool>>,
    busy: &Rc<Cell<bool>>,
) -> SessionDetail {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let heading = gtk::Box::new(gtk::Orientation::Vertical, 4);
    heading.set_margin_top(18);
    heading.set_margin_bottom(10);
    heading.set_margin_start(20);
    heading.set_margin_end(20);
    let top = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    let title = gtk::Label::new(Some("Select a session"));
    title.set_xalign(0.0);
    title.set_hexpand(true);
    title.set_ellipsize(gtk::pango::EllipsizeMode::End);
    title.add_css_class("title-2");
    top.append(&title);
    let status = status_pill("", "idle");
    status.set_visible(false);
    top.append(&status);
    let process_button = gtk::Button::with_label("Process");
    process_button.add_css_class("suggested-action");
    process_button.set_visible(false);
    top.append(&process_button);
    let done_button = gtk::Button::with_label("Mark done");
    done_button.set_visible(false);
    top.append(&done_button);
    let archive_button = gtk::Button::with_label("Archive…");
    archive_button.set_tooltip_text(Some(
        "Convert the recorded audio to 16-bit FLAC to save disk space",
    ));
    archive_button.set_visible(false);
    top.append(&archive_button);
    let meeting_button = gtk::Button::with_label("Meeting details…");
    meeting_button.set_visible(false);
    top.append(&meeting_button);
    let rename_button = gtk::Button::from_icon_name("document-edit-symbolic");
    rename_button.set_tooltip_text(Some("Rename session"));
    rename_button.add_css_class("flat");
    rename_button.set_visible(false);
    top.append(&rename_button);
    let delete_button = gtk::Button::from_icon_name("user-trash-symbolic");
    delete_button.set_tooltip_text(Some("Delete session"));
    delete_button.add_css_class("flat");
    delete_button.set_visible(false);
    top.append(&delete_button);
    heading.append(&top);
    let subtitle = gtk::Label::new(Some("Recorded meetings appear in the sidebar."));
    subtitle.set_xalign(0.0);
    subtitle.add_css_class("dim-label");
    heading.append(&subtitle);
    root.append(&heading);
    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_vexpand(true);
    paned.set_position(650);
    paned.set_resize_start_child(true);
    paned.set_shrink_start_child(false);
    paned.set_resize_end_child(false);
    paned.set_shrink_end_child(false);
    let transcript_frame = gtk::Box::new(gtk::Orientation::Vertical, 6);
    transcript_frame.set_margin_top(12);
    let transcript_header = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    transcript_header.set_margin_start(16);
    transcript_header.set_margin_end(16);
    let transcript_heading = gtk::Label::new(Some("Transcript"));
    transcript_heading.add_css_class("title-4");
    transcript_heading.set_xalign(0.0);
    transcript_heading.set_hexpand(true);
    transcript_header.append(&transcript_heading);
    let transcript_toggles = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    transcript_toggles.set_layout_manager(Some(WrapLayout::new(8, 4)));
    transcript_header.append(&transcript_toggles);
    let hide_mic = gtk::ToggleButton::with_label("Hide microphone");
    hide_mic.set_tooltip_text(Some(
        "Leave microphone lines out of transcript.jsonl and transcript.txt. Nothing is deleted.",
    ));
    hide_mic.set_visible(false);
    transcript_toggles.append(&hide_mic);
    let hide_system = gtk::ToggleButton::with_label("Hide system");
    hide_system.set_tooltip_text(Some(
        "Leave system audio lines out of transcript.jsonl and transcript.txt. Nothing is deleted.",
    ));
    hide_system.set_visible(false);
    transcript_toggles.append(&hide_system);
    transcript_frame.append(&transcript_header);
    let transcript_panes = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    transcript_panes.add_css_class("timeline-pane-headers");
    transcript_panes.set_visible(false);
    transcript_frame.append(&transcript_panes);
    let transcript = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let transcript_scroll = gtk::ScrolledWindow::builder()
        .vexpand(true)
        .child(&transcript)
        .build();
    transcript_frame.append(&transcript_scroll);
    let processing = processing_view(&backend::current());
    let processing_scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .child(&processing.root)
        .build();
    let body_stack = gtk::Stack::new();
    body_stack.set_vhomogeneous(false);
    body_stack.add_named(&transcript_frame, Some("transcript"));
    body_stack.add_named(&processing_scroll, Some("processing"));
    body_stack.set_visible_child_name("transcript");
    paned.set_start_child(Some(&body_stack));
    let side = gtk::Box::new(gtk::Orientation::Vertical, 14);
    side.set_margin_top(12);
    side.set_margin_bottom(12);
    side.set_margin_start(12);
    side.set_margin_end(12);
    let shots_heading = gtk::Label::new(Some("Screenshots"));
    shots_heading.add_css_class("title-4");
    shots_heading.set_xalign(0.0);
    side.append(&shots_heading);
    let screenshots = gtk::FlowBox::new();
    screenshots.set_selection_mode(gtk::SelectionMode::None);
    screenshots.set_max_children_per_line(2);
    screenshots.set_row_spacing(8);
    screenshots.set_column_spacing(8);
    side.append(&screenshots);
    let session_heading = gtk::Label::new(Some("Session"));
    session_heading.add_css_class("title-4");
    session_heading.set_xalign(0.0);
    side.append(&session_heading);
    let metadata = gtk::Box::new(gtk::Orientation::Vertical, 6);
    side.append(&metadata);
    paned.set_end_child(Some(
        &gtk::ScrolledWindow::builder()
            .min_content_width(280)
            .child(&side)
            .build(),
    ));
    root.append(&paned);

    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    let files = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    files.set_margin_top(8);
    files.set_margin_bottom(10);
    files.set_margin_start(16);
    files.set_margin_end(16);
    root.append(&files);

    let detail = SessionDetail {
        root,
        title,
        subtitle,
        status,
        process_button,
        done_button,
        archive_button,
        meeting_button,
        rename_button,
        delete_button,
        hide_mic,
        hide_system,
        loading_hidden_sources: Rc::new(Cell::new(false)),
        body_stack,
        transcript_panes,
        transcript,
        empty_state: Rc::new(RefCell::new(None)),
        screenshots,
        metadata,
        files,
        selected: Rc::new(RefCell::new(None)),
        playback: PlaybackController::default(),
        window: window.clone(),
        queue: queue.clone(),
        cancel_requested: cancel_requested.clone(),
        busy: busy.clone(),
        processing,
    };
    for button in [&detail.hide_mic, &detail.hide_system] {
        let detail_for_toggle = detail.clone();
        button.connect_toggled(move |_| {
            if !detail_for_toggle.loading_hidden_sources.get() {
                start_hidden_sources_update(&detail_for_toggle);
            }
        });
    }
    detail
}

#[derive(Clone)]
struct SpeakersPage {
    root: gtk::Box,
    group: adw::PreferencesGroup,
    rows: Rc<RefCell<Vec<adw::ActionRow>>>,
    database_path: PathBuf,
}

fn build_speakers_page(window: &adw::ApplicationWindow) -> SpeakersPage {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let scroll = gtk::ScrolledWindow::builder().vexpand(true).build();
    let clamp = adw::Clamp::builder()
        .maximum_size(580)
        .margin_top(24)
        .margin_bottom(24)
        .margin_start(12)
        .margin_end(12)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 16);
    let group = adw::PreferencesGroup::builder()
        .title("Learned voices")
        .description("Voices are matched locally; uncertain matches stay anonymous")
        .build();
    let db_path = speakers_database_path();
    body.append(&group);
    let path = gtk::Label::new(Some(&format!("Database: {}", db_path.display())));
    path.set_xalign(0.0);
    path.set_wrap(true);
    path.add_css_class("dim-label");
    path.add_css_class("caption");
    body.append(&path);
    clamp.set_child(Some(&body));
    scroll.set_child(Some(&clamp));
    root.append(&scroll);
    let page = SpeakersPage {
        root,
        group,
        rows: Rc::new(RefCell::new(Vec::new())),
        database_path: db_path,
    };
    page.refresh(window);
    page
}

impl SpeakersPage {
    fn refresh(&self, window: &adw::ApplicationWindow) {
        for row in self.rows.borrow_mut().drain(..) {
            self.group.remove(&row);
        }
        let mut rows = Vec::new();
        match SpeakerDatabase::load(&self.database_path) {
            Ok(database) if !database.speakers.is_empty() => {
                for (name, speaker) in database.speakers {
                    let row = adw::ActionRow::builder()
                        .title(&name)
                        .subtitle(format!(
                            "{} voice sample{}",
                            speaker.embeddings.len(),
                            if speaker.embeddings.len() == 1 {
                                ""
                            } else {
                                "s"
                            }
                        ))
                        .build();
                    row.add_prefix(&gtk::Image::from_icon_name("avatar-default-symbolic"));
                    let remove = gtk::Button::from_icon_name("user-trash-symbolic");
                    remove.set_tooltip_text(Some("Remove learned voice"));
                    remove.add_css_class("flat");
                    remove.set_valign(gtk::Align::Center);
                    row.add_suffix(&remove);
                    let page = self.clone();
                    let parent = window.clone();
                    remove.connect_clicked(move |_| {
                        confirm_delete_speaker(&parent, &page, &name);
                    });
                    self.group.add(&row);
                    rows.push(row);
                }
            }
            Ok(_) | Err(_) => {
                let row = adw::ActionRow::builder()
                    .title("No learned voices yet")
                    .subtitle("Name someone in a transcript to teach Singstone their voice")
                    .build();
                row.add_prefix(&gtk::Image::from_icon_name("avatar-default-symbolic"));
                self.group.add(&row);
                rows.push(row);
            }
        }
        *self.rows.borrow_mut() = rows;
    }
}

fn confirm_delete_speaker(parent: &adw::ApplicationWindow, page: &SpeakersPage, name: &str) {
    let dialog = adw::AlertDialog::new(
        Some("Remove learned voice?"),
        Some(&format!(
            "Future sessions will no longer recognize {name}. Existing transcripts are unchanged."
        )),
    );
    dialog.add_responses(&[("cancel", "Cancel"), ("remove", "Remove")]);
    dialog.set_default_response(Some("cancel"));
    dialog.set_close_response("cancel");
    dialog.set_response_appearance("remove", adw::ResponseAppearance::Destructive);
    let page = page.clone();
    let parent_for_response = parent.clone();
    let name = name.to_owned();
    dialog.connect_response(Some("remove"), move |_, _| {
        let result = SpeakerDatabase::load(&page.database_path).and_then(|mut database| {
            database.speakers.remove(&name);
            database.save(&page.database_path)
        });
        match result {
            Ok(()) => page.refresh(&parent_for_response),
            Err(error) => show_error(
                &parent_for_response,
                "Could not remove speaker",
                &error.to_string(),
            ),
        }
    });
    dialog.present(Some(parent));
}

struct SettingsPage {
    root: gtk::Box,
    meetings: adw::ActionRow,
    screenshots: adw::ActionRow,
    context: adw::ActionRow,
    swedish_transcription: gtk::Switch,
    meetings_change: gtk::Button,
    screenshots_change: gtk::Button,
    context_change: gtk::Button,
    context_clear: gtk::Button,
}

fn build_settings_page(config: &GuiConfig) -> SettingsPage {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    let scroll = gtk::ScrolledWindow::builder().vexpand(true).build();
    let clamp = adw::Clamp::builder()
        .maximum_size(600)
        .margin_top(24)
        .margin_bottom(24)
        .margin_start(12)
        .margin_end(12)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 18);
    let storage = adw::PreferencesGroup::builder().title("Storage").build();
    let meetings = adw::ActionRow::builder()
        .title("Meetings folder")
        .subtitle(config.meetings_dir.display().to_string())
        .build();
    meetings.add_prefix(&gtk::Image::from_icon_name("folder-symbolic"));
    let meetings_change = gtk::Button::with_label("Change…");
    meetings_change.set_valign(gtk::Align::Center);
    meetings.add_suffix(&meetings_change);
    storage.add(&meetings);
    let shots = adw::ActionRow::builder()
        .title("Screenshot watch folder")
        .subtitle(config.screenshots_dir.display().to_string())
        .build();
    shots.add_prefix(&gtk::Image::from_icon_name("image-x-generic-symbolic"));
    let screenshots_change = gtk::Button::with_label("Change…");
    screenshots_change.set_valign(gtk::Align::Center);
    shots.add_suffix(&screenshots_change);
    storage.add(&shots);
    let context = adw::ActionRow::builder()
        .title("Meeting context file")
        .subtitle(
            config
                .context_file
                .as_ref()
                .map_or_else(|| "Not set".into(), |path| path.display().to_string()),
        )
        .build();
    context.add_prefix(&gtk::Image::from_icon_name("text-x-generic-symbolic"));
    let context_clear = gtk::Button::from_icon_name("edit-clear-symbolic");
    context_clear.set_tooltip_text(Some("Clear meeting context file"));
    context_clear.set_valign(gtk::Align::Center);
    context_clear.set_sensitive(config.context_file.is_some());
    context.add_suffix(&context_clear);
    let context_change = gtk::Button::with_label("Change…");
    context_change.set_valign(gtk::Align::Center);
    context.add_suffix(&context_change);
    storage.add(&context);
    body.append(&storage);

    let compute_group = adw::PreferencesGroup::builder()
        .title("Transcription compute")
        .build();
    let backend = backend::current();
    let compute = adw::ActionRow::builder()
        .title("Whisper backend")
        .subtitle(&backend.description)
        .build();
    compute.add_prefix(&gtk::Image::from_icon_name("computer-symbolic"));
    compute.add_suffix(&status_pill(
        if backend.accelerated { "GPU" } else { "CPU" },
        if backend.accelerated { "ok" } else { "idle" },
    ));
    compute_group.add(&compute);
    let swedish_row = adw::ActionRow::builder()
        .title("Swedish transcription")
        .subtitle("Use KB-Whisper Small; turn off for multilingual Whisper Small")
        .activatable(true)
        .build();
    let swedish_transcription = gtk::Switch::builder()
        .active(config.swedish_transcription)
        .valign(gtk::Align::Center)
        .build();
    swedish_row.add_suffix(&swedish_transcription);
    swedish_row.set_activatable_widget(Some(&swedish_transcription));
    compute_group.add(&swedish_row);
    body.append(&compute_group);

    let model_group = adw::PreferencesGroup::builder()
        .title("Local models")
        .description("Downloaded once, verified by SHA-256, and cached per user")
        .build();
    for (title, env) in [
        (
            "KB-Whisper Small — Swedish",
            "SINGSTONE_WHISPER_MODEL_SWEDISH",
        ),
        (
            "Whisper Small — multilingual",
            "SINGSTONE_WHISPER_MODEL_MULTILINGUAL",
        ),
        (
            "pyannote segmentation — diarization",
            "SINGSTONE_SEGMENTATION_MODEL",
        ),
        ("TitaNet — speaker embeddings", "SINGSTONE_EMBEDDING_MODEL"),
    ] {
        let path = std::env::var_os(env).map(PathBuf::from);
        let ready = path.as_ref().is_some_and(|path| path.is_file());
        let row = adw::ActionRow::builder()
            .title(title)
            .subtitle(path.as_ref().map_or_else(
                || "Not configured".into(),
                |path| path.display().to_string(),
            ))
            .build();
        row.add_suffix(&status_pill(
            if ready { "Ready" } else { "Needed" },
            if ready { "ok" } else { "idle" },
        ));
        model_group.add(&row);
    }
    body.append(&model_group);

    let privacy = adw::PreferencesGroup::builder().title("Privacy").build();
    let privacy_row = adw::ActionRow::builder()
        .title("Recording and processing never use the network")
        .subtitle("Only the one-time model download service is allowed online")
        .build();
    privacy_row.add_prefix(&gtk::Image::from_icon_name("network-offline-symbolic"));
    privacy.add(&privacy_row);
    body.append(&privacy);
    clamp.set_child(Some(&body));
    scroll.set_child(Some(&clamp));
    root.append(&scroll);
    SettingsPage {
        root,
        meetings,
        screenshots: shots,
        context,
        swedish_transcription,
        meetings_change,
        screenshots_change,
        context_change,
        context_clear,
    }
}

#[allow(clippy::too_many_arguments)]
fn wire_settings(
    window: &adw::ApplicationWindow,
    page: &SettingsPage,
    recording: &RecordingPage,
    config: &Rc<RefCell<GuiConfig>>,
    sessions: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
    queue: &Rc<RefCell<ProcessingQueue>>,
) {
    let parent = window.clone();
    let config_for_meetings = config.clone();
    let row_for_meetings = page.meetings.clone();
    let hint_for_meetings = recording.hint.clone();
    let sessions_for_meetings = sessions.clone();
    let paths_for_meetings = paths.clone();
    let search_for_meetings = search.clone();
    let queue_for_meetings = queue.clone();
    page.meetings_change.connect_clicked(move |_| {
        let chooser = gtk::FileDialog::builder()
            .title("Choose meetings folder")
            .accept_label("Use folder")
            .modal(true)
            .build();
        let parent = parent.clone();
        let config = config_for_meetings.clone();
        let row = row_for_meetings.clone();
        let hint = hint_for_meetings.clone();
        let sessions = sessions_for_meetings.clone();
        let paths = paths_for_meetings.clone();
        let search = search_for_meetings.clone();
        let queue = queue_for_meetings.clone();
        let parent_for_result = parent.clone();
        chooser.select_folder(Some(&parent), None::<&gio::Cancellable>, move |result| {
            let Ok(folder) = result else { return };
            let Some(path) = folder.path() else { return };
            let updated = {
                let mut updated = config.borrow().clone();
                updated.meetings_dir = path.clone();
                updated
            };
            if let Err(error) = updated.save() {
                show_error(
                    &parent_for_result,
                    "Could not save settings",
                    &error.to_string(),
                );
                return;
            }
            *config.borrow_mut() = updated;
            row.set_subtitle(&path.display().to_string());
            hint.set_label(&format!("Creates a new session in {}", path.display()));
            populate_sessions(&sessions, &paths, &path, &search.text(), &queue);
            if let Some(first) = sessions.row_at_index(0) {
                sessions.select_row(Some(&first));
            }
        });
    });

    let parent = window.clone();
    let config_for_shots = config.clone();
    let row_for_shots = page.screenshots.clone();
    let recording_shots = recording.screenshots.clone();
    page.screenshots_change.connect_clicked(move |_| {
        let chooser = gtk::FileDialog::builder()
            .title("Choose screenshot watch folder")
            .accept_label("Watch folder")
            .modal(true)
            .build();
        let parent = parent.clone();
        let config = config_for_shots.clone();
        let row = row_for_shots.clone();
        let recording_shots = recording_shots.clone();
        let parent_for_result = parent.clone();
        chooser.select_folder(Some(&parent), None::<&gio::Cancellable>, move |result| {
            let Ok(folder) = result else { return };
            let Some(path) = folder.path() else { return };
            let updated = {
                let mut updated = config.borrow().clone();
                updated.screenshots_dir = path.clone();
                updated
            };
            if let Err(error) = updated.save() {
                show_error(
                    &parent_for_result,
                    "Could not save settings",
                    &error.to_string(),
                );
                return;
            }
            *config.borrow_mut() = updated;
            row.set_subtitle(&path.display().to_string());
            recording_shots.set_subtitle(&path.display().to_string());
        });
    });

    let parent = window.clone();
    let config_for_context = config.clone();
    let row_for_context = page.context.clone();
    let clear_for_context = page.context_clear.clone();
    page.context_change.connect_clicked(move |_| {
        let json_filter = gtk::FileFilter::new();
        json_filter.set_name(Some("JSON files"));
        json_filter.add_suffix("json");
        let filters = gio::ListStore::new::<gtk::FileFilter>();
        filters.append(&json_filter);
        let chooser = gtk::FileDialog::builder()
            .title("Choose meeting context file")
            .accept_label("Use file")
            .modal(true)
            .filters(&filters)
            .default_filter(&json_filter)
            .build();
        if let Some(current) = config_for_context.borrow().context_file.as_deref() {
            chooser.set_initial_file(Some(&gio::File::for_path(current)));
        }
        let parent_for_result = parent.clone();
        let config = config_for_context.clone();
        let row = row_for_context.clone();
        let clear = clear_for_context.clone();
        chooser.open(Some(&parent), None::<&gio::Cancellable>, move |result| {
            let Ok(file) = result else { return };
            let Some(path) = file.path() else { return };
            let mut updated = config.borrow().clone();
            updated.context_file = Some(path.clone());
            if let Err(error) = updated.save() {
                show_error(
                    &parent_for_result,
                    "Could not save settings",
                    &error.to_string(),
                );
                return;
            }
            *config.borrow_mut() = updated;
            row.set_subtitle(&path.display().to_string());
            clear.set_sensitive(true);
        });
    });

    let parent = window.clone();
    let config_for_context = config.clone();
    let row_for_context = page.context.clone();
    let clear_for_context = page.context_clear.clone();
    page.context_clear.connect_clicked(move |_| {
        let mut updated = config_for_context.borrow().clone();
        updated.context_file = None;
        if let Err(error) = updated.save() {
            show_error(&parent, "Could not save settings", &error.to_string());
            return;
        }
        *config_for_context.borrow_mut() = updated;
        row_for_context.set_subtitle("Not set");
        clear_for_context.set_sensitive(false);
    });

    let parent = window.clone();
    let config_for_language = config.clone();
    let changing_language = Rc::new(Cell::new(false));
    let changing_language_for_notify = changing_language.clone();
    page.swedish_transcription
        .connect_active_notify(move |toggle| {
            if changing_language_for_notify.get() {
                return;
            }
            let mut updated = config_for_language.borrow().clone();
            updated.swedish_transcription = toggle.is_active();
            if let Err(error) = updated.save() {
                changing_language_for_notify.set(true);
                toggle.set_active(config_for_language.borrow().swedish_transcription);
                changing_language_for_notify.set(false);
                show_error(&parent, "Could not save settings", &error.to_string());
            } else {
                *config_for_language.borrow_mut() = updated;
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn wire_session_browser(
    window: &adw::ApplicationWindow,
    config: &Rc<RefCell<GuiConfig>>,
    search: &gtk::SearchEntry,
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    detail: &SessionDetail,
    stack: &adw::ViewStack,
    queue: &Rc<RefCell<ProcessingQueue>>,
) {
    populate_sessions(list, paths, &config.borrow().meetings_dir, "", queue);
    let list_for_search = list.clone();
    let paths_for_search = paths.clone();
    let config_for_search = config.clone();
    let queue_for_search = queue.clone();
    search.connect_search_changed(move |entry| {
        populate_sessions(
            &list_for_search,
            &paths_for_search,
            &config_for_search.borrow().meetings_dir,
            &entry.text(),
            &queue_for_search,
        );
    });

    let detail_for_select = detail.clone();
    let paths_for_select = paths.clone();
    let stack_for_select = stack.clone();
    let window_for_select = window.clone();
    list.connect_row_selected(move |_, row| {
        let Some(row) = row else { return };
        let Some(path) = paths_for_select.borrow().get(row.index() as usize).cloned() else {
            return;
        };
        if let Err(error) = detail_for_select.load(&path) {
            show_error(
                &window_for_select,
                "Could not open session",
                &error.to_string(),
            );
        } else {
            stack_for_select.set_visible_child_name("session");
        }
    });
    if let Some(first) = list.row_at_index(0) {
        list.select_row(Some(&first));
    }
}

fn select_session_row(list: &gtk::ListBox, index: usize) {
    if let Some(row) = i32::try_from(index)
        .ok()
        .and_then(|index| list.row_at_index(index))
    {
        list.select_row(Some(&row));
    }
}

fn session_row_labels(row: &gtk::ListBoxRow) -> Option<(gtk::Label, gtk::Label)> {
    let body = row.child()?.downcast::<gtk::Box>().ok()?;
    let top = body.first_child()?.downcast::<gtk::Box>().ok()?;
    let title = top.first_child()?.downcast::<gtk::Label>().ok()?;
    let status = top.last_child()?.downcast::<gtk::Label>().ok()?;
    Some((title, status))
}

fn update_session_row(
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    path: &Path,
    queue: &ProcessingQueue,
) {
    let Some(index) = paths
        .borrow()
        .iter()
        .position(|candidate| candidate == path)
    else {
        return;
    };
    let Some(row) = i32::try_from(index)
        .ok()
        .and_then(|index| list.row_at_index(index))
    else {
        return;
    };
    let Some((title, status)) = session_row_labels(&row) else {
        return;
    };
    let (text, class) = session_path_status(path, queue);
    set_status(&status, text, class);
    if let Ok(session) = Session::open(path)
        && session.read_manifest().is_ok()
    {
        title.set_label(&session_display_title(&session, path));
    }
}

#[allow(clippy::too_many_arguments)]
fn wire_session_actions(
    window: &adw::ApplicationWindow,
    detail: &SessionDetail,
    config: &Rc<RefCell<GuiConfig>>,
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
    job: &Rc<RefCell<Option<RecordingJob>>>,
    queue: &Rc<RefCell<ProcessingQueue>>,
    controller: &Rc<QueueController>,
) {
    let parent = window.clone();
    let detail_for_done = detail.clone();
    let list_for_done = list.clone();
    let paths_for_done = paths.clone();
    let queue_for_done = queue.clone();
    detail.done_button.connect_clicked(move |_| {
        let Some(session_path) = detail_for_done.selected.borrow().clone() else {
            return;
        };
        if let Err(error) =
            Session::open(&session_path).and_then(|session| session.set_done(!session.is_done()))
        {
            show_error(&parent, "Could not update session", &error.to_string());
        }
        update_session_row(
            &list_for_done,
            &paths_for_done,
            &session_path,
            &queue_for_done.borrow(),
        );
        detail_for_done.refresh_queue_state();
    });
    let parent = window.clone();
    let detail_for_rename = detail.clone();
    let config_for_rename = config.clone();
    let list_for_rename = list.clone();
    let paths_for_rename = paths.clone();
    let search_for_rename = search.clone();
    let queue_for_rename = queue.clone();
    let controller_for_rename = controller.clone();
    detail.rename_button.connect_clicked(move |_| {
        let Some(session_path) = detail_for_rename.selected.borrow().clone() else {
            return;
        };
        let stored = match Session::open(&session_path)
            .and_then(|session| meeting::read_details(&session.meeting_path()))
        {
            Ok(details) => details.map(|details| details.title).unwrap_or_default(),
            Err(error) => {
                show_error(&parent, "Could not read session", &error.to_string());
                return;
            }
        };
        let dialog = adw::AlertDialog::new(
            Some("Rename session"),
            Some("Leave the name empty to use the default name."),
        );
        let entry = gtk::Entry::builder()
            .text(stored)
            .placeholder_text(session_title(&session_path))
            .activates_default(true)
            .build();
        dialog.set_extra_child(Some(&entry));
        dialog.add_responses(&[("cancel", "Cancel"), ("rename", "Rename")]);
        dialog.set_default_response(Some("rename"));
        dialog.set_close_response("cancel");
        dialog.set_response_appearance("rename", adw::ResponseAppearance::Suggested);
        dialog.set_focus(Some(&entry));
        let parent_for_response = parent.clone();
        let detail = detail_for_rename.clone();
        let config = config_for_rename.clone();
        let list = list_for_rename.clone();
        let paths = paths_for_rename.clone();
        let search = search_for_rename.clone();
        let queue = queue_for_rename.clone();
        let controller = controller_for_rename.clone();
        dialog.connect_response(Some("rename"), move |_, _| {
            if let Err(error) = Session::open(&session_path)
                .and_then(|session| meeting::set_title(&session.meeting_path(), &entry.text()))
            {
                show_error(
                    &parent_for_response,
                    "Could not rename session",
                    &error.to_string(),
                );
                return;
            }
            populate_sessions(
                &list,
                &paths,
                &config.borrow().meetings_dir,
                &search.text(),
                &queue,
            );
            let index = paths.borrow().iter().position(|path| *path == session_path);
            match index {
                Some(index) => select_session_row(&list, index),
                // The new name no longer matches the search filter.
                None => {
                    let _ = detail.load(&session_path);
                }
            }
            controller.render_panel_if_visible();
        });
        dialog.present(Some(&parent));
    });

    let parent = window.clone();
    let detail_for_delete = detail.clone();
    let config_for_delete = config.clone();
    let list_for_delete = list.clone();
    let paths_for_delete = paths.clone();
    let search_for_delete = search.clone();
    let job = job.clone();
    let queue_for_delete = queue.clone();
    detail.delete_button.connect_clicked(move |_| {
        let Some(session_path) = detail_for_delete.selected.borrow().clone() else {
            return;
        };
        let recording = job.borrow().is_some()
            && Session::open(&session_path)
                .and_then(|session| session.read_manifest())
                .is_ok_and(|manifest| manifest.state == SessionState::Recording);
        if recording {
            show_error(
                &parent,
                "Recording in progress",
                "Stop the recording before deleting its session.",
            );
            return;
        }
        let dialog = adw::AlertDialog::new(
            Some("Delete this session?"),
            Some(&format!(
                "“{}” will be permanently deleted, including its recorded audio, transcript, and screenshots. This cannot be undone.",
                detail_for_delete.title.label()
            )),
        );
        dialog.add_responses(&[("cancel", "Cancel"), ("delete", "Delete")]);
        dialog.set_default_response(Some("cancel"));
        dialog.set_close_response("cancel");
        dialog.set_response_appearance("delete", adw::ResponseAppearance::Destructive);
        let parent_for_response = parent.clone();
        let detail = detail_for_delete.clone();
        let config = config_for_delete.clone();
        let list = list_for_delete.clone();
        let paths = paths_for_delete.clone();
        let search = search_for_delete.clone();
        let queue = queue_for_delete.clone();
        dialog.connect_response(Some("delete"), move |_, _| {
            detail.playback.stop();
            let index = paths.borrow().iter().position(|path| *path == session_path);
            if let Err(error) = Session::open(&session_path).and_then(Session::delete) {
                show_error(
                    &parent_for_response,
                    "Could not delete session",
                    &error.to_string(),
                );
                return;
            }
            populate_sessions(
                &list,
                &paths,
                &config.borrow().meetings_dir,
                &search.text(),
                &queue,
            );
            let remaining = paths.borrow().len();
            if remaining == 0 {
                detail.clear();
            } else {
                select_session_row(&list, index.unwrap_or(0).min(remaining - 1));
            }
        });
        dialog.present(Some(&parent));
    });
}

#[allow(clippy::too_many_arguments)]
fn wire_recording(
    window: &adw::ApplicationWindow,
    page: &RecordingPage,
    live: &LiveBar,
    header_live: &gtk::Box,
    quick_record: &gtk::Button,
    banner: &adw::Banner,
    config: &Rc<RefCell<GuiConfig>>,
    session_list: &gtk::ListBox,
    session_paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
    detail: &SessionDetail,
    stack: &adw::ViewStack,
    close_when_stopped: &Rc<Cell<bool>>,
    recorder_connection: Option<gio::DBusConnection>,
    queue: &Rc<RefCell<ProcessingQueue>>,
) -> remote::RecorderHandlers {
    let job_for_stop = page.job.clone();
    let button_for_stop = page.button.clone();
    let stop_recording: Rc<dyn Fn()> = Rc::new(move || {
        if let Some(job) = job_for_stop.borrow().as_ref() {
            job.stop.store(true, Ordering::Release);
            button_for_stop.set_label("Stopping…");
        }
    });

    let stop_recording_for_live = stop_recording.clone();
    live.stop.connect_clicked(move |_| {
        stop_recording_for_live();
    });

    let window_for_start = window.clone();
    let config_for_start = config.clone();
    let mic = page.mic.clone();
    let system = page.system.clone();
    let screenshots = page.screenshots.clone();
    let name = page.name.clone();
    let button = page.button.clone();
    let hint = page.hint.clone();
    let mic_targets = page.mic_targets.clone();
    let system_targets = page.system_targets.clone();
    let job = page.job.clone();
    let live_root = live.root.clone();
    let live_mic_group = live.mic_group.clone();
    let live_system_group = live.system_group.clone();
    let header_live_for_start = header_live.clone();
    let quick_record_for_start = quick_record.clone();
    let connection_for_start = recorder_connection.clone();
    let start_recording: Rc<dyn Fn() -> bool> = Rc::new(move || {
        let mic_target = selected_target(&mic, &mic_targets);
        let system_target = selected_target(&system, &system_targets);
        if mic_target == "none" && system_target == "none" {
            show_error(
                &window_for_start,
                "No audio source selected",
                "Choose a microphone, system audio, or both.",
            );
            return false;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let telemetry = record::RecordingTelemetry::default();
        let result = Arc::new(Mutex::new(None));
        let settings = config_for_start.borrow().clone();
        if name.text().trim().is_empty() {
            show_error(
                &window_for_start,
                "Your name is required",
                "Enter the name to use for microphone speech in the transcript.",
            );
            return false;
        }
        {
            let mut config = config_for_start.borrow_mut();
            config.local_speaker = name.text().trim().to_owned();
            if let Err(error) = config.save() {
                show_error(
                    &window_for_start,
                    "Could not save settings",
                    &error.to_string(),
                );
                return false;
            }
        }
        let args = RecordArgs {
            mic: mic_target.clone(),
            system: system_target.clone(),
            screenshots: screenshots.is_active().then_some(settings.screenshots_dir),
            screenshot_ext: vec!["png".into(), "jpg".into(), "jpeg".into(), "webp".into()],
            local_speaker: name.text().to_string(),
            output_dir: settings.meetings_dir,
            duration: None,
        };
        let thread_stop = stop.clone();
        let thread_telemetry = telemetry.clone();
        let thread_result = result.clone();
        std::thread::spawn(move || {
            let value = record::run_with_telemetry(args, thread_stop, thread_telemetry)
                .map(|session| session.dir)
                .map_err(|error| error.to_string());
            *thread_result.lock().expect("record result mutex") = Some(value);
        });
        *job.borrow_mut() = Some(RecordingJob {
            stop,
            result,
            started: Instant::now(),
            telemetry,
            mic: mic_target != "none",
            system: system_target != "none",
        });
        button.set_label("Stop recording");
        button.remove_css_class("suggested-action");
        button.add_css_class("destructive-action");
        hint.set_label("Recording… press Stop when the meeting is finished");
        mic.set_sensitive(false);
        system.set_sensitive(false);
        screenshots.set_sensitive(false);
        name.set_sensitive(false);
        live_mic_group.set_visible(mic_target != "none");
        live_system_group.set_visible(system_target != "none");
        live_root.set_visible(true);
        header_live_for_start.set_visible(true);
        quick_record_for_start.set_visible(false);
        if let Some(connection) = connection_for_start.as_ref() {
            let status = recorder_status(job.borrow().as_ref());
            remote::emit_status(connection, status);
        }
        true
    });

    let job_for_button = page.job.clone();
    let start_recording_for_button = start_recording.clone();
    let stop_recording_for_button = stop_recording.clone();
    page.button.connect_clicked(move |_| {
        if job_for_button.borrow().is_some() {
            stop_recording_for_button();
            return;
        }
        start_recording_for_button();
    });

    let session_list_for_quick = session_list.clone();
    let stack_for_quick = stack.clone();
    let start_recording_for_quick = start_recording.clone();
    quick_record.connect_clicked(move |_| {
        session_list_for_quick.unselect_all();
        stack_for_quick.set_visible_child_name("new");
        start_recording_for_quick();
    });

    let time = live.time.clone();
    let mic_level = live.mic_level.clone();
    let system_level = live.system_level.clone();
    let screenshot_count = live.screenshot_count.clone();
    let job_for_poll = page.job.clone();
    let button_for_poll = page.button.clone();
    let hint_for_poll = page.hint.clone();
    let mic_for_poll = page.mic.clone();
    let system_for_poll = page.system.clone();
    let shots_for_poll = page.screenshots.clone();
    let name_for_poll = page.name.clone();
    let live_for_poll = live.root.clone();
    let header_for_poll = header_live.clone();
    let quick_record_for_poll = quick_record.clone();
    let window_for_poll = window.clone();
    let banner_for_poll = banner.clone();
    let config_for_poll = config.clone();
    let list_for_poll = session_list.clone();
    let paths_for_poll = session_paths.clone();
    let search_for_poll = search.clone();
    let detail_for_poll = detail.clone();
    let stack_for_poll = stack.clone();
    let close_when_stopped = close_when_stopped.clone();
    let connection_for_poll = recorder_connection.clone();
    let queue_for_poll = queue.clone();
    glib::timeout_add_local(Duration::from_millis(200), move || {
        let (completed, status) = {
            let jobs = job_for_poll.borrow();
            let Some(active) = jobs.as_ref() else {
                return glib::ControlFlow::Continue;
            };
            let seconds = active.started.elapsed().as_secs();
            time.set_label(&format!("{:02}:{:02}", seconds / 60, seconds % 60));
            mic_level.set_value(f64::from(record::RecordingTelemetry::level(
                &active.telemetry.mic_level,
            )));
            system_level.set_value(f64::from(record::RecordingTelemetry::level(
                &active.telemetry.system_level,
            )));
            let count = active.telemetry.screenshots.load(Ordering::Relaxed);
            screenshot_count.set_label(&format!(
                "{count} screenshot{}",
                if count == 1 { "" } else { "s" }
            ));
            let completed = active.result.lock().expect("record result mutex").take();
            (completed, recorder_status(Some(active)))
        };
        if let Some(connection) = connection_for_poll.as_ref() {
            remote::emit_status(connection, status);
        }
        let Some(result) = completed else {
            return glib::ControlFlow::Continue;
        };
        job_for_poll.borrow_mut().take();
        if let Some(connection) = connection_for_poll.as_ref() {
            remote::emit_status(connection, remote::RecorderStatus::default());
        }
        if close_when_stopped.replace(false) {
            window_for_poll.close();
            if !window_for_poll.is_visible() {
                return glib::ControlFlow::Break;
            }
        }
        button_for_poll.set_label("Start recording");
        button_for_poll.remove_css_class("destructive-action");
        button_for_poll.add_css_class("suggested-action");
        hint_for_poll.set_label(&format!(
            "Creates a new session in {}",
            config_for_poll.borrow().meetings_dir.display()
        ));
        mic_for_poll.set_sensitive(true);
        system_for_poll.set_sensitive(true);
        shots_for_poll.set_sensitive(true);
        name_for_poll.set_sensitive(true);
        live_for_poll.set_visible(false);
        header_for_poll.set_visible(false);
        quick_record_for_poll.set_visible(true);
        time.set_label("00:00");
        mic_level.set_value(0.0);
        system_level.set_value(0.0);
        screenshot_count.set_label("0 screenshots");
        populate_sessions(
            &list_for_poll,
            &paths_for_poll,
            &config_for_poll.borrow().meetings_dir,
            &search_for_poll.text(),
            &queue_for_poll,
        );
        match result {
            Ok(path) => {
                let _ = detail_for_poll.load(&path);
                stack_for_poll.set_visible_child_name("session");
                banner_for_poll.set_title("Recording saved — ready to process");
                banner_for_poll.set_revealed(true);
                if let Some(first) = list_for_poll.row_at_index(0) {
                    list_for_poll.select_row(Some(&first));
                }
            }
            Err(error) => show_error(&window_for_poll, "Recording stopped with an error", &error),
        }
        glib::ControlFlow::Continue
    });

    let job_for_remote_start = page.job.clone();
    let session_list_for_remote = session_list.clone();
    let stack_for_remote = stack.clone();
    let start_recording_for_remote = start_recording.clone();
    let remote_start: Rc<dyn Fn() -> bool> = Rc::new(move || {
        if job_for_remote_start.borrow().is_some() {
            return true;
        }
        session_list_for_remote.unselect_all();
        stack_for_remote.set_visible_child_name("new");
        start_recording_for_remote()
    });
    let job_for_status = page.job.clone();
    let status: Rc<dyn Fn() -> remote::RecorderStatus> =
        Rc::new(move || recorder_status(job_for_status.borrow().as_ref()));
    let window_for_remote_quit = window.clone();
    let quit: Rc<dyn Fn()> = Rc::new(move || window_for_remote_quit.close());

    remote::RecorderHandlers {
        start: remote_start,
        stop: stop_recording,
        quit,
        status,
    }
}

fn recorder_status(job: Option<&RecordingJob>) -> remote::RecorderStatus {
    let Some(job) = job else {
        return remote::RecorderStatus::default();
    };
    remote::RecorderStatus {
        recording: true,
        stopping: job.stop.load(Ordering::Acquire),
        mic: job.mic,
        system: job.system,
        mic_level: f64::from(record::RecordingTelemetry::level(&job.telemetry.mic_level)),
        system_level: f64::from(record::RecordingTelemetry::level(
            &job.telemetry.system_level,
        )),
        elapsed: u32::try_from(job.started.elapsed().as_secs()).unwrap_or(u32::MAX),
        screenshots: u32::try_from(job.telemetry.screenshots.load(Ordering::Relaxed))
            .unwrap_or(u32::MAX),
    }
}

const PLACE_LOCAL: u32 = 0;
const PLACE_REMOTE: u32 = 1;

struct MeetingPrefill {
    details: MeetingDetails,
    /// Calendar attendees still to be placed in the room or on the remote end.
    attendees: Vec<(String, u32)>,
    attendee_note: Option<String>,
    caption: Option<String>,
}

struct MeetingDetailsDialog {
    dialog: adw::Dialog,
    title: adw::EntryRow,
    attendee_rows: Vec<(String, adw::ComboRow)>,
    local_names: adw::EntryRow,
    local_unknown: adw::SpinRow,
    remote_names: adw::EntryRow,
    remote_unknown: adw::SpinRow,
    primary: gtk::Button,
}

impl MeetingDetailsDialog {
    fn details(&self) -> MeetingDetails {
        let placed = |place: u32| {
            self.attendee_rows
                .iter()
                .filter(move |(_, row)| row.selected() == place)
                .map(|(name, _)| name.clone())
        };
        MeetingDetails::new(
            self.title.text().to_string(),
            Attendees {
                known: placed(PLACE_LOCAL)
                    .chain(comma_separated_names(&self.local_names.text()))
                    .collect(),
                unknown: self.local_unknown.value() as u32,
            },
            Attendees {
                known: placed(PLACE_REMOTE)
                    .chain(comma_separated_names(&self.remote_names.text()))
                    .collect(),
                unknown: self.remote_unknown.value() as u32,
            },
        )
    }
}

fn meeting_details_dialog(prefill: &MeetingPrefill, primary_label: &str) -> MeetingDetailsDialog {
    let details = &prefill.details;
    let prefill_caption = prefill.caption.as_deref();
    let dialog = adw::Dialog::builder()
        .title("Meeting details")
        .content_width(520)
        .follows_content_size(true)
        .presentation_mode(adw::DialogPresentationMode::Floating)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 16);
    body.set_margin_top(24);
    body.set_margin_bottom(24);
    body.set_margin_start(24);
    body.set_margin_end(24);

    let title_group = adw::PreferencesGroup::new();
    let title = adw::EntryRow::builder()
        .title("Meeting title")
        .text(&details.title)
        .build();
    title_group.add(&title);
    body.append(&title_group);

    let mut attendee_rows = Vec::new();
    if !prefill.attendees.is_empty() {
        let attendee_group = adw::PreferencesGroup::builder()
            .title("Attendees")
            .description("Where was each invited person? Add anyone missing below.")
            .build();
        for (name, place) in &prefill.attendees {
            let row = adw::ComboRow::builder().title(name).build();
            row.set_model(Some(&gtk::StringList::new(&[
                "In the room",
                "Remote",
                "Did not attend",
            ])));
            row.set_selected(*place);
            attendee_group.add(&row);
            attendee_rows.push((name.clone(), row));
        }
        body.append(&attendee_group);
        if let Some(note) = &prefill.attendee_note {
            let label = gtk::Label::new(Some(note));
            label.set_xalign(0.0);
            label.set_wrap(true);
            label.set_max_width_chars(60);
            label.add_css_class("dim-label");
            label.add_css_class("caption");
            body.append(&label);
        }
    }
    let names_title = if attendee_rows.is_empty() {
        "Names, comma separated"
    } else {
        "Other names, comma separated"
    };

    let local_group = adw::PreferencesGroup::builder()
        .title("In the room (microphone)")
        .build();
    let local_names = adw::EntryRow::builder()
        .title(names_title)
        .text(details.local.known.join(", "))
        .build();
    local_group.add(&local_names);
    let local_unknown = adw::SpinRow::with_range(0.0, 50.0, 1.0);
    local_unknown.set_title("Unnamed people");
    local_unknown.set_value(f64::from(details.local.unknown));
    local_group.add(&local_unknown);
    body.append(&local_group);

    let remote_group = adw::PreferencesGroup::builder()
        .title("Remote (system audio)")
        .build();
    let remote_names = adw::EntryRow::builder()
        .title(names_title)
        .text(details.remote.known.join(", "))
        .build();
    remote_group.add(&remote_names);
    let remote_unknown = adw::SpinRow::with_range(0.0, 50.0, 1.0);
    remote_unknown.set_title("Unnamed people");
    remote_unknown.set_value(f64::from(details.remote.unknown));
    remote_group.add(&remote_unknown);
    body.append(&remote_group);

    if let Some(caption) = prefill_caption {
        let caption = gtk::Label::new(Some(caption));
        caption.set_xalign(0.0);
        caption.add_css_class("dim-label");
        caption.add_css_class("caption");
        body.append(&caption);
    }
    let actions = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    actions.set_halign(gtk::Align::End);
    let cancel = gtk::Button::with_label("Cancel");
    let dialog_for_cancel = dialog.clone();
    cancel.connect_clicked(move |_| {
        dialog_for_cancel.close();
    });
    actions.append(&cancel);
    let primary = gtk::Button::with_label(primary_label);
    primary.add_css_class("suggested-action");
    actions.append(&primary);
    let scroll = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .propagate_natural_width(true)
        .propagate_natural_height(true)
        .max_content_height(520)
        .child(&body)
        .build();
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.append(&scroll);
    actions.set_margin_top(8);
    actions.set_margin_bottom(24);
    actions.set_margin_start(24);
    actions.set_margin_end(24);
    root.append(&actions);
    dialog.set_child(Some(&root));
    MeetingDetailsDialog {
        dialog,
        title,
        attendee_rows,
        local_names,
        local_unknown,
        remote_names,
        remote_unknown,
        primary,
    }
}

fn comma_separated_names(value: &str) -> Vec<String> {
    value.split(',').map(str::trim).map(str::to_owned).collect()
}

fn meeting_prefill(
    session_path: &Path,
    config: &GuiConfig,
) -> Result<MeetingPrefill, Box<dyn std::error::Error>> {
    let session = Session::open(session_path)?;
    // Renaming an unprocessed session stores a title without attendees; keep
    // that title but prefill the attendees as for a session without details.
    // Once processed, an empty roster is what the user confirmed.
    let confirmed = session.transcript_path().is_file();
    let renamed = match meeting::read_details(&session.meeting_path())? {
        Some(details) if confirmed || details.local_count() > 0 || details.remote_count() > 0 => {
            return Ok(MeetingPrefill {
                details,
                attendees: Vec::new(),
                attendee_note: None,
                caption: Some("Previous details".into()),
            });
        }
        stored => stored
            .map(|details| details.title)
            .filter(|title| !title.is_empty()),
    };
    let me = config.local_speaker.trim();
    let manifest = session.read_manifest()?;
    if let Some(context_file) = config.context_file.as_deref()
        && let Some((context, source)) =
            meeting::match_context_with_source(context_file, &manifest.started_wallclock)
    {
        let filename = source.file_name().map_or_else(
            || source.display().to_string(),
            |name| name.to_string_lossy().into(),
        );
        let invited_me = context.attendees.iter().any(|name| name == me);
        let attendees = context
            .attendees
            .into_iter()
            .map(|name| {
                let place = if name == me {
                    PLACE_LOCAL
                } else {
                    PLACE_REMOTE
                };
                (name, place)
            })
            .collect();
        let attendee_note = (!invited_me && !me.is_empty()).then(|| {
            format!(
                "Your name in Settings, {me}, is not on the invite list; pick your own row as \"In the room\"."
            )
        });
        return Ok(MeetingPrefill {
            details: MeetingDetails::new(
                renamed.unwrap_or(context.title),
                Attendees::default(),
                Attendees::default(),
            ),
            attendees,
            attendee_note,
            caption: Some(format!("From meeting-context file {filename}")),
        });
    }
    Ok(MeetingPrefill {
        details: MeetingDetails::new(
            renamed.unwrap_or_default(),
            Attendees {
                known: vec![config.local_speaker.clone()],
                unknown: 0,
            },
            Attendees::default(),
        ),
        attendees: Vec::new(),
        attendee_note: None,
        caption: None,
    })
}

struct RunningProcessing {
    path: PathBuf,
    progress: Arc<Mutex<ProcessingProgress>>,
    transcription_metrics: Arc<Mutex<Option<TranscriptionProgress>>>,
    cancellation_requested: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    paused: Cell<bool>,
    paused_at: Cell<Option<Instant>>,
    metrics_source: Cell<Option<AudioSource>>,
    paused_for_source: Cell<Duration>,
    last_worker_metrics: Cell<Option<TranscriptionProgress>>,
    display_metrics: Cell<Option<TranscriptionProgress>>,
    result: Arc<Mutex<Option<ProcessingOutcome>>>,
}

struct ProgressSnapshot {
    path: PathBuf,
    update: ProcessingProgress,
    metrics: Option<TranscriptionProgress>,
    cancelled: bool,
    paused: bool,
}

struct QueueController {
    app: gtk::Application,
    window: adw::ApplicationWindow,
    queue: Rc<RefCell<ProcessingQueue>>,
    running: RefCell<Option<RunningProcessing>>,
    timer_running: Cell<bool>,
    recording_held: Cell<bool>,
    inhibit_cookie: Cell<Option<u32>>,
    panel: QueuePanel,
    detail: SessionDetail,
    banner: adw::Banner,
    config: Rc<RefCell<GuiConfig>>,
    list: gtk::ListBox,
    paths: Rc<RefCell<Vec<PathBuf>>>,
    stack: adw::ViewStack,
    recording_job: Rc<RefCell<Option<RecordingJob>>>,
    busy: Rc<Cell<bool>>,
}

impl QueueController {
    #[allow(clippy::too_many_arguments)]
    fn new(
        app: &adw::Application,
        window: &adw::ApplicationWindow,
        queue: &Rc<RefCell<ProcessingQueue>>,
        panel: &QueuePanel,
        detail: &SessionDetail,
        banner: &adw::Banner,
        config: &Rc<RefCell<GuiConfig>>,
        list: &gtk::ListBox,
        paths: &Rc<RefCell<Vec<PathBuf>>>,
        stack: &adw::ViewStack,
        recording_job: &Rc<RefCell<Option<RecordingJob>>>,
        busy: &Rc<Cell<bool>>,
    ) -> Rc<Self> {
        let controller = Rc::new(Self {
            app: app.clone().upcast(),
            window: window.clone(),
            queue: queue.clone(),
            running: RefCell::new(None),
            timer_running: Cell::new(false),
            recording_held: Cell::new(false),
            inhibit_cookie: Cell::new(None),
            panel: panel.clone(),
            detail: detail.clone(),
            banner: banner.clone(),
            config: config.clone(),
            list: list.clone(),
            paths: paths.clone(),
            stack: stack.clone(),
            recording_job: recording_job.clone(),
            busy: busy.clone(),
        });
        let weak = Rc::downgrade(&controller);
        panel.set_handler(Rc::new(move |action| {
            let Some(controller) = weak.upgrade() else {
                return;
            };
            match action {
                QueuePanelAction::Select(path) => controller.select(&path),
                QueuePanelAction::Cancel => controller.cancel(),
                QueuePanelAction::Remove(path) => controller.remove(&path),
                QueuePanelAction::Retry(path) => {
                    controller.enqueue(path);
                }
                QueuePanelAction::Details(error) => {
                    show_error(&controller.window, "Processing failed", &error)
                }
                QueuePanelAction::Clear => {
                    let failed = controller
                        .queue
                        .borrow()
                        .failed()
                        .iter()
                        .map(|job| job.path.clone())
                        .collect::<Vec<_>>();
                    controller.queue.borrow_mut().clear_failed();
                    for path in failed {
                        controller.update_row(&path);
                    }
                    controller.detail.refresh_queue_state();
                    controller.render_panel();
                }
            }
        }));
        controller.render_panel();
        controller
    }

    fn enqueue(self: &Rc<Self>, path: PathBuf) -> bool {
        if !self.queue.borrow_mut().enqueue(path.clone()) {
            return false;
        }
        self.ensure_inhibited();
        self.update_row(&path);
        self.detail.refresh_queue_state();
        self.render_panel();
        self.start_timer();
        true
    }

    fn remove(&self, path: &Path) {
        if !self.queue.borrow_mut().remove(path) {
            return;
        }
        self.update_row(path);
        self.detail.refresh_queue_state();
        self.render_panel();
        if self.queue.borrow().is_idle() {
            self.release_inhibit();
        }
    }

    fn cancel(&self) {
        {
            let running = self.running.borrow();
            let Some(running) = running.as_ref() else {
                return;
            };
            running
                .cancellation_requested
                .store(true, Ordering::Release);
            if let Err(error) = signal_child(&running.child, Signal::KILL) {
                eprintln!("warning: cannot cancel processing child: {error}");
            }
        }
        self.detail.cancel_requested.set(true);
        self.detail.refresh_queue_state();
        self.render_panel();
    }

    fn select(&self, path: &Path) {
        if let Some(index) = self
            .paths
            .borrow()
            .iter()
            .position(|candidate| candidate == path)
        {
            select_session_row(&self.list, index);
        } else {
            self.list.unselect_all();
            if self.detail.load(path).is_ok() {
                self.stack.set_visible_child_name("session");
            } else if self.queue.borrow_mut().remove_failed(path) {
                self.detail.clear();
                self.render_panel();
            }
        }
    }

    fn ensure_inhibited(&self) {
        if self.inhibit_cookie.get().is_none() {
            let cookie = self.app.inhibit(
                Some(&self.window),
                gtk::ApplicationInhibitFlags::SUSPEND,
                Some("Processing recordings"),
            );
            if cookie != 0 {
                self.inhibit_cookie.set(Some(cookie));
            }
        }
    }

    fn release_inhibit(&self) {
        if let Some(cookie) = self.inhibit_cookie.take() {
            self.app.uninhibit(cookie);
        }
    }

    fn start_timer(self: &Rc<Self>) {
        if self.timer_running.replace(true) {
            return;
        }
        let weak = Rc::downgrade(self);
        glib::timeout_add_local(Duration::from_millis(150), move || {
            let Some(controller) = weak.upgrade() else {
                return glib::ControlFlow::Break;
            };
            let flow = controller.tick();
            if flow == glib::ControlFlow::Break {
                controller.timer_running.set(false);
            }
            flow
        });
    }

    fn tick(self: &Rc<Self>) -> glib::ControlFlow {
        self.apply_progress_snapshot();
        let completed = {
            let running = self.running.borrow();
            running.as_ref().and_then(|running| {
                running
                    .result
                    .lock()
                    .expect("processing result mutex")
                    .take()
            })
        };
        if let Some(outcome) = completed {
            self.complete(outcome);
            return glib::ControlFlow::Continue;
        }

        self.update_pause_state();

        if self.running.borrow().is_none() && !self.queue.borrow().waiting().is_empty() {
            if self.recording_job.borrow().is_some() {
                if !self.recording_held.replace(true) {
                    self.render_panel();
                }
                return glib::ControlFlow::Continue;
            }
            if self.recording_held.replace(false) {
                self.render_panel();
            }
            let dialog_holds_selected = {
                let queue = self.queue.borrow();
                let selected = self.detail.selected.borrow();
                queue.waiting().front().is_some_and(|path| {
                    selected.as_deref() == Some(path.as_path())
                        && self.window.visible_dialog().is_some()
                })
            };
            if !self.busy.get() && !dialog_holds_selected {
                self.start_next();
            }
        }

        if self.queue.borrow().is_idle() {
            self.release_inhibit();
            glib::ControlFlow::Break
        } else {
            glib::ControlFlow::Continue
        }
    }

    fn update_pause_state(&self) {
        let recording = self.recording_job.borrow().is_some();
        let rerender = {
            let running = self.running.borrow();
            let Some(running) = running.as_ref() else {
                return;
            };
            if recording && !running.paused.get() {
                match signal_child(&running.child, Signal::STOP) {
                    Ok(true) => {
                        running.paused.set(true);
                        running.paused_at.set(Some(Instant::now()));
                        true
                    }
                    Ok(false) => false,
                    Err(error) => {
                        eprintln!("warning: cannot pause processing child: {error}");
                        false
                    }
                }
            } else if !recording && running.paused.get() {
                match signal_child(&running.child, Signal::CONT) {
                    Ok(true) => {
                        if running.metrics_source.get().is_some()
                            && let Some(started) = running.paused_at.get()
                        {
                            running
                                .paused_for_source
                                .set(running.paused_for_source.get() + started.elapsed());
                        }
                        running.paused.set(false);
                        running.paused_at.set(None);
                        true
                    }
                    Ok(false) => false,
                    Err(error) => {
                        eprintln!("warning: cannot resume processing child: {error}");
                        false
                    }
                }
            } else {
                false
            }
        };
        if rerender {
            self.render_panel();
        }
    }

    fn start_next(&self) {
        let Some(path) = self.queue.borrow_mut().start_next() else {
            return;
        };
        self.detail.cancel_requested.set(false);
        let progress = Arc::new(Mutex::new(ProcessingProgress {
            stage: ProcessingStage::Preparing,
            fraction: None,
        }));
        let transcription_metrics = Arc::new(Mutex::new(None));
        let cancellation_requested = Arc::new(AtomicBool::new(false));
        let child = Arc::new(Mutex::new(None));
        let result = Arc::new(Mutex::new(None));
        *self.running.borrow_mut() = Some(RunningProcessing {
            path: path.clone(),
            progress: progress.clone(),
            transcription_metrics: transcription_metrics.clone(),
            cancellation_requested: cancellation_requested.clone(),
            child: child.clone(),
            paused: Cell::new(false),
            paused_at: Cell::new(None),
            metrics_source: Cell::new(None),
            paused_for_source: Cell::new(Duration::ZERO),
            last_worker_metrics: Cell::new(None),
            display_metrics: Cell::new(None),
            result: result.clone(),
        });
        reset_processing_view(&self.detail.processing);
        self.update_row(&path);
        self.detail.refresh_queue_state();
        self.render_panel();

        let swedish_transcription = self.config.borrow().swedish_transcription;
        std::thread::spawn(move || {
            let prepared = (|| -> Result<ProcessArgs, String> {
                let mut args = ProcessArgs::for_session(path);
                args.diarize_mic = Session::open(&args.session)
                    .and_then(|session| session.read_manifest())
                    .map_err(|error| error.to_string())?
                    .mic
                    .enabled;
                let model_env = if swedish_transcription {
                    "SINGSTONE_WHISPER_MODEL_SWEDISH"
                } else {
                    "SINGSTONE_WHISPER_MODEL_MULTILINGUAL"
                };
                if let Some(model) = std::env::var_os(model_env) {
                    args.whisper_model = Some(PathBuf::from(model));
                }
                args.language = if swedish_transcription { "sv" } else { "auto" }.into();
                args.events = true;
                Ok(args)
            })();
            let outcome = match prepared {
                Err(_) if cancellation_requested.load(Ordering::Acquire) => {
                    ProcessingOutcome::Cancelled
                }
                Err(error) => ProcessingOutcome::Failed(error),
                Ok(_) if cancellation_requested.load(Ordering::Acquire) => {
                    ProcessingOutcome::Cancelled
                }
                Ok(args) => {
                    let spawned = std::env::current_exe()
                        .map_err(|error| format!("cannot find the processing executable: {error}"))
                        .and_then(|executable| {
                            ProcessCommand::new(executable)
                                .args(args.command_args())
                                .stdin(Stdio::null())
                                .stdout(Stdio::piped())
                                .stderr(Stdio::inherit())
                                .spawn()
                                .map_err(|error| format!("cannot start processing worker: {error}"))
                        });
                    match spawned {
                        Err(_) if cancellation_requested.load(Ordering::Acquire) => {
                            ProcessingOutcome::Cancelled
                        }
                        Err(error) => ProcessingOutcome::Failed(error),
                        Ok(mut spawned) => {
                            let stdout = spawned.stdout.take();
                            *child.lock().expect("processing child mutex") = Some(spawned);
                            if cancellation_requested.load(Ordering::Acquire) {
                                let _ = signal_child(&child, Signal::KILL);
                            }

                            let failure = stdout.and_then(|stdout| {
                                read_worker_events(stdout, &progress, &transcription_metrics)
                            });

                            let status = {
                                let mut child = child.lock().expect("processing child mutex");
                                let status = child
                                    .as_mut()
                                    .expect("processing child stored before event read")
                                    .wait();
                                *child = None;
                                status
                            };
                            match status {
                                Ok(status) => processing_outcome(
                                    status,
                                    failure,
                                    cancellation_requested.load(Ordering::Acquire),
                                ),
                                Err(_) if cancellation_requested.load(Ordering::Acquire) => {
                                    ProcessingOutcome::Cancelled
                                }
                                Err(error) => ProcessingOutcome::Failed(format!(
                                    "cannot reap processing worker: {error}"
                                )),
                            }
                        }
                    }
                }
            };
            *result.lock().expect("processing result mutex") = Some(outcome);
        });
    }

    fn complete(&self, outcome: ProcessingOutcome) {
        let Some(running) = self.running.borrow_mut().take() else {
            return;
        };
        let path = running.path;
        let model_outcome = match &outcome {
            ProcessingOutcome::Finished => QueueOutcome::Finished,
            ProcessingOutcome::Cancelled => QueueOutcome::Cancelled,
            ProcessingOutcome::Failed(error) => QueueOutcome::Failed(error.clone()),
        };
        self.queue.borrow_mut().finish(model_outcome);
        self.detail.cancel_requested.set(false);
        self.update_row(&path);
        if self.detail.selected.borrow().as_deref() == Some(path.as_path()) {
            if self.detail.load(&path).is_err() {
                self.detail.clear();
            }
        } else {
            self.detail.refresh_queue_state();
        }
        self.render_panel();

        let queue = self.queue.borrow();
        let idle = queue.is_idle();
        match outcome {
            ProcessingOutcome::Finished if idle => {
                if queue.completed_count() == 1
                    && queue.processed_count() == 1
                    && queue.failed_count() == 0
                {
                    self.banner.set_title(
                        "Processing finished — transcript.jsonl and transcript.txt are ready",
                    );
                } else {
                    self.banner.set_title(&finished_banner_text(
                        queue.processed_count(),
                        queue.failed_count(),
                    ));
                }
                self.banner.set_revealed(true);
            }
            ProcessingOutcome::Cancelled => {
                self.banner.set_title(
                    "Processing cancelled — recorded audio is unchanged; completed stages may have refreshed outputs",
                );
                self.banner.set_revealed(true);
            }
            ProcessingOutcome::Failed(_) if idle && queue.processed_count() > 0 => {
                self.banner.set_title(&finished_banner_text(
                    queue.processed_count(),
                    queue.failed_count(),
                ));
                self.banner.set_revealed(true);
            }
            _ => {}
        }
        drop(queue);
        if idle {
            self.release_inhibit();
        }
    }

    fn update_row(&self, path: &Path) {
        update_session_row(&self.list, &self.paths, path, &self.queue.borrow());
    }

    fn render_panel(&self) {
        self.panel.render(
            &self.queue.borrow(),
            self.recording_job.borrow().is_some(),
            self.detail.cancel_requested.get(),
        );
        self.apply_progress_snapshot();
    }

    fn render_panel_if_visible(&self) {
        let queue = self.queue.borrow();
        let visible = !queue.is_idle() || !queue.failed().is_empty();
        drop(queue);
        if visible {
            self.render_panel();
        }
    }

    fn progress_snapshot(&self) -> Option<ProgressSnapshot> {
        self.running.borrow().as_ref().map(|running| {
            let metrics = *running
                .transcription_metrics
                .lock()
                .expect("transcription metrics mutex");
            if metrics != running.last_worker_metrics.get() {
                if let Some(metrics) = metrics {
                    if running.metrics_source.get() != Some(metrics.source) {
                        running.metrics_source.set(Some(metrics.source));
                        running.paused_for_source.set(Duration::ZERO);
                    }
                    running.display_metrics.set(Some(compensate_paused_elapsed(
                        metrics,
                        running.paused_for_source.get(),
                    )));
                } else {
                    running.display_metrics.set(None);
                }
                running.last_worker_metrics.set(metrics);
            }
            ProgressSnapshot {
                path: running.path.clone(),
                update: *running.progress.lock().expect("processing progress mutex"),
                metrics: running.display_metrics.get(),
                cancelled: self.detail.cancel_requested.get(),
                paused: running.paused.get(),
            }
        })
    }

    fn apply_progress_snapshot(&self) {
        let Some(snapshot) = self.progress_snapshot() else {
            return;
        };
        let current_metrics = snapshot.metrics.filter(|metrics| {
            matches!(
                (snapshot.update.stage, metrics.source),
                (ProcessingStage::TranscribingMic, AudioSource::Mic)
                    | (ProcessingStage::TranscribingSystem, AudioSource::System)
            )
        });
        self.panel.update_progress(
            snapshot.update,
            current_metrics,
            snapshot.cancelled,
            snapshot.paused,
            self.queue.borrow().batch_position_total(),
        );
        if self.detail.selected.borrow().as_deref() == Some(snapshot.path.as_path()) {
            update_processing_view(
                &self.detail.processing,
                snapshot.update,
                current_metrics,
                snapshot.cancelled,
                snapshot.paused,
            );
        }
    }

    fn will_wait(&self) -> bool {
        !self.queue.borrow().is_idle() || self.recording_job.borrow().is_some()
    }

    fn stop_for_quit(&self) {
        let waiting = self
            .queue
            .borrow()
            .waiting()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        self.queue.borrow_mut().clear_waiting();
        for path in waiting {
            self.update_row(&path);
        }
        if let Some(running) = self.running.borrow().as_ref() {
            running
                .cancellation_requested
                .store(true, Ordering::Release);
            if let Err(error) = signal_child(&running.child, Signal::KILL) {
                eprintln!("warning: cannot stop processing child during quit: {error}");
            }
            self.detail.cancel_requested.set(true);
        }
        self.detail.refresh_queue_state();
        self.render_panel();
        self.release_inhibit();
    }
}

fn update_processing_view(
    view: &ProcessingView,
    update: ProcessingProgress,
    metrics: Option<TranscriptionProgress>,
    cancelled: bool,
    paused: bool,
) {
    view.label.set_label(if cancelled {
        "Cancelling…"
    } else if paused {
        "Paused while recording"
    } else {
        update.stage.label()
    });
    view.spinner.set_spinning(!paused);
    update_processing_stages(&view.stages, update.stage);
    view.metrics.set_visible(metrics.is_some());
    if let Some(metrics) = metrics {
        update_transcription_metrics(&view.throughput, &view.throughput_detail, metrics);
        update_transcription_progress(&view.progress, metrics);
    } else if let Some(fraction) = update.fraction {
        view.progress.set_fraction(fraction);
        view.progress
            .set_text(Some(&format!("{:.0}%", fraction * 100.0)));
    } else if !paused {
        view.progress.pulse();
        view.progress.set_text(Some("Working…"));
    }
}

fn reset_processing_view(view: &ProcessingView) {
    view.spinner.set_spinning(true);
    view.label.set_label(ProcessingStage::Preparing.label());
    update_processing_stages(&view.stages, ProcessingStage::Preparing);
    view.metrics.set_visible(false);
    view.progress.set_fraction(0.0);
    view.progress.set_text(Some("Working…"));
}

#[allow(clippy::too_many_arguments)]
fn wire_processing(
    app: &adw::Application,
    window: &adw::ApplicationWindow,
    detail: &SessionDetail,
    banner: &adw::Banner,
    config: &Rc<RefCell<GuiConfig>>,
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
    queue: &Rc<RefCell<ProcessingQueue>>,
    panel: &QueuePanel,
    stack: &adw::ViewStack,
    recording_job: &Rc<RefCell<Option<RecordingJob>>>,
    busy: &Rc<Cell<bool>>,
) -> Rc<QueueController> {
    wire_archiving(
        window, detail, banner, config, list, paths, search, busy, queue,
    );
    let controller = QueueController::new(
        app,
        window,
        queue,
        panel,
        detail,
        banner,
        config,
        list,
        paths,
        stack,
        recording_job,
        busy,
    );

    let controller_for_process = controller.clone();
    detail.process_button.connect_clicked(move |_| {
        if controller_for_process.busy.get() {
            return;
        }
        let Some(path) = controller_for_process.detail.selected.borrow().clone() else {
            return;
        };
        let state = controller_for_process.queue.borrow().state(&path);
        match state {
            Some(JobState::Running) => controller_for_process.cancel(),
            Some(JobState::Waiting { .. }) => controller_for_process.remove(&path),
            None => begin_process_flow(controller_for_process.clone(), path),
        }
    });

    wire_meeting_details(&controller, search);
    controller
}

fn begin_process_flow(controller: Rc<QueueController>, path: PathBuf) {
    if path.join("transcript.jsonl").is_file() {
        let dialog = adw::AlertDialog::new(
            Some("Reprocess this recording?"),
            Some(
                "This replaces the transcript, intermediate processing files, and automatic speaker matches. Recorded audio is unchanged, and speaker names you confirmed in this version are kept. Names assigned with an older version are replaced.",
            ),
        );
        dialog.add_responses(&[("cancel", "Cancel"), ("reprocess", "Reprocess")]);
        dialog.set_default_response(Some("reprocess"));
        dialog.set_close_response("cancel");
        dialog.set_response_appearance("reprocess", adw::ResponseAppearance::Suggested);
        let controller_for_response = controller.clone();
        dialog.connect_response(Some("reprocess"), move |_, _| {
            show_process_details(controller_for_response.clone(), path.clone());
        });
        dialog.present(Some(&controller.window));
    } else {
        show_process_details(controller, path);
    }
}

fn show_process_details(controller: Rc<QueueController>, path: PathBuf) {
    let prefill = match meeting_prefill(&path, &controller.config.borrow()) {
        Ok(prefill) => prefill,
        Err(error) => {
            show_error(
                &controller.window,
                "Could not read meeting details",
                &error.to_string(),
            );
            return;
        }
    };
    let primary = if controller.will_wait() {
        "Add to queue"
    } else {
        "Process"
    };
    let dialog = Rc::new(meeting_details_dialog(&prefill, primary));
    let dialog_for_save = dialog.clone();
    let controller_for_save = controller.clone();
    dialog.primary.connect_clicked(move |_| {
        let details = dialog_for_save.details();
        match Session::open(&path)
            .and_then(|session| meeting::write_details_atomic(&session.meeting_path(), &details))
        {
            Ok(()) => {
                dialog_for_save.dialog.force_close();
                controller_for_save.enqueue(path.clone());
            }
            Err(error) => show_error(
                &controller_for_save.window,
                "Could not save meeting details",
                &error.to_string(),
            ),
        }
    });
    dialog.dialog.present(Some(&controller.window));
}

fn wire_meeting_details(controller: &Rc<QueueController>, search: &gtk::SearchEntry) {
    let controller = controller.clone();
    let search = search.clone();
    let detail = controller.detail.clone();
    detail.meeting_button.connect_clicked(move |_| {
        if controller.busy.get() {
            return;
        }
        let Some(path) = controller.detail.selected.borrow().clone() else {
            return;
        };
        let processed = path.join("transcript.jsonl").is_file();
        let prefill = match meeting_prefill(&path, &controller.config.borrow()) {
            Ok(prefill) => prefill,
            Err(error) => {
                show_error(
                    &controller.window,
                    "Could not read meeting details",
                    &error.to_string(),
                );
                return;
            }
        };
        let dialog = Rc::new(meeting_details_dialog(
            &prefill,
            if processed {
                "Save and re-render"
            } else {
                "Save"
            },
        ));
        let dialog_for_save = dialog.clone();
        let controller_for_save = controller.clone();
        let search_for_save = search.clone();
        dialog.primary.connect_clicked(move |_| {
            let details = dialog_for_save.details();
            let session = match Session::open(&path) {
                Ok(session) => session,
                Err(error) => {
                    show_error(
                        &controller_for_save.window,
                        "Could not open session",
                        &error.to_string(),
                    );
                    return;
                }
            };
            if let Err(error) = meeting::write_details_atomic(&session.meeting_path(), &details) {
                show_error(
                    &controller_for_save.window,
                    "Could not save meeting details",
                    &error.to_string(),
                );
                return;
            }
            dialog_for_save.dialog.force_close();
            controller_for_save.render_panel_if_visible();
            if !processed {
                let _ = controller_for_save.detail.load(&path);
                controller_for_save.update_row(&path);
                return;
            }

            controller_for_save.busy.set(true);
            let progress = progress_window(
                &controller_for_save.window,
                "Rendering transcript",
                "Saving meeting details and re-rendering the transcript…",
            );
            let result = Arc::new(Mutex::new(None));
            let thread_result = result.clone();
            let render_path = path.clone();
            std::thread::spawn(move || {
                let value = process::run_render(RenderArgs {
                    session: render_path,
                    diarize_mic: None,
                })
                .map_err(|error| error.to_string());
                *thread_result.lock().expect("render result mutex") = Some(value);
            });
            let controller = controller_for_save.clone();
            let reload_path = path.clone();
            let search = search_for_save.clone();
            glib::timeout_add_local(Duration::from_millis(150), move || {
                let Some(result) = result.lock().expect("render result mutex").take() else {
                    return glib::ControlFlow::Continue;
                };
                controller.busy.set(false);
                progress.close();
                match result {
                    Ok(()) => {
                        let _ = controller.detail.load(&reload_path);
                        populate_sessions(
                            &controller.list,
                            &controller.paths,
                            &controller.config.borrow().meetings_dir,
                            &search.text(),
                            &controller.queue,
                        );
                        controller
                            .banner
                            .set_title("Meeting details saved and transcript re-rendered");
                        controller.banner.set_revealed(true);
                    }
                    Err(error) => {
                        show_error(&controller.window, "Could not re-render transcript", &error)
                    }
                }
                glib::ControlFlow::Break
            });
        });
        dialog.dialog.present(Some(&controller.window));
    });
}

#[allow(clippy::too_many_arguments)]
fn wire_archiving(
    window: &adw::ApplicationWindow,
    detail: &SessionDetail,
    banner: &adw::Banner,
    config: &Rc<RefCell<GuiConfig>>,
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
    busy: &Rc<Cell<bool>>,
    queue: &Rc<RefCell<ProcessingQueue>>,
) {
    let parent = window.clone();
    let detail_for_archive = detail.clone();
    let banner = banner.clone();
    let config = config.clone();
    let list = list.clone();
    let paths = paths.clone();
    let search = search.clone();
    let busy = busy.clone();
    let queue = queue.clone();
    detail.archive_button.connect_clicked(move |_| {
        if busy.get() {
            return;
        }
        let Some(session_path) = detail_for_archive.selected.borrow().clone() else {
            return;
        };
        let dialog = adw::AlertDialog::new(
            Some("Archive this session's audio?"),
            Some(
                "The recorded audio is converted to 16-bit FLAC, which typically takes a fifth of the space. Playback and reprocessing keep working. The original float recording is deleted once the copy is verified; this cannot be undone.",
            ),
        );
        dialog.add_responses(&[("cancel", "Cancel"), ("archive", "Archive")]);
        dialog.set_default_response(Some("archive"));
        dialog.set_close_response("cancel");
        dialog.set_response_appearance("archive", adw::ResponseAppearance::Suggested);
        let parent_for_response = parent.clone();
        let detail = detail_for_archive.clone();
        let banner = banner.clone();
        let config = config.clone();
        let list = list.clone();
        let paths = paths.clone();
        let search = search.clone();
        let busy = busy.clone();
        let queue = queue.clone();
        dialog.connect_response(Some("archive"), move |_, _| {
            detail.playback.stop();
            busy.set(true);
            let progress = gtk::Window::builder()
                .title("Archiving audio")
                .transient_for(&parent_for_response)
                .modal(true)
                .deletable(false)
                .default_width(360)
                .build();
            let body = gtk::Box::new(gtk::Orientation::Vertical, 12);
            body.set_margin_top(24);
            body.set_margin_bottom(24);
            body.set_margin_start(24);
            body.set_margin_end(24);
            let spinner = gtk::Spinner::new();
            spinner.set_spinning(true);
            body.append(&spinner);
            body.append(&gtk::Label::new(Some(
                "Converting the recorded audio to FLAC and verifying the copy…",
            )));
            progress.set_child(Some(&body));
            progress.present();

            let result = Arc::new(Mutex::new(None));
            let thread_result = result.clone();
            let archive_session = session_path.clone();
            std::thread::spawn(move || {
                let value = Session::open(&archive_session)
                    .and_then(|session| archive::archive_session(&session))
                    .map_err(|error| error.to_string());
                *thread_result.lock().expect("archive result mutex") = Some(value);
            });
            let parent = parent_for_response.clone();
            let detail = detail.clone();
            let banner = banner.clone();
            let config = config.clone();
            let list = list.clone();
            let paths = paths.clone();
            let search = search.clone();
            let busy = busy.clone();
            let queue = queue.clone();
            let session_path = session_path.clone();
            glib::timeout_add_local(Duration::from_millis(150), move || {
                let Some(result) = result.lock().expect("archive result mutex").take() else {
                    return glib::ControlFlow::Continue;
                };
                busy.set(false);
                progress.close();
                populate_sessions(
                    &list,
                    &paths,
                    &config.borrow().meetings_dir,
                    &search.text(),
                    &queue,
                );
                let index = paths.borrow().iter().position(|path| *path == session_path);
                match index {
                    Some(index) => select_session_row(&list, index),
                    None => {
                        let _ = detail.load(&session_path);
                    }
                }
                match result {
                    Ok(summary) => {
                        banner.set_title(&format!(
                            "Audio archived — {} is now {}",
                            archive::format_size(summary.bytes_before),
                            archive::format_size(summary.bytes_after)
                        ));
                        banner.set_revealed(true);
                    }
                    Err(error) => show_error(&parent, "Could not archive audio", &error),
                }
                glib::ControlFlow::Break
            });
        });
        dialog.present(Some(&parent));
    });
}

#[derive(Debug, PartialEq)]
enum ProcessingOutcome {
    Finished,
    Cancelled,
    Failed(String),
}

fn signal_child(child: &Mutex<Option<Child>>, signal: Signal) -> io::Result<bool> {
    let child = child
        .lock()
        .map_err(|_| io::Error::other("processing child lock was poisoned"))?;
    let Some(child) = child.as_ref() else {
        return Ok(false);
    };
    let raw_pid = i32::try_from(child.id())
        .ok()
        .and_then(Pid::from_raw)
        .ok_or_else(|| io::Error::other("processing child has an invalid process id"))?;
    rustix::process::kill_process(raw_pid, signal)?;
    Ok(true)
}

/// Applies the worker's events until its stdout closes and returns the failure it reported.
/// Lines that are not events come from native libraries and are passed on to stderr.
fn read_worker_events(
    stdout: impl Read,
    progress: &Mutex<ProcessingProgress>,
    transcription_metrics: &Mutex<Option<TranscriptionProgress>>,
) -> Option<String> {
    let mut failure = None;
    let mut reader = BufReader::new(stdout);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                eprintln!("warning: cannot read processing worker events: {error}");
                break;
            }
        }
        let text = String::from_utf8_lossy(&line);
        match events::parse(text.trim_end()) {
            Ok(ProcessingEvent::Progress { progress: update }) => {
                *progress.lock().expect("processing progress mutex") = update;
            }
            Ok(ProcessingEvent::Transcription { progress: metrics }) => {
                *transcription_metrics
                    .lock()
                    .expect("transcription metrics mutex") = Some(metrics);
            }
            Ok(ProcessingEvent::Failure { error }) => failure = Some(error),
            Err(_) => eprintln!("{}", text.trim_end()),
        }
    }
    failure
}

fn processing_outcome(
    status: ExitStatus,
    failure: Option<String>,
    cancellation_requested: bool,
) -> ProcessingOutcome {
    if status.success() {
        return ProcessingOutcome::Finished;
    }
    if cancellation_requested {
        return ProcessingOutcome::Cancelled;
    }
    if let Some(error) = failure {
        return ProcessingOutcome::Failed(error);
    }
    let detail = if let Some(signal) = status.signal() {
        format!("signal {signal}")
    } else if let Some(code) = status.code() {
        format!("exit status {code}")
    } else {
        "an unknown status".into()
    };
    ProcessingOutcome::Failed(format!("processing stopped unexpectedly ({detail})"))
}

fn compensate_paused_elapsed(
    mut metrics: TranscriptionProgress,
    paused: Duration,
) -> TranscriptionProgress {
    metrics.elapsed_seconds = (metrics.elapsed_seconds - paused.as_secs_f64()).max(0.0);
    metrics
}

fn processing_view(backend: &backend::WhisperBackend) -> ProcessingView {
    let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
    root.set_hexpand(true);
    let body = gtk::Box::new(gtk::Orientation::Vertical, 12);
    body.set_margin_top(16);
    body.set_margin_bottom(16);
    body.set_margin_start(16);
    body.set_margin_end(16);
    let spinner = gtk::Spinner::new();
    spinner.set_spinning(true);
    spinner.set_size_request(38, 38);
    body.append(&spinner);
    let label = gtk::Label::new(Some(ProcessingStage::Preparing.label()));
    label.add_css_class("title-3");
    label.set_wrap(true);
    body.append(&label);
    let compute = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    compute.set_halign(gtk::Align::Center);
    compute.append(&status_pill(
        if backend.accelerated { "GPU" } else { "CPU" },
        if backend.accelerated { "ok" } else { "idle" },
    ));
    let compute_description = gtk::Label::new(Some(&backend.description));
    compute_description.add_css_class("dim-label");
    compute_description.set_wrap(true);
    compute.append(&compute_description);
    body.append(&compute);
    let metrics = gtk::Box::new(gtk::Orientation::Vertical, 2);
    metrics.set_halign(gtk::Align::Center);
    metrics.set_visible(false);
    let throughput = gtk::Label::new(Some("Measuring decoder throughput…"));
    throughput.add_css_class("title-4");
    throughput.set_tooltip_text(Some(
        "Token rate is for the latest completed chunk. Realtime speed is the audio processed per second since the current track started.",
    ));
    metrics.append(&throughput);
    let throughput_detail = gtk::Label::new(Some("Starts when Whisper begins transcribing"));
    throughput_detail.add_css_class("dim-label");
    metrics.append(&throughput_detail);
    body.append(&metrics);
    let progress = gtk::ProgressBar::new();
    progress.set_pulse_step(0.04);
    progress.set_show_text(true);
    progress.set_text(Some("Working…"));
    body.append(&progress);
    let stages_box = gtk::Box::new(gtk::Orientation::Vertical, 7);
    stages_box.set_margin_top(4);
    let mut stages = Vec::new();
    for stage in ProcessingStage::ALL
        .into_iter()
        .filter(|stage| *stage != ProcessingStage::Finished)
    {
        let row = gtk::Label::new(Some(&format!("○  {}", stage.label())));
        row.set_xalign(0.0);
        row.add_css_class("dim-label");
        stages_box.append(&row);
        stages.push(row);
    }
    body.append(&stages_box);
    let note = gtk::Label::new(Some(
        "This runs in the background. You can open other sessions, review transcripts, or add more recordings to the queue.",
    ));
    note.add_css_class("queue-progress-hint");
    note.set_wrap(true);
    body.append(&note);
    let clamp = adw::Clamp::builder().maximum_size(470).child(&body).build();
    clamp.set_valign(gtk::Align::Center);
    clamp.set_vexpand(true);
    root.append(&clamp);
    ProcessingView {
        root,
        spinner,
        label,
        progress,
        stages,
        metrics,
        throughput,
        throughput_detail,
    }
}

fn update_transcription_metrics(
    throughput: &gtk::Label,
    detail: &gtk::Label,
    metrics: TranscriptionProgress,
) {
    let (throughput_text, detail_text) = transcription_metrics_text(metrics);
    throughput.set_label(&throughput_text);
    detail.set_label(&detail_text);
}

fn update_transcription_progress(bar: &gtk::ProgressBar, metrics: TranscriptionProgress) {
    let fraction = metrics.fraction();
    bar.set_fraction(fraction);
    bar.set_text(Some(&transcription_progress_text(metrics)));
}

fn transcription_progress_text(metrics: TranscriptionProgress) -> String {
    format!(
        "{:.0} / {:.0} s  ·  {:.0}%",
        metrics.processed_seconds,
        metrics.total_seconds,
        metrics.fraction() * 100.0
    )
}

fn transcription_metrics_text(metrics: TranscriptionProgress) -> (String, String) {
    let speed = metrics.realtime_speed();
    let token_rate = metrics
        .tokens_per_second
        .map(|rate| format!("{rate:.1} tokens/s"))
        .unwrap_or_else(|| "Measuring tokens/s".into());
    (
        format!("{token_rate}  ·  {speed:.2}× realtime"),
        format!(
            "{} · {:.1} s audio in {:.1} s · {} tokens",
            metrics.source, metrics.audio_seconds, metrics.elapsed_seconds, metrics.decoded_tokens
        ),
    )
}

fn update_processing_stages(rows: &[gtk::Label], current: ProcessingStage) {
    let current = current.index();
    for (index, row) in rows.iter().enumerate() {
        let stage = ProcessingStage::ALL[index];
        let marker = if index < current {
            "✓"
        } else if index == current {
            "●"
        } else {
            "○"
        };
        row.set_label(&format!("{marker}  {}", stage.label()));
        if index <= current {
            row.remove_css_class("dim-label");
        } else {
            row.add_css_class("dim-label");
        }
    }
}

impl PlaybackController {
    fn allocate_id(&self) -> u64 {
        let id = self.next_id.get().wrapping_add(1);
        self.next_id.set(id);
        id
    }

    fn toggle(
        &self,
        row_id: u64,
        button: &gtk::Button,
        window: &adw::ApplicationWindow,
        audio_path: &Path,
        start_ms: u64,
        end_ms: u64,
    ) {
        if self
            .active
            .borrow()
            .as_ref()
            .is_some_and(|active| active.row_id == row_id)
        {
            self.stop();
            return;
        }
        self.stop();

        let generation = self.allocate_id();
        match start_audio_playback(audio_path, start_ms, end_ms) {
            Ok(spawned) => {
                set_playback_button(button, true);
                *self.active.borrow_mut() = Some(ActivePlayback {
                    row_id,
                    generation,
                    child: spawned.child,
                    button: button.clone(),
                    writer_error: spawned.writer_error,
                    stderr: spawned.stderr,
                });
                self.poll(generation, window.clone());
            }
            Err(error) => show_error(window, "Could not play audio", &error.to_string()),
        }
    }

    fn poll(&self, generation: u64, window: adw::ApplicationWindow) {
        let controller = self.clone();
        glib::timeout_add_local(Duration::from_millis(100), move || {
            let completed = {
                let mut slot = controller.active.borrow_mut();
                let Some(active) = slot.as_mut() else {
                    return glib::ControlFlow::Break;
                };
                if active.generation != generation {
                    return glib::ControlFlow::Break;
                }
                match active.child.try_wait() {
                    Ok(None) => None,
                    Ok(Some(status)) => Some(Ok(status)),
                    Err(error) => Some(Err(error)),
                }
            };

            let Some(completed) = completed else {
                return glib::ControlFlow::Continue;
            };
            let Some(active) = controller.active.borrow_mut().take() else {
                return glib::ControlFlow::Break;
            };
            set_playback_button(&active.button, false);
            let writer_error = active
                .writer_error
                .lock()
                .ok()
                .and_then(|mut result| result.take());
            let stderr = active
                .stderr
                .lock()
                .map(|output| output.trim().to_owned())
                .unwrap_or_default();
            let error = match (completed, writer_error, stderr.as_str()) {
                (Ok(status), None, _) if status.success() => None,
                (Ok(status), Some(error), _) if status.success() => Some(error),
                (Ok(status), _, stderr) if !stderr.is_empty() => {
                    Some(format!("PipeWire playback exited with {status}:\n{stderr}"))
                }
                (Ok(status), Some(error), _) => Some(format!(
                    "PipeWire playback exited with {status}. Audio streaming failed: {error}"
                )),
                (Ok(status), None, _) => Some(format!("PipeWire playback exited with {status}.")),
                (Err(error), _, _) => Some(format!("Could not monitor PipeWire playback: {error}")),
            };
            if let Some(error) = error {
                show_error(&window, "Audio playback stopped", &error);
            }
            glib::ControlFlow::Break
        });
    }

    fn stop(&self) {
        let Some(mut active) = self.active.borrow_mut().take() else {
            return;
        };
        if active.child.try_wait().ok().flatten().is_none() {
            let _ = active.child.kill();
        }
        let _ = active.child.wait();
        set_playback_button(&active.button, false);
    }
}

fn start_audio_playback(
    audio_path: &Path,
    start_ms: u64,
    end_ms: u64,
) -> io::Result<SpawnedPlayback> {
    let mut segment: Box<dyn Read + Send> = if archive::is_archived(audio_path) {
        let stored_bytes = archive::sample_count(audio_path)?.saturating_mul(4);
        let (offset, byte_count) = audio_byte_range(start_ms, end_ms, stored_bytes)?;
        let samples = archive::read_range(audio_path, offset / 4, (offset + byte_count) / 4)?;
        Box::new(io::Cursor::new(
            samples
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<u8>>(),
        ))
    } else {
        let mut audio = fs::File::open(audio_path)?;
        let (offset, byte_count) = audio_byte_range(start_ms, end_ms, audio.metadata()?.len())?;
        audio.seek(SeekFrom::Start(offset))?;
        Box::new(audio.take(byte_count))
    };

    let executable = pw_play_executable();
    let mut child = pw_play_command(&executable)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("could not start {}: {error}", executable.display()),
            )
        })?;
    let Some(mut stdin) = child.stdin.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "PipeWire playback did not provide an audio input",
        ));
    };
    let Some(mut child_stderr) = child.stderr.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::other(
            "PipeWire playback did not provide diagnostic output",
        ));
    };
    let stderr = Arc::new(Mutex::new(String::new()));
    let stderr_for_thread = stderr.clone();
    std::thread::spawn(move || {
        let mut buffer = [0u8; 1_024];
        while let Ok(count) = child_stderr.read(&mut buffer) {
            if count == 0 {
                break;
            }
            if let Ok(mut output) = stderr_for_thread.lock()
                && output.len() < 16_384
            {
                let remaining = 16_384 - output.len();
                output.push_str(&String::from_utf8_lossy(&buffer[..count.min(remaining)]));
            }
        }
    });
    let writer_error = Arc::new(Mutex::new(None));
    let writer_error_for_thread = writer_error.clone();
    std::thread::spawn(move || {
        let result = io::copy(&mut segment, &mut stdin)
            .and_then(|_| stdin.flush())
            .map_err(|error| error.to_string());
        if let Err(error) = result
            && let Ok(mut slot) = writer_error_for_thread.lock()
        {
            *slot = Some(error);
        }
    });
    Ok(SpawnedPlayback {
        child,
        writer_error,
        stderr,
    })
}

fn pw_play_command(executable: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new(executable);
    command.args([
        "--playback",
        &format!("--rate={SAMPLE_RATE}"),
        "--channels=1",
        "--channel-map=mono",
        "--format=f32",
        "--media-role=Communication",
    ]);
    command
}

fn pw_play_executable() -> PathBuf {
    if let Some(executable) = std::env::var_os("SINGSTONE_PW_PLAY") {
        return PathBuf::from(executable);
    }
    if let Some(snap) = std::env::var_os("SNAP") {
        let bundled = PathBuf::from(snap).join("usr/bin/pw-play");
        if bundled.is_file() {
            return bundled;
        }
    }
    PathBuf::from("pw-play")
}

fn audio_byte_range(start_ms: u64, end_ms: u64, file_len: u64) -> io::Result<(u64, u64)> {
    if end_ms <= start_ms {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "audio segment has no duration",
        ));
    }
    const BYTES_PER_SAMPLE: u128 = size_of::<f32>() as u128;
    let sample_rate = u128::from(SAMPLE_RATE);
    let start = u128::from(start_ms)
        .saturating_mul(sample_rate)
        .saturating_mul(BYTES_PER_SAMPLE)
        / 1_000;
    let end = u128::from(end_ms)
        .saturating_mul(sample_rate)
        .saturating_add(999)
        / 1_000
        * BYTES_PER_SAMPLE;
    let start = u64::try_from(start).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "audio segment offset is too large",
        )
    })?;
    let end = u64::try_from(end).unwrap_or(u64::MAX);
    let aligned_file_len = file_len - file_len % size_of::<f32>() as u64;
    let end = end.min(aligned_file_len);
    if start >= end {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "audio segment is outside the recorded audio",
        ));
    }
    Ok((start, end - start))
}

fn set_playback_button(button: &gtk::Button, playing: bool) {
    let (icon, label) = if playing {
        ("media-playback-stop-symbolic", "Stop playback")
    } else {
        ("media-playback-start-symbolic", "Play this audio segment")
    };
    button.set_icon_name(icon);
    button.set_tooltip_text(Some(label));
    button.update_property(&[gtk::accessible::Property::Label(label)]);
}

impl SessionDetail {
    fn load(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.playback.stop();
        let session = Session::open(path)?;
        let manifest = session.read_manifest()?;
        *self.selected.borrow_mut() = Some(path.to_owned());
        let title = session_display_title(&session, path);
        self.title.set_label(&title);
        let duration = session_duration_ms(&session, &manifest);
        self.subtitle.set_label(&format!(
            "{} · {}",
            display_started(&manifest.started_wallclock),
            format_duration(duration)
        ));
        let processed = session.transcript_path().is_file();
        let queue_state = self.queue.borrow().state(path);
        self.apply_queue_state(&session, &manifest, processed, queue_state);
        let hidden = process::read_hidden_sources(&session).unwrap_or_default();
        let mic_hidden = hidden.contains(&AudioSource::Mic);
        let system_hidden = hidden.contains(&AudioSource::System);
        self.loading_hidden_sources.set(true);
        self.hide_mic.set_active(mic_hidden);
        self.hide_system.set_active(system_hidden);
        self.loading_hidden_sources.set(false);
        let can_hide_sources = processed && manifest.state != SessionState::Recording;
        self.hide_mic
            .set_visible(can_hide_sources && manifest.mic.enabled);
        self.hide_system
            .set_visible(can_hide_sources && manifest.system.enabled);

        clear_box(&self.transcript);
        *self.empty_state.borrow_mut() = None;
        clear_box(&self.transcript_panes);
        self.transcript_panes.set_visible(false);
        if processed {
            let utterances: Vec<Utterance> = jsonl::read_all(&session.transcript_path())?;
            if utterances.is_empty() {
                append_empty(
                    &self.transcript,
                    if mic_hidden || system_hidden {
                        "Every line of this transcript is hidden."
                    } else {
                        "The transcript is empty."
                    },
                );
            } else {
                let frequent = Rc::new(frequent_speakers(&utterances, 3));
                let learned =
                    process::learned_lines(&session, &speakers_database_path(), &utterances);
                let has_mic = utterances
                    .iter()
                    .any(|utterance| utterance.source == AudioSource::Mic);
                let has_system = utterances
                    .iter()
                    .any(|utterance| utterance.source == AudioSource::System);
                let one_sided = !(has_mic && has_system);
                populate_timeline_headers(&self.transcript_panes, has_mic, has_system, one_sided);
                self.transcript_panes.set_visible(true);
                for row in timeline_rows(&utterances) {
                    self.transcript.append(&transcript_timeline_row(
                        &row,
                        &utterances,
                        &learned,
                        self,
                        &session,
                        &frequent,
                        one_sided,
                    ));
                }
            }
        } else {
            let empty = append_empty(
                &self.transcript,
                if matches!(queue_state, Some(JobState::Waiting { .. })) {
                    "Waiting in the processing queue."
                } else {
                    "This recording has not been processed yet."
                },
            );
            *self.empty_state.borrow_mut() = Some(empty);
        }

        clear_flow(&self.screenshots);
        let screenshots: Vec<ScreenshotEntry> =
            jsonl::read_all(&session.screenshots_index_path()).unwrap_or_default();
        if screenshots.is_empty() {
            let empty = gtk::Label::new(Some("No screenshots"));
            empty.add_css_class("dim-label");
            self.screenshots.insert(&empty, -1);
        } else {
            for screenshot in screenshots {
                if let Some(file) = safe_session_path(&session.dir, &screenshot.file) {
                    self.screenshots
                        .insert(&screenshot_card(&file, screenshot.time_ms), -1);
                }
            }
        }

        clear_box(&self.metadata);
        self.metadata
            .append(&property_row("Folder", &session.dir.display().to_string()));
        self.metadata
            .append(&property_row("Duration", &format_duration(duration)));
        let audio = match (manifest.mic.enabled, manifest.system.enabled) {
            (true, true) => "Microphone + system",
            (true, false) => "Microphone",
            (false, true) => "System",
            (false, false) => "None",
        };
        self.metadata.append(&property_row("Audio", audio));
        let stored = [AudioSource::Mic, AudioSource::System]
            .map(|source| session.stored_audio_path(source))
            .into_iter()
            .filter_map(|path| Some((fs::metadata(&path).ok()?.len(), path)))
            .collect::<Vec<_>>();
        if !stored.is_empty() {
            let raw = stored.iter().any(|(_, path)| !archive::is_archived(path));
            self.metadata.append(&property_row(
                "Audio storage",
                &format!(
                    "{} · {}",
                    archive::format_size(stored.iter().map(|(size, _)| size).sum()),
                    if raw { "raw float" } else { "FLAC" }
                ),
            ));
        }

        clear_box(&self.files);
        let outputs = gtk::Label::new(Some("Outputs:"));
        outputs.add_css_class("dim-label");
        self.files.append(&outputs);
        for filename in ["transcript.jsonl", "transcript.txt", "screenshots.jsonl"] {
            let file = session.dir.join(filename);
            if !file.is_file() {
                continue;
            }
            let button = gtk::Button::with_label(filename);
            button.add_css_class("flat");
            button.add_css_class("mono-button");
            let file_for_click = file.clone();
            button.connect_clicked(move |_| open_path(&file_for_click));
            self.files.append(&button);
        }
        let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        spacer.set_hexpand(true);
        self.files.append(&spacer);
        let open = gtk::Button::with_label("Open folder");
        open.add_css_class("flat");
        let folder = session.dir.clone();
        open.connect_clicked(move |_| open_path(&folder));
        self.files.append(&open);
        Ok(())
    }

    fn apply_queue_state(
        &self,
        session: &Session,
        manifest: &Manifest,
        processed: bool,
        queue_state: Option<JobState>,
    ) {
        let recording = manifest.state == SessionState::Recording;
        self.status.set_visible(true);
        self.process_button.set_visible(!recording);
        self.process_button.remove_css_class("suggested-action");
        match queue_state {
            Some(JobState::Running) => {
                set_status(&self.status, "Processing", "busy");
                self.process_button
                    .set_label(if self.cancel_requested.get() {
                        "Cancelling…"
                    } else {
                        "Cancel processing"
                    });
                self.process_button
                    .set_sensitive(!self.cancel_requested.get());
                self.process_button
                    .set_tooltip_text(Some("Stop processing this recording"));
                self.body_stack.set_visible_child_name("processing");
            }
            Some(JobState::Waiting { position }) => {
                set_status(
                    &self.status,
                    &format!("Queued · {}", ordinal(position)),
                    "queued",
                );
                self.process_button.set_label("Remove from queue");
                self.process_button.set_sensitive(true);
                self.process_button
                    .set_tooltip_text(Some("Remove this session from the processing queue"));
                self.body_stack.set_visible_child_name("transcript");
            }
            None => {
                let (status, class) = session_path_status(&session.dir, &self.queue.borrow());
                set_status(&self.status, status, class);
                self.process_button
                    .set_label(if processed { "Reprocess" } else { "Process" });
                self.process_button.set_sensitive(true);
                self.process_button.add_css_class("suggested-action");
                self.process_button.set_tooltip_text(Some(if processed {
                    "Replace the transcript, intermediate files, and automatic speaker matches. Confirmed names are kept."
                } else {
                    "Process this recording"
                }));
                self.body_stack.set_visible_child_name("transcript");
            }
        }

        let running = matches!(queue_state, Some(JobState::Running));
        let waiting = matches!(queue_state, Some(JobState::Waiting { .. }));
        let archived = session.is_archived();
        let done = session.is_done();
        self.done_button.set_visible(
            processed
                && !recording
                && !archived
                && queue_state.is_none()
                && !self.queue.borrow().has_failed(&session.dir),
        );
        self.done_button
            .set_label(if done { "Unmark done" } else { "Mark done" });
        self.done_button.set_tooltip_text(Some(if done {
            "Move this session back to Processed"
        } else {
            "Flag this session as finished"
        }));
        self.archive_button
            .set_visible(processed && !recording && !archived);
        self.archive_button.set_sensitive(!running);
        self.meeting_button
            .set_visible((processed || waiting || running) && !recording);
        self.meeting_button.set_sensitive(!running);
        self.rename_button.set_visible(true);
        self.rename_button.set_sensitive(!running);
        self.delete_button.set_visible(true);
        self.delete_button.set_sensitive(!running && !waiting);
        self.delete_button.set_tooltip_text(Some(if waiting {
            "Remove the session from the queue before deleting it"
        } else {
            "Delete session"
        }));
        self.hide_mic.set_sensitive(!running);
        self.hide_system.set_sensitive(!running);
    }

    fn refresh_queue_state(&self) {
        let Some(path) = self.selected.borrow().clone() else {
            return;
        };
        let (session, manifest) = match Session::open(&path)
            .and_then(|session| session.read_manifest().map(|manifest| (session, manifest)))
        {
            Ok(value) => value,
            Err(_) => {
                self.clear();
                return;
            }
        };
        let processed = session.transcript_path().is_file();
        let queue_state = self.queue.borrow().state(&path);
        self.apply_queue_state(&session, &manifest, processed, queue_state);
        if let Some(empty) = self.empty_state.borrow().as_ref() {
            empty.set_label(if matches!(queue_state, Some(JobState::Waiting { .. })) {
                "Waiting in the processing queue."
            } else {
                "This recording has not been processed yet."
            });
        }
    }

    /// Back to the empty state shown before any session is selected.
    fn clear(&self) {
        self.playback.stop();
        *self.selected.borrow_mut() = None;
        self.title.set_label("Select a session");
        self.subtitle
            .set_label("Recorded meetings appear in the sidebar.");
        self.body_stack.set_visible_child_name("transcript");
        for widget in [
            self.status.upcast_ref::<gtk::Widget>(),
            self.process_button.upcast_ref(),
            self.done_button.upcast_ref(),
            self.archive_button.upcast_ref(),
            self.meeting_button.upcast_ref(),
            self.rename_button.upcast_ref(),
            self.delete_button.upcast_ref(),
            self.hide_mic.upcast_ref(),
            self.hide_system.upcast_ref(),
        ] {
            widget.set_visible(false);
        }
        clear_box(&self.transcript);
        *self.empty_state.borrow_mut() = None;
        clear_box(&self.transcript_panes);
        self.transcript_panes.set_visible(false);
        clear_flow(&self.screenshots);
        clear_box(&self.metadata);
        clear_box(&self.files);
    }
}

fn populate_sessions(
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    root: &Path,
    filter: &str,
    queue: &Rc<RefCell<ProcessingQueue>>,
) {
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
    let needle = filter.to_lowercase();
    let summaries = load_sessions(root, &queue.borrow())
        .into_iter()
        .filter(|item| item.title.to_lowercase().contains(&needle))
        .collect::<Vec<_>>();
    *paths.borrow_mut() = summaries.iter().map(|item| item.path.clone()).collect();
    for summary in summaries {
        list.append(&session_row(&summary));
    }
    if paths.borrow().is_empty() {
        let empty = gtk::Label::new(Some("No sessions yet"));
        empty.set_margin_top(24);
        empty.add_css_class("dim-label");
        list.append(&empty);
    }
}

fn load_sessions(root: &Path, queue: &ProcessingQueue) -> Vec<SessionSummary> {
    let mut sessions = fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let session = Session::open(entry.path()).ok()?;
            let manifest = session.read_manifest().ok();
            let (status, status_class) = session_path_status(&entry.path(), queue);
            Some(SessionSummary {
                title: session_display_title(&session, &entry.path()),
                started: manifest
                    .as_ref()
                    .map(|manifest| manifest.started_wallclock.clone())
                    .unwrap_or_else(|| session_title(&entry.path())),
                duration_ms: manifest
                    .as_ref()
                    .map(|manifest| session_duration_ms(&session, manifest))
                    .unwrap_or_default(),
                path: entry.path(),
                status,
                status_class,
            })
        })
        .collect::<Vec<_>>();
    sessions.sort_by(|a, b| b.started.cmp(&a.started));
    sessions
}

fn session_display_title(session: &Session, path: &Path) -> String {
    meeting::read_details(&session.meeting_path())
        .ok()
        .flatten()
        .map(|details| details.title)
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| session_title(path))
}

fn session_status(session: &Session, manifest: &Manifest) -> (&'static str, &'static str) {
    if manifest.state == SessionState::Recording {
        ("Recording", "busy")
    } else if !session.transcript_path().is_file() {
        ("Recorded", "idle")
    } else if session.is_archived() {
        ("Archived", "archived")
    } else if session.is_done() {
        ("Done", "done")
    } else {
        ("Processed", "ok")
    }
}

fn session_path_status(path: &Path, queue: &ProcessingQueue) -> (&'static str, &'static str) {
    match queue.state(path) {
        Some(JobState::Running) => ("Processing", "busy"),
        Some(JobState::Waiting { .. }) => ("Queued", "queued"),
        None if queue.has_failed(path) => ("Failed", "failed"),
        None => Session::open(path)
            .and_then(|session| {
                session
                    .read_manifest()
                    .map(|manifest| session_status(&session, &manifest))
            })
            .unwrap_or(("Unavailable", "queued")),
    }
}

fn session_row(summary: &SessionSummary) -> gtk::ListBoxRow {
    let row = gtk::ListBoxRow::new();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 4);
    body.set_margin_top(10);
    body.set_margin_bottom(10);
    body.set_margin_start(12);
    body.set_margin_end(12);
    let top = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let title = gtk::Label::new(Some(&summary.title));
    title.set_xalign(0.0);
    title.set_hexpand(true);
    title.set_ellipsize(gtk::pango::EllipsizeMode::End);
    title.add_css_class("heading");
    top.append(&title);
    top.append(&status_pill(summary.status, summary.status_class));
    body.append(&top);
    let meta = gtk::Label::new(Some(&format!(
        "{} · {}",
        display_started(&summary.started),
        format_duration(summary.duration_ms)
    )));
    meta.set_xalign(0.0);
    meta.add_css_class("dim-label");
    meta.add_css_class("caption");
    body.append(&meta);
    row.set_child(Some(&body));
    row
}

fn populate_timeline_headers(headers: &gtk::Box, has_mic: bool, has_system: bool, one_sided: bool) {
    headers.set_layout_manager(Some(TimelineLayout::new(one_sided)));

    let left = timeline_pane_header(
        "Microphone",
        "audio-input-microphone-symbolic",
        if one_sided {
            gtk::Align::Start
        } else {
            gtk::Align::End
        },
    );
    left.set_visible(has_mic);
    headers.append(&left);

    let marker = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    marker.add_css_class("timeline-marker");
    headers.append(&marker);

    let right = timeline_pane_header("System audio", "audio-speakers-symbolic", gtk::Align::Start);
    right.set_visible(has_system);
    headers.append(&right);
}

fn timeline_pane_header(text: &str, icon_name: &str, align: gtk::Align) -> gtk::Box {
    let cell = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    cell.add_css_class("timeline-pane-header");
    let content = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    content.set_valign(gtk::Align::Center);
    let icon = gtk::Image::from_icon_name(icon_name);
    icon.set_pixel_size(15);
    icon.add_css_class("dim-label");
    content.append(&icon);
    let label = gtk::Label::new(Some(text));
    label.add_css_class("heading");
    label.add_css_class("caption");
    label.add_css_class("dim-label");
    content.append(&label);
    if align == gtk::Align::End {
        let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        spacer.set_hexpand(true);
        cell.append(&spacer);
    }
    cell.append(&content);
    cell
}

fn transcript_timeline_row(
    timeline_row: &TimelineRow,
    utterances: &[Utterance],
    learned: &[bool],
    detail: &SessionDetail,
    session: &Session,
    frequent: &Rc<Vec<String>>,
    one_sided: bool,
) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    row.set_layout_manager(Some(TimelineLayout::new(one_sided)));

    let mic = gtk::Box::new(gtk::Orientation::Vertical, 6);
    mic.add_css_class("timeline-cell");
    mic.set_visible(!one_sided || !timeline_row.mic.is_empty());
    for &index in &timeline_row.mic {
        let utterance = &utterances[index];
        let card = transcript_card(
            utterance,
            learned[index],
            detail,
            session.stored_audio_path(utterance.source),
            frequent,
            !one_sided,
            format_timeline_timestamp(utterance.start_ms)
                != format_timeline_timestamp(timeline_row.start_ms),
        );
        card.set_halign(if one_sided {
            gtk::Align::Start
        } else {
            gtk::Align::End
        });
        mic.append(&card);
    }
    row.append(&mic);

    row.append(&timeline_marker(timeline_row.start_ms));

    let system = gtk::Box::new(gtk::Orientation::Vertical, 6);
    system.add_css_class("timeline-cell");
    system.set_visible(!one_sided || !timeline_row.system.is_empty());
    for &index in &timeline_row.system {
        let utterance = &utterances[index];
        let card = transcript_card(
            utterance,
            learned[index],
            detail,
            session.stored_audio_path(utterance.source),
            frequent,
            false,
            format_timeline_timestamp(utterance.start_ms)
                != format_timeline_timestamp(timeline_row.start_ms),
        );
        card.set_halign(gtk::Align::Start);
        system.append(&card);
    }
    row.append(&system);
    row
}

fn timeline_marker(start_ms: u64) -> gtk::Overlay {
    let marker = gtk::Overlay::new();
    marker.add_css_class("timeline-marker");
    let line = gtk::Separator::new(gtk::Orientation::Vertical);
    line.set_halign(gtk::Align::Center);
    line.set_vexpand(true);
    marker.set_child(Some(&line));
    let timestamp = gtk::Label::new(Some(&format_timeline_timestamp(start_ms)));
    timestamp.add_css_class("timeline-time");
    timestamp.add_css_class("caption");
    timestamp.set_width_chars(5);
    timestamp.set_halign(gtk::Align::Center);
    timestamp.set_valign(gtk::Align::Start);
    timestamp.set_margin_top(8);
    marker.add_overlay(&timestamp);
    marker
}

fn transcript_card(
    utterance: &Utterance,
    learned: bool,
    detail: &SessionDetail,
    audio_path: PathBuf,
    frequent: &Rc<Vec<String>>,
    mirrored: bool,
    show_timestamp: bool,
) -> gtk::Box {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 5);
    card.add_css_class("transcript-card");
    if utterance.echo {
        card.add_css_class("echo-row");
    }
    let avatar = gtk::Image::from_icon_name("avatar-default-symbolic");
    avatar.set_pixel_size(18);
    avatar.add_css_class("speaker-avatar");
    if is_anonymous_speaker(utterance) {
        avatar.add_css_class("speaker-unknown");
    } else if utterance.source == AudioSource::Mic {
        avatar.add_css_class("speaker-2");
    } else {
        let speaker_number = utterance
            .speaker_id
            .bytes()
            .fold(0usize, |total, value| total.wrapping_add(value as usize))
            % 3;
        avatar.add_css_class(&format!("speaker-{speaker_number}"));
    }
    avatar.set_valign(gtk::Align::Start);
    let head = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    head.set_layout_manager(Some(if mirrored {
        WrapLayout::new_end_aligned(6, 4)
    } else {
        WrapLayout::new(6, 4)
    }));
    head.set_hexpand(true);
    let identity = gtk::Box::new(gtk::Orientation::Horizontal, 4);
    let speaker = gtk::Label::new(Some(&utterance.speaker));
    speaker.add_css_class("heading");
    speaker.set_xalign(if mirrored { 1.0 } else { 0.0 });
    speaker.set_single_line_mode(true);
    if utterance.speaker.chars().count() > 6 {
        speaker.set_ellipsize(gtk::pango::EllipsizeMode::End);
        speaker.set_width_chars(6);
    }
    if mirrored {
        identity.append(&speaker);
        identity.append(&avatar);
    } else {
        identity.append(&avatar);
        identity.append(&speaker);
    }
    if is_anonymous_speaker(utterance) {
        speaker.add_css_class("warning-text");
    }
    head.append(&identity);
    if utterance.locked {
        let lock = gtk::Image::from_icon_name("changes-prevent-symbolic");
        lock.set_pixel_size(14);
        lock.set_tooltip_text(Some("Confirmed by you"));
        lock.update_property(&[gtk::accessible::Property::Label("Confirmed by you")]);
        lock.add_css_class("dim-label");
        lock.set_valign(gtk::Align::Center);
        head.append(&lock);
    }
    if learned {
        let voice = gtk::Image::from_icon_name("auth-fingerprint-symbolic");
        voice.set_pixel_size(14);
        voice.set_tooltip_text(Some("Voice learned from this line"));
        voice.update_property(&[gtk::accessible::Property::Label(
            "Voice learned from this line",
        )]);
        voice.add_css_class("dim-label");
        voice.set_valign(gtk::Align::Center);
        head.append(&voice);
    }
    if show_timestamp {
        let timestamp = gtk::Label::new(Some(&format_timeline_timestamp(utterance.start_ms)));
        timestamp.add_css_class("dim-label");
        timestamp.add_css_class("caption");
        head.append(&timestamp);
    }
    if utterance.echo {
        let echo = status_pill("Echo", "idle");
        echo.add_css_class("caption");
        echo.set_tooltip_text(Some("Loudspeaker sound picked up by the microphone"));
        head.append(&echo);
    }
    let play = gtk::Button::new();
    play.add_css_class("flat");
    play.add_css_class("transcript-icon");
    play.set_valign(gtk::Align::Center);
    set_playback_button(&play, false);
    let playback = detail.playback.clone();
    let row_id = playback.allocate_id();
    let window = detail.window.clone();
    let start_ms = utterance.start_ms;
    let end_ms = utterance.end_ms;
    play.connect_clicked(move |button| {
        playback.toggle(row_id, button, &window, &audio_path, start_ms, end_ms);
    });
    head.append(&play);
    if has_assignable_id(utterance) {
        let anonymous = is_anonymous_speaker(utterance);
        let edit = gtk::Button::from_icon_name("document-edit-symbolic");
        edit.add_css_class("flat");
        edit.add_css_class("transcript-icon");
        let edit_label = if anonymous {
            "Assign speaker…"
        } else {
            "Change speaker…"
        };
        edit.set_tooltip_text(Some(edit_label));
        edit.update_property(&[gtk::accessible::Property::Label(edit_label)]);
        if anonymous {
            let detail = detail.clone();
            let source = utterance.source;
            let start_ms = utterance.start_ms;
            let end_ms = utterance.end_ms;
            let frequent = frequent.clone();
            edit.connect_clicked(move |_| {
                show_assignment_dialog(&detail, source, start_ms, end_ms, None, &frequent);
            });
        } else {
            let detail = detail.clone();
            let source = utterance.source;
            let start_ms = utterance.start_ms;
            let end_ms = utterance.end_ms;
            let current = utterance.speaker.clone();
            let frequent = frequent.clone();
            edit.connect_clicked(move |_| {
                show_assignment_dialog(
                    &detail,
                    source,
                    start_ms,
                    end_ms,
                    Some(&current),
                    &frequent,
                );
            });
        }
        head.append(&edit);
    }
    if has_assignable_id(utterance) && is_anonymous_speaker(utterance) && !utterance.echo {
        for name in frequent.iter() {
            let shortcut = shortcut_button(name);
            shortcut.set_tooltip_text(Some(&format!("Assign this voice to {name}")));
            let detail = detail.clone();
            let source = utterance.source;
            let start_ms = utterance.start_ms;
            let end_ms = utterance.end_ms;
            let name = name.clone();
            shortcut.connect_clicked(move |_| {
                start_assignment(&detail, source, start_ms, end_ms, name.clone());
            });
            head.append(&shortcut);
        }
    }
    card.append(&head);
    let text = gtk::Label::new(Some(&utterance.text));
    text.set_xalign(0.0);
    text.set_wrap(true);
    text.set_wrap_mode(gtk::pango::WrapMode::WordChar);
    text.set_selectable(true);
    card.append(&text);
    card
}

fn caption_button(text: &str) -> gtk::Button {
    let label = gtk::Label::new(Some(text));
    label.set_ellipsize(gtk::pango::EllipsizeMode::End);
    label.set_max_width_chars(14);
    let button = gtk::Button::new();
    button.set_child(Some(&label));
    button.add_css_class("flat");
    button.add_css_class("caption");
    button
}

fn shortcut_button(name: &str) -> gtk::Button {
    let button = caption_button(name);
    button.add_css_class("speaker-shortcut");
    button
}

fn is_anonymous_speaker(utterance: &Utterance) -> bool {
    utterance.speaker.starts_with("SPEAKER_") || utterance.speaker == "unknown"
}

fn has_assignable_id(utterance: &Utterance) -> bool {
    parse_assignable_id(&utterance.speaker_id).is_some()
}

fn parse_assignable_id(value: &str) -> Option<(AudioSource, u32)> {
    if let Some(cluster) = value.strip_prefix("speaker-") {
        return Some((AudioSource::System, cluster.parse().ok()?));
    }
    let (prefix, cluster) = value.rsplit_once('_')?;
    let source = match prefix {
        "mic" => AudioSource::Mic,
        "spk" => AudioSource::System,
        _ => return None,
    };
    Some((source, cluster.parse().ok()?))
}

fn frequent_speakers(utterances: &[Utterance], limit: usize) -> Vec<String> {
    let mut durations = BTreeMap::new();
    for utterance in utterances {
        if is_anonymous_speaker(utterance) || utterance.speaker_id == "local" {
            continue;
        }
        let duration = utterance.end_ms.saturating_sub(utterance.start_ms);
        let total = durations.entry(utterance.speaker.clone()).or_insert(0u64);
        *total = total.saturating_add(duration);
    }
    let mut ranked = durations.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked
        .into_iter()
        .take(limit)
        .map(|(name, _)| name)
        .collect()
}

fn show_assignment_dialog(
    detail: &SessionDetail,
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
    current: Option<&str>,
    frequent: &[String],
) {
    let (heading, body) = if let Some(current) = current {
        (
            "Change this speaker",
            format!(
                "Currently {current}. The new name is locked on this line. Later lines that Singstone recognizes as the same voice get the name too, without a lock. Earlier lines stay as they are. When the local voice model is available, it also learns the voice under the new name."
            ),
        )
    } else {
        (
            "Who is this speaker?",
            "The name is locked on this line. Later lines that Singstone recognizes as the same voice get the name too, without a lock. Earlier lines stay as they are. When the local voice model is available, it also learns this voice for future meetings.".to_owned(),
        )
    };
    let dialog = adw::AlertDialog::new(Some(heading), Some(&body));
    let content = gtk::Box::new(gtk::Orientation::Vertical, 8);
    let entry = gtk::Entry::builder()
        .placeholder_text("Speaker name")
        .activates_default(true)
        .build();
    let meeting_names = frequent
        .iter()
        .filter(|name| current != Some(name.as_str()))
        .collect::<Vec<_>>();
    if !meeting_names.is_empty() {
        let meeting = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        meeting.set_layout_manager(Some(WrapLayout::new(8, 4)));
        let label = gtk::Label::new(Some("In this meeting:"));
        label.add_css_class("dim-label");
        label.add_css_class("caption");
        meeting.append(&label);
        for name in meeting_names {
            let shortcut = shortcut_button(name);
            shortcut.set_tooltip_text(Some(&format!("Use {name}")));
            let entry = entry.clone();
            let name = name.clone();
            shortcut.connect_clicked(move |_| entry.set_text(&name));
            meeting.append(&shortcut);
        }
        content.append(&meeting);
    }
    content.append(&entry);

    let suggestions = gtk::ListBox::new();
    suggestions.add_css_class("boxed-list");
    suggestions.set_activate_on_single_click(true);
    suggestions.set_selection_mode(gtk::SelectionMode::Single);
    let suggestion_names = assignment_suggestion_names(detail);
    update_assignment_suggestions(&suggestions, &suggestion_names, "");
    content.append(&suggestions);

    dialog.set_extra_child(Some(&content));
    let response = if current.is_some() {
        "Reassign"
    } else {
        "Assign"
    };
    dialog.add_responses(&[("cancel", "Cancel"), ("assign", response)]);
    dialog.set_default_response(Some("assign"));
    dialog.set_close_response("cancel");
    dialog.set_response_appearance("assign", adw::ResponseAppearance::Suggested);
    dialog.set_response_enabled("assign", false);

    let suggestions_for_change = suggestions.clone();
    let suggestion_names_for_change = suggestion_names.clone();
    let dialog_for_change = dialog.downgrade();
    entry.connect_changed(move |entry| {
        suggestions_for_change.unselect_all();
        update_assignment_suggestions(
            &suggestions_for_change,
            &suggestion_names_for_change,
            &entry.text(),
        );
        if let Some(dialog) = dialog_for_change.upgrade() {
            dialog.set_response_enabled("assign", !entry.text().trim().is_empty());
        }
    });

    let entry_keys = gtk::EventControllerKey::new();
    let suggestions_for_entry_keys = suggestions.clone();
    entry_keys.connect_key_pressed(move |_, key, _, _| {
        if key == gdk::Key::Down
            && let Some(row) = suggestions_for_entry_keys.row_at_index(0)
        {
            suggestions_for_entry_keys.select_row(Some(&row));
            row.grab_focus();
            return glib::Propagation::Stop;
        }
        glib::Propagation::Proceed
    });
    entry.add_controller(entry_keys);

    let list_keys = gtk::EventControllerKey::new();
    let suggestions_for_keys = suggestions.clone();
    let entry_for_keys = entry.clone();
    let dialog_for_keys = dialog.downgrade();
    let detail_for_keys = detail.clone();
    let current_for_keys = current.map(str::to_owned);
    list_keys.connect_key_pressed(move |_, key, _, _| {
        let selected = suggestions_for_keys.selected_row();
        if key == gdk::Key::Up {
            if selected.as_ref().is_some_and(|row| row.index() == 0) {
                suggestions_for_keys.unselect_all();
                entry_for_keys.grab_focus();
            } else if let Some(row) = selected
                && let Some(previous) = suggestions_for_keys.row_at_index(row.index() - 1)
            {
                suggestions_for_keys.select_row(Some(&previous));
                previous.grab_focus();
            }
            return glib::Propagation::Stop;
        }
        if key == gdk::Key::Down {
            if let Some(row) = selected
                && let Some(next) = suggestions_for_keys.row_at_index(row.index() + 1)
            {
                suggestions_for_keys.select_row(Some(&next));
                next.grab_focus();
            }
            return glib::Propagation::Stop;
        }
        if matches!(key, gdk::Key::Return | gdk::Key::KP_Enter)
            && let Some(row) = selected
            && let Some(name) = assignment_suggestion_name(&row)
        {
            entry_for_keys.set_text(&name);
            if let Some(dialog) = dialog_for_keys.upgrade() {
                submit_assignment(
                    &dialog,
                    &detail_for_keys,
                    source,
                    start_ms,
                    end_ms,
                    current_for_keys.as_deref(),
                    name,
                );
            }
            return glib::Propagation::Stop;
        }
        glib::Propagation::Proceed
    });
    suggestions.add_controller(list_keys);

    let entry_for_row = entry.clone();
    let dialog_for_row = dialog.downgrade();
    let detail_for_row = detail.clone();
    let current_for_row = current.map(str::to_owned);
    suggestions.connect_row_activated(move |_, row| {
        let Some(name) = assignment_suggestion_name(row) else {
            return;
        };
        entry_for_row.set_text(&name);
        if let Some(dialog) = dialog_for_row.upgrade() {
            submit_assignment(
                &dialog,
                &detail_for_row,
                source,
                start_ms,
                end_ms,
                current_for_row.as_deref(),
                name,
            );
        }
    });

    let detail_for_response = detail.clone();
    let current = current.map(str::to_owned);
    let entry_for_response = entry.clone();
    dialog.connect_response(Some("assign"), move |dialog, _| {
        submit_assignment(
            dialog,
            &detail_for_response,
            source,
            start_ms,
            end_ms,
            current.as_deref(),
            entry_for_response.text().to_string(),
        );
    });
    dialog.present(Some(&detail.window));
    entry.grab_focus();
}

fn speakers_database_path() -> PathBuf {
    std::env::var_os("SINGSTONE_SPEAKERS_DB")
        .map(PathBuf::from)
        .unwrap_or_else(database::default_path)
}

fn assignment_suggestion_names(detail: &SessionDetail) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = BTreeMap::new();
    let database_path = speakers_database_path();
    if let Ok(database) = SpeakerDatabase::load(&database_path) {
        for name in database.speakers.into_keys() {
            if seen.insert(canonical_name_key(&name), ()).is_none() {
                names.push(name);
            }
        }
    }
    if let Some(session_dir) = detail.selected.borrow().as_ref()
        && let Ok(session) = Session::open(session_dir)
        && let Ok(utterances) = jsonl::read_all::<Utterance>(&session.transcript_path())
    {
        for utterance in utterances {
            if is_anonymous_speaker(&utterance) {
                continue;
            }
            let name = utterance.speaker;
            if seen.insert(canonical_name_key(&name), ()).is_none() {
                names.push(name);
            }
        }
    }
    names
}

fn update_assignment_suggestions(list: &gtk::ListBox, names: &[String], typed: &str) {
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
    let typed_key = canonical_name_key(typed);
    let mut matches = names
        .iter()
        .filter_map(|name| {
            let key = canonical_name_key(name);
            let rank = if key == typed_key {
                0
            } else if key.starts_with(&typed_key) {
                1
            } else if key.contains(&typed_key) {
                2
            } else {
                return None;
            };
            Some((rank, name))
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|(rank, _)| *rank);
    for (_, name) in matches.into_iter().take(8) {
        let label = gtk::Label::new(Some(name));
        label.set_xalign(0.0);
        label.set_margin_top(6);
        label.set_margin_bottom(6);
        label.set_margin_start(8);
        label.set_margin_end(8);
        list.append(&label);
    }
    list.set_visible(list.first_child().is_some());
}

fn assignment_suggestion_name(row: &gtk::ListBoxRow) -> Option<String> {
    row.child()?
        .downcast::<gtk::Label>()
        .ok()
        .map(|label| label.text().to_string())
}

fn submit_assignment(
    dialog: &adw::AlertDialog,
    detail: &SessionDetail,
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
    current: Option<&str>,
    name: String,
) {
    if name.trim().is_empty() {
        show_error(
            &detail.window,
            "Speaker name required",
            "Enter or choose a name for this voice.",
        );
        return;
    }
    if current.is_some_and(|current| canonical_name_key(current) == canonical_name_key(&name)) {
        show_error(
            &detail.window,
            "Same name",
            "Choose a different name to reassign this voice.",
        );
        return;
    }
    dialog.close();
    start_assignment(detail, source, start_ms, end_ms, name);
}

fn start_assignment(
    detail: &SessionDetail,
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
    name: String,
) {
    if detail.busy.get() {
        return;
    }
    let Some(session) = detail.selected.borrow().clone() else {
        return;
    };
    detail.busy.set(true);
    let progress = progress_window(
        &detail.window,
        "Assigning speaker",
        "Updating transcript and learning the voice locally…",
    );

    let result = Arc::new(Mutex::new(None));
    let thread_result = result.clone();
    std::thread::spawn(move || {
        let value = process::correct_speaker_forward(
            ProcessArgs::for_session(session),
            &process::CorrectionRequest {
                source,
                start_ms,
                end_ms,
                speaker: name,
            },
        )
        .map_err(|error| error.to_string());
        *thread_result
            .lock()
            .expect("speaker assignment result mutex") = Some(value);
    });
    let detail = detail.clone();
    glib::timeout_add_local(Duration::from_millis(150), move || {
        let Some(result) = result
            .lock()
            .expect("speaker assignment result mutex")
            .take()
        else {
            return glib::ControlFlow::Continue;
        };
        detail.busy.set(false);
        progress.close();
        reload_session(&detail);
        match result {
            Ok(outcome) => {
                show_assignment_notice(&detail, outcome.learning_available);
            }
            Err(error) => show_error(&detail.window, "Could not assign speaker", &error),
        }
        glib::ControlFlow::Break
    });
}

fn progress_window(parent: &adw::ApplicationWindow, title: &str, text: &str) -> gtk::Window {
    let progress = gtk::Window::builder()
        .title(title)
        .transient_for(parent)
        .modal(true)
        .deletable(false)
        .default_width(360)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 12);
    body.set_margin_top(24);
    body.set_margin_bottom(24);
    body.set_margin_start(24);
    body.set_margin_end(24);
    let spinner = gtk::Spinner::new();
    spinner.set_spinning(true);
    body.append(&spinner);
    let label = gtk::Label::new(Some(text));
    label.set_wrap(true);
    body.append(&label);
    progress.set_child(Some(&body));
    progress.present();
    progress
}

fn start_hidden_sources_update(detail: &SessionDetail) {
    if detail.busy.get() {
        return;
    }
    let Some(session) = detail.selected.borrow().clone() else {
        return;
    };
    detail.busy.set(true);
    let hidden = [
        (AudioSource::Mic, detail.hide_mic.is_active()),
        (AudioSource::System, detail.hide_system.is_active()),
    ]
    .into_iter()
    .filter_map(|(source, hidden)| hidden.then_some(source))
    .collect::<Vec<_>>();
    let progress = progress_window(
        &detail.window,
        "Updating transcript",
        "Updating transcript…",
    );
    let result = Arc::new(Mutex::new(None));
    let thread_result = result.clone();
    std::thread::spawn(move || {
        let value =
            process::set_hidden_sources(&session, &hidden).map_err(|error| error.to_string());
        *thread_result.lock().expect("hidden sources result mutex") = Some(value);
    });
    let detail = detail.clone();
    glib::timeout_add_local(Duration::from_millis(150), move || {
        let Some(result) = result.lock().expect("hidden sources result mutex").take() else {
            return glib::ControlFlow::Continue;
        };
        detail.busy.set(false);
        progress.close();
        reload_session(&detail);
        if let Err(error) = result {
            show_error(&detail.window, "Could not update the transcript", &error);
        }
        glib::ControlFlow::Break
    });
}

fn reload_session(detail: &SessionDetail) {
    let selected = detail.selected.borrow().clone();
    if let Some(path) = selected {
        let _ = detail.load(&path);
    }
}

fn show_assignment_notice(detail: &SessionDetail, learning_available: bool) {
    if learning_available {
        return;
    }
    let notice = adw::AlertDialog::new(
        Some("Speaker assigned"),
        Some("The line is locked with this name, but its voice could not be learned."),
    );
    notice.add_response("close", "Close");
    notice.present(Some(&detail.window));
}

fn screenshot_card(path: &Path, time_ms: u64) -> gtk::Box {
    let card = gtk::Box::new(gtk::Orientation::Vertical, 4);
    card.add_css_class("shot-card");
    let picture = gtk::Picture::for_filename(path);
    picture.set_size_request(120, 80);
    picture.set_content_fit(gtk::ContentFit::Cover);
    card.append(&picture);
    let filename = gtk::Label::new(path.file_name().and_then(|name| name.to_str()));
    filename.set_ellipsize(gtk::pango::EllipsizeMode::End);
    filename.add_css_class("caption");
    card.append(&filename);
    let timestamp = gtk::Label::new(Some(&format_timestamp(time_ms)));
    timestamp.add_css_class("dim-label");
    timestamp.add_css_class("caption");
    card.append(&timestamp);
    card
}

fn property_row(title: &str, value: &str) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let title = gtk::Label::new(Some(title));
    title.set_xalign(0.0);
    title.add_css_class("heading");
    row.append(&title);
    let value = gtk::Label::new(Some(value));
    value.set_xalign(0.0);
    value.set_wrap(true);
    value.add_css_class("dim-label");
    row.append(&value);
    row
}

fn status_pill(text: &str, class: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.add_css_class("pill");
    label.add_css_class(&format!("pill-{class}"));
    label.set_valign(gtk::Align::Center);
    label
}

fn header_action_button(icon_name: &str, label: &str) -> gtk::Button {
    let content = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    content.append(&gtk::Image::from_icon_name(icon_name));
    content.append(&gtk::Label::new(Some(label)));
    let button = gtk::Button::builder().child(&content).build();
    button.add_css_class("flat");
    button
}

fn set_status(label: &gtk::Label, text: &str, class: &str) {
    label.set_label(text);
    for name in [
        "pill-ok",
        "pill-idle",
        "pill-busy",
        "pill-queued",
        "pill-failed",
        "pill-done",
        "pill-archived",
    ] {
        label.remove_css_class(name);
    }
    label.add_css_class(&format!("pill-{class}"));
}

fn selected_target(row: &adw::ComboRow, values: &Rc<RefCell<Vec<String>>>) -> String {
    values
        .borrow()
        .get(row.selected() as usize)
        .cloned()
        .unwrap_or_else(|| "default".into())
}

fn session_title(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Session")
        .replace("session-", "Session ")
}

fn stored_sample_count(path: &Path) -> io::Result<u64> {
    if archive::is_archived(path) {
        archive::sample_count(path)
    } else {
        Ok(fs::metadata(path)?.len() / 4)
    }
}

fn session_duration_ms(session: &Session, manifest: &Manifest) -> u64 {
    [
        (AudioSource::Mic, manifest.mic.enabled),
        (AudioSource::System, manifest.system.enabled),
    ]
    .into_iter()
    .filter(|(_, enabled)| *enabled)
    .filter_map(|(source, _)| stored_sample_count(&session.stored_audio_path(source)).ok())
    .map(|samples| samples * 1000 / u64::from(SAMPLE_RATE))
    .max()
    .unwrap_or(0)
}

fn display_started(value: &str) -> String {
    value
        .get(..16)
        .unwrap_or(value)
        .replace('T', " ")
        .replace('-', "‑")
}

fn format_duration(ms: u64) -> String {
    let seconds = ms / 1000;
    if seconds >= 3600 {
        format!("{} h {:02} min", seconds / 3600, seconds % 3600 / 60)
    } else if seconds >= 60 {
        format!("{} min {:02} s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds} s")
    }
}

fn format_timestamp(ms: u64) -> String {
    let seconds = ms / 1000;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds % 3600 / 60,
        seconds % 60
    )
}

fn format_timeline_timestamp(ms: u64) -> String {
    let seconds = ms / 1000;
    if seconds < 3600 {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    } else {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds % 3600 / 60,
            seconds % 60
        )
    }
}

fn safe_session_path(root: &Path, relative: &str) -> Option<PathBuf> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return None;
    }
    Some(root.join(relative))
}

fn clear_box(container: &gtk::Box) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
}

fn clear_flow(container: &gtk::FlowBox) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
}

fn append_empty(container: &gtk::Box, text: &str) -> gtk::Label {
    let label = gtk::Label::new(Some(text));
    label.set_margin_top(28);
    label.set_margin_start(16);
    label.set_margin_end(16);
    label.set_wrap(true);
    label.add_css_class("dim-label");
    container.append(&label);
    label
}

fn open_path(path: &Path) {
    let file = gio::File::for_path(path);
    let _ = gio::AppInfo::launch_default_for_uri(&file.uri(), None::<&gio::AppLaunchContext>);
}

fn show_error(parent: &adw::ApplicationWindow, heading: &str, body: &str) {
    // Also reach the terminal or journal, where a dismissed dialog can be found again.
    eprintln!("error: {heading}: {body}");
    let dialog = adw::AlertDialog::new(Some(heading), Some(body));
    dialog.add_response("close", "Close");
    dialog.set_default_response(Some("close"));
    dialog.present(Some(parent));
}

fn show_about(parent: &adw::ApplicationWindow) {
    let dialog = adw::AboutDialog::builder()
        .application_name("Singstone")
        .application_icon("audio-input-microphone-symbolic")
        .version(env!("CARGO_PKG_VERSION"))
        .comments("Local-first meeting recording, transcription, and speaker diarization for Linux and PipeWire.")
        .license_type(gtk::License::MitX11)
        .website("https://github.com/nsg/singstone")
        .build();
    dialog.present(Some(parent));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterance(speaker: &str, speaker_id: &str, start_ms: u64, end_ms: u64) -> Utterance {
        Utterance {
            start_ms,
            end_ms,
            source: AudioSource::System,
            speaker_id: speaker_id.into(),
            speaker: speaker.into(),
            text: "text".into(),
            locked: false,
            echo: false,
        }
    }

    #[test]
    fn frequent_speakers_rank_duration_then_name() {
        let utterances = vec![
            utterance("Alice", "spk_1", 0, 500),
            utterance("Alice", "spk_1", 700, 900),
            utterance("Bob", "spk_2", 0, 700),
            utterance("Carol", "spk_3", 0, 600),
            utterance("Dana", "spk_4", 0, 550),
            utterance("SPEAKER_00", "spk_5", 0, 2_000),
            utterance("unknown", "spk_6", 0, 2_000),
            utterance("Local owner", "local", 0, 2_000),
        ];

        assert_eq!(
            frequent_speakers(&utterances, 3),
            vec!["Alice", "Bob", "Carol"]
        );
    }

    #[test]
    fn assignable_ids_match_correction_parser_formats() {
        assert_eq!(parse_assignable_id("mic_3"), Some((AudioSource::Mic, 3)));
        assert_eq!(
            parse_assignable_id("spk_42"),
            Some((AudioSource::System, 42))
        );
        assert_eq!(
            parse_assignable_id("speaker-2"),
            Some((AudioSource::System, 2))
        );
        assert_eq!(parse_assignable_id("spk_nope"), None);
        assert_eq!(parse_assignable_id("other_1"), None);
    }

    #[test]
    fn session_paths_cannot_escape_the_session() {
        let root = Path::new("/meetings/session");
        assert_eq!(
            safe_session_path(root, "screenshots/shot.png"),
            Some(root.join("screenshots/shot.png"))
        );
        assert_eq!(safe_session_path(root, "../secret"), None);
        assert_eq!(safe_session_path(root, "/etc/passwd"), None);
    }

    #[test]
    fn durations_are_human_readable() {
        assert_eq!(format_duration(42_000), "42 s");
        assert_eq!(format_duration(125_000), "2 min 05 s");
        assert_eq!(format_timestamp(3_723_000), "01:02:03");
    }

    #[test]
    fn timeline_timestamps_are_compact_below_one_hour() {
        assert_eq!(format_timeline_timestamp(4_999), "0:04");
        assert_eq!(format_timeline_timestamp(751_000), "12:31");
        assert_eq!(format_timeline_timestamp(3_723_000), "1:02:03");
    }

    #[test]
    fn decoder_metrics_are_human_readable() {
        let metrics = TranscriptionProgress {
            source: AudioSource::System,
            processed_seconds: 45.0,
            total_seconds: 60.0,
            audio_seconds: 45.0,
            elapsed_seconds: 30.0,
            decoded_tokens: 420,
            tokens_per_second: Some(14.25),
        };
        let (throughput, detail) = transcription_metrics_text(metrics);
        assert_eq!(throughput, "14.2 tokens/s  ·  1.50× realtime");
        assert_eq!(detail, "system · 45.0 s audio in 30.0 s · 420 tokens");
        let progress = TranscriptionProgress {
            processed_seconds: 1_498.0,
            total_seconds: 1_986.0,
            ..metrics
        };
        assert_eq!(
            transcription_progress_text(progress),
            "1498 / 1986 s  ·  75%"
        );
    }

    #[test]
    fn worker_exit_maps_to_processing_outcome() {
        let success = ExitStatus::from_raw(0);
        assert_eq!(
            processing_outcome(success, None, false),
            ProcessingOutcome::Finished
        );

        let failure = ExitStatus::from_raw(7 << 8);
        assert_eq!(
            processing_outcome(failure, Some("model missing".into()), false),
            ProcessingOutcome::Failed("model missing".into())
        );
        assert_eq!(
            processing_outcome(failure, None, true),
            ProcessingOutcome::Cancelled
        );

        let signalled = ExitStatus::from_raw(9);
        assert_eq!(
            processing_outcome(signalled, None, false),
            ProcessingOutcome::Failed("processing stopped unexpectedly (signal 9)".into())
        );
    }

    #[test]
    fn worker_output_that_is_not_an_event_is_skipped() {
        let progress = Mutex::new(ProcessingProgress {
            stage: ProcessingStage::Preparing,
            fraction: None,
        });
        let metrics = Mutex::new(None);
        let output = concat!(
            "native library banner\n",
            "{\"event\":\"progress\",\"progress\":{\"stage\":\"Diarizing\",\"fraction\":null}}\n",
            "\u{fffd}\n",
            "{\"event\":\"failure\",\"error\":\"model missing\"}\n",
        );

        let failure = read_worker_events(output.as_bytes(), &progress, &metrics);

        assert_eq!(failure.as_deref(), Some("model missing"));
        assert_eq!(progress.lock().unwrap().stage, ProcessingStage::Diarizing);
    }

    #[test]
    fn paused_time_is_removed_from_transcription_elapsed_time() {
        let metrics = TranscriptionProgress {
            source: AudioSource::System,
            processed_seconds: 45.0,
            total_seconds: 60.0,
            audio_seconds: 45.0,
            elapsed_seconds: 30.0,
            decoded_tokens: 420,
            tokens_per_second: Some(14.25),
        };

        let adjusted = compensate_paused_elapsed(metrics, Duration::from_millis(12_500));
        assert_eq!(adjusted.elapsed_seconds, 17.5);
        assert_eq!(
            compensate_paused_elapsed(metrics, Duration::from_secs(40)).elapsed_seconds,
            0.0
        );
    }

    #[test]
    fn playback_range_uses_utterance_timestamps() {
        assert_eq!(audio_byte_range(0, 1_000, 80_000).unwrap(), (0, 64_000));
        assert_eq!(
            audio_byte_range(1_250, 1_750, 200_000).unwrap(),
            (80_000, 32_000)
        );
    }

    #[test]
    fn playback_range_clamps_to_complete_recorded_samples() {
        assert_eq!(
            audio_byte_range(1_000, 2_000, 80_003).unwrap(),
            (64_000, 16_000)
        );
    }

    #[test]
    fn playback_range_rejects_empty_or_missing_audio() {
        assert!(audio_byte_range(1_000, 1_000, 100_000).is_err());
        assert!(audio_byte_range(2_000, 3_000, 100_000).is_err());
    }
}
