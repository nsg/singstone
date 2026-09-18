mod config;

use crate::audio::{devices, record};
use crate::cli::{Command, EnrollArgs, ProcessArgs, RecordArgs};
use crate::format::jsonl;
use crate::merge::process::{self, ProcessingStage};
use crate::model_setup;
use crate::session::Session;
use crate::speaker::database::{self, SpeakerDatabase};
use crate::speaker::enroll;
use crate::transcription::{TranscriptionProgress, backend};
use crate::types::{AudioSource, Manifest, SAMPLE_RATE, ScreenshotEntry, SessionState, Utterance};
use adw::prelude::*;
use gtk::{gdk, gio, glib};
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::{Cell, RefCell};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use self::config::GuiConfig;

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
.recording-dot { color: #e01b24; }
.live-bar { background: alpha(#e01b24, 0.10); padding: 7px 12px; }
.pill-button { border-radius: 999px; padding: 8px 26px; font-weight: 600; }
.navigation-sidebar row { border-radius: 10px; margin: 2px 6px; }
.shot-card { padding: 8px; border-radius: 12px; background: alpha(currentColor, 0.05); }
.mono-button { font-family: monospace; font-size: 12px; }
.transcript-row { padding: 8px 12px; }
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
}

#[derive(Clone)]
struct SessionDetail {
    root: gtk::Box,
    title: gtk::Label,
    subtitle: gtk::Label,
    status: gtk::Label,
    process_button: gtk::Button,
    transcript: gtk::Box,
    screenshots: gtk::FlowBox,
    metadata: gtk::Box,
    files: gtk::Box,
    selected: Rc<RefCell<Option<PathBuf>>>,
    window: adw::ApplicationWindow,
}

pub fn run() -> ExitCode {
    let app = adw::Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_window);
    let _ = app.run_with_args(&["singstone"]);
    ExitCode::SUCCESS
}

fn build_window(app: &adw::Application) {
    install_css();

    let config = Rc::new(RefCell::new(GuiConfig::load().unwrap_or_default()));
    let style_manager = adw::StyleManager::default();
    if let Some(dark_mode) = config.borrow().dark_mode {
        set_dark_mode(&style_manager, dark_mode);
    }

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Singstone")
        .default_width(1080)
        .default_height(700)
        .build();

    let toolbar = adw::ToolbarView::new();
    let header = adw::HeaderBar::new();
    let new_recording = header_action_button("media-record-symbolic", "New recording");
    new_recording.set_tooltip_text(Some("Create a new recording"));
    header.pack_start(&new_recording);
    let speakers = header_action_button("system-users-symbolic", "Speakers");
    speakers.set_tooltip_text(Some("Manage enrolled speakers"));
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
    let sidebar = build_sidebar(&search, &session_list);

    let detail = build_session_detail(&window);
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
    stack.connect_visible_child_name_notify(move |stack| {
        if stack.visible_child_name().as_deref() == Some("speakers") {
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
    );
    wire_recording(
        &window,
        &recording_page,
        &live_bar,
        &header_live,
        &banner,
        &config,
        &session_list,
        &session_paths,
        &search,
        &detail,
        &stack,
        &close_when_recording_stops,
    );
    wire_processing(
        &window,
        &detail,
        &banner,
        &config,
        &session_list,
        &session_paths,
        &search,
    );
    wire_settings(
        &window,
        &settings_page,
        &recording_page,
        &config,
        &session_list,
        &session_paths,
        &search,
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

    let stop_on_close = recording_page.job.clone();
    let close_after_stop = close_when_recording_stops.clone();
    window.connect_close_request(move |_| {
        if let Some(job) = stop_on_close.borrow().as_ref() {
            job.stop.store(true, Ordering::Release);
            close_after_stop.set(true);
            return glib::Propagation::Stop;
        }
        glib::Propagation::Proceed
    });

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

fn build_sidebar(search: &gtk::SearchEntry, list: &gtk::ListBox) -> gtk::Box {
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
    root
}

struct RecordingPage {
    root: gtk::Box,
    mic: adw::ComboRow,
    diarize_mic: adw::SwitchRow,
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
    let diarize_mic = adw::SwitchRow::builder()
        .title("Several people share the microphone")
        .subtitle("Separate microphone speech by speaker during processing")
        .active(config.diarize_mic)
        .build();
    diarize_mic.add_prefix(&gtk::Image::from_icon_name("system-users-symbolic"));
    system.set_model(Some(&gtk::StringList::new(&[
        "Default (WirePlumber)",
        "None",
    ])));
    audio.add(&mic);
    audio.add(&diarize_mic);
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
        .description("Used when microphone speaker separation is off")
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
        diarize_mic,
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

fn build_session_detail(window: &adw::ApplicationWindow) -> SessionDetail {
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
    heading.append(&top);
    let subtitle = gtk::Label::new(Some("Recorded meetings appear in the sidebar."));
    subtitle.set_xalign(0.0);
    subtitle.add_css_class("dim-label");
    heading.append(&subtitle);
    root.append(&heading);
    root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_vexpand(true);
    paned.set_position(570);
    paned.set_resize_start_child(true);
    paned.set_shrink_start_child(false);
    paned.set_resize_end_child(false);
    paned.set_shrink_end_child(false);
    let transcript_frame = gtk::Box::new(gtk::Orientation::Vertical, 6);
    transcript_frame.set_margin_top(12);
    let transcript_heading = gtk::Label::new(Some("Transcript"));
    transcript_heading.add_css_class("title-4");
    transcript_heading.set_xalign(0.0);
    transcript_heading.set_margin_start(16);
    transcript_frame.append(&transcript_heading);
    let transcript = gtk::Box::new(gtk::Orientation::Vertical, 0);
    transcript_frame.append(
        &gtk::ScrolledWindow::builder()
            .vexpand(true)
            .child(&transcript)
            .build(),
    );
    paned.set_start_child(Some(&transcript_frame));

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

    SessionDetail {
        root,
        title,
        subtitle,
        status,
        process_button,
        transcript,
        screenshots,
        metadata,
        files,
        selected: Rc::new(RefCell::new(None)),
        window: window.clone(),
    }
}

#[derive(Clone)]
struct SpeakersPage {
    root: gtk::Box,
    group: adw::PreferencesGroup,
    rows: Rc<RefCell<Vec<adw::ActionRow>>>,
    add_row: adw::ActionRow,
    add: gtk::Button,
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
        .title("Enrolled voices")
        .description("Voices are matched locally; uncertain matches stay anonymous")
        .build();
    let db_path = std::env::var_os("SINGSTONE_SPEAKERS_DB")
        .map(PathBuf::from)
        .unwrap_or_else(database::default_path);
    body.append(&group);
    let add_row = adw::ActionRow::builder()
        .title("Enroll a new voice")
        .subtitle("Pick 10–30 seconds of clean speech")
        .build();
    add_row.add_prefix(&gtk::Image::from_icon_name("contact-new-symbolic"));
    let add = gtk::Button::with_label("Add…");
    add.set_valign(gtk::Align::Center);
    add_row.add_suffix(&add);
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
        add_row,
        add,
        database_path: db_path,
    };
    page.refresh(window);
    let page_for_add = page.clone();
    let window_for_add = window.clone();
    page.add.connect_clicked(move |_| {
        show_enroll_dialog(&window_for_add, &page_for_add);
    });
    page
}

impl SpeakersPage {
    fn refresh(&self, window: &adw::ApplicationWindow) {
        if self.add_row.parent().is_some() {
            self.group.remove(&self.add_row);
        }
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
                    remove.set_tooltip_text(Some("Remove enrolled voice"));
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
                    .title("No enrolled voices yet")
                    .subtitle("Add 10–30 seconds of clean speech to recognize someone")
                    .build();
                row.add_prefix(&gtk::Image::from_icon_name("contact-new-symbolic"));
                self.group.add(&row);
                rows.push(row);
            }
        }
        self.group.add(&self.add_row);
        *self.rows.borrow_mut() = rows;
    }
}

fn confirm_delete_speaker(parent: &adw::ApplicationWindow, page: &SpeakersPage, name: &str) {
    let dialog = adw::AlertDialog::new(
        Some("Remove enrolled voice?"),
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

fn show_enroll_dialog(parent: &adw::ApplicationWindow, page: &SpeakersPage) {
    let dialog = adw::AlertDialog::new(
        Some("Enroll a new voice"),
        Some(
            "Name the speaker, then choose one or more clean 16 kHz WAV or raw f32le recordings (10–30 seconds works well).",
        ),
    );
    let entry = gtk::Entry::builder()
        .placeholder_text("Speaker name")
        .activates_default(true)
        .build();
    dialog.set_extra_child(Some(&entry));
    dialog.add_responses(&[("cancel", "Cancel"), ("choose", "Choose audio…")]);
    dialog.set_default_response(Some("choose"));
    dialog.set_close_response("cancel");
    dialog.set_response_appearance("choose", adw::ResponseAppearance::Suggested);
    let parent_for_response = parent.clone();
    let page_for_response = page.clone();
    dialog.connect_response(Some("choose"), move |_, _| {
        let name = entry.text().trim().to_owned();
        if name.is_empty() {
            show_error(
                &parent_for_response,
                "Speaker name required",
                "Enter the name that should appear in transcripts.",
            );
            return;
        }
        choose_enrollment_audio(&parent_for_response, &page_for_response, name);
    });
    dialog.present(Some(parent));
}

fn choose_enrollment_audio(parent: &adw::ApplicationWindow, page: &SpeakersPage, name: String) {
    let Some(embedding_model) = std::env::var_os("SINGSTONE_EMBEDDING_MODEL").map(PathBuf::from)
    else {
        show_error(
            parent,
            "Speaker model is not configured",
            "Process a session after installing the local models, then try again.",
        );
        return;
    };
    let chooser = gtk::FileDialog::builder()
        .title("Choose clean voice recordings")
        .accept_label("Enroll")
        .modal(true)
        .build();
    let parent_for_choose = parent.clone();
    let page_for_choose = page.clone();
    chooser.open_multiple(Some(parent), None::<&gio::Cancellable>, move |result| {
        let Ok(files) = result else { return };
        let samples = (0..files.n_items())
            .filter_map(|index| files.item(index))
            .filter_map(|item| item.downcast::<gio::File>().ok())
            .filter_map(|file| file.path())
            .collect::<Vec<_>>();
        if samples.is_empty() {
            return;
        }
        let args = EnrollArgs {
            name,
            samples,
            embedding_model,
            models_lock: std::env::var_os("SINGSTONE_MODELS_LOCK").map(PathBuf::from),
            allow_unverified_models: false,
            speakers_db: Some(page_for_choose.database_path.clone()),
            replace: false,
        };
        start_enrollment(&parent_for_choose, &page_for_choose, args);
    });
}

fn start_enrollment(parent: &adw::ApplicationWindow, page: &SpeakersPage, args: EnrollArgs) {
    page.add.set_sensitive(false);
    page.add.set_label("Enrolling…");
    let result = Arc::new(Mutex::new(None));
    let thread_result = result.clone();
    std::thread::spawn(move || {
        let value = (|| -> Result<(), String> {
            let command = Command::Enroll(args);
            model_setup::ensure_available(&command).map_err(|error| error.to_string())?;
            let Command::Enroll(args) = command else {
                unreachable!()
            };
            enroll::run(args).map_err(|error| error.to_string())
        })();
        *thread_result.lock().expect("enrollment result mutex") = Some(value);
    });
    let parent = parent.clone();
    let page = page.clone();
    glib::timeout_add_local(Duration::from_millis(150), move || {
        let Some(result) = result.lock().expect("enrollment result mutex").take() else {
            return glib::ControlFlow::Continue;
        };
        page.add.set_sensitive(true);
        page.add.set_label("Add a voice…");
        match result {
            Ok(()) => page.refresh(&parent),
            Err(error) => show_error(&parent, "Could not enroll voice", &error),
        }
        glib::ControlFlow::Break
    });
}

struct SettingsPage {
    root: gtk::Box,
    meetings: adw::ActionRow,
    screenshots: adw::ActionRow,
    swedish_transcription: gtk::Switch,
    meetings_change: gtk::Button,
    screenshots_change: gtk::Button,
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
        swedish_transcription,
        meetings_change,
        screenshots_change,
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
) {
    let parent = window.clone();
    let config_for_meetings = config.clone();
    let row_for_meetings = page.meetings.clone();
    let hint_for_meetings = recording.hint.clone();
    let sessions_for_meetings = sessions.clone();
    let paths_for_meetings = paths.clone();
    let search_for_meetings = search.clone();
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
            populate_sessions(&sessions, &paths, &path, &search.text());
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

fn wire_session_browser(
    window: &adw::ApplicationWindow,
    config: &Rc<RefCell<GuiConfig>>,
    search: &gtk::SearchEntry,
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    detail: &SessionDetail,
    stack: &adw::ViewStack,
) {
    populate_sessions(list, paths, &config.borrow().meetings_dir, "");
    let list_for_search = list.clone();
    let paths_for_search = paths.clone();
    let config_for_search = config.clone();
    search.connect_search_changed(move |entry| {
        populate_sessions(
            &list_for_search,
            &paths_for_search,
            &config_for_search.borrow().meetings_dir,
            &entry.text(),
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

#[allow(clippy::too_many_arguments)]
fn wire_recording(
    window: &adw::ApplicationWindow,
    page: &RecordingPage,
    live: &LiveBar,
    header_live: &gtk::Box,
    banner: &adw::Banner,
    config: &Rc<RefCell<GuiConfig>>,
    session_list: &gtk::ListBox,
    session_paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
    detail: &SessionDetail,
    stack: &adw::ViewStack,
    close_when_stopped: &Rc<Cell<bool>>,
) {
    let updating_diarize_mic = Rc::new(Cell::new(false));
    let updating_diarize_mic_for_notify = updating_diarize_mic.clone();
    let config_for_diarize_mic = config.clone();
    let window_for_diarize_mic = window.clone();
    page.diarize_mic.connect_active_notify(move |row| {
        if updating_diarize_mic_for_notify.replace(true) {
            return;
        }
        let mut updated = config_for_diarize_mic.borrow().clone();
        updated.diarize_mic = row.is_active();
        if let Err(error) = updated.save() {
            show_error(
                &window_for_diarize_mic,
                "Could not save settings",
                &error.to_string(),
            );
            row.set_active(config_for_diarize_mic.borrow().diarize_mic);
        } else {
            *config_for_diarize_mic.borrow_mut() = updated;
        }
        updating_diarize_mic_for_notify.set(false);
    });

    let stop_for_button = page.job.clone();
    let record_button_for_stop = page.button.clone();
    live.stop.connect_clicked(move |_| {
        if let Some(job) = stop_for_button.borrow().as_ref() {
            job.stop.store(true, Ordering::Release);
            record_button_for_stop.set_label("Stopping…");
        }
    });

    let window_for_start = window.clone();
    let config_for_start = config.clone();
    let mic = page.mic.clone();
    let diarize_mic = page.diarize_mic.clone();
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
    page.button.connect_clicked(move |_| {
        if let Some(active) = job.borrow().as_ref() {
            active.stop.store(true, Ordering::Release);
            button.set_label("Stopping…");
            return;
        }
        let mic_target = selected_target(&mic, &mic_targets);
        let system_target = selected_target(&system, &system_targets);
        if mic_target == "none" && system_target == "none" {
            show_error(
                &window_for_start,
                "No audio source selected",
                "Choose a microphone, system audio, or both.",
            );
            return;
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
            return;
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
                return;
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
        });
        button.set_label("Stop recording");
        button.remove_css_class("suggested-action");
        button.add_css_class("destructive-action");
        hint.set_label("Recording… press Stop when the meeting is finished");
        mic.set_sensitive(false);
        diarize_mic.set_sensitive(false);
        system.set_sensitive(false);
        screenshots.set_sensitive(false);
        name.set_sensitive(false);
        live_mic_group.set_visible(mic_target != "none");
        live_system_group.set_visible(system_target != "none");
        live_root.set_visible(true);
        header_live_for_start.set_visible(true);
    });

    let time = live.time.clone();
    let mic_level = live.mic_level.clone();
    let system_level = live.system_level.clone();
    let screenshot_count = live.screenshot_count.clone();
    let job_for_poll = page.job.clone();
    let button_for_poll = page.button.clone();
    let hint_for_poll = page.hint.clone();
    let mic_for_poll = page.mic.clone();
    let diarize_mic_for_poll = page.diarize_mic.clone();
    let system_for_poll = page.system.clone();
    let shots_for_poll = page.screenshots.clone();
    let name_for_poll = page.name.clone();
    let live_for_poll = live.root.clone();
    let header_for_poll = header_live.clone();
    let window_for_poll = window.clone();
    let banner_for_poll = banner.clone();
    let config_for_poll = config.clone();
    let list_for_poll = session_list.clone();
    let paths_for_poll = session_paths.clone();
    let search_for_poll = search.clone();
    let detail_for_poll = detail.clone();
    let stack_for_poll = stack.clone();
    let close_when_stopped = close_when_stopped.clone();
    glib::timeout_add_local(Duration::from_millis(200), move || {
        let completed = {
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
            active.result.lock().expect("record result mutex").take()
        };
        let Some(result) = completed else {
            return glib::ControlFlow::Continue;
        };
        job_for_poll.borrow_mut().take();
        if close_when_stopped.replace(false) {
            window_for_poll.close();
            return glib::ControlFlow::Break;
        }
        button_for_poll.set_label("Start recording");
        button_for_poll.remove_css_class("destructive-action");
        button_for_poll.add_css_class("suggested-action");
        hint_for_poll.set_label(&format!(
            "Creates a new session in {}",
            config_for_poll.borrow().meetings_dir.display()
        ));
        mic_for_poll.set_sensitive(true);
        diarize_mic_for_poll.set_sensitive(true);
        system_for_poll.set_sensitive(true);
        shots_for_poll.set_sensitive(true);
        name_for_poll.set_sensitive(true);
        live_for_poll.set_visible(false);
        header_for_poll.set_visible(false);
        time.set_label("00:00");
        mic_level.set_value(0.0);
        system_level.set_value(0.0);
        screenshot_count.set_label("0 screenshots");
        populate_sessions(
            &list_for_poll,
            &paths_for_poll,
            &config_for_poll.borrow().meetings_dir,
            &search_for_poll.text(),
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
}

#[allow(clippy::too_many_arguments)]
fn wire_processing(
    window: &adw::ApplicationWindow,
    detail: &SessionDetail,
    banner: &adw::Banner,
    config: &Rc<RefCell<GuiConfig>>,
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    search: &gtk::SearchEntry,
) {
    let busy = Rc::new(Cell::new(false));
    let confirmed_reprocess = Rc::new(Cell::new(false));
    let selected = detail.selected.clone();
    let parent = window.clone();
    let detail_for_done = detail.clone();
    let banner_for_done = banner.clone();
    let config_for_done = config.clone();
    let list_for_done = list.clone();
    let paths_for_done = paths.clone();
    let search_for_done = search.clone();
    detail.process_button.connect_clicked(move |button| {
        if busy.get() {
            return;
        }
        let Some(session_path) = selected.borrow().clone() else {
            return;
        };
        if session_path.join("transcript.jsonl").is_file()
            && !confirmed_reprocess.replace(false)
        {
            let dialog = adw::AlertDialog::new(
                Some("Reprocess this recording?"),
                Some(
                    "This replaces the transcript, intermediate processing files, and assigned speaker names. Recorded audio is preserved.",
                ),
            );
            dialog.add_responses(&[("cancel", "Cancel"), ("reprocess", "Reprocess")]);
            dialog.set_default_response(Some("reprocess"));
            dialog.set_close_response("cancel");
            dialog.set_response_appearance("reprocess", adw::ResponseAppearance::Suggested);
            let button = button.clone();
            let confirmed_reprocess = confirmed_reprocess.clone();
            dialog.connect_response(Some("reprocess"), move |_, _| {
                confirmed_reprocess.set(true);
                button.emit_clicked();
            });
            dialog.present(Some(&parent));
            return;
        }
        busy.set(true);
        let dialog = processing_dialog(&backend::current());
        let progress = Arc::new(Mutex::new(ProcessingStage::Preparing));
        let transcription_metrics = Arc::new(Mutex::new(None));
        let cancelled = Arc::new(AtomicBool::new(false));
        let result = Arc::new(Mutex::new(None));
        let thread_progress = progress.clone();
        let thread_transcription_metrics = transcription_metrics.clone();
        let thread_result = result.clone();
        let thread_cancelled = cancelled.clone();
        let (diarize_mic, swedish_transcription) = {
            let config = config_for_done.borrow();
            (config.diarize_mic, config.swedish_transcription)
        };
        std::thread::spawn(move || {
            let prepared = (|| -> Result<ProcessArgs, String> {
                let mut args = ProcessArgs::for_session(session_path);
                args.diarize_mic = diarize_mic;
                let model_env = if swedish_transcription {
                    "SINGSTONE_WHISPER_MODEL_SWEDISH"
                } else {
                    "SINGSTONE_WHISPER_MODEL_MULTILINGUAL"
                };
                if let Some(model) = std::env::var_os(model_env) {
                    args.whisper_model = Some(PathBuf::from(model));
                }
                args.language = if swedish_transcription { "sv" } else { "auto" }.into();
                let command = Command::Process(args);
                model_setup::ensure_available(&command).map_err(|error| error.to_string())?;
                let Command::Process(args) = command else {
                    unreachable!()
                };
                Ok(args)
            })();
            let outcome = match prepared {
                Err(error) => ProcessingOutcome::Failed(error),
                Ok(args) => match process::run_with_control_and_metrics(
                    args,
                    |stage| {
                        *thread_progress.lock().expect("processing progress mutex") = stage;
                    },
                    thread_cancelled,
                    Arc::new(move |metrics| {
                        *thread_transcription_metrics
                            .lock()
                            .expect("transcription metrics mutex") = Some(metrics);
                    }),
                ) {
                    Ok(()) => ProcessingOutcome::Finished,
                    Err(error)
                        if error
                            .downcast_ref::<std::io::Error>()
                            .is_some_and(|error| error.kind() == std::io::ErrorKind::Interrupted) =>
                    {
                        ProcessingOutcome::Cancelled
                    }
                    Err(error) => ProcessingOutcome::Failed(error.to_string()),
                },
            };
            *thread_result.lock().expect("processing result mutex") = Some(outcome);
        });

        let label = dialog.label.clone();
        let bar = dialog.progress.clone();
        let stages = dialog.stages.clone();
        let throughput = dialog.throughput.clone();
        let throughput_detail = dialog.throughput_detail.clone();
        let cancel_button = dialog.cancel.clone();
        let cancelled_for_click = cancelled.clone();
        let label_for_cancel = dialog.label.clone();
        dialog.cancel.connect_clicked(move |button| {
            cancelled_for_click.store(true, Ordering::Release);
            button.set_sensitive(false);
            button.set_label("Cancelling…");
            label_for_cancel.set_label("Cancelling after the current stage…");
        });
        let processing_dialog = dialog.dialog.clone();
        let busy_for_poll = busy.clone();
        let parent_for_poll = parent.clone();
        let detail_for_poll = detail_for_done.clone();
        let banner_for_poll = banner_for_done.clone();
        let config_for_poll = config_for_done.clone();
        let list_for_poll = list_for_done.clone();
        let paths_for_poll = paths_for_done.clone();
        let search_for_poll = search_for_done.clone();
        let selected_for_poll = selected.clone();
        glib::timeout_add_local(Duration::from_millis(150), move || {
            let stage = *progress.lock().expect("processing progress mutex");
            if !cancelled.load(Ordering::Acquire) {
                label.set_label(stage.label());
            }
            bar.set_fraction(stage.fraction());
            update_processing_stages(&stages, stage);
            if let Some(metrics) = *transcription_metrics
                .lock()
                .expect("transcription metrics mutex")
            {
                update_transcription_metrics(&throughput, &throughput_detail, metrics);
            }
            let completed = result.lock().expect("processing result mutex").take();
            let Some(result) = completed else {
                return glib::ControlFlow::Continue;
            };
            busy_for_poll.set(false);
            cancel_button.set_sensitive(false);
            processing_dialog.force_close();
            match result {
                ProcessingOutcome::Finished => {
                    let selected_path = selected_for_poll.borrow().clone();
                    if let Some(path) = selected_path {
                        let _ = detail_for_poll.load(&path);
                    }
                    populate_sessions(
                        &list_for_poll,
                        &paths_for_poll,
                        &config_for_poll.borrow().meetings_dir,
                        &search_for_poll.text(),
                    );
                    banner_for_poll.set_title(
                        "Processing finished — transcript.jsonl and transcript.txt are ready",
                    );
                    banner_for_poll.set_revealed(true);
                }
                ProcessingOutcome::Cancelled => {
                    banner_for_poll.set_title(
                        "Processing cancelled — recorded audio is unchanged; completed stages may have refreshed outputs",
                    );
                    banner_for_poll.set_revealed(true);
                }
                ProcessingOutcome::Failed(error) => {
                    show_error(&parent_for_poll, "Processing failed", &error)
                }
            }
            glib::ControlFlow::Break
        });
        dialog.dialog.present(Some(&parent));
    });
}

struct ProcessingDialog {
    dialog: adw::Dialog,
    label: gtk::Label,
    progress: gtk::ProgressBar,
    stages: Vec<gtk::Label>,
    throughput: gtk::Label,
    throughput_detail: gtk::Label,
    cancel: gtk::Button,
}

enum ProcessingOutcome {
    Finished,
    Cancelled,
    Failed(String),
}

fn processing_dialog(backend: &backend::WhisperBackend) -> ProcessingDialog {
    let dialog = adw::Dialog::builder()
        .title("Processing session")
        .can_close(false)
        .content_width(470)
        .follows_content_size(true)
        .presentation_mode(adw::DialogPresentationMode::Floating)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 16);
    body.set_margin_top(28);
    body.set_margin_bottom(28);
    body.set_margin_start(28);
    body.set_margin_end(28);
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
    progress.set_fraction(ProcessingStage::Preparing.fraction());
    progress.set_show_text(true);
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
        "Audio stays on this device. Processing can take several minutes.",
    ));
    note.add_css_class("dim-label");
    note.set_wrap(true);
    body.append(&note);
    let cancel = gtk::Button::with_label("Cancel");
    cancel.set_halign(gtk::Align::Center);
    body.append(&cancel);
    dialog.set_child(Some(&body));
    ProcessingDialog {
        dialog,
        label,
        progress,
        stages,
        throughput,
        throughput_detail,
        cancel,
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

fn transcription_metrics_text(metrics: TranscriptionProgress) -> (String, String) {
    let speed = metrics.realtime_speed();
    let token_rate = metrics
        .tokens_per_second
        .map(|rate| format!("{rate:.1} tokens/s"))
        .unwrap_or_else(|| "Measuring tokens/s".into());
    (
        format!("{token_rate}  ·  {speed:.2}× realtime"),
        format!(
            "{} · {}% of current chunk · {:.1} s audio in {:.1} s · {} tokens",
            metrics.source,
            metrics.chunk_percent,
            metrics.audio_seconds,
            metrics.elapsed_seconds,
            metrics.decoded_tokens
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

impl SessionDetail {
    fn load(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let session = Session::open(path)?;
        let manifest = session.read_manifest()?;
        *self.selected.borrow_mut() = Some(path.to_owned());
        self.title.set_label(&session_title(path));
        let duration = session_duration_ms(&session, &manifest);
        self.subtitle.set_label(&format!(
            "{} · {}",
            display_started(&manifest.started_wallclock),
            format_duration(duration)
        ));
        let processed = session.transcript_path().is_file();
        let (status, class) = if manifest.state == SessionState::Recording {
            ("Recording", "busy")
        } else if processed {
            ("Processed", "ok")
        } else {
            ("Recorded", "idle")
        };
        set_status(&self.status, status, class);
        self.status.set_visible(true);
        self.process_button
            .set_label(if processed { "Reprocess" } else { "Process" });
        self.process_button.set_tooltip_text(Some(if processed {
            "Replace the transcript, intermediate files, and speaker assignments. Recorded audio is preserved."
        } else {
            "Process this recording"
        }));
        self.process_button
            .set_visible(manifest.state != SessionState::Recording);

        clear_box(&self.transcript);
        if processed {
            let utterances: Vec<Utterance> = jsonl::read_all(&session.transcript_path())?;
            if utterances.is_empty() {
                append_empty(&self.transcript, "The transcript is empty.");
            } else {
                for utterance in utterances {
                    self.transcript.append(&transcript_row(&utterance, self));
                }
            }
        } else {
            append_empty(
                &self.transcript,
                "This recording has not been processed yet.",
            );
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
}

fn populate_sessions(
    list: &gtk::ListBox,
    paths: &Rc<RefCell<Vec<PathBuf>>>,
    root: &Path,
    filter: &str,
) {
    while let Some(child) = list.first_child() {
        list.remove(&child);
    }
    let needle = filter.to_lowercase();
    let summaries = load_sessions(root)
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

fn load_sessions(root: &Path) -> Vec<SessionSummary> {
    let mut sessions = fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let session = Session::open(entry.path()).ok()?;
            let manifest = session.read_manifest().ok()?;
            let processed = session.transcript_path().is_file();
            let (status, status_class) = if manifest.state == SessionState::Recording {
                ("Recording", "busy")
            } else if processed {
                ("Processed", "ok")
            } else {
                ("Recorded", "idle")
            };
            Some(SessionSummary {
                title: session_title(&entry.path()),
                started: manifest.started_wallclock.clone(),
                duration_ms: session_duration_ms(&session, &manifest),
                path: entry.path(),
                status,
                status_class,
            })
        })
        .collect::<Vec<_>>();
    sessions.sort_by(|a, b| b.started.cmp(&a.started));
    sessions
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

fn transcript_row(utterance: &Utterance, detail: &SessionDetail) -> gtk::Box {
    let row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    row.add_css_class("transcript-row");
    let avatar = gtk::Image::from_icon_name("avatar-default-symbolic");
    avatar.set_pixel_size(28);
    avatar.add_css_class("speaker-avatar");
    if utterance.speaker.starts_with("SPEAKER_") || utterance.speaker == "unknown" {
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
    row.append(&avatar);
    let column = gtk::Box::new(gtk::Orientation::Vertical, 2);
    column.set_hexpand(true);
    let head = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    let speaker = gtk::Label::new(Some(&utterance.speaker));
    speaker.add_css_class("heading");
    if utterance.speaker.starts_with("SPEAKER_") || utterance.speaker == "unknown" {
        speaker.add_css_class("warning-text");
    }
    head.append(&speaker);
    let timestamp = gtk::Label::new(Some(&format_timestamp(utterance.start_ms)));
    timestamp.add_css_class("dim-label");
    timestamp.add_css_class("caption");
    head.append(&timestamp);
    if is_assignable_speaker(utterance) {
        let assign = gtk::Button::with_label("Assign…");
        assign.add_css_class("flat");
        assign.add_css_class("caption");
        let detail = detail.clone();
        let speaker_id = utterance.speaker_id.clone();
        assign.connect_clicked(move |_| show_assignment_dialog(&detail, &speaker_id));
        head.append(&assign);
    }
    column.append(&head);
    let text = gtk::Label::new(Some(&utterance.text));
    text.set_xalign(0.0);
    text.set_wrap(true);
    text.set_selectable(true);
    column.append(&text);
    row.append(&column);
    row
}

fn is_assignable_speaker(utterance: &Utterance) -> bool {
    (utterance.speaker.starts_with("SPEAKER_") || utterance.speaker == "unknown")
        && (utterance
            .speaker_id
            .rsplit_once('_')
            .is_some_and(|(prefix, cluster)| {
                matches!(prefix, "mic" | "spk") && cluster.parse::<u32>().is_ok()
            })
            || utterance
                .speaker_id
                .strip_prefix("speaker-")
                .is_some_and(|cluster| cluster.parse::<u32>().is_ok()))
}

fn show_assignment_dialog(detail: &SessionDetail, speaker_id: &str) {
    let dialog = adw::AlertDialog::new(
        Some("Who is this speaker?"),
        Some(
            "The name updates this transcript. When the local voice model is available, Singstone also learns this voice for future meetings.",
        ),
    );
    let content = gtk::Box::new(gtk::Orientation::Vertical, 8);
    let entry = gtk::Entry::builder()
        .placeholder_text("Speaker name")
        .activates_default(true)
        .build();
    let database_path = std::env::var_os("SINGSTONE_SPEAKERS_DB")
        .map(PathBuf::from)
        .unwrap_or_else(database::default_path);
    if let Ok(database) = SpeakerDatabase::load(&database_path)
        && !database.speakers.is_empty()
    {
        let names = database.speakers.keys().cloned().collect::<Vec<_>>();
        let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        let choices = gtk::DropDown::from_strings(&refs);
        choices.set_tooltip_text(Some("Choose an enrolled speaker"));
        if let Some(first) = names.first() {
            entry.set_text(first);
        }
        let entry_for_choice = entry.clone();
        choices.connect_selected_notify(move |choices| {
            if let Some(value) = choices.selected_item().and_downcast::<gtk::StringObject>() {
                entry_for_choice.set_text(&value.string());
            }
        });
        content.append(&choices);
    }
    content.append(&entry);
    dialog.set_extra_child(Some(&content));
    dialog.add_responses(&[("cancel", "Cancel"), ("assign", "Assign and learn")]);
    dialog.set_default_response(Some("assign"));
    dialog.set_close_response("cancel");
    dialog.set_response_appearance("assign", adw::ResponseAppearance::Suggested);
    let detail_for_response = detail.clone();
    let speaker_id = speaker_id.to_owned();
    dialog.connect_response(Some("assign"), move |_, _| {
        let name = entry.text().trim().to_owned();
        if name.is_empty() {
            show_error(
                &detail_for_response.window,
                "Speaker name required",
                "Enter or choose a name for this voice.",
            );
            return;
        }
        start_assignment(&detail_for_response, speaker_id.clone(), name);
    });
    dialog.present(Some(&detail.window));
}

fn start_assignment(detail: &SessionDetail, speaker_id: String, name: String) {
    let Some(session) = detail.selected.borrow().clone() else {
        return;
    };
    let progress = gtk::Window::builder()
        .title("Assigning speaker")
        .transient_for(&detail.window)
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
    let label = gtk::Label::new(Some("Updating transcript and learning the voice locally…"));
    label.set_wrap(true);
    body.append(&label);
    progress.set_child(Some(&body));
    progress.present();

    let result = Arc::new(Mutex::new(None));
    let thread_result = result.clone();
    std::thread::spawn(move || {
        let value = process::assign_speaker(ProcessArgs::for_session(session), &speaker_id, &name)
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
        progress.close();
        match result {
            Ok(outcome) => {
                let selected_path = detail.selected.borrow().clone();
                if let Some(path) = selected_path {
                    let _ = detail.load(&path);
                }
                if !outcome.learned {
                    let notice = adw::AlertDialog::new(
                        Some("Speaker assigned"),
                        Some(
                            "The transcript was updated. The voice model was unavailable, so this name applies to this session only.",
                        ),
                    );
                    notice.add_response("close", "Close");
                    notice.present(Some(&detail.window));
                }
            }
            Err(error) => show_error(&detail.window, "Could not assign speaker", &error),
        }
        glib::ControlFlow::Break
    });
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
    for name in ["pill-ok", "pill-idle", "pill-busy"] {
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

fn session_duration_ms(session: &Session, manifest: &Manifest) -> u64 {
    [
        (AudioSource::Mic, manifest.mic.enabled),
        (AudioSource::System, manifest.system.enabled),
    ]
    .into_iter()
    .filter(|(_, enabled)| *enabled)
    .filter_map(|(source, _)| fs::metadata(session.audio_path(source)).ok())
    .map(|metadata| metadata.len() / 4 * 1000 / u64::from(SAMPLE_RATE))
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

fn append_empty(container: &gtk::Box, text: &str) {
    let label = gtk::Label::new(Some(text));
    label.set_margin_top(28);
    label.set_margin_start(16);
    label.set_margin_end(16);
    label.set_wrap(true);
    label.add_css_class("dim-label");
    container.append(&label);
}

fn open_path(path: &Path) {
    let file = gio::File::for_path(path);
    let _ = gio::AppInfo::launch_default_for_uri(&file.uri(), None::<&gio::AppLaunchContext>);
}

fn show_error(parent: &adw::ApplicationWindow, heading: &str, body: &str) {
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
    fn decoder_metrics_are_human_readable() {
        let (throughput, detail) = transcription_metrics_text(TranscriptionProgress {
            source: AudioSource::System,
            chunk_percent: 75,
            audio_seconds: 45.0,
            elapsed_seconds: 30.0,
            decoded_tokens: 420,
            tokens_per_second: Some(14.25),
        });
        assert_eq!(throughput, "14.2 tokens/s  ·  1.50× realtime");
        assert_eq!(
            detail,
            "system · 75% of current chunk · 45.0 s audio in 30.0 s · 420 tokens"
        );
    }
}
