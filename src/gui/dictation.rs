//! Dictation dialog: drives a `singstone dictate --events` child and shows its text.
//! Nothing is stored; closing the dialog kills the child and drops the text.

use super::config::GuiConfig;
use crate::audio::dictate::{self, DictationEvent};
use adw::prelude::*;
use gtk::glib;
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const POLL: Duration = Duration::from_millis(50);

pub fn present(
    window: &adw::ApplicationWindow,
    config: &Rc<RefCell<GuiConfig>>,
    on_active: Rc<dyn Fn(bool)>,
) {
    let dialog = DictationDialog::new(config, on_active);
    dialog.dialog.present(Some(window));
    dialog.start();
}

#[derive(Default)]
struct Inbox {
    events: VecDeque<DictationEvent>,
    exit: Option<String>,
}

struct Run {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Option<ChildStdin>,
    inbox: Arc<Mutex<Inbox>>,
    settled: bool,
}

struct DictationDialog {
    dialog: adw::Dialog,
    pages: gtk::Stack,
    status: gtk::Label,
    elapsed: gtk::Label,
    level: gtk::LevelBar,
    swedish: gtk::ToggleButton,
    stop: gtk::Button,
    text_card: gtk::Overlay,
    buffer: gtk::TextBuffer,
    empty: gtk::Label,
    copy: gtk::Button,
    error: gtk::Label,
    config: Rc<RefCell<GuiConfig>>,
    on_active: Rc<dyn Fn(bool)>,
    active: Cell<bool>,
    generation: Cell<u64>,
    run: RefCell<Option<Run>>,
}

impl DictationDialog {
    fn new(config: &Rc<RefCell<GuiConfig>>, on_active: Rc<dyn Fn(bool)>) -> Rc<Self> {
        let dialog = adw::Dialog::builder()
            .title("Dictation")
            .content_width(460)
            .presentation_mode(adw::DialogPresentationMode::Floating)
            .build();
        let pages = gtk::Stack::builder()
            .transition_type(gtk::StackTransitionType::Crossfade)
            .margin_top(12)
            .margin_bottom(24)
            .margin_start(24)
            .margin_end(24)
            .build();

        let listening = gtk::Box::new(gtk::Orientation::Vertical, 14);
        listening.set_valign(gtk::Align::Center);
        let status = gtk::Label::new(None);
        status.add_css_class("dim-label");
        status.set_wrap(true);
        let elapsed = gtk::Label::new(Some("0:00"));
        elapsed.add_css_class("title-1");
        elapsed.add_css_class("numeric");
        let level = gtk::LevelBar::for_interval(0.0, 1.0);
        level.set_value(0.0);
        let english = gtk::ToggleButton::with_label("English");
        let swedish = gtk::ToggleButton::with_label("Swedish");
        swedish.set_group(Some(&english));
        if config.borrow().dictation_swedish {
            swedish.set_active(true);
        } else {
            english.set_active(true);
        }
        let languages = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        languages.add_css_class("linked");
        languages.set_halign(gtk::Align::Center);
        languages.append(&english);
        languages.append(&swedish);
        let stop = gtk::Button::with_label("Stop");
        stop.add_css_class("suggested-action");
        stop.add_css_class("pill");
        stop.set_halign(gtk::Align::Center);
        stop.set_size_request(120, -1);
        listening.append(&elapsed);
        listening.append(&status);
        listening.append(&level);
        listening.append(&languages);
        listening.append(&stop);
        pages.add_named(&listening, Some("listening"));

        let transcribing = gtk::Box::new(gtk::Orientation::Vertical, 14);
        transcribing.set_valign(gtk::Align::Center);
        let spinner = gtk::Spinner::new();
        spinner.set_size_request(32, 32);
        spinner.start();
        transcribing.append(&spinner);
        transcribing.append(&gtk::Label::new(Some("Transcribing…")));
        pages.add_named(&transcribing, Some("transcribing"));

        let result = gtk::Box::new(gtk::Orientation::Vertical, 14);
        let text_view = gtk::TextView::builder()
            .wrap_mode(gtk::WrapMode::WordChar)
            .top_margin(10)
            .bottom_margin(10)
            .left_margin(10)
            .right_margin(48)
            .build();
        let buffer = text_view.buffer();
        let text_scroller = gtk::ScrolledWindow::builder()
            .child(&text_view)
            .hscrollbar_policy(gtk::PolicyType::Never)
            .min_content_height(160)
            .max_content_height(360)
            .propagate_natural_height(true)
            .build();
        text_scroller.add_css_class("card");
        let copy = icon_button("edit-copy-symbolic", "Copy");
        copy.set_halign(gtk::Align::End);
        copy.set_valign(gtk::Align::Start);
        copy.set_margin_top(6);
        copy.set_margin_end(6);
        let text_card = gtk::Overlay::builder().child(&text_scroller).build();
        text_card.add_overlay(&copy);
        text_card.set_vexpand(true);
        let empty = gtk::Label::new(Some("No speech detected"));
        empty.add_css_class("title-4");
        empty.add_css_class("dim-label");
        empty.set_vexpand(true);
        let again = icon_button("media-record-symbolic", "Dictate again");
        again.add_css_class("circular");
        again.add_css_class("destructive-action");
        again.set_size_request(48, 48);
        again.set_halign(gtk::Align::Center);
        result.append(&text_card);
        result.append(&empty);
        result.append(&again);
        pages.add_named(&result, Some("result"));

        let failure = gtk::Box::new(gtk::Orientation::Vertical, 12);
        failure.set_valign(gtk::Align::Center);
        let icon = gtk::Image::from_icon_name("dialog-error-symbolic");
        icon.set_pixel_size(48);
        icon.add_css_class("dim-label");
        let heading = gtk::Label::new(Some("Dictation failed"));
        heading.add_css_class("title-3");
        let error = gtk::Label::new(None);
        error.set_wrap(true);
        error.set_justify(gtk::Justification::Center);
        let retry = gtk::Button::with_label("Try again");
        retry.add_css_class("pill");
        retry.set_halign(gtk::Align::Center);
        failure.append(&icon);
        failure.append(&heading);
        failure.append(&error);
        failure.append(&retry);
        pages.add_named(&failure, Some("error"));

        let toolbar = adw::ToolbarView::new();
        toolbar.add_top_bar(&adw::HeaderBar::new());
        toolbar.set_content(Some(&pages));
        dialog.set_child(Some(&toolbar));

        let this = Rc::new(Self {
            dialog,
            pages,
            status,
            elapsed,
            level,
            swedish,
            stop,
            text_card,
            buffer,
            empty,
            copy,
            error,
            config: config.clone(),
            on_active,
            active: Cell::new(false),
            generation: Cell::new(0),
            run: RefCell::new(None),
        });

        let weak = Rc::downgrade(&this);
        this.swedish.connect_toggled(move |button| {
            if let Some(this) = weak.upgrade() {
                this.save_language(button.is_active());
            }
        });
        let weak = Rc::downgrade(&this);
        this.stop.connect_clicked(move |_| {
            if let Some(this) = weak.upgrade() {
                this.request_stop();
            }
        });
        for button in [&again, &retry] {
            let weak = Rc::downgrade(&this);
            button.connect_clicked(move |_| {
                if let Some(this) = weak.upgrade() {
                    this.start();
                }
            });
        }
        let weak = Rc::downgrade(&this);
        this.copy.connect_clicked(move |_| {
            if let Some(this) = weak.upgrade() {
                this.copy_text();
            }
        });
        // Ctrl+C copies the selection, or all of the text when nothing is selected.
        let weak = Rc::downgrade(&this);
        let copy_shortcut = gtk::CallbackAction::new(move |_, _| match weak.upgrade() {
            Some(this) if this.text_card.is_visible() => {
                if this.buffer.has_selection() {
                    this.buffer.copy_clipboard(&this.copy.clipboard());
                } else {
                    this.copy_text();
                }
                glib::Propagation::Stop
            }
            _ => glib::Propagation::Proceed,
        });
        let shortcuts = gtk::ShortcutController::new();
        shortcuts.set_propagation_phase(gtk::PropagationPhase::Capture);
        shortcuts.add_shortcut(gtk::Shortcut::new(
            gtk::ShortcutTrigger::parse_string("<Control>c"),
            Some(copy_shortcut),
        ));
        result.add_controller(shortcuts);
        // The handler owns the dialog state, so it lives exactly as long as the dialog is open.
        let owner = RefCell::new(Some(this.clone()));
        this.dialog.connect_closed(move |_| {
            if let Some(this) = owner.borrow_mut().take() {
                this.discard();
            }
        });
        this
    }

    fn start(self: &Rc<Self>) {
        self.end_run();
        self.status.set_label("Starting the microphone…");
        self.elapsed.set_label("0:00");
        self.level.set_value(0.0);
        self.stop.set_sensitive(true);
        self.buffer.set_text("");
        self.pages.set_visible_child_name("listening");

        let language = if self.swedish.is_active() { "sv" } else { "en" };
        let spawned = std::env::current_exe()
            .map_err(|error| format!("cannot find the Singstone executable: {error}"))
            .and_then(|executable| {
                Command::new(executable)
                    .args(["dictate", "--events", "--language", language])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|error| format!("cannot start dictation: {error}"))
            });
        let mut child = match spawned {
            Ok(child) => child,
            Err(error) => {
                self.show_error(&error);
                return;
            }
        };
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let child = Arc::new(Mutex::new(Some(child)));
        let inbox = Arc::new(Mutex::new(Inbox::default()));
        {
            let child = child.clone();
            let inbox = inbox.clone();
            std::thread::spawn(move || read_events(stdout, &child, &inbox));
        }
        *self.run.borrow_mut() = Some(Run {
            child,
            stdin,
            inbox,
            settled: false,
        });
        self.set_active(true);

        let generation = self.generation.get() + 1;
        self.generation.set(generation);
        let weak = Rc::downgrade(self);
        glib::timeout_add_local(POLL, move || match weak.upgrade() {
            Some(this) if this.generation.get() == generation => this.poll(),
            _ => glib::ControlFlow::Break,
        });
    }

    fn poll(&self) -> glib::ControlFlow {
        let generation = self.generation.get();
        let (events, exit) = {
            let run = self.run.borrow();
            let Some(run) = run.as_ref() else {
                return glib::ControlFlow::Break;
            };
            let mut inbox = run.inbox.lock().expect("dictation inbox mutex");
            (std::mem::take(&mut inbox.events), inbox.exit.take())
        };
        for event in events {
            self.apply(event);
            if self.generation.get() != generation {
                return glib::ControlFlow::Break;
            }
        }
        let Some(exit) = exit else {
            return glib::ControlFlow::Continue;
        };
        let settled = self.run.borrow_mut().take().is_some_and(|run| run.settled);
        self.set_active(false);
        if !settled {
            self.show_error(&format!("Dictation stopped unexpectedly ({exit})."));
        }
        glib::ControlFlow::Break
    }

    fn apply(&self, event: DictationEvent) {
        match event {
            DictationEvent::Listening => self.status.set_label("Listening"),
            DictationEvent::Level { peak, seconds } => {
                self.level.set_value(f64::from(peak));
                let seconds = seconds as u64;
                self.elapsed
                    .set_label(&format!("{}:{:02}", seconds / 60, seconds % 60));
            }
            DictationEvent::LimitReached => {
                self.status
                    .set_label("Reached the 10-minute limit. Press Stop to transcribe.");
                self.level.set_value(0.0);
            }
            DictationEvent::Transcribing => self.pages.set_visible_child_name("transcribing"),
            DictationEvent::Text { text } => {
                self.settle();
                let empty = text.trim().is_empty();
                self.buffer.set_text(&text);
                self.text_card.set_visible(!empty);
                self.empty.set_visible(empty);
                self.pages.set_visible_child_name("result");
            }
            DictationEvent::Failure { error } => {
                self.settle();
                self.show_error(&error);
            }
        }
    }

    fn settle(&self) {
        if let Some(run) = self.run.borrow_mut().as_mut() {
            run.settled = true;
        }
    }

    fn request_stop(&self) {
        let language = if self.swedish.is_active() { "sv" } else { "en" };
        let written = self
            .run
            .borrow_mut()
            .as_mut()
            .and_then(|run| run.stdin.as_mut())
            .map(|stdin| writeln!(stdin, "{language}").and_then(|()| stdin.flush()));
        match written {
            Some(Ok(())) => {
                self.stop.set_sensitive(false);
                self.pages.set_visible_child_name("transcribing");
            }
            Some(Err(error)) => self.show_error(&format!("Cannot stop dictation: {error}")),
            None => {}
        }
    }

    fn copy_text(&self) {
        let (start, end) = self.buffer.bounds();
        self.copy
            .clipboard()
            .set_text(&self.buffer.text(&start, &end, false));
        self.copy.set_icon_name("object-select-symbolic");
        self.copy.set_tooltip_text(Some("Copied"));
        let button = self.copy.downgrade();
        glib::timeout_add_local_once(Duration::from_millis(1500), move || {
            if let Some(button) = button.upgrade() {
                button.set_icon_name("edit-copy-symbolic");
                button.set_tooltip_text(Some("Copy"));
            }
        });
    }

    fn save_language(&self, swedish: bool) {
        let mut updated = self.config.borrow().clone();
        if updated.dictation_swedish == swedish {
            return;
        }
        updated.dictation_swedish = swedish;
        match updated.save() {
            Ok(()) => *self.config.borrow_mut() = updated,
            Err(error) => eprintln!("warning: cannot save the dictation language: {error}"),
        }
    }

    fn show_error(&self, message: &str) {
        self.end_run();
        self.error.set_label(message);
        self.pages.set_visible_child_name("error");
    }

    fn end_run(&self) {
        self.generation.set(self.generation.get() + 1);
        if let Some(run) = self.run.borrow_mut().take()
            && let Ok(mut child) = run.child.lock()
            && let Some(child) = child.as_mut()
        {
            // The event reader reaps it once its stdout closes.
            let _ = child.kill();
        }
        self.set_active(false);
    }

    fn discard(&self) {
        self.end_run();
        self.buffer.set_text("");
    }

    fn set_active(&self, active: bool) {
        if self.active.replace(active) != active {
            (self.on_active)(active);
        }
    }
}

fn icon_button(icon: &str, label: &str) -> gtk::Button {
    let button = gtk::Button::from_icon_name(icon);
    button.set_tooltip_text(Some(label));
    button.update_property(&[gtk::accessible::Property::Label(label)]);
    button
}

fn read_events(
    stdout: Option<impl std::io::Read>,
    child: &Mutex<Option<Child>>,
    inbox: &Mutex<Inbox>,
) {
    if let Some(stdout) = stdout {
        let mut reader = BufReader::new(stdout);
        let mut line = Vec::new();
        while matches!(reader.read_until(b'\n', &mut line), Ok(count) if count > 0) {
            // Anything that is not an event could hold dictated text, so it is dropped unread.
            if let Ok(event) = dictate::parse_event(String::from_utf8_lossy(&line).trim_end()) {
                inbox
                    .lock()
                    .expect("dictation inbox mutex")
                    .events
                    .push_back(event);
            }
            line.clear();
        }
    }
    let status = child
        .lock()
        .ok()
        .and_then(|mut child| child.take())
        .map(|mut child| child.wait());
    let exit = match status {
        Some(Ok(status)) => match (status.code(), status.signal()) {
            (Some(code), _) => format!("exit status {code}"),
            (None, Some(signal)) => format!("signal {signal}"),
            _ => "unknown status".into(),
        },
        Some(Err(error)) => format!("cannot reap the dictation process: {error}"),
        None => "no process".into(),
    };
    inbox.lock().expect("dictation inbox mutex").exit = Some(exit);
}
