//! Transient dictation: microphone audio stays in memory, is transcribed once
//! and is never written anywhere.

use crate::audio::{capture, pipewire as pw_helpers};
use crate::cli::DictateArgs;
use crate::merge::{process, utterances};
use crate::transcription::{self, Transcriber, whisper::WhisperTranscriber};
use crate::types::{AudioSource, SAMPLE_RATE};
use crate::{model_setup, models};
use crossbeam_queue::ArrayQueue;
use pipewire as pw;
use pw::properties::properties;
use rustix::process::{Pid, Signal};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::ffi::OsString;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

const MAX_SAMPLES: usize = 10 * 60 * SAMPLE_RATE as usize;
const TICK: Duration = Duration::from_millis(100);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const SILENT_STREAM_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictationLanguage {
    English,
    Swedish,
}

impl DictationLanguage {
    pub fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Swedish => "sv",
        }
    }
}

impl FromStr for DictationLanguage {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "en" => Ok(Self::English),
            "sv" => Ok(Self::Swedish),
            _ => Err(format!("unknown language {value:?}; use `en` or `sv`")),
        }
    }
}

/// An empty stop line keeps the default language; otherwise it must name one.
pub fn parse_stop_line(line: &str) -> Result<Option<DictationLanguage>, String> {
    let line = line.trim();
    if line.is_empty() {
        Ok(None)
    } else {
        line.parse().map(Some)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum DictationEvent {
    Listening,
    Level { peak: f32, seconds: f64 },
    LimitReached,
    Transcribing,
    Text { text: String },
    Failure { error: String },
}

pub fn parse_event(line: &str) -> serde_json::Result<DictationEvent> {
    serde_json::from_str(line)
}

enum Control {
    Stop(Option<DictationLanguage>),
    Cancel,
    Failed(String),
    Transcribed(Result<String, String>),
}

enum Outcome {
    Text(String),
    Cancelled,
}

#[derive(Clone, Copy)]
struct Output {
    events: bool,
}

impl Output {
    fn event(self, event: &DictationEvent) {
        if !self.events {
            return;
        }
        let written = serde_json::to_string(event)
            .map_err(io::Error::from)
            .and_then(|line| {
                let mut stdout = io::stdout().lock();
                writeln!(stdout, "{line}")?;
                stdout.flush()
            });
        // Nobody reads the events once the GUI is gone.
        if written.is_err() {
            std::process::exit(1);
        }
    }

    fn note(self, message: &str) {
        if !self.events {
            eprintln!("{message}");
        }
    }
}

pub fn run(args: DictateArgs) -> ExitCode {
    let output = Output {
        events: args.events,
    };
    let result = if args.events {
        let death_signal = rustix::process::set_parent_process_death_signal(Some(Signal::KILL));
        let orphaned = rustix::process::getppid().is_none_or(|parent| parent == Pid::INIT);
        match death_signal {
            Err(error) => Err(format!("cannot arm parent-death signal: {error}")),
            Ok(()) if orphaned => Err("dictation was orphaned before it started".into()),
            Ok(()) => dictate(&args, output),
        }
    } else {
        dictate(&args, output)
    };
    match result {
        Ok(Outcome::Text(text)) => {
            if args.events {
                output.event(&DictationEvent::Text { text });
            } else if text.is_empty() {
                eprintln!("no speech detected");
            } else {
                println!("{text}");
            }
            ExitCode::SUCCESS
        }
        Ok(Outcome::Cancelled) => {
            output.note("dictation cancelled");
            ExitCode::FAILURE
        }
        Err(error) => {
            if args.events {
                output.event(&DictationEvent::Failure { error });
            } else {
                eprintln!("error: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

struct ModelOptions {
    flag: Option<PathBuf>,
    lock: Option<PathBuf>,
    allow_unverified: bool,
}

impl ModelOptions {
    fn model(&self, language: DictationLanguage) -> Result<PathBuf, String> {
        dictation_model(self.flag.as_deref(), language, |name| std::env::var_os(name)).ok_or_else(
            || {
                format!(
                    "no Whisper model is configured for {}; set --whisper-model or SINGSTONE_WHISPER_MODEL",
                    language.code()
                )
            },
        )
    }

    fn prepare(&self, model: &Path) -> Result<(), String> {
        model_setup::ensure_models(&[("transcription", model)])
            .map_err(|error| error.to_string())?;
        models::verify_model(
            model,
            "transcription",
            self.lock.as_deref(),
            self.allow_unverified,
        )
        .map_err(|error| format!("Whisper model {}: {error}", model.display()))
    }
}

/// `--whisper-model`, else the language's own model, else `SINGSTONE_WHISPER_MODEL`.
fn dictation_model(
    flag: Option<&Path>,
    language: DictationLanguage,
    env: impl Fn(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    flag.map(Path::to_path_buf).or_else(|| {
        let generic = env("SINGSTONE_WHISPER_MODEL").map(PathBuf::from);
        transcription::language_model_from(language == DictationLanguage::Swedish, generic, env)
    })
}

struct State {
    samples: Vec<f32>,
    started: Instant,
    listening: bool,
    limited: bool,
    last_audio: Instant,
    transcribing: bool,
    outcome: Option<Result<Outcome, String>>,
}

type CaptureSlot = RefCell<
    Option<(
        pw::stream::StreamListener<capture::CaptureData>,
        pw::stream::StreamRc,
    )>,
>;

fn dictate(args: &DictateArgs, output: Output) -> Result<Outcome, String> {
    let models = Arc::new(ModelOptions {
        flag: args.whisper_model.clone(),
        lock: args.models_lock.clone(),
        allow_unverified: args.allow_unverified_models,
    });
    let default_language = args.language;
    let threads = process::thread_count(args.threads);

    pw::init();
    let pipewire_error = |error: pw::Error| format!("cannot connect to PipeWire: {error}");
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pipewire_error)?;
    let state = Rc::new(RefCell::new(State {
        samples: Vec::new(),
        started: Instant::now(),
        listening: false,
        limited: false,
        last_audio: Instant::now(),
        transcribing: false,
        outcome: None,
    }));
    // Registered before any thread starts, so every thread inherits the blocked signals.
    let signal_sources = [pw::loop_::Signal::INT, pw::loop_::Signal::TERM].map(|signal| {
        let state = state.clone();
        let weak_loop = mainloop.downgrade();
        mainloop.loop_().add_signal_local(signal, move || {
            state
                .borrow_mut()
                .outcome
                .get_or_insert(Ok(Outcome::Cancelled));
            if let Some(mainloop) = weak_loop.upgrade() {
                mainloop.quit();
            }
        })
    });

    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pipewire_error)?;
    let core = context
        .connect_rc(Some(properties! {
            *pw::keys::REMOTE_NAME => "pipewire-0"
        }))
        .map_err(pipewire_error)?;

    let queue = Arc::new(ArrayQueue::new(capture::QUEUE_CAPACITY));
    let format_ok = Arc::new(AtomicBool::new(false));
    let capture_error = Arc::new(Mutex::new(None));
    let stream = pw_helpers::new_capture_stream(&core, "singstone-dictation", &args.mic, false)
        .map_err(|error| format!("cannot open the microphone: {error}"))?;
    let listener = capture::listener(
        &stream,
        capture::CaptureData::new(
            queue.clone(),
            Arc::new(AtomicU64::new(0)),
            format_ok.clone(),
            capture_error.clone(),
            Arc::new(AtomicU32::new(0)),
        ),
    )
    .map_err(|error| format!("cannot open the microphone: {error}"))?;
    pw_helpers::connect(&stream).map_err(|error| format!("cannot open the microphone: {error}"))?;
    // Tuple fields drop in order, so the listener detaches before its stream is freed.
    let capture: Rc<CaptureSlot> = Rc::new(RefCell::new(Some((listener, stream))));

    let (control, inbox) = mpsc::channel();
    {
        let control = control.clone();
        std::thread::spawn(move || {
            let _ = control.send(read_stop(io::stdin().lock()));
        });
    }

    let tick_state = state.clone();
    let tick_loop = mainloop.downgrade();
    let timer = mainloop.loop_().add_timer(move |_| {
        let mut state = tick_state.borrow_mut();
        if state.outcome.is_none() {
            state.outcome = tick(
                &mut state,
                &Tick {
                    queue: &queue,
                    format_ok: &format_ok,
                    capture_error: &capture_error,
                    capture: &capture,
                    inbox: &inbox,
                    control: &control,
                    models: &models,
                    default_language,
                    threads,
                    output,
                },
            );
        }
        if state.outcome.is_some()
            && let Some(mainloop) = tick_loop.upgrade()
        {
            mainloop.quit();
        }
    });
    let _ = timer.update_timer(Some(TICK), Some(TICK));

    mainloop.run();
    drop(timer);
    drop(signal_sources);
    let outcome = state.borrow_mut().outcome.take();
    outcome.unwrap_or(Ok(Outcome::Cancelled))
}

struct Tick<'a> {
    queue: &'a ArrayQueue<capture::Block>,
    format_ok: &'a AtomicBool,
    capture_error: &'a Mutex<Option<String>>,
    capture: &'a CaptureSlot,
    inbox: &'a mpsc::Receiver<Control>,
    control: &'a mpsc::Sender<Control>,
    models: &'a Arc<ModelOptions>,
    default_language: DictationLanguage,
    threads: usize,
    output: Output,
}

fn tick(state: &mut State, context: &Tick) -> Option<Result<Outcome, String>> {
    if !state.transcribing {
        if let Some(error) = context
            .capture_error
            .lock()
            .ok()
            .and_then(|mut error| error.take())
        {
            return Some(Err(format!("microphone: {error}")));
        }
        let peak = drain(context.queue, &mut state.samples);
        if peak.is_some() {
            state.last_audio = Instant::now();
        }
        if !state.listening && context.format_ok.load(Ordering::Acquire) {
            state.listening = true;
            state.last_audio = Instant::now();
            context.output.event(&DictationEvent::Listening);
            context
                .output
                .note("Listening. Press Enter to stop, Ctrl+C to cancel.");
        }
        if !state.listening && state.started.elapsed() > STARTUP_TIMEOUT {
            return Some(Err("the microphone did not start".into()));
        }
        // A disconnected stream goes quiet instead of reporting an error.
        if state.listening && !state.limited && state.last_audio.elapsed() > SILENT_STREAM_TIMEOUT {
            return Some(Err("the microphone stopped delivering audio".into()));
        }
        if state.listening {
            context.output.event(&DictationEvent::Level {
                peak: peak.unwrap_or(0.0),
                seconds: state.samples.len() as f64 / f64::from(SAMPLE_RATE),
            });
        }
        if !state.limited && state.samples.len() >= MAX_SAMPLES {
            state.limited = true;
            context.capture.borrow_mut().take();
            context.output.event(&DictationEvent::LimitReached);
            context
                .output
                .note("Reached the 10-minute limit. Press Enter to transcribe, Ctrl+C to cancel.");
        }
    }
    while let Ok(message) = context.inbox.try_recv() {
        match message {
            Control::Failed(error) => return Some(Err(error)),
            Control::Cancel if !state.transcribing => return Some(Ok(Outcome::Cancelled)),
            Control::Stop(language) if !state.transcribing => {
                context.capture.borrow_mut().take();
                drain(context.queue, &mut state.samples);
                state.transcribing = true;
                let language = language.unwrap_or(context.default_language);
                let model = match context.models.model(language) {
                    Ok(model) => model,
                    Err(error) => return Some(Err(error)),
                };
                context.output.event(&DictationEvent::Transcribing);
                context.output.note("Transcribing.");
                let samples = std::mem::take(&mut state.samples);
                let control = context.control.clone();
                let models = context.models.clone();
                let threads = context.threads;
                std::thread::spawn(move || {
                    let result = (|| {
                        models.prepare(&model)?;
                        transcribe(&model, language, threads, &samples)
                    })();
                    let _ = control.send(Control::Transcribed(result));
                });
            }
            Control::Transcribed(result) => return Some(result.map(Outcome::Text)),
            Control::Stop(_) | Control::Cancel => {}
        }
    }
    None
}

/// Moves captured blocks into `samples`; the peak is `None` when no block arrived.
fn drain(queue: &ArrayQueue<capture::Block>, samples: &mut Vec<f32>) -> Option<f32> {
    let mut peak = None;
    while let Some(block) = queue.pop() {
        let block = &block.samples[..block.len as usize];
        let peak = peak.get_or_insert(0.0f32);
        *peak = block
            .iter()
            .fold(*peak, |peak, sample| peak.max(sample.abs()));
        let room = MAX_SAMPLES.saturating_sub(samples.len());
        samples.extend_from_slice(&block[..block.len().min(room)]);
    }
    peak.map(|peak| peak.min(1.0))
}

fn transcribe(
    model: &Path,
    language: DictationLanguage,
    threads: usize,
    samples: &[f32],
) -> Result<String, String> {
    let mut transcriber = WhisperTranscriber::new(
        model,
        AudioSource::Mic,
        language.code().into(),
        threads,
        None,
    )
    .map_err(|error| format!("cannot load Whisper model {}: {error}", model.display()))?;
    let words = transcriber
        .transcribe(samples)
        .map_err(|error| format!("transcription failed: {error}"))?;
    Ok(utterances::words_text(&words))
}

/// Reads the stop line; end of input before a complete line cancels.
fn read_stop(mut input: impl BufRead) -> Control {
    let mut line = String::new();
    match input.read_line(&mut line) {
        Ok(_) if !line.ends_with('\n') => Control::Cancel,
        Ok(_) => match parse_stop_line(&line) {
            Ok(language) => Control::Stop(language),
            Err(error) => Control::Failed(format!("invalid stop line: {error}")),
        },
        Err(error) => Control::Failed(format!("cannot read the stop line: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_line_picks_the_language_or_keeps_the_default() {
        assert_eq!(parse_stop_line("\n"), Ok(None));
        assert_eq!(
            parse_stop_line("sv\n"),
            Ok(Some(DictationLanguage::Swedish))
        );
        assert_eq!(
            parse_stop_line("en\r\n"),
            Ok(Some(DictationLanguage::English))
        );
        assert!(parse_stop_line("stop\n").is_err());
    }

    #[test]
    fn end_of_input_before_a_complete_stop_line_cancels() {
        assert!(matches!(read_stop(&b""[..]), Control::Cancel));
        assert!(matches!(read_stop(&b"en"[..]), Control::Cancel));
        assert!(matches!(
            read_stop(&b"sv\n"[..]),
            Control::Stop(Some(DictationLanguage::Swedish))
        ));
    }

    #[test]
    fn explicit_model_beats_language_model_beats_generic_model() {
        let env = |name: &str| match name {
            "SINGSTONE_WHISPER_MODEL" => Some(OsString::from("generic.bin")),
            "SINGSTONE_WHISPER_MODEL_SWEDISH" => Some(OsString::from("kb.bin")),
            _ => None,
        };
        let flag = Some(Path::new("custom.bin"));
        assert_eq!(
            dictation_model(flag, DictationLanguage::Swedish, env),
            Some(PathBuf::from("custom.bin"))
        );
        assert_eq!(
            dictation_model(None, DictationLanguage::Swedish, env),
            Some(PathBuf::from("kb.bin"))
        );
        assert_eq!(
            dictation_model(None, DictationLanguage::English, env),
            Some(PathBuf::from("generic.bin"))
        );
    }
}
