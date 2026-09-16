use crate::audio::{capture, pipewire as pw_helpers, writer};
use crate::cli::RecordArgs;
use crate::format::jsonl::JsonlAppender;
use crate::screenshot::watcher::WatcherHandle;
use crate::session::{LocalTime, Session, local_time_from_unix, session_dir_name};
use crate::types::{AudioSource, FORMAT_VERSION, Manifest, SAMPLE_RATE, SessionState, StreamInfo};
use pipewire as pw;
use pw::properties::properties;
use std::cell::RefCell;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type StreamStats = (
    std::sync::Arc<std::sync::atomic::AtomicU64>,
    std::sync::Arc<std::sync::atomic::AtomicU64>,
    std::sync::Arc<std::sync::Mutex<Option<String>>>,
);
type StreamStartup = (
    std::sync::Arc<std::sync::atomic::AtomicU64>,
    std::sync::Arc<std::sync::Mutex<Option<String>>>,
);

pub fn run(args: RecordArgs) -> Result<(), Box<dyn std::error::Error>> {
    run_inner(args, None, None).map(|_| ())
}

/// Lock-free capture health read by the GUI while PipeWire owns the RT path.
#[derive(Clone, Default)]
pub struct RecordingTelemetry {
    pub mic_level: Arc<AtomicU32>,
    pub system_level: Arc<AtomicU32>,
    pub screenshots: Arc<AtomicU64>,
    pub session_dir: Arc<Mutex<Option<PathBuf>>>,
}

impl RecordingTelemetry {
    pub fn level(value: &AtomicU32) -> f32 {
        f32::from_bits(value.load(Ordering::Relaxed)).clamp(0.0, 1.0)
    }
}

pub fn run_with_telemetry(
    args: RecordArgs,
    stop: Arc<AtomicBool>,
    telemetry: RecordingTelemetry,
) -> Result<Session, Box<dyn std::error::Error>> {
    run_inner(args, Some(stop), Some(telemetry))
}

fn run_inner(
    args: RecordArgs,
    stop: Option<Arc<AtomicBool>>,
    telemetry: Option<RecordingTelemetry>,
) -> Result<Session, Box<dyn std::error::Error>> {
    if args.mic == "none" && args.system == "none" {
        return Err("--mic and --system cannot both be `none`".into());
    }
    if let Some(duration) = args.duration
        && (!duration.is_finite() || duration <= 0.0)
    {
        return Err("--duration must be a finite positive number".into());
    }

    let local_time = local_now()?;
    let session = Session::create(args.output_dir.join(session_dir_name(&local_time)))?;
    if let Some(telemetry) = &telemetry
        && let Ok(mut path) = telemetry.session_dir.lock()
    {
        *path = Some(session.dir.clone());
    }
    let mut manifest = Manifest {
        format_version: FORMAT_VERSION,
        state: SessionState::Recording,
        started_wallclock: local_time.rfc3339(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
        sample_format: "f32le".to_owned(),
        mic: stream_info(&args.mic),
        system: stream_info(&args.system),
        local_speaker: args.local_speaker,
        screenshot_dir: args
            .screenshots
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
    };
    session.write_manifest(&manifest)?;
    drop(JsonlAppender::create(&session.screenshots_index_path())?);

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(Some(properties! {
        *pw::keys::REMOTE_NAME => "pipewire-0"
    }))?;
    let t0_ns = pw_helpers::monotonic_ns();

    let mut mic_writer = if manifest.mic.enabled {
        Some(writer::WriterHandle::start(
            &session,
            AudioSource::Mic,
            t0_ns,
        )?)
    } else {
        None
    };
    let mut system_writer = if manifest.system.enabled {
        Some(writer::WriterHandle::start(
            &session,
            AudioSource::System,
            t0_ns,
        )?)
    } else {
        None
    };

    let mic_slot = Rc::new(RefCell::new(None));
    let system_slot = Rc::new(RefCell::new(None));
    let mic_start = mic_writer.as_ref().map(|writer| StreamStart {
        name: "singstone-microphone",
        target: args.mic.clone(),
        capture_sink: false,
        data: writer.capture_data_with_level(
            telemetry
                .as_ref()
                .map_or_else(|| writer.level.clone(), |value| value.mic_level.clone()),
        ),
    });
    let system_start = system_writer.as_ref().map(|writer| StreamStart {
        name: "singstone-system",
        target: args.system.clone(),
        capture_sink: true,
        data: writer.capture_data_with_level(
            telemetry
                .as_ref()
                .map_or_else(|| writer.level.clone(), |value| value.system_level.clone()),
        ),
    });
    let connect_core = core.clone();
    let connect_mic_slot = Rc::clone(&mic_slot);
    let connect_system_slot = Rc::clone(&system_slot);
    let connect_timer = mainloop.loop_().add_timer(move |_| {
        start_stream(&connect_core, &connect_mic_slot, mic_start.as_ref());
        start_stream(&connect_core, &connect_system_slot, system_start.as_ref());
    });
    let _ = connect_timer.update_timer(Some(Duration::from_millis(10)), None);

    let mut screenshot_watcher = match args.screenshots.as_deref() {
        Some(path) => match WatcherHandle::start_with_counter(
            path,
            &session,
            args.screenshot_ext,
            t0_ns,
            telemetry.as_ref().map_or_else(
                || Arc::new(AtomicU64::new(0)),
                |value| value.screenshots.clone(),
            ),
        ) {
            Ok(watcher) => Some(watcher),
            Err(error) => {
                eprintln!("warning: screenshot watcher could not start: {error}");
                None
            }
        },
        None => None,
    };

    let loop_for_int = mainloop.downgrade();
    let _sig_int = mainloop
        .loop_()
        .add_signal_local(pw::loop_::Signal::INT, move || {
            if let Some(mainloop) = loop_for_int.upgrade() {
                mainloop.quit();
            }
        });
    let loop_for_term = mainloop.downgrade();
    let _sig_term = mainloop
        .loop_()
        .add_signal_local(pw::loop_::Signal::TERM, move || {
            if let Some(mainloop) = loop_for_term.upgrade() {
                mainloop.quit();
            }
        });

    let duration_timer = args.duration.map(|seconds| {
        let loop_for_timer = mainloop.downgrade();
        let timer = mainloop.loop_().add_timer(move |_| {
            if let Some(mainloop) = loop_for_timer.upgrade() {
                mainloop.quit();
            }
        });
        let _ = timer.update_timer(Some(Duration::from_secs_f64(seconds)), None);
        timer
    });
    let stop_timer = stop.map(|stop| {
        let loop_for_stop = mainloop.downgrade();
        let timer = mainloop.loop_().add_timer(move |_| {
            if stop.load(Ordering::Acquire)
                && let Some(mainloop) = loop_for_stop.upgrade()
            {
                mainloop.quit();
            }
        });
        let _ = timer.update_timer(
            Some(Duration::from_millis(100)),
            Some(Duration::from_millis(100)),
        );
        timer
    });

    let mic_stats = mic_writer.as_ref().map(|writer| {
        (
            writer.samples.clone(),
            writer.dropped.clone(),
            writer.error.clone(),
        )
    });
    let system_stats = system_writer.as_ref().map(|writer| {
        (
            writer.samples.clone(),
            writer.dropped.clone(),
            writer.error.clone(),
        )
    });
    let mic_health = mic_writer.as_ref().map(|writer| writer.error.clone());
    let system_health = system_writer.as_ref().map(|writer| writer.error.clone());
    let mic_startup = mic_writer
        .as_ref()
        .map(|writer| (writer.samples.clone(), writer.error.clone()));
    let system_startup = system_writer
        .as_ref()
        .map(|writer| (writer.samples.clone(), writer.error.clone()));
    let startup_timer = mainloop.loop_().add_timer(move |_| {
        mark_no_audio(&mic_startup);
        mark_no_audio(&system_startup);
    });
    let _ = startup_timer.update_timer(Some(Duration::from_secs(2)), None);
    let loop_for_health = mainloop.downgrade();
    let health_timer = mainloop.loop_().add_timer(move |_| {
        if all_streams_failed(&mic_health, &system_health)
            && let Some(mainloop) = loop_for_health.upgrade()
        {
            mainloop.quit();
        }
    });
    let _ = health_timer.update_timer(
        Some(Duration::from_millis(100)),
        Some(Duration::from_millis(100)),
    );
    let status_start = std::time::Instant::now();
    let status_timer = mainloop.loop_().add_timer(move |_| {
        let (mic_samples, mic_drops, mic_error) = stats(&mic_stats);
        let (system_samples, system_drops, system_error) = stats(&system_stats);
        eprintln!(
            "recording {:.0}s: mic {} samples/{} drops{}, system {} samples/{} drops{}",
            status_start.elapsed().as_secs_f64(),
            mic_samples,
            mic_drops,
            status_error(&mic_error),
            system_samples,
            system_drops,
            status_error(&system_error),
        );
    });
    let _ = status_timer.update_timer(Some(Duration::from_secs(5)), Some(Duration::from_secs(5)));

    eprintln!("recording to {}", session.dir.display());
    if mic_writer.is_some() || system_writer.is_some() {
        mainloop.run();
    }
    drop(stop_timer);
    drop(duration_timer);
    drop(status_timer);
    drop(health_timer);
    drop(startup_timer);
    drop(connect_timer);
    mic_slot.borrow_mut().take();
    system_slot.borrow_mut().take();

    if let Some(watcher) = &mut screenshot_watcher
        && let Err(error) = watcher.finish()
    {
        eprintln!("warning: screenshot watcher stopped with an error: {error}");
    }
    let mut finish_errors = Vec::new();
    if let Some(writer) = &mut mic_writer {
        mark_missing_format(writer);
        if let Err(error) = writer.finish() {
            let message = format!("mic writer: {error}");
            capture::set_error(&writer.error, message.clone());
            finish_errors.push(message);
        }
        manifest.mic.error = writer.error_message();
    }
    if let Some(writer) = &mut system_writer {
        mark_missing_format(writer);
        if let Err(error) = writer.finish() {
            let message = format!("system writer: {error}");
            capture::set_error(&writer.error, message.clone());
            finish_errors.push(message);
        }
        manifest.system.error = writer.error_message();
    }
    manifest.state = SessionState::Stopped;
    if let Err(error) = session.write_manifest(&manifest) {
        finish_errors.push(format!("manifest: {error}"));
    }
    report_stream_error("mic", &manifest.mic);
    report_stream_error("system", &manifest.system);
    let mic_samples = mic_writer
        .as_ref()
        .map_or(0, |writer| writer.samples.load(Ordering::Relaxed));
    let system_samples = system_writer
        .as_ref()
        .map_or(0, |writer| writer.samples.load(Ordering::Relaxed));
    let every_stream_failed = [
        (manifest.mic.enabled, &manifest.mic, mic_samples),
        (manifest.system.enabled, &manifest.system, system_samples),
    ]
    .into_iter()
    .filter(|(enabled, _, _)| *enabled)
    .all(|(_, info, samples)| info.error.is_some() || samples == 0);
    eprintln!("stopped {}", session.dir.display());
    if every_stream_failed {
        finish_errors.push("all enabled streams failed or recorded zero samples".to_owned());
    }
    if finish_errors.is_empty() {
        Ok(session)
    } else {
        Err(io::Error::other(finish_errors.join("; ")).into())
    }
}

fn mark_missing_format(writer: &writer::WriterHandle) {
    if !writer.format_ok.load(Ordering::Acquire) && writer.error_message().is_none() {
        capture::set_error(
            &writer.error,
            "stream stopped before format negotiation completed".to_owned(),
        );
    }
}

struct StreamStart {
    name: &'static str,
    target: String,
    capture_sink: bool,
    data: capture::CaptureData,
}

struct StreamSlot {
    // Fields drop in declaration order; detach the listener before freeing its stream.
    _listener: pw::stream::StreamListener<capture::CaptureData>,
    _stream: pw::stream::StreamRc,
}

fn start_stream(
    core: &pw::core::CoreRc,
    slot: &RefCell<Option<StreamSlot>>,
    start: Option<&StreamStart>,
) {
    let Some(start) = start else {
        return;
    };
    let result = (|| {
        let stream =
            pw_helpers::new_capture_stream(core, start.name, &start.target, start.capture_sink)?;
        let listener = capture::listener(
            &stream,
            capture::CaptureData::new(
                start.data.queue.clone(),
                start.data.dropped.clone(),
                start.data.format_ok.clone(),
                start.data.error.clone(),
                start.data.level.clone(),
            ),
        )?;
        pw_helpers::connect(&stream)?;
        Ok::<_, Box<dyn std::error::Error>>(StreamSlot {
            _listener: listener,
            _stream: stream,
        })
    })();
    match result {
        Ok(active) => *slot.borrow_mut() = Some(active),
        Err(error) => {
            capture::set_error(
                &start.data.error,
                format!("failed to start PipeWire stream: {error}"),
            );
            start.data.format_ok.store(false, Ordering::Release);
        }
    }
}

fn stream_info(value: &str) -> StreamInfo {
    StreamInfo {
        enabled: value != "none",
        pipewire_node: (value != "none").then(|| value.to_owned()),
        error: None,
    }
}

fn stats(stats: &Option<StreamStats>) -> (u64, u64, Option<String>) {
    stats
        .as_ref()
        .map(|(samples, drops, error)| {
            (
                samples.load(Ordering::Relaxed),
                drops.load(Ordering::Relaxed),
                error.lock().ok().and_then(|error| error.clone()),
            )
        })
        .unwrap_or((0, 0, None))
}

fn all_streams_failed(
    mic: &Option<std::sync::Arc<std::sync::Mutex<Option<String>>>>,
    system: &Option<std::sync::Arc<std::sync::Mutex<Option<String>>>>,
) -> bool {
    let mut enabled = 0;
    let mut failed = 0;
    for error in [mic, system].into_iter().flatten() {
        enabled += 1;
        failed += usize::from(error.lock().is_ok_and(|error| error.is_some()));
    }
    enabled > 0 && failed == enabled
}

fn status_error(error: &Option<String>) -> String {
    error
        .as_ref()
        .map(|message| format!(" [error: {message}]"))
        .unwrap_or_default()
}

fn report_stream_error(source: &str, info: &StreamInfo) {
    if let Some(error) = &info.error {
        eprintln!("warning: {source} stream: {error}");
    }
}

fn mark_no_audio(stream: &Option<StreamStartup>) {
    if let Some((samples, error)) = stream
        && samples.load(Ordering::Relaxed) == 0
    {
        capture::set_error(error, "stream produced no audio samples".to_owned());
    }
}

fn local_now() -> Result<LocalTime, io::Error> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| io::Error::other(format!("system clock is before Unix epoch: {error}")))?
        .as_secs();
    Ok(local_time_from_unix(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::DirBuilderExt;

    #[test]
    fn manifest_round_trip() {
        let root = std::env::temp_dir().join(format!("singstone-session-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .expect("create temp root");
        let session = Session::create(root.join("session")).expect("create session");
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            state: SessionState::Recording,
            started_wallclock: "2026-09-14T10:30:00+00:00".to_owned(),
            sample_rate: SAMPLE_RATE,
            channels: 1,
            sample_format: "f32le".to_owned(),
            mic: stream_info("default"),
            system: stream_info("none"),
            local_speaker: "Me".to_owned(),
            screenshot_dir: None,
        };
        session.write_manifest(&manifest).expect("write manifest");
        assert_eq!(session.read_manifest().expect("read manifest"), manifest);
        fs::remove_dir_all(root).expect("remove temp session");
    }
}
