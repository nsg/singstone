#![forbid(unsafe_code)]

mod audio;
mod cli;
mod diarization;
mod format;
mod gui;
mod meeting;
mod merge;
mod model_setup;
mod models;
mod screenshot;
mod session;
mod speaker;
mod transcription;
mod types;

use clap::Parser;
use rustix::process::{Pid, Signal};
use std::io::{self, BufWriter};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

fn main() -> ExitCode {
    if std::env::args_os().len() == 1 {
        return gui::run();
    }
    let cli = cli::Cli::parse();
    if matches!(&cli.command, cli::Command::Process(args) if args.events) {
        let cli::Command::Process(args) = cli.command else {
            unreachable!()
        };
        return run_event_worker(args);
    }
    if !matches!(
        cli.command,
        cli::Command::ModelSetup | cli::Command::ModelSetupCheck
    ) && let Err(e) = model_setup::ensure_available(&cli.command)
    {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let result = match cli.command {
        cli::Command::Gui => return gui::run(),
        cli::Command::ModelSetup => model_setup::download().map_err(Into::into),
        cli::Command::ModelSetupCheck => model_setup::check_service().map_err(Into::into),
        cli::Command::Devices => audio::devices::list(),
        cli::Command::Record(args) => audio::record::run(args),
        cli::Command::Process(args) => merge::process::run(args),
        cli::Command::Transcribe(args) => merge::process::run_transcribe(args),
        cli::Command::Diarize(args) => merge::process::run_diarize(args),
        cli::Command::Recognize(args) => merge::process::run_recognize(args),
        cli::Command::Correct(args) => merge::process::run_correct(args),
        cli::Command::Render(args) => merge::process::run_render(args),
        cli::Command::Archive(args) => audio::archive::run(args),
        cli::Command::Speakers(args) => speaker::speakers::list(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_event_worker(args: cli::ProcessArgs) -> ExitCode {
    let death_signal = rustix::process::set_parent_process_death_signal(Some(Signal::KILL));
    let orphaned = rustix::process::getppid().is_none_or(|parent| parent == Pid::INIT);
    let output = Arc::new(Mutex::new(BufWriter::new(io::stdout())));
    let fail = |error: String| {
        let _ = write_worker_event(&output, &merge::events::ProcessingEvent::Failure { error });
        ExitCode::FAILURE
    };

    if let Err(error) = death_signal {
        return fail(format!("cannot arm worker parent-death signal: {error}"));
    }
    if orphaned {
        return fail("processing worker was orphaned before it started".into());
    }

    let command = cli::Command::Process(args);
    if let Err(error) = model_setup::ensure_available(&command) {
        return fail(error.to_string());
    }
    let cli::Command::Process(args) = command else {
        unreachable!()
    };

    // Nobody reads the events once the parent is gone, so the work is abandoned with it.
    let report = {
        let output = output.clone();
        move |event| {
            if write_worker_event(&output, &event).is_err() {
                std::process::exit(1);
            }
        }
    };
    let report_transcription = report.clone();
    let result = merge::process::run_with_metrics(
        args,
        move |progress| report(merge::events::ProcessingEvent::Progress { progress }),
        Arc::new(move |progress| {
            report_transcription(merge::events::ProcessingEvent::Transcription { progress })
        }),
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => fail(error.to_string()),
    }
}

fn write_worker_event(
    output: &Mutex<BufWriter<io::Stdout>>,
    event: &merge::events::ProcessingEvent,
) -> io::Result<()> {
    let mut output = output
        .lock()
        .map_err(|_| io::Error::other("worker event output lock was poisoned"))?;
    merge::events::write_line(&mut *output, event)
}
