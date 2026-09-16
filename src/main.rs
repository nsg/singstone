#![forbid(unsafe_code)]

mod audio;
mod cli;
mod diarization;
mod format;
mod merge;
mod model_setup;
mod models;
mod screenshot;
mod session;
mod speaker;
mod transcription;
mod types;

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = cli::Cli::parse();
    if !matches!(cli.command, cli::Command::ModelSetup)
        && let Err(e) = model_setup::ensure_available(&cli.command)
    {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let result = match cli.command {
        cli::Command::ModelSetup => model_setup::download().map_err(Into::into),
        cli::Command::Devices => audio::devices::list(),
        cli::Command::Record(args) => audio::record::run(args),
        cli::Command::Process(args) => merge::process::run(args),
        cli::Command::Transcribe(args) => merge::process::run_transcribe(args),
        cli::Command::Diarize(args) => merge::process::run_diarize(args),
        cli::Command::Recognize(args) => merge::process::run_recognize(args),
        cli::Command::Render(args) => merge::process::run_render(args),
        cli::Command::Enroll(args) => speaker::enroll::run(args),
        cli::Command::Speakers(args) => speaker::enroll::list(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
