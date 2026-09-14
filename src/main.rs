#![forbid(unsafe_code)]

mod audio;
mod cli;
mod diarization;
mod format;
mod merge;
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
    let result = match cli.command {
        cli::Command::Devices => audio::devices::list(),
        cli::Command::Record(args) => audio::record::run(args),
        cli::Command::Process(args) => merge::process::run(args),
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
