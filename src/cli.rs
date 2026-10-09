use crate::types::{AudioSource, DEFAULT_SPEAKER_THRESHOLD};
use clap::{ArgAction, Args, Parser, Subcommand};
use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "singstone",
    version,
    about = "Local-only meeting recorder, transcriber and speaker diarizer for Linux/PipeWire"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Launch the graphical application.
    Gui,
    /// Download and verify the Snap's pinned model cache.
    #[command(hide = true)]
    ModelSetup,
    /// Verify the Snap's model setup activation channel.
    #[command(hide = true)]
    ModelSetupCheck,
    /// List PipeWire audio sources and sinks usable for recording.
    Devices,
    /// Record microphone and/or system audio into a new session directory.
    Record(RecordArgs),
    /// Transcribe, diarize and merge a recorded session offline.
    Process(ProcessArgs),
    /// Transcribe a recorded session into words.jsonl.
    Transcribe(TranscribeArgs),
    /// Diarize a recorded session into diarization.jsonl.
    Diarize(DiarizeArgs),
    /// Match diarized clusters and write speaker-assignments.json.
    Recognize(RecognizeArgs),
    /// Correct one transcript line and learn its speaker.
    Correct(CorrectArgs),
    /// Render persistent processing artifacts into transcript files.
    Render(RenderArgs),
    /// Convert a processed session's raw audio to 16-bit FLAC to save disk space.
    Archive(ArchiveArgs),
    /// List learned speakers.
    Speakers(SpeakersArgs),
}

#[derive(Args, Debug)]
pub struct RecordArgs {
    /// Microphone: `default`, `none`, a PipeWire node id or node name.
    #[arg(long, default_value = "default")]
    pub mic: String,
    /// System audio sink to monitor: `default`, `none`, a node id or node name.
    #[arg(long, default_value = "default")]
    pub system: String,
    /// Directory to watch for new screenshots.
    #[arg(long)]
    pub screenshots: Option<PathBuf>,
    /// Comma-separated screenshot extensions to accept.
    #[arg(long, default_value = "png,jpg,jpeg,webp", value_delimiter = ',')]
    pub screenshot_ext: Vec<String>,
    /// Name assigned to microphone speech.
    #[arg(long, default_value = "Me")]
    pub local_speaker: String,
    /// Parent directory for the new session directory.
    #[arg(long, default_value = ".")]
    pub output_dir: PathBuf,
    /// Stop automatically after this many seconds (mainly for testing).
    #[arg(long)]
    pub duration: Option<f64>,
}

#[derive(Args, Debug, PartialEq)]
pub struct ProcessArgs {
    /// Session directory.
    pub session: PathBuf,
    /// Meeting details file to validate and copy into the session.
    #[arg(long)]
    pub meeting: Option<PathBuf>,
    /// Path to a whisper.cpp GGML model.
    #[arg(long, env = "SINGSTONE_WHISPER_MODEL")]
    pub whisper_model: Option<PathBuf>,
    /// Path to the sherpa-onnx pyannote segmentation model (model.onnx).
    #[arg(long, env = "SINGSTONE_SEGMENTATION_MODEL")]
    pub segmentation_model: Option<PathBuf>,
    /// Path to the sherpa-onnx speaker embedding model.
    #[arg(long, env = "SINGSTONE_EMBEDDING_MODEL")]
    pub embedding_model: Option<PathBuf>,
    /// Trusted model manifest with SHA-256 hashes.
    #[arg(long, env = "SINGSTONE_MODELS_LOCK")]
    pub models_lock: Option<PathBuf>,
    /// Skip model hash verification (development only).
    #[arg(long)]
    pub allow_unverified_models: bool,
    /// Diarize the microphone track (`--diarize-mic=false` to disable).
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        require_equals = true,
        action = ArgAction::Set
    )]
    pub diarize_mic: bool,
    /// Skip diarization entirely; every system utterance is `unknown`.
    #[arg(long)]
    pub no_diarize: bool,
    /// Reuse the session's existing words.jsonl instead of running whisper
    /// (re-diarize or re-merge with different settings).
    #[arg(long)]
    pub skip_transcription: bool,
    /// Whisper language code (`auto` to detect).
    #[arg(long, env = "SINGSTONE_WHISPER_LANGUAGE", default_value = "auto")]
    pub language: String,
    /// Inference threads (defaults to available parallelism).
    #[arg(long)]
    pub threads: Option<usize>,
    /// Learned-speaker database (default: $XDG_DATA_HOME/singstone/speakers.json).
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
    /// Minimum speaker-matching score required to name a diarized cluster.
    #[arg(long, default_value_t = DEFAULT_SPEAKER_THRESHOLD)]
    pub speaker_threshold: f32,
    /// Sherpa agglomerative clustering distance threshold (lower = more
    /// speakers; 0.9-1.1 works with titanet-small on meeting audio).
    #[arg(long, default_value_t = 1.0)]
    pub cluster_threshold: f32,
    /// Fix the number of speakers instead of estimating it.
    #[arg(long)]
    pub num_speakers: Option<u32>,
    /// Stream machine-readable processing events to stdout.
    #[arg(long, hide = true)]
    pub events: bool,
}

impl ProcessArgs {
    pub fn for_session(session: PathBuf) -> Self {
        Self {
            session,
            meeting: None,
            whisper_model: std::env::var_os("SINGSTONE_WHISPER_MODEL").map(PathBuf::from),
            segmentation_model: std::env::var_os("SINGSTONE_SEGMENTATION_MODEL").map(PathBuf::from),
            embedding_model: std::env::var_os("SINGSTONE_EMBEDDING_MODEL").map(PathBuf::from),
            models_lock: std::env::var_os("SINGSTONE_MODELS_LOCK").map(PathBuf::from),
            allow_unverified_models: false,
            diarize_mic: true,
            no_diarize: false,
            skip_transcription: false,
            language: std::env::var("SINGSTONE_WHISPER_LANGUAGE")
                .ok()
                .filter(|language| !language.trim().is_empty())
                .unwrap_or_else(|| "auto".into()),
            threads: None,
            speakers_db: std::env::var_os("SINGSTONE_SPEAKERS_DB").map(PathBuf::from),
            speaker_threshold: DEFAULT_SPEAKER_THRESHOLD,
            cluster_threshold: 1.0,
            num_speakers: None,
            events: false,
        }
    }

    pub fn command_args(&self) -> Vec<OsString> {
        let mut args = vec![
            OsString::from("process"),
            self.session.as_os_str().to_owned(),
        ];
        push_path_option(&mut args, "--meeting", self.meeting.as_deref());
        push_path_option(&mut args, "--whisper-model", self.whisper_model.as_deref());
        push_path_option(
            &mut args,
            "--segmentation-model",
            self.segmentation_model.as_deref(),
        );
        push_path_option(
            &mut args,
            "--embedding-model",
            self.embedding_model.as_deref(),
        );
        push_path_option(&mut args, "--models-lock", self.models_lock.as_deref());
        if self.allow_unverified_models {
            args.push("--allow-unverified-models".into());
        }
        args.push(format!("--diarize-mic={}", self.diarize_mic).into());
        if self.no_diarize {
            args.push("--no-diarize".into());
        }
        if self.skip_transcription {
            args.push("--skip-transcription".into());
        }
        args.extend([OsString::from("--language"), self.language.clone().into()]);
        if let Some(threads) = self.threads {
            args.extend([OsString::from("--threads"), threads.to_string().into()]);
        }
        push_path_option(&mut args, "--speakers-db", self.speakers_db.as_deref());
        args.extend([
            OsString::from("--speaker-threshold"),
            self.speaker_threshold.to_string().into(),
            OsString::from("--cluster-threshold"),
            self.cluster_threshold.to_string().into(),
        ]);
        if let Some(num_speakers) = self.num_speakers {
            args.extend([
                OsString::from("--num-speakers"),
                num_speakers.to_string().into(),
            ]);
        }
        if self.events {
            args.push("--events".into());
        }
        args
    }
}

fn push_path_option(args: &mut Vec<OsString>, flag: &str, value: Option<&std::path::Path>) {
    if let Some(value) = value {
        args.extend([OsString::from(flag), value.as_os_str().to_owned()]);
    }
}

#[derive(Args, Debug)]
pub struct TranscribeArgs {
    /// Session directory.
    pub session: PathBuf,
    /// Path to a whisper.cpp GGML model.
    #[arg(long, env = "SINGSTONE_WHISPER_MODEL")]
    pub whisper_model: PathBuf,
    /// Trusted model manifest with SHA-256 hashes.
    #[arg(long, env = "SINGSTONE_MODELS_LOCK")]
    pub models_lock: Option<PathBuf>,
    /// Skip model hash verification (development only).
    #[arg(long)]
    pub allow_unverified_models: bool,
    /// Whisper language code (`auto` to detect).
    #[arg(long, env = "SINGSTONE_WHISPER_LANGUAGE", default_value = "auto")]
    pub language: String,
    /// Inference threads (defaults to available parallelism).
    #[arg(long)]
    pub threads: Option<usize>,
}

#[derive(Args, Debug)]
pub struct DiarizeArgs {
    /// Session directory.
    pub session: PathBuf,
    /// Path to the sherpa-onnx pyannote segmentation model (model.onnx).
    #[arg(long, env = "SINGSTONE_SEGMENTATION_MODEL")]
    pub segmentation_model: PathBuf,
    /// Path to the sherpa-onnx speaker embedding model.
    #[arg(long, env = "SINGSTONE_EMBEDDING_MODEL")]
    pub embedding_model: PathBuf,
    /// Trusted model manifest with SHA-256 hashes.
    #[arg(long, env = "SINGSTONE_MODELS_LOCK")]
    pub models_lock: Option<PathBuf>,
    /// Skip model hash verification (development only).
    #[arg(long)]
    pub allow_unverified_models: bool,
    /// Diarize the microphone track (`--diarize-mic=false` to disable).
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        require_equals = true,
        action = ArgAction::Set
    )]
    pub diarize_mic: bool,
    /// Inference threads (defaults to available parallelism).
    #[arg(long)]
    pub threads: Option<usize>,
    /// Sherpa agglomerative clustering distance threshold.
    #[arg(long, default_value_t = 1.0)]
    pub cluster_threshold: f32,
    /// Fix the number of speakers instead of estimating it.
    #[arg(long)]
    pub num_speakers: Option<u32>,
}

#[derive(Args, Debug)]
pub struct RecognizeArgs {
    /// Session directory.
    pub session: PathBuf,
    /// Path to the sherpa-onnx speaker embedding model.
    #[arg(long, env = "SINGSTONE_EMBEDDING_MODEL")]
    pub embedding_model: PathBuf,
    /// Trusted model manifest with SHA-256 hashes.
    #[arg(long, env = "SINGSTONE_MODELS_LOCK")]
    pub models_lock: Option<PathBuf>,
    /// Skip model hash verification (development only).
    #[arg(long)]
    pub allow_unverified_models: bool,
    /// Inference threads (defaults to available parallelism).
    #[arg(long)]
    pub threads: Option<usize>,
    /// Learned-speaker database (default: $XDG_DATA_HOME/singstone/speakers.json).
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
    /// Minimum speaker-matching score required to name a diarized cluster.
    #[arg(long, default_value_t = DEFAULT_SPEAKER_THRESHOLD)]
    pub speaker_threshold: f32,
}

#[derive(Args, Debug)]
pub struct CorrectArgs {
    /// Session directory.
    pub session: PathBuf,
    /// Audio source of the rendered transcript line.
    #[arg(long, value_parser = parse_audio_source)]
    pub source: AudioSource,
    /// Exact rendered start time of the transcript line.
    #[arg(long)]
    pub start_ms: u64,
    /// Exact rendered end time of the transcript line.
    #[arg(long)]
    pub end_ms: u64,
    /// Speaker name to lock onto the line.
    #[arg(long)]
    pub name: String,
    /// Path to the sherpa-onnx speaker embedding model.
    #[arg(long, env = "SINGSTONE_EMBEDDING_MODEL")]
    pub embedding_model: Option<PathBuf>,
    /// Trusted model manifest with SHA-256 hashes.
    #[arg(long, env = "SINGSTONE_MODELS_LOCK")]
    pub models_lock: Option<PathBuf>,
    /// Skip model hash verification (development only).
    #[arg(long)]
    pub allow_unverified_models: bool,
    /// Inference threads (defaults to available parallelism).
    #[arg(long)]
    pub threads: Option<usize>,
    /// Learned-speaker database (default: $XDG_DATA_HOME/singstone/speakers.json).
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
    /// Minimum score for an unnamed forward proposal.
    #[arg(long, default_value_t = DEFAULT_SPEAKER_THRESHOLD)]
    pub speaker_threshold: f32,
}

fn parse_audio_source(value: &str) -> Result<AudioSource, String> {
    match value {
        "mic" => Ok(AudioSource::Mic),
        "system" => Ok(AudioSource::System),
        _ => Err("source must be 'system' or 'mic'".into()),
    }
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    /// Session directory.
    pub session: PathBuf,
    /// Override whether microphone words are diarized (`--diarize-mic=false` to disable).
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    pub diarize_mic: Option<bool>,
}

#[derive(Args, Debug)]
pub struct ArchiveArgs {
    /// Session directory.
    pub session: PathBuf,
}

#[derive(Args, Debug)]
pub struct SpeakersArgs {
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_processing_stage_command() {
        assert_eq!(
            ProcessArgs::for_session(PathBuf::from("session")).language,
            "auto"
        );
        assert!(ProcessArgs::for_session(PathBuf::from("session")).diarize_mic);
        let process =
            Cli::try_parse_from(["singstone", "process", "session"]).expect("parse process");
        assert!(matches!(
            process.command,
            Command::Process(ProcessArgs {
                language,
                diarize_mic: true,
                ..
            }) if language == "auto"
        ));
        let process_without_mic =
            Cli::try_parse_from(["singstone", "process", "session", "--diarize-mic=false"])
                .expect("parse process microphone opt-out");
        assert!(matches!(
            process_without_mic.command,
            Command::Process(ProcessArgs {
                diarize_mic: false,
                ..
            })
        ));

        let transcribe = Cli::try_parse_from([
            "singstone",
            "transcribe",
            "session",
            "--whisper-model",
            "whisper.bin",
        ])
        .expect("parse transcribe");
        assert!(matches!(
            transcribe.command,
            Command::Transcribe(TranscribeArgs { language, .. }) if language == "auto"
        ));

        let diarize = Cli::try_parse_from([
            "singstone",
            "diarize",
            "session",
            "--segmentation-model",
            "seg.onnx",
            "--embedding-model",
            "embed.onnx",
        ])
        .expect("parse diarize");
        assert!(matches!(
            diarize.command,
            Command::Diarize(DiarizeArgs {
                diarize_mic: true,
                ..
            })
        ));
        let diarize_without_mic = Cli::try_parse_from([
            "singstone",
            "diarize",
            "session",
            "--segmentation-model",
            "seg.onnx",
            "--embedding-model",
            "embed.onnx",
            "--diarize-mic=false",
        ])
        .expect("parse diarize microphone opt-out");
        assert!(matches!(
            diarize_without_mic.command,
            Command::Diarize(DiarizeArgs {
                diarize_mic: false,
                ..
            })
        ));

        let recognize = Cli::try_parse_from([
            "singstone",
            "recognize",
            "session",
            "--embedding-model",
            "embed.onnx",
        ])
        .expect("parse recognize");
        assert!(matches!(recognize.command, Command::Recognize(_)));

        let correct = Cli::try_parse_from([
            "singstone",
            "correct",
            "session",
            "--source",
            "mic",
            "--start-ms",
            "100",
            "--end-ms",
            "200",
            "--name",
            "Alice",
        ])
        .expect("parse correction without learning model");
        assert!(matches!(
            correct.command,
            Command::Correct(CorrectArgs {
                source: AudioSource::Mic,
                start_ms: 100,
                end_ms: 200,
                ref name,
                embedding_model: None,
                ..
            }) if name == "Alice"
        ));

        let render = Cli::try_parse_from(["singstone", "render", "session", "--diarize-mic"])
            .expect("parse render");
        assert!(matches!(
            render.command,
            Command::Render(RenderArgs {
                diarize_mic: Some(true),
                ..
            })
        ));
        let inferred =
            Cli::try_parse_from(["singstone", "render", "session"]).expect("parse inferred render");
        assert!(matches!(
            inferred.command,
            Command::Render(RenderArgs {
                diarize_mic: None,
                ..
            })
        ));
    }

    #[test]
    fn process_arguments_round_trip_to_command_line() {
        let expected = ProcessArgs {
            session: "session path".into(),
            meeting: Some("meeting.json".into()),
            whisper_model: Some("whisper.bin".into()),
            segmentation_model: Some("segmentation.onnx".into()),
            embedding_model: Some("embedding.onnx".into()),
            models_lock: Some("models.lock".into()),
            allow_unverified_models: true,
            diarize_mic: false,
            no_diarize: true,
            skip_transcription: true,
            language: "sv".into(),
            threads: Some(3),
            speakers_db: Some("speakers.json".into()),
            speaker_threshold: 0.81,
            cluster_threshold: 0.95,
            num_speakers: Some(4),
            events: true,
        };
        let mut command = vec![OsString::from("singstone")];
        command.extend(expected.command_args());

        let parsed = Cli::try_parse_from(command).expect("parse generated process arguments");
        let Command::Process(actual) = parsed.command else {
            panic!("expected process command");
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn parses_internal_model_setup_command() {
        let setup = Cli::try_parse_from(["singstone", "model-setup"]).expect("parse setup");
        assert!(matches!(setup.command, Command::ModelSetup));
        let check =
            Cli::try_parse_from(["singstone", "model-setup-check"]).expect("parse setup check");
        assert!(matches!(check.command, Command::ModelSetupCheck));
    }

    #[test]
    fn parses_gui_command() {
        let gui = Cli::try_parse_from(["singstone", "gui"]).expect("parse gui");
        assert!(matches!(gui.command, Command::Gui));
    }
}
