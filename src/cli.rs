use clap::{Args, Parser, Subcommand};
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
    /// Render persistent processing artifacts into transcript files.
    Render(RenderArgs),
    /// Enroll a known speaker from audio samples (raw f32le or WAV).
    Enroll(EnrollArgs),
    /// List enrolled speakers.
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

#[derive(Args, Debug)]
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
    /// Also diarize the microphone track (several people in the room).
    #[arg(long)]
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
    /// Enrolled-speaker database (default: $XDG_DATA_HOME/singstone/speakers.json).
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
    /// Minimum cosine similarity to name a diarized cluster after an enrolled speaker.
    #[arg(long, default_value_t = 0.6)]
    pub speaker_threshold: f32,
    /// Sherpa agglomerative clustering distance threshold (lower = more
    /// speakers; 0.9-1.1 works with titanet-small on meeting audio).
    #[arg(long, default_value_t = 1.0)]
    pub cluster_threshold: f32,
    /// Fix the number of speakers instead of estimating it.
    #[arg(long)]
    pub num_speakers: Option<u32>,
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
            diarize_mic: false,
            no_diarize: false,
            skip_transcription: false,
            language: std::env::var("SINGSTONE_WHISPER_LANGUAGE")
                .ok()
                .filter(|language| !language.trim().is_empty())
                .unwrap_or_else(|| "auto".into()),
            threads: None,
            speakers_db: std::env::var_os("SINGSTONE_SPEAKERS_DB").map(PathBuf::from),
            speaker_threshold: 0.6,
            cluster_threshold: 1.0,
            num_speakers: None,
        }
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
    /// Also diarize the microphone track (several people in the room).
    #[arg(long)]
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
    /// Enrolled-speaker database (default: $XDG_DATA_HOME/singstone/speakers.json).
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
    /// Minimum cosine similarity to name a diarized cluster after an enrolled speaker.
    #[arg(long, default_value_t = 0.6)]
    pub speaker_threshold: f32,
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
pub struct EnrollArgs {
    /// Speaker name.
    pub name: String,
    /// Audio samples: raw 16 kHz mono f32le, or 16 kHz mono WAV.
    #[arg(required = true)]
    pub samples: Vec<PathBuf>,
    #[arg(long, env = "SINGSTONE_EMBEDDING_MODEL")]
    pub embedding_model: PathBuf,
    #[arg(long, env = "SINGSTONE_MODELS_LOCK")]
    pub models_lock: Option<PathBuf>,
    #[arg(long)]
    pub allow_unverified_models: bool,
    #[arg(long, env = "SINGSTONE_SPEAKERS_DB")]
    pub speakers_db: Option<PathBuf>,
    /// Replace existing embeddings for this name instead of adding to them.
    #[arg(long)]
    pub replace: bool,
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
        let process =
            Cli::try_parse_from(["singstone", "process", "session"]).expect("parse process");
        assert!(matches!(
            process.command,
            Command::Process(ProcessArgs { language, .. }) if language == "auto"
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
        assert!(matches!(diarize.command, Command::Diarize(_)));

        let recognize = Cli::try_parse_from([
            "singstone",
            "recognize",
            "session",
            "--embedding-model",
            "embed.onnx",
        ])
        .expect("parse recognize");
        assert!(matches!(recognize.command, Command::Recognize(_)));

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
