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
    /// List PipeWire audio sources and sinks usable for recording.
    Devices,
    /// Record microphone and/or system audio into a new session directory.
    Record(RecordArgs),
    /// Transcribe, diarize and merge a recorded session offline.
    Process(ProcessArgs),
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
    #[arg(long, default_value = "en")]
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
