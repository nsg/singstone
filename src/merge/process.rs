use crate::cli::{DiarizeArgs, ProcessArgs, RecognizeArgs, RenderArgs, TranscribeArgs};
use crate::diarization::Diarizer;
use crate::diarization::sherpa::SherpaDiarizer;
use crate::format::jsonl;
use crate::merge::utterances::{self, DEFAULT_NEAREST_TOLERANCE_MS};
use crate::models;
use crate::session::Session;
use crate::speaker::database::{self, SpeakerDatabase};
use crate::speaker::embedding::{self, ClusterCandidate, EmbeddingExtractor};
use crate::transcription::Transcriber;
use crate::transcription::whisper::WhisperTranscriber;
use crate::types::{
    AudioSource, Manifest, SAMPLE_RATE, SessionState, SpeakerAssignment,
    SpeakerAssignmentProvenance, SpeakerAssignments, SpeakerSegment, TimedWord, Utterance,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const SPEAKER_ASSIGNMENTS_FORMAT_VERSION: u32 = 1;
const STAGE_METADATA_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingStage {
    Preparing,
    TranscribingMic,
    TranscribingSystem,
    Diarizing,
    Recognizing,
    Merging,
    Writing,
    Finished,
}

impl ProcessingStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Preparing => "Checking session and local models",
            Self::TranscribingMic => "Transcribing microphone audio",
            Self::TranscribingSystem => "Transcribing system audio",
            Self::Diarizing => "Separating speakers",
            Self::Recognizing => "Recognizing enrolled voices",
            Self::Merging => "Merging the meeting timeline",
            Self::Writing => "Writing transcript files",
            Self::Finished => "Processing complete",
        }
    }

    pub fn fraction(self) -> f64 {
        match self {
            Self::Preparing => 0.05,
            Self::TranscribingMic => 0.12,
            Self::TranscribingSystem => 0.3,
            Self::Diarizing => 0.55,
            Self::Recognizing => 0.78,
            Self::Merging => 0.88,
            Self::Writing => 0.95,
            Self::Finished => 1.0,
        }
    }

    pub const ALL: [Self; 8] = [
        Self::Preparing,
        Self::TranscribingMic,
        Self::TranscribingSystem,
        Self::Diarizing,
        Self::Recognizing,
        Self::Merging,
        Self::Writing,
        Self::Finished,
    ];

    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|stage| *stage == self)
            .unwrap_or(0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptionMetadata {
    format_version: u32,
    output_file: String,
    output_sha256: String,
    mic_audio_sha256: Option<String>,
    system_audio_sha256: Option<String>,
    whisper_model_sha256: String,
    language: String,
    threads: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiarizationMetadata {
    format_version: u32,
    output_file: String,
    output_sha256: String,
    mic_audio_sha256: Option<String>,
    system_audio_sha256: Option<String>,
    segmentation_model_sha256: Option<String>,
    embedding_model_sha256: Option<String>,
    diarize_mic: bool,
    cluster_threshold: f32,
    num_speakers: Option<u32>,
    threads: usize,
}

pub fn run(args: ProcessArgs) -> Result<(), Box<dyn std::error::Error>> {
    run_with_progress(args, |_| {})
}

pub fn run_with_progress(
    args: ProcessArgs,
    progress: impl Fn(ProcessingStage),
) -> Result<(), Box<dyn std::error::Error>> {
    run_with_control(args, progress, Arc::new(AtomicBool::new(false)))
}

pub fn run_with_control(
    args: ProcessArgs,
    progress: impl Fn(ProcessingStage),
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    progress(ProcessingStage::Preparing);
    ensure_not_cancelled(&cancelled)?;
    let (session, manifest) = open_session(&args.session)?;
    let threads = thread_count(args.threads);
    if args.skip_transcription {
        let words: Vec<TimedWord> = read_jsonl_artifact(&session.words_path(), "word input")?;
        eprintln!("transcribe: reusing {} words from words.jsonl", words.len());
    } else {
        let mut words = Vec::new();
        transcribe_sources(
            &args,
            &session,
            &manifest,
            threads,
            &mut words,
            false,
            Some(&progress),
            Some(&cancelled),
        )?;
    }

    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingStage::Diarizing);
    let diarization_started = Instant::now();
    let mut segments = diarize_sources(&args, &session, &manifest, threads, false)?;
    sort_segments(&mut segments);
    jsonl::write_all_atomic(&session.diarization_path(), &segments)?;
    write_diarization_metadata(&args, &session, &manifest, threads)?;
    eprintln!(
        "diarize: {} segment(s) in {:.1} s",
        segments.len(),
        diarization_started.elapsed().as_secs_f64()
    );

    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingStage::Recognizing);
    let recognition_started = Instant::now();
    let recognition = recognize_speakers(&args, &session, &segments, threads, false)?;
    write_speaker_assignments(&args, &session, &segments, &recognition)?;
    eprintln!(
        "recognize speakers: {:.1} s",
        recognition_started.elapsed().as_secs_f64()
    );

    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingStage::Merging);
    render_artifacts_with_hook(&session, &manifest, args.diarize_mic, || {
        progress(ProcessingStage::Writing);
    })?;
    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingStage::Finished);
    Ok(())
}

fn ensure_not_cancelled(cancelled: &AtomicBool) -> io::Result<()> {
    if cancelled.load(Ordering::Acquire) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "processing cancelled",
        ))
    } else {
        Ok(())
    }
}

pub fn run_transcribe(args: TranscribeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut process = stage_process_args(args.session);
    process.whisper_model = Some(args.whisper_model);
    process.models_lock = args.models_lock;
    process.allow_unverified_models = args.allow_unverified_models;
    process.language = args.language;
    process.threads = args.threads;
    let (session, manifest) = open_session(&process.session)?;
    validate_enabled_audio_files(
        &session,
        [
            (AudioSource::Mic, manifest.mic.enabled),
            (AudioSource::System, manifest.system.enabled),
        ],
    )?;
    let mut words = Vec::new();
    transcribe_sources(
        &process,
        &session,
        &manifest,
        thread_count(process.threads),
        &mut words,
        true,
        None,
        None,
    )
}

pub fn run_diarize(args: DiarizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut process = stage_process_args(args.session);
    process.segmentation_model = Some(args.segmentation_model);
    process.embedding_model = Some(args.embedding_model);
    process.models_lock = args.models_lock;
    process.allow_unverified_models = args.allow_unverified_models;
    process.diarize_mic = args.diarize_mic;
    process.threads = args.threads;
    process.cluster_threshold = args.cluster_threshold;
    process.num_speakers = args.num_speakers;
    let (session, manifest) = open_session(&process.session)?;
    validate_enabled_audio_files(
        &session,
        [
            (AudioSource::System, manifest.system.enabled),
            (
                AudioSource::Mic,
                manifest.mic.enabled && process.diarize_mic,
            ),
        ],
    )?;
    let started = Instant::now();
    let mut segments = diarize_sources(
        &process,
        &session,
        &manifest,
        thread_count(process.threads),
        true,
    )?;
    sort_segments(&mut segments);
    jsonl::write_all_atomic(&session.diarization_path(), &segments)?;
    write_diarization_metadata(&process, &session, &manifest, thread_count(process.threads))?;
    eprintln!(
        "diarize: {} segment(s) in {:.1} s",
        segments.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn run_recognize(args: RecognizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut process = stage_process_args(args.session);
    process.embedding_model = Some(args.embedding_model);
    process.models_lock = args.models_lock;
    process.allow_unverified_models = args.allow_unverified_models;
    process.threads = args.threads;
    process.speakers_db = args.speakers_db;
    process.speaker_threshold = args.speaker_threshold;
    let (session, _) = open_session(&process.session)?;
    let segments: Vec<SpeakerSegment> =
        read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    let recognition = recognize_speakers(
        &process,
        &session,
        &segments,
        thread_count(process.threads),
        true,
    )?;
    write_speaker_assignments(&process, &session, &segments, &recognition)?;
    Ok(())
}

pub fn run_render(args: RenderArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (session, manifest) = open_session(&args.session)?;
    let segments: Vec<SpeakerSegment> =
        read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    let diarize_mic = resolve_diarize_mic(&session, &segments, args.diarize_mic)?;
    render_artifacts_with_segments(&session, &manifest, diarize_mic, segments)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssignmentOutcome {
    /// Whether the cluster embedding was also added to the persistent speaker
    /// database, allowing later sessions to recognize the voice.
    pub learned: bool,
}

/// Assign a diarized cluster from the transcript and, when the embedding model
/// is available, use that cluster as a new local enrollment sample.
pub fn assign_speaker(
    args: ProcessArgs,
    speaker_id: &str,
    name: &str,
) -> Result<AssignmentOutcome, Box<dyn std::error::Error>> {
    let name = name.trim();
    if name.is_empty() {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "speaker name cannot be empty").into(),
        );
    }
    let (source, cluster) = parse_speaker_id(speaker_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{speaker_id:?} is not an assignable diarized speaker"),
        )
    })?;
    let (session, _) = open_session(&args.session)?;
    let mut artifact: SpeakerAssignments = read_json(&session.speaker_assignments_path())?;
    if artifact.format_version != SPEAKER_ASSIGNMENTS_FORMAT_VERSION
        || artifact.provenance.diarization_file != "diarization.jsonl"
        || !artifact
            .provenance
            .diarization_sha256
            .eq_ignore_ascii_case(&models::sha256_file(&session.diarization_path())?)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "speaker assignments are stale; process the session again before assigning a name",
        )
        .into());
    }
    let assignment = artifact
        .assignments
        .iter_mut()
        .find(|value| value.source == source && value.cluster == cluster)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("speaker cluster {speaker_id} is not in this session"),
            )
        })?;
    assignment.speaker = Some(name.to_owned());

    let learned = match learn_cluster(&args, &session, source, cluster, name) {
        Ok(learned) => learned,
        Err(error) => {
            eprintln!("warning: assigned {speaker_id} but could not learn its voice: {error}");
            false
        }
    };
    if learned {
        assignment.best_candidate = Some(name.to_owned());
        assignment.score = Some(1.0);
        let database_path = args
            .speakers_db
            .clone()
            .unwrap_or_else(database::default_path);
        artifact.provenance.speakers_database_sha256 = Some(models::sha256_file(&database_path)?);
    }
    write_json_atomic(&session.speaker_assignments_path(), &artifact)?;
    run_render(RenderArgs {
        session: session.dir,
        diarize_mic: None,
    })?;
    Ok(AssignmentOutcome { learned })
}

fn parse_speaker_id(value: &str) -> Option<(AudioSource, u32)> {
    if let Some(cluster) = value.strip_prefix("speaker-") {
        return Some((AudioSource::System, cluster.parse().ok()?));
    }
    let (prefix, cluster) = value.rsplit_once('_')?;
    let source = match prefix {
        "mic" => AudioSource::Mic,
        "spk" => AudioSource::System,
        _ => return None,
    };
    Some((source, cluster.parse().ok()?))
}

fn learn_cluster(
    args: &ProcessArgs,
    session: &Session,
    source: AudioSource,
    cluster: u32,
    name: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(model) = args.embedding_model.as_deref() else {
        return Ok(false);
    };
    models::verify_model(
        model,
        "speaker-embedding",
        args.models_lock.as_deref(),
        args.allow_unverified_models,
    )?;
    let extractor = EmbeddingExtractor::new(model, thread_count(args.threads))?;
    let segments: Vec<SpeakerSegment> =
        read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    let source_segments = segments
        .iter()
        .filter(|segment| segment.source == source && segment.cluster == cluster)
        .collect::<Vec<_>>();
    let samples = session.read_audio(source)?;
    let Some(embedding) = embed_clusters(&extractor, &samples, &source_segments).remove(&cluster)
    else {
        return Ok(false);
    };
    let identity = database::identity(model, extractor.dimension())?;
    let path = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    let mut database = match SpeakerDatabase::load_checked(&path, &identity) {
        Ok(database) => database,
        Err(error) if error.kind() == io::ErrorKind::NotFound => SpeakerDatabase::empty(identity),
        Err(error) => return Err(error.into()),
    };
    database
        .speakers
        .entry(name.to_owned())
        .or_default()
        .embeddings
        .push(embedding);
    database.save(&path)?;
    Ok(true)
}

fn stage_process_args(session: PathBuf) -> ProcessArgs {
    ProcessArgs {
        session,
        whisper_model: None,
        segmentation_model: None,
        embedding_model: None,
        models_lock: None,
        allow_unverified_models: false,
        diarize_mic: false,
        no_diarize: false,
        skip_transcription: false,
        language: "auto".into(),
        threads: None,
        speakers_db: None,
        speaker_threshold: 0.6,
        cluster_threshold: 1.0,
        num_speakers: None,
    }
}

fn open_session(path: &Path) -> Result<(Session, Manifest), Box<dyn std::error::Error>> {
    let session = Session::open(path)?;
    let manifest = session.read_manifest()?;
    if manifest.state == SessionState::Recording {
        eprintln!(
            "warning: session manifest is still in recording state; processing recovered audio"
        );
    }
    if manifest.sample_rate != SAMPLE_RATE
        || manifest.channels != 1
        || manifest.sample_format != "f32le"
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported session audio format: {} Hz, {} channel(s), {}",
                manifest.sample_rate, manifest.channels, manifest.sample_format
            ),
        )
        .into());
    }
    Ok((session, manifest))
}

fn thread_count(configured: Option<usize>) -> usize {
    configured
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
        .max(1)
}

#[allow(clippy::too_many_arguments)]
fn transcribe_sources(
    args: &ProcessArgs,
    session: &Session,
    manifest: &Manifest,
    threads: usize,
    words: &mut Vec<TimedWord>,
    strict_audio: bool,
    progress: Option<&dyn Fn(ProcessingStage)>,
    cancelled: Option<&AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = args.whisper_model.as_deref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--whisper-model is required unless --skip-transcription is used",
        )
    })?;
    models::verify_model(
        model,
        "transcription",
        args.models_lock.as_deref(),
        args.allow_unverified_models,
    )?;
    let mut transcriber =
        WhisperTranscriber::new(model, AudioSource::Mic, args.language.clone(), threads)?;
    for (source, enabled) in [
        (AudioSource::Mic, manifest.mic.enabled),
        (AudioSource::System, manifest.system.enabled),
    ] {
        if let Some(cancelled) = cancelled {
            ensure_not_cancelled(cancelled)?;
        }
        let samples = read_enabled_audio(session, source, enabled, strict_audio)?;
        if samples.is_empty() {
            continue;
        }
        if let Some(progress) = progress {
            progress(match source {
                AudioSource::Mic => ProcessingStage::TranscribingMic,
                AudioSource::System => ProcessingStage::TranscribingSystem,
            });
        }
        let started = Instant::now();
        transcriber.set_source(source);
        words.extend(transcriber.transcribe(&samples)?);
        eprintln!(
            "transcribe {source}: {:.1} s audio in {:.1} s",
            samples.len() as f64 / SAMPLE_RATE as f64,
            started.elapsed().as_secs_f64()
        );
    }
    sort_words(words);
    jsonl::write_all_atomic(&session.words_path(), words)?;
    write_transcription_metadata(args, session, manifest, threads)?;
    Ok(())
}

fn read_enabled_audio(
    session: &Session,
    source: AudioSource,
    enabled: bool,
    strict: bool,
) -> io::Result<Vec<f32>> {
    if !enabled {
        return Ok(Vec::new());
    }
    match session.read_audio(source) {
        Ok(samples) => Ok(samples),
        Err(error) if error.kind() == io::ErrorKind::NotFound && !strict => {
            eprintln!("warning: enabled {source} track is missing; skipping it");
            Ok(Vec::new())
        }
        Err(error) => Err(io::Error::new(
            error.kind(),
            format!(
                "cannot read enabled {source} audio {}: {error}",
                session.audio_path(source).display()
            ),
        )),
    }
}

fn validate_enabled_audio_files(
    session: &Session,
    tracks: impl IntoIterator<Item = (AudioSource, bool)>,
) -> io::Result<()> {
    for (source, enabled) in tracks {
        if enabled {
            let path = session.audio_path(source);
            fs::metadata(&path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "cannot read enabled {source} audio {}: {error}",
                        path.display()
                    ),
                )
            })?;
        }
    }
    Ok(())
}

fn diarize_sources(
    args: &ProcessArgs,
    session: &Session,
    manifest: &crate::types::Manifest,
    threads: usize,
    strict: bool,
) -> Result<Vec<SpeakerSegment>, Box<dyn std::error::Error>> {
    if args.no_diarize {
        eprintln!("diarization disabled");
        return Ok(Vec::new());
    }
    let (Some(segmentation_model), Some(embedding_model)) =
        (&args.segmentation_model, &args.embedding_model)
    else {
        eprintln!(
            "warning: diarization models were not both configured; continuing without diarization"
        );
        return Ok(Vec::new());
    };
    if strict {
        for model in [segmentation_model, embedding_model] {
            if !model.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("model {} does not exist", model.display()),
                )
                .into());
            }
        }
    }
    for (model, purpose) in [
        (segmentation_model.as_path(), "diarization-segmentation"),
        (embedding_model.as_path(), "speaker-embedding"),
    ] {
        if let Err(error) = models::verify_model(
            model,
            purpose,
            args.models_lock.as_deref(),
            args.allow_unverified_models,
        ) {
            if strict {
                return Err(error.into());
            }
            eprintln!("warning: diarization unavailable: {error}");
            return Ok(Vec::new());
        }
    }
    let mut all_segments = Vec::new();
    for (source, enabled) in [
        (AudioSource::System, manifest.system.enabled),
        (AudioSource::Mic, manifest.mic.enabled && args.diarize_mic),
    ] {
        if !enabled {
            continue;
        }
        let samples = match read_enabled_audio(session, source, true, strict) {
            Ok(samples) if !samples.is_empty() => samples,
            Ok(_) => continue,
            Err(error) => {
                if strict {
                    return Err(error.into());
                }
                eprintln!("warning: cannot read {source} audio for diarization: {error}");
                continue;
            }
        };
        let started = Instant::now();
        let result = SherpaDiarizer::new(
            segmentation_model,
            embedding_model,
            source,
            threads,
            args.cluster_threshold,
            args.num_speakers,
        )
        .and_then(|mut diarizer| diarizer.diarize(&samples));
        match result {
            Ok(mut segments) => {
                eprintln!(
                    "diarize {source}: {:.1} s audio, {} segment(s) in {:.1} s",
                    samples.len() as f64 / SAMPLE_RATE as f64,
                    segments.len(),
                    started.elapsed().as_secs_f64()
                );
                all_segments.append(&mut segments);
            }
            Err(error) if strict => return Err(error),
            Err(error) => eprintln!("warning: {source} diarization failed: {error}"),
        }
    }
    Ok(all_segments)
}

#[derive(Default)]
struct RecognitionResult {
    matches: HashMap<(AudioSource, u32), String>,
    best: HashMap<(AudioSource, u32), (String, f32)>,
}

fn recognize_speakers(
    args: &ProcessArgs,
    session: &Session,
    segments: &[SpeakerSegment],
    threads: usize,
    strict: bool,
) -> Result<RecognitionResult, Box<dyn std::error::Error>> {
    if segments.is_empty() {
        eprintln!("speaker recognition: no diarized clusters; skipped");
        return Ok(RecognitionResult::default());
    }
    let path = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    if !path.is_file() {
        if strict {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("speaker database {} does not exist", path.display()),
            )
            .into());
        }
        eprintln!("speaker recognition: no database; skipped");
        return Ok(RecognitionResult::default());
    }
    let database = match SpeakerDatabase::load(&path) {
        Ok(database) => database,
        Err(error) => {
            if strict {
                return Err(error.into());
            }
            eprintln!("warning: speaker recognition skipped: {error}");
            return Ok(RecognitionResult::default());
        }
    };
    if database
        .speakers
        .values()
        .all(|speaker| speaker.embeddings.is_empty())
    {
        if strict {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "speaker database has no embeddings",
            )
            .into());
        }
        eprintln!("speaker recognition: database has no embeddings; skipped");
        return Ok(RecognitionResult::default());
    }
    let Some(model) = args.embedding_model.as_deref() else {
        if strict {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--embedding-model is required",
            )
            .into());
        }
        eprintln!("speaker recognition: no embedding model configured; skipped");
        return Ok(RecognitionResult::default());
    };
    if let Err(error) = models::verify_model(
        model,
        "speaker-embedding",
        args.models_lock.as_deref(),
        args.allow_unverified_models,
    ) {
        if strict {
            return Err(error.into());
        }
        eprintln!("warning: speaker recognition skipped: {error}");
        return Ok(RecognitionResult::default());
    }
    let extractor = match EmbeddingExtractor::new(model, threads) {
        Ok(extractor) => extractor,
        Err(error) => {
            if strict {
                return Err(error);
            }
            eprintln!("warning: speaker recognition unavailable: {error}");
            return Ok(RecognitionResult::default());
        }
    };
    let identity = match database::identity(model, extractor.dimension()) {
        Ok(identity) => identity,
        Err(error) => {
            if strict {
                return Err(error.into());
            }
            eprintln!("warning: speaker recognition skipped: {error}");
            return Ok(RecognitionResult::default());
        }
    };
    if let Err(error) = database.validate_identity(&identity) {
        if strict {
            return Err(error.into());
        }
        eprintln!("warning: speaker recognition skipped: {error}");
        return Ok(RecognitionResult::default());
    }

    let mut result = RecognitionResult::default();
    for source in [AudioSource::System, AudioSource::Mic] {
        let source_segments = segments
            .iter()
            .filter(|segment| segment.source == source)
            .collect::<Vec<_>>();
        if source_segments.is_empty() {
            continue;
        }
        let samples = match session.read_audio(source) {
            Ok(samples) => samples,
            Err(error) => {
                if strict {
                    return Err(error.into());
                }
                eprintln!("warning: cannot read {source} audio for speaker recognition: {error}");
                continue;
            }
        };
        let cluster_embeddings = embed_clusters(&extractor, &samples, &source_segments);
        let mut candidates = Vec::new();
        eprintln!("speaker recognition ({source}):");
        eprintln!("cluster\tbest candidate\tscore\tassigned");
        for (cluster, embedding) in cluster_embeddings {
            let best = best_candidate(&database, &embedding);
            match best {
                Some((name, score)) => {
                    let assigned = if score >= args.speaker_threshold {
                        name.as_str()
                    } else {
                        "-"
                    };
                    eprintln!("{cluster}\t{name}\t{score:.3}\t{assigned}");
                    result.best.insert((source, cluster), (name.clone(), score));
                    candidates.push(ClusterCandidate {
                        cluster,
                        name,
                        score,
                    });
                }
                None => eprintln!("{cluster}\t-\t-\t-"),
            }
        }
        for (cluster, name) in embedding::unique_matches(&candidates, args.speaker_threshold) {
            result.matches.insert((source, cluster), name);
        }
    }
    Ok(result)
}

fn write_transcription_metadata(
    args: &ProcessArgs,
    session: &Session,
    manifest: &Manifest,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = args.whisper_model.as_deref().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "--whisper-model is required")
    })?;
    write_json_atomic(
        &session.words_metadata_path(),
        &TranscriptionMetadata {
            format_version: STAGE_METADATA_FORMAT_VERSION,
            output_file: "words.jsonl".into(),
            output_sha256: models::sha256_file(&session.words_path())?,
            mic_audio_sha256: hash_audio(session, AudioSource::Mic, manifest.mic.enabled)?,
            system_audio_sha256: hash_audio(session, AudioSource::System, manifest.system.enabled)?,
            whisper_model_sha256: models::sha256_file(model)?,
            language: args.language.clone(),
            threads,
        },
    )?;
    Ok(())
}

fn write_diarization_metadata(
    args: &ProcessArgs,
    session: &Session,
    manifest: &Manifest,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let optional_hash = |path: Option<&Path>| {
        path.filter(|path| path.is_file())
            .map(models::sha256_file)
            .transpose()
    };
    write_json_atomic(
        &session.diarization_metadata_path(),
        &DiarizationMetadata {
            format_version: STAGE_METADATA_FORMAT_VERSION,
            output_file: "diarization.jsonl".into(),
            output_sha256: models::sha256_file(&session.diarization_path())?,
            mic_audio_sha256: hash_audio(
                session,
                AudioSource::Mic,
                manifest.mic.enabled && args.diarize_mic,
            )?,
            system_audio_sha256: hash_audio(session, AudioSource::System, manifest.system.enabled)?,
            segmentation_model_sha256: optional_hash(args.segmentation_model.as_deref())?,
            embedding_model_sha256: optional_hash(args.embedding_model.as_deref())?,
            diarize_mic: args.diarize_mic,
            cluster_threshold: args.cluster_threshold,
            num_speakers: args.num_speakers,
            threads,
        },
    )?;
    Ok(())
}

fn hash_audio(
    session: &Session,
    source: AudioSource,
    included: bool,
) -> io::Result<Option<String>> {
    let path = session.audio_path(source);
    (included && path.is_file())
        .then(|| models::sha256_file(&path))
        .transpose()
}

fn write_speaker_assignments(
    args: &ProcessArgs,
    session: &Session,
    segments: &[SpeakerSegment],
    recognition: &RecognitionResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut clusters = segments
        .iter()
        .map(|segment| (segment.source, segment.cluster))
        .collect::<Vec<_>>();
    clusters.sort_by_key(|(source, cluster)| (source_order(*source), *cluster));
    clusters.dedup();
    let assignments = clusters
        .into_iter()
        .map(|(source, cluster)| {
            let best = recognition.best.get(&(source, cluster));
            SpeakerAssignment {
                source,
                cluster,
                best_candidate: best.map(|(name, _)| name.clone()),
                score: best.map(|(_, score)| *score),
                speaker: recognition.matches.get(&(source, cluster)).cloned(),
            }
        })
        .collect();
    let database = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    let artifact = SpeakerAssignments {
        format_version: SPEAKER_ASSIGNMENTS_FORMAT_VERSION,
        provenance: SpeakerAssignmentProvenance {
            diarization_file: "diarization.jsonl".into(),
            diarization_sha256: models::sha256_file(&session.diarization_path())?,
            embedding_model_sha256: args
                .embedding_model
                .as_deref()
                .filter(|path| path.is_file())
                .map(models::sha256_file)
                .transpose()?,
            speakers_database_sha256: database
                .is_file()
                .then(|| models::sha256_file(&database))
                .transpose()?,
            speaker_threshold: args.speaker_threshold,
        },
        assignments,
    };
    write_json_atomic(&session.speaker_assignments_path(), &artifact)?;
    Ok(())
}

fn render_artifacts_with_hook(
    session: &Session,
    manifest: &Manifest,
    diarize_mic: bool,
    before_write: impl FnOnce(),
) -> Result<(), Box<dyn std::error::Error>> {
    let segments = read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    render_artifacts_with_segments_and_hook(session, manifest, diarize_mic, segments, before_write)
}

fn render_artifacts_with_segments(
    session: &Session,
    manifest: &Manifest,
    diarize_mic: bool,
    segments: Vec<SpeakerSegment>,
) -> Result<(), Box<dyn std::error::Error>> {
    render_artifacts_with_segments_and_hook(session, manifest, diarize_mic, segments, || {})
}

fn render_artifacts_with_segments_and_hook(
    session: &Session,
    manifest: &Manifest,
    diarize_mic: bool,
    segments: Vec<SpeakerSegment>,
    before_write: impl FnOnce(),
) -> Result<(), Box<dyn std::error::Error>> {
    let merge_started = Instant::now();
    let words: Vec<TimedWord> = read_jsonl_artifact(&session.words_path(), "word input")?;
    warn_transcription_provenance(session);
    warn_diarization_provenance(session, diarize_mic);
    let recognized = read_speaker_assignments(session)?;
    let utterances = utterances::build_utterances(
        &words,
        &segments,
        &recognized,
        &manifest.local_speaker,
        diarize_mic,
        DEFAULT_NEAREST_TOLERANCE_MS,
    );
    before_write();
    jsonl::write_all_atomic(&session.transcript_path(), &utterances)?;
    write_transcript_text(&session.transcript_text_path(), &utterances)?;
    eprintln!(
        "merge: {} utterance(s) in {:.1} s",
        utterances.len(),
        merge_started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn resolve_diarize_mic(
    session: &Session,
    segments: &[SpeakerSegment],
    explicit: Option<bool>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let path = session.diarization_metadata_path();
    let recorded = match read_json::<DiarizationMetadata>(&path) {
        Ok(metadata) => {
            if metadata.format_version != STAGE_METADATA_FORMAT_VERSION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported diarization metadata format version {} in {}",
                        metadata.format_version,
                        path.display()
                    ),
                )
                .into());
            }
            Some(metadata.diarize_mic)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    match (explicit, recorded) {
        (Some(requested), Some(actual)) if requested != actual => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "--diarize-mic={requested} conflicts with diarization metadata {} (diarize_mic={actual})",
                path.display()
            ),
        )
        .into()),
        (Some(requested), _) => Ok(requested),
        (None, Some(actual)) => Ok(actual),
        (None, None) => Ok(segments
            .iter()
            .any(|segment| segment.source == AudioSource::Mic)),
    }
}

fn read_speaker_assignments(
    session: &Session,
) -> Result<HashMap<(AudioSource, u32), String>, Box<dyn std::error::Error>> {
    let path = session.speaker_assignments_path();
    let artifact: SpeakerAssignments = match read_json(&path) {
        Ok(artifact) => artifact,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            eprintln!(
                "warning: {} is missing; rendering diarized clusters anonymously",
                path.display()
            );
            return Ok(HashMap::new());
        }
        Err(error) => return Err(error.into()),
    };
    if artifact.format_version != SPEAKER_ASSIGNMENTS_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported speaker assignments format version {}",
                artifact.format_version
            ),
        )
        .into());
    }
    if artifact.provenance.diarization_file != "diarization.jsonl" {
        eprintln!(
            "warning: speaker-assignments.json references {}; expected diarization.jsonl; ignoring assignments",
            artifact.provenance.diarization_file
        );
        return Ok(HashMap::new());
    }
    let actual = models::sha256_file(&session.diarization_path())?;
    if !actual.eq_ignore_ascii_case(&artifact.provenance.diarization_sha256) {
        eprintln!(
            "warning: speaker-assignments.json was produced from a different diarization.jsonl; ignoring stale assignments"
        );
        return Ok(HashMap::new());
    }
    let mut recognized = HashMap::new();
    let mut seen = HashSet::new();
    for assignment in artifact.assignments {
        let key = (assignment.source, assignment.cluster);
        if !seen.insert(key) {
            eprintln!(
                "warning: duplicate speaker assignment for {} cluster {}; keeping the first",
                assignment.source, assignment.cluster
            );
            continue;
        }
        if let Some(speaker) = assignment.speaker {
            recognized.insert(key, speaker);
        }
    }
    Ok(recognized)
}

fn warn_transcription_provenance(session: &Session) {
    let path = session.words_metadata_path();
    let metadata: TranscriptionMetadata = match read_json(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return,
        Err(error) => {
            eprintln!("warning: cannot read {}: {error}", path.display());
            return;
        }
    };
    if metadata.format_version != STAGE_METADATA_FORMAT_VERSION {
        eprintln!(
            "warning: {} has unsupported format version {}",
            path.display(),
            metadata.format_version
        );
        return;
    }
    warn_output_hash(
        &session.words_path(),
        &metadata.output_sha256,
        "words.meta.json",
    );
    warn_input_hash(
        session,
        AudioSource::Mic,
        metadata.mic_audio_sha256.as_deref(),
        "words.meta.json",
    );
    warn_input_hash(
        session,
        AudioSource::System,
        metadata.system_audio_sha256.as_deref(),
        "words.meta.json",
    );
}

fn warn_diarization_provenance(session: &Session, diarize_mic: bool) {
    let path = session.diarization_metadata_path();
    let metadata: DiarizationMetadata = match read_json(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return,
        Err(error) => {
            eprintln!("warning: cannot read {}: {error}", path.display());
            return;
        }
    };
    if metadata.format_version != STAGE_METADATA_FORMAT_VERSION {
        eprintln!(
            "warning: {} has unsupported format version {}",
            path.display(),
            metadata.format_version
        );
        return;
    }
    warn_output_hash(
        &session.diarization_path(),
        &metadata.output_sha256,
        "diarization.meta.json",
    );
    warn_input_hash(
        session,
        AudioSource::Mic,
        metadata.mic_audio_sha256.as_deref(),
        "diarization.meta.json",
    );
    warn_input_hash(
        session,
        AudioSource::System,
        metadata.system_audio_sha256.as_deref(),
        "diarization.meta.json",
    );
    if metadata.diarize_mic != diarize_mic {
        eprintln!(
            "warning: render --diarize-mic={} differs from the diarization stage setting {}",
            diarize_mic, metadata.diarize_mic
        );
    }
}

fn warn_output_hash(path: &Path, expected: &str, metadata_name: &str) {
    match models::sha256_file(path) {
        Ok(actual) if !actual.eq_ignore_ascii_case(expected) => eprintln!(
            "warning: {metadata_name} was produced for a different {}; the artifact was edited or replaced",
            path.file_name().unwrap_or_default().to_string_lossy()
        ),
        Err(error) => eprintln!("warning: cannot fingerprint {}: {error}", path.display()),
        _ => {}
    }
}

fn warn_input_hash(
    session: &Session,
    source: AudioSource,
    expected: Option<&str>,
    metadata_name: &str,
) {
    let Some(expected) = expected else {
        return;
    };
    let path = session.audio_path(source);
    match models::sha256_file(&path) {
        Ok(actual) if !actual.eq_ignore_ascii_case(expected) => eprintln!(
            "warning: {metadata_name} was produced from different {source} audio; its output may be stale"
        ),
        Err(error) => eprintln!("warning: cannot fingerprint {}: {error}", path.display()),
        _ => {}
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let tmp = jsonl::tmp_path(path);
    {
        let mut file = crate::session::create_private_file(&tmp)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(tmp, path)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> io::Result<T> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid JSON in {}: {error}", path.display()),
        )
    })
}

fn read_jsonl_artifact<T: DeserializeOwned>(path: &Path, description: &str) -> io::Result<Vec<T>> {
    jsonl::read_all(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("cannot read {description} {}: {error}", path.display()),
        )
    })
}

fn embed_clusters(
    extractor: &EmbeddingExtractor,
    samples: &[f32],
    segments: &[&SpeakerSegment],
) -> BTreeMap<u32, Vec<f32>> {
    let mut grouped: BTreeMap<u32, Vec<&SpeakerSegment>> = BTreeMap::new();
    for segment in segments
        .iter()
        .copied()
        .filter(|segment| segment.end_ms.saturating_sub(segment.start_ms) >= 1_500)
    {
        grouped.entry(segment.cluster).or_default().push(segment);
    }
    let mut output = BTreeMap::new();
    for (cluster, mut cluster_segments) in grouped {
        cluster_segments
            .sort_by_key(|segment| std::cmp::Reverse(segment.end_ms - segment.start_ms));
        let mut total_ms = 0u64;
        let mut embeddings = Vec::new();
        for segment in cluster_segments {
            let remaining = 30_000u64.saturating_sub(total_ms);
            if remaining < 1_500 {
                break;
            }
            let duration = (segment.end_ms - segment.start_ms).min(remaining);
            let start = ms_to_index(segment.start_ms, samples.len());
            let end = ms_to_index(segment.start_ms + duration, samples.len());
            if end > start
                && let Some(embedding) = extractor.embed(&samples[start..end])
            {
                embeddings.push(embedding);
                total_ms += duration;
            }
        }
        if let Some(mean) = embedding::mean_normalized(&embeddings) {
            output.insert(cluster, mean);
        }
    }
    output
}

fn best_candidate(database: &SpeakerDatabase, cluster: &[f32]) -> Option<(String, f32)> {
    database
        .speakers
        .iter()
        .flat_map(|(name, speaker)| {
            speaker.embeddings.iter().filter_map(move |enrolled| {
                let mut enrolled = enrolled.clone();
                embedding::normalize(&mut enrolled)
                    .then(|| embedding::cosine(cluster, &enrolled))
                    .flatten()
                    .map(|score| (name.clone(), score))
            })
        })
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
}

fn ms_to_index(milliseconds: u64, audio_len: usize) -> usize {
    milliseconds
        .saturating_mul(SAMPLE_RATE as u64)
        .saturating_div(1_000)
        .try_into()
        .unwrap_or(usize::MAX)
        .min(audio_len)
}

fn sort_words(words: &mut [TimedWord]) {
    words.sort_by_key(|word| (word.start_ms, source_order(word.source), word.end_ms));
}

fn sort_segments(segments: &mut [SpeakerSegment]) {
    segments.sort_by_key(|segment| {
        (
            segment.start_ms,
            source_order(segment.source),
            segment.cluster,
        )
    });
}

fn source_order(source: AudioSource) -> u8 {
    match source {
        AudioSource::Mic => 0,
        AudioSource::System => 1,
    }
}

fn write_transcript_text(path: &Path, utterances: &[Utterance]) -> io::Result<()> {
    let tmp = jsonl::tmp_path(path);
    {
        let mut file = crate::session::create_private_file(&tmp)?;
        for utterance in utterances {
            writeln!(
                file,
                "[{}] {}: {}",
                format_timestamp(utterance.start_ms),
                utterance.speaker,
                utterance.text
            )?;
        }
        file.sync_all()?;
    }
    fs::rename(tmp, path)
}

fn format_timestamp(milliseconds: u64) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = milliseconds / 60_000 % 60;
    let seconds = milliseconds / 1_000 % 60;
    let millis = milliseconds % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker::database::{EmbeddingModelIdentity, SpeakerRecord};
    use crate::types::{ScreenshotEntry, StreamInfo, TimelineEvent};
    use serde::{Serialize, de::DeserializeOwned};

    #[test]
    fn best_score_uses_max_enrollment_embedding() {
        let database = SpeakerDatabase {
            embedding_model: EmbeddingModelIdentity {
                name: "m".into(),
                sha256: "x".into(),
                dimension: 2,
            },
            speakers: BTreeMap::from([
                (
                    "Alice".into(),
                    SpeakerRecord {
                        embeddings: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    },
                ),
                (
                    "Bob".into(),
                    SpeakerRecord {
                        embeddings: vec![vec![-1.0, 0.0]],
                    },
                ),
            ]),
        };
        let (name, score) = best_candidate(&database, &[0.0, 1.0]).expect("candidate");
        assert_eq!(name, "Alice");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn timestamp_format_is_fixed_width() {
        assert_eq!(format_timestamp(3_723_004), "01:02:03.004");
    }

    #[test]
    fn jsonl_round_trips_all_persistent_line_types() {
        let base = std::env::temp_dir().join(format!(
            "singstone-jsonl-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        round_trip(
            &base.with_extension("words"),
            &[TimedWord {
                source: AudioSource::Mic,
                start_ms: 10,
                end_ms: 20,
                text: "hello".into(),
            }],
        );
        round_trip(
            &base.with_extension("segments"),
            &[SpeakerSegment {
                source: AudioSource::System,
                start_ms: 1,
                end_ms: 2,
                cluster: 3,
            }],
        );
        round_trip(
            &base.with_extension("utterances"),
            &[Utterance {
                start_ms: 1,
                end_ms: 2,
                source: AudioSource::System,
                speaker_id: "spk_3".into(),
                speaker: "SPEAKER_00".into(),
                text: "hello".into(),
            }],
        );
        round_trip(
            &base.with_extension("screenshots"),
            &[ScreenshotEntry {
                time_ms: 42,
                file: "screenshots/000042.png".into(),
                original: "/tmp/shot.png".into(),
            }],
        );
        round_trip(
            &base.with_extension("timeline"),
            &[
                TimelineEvent::Xrun {
                    sample: 160,
                    time_ms: 10,
                    missing_samples: 80,
                },
                TimelineEvent::ClockJump {
                    sample: 160,
                    time_ms: 10,
                    requested_samples: 2_000_000,
                },
            ],
        );
    }

    fn round_trip<T>(path: &Path, values: &[T])
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        jsonl::write_all_atomic(path, values).expect("write JSONL");
        let actual: Vec<T> = jsonl::read_all(path).expect("read JSONL");
        assert_eq!(actual, values);
        std::fs::remove_file(path).expect("remove test JSONL");
    }

    fn empty_session(root: &Path) -> Session {
        let session = Session::create(root.join("session")).expect("create session");
        session
            .write_manifest(&Manifest {
                format_version: crate::types::FORMAT_VERSION,
                state: SessionState::Stopped,
                started_wallclock: "2026-09-14T10:30:00+00:00".into(),
                sample_rate: SAMPLE_RATE,
                channels: 1,
                sample_format: "f32le".into(),
                mic: StreamInfo {
                    enabled: false,
                    pipewire_node: None,
                    error: None,
                },
                system: StreamInfo {
                    enabled: false,
                    pipewire_node: None,
                    error: None,
                },
                local_speaker: "Me".into(),
                screenshot_dir: None,
            })
            .expect("write manifest");
        session
    }

    #[test]
    fn diarization_pipeline_smoke_test_when_fixtures_are_configured() {
        let (Some(models), Some(samples)) = (
            std::env::var_os("SINGSTONE_TEST_MODELS"),
            std::env::var_os("SINGSTONE_TEST_SAMPLES"),
        ) else {
            return;
        };
        let models = std::path::PathBuf::from(models);
        let session = Session::open(std::path::PathBuf::from(samples).join("session-ami-3min"))
            .expect("fixture session");
        let manifest = session.read_manifest().expect("manifest");
        let args = ProcessArgs {
            session: session.dir.clone(),
            whisper_model: Some(models.join("ggml-base.en.bin")),
            segmentation_model: Some(
                models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
            ),
            embedding_model: Some(models.join("nemo_en_titanet_small.onnx")),
            models_lock: None,
            allow_unverified_models: true,
            skip_transcription: false,
            diarize_mic: false,
            no_diarize: false,
            language: "en".into(),
            threads: Some(1),
            speakers_db: None,
            speaker_threshold: 0.6,
            cluster_threshold: 0.5,
            num_speakers: None,
        };
        let segments =
            diarize_sources(&args, &session, &manifest, 1, true).expect("diarize fixture");
        assert!(!segments.is_empty());
    }

    #[test]
    fn skipped_stages_do_not_require_or_touch_models() {
        let root =
            std::env::temp_dir().join(format!("singstone-skipped-models-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let session = empty_session(&root);
        jsonl::write_all_atomic::<TimedWord>(&session.words_path(), &[]).expect("write words");
        run(ProcessArgs {
            session: session.dir.clone(),
            whisper_model: None,
            segmentation_model: Some(root.join("missing-segmentation.onnx")),
            embedding_model: Some(root.join("missing-embedding.onnx")),
            models_lock: Some(root.join("missing-models.lock")),
            allow_unverified_models: false,
            diarize_mic: false,
            no_diarize: true,
            skip_transcription: true,
            language: "en".into(),
            threads: Some(1),
            speakers_db: None,
            speaker_threshold: 0.6,
            cluster_threshold: 1.0,
            num_speakers: None,
        })
        .expect("process with skipped stages");
        run(ProcessArgs {
            session: session.dir.clone(),
            whisper_model: None,
            segmentation_model: Some(root.join("missing-segmentation.onnx")),
            embedding_model: Some(root.join("missing-embedding.onnx")),
            models_lock: Some(root.join("missing-models.lock")),
            allow_unverified_models: false,
            diarize_mic: false,
            no_diarize: false,
            skip_transcription: true,
            language: "en".into(),
            threads: Some(1),
            speakers_db: None,
            speaker_threshold: 0.6,
            cluster_threshold: 1.0,
            num_speakers: None,
        })
        .expect("fall back when diarization models cannot be verified");
        assert!(session.diarization_metadata_path().is_file());
        assert!(session.speaker_assignments_path().is_file());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn render_consumes_current_assignments_and_ignores_stale_ones() {
        let root = std::env::temp_dir().join(format!(
            "singstone-render-stage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        jsonl::write_all_atomic(
            &session.words_path(),
            &[TimedWord {
                source: AudioSource::System,
                start_ms: 10,
                end_ms: 20,
                text: "hello".into(),
            }],
        )
        .expect("write words");
        let segment = SpeakerSegment {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 30,
            cluster: 7,
        };
        jsonl::write_all_atomic(&session.diarization_path(), std::slice::from_ref(&segment))
            .expect("write diarization");
        let artifact = SpeakerAssignments {
            format_version: SPEAKER_ASSIGNMENTS_FORMAT_VERSION,
            provenance: SpeakerAssignmentProvenance {
                diarization_file: "diarization.jsonl".into(),
                diarization_sha256: models::sha256_file(&session.diarization_path())
                    .expect("hash diarization"),
                embedding_model_sha256: Some("model-hash".into()),
                speakers_database_sha256: Some("database-hash".into()),
                speaker_threshold: 0.6,
            },
            assignments: vec![
                SpeakerAssignment {
                    source: AudioSource::System,
                    cluster: 7,
                    best_candidate: Some("Alice".into()),
                    score: Some(0.9),
                    speaker: Some("Alice".into()),
                },
                SpeakerAssignment {
                    source: AudioSource::System,
                    cluster: 7,
                    best_candidate: Some("Bob".into()),
                    score: Some(0.8),
                    speaker: Some("Bob".into()),
                },
            ],
        };
        write_json_atomic(&session.speaker_assignments_path(), &artifact)
            .expect("write assignments");

        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: Some(false),
        })
        .expect("render current artifacts");
        let utterances: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read transcript");
        assert_eq!(utterances[0].speaker, "Alice");

        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                end_ms: 31,
                ..segment
            }],
        )
        .expect("edit diarization");
        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: Some(false),
        })
        .expect("stale provenance is a warning");
        let utterances: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read stale transcript");
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn manual_assignment_rerenders_without_a_voice_model() {
        let root = std::env::temp_dir().join(format!(
            "singstone-manual-assignment-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        jsonl::write_all_atomic(
            &session.words_path(),
            &[TimedWord {
                source: AudioSource::System,
                start_ms: 10,
                end_ms: 20,
                text: "hello".into(),
            }],
        )
        .expect("write words");
        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 30,
                cluster: 7,
            }],
        )
        .expect("write diarization");
        write_json_atomic(
            &session.speaker_assignments_path(),
            &SpeakerAssignments {
                format_version: SPEAKER_ASSIGNMENTS_FORMAT_VERSION,
                provenance: SpeakerAssignmentProvenance {
                    diarization_file: "diarization.jsonl".into(),
                    diarization_sha256: models::sha256_file(&session.diarization_path())
                        .expect("hash diarization"),
                    embedding_model_sha256: None,
                    speakers_database_sha256: None,
                    speaker_threshold: 0.6,
                },
                assignments: vec![SpeakerAssignment {
                    source: AudioSource::System,
                    cluster: 7,
                    best_candidate: None,
                    score: None,
                    speaker: None,
                }],
            },
        )
        .expect("write assignments");
        let outcome = assign_speaker(stage_process_args(session.dir.clone()), "spk_7", "Carol")
            .expect("assign speaker");
        assert!(!outcome.learned);
        let transcript: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read transcript");
        assert_eq!(transcript[0].speaker, "Carol");
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn render_resolves_diarize_mic_policy_and_rejects_conflicts() {
        let root = std::env::temp_dir().join(format!(
            "singstone-render-policy-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        jsonl::write_all_atomic(
            &session.words_path(),
            &[TimedWord {
                source: AudioSource::Mic,
                start_ms: 10,
                end_ms: 20,
                text: "hello".into(),
            }],
        )
        .expect("write words");
        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                source: AudioSource::Mic,
                start_ms: 0,
                end_ms: 30,
                cluster: 4,
            }],
        )
        .expect("write diarization");
        write_json_atomic(
            &session.diarization_metadata_path(),
            &DiarizationMetadata {
                format_version: STAGE_METADATA_FORMAT_VERSION,
                output_file: "diarization.jsonl".into(),
                output_sha256: models::sha256_file(&session.diarization_path())
                    .expect("hash diarization"),
                mic_audio_sha256: None,
                system_audio_sha256: None,
                segmentation_model_sha256: Some("segmentation".into()),
                embedding_model_sha256: Some("embedding".into()),
                diarize_mic: true,
                cluster_threshold: 1.0,
                num_speakers: None,
                threads: 1,
            },
        )
        .expect("write metadata");

        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: None,
        })
        .expect("infer policy from metadata");
        let utterances: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read transcript");
        assert_eq!(utterances[0].speaker, "SPEAKER_00");

        let conflict = run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: Some(false),
        })
        .expect_err("explicit conflict fails");
        assert!(conflict.to_string().contains("diarization.meta.json"));

        fs::remove_file(session.diarization_metadata_path()).expect("remove metadata");
        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: None,
        })
        .expect("legacy mic segments imply diarization");
        let utterances: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read legacy transcript");
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn missing_upstream_artifacts_name_their_paths() {
        let root = std::env::temp_dir().join(format!(
            "singstone-missing-upstream-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        let recognize_error = run_recognize(RecognizeArgs {
            session: session.dir.clone(),
            embedding_model: root.join("embedding.onnx"),
            models_lock: None,
            allow_unverified_models: true,
            threads: Some(1),
            speakers_db: Some(root.join("speakers.json")),
            speaker_threshold: 0.6,
        })
        .expect_err("missing diarization fails");
        assert!(
            recognize_error
                .to_string()
                .contains(session.diarization_path().to_string_lossy().as_ref())
        );

        jsonl::write_all_atomic::<SpeakerSegment>(&session.diarization_path(), &[])
            .expect("write diarization");
        let render_error = run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: None,
        })
        .expect_err("missing words fails");
        assert!(
            render_error
                .to_string()
                .contains(session.words_path().to_string_lossy().as_ref())
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn failed_standalone_stages_preserve_prior_outputs() {
        let root = std::env::temp_dir().join(format!(
            "singstone-stage-failure-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        let mut manifest = session.read_manifest().expect("read manifest");
        manifest.system.enabled = true;
        session
            .write_manifest(&manifest)
            .expect("enable system audio");
        fs::write(session.words_path(), b"prior words\n").expect("write prior words");
        let transcription_error = run_transcribe(TranscribeArgs {
            session: session.dir.clone(),
            whisper_model: root.join("unused-whisper.bin"),
            models_lock: None,
            allow_unverified_models: true,
            language: "en".into(),
            threads: Some(1),
        })
        .expect_err("missing enabled audio fails transcription");
        assert!(
            transcription_error.to_string().contains(
                session
                    .audio_path(AudioSource::System)
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            fs::read(session.words_path()).expect("read prior words"),
            b"prior words\n"
        );

        fs::write(session.diarization_path(), b"prior diarization\n")
            .expect("write prior diarization");
        let error = run_diarize(DiarizeArgs {
            session: session.dir.clone(),
            segmentation_model: root.join("unused-segmentation.onnx"),
            embedding_model: root.join("unused-embedding.onnx"),
            models_lock: None,
            allow_unverified_models: true,
            diarize_mic: false,
            threads: Some(1),
            cluster_threshold: 1.0,
            num_speakers: None,
        })
        .expect_err("missing enabled audio fails diarization");
        assert!(
            error.to_string().contains(
                session
                    .audio_path(AudioSource::System)
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            fs::read(session.diarization_path()).expect("read prior diarization"),
            b"prior diarization\n"
        );

        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 10,
                cluster: 0,
            }],
        )
        .expect("write valid diarization");
        fs::write(session.speaker_assignments_path(), b"prior assignments\n")
            .expect("write prior assignments");
        run_recognize(RecognizeArgs {
            session: session.dir.clone(),
            embedding_model: root.join("missing-embedding.onnx"),
            models_lock: None,
            allow_unverified_models: true,
            threads: Some(1),
            speakers_db: Some(root.join("missing-speakers.json")),
            speaker_threshold: 0.6,
        })
        .expect_err("missing database fails");
        assert_eq!(
            fs::read(session.speaker_assignments_path()).expect("read prior assignments"),
            b"prior assignments\n"
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }
}
#[test]
fn parses_assignable_transcript_speaker_ids() {
    assert_eq!(parse_speaker_id("mic_3"), Some((AudioSource::Mic, 3)));
    assert_eq!(parse_speaker_id("spk_42"), Some((AudioSource::System, 42)));
    assert_eq!(parse_speaker_id("unknown"), None);
    assert_eq!(
        parse_speaker_id("speaker-2"),
        Some((AudioSource::System, 2))
    );
    assert_eq!(parse_speaker_id("spk_nope"), None);
}
