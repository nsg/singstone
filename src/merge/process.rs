use crate::audio::archive;
use crate::cli::{
    CorrectArgs, DiarizeArgs, ProcessArgs, RecognizeArgs, RenderArgs, TranscribeArgs,
};
use crate::diarization::Diarizer;
use crate::diarization::sherpa::SherpaDiarizer;
use crate::format::jsonl;
use crate::meeting::{self, MeetingDetails};
use crate::merge::leakage::{self, AudioEnvelopes};
use crate::merge::utterances::{self, DEFAULT_NEAREST_TOLERANCE_MS};
use crate::models;
use crate::session::Session;
use crate::speaker::database::{self, EmbeddingModelIdentity, SpeakerDatabase};
use crate::speaker::embedding::{self, ClusterCandidate, EmbeddingExtractor};
use crate::transcription::whisper::WhisperTranscriber;
use crate::transcription::{ProgressReporter, Transcriber};
use crate::types::{
    AudioSource, DEFAULT_SPEAKER_THRESHOLD, EmbeddingChunk, HiddenSources, Manifest, SAMPLE_RATE,
    SessionState, SpeakerAssignment, SpeakerAssignmentProvenance, SpeakerAssignments,
    SpeakerCorrection, SpeakerCorrections, SpeakerSegment, TimedWord, Utterance,
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
const EMBEDDINGS_FORMAT_VERSION: u32 = 1;
const SPEAKER_CORRECTIONS_FORMAT_VERSION: u32 = 1;
const HIDDEN_SOURCES_FORMAT_VERSION: u32 = 1;
const EMBEDDING_WINDOW_MS: u64 = 10_000;
const MIN_LEARN_MS: u64 = 1_500;
const PROPOSAL_MARGIN: f32 = 0.05;
const SAME_VECTOR_SIMILARITY: f32 = 0.999;

struct CorrectionLearning {
    chunks: Vec<EmbeddingChunk>,
    database: SpeakerDatabase,
    database_path: PathBuf,
    migrate_legacy_database: bool,
}

struct CorrectionLearningUpdate {
    speaker: String,
    old_names: HashSet<String>,
    vectors: Vec<Vec<f32>>,
}

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
            Self::Recognizing => "Recognizing learned voices",
            Self::Merging => "Merging the meeting timeline",
            Self::Writing => "Writing transcript files",
            Self::Finished => "Processing complete",
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessingProgress {
    pub stage: ProcessingStage,
    pub fraction: Option<f64>,
}

impl ProcessingProgress {
    fn indeterminate(stage: ProcessingStage) -> Self {
        Self {
            stage,
            fraction: None,
        }
    }

    fn determinate(stage: ProcessingStage, fraction: f64) -> Self {
        Self {
            stage,
            fraction: Some(fraction.clamp(0.0, 1.0)),
        }
    }
}

type ProcessingProgressReporter = Arc<dyn Fn(ProcessingProgress) + Send + Sync>;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mic_num_speakers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system_num_speakers: Option<u32>,
    threads: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingsMetadata {
    format_version: u32,
    output_file: String,
    output_sha256: String,
    diarization_sha256: String,
    mic_audio_sha256: Option<String>,
    system_audio_sha256: Option<String>,
    embedding_model_sha256: String,
    threads: usize,
    window_ms: u64,
    min_learn_ms: u64,
}

pub fn run(args: ProcessArgs) -> Result<(), Box<dyn std::error::Error>> {
    run_with_progress(args, |_| {})
}

pub fn run_with_progress(
    args: ProcessArgs,
    progress: impl Fn(ProcessingStage) + Send + Sync + 'static,
) -> Result<(), Box<dyn std::error::Error>> {
    run_with_control(args, progress, Arc::new(AtomicBool::new(false)))
}

pub fn run_with_control(
    args: ProcessArgs,
    progress: impl Fn(ProcessingStage) + Send + Sync + 'static,
    cancelled: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_with_control_and_metrics(
        args,
        move |update| progress(update.stage),
        cancelled,
        Arc::new(|_| {}),
    )
}

pub fn run_with_control_and_metrics(
    args: ProcessArgs,
    progress: impl Fn(ProcessingProgress) + Send + Sync + 'static,
    cancelled: Arc<AtomicBool>,
    transcription_progress: ProgressReporter,
) -> Result<(), Box<dyn std::error::Error>> {
    let progress: ProcessingProgressReporter = Arc::new(progress);
    progress(ProcessingProgress::indeterminate(
        ProcessingStage::Preparing,
    ));
    ensure_not_cancelled(&cancelled)?;
    let (session, manifest) = open_session(&args.session)?;
    prepare_meeting(&args, &session)?;
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
            Some(progress.clone()),
            Some(&cancelled),
            Some(transcription_progress),
        )?;
    }

    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingProgress::indeterminate(
        ProcessingStage::Diarizing,
    ));
    let diarization_started = Instant::now();
    let mut segments = diarize_sources(&args, &session, &manifest, threads, false)?;
    sort_segments(&mut segments);
    jsonl::write_all_atomic(&session.diarization_path(), &segments)?;
    let diarize_mic = effective_diarize_mic(args.diarize_mic, &segments);
    write_diarization_metadata(&args, &session, &manifest, diarize_mic, threads)?;
    eprintln!(
        "diarize: {} segment(s) in {:.1} s",
        segments.len(),
        diarization_started.elapsed().as_secs_f64()
    );

    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingProgress::indeterminate(
        ProcessingStage::Recognizing,
    ));
    let recognition_started = Instant::now();
    let recognition_progress = progress.clone();
    let recognition = recognize_speakers(
        &args,
        &session,
        &segments,
        threads,
        false,
        Some(&move |fraction| {
            recognition_progress(ProcessingProgress::determinate(
                ProcessingStage::Recognizing,
                fraction,
            ));
        }),
    )?;
    write_speaker_assignments(&args, &session, &segments, &recognition)?;
    eprintln!(
        "recognize speakers: {:.1} s",
        recognition_started.elapsed().as_secs_f64()
    );

    ensure_not_cancelled(&cancelled)?;
    progress(ProcessingProgress::indeterminate(ProcessingStage::Merging));
    render_artifacts_with_hook(&session, &manifest, diarize_mic, || {
        progress(ProcessingProgress::indeterminate(ProcessingStage::Writing));
    })?;
    progress(ProcessingProgress::determinate(
        ProcessingStage::Finished,
        1.0,
    ));
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
    let diarize_mic = effective_diarize_mic(process.diarize_mic, &segments);
    write_diarization_metadata(
        &process,
        &session,
        &manifest,
        diarize_mic,
        thread_count(process.threads),
    )?;
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
    let outputs = [
        session.embeddings_path(),
        session.embeddings_metadata_path(),
        session.speaker_assignments_path(),
    ];
    let previous_outputs = outputs
        .iter()
        .map(|path| read_optional_file(path))
        .collect::<Result<Vec<_>, _>>()?;
    let result = (|| {
        let recognition = recognize_speakers(
            &process,
            &session,
            &segments,
            thread_count(process.threads),
            true,
            None,
        )?;
        write_speaker_assignments(&process, &session, &segments, &recognition)
    })();
    if let Err(error) = result {
        for (path, previous) in outputs.iter().zip(&previous_outputs) {
            if let Err(restore_error) = restore_file(path, previous.as_deref()) {
                return Err(io::Error::other(format!(
                    "recognition failed: {error}; restoring {} also failed: {restore_error}",
                    path.display()
                ))
                .into());
            }
        }
        return Err(error);
    }
    Ok(())
}

pub fn run_correct(args: CorrectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut process = stage_process_args(args.session);
    process.embedding_model = args.embedding_model;
    process.models_lock = args.models_lock;
    process.allow_unverified_models = args.allow_unverified_models;
    process.threads = args.threads;
    process.speakers_db = args.speakers_db;
    process.speaker_threshold = args.speaker_threshold;
    let request = CorrectionRequest {
        source: args.source,
        start_ms: args.start_ms,
        end_ms: args.end_ms,
        speaker: args.name,
    };
    let mut outcome = correct_speaker(process, &request)?;
    let proposals = std::mem::take(&mut outcome.proposals);
    println!("{}", serde_json::to_string(&outcome)?);
    for proposal in proposals {
        println!("{}", serde_json::to_string(&proposal)?);
    }
    Ok(())
}

pub fn run_render(args: RenderArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (session, manifest) = open_session(&args.session)?;
    let segments: Vec<SpeakerSegment> =
        read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    let diarize_mic = resolve_diarize_mic(&session, &segments, args.diarize_mic)?;
    render_artifacts_with_segments(&session, &manifest, diarize_mic, segments)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionRequest {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub source: AudioSource,
    pub cluster: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub current_speaker: Option<String>,
    pub score_new: f32,
    pub score_old: Option<f32>,
    pub proposed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionOutcome {
    pub learned: usize,
    pub forgotten: usize,
    pub learning_available: bool,
    pub echo: bool,
    pub proposals: Vec<Proposal>,
}

pub fn correct_speaker(
    args: ProcessArgs,
    request: &CorrectionRequest,
) -> Result<CorrectionOutcome, Box<dyn std::error::Error>> {
    correct_speaker_resolved(args, request).map(|(outcome, _)| outcome)
}

pub fn correct_speaker_forward(
    args: ProcessArgs,
    request: &CorrectionRequest,
) -> Result<CorrectionOutcome, Box<dyn std::error::Error>> {
    let session = Session::open(args.session.clone())?;
    let (outcome, request) = correct_speaker_resolved(args, request)?;
    let later = inferred_corrections(&outcome.proposals, &request.speaker);
    if !later.is_empty() {
        infer_later_lines(&session, later).map_err(|error| {
            io::Error::other(format!(
                "the line was named, but later lines of the same voice were not: {error}"
            ))
        })?;
    }
    Ok(outcome)
}

fn inferred_corrections(proposals: &[Proposal], speaker: &str) -> Vec<SpeakerCorrection> {
    proposals
        .iter()
        .filter(|proposal| proposal.proposed)
        .map(|proposal| SpeakerCorrection {
            source: proposal.source,
            start_ms: proposal.start_ms,
            end_ms: proposal.end_ms,
            speaker: speaker.to_owned(),
            inferred: true,
        })
        .collect()
}

fn infer_later_lines(
    session: &Session,
    later: Vec<SpeakerCorrection>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut corrections = read_speaker_corrections(session)?;
    for correction in later {
        upsert_correction(&mut corrections, correction);
    }
    save_corrections_and_render(session, &corrections)
}

fn save_corrections_and_render(
    session: &Session,
    corrections: &SpeakerCorrections,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = session.speaker_corrections_path();
    let previous = match fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    write_json_atomic(&path, corrections)?;
    if let Err(render_error) = run_render(RenderArgs {
        session: session.dir.clone(),
        diarize_mic: None,
    }) {
        if let Err(rollback_error) = restore_file(&path, previous.as_deref()) {
            return Err(io::Error::other(format!(
                "render failed: {render_error}; restoring {} also failed: {rollback_error}",
                path.display()
            ))
            .into());
        }
        return Err(render_error);
    }
    Ok(())
}

fn correct_speaker_resolved(
    args: ProcessArgs,
    request: &CorrectionRequest,
) -> Result<(CorrectionOutcome, CorrectionRequest), Box<dyn std::error::Error>> {
    let request = validate_correction_request(request)?;
    let session = Session::open(args.session.clone())?;
    let segments: Vec<SpeakerSegment> =
        read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    let prior_utterances = match read_jsonl_artifact(&session.transcript_path(), "transcript") {
        Ok(utterances) => utterances,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let proposal_cluster = cluster_for_request(&request, &prior_utterances, &segments);
    let learning_configured = args.embedding_model.is_some();
    let database_path = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    let threshold = args.speaker_threshold;
    let (mut outcome, mut requests) =
        apply_corrections_inner(args, std::slice::from_ref(&request))?;
    let request = requests.remove(0);
    if outcome.learning_available
        && let Some(cluster) = proposal_cluster
        && learning_configured
        && database_path.is_file()
    {
        let chunks: Vec<EmbeddingChunk> =
            read_jsonl_artifact(&session.embeddings_path(), "speaker embeddings")?;
        let database = SpeakerDatabase::load(&database_path)?;
        outcome.proposals =
            build_proposals(&session, &request, cluster, &chunks, &database, threshold)?;
    }
    Ok((outcome, request))
}

fn apply_corrections_inner(
    args: ProcessArgs,
    requests: &[CorrectionRequest],
) -> Result<(CorrectionOutcome, Vec<CorrectionRequest>), Box<dyn std::error::Error>> {
    let mut requests = requests
        .iter()
        .map(validate_correction_request)
        .collect::<Result<Vec<_>, _>>()?;
    requests.sort_by_key(|request| {
        (
            request.start_ms,
            source_order(request.source),
            request.end_ms,
        )
    });
    let (session, _) = open_session(&args.session)?;
    let segments: Vec<SpeakerSegment> =
        read_jsonl_artifact(&session.diarization_path(), "diarization input")?;
    let recognized = read_speaker_assignments(&session)?;
    let prior_utterances = match read_jsonl_artifact(&session.transcript_path(), "transcript") {
        Ok(utterances) => utterances,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let mut corrections = read_speaker_corrections(&session)?;
    let mut learning = prepare_correction_learning(&args, &session, &segments)?;
    if let Some(learning) = learning.as_ref() {
        for request in &mut requests {
            request.speaker = learning.database.resolve_name(&request.speaker);
        }
    }
    let mut replaced = Vec::with_capacity(requests.len());
    for request in &requests {
        replaced.push(upsert_correction(
            &mut corrections,
            SpeakerCorrection {
                source: request.source,
                start_ms: request.start_ms,
                end_ms: request.end_ms,
                speaker: request.speaker.clone(),
                inferred: false,
            },
        ));
    }
    let learning_updates = learning
        .as_ref()
        .map(|learning| {
            requests
                .iter()
                .zip(&replaced)
                .map(|(request, replaced)| {
                    let mut old_names = replaced
                        .iter()
                        .filter(|correction| !correction.inferred)
                        .map(|correction| correction.speaker.clone())
                        .collect::<HashSet<_>>();
                    if let Some(cluster) =
                        cluster_for_request(request, &prior_utterances, &segments)
                        && let Some(name) = recognized.get(&(request.source, cluster))
                    {
                        old_names.insert(name.clone());
                    }
                    old_names.remove(&request.speaker);
                    CorrectionLearningUpdate {
                        speaker: request.speaker.clone(),
                        old_names,
                        vectors: embeddings_in_range(
                            &learning.chunks,
                            request.source,
                            request.start_ms,
                            request.end_ms,
                        ),
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    save_corrections_and_render(&session, &corrections)?;

    let learning_available = learning.is_some();
    let (mut learned, mut forgotten) = (0, 0);
    if let Some(learning) = learning.as_mut() {
        if learning.migrate_legacy_database {
            SpeakerDatabase::load(&learning.database_path)?;
        }
        (learned, forgotten) = apply_correction_learning(&mut learning.database, &learning_updates);
        if learned > 0 || forgotten > 0 {
            learning.database.save(&learning.database_path)?;
        }
    }
    let echo = correction_is_system_echo(&requests, &recognized, &corrections.corrections);
    Ok((
        CorrectionOutcome {
            learned,
            forgotten,
            learning_available,
            echo,
            proposals: Vec::new(),
        },
        requests,
    ))
}

fn validate_correction_request(
    request: &CorrectionRequest,
) -> Result<CorrectionRequest, Box<dyn std::error::Error>> {
    let speaker = request
        .speaker
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if speaker.is_empty() {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "speaker name cannot be empty").into(),
        );
    }
    if request.end_ms < request.start_ms {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "speaker correction end must not be before its start",
        )
        .into());
    }
    Ok(CorrectionRequest {
        source: request.source,
        start_ms: request.start_ms,
        end_ms: request.end_ms,
        speaker,
    })
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

fn read_speaker_corrections(session: &Session) -> io::Result<SpeakerCorrections> {
    let path = session.speaker_corrections_path();
    let artifact: SpeakerCorrections = match read_json(&path) {
        Ok(artifact) => artifact,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(SpeakerCorrections {
                format_version: SPEAKER_CORRECTIONS_FORMAT_VERSION,
                corrections: Vec::new(),
            });
        }
        Err(error) => return Err(error),
    };
    if artifact.format_version != SPEAKER_CORRECTIONS_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported speaker corrections format version {} in {}",
                artifact.format_version,
                path.display()
            ),
        ));
    }
    Ok(artifact)
}

/// The sources left out of the transcript; none when the session has no file.
pub fn read_hidden_sources(session: &Session) -> io::Result<Vec<AudioSource>> {
    let path = session.hidden_sources_path();
    let artifact: HiddenSources = match read_json(&path) {
        Ok(artifact) => artifact,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    if artifact.format_version != HIDDEN_SOURCES_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported hidden sources format version {} in {}",
                artifact.format_version,
                path.display()
            ),
        ));
    }
    Ok(normalize_hidden_sources(&artifact.hidden))
}

fn normalize_hidden_sources(hidden: &[AudioSource]) -> Vec<AudioSource> {
    [AudioSource::Mic, AudioSource::System]
        .into_iter()
        .filter(|source| hidden.contains(source))
        .collect()
}

fn upsert_correction(
    artifact: &mut SpeakerCorrections,
    correction: SpeakerCorrection,
) -> Vec<SpeakerCorrection> {
    let old = std::mem::take(&mut artifact.corrections);
    let mut replaced = Vec::new();
    for existing in old {
        if existing.source == correction.source
            && (existing.inferred || !correction.inferred)
            && locks_collide(
                existing.start_ms,
                existing.end_ms,
                correction.start_ms,
                correction.end_ms,
            )
        {
            replaced.push(existing);
        } else {
            artifact.corrections.push(existing);
        }
    }
    artifact.corrections.push(correction);
    replaced
}

fn ranges_overlap(left_start: u64, left_end: u64, right_start: u64, right_end: u64) -> bool {
    left_start <= right_end && right_start <= left_end
}

/// Adjacent lines share a boundary timestamp, so a lock only replaces locks it
/// strictly overlaps, or an identical range.
fn locks_collide(left_start: u64, left_end: u64, right_start: u64, right_end: u64) -> bool {
    (left_start == right_start && left_end == right_end)
        || (left_start < right_end && right_start < left_end)
}

fn cluster_for_request(
    request: &CorrectionRequest,
    utterances: &[Utterance],
    segments: &[SpeakerSegment],
) -> Option<u32> {
    utterances
        .iter()
        .find(|utterance| {
            utterance.source == request.source
                && utterance.start_ms == request.start_ms
                && utterance.end_ms == request.end_ms
        })
        .and_then(|utterance| parse_speaker_id(&utterance.speaker_id))
        .filter(|(source, _)| *source == request.source)
        .map(|(_, cluster)| cluster)
        .or_else(|| {
            segments
                .iter()
                .filter(|segment| segment.source == request.source)
                .filter_map(|segment| {
                    let overlap = segment
                        .end_ms
                        .min(request.end_ms)
                        .saturating_sub(segment.start_ms.max(request.start_ms));
                    ranges_overlap(
                        segment.start_ms,
                        segment.end_ms,
                        request.start_ms,
                        request.end_ms,
                    )
                    .then_some((overlap, segment.start_ms, segment.cluster))
                })
                .max_by_key(|(overlap, start, cluster)| {
                    (
                        *overlap,
                        std::cmp::Reverse(*start),
                        std::cmp::Reverse(*cluster),
                    )
                })
                .map(|(_, _, cluster)| cluster)
        })
}

fn correction_is_system_echo(
    requests: &[CorrectionRequest],
    recognized: &HashMap<(AudioSource, u32), String>,
    corrections: &[SpeakerCorrection],
) -> bool {
    let system_speakers = recognized
        .iter()
        .filter(|((source, _), _)| *source == AudioSource::System)
        .map(|(_, speaker)| speaker.as_str())
        .chain(
            corrections
                .iter()
                .filter(|correction| correction.source == AudioSource::System)
                .map(|correction| correction.speaker.as_str()),
        )
        .collect::<HashSet<_>>();
    requests.iter().any(|request| {
        request.source == AudioSource::Mic && system_speakers.contains(request.speaker.as_str())
    })
}

fn build_proposals(
    session: &Session,
    request: &CorrectionRequest,
    cluster: u32,
    chunks: &[EmbeddingChunk],
    database: &SpeakerDatabase,
    threshold: f32,
) -> Result<Vec<Proposal>, Box<dyn std::error::Error>> {
    let utterances: Vec<Utterance> = read_jsonl_artifact(&session.transcript_path(), "transcript")?;
    let mut proposals = utterances
        .into_iter()
        .filter_map(|utterance| {
            proposal_for_utterance(request, cluster, utterance, chunks, database, threshold)
        })
        .collect::<Vec<_>>();
    proposals.sort_by_key(|proposal| (proposal.start_ms, proposal.end_ms));
    Ok(proposals)
}

fn proposal_for_utterance(
    request: &CorrectionRequest,
    cluster: u32,
    utterance: Utterance,
    chunks: &[EmbeddingChunk],
    database: &SpeakerDatabase,
    threshold: f32,
) -> Option<Proposal> {
    if utterance.source != request.source
        || utterance.start_ms < request.end_ms
        || utterance.locked
        || parse_speaker_id(&utterance.speaker_id) != Some((request.source, cluster))
    {
        return None;
    }
    let query = embeddings_in_range(
        chunks,
        utterance.source,
        utterance.start_ms,
        utterance.end_ms,
    );
    let score_new = score_speaker_name(database, &request.speaker, &query)?;
    let current_speaker =
        (!is_anonymous_speaker(&utterance.speaker)).then(|| utterance.speaker.clone());
    let score_old = current_speaker
        .as_deref()
        .and_then(|name| score_speaker_name(database, name, &query));
    let proposed = score_old.map_or(score_new >= threshold, |score_old| {
        score_new - score_old >= PROPOSAL_MARGIN
    });
    Some(Proposal {
        source: utterance.source,
        cluster,
        start_ms: utterance.start_ms,
        end_ms: utterance.end_ms,
        text: utterance.text,
        current_speaker,
        score_new,
        score_old,
        proposed,
    })
}

fn is_anonymous_speaker(speaker: &str) -> bool {
    speaker.starts_with("SPEAKER_") || speaker == "unknown"
}

fn prepare_correction_learning(
    args: &ProcessArgs,
    session: &Session,
    segments: &[SpeakerSegment],
) -> Result<Option<CorrectionLearning>, Box<dyn std::error::Error>> {
    let Some(model) = args.embedding_model.as_deref() else {
        return Ok(None);
    };
    models::verify_model(
        model,
        "speaker-embedding",
        args.models_lock.as_deref(),
        args.allow_unverified_models,
    )?;
    let threads = thread_count(args.threads);
    let extractor = EmbeddingExtractor::new(model, threads)?;
    let identity = database::identity(model, extractor.dimension())?;
    let path = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    let (database, migrate_legacy_database) = load_correction_database(&path, &identity)?;
    let chunks = ensure_embeddings(
        session,
        segments,
        &extractor,
        &database.embedding_model,
        threads,
    )?;
    Ok(Some(CorrectionLearning {
        chunks,
        database,
        database_path: path,
        migrate_legacy_database,
    }))
}

fn load_correction_database(
    path: &Path,
    identity: &EmbeddingModelIdentity,
) -> io::Result<(SpeakerDatabase, bool)> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((SpeakerDatabase::empty(identity.clone()), false));
        }
        Err(error) => return Err(error),
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid speaker database {}: {error}", path.display()),
        )
    })?;
    if value.get("format_version").is_none() {
        let (database, _, _) = SpeakerDatabase::import_legacy(value, path)?;
        database.validate_identity(identity)?;
        return Ok((database, true));
    }
    SpeakerDatabase::load_checked(path, identity).map(|database| (database, false))
}

fn apply_correction_learning(
    database: &mut SpeakerDatabase,
    updates: &[CorrectionLearningUpdate],
) -> (usize, usize) {
    let mut learned = 0;
    let mut forgotten = 0;
    for update in updates {
        for old_name in &update.old_names {
            for vector in &update.vectors {
                forgotten += database.forget_matching(old_name, vector, SAME_VECTOR_SIMILARITY);
            }
        }
        let embeddings = &mut database
            .speakers
            .entry(update.speaker.clone())
            .or_default()
            .embeddings;
        for vector in &update.vectors {
            if embeddings.iter().any(|existing| {
                embedding::cosine(existing, vector)
                    .is_some_and(|score| score >= SAME_VECTOR_SIMILARITY)
            }) {
                continue;
            }
            embeddings.push(vector.clone());
            learned += 1;
        }
    }
    database.refresh_centroids();
    (learned, forgotten)
}

fn stage_process_args(session: PathBuf) -> ProcessArgs {
    ProcessArgs {
        session,
        meeting: None,
        whisper_model: None,
        segmentation_model: None,
        embedding_model: None,
        models_lock: None,
        allow_unverified_models: false,
        diarize_mic: true,
        no_diarize: false,
        skip_transcription: false,
        language: "auto".into(),
        threads: None,
        speakers_db: None,
        speaker_threshold: DEFAULT_SPEAKER_THRESHOLD,
        cluster_threshold: 1.0,
        num_speakers: None,
    }
}

fn prepare_meeting(
    args: &ProcessArgs,
    session: &Session,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = args.meeting.as_deref() {
        let details = meeting::read_details(path)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("meeting details {} do not exist", path.display()),
            )
        })?;
        meeting::write_details_atomic(&session.meeting_path(), &details)?;
        eprintln!("meeting details: copied {}", path.display());
    }
    Ok(())
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
    // Stages that tolerate unreadable audio would otherwise quietly produce
    // emptier results from an archived session.
    if [AudioSource::Mic, AudioSource::System]
        .into_iter()
        .any(|source| archive::is_archived(&session.stored_audio_path(source)))
    {
        archive::ensure_decoder()?;
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
    processing_progress: Option<ProcessingProgressReporter>,
    cancelled: Option<&AtomicBool>,
    transcription_progress: Option<ProgressReporter>,
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
    let processing_for_metrics = processing_progress.clone();
    let transcription_for_metrics = transcription_progress.clone();
    let combined_progress =
        if processing_for_metrics.is_some() || transcription_for_metrics.is_some() {
            Some(Arc::new(
                move |metrics: crate::transcription::TranscriptionProgress| {
                    if let Some(progress) = &processing_for_metrics {
                        let stage = match metrics.source {
                            AudioSource::Mic => ProcessingStage::TranscribingMic,
                            AudioSource::System => ProcessingStage::TranscribingSystem,
                        };
                        progress(ProcessingProgress::determinate(stage, metrics.fraction()));
                    }
                    if let Some(progress) = &transcription_for_metrics {
                        progress(metrics);
                    }
                },
            ) as ProgressReporter)
        } else {
            None
        };
    let mut transcriber = WhisperTranscriber::new(
        model,
        AudioSource::Mic,
        args.language.clone(),
        threads,
        combined_progress,
    )?;
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
        if let Some(progress) = &processing_progress {
            progress(ProcessingProgress::indeterminate(match source {
                AudioSource::Mic => ProcessingStage::TranscribingMic,
                AudioSource::System => ProcessingStage::TranscribingSystem,
            }));
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
        Err(error) if !strict => {
            eprintln!("warning: cannot read enabled {source} audio; skipping it: {error}");
            Ok(Vec::new())
        }
        Err(error) => Err(io::Error::new(
            error.kind(),
            format!(
                "cannot read enabled {source} audio {}: {error}",
                session.stored_audio_path(source).display()
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
            let path = session.stored_audio_path(source);
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
    let meeting = meeting::read_details(&session.meeting_path())?;
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
        let num_speakers = effective_num_speakers(args, meeting.as_ref(), source);
        if let Some(num_speakers) = num_speakers {
            eprintln!("diarize {source}: fixed speaker count {num_speakers}");
        }
        let result = SherpaDiarizer::new(
            segmentation_model,
            embedding_model,
            source,
            threads,
            args.cluster_threshold,
            num_speakers,
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

fn effective_num_speakers(
    args: &ProcessArgs,
    meeting: Option<&MeetingDetails>,
    source: AudioSource,
) -> Option<u32> {
    args.num_speakers.or_else(|| {
        (source == AudioSource::System)
            .then(|| meeting.map(MeetingDetails::remote_count))
            .flatten()
            .filter(|count| *count > 0)
    })
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
    progress: Option<&dyn Fn(f64)>,
) -> Result<RecognitionResult, Box<dyn std::error::Error>> {
    if segments.is_empty() {
        eprintln!("speaker recognition: no diarized clusters; skipped");
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
    let meeting = meeting::read_details(&session.meeting_path())?;
    let path = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    let strict_database = if strict {
        if !path.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("speaker database {} does not exist", path.display()),
            )
            .into());
        }
        let database = SpeakerDatabase::load(&path)?;
        validate_recognition_database(&database, &identity)?;
        for source in [AudioSource::System, AudioSource::Mic] {
            if segments.iter().any(|segment| segment.source == source) {
                read_enabled_audio(session, source, true, true)?;
            }
        }
        Some(database)
    } else {
        None
    };
    let chunks = match write_embeddings(
        session, segments, &extractor, &identity, threads, strict, progress,
    ) {
        Ok(chunks) => chunks,
        Err(error) => {
            if strict {
                return Err(error);
            }
            eprintln!("warning: speaker embeddings unavailable: {error}");
            return Ok(RecognitionResult::default());
        }
    };
    if !strict && !path.is_file() {
        eprintln!("speaker recognition: no database; skipped");
        return Ok(RecognitionResult::default());
    }
    let database = if let Some(database) = strict_database {
        database
    } else {
        match SpeakerDatabase::load(&path) {
            Ok(database) => database,
            Err(error) => {
                eprintln!("warning: speaker recognition skipped: {error}");
                return Ok(RecognitionResult::default());
            }
        }
    };
    if let Err(error) = validate_recognition_database(&database, &identity) {
        if strict {
            return Err(error.into());
        }
        eprintln!("warning: speaker recognition skipped: {error}");
        return Ok(RecognitionResult::default());
    }
    let mut result = RecognitionResult::default();
    for source in [AudioSource::System, AudioSource::Mic] {
        let mut cluster_embeddings = BTreeMap::<u32, Vec<Vec<f32>>>::new();
        for chunk in chunks.iter().filter(|chunk| {
            chunk.source == source && chunk.end_ms.saturating_sub(chunk.start_ms) >= MIN_LEARN_MS
        }) {
            cluster_embeddings
                .entry(chunk.cluster)
                .or_default()
                .push(chunk.embedding.clone());
        }
        if cluster_embeddings.is_empty() {
            continue;
        }
        let allowed = meeting
            .as_ref()
            .and_then(|meeting| allowed_candidate_names(meeting, source));
        let mut candidates = Vec::new();
        eprintln!("speaker recognition ({source}):");
        if let Some(allowed) = &allowed {
            let mut names = allowed.iter().cloned().collect::<Vec<_>>();
            names.sort();
            eprintln!("candidates\t{}", names.join(", "));
        } else {
            eprintln!("candidates\tall learned speakers");
        }
        eprintln!("cluster\tbest candidate\tscore\tassigned");
        for (cluster, embeddings) in cluster_embeddings {
            let best = best_candidate(&database, &embeddings, allowed.as_ref());
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

fn validate_recognition_database(
    database: &SpeakerDatabase,
    identity: &EmbeddingModelIdentity,
) -> io::Result<()> {
    if database
        .speakers
        .values()
        .all(|speaker| speaker.embeddings.is_empty())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "speaker database has no embeddings",
        ));
    }
    database.validate_identity(identity)
}

fn allowed_candidate_names(
    meeting: &MeetingDetails,
    source: AudioSource,
) -> Option<HashSet<String>> {
    match source {
        AudioSource::System if meeting.remote.unknown == 0 && !meeting.remote.known.is_empty() => {
            Some(meeting.remote.known.iter().cloned().collect())
        }
        AudioSource::Mic
            if meeting.local_count() > 0
                && meeting.local.unknown == 0
                && !meeting.local.known.is_empty() =>
        {
            Some(
                meeting
                    .local
                    .known
                    .iter()
                    .chain(&meeting.remote.known)
                    .cloned()
                    .collect(),
            )
        }
        _ => None,
    }
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
    diarize_mic: bool,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let meeting = meeting::read_details(&session.meeting_path())?;
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
                manifest.mic.enabled && diarize_mic,
            )?,
            system_audio_sha256: hash_audio(session, AudioSource::System, manifest.system.enabled)?,
            segmentation_model_sha256: optional_hash(args.segmentation_model.as_deref())?,
            embedding_model_sha256: optional_hash(args.embedding_model.as_deref())?,
            diarize_mic,
            cluster_threshold: args.cluster_threshold,
            num_speakers: args.num_speakers,
            mic_num_speakers: diarize_mic
                .then(|| effective_num_speakers(args, meeting.as_ref(), AudioSource::Mic))
                .flatten(),
            system_num_speakers: effective_num_speakers(
                args,
                meeting.as_ref(),
                AudioSource::System,
            ),
            threads,
        },
    )?;
    Ok(())
}

fn effective_diarize_mic(requested: bool, segments: &[SpeakerSegment]) -> bool {
    requested
        && segments
            .iter()
            .any(|segment| segment.source == AudioSource::Mic)
}

fn hash_audio(
    session: &Session,
    source: AudioSource,
    included: bool,
) -> io::Result<Option<String>> {
    let path = session.stored_audio_path(source);
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
    let (embedding_model_sha256, speakers_database_sha256) = if segments.is_empty() {
        (None, None)
    } else {
        (
            args.embedding_model
                .as_deref()
                .filter(|path| path.is_file())
                .map(models::sha256_file)
                .transpose()?,
            database
                .is_file()
                .then(|| models::sha256_file(&database))
                .transpose()?,
        )
    };
    let artifact = SpeakerAssignments {
        format_version: SPEAKER_ASSIGNMENTS_FORMAT_VERSION,
        provenance: SpeakerAssignmentProvenance {
            diarization_file: "diarization.jsonl".into(),
            diarization_sha256: models::sha256_file(&session.diarization_path())?,
            embedding_model_sha256,
            speakers_database_sha256,
            meeting_sha256: session
                .meeting_path()
                .is_file()
                .then(|| models::sha256_file(&session.meeting_path()))
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
    let corrections = read_speaker_corrections(session)?;
    let hidden = read_hidden_sources(session)?;
    let audio = match AudioEnvelopes::read(
        &session.stored_audio_path(AudioSource::Mic),
        &session.stored_audio_path(AudioSource::System),
    ) {
        Ok(audio) => Some(audio),
        Err(error) => {
            eprintln!(
                "warning: cannot compare microphone and system audio for leakage: {error}; using conservative text matching"
            );
            None
        }
    };
    let leakage = leakage::suppress_leaked_mic_words(&words, audio.as_ref());
    let meeting = meeting::read_details(&session.meeting_path())?;
    let all_utterances = utterances::build_utterances(
        &leakage.words,
        &segments,
        &recognized,
        &corrections.corrections,
        &manifest.local_speaker,
        diarize_mic,
        DEFAULT_NEAREST_TOLERANCE_MS,
        meeting.as_ref(),
    );
    let all_utterance_count = all_utterances.len();
    let utterances = all_utterances
        .into_iter()
        .filter(|utterance| !hidden.contains(&utterance.source))
        .collect::<Vec<_>>();
    let hidden_utterances = all_utterance_count - utterances.len();
    before_write();
    jsonl::write_all_atomic(&session.leakage_suppressions_path(), &leakage.suppressions)?;
    jsonl::write_all_atomic(&session.transcript_path(), &utterances)?;
    write_transcript_text(&session.transcript_text_path(), &utterances)?;
    let suppressed_words = leakage
        .suppressions
        .iter()
        .map(|suppression| suppression.suppressed_words)
        .sum::<usize>();
    let echo_utterances = utterances
        .iter()
        .filter(|utterance| utterance.echo.is_some())
        .count();
    let hidden_summary = if hidden.is_empty() {
        String::new()
    } else {
        let sources = hidden
            .iter()
            .map(|source| source.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!(", hid {hidden_utterances} utterance(s) of hidden sources ({sources})")
    };
    eprintln!(
        "merge: {} utterance(s), {} echo utterance(s), suppressed {} leaked microphone word(s) in {} region(s), in {:.1} s{hidden_summary}",
        utterances.len(),
        echo_utterances,
        suppressed_words,
        leakage.suppressions.len(),
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
    let path = session.stored_audio_path(source);
    match models::sha256_file(&path) {
        Ok(actual) if !actual.eq_ignore_ascii_case(expected) => eprintln!(
            "warning: {metadata_name} was produced from different {source} audio; its output may be stale"
        ),
        Err(error) => eprintln!("warning: cannot fingerprint {}: {error}", path.display()),
        _ => {}
    }
}

/// Archiving replaces a track's file. Stage metadata that fingerprinted the
/// old file follows it to the new one, so its outputs are not reported stale.
pub fn rebase_audio_provenance(session: &Session, source: AudioSource, old: &str, new: &str) {
    fn rebase<T: Serialize + DeserializeOwned>(
        path: &Path,
        old: &str,
        new: &str,
        hash: impl Fn(&mut T) -> &mut Option<String>,
    ) -> io::Result<()> {
        let mut metadata: T = match read_json(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let hash = hash(&mut metadata);
        if !hash
            .as_deref()
            .is_some_and(|hash| hash.eq_ignore_ascii_case(old))
        {
            return Ok(());
        }
        *hash = Some(new.to_owned());
        write_json_atomic(path, &metadata)
    }
    let words = session.words_metadata_path();
    let diarization = session.diarization_metadata_path();
    let embeddings = session.embeddings_metadata_path();
    let results = [
        (
            &words,
            rebase(
                &words,
                old,
                new,
                |metadata: &mut TranscriptionMetadata| match source {
                    AudioSource::Mic => &mut metadata.mic_audio_sha256,
                    AudioSource::System => &mut metadata.system_audio_sha256,
                },
            ),
        ),
        (
            &diarization,
            rebase(
                &diarization,
                old,
                new,
                |metadata: &mut DiarizationMetadata| match source {
                    AudioSource::Mic => &mut metadata.mic_audio_sha256,
                    AudioSource::System => &mut metadata.system_audio_sha256,
                },
            ),
        ),
        (
            &embeddings,
            rebase(
                &embeddings,
                old,
                new,
                |metadata: &mut EmbeddingsMetadata| match source {
                    AudioSource::Mic => &mut metadata.mic_audio_sha256,
                    AudioSource::System => &mut metadata.system_audio_sha256,
                },
            ),
        ),
    ];
    for (path, result) in results {
        if let Err(error) = result {
            eprintln!("warning: cannot update {}: {error}", path.display());
        }
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

fn read_optional_file(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn restore_file(path: &Path, previous: Option<&[u8]>) -> io::Result<()> {
    match (fs::read(path), previous) {
        (Ok(current), Some(previous)) if current == previous => return Ok(()),
        (Err(error), None) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        _ => {}
    }
    let Some(previous) = previous else {
        return match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        };
    };
    let tmp = jsonl::tmp_path(path);
    {
        let mut file = crate::session::create_private_file(&tmp)?;
        file.write_all(previous)?;
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

fn ensure_embeddings(
    session: &Session,
    segments: &[SpeakerSegment],
    extractor: &EmbeddingExtractor,
    identity: &EmbeddingModelIdentity,
    threads: usize,
) -> Result<Vec<EmbeddingChunk>, Box<dyn std::error::Error>> {
    ensure_embeddings_with(session, segments, identity, || {
        write_embeddings(session, segments, extractor, identity, threads, true, None)
    })
}

fn ensure_embeddings_with(
    session: &Session,
    segments: &[SpeakerSegment],
    identity: &EmbeddingModelIdentity,
    regenerate: impl FnOnce() -> Result<Vec<EmbeddingChunk>, Box<dyn std::error::Error>>,
) -> Result<Vec<EmbeddingChunk>, Box<dyn std::error::Error>> {
    let current = (|| -> Result<Option<Vec<EmbeddingChunk>>, Box<dyn std::error::Error>> {
        let metadata: EmbeddingsMetadata = read_json(&session.embeddings_metadata_path())?;
        if !embeddings_metadata_is_current(session, segments, identity, &metadata)? {
            return Ok(None);
        }
        let chunks = read_jsonl_artifact(&session.embeddings_path(), "speaker embeddings")?;
        Ok(Some(chunks))
    })();
    if let Ok(Some(chunks)) = current {
        return Ok(chunks);
    }
    regenerate()
}

fn write_embeddings(
    session: &Session,
    segments: &[SpeakerSegment],
    extractor: &EmbeddingExtractor,
    identity: &EmbeddingModelIdentity,
    threads: usize,
    strict_audio: bool,
    progress: Option<&dyn Fn(f64)>,
) -> Result<Vec<EmbeddingChunk>, Box<dyn std::error::Error>> {
    let total_chunks = segments
        .iter()
        .map(|segment| chunk_ranges(segment).len())
        .sum::<usize>();
    if let Some(progress) = progress {
        progress(0.0);
    }
    let mut completed = 0usize;
    let mut chunks = Vec::with_capacity(total_chunks);
    let mut readable_sources = HashSet::new();
    for source in [AudioSource::System, AudioSource::Mic] {
        let source_segments = segments
            .iter()
            .filter(|segment| segment.source == source)
            .collect::<Vec<_>>();
        if source_segments.is_empty() {
            continue;
        }
        let Some(samples) = read_embedding_audio(session, source, strict_audio)? else {
            completed += source_segments
                .iter()
                .map(|segment| chunk_ranges(segment).len())
                .sum::<usize>();
            if let Some(progress) = progress
                && total_chunks > 0
            {
                progress(completed as f64 / total_chunks as f64);
            }
            continue;
        };
        readable_sources.insert(source);
        for segment in source_segments {
            for (start_ms, end_ms) in chunk_ranges(segment) {
                let start = ms_to_index(start_ms, samples.len());
                let end = ms_to_index(end_ms, samples.len());
                if end > start {
                    if let Some(embedding) = extractor.embed(&samples[start..end]) {
                        chunks.push(EmbeddingChunk {
                            source,
                            start_ms,
                            end_ms,
                            cluster: segment.cluster,
                            embedding,
                        });
                    } else {
                        eprintln!("warning: could not embed {source} chunk {start_ms}..{end_ms}");
                    }
                }
                completed += 1;
                if let Some(progress) = progress
                    && total_chunks > 0
                {
                    progress(completed as f64 / total_chunks as f64);
                }
            }
        }
    }
    chunks.sort_by_key(|chunk| {
        (
            chunk.start_ms,
            source_order(chunk.source),
            chunk.cluster,
            chunk.end_ms,
        )
    });
    let mic_audio_sha256 = hash_audio(
        session,
        AudioSource::Mic,
        readable_sources.contains(&AudioSource::Mic),
    )?;
    let system_audio_sha256 = hash_audio(
        session,
        AudioSource::System,
        readable_sources.contains(&AudioSource::System),
    )?;
    let diarization_sha256 = models::sha256_file(&session.diarization_path())?;
    let outputs = [
        session.embeddings_path(),
        session.embeddings_metadata_path(),
    ];
    let previous_outputs = outputs
        .iter()
        .map(|path| read_optional_file(path))
        .collect::<Result<Vec<_>, _>>()?;
    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        jsonl::write_all_atomic(&session.embeddings_path(), &chunks)?;
        let metadata = EmbeddingsMetadata {
            format_version: EMBEDDINGS_FORMAT_VERSION,
            output_file: "embeddings.jsonl".into(),
            output_sha256: models::sha256_file(&session.embeddings_path())?,
            diarization_sha256,
            mic_audio_sha256,
            system_audio_sha256,
            embedding_model_sha256: identity.sha256.clone(),
            threads,
            window_ms: EMBEDDING_WINDOW_MS,
            min_learn_ms: MIN_LEARN_MS,
        };
        write_json_atomic(&session.embeddings_metadata_path(), &metadata)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        for (path, previous) in outputs.iter().zip(&previous_outputs) {
            if let Err(restore_error) = restore_file(path, previous.as_deref()) {
                return Err(io::Error::other(format!(
                    "writing speaker embeddings failed: {error}; restoring {} also failed: {restore_error}",
                    path.display()
                ))
                .into());
            }
        }
        return Err(error);
    }
    Ok(chunks)
}

fn read_embedding_audio(
    session: &Session,
    source: AudioSource,
    strict: bool,
) -> io::Result<Option<Vec<f32>>> {
    match session.read_audio(source) {
        Ok(samples) => Ok(Some(samples)),
        Err(error) if strict => Err(io::Error::new(
            error.kind(),
            format!(
                "cannot read {source} audio {}: {error}",
                session.stored_audio_path(source).display()
            ),
        )),
        Err(error) => {
            eprintln!("warning: cannot read {source} audio for speaker recognition: {error}");
            Ok(None)
        }
    }
}

fn embeddings_metadata_is_current(
    session: &Session,
    segments: &[SpeakerSegment],
    identity: &EmbeddingModelIdentity,
    metadata: &EmbeddingsMetadata,
) -> io::Result<bool> {
    if metadata.format_version != EMBEDDINGS_FORMAT_VERSION
        || metadata.output_file != "embeddings.jsonl"
        || metadata.window_ms != EMBEDDING_WINDOW_MS
        || metadata.min_learn_ms != MIN_LEARN_MS
        || !metadata
            .embedding_model_sha256
            .eq_ignore_ascii_case(&identity.sha256)
        || !models::sha256_file(&session.diarization_path())?
            .eq_ignore_ascii_case(&metadata.diarization_sha256)
        || !models::sha256_file(&session.embeddings_path())?
            .eq_ignore_ascii_case(&metadata.output_sha256)
    {
        return Ok(false);
    }
    let (mic, system) = embeddings_audio_hashes(session, segments)?;
    Ok(
        option_hashes_equal(mic.as_deref(), metadata.mic_audio_sha256.as_deref())
            && option_hashes_equal(system.as_deref(), metadata.system_audio_sha256.as_deref()),
    )
}

fn embeddings_audio_hashes(
    session: &Session,
    segments: &[SpeakerSegment],
) -> io::Result<(Option<String>, Option<String>)> {
    Ok((
        hash_audio(
            session,
            AudioSource::Mic,
            segments
                .iter()
                .any(|segment| segment.source == AudioSource::Mic),
        )?,
        hash_audio(
            session,
            AudioSource::System,
            segments
                .iter()
                .any(|segment| segment.source == AudioSource::System),
        )?,
    ))
}

fn option_hashes_equal(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
        (None, None) => true,
        _ => false,
    }
}

fn chunk_ranges(segment: &SpeakerSegment) -> Vec<(u64, u64)> {
    let duration = segment.end_ms.saturating_sub(segment.start_ms);
    if duration == 0 {
        return Vec::new();
    }
    let count = duration.div_ceil(EMBEDDING_WINDOW_MS);
    (0..count)
        .map(|index| {
            let offset =
                |part: u64| (u128::from(duration) * u128::from(part) / u128::from(count)) as u64;
            (
                segment.start_ms.saturating_add(offset(index)),
                segment.start_ms.saturating_add(offset(index + 1)),
            )
        })
        .collect()
}

fn embeddings_in_range(
    chunks: &[EmbeddingChunk],
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
) -> Vec<Vec<f32>> {
    chunks
        .iter()
        .filter(|chunk| {
            if chunk.source != source {
                return false;
            }
            let duration = chunk.end_ms.saturating_sub(chunk.start_ms);
            if duration < MIN_LEARN_MS {
                return false;
            }
            let overlap = chunk
                .end_ms
                .min(end_ms)
                .saturating_sub(chunk.start_ms.max(start_ms));
            overlap.saturating_mul(2) >= duration
        })
        .map(|chunk| chunk.embedding.clone())
        .collect()
}

/// Marks the locked lines whose voice is stored under their name in the database.
pub fn learned_lines(
    session: &Session,
    database_path: &Path,
    utterances: &[Utterance],
) -> Vec<bool> {
    if !utterances.iter().any(|utterance| utterance.locked) {
        return vec![false; utterances.len()];
    }
    let chunks: Vec<EmbeddingChunk> =
        jsonl::read_all(&session.embeddings_path()).unwrap_or_default();
    let Ok(database) = SpeakerDatabase::load(database_path) else {
        return vec![false; utterances.len()];
    };
    utterances
        .iter()
        .map(|utterance| line_is_learned(utterance, &chunks, &database))
        .collect()
}

fn line_is_learned(
    utterance: &Utterance,
    chunks: &[EmbeddingChunk],
    database: &SpeakerDatabase,
) -> bool {
    if !utterance.locked {
        return false;
    }
    let Some(record) = database.speakers.get(&utterance.speaker) else {
        return false;
    };
    embeddings_in_range(
        chunks,
        utterance.source,
        utterance.start_ms,
        utterance.end_ms,
    )
    .iter()
    .any(|vector| {
        record.embeddings.iter().any(|stored| {
            embedding::cosine(stored, vector).is_some_and(|score| score >= SAME_VECTOR_SIMILARITY)
        })
    })
}

fn score_speaker_name(database: &SpeakerDatabase, name: &str, query: &[Vec<f32>]) -> Option<f32> {
    embedding::speaker_score(query, &database.speakers.get(name)?.embeddings)
}

fn centroid_prefilter<'a>(
    database: &'a SpeakerDatabase,
    query: &[Vec<f32>],
    allowed: Option<&HashSet<String>>,
) -> Vec<(&'a str, &'a database::SpeakerRecord)> {
    let Some(query_centroid) = embedding::mean_normalized(query) else {
        return Vec::new();
    };
    let mut candidates = database
        .speakers
        .iter()
        .filter(|(name, _)| allowed.is_none_or(|allowed| allowed.contains(*name)))
        .filter_map(|(name, speaker)| {
            embedding::cosine(&query_centroid, speaker.centroid.as_deref()?)
                .map(|score| (name.as_str(), speaker, score))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.2.total_cmp(&left.2).then_with(|| left.0.cmp(right.0)));
    candidates.truncate(5);
    candidates
        .into_iter()
        .map(|(name, speaker, _)| (name, speaker))
        .collect()
}

fn best_candidate(
    database: &SpeakerDatabase,
    query: &[Vec<f32>],
    allowed: Option<&HashSet<String>>,
) -> Option<(String, f32)> {
    centroid_prefilter(database, query, allowed)
        .into_iter()
        .filter_map(|(name, _)| {
            score_speaker_name(database, name, query).map(|score| (name.to_owned(), score))
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
                "[{}] {}{}: {}",
                format_timestamp(utterance.start_ms),
                utterance.speaker,
                if utterance.echo.is_some() {
                    " [echo]"
                } else {
                    ""
                },
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
    fn stage_progress_is_indeterminate_or_clamped() {
        assert_eq!(
            ProcessingProgress::indeterminate(ProcessingStage::Diarizing).fraction,
            None
        );
        assert_eq!(
            ProcessingProgress::determinate(ProcessingStage::Recognizing, 1.5).fraction,
            Some(1.0)
        );
        assert_eq!(
            ProcessingProgress::determinate(ProcessingStage::TranscribingMic, -0.5).fraction,
            Some(0.0)
        );
    }

    #[test]
    fn chunk_windows_are_equal_and_keep_short_segments() {
        let ranges = |duration: u64| {
            chunk_ranges(&SpeakerSegment {
                source: AudioSource::System,
                start_ms: 100,
                end_ms: 100 + duration,
                cluster: 0,
            })
            .into_iter()
            .map(|(start, end)| end - start)
            .collect::<Vec<_>>()
        };
        assert!(ranges(0).is_empty());
        assert_eq!(ranges(1_499), [1_499]);
        assert_eq!(ranges(1_500), [1_500]);
        assert_eq!(ranges(10_000), [10_000]);
        assert_eq!(ranges(10_001), [5_000, 5_001]);
        assert_eq!(ranges(20_000), [10_000, 10_000]);
        assert_eq!(ranges(20_001), [6_667, 6_667, 6_667]);
    }

    #[test]
    fn embedding_range_requires_half_overlap_and_minimum_duration() {
        let chunks = [
            EmbeddingChunk {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                cluster: 1,
                embedding: vec![1.0, 0.0],
            },
            EmbeddingChunk {
                source: AudioSource::System,
                start_ms: 2_000,
                end_ms: 3_499,
                cluster: 1,
                embedding: vec![0.0, 1.0],
            },
            EmbeddingChunk {
                source: AudioSource::Mic,
                start_ms: 0,
                end_ms: 2_000,
                cluster: 1,
                embedding: vec![-1.0, 0.0],
            },
        ];

        assert_eq!(
            embeddings_in_range(&chunks, AudioSource::System, 1_000, 2_000),
            vec![vec![1.0, 0.0]]
        );
        assert!(embeddings_in_range(&chunks, AudioSource::System, 1_001, 2_000).is_empty());
    }

    #[test]
    fn cluster_for_request_prefers_an_exact_transcript_row() {
        let request = CorrectionRequest {
            source: AudioSource::System,
            start_ms: 10,
            end_ms: 20,
            speaker: "Alice".into(),
        };
        let utterances = [Utterance {
            start_ms: 10,
            end_ms: 20,
            source: AudioSource::System,
            speaker_id: "spk_9".into(),
            speaker: "SPEAKER_09".into(),
            text: "hello".into(),
            locked: false,
            echo: None,
        }];
        let segments = [SpeakerSegment {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 30,
            cluster: 7,
        }];

        assert_eq!(
            cluster_for_request(&request, &utterances, &segments),
            Some(9)
        );
    }

    #[test]
    fn cluster_for_request_falls_back_to_closed_segment_overlap() {
        let request = CorrectionRequest {
            source: AudioSource::System,
            start_ms: 30,
            end_ms: 30,
            speaker: "Alice".into(),
        };
        let segments = [SpeakerSegment {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 30,
            cluster: 7,
        }];

        assert_eq!(cluster_for_request(&request, &[], &segments), Some(7));
    }

    #[test]
    fn correction_validation_accepts_a_point_range() {
        let request = CorrectionRequest {
            source: AudioSource::System,
            start_ms: 30,
            end_ms: 30,
            speaker: "Alice".into(),
        };

        assert_eq!(validate_correction_request(&request).unwrap(), request);
    }

    #[test]
    fn correction_echo_uses_recognized_system_names() {
        let requests = [CorrectionRequest {
            source: AudioSource::Mic,
            start_ms: 0,
            end_ms: 0,
            speaker: "Laura".into(),
        }];
        let recognized = HashMap::from([((AudioSource::System, 2), "Laura".into())]);

        assert!(correction_is_system_echo(&requests, &recognized, &[]));
    }

    #[test]
    fn correction_echo_uses_hand_corrected_system_names() {
        let requests = [CorrectionRequest {
            source: AudioSource::Mic,
            start_ms: 0,
            end_ms: 0,
            speaker: "Laura".into(),
        }];
        let corrections = [SpeakerCorrection {
            source: AudioSource::System,
            start_ms: 10,
            end_ms: 20,
            speaker: "Laura".into(),
            inferred: false,
        }];

        assert!(correction_is_system_echo(
            &requests,
            &HashMap::new(),
            &corrections
        ));
    }

    #[test]
    fn centroid_prefilter_keeps_only_five_nearest_speakers() {
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "m".into(),
            sha256: "x".into(),
            dimension: 2,
        });
        for (name, x) in [
            ("A", 1.0f32),
            ("B", 0.9),
            ("C", 0.8),
            ("D", 0.7),
            ("E", 0.6),
            ("F", -1.0),
        ] {
            database.speakers.insert(
                name.into(),
                SpeakerRecord {
                    embeddings: vec![vec![x, (1.0 - x * x).max(0.0).sqrt()]],
                    centroid: None,
                },
            );
        }
        database.refresh_centroids();
        let names = centroid_prefilter(&database, &[vec![1.0, 0.0]], None)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["A", "B", "C", "D", "E"]);
    }

    #[test]
    fn meeting_speaker_count_only_applies_to_system_and_explicit_wins() {
        let mut args = stage_process_args(PathBuf::from("session"));
        let meeting = MeetingDetails::new(
            String::new(),
            crate::meeting::Attendees {
                known: vec!["Laura".into()],
                unknown: 0,
            },
            crate::meeting::Attendees {
                known: vec!["Andrew".into(), "Craig".into()],
                unknown: 1,
            },
        );
        assert_eq!(
            effective_num_speakers(&args, Some(&meeting), AudioSource::System),
            Some(3)
        );
        assert_eq!(
            effective_num_speakers(&args, Some(&meeting), AudioSource::Mic),
            None
        );
        args.num_speakers = Some(4);
        assert_eq!(
            effective_num_speakers(&args, Some(&meeting), AudioSource::System),
            Some(4)
        );
        assert_eq!(
            effective_num_speakers(&args, Some(&meeting), AudioSource::Mic),
            Some(4)
        );
    }

    #[test]
    fn meeting_candidate_restriction_filters_database_names() {
        let meeting = MeetingDetails::new(
            String::new(),
            crate::meeting::Attendees {
                known: vec!["Laura".into()],
                unknown: 0,
            },
            crate::meeting::Attendees {
                known: vec!["Bob".into()],
                unknown: 0,
            },
        );
        assert_eq!(
            allowed_candidate_names(&meeting, AudioSource::System),
            Some(HashSet::from(["Bob".into()]))
        );
        assert_eq!(
            allowed_candidate_names(&meeting, AudioSource::Mic),
            Some(HashSet::from(["Bob".into(), "Laura".into()]))
        );
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "m".into(),
            sha256: "x".into(),
            dimension: 2,
        });
        database.speakers.insert(
            "Alice".into(),
            SpeakerRecord {
                embeddings: vec![vec![1.0, 0.0]],
                centroid: None,
            },
        );
        database.speakers.insert(
            "Bob".into(),
            SpeakerRecord {
                embeddings: vec![vec![0.0, 1.0]],
                centroid: None,
            },
        );
        database.refresh_centroids();
        let allowed = HashSet::from(["Bob".into()]);
        assert_eq!(
            best_candidate(&database, &[vec![1.0, 0.0]], Some(&allowed))
                .expect("filtered candidate")
                .0,
            "Bob"
        );
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
                locked: false,
                echo: None,
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

    fn hidden_sources_fixture(label: &str) -> (PathBuf, Session) {
        let root = std::env::temp_dir().join(format!(
            "singstone-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        jsonl::write_all_atomic(
            &session.words_path(),
            &[
                TimedWord {
                    source: AudioSource::Mic,
                    start_ms: 100,
                    end_ms: 300,
                    text: "microphone".into(),
                },
                TimedWord {
                    source: AudioSource::System,
                    start_ms: 1_000,
                    end_ms: 1_300,
                    text: "system".into(),
                },
            ],
        )
        .expect("write words");
        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[
                SpeakerSegment {
                    source: AudioSource::Mic,
                    start_ms: 0,
                    end_ms: 500,
                    cluster: 7,
                },
                SpeakerSegment {
                    source: AudioSource::System,
                    start_ms: 900,
                    end_ms: 1_500,
                    cluster: 3,
                },
            ],
        )
        .expect("write diarization");
        (root, session)
    }

    fn embedding_cache_fixture(
        label: &str,
    ) -> (
        PathBuf,
        Session,
        Vec<SpeakerSegment>,
        EmbeddingModelIdentity,
        Vec<EmbeddingChunk>,
    ) {
        let root = std::env::temp_dir().join(format!(
            "singstone-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        let segments = vec![SpeakerSegment {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            cluster: 7,
        }];
        jsonl::write_all_atomic(&session.diarization_path(), &segments).expect("write diarization");
        fs::write(
            session.audio_path(AudioSource::System),
            [0.0f32, 0.25, -0.25, 0.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .expect("write audio");
        let chunks = vec![EmbeddingChunk {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            cluster: 7,
            embedding: vec![1.0, 0.0],
        }];
        jsonl::write_all_atomic(&session.embeddings_path(), &chunks).expect("write embeddings");
        let identity = EmbeddingModelIdentity {
            name: "model".into(),
            sha256: "model-hash".into(),
            dimension: 2,
        };
        write_json_atomic(
            &session.embeddings_metadata_path(),
            &EmbeddingsMetadata {
                format_version: EMBEDDINGS_FORMAT_VERSION,
                output_file: "embeddings.jsonl".into(),
                output_sha256: models::sha256_file(&session.embeddings_path())
                    .expect("hash embeddings"),
                diarization_sha256: models::sha256_file(&session.diarization_path())
                    .expect("hash diarization"),
                mic_audio_sha256: None,
                system_audio_sha256: Some(
                    models::sha256_file(&session.audio_path(AudioSource::System))
                        .expect("hash audio"),
                ),
                embedding_model_sha256: identity.sha256.clone(),
                threads: 1,
                window_ms: EMBEDDING_WINDOW_MS,
                min_learn_ms: MIN_LEARN_MS,
            },
        )
        .expect("write metadata");
        (root, session, segments, identity, chunks)
    }

    #[test]
    fn ensure_embeddings_reuses_current_artifacts() {
        let (root, session, segments, identity, chunks) =
            embedding_cache_fixture("embeddings-reuse");
        let regenerated = std::cell::Cell::new(false);

        let actual = ensure_embeddings_with(&session, &segments, &identity, || {
            regenerated.set(true);
            Ok(Vec::new())
        })
        .expect("reuse embeddings");

        assert_eq!(actual, chunks);
        assert!(!regenerated.get());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn ensure_embeddings_regenerates_after_diarization_changes() {
        let (root, session, mut segments, identity, _) =
            embedding_cache_fixture("embeddings-regenerate");
        segments[0].cluster = 8;
        jsonl::write_all_atomic(&session.diarization_path(), &segments)
            .expect("replace diarization");
        let replacement = vec![EmbeddingChunk {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            cluster: 8,
            embedding: vec![0.0, 1.0],
        }];
        let regenerated = std::cell::Cell::new(false);

        let actual = ensure_embeddings_with(&session, &segments, &identity, || {
            regenerated.set(true);
            Ok(replacement.clone())
        })
        .expect("regenerate embeddings");

        assert_eq!(actual, replacement);
        assert!(regenerated.get());
        fs::remove_dir_all(root).expect("remove fixture");
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
            meeting: None,
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
            meeting: None,
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
            meeting: None,
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
    fn reprocessing_preserves_recorded_audio() {
        let root = std::env::temp_dir().join(format!(
            "singstone-reprocess-audio-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        let session = empty_session(&root);
        let mut manifest = session.read_manifest().expect("read manifest");
        manifest.mic.enabled = true;
        manifest.system.enabled = true;
        session
            .write_manifest(&manifest)
            .expect("enable audio tracks");

        let mic_audio = (0..64)
            .flat_map(|sample| ((sample as f32 - 32.0) / 64.0).to_le_bytes())
            .collect::<Vec<_>>();
        let system_audio = (0..96)
            .flat_map(|sample| ((48.0 - sample as f32) / 96.0).to_le_bytes())
            .collect::<Vec<_>>();
        let mic_timeline = b"mic timeline sentinel\n".to_vec();
        let system_timeline = b"system timeline sentinel\n".to_vec();
        fs::write(session.audio_path(AudioSource::Mic), &mic_audio).expect("write mic audio");
        fs::write(session.audio_path(AudioSource::System), &system_audio)
            .expect("write system audio");
        fs::write(session.timeline_path(AudioSource::Mic), &mic_timeline)
            .expect("write mic timeline");
        fs::write(session.timeline_path(AudioSource::System), &system_timeline)
            .expect("write system timeline");
        jsonl::write_all_atomic::<TimedWord>(&session.words_path(), &[]).expect("write words");
        fs::write(session.transcript_path(), b"stale transcript\n")
            .expect("write stale transcript");
        let corrections = SpeakerCorrections {
            format_version: SPEAKER_CORRECTIONS_FORMAT_VERSION,
            corrections: vec![SpeakerCorrection {
                source: AudioSource::Mic,
                start_ms: 10,
                end_ms: 20,
                speaker: "Alice".into(),
                inferred: false,
            }],
        };
        write_json_atomic(&session.speaker_corrections_path(), &corrections)
            .expect("write corrections");
        let correction_bytes =
            fs::read(session.speaker_corrections_path()).expect("read corrections");

        run(ProcessArgs {
            session: session.dir.clone(),
            meeting: None,
            whisper_model: None,
            segmentation_model: None,
            embedding_model: None,
            models_lock: None,
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
        .expect("reprocess session");

        assert_eq!(
            fs::read(session.audio_path(AudioSource::Mic)).expect("read mic audio"),
            mic_audio
        );
        assert_eq!(
            fs::read(session.audio_path(AudioSource::System)).expect("read system audio"),
            system_audio
        );
        assert_eq!(
            fs::read(session.timeline_path(AudioSource::Mic)).expect("read mic timeline"),
            mic_timeline
        );
        assert_eq!(
            fs::read(session.timeline_path(AudioSource::System)).expect("read system timeline"),
            system_timeline
        );
        assert_ne!(
            fs::read(session.transcript_path()).expect("read replaced transcript"),
            b"stale transcript\n"
        );
        assert_eq!(
            fs::read(session.speaker_corrections_path()).expect("reread corrections"),
            correction_bytes
        );
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
                meeting_sha256: None,
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

    fn assignment_fixture(label: &str, speaker: Option<&str>) -> (PathBuf, Session) {
        let root = std::env::temp_dir().join(format!(
            "singstone-{label}-{}-{}",
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
                    meeting_sha256: None,
                    speaker_threshold: 0.6,
                },
                assignments: vec![SpeakerAssignment {
                    source: AudioSource::System,
                    cluster: 7,
                    best_candidate: None,
                    score: None,
                    speaker: speaker.map(str::to_owned),
                }],
            },
        )
        .expect("write assignments");
        (root, session)
    }

    #[test]
    fn correction_without_model_locks_and_rerenders_only_its_range() {
        let (root, session) = assignment_fixture("speaker-correction", None);
        let request = CorrectionRequest {
            source: AudioSource::System,
            start_ms: 10,
            end_ms: 20,
            speaker: " Carol ".into(),
        };
        let outcome = correct_speaker(stage_process_args(session.dir.clone()), &request)
            .expect("correct speaker");
        assert_eq!(outcome.learned, 0);
        assert_eq!(outcome.forgotten, 0);
        assert!(!outcome.learning_available);
        assert!(outcome.proposals.is_empty());
        let corrections: SpeakerCorrections =
            read_json(&session.speaker_corrections_path()).expect("read corrections");
        assert_eq!(corrections.corrections[0].speaker, "Carol");
        let transcript: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read transcript");
        assert_eq!(transcript[0].speaker, "Carol");
        assert!(transcript[0].locked);
        let assignments: SpeakerAssignments =
            read_json(&session.speaker_assignments_path()).expect("read assignments");
        assert_eq!(assignments.assignments[0].speaker, None);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn correction_resolves_existing_spelling_for_lock_and_learning() {
        let Some(models) = std::env::var_os("SINGSTONE_TEST_MODELS") else {
            return;
        };
        let (root, session) = assignment_fixture("correction-existing-spelling", None);
        let model = PathBuf::from(models).join("nemo_en_titanet_small.onnx");
        let extractor = EmbeddingExtractor::new(&model, 1).expect("open embedding model");
        let identity = database::identity(&model, extractor.dimension()).expect("model identity");
        let database_path = root.join("speakers.json");
        let mut database = SpeakerDatabase::empty(identity.clone());
        database.speakers.insert("Laura".into(), Default::default());
        database.save(&database_path).expect("write database");

        jsonl::write_all_atomic(
            &session.words_path(),
            &[TimedWord {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                text: "hello".into(),
            }],
        )
        .expect("replace words");
        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                cluster: 7,
            }],
        )
        .expect("replace diarization");
        let mut assignments: SpeakerAssignments =
            read_json(&session.speaker_assignments_path()).expect("read assignments");
        assignments.provenance.diarization_sha256 =
            models::sha256_file(&session.diarization_path()).expect("hash diarization");
        write_json_atomic(&session.speaker_assignments_path(), &assignments)
            .expect("update assignments");

        let mut vector = vec![0.0; identity.dimension];
        vector[0] = 1.0;
        jsonl::write_all_atomic(
            &session.embeddings_path(),
            &[EmbeddingChunk {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                cluster: 7,
                embedding: vector,
            }],
        )
        .expect("write cached embeddings");
        write_json_atomic(
            &session.embeddings_metadata_path(),
            &EmbeddingsMetadata {
                format_version: EMBEDDINGS_FORMAT_VERSION,
                output_file: "embeddings.jsonl".into(),
                output_sha256: models::sha256_file(&session.embeddings_path())
                    .expect("hash embeddings"),
                diarization_sha256: models::sha256_file(&session.diarization_path())
                    .expect("hash diarization"),
                mic_audio_sha256: None,
                system_audio_sha256: None,
                embedding_model_sha256: identity.sha256,
                threads: 1,
                window_ms: EMBEDDING_WINDOW_MS,
                min_learn_ms: MIN_LEARN_MS,
            },
        )
        .expect("write embeddings metadata");
        let mut args = stage_process_args(session.dir.clone());
        args.embedding_model = Some(model);
        args.allow_unverified_models = true;
        args.threads = Some(1);
        args.speakers_db = Some(database_path.clone());

        let outcome = correct_speaker(
            args,
            &CorrectionRequest {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                speaker: "laura".into(),
            },
        )
        .expect("correct speaker");

        assert_eq!(outcome.learned, 1);
        let corrections: SpeakerCorrections =
            read_json(&session.speaker_corrections_path()).expect("read corrections");
        assert_eq!(corrections.corrections[0].speaker, "Laura");
        let transcript: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read transcript");
        assert_eq!(transcript[0].speaker, "Laura");
        assert!(transcript[0].locked);
        let database = SpeakerDatabase::load(&database_path).expect("read learned database");
        assert_eq!(database.speakers["Laura"].embeddings.len(), 1);
        assert!(!database.speakers.contains_key("laura"));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn correction_learning_failure_does_not_write_correction() {
        let (root, session) = assignment_fixture("correction-learning-failure", None);
        let mut args = stage_process_args(session.dir.clone());
        args.embedding_model = Some(root.join("missing-embedding.onnx"));
        args.allow_unverified_models = true;

        correct_speaker(
            args,
            &CorrectionRequest {
                source: AudioSource::System,
                start_ms: 10,
                end_ms: 20,
                speaker: "Carol".into(),
            },
        )
        .expect_err("invalid learning setup fails correction");

        assert!(!session.speaker_corrections_path().exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn failed_render_restores_previous_corrections() {
        let (root, session) = assignment_fixture("correction-render-rollback", None);
        let previous = SpeakerCorrections {
            format_version: SPEAKER_CORRECTIONS_FORMAT_VERSION,
            corrections: vec![SpeakerCorrection {
                source: AudioSource::Mic,
                start_ms: 1,
                end_ms: 2,
                speaker: "Pat".into(),
                inferred: false,
            }],
        };
        write_json_atomic(&session.speaker_corrections_path(), &previous)
            .expect("write previous corrections");
        let previous_bytes =
            fs::read(session.speaker_corrections_path()).expect("read previous corrections");
        fs::write(
            session.diarization_metadata_path(),
            br#"{"format_version":999}"#,
        )
        .expect("write invalid metadata version");

        correct_speaker(
            stage_process_args(session.dir.clone()),
            &CorrectionRequest {
                source: AudioSource::System,
                start_ms: 10,
                end_ms: 20,
                speaker: "Carol".into(),
            },
        )
        .expect_err("render failure rolls back correction");

        assert_eq!(
            fs::read(session.speaker_corrections_path()).expect("read restored corrections"),
            previous_bytes
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn failed_render_removes_a_new_corrections_file() {
        let (root, session) = assignment_fixture("correction-render-delete", None);
        fs::write(
            session.diarization_metadata_path(),
            br#"{"format_version":999}"#,
        )
        .expect("write invalid metadata version");

        correct_speaker(
            stage_process_args(session.dir.clone()),
            &CorrectionRequest {
                source: AudioSource::System,
                start_ms: 10,
                end_ms: 20,
                speaker: "Carol".into(),
            },
        )
        .expect_err("render failure rolls back correction");

        assert!(!session.speaker_corrections_path().exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn failed_render_does_not_apply_prepared_learning() {
        let Some(models) = std::env::var_os("SINGSTONE_TEST_MODELS") else {
            return;
        };
        let (root, session) = assignment_fixture("correction-render-learning", None);
        let model = PathBuf::from(models).join("nemo_en_titanet_small.onnx");
        let extractor = EmbeddingExtractor::new(&model, 1).expect("open embedding model");
        let identity = database::identity(&model, extractor.dimension()).expect("model identity");
        let database_path = root.join("speakers.json");
        SpeakerDatabase::empty(identity.clone())
            .save(&database_path)
            .expect("write database");
        let previous_database = fs::read(&database_path).expect("read database");
        let segments = [SpeakerSegment {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            cluster: 7,
        }];
        jsonl::write_all_atomic(&session.diarization_path(), &segments)
            .expect("replace diarization");
        let mut vector = vec![0.0; identity.dimension];
        vector[0] = 1.0;
        let chunks = [EmbeddingChunk {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            cluster: 7,
            embedding: vector,
        }];
        jsonl::write_all_atomic(&session.embeddings_path(), &chunks)
            .expect("write cached embeddings");
        write_json_atomic(
            &session.embeddings_metadata_path(),
            &EmbeddingsMetadata {
                format_version: EMBEDDINGS_FORMAT_VERSION,
                output_file: "embeddings.jsonl".into(),
                output_sha256: models::sha256_file(&session.embeddings_path())
                    .expect("hash embeddings"),
                diarization_sha256: models::sha256_file(&session.diarization_path())
                    .expect("hash diarization"),
                mic_audio_sha256: None,
                system_audio_sha256: None,
                embedding_model_sha256: identity.sha256,
                threads: 1,
                window_ms: EMBEDDING_WINDOW_MS,
                min_learn_ms: MIN_LEARN_MS,
            },
        )
        .expect("write embeddings metadata");
        fs::write(
            session.diarization_metadata_path(),
            br#"{"format_version":999}"#,
        )
        .expect("write invalid diarization metadata");
        let mut args = stage_process_args(session.dir.clone());
        args.embedding_model = Some(model);
        args.allow_unverified_models = true;
        args.threads = Some(1);
        args.speakers_db = Some(database_path.clone());

        correct_speaker(
            args,
            &CorrectionRequest {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                speaker: "Carol".into(),
            },
        )
        .expect_err("render failure prevents learning");

        assert_eq!(
            fs::read(database_path).expect("reread database"),
            previous_database
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn correcting_same_voice_twice_does_not_duplicate_learning() {
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "model".into(),
            sha256: "hash".into(),
            dimension: 2,
        });
        let updates = [CorrectionLearningUpdate {
            speaker: "Carol".into(),
            old_names: HashSet::new(),
            vectors: vec![vec![1.0, 0.0]],
        }];

        assert_eq!(apply_correction_learning(&mut database, &updates), (1, 0));
        let after_first = serde_json::to_vec(&database).expect("serialize database");
        assert_eq!(apply_correction_learning(&mut database, &updates), (0, 0));
        assert_eq!(
            serde_json::to_vec(&database).expect("serialize database again"),
            after_first
        );
    }

    #[test]
    fn correction_preflight_imports_legacy_vectors_without_touching_the_file() {
        let root = std::env::temp_dir().join(format!(
            "singstone-correction-legacy-preflight-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create fixture");
        let path = root.join("speakers.json");
        let legacy = serde_json::json!({
            "embedding_model": {
                "name": "old-model",
                "sha256": "old-hash",
                "dimension": 2
            },
            "speakers": {
                "Alice": { "embeddings": [[1.0, 0.0]] }
            }
        });
        fs::write(
            &path,
            serde_json::to_vec_pretty(&legacy).expect("serialize legacy database"),
        )
        .expect("write legacy database");
        let previous = fs::read(&path).expect("read legacy database");
        let expected = EmbeddingModelIdentity {
            name: "old-model".into(),
            sha256: "old-hash".into(),
            dimension: 2,
        };

        let (database, migrate) =
            load_correction_database(&path, &expected).expect("load correction database");

        assert!(migrate);
        assert_eq!(database.embedding_model, expected);
        assert_eq!(database.speakers["Alice"].embeddings, vec![vec![1.0, 0.0]]);
        assert_eq!(fs::read(&path).expect("reread legacy database"), previous);
        assert!(!root.join("speakers.json.v1.bak").exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn correction_upsert_replaces_only_strictly_overlapping_locks() {
        let lock = |source, start_ms, end_ms, speaker: &str| SpeakerCorrection {
            source,
            start_ms,
            end_ms,
            speaker: speaker.into(),
            inferred: false,
        };
        let mut artifact = SpeakerCorrections {
            format_version: SPEAKER_CORRECTIONS_FORMAT_VERSION,
            corrections: vec![
                lock(AudioSource::System, 0, 100, "Old"),
                lock(AudioSource::System, 100, 200, "Adjacent"),
                lock(AudioSource::System, 300, 300, "Point"),
                lock(AudioSource::Mic, 0, 100, "Other source"),
            ],
        };

        let replaced = upsert_correction(&mut artifact, lock(AudioSource::System, 50, 100, "New"));
        assert_eq!(
            replaced
                .iter()
                .map(|c| c.speaker.as_str())
                .collect::<Vec<_>>(),
            ["Old"]
        );
        let names = |artifact: &SpeakerCorrections| {
            artifact
                .corrections
                .iter()
                .map(|c| c.speaker.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(&artifact),
            ["Adjacent", "Point", "Other source", "New"]
        );

        let replaced = upsert_correction(
            &mut artifact,
            lock(AudioSource::System, 300, 300, "Point again"),
        );
        assert_eq!(replaced.len(), 1);
        assert_eq!(replaced[0].speaker, "Point");
        assert_eq!(artifact.corrections.len(), 4);
    }

    #[test]
    fn proposals_are_forward_unlocked_cluster_matches_with_margin() {
        let request = CorrectionRequest {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            speaker: "Alice".into(),
        };
        let chunks = [EmbeddingChunk {
            source: AudioSource::System,
            start_ms: 2_000,
            end_ms: 4_000,
            cluster: 7,
            embedding: vec![1.0, 0.0],
        }];
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "model".into(),
            sha256: "hash".into(),
            dimension: 2,
        });
        database.speakers.insert(
            "Alice".into(),
            SpeakerRecord {
                embeddings: vec![vec![1.0, 0.0]],
                centroid: None,
            },
        );
        database.speakers.insert(
            "Bob".into(),
            SpeakerRecord {
                embeddings: vec![vec![0.96, 0.28]],
                centroid: None,
            },
        );
        let candidate = Utterance {
            start_ms: 2_000,
            end_ms: 4_000,
            source: AudioSource::System,
            speaker_id: "spk_7".into(),
            speaker: "Bob".into(),
            text: "later".into(),
            locked: false,
            echo: None,
        };

        let proposal =
            proposal_for_utterance(&request, 7, candidate.clone(), &chunks, &database, 0.6)
                .expect("proposal");
        assert_eq!(proposal.current_speaker.as_deref(), Some("Bob"));
        assert_eq!(proposal.score_new, 1.0);
        assert!(proposal.score_old.is_some_and(|score| score > 0.95));
        assert!(!proposal.proposed);

        database.speakers.get_mut("Bob").unwrap().embeddings = vec![vec![0.8, 0.6]];
        assert!(
            proposal_for_utterance(&request, 7, candidate.clone(), &chunks, &database, 0.6)
                .expect("positive margin proposal")
                .proposed
        );
        let anonymous = Utterance {
            speaker: "SPEAKER_00".into(),
            ..candidate.clone()
        };
        let anonymous_proposal =
            proposal_for_utterance(&request, 7, anonymous, &chunks, &database, 0.99)
                .expect("anonymous threshold proposal");
        assert_eq!(anonymous_proposal.score_old, None);
        assert!(anonymous_proposal.proposed);

        assert!(
            proposal_for_utterance(
                &request,
                7,
                Utterance {
                    locked: true,
                    ..candidate.clone()
                },
                &chunks,
                &database,
                0.6,
            )
            .is_none()
        );
        assert!(
            proposal_for_utterance(
                &request,
                7,
                Utterance {
                    start_ms: 1_999,
                    ..candidate.clone()
                },
                &chunks,
                &database,
                0.6,
            )
            .is_none()
        );
        assert!(proposal_for_utterance(&request, 8, candidate, &chunks, &database, 0.6).is_none());
    }

    #[test]
    fn learned_lines_are_locked_lines_whose_voice_is_stored_under_their_name() {
        let chunks = [EmbeddingChunk {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 2_000,
            cluster: 7,
            embedding: vec![1.0, 0.0],
        }];
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "model".into(),
            sha256: "hash".into(),
            dimension: 2,
        });
        let record = |embedding| SpeakerRecord {
            embeddings: vec![embedding],
            centroid: None,
        };
        database
            .speakers
            .insert("Alice".into(), record(vec![1.0, 0.0]));
        database
            .speakers
            .insert("Bob".into(), record(vec![0.0, 1.0]));
        let line = Utterance {
            start_ms: 0,
            end_ms: 2_000,
            source: AudioSource::System,
            speaker_id: "spk_7".into(),
            speaker: "Alice".into(),
            text: "line".into(),
            locked: true,
            echo: None,
        };

        assert!(line_is_learned(&line, &chunks, &database));
        let unlocked = Utterance {
            locked: false,
            ..line.clone()
        };
        assert!(!line_is_learned(&unlocked, &chunks, &database));
        let other_voice = Utterance {
            speaker: "Bob".into(),
            ..line
        };
        assert!(!line_is_learned(&other_voice, &chunks, &database));
    }

    #[test]
    fn only_proposed_later_lines_become_inferred_corrections() {
        let proposal = |start_ms, proposed| Proposal {
            source: AudioSource::System,
            cluster: 7,
            start_ms,
            end_ms: start_ms + 1_000,
            text: "later".into(),
            current_speaker: None,
            score_new: 0.7,
            score_old: None,
            proposed,
        };
        assert_eq!(
            inferred_corrections(&[proposal(2_000, true), proposal(4_000, false)], "Alice"),
            [SpeakerCorrection {
                source: AudioSource::System,
                start_ms: 2_000,
                end_ms: 3_000,
                speaker: "Alice".into(),
                inferred: true,
            }]
        );
    }

    #[test]
    fn inferred_corrections_replace_each_other_but_never_a_lock() {
        let correction = |speaker: &str, inferred| SpeakerCorrection {
            source: AudioSource::System,
            start_ms: 0,
            end_ms: 100,
            speaker: speaker.into(),
            inferred,
        };
        let mut artifact = SpeakerCorrections {
            format_version: SPEAKER_CORRECTIONS_FORMAT_VERSION,
            corrections: vec![correction("Locked", false), correction("Guess", true)],
        };

        upsert_correction(&mut artifact, correction("New guess", true));
        assert_eq!(
            artifact.corrections,
            [correction("Locked", false), correction("New guess", true)]
        );

        upsert_correction(&mut artifact, correction("Chosen", false));
        assert_eq!(artifact.corrections, [correction("Chosen", false)]);
    }

    #[test]
    fn render_suppresses_leakage_without_changing_word_artifact() {
        let root = std::env::temp_dir().join(format!(
            "singstone-render-leakage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        let mut words = ["we", "should", "release", "Friday"]
            .iter()
            .enumerate()
            .map(|(index, text)| TimedWord {
                source: AudioSource::System,
                start_ms: 1_000 + index as u64 * 350,
                end_ms: 1_300 + index as u64 * 350,
                text: (*text).into(),
            })
            .collect::<Vec<_>>();
        words.push(TimedWord {
            source: AudioSource::Mic,
            start_ms: 770,
            end_ms: 970,
            text: "yes".into(),
        });
        words.extend(
            ["we", "should", "release", "Friday"]
                .iter()
                .enumerate()
                .map(|(index, text)| TimedWord {
                    source: AudioSource::Mic,
                    start_ms: 1_120 + index as u64 * 350,
                    end_ms: 1_420 + index as u64 * 350,
                    text: (*text).into(),
                }),
        );
        jsonl::write_all_atomic(&session.words_path(), &words).expect("write words");
        let original_words = fs::read(session.words_path()).expect("read original words");
        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                source: AudioSource::System,
                start_ms: 900,
                end_ms: 2_500,
                cluster: 0,
            }],
        )
        .expect("write diarization");

        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: Some(false),
        })
        .expect("render leakage");
        assert_eq!(
            fs::read(session.words_path()).expect("reread words"),
            original_words
        );
        let first_transcript = fs::read(session.transcript_path()).expect("read transcript");
        let transcript: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("parse transcript");
        assert_eq!(transcript.len(), 2);
        assert_eq!(transcript[0].text, "yes");
        assert_eq!(transcript[1].text, "we should release Friday");
        let suppressions: Vec<serde_json::Value> =
            jsonl::read_all(&session.leakage_suppressions_path()).expect("read suppressions");
        assert_eq!(suppressions.len(), 1);
        assert_eq!(suppressions[0]["suppressed_words"], 4);

        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: Some(false),
        })
        .expect("rerender leakage");
        assert_eq!(
            fs::read(session.transcript_path()).expect("reread transcript"),
            first_transcript
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn render_omits_hidden_mic_utterances_only_from_transcripts() {
        let (root, session) = hidden_sources_fixture("render-hidden-mic");
        let original_words = fs::read(session.words_path()).expect("read words");
        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: None,
        })
        .expect("render all sources");
        let all: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read full transcript");
        let expected_system = all
            .into_iter()
            .filter(|utterance| utterance.source == AudioSource::System)
            .collect::<Vec<_>>();
        write_json_atomic(
            &session.hidden_sources_path(),
            &HiddenSources {
                format_version: HIDDEN_SOURCES_FORMAT_VERSION,
                hidden: vec![AudioSource::Mic],
            },
        )
        .expect("write hidden sources");

        run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: None,
        })
        .expect("render hidden microphone");

        let visible: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).expect("read filtered transcript");
        assert_eq!(visible, expected_system);
        let text =
            fs::read_to_string(session.transcript_text_path()).expect("read transcript text");
        assert!(text.contains("system"));
        assert!(!text.contains("microphone"));
        assert_eq!(
            fs::read(session.words_path()).expect("reread words"),
            original_words
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn unsupported_hidden_sources_version_fails_render_with_path() {
        let (root, session) = hidden_sources_fixture("hidden-sources-version");
        let path = session.hidden_sources_path();
        fs::write(&path, br#"{"format_version":999,"hidden":[]}"#)
            .expect("write unsupported hidden sources");

        let error = run_render(RenderArgs {
            session: session.dir.clone(),
            diarize_mic: None,
        })
        .expect_err("unsupported version fails render");

        assert_eq!(
            error.to_string(),
            format!(
                "unsupported hidden sources format version 999 in {}",
                path.display()
            )
        );
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
                mic_num_speakers: None,
                system_num_speakers: None,
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
    fn empty_diarization_skips_recognition_before_model_and_database() {
        let root = std::env::temp_dir().join(format!(
            "singstone-empty-recognition-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        jsonl::write_all_atomic::<SpeakerSegment>(&session.diarization_path(), &[])
            .expect("write empty diarization");
        fs::write(session.embeddings_path(), b"prior embeddings\n")
            .expect("write prior embeddings");
        let model_path = root.join("unused-embedding.onnx");
        let database_path = root.join("unused-speakers.json");
        fs::write(&model_path, b"not a model").expect("write unused model");
        fs::write(&database_path, b"not a database").expect("write unused database");

        run_recognize(RecognizeArgs {
            session: session.dir.clone(),
            embedding_model: model_path,
            models_lock: None,
            allow_unverified_models: false,
            threads: Some(1),
            speakers_db: Some(database_path),
            speaker_threshold: 0.6,
        })
        .expect("empty diarization needs no recognition inputs");

        assert_eq!(
            fs::read(session.embeddings_path()).expect("read prior embeddings"),
            b"prior embeddings\n"
        );
        let assignments: SpeakerAssignments =
            read_json(&session.speaker_assignments_path()).expect("read empty assignments");
        assert!(assignments.assignments.is_empty());
        assert_eq!(assignments.provenance.embedding_model_sha256, None);
        assert_eq!(assignments.provenance.speakers_database_sha256, None);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn non_strict_embedding_audio_keeps_a_readable_source_and_skips_an_unreadable_one() {
        let root = std::env::temp_dir().join(format!(
            "singstone-unreadable-recognition-audio-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        let system_samples = [0.25f32, -0.25, 0.5, -0.5];
        fs::write(
            session.audio_path(AudioSource::System),
            system_samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .expect("write readable system audio");
        fs::create_dir(session.audio_path(AudioSource::Mic))
            .expect("put directory where audio file belongs");

        assert_eq!(
            read_embedding_audio(&session, AudioSource::System, false)
                .expect("readable non-strict source")
                .expect("system source remains available"),
            system_samples
        );
        assert!(
            read_embedding_audio(&session, AudioSource::Mic, false)
                .expect("non-strict read")
                .is_none()
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn strict_recognition_validates_database_before_writing_outputs() {
        let Some(models) = std::env::var_os("SINGSTONE_TEST_MODELS") else {
            return;
        };
        let root = std::env::temp_dir().join(format!(
            "singstone-recognition-preflight-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = empty_session(&root);
        jsonl::write_all_atomic(
            &session.diarization_path(),
            &[SpeakerSegment {
                source: AudioSource::System,
                start_ms: 0,
                end_ms: 2_000,
                cluster: 0,
            }],
        )
        .expect("write diarization");
        fs::write(
            session.audio_path(AudioSource::System),
            vec![0u8; SAMPLE_RATE as usize * 2 * std::mem::size_of::<f32>()],
        )
        .expect("write audio");
        let database_path = root.join("speakers.json");
        fs::write(&database_path, b"not JSON\n").expect("write invalid database");
        fs::write(session.embeddings_path(), b"prior embeddings\n")
            .expect("write prior embeddings");
        fs::write(session.embeddings_metadata_path(), b"prior metadata\n")
            .expect("write prior metadata");
        fs::write(session.speaker_assignments_path(), b"prior assignments\n")
            .expect("write prior assignments");

        run_recognize(RecognizeArgs {
            session: session.dir.clone(),
            embedding_model: PathBuf::from(models).join("nemo_en_titanet_small.onnx"),
            models_lock: None,
            allow_unverified_models: true,
            threads: Some(1),
            speakers_db: Some(database_path),
            speaker_threshold: 0.6,
        })
        .expect_err("invalid database fails preflight");

        assert_eq!(
            fs::read(session.embeddings_path()).expect("read embeddings"),
            b"prior embeddings\n"
        );
        assert_eq!(
            fs::read(session.embeddings_metadata_path()).expect("read metadata"),
            b"prior metadata\n"
        );
        assert_eq!(
            fs::read(session.speaker_assignments_path()).expect("read assignments"),
            b"prior assignments\n"
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
