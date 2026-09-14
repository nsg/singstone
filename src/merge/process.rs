use crate::cli::ProcessArgs;
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
    AudioSource, Manifest, SAMPLE_RATE, SessionState, SpeakerSegment, TimedWord, Utterance,
};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

pub fn run(args: ProcessArgs) -> Result<(), Box<dyn std::error::Error>> {
    let session = Session::open(&args.session)?;
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
    let threads = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
        .max(1);
    let mut words = Vec::new();
    if args.skip_transcription {
        words = jsonl::read_all(&session.words_path())?;
        eprintln!("transcribe: reusing {} words from words.jsonl", words.len());
    } else {
        transcribe_sources(&args, &session, &manifest, threads, &mut words)?;
    }

    let diarization_started = Instant::now();
    let mut segments = diarize_sources(&args, &session, &manifest, threads);
    segments.sort_by_key(|segment| {
        (
            segment.start_ms,
            source_order(segment.source),
            segment.cluster,
        )
    });
    jsonl::write_all_atomic(&session.diarization_path(), &segments)?;
    eprintln!(
        "diarize: {} segment(s) in {:.1} s",
        segments.len(),
        diarization_started.elapsed().as_secs_f64()
    );

    let recognition_started = Instant::now();
    let recognized = recognize_speakers(&args, &session, &segments, threads)?;
    eprintln!(
        "recognize speakers: {:.1} s",
        recognition_started.elapsed().as_secs_f64()
    );

    let merge_started = Instant::now();
    let utterances = utterances::build_utterances(
        &words,
        &segments,
        &recognized,
        &manifest.local_speaker,
        args.diarize_mic,
        DEFAULT_NEAREST_TOLERANCE_MS,
    );
    jsonl::write_all_atomic(&session.transcript_path(), &utterances)?;
    write_transcript_text(&session.transcript_text_path(), &utterances)?;
    eprintln!(
        "merge: {} utterance(s) in {:.1} s",
        utterances.len(),
        merge_started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn transcribe_sources(
    args: &ProcessArgs,
    session: &Session,
    manifest: &Manifest,
    threads: usize,
    words: &mut Vec<TimedWord>,
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
        let samples = read_enabled_audio(session, source, enabled)?;
        if samples.is_empty() {
            continue;
        }
        let started = Instant::now();
        transcriber.set_source(source);
        words.extend(transcriber.transcribe(&samples)?);
        sort_words(words);
        jsonl::write_all_atomic(&session.words_path(), words)?;
        eprintln!(
            "transcribe {source}: {:.1} s audio in {:.1} s",
            samples.len() as f64 / SAMPLE_RATE as f64,
            started.elapsed().as_secs_f64()
        );
    }
    sort_words(words);
    jsonl::write_all_atomic(&session.words_path(), words)?;
    Ok(())
}

fn read_enabled_audio(
    session: &Session,
    source: AudioSource,
    enabled: bool,
) -> io::Result<Vec<f32>> {
    if !enabled {
        return Ok(Vec::new());
    }
    match session.read_audio(source) {
        Ok(samples) => Ok(samples),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            eprintln!("warning: enabled {source} track is missing; skipping it");
            Ok(Vec::new())
        }
        Err(error) => Err(error),
    }
}

fn diarize_sources(
    args: &ProcessArgs,
    session: &Session,
    manifest: &crate::types::Manifest,
    threads: usize,
) -> Vec<SpeakerSegment> {
    if args.no_diarize {
        eprintln!("diarization disabled");
        return Vec::new();
    }
    let (Some(segmentation_model), Some(embedding_model)) =
        (&args.segmentation_model, &args.embedding_model)
    else {
        eprintln!(
            "warning: diarization models were not both configured; continuing without diarization"
        );
        return Vec::new();
    };
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
            eprintln!("warning: diarization unavailable: {error}");
            return Vec::new();
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
        let samples = match read_enabled_audio(session, source, true) {
            Ok(samples) if !samples.is_empty() => samples,
            Ok(_) => continue,
            Err(error) => {
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
            Err(error) => eprintln!("warning: {source} diarization failed: {error}"),
        }
    }
    all_segments
}

fn recognize_speakers(
    args: &ProcessArgs,
    session: &Session,
    segments: &[SpeakerSegment],
    threads: usize,
) -> io::Result<HashMap<(AudioSource, u32), String>> {
    if segments.is_empty() {
        eprintln!("speaker recognition: no diarized clusters; skipped");
        return Ok(HashMap::new());
    }
    let path = args
        .speakers_db
        .clone()
        .unwrap_or_else(database::default_path);
    if !path.is_file() {
        eprintln!("speaker recognition: no database; skipped");
        return Ok(HashMap::new());
    }
    let database = match SpeakerDatabase::load(&path) {
        Ok(database) => database,
        Err(error) => {
            eprintln!("warning: speaker recognition skipped: {error}");
            return Ok(HashMap::new());
        }
    };
    if database
        .speakers
        .values()
        .all(|speaker| speaker.embeddings.is_empty())
    {
        eprintln!("speaker recognition: database has no embeddings; skipped");
        return Ok(HashMap::new());
    }
    let Some(model) = args.embedding_model.as_deref() else {
        eprintln!("speaker recognition: no embedding model configured; skipped");
        return Ok(HashMap::new());
    };
    if let Err(error) = models::verify_model(
        model,
        "speaker-embedding",
        args.models_lock.as_deref(),
        args.allow_unverified_models,
    ) {
        eprintln!("warning: speaker recognition skipped: {error}");
        return Ok(HashMap::new());
    }
    let extractor = match EmbeddingExtractor::new(model, threads) {
        Ok(extractor) => extractor,
        Err(error) => {
            eprintln!("warning: speaker recognition unavailable: {error}");
            return Ok(HashMap::new());
        }
    };
    let identity = match database::identity(model, extractor.dimension()) {
        Ok(identity) => identity,
        Err(error) => {
            eprintln!("warning: speaker recognition skipped: {error}");
            return Ok(HashMap::new());
        }
    };
    if let Err(error) = database.validate_identity(&identity) {
        eprintln!("warning: speaker recognition skipped: {error}");
        return Ok(HashMap::new());
    }

    let mut matches = HashMap::new();
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
            matches.insert((source, cluster), name);
        }
    }
    Ok(matches)
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
        let segments = diarize_sources(&args, &session, &manifest, 1);
        assert!(!segments.is_empty());
    }

    #[test]
    fn skipped_stages_do_not_require_or_touch_models() {
        let root =
            std::env::temp_dir().join(format!("singstone-skipped-models-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
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
        fs::remove_dir_all(root).expect("remove fixture");
    }
}
