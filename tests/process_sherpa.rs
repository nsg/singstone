#![allow(dead_code)]

#[path = "../src/diarization/mod.rs"]
mod diarization;
#[path = "../src/speaker/embedding.rs"]
mod embedding;
#[path = "../src/types.rs"]
mod types;

use diarization::Diarizer;
use diarization::sherpa::SherpaDiarizer;
use embedding::EmbeddingExtractor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use types::{AudioSource, DEFAULT_SPEAKER_THRESHOLD, SAMPLE_RATE, SpeakerSegment};

#[derive(Debug, Deserialize)]
struct SpeakerAssignments {
    assignments: Vec<SpeakerAssignment>,
}

#[derive(Debug, Deserialize)]
struct SpeakerAssignment {
    source: String,
    cluster: u32,
    score: Option<f32>,
    speaker: Option<String>,
}

#[test]
fn isolated_sherpa_diarization_and_embedding_calibration() {
    let Some((models, samples)) = configured() else {
        eprintln!("skipped: SINGSTONE_TEST_MODELS and SINGSTONE_TEST_SAMPLES are required");
        return;
    };
    let audio = read_f32le(&samples.join("session-ami-3min/audio/system.f32le"))
        .expect("read system audio");
    let started = Instant::now();
    let segments = diarize(&models, &audio).expect("diarize fixture");
    let elapsed = started.elapsed();
    let clusters = segments
        .iter()
        .map(|segment| segment.cluster)
        .collect::<BTreeSet<_>>();
    let coverage = segments
        .iter()
        .map(|segment| segment.end_ms.saturating_sub(segment.start_ms))
        .sum::<u64>();
    eprintln!(
        "isolated sherpa: {} segments, {} clusters, {:.1}% summed coverage in {:.1} s",
        segments.len(),
        clusters.len(),
        coverage as f64 / 180_000.0 * 100.0,
        elapsed.as_secs_f64()
    );
    assert!(clusters.len() >= 2);
    assert!(coverage >= 72_000);

    let model = models.join("nemo_en_titanet_small.onnx");
    let extractor = EmbeddingExtractor::new(&model, 4).expect("create extractor");
    let meeting =
        read_wav_pcm16(&samples.join("ES2002a.Mix-Headset.wav")).expect("read AMI meeting");
    let learned = learn_ground_truth_speakers(&samples, &meeting, &extractor)
        .expect("learn ground-truth speakers");
    assert!(learned.values().all(|dots| !dots.is_empty()));
    let truth = ground_truth_segments(&samples).expect("parse ground truth segments");
    let root = unique_dir("singstone-sherpa-calibration").expect("create calibration directory");
    let session = root.join("session");
    copy_dir(&samples.join("session-ami-3min"), &session).expect("copy AMI session");
    write_jsonl(&session.join("diarization.jsonl"), &segments).expect("write diarization");

    let database = root.join("speakers.json");
    let mut scores_by_speaker = BTreeMap::new();
    for speaker in ['A', 'B', 'C', 'D'] {
        let singleton = BTreeMap::from([(speaker, learned[&speaker].clone())]);
        write_speaker_database(&database, &model, extractor.dimension(), &singleton)
            .expect("write singleton speaker database");
        run_recognize(&session, &model, &database).expect("score singleton database");
        scores_by_speaker.insert(
            speaker,
            read_system_assignments(&session)
                .expect("read singleton assignments")
                .into_iter()
                .filter_map(|(cluster, assignment)| assignment.score.map(|score| (cluster, score)))
                .collect::<BTreeMap<_, _>>(),
        );
    }

    write_speaker_database(&database, &model, extractor.dimension(), &learned)
        .expect("write complete speaker database");
    run_recognize(&session, &model, &database).expect("recognize complete database");
    let assignments = read_system_assignments(&session).expect("read complete assignments");
    let mut same_scores = Vec::new();
    let mut different_scores = Vec::new();
    let mut recognized_names = BTreeSet::new();
    for cluster in clusters {
        if !scores_by_speaker
            .values()
            .all(|scores| scores.contains_key(&cluster))
        {
            eprintln!("cluster {cluster}: no usable production embedding score");
            continue;
        }
        let overlaps = speaker_overlaps(cluster, &segments, &truth);
        let (&speaker, &dominant_overlap) = overlaps
            .iter()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .filter(|(_, overlap)| **overlap > 0.0)
            .expect("dominant speaker");
        let same = scores_by_speaker[&speaker][&cluster];
        let (wrong_speaker, different) = scores_by_speaker
            .iter()
            .filter(|(name, _)| **name != speaker)
            .map(|(name, scores)| (*name, scores[&cluster]))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("different-speaker score");
        let assignment = &assignments[&cluster];
        let expected = format!("AMI {speaker}");
        same_scores.push(same);
        different_scores.push(different);
        eprintln!(
            "cluster {cluster}: ground truth {speaker} ({dominant_overlap:.3}s), correct {same:.6}, best wrong {wrong_speaker} {different:.6}, assigned {:?}, overlaps {:?}",
            assignment.speaker, overlaps
        );
        if different >= DEFAULT_SPEAKER_THRESHOLD {
            eprintln!(
                "cluster {cluster}: wrong speaker {wrong_speaker} is above the default threshold ({different:.6} >= {DEFAULT_SPEAKER_THRESHOLD:.3})"
            );
        }
        if same >= DEFAULT_SPEAKER_THRESHOLD {
            assert_eq!(
                assignment.speaker.as_deref(),
                Some(expected.as_str()),
                "cluster {cluster} has a qualifying correct score but was not named correctly"
            );
        }
        if let Some(name) = assignment.speaker.as_deref() {
            assert_eq!(name, expected, "cluster {cluster} was named incorrectly");
            recognized_names.insert(name.to_owned());
        }
    }
    assert!(
        recognized_names.len() >= 2,
        "threshold must still recognize multiple speakers"
    );
    eprintln!(
        "calibration ranges: correct {}; best-wrong {}; default threshold {:.3}",
        score_range(&same_scores),
        score_range(&different_scores),
        DEFAULT_SPEAKER_THRESHOLD
    );
}

#[test]
fn isolated_full_diarization_when_requested() {
    if std::env::var_os("SINGSTONE_TEST_FULL").is_none() {
        return;
    }
    let (models, samples) = configured().expect("fixture variables required with full test");
    let audio =
        read_f32le(&samples.join("session-ami-full/audio/system.f32le")).expect("read full audio");
    let started = Instant::now();
    let segments = diarize(&models, &audio).expect("diarize full fixture");
    let clusters = segments
        .iter()
        .map(|segment| segment.cluster)
        .collect::<BTreeSet<_>>();
    eprintln!(
        "isolated full sherpa: {:.1} s audio, {} segments, {} speakers in {:.1} s",
        audio.len() as f64 / SAMPLE_RATE as f64,
        segments.len(),
        clusters.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(!segments.is_empty());
}

fn configured() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("SINGSTONE_TEST_MODELS")?.into(),
        std::env::var_os("SINGSTONE_TEST_SAMPLES")?.into(),
    ))
}

fn diarize(
    models: &Path,
    audio: &[f32],
) -> Result<Vec<SpeakerSegment>, Box<dyn std::error::Error>> {
    let mut diarizer = SherpaDiarizer::new(
        &models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
        &models.join("nemo_en_titanet_small.onnx"),
        AudioSource::System,
        4,
        0.5,
        None,
    )?;
    diarizer.diarize(audio)
}

fn learn_ground_truth_speakers(
    samples: &Path,
    meeting: &[f32],
    extractor: &EmbeddingExtractor,
) -> io::Result<BTreeMap<char, Vec<Vec<f32>>>> {
    let mut output = BTreeMap::new();
    for speaker in ['A', 'B', 'C', 'D'] {
        let xml = fs::read_to_string(
            samples.join(format!("ami/segments/ES2002a.{speaker}.segments.xml")),
        )?;
        let mut dots = Vec::new();
        for line in xml
            .lines()
            .filter(|line| line.trim_start().starts_with("<segment "))
        {
            let Some(start) = attribute(line, "transcriber_start").and_then(parse_number) else {
                continue;
            };
            let Some(end) = attribute(line, "transcriber_end").and_then(parse_number) else {
                continue;
            };
            if start < 250.0 {
                continue;
            }
            let start_ms = (start * 1_000.0).round() as u64;
            let end_ms = (end * 1_000.0).round() as u64;
            for (window_start, window_end) in fixed_windows(start_ms, end_ms) {
                if window_end.saturating_sub(window_start) < 1_500 {
                    continue;
                }
                let first = to_sample(window_start).min(meeting.len());
                let last = to_sample(window_end).min(meeting.len());
                if let Some(value) = extractor.embed(&meeting[first..last]) {
                    dots.push(value);
                }
            }
        }
        if dots.is_empty() {
            return Err(io::Error::other(format!(
                "could not embed AMI speaker {speaker}"
            )));
        }
        eprintln!("learned AMI speaker {speaker} from {} dots", dots.len());
        output.insert(speaker, dots);
    }
    Ok(output)
}

fn fixed_windows(start_ms: u64, end_ms: u64) -> Vec<(u64, u64)> {
    let duration = end_ms.saturating_sub(start_ms);
    if duration == 0 {
        return Vec::new();
    }
    let count = duration.div_ceil(10_000);
    (0..count)
        .map(|index| {
            let offset =
                |part: u64| (u128::from(duration) * u128::from(part) / u128::from(count)) as u64;
            (
                start_ms.saturating_add(offset(index)),
                start_ms.saturating_add(offset(index + 1)),
            )
        })
        .collect()
}

fn ground_truth_segments(samples: &Path) -> io::Result<BTreeMap<char, Vec<(f64, f64)>>> {
    let mut output = BTreeMap::new();
    for speaker in ['A', 'B', 'C', 'D'] {
        let xml = fs::read_to_string(
            samples.join(format!("ami/segments/ES2002a.{speaker}.segments.xml")),
        )?;
        let segments = xml
            .lines()
            .filter(|line| line.trim_start().starts_with("<segment "))
            .filter_map(|line| {
                let start = attribute(line, "transcriber_start").and_then(parse_number)?;
                let end = attribute(line, "transcriber_end").and_then(parse_number)?;
                Some((start - 70.0, end - 70.0))
            })
            .collect();
        output.insert(speaker, segments);
    }
    Ok(output)
}

fn speaker_overlaps(
    cluster: u32,
    segments: &[SpeakerSegment],
    truth: &BTreeMap<char, Vec<(f64, f64)>>,
) -> BTreeMap<char, f64> {
    truth
        .iter()
        .map(|(speaker, truth_segments)| {
            let overlap = segments
                .iter()
                .filter(|segment| segment.cluster == cluster)
                .flat_map(|segment| {
                    let start = segment.start_ms as f64 / 1_000.0;
                    let end = segment.end_ms as f64 / 1_000.0;
                    truth_segments.iter().map(move |(truth_start, truth_end)| {
                        (end.min(*truth_end) - start.max(*truth_start)).max(0.0)
                    })
                })
                .sum::<f64>();
            (*speaker, overlap)
        })
        .collect()
}

fn score_range(scores: &[f32]) -> String {
    let minimum = scores.iter().copied().min_by(f32::total_cmp).unwrap_or(0.0);
    let maximum = scores.iter().copied().max_by(f32::total_cmp).unwrap_or(0.0);
    format!("{minimum:.3}..{maximum:.3}")
}

fn to_sample(milliseconds: u64) -> usize {
    milliseconds.saturating_mul(SAMPLE_RATE as u64) as usize / 1_000
}

fn read_f32le(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "raw audio length is not a multiple of four",
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn read_wav_pcm16(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid WAV chunk"))?,
        ) as usize;
        let start = offset + 8;
        let end = start
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated WAV chunk"))?;
        if &bytes[offset..offset + 4] == b"data" {
            return Ok(bytes[start..end]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|sample| i16::from_le_bytes(*sample) as f32 / 32_768.0)
                .collect());
        }
        offset = end + (size & 1);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "WAV has no data chunk",
    ))
}

fn write_speaker_database(
    path: &Path,
    model: &Path,
    dimension: usize,
    learned: &BTreeMap<char, Vec<Vec<f32>>>,
) -> io::Result<()> {
    let speakers = learned
        .iter()
        .map(|(speaker, embeddings)| {
            (
                format!("AMI {speaker}"),
                serde_json::json!({ "embeddings": embeddings }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let value = serde_json::json!({
        "format_version": 2,
        "embedding_model": {
            "name": model.file_stem().and_then(|name| name.to_str()).unwrap_or("model"),
            "sha256": sha256_file(model)?,
            "dimension": dimension,
        },
        "speakers": speakers,
    });
    fs::write(
        path,
        serde_json::to_vec_pretty(&value).map_err(io::Error::other)?,
    )
}

fn run_recognize(session: &Path, model: &Path, database: &Path) -> io::Result<()> {
    let status = Command::new(env!("CARGO_BIN_EXE_singstone"))
        .arg("recognize")
        .arg(session)
        .arg("--embedding-model")
        .arg(model)
        .arg("--speakers-db")
        .arg(database)
        .arg("--allow-unverified-models")
        .args(["--threads", "4"])
        .status()?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| io::Error::other(format!("recognize exited with {status}")))
}

fn read_system_assignments(session: &Path) -> io::Result<BTreeMap<u32, SpeakerAssignment>> {
    let artifact: SpeakerAssignments =
        serde_json::from_slice(&fs::read(session.join("speaker-assignments.json"))?)
            .map_err(io::Error::other)?;
    Ok(artifact
        .assignments
        .into_iter()
        .filter(|assignment| assignment.source == "system")
        .map(|assignment| (assignment.cluster, assignment))
        .collect())
}

fn write_jsonl<T: serde::Serialize>(path: &Path, values: &[T]) -> io::Result<()> {
    let mut output = String::new();
    for value in values {
        output.push_str(&serde_json::to_string(value).map_err(io::Error::other)?);
        output.push('\n');
    }
    fs::write(path, output)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn unique_dir(prefix: &str) -> io::Result<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos()
    ));
    fs::create_dir(&path)?;
    Ok(path)
}

fn copy_dir(source: &Path, destination: &Path) -> io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn attribute<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("{name}=\"");
    let value = line.split_once(&marker)?.1;
    Some(value.split_once('"')?.0)
}

fn parse_number(value: &str) -> Option<f64> {
    value.parse().ok()
}
