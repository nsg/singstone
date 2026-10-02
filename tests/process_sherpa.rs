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
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;
use types::{AudioSource, SAMPLE_RATE, SpeakerSegment};

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

    let extractor = EmbeddingExtractor::new(&models.join("nemo_en_titanet_small.onnx"), 4)
        .expect("create extractor");
    let meeting =
        read_wav_pcm16(&samples.join("ES2002a.Mix-Headset.wav")).expect("read AMI meeting");
    let learned = learn_ground_truth_speakers(&samples, &meeting, &extractor)
        .expect("learn ground-truth speakers");
    assert!(learned.values().all(|dots| !dots.is_empty()));
    let cluster_embeddings = embed_cluster_chunks(&segments, &audio, &extractor);
    let truth = ground_truth_segments(&samples).expect("parse ground truth segments");
    let mut same_scores = Vec::new();
    let mut different_scores = Vec::new();
    for (cluster, cluster_embedding) in cluster_embeddings {
        let speaker = dominant_speaker(cluster, &segments, &truth).expect("dominant speaker");
        let scores = learned
            .iter()
            .filter_map(|(name, learned_dots)| {
                embedding::speaker_score(&cluster_embedding, learned_dots)
                    .map(|score| (*name, score))
            })
            .collect::<Vec<_>>();
        let same = scores
            .iter()
            .find(|(name, _)| *name == speaker)
            .map(|(_, score)| *score)
            .expect("same-speaker score");
        let different = scores
            .iter()
            .filter(|(name, _)| *name != speaker)
            .map(|(_, score)| *score)
            .max_by(f32::total_cmp)
            .expect("different-speaker score");
        same_scores.push(same);
        different_scores.push(different);
        eprintln!(
            "cluster {cluster}: ground truth {speaker}, correct {same:.6}, best wrong {different:.6}"
        );
    }
    eprintln!(
        "calibration ranges: same {}; best-different {}",
        score_range(&same_scores),
        score_range(&different_scores)
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

fn embed_cluster_chunks(
    segments: &[SpeakerSegment],
    audio: &[f32],
    extractor: &EmbeddingExtractor,
) -> BTreeMap<u32, Vec<Vec<f32>>> {
    let mut output = BTreeMap::<u32, Vec<Vec<f32>>>::new();
    for segment in segments {
        for (start_ms, end_ms) in fixed_windows(segment.start_ms, segment.end_ms) {
            if end_ms.saturating_sub(start_ms) < 1_500 {
                continue;
            }
            let start = to_sample(start_ms).min(audio.len());
            let end = to_sample(end_ms).min(audio.len());
            if let Some(value) = extractor.embed(&audio[start..end]) {
                output.entry(segment.cluster).or_default().push(value);
            }
        }
    }
    output
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

fn dominant_speaker(
    cluster: u32,
    segments: &[SpeakerSegment],
    truth: &BTreeMap<char, Vec<(f64, f64)>>,
) -> Option<char> {
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
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .filter(|(_, overlap)| *overlap > 0.0)
        .map(|(speaker, _)| speaker)
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

fn attribute<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("{name}=\"");
    let value = line.split_once(&marker)?.1;
    Some(value.split_once('"')?.0)
}

fn parse_number(value: &str) -> Option<f64> {
    value.parse().ok()
}
