use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

const RATE: usize = 16_000;

#[derive(Debug, Deserialize)]
struct Word {
    source: String,
    start_ms: u64,
    end_ms: u64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Segment {
    source: String,
    start_ms: u64,
    end_ms: u64,
    cluster: u32,
}

#[derive(Debug, Deserialize)]
struct Utterance {
    start_ms: u64,
    end_ms: u64,
    source: String,
    speaker: String,
    text: String,
}

fn configured() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("SINGSTONE_TEST_MODELS")?.into(),
        std::env::var_os("SINGSTONE_TEST_SAMPLES")?.into(),
    ))
}

fn process_fixture() -> Result<PathBuf, String> {
    static PROCESSED: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    PROCESSED
        .get_or_init(|| {
            let (models, samples) = configured().ok_or("model fixtures are not configured")?;
            let destination = unique_dir("singstone-process-ami").map_err(|e| e.to_string())?;
            copy_dir(&samples.join("session-ami-3min"), &destination).map_err(|e| e.to_string())?;
            run_process(&destination, &models, None).map_err(|e| e.to_string())?;
            Ok(destination)
        })
        .clone()
}

#[test]
fn processes_ami_with_timed_words_and_diarization() {
    if configured().is_none() {
        eprintln!("skipped: SINGSTONE_TEST_MODELS and SINGSTONE_TEST_SAMPLES are required");
        return;
    }
    let (_, samples) = configured().expect("checked above");
    let session = process_fixture().expect("process fixture");
    let words: Vec<Word> = read_jsonl(&session.join("words.jsonl")).expect("read words");
    assert!(!words.is_empty());
    assert!(words.windows(2).all(|pair| {
        (pair[0].start_ms, &pair[0].source, pair[0].end_ms)
            <= (pair[1].start_ms, &pair[1].source, pair[1].end_ms)
    }));
    assert!(words.iter().all(|word| word.start_ms <= word.end_ms));

    let utterances: Vec<Utterance> =
        read_jsonl(&session.join("transcript.jsonl")).expect("read transcript");
    let mic_text = utterances
        .iter()
        .filter(|utterance| utterance.source == "mic")
        .map(|utterance| utterance.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        mic_text.contains("ask not what your country"),
        "mic transcript: {mic_text}"
    );
    assert!(utterances.iter().any(|utterance| {
        utterance.source == "mic"
            && utterance.speaker == "Me"
            && (5_000..=7_000).contains(&utterance.start_ms)
    }));

    let segments: Vec<Segment> =
        read_jsonl(&session.join("diarization.jsonl")).expect("read diarization");
    let clusters = segments
        .iter()
        .filter(|segment| segment.source == "system")
        .map(|segment| segment.cluster)
        .collect::<HashSet<_>>();
    assert!(
        clusters.len() >= 2,
        "only {} system cluster(s)",
        clusters.len()
    );
    let coverage = segments
        .iter()
        .filter(|segment| segment.source == "system")
        .map(|segment| segment.end_ms.saturating_sub(segment.start_ms))
        .sum::<u64>();
    assert!(
        coverage >= 72_000,
        "system diarization coverage was {coverage} ms"
    );

    let recognized = words
        .iter()
        .filter(|word| word.source == "system")
        .flat_map(|word| tokens(&word.text))
        .collect::<HashSet<_>>();
    let truth = ground_truth_tokens(&samples.join("ami/words")).expect("parse ground truth");
    let shared = truth.intersection(&recognized).count();
    let overlap = shared as f64 / truth.len() as f64;
    eprintln!(
        "AMI distinct-token overlap: {shared}/{} = {:.1}%; {} system clusters; {:.1}% diarization coverage",
        truth.len(),
        overlap * 100.0,
        clusters.len(),
        coverage as f64 / 180_000.0 * 100.0
    );
    assert!(
        overlap >= 0.50,
        "distinct-token overlap was {:.1}%",
        overlap * 100.0
    );
}

#[test]
fn enrollment_names_david_from_disjoint_audio() {
    let Some((models, samples)) = configured() else {
        eprintln!("skipped: SINGSTONE_TEST_MODELS and SINGSTONE_TEST_SAMPLES are required");
        return;
    };
    let enrollment = unique_dir("singstone-enroll-ami").expect("create enrollment directory");
    let sample_path = enrollment.join("david.f32le");
    make_enrollment_sample(&samples, &sample_path).expect("make enrollment sample");
    let database = enrollment.join("speakers.json");
    let status = Command::new(env!("CARGO_BIN_EXE_singstone"))
        .args(["enroll", "David"])
        .arg(&sample_path)
        .arg("--embedding-model")
        .arg(models.join("nemo_en_titanet_small.onnx"))
        .arg("--speakers-db")
        .arg(&database)
        .arg("--allow-unverified-models")
        .status()
        .expect("run enroll");
    assert!(status.success());

    let session = enrollment.join("session");
    copy_dir(&samples.join("session-ami-3min"), &session).expect("copy session");
    run_process(&session, &models, Some(&database)).expect("process enrolled fixture");
    let utterances: Vec<Utterance> =
        read_jsonl(&session.join("transcript.jsonl")).expect("read transcript");
    let david = utterances
        .iter()
        .filter(|utterance| utterance.speaker == "David")
        .collect::<Vec<_>>();
    assert!(!david.is_empty(), "no utterance was recognized as David");
    let speaker_a = ground_truth_segments(&samples, "A", 70.0).expect("read speaker A segments");
    let david_ms: u64 = david
        .iter()
        .map(|utterance| utterance.end_ms - utterance.start_ms)
        .sum();
    let overlap_ms: u64 = david
        .iter()
        .map(|utterance| {
            speaker_a
                .iter()
                .map(|(start, end)| {
                    utterance
                        .end_ms
                        .min(*end)
                        .saturating_sub(utterance.start_ms.max(*start))
                })
                .sum::<u64>()
        })
        .sum();
    eprintln!(
        "David: {} utterances, {david_ms} ms, {overlap_ms} ms inside speaker A ground truth",
        david.len()
    );
    assert!(
        overlap_ms * 100 >= david_ms * 60,
        "utterances named David mostly fall outside speaker A's ground truth"
    );
}

/// Ground-truth (start_ms, end_ms) of one AMI speaker, shifted by `offset_s`.
fn ground_truth_segments(
    samples: &Path,
    speaker: &str,
    offset_s: f64,
) -> io::Result<Vec<(u64, u64)>> {
    let xml =
        fs::read_to_string(samples.join(format!("ami/segments/ES2002a.{speaker}.segments.xml")))?;
    Ok(xml
        .lines()
        .filter(|line| line.trim_start().starts_with("<segment "))
        .filter_map(|line| {
            let start = attribute(line, "transcriber_start")?.parse::<f64>().ok()?;
            let end = attribute(line, "transcriber_end")?.parse::<f64>().ok()?;
            let to_ms = |t: f64| ((t - offset_s).max(0.0) * 1000.0) as u64;
            (end > offset_s).then(|| (to_ms(start), to_ms(end)))
        })
        .collect())
}

fn run_process(session: &Path, models: &Path, database: Option<&Path>) -> io::Result<()> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_singstone"));
    command
        .arg("process")
        .arg(session)
        .arg("--whisper-model")
        .arg(models.join("ggml-base.en.bin"))
        .arg("--segmentation-model")
        .arg(models.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"))
        .arg("--embedding-model")
        .arg(models.join("nemo_en_titanet_small.onnx"))
        .arg("--allow-unverified-models")
        .args(["--language", "en", "--threads", "4"]);
    if let Some(database) = database {
        command.arg("--speakers-db").arg(database);
    } else {
        command
            .arg("--speakers-db")
            .arg(session.join("absent-speakers.json"));
    }
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("process exited with {status}")))
    }
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<Vec<T>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(io::Error::other))
        .collect()
}

fn ground_truth_tokens(words_dir: &Path) -> io::Result<HashSet<String>> {
    let mut result = HashSet::new();
    for speaker in ['A', 'B', 'C', 'D'] {
        let xml = fs::read_to_string(words_dir.join(format!("ES2002a.{speaker}.words.xml")))?;
        for line in xml
            .lines()
            .filter(|line| line.trim_start().starts_with("<w "))
        {
            let start = attribute(line, "starttime").and_then(|value| value.parse::<f64>().ok());
            let Some(start) = start else { continue };
            if !(70.0..250.0).contains(&start) {
                continue;
            }
            if let Some(text) = element_text(line) {
                result.extend(tokens(&text.replace("&#39;", "'")));
            }
        }
    }
    Ok(result)
}

fn make_enrollment_sample(samples: &Path, output: &Path) -> io::Result<()> {
    let wav = fs::read(samples.join("ES2002a.Mix-Headset.wav"))?;
    let pcm = wav_data(&wav)?;
    let xml = fs::read_to_string(samples.join("ami/segments/ES2002a.A.segments.xml"))?;
    let mut segments = xml
        .lines()
        .filter(|line| line.trim_start().starts_with("<segment "))
        .filter_map(|line| {
            let start = attribute(line, "transcriber_start")?.parse::<f64>().ok()?;
            let end = attribute(line, "transcriber_end")?.parse::<f64>().ok()?;
            (start >= 250.0 && end - start >= 2.0).then_some((start, end))
        })
        .collect::<Vec<_>>();
    segments.sort_by(|a, b| (b.1 - b.0).total_cmp(&(a.1 - a.0)));
    let mut selected = Vec::new();
    let mut duration = 0.0;
    for (start, end) in segments {
        if duration >= 25.0 {
            break;
        }
        let take = (end - start).min(25.0 - duration);
        let first = (start * RATE as f64) as usize;
        let last = ((start + take) * RATE as f64) as usize;
        selected.extend_from_slice(&pcm[first.min(pcm.len())..last.min(pcm.len())]);
        duration += take;
    }
    assert!(
        duration >= 20.0,
        "only {duration:.1} s available for enrollment"
    );
    let bytes = selected
        .into_iter()
        .flat_map(|sample| (sample as f32 / 32768.0).to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(output, bytes)
}

fn wav_data(bytes: &[u8]) -> io::Result<Vec<i16>> {
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(io::Error::other)?,
        ) as usize;
        let start = offset + 8;
        let end = start + size;
        if &bytes[offset..offset + 4] == b"data" {
            return Ok(bytes[start..end]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
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
    let rest = line.split_once(&marker)?.1;
    Some(rest.split_once('"')?.0)
}

fn element_text(line: &str) -> Option<String> {
    let start = line.find('>')? + 1;
    let end = line.rfind('<')?;
    (end >= start).then(|| line[start..end].to_string())
}

fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn unique_dir(prefix: &str) -> io::Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
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
