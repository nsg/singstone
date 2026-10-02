use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

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
    speaker_id: String,
    speaker: String,
    text: String,
    #[serde(default)]
    locked: bool,
}

#[derive(Debug, Deserialize)]
struct EmbeddingChunk {
    source: String,
    start_ms: u64,
    end_ms: u64,
    cluster: u32,
}

#[derive(Debug, Deserialize)]
struct SpeakerAssignments {
    assignments: Vec<SpeakerAssignment>,
}

#[derive(Debug, Deserialize)]
struct SpeakerAssignment {
    source: String,
    cluster: u32,
    best_candidate: Option<String>,
    speaker: Option<String>,
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
            && utterance.speaker_id.starts_with("mic_")
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
fn corrections_learn_and_recognize_ami_speaker() {
    let Some((models, samples)) = configured() else {
        eprintln!("skipped: SINGSTONE_TEST_MODELS and SINGSTONE_TEST_SAMPLES are required");
        return;
    };
    let base = process_fixture().expect("process fixture");
    let root = unique_dir("singstone-correct-ami").expect("create correction directory");
    let session = root.join("session");
    copy_dir(&base, &session).expect("copy processed session");
    let database = root.join("speakers.json");

    assert!(session.join("embeddings.jsonl").is_file());
    assert!(session.join("embeddings.meta.json").is_file());
    assert!(session.join("speaker-assignments.json").is_file());
    let chunks: Vec<EmbeddingChunk> =
        read_jsonl(&session.join("embeddings.jsonl")).expect("read embeddings");
    assert!(!chunks.is_empty());
    let utterances: Vec<Utterance> =
        read_jsonl(&session.join("transcript.jsonl")).expect("read transcript");
    let truth = ['A', 'B', 'C', 'D']
        .into_iter()
        .map(|speaker| {
            ground_truth_segments(&samples, &speaker.to_string(), 70.0)
                .map(|segments| (speaker, segments))
        })
        .collect::<io::Result<BTreeMap<_, _>>>()
        .expect("read ground truth");
    let (target, truth_speaker, truth_overlap) =
        correction_target(&utterances, &chunks, &truth).expect("eligible correction target");
    let cluster = target
        .speaker_id
        .strip_prefix("spk_")
        .expect("system cluster id")
        .parse::<u32>()
        .expect("numeric cluster");
    let temporary_name = "Temporary speaker";
    let learned_name = format!("AMI {truth_speaker}");

    let first = run_correct(&session, &models, &database, target, temporary_name)
        .expect("run first correction");
    assert_eq!(first["learning_available"], true);
    assert!(first["learned"].as_u64().is_some_and(|count| count > 0));
    let corrected: Vec<Utterance> =
        read_jsonl(&session.join("transcript.jsonl")).expect("read corrected transcript");
    assert!(corrected.iter().any(|utterance| {
        utterance.source == target.source
            && utterance.start_ms == target.start_ms
            && utterance.end_ms == target.end_ms
            && utterance.speaker == temporary_name
            && utterance.locked
    }));

    let second = run_correct(&session, &models, &database, target, &learned_name)
        .expect("run replacement correction");
    assert!(second["learned"].as_u64().is_some_and(|count| count > 0));
    assert!(second["forgotten"].as_u64().is_some_and(|count| count > 0));
    let database_json: serde_json::Value =
        serde_json::from_slice(&fs::read(&database).expect("read speaker database"))
            .expect("parse speaker database");
    assert_eq!(database_json["format_version"], 2);
    assert_eq!(
        database_json["speakers"][temporary_name]["embeddings"]
            .as_array()
            .map(Vec::len),
        Some(0)
    );
    assert!(
        database_json["speakers"][&learned_name]["embeddings"]
            .as_array()
            .is_some_and(|dots| !dots.is_empty())
    );

    run_recognize(&session, &models, &database).expect("recognize learned speaker");
    run_render(&session).expect("render recognized transcript");
    let assignments: SpeakerAssignments = serde_json::from_slice(
        &fs::read(session.join("speaker-assignments.json")).expect("read assignments"),
    )
    .expect("parse assignments");
    let assignment = assignments
        .assignments
        .iter()
        .find(|assignment| assignment.source == "system" && assignment.cluster == cluster)
        .expect("corrected cluster assignment");
    assert_eq!(
        assignment.best_candidate.as_deref(),
        Some(learned_name.as_str())
    );
    assert_eq!(assignment.speaker.as_deref(), Some(learned_name.as_str()));
    let rendered: Vec<Utterance> =
        read_jsonl(&session.join("transcript.jsonl")).expect("read rendered transcript");
    assert!(rendered.iter().any(|utterance| {
        utterance.source == target.source
            && utterance.start_ms == target.start_ms
            && utterance.end_ms == target.end_ms
            && utterance.speaker == learned_name
            && utterance.locked
    }));
    eprintln!(
        "corrected {}..{} as {learned_name} ({truth_overlap} ms ground-truth overlap)",
        target.start_ms, target.end_ms
    );
}

fn correction_target<'a>(
    utterances: &'a [Utterance],
    chunks: &[EmbeddingChunk],
    truth: &BTreeMap<char, Vec<(u64, u64)>>,
) -> Option<(&'a Utterance, char, u64)> {
    utterances
        .iter()
        .filter(|utterance| utterance.source == "system")
        .filter_map(|utterance| {
            let cluster = utterance
                .speaker_id
                .strip_prefix("spk_")?
                .parse::<u32>()
                .ok()?;
            let has_dot = chunks.iter().any(|chunk| {
                let duration = chunk.end_ms.saturating_sub(chunk.start_ms);
                let overlap = chunk
                    .end_ms
                    .min(utterance.end_ms)
                    .saturating_sub(chunk.start_ms.max(utterance.start_ms));
                chunk.source == "system"
                    && chunk.cluster == cluster
                    && duration >= 1_500
                    && overlap.saturating_mul(2) >= duration
            });
            has_dot.then_some((utterance, cluster))
        })
        .flat_map(|(utterance, _)| {
            truth.iter().map(move |(speaker, segments)| {
                let overlap = segments
                    .iter()
                    .map(|(start, end)| {
                        utterance
                            .end_ms
                            .min(*end)
                            .saturating_sub(utterance.start_ms.max(*start))
                    })
                    .sum::<u64>();
                (utterance, *speaker, overlap)
            })
        })
        .filter(|(_, _, overlap)| *overlap > 0)
        .max_by_key(|(_, _, overlap)| *overlap)
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

fn run_correct(
    session: &Path,
    models: &Path,
    database: &Path,
    target: &Utterance,
    name: &str,
) -> io::Result<serde_json::Value> {
    let output = Command::new(env!("CARGO_BIN_EXE_singstone"))
        .arg("correct")
        .arg(session)
        .args(["--source", &target.source])
        .args(["--start-ms", &target.start_ms.to_string()])
        .args(["--end-ms", &target.end_ms.to_string()])
        .args(["--name", name])
        .arg("--embedding-model")
        .arg(models.join("nemo_en_titanet_small.onnx"))
        .arg("--speakers-db")
        .arg(database)
        .arg("--allow-unverified-models")
        .args(["--threads", "4"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "correct exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let line = String::from_utf8(output.stdout)
        .map_err(io::Error::other)?
        .lines()
        .next()
        .ok_or_else(|| io::Error::other("correct produced no outcome"))?
        .to_owned();
    serde_json::from_str(&line).map_err(io::Error::other)
}

fn run_recognize(session: &Path, models: &Path, database: &Path) -> io::Result<()> {
    let status = Command::new(env!("CARGO_BIN_EXE_singstone"))
        .arg("recognize")
        .arg(session)
        .arg("--embedding-model")
        .arg(models.join("nemo_en_titanet_small.onnx"))
        .arg("--speakers-db")
        .arg(database)
        .arg("--speaker-threshold")
        .arg("0")
        .arg("--allow-unverified-models")
        .args(["--threads", "4"])
        .status()?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| io::Error::other(format!("recognize exited with {status}")))
}

fn run_render(session: &Path) -> io::Result<()> {
    let status = Command::new(env!("CARGO_BIN_EXE_singstone"))
        .arg("render")
        .arg(session)
        .status()?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| io::Error::other(format!("render exited with {status}")))
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
