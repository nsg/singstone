//! Diarization test cases: the audio of one transcript line with who speaks
//! when labeled by hand, saved as a folder that the diarization scorer reads
//! (`clip.wav`, `truth.json`, `words.jsonl`). Nothing here changes the
//! session; it only keeps a small index of its saved cases.

use crate::audio::track::TrackReader;
use crate::format::jsonl;
use crate::merge::process;
use crate::session::{self, Session};
use crate::types::{AudioSource, SAMPLE_RATE, SpeakerSegment, TimedWord, Utterance};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 1;
pub const DEFAULT_CONTEXT_MS: u64 = 1_000;
pub const MAX_CONTEXT_MS: u64 = 5_000;
pub const CONTEXT_STEP_MS: u64 = 1_000;
pub const DEFAULT_SPEAKERS: u8 = 2;
pub const MAX_SPEAKERS: u8 = 9;
/// The lane of stretches nobody is sure about; speakers are lanes 1 to 9.
pub const NOT_SURE: u8 = 0;
/// A bar shorter than this is a slip of the mouse or key, not speech.
pub const MIN_BAR_MS: u64 = 50;
const SAMPLES_PER_MS: u64 = SAMPLE_RATE as u64 / 1000;

/// The audio a case is labeled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Track {
    System,
    Mic,
    Both,
}

impl Track {
    pub fn of(source: AudioSource) -> Self {
        match source {
            AudioSource::Mic => Self::Mic,
            AudioSource::System => Self::System,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Mic => "mic",
            Self::Both => "both",
        }
    }

    pub fn includes(self, source: AudioSource) -> bool {
        self == Self::Both || self == Self::of(source)
    }
}

/// A transcript line by its track and span in the recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LineRef {
    pub source: AudioSource,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// One labeled stretch in recording time: `lane` 1 to 9 is a speaker, 0 is
/// "not sure". Saved as `[lane, start_ms, end_ms]`; `id` only tells bars apart
/// while editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "(u8, u64, u64)", into = "(u8, u64, u64)")]
pub struct Bar {
    pub lane: u8,
    pub start_ms: u64,
    pub end_ms: u64,
    pub id: u64,
}

impl From<(u8, u64, u64)> for Bar {
    fn from((lane, start_ms, end_ms): (u8, u64, u64)) -> Self {
        Self {
            lane,
            start_ms,
            end_ms,
            id: 0,
        }
    }
}

impl From<Bar> for (u8, u64, u64) {
    fn from(bar: Bar) -> Self {
        (bar.lane, bar.start_ms, bar.end_ms)
    }
}

/// What the labeling view shows: everything needed to open a case again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    pub track: Track,
    pub context_ms: u64,
    /// Speaker lanes shown, 1 to 9.
    pub lanes: u8,
    pub bars: Vec<Bar>,
    /// Saved as the case's `notes`.
    #[serde(skip)]
    pub note: String,
}

impl Labels {
    pub fn new(source: AudioSource) -> Self {
        Self {
            track: Track::of(source),
            context_ms: DEFAULT_CONTEXT_MS,
            lanes: DEFAULT_SPEAKERS,
            bars: Vec::new(),
            note: String::new(),
        }
    }
}

/// The clip of `line` with `context_ms` on each side, within the recording.
pub fn clip_range(line: LineRef, context_ms: u64, recorded_ms: u64) -> (u64, u64) {
    let end = line.end_ms.saturating_add(context_ms).min(recorded_ms);
    let start = line.start_ms.saturating_sub(context_ms).min(end);
    (start, end)
}

/// Both tracks of a stretch of the recording, read once so that the clip can
/// change track and context without reading again.
#[derive(Debug, Clone, Default)]
pub struct Audio {
    pub start_ms: u64,
    mic: Vec<f32>,
    system: Vec<f32>,
}

impl Audio {
    /// `[start_ms, end_ms)` of both tracks, silent where a track has no audio.
    pub fn read(session: &Session, start_ms: u64, end_ms: u64) -> io::Result<Self> {
        Ok(Self {
            start_ms,
            mic: read_span(session, AudioSource::Mic, start_ms, end_ms)?,
            system: read_span(session, AudioSource::System, start_ms, end_ms)?,
        })
    }

    /// The samples of `track` in `[start_ms, end_ms)`; both tracks are summed
    /// and every sample is kept within [-1, 1].
    pub fn clip(&self, track: Track, (start_ms, end_ms): (u64, u64)) -> Vec<f32> {
        let count = (end_ms.saturating_sub(start_ms) * SAMPLES_PER_MS) as usize;
        let offset = (start_ms.saturating_sub(self.start_ms) * SAMPLES_PER_MS) as usize;
        let sample = |samples: &[f32], index: usize| samples.get(offset + index).copied();
        (0..count)
            .map(|index| {
                let mic = sample(&self.mic, index).filter(|_| track.includes(AudioSource::Mic));
                let system =
                    sample(&self.system, index).filter(|_| track.includes(AudioSource::System));
                (mic.unwrap_or(0.0) + system.unwrap_or(0.0)).clamp(-1.0, 1.0)
            })
            .collect()
    }
}

fn read_span(
    session: &Session,
    source: AudioSource,
    start_ms: u64,
    end_ms: u64,
) -> io::Result<Vec<f32>> {
    let count = end_ms.saturating_sub(start_ms) * SAMPLES_PER_MS;
    let mut samples = Vec::with_capacity(count as usize);
    if let Some(mut reader) = TrackReader::open(session, source, start_ms * SAMPLES_PER_MS)? {
        let mut chunk = Vec::new();
        while (samples.len() as u64) < count {
            let wanted = (count - samples.len() as u64).min(1 << 14) as usize;
            reader.read(&mut chunk, wanted)?;
            if chunk.is_empty() {
                break;
            }
            samples.extend(
                chunk
                    .iter()
                    .map(|sample| if sample.is_finite() { *sample } else { 0.0 }),
            );
        }
    }
    samples.resize(count as usize, 0.0);
    Ok(samples)
}

/// Writes `samples` as a 16 kHz mono 16-bit PCM WAV file, quantized the way
/// archiving does.
pub fn write_wav(out: &mut impl Write, samples: &[f32]) -> io::Result<()> {
    let too_long = || io::Error::new(io::ErrorKind::InvalidInput, "clip too long for WAV");
    let data_len = u32::try_from(samples.len() * 2).map_err(|_| too_long())?;
    let riff_len = data_len.checked_add(36).ok_or_else(too_long)?;
    out.write_all(b"RIFF")?;
    out.write_all(&riff_len.to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?;
    out.write_all(&SAMPLE_RATE.to_le_bytes())?;
    out.write_all(&(SAMPLE_RATE * 2).to_le_bytes())?;
    out.write_all(&2u16.to_le_bytes())?;
    out.write_all(&16u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        // The cast saturates and maps NaN to silence.
        out.write_all(&((sample * 32_768.0).round() as i16).to_le_bytes())?;
    }
    Ok(())
}

/// Joins bar `id` with every bar of its lane that it touches or overlaps.
pub fn merge_lane(bars: &mut Vec<Bar>, id: u64) {
    let Some(mut joined) = bars.iter().find(|bar| bar.id == id).copied() else {
        return;
    };
    while let Some(index) = bars.iter().position(|bar| {
        bar.id != id
            && bar.lane == joined.lane
            && bar.start_ms <= joined.end_ms
            && bar.end_ms >= joined.start_ms
    }) {
        let other = bars.remove(index);
        joined.start_ms = joined.start_ms.min(other.start_ms);
        joined.end_ms = joined.end_ms.max(other.end_ms);
    }
    if let Some(bar) = bars.iter_mut().find(|bar| bar.id == id) {
        *bar = joined;
    }
}

/// The bars as a case saves them: cut to the clip, those of one lane that
/// touch or overlap joined, by start.
pub fn clip_bars(bars: &[Bar], (start_ms, end_ms): (u64, u64)) -> Vec<Bar> {
    let mut clipped = bars
        .iter()
        .map(|bar| Bar {
            start_ms: bar.start_ms.max(start_ms),
            end_ms: bar.end_ms.min(end_ms),
            ..*bar
        })
        .filter(|bar| bar.end_ms > bar.start_ms)
        .collect::<Vec<_>>();
    clipped.sort_by_key(|bar| (bar.lane, bar.start_ms));
    let mut joined: Vec<Bar> = Vec::new();
    for bar in clipped {
        match joined.last_mut() {
            Some(last) if last.lane == bar.lane && bar.start_ms <= last.end_ms => {
                last.end_ms = last.end_ms.max(bar.end_ms);
            }
            _ => joined.push(bar),
        }
    }
    joined.sort_by_key(|bar| (bar.start_ms, bar.end_ms, bar.lane));
    joined
}

/// Where two or more speakers talk at once: start, end, and the two lowest
/// speaker lanes then.
pub fn overlaps(bars: &[Bar]) -> Vec<(u64, u64, u8, u8)> {
    let speaking = bars
        .iter()
        .filter(|bar| bar.lane != NOT_SURE && bar.end_ms > bar.start_ms)
        .collect::<Vec<_>>();
    let mut edges = speaking
        .iter()
        .flat_map(|bar| [bar.start_ms, bar.end_ms])
        .collect::<Vec<_>>();
    edges.sort_unstable();
    edges.dedup();
    let mut found: Vec<(u64, u64, u8, u8)> = Vec::new();
    for pair in edges.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let mut lanes = speaking
            .iter()
            .filter(|bar| bar.start_ms <= from && bar.end_ms >= to)
            .map(|bar| bar.lane)
            .collect::<Vec<_>>();
        lanes.sort_unstable();
        lanes.dedup();
        if lanes.len() < 2 {
            continue;
        }
        match found.last_mut() {
            Some(last) if last.1 == from && (last.2, last.3) == (lanes[0], lanes[1]) => {
                last.1 = to;
            }
            _ => found.push((from, to, lanes[0], lanes[1])),
        }
    }
    found
}

/// `truth.json`, in the shape of the hand-labeled clips the scorer reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    pub format_version: u32,
    pub id: String,
    pub source: TruthSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub speakers: BTreeMap<String, String>,
    pub segments: Vec<(f64, f64, String)>,
    pub exclude: Vec<(f64, f64)>,
    pub app: AppSaid,
    pub notes: Vec<String>,
    pub labeled_at: String,
    pub editor: Editor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthSource {
    pub session: String,
    /// The recording's start (`manifest.json`), which with `session` tells
    /// recordings in folders of the same name apart.
    #[serde(default)]
    pub session_started: String,
    pub track: Track,
    pub offset_s: f64,
    pub duration_s: f64,
    pub line: TruthLine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthLine {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: String,
}

/// What the app's diarization said about the clip when it was labeled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSaid {
    pub version: String,
    pub segments: Vec<(f64, f64, String)>,
}

/// The labels in recording time, to open the case again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Editor {
    pub line: LineRef,
    #[serde(flatten)]
    pub labels: Labels,
}

/// One line of a case's `words.jsonl`, on the clip's time axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipWord {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<AudioSource>,
}

/// The session's words, diarization and transcript near a line.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub words: Vec<TimedWord>,
    pub diarization: Vec<SpeakerSegment>,
    pub transcript: Vec<Utterance>,
    pub language: Option<String>,
}

impl Inputs {
    /// What the session has about `[start_ms, end_ms]`; a missing file reads
    /// as nothing.
    pub fn read(session: &Session, (start_ms, end_ms): (u64, u64)) -> Self {
        let near = |from: u64, to: u64| to >= start_ms && from <= end_ms;
        let mut transcript: Vec<Utterance> =
            jsonl::read_all(&session.transcript_path()).unwrap_or_default();
        transcript.extend(process::read_outside_meeting(session).unwrap_or_default());
        transcript.extend(process::read_hidden_lines(session).unwrap_or_default());
        transcript.retain(|line| near(line.start_ms, line.end_ms));
        let language = fs::read(session.words_metadata_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|meta| meta.get("language")?.as_str().map(str::to_owned))
            .filter(|language| !language.is_empty() && language != "auto");
        Self {
            words: jsonl::read_all::<TimedWord>(&session.words_path())
                .unwrap_or_default()
                .into_iter()
                .filter(|word| near(word.start_ms, word.end_ms))
                .collect(),
            diarization: jsonl::read_all::<SpeakerSegment>(&session.diarization_path())
                .unwrap_or_default()
                .into_iter()
                .filter(|segment| near(segment.start_ms, segment.end_ms))
                .collect(),
            transcript,
            language,
        }
    }
}

/// Everything a save needs.
pub struct Draft<'a> {
    pub session: &'a Session,
    pub line: LineRef,
    pub line_speaker: &'a str,
    pub labels: &'a Labels,
    pub recorded_ms: u64,
    pub audio: &'a Audio,
    pub inputs: &'a Inputs,
    pub labeled_at: &'a str,
}

impl Draft<'_> {
    fn clip(&self) -> (u64, u64) {
        clip_range(self.line, self.labels.context_ms, self.recorded_ms)
    }

    fn truth(&self, id: &str, owner: &Owner) -> Truth {
        let clip = self.clip();
        let at = |ms: u64| seconds(ms - clip.0);
        let bars = clip_bars(&self.labels.bars, clip);
        let segments = bars
            .iter()
            .filter(|bar| bar.lane != NOT_SURE)
            .map(|bar| (at(bar.start_ms), at(bar.end_ms), format!("P{}", bar.lane)))
            .filter(|(start, end, _)| end > start)
            .collect::<Vec<_>>();
        let exclude = bars
            .iter()
            .filter(|bar| bar.lane == NOT_SURE)
            .map(|bar| (at(bar.start_ms), at(bar.end_ms)))
            .filter(|(start, end)| end > start)
            .collect();
        let mut editor = Labels {
            note: String::new(),
            ..self.labels.clone()
        };
        editor
            .bars
            .sort_by_key(|bar| (bar.start_ms, bar.end_ms, bar.lane));
        let speakers = segments
            .iter()
            .map(|(_, _, label)| (label.clone(), format!("Speaker {}", &label[1..])))
            .collect();
        Truth {
            format_version: FORMAT_VERSION,
            id: id.to_owned(),
            source: TruthSource {
                session: owner.session.clone(),
                session_started: owner.started.clone(),
                track: self.labels.track,
                offset_s: seconds(clip.0),
                duration_s: seconds(clip.1 - clip.0),
                line: TruthLine {
                    start_s: seconds(self.line.start_ms),
                    end_s: seconds(self.line.end_ms),
                    speaker: self.line_speaker.to_owned(),
                },
            },
            language: self.inputs.language.clone(),
            speakers,
            segments,
            exclude,
            app: AppSaid {
                version: env!("CARGO_PKG_VERSION").to_owned(),
                segments: app_segments(self.inputs, self.labels.track, clip),
            },
            notes: self
                .labels
                .note
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect(),
            labeled_at: self.labeled_at.to_owned(),
            editor: Editor {
                line: self.line,
                labels: editor,
            },
        }
    }
}

/// Seconds, to the nearest 10 ms.
fn seconds(ms: u64) -> f64 {
    to_10ms(ms) as f64 / 1000.0
}

fn to_10ms(ms: u64) -> u64 {
    ms.saturating_add(5) / 10 * 10
}

fn session_name(session: &Session) -> String {
    session
        .dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "session".into())
}

/// The words of `track` whose middle is in the clip, re-timed to it. A word
/// that rounds to no time at all, such as trailing punctuation, lasts 10 ms.
pub fn clip_words(
    words: &[TimedWord],
    track: Track,
    (start_ms, end_ms): (u64, u64),
) -> Vec<ClipWord> {
    let mut inside = words
        .iter()
        .filter(|word| {
            track.includes(word.source)
                && (start_ms..end_ms).contains(&((word.start_ms + word.end_ms) / 2))
        })
        .collect::<Vec<_>>();
    inside.sort_by_key(|word| (word.start_ms, word.end_ms));
    let length = to_10ms(end_ms - start_ms);
    inside
        .into_iter()
        .filter_map(|word| {
            let start = to_10ms(word.start_ms.max(start_ms) - start_ms);
            let end = to_10ms(word.end_ms.min(end_ms) - start_ms)
                .max(start + 10)
                .min(length);
            (end > start).then(|| ClipWord {
                start: start as f64 / 1000.0,
                end: end as f64 / 1000.0,
                text: word.text.trim().to_owned(),
                source: (track == Track::Both).then_some(word.source),
            })
        })
        .collect()
}

/// The diarization of `track` cut to the clip and re-timed to it. Each
/// segment is named like the transcript line of its track it overlaps most,
/// or by its cluster (`spk_N`, `mic_N`) when it overlaps none.
fn app_segments(
    inputs: &Inputs,
    track: Track,
    (start_ms, end_ms): (u64, u64),
) -> Vec<(f64, f64, String)> {
    let mut segments = inputs
        .diarization
        .iter()
        .filter(|segment| {
            track.includes(segment.source) && segment.end_ms > start_ms && segment.start_ms < end_ms
        })
        .map(|segment| {
            let (from, to) = (segment.start_ms.max(start_ms), segment.end_ms.min(end_ms));
            let overlap = |line: &Utterance| {
                line.end_ms
                    .min(segment.end_ms)
                    .saturating_sub(line.start_ms.max(segment.start_ms))
            };
            let name = inputs
                .transcript
                .iter()
                .filter(|line| line.source == segment.source && overlap(line) > 0)
                .max_by_key(|line| overlap(line))
                .map(|line| line.speaker.clone())
                .unwrap_or_else(|| {
                    let prefix = match segment.source {
                        AudioSource::Mic => "mic",
                        AudioSource::System => "spk",
                    };
                    format!("{prefix}_{}", segment.cluster)
                });
            (from, to, name)
        })
        .collect::<Vec<_>>();
    segments.sort_by_key(|(from, to, _)| (*from, *to));
    segments
        .into_iter()
        .map(|(from, to, name)| (seconds(from - start_ms), seconds(to - start_ms), name))
        .filter(|(from, to, _)| to > from)
        .collect()
}

/// `<session>_<m>m<ss>s` from the clip start, with the track when it is not
/// the line's own.
pub fn folder_base(session: &str, line: LineRef, track: Track, clip_start_ms: u64) -> String {
    let seconds = clip_start_ms / 1000;
    let track = if track == Track::of(line.source) {
        String::new()
    } else {
        format!("-{}", track.as_str())
    };
    format!("{session}_{}m{:02}s{track}", seconds / 60, seconds % 60)
}

/// Which recording a case belongs to. A folder name alone does not tell:
/// two recordings may have folders of the same name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owner {
    pub session: String,
    pub started: String,
}

impl Owner {
    pub fn of(session: &Session) -> io::Result<Self> {
        Ok(Self {
            session: session_name(session),
            started: session.read_manifest()?.started_wallclock,
        })
    }

    /// Whether `folder` is a real directory, not a link, holding this
    /// recording's case of `line`.
    fn owns(&self, folder: &Path, line: LineRef) -> bool {
        fs::symlink_metadata(folder).is_ok_and(|metadata| metadata.is_dir())
            && read_truth(folder).is_ok_and(|truth| {
                truth.source.session == self.session
                    && truth.source.session_started == self.started
                    && truth.editor.line == line
            })
    }
}

/// `folder` as a case folder directly in `root` (canonical): a plain,
/// visible name in a parent that is `root`. Anything else is never replaced
/// or removed.
fn in_root(root: &Path, folder: &Path) -> Option<PathBuf> {
    let name = folder.file_name()?;
    let plain = Path::new(name)
        .components()
        .eq([std::path::Component::Normal(name)])
        && !name.to_string_lossy().starts_with('.');
    (plain && fs::canonicalize(folder.parent()?).ok()? == root).then(|| root.join(name))
}

/// The folder for a case: `base`, or `base-2`, `base-3`… when the name is
/// taken. A folder that already holds `owner`'s case of `line` may be
/// replaced, when `replace` allows it.
fn choose_folder(root: &Path, base: &str, owner: &Owner, line: LineRef, replace: bool) -> PathBuf {
    (1u32..)
        .map(|number| match number {
            1 => root.join(base),
            _ => root.join(format!("{base}-{number}")),
        })
        .find(|candidate| match fs::symlink_metadata(candidate) {
            Err(error) => error.kind() == io::ErrorKind::NotFound,
            Ok(_) => replace && owner.owns(candidate, line),
        })
        .expect("an unbounded range has a free name")
}

pub fn read_truth(folder: &Path) -> io::Result<Truth> {
    let bytes = fs::read(folder.join("truth.json"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// The labels of `line`'s saved case, to edit them again. Bars that do not
/// fit the saved lanes or clip, as in a case edited by hand, are left out.
pub fn reopen(folder: &Path, line: LineRef, recorded_ms: u64) -> io::Result<Labels> {
    let truth = read_truth(folder)?;
    if truth.format_version != FORMAT_VERSION || truth.editor.line != line {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} does not hold this line's case", folder.display()),
        ));
    }
    let mut labels = truth.editor.labels;
    labels.note = truth.notes.join("\n");
    labels.lanes = labels.lanes.clamp(1, MAX_SPEAKERS);
    labels.context_ms = labels.context_ms.min(MAX_CONTEXT_MS) / CONTEXT_STEP_MS * CONTEXT_STEP_MS;
    let (start_ms, end_ms) = clip_range(line, labels.context_ms, recorded_ms);
    let lanes = labels.lanes;
    labels.bars = labels
        .bars
        .into_iter()
        .filter(|bar| bar.lane == NOT_SURE || bar.lane <= lanes)
        .map(|bar| Bar {
            start_ms: bar.start_ms.max(start_ms),
            end_ms: bar.end_ms.min(end_ms),
            ..bar
        })
        .filter(|bar| bar.end_ms >= bar.start_ms.saturating_add(MIN_BAR_MS))
        .enumerate()
        .map(|(index, bar)| Bar {
            id: index as u64 + 1,
            ..bar
        })
        .collect();
    Ok(labels)
}

/// A saved case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Saved {
    #[serde(flatten)]
    pub line: LineRef,
    pub folder: PathBuf,
    /// Speakers labeled, for the line's button.
    pub speakers: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    format_version: u32,
    cases: Vec<Saved>,
}

/// The session's index; none yet is empty, an unreadable one an error.
fn read_index(session: &Session) -> io::Result<Vec<Saved>> {
    let path = session.test_cases_path();
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<Index>(&bytes)
            .map(|index| index.cases)
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{} is unreadable ({error}); fix or remove it",
                        path.display()
                    ),
                )
            }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error),
    }
}

/// The session's saved cases whose folders still hold them. An entry whose
/// folder is gone or now holds another case counts as unsaved.
pub fn saved_cases(session: &Session) -> Vec<Saved> {
    let Ok(owner) = Owner::of(session) else {
        return Vec::new();
    };
    read_index(session)
        .unwrap_or_default()
        .into_iter()
        .filter(|saved| owner.owns(&saved.folder, saved.line))
        .collect()
}

/// Saves the case in its own folder under `root` and notes it in the
/// session's index.
///
/// The case is written beside its place and moved in whole. It replaces the
/// line's earlier case of the same name in one step, by swapping the two
/// folders; where the file system cannot swap, it goes to a new name
/// instead. Only once the index points to the new case is the old one
/// removed, so a failure or crash at any point leaves the line with a case
/// it can reach. Only folders directly in `root` that hold this recording's
/// case of the line are ever replaced or removed.
pub fn save(root: &Path, draft: &Draft) -> io::Result<Saved> {
    save_with(root, draft, true)
}

fn save_with(root: &Path, draft: &Draft, swap: bool) -> io::Result<Saved> {
    let clip = draft.clip();
    if draft.line.start_ms >= draft.recorded_ms || clip.1 <= clip.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the recording ends before this line",
        ));
    }
    let owner = Owner::of(draft.session)?;
    let mut cases = read_index(draft.session)?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(root)?;
    let root = fs::canonicalize(root)?;
    let previous = cases
        .iter()
        .find(|saved| saved.line == draft.line)
        .and_then(|saved| in_root(&root, &saved.folder))
        .filter(|folder| owner.owns(folder, draft.line));
    let base = folder_base(&owner.session, draft.line, draft.labels.track, clip.0);
    let mut target = choose_folder(&root, &base, &owner, draft.line, swap);
    let name = |folder: &Path| {
        folder
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let truth = draft.truth(&name(&target), &owner);
    let staged = root.join(format!(".{}.saving", name(&target)));
    // Left by an interrupted save of this same case folder.
    if fs::symlink_metadata(&staged).is_ok_and(|meta| meta.is_dir()) {
        fs::remove_dir_all(&staged)?;
    }
    let installed = session::create_private_dir(&staged)
        .and_then(|()| write_case(&staged, draft, &truth, clip))
        .and_then(|()| {
            if fs::symlink_metadata(&target).is_err() {
                return fs::rename(&staged, &target).map(|()| false);
            }
            if exchange(&staged, &target).is_ok() {
                return Ok(true);
            }
            // No swapping here: leave the old case alone and use a new name.
            target = choose_folder(&root, &base, &owner, draft.line, false);
            let truth = draft.truth(&name(&target), &owner);
            write_file(&staged, "truth.json", &|file| write_truth(file, &truth))?;
            fs::rename(&staged, &target).map(|()| false)
        });
    let swapped = match installed {
        Ok(swapped) => swapped,
        Err(error) => {
            let _ = fs::remove_dir_all(&staged);
            return Err(error);
        }
    };
    let synced = sync_dir(&root);
    let saved = Saved {
        line: draft.line,
        folder: target.clone(),
        speakers: truth.speakers.len(),
    };
    cases.retain(|case| case.line != draft.line);
    cases.push(saved.clone());
    cases.sort_by_key(|case| (case.line.start_ms, case.line.source == AudioSource::System));
    let index_path = draft.session.test_cases_path();
    let indexed = synced.and_then(|()| {
        process::write_json_atomic(
            &index_path,
            &Index {
                format_version: FORMAT_VERSION,
                cases,
            },
        )
    });
    if let Err(error) = indexed {
        // Put back the case the index still points to.
        if !swapped {
            let _ = fs::remove_dir_all(&target);
        } else if exchange(&staged, &target).is_ok() {
            let _ = fs::remove_dir_all(&staged);
        } else {
            // The earlier case is in the staging folder now: keep it where
            // no later save sweeps it up.
            let kept = root.join(format!("{}.previous", name(&target)));
            let kept = if fs::rename(&staged, &kept).is_ok() {
                kept
            } else {
                staged
            };
            return Err(io::Error::new(
                error.kind(),
                format!("{error}; the earlier labels are kept in {}", kept.display()),
            ));
        }
        return Err(error);
    }
    if swapped {
        let _ = fs::remove_dir_all(&staged);
    }
    // Drop the superseded case only once the index pointing past it is on disk.
    let committed = index_path.parent().is_some_and(|dir| sync_dir(dir).is_ok());
    if let Some(previous) = previous.filter(|previous| committed && *previous != target) {
        let _ = fs::remove_dir_all(previous);
    }
    Ok(saved)
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    fs::File::open(dir)?.sync_all()
}

/// Swaps two folders in one step, where the file system can.
fn exchange(a: &Path, b: &Path) -> io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        a,
        rustix::fs::CWD,
        b,
        rustix::fs::RenameFlags::EXCHANGE,
    )
    .map_err(io::Error::from)
}

type Body<'a> = &'a dyn Fn(&mut BufWriter<fs::File>) -> io::Result<()>;

fn write_file(folder: &Path, name: &str, body: Body) -> io::Result<()> {
    let mut file = BufWriter::new(session::create_private_file(&folder.join(name))?);
    body(&mut file)?;
    file.flush()?;
    file.get_ref().sync_all()
}

fn write_truth(file: &mut BufWriter<fs::File>, truth: &Truth) -> io::Result<()> {
    file.write_all(to_json(truth)?.as_bytes())?;
    file.write_all(b"\n")
}

fn write_case(folder: &Path, draft: &Draft, truth: &Truth, clip: (u64, u64)) -> io::Result<()> {
    write_file(folder, "clip.wav", &|file| {
        write_wav(file, &draft.audio.clip(draft.labels.track, clip))
    })?;
    write_file(folder, "truth.json", &|file| write_truth(file, truth))?;
    write_file(folder, "words.jsonl", &|file| {
        for word in clip_words(&draft.inputs.words, draft.labels.track, clip) {
            serde_json::to_writer(&mut *file, &word)?;
            file.write_all(b"\n")?;
        }
        Ok(())
    })?;
    fs::File::open(folder)?.sync_all()
}

/// Pretty JSON with every list of plain values, such as a segment, on one
/// line, like the hand-labeled clips.
fn to_json<T: Serialize>(value: &T) -> io::Result<String> {
    let pretty = serde_json::to_string_pretty(value)?;
    let mut out = String::with_capacity(pretty.len());
    // The list being read, as written and as one line, while it holds no
    // list or object.
    let mut flat: Option<(String, String)> = None;
    let (mut quoted, mut escaped, mut indent) = (false, false, false);
    for character in pretty.chars() {
        let plain = !quoted;
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
        } else if character == '"' {
            quoted = true;
        }
        if plain && matches!(character, '[' | '{' | ']') {
            if let Some((written, one_line)) = flat.take() {
                out.push_str(if character == ']' {
                    &one_line
                } else {
                    &written
                });
            }
            if character == '[' {
                flat = Some(("[".into(), "[".into()));
                continue;
            }
            out.push(character);
            continue;
        }
        let Some((written, one_line)) = flat.as_mut() else {
            out.push(character);
            continue;
        };
        written.push(character);
        if plain && character == '\n' {
            indent = true;
        } else if !(indent && plain && character == ' ') {
            indent = false;
            one_line.push(character);
            if plain && character == ',' {
                one_line.push(' ');
            }
        }
    }
    if let Some((written, _)) = flat {
        out.push_str(&written);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FORMAT_VERSION as SESSION_FORMAT, Manifest, SessionState, StreamInfo};

    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "singstone-testcase-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("fixture root");
        root
    }

    /// A session with `mic` and `system` as its raw tracks.
    fn session_with(root: &Path, mic: &[f32], system: &[f32]) -> Session {
        let session = Session::create(root.join("session-20260914-103000")).expect("session");
        let stream = StreamInfo {
            enabled: true,
            pipewire_node: None,
            error: None,
        };
        session
            .write_manifest(&Manifest {
                format_version: SESSION_FORMAT,
                state: SessionState::Stopped,
                started_wallclock: "2026-09-14T10:30:00+02:00".into(),
                sample_rate: SAMPLE_RATE,
                channels: 1,
                sample_format: "f32le".into(),
                mic: stream.clone(),
                system: stream,
                local_speaker: "Me".into(),
                screenshot_dir: None,
            })
            .expect("manifest");
        for (source, samples) in [(AudioSource::Mic, mic), (AudioSource::System, system)] {
            let bytes = samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>();
            fs::write(session.audio_path(source), bytes).expect("audio");
        }
        session
    }

    fn bar(id: u64, lane: u8, start_ms: u64, end_ms: u64) -> Bar {
        Bar {
            lane,
            start_ms,
            end_ms,
            id,
        }
    }

    const LINE: LineRef = LineRef {
        source: AudioSource::System,
        start_ms: 38_000,
        end_ms: 51_000,
    };

    #[test]
    fn clip_takes_context_within_the_recording() {
        assert_eq!(clip_range(LINE, 1_000, 180_000), (37_000, 52_000));
        assert_eq!(clip_range(LINE, 5_000, 53_000), (33_000, 53_000));
        let first = LineRef {
            start_ms: 400,
            ..LINE
        };
        assert_eq!(clip_range(first, 1_000, 180_000), (0, 52_000));
    }

    #[test]
    fn both_tracks_are_summed_clipped_and_silent_past_their_end() {
        let root = fixture_root("mix");
        let session = session_with(&root, &[0.25; 32], &[0.5, 0.9, -0.9, 0.1]);
        let audio = Audio::read(&session, 0, 2).expect("read");
        let both = audio.clip(Track::Both, (0, 2));
        assert_eq!(both.len(), 32);
        assert_eq!(&both[..5], &[0.75, 1.0, -0.65, 0.35, 0.25]);
        assert_eq!(audio.clip(Track::System, (0, 1))[4..], [0.0; 12]);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn archived_tracks_read_like_the_recording() {
        crate::audio::archive::ensure_decoder().expect("the flac tool is installed");
        let root = fixture_root("archived");
        let system = (0..32_000)
            .map(|index| ((index % 200) as f32 / 200.0 - 0.5) * 0.8)
            .collect::<Vec<_>>();
        let session = session_with(&root, &[0.1; 16_000], &system);
        fs::write(session.transcript_path(), "").expect("processed");
        crate::audio::archive::archive_session(&session).expect("archive");
        let clip = Audio::read(&session, 1_000, 1_500)
            .expect("read")
            .clip(Track::System, (1_000, 1_500));
        assert_eq!(clip.len(), 8_000);
        for (got, wanted) in clip.iter().zip(&system[16_000..]) {
            assert!((got - wanted).abs() < 1e-4, "{got} vs {wanted}");
        }
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn wav_is_16_bit_mono_at_the_recording_rate() {
        let mut wav = Vec::new();
        write_wav(&mut wav, &[0.0, 0.5, -1.0, 2.0]).expect("wav");
        assert_eq!(wav.len(), 44 + 8);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 44);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 8);
        let samples = wav[44..]
            .chunks(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        assert_eq!(samples, [0, 16_384, -32_768, 32_767]);
    }

    #[test]
    fn a_bar_joins_the_bars_of_its_lane_it_touches() {
        let mut bars = vec![
            bar(1, 1, 1_000, 2_000),
            bar(2, 1, 2_000, 3_000),
            bar(3, 2, 1_500, 2_500),
            bar(4, 1, 2_800, 4_000),
            bar(5, 1, 4_100, 5_000),
        ];
        merge_lane(&mut bars, 2);
        assert_eq!(
            bars,
            [
                bar(2, 1, 1_000, 4_000),
                bar(3, 2, 1_500, 2_500),
                bar(5, 1, 4_100, 5_000)
            ]
        );
    }

    #[test]
    fn saved_bars_are_cut_to_the_clip_joined_and_by_start() {
        let bars = [
            bar(1, 2, 50_000, 60_000),
            bar(2, 1, 30_000, 39_000),
            bar(3, 1, 38_500, 40_000),
            bar(4, 0, 10_000, 20_000),
        ];
        assert_eq!(
            clip_bars(&bars, (37_000, 52_000)),
            [bar(2, 1, 37_000, 40_000), bar(1, 2, 50_000, 52_000)]
        );
    }

    #[test]
    fn overlap_is_where_two_speakers_talk_at_once() {
        let bars = [
            bar(1, 1, 0, 3_000),
            bar(2, 2, 2_000, 5_000),
            bar(3, 3, 4_000, 6_000),
            bar(4, 0, 0, 6_000),
        ];
        assert_eq!(
            overlaps(&bars),
            [(2_000, 3_000, 1, 2), (4_000, 5_000, 2, 3)]
        );
    }

    fn inputs() -> Inputs {
        let line = |start_ms, end_ms, speaker: &str| Utterance {
            start_ms,
            end_ms,
            source: AudioSource::System,
            speaker_id: "spk_1".into(),
            speaker: speaker.into(),
            text: String::new(),
            locked: false,
            echo: false,
            group: None,
        };
        let segment = |source, start_ms, end_ms, cluster| SpeakerSegment {
            source,
            start_ms,
            end_ms,
            cluster,
        };
        let word = |source, start_ms, end_ms, text: &str| TimedWord {
            source,
            start_ms,
            end_ms,
            text: text.into(),
        };
        Inputs {
            words: vec![
                word(AudioSource::System, 38_100, 38_400, " Did"),
                word(AudioSource::Mic, 39_000, 39_300, "Yes"),
                word(AudioSource::System, 36_000, 37_600, "early"),
            ],
            diarization: vec![
                segment(AudioSource::System, 30_000, 45_004, 1),
                segment(AudioSource::System, 45_004, 60_000, 3),
                segment(AudioSource::Mic, 39_000, 40_000, 0),
            ],
            transcript: vec![line(38_000, 45_000, "Laura")],
            language: Some("en".into()),
        }
    }

    fn draft<'a>(
        session: &'a Session,
        labels: &'a Labels,
        audio: &'a Audio,
        inputs: &'a Inputs,
    ) -> Draft<'a> {
        Draft {
            session,
            line: LINE,
            line_speaker: "Laura",
            labels,
            recorded_ms: 180_000,
            audio,
            inputs,
            labeled_at: "2026-11-04T14:12:00+01:00",
        }
    }

    fn owner(session: &str) -> Owner {
        Owner {
            session: session.into(),
            started: "2026-09-14T10:30:00+02:00".into(),
        }
    }

    #[test]
    fn truth_is_on_the_clip_axis_rounded_and_sorted() {
        let session = Session {
            dir: PathBuf::from("/meetings/ami-3min"),
        };
        let labels = Labels {
            bars: vec![
                bar(1, 2, 40_454, 43_000),
                bar(2, 1, 38_104, 40_600),
                bar(3, 0, 46_400, 46_800),
                bar(4, 3, 60_000, 61_000),
            ],
            lanes: 3,
            note: "P2 laughs\n\n".into(),
            ..Labels::new(AudioSource::System)
        };
        let (audio, inputs) = (Audio::default(), inputs());
        let truth =
            draft(&session, &labels, &audio, &inputs).truth("ami-3min_0m37s", &owner("ami-3min"));
        assert_eq!(truth.source.offset_s, 37.0);
        assert_eq!(truth.source.duration_s, 15.0);
        assert_eq!(
            truth.segments,
            [(1.1, 3.6, "P1".to_owned()), (3.45, 6.0, "P2".to_owned())]
        );
        assert_eq!(truth.exclude, [(9.4, 9.8)]);
        assert_eq!(truth.speakers.len(), 2);
        assert_eq!(truth.speakers["P2"], "Speaker 2");
        assert_eq!(truth.notes, ["P2 laughs"]);
        assert_eq!(
            truth.app.segments,
            [
                (0.0, 8.0, "Laura".to_owned()),
                (8.0, 15.0, "spk_3".to_owned())
            ]
        );
        let json = to_json(&truth).expect("json");
        assert!(json.contains(
            "\n  \"segments\": [\n    [1.1, 3.6, \"P1\"],\n    [3.45, 6.0, \"P2\"]\n  ],\n"
        ));
        assert!(json.contains("\"speakers\": {\n    \"P1\": \"Speaker 1\",\n"));
        assert!(json.contains("\"notes\": [\"P2 laughs\"],\n"));
        let parsed: Truth = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.segments, truth.segments);
        let tricky = serde_json::json!({
            "notes": [r#"said "[ a, b ]" here \ {x}"#, "\\\"", "]"],
            "list": [1, 2]
        });
        let parsed: serde_json::Value =
            serde_json::from_str(&to_json(&tricky).expect("json")).expect("parse");
        assert_eq!(parsed, tricky);
    }

    #[test]
    fn both_tracks_name_each_diarized_voice_by_its_own_track() {
        let names = app_segments(&inputs(), Track::Both, (37_000, 52_000))
            .into_iter()
            .map(|(_, _, name)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["Laura", "mic_0", "spk_3"]);
    }

    #[test]
    fn words_are_those_of_the_track_retimed_to_the_clip() {
        let words = inputs().words;
        let system = clip_words(&words, Track::System, (37_000, 52_000));
        assert_eq!(
            system,
            [ClipWord {
                start: 1.1,
                end: 1.4,
                text: "Did".into(),
                source: None
            }]
        );
        let both = clip_words(&words, Track::Both, (37_000, 52_000));
        assert_eq!(both.len(), 2);
        assert_eq!(both[1].source, Some(AudioSource::Mic));
    }

    #[test]
    fn a_word_too_short_to_round_lasts_10_ms_within_the_clip() {
        let word = |start_ms, end_ms| TimedWord {
            source: AudioSource::System,
            start_ms,
            end_ms,
            text: ".".into(),
        };
        let words = clip_words(
            &[word(1_001, 1_004), word(1_990, 1_993), word(1_996, 1_999)],
            Track::System,
            (1_000, 2_000),
        );
        let times = words
            .iter()
            .map(|word| (word.start, word.end))
            .collect::<Vec<_>>();
        assert_eq!(times, [(0.0, 0.01), (0.99, 1.0)]);
    }

    fn write_truth_at(folder: &Path, truth: &Truth) {
        fs::create_dir_all(folder).expect("folder");
        fs::write(folder.join("truth.json"), to_json(truth).expect("json")).expect("truth");
    }

    fn truth_of(owner: &Owner, line: LineRef) -> Truth {
        let truth = draft(
            &Session {
                dir: PathBuf::from("/meetings").join(&owner.session),
            },
            &Labels::new(line.source),
            &Audio::default(),
            &Inputs::default(),
        )
        .truth("case", owner);
        Truth {
            editor: Editor {
                line,
                labels: Labels::new(line.source),
            },
            ..truth
        }
    }

    #[test]
    fn a_folder_holding_another_case_gets_a_number() {
        let root = fixture_root("names");
        assert_eq!(
            folder_base("ami-3min", LINE, Track::Both, 97_400),
            "ami-3min_1m37s-both"
        );
        let ours = owner("ami-3min");
        let other_line = LineRef {
            start_ms: 37_500,
            ..LINE
        };
        let taken = root.join("ami-3min_0m37s");
        write_truth_at(&taken, &truth_of(&ours, other_line));
        let other_recording = Owner {
            started: "2026-09-15T09:00:00+02:00".into(),
            ..ours.clone()
        };
        let choose =
            |owner: &Owner, line| choose_folder(&root, "ami-3min_0m37s", owner, line, true);
        assert_eq!(choose(&ours, LINE), root.join("ami-3min_0m37s-2"));
        assert_eq!(
            choose(&other_recording, other_line),
            root.join("ami-3min_0m37s-2")
        );
        assert_eq!(choose(&ours, other_line), taken);
        fs::remove_dir_all(root).expect("cleanup");
    }

    const LINE_10S: LineRef = LineRef {
        source: AudioSource::System,
        start_ms: 10_000,
        end_ms: 20_000,
    };

    /// A recording of a minute with mic 0.1 and system 0.2 throughout.
    fn minute(root: &Path) -> (Session, Audio) {
        let session = session_with(root, &[0.1; 16_000 * 60], &[0.2; 16_000 * 60]);
        let audio = Audio::read(&session, 5_000, 25_000).expect("audio");
        (session, audio)
    }

    fn labels_with(bars: Vec<Bar>) -> Labels {
        Labels {
            bars,
            ..Labels::new(AudioSource::System)
        }
    }

    fn save_line(
        cases: &Path,
        session: &Session,
        audio: &Audio,
        labels: &Labels,
        swap: bool,
    ) -> io::Result<Saved> {
        let inputs = Inputs::default();
        save_with(
            cases,
            &Draft {
                line: LINE_10S,
                ..draft(session, labels, audio, &inputs)
            },
            swap,
        )
    }

    fn names_in(folder: &Path) -> Vec<String> {
        let mut names = fs::read_dir(folder)
            .expect("folder")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).expect("metadata").permissions().mode() & 0o777
    }

    #[test]
    fn saving_under_a_new_name_moves_the_case_and_reopens_as_saved() {
        let root = fixture_root("save");
        let (session, audio) = minute(&root);
        let cases = root.join("Diarization test cases");
        let mut labels = labels_with(vec![bar(1, 1, 9_500, 15_000), bar(2, 2, 14_000, 20_000)]);
        let first = save_line(&cases, &session, &audio, &labels, true).expect("save");
        let cases = fs::canonicalize(&cases).expect("cases");
        assert_eq!(first.folder, cases.join("session-20260914-103000_0m09s"));
        assert_eq!(first.speakers, 2);
        labels.context_ms = 2_000;
        labels.track = Track::Both;
        labels.lanes = 3;
        labels.bars.push(bar(3, 3, 19_000, 21_000));
        labels.note = "Third voice at the end".into();
        let second = save_line(&cases, &session, &audio, &labels, true).expect("save");
        assert_eq!(
            second.folder,
            cases.join("session-20260914-103000_0m08s-both")
        );
        assert_eq!(names_in(&cases), ["session-20260914-103000_0m08s-both"]);
        assert_eq!(saved_cases(&session), std::slice::from_ref(&second));
        let wav = fs::read(second.folder.join("clip.wav")).expect("wav");
        assert_eq!(wav.len(), 44 + 2 * 16 * 14_000);
        assert_eq!(i16::from_le_bytes([wav[44], wav[45]]), 9_830);
        assert_eq!(mode(&second.folder), 0o700);
        assert_eq!(mode(&second.folder.join("truth.json")), 0o600);
        let mut reopened = reopen(&second.folder, LINE_10S, 60_000).expect("reopen");
        for bar in reopened.bars.iter_mut().chain(&mut labels.bars) {
            bar.id = 0;
        }
        labels.bars.sort_by_key(|bar| bar.start_ms);
        assert_eq!(reopened, labels);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn saving_again_under_the_same_name_swaps_the_case_in() {
        use std::os::unix::fs::MetadataExt;
        let root = fixture_root("swap");
        let (session, audio) = minute(&root);
        let cases = root.join("cases");
        let first = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 1, 10_000, 12_000)]),
            true,
        )
        .expect("save");
        let inode = fs::metadata(&first.folder).expect("first").ino();
        let own = first
            .folder
            .with_file_name(".session-20260914-103000_0m09s.saving");
        let foreign = first.folder.with_file_name(".foreign.saving");
        fs::create_dir(&own).expect("leftover");
        fs::create_dir(&foreign).expect("foreign");
        let second = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 2, 10_000, 12_000)]),
            true,
        )
        .expect("save");
        assert_eq!(second.folder, first.folder);
        assert_ne!(fs::metadata(&second.folder).expect("second").ino(), inode);
        assert_eq!(
            read_truth(&second.folder).expect("truth").segments[0].2,
            "P2"
        );
        assert_eq!(
            names_in(&fs::canonicalize(&cases).expect("cases")),
            [".foreign.saving", "session-20260914-103000_0m09s"]
        );
        assert_eq!(mode(&second.folder), 0o700);
        assert_eq!(mode(&second.folder.join("clip.wav")), 0o600);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn without_swapping_a_new_save_takes_a_new_name_and_then_drops_the_old() {
        let root = fixture_root("noswap");
        let (session, audio) = minute(&root);
        let cases = root.join("cases");
        let first = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 1, 10_000, 12_000)]),
            false,
        )
        .expect("save");
        let second = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 2, 10_000, 12_000)]),
            false,
        )
        .expect("save");
        assert_eq!(
            second.folder.file_name().expect("name"),
            "session-20260914-103000_0m09s-2"
        );
        assert_eq!(
            read_truth(&second.folder).expect("truth").id,
            "session-20260914-103000_0m09s-2"
        );
        assert!(!first.folder.exists());
        assert_eq!(saved_cases(&session), [second]);
        assert_eq!(
            names_in(&fs::canonicalize(&cases).expect("cases")),
            ["session-20260914-103000_0m09s-2"]
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn a_failed_index_write_keeps_the_case_the_index_points_to() {
        let root = fixture_root("index-fails");
        let (session, audio) = minute(&root);
        let cases = root.join("cases");
        let first = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 1, 10_000, 12_000)]),
            true,
        )
        .expect("save");
        let before = fs::read(first.folder.join("truth.json")).expect("truth");
        // The index's temporary file cannot be created.
        fs::create_dir(jsonl::tmp_path(&session.test_cases_path())).expect("block");
        for context_ms in [1_000, 3_000] {
            let labels = Labels {
                context_ms,
                ..labels_with(vec![bar(1, 2, 10_000, 12_000)])
            };
            save_line(&cases, &session, &audio, &labels, true)
                .expect_err("index cannot be written");
            assert_eq!(saved_cases(&session), std::slice::from_ref(&first));
            assert_eq!(
                fs::read(first.folder.join("truth.json")).expect("truth"),
                before
            );
            assert_eq!(
                names_in(&fs::canonicalize(&cases).expect("cases")),
                ["session-20260914-103000_0m09s"]
            );
        }
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn an_unreadable_index_stops_the_save_and_is_kept() {
        let root = fixture_root("corrupt");
        let (session, audio) = minute(&root);
        let cases = root.join("cases");
        fs::write(session.test_cases_path(), "{ not json").expect("corrupt");
        let error = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 1, 10_000, 12_000)]),
            true,
        )
        .expect_err("corrupt index");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            fs::read_to_string(session.test_cases_path()).expect("index"),
            "{ not json"
        );
        assert!(!cases.exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn an_index_entry_outside_the_folder_is_never_removed() {
        let root = fixture_root("traversal");
        let (session, audio) = minute(&root);
        let cases = root.join("chosen/cases");
        fs::create_dir_all(&cases).expect("cases");
        let sentinel = root.join("chosen/keep.txt");
        fs::write(&sentinel, "keep").expect("sentinel");
        let owner = Owner::of(&session).expect("owner");
        write_truth_at(&root.join("chosen"), &truth_of(&owner, LINE_10S));
        process::write_json_atomic(
            &session.test_cases_path(),
            &Index {
                format_version: FORMAT_VERSION,
                cases: vec![Saved {
                    line: LINE_10S,
                    folder: cases.join(".."),
                    speakers: 1,
                }],
            },
        )
        .expect("index");
        let saved = save_line(
            &cases,
            &session,
            &audio,
            &labels_with(vec![bar(1, 1, 10_000, 12_000)]),
            true,
        )
        .expect("save");
        assert!(sentinel.exists());
        assert!(
            saved
                .folder
                .starts_with(fs::canonicalize(&cases).expect("cases"))
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn a_stale_entry_whose_folder_holds_another_line_is_left_alone() {
        let root = fixture_root("stale");
        let (session, audio) = minute(&root);
        let cases = root.join("cases");
        let labels = labels_with(vec![bar(1, 1, 10_000, 12_000)]);
        let first = save_line(&cases, &session, &audio, &labels, true).expect("save");
        // The folder is removed by hand and another line of the same second
        // takes its name.
        fs::remove_dir_all(&first.folder).expect("remove");
        let other = LineRef {
            start_ms: 10_400,
            ..LINE_10S
        };
        let inputs = Inputs::default();
        let theirs = save(
            &cases,
            &Draft {
                line: other,
                ..draft(&session, &labels, &audio, &inputs)
            },
        )
        .expect("save other");
        assert_eq!(theirs.folder, first.folder);
        let index = read_index(&session).expect("index");
        assert_eq!(index.len(), 2);
        assert_eq!(saved_cases(&session), std::slice::from_ref(&theirs));
        let again = save_line(&cases, &session, &audio, &labels, true).expect("save again");
        assert_ne!(again.folder, theirs.folder);
        assert_eq!(
            read_truth(&theirs.folder).expect("theirs").editor.line,
            other
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn two_recordings_in_folders_of_the_same_name_keep_their_own_cases() {
        let root = fixture_root("twins");
        let cases = root.join("cases");
        let labels = labels_with(vec![bar(1, 1, 10_000, 12_000)]);
        let mut saved = Vec::new();
        for (meetings, started) in [
            ("a", "2026-09-14T10:30:00+02:00"),
            ("b", "2026-10-01T08:00:00+02:00"),
        ] {
            let (session, audio) = minute(&root.join(meetings));
            let mut manifest = session.read_manifest().expect("manifest");
            manifest.started_wallclock = started.into();
            session.write_manifest(&manifest).expect("manifest");
            saved.push((
                session.clone(),
                save_line(&cases, &session, &audio, &labels, true).expect("save"),
            ));
        }
        assert_ne!(saved[0].1.folder, saved[1].1.folder);
        for (session, case) in &saved {
            assert_eq!(saved_cases(session), std::slice::from_ref(case));
        }
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn reopening_leaves_out_bars_that_do_not_fit() {
        let root = fixture_root("reopen");
        let ours = owner("ami-3min");
        let mut truth = truth_of(&ours, LINE);
        truth.editor.labels = Labels {
            lanes: 12,
            context_ms: 1_500,
            bars: vec![
                bar(0, 10, 39_000, 40_000),
                bar(0, 1, 40_000, 39_000),
                bar(0, 2, 36_000, 38_500),
                bar(0, 0, 60_000, 61_000),
                bar(0, 3, 41_000, 41_010),
            ],
            ..Labels::new(AudioSource::System)
        };
        let folder = root.join("case");
        write_truth_at(&folder, &truth);
        let labels = reopen(&folder, LINE, 180_000).expect("reopen");
        assert_eq!(labels.lanes, MAX_SPEAKERS);
        assert_eq!(labels.context_ms, 1_000);
        assert_eq!(labels.bars, [bar(1, 2, 37_000, 38_500)]);
        let elsewhere = LineRef {
            start_ms: 1_000,
            ..LINE
        };
        assert_eq!(
            reopen(&folder, elsewhere, 180_000)
                .expect_err("another line")
                .kind(),
            io::ErrorKind::InvalidData
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn a_line_after_the_end_of_the_recording_cannot_be_saved() {
        let root = fixture_root("empty");
        let (session, audio) = minute(&root);
        let inputs = Inputs::default();
        let labels = labels_with(Vec::new());
        let late = LineRef {
            start_ms: 61_000,
            end_ms: 65_000,
            ..LINE
        };
        let error = save(
            &root.join("cases"),
            &Draft {
                line: late,
                recorded_ms: 60_000,
                ..draft(&session, &labels, &audio, &inputs)
            },
        )
        .expect_err("no audio");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        fs::remove_dir_all(root).expect("cleanup");
    }
}
