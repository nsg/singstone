//! Merging recordings that have not been processed yet into one new session,
//! in a chosen order. The pieces stay as archived backups.

use crate::audio::archive::{self, ArchiveSummary};
use crate::audio::track::TrackReader;
use crate::cli::MergeArgs;
use crate::format::jsonl;
use crate::meeting::{self, Attendees, MeetingDetails};
use crate::merge::process;
use crate::session::{self, Session, WallClock, WallClockMap};
use crate::split::{self, Undo};
use crate::types::{
    AudioSource, HiddenSources, KeptRange, Manifest, MeetingSpan, MeetingSpanFile, SAMPLE_RATE,
    ScreenshotEntry, SessionState, StreamInfo, TimelineEvent, samples_to_ms,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const MERGE_FORMAT_VERSION: u32 = 1;
/// Digital silence between two pieces, so that no transcript line runs
/// across a seam.
pub const SEAM_MS: u64 = 1_000;
const SAMPLES_PER_MS: u64 = SAMPLE_RATE as u64 / 1000;
const SEAM_SAMPLES: u64 = SEAM_MS * SAMPLES_PER_MS;
const CHUNK_SAMPLES: usize = 16_384;
/// Free space kept in reserve beyond what the merge itself writes.
const SPACE_MARGIN: u64 = 64 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeOrder {
    /// By start time.
    Recorded,
    /// As the user put them.
    Chosen,
}

/// `merged-from.json` in the new session: its pieces, in the merged order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedFrom {
    pub format_version: u32,
    pub seam_ms: u64,
    pub order: MergeOrder,
    pub pieces: Vec<MergedPiece>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedPiece {
    /// Folder name of the piece, next to this session.
    pub session: String,
    /// The piece's title when it was merged.
    pub title: String,
    pub started_wallclock: String,
    /// Where the piece starts in the merged recording.
    pub offset_ms: u64,
    pub length_ms: u64,
}

/// `merged-into.json` in each piece: every merge it went into.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedInto {
    pub format_version: u32,
    pub merges: Vec<MergeRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeRecord {
    /// Folder name of the merged session.
    pub session: String,
    /// This piece's place in it, from 1.
    pub piece: usize,
}

#[derive(Debug)]
pub struct MergeOutcome {
    pub session: PathBuf,
    pub archived: Vec<ArchiveSummary>,
}

/// A session about to be merged, with what the merge needs to know of it.
#[derive(Debug, Clone)]
pub struct Piece {
    pub session: Session,
    pub manifest: Manifest,
    pub title: String,
    /// Stored sample count per track.
    pub tracks: Vec<(AudioSource, u64)>,
    /// The longest track: the piece's length.
    pub samples: u64,
    pub clock: WallClockMap,
}

impl Piece {
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let session = Session::open(dir)?;
        let manifest = session.read_manifest()?;
        let tracks = [AudioSource::Mic, AudioSource::System]
            .into_iter()
            .filter_map(|source| {
                let path = session.stored_audio_path(source);
                path.is_file().then_some((source, path))
            })
            .map(|(source, path)| Ok((source, process::stored_sample_count(&path)?)))
            .collect::<io::Result<Vec<_>>>()?;
        let samples = tracks
            .iter()
            .map(|(_, samples)| *samples)
            .max()
            .unwrap_or(0);
        let clock = session.wallclock_map(&manifest)?;
        Ok(Self {
            title: session.title(),
            session,
            manifest,
            tracks,
            samples,
            clock,
        })
    }

    pub fn length_ms(&self) -> u64 {
        samples_to_ms(self.samples)
    }

    pub fn name(&self) -> String {
        folder_name(&self.session.dir)
    }

    pub fn has_track(&self, source: AudioSource) -> bool {
        self.tracks
            .iter()
            .any(|(candidate, _)| *candidate == source)
    }

    pub fn started_unix_ms(&self) -> Option<i64> {
        session::unix_ms(self.clock.started())
    }

    /// When each stretch of the piece was recorded, as Unix milliseconds.
    pub fn stretches(&self) -> Vec<(i64, i64)> {
        let length_ms = self.length_ms();
        let Some(clock) = self.clock.slice(0, length_ms) else {
            return Vec::new();
        };
        clock
            .segments()
            .iter()
            .filter_map(|segment| {
                let start = session::unix_ms(&segment.started_wallclock)?;
                Some((
                    start,
                    start.saturating_add(i64::try_from(segment.length_ms).ok()?),
                ))
            })
            .collect()
    }

    /// "2026-09-14 09:30–09:58".
    pub fn range_label(&self) -> String {
        match (self.clock.at(0), self.clock.at(self.length_ms())) {
            (Some(start), Some(end)) if date_label(&start) == date_label(&end) => format!(
                "{} {}–{}",
                date_label(&start),
                time_label(&start),
                time_label(&end)
            ),
            (Some(start), Some(end)) => format!(
                "{} {} – {} {}",
                date_label(&start),
                time_label(&start),
                date_label(&end),
                time_label(&end)
            ),
            _ => self.manifest.started_wallclock.clone(),
        }
    }
}

/// Why `session` cannot be merged.
pub fn refusal(session: &Session, manifest: &Manifest) -> Option<&'static str> {
    if manifest.state == SessionState::Recording {
        Some("it is still being recorded")
    } else if session.transcript_path().is_file() {
        Some("it is already processed; merging is for recordings that have not been processed yet")
    } else if session.is_archived() {
        Some("it is archived; archived sessions are kept as backups")
    } else {
        None
    }
}

/// Whether the two pieces were recording at the same time.
pub fn overlap(first: &Piece, second: &Piece) -> bool {
    let theirs = second.stretches();
    first.stretches().iter().any(|(start, end)| {
        theirs
            .iter()
            .any(|(other_start, other_end)| start < other_end && other_start < end)
    })
}

/// Earliest first.
pub fn sort_by_start(pieces: &mut [Piece]) {
    pieces.sort_by_key(|piece| (piece.started_unix_ms().unwrap_or(i64::MAX), piece.name()));
}

/// Where each piece starts in the merged recording, in samples.
pub fn offsets(samples: &[u64]) -> Vec<u64> {
    let mut offset = 0;
    samples
        .iter()
        .map(|samples| {
            let start = offset;
            offset = start + samples + SEAM_SAMPLES;
            start
        })
        .collect()
}

/// The time of day across the merged recording.
pub fn merged_clock<'a>(pieces: impl IntoIterator<Item = &'a Piece>) -> Option<WallClockMap> {
    let pieces = pieces.into_iter().collect::<Vec<_>>();
    let offsets = offsets(&pieces.iter().map(|piece| piece.samples).collect::<Vec<_>>());
    WallClockMap::joined(
        &pieces
            .iter()
            .zip(&offsets)
            .map(|(piece, offset)| (&piece.clock, samples_to_ms(*offset), piece.length_ms()))
            .collect::<Vec<_>>(),
    )
}

/// Whether `pieces` can be merged in this order; the error says why not.
pub fn check(pieces: &[Piece]) -> io::Result<()> {
    let refuse = |message: String| Err(io::Error::new(io::ErrorKind::InvalidInput, message));
    if pieces.len() < 2 {
        return refuse("pick at least two recordings to merge".into());
    }
    let mut seen = Vec::new();
    let mut root = None;
    for piece in pieces {
        let dir = fs::canonicalize(&piece.session.dir)?;
        if seen.contains(&dir) {
            return refuse(format!("“{}” is picked twice", piece.title));
        }
        let parent = dir.parent().map(Path::to_path_buf);
        if *root.get_or_insert_with(|| parent.clone()) != parent {
            return refuse("the sessions to merge must be in the same folder".into());
        }
        seen.push(dir);
        if let Some(reason) = refusal(&piece.session, &piece.manifest) {
            return refuse(format!("“{}” cannot be merged: {reason}", piece.title));
        }
        if piece.samples == 0 {
            return refuse(format!("“{}” has no recorded audio", piece.title));
        }
        if piece.clock.at(0).is_none() || piece.started_unix_ms().is_none() {
            return refuse(format!(
                "“{}” has no valid start time ({:?})",
                piece.title, piece.manifest.started_wallclock
            ));
        }
    }
    for (index, first) in pieces.iter().enumerate() {
        for second in &pieces[index + 1..] {
            if overlap(first, second) {
                return refuse(format!(
                    "“{}” ({}) and “{}” ({}) overlap in time. They are the same moment recorded twice, so only one of them can go into a merge.",
                    first.title,
                    first.range_label(),
                    second.title,
                    second.range_label()
                ));
            }
        }
    }
    Ok(())
}

pub fn read_merged_from(session: &Session) -> io::Result<Option<MergedFrom>> {
    split::read_optional(&session.merged_from_path())
}

pub fn read_merged_into(session: &Session) -> io::Result<Option<MergedInto>> {
    split::read_optional(&session.merged_into_path())
}

pub fn run(args: MergeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut pieces = args
        .sessions
        .iter()
        .map(Piece::open)
        .collect::<io::Result<Vec<_>>>()?;
    let order = if args.keep_order {
        MergeOrder::Chosen
    } else {
        sort_by_start(&mut pieces);
        MergeOrder::Recorded
    };
    let title = args.title.unwrap_or_else(|| pieces[0].title.clone());
    let outcome = merge_sessions(&pieces, &title, order)?;
    println!("{}", outcome.session.display());
    let length_ms = samples_to_ms(
        offsets(&pieces.iter().map(|piece| piece.samples).collect::<Vec<_>>())
            .last()
            .copied()
            .unwrap_or(0)
            + pieces.last().map_or(0, |piece| piece.samples),
    );
    eprintln!(
        "merged {} recordings into {} ({}):",
        pieces.len(),
        folder_name(&outcome.session),
        split::clock(length_ms)
    );
    for piece in &pieces {
        eprintln!("  {} ({})", piece.name(), piece.range_label());
    }
    let (before, after) = outcome
        .archived
        .iter()
        .fold((0, 0), |(before, after), summary| {
            (before + summary.bytes_before, after + summary.bytes_after)
        });
    eprintln!(
        "kept the pieces as archived backups: {} -> {}",
        archive::format_size(before),
        archive::format_size(after)
    );
    Ok(())
}

/// Write `pieces`, in this order, into one new session next to them, read
/// it back against them, then archive the pieces as its backups. Nothing
/// changes unless every step succeeds.
pub fn merge_sessions(
    pieces: &[Piece],
    title: &str,
    order: MergeOrder,
) -> io::Result<MergeOutcome> {
    let mut undo = Undo::default();
    match write_merge(pieces, title, order, &mut undo) {
        Ok(outcome) => {
            undo.disarm();
            Ok(outcome)
        }
        Err(error) => Err(undo.fail(error)),
    }
}

fn write_merge(
    pieces: &[Piece],
    title: &str,
    order: MergeOrder,
    undo: &mut Undo,
) -> io::Result<MergeOutcome> {
    check(pieces)?;
    let samples = pieces.iter().map(|piece| piece.samples).collect::<Vec<_>>();
    let offsets = offsets(&samples);
    let total = offsets[offsets.len() - 1] + samples[samples.len() - 1];
    let sources = [AudioSource::Mic, AudioSource::System]
        .into_iter()
        .filter(|source| pieces.iter().any(|piece| piece.has_track(*source)))
        .collect::<Vec<_>>();
    let raw_bytes = pieces
        .iter()
        .flat_map(|piece| {
            piece.tracks.iter().filter_map(|(source, samples)| {
                (!archive::is_archived(&piece.session.stored_audio_path(*source)))
                    .then_some(samples * 4)
            })
        })
        .sum::<u64>();
    archive::ensure_decoder()?;
    let screenshots = pieces
        .iter()
        .map(
            |piece| match jsonl::read_all(&piece.session.screenshots_index_path()) {
                Ok(entries) => Ok(entries),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
                Err(error) => Err(error),
            },
        )
        .collect::<io::Result<Vec<Vec<ScreenshotEntry>>>>()?;
    let screenshot_bytes = pieces
        .iter()
        .zip(&screenshots)
        .flat_map(|(piece, entries)| {
            entries.iter().filter_map(|entry| {
                split::open_screenshot(&piece.session, &entry.file)?
                    .1
                    .metadata()
                    .ok()
            })
        })
        .map(|metadata| metadata.len())
        .sum::<u64>();

    let first = &pieces[0];
    let root = match first.session.dir.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    // The merged tracks in raw float, the screenshots again, and the pieces'
    // FLAC copies, which are at most half their raw size.
    let needed = sources.len() as u64 * total * 4 + screenshot_bytes + raw_bytes / 2 + SPACE_MARGIN;
    let stats = rustix::fs::statvfs(&root).map_err(io::Error::from)?;
    let free = stats.f_bavail.saturating_mul(stats.f_frsize);
    if free < needed {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "not enough free disk space: merging needs about {}, {} is free",
                archive::format_size(needed),
                archive::format_size(free)
            ),
        ));
    }
    let invalid_time = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "the pieces' start times are not valid times",
        )
    };
    let clock = merged_clock(pieces).ok_or_else(invalid_time)?;
    let started = clock.at(0).ok_or_else(invalid_time)?;

    let staging = session::reserve_dir(&root.join(format!(".{}.merge", first.name())))?;
    undo.directories.push(staging.clone());
    let merged = Session::create(staging.join("merged"))?;
    let placements = pieces
        .iter()
        .zip(&offsets)
        .map(|(piece, offset)| Placement {
            session: piece.session.clone(),
            offset: *offset,
            samples: piece.samples,
        })
        .collect::<Vec<_>>();
    for source in &sources {
        let written = write_track(&placements, *source, &merged.audio_path(*source))?;
        if written != total {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the merged {source} track holds {written} samples instead of {total}"),
            ));
        }
        join_timeline(pieces, &offsets, *source, &merged)?;
    }
    join_screenshots(pieces, &screenshots, &offsets, &merged)?;

    let title = title.trim();
    let details = match meeting::read_details(&first.session.meeting_path())? {
        Some(details) => Some(MeetingDetails {
            title: title.to_owned(),
            ..details
        }),
        None if title.is_empty() => None,
        None => Some(MeetingDetails::new(
            title.to_owned(),
            Attendees::default(),
            Attendees::default(),
        )),
    };
    if let Some(details) = details {
        meeting::write_details_atomic(&merged.meeting_path(), &details)?;
    }
    let span = join_span(pieces, &offsets);
    if !span.is_whole() {
        process::write_json_atomic(
            &merged.meeting_span_path(),
            &MeetingSpanFile {
                format_version: process::MEETING_SPAN_FORMAT_VERSION,
                span,
            },
        )?;
    }
    let hidden = join_hidden_sources(
        &pieces
            .iter()
            .map(|piece| process::read_hidden_sources(&piece.session))
            .collect::<io::Result<Vec<_>>>()?,
        &offsets,
    );
    if !hidden.is_default() {
        process::write_json_atomic(&merged.hidden_sources_path(), &hidden)?;
    }
    process::write_json_atomic(
        &merged.merged_from_path(),
        &MergedFrom {
            format_version: MERGE_FORMAT_VERSION,
            seam_ms: SEAM_MS,
            order,
            pieces: pieces
                .iter()
                .zip(&offsets)
                .map(|(piece, offset)| MergedPiece {
                    session: piece.name(),
                    title: piece.title.clone(),
                    started_wallclock: piece.manifest.started_wallclock.clone(),
                    offset_ms: samples_to_ms(*offset),
                    length_ms: piece.length_ms(),
                })
                .collect(),
        },
    )?;
    merged.write_wallclock_map(&clock, samples_to_ms(total))?;
    let recorded = |source: AudioSource, stream: fn(&Manifest) -> &StreamInfo| {
        pieces
            .iter()
            .any(|piece| stream(&piece.manifest).enabled && piece.has_track(source))
    };
    merged.write_manifest(&Manifest {
        state: SessionState::Stopped,
        started_wallclock: clock.started().to_owned(),
        mic: StreamInfo {
            enabled: recorded(AudioSource::Mic, |manifest| &manifest.mic),
            ..first.manifest.mic.clone()
        },
        system: StreamInfo {
            enabled: recorded(AudioSource::System, |manifest| &manifest.system),
            ..first.manifest.system.clone()
        },
        ..first.manifest.clone()
    })?;
    split::sync_directory(&merged.dir.join("audio"))?;
    split::sync_directory(&merged.screenshots_dir())?;
    split::sync_directory(&merged.dir)?;

    for source in &sources {
        verify_track(pieces, &offsets, *source, &merged.audio_path(*source))?;
    }

    let target = session::reserve_dir(&root.join(session::session_dir_name(&started.time)))?;
    undo.directories.push(target.clone());
    // Replaces the empty directory that reserved the name.
    fs::rename(&merged.dir, &target)?;
    fs::remove_dir(&staging)?;
    split::sync_directory(&root)?;

    let merged_name = folder_name(&target);
    for (index, piece) in pieces.iter().enumerate() {
        let into_path = piece.session.merged_into_path();
        let mut record = read_merged_into(&piece.session)?.unwrap_or(MergedInto {
            format_version: MERGE_FORMAT_VERSION,
            merges: Vec::new(),
        });
        record.merges.push(MergeRecord {
            session: merged_name.clone(),
            piece: index + 1,
        });
        let recording_failed = |error: io::Error| {
            io::Error::new(
                error.kind(),
                format!(
                    "could not record the merge in “{}”, so nothing was merged: {error}",
                    piece.title
                ),
            )
        };
        undo.keep_record(&into_path).map_err(recording_failed)?;
        process::write_json_atomic(&into_path, &record).map_err(recording_failed)?;
        undo.record_installed();
        split::sync_directory(&piece.session.dir).map_err(recording_failed)?;
    }
    // Every piece is archived or none is, so a failure here leaves all of
    // them as they were.
    let archived = archive::archive_backups(
        &pieces
            .iter()
            .map(|piece| &piece.session)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("could not archive the pieces, so nothing was merged: {error}"),
        )
    })?;
    Ok(MergeOutcome {
        session: target,
        archived,
    })
}

/// A piece's place in a merged recording, in samples.
#[derive(Debug, Clone)]
pub struct Placement {
    pub session: Session,
    pub offset: u64,
    pub samples: u64,
}

/// One track across pieces placed one after another: each piece's audio,
/// silence where it lacks the track or the track ends early, and silence
/// between pieces.
pub struct JoinedTrack {
    placements: Vec<Placement>,
    source: AudioSource,
    position: u64,
    end: u64,
    reader: Option<(usize, Option<TrackReader>)>,
    chunk: Vec<f32>,
}

impl JoinedTrack {
    /// From sample `start` of the merged recording on.
    pub fn new(placements: Vec<Placement>, source: AudioSource, start: u64) -> Self {
        let end = placements
            .last()
            .map_or(0, |last| last.offset.saturating_add(last.samples));
        Self {
            placements,
            source,
            position: start.min(end),
            end,
            reader: None,
            chunk: Vec::new(),
        }
    }

    /// Replace `out` with up to `max` next samples; empty at the end.
    pub fn read(&mut self, out: &mut Vec<f32>, max: usize) -> io::Result<()> {
        out.clear();
        while out.len() < max && self.position < self.end {
            let wanted = (max - out.len()) as u64;
            let index = self
                .placements
                .iter()
                .rposition(|placement| placement.offset <= self.position)
                .unwrap_or(0);
            let placement = &self.placements[index];
            let piece_end = placement.offset.saturating_add(placement.samples);
            if self.position < placement.offset || self.position >= piece_end {
                let next = self
                    .placements
                    .get(index + 1)
                    .map_or(self.end, |next| next.offset)
                    .max(self.position + 1);
                let count = wanted.min(next - self.position);
                out.resize(out.len() + count as usize, 0.0);
                self.position += count;
                continue;
            }
            if self.reader.as_ref().map(|(open, _)| *open) != Some(index) {
                let reader = TrackReader::open(
                    &placement.session,
                    self.source,
                    self.position - placement.offset,
                )?;
                self.reader = Some((index, reader));
            }
            let count = wanted.min(piece_end - self.position);
            let reader = &mut self.reader.as_mut().expect("reader just opened").1;
            match reader {
                Some(track) => {
                    track.read(&mut self.chunk, count as usize)?;
                    if self.chunk.is_empty() {
                        *reader = None;
                        continue;
                    }
                    out.extend_from_slice(&self.chunk);
                    self.position += self.chunk.len() as u64;
                }
                None => {
                    out.resize(out.len() + count as usize, 0.0);
                    self.position += count;
                }
            }
        }
        Ok(())
    }
}

fn write_track(placements: &[Placement], source: AudioSource, path: &Path) -> io::Result<u64> {
    let mut reader = JoinedTrack::new(placements.to_vec(), source, 0);
    let mut writer = BufWriter::new(session::create_private_file(path)?);
    let mut chunk = Vec::with_capacity(CHUNK_SAMPLES);
    let mut bytes = Vec::with_capacity(CHUNK_SAMPLES * 4);
    loop {
        reader.read(&mut chunk, CHUNK_SAMPLES)?;
        if chunk.is_empty() {
            break;
        }
        bytes.clear();
        bytes.extend(chunk.iter().flat_map(|sample| sample.to_le_bytes()));
        writer.write_all(&bytes)?;
    }
    writer
        .into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    Ok(fs::metadata(path)?.len() / 4)
}

/// Read the merged track back from disk and compare it, a chunk at a time,
/// with each piece's own track at its offset; everything else must be
/// silence.
fn verify_track(
    pieces: &[Piece],
    offsets: &[u64],
    source: AudioSource,
    path: &Path,
) -> io::Result<()> {
    let mismatch = |piece: &Piece, at: u64, what: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "the merged {source} track does not match “{}” {what} at {}",
                piece.title,
                split::clock(samples_to_ms(at))
            ),
        )
    };
    let mut file = io::BufReader::with_capacity(1 << 16, File::open(path)?);
    let last = pieces.len() - 1;
    let total = offsets[last] + pieces[last].samples;
    let stored = file.get_ref().metadata()?.len();
    if stored != total * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "the merged {source} track holds {} samples instead of {total}",
                stored / 4
            ),
        ));
    }
    let mut chunk = Vec::with_capacity(CHUNK_SAMPLES);
    let mut expected = Vec::with_capacity(CHUNK_SAMPLES * 4);
    let mut actual = vec![0u8; CHUNK_SAMPLES * 4];
    for (index, piece) in pieces.iter().enumerate() {
        let recorded = piece
            .tracks
            .iter()
            .find(|(candidate, _)| *candidate == source)
            .map_or(0, |(_, samples)| *samples);
        let mut position = 0u64;
        if let Some(mut reader) = TrackReader::open(&piece.session, source, 0)? {
            loop {
                reader.read(&mut chunk, CHUNK_SAMPLES)?;
                if chunk.is_empty() {
                    break;
                }
                if position + chunk.len() as u64 > piece.samples {
                    return Err(mismatch(piece, position, "in length"));
                }
                expected.clear();
                expected.extend(chunk.iter().flat_map(|sample| sample.to_le_bytes()));
                let actual = &mut actual[..expected.len()];
                file.read_exact(actual)?;
                if *actual != *expected {
                    return Err(mismatch(piece, offsets[index] + position, "audio"));
                }
                position += chunk.len() as u64;
            }
        }
        if position != recorded {
            return Err(mismatch(piece, offsets[index] + position, "in length"));
        }
        let silence_end = if index == last {
            piece.samples
        } else {
            offsets[index + 1] - offsets[index]
        };
        while position < silence_end {
            let count = (silence_end - position).min(CHUNK_SAMPLES as u64) as usize;
            let actual = &mut actual[..count * 4];
            file.read_exact(actual)?;
            if actual.iter().any(|byte| *byte != 0) {
                return Err(mismatch(
                    piece,
                    offsets[index] + position,
                    "in the silence after it",
                ));
            }
            position += count as u64;
        }
    }
    Ok(())
}

/// Each piece's timing events at its offset. Processing never reads them;
/// they are a capture log, so each piece's own start and stop stay as they
/// were recorded.
fn join_timeline(
    pieces: &[Piece],
    offsets: &[u64],
    source: AudioSource,
    merged: &Session,
) -> io::Result<()> {
    let mut joined = Vec::new();
    for (piece, offset) in pieces.iter().zip(offsets) {
        let events: Vec<TimelineEvent> = match jsonl::read_all(&piece.session.timeline_path(source))
        {
            Ok(events) => events,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        joined.extend(events.into_iter().map(|event| {
            let sample = split::event_sample(&event);
            split::at_sample(event, sample + offset)
        }));
    }
    if joined.is_empty() {
        return Ok(());
    }
    jsonl::write_all_atomic(&merged.timeline_path(source), &joined)
}

/// Every piece's screenshots, renamed after their time in the merged
/// recording, with the same rules as a split.
fn join_screenshots(
    pieces: &[Piece],
    screenshots: &[Vec<ScreenshotEntry>],
    offsets: &[u64],
    merged: &Session,
) -> io::Result<()> {
    let mut joined = Vec::new();
    for ((piece, entries), offset) in pieces.iter().zip(screenshots).zip(offsets) {
        for entry in entries {
            let Some((name, mut source)) = split::open_screenshot(&piece.session, &entry.file)
            else {
                eprintln!(
                    "warning: leaving out screenshot {:?} of {}: not a regular file in the session's screenshots folder",
                    entry.file,
                    piece.name()
                );
                continue;
            };
            let time_ms = entry.time_ms + samples_to_ms(*offset);
            let target = split::free_screenshot_name(&merged.screenshots_dir(), &name, time_ms);
            split::copy_private(&mut source, &target)?;
            let name = target.file_name().unwrap_or_default().to_string_lossy();
            joined.push(ScreenshotEntry {
                time_ms,
                file: format!("screenshots/{name}"),
                original: entry.original.clone(),
            });
        }
    }
    if joined.is_empty() {
        return Ok(());
    }
    jsonl::write_all_atomic(&merged.screenshots_index_path(), &joined)
}

/// The first piece's start mark and the last piece's end mark; marks inside
/// the merged recording would cut it short.
fn join_span(pieces: &[Piece], offsets: &[u64]) -> MeetingSpan {
    let read = |piece: &Piece| process::read_meeting_span(&piece.session).unwrap_or_default();
    let first = &pieces[0];
    let last = pieces.len() - 1;
    let span = MeetingSpan {
        start_ms: read(first)
            .start_ms
            .filter(|start| *start < first.length_ms()),
        end_ms: read(&pieces[last])
            .end_ms
            .filter(|end| *end > 0)
            .map(|end| end.min(pieces[last].length_ms()) + samples_to_ms(offsets[last])),
    };
    if span.is_valid() {
        span
    } else {
        MeetingSpan::default()
    }
}

/// The first piece decides which sources are hidden and whose lines stay,
/// since its meeting details are the ones kept too. Lines kept by hand stay
/// kept in every piece, moved to the merged clock.
fn join_hidden_sources(hidden: &[HiddenSources], offsets: &[u64]) -> HiddenSources {
    let keep = hidden
        .iter()
        .zip(offsets)
        .flat_map(|(hidden, offset)| {
            let offset_ms = samples_to_ms(*offset);
            hidden.keep.iter().map(move |range| KeptRange {
                start_ms: range.start_ms + offset_ms,
                end_ms: range.end_ms + offset_ms,
                ..*range
            })
        })
        .collect();
    process::normalize_hidden_sources(HiddenSources {
        keep,
        ..hidden[0].clone()
    })
}

/// "09:58".
pub fn time_label(clock: &WallClock) -> String {
    format!("{:02}:{:02}", clock.time.hour, clock.time.minute)
}

/// "2026-09-14".
pub fn date_label(clock: &WallClock) -> String {
    format!(
        "{:04}-{:02}-{:02}",
        clock.time.year, clock.time.month, clock.time.day
    )
}

/// The jump in time of day at a seam: "09:58 → 10:21", with the date of the
/// later side when the day changes.
pub fn seam_label(before: Option<WallClock>, after: Option<WallClock>) -> String {
    match (before, after) {
        (Some(before), Some(after)) if date_label(&before) == date_label(&after) => {
            format!("{} → {}", time_label(&before), time_label(&after))
        }
        (Some(before), Some(after)) => format!(
            "{} → {} {}",
            time_label(&before),
            date_label(&after),
            time_label(&after)
        ),
        _ => "seam".into(),
    }
}

fn folder_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FORMAT_VERSION;

    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "singstone-merge-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create fixture root");
        root
    }

    fn samples(count: u64, offset: f32) -> Vec<u8> {
        (0..count)
            .flat_map(|index| (offset + index as f32 * 1e-6).to_le_bytes())
            .collect()
    }

    /// A stopped, unprocessed session with the given tracks, in samples.
    fn recorded(root: &Path, name: &str, started: &str, mic: u64, system: u64) -> Session {
        let session = Session::create(root.join(name)).expect("create");
        let stream = |enabled| StreamInfo {
            enabled,
            pipewire_node: None,
            error: None,
        };
        session
            .write_manifest(&Manifest {
                format_version: FORMAT_VERSION,
                state: SessionState::Stopped,
                started_wallclock: started.into(),
                sample_rate: SAMPLE_RATE,
                channels: 1,
                sample_format: "f32le".into(),
                mic: stream(mic > 0),
                system: stream(system > 0),
                local_speaker: "Me".into(),
                screenshot_dir: None,
            })
            .expect("write manifest");
        for (source, count, value) in [
            (AudioSource::Mic, mic, 0.1),
            (AudioSource::System, system, 0.2),
        ] {
            if count > 0 {
                fs::write(session.audio_path(source), samples(count, value)).expect("track");
            }
        }
        session
    }

    fn open(sessions: &[&Session]) -> Vec<Piece> {
        sessions
            .iter()
            .map(|session| Piece::open(&session.dir).expect("piece"))
            .collect()
    }

    fn seconds(count: u64) -> u64 {
        count * SAMPLE_RATE as u64
    }

    #[test]
    fn tracks_are_joined_at_the_same_offsets_with_silent_seams() {
        let root = fixture_root("tracks");
        let first = recorded(
            &root,
            "session-20260914-093000",
            "2026-09-14T09:30:00+02:00",
            seconds(3),
            seconds(2),
        );
        let second = recorded(
            &root,
            "session-20260914-102100",
            "2026-09-14T10:21:00+02:00",
            seconds(1),
            0,
        );
        let outcome = merge_sessions(
            &open(&[&first, &second]),
            "Weekly sync",
            MergeOrder::Recorded,
        )
        .expect("merge");
        assert_eq!(
            folder_name(&outcome.session),
            "session-20260914-093000-1",
            "named after the first piece's start, next to it"
        );
        let merged = Session::open(&outcome.session).expect("merged");
        let zeros = |count: u64| vec![0u8; count as usize * 4];
        let mut mic = samples(seconds(3), 0.1);
        mic.extend(zeros(SEAM_SAMPLES));
        mic.extend(samples(seconds(1), 0.1));
        assert_eq!(
            fs::read(merged.audio_path(AudioSource::Mic)).expect("mic"),
            mic
        );
        let mut system = samples(seconds(2), 0.2);
        system.extend(zeros(seconds(1) + SEAM_SAMPLES + seconds(1)));
        assert_eq!(
            fs::read(merged.audio_path(AudioSource::System)).expect("system"),
            system
        );
        let manifest = merged.read_manifest().expect("manifest");
        assert!(manifest.mic.enabled && manifest.system.enabled);
        assert_eq!(manifest.started_wallclock, "2026-09-14T09:30:00+02:00");
        assert_eq!(
            meeting::read_details(&merged.meeting_path())
                .expect("details")
                .expect("title")
                .title,
            "Weekly sync"
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn pieces_are_kept_as_archived_backups_that_point_to_the_merge() {
        let root = fixture_root("provenance");
        let first = recorded(
            &root,
            "session-20260914-093000",
            "2026-09-14T09:30:00+02:00",
            seconds(2),
            0,
        );
        let second = recorded(
            &root,
            "session-20260914-102100",
            "2026-09-14T10:21:00+02:00",
            seconds(1),
            0,
        );
        let outcome = merge_sessions(&open(&[&second, &first]), "Weekly sync", MergeOrder::Chosen)
            .expect("merge");
        let merged = Session::open(&outcome.session).expect("merged");
        let from = read_merged_from(&merged)
            .expect("read")
            .expect("provenance");
        assert_eq!(from.order, MergeOrder::Chosen);
        assert_eq!(
            from.pieces
                .iter()
                .map(|piece| (piece.session.as_str(), piece.offset_ms, piece.length_ms))
                .collect::<Vec<_>>(),
            [
                ("session-20260914-102100", 0, 1_000),
                ("session-20260914-093000", 2_000, 2_000)
            ]
        );
        for (piece, place) in [(&second, 1), (&first, 2)] {
            assert!(piece.is_archived());
            assert_eq!(
                read_merged_into(piece)
                    .expect("read")
                    .expect("record")
                    .merges,
                [MergeRecord {
                    session: folder_name(&outcome.session),
                    piece: place,
                }]
            );
        }
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn default_order_is_by_start_time() {
        let root = fixture_root("order");
        let late = recorded(
            &root,
            "session-20260915-140000",
            "2026-09-15T14:00:00+02:00",
            seconds(1),
            0,
        );
        let early = recorded(
            &root,
            "session-20260914-093000",
            "2026-09-14T09:30:00.500+02:00",
            seconds(1),
            0,
        );
        let mut pieces = open(&[&late, &early]);
        sort_by_start(&mut pieces);
        assert_eq!(
            pieces.iter().map(Piece::name).collect::<Vec<_>>(),
            ["session-20260914-093000", "session-20260915-140000"]
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn read_back_catches_a_changed_byte_or_noise_in_a_seam() {
        let root = fixture_root("verify");
        let first = recorded(&root, "a", "2026-09-14T09:30:00+02:00", seconds(1), 0);
        let second = recorded(&root, "b", "2026-09-14T10:00:00+02:00", seconds(1), 0);
        let pieces = open(&[&first, &second]);
        let offsets = offsets(&[seconds(1), seconds(1)]);
        let placements = pieces
            .iter()
            .zip(&offsets)
            .map(|(piece, offset)| Placement {
                session: piece.session.clone(),
                offset: *offset,
                samples: piece.samples,
            })
            .collect::<Vec<_>>();
        let path = root.join("merged.f32le");
        write_track(&placements, AudioSource::Mic, &path).expect("write");
        verify_track(&pieces, &offsets, AudioSource::Mic, &path).expect("intact");
        let intact = fs::read(&path).expect("read");
        for at in [4 * 100 + 1, 4 * (seconds(1) as usize + 10)] {
            let mut changed = intact.clone();
            changed[at] ^= 1;
            fs::write(&path, changed).expect("corrupt");
            assert_eq!(
                verify_track(&pieces, &offsets, AudioSource::Mic, &path)
                    .expect_err("mismatch")
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
        fs::remove_dir_all(root).expect("remove fixture");
    }

    /// Every entry under `dir` but temporaries: its contents, inode and
    /// change time, to tell a record put back from one written again.
    fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>, u64, i64)> {
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::fs::MetadataExt;
        let mut entries = Vec::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(next) = pending.pop() {
            for entry in fs::read_dir(&next).expect("read dir") {
                let path = entry.expect("entry").path();
                if path.extension().is_some_and(|extension| extension == "tmp") {
                    continue;
                }
                let metadata = fs::symlink_metadata(&path).expect("metadata");
                let contents = if metadata.is_dir() {
                    pending.push(path.clone());
                    b"dir".to_vec()
                } else if metadata.is_symlink() {
                    fs::read_link(&path)
                        .expect("link")
                        .into_os_string()
                        .into_vec()
                } else {
                    fs::read(&path).expect("file")
                };
                let changed = if metadata.is_file() {
                    metadata.mtime_nsec()
                } else {
                    0
                };
                entries.push((path, contents, metadata.ino(), changed));
            }
        }
        entries.sort();
        entries
    }

    /// Two pieces that were merged before, so each has a record to keep.
    fn merged_before(root: &Path) -> [Session; 2] {
        let pieces = [
            recorded(
                root,
                "session-20260914-093000",
                "2026-09-14T09:30:00+02:00",
                seconds(1),
                seconds(1),
            ),
            recorded(
                root,
                "session-20260914-102100",
                "2026-09-14T10:21:00+02:00",
                seconds(1),
                0,
            ),
        ];
        for piece in &pieces {
            process::write_json_atomic(
                &piece.merged_into_path(),
                &MergedInto {
                    format_version: MERGE_FORMAT_VERSION,
                    merges: vec![MergeRecord {
                        session: "session-20260901-090000".into(),
                        piece: 1,
                    }],
                },
            )
            .expect("record");
        }
        pieces
    }

    #[test]
    fn a_failed_merge_leaves_the_pieces_untouched() {
        let root = fixture_root("failed");
        let [first, second] = merged_before(&root);
        // A planted link must not lead the first record into its audio, and
        // the second record cannot be written at all.
        std::os::unix::fs::symlink(
            first.audio_path(AudioSource::Mic),
            jsonl::tmp_path(&first.merged_into_path()),
        )
        .expect("plant a link");
        fs::create_dir(jsonl::tmp_path(&second.merged_into_path())).expect("block the record");
        // An interrupted run's copy of an older record must survive.
        let mut left = first.merged_into_path().into_os_string();
        left.push(".undo");
        fs::write(left, b"older record").expect("leave an undo copy");
        let entries = fs::read_dir(&root).expect("root").count();
        let before = [&first, &second].map(|piece| snapshot(&piece.dir));

        let error = merge_sessions(&open(&[&first, &second]), "x", MergeOrder::Recorded)
            .expect_err("blocked");
        assert!(
            error
                .to_string()
                .starts_with("could not record the merge in"),
            "{error}"
        );
        assert_eq!(fs::read_dir(&root).expect("root").count(), entries);
        assert_eq!([&first, &second].map(|piece| snapshot(&piece.dir)), before);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn a_failed_archive_leaves_every_piece_as_it_was() {
        let root = fixture_root("archive");
        let [first, second] = merged_before(&root);
        // A copy left by an earlier archive of the second piece is in the way.
        fs::create_dir_all(second.archived_audio_path(AudioSource::Mic).join("full"))
            .expect("block the copy");
        let entries = fs::read_dir(&root).expect("root").count();
        let before = [&first, &second].map(|piece| snapshot(&piece.dir));

        let error = merge_sessions(&open(&[&first, &second]), "x", MergeOrder::Recorded)
            .expect_err("blocked");
        assert!(error.to_string().contains("could not archive"), "{error}");
        assert_eq!(fs::read_dir(&root).expect("root").count(), entries);
        assert_eq!([&first, &second].map(|piece| snapshot(&piece.dir)), before);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn processed_archived_overlapping_or_single_pieces_are_refused() {
        let root = fixture_root("refused");
        let first = recorded(&root, "a", "2026-09-14T09:30:00+02:00", seconds(60), 0);
        let overlapping = recorded(&root, "b", "2026-09-14T09:30:59+02:00", seconds(1), 0);
        let after = recorded(&root, "c", "2026-09-14T09:31:00+02:00", seconds(1), 0);
        let refused = |pieces: &[&Session]| check(&open(pieces)).expect_err("refused").to_string();
        assert!(check(&open(&[&first, &after])).is_ok());
        assert!(refused(&[&first]).contains("at least two"));
        assert!(refused(&[&first, &overlapping]).contains("same moment recorded twice"));
        fs::write(after.transcript_path(), b"").expect("process");
        assert!(refused(&[&first, &after]).contains("already processed"));
        fs::remove_file(after.transcript_path()).expect("unprocess");
        archive::archive_backups(&[&after]).expect("archive");
        assert!(refused(&[&first, &after]).contains("archived"));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn screenshots_timeline_marks_and_kept_lines_move_to_the_merged_clock() {
        let root = fixture_root("rebase");
        let first = recorded(&root, "a", "2026-09-14T09:30:00+02:00", seconds(3), 0);
        let second = recorded(&root, "b", "2026-09-14T10:21:00+02:00", seconds(2), 0);
        for (session, start_ms, end_ms) in [(&first, 500, 2_500), (&second, 200, 1_500)] {
            let name = format!("{start_ms:09}-shot.png");
            fs::write(session.screenshots_dir().join(&name), "png").expect("screenshot");
            jsonl::write_all_atomic(
                &session.screenshots_index_path(),
                &[ScreenshotEntry {
                    time_ms: start_ms,
                    file: format!("screenshots/{name}"),
                    original: format!("/shots/{name}"),
                }],
            )
            .expect("index");
            jsonl::write_all_atomic(
                &session.timeline_path(AudioSource::Mic),
                &[
                    TimelineEvent::Start {
                        sample: 0,
                        time_ms: 0,
                    },
                    TimelineEvent::Stop {
                        sample: end_ms * SAMPLES_PER_MS,
                        time_ms: end_ms,
                    },
                ],
            )
            .expect("timeline");
            process::write_json_atomic(
                &session.meeting_span_path(),
                &MeetingSpanFile {
                    format_version: process::MEETING_SPAN_FORMAT_VERSION,
                    span: MeetingSpan {
                        start_ms: Some(start_ms),
                        end_ms: Some(end_ms),
                    },
                },
            )
            .expect("span");
            process::write_json_atomic(
                &session.hidden_sources_path(),
                &HiddenSources {
                    hidden: vec![AudioSource::Mic],
                    keep_local_speaker: start_ms == 500,
                    keep: vec![KeptRange {
                        source: AudioSource::Mic,
                        start_ms,
                        end_ms,
                    }],
                    ..HiddenSources::default()
                },
            )
            .expect("hidden");
        }
        let outcome =
            merge_sessions(&open(&[&first, &second]), "", MergeOrder::Recorded).expect("merge");
        let merged = Session::open(&outcome.session).expect("merged");

        let shots: Vec<ScreenshotEntry> =
            jsonl::read_all(&merged.screenshots_index_path()).expect("screenshots");
        assert_eq!(
            shots
                .iter()
                .map(|shot| (shot.time_ms, shot.file.as_str()))
                .collect::<Vec<_>>(),
            [
                (500, "screenshots/000000500-shot.png"),
                (4_200, "screenshots/000004200-shot.png")
            ]
        );
        assert!(merged.dir.join(&shots[1].file).is_file());
        let timeline: Vec<TimelineEvent> =
            jsonl::read_all(&merged.timeline_path(AudioSource::Mic)).expect("timeline");
        assert_eq!(
            timeline.iter().map(split::event_sample).collect::<Vec<_>>(),
            [0, 40_000, 64_000, 88_000]
        );
        assert_eq!(
            process::read_meeting_span(&merged).expect("span"),
            MeetingSpan {
                start_ms: Some(500),
                end_ms: Some(5_500),
            }
        );
        let hidden = process::read_hidden_sources(&merged).expect("hidden");
        assert!(hidden.keep_local_speaker);
        assert_eq!(
            hidden
                .keep
                .iter()
                .map(|range| (range.start_ms, range.end_ms))
                .collect::<Vec<_>>(),
            [(500, 2_500), (4_200, 5_500)]
        );
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn time_of_day_survives_merging_splitting_and_merging_again() {
        let root = fixture_root("clock");
        let first = recorded(
            &root,
            "session-20260914-093000",
            "2026-09-14T09:30:00+02:00",
            seconds(60),
            0,
        );
        let second = recorded(
            &root,
            "session-20260914-102100",
            "2026-09-14T10:21:00+02:00",
            seconds(60),
            0,
        );
        let merged = Session::open(
            merge_sessions(&open(&[&first, &second]), "", MergeOrder::Recorded)
                .expect("merge")
                .session,
        )
        .expect("merged");
        let at = |session: &Session, offset_ms| {
            session
                .wallclock_at(offset_ms)
                .expect("time of day")
                .rfc3339()
        };
        assert_eq!(at(&merged, 30_000), "2026-09-14T09:30:30+02:00");
        assert_eq!(at(&merged, 60_500), "2026-09-14T09:31:00+02:00", "seam");
        assert_eq!(at(&merged, 71_000), "2026-09-14T10:21:10+02:00");

        // A cut 10 s into the second piece: the first part keeps the seam.
        let parts = split::split_session(&merged, 71_000, ["a", "b"])
            .expect("split")
            .parts
            .map(|part| Session::open(part).expect("part"));
        assert_eq!(
            folder_name(&parts[1].dir),
            "session-20260914-102110",
            "named after its real start"
        );
        assert_eq!(at(&parts[1], 0), "2026-09-14T10:21:10+02:00");
        assert!(!parts[1].wallclock_path().exists());
        assert_eq!(at(&parts[0], 70_000), "2026-09-14T10:21:09+02:00");

        // The first part, merged with a piece from the next day, keeps both
        // of its seams.
        let late = recorded(
            &root,
            "session-20260915-140000",
            "2026-09-15T14:00:00+02:00",
            seconds(5),
            0,
        );
        let again = Session::open(
            merge_sessions(&open(&[&parts[0], &late]), "", MergeOrder::Recorded)
                .expect("merge again")
                .session,
        )
        .expect("merged again");
        let manifest = again.read_manifest().expect("manifest");
        let seams = again.wallclock_map(&manifest).expect("map").seams();
        assert_eq!(
            seams
                .iter()
                .map(|seam| (seam.end_ms, seam_label(seam.before, seam.after)))
                .collect::<Vec<_>>(),
            [
                (61_000, "09:31 → 10:21".to_owned()),
                (72_000, "10:21 → 2026-09-15 14:00".to_owned())
            ]
        );
        assert_eq!(at(&again, 73_000), "2026-09-15T14:00:01+02:00");
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn a_cut_at_or_inside_a_seam_keeps_the_time_of_day_still_there() {
        for cut_ms in [3_000, 3_500, 4_000] {
            let root = fixture_root("seam-cut");
            let first = recorded(&root, "a", "2026-09-14T09:30:00+02:00", seconds(3), 0);
            let second = recorded(&root, "b", "2026-09-14T10:00:00+02:00", seconds(3), 0);
            let merged = Session::open(
                merge_sessions(&open(&[&first, &second]), "", MergeOrder::Recorded)
                    .expect("merge")
                    .session,
            )
            .expect("merged");
            let [before, after] = split::split_session(&merged, cut_ms, ["a", "b"])
                .expect("split")
                .parts
                .map(|part| Session::open(part).expect("part"));
            let at = |session: &Session, offset_ms| {
                session
                    .wallclock_at(offset_ms)
                    .expect("time of day")
                    .rfc3339()
            };
            assert_eq!(at(&before, cut_ms), "2026-09-14T09:30:03+02:00", "{cut_ms}");
            assert_eq!(
                at(&after, 0),
                if cut_ms < 4_000 {
                    "2026-09-14T09:30:03+02:00"
                } else {
                    "2026-09-14T10:00:00+02:00"
                },
                "{cut_ms}"
            );

            // A recording that starts where the first piece ended does not
            // overlap the silence after it.
            let next = recorded(&root, "c", "2026-09-14T09:30:03+02:00", seconds(1), 0);
            let again = Session::open(
                merge_sessions(&open(&[&before, &next]), "", MergeOrder::Recorded)
                    .expect("merge again")
                    .session,
            )
            .expect("merged again");
            assert_eq!(at(&again, cut_ms + 500), "2026-09-14T09:30:03+02:00");
            assert_eq!(
                at(&again, cut_ms + SEAM_MS + 500),
                "2026-09-14T09:30:03.500+02:00"
            );
            fs::remove_dir_all(root).expect("remove fixture");
        }
    }
}
