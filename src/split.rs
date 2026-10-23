//! Cutting a recording that has not been processed yet into two new
//! sessions. The original stays as an archived backup.

use crate::audio::archive::{self, ArchiveSummary};
use crate::audio::track::TrackReader;
use crate::cli::SplitArgs;
use crate::format::jsonl;
use crate::meeting::{self, Attendees, MeetingDetails};
use crate::merge::process;
use crate::screenshot::inotify::screenshot_file_name;
use crate::session::{self, Session};
use crate::types::{
    AudioSource, Manifest, MeetingSpan, MeetingSpanFile, SAMPLE_RATE, ScreenshotEntry,
    SessionState, StreamInfo, TimelineEvent, samples_to_ms,
};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

pub const SPLIT_FORMAT_VERSION: u32 = 1;
/// Neither part may be shorter than this.
pub const MIN_PART_MS: u64 = 1_000;
const SAMPLES_PER_MS: u64 = SAMPLE_RATE as u64 / 1000;
const CHUNK_SAMPLES: usize = 16_384;
/// Free space kept in reserve beyond what the split itself writes.
const SPACE_MARGIN: u64 = 64 << 20;

/// `split-from.json` in each new session: where its audio came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitFrom {
    pub format_version: u32,
    /// Folder name of the original session, next to this one.
    pub session: String,
    /// The original's title when it was split.
    pub title: String,
    /// Where in the original the cut was.
    pub at_ms: u64,
    /// 1 for the part before the cut, 2 for the part after it.
    pub part: u8,
}

/// `split-into.json` in the original: every split made from it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitInto {
    pub format_version: u32,
    pub splits: Vec<SplitRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitRecord {
    pub at_ms: u64,
    /// Folder names of the two new sessions.
    pub parts: [String; 2],
}

#[derive(Debug)]
pub struct SplitOutcome {
    pub parts: [PathBuf; 2],
    pub archived: ArchiveSummary,
}

pub fn run(args: SplitArgs) -> Result<(), Box<dyn std::error::Error>> {
    let session = Session::open(&args.session)?;
    let at_ms = parse_time(&args.at)?;
    let [first, second] = default_titles(&session);
    let titles = [
        args.first_title.unwrap_or(first),
        args.second_title.unwrap_or(second),
    ];
    let outcome = split_session(&session, at_ms, [&titles[0], &titles[1]])?;
    eprintln!("split at {}:", clock(at_ms));
    for part in &outcome.parts {
        println!("{}", part.display());
    }
    let name = session.dir.file_name().map_or_else(
        || session.dir.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    if outcome.archived.tracks.is_empty() {
        eprintln!("kept {name} as a backup; its audio was already archived");
    } else {
        eprintln!(
            "kept {name} as an archived backup: {} -> {}",
            archive::format_size(outcome.archived.bytes_before),
            archive::format_size(outcome.archived.bytes_after)
        );
    }
    Ok(())
}

/// "<title>" and "<title> (2)".
pub fn default_titles(session: &Session) -> [String; 2] {
    let title = session.title();
    [title.clone(), format!("{title} (2)")]
}

/// Why `session` cannot be split at any point.
pub fn refusal(session: &Session, manifest: &Manifest) -> Option<&'static str> {
    if manifest.state == SessionState::Recording {
        Some("the session is still being recorded")
    } else if session.transcript_path().is_file() {
        Some(
            "the session is already processed; splitting is for recordings that have not been processed yet",
        )
    } else {
        None
    }
}

/// `41:12`, `1:07:20`, `2472` or `2472.5` (seconds), as milliseconds.
pub fn parse_time(value: &str) -> Result<u64, String> {
    let invalid = || format!("invalid time {value:?}: use seconds, M:SS or H:MM:SS");
    let digits = |field: &str| !field.is_empty() && field.bytes().all(|byte| byte.is_ascii_digit());
    let fields = value.trim().split(':').collect::<Vec<_>>();
    let (last, leading) = fields.split_last().ok_or_else(invalid)?;
    let (seconds, fraction) = last.split_once('.').unwrap_or((last, ""));
    if leading.len() > 2
        || !leading.iter().all(|field| digits(field))
        || !digits(seconds)
        || !(fraction.is_empty() || digits(fraction))
        || (!leading.is_empty() && seconds.len() != 2)
        || (leading.len() == 2 && leading[1].len() != 2)
    {
        return Err(invalid());
    }
    let number = |field: &str| field.parse::<u64>().map_err(|_| invalid());
    let mut total = 0u64;
    for (index, field) in leading.iter().enumerate() {
        let value = number(field)?;
        if index > 0 && value >= 60 {
            return Err(invalid());
        }
        total = total
            .checked_mul(60)
            .and_then(|total| total.checked_add(value))
            .ok_or_else(invalid)?;
    }
    let seconds = number(seconds)?;
    if !leading.is_empty() && seconds >= 60 {
        return Err(invalid());
    }
    let fraction_ms = format!("{fraction:0<3}")[..3]
        .parse::<u64>()
        .map_err(|_| invalid())?;
    total
        .checked_mul(60)
        .and_then(|total| total.checked_add(seconds))
        .and_then(|total| total.checked_mul(1000))
        .and_then(|total| total.checked_add(fraction_ms))
        .ok_or_else(invalid)
}

pub fn read_split_from(session: &Session) -> io::Result<Option<SplitFrom>> {
    read_optional(&session.split_from_path())
}

pub fn read_split_into(session: &Session) -> io::Result<Option<SplitInto>> {
    read_optional(&session.split_into_path())
}

fn read_optional<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<Option<T>> {
    match process::read_json(path) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Write the audio before and after `at_ms` into two new sessions next to
/// `session`, then archive `session` as their backup. Nothing changes unless
/// every step succeeds.
pub fn split_session(session: &Session, at_ms: u64, titles: [&str; 2]) -> io::Result<SplitOutcome> {
    let manifest = session.read_manifest()?;
    if let Some(reason) = refusal(session, &manifest) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, reason));
    }
    let length_ms = process::recorded_duration_ms(session, &manifest);
    if at_ms < MIN_PART_MS || at_ms.saturating_add(MIN_PART_MS) > length_ms {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "the cut at {} must leave at least {} s on each side of the {} recording",
                clock(at_ms),
                MIN_PART_MS / 1000,
                clock(length_ms)
            ),
        ));
    }
    let tracks = [AudioSource::Mic, AudioSource::System]
        .into_iter()
        .filter_map(|source| {
            let path = session.stored_audio_path(source);
            path.is_file().then_some((source, path))
        })
        .map(|(source, path)| Ok((source, process::stored_sample_count(&path)?, path)))
        .collect::<io::Result<Vec<_>>>()?;
    let raw_bytes = tracks
        .iter()
        .filter(|(_, _, path)| !archive::is_archived(path))
        .map(|(_, samples, _)| samples * 4)
        .sum::<u64>();
    if raw_bytes > 0 {
        archive::ensure_decoder()?;
    }
    let screenshots: Vec<ScreenshotEntry> = match jsonl::read_all(&session.screenshots_index_path())
    {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error),
    };
    let screenshot_bytes = screenshots
        .iter()
        .filter_map(|entry| open_screenshot(session, &entry.file)?.1.metadata().ok())
        .map(|metadata| metadata.len())
        .sum::<u64>();

    let root = match session.dir.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    // Both parts in raw float, the screenshots again, and the original's
    // FLAC copy, which is at most half its raw size.
    let needed = tracks
        .iter()
        .map(|(_, samples, _)| samples * 4)
        .sum::<u64>()
        + screenshot_bytes
        + raw_bytes / 2
        + SPACE_MARGIN;
    let stats = rustix::fs::statvfs(&root).map_err(io::Error::from)?;
    let free = stats.f_bavail.saturating_mul(stats.f_frsize);
    if free < needed {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "not enough free disk space: splitting needs about {}, {} is free",
                archive::format_size(needed),
                archive::format_size(free)
            ),
        ));
    }

    let started = [
        session::wallclock_after(&manifest.started_wallclock, 0),
        session::wallclock_after(&manifest.started_wallclock, at_ms),
    ];
    let [Some(first_start), Some(second_start)] = started else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "the session start {:?} is not a valid time",
                manifest.started_wallclock
            ),
        ));
    };
    let original_name = session
        .dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut undo = Undo::default();
    let staging = session::reserve_dir(&root.join(format!(".{original_name}.split")))?;
    undo.directories.push(staging.clone());
    let parts = [
        Session::create(staging.join("part-1"))?,
        Session::create(staging.join("part-2"))?,
    ];
    let cut = at_ms * SAMPLES_PER_MS;
    for (source, total, _) in &tracks {
        let written = split_track(session, *source, cut, &parts)?;
        let expected_first = cut.min(*total);
        if written != [expected_first, total - expected_first] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "the {source} track was split into {} and {} samples instead of {} and {}",
                    written[0],
                    written[1],
                    expected_first,
                    total - expected_first
                ),
            ));
        }
        split_timeline(session, *source, cut, &parts, written)?;
    }
    split_screenshots(session, &screenshots, at_ms, &parts)?;
    let details = meeting::read_details(&session.meeting_path())?;
    let span = process::read_meeting_span(session).unwrap_or_default();
    let spans = split_span(span, at_ms);
    let original_title = session.title();
    for (index, part) in parts.iter().enumerate() {
        let title = titles[index].trim();
        let part_details = match &details {
            Some(details) => Some(MeetingDetails {
                title: title.to_owned(),
                ..details.clone()
            }),
            None if title.is_empty() => None,
            None => Some(MeetingDetails::new(
                title.to_owned(),
                Attendees::default(),
                Attendees::default(),
            )),
        };
        if let Some(part_details) = part_details {
            meeting::write_details_atomic(&part.meeting_path(), &part_details)?;
        }
        if session.hidden_sources_path().is_file() {
            copy_private(
                &mut File::open(session.hidden_sources_path())?,
                &part.hidden_sources_path(),
            )?;
        }
        if !spans[index].is_whole() {
            process::write_json_atomic(
                &part.meeting_span_path(),
                &MeetingSpanFile {
                    format_version: process::MEETING_SPAN_FORMAT_VERSION,
                    span: spans[index],
                },
            )?;
        }
        process::write_json_atomic(
            &part.split_from_path(),
            &SplitFrom {
                format_version: SPLIT_FORMAT_VERSION,
                session: original_name.clone(),
                title: original_title.clone(),
                at_ms,
                part: index as u8 + 1,
            },
        )?;
        // A track that ends before the second part starts recorded nothing there.
        let recorded = |source| {
            tracks
                .iter()
                .any(|(candidate, total, _)| *candidate == source && *total > cut * index as u64)
        };
        part.write_manifest(&Manifest {
            state: SessionState::Stopped,
            started_wallclock: if index == 0 {
                manifest.started_wallclock.clone()
            } else {
                second_start.rfc3339()
            },
            mic: StreamInfo {
                enabled: manifest.mic.enabled && recorded(AudioSource::Mic),
                ..manifest.mic.clone()
            },
            system: StreamInfo {
                enabled: manifest.system.enabled && recorded(AudioSource::System),
                ..manifest.system.clone()
            },
            ..manifest.clone()
        })?;
        sync_directory(&part.dir.join("audio"))?;
        sync_directory(&part.screenshots_dir())?;
        sync_directory(&part.dir)?;
    }

    let mut finished = Vec::new();
    for (part, start) in parts.iter().zip([first_start, second_start]) {
        let target = session::reserve_dir(&root.join(session::session_dir_name(&start.time)))?;
        undo.directories.push(target.clone());
        // Replaces the empty directory that reserved the name.
        fs::rename(&part.dir, &target)?;
        finished.push(target);
    }
    fs::remove_dir(&staging)?;
    sync_directory(&root)?;

    let into_path = session.split_into_path();
    let previous = match fs::read(&into_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let mut record = read_split_into(session)?.unwrap_or(SplitInto {
        format_version: SPLIT_FORMAT_VERSION,
        splits: Vec::new(),
    });
    record.splits.push(SplitRecord {
        at_ms,
        parts: [0, 1].map(|index| {
            finished[index]
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        }),
    });
    undo.split_into = Some((into_path.clone(), previous));
    process::write_json_atomic(&into_path, &record)?;
    sync_directory(&session.dir)?;
    let archived = archive::archive_split_backup(session).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("could not archive the original, so it was not split: {error}"),
        )
    })?;
    undo.disarm();
    let [first, second] = <[PathBuf; 2]>::try_from(finished).expect("two parts");
    Ok(SplitOutcome {
        parts: [first, second],
        archived,
    })
}

/// Removes what a failed split left behind and restores the original's record.
#[derive(Default)]
struct Undo {
    directories: Vec<PathBuf>,
    split_into: Option<(PathBuf, Option<Vec<u8>>)>,
}

impl Undo {
    fn disarm(&mut self) {
        self.directories.clear();
        self.split_into = None;
    }
}

impl Drop for Undo {
    fn drop(&mut self) {
        for directory in self.directories.iter().rev() {
            let _ = fs::remove_dir_all(directory);
        }
        if let Some((path, previous)) = self.split_into.take() {
            let _ = match previous {
                Some(bytes) => fs::write(&path, bytes),
                None => fs::remove_file(&path),
            };
        }
    }
}

/// Write the samples before `cut` to the first part and the rest to the
/// second; returns how many samples each part holds. A part without any
/// samples of the track gets no file.
fn split_track(
    session: &Session,
    source: AudioSource,
    cut: u64,
    parts: &[Session; 2],
) -> io::Result<[u64; 2]> {
    let path = session.stored_audio_path(source);
    let outputs = [&parts[0], &parts[1]].map(|part| part.audio_path(source));
    if archive::is_archived(&path) {
        let mut reader = TrackReader::open(session, source, 0)?
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "track disappeared"))?;
        let mut writers = [
            BufWriter::new(session::create_private_file(&outputs[0])?),
            BufWriter::new(session::create_private_file(&outputs[1])?),
        ];
        let mut position = 0u64;
        let mut chunk = Vec::with_capacity(CHUNK_SAMPLES);
        let mut bytes = Vec::with_capacity(CHUNK_SAMPLES * 4);
        loop {
            reader.read(&mut chunk, CHUNK_SAMPLES)?;
            if chunk.is_empty() {
                break;
            }
            let first = (cut.saturating_sub(position) as usize).min(chunk.len());
            for (writer, samples) in writers.iter_mut().zip([&chunk[..first], &chunk[first..]]) {
                bytes.clear();
                bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
                writer.write_all(&bytes)?;
            }
            position += chunk.len() as u64;
        }
        for writer in writers {
            writer
                .into_inner()
                .map_err(|error| error.into_error())?
                .sync_all()?;
        }
    } else {
        let mut input = File::open(&path)?;
        let total = input.metadata()?.len() / 4;
        let first = cut.min(total);
        for (output, samples) in outputs.iter().zip([first, total - first]) {
            let mut file = session::create_private_file(output)?;
            let copied = io::copy(&mut (&mut input).take(samples * 4), &mut file)?;
            file.sync_all()?;
            if copied != samples * 4 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("the {source} recording ended early"),
                ));
            }
        }
    }
    let mut written = [0u64; 2];
    for (count, output) in written.iter_mut().zip(&outputs) {
        *count = fs::metadata(output)?.len() / 4;
        if *count == 0 {
            fs::remove_file(output)?;
        }
    }
    Ok(written)
}

/// Each timing event goes to the part whose audio it describes; a clean stop
/// after the cut also ends the first part.
fn split_timeline(
    session: &Session,
    source: AudioSource,
    cut: u64,
    parts: &[Session; 2],
    written: [u64; 2],
) -> io::Result<()> {
    let events: Vec<TimelineEvent> = match jsonl::read_all(&session.timeline_path(source)) {
        Ok(events) => events,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    let mut split: [Vec<TimelineEvent>; 2] = Default::default();
    for event in events {
        let sample = event_sample(&event);
        if sample < cut {
            split[0].push(event);
            continue;
        }
        if matches!(event, TimelineEvent::Stop { .. }) {
            split[0].push(at_sample(event.clone(), written[0]));
        }
        split[1].push(at_sample(event, sample - cut));
    }
    for ((part, events), samples) in parts.iter().zip(split).zip(written) {
        if samples > 0 {
            jsonl::write_all_atomic(&part.timeline_path(source), &events)?;
        }
    }
    Ok(())
}

fn event_sample(event: &TimelineEvent) -> u64 {
    match event {
        TimelineEvent::Start { sample, .. }
        | TimelineEvent::Xrun { sample, .. }
        | TimelineEvent::ClockJump { sample, .. }
        | TimelineEvent::Dropped { sample, .. }
        | TimelineEvent::Overlap { sample, .. }
        | TimelineEvent::Error { sample, .. }
        | TimelineEvent::Stop { sample, .. } => *sample,
    }
}

fn at_sample(mut event: TimelineEvent, new: u64) -> TimelineEvent {
    match &mut event {
        TimelineEvent::Start { sample, time_ms }
        | TimelineEvent::Xrun {
            sample, time_ms, ..
        }
        | TimelineEvent::ClockJump {
            sample, time_ms, ..
        }
        | TimelineEvent::Dropped {
            sample, time_ms, ..
        }
        | TimelineEvent::Overlap {
            sample, time_ms, ..
        }
        | TimelineEvent::Error {
            sample, time_ms, ..
        }
        | TimelineEvent::Stop { sample, time_ms } => {
            *sample = new;
            *time_ms = samples_to_ms(new);
        }
    }
    event
}

/// Each screenshot goes to the part it was taken in, renamed after its time
/// there. Entries whose file is missing, is a symlink, or lies outside the
/// session's `screenshots/` folder are left out.
fn split_screenshots(
    session: &Session,
    screenshots: &[ScreenshotEntry],
    at_ms: u64,
    parts: &[Session; 2],
) -> io::Result<()> {
    let mut split: [Vec<ScreenshotEntry>; 2] = Default::default();
    for entry in screenshots {
        let Some((name, mut source)) = open_screenshot(session, &entry.file) else {
            eprintln!(
                "warning: leaving out screenshot {:?}: not a regular file in the session's screenshots folder",
                entry.file
            );
            continue;
        };
        let index = usize::from(entry.time_ms >= at_ms);
        let time_ms = entry.time_ms - [0, at_ms][index];
        let target = free_screenshot_name(&parts[index].screenshots_dir(), &name, time_ms);
        copy_private(&mut source, &target)?;
        let name = target.file_name().unwrap_or_default().to_string_lossy();
        split[index].push(ScreenshotEntry {
            time_ms,
            file: format!("screenshots/{name}"),
            original: entry.original.clone(),
        });
    }
    for (part, entries) in parts.iter().zip(split) {
        if !entries.is_empty() {
            jsonl::write_all_atomic(&part.screenshots_index_path(), &entries)?;
        }
    }
    Ok(())
}

/// `screenshots/<name>` opened without following symlinks, if it is a
/// regular file directly in a real `screenshots` folder of the session.
fn open_screenshot(session: &Session, relative: &str) -> Option<(std::ffi::OsString, File)> {
    let mut components = Path::new(relative).components();
    let (Some(Component::Normal(folder)), Some(Component::Normal(name)), None) =
        (components.next(), components.next(), components.next())
    else {
        return None;
    };
    if folder != "screenshots"
        || !fs::symlink_metadata(session.screenshots_dir())
            .ok()?
            .is_dir()
    {
        return None;
    }
    let file = fs::OpenOptions::new()
        .read(true)
        // Neither follows a symlink nor waits on a FIFO or device.
        .custom_flags((rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits() as i32)
        .open(session.screenshots_dir().join(name))
        .ok()?;
    file.metadata()
        .ok()?
        .is_file()
        .then(|| (name.to_os_string(), file))
}

/// Copy `source` into a new owner-only file and flush it to disk.
fn copy_private(source: &mut File, target: &Path) -> io::Result<()> {
    let mut file = session::create_private_file(target)?;
    io::copy(source, &mut file)?;
    file.sync_all()
}

/// `<time>-<name>` as the screenshot watcher names files, with the time
/// replaced; other names are kept. Either gets a `-N` suffix when taken.
fn free_screenshot_name(directory: &Path, name: &OsStr, time_ms: u64) -> PathBuf {
    let bytes = name.as_bytes();
    let timed = bytes.len() > 10 && bytes[..9].iter().all(u8::is_ascii_digit) && bytes[9] == b'-';
    (0u32..)
        .map(|suffix| {
            if timed {
                screenshot_file_name(time_ms, OsStr::from_bytes(&bytes[10..]), suffix)
            } else {
                let mut candidate = name.to_os_string();
                if suffix > 0 {
                    candidate.push(format!("-{suffix}"));
                }
                candidate
            }
        })
        .map(|candidate| directory.join(candidate))
        .find(|candidate| !candidate.exists())
        .expect("a free screenshot name")
}

/// The meeting marks as seen from each part. A mark outside a part opens that
/// side; a meeting entirely outside a part leaves it unmarked.
fn split_span(span: MeetingSpan, at_ms: u64) -> [MeetingSpan; 2] {
    let first = if span.start_ms.is_some_and(|start| start >= at_ms) {
        MeetingSpan::default()
    } else {
        MeetingSpan {
            start_ms: span.start_ms,
            end_ms: span.end_ms.filter(|end| *end < at_ms),
        }
    };
    let second = if span.end_ms.is_some_and(|end| end <= at_ms) {
        MeetingSpan::default()
    } else {
        MeetingSpan {
            start_ms: span
                .start_ms
                .filter(|start| *start > at_ms)
                .map(|start| start - at_ms),
            end_ms: span.end_ms.map(|end| end - at_ms),
        }
    };
    [first, second].map(|span| {
        if span.is_valid() {
            span
        } else {
            MeetingSpan::default()
        }
    })
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn clock(ms: u64) -> String {
    let seconds = ms / 1000;
    if seconds < 3600 {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    } else {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds % 3600 / 60,
            seconds % 60
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FORMAT_VERSION;
    use std::os::unix::fs::PermissionsExt;

    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "singstone-split-{label}-{}-{}",
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

    /// A stopped, unprocessed session: 3 s of microphone, 5 s of system audio
    /// plus a stray partial sample, and a screenshot on each side of 3.5 s.
    fn recorded_session(root: &Path) -> Session {
        let session = Session::create(root.join("session-20260914-103000")).expect("create");
        let stream = StreamInfo {
            enabled: true,
            pipewire_node: None,
            error: None,
        };
        session
            .write_manifest(&Manifest {
                format_version: FORMAT_VERSION,
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
            .expect("write manifest");
        fs::write(session.audio_path(AudioSource::Mic), samples(48_000, 0.1)).expect("mic");
        let mut system = samples(80_000, 0.2);
        system.extend_from_slice(&[1, 2]);
        fs::write(session.audio_path(AudioSource::System), system).expect("system");
        jsonl::write_all_atomic(
            &session.timeline_path(AudioSource::System),
            &[
                TimelineEvent::Start {
                    sample: 0,
                    time_ms: 0,
                },
                TimelineEvent::Xrun {
                    sample: 64_000,
                    time_ms: 4_000,
                    missing_samples: 10,
                },
                TimelineEvent::Stop {
                    sample: 80_000,
                    time_ms: 5_000,
                },
            ],
        )
        .expect("timeline");
        let mut entries = Vec::new();
        for (time_ms, name) in [(1_000, "000001000-a.png"), (4_200, "000004200-b.png")] {
            fs::write(session.screenshots_dir().join(name), name).expect("screenshot");
            entries.push(ScreenshotEntry {
                time_ms,
                file: format!("screenshots/{name}"),
                original: format!("/shots/{name}"),
            });
        }
        jsonl::write_all_atomic(&session.screenshots_index_path(), &entries).expect("index");
        meeting::write_details_atomic(
            &session.meeting_path(),
            &MeetingDetails::new(
                "Weekly".into(),
                Attendees {
                    known: vec!["Ada".into()],
                    unknown: 1,
                },
                Attendees::default(),
            ),
        )
        .expect("details");
        session
    }

    #[test]
    fn split_writes_two_sessions_and_archives_the_original() {
        let root = fixture_root("whole");
        let original = recorded_session(&root);
        // Left behind by a crashed split; a new split works around it.
        let stale = root.join(".session-20260914-103000.split");
        fs::create_dir_all(stale.join("part-1")).expect("stale staging");
        // A symlink, a FIFO and a path out of the screenshots folder are left out.
        let secret = root.join("secret.png");
        fs::write(&secret, "secret").expect("secret");
        std::os::unix::fs::symlink(
            &secret,
            original.screenshots_dir().join("000002000-link.png"),
        )
        .expect("symlink");
        let mut entries: Vec<ScreenshotEntry> =
            jsonl::read_all(&original.screenshots_index_path()).expect("index");
        rustix::fs::mknodat(
            rustix::fs::CWD,
            original.screenshots_dir().join("000002000-fifo.png"),
            rustix::fs::FileType::Fifo,
            rustix::fs::Mode::from_raw_mode(0o600),
            0,
        )
        .expect("fifo");
        for file in [
            "screenshots/000002000-link.png",
            "screenshots/000002000-fifo.png",
            "screenshots/../secret.png",
        ] {
            entries.push(ScreenshotEntry {
                time_ms: 2_000,
                file: file.into(),
                original: "/shots/x.png".into(),
            });
        }
        jsonl::write_all_atomic(&original.screenshots_index_path(), &entries).expect("index");
        let outcome =
            split_session(&original, 3_500, ["Weekly", " Weekly, part two "]).expect("split");

        let names = outcome
            .parts
            .iter()
            .map(|part| {
                part.file_name()
                    .expect("name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["session-20260914-103000-1", "session-20260914-103003"]
        );
        let [first, second] = outcome
            .parts
            .map(|part| Session::open(part).expect("open part"));
        assert_eq!(
            second.read_manifest().expect("manifest").started_wallclock,
            "2026-09-14T10:30:03.500+02:00"
        );

        let raw = |session: &Session, source| fs::read(session.audio_path(source)).ok();
        assert_eq!(raw(&first, AudioSource::Mic), Some(samples(48_000, 0.1)));
        assert_eq!(raw(&second, AudioSource::Mic), None);
        assert!(!second.read_manifest().expect("manifest").mic.enabled);
        let system = samples(80_000, 0.2);
        assert_eq!(
            raw(&first, AudioSource::System).as_deref(),
            Some(&system[..56_000 * 4])
        );
        assert_eq!(
            raw(&second, AudioSource::System).as_deref(),
            Some(&system[56_000 * 4..])
        );

        let timeline = |session: &Session| -> Vec<TimelineEvent> {
            jsonl::read_all(&session.timeline_path(AudioSource::System)).expect("timeline")
        };
        assert!(matches!(
            timeline(&first)[..],
            [
                TimelineEvent::Start { .. },
                TimelineEvent::Stop {
                    sample: 56_000,
                    time_ms: 3_500
                }
            ]
        ));
        assert!(matches!(
            timeline(&second)[..],
            [
                TimelineEvent::Xrun {
                    sample: 8_000,
                    time_ms: 500,
                    ..
                },
                TimelineEvent::Stop { sample: 24_000, .. }
            ]
        ));

        let shots = |session: &Session| -> Vec<ScreenshotEntry> {
            jsonl::read_all(&session.screenshots_index_path()).expect("screenshots")
        };
        assert_eq!(
            shots(&first)
                .iter()
                .map(|shot| shot.file.as_str())
                .collect::<Vec<_>>(),
            ["screenshots/000001000-a.png"]
        );
        assert_eq!(
            fs::metadata(first.dir.join("screenshots/000001000-a.png"))
                .expect("copied screenshot")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        let moved = &shots(&second)[0];
        assert_eq!(
            (moved.time_ms, moved.file.as_str()),
            (700, "screenshots/000000700-b.png")
        );
        assert!(second.dir.join(&moved.file).is_file());

        let details = |session: &Session| {
            meeting::read_details(&session.meeting_path())
                .expect("details")
                .expect("details exist")
        };
        assert_eq!(details(&second).title, "Weekly, part two");
        assert_eq!(details(&second).local.known, ["Ada"]);
        assert_eq!(
            read_split_from(&second).expect("provenance"),
            Some(SplitFrom {
                format_version: SPLIT_FORMAT_VERSION,
                session: "session-20260914-103000".into(),
                title: "Weekly".into(),
                at_ms: 3_500,
                part: 2,
            })
        );
        assert_eq!(
            read_split_into(&original)
                .expect("record")
                .expect("split")
                .splits,
            [SplitRecord {
                at_ms: 3_500,
                parts: names.try_into().expect("two names"),
            }]
        );

        assert!(original.is_archived());
        assert_eq!(
            original
                .read_audio(AudioSource::Mic)
                .expect("archived mic")
                .len(),
            48_000
        );
        let again = split_session(&original, 2_000, ["a", "b"]).expect("split the backup");
        let again_second = Session::open(&again.parts[1]).expect("open part");
        assert_eq!(
            fs::metadata(again_second.audio_path(AudioSource::System))
                .expect("decoded part")
                .len(),
            48_000 * 4
        );
        assert_eq!(
            read_split_into(&original)
                .expect("record")
                .expect("splits")
                .splits
                .len(),
            2
        );
        let hidden = fs::read_dir(&root)
            .expect("root")
            .map(|entry| entry.expect("entry").path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.as_bytes()[0] == b'.')
            })
            .collect::<Vec<_>>();
        assert_eq!(hidden, [stale]);
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn a_failed_split_leaves_only_the_original() {
        let root = fixture_root("failed");
        let original = recorded_session(&root);
        let before = fs::read_dir(&root).expect("root").count();
        fs::create_dir(jsonl::tmp_path(&original.split_into_path())).expect("block the record");

        assert!(split_session(&original, 3_500, ["a", "b"]).is_err());
        assert_eq!(fs::read_dir(&root).expect("root").count(), before);
        assert!(!original.split_into_path().exists());
        assert!(original.audio_path(AudioSource::System).is_file());
        assert!(!original.is_archived());

        fs::write(original.transcript_path(), b"").expect("process");
        assert!(split_session(&original, 3_500, ["a", "b"]).is_err());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn cut_bounds_leave_a_second_on_each_side() {
        let root = fixture_root("bounds");
        let original = recorded_session(&root);
        for at_ms in [999, 4_001] {
            assert_eq!(
                split_session(&original, at_ms, ["a", "b"])
                    .expect_err("too close to an end")
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn meeting_marks_are_clipped_to_each_part() {
        let span = |start_ms, end_ms| MeetingSpan { start_ms, end_ms };
        assert_eq!(
            split_span(span(Some(1_000), Some(9_000)), 5_000),
            [span(Some(1_000), None), span(None, Some(4_000))]
        );
        assert_eq!(
            split_span(span(Some(6_000), None), 5_000),
            [span(None, None), span(Some(1_000), None)]
        );
        assert_eq!(
            split_span(span(None, Some(5_000)), 5_000),
            [span(None, None), span(None, None)]
        );
    }

    #[test]
    fn cut_times_read_as_seconds_or_clock_time() {
        assert_eq!(parse_time("2472"), Ok(2_472_000));
        assert_eq!(parse_time("2472.25"), Ok(2_472_250));
        assert_eq!(parse_time("41:12"), Ok(2_472_000));
        assert_eq!(parse_time("1:07:20.5"), Ok(4_040_500));
        for invalid in ["", "1:2", "41:60", "1:60:00", "1:2:3:4", "-5", "4x"] {
            assert!(parse_time(invalid).is_err(), "{invalid}");
        }
    }
}
