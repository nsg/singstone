use crate::format::jsonl;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const MEETING_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Attendees {
    pub known: Vec<String>,
    pub unknown: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeetingDetails {
    pub format_version: u32,
    pub title: String,
    #[serde(default)]
    pub local: Attendees,
    #[serde(default)]
    pub remote: Attendees,
}

impl MeetingDetails {
    pub fn new(title: String, local: Attendees, remote: Attendees) -> Self {
        Self {
            format_version: MEETING_FORMAT_VERSION,
            title: title.trim().to_owned(),
            local: normalize_attendees(local),
            remote: normalize_attendees(remote),
        }
    }

    pub fn local_count(&self) -> u32 {
        attendee_count(&self.local)
    }

    pub fn remote_count(&self) -> u32 {
        attendee_count(&self.remote)
    }

    pub fn is_remote_attendee(&self, name: &str) -> bool {
        self.remote.known.iter().any(|known| known == name)
    }

    pub fn is_local_attendee(&self, name: &str) -> bool {
        self.local.known.iter().any(|known| known == name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeetingContextFile {
    pub format_version: u32,
    pub meetings: Vec<MeetingContextEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeetingContextEntry {
    pub title: String,
    pub start: String,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub local: Attendees,
    #[serde(default)]
    pub remote: Attendees,
}

pub fn read_details(path: &Path) -> io::Result<Option<MeetingDetails>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut details: MeetingDetails = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid meeting details {}: {error}", path.display()),
        )
    })?;
    if details.format_version != MEETING_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported meeting details format version {} in {}",
                details.format_version,
                path.display()
            ),
        ));
    }
    details.title = details.title.trim().to_owned();
    details.local = normalize_attendees(details.local);
    details.remote = normalize_attendees(details.remote);
    Ok(Some(details))
}

pub fn write_details_atomic(path: &Path, details: &MeetingDetails) -> io::Result<()> {
    if details.format_version != MEETING_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unsupported meeting details format version {}",
                details.format_version
            ),
        ));
    }
    let normalized = MeetingDetails::new(
        details.title.trim().to_owned(),
        details.local.clone(),
        details.remote.clone(),
    );
    let temporary = jsonl::tmp_path(path);
    {
        let mut file = crate::session::create_private_file(&temporary)?;
        serde_json::to_writer_pretty(&mut file, &normalized)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(temporary, path)
}

/// `source` is either one meeting-context JSON file or a folder of them.
pub fn match_context_with_source(
    source: &Path,
    started_wallclock: &str,
) -> Option<(MeetingDetails, PathBuf)> {
    let session_start = match parse_rfc3339(started_wallclock) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("warning: cannot match meeting context: session start is invalid: {error}");
            return None;
        }
    };
    let mut files = if source.is_file() {
        vec![source.to_path_buf()]
    } else {
        match fs::read_dir(source) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file()
                        && path
                            .extension()
                            .is_some_and(|extension| extension == "json")
                })
                .collect::<Vec<_>>(),
            Err(error) => {
                eprintln!(
                    "warning: cannot read meeting context {}: {error}",
                    source.display()
                );
                return None;
            }
        }
    };
    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));

    let mut best: Option<(u64, MeetingDetails, PathBuf)> = None;
    for path in files {
        let entries = match read_context_file(&path) {
            Some(entries) => entries,
            None => continue,
        };
        for entry in entries.meetings {
            let start = match parse_context_entry(&path, &entry) {
                Some(start) => start,
                None => continue,
            };
            let end = entry
                .end
                .as_deref()
                .and_then(|value| parse_rfc3339(value).ok())
                .unwrap_or_else(|| start.saturating_add(60 * 60));
            if session_start < start.saturating_sub(15 * 60) || session_start > end {
                continue;
            }
            let distance = session_start.abs_diff(start);
            if best
                .as_ref()
                .is_none_or(|(best_distance, _, _)| distance < *best_distance)
            {
                best = Some((
                    distance,
                    MeetingDetails::new(entry.title, entry.local, entry.remote),
                    path.clone(),
                ));
            }
        }
    }
    best.map(|(_, details, path)| (details, path))
}

fn read_context_file(path: &Path) -> Option<MeetingContextFile> {
    let value: serde_json::Value = match fs::read(path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
    {
        Ok(value) => value,
        Err(error) => {
            eprintln!(
                "warning: skipping meeting context file {}: {error}",
                path.display()
            );
            return None;
        }
    };
    let version = value.get("format_version").and_then(|value| value.as_u64());
    if version != Some(u64::from(MEETING_FORMAT_VERSION)) {
        eprintln!(
            "warning: skipping meeting context file {}: format_version must be {}",
            path.display(),
            MEETING_FORMAT_VERSION
        );
        return None;
    }
    let Some(meetings) = value.get("meetings").and_then(|value| value.as_array()) else {
        eprintln!(
            "warning: skipping meeting context file {}: meetings must be an array",
            path.display()
        );
        return None;
    };
    let mut parsed = Vec::new();
    for (index, meeting) in meetings.iter().enumerate() {
        match serde_json::from_value::<MeetingContextEntry>(meeting.clone()) {
            Ok(entry) => parsed.push(entry),
            Err(error) => eprintln!(
                "warning: skipping meeting entry {} in {}: {error}",
                index + 1,
                path.display()
            ),
        }
    }
    Some(MeetingContextFile {
        format_version: MEETING_FORMAT_VERSION,
        meetings: parsed,
    })
}

fn parse_context_entry(path: &Path, entry: &MeetingContextEntry) -> Option<i64> {
    if entry.title.trim().is_empty() {
        eprintln!(
            "warning: skipping meeting entry in {}: title must not be empty",
            path.display()
        );
        return None;
    }
    let start = match parse_rfc3339(&entry.start) {
        Ok(start) => start,
        Err(error) => {
            eprintln!(
                "warning: skipping meeting entry {:?} in {}: invalid start: {error}",
                entry.title,
                path.display()
            );
            return None;
        }
    };
    if let Some(end) = entry.end.as_deref() {
        match parse_rfc3339(end) {
            Ok(end) if end >= start => {}
            Ok(_) => {
                eprintln!(
                    "warning: skipping meeting entry {:?} in {}: end is before start",
                    entry.title,
                    path.display()
                );
                return None;
            }
            Err(error) => {
                eprintln!(
                    "warning: skipping meeting entry {:?} in {}: invalid end: {error}",
                    entry.title,
                    path.display()
                );
                return None;
            }
        }
    }
    Some(start)
}

fn attendee_count(attendees: &Attendees) -> u32 {
    u32::try_from(attendees.known.len())
        .unwrap_or(u32::MAX)
        .saturating_add(attendees.unknown)
}

fn normalize_attendees(mut attendees: Attendees) -> Attendees {
    let mut normalized = Vec::new();
    for name in attendees.known {
        let name = name.trim();
        if !name.is_empty() && !normalized.iter().any(|existing| existing == name) {
            normalized.push(name.to_owned());
        }
    }
    attendees.known = normalized;
    attendees
}

pub fn parse_rfc3339(value: &str) -> Result<i64, String> {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return Err("expected YYYY-MM-DDTHH:MM:SS with Z or numeric offset".into());
    }
    let year = decimal(bytes, 0, 4)? as i32;
    let month = decimal(bytes, 5, 2)?;
    let day = decimal(bytes, 8, 2)?;
    let hour = decimal(bytes, 11, 2)?;
    let minute = decimal(bytes, 14, 2)?;
    let second = decimal(bytes, 17, 2)?;
    if !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err("date or time component is out of range".into());
    }
    let mut index = 19;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == fraction_start {
            return Err("fractional seconds require at least one digit".into());
        }
    }
    let offset_seconds = match bytes.get(index) {
        Some(b'Z') if index + 1 == bytes.len() => 0i64,
        Some(sign @ (b'+' | b'-'))
            if index + 6 == bytes.len() && bytes.get(index + 3) == Some(&b':') =>
        {
            let hours = decimal(bytes, index + 1, 2)?;
            let minutes = decimal(bytes, index + 4, 2)?;
            if hours > 23 || minutes > 59 {
                return Err("UTC offset is out of range".into());
            }
            let seconds = i64::from(hours * 3_600 + minutes * 60);
            if *sign == b'-' { -seconds } else { seconds }
        }
        _ => return Err("timestamp must end with Z or a numeric UTC offset".into()),
    };
    let days = days_from_civil(year, month, day);
    Ok(days
        .saturating_mul(86_400)
        .saturating_add(i64::from(hour * 3_600 + minute * 60 + second))
        .saturating_sub(offset_seconds))
}

fn decimal(bytes: &[u8], start: usize, length: usize) -> Result<u32, String> {
    let digits = bytes
        .get(start..start + length)
        .ok_or_else(|| "timestamp is truncated".to_owned())?;
    if !digits.iter().all(u8::is_ascii_digit) {
        return Err("timestamp contains a non-numeric component".into());
    }
    Ok(digits
        .iter()
        .fold(0u32, |value, digit| value * 10 + u32::from(*digit - b'0')))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 31,
    }
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month = i64::from(month);
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "singstone-meeting-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&path).expect("create fixture");
        path
    }

    fn write_context(dir: &Path, name: &str, meetings: serde_json::Value) {
        fs::write(
            dir.join(name),
            serde_json::to_vec(&serde_json::json!({
                "format_version": 1,
                "meetings": meetings,
            }))
            .expect("serialize fixture"),
        )
        .expect("write fixture");
    }

    #[test]
    fn parses_supported_rfc3339_forms() {
        assert_eq!(
            parse_rfc3339("2026-09-25T10:00:00+02:00"),
            parse_rfc3339("2026-09-25T08:00:00Z")
        );
        assert_eq!(
            parse_rfc3339("2026-09-25T08:00:00.123456Z"),
            parse_rfc3339("2026-09-25T08:00:00Z")
        );
        for invalid in [
            "garbage",
            "2026-09-25T08:00Z",
            "2026-02-29T08:00:00Z",
            "2026-09-25T08:00:00",
            "2026-09-25T08:00:00.Z",
        ] {
            assert!(parse_rfc3339(invalid).is_err(), "accepted {invalid}");
        }
    }

    fn match_context(source: &Path, started_wallclock: &str) -> Option<MeetingDetails> {
        match_context_with_source(source, started_wallclock).map(|(details, _)| details)
    }

    #[test]
    fn matches_a_single_file_and_reports_it() {
        let dir = test_dir("single-file");
        write_context(
            &dir,
            "other.json",
            serde_json::json!([{"title":"Other","start":"2026-09-25T10:00:00Z"}]),
        );
        write_context(
            &dir,
            "chosen.json",
            serde_json::json!([{"title":"Chosen","start":"2026-09-25T10:00:00Z"}]),
        );
        let file = dir.join("chosen.json");
        let (details, source) =
            match_context_with_source(&file, "2026-09-25T10:00:00Z").expect("match");
        assert_eq!(details.title, "Chosen");
        assert_eq!(source, file);
        assert!(match_context(&dir.join("missing.json"), "2026-09-25T10:00:00Z").is_none());
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn matches_windows_and_chooses_closest_start() {
        let dir = test_dir("windows");
        write_context(
            &dir,
            "context.json",
            serde_json::json!([
                {"title":"Earlier","start":"2026-09-25T09:30:00Z","end":"2026-09-25T11:00:00Z"},
                {"title":"Closest","start":"2026-09-25T10:05:00Z","end":"2026-09-25T11:00:00Z"}
            ]),
        );
        assert_eq!(
            match_context(&dir, "2026-09-25T10:00:00Z")
                .expect("match")
                .title,
            "Closest"
        );
        assert!(match_context(&dir, "2026-09-25T09:14:59Z").is_none());
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn missing_end_uses_one_hour_and_bad_input_is_skipped() {
        let dir = test_dir("default-end");
        fs::write(dir.join("bad.json"), b"not json").expect("write bad fixture");
        write_context(
            &dir,
            "valid.json",
            serde_json::json!([
                {"title":7,"start":"bad"},
                {"title":"Default end","start":"2026-09-25T10:00:00Z",
                 "local":{"known":[" Me ","Me",""],"unknown":0}}
            ]),
        );
        let matched = match_context(&dir, "2026-09-25T11:00:00Z").expect("boundary match");
        assert_eq!(matched.title, "Default end");
        assert_eq!(matched.local.known, ["Me"]);
        assert!(match_context(&dir, "2026-09-25T11:00:01Z").is_none());
        fs::remove_dir_all(dir).expect("remove fixture");
    }
}
