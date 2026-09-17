//! Session directory layout and manifest handling.

use crate::format::jsonl;
use crate::types::Manifest;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

pub const MANIFEST: &str = "manifest.json";

#[derive(Debug, Clone)]
pub struct Session {
    pub dir: PathBuf,
}

impl Session {
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        if !dir.join(MANIFEST).is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{} is not a session directory (no manifest.json)",
                    dir.display()
                ),
            ));
        }
        Ok(Self { dir })
    }

    /// Create a fresh, private (0700) session directory tree.
    pub fn create(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let requested = dir.into();
        let parent = requested
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
        let mut suffix = 0u64;
        let dir = loop {
            let mut candidate = requested.clone();
            if suffix > 0 {
                let name = requested.file_name().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "session path has no file name")
                })?;
                let mut suffixed = name.to_os_string();
                suffixed.push(format!("-{suffix}"));
                candidate.set_file_name(suffixed);
            }
            match create_private_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    suffix = suffix.checked_add(1).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::AlreadyExists, "too many session collisions")
                    })?;
                }
                Err(error) => return Err(error),
            }
        };
        create_private_dir(&dir.join("audio"))?;
        create_private_dir(&dir.join("screenshots"))?;
        Ok(Self { dir })
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join(MANIFEST)
    }
    pub fn audio_path(&self, source: crate::types::AudioSource) -> PathBuf {
        self.dir.join("audio").join(format!("{source}.f32le"))
    }
    pub fn timeline_path(&self, source: crate::types::AudioSource) -> PathBuf {
        self.dir
            .join("audio")
            .join(format!("{source}.timeline.jsonl"))
    }
    pub fn screenshots_dir(&self) -> PathBuf {
        self.dir.join("screenshots")
    }
    pub fn screenshots_index_path(&self) -> PathBuf {
        self.dir.join("screenshots.jsonl")
    }
    pub fn words_path(&self) -> PathBuf {
        self.dir.join("words.jsonl")
    }
    pub fn words_metadata_path(&self) -> PathBuf {
        self.dir.join("words.meta.json")
    }
    pub fn diarization_path(&self) -> PathBuf {
        self.dir.join("diarization.jsonl")
    }
    pub fn diarization_metadata_path(&self) -> PathBuf {
        self.dir.join("diarization.meta.json")
    }
    pub fn speaker_assignments_path(&self) -> PathBuf {
        self.dir.join("speaker-assignments.json")
    }
    pub fn leakage_suppressions_path(&self) -> PathBuf {
        self.dir.join("leakage-suppressions.jsonl")
    }
    pub fn transcript_path(&self) -> PathBuf {
        self.dir.join("transcript.jsonl")
    }
    pub fn transcript_text_path(&self) -> PathBuf {
        self.dir.join("transcript.txt")
    }

    pub fn read_manifest(&self) -> io::Result<Manifest> {
        let data = fs::read(self.manifest_path())?;
        serde_json::from_slice(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Atomically replace the manifest.
    pub fn write_manifest(&self, manifest: &Manifest) -> io::Result<()> {
        let path = self.manifest_path();
        let tmp = jsonl::tmp_path(&path);
        let mut f = create_private_file(&tmp)?;
        serde_json::to_writer_pretty(&mut f, manifest)?;
        use io::Write;
        f.write_all(b"\n")?;
        f.sync_all()?;
        fs::rename(tmp, path)
    }

    /// Read a raw f32le audio file fully into memory.
    pub fn read_audio(&self, source: crate::types::AudioSource) -> io::Result<Vec<f32>> {
        let bytes = fs::read(self.audio_path(source))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

/// Session directory name for a wall-clock start time: `session-YYYYMMDD-HHMMSS`.
pub fn session_dir_name(local: &LocalTime) -> String {
    format!(
        "session-{:04}{:02}{:02}-{:02}{:02}{:02}",
        local.year, local.month, local.day, local.hour, local.minute, local.second
    )
}

/// Broken-down local time with UTC offset, derived without a time-zone crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub offset_minutes: i32,
}

impl LocalTime {
    pub fn rfc3339(&self) -> String {
        let sign = if self.offset_minutes < 0 { '-' } else { '+' };
        let off = self.offset_minutes.abs();
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{:02}:{:02}",
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            sign,
            off / 60,
            off % 60
        )
    }
}

pub fn local_time_from_unix(seconds: u64) -> LocalTime {
    let unix_seconds = seconds.min(i64::MAX as u64) as i64;
    let offset_seconds = local_offset_seconds(unix_seconds).unwrap_or(0);
    broken_down_time(
        unix_seconds.saturating_add(i64::from(offset_seconds)),
        offset_seconds / 60,
    )
}

fn local_offset_seconds(unix_seconds: i64) -> Option<i32> {
    let path = match std::env::var_os("TZ") {
        Some(value) if !value.is_empty() => {
            let zone = Path::new(&value);
            if zone.is_absolute()
                || zone
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return None;
            }
            let directory = std::env::var_os("TZDIR")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/usr/share/zoneinfo"));
            directory.join(zone)
        }
        _ => PathBuf::from("/etc/localtime"),
    };
    let bytes = fs::read(path).ok()?;
    parse_tzif_offset(&bytes, unix_seconds).ok()
}

fn broken_down_time(seconds: i64, offset_minutes: i32) -> LocalTime {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400) as u64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    LocalTime {
        year: year as i32,
        month: month as u32,
        day: day as u32,
        hour: (seconds_of_day / 3_600) as u32,
        minute: ((seconds_of_day % 3_600) / 60) as u32,
        second: (seconds_of_day % 60) as u32,
        offset_minutes,
    }
}

fn parse_tzif_offset(bytes: &[u8], unix_seconds: i64) -> io::Result<i32> {
    const HEADER_LEN: usize = 44;
    if bytes.len() < HEADER_LEN || &bytes[..4] != b"TZif" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid TZif header",
        ));
    }
    let read_count = |offset| read_be_u32(bytes, offset).map(|value| value as usize);
    let ttisgmtcnt = read_count(20)?;
    let ttisstdcnt = read_count(24)?;
    let leapcnt = read_count(28)?;
    let timecnt = read_count(32)?;
    let typecnt = read_count(36)?;
    let charcnt = read_count(40)?;
    if typecnt == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "TZif has no local time types",
        ));
    }
    let transitions_offset = HEADER_LEN;
    let indices_offset = transitions_offset
        .checked_add(timecnt.checked_mul(4).ok_or_else(invalid_tzif)?)
        .ok_or_else(invalid_tzif)?;
    let ttinfo_offset = indices_offset
        .checked_add(timecnt)
        .ok_or_else(invalid_tzif)?;
    let ttinfo_end = ttinfo_offset
        .checked_add(typecnt.checked_mul(6).ok_or_else(invalid_tzif)?)
        .ok_or_else(invalid_tzif)?;
    let block_end = ttinfo_end
        .checked_add(charcnt)
        .and_then(|end| end.checked_add(leapcnt.checked_mul(8)?))
        .and_then(|end| end.checked_add(ttisstdcnt))
        .and_then(|end| end.checked_add(ttisgmtcnt))
        .ok_or_else(invalid_tzif)?;
    if block_end > bytes.len() {
        return Err(invalid_tzif());
    }

    let type_info = |index: usize| -> io::Result<(i32, bool)> {
        if index >= typecnt {
            return Err(invalid_tzif());
        }
        let offset = ttinfo_offset + index * 6;
        Ok((read_be_i32(bytes, offset)?, bytes[offset + 4] != 0))
    };
    let mut selected = None;
    for index in 0..timecnt {
        let transition = i64::from(read_be_i32(bytes, transitions_offset + index * 4)?);
        let type_index = bytes[indices_offset + index] as usize;
        if type_index >= typecnt {
            return Err(invalid_tzif());
        }
        if transition <= unix_seconds {
            selected = Some(type_index);
        } else {
            break;
        }
    }
    let selected = selected.unwrap_or_else(|| {
        (0..typecnt)
            .find(|index| type_info(*index).is_ok_and(|(_, is_dst)| !is_dst))
            .unwrap_or(0)
    });
    type_info(selected).map(|(offset, _)| offset)
}

fn read_be_u32(bytes: &[u8], offset: usize) -> io::Result<u32> {
    let value = bytes.get(offset..offset + 4).ok_or_else(invalid_tzif)?;
    Ok(u32::from_be_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_be_i32(bytes: &[u8], offset: usize) -> io::Result<i32> {
    read_be_u32(bytes, offset).map(|value| value as i32)
}

fn invalid_tzif() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "truncated or invalid TZif data")
}

pub fn create_private_dir(path: &Path) -> io::Result<()> {
    fs::DirBuilder::new().mode(0o700).create(path)
}

/// Create (truncating) a file readable only by the owner.
pub fn create_private_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn session_creation_recurses_and_avoids_collisions() {
        let root =
            std::env::temp_dir().join(format!("singstone-session-create-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let requested = root.join("missing/parent/session");
        let first = Session::create(&requested).expect("create first session");
        let second = Session::create(&requested).expect("create colliding session");
        assert_eq!(first.dir, requested);
        assert_eq!(
            second.dir.file_name().and_then(|name| name.to_str()),
            Some("session-1")
        );
        assert_eq!(
            fs::metadata(root.join("missing"))
                .expect("parent metadata")
                .mode()
                & 0o777,
            0o700
        );
        fs::remove_dir_all(root).expect("remove sessions");
    }

    #[test]
    fn parses_v1_tzif_transitions() {
        let mut bytes = vec![0u8; 44];
        bytes[..5].copy_from_slice(b"TZif\0");
        bytes[32..36].copy_from_slice(&2u32.to_be_bytes());
        bytes[36..40].copy_from_slice(&2u32.to_be_bytes());
        bytes[40..44].copy_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&100i32.to_be_bytes());
        bytes.extend_from_slice(&200i32.to_be_bytes());
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&0i32.to_be_bytes());
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&7_200i32.to_be_bytes());
        bytes.extend_from_slice(&[1, 0]);
        bytes.push(0);

        assert_eq!(parse_tzif_offset(&bytes, 50).expect("pre-transition"), 0);
        assert_eq!(
            parse_tzif_offset(&bytes, 150).expect("DST transition"),
            7_200
        );
        assert_eq!(
            parse_tzif_offset(&bytes, 250).expect("standard transition"),
            0
        );
    }

    #[test]
    fn broken_down_time_applies_offset() {
        assert_eq!(
            broken_down_time(1_789_381_800 + 7_200, 120),
            LocalTime {
                year: 2026,
                month: 9,
                day: 14,
                hour: 12,
                minute: 30,
                second: 0,
                offset_minutes: 120,
            }
        );
    }
}
