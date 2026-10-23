//! Reading a recorded track a chunk at a time, whether it is still raw float
//! or archived as FLAC, and the level envelope drawn from it.

use crate::audio::archive;
use crate::session::Session;
use crate::types::{AudioSource, SAMPLE_RATE};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, Ordering};

const CHUNK_SAMPLES: usize = 16_384;

pub enum TrackReader {
    Raw {
        file: BufReader<File>,
        remaining: u64,
        bytes: Vec<u8>,
    },
    Archived(archive::SampleStream),
}

impl TrackReader {
    /// The track from sample `start` on, or `None` when it has no stored audio.
    pub fn open(session: &Session, source: AudioSource, start: u64) -> io::Result<Option<Self>> {
        let path = session.stored_audio_path(source);
        if !path.is_file() {
            return Ok(None);
        }
        if archive::is_archived(&path) {
            return archive::stream(&path, start).map(|stream| Some(Self::Archived(stream)));
        }
        let mut file = File::open(&path)?;
        let total = file.metadata()?.len() / 4;
        let start = start.min(total);
        file.seek(SeekFrom::Start(start * 4))?;
        Ok(Some(Self::Raw {
            file: BufReader::with_capacity(1 << 16, file),
            remaining: total - start,
            bytes: Vec::new(),
        }))
    }

    /// Replace `out` with up to `max` next samples; empty at the end.
    pub fn read(&mut self, out: &mut Vec<f32>, max: usize) -> io::Result<()> {
        match self {
            Self::Archived(stream) => stream.read(out, max),
            Self::Raw {
                file,
                remaining,
                bytes,
            } => {
                out.clear();
                let wanted = (max as u64).min(*remaining) as usize;
                bytes.resize(wanted * 4, 0);
                let mut filled = 0;
                while filled < bytes.len() {
                    match file.read(&mut bytes[filled..]) {
                        Ok(0) => break,
                        Ok(count) => filled += count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(error) => return Err(error),
                    }
                }
                let whole = filled / 4;
                out.extend(
                    bytes[..whole * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|sample| f32::from_le_bytes(*sample)),
                );
                *remaining = if whole < wanted {
                    0
                } else {
                    *remaining - whole as u64
                };
                Ok(())
            }
        }
    }
}

/// Peak and RMS of one stretch of samples.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Level {
    pub peak: f32,
    pub rms: f32,
}

pub const BUCKET_MS: u64 = 50;
const BUCKET_SAMPLES: usize = (SAMPLE_RATE as u64 * BUCKET_MS / 1000) as usize;
/// A pause shorter than this is not a place to cut.
const MIN_QUIET_MS: u64 = 600;
/// How far inside a quiet stretch a snapped cut stays from speech.
const QUIET_MARGIN_MS: u64 = 1_000;

/// Each track's level per `BUCKET_MS`.
#[derive(Debug, Clone, Default)]
pub struct Envelope {
    pub tracks: Vec<(AudioSource, Vec<Level>)>,
}

impl Envelope {
    /// Read every stored track of `session`; `None` once `cancelled` is set.
    pub fn compute(session: &Session, cancelled: &AtomicBool) -> io::Result<Option<Self>> {
        let mut tracks = Vec::new();
        for source in [AudioSource::Mic, AudioSource::System] {
            let Some(mut reader) = TrackReader::open(session, source, 0)? else {
                continue;
            };
            let mut levels = Vec::new();
            let mut chunk = Vec::with_capacity(CHUNK_SAMPLES);
            let (mut peak, mut squares, mut count) = (0f32, 0f64, 0usize);
            loop {
                if cancelled.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                reader.read(&mut chunk, CHUNK_SAMPLES)?;
                if chunk.is_empty() {
                    break;
                }
                for &sample in &chunk {
                    let sample = if sample.is_finite() { sample } else { 0.0 };
                    peak = peak.max(sample.abs());
                    squares += f64::from(sample) * f64::from(sample);
                    count += 1;
                    if count == BUCKET_SAMPLES {
                        levels.push(level(peak, squares, count));
                        (peak, squares, count) = (0.0, 0.0, 0);
                    }
                }
            }
            if count > 0 {
                levels.push(level(peak, squares, count));
            }
            tracks.push((source, levels));
        }
        Ok(Some(Self { tracks }))
    }

    /// Stretches, in ms, where every track is near its own silence for at
    /// least `MIN_QUIET_MS`. Past a track's end counts as silent.
    pub fn quiet_stretches(&self) -> Vec<(u64, u64)> {
        let buckets = self
            .tracks
            .iter()
            .map(|(_, levels)| levels.len())
            .max()
            .unwrap_or(0);
        let thresholds = self
            .tracks
            .iter()
            .map(|(_, levels)| quiet_threshold(levels))
            .collect::<Vec<_>>();
        let quiet = |bucket: usize| {
            self.tracks
                .iter()
                .zip(&thresholds)
                .all(|((_, levels), threshold)| {
                    levels
                        .get(bucket)
                        .is_none_or(|level| level.rms <= *threshold)
                })
        };
        let mut stretches = Vec::new();
        let mut start = None;
        for bucket in 0..=buckets {
            match (bucket < buckets && quiet(bucket), start) {
                (true, None) => start = Some(bucket),
                (false, Some(first)) => {
                    let (from, to) = (first as u64 * BUCKET_MS, bucket as u64 * BUCKET_MS);
                    if to - from >= MIN_QUIET_MS {
                        stretches.push((from, to));
                    }
                    start = None;
                }
                _ => {}
            }
        }
        stretches
    }
}

fn level(peak: f32, squares: f64, count: usize) -> Level {
    Level {
        peak,
        rms: (squares / count as f64).sqrt() as f32,
    }
}

/// Near silence for one track: a few times its quietest tenth, within sane
/// bounds so that neither digital silence nor constant noise breaks it.
fn quiet_threshold(levels: &[Level]) -> f32 {
    let mut rms = levels.iter().map(|level| level.rms).collect::<Vec<_>>();
    if rms.is_empty() {
        return 0.0;
    }
    let tenth = rms.len() / 10;
    let (_, floor, _) = rms.select_nth_unstable_by(tenth, f32::total_cmp);
    (*floor * 4.0).clamp(0.003, 0.03)
}

/// Whether `at_ms` lies in one of `stretches`.
pub fn is_quiet(stretches: &[(u64, u64)], at_ms: u64) -> bool {
    stretches
        .iter()
        .any(|&(start, end)| (start..end).contains(&at_ms))
}

/// The point of the nearest quiet stretch closest to `at_ms`, kept up to
/// `QUIET_MARGIN_MS` away from the speech around it.
pub fn snap_to_quiet(stretches: &[(u64, u64)], at_ms: u64) -> Option<u64> {
    stretches
        .iter()
        .map(|&(start, end)| {
            let margin = ((end - start) / 2).min(QUIET_MARGIN_MS);
            at_ms.clamp(start + margin, end - margin)
        })
        .min_by_key(|snapped| snapped.abs_diff(at_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(tracks: &[&[f32]]) -> Envelope {
        Envelope {
            tracks: tracks
                .iter()
                .map(|levels| {
                    (
                        AudioSource::System,
                        levels.iter().map(|&rms| Level { peak: rms, rms }).collect(),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn quiet_needs_every_track_silent_for_long_enough() {
        let mut talk = vec![0.2f32; 100];
        talk[20..50].fill(0.0);
        talk[60..65].fill(0.0);
        let mut other = vec![0.0f32; 100];
        other[25..30].fill(0.3);
        // The other track talks over the first pause, the second pause is
        // too short, and the other track ending early leaves the tail quiet.
        other.truncate(90);
        talk[92..].fill(0.0);
        assert_eq!(
            envelope(&[&talk, &other]).quiet_stretches(),
            [(30 * BUCKET_MS, 50 * BUCKET_MS)]
        );
        talk[80..].fill(0.0);
        assert_eq!(
            envelope(&[&talk, &other]).quiet_stretches().last(),
            Some(&(80 * BUCKET_MS, 100 * BUCKET_MS))
        );
    }

    #[test]
    fn snapping_moves_into_the_nearest_pause_away_from_speech() {
        let stretches = [(10_000, 10_800), (20_000, 30_000)];
        assert_eq!(snap_to_quiet(&stretches, 12_000), Some(10_400));
        assert_eq!(snap_to_quiet(&stretches, 17_000), Some(21_000));
        assert_eq!(snap_to_quiet(&stretches, 25_000), Some(25_000));
        assert_eq!(snap_to_quiet(&[], 25_000), None);
        assert!(is_quiet(&stretches, 25_000));
        assert!(!is_quiet(&stretches, 30_000));
    }
}
