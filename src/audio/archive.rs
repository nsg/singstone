//! Archived session audio: the raw f32le recording converted to 16-bit FLAC
//! by the `flac` program, and read back through it.

use crate::cli::ArchiveArgs;
use crate::format::jsonl;
use crate::merge::process;
use crate::models;
use crate::session::Session;
use crate::types::{AudioSource, SAMPLE_RATE, SessionState};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;

const RAW_PCM: [&str; 4] = [
    "--silent",
    "--force-raw-format",
    "--endian=little",
    "--sign=signed",
];
const CHUNK_SAMPLES: usize = 16_384;
const STREAM_HEADER_LEN: usize = 42;

/// What archiving changed; `tracks` is empty when the audio already was archived.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ArchiveSummary {
    pub tracks: Vec<AudioSource>,
    pub bytes_before: u64,
    pub bytes_after: u64,
}

pub fn run(args: ArchiveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let summary = archive_session(&Session::open(&args.session)?)?;
    if summary.tracks.is_empty() {
        eprintln!("audio is already archived");
    } else {
        eprintln!(
            "archived {}: {} -> {}",
            summary
                .tracks
                .iter()
                .map(|source| source.as_str())
                .collect::<Vec<_>>()
                .join(" and "),
            format_size(summary.bytes_before),
            format_size(summary.bytes_after)
        );
    }
    Ok(())
}

/// Replace each raw track of a processed session with a verified FLAC copy.
/// The raw file is deleted only after its copy decodes to the same samples.
pub fn archive_session(session: &Session) -> io::Result<ArchiveSummary> {
    if session.read_manifest()?.state == SessionState::Recording {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the session is still being recorded",
        ));
    }
    if !session.transcript_path().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "process the session before archiving its audio",
        ));
    }
    let mut summary = ArchiveSummary::default();
    for source in [AudioSource::Mic, AudioSource::System] {
        let raw = session.audio_path(source);
        if !raw.is_file() {
            continue;
        }
        let archived = session.archived_audio_path(source);
        let temporary = jsonl::tmp_path(&archived);
        if let Err(error) = encode_verified(&raw, &temporary) {
            let _ = fs::remove_file(&temporary);
            return Err(io::Error::new(
                error.kind(),
                format!("cannot archive {source} audio: {error}"),
            ));
        }
        let raw_sha256 = models::sha256_file(&raw)?;
        summary.bytes_before += fs::metadata(&raw)?.len();
        summary.bytes_after += fs::metadata(&temporary)?.len();
        fs::rename(&temporary, &archived)?;
        if let Some(directory) = archived.parent() {
            File::open(directory)?.sync_all()?;
        }
        process::rebase_audio_provenance(
            session,
            source,
            &raw_sha256,
            &models::sha256_file(&archived)?,
        );
        fs::remove_file(&raw)?;
        summary.tracks.push(source);
    }
    if summary.tracks.is_empty() && !session.is_archived() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "the session has no recorded audio",
        ));
    }
    Ok(summary)
}

/// Fails with an explanation when the `flac` program cannot be run.
pub fn ensure_decoder() -> io::Result<()> {
    spawn(
        flac_command()
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )?
    .finish()
}

pub fn is_archived(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "flac")
}

/// Number of samples in an archived track, from its stream header.
pub fn sample_count(path: &Path) -> io::Result<u64> {
    let mut header = [0u8; STREAM_HEADER_LEN];
    File::open(path)?.read_exact(&mut header)?;
    parse_stream_header(&header).map_err(|message| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {message}", path.display()),
        )
    })
}

pub fn read(path: &Path) -> io::Result<Vec<f32>> {
    let mut samples = Vec::new();
    decode_each(path, CHUNK_SAMPLES, |chunk| {
        samples.extend_from_slice(chunk)
    })?;
    Ok(samples)
}

/// Samples `start..end`; the range must lie inside the track.
pub fn read_range(path: &Path, start: u64, end: u64) -> io::Result<Vec<f32>> {
    let mut samples = Vec::new();
    decode(path, Some((start, end)), CHUNK_SAMPLES, |chunk| {
        samples.extend(pcm_samples(chunk))
    })?;
    Ok(samples)
}

/// Decode the whole track, passing `chunk_samples` samples at a time; only
/// the last chunk may be shorter.
pub fn decode_each(
    path: &Path,
    chunk_samples: usize,
    mut sink: impl FnMut(&[f32]),
) -> io::Result<()> {
    let mut samples = Vec::with_capacity(chunk_samples);
    decode(path, None, chunk_samples, |chunk| {
        samples.clear();
        samples.extend(pcm_samples(chunk));
        sink(&samples);
    })
}

pub fn format_size(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

fn quantize(sample: f32) -> i16 {
    // The cast saturates and maps NaN to silence.
    (sample * 32_768.0).round() as i16
}

fn pcm_samples(pcm: &[u8]) -> impl Iterator<Item = f32> {
    pcm.as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f32::from(i16::from_le_bytes(*bytes)) / 32_768.0)
}

/// Encode the raw f32le file at `raw` into `output`, flush it to disk, and
/// check that it decodes back to exactly the 16-bit samples that went in.
fn encode_verified(raw: &Path, output: &Path) -> io::Result<()> {
    let mut output_argument = OsString::from("--output-name=");
    output_argument.push(std::path::absolute(output)?);
    let mut flac = spawn(
        flac_command()
            .args(RAW_PCM)
            .args([
                "--verify",
                "--force",
                "--no-padding",
                "--channels=1",
                "--bps=16",
            ])
            .arg(format!("--sample-rate={SAMPLE_RATE}"))
            .arg(output_argument)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::null()),
    )?;
    let mut stdin = flac.child.stdin.take().expect("piped flac input");
    let mut written = Sha256::new();
    let mut samples = 0u64;
    let fed = (|| {
        let mut audio = File::open(raw)?;
        let mut bytes = vec![0u8; CHUNK_SAMPLES * 4];
        let mut pcm = Vec::with_capacity(CHUNK_SAMPLES * 2);
        loop {
            let filled = read_full(&mut audio, &mut bytes)?;
            pcm.clear();
            for sample in bytes[..filled].as_chunks::<4>().0 {
                pcm.extend_from_slice(&quantize(f32::from_le_bytes(*sample)).to_le_bytes());
            }
            stdin.write_all(&pcm)?;
            written.update(&pcm);
            samples += (pcm.len() / 2) as u64;
            if filled < bytes.len() {
                return stdin.flush();
            }
        }
    })();
    drop(stdin);
    // A failed encoder closes its input; its own message explains more than
    // the resulting broken pipe.
    flac.finish().and(fed)?;

    fs::set_permissions(output, fs::Permissions::from_mode(0o600))?;
    File::open(output)?.sync_all()?;
    let stored = sample_count(output)?;
    let mut decoded = Sha256::new();
    decode(output, None, CHUNK_SAMPLES, |chunk| decoded.update(chunk))?;
    if stored != samples || decoded.finalize() != written.finalize() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the archived copy does not decode back to the recording",
        ));
    }
    Ok(())
}

/// Decode to 16-bit little-endian PCM, passing `chunk_samples` samples at a
/// time; only the last chunk may be shorter.
fn decode(
    path: &Path,
    range: Option<(u64, u64)>,
    chunk_samples: usize,
    mut sink: impl FnMut(&[u8]),
) -> io::Result<()> {
    let stored = sample_count(path)?;
    let expected = range.map_or(stored, |(start, end)| end.saturating_sub(start));
    let mut command = flac_command();
    command.args(RAW_PCM).args(["--decode", "--stdout"]);
    if let Some((start, end)) = range {
        command.arg(format!("--skip={start}"));
        command.arg(format!("--until={end}"));
    }
    let mut flac = spawn(
        command
            .arg(std::path::absolute(path)?)
            .stdin(Stdio::null())
            .stdout(Stdio::piped()),
    )?;
    let mut stdout = flac.child.stdout.take().expect("piped flac output");
    let mut pcm = vec![0u8; chunk_samples * 2];
    let mut decoded = 0u64;
    let read = (|| {
        loop {
            let filled = read_full(&mut stdout, &mut pcm)?;
            decoded += filled as u64;
            if filled > 0 {
                sink(&pcm[..filled]);
            }
            if filled < pcm.len() {
                break;
            }
        }
        if decoded != expected.saturating_mul(2) {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "archived audio decoded to {} of {expected} samples",
                    decoded / 2
                ),
            ));
        }
        Ok(())
    })();
    drop(stdout);
    flac.finish().and(read)
}

fn parse_stream_header(header: &[u8; STREAM_HEADER_LEN]) -> Result<u64, String> {
    // "fLaC", then a STREAMINFO block: 4 header bytes, 10 bytes of block and
    // frame sizes, then sample rate (20 bits), channels - 1 (3 bits), bits
    // per sample - 1 (5 bits) and the sample count (36 bits).
    if &header[..4] != b"fLaC" || header[4] & 0x7f != 0 {
        return Err("not a FLAC stream".into());
    }
    let packed = u64::from_be_bytes(header[18..26].try_into().expect("eight bytes"));
    let sample_rate = packed >> 44;
    let channels = (packed >> 41 & 0x7) + 1;
    let bits_per_sample = (packed >> 36 & 0x1f) + 1;
    if sample_rate != u64::from(SAMPLE_RATE) || channels != 1 || bits_per_sample != 16 {
        return Err(format!(
            "unsupported archived audio format: {sample_rate} Hz, {channels} channel(s), {bits_per_sample} bits"
        ));
    }
    Ok(packed & ((1 << 36) - 1))
}

/// `SINGSTONE_FLAC`, else the copy bundled in the snap, else `flac` on `PATH`.
/// A snap without its bundled copy does not fall back: under confinement
/// `PATH` only reaches the base snap, and the error should name the real gap.
fn flac_command() -> Command {
    let executable = std::env::var_os("SINGSTONE_FLAC")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(std::env::var_os("SNAP")?).join("usr/bin/flac")))
        .unwrap_or_else(|| PathBuf::from("flac"));
    Command::new(executable)
}

/// A running `flac` whose diagnostics are drained in the background, so a
/// chatty failure can never block on a full pipe while we use its other pipes.
struct Flac {
    child: Child,
    diagnostics: JoinHandle<String>,
}

fn spawn(command: &mut Command) -> io::Result<Flac> {
    let mut child = command.stderr(Stdio::piped()).spawn().map_err(|error| {
        let program = command.get_program().to_string_lossy();
        if error.kind() == io::ErrorKind::NotFound {
            // Not `NotFound`: processing takes that to mean the track itself
            // is absent and carries on without it.
            io::Error::other(format!(
                "archived audio needs the {program} program, which is not installed"
            ))
        } else {
            io::Error::new(error.kind(), format!("could not start {program}: {error}"))
        }
    })?;
    let mut stderr = child.stderr.take().expect("piped flac diagnostics");
    let diagnostics = std::thread::spawn(move || {
        let mut kept = Vec::new();
        let _ = stderr.by_ref().take(16_384).read_to_end(&mut kept);
        let _ = io::copy(&mut stderr, &mut io::sink());
        String::from_utf8_lossy(&kept).trim().to_owned()
    });
    Ok(Flac { child, diagnostics })
}

impl Flac {
    /// Wait for the exit, turning a failure into the program's own message.
    fn finish(mut self) -> io::Result<()> {
        let status = self.child.wait()?;
        let diagnostics = self.diagnostics.join().unwrap_or_default();
        if status.success() {
            return Ok(());
        }
        Err(io::Error::other(if diagnostics.is_empty() {
            format!("flac failed: {status}")
        } else {
            format!("flac failed: {diagnostics}")
        }))
    }
}

/// Fill `buffer` unless the input ends first; returns the bytes read.
fn read_full(input: &mut impl Read, buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match input.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FORMAT_VERSION, Manifest, StreamInfo};

    /// A stopped, processed session whose microphone track holds `samples`
    /// followed by a stray partial sample.
    fn processed_session(label: &str, samples: &[f32]) -> Session {
        let root = std::env::temp_dir().join(format!(
            "singstone-archive-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let session = Session::create(root.join("session")).expect("create session");
        let stream = |enabled| StreamInfo {
            enabled,
            pipewire_node: None,
            error: None,
        };
        session
            .write_manifest(&Manifest {
                format_version: FORMAT_VERSION,
                state: SessionState::Stopped,
                started_wallclock: "2026-09-14T10:30:00+00:00".into(),
                sample_rate: SAMPLE_RATE,
                channels: 1,
                sample_format: "f32le".into(),
                mic: stream(true),
                system: stream(false),
                local_speaker: "Me".into(),
                screenshot_dir: None,
            })
            .expect("write manifest");
        let mut bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<u8>>();
        bytes.extend_from_slice(&[1, 2]);
        fs::write(session.audio_path(AudioSource::Mic), bytes).expect("write audio");
        fs::write(session.transcript_path(), b"").expect("write transcript");
        session
    }

    fn remove(session: Session) {
        fs::remove_dir_all(session.dir.parent().expect("fixture root")).expect("remove fixture");
    }

    fn speech_like(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| (index as f32 * 0.37).sin() * 1.2)
            .collect()
    }

    #[test]
    fn archiving_replaces_raw_audio_with_an_equivalent_copy() {
        let mut samples = speech_like(40_000);
        samples[7] = f32::NAN;
        let session = processed_session("roundtrip", &samples);
        let raw = session.audio_path(AudioSource::Mic);
        let archived = session.archived_audio_path(AudioSource::Mic);
        fs::write(
            session.words_metadata_path(),
            serde_json::to_vec(&serde_json::json!({
                "format_version": 1,
                "output_file": "words.jsonl",
                "output_sha256": "00",
                "mic_audio_sha256": models::sha256_file(&raw).expect("hash raw"),
                "system_audio_sha256": null,
                "whisper_model_sha256": "00",
                "language": "auto",
                "threads": 1,
            }))
            .expect("serialize metadata"),
        )
        .expect("write metadata");

        let summary = archive_session(&session).expect("archive");
        assert_eq!(summary.tracks, [AudioSource::Mic]);
        assert!(summary.bytes_after < summary.bytes_before);
        assert!(!raw.exists());
        assert!(session.is_archived());
        assert_eq!(
            fs::metadata(&archived)
                .expect("archived metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );

        // Full scale is clamped and NaN becomes silence; nothing else changes
        // beyond rounding to 16 bits.
        let expected = samples
            .iter()
            .map(|sample| f32::from(quantize(*sample)) / 32_768.0)
            .collect::<Vec<_>>();
        assert_eq!(expected[7], 0.0);
        assert!(expected.iter().all(|sample| sample.abs() <= 1.0));
        assert_eq!(
            session.read_audio(AudioSource::Mic).expect("read archived"),
            expected
        );
        assert_eq!(
            read_range(&archived, 100, 17_000).expect("read range"),
            expected[100..17_000]
        );
        assert_eq!(sample_count(&archived).expect("count"), 40_000);

        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(session.words_metadata_path()).expect("metadata"))
                .expect("parse metadata");
        assert_eq!(
            metadata["mic_audio_sha256"],
            models::sha256_file(&archived).expect("hash archived")
        );

        assert!(archive_session(&session).expect("repeat").tracks.is_empty());
        remove(session);
    }

    #[test]
    fn unprocessed_sessions_keep_their_raw_audio() {
        let session = processed_session("unprocessed", &speech_like(100));
        fs::remove_file(session.transcript_path()).expect("remove transcript");
        assert!(archive_session(&session).is_err());
        assert!(session.audio_path(AudioSource::Mic).is_file());
        assert!(!session.archived_audio_path(AudioSource::Mic).exists());
        remove(session);
    }

    #[test]
    fn damaged_archives_are_errors_not_short_audio() {
        let session = processed_session("damaged", &speech_like(40_000));
        archive_session(&session).expect("archive");
        let archived = session.archived_audio_path(AudioSource::Mic);
        let bytes = fs::read(&archived).expect("read archived");
        fs::write(&archived, &bytes[..bytes.len() / 2]).expect("truncate");
        assert!(session.read_audio(AudioSource::Mic).is_err());
        remove(session);
    }

    #[test]
    fn a_missing_flac_program_is_not_a_missing_track() {
        let error = spawn(&mut Command::new("/nonexistent/flac"))
            .err()
            .expect("spawn fails");
        assert_ne!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn stream_header_accepts_only_our_format() {
        fn header(sample_rate: u64, channels: u64, bits: u64, samples: u64) -> [u8; 42] {
            let mut header = [0u8; STREAM_HEADER_LEN];
            header[..4].copy_from_slice(b"fLaC");
            header[7] = 34;
            let packed = sample_rate << 44 | (channels - 1) << 41 | (bits - 1) << 36 | samples;
            header[18..26].copy_from_slice(&packed.to_be_bytes());
            header
        }
        assert_eq!(
            parse_stream_header(&header(16_000, 1, 16, 123_456)),
            Ok(123_456)
        );
        assert!(parse_stream_header(&header(44_100, 1, 16, 1)).is_err());
        assert!(parse_stream_header(&header(16_000, 2, 16, 1)).is_err());
        assert!(parse_stream_header(&header(16_000, 1, 24, 1)).is_err());
        assert!(parse_stream_header(&[0u8; STREAM_HEADER_LEN]).is_err());
    }
}
