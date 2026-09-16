//! Snap model setup service and foreground wait/progress UI.

use crate::cli::Command;
use crate::models::{self, ModelEntry, ModelsLock};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const STATUS_FILE: &str = ".download-status.json";
const LOCK_FILE: &str = ".download.lock";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Status {
    state: String,
    current_model: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    error: Option<String>,
    updated_unix_ms: u128,
}

pub fn ensure_available(command: &Command) -> io::Result<()> {
    let Some(service) = std::env::var_os("SINGSTONE_MODEL_SETUP_SERVICE") else {
        return Ok(());
    };
    let Some(models_dir) = std::env::var_os("SINGSTONE_MODELS_DIR").map(PathBuf::from) else {
        return Ok(());
    };
    let Some(lock_path) = std::env::var_os("SINGSTONE_MODELS_LOCK").map(PathBuf::from) else {
        return Ok(());
    };
    let lock = ModelsLock::load(&lock_path)?;
    let purposes = required_default_purposes(command, &models_dir, &lock);
    if purposes.is_empty()
        || purposes
            .iter()
            .all(|purpose| purpose_ready(&lock, &models_dir, purpose))
    {
        return Ok(());
    }

    fs::create_dir_all(&models_dir)?;
    let _ = fs::remove_file(models_dir.join(STATUS_FILE));
    let started = now_ms();
    let result = ProcessCommand::new("snapctl")
        .args(["start", &service.to_string_lossy()])
        .output()?;
    if !result.status.success() {
        return Err(io::Error::other(format!(
            "could not start model setup service: {}",
            String::from_utf8_lossy(&result.stderr).trim()
        )));
    }

    wait_for_download(&models_dir, started)?;
    if purposes
        .iter()
        .all(|purpose| purpose_ready(&lock, &models_dir, purpose))
    {
        Ok(())
    } else {
        Err(io::Error::other(
            "model setup finished without the required models",
        ))
    }
}

fn required_default_purposes<'a>(
    command: &'a Command,
    models_dir: &Path,
    lock: &'a ModelsLock,
) -> Vec<&'a str> {
    let mut requested: Vec<(&str, &Path)> = Vec::new();
    match command {
        Command::Process(args) => {
            if !args.skip_transcription
                && let Some(path) = args.whisper_model.as_deref()
            {
                requested.push(("transcription", path));
            }
            if !args.no_diarize {
                if let Some(path) = args.segmentation_model.as_deref() {
                    requested.push(("diarization-segmentation", path));
                }
                if let Some(path) = args.embedding_model.as_deref() {
                    requested.push(("speaker-embedding", path));
                }
            }
        }
        Command::Transcribe(args) => requested.push(("transcription", &args.whisper_model)),
        Command::Diarize(args) => {
            requested.push(("diarization-segmentation", &args.segmentation_model));
            requested.push(("speaker-embedding", &args.embedding_model));
        }
        Command::Recognize(args) => requested.push(("speaker-embedding", &args.embedding_model)),
        Command::Enroll(args) => requested.push(("speaker-embedding", &args.embedding_model)),
        _ => {}
    }
    requested
        .into_iter()
        .filter_map(|(purpose, path)| {
            let entry = lock.models.iter().find(|entry| entry.purpose == purpose)?;
            (path == models_dir.join(&entry.filename)).then_some(purpose)
        })
        .collect()
}

fn purpose_ready(lock: &ModelsLock, models_dir: &Path, purpose: &str) -> bool {
    lock.models
        .iter()
        .find(|entry| entry.purpose == purpose)
        .is_some_and(|entry| {
            verified(&models_dir.join(&entry.filename), entry.size, &entry.sha256).unwrap_or(false)
        })
}

fn wait_for_download(models_dir: &Path, started: u128) -> io::Result<()> {
    let status_path = models_dir.join(STATUS_FILE);
    let terminal = io::stderr().is_terminal();
    let mut last_update = Instant::now();
    let mut last_status_ms = 0;
    let mut last_line = String::new();
    loop {
        if let Ok(data) = fs::read(&status_path)
            && let Ok(status) = serde_json::from_slice::<Status>(&data)
            && status.updated_unix_ms >= started.saturating_sub(2_000)
        {
            if status.updated_unix_ms != last_status_ms {
                last_status_ms = status.updated_unix_ms;
                last_update = Instant::now();
            }
            render_progress(&status, terminal, &mut last_line)?;
            match status.state.as_str() {
                "complete" => {
                    if terminal {
                        eprintln!();
                    }
                    return Ok(());
                }
                "error" => {
                    if terminal {
                        eprintln!();
                    }
                    return Err(io::Error::other(
                        status
                            .error
                            .unwrap_or_else(|| "model download failed".into()),
                    ));
                }
                _ => {}
            }
        }
        if last_update.elapsed() > Duration::from_secs(120) {
            if terminal {
                eprintln!();
            }
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "model setup stopped reporting progress",
            ));
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn render_progress(status: &Status, terminal: bool, last_line: &mut String) -> io::Result<()> {
    let percent = status
        .downloaded_bytes
        .saturating_mul(100)
        .checked_div(status.total_bytes)
        .unwrap_or_else(|| u64::from(status.state == "complete") * 100)
        .min(100);
    let filled = (percent * 24 / 100) as usize;
    let bar = format!("{}{}", "=".repeat(filled), " ".repeat(24 - filled));
    let line = format!(
        "Downloading models [{bar}] {percent:3}% {}/{} MiB — {}",
        status.downloaded_bytes / (1024 * 1024),
        status.total_bytes / (1024 * 1024),
        status.current_model
    );
    if terminal {
        eprint!("\r\x1b[2K{line}");
        io::stderr().flush()
    } else {
        let key = format!("{}:{}", status.current_model, percent / 10);
        if key == *last_line && status.state == "downloading" {
            return Ok(());
        }
        eprintln!("{line}");
        *last_line = key;
        Ok(())
    }
}

pub fn download() -> io::Result<()> {
    let models_dir = required_env_path("SINGSTONE_MODELS_DIR")?;
    let lock_path = required_env_path("SINGSTONE_MODELS_LOCK")?;
    fs::create_dir_all(&models_dir)?;
    let status_path = models_dir.join(STATUS_FILE);
    write_status(
        &status_path,
        &Status {
            state: "starting".into(),
            current_model: "preparing".into(),
            downloaded_bytes: 0,
            total_bytes: 0,
            error: None,
            updated_unix_ms: now_ms(),
        },
    )?;
    let result = download_inner(&models_dir, &lock_path, &status_path);
    if let Err(error) = &result {
        let _ = write_status(
            &status_path,
            &Status {
                state: "error".into(),
                current_model: "setup failed".into(),
                downloaded_bytes: 0,
                total_bytes: 0,
                error: Some(error.to_string()),
                updated_unix_ms: now_ms(),
            },
        );
    }
    result
}

fn download_inner(models_dir: &Path, lock_path: &Path, status_path: &Path) -> io::Result<()> {
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(models_dir.join(LOCK_FILE))?;
    lock_file
        .try_lock()
        .map_err(|_| io::Error::other("another model setup process is already running"))?;

    let lock = ModelsLock::load(lock_path)?;
    let total = lock.models.iter().map(ModelEntry::artifact_size).sum();
    download_all(&lock, models_dir, status_path, total)
}

fn download_all(
    lock: &ModelsLock,
    models_dir: &Path,
    status_path: &Path,
    total: u64,
) -> io::Result<()> {
    let mut completed = 0;
    for entry in &lock.models {
        let destination = models_dir.join(&entry.filename);
        if verified(&destination, entry.size, &entry.sha256)? {
            completed += entry.artifact_size();
            continue;
        }
        let artifact = models_dir.join(format!("{}.download.part", entry.filename));
        let mut artifact_valid =
            verified(&artifact, entry.artifact_size(), entry.artifact_sha256())?;
        if !artifact_valid
            && fs::metadata(&artifact).is_ok_and(|m| m.len() >= entry.artifact_size())
        {
            fs::remove_file(&artifact)?;
        }
        if !artifact_valid {
            for attempt in 0..2 {
                download_artifact(entry, &artifact, status_path, completed, total)?;
                if verified(&artifact, entry.artifact_size(), entry.artifact_sha256())? {
                    artifact_valid = true;
                    break;
                }
                let _ = fs::remove_file(&artifact);
                if attempt == 1 {
                    break;
                }
            }
        }
        if !artifact_valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "downloaded artifact for {} failed size or SHA-256 verification",
                    entry.name
                ),
            ));
        }
        if let Some(member) = &entry.archive_member {
            extract_archive(entry, &artifact, member, &destination)?;
            fs::remove_file(&artifact)?;
        } else {
            fs::rename(&artifact, &destination)?;
        }
        if !verified(&destination, entry.size, &entry.sha256)? {
            let _ = fs::remove_file(&destination);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "installed model {} failed size or SHA-256 verification",
                    entry.name
                ),
            ));
        }
        completed += entry.artifact_size();
    }
    write_status(
        status_path,
        &Status {
            state: "complete".into(),
            current_model: "ready".into(),
            downloaded_bytes: total,
            total_bytes: total,
            error: None,
            updated_unix_ms: now_ms(),
        },
    )
}

fn download_artifact(
    entry: &ModelEntry,
    artifact: &Path,
    status_path: &Path,
    completed: u64,
    total: u64,
) -> io::Result<()> {
    let mut child = ProcessCommand::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--continue-at",
            "-",
        ])
        .arg("--output")
        .arg(artifact)
        .arg(&entry.url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    loop {
        let current = fs::metadata(artifact)
            .map_or(0, |m| m.len())
            .min(entry.artifact_size());
        write_status(
            status_path,
            &Status {
                state: "downloading".into(),
                current_model: entry.name.clone(),
                downloaded_bytes: completed + current,
                total_bytes: total,
                error: None,
                updated_unix_ms: now_ms(),
            },
        )?;
        if let Some(exit) = child.try_wait()? {
            if exit.success() {
                return Ok(());
            }
            let output = child.wait_with_output()?;
            return Err(io::Error::other(format!(
                "download of {} failed: {}",
                entry.name,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn extract_archive(
    entry: &ModelEntry,
    archive: &Path,
    member: &str,
    destination: &Path,
) -> io::Result<()> {
    let temporary = destination.with_extension("install.part");
    let output = File::create(&temporary)?;
    let result = ProcessCommand::new("tar")
        .args(["-xjf"])
        .arg(archive)
        .arg("-O")
        .arg(member)
        .stdout(Stdio::from(output))
        .stderr(Stdio::piped())
        .output()?;
    if !result.status.success() {
        let _ = fs::remove_file(&temporary);
        return Err(io::Error::other(format!(
            "could not extract {}: {}",
            entry.name,
            String::from_utf8_lossy(&result.stderr).trim()
        )));
    }
    fs::rename(temporary, destination)
}

fn verified(path: &Path, size: u64, sha256: &str) -> io::Result<bool> {
    if !file_has_size(path, size) {
        return Ok(false);
    }
    Ok(models::sha256_file(path)?.eq_ignore_ascii_case(sha256))
}

fn file_has_size(path: &Path, size: u64) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == size)
}

fn write_status(path: &Path, status: &Status) -> io::Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let data = serde_json::to_vec(status).map_err(io::Error::other)?;
    fs::write(&temporary, data)?;
    fs::rename(temporary, path)
}

fn required_env_path(name: &str) -> io::Result<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{name} is not set")))
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "singstone-download-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).expect("create fixture");
        path
    }

    #[test]
    fn progress_is_clamped() {
        let status = Status {
            state: "complete".into(),
            current_model: "ready".into(),
            downloaded_bytes: 2,
            total_bytes: 1,
            error: None,
            updated_unix_ms: 0,
        };
        let mut last = String::new();
        render_progress(&status, false, &mut last).expect("render progress");
        assert_eq!(last, "ready:10");
    }

    #[test]
    fn installs_only_a_verified_download() {
        let root = fixture_dir();
        let source = root.join("source.bin");
        fs::write(&source, b"abc").expect("write source");
        let models_dir = root.join("models");
        fs::create_dir(&models_dir).expect("create model cache");
        let entry = ModelEntry {
            name: "fixture".into(),
            purpose: "transcription".into(),
            upstream: "fixture".into(),
            url: format!("file://{}", source.display()),
            revision: "1".into(),
            license: "MIT".into(),
            filename: "model.bin".into(),
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            download_size: None,
            download_sha256: None,
            archive_member: None,
        };
        let lock = ModelsLock {
            models: vec![entry],
        };
        download_all(&lock, &models_dir, &models_dir.join(STATUS_FILE), 3).expect("download model");
        assert_eq!(
            fs::read(models_dir.join("model.bin")).expect("read installed model"),
            b"abc"
        );
        assert!(!models_dir.join("model.bin.download.part").exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }
}
