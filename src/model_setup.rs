//! Snap model setup service and foreground wait/progress UI.

use crate::cli::Command;
use crate::models::{self, ModelEntry, ModelsLock};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::os::fd::AsFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const STATUS_FILE: &str = ".download-status.json";
const LOCK_FILE: &str = ".download.lock";
const ACTIVATED_ENV: &str = "SINGSTONE_MODEL_SETUP_ACTIVATED";
const SOCKET_ENV: &str = "SINGSTONE_MODEL_SETUP_SOCKET";
const MAX_MESSAGE_BYTES: u64 = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Status {
    #[serde(default)]
    request_id: String,
    state: String,
    current_model: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    error: Option<String>,
    updated_unix_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
struct ServiceMessage {
    state: String,
    error: Option<String>,
}

pub fn ensure_available(command: &Command) -> io::Result<()> {
    let Some(socket_path) = std::env::var_os(SOCKET_ENV).map(PathBuf::from) else {
        return Ok(());
    };
    let Some(models_dir) = std::env::var_os("SINGSTONE_MODELS_DIR").map(PathBuf::from) else {
        return Ok(());
    };
    let Some(lock_path) = std::env::var_os("SINGSTONE_MODELS_LOCK").map(PathBuf::from) else {
        return Ok(());
    };
    let lock = ModelsLock::load(&lock_path)?;
    let required_models = required_default_models(command, &models_dir, &lock);
    if required_models.is_empty()
        || required_models
            .iter()
            .all(|entry| model_ready(&models_dir, entry))
    {
        return Ok(());
    }

    fs::create_dir_all(&models_dir)?;
    let request_id = format!("{}-{}", std::process::id(), now_ms());
    let mut service = request_download(&socket_path, &request_id).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("could not contact model setup service: {error}"),
        )
    })?;

    wait_for_download(&models_dir, &request_id, &mut service)?;
    if required_models
        .iter()
        .all(|entry| model_ready(&models_dir, entry))
    {
        Ok(())
    } else {
        Err(io::Error::other(
            "model setup finished without the required models",
        ))
    }
}

fn request_download(socket_path: &Path, request_id: &str) -> io::Result<UnixStream> {
    let mut stream = UnixStream::connect(socket_path)?;
    writeln!(stream, "ensure {request_id}")?;
    stream.set_nonblocking(true)?;
    Ok(stream)
}

pub fn check_service() -> io::Result<()> {
    let socket_path = required_env_path(SOCKET_ENV)?;
    let mut stream = UnixStream::connect(socket_path)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    stream.write_all(b"ping\n")?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    for expected in ["accepted", "complete"] {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "model setup service stopped during its activation check",
            ));
        }
        let message: ServiceMessage = serde_json::from_str(line.trim_end()).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid model setup service response: {error}"),
            )
        })?;
        if message.state != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model setup service returned {:?}, expected {expected:?}",
                    message.state
                ),
            ));
        }
    }
    Ok(())
}

fn required_default_models<'a>(
    command: &'a Command,
    models_dir: &Path,
    lock: &'a ModelsLock,
) -> Vec<&'a ModelEntry> {
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
            lock.models
                .iter()
                .find(|entry| entry.purpose == purpose && path == models_dir.join(&entry.filename))
        })
        .collect()
}

fn model_ready(models_dir: &Path, entry: &ModelEntry) -> bool {
    verified(&models_dir.join(&entry.filename), entry.size, &entry.sha256).unwrap_or(false)
}

fn wait_for_download(
    models_dir: &Path,
    request_id: &str,
    service: &mut UnixStream,
) -> io::Result<()> {
    let status_path = models_dir.join(STATUS_FILE);
    let terminal = io::stderr().is_terminal();
    let mut last_update = Instant::now();
    let mut last_status_ms = 0;
    let mut last_line = String::new();
    let mut accepted = false;
    let mut response = Vec::new();
    let mut last_activation_activity = Instant::now();
    let mut last_other_status_ms = 0;
    loop {
        let (messages, closed) = read_service_messages(service, &mut response)?;
        for message in messages {
            match message.state.as_str() {
                "accepted" => {
                    accepted = true;
                    last_update = Instant::now();
                }
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
                        message
                            .error
                            .unwrap_or_else(|| "model download failed".into()),
                    ));
                }
                state => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("model setup service returned unknown state {state:?}"),
                    ));
                }
            }
        }
        if closed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "model setup service stopped without reporting a result",
            ));
        }
        if let Ok(data) = fs::read(&status_path)
            && let Ok(status) = serde_json::from_slice::<Status>(&data)
        {
            if status.request_id == request_id {
                if status.updated_unix_ms != last_status_ms {
                    last_status_ms = status.updated_unix_ms;
                    last_update = Instant::now();
                }
                render_progress(&status, terminal, &mut last_line)?;
            } else if !accepted
                && matches!(status.state.as_str(), "starting" | "downloading")
                && status.updated_unix_ms != last_other_status_ms
                && now_ms().saturating_sub(status.updated_unix_ms) <= 2_000
            {
                last_other_status_ms = status.updated_unix_ms;
                last_activation_activity = Instant::now();
            }
        }
        if !accepted && last_activation_activity.elapsed() > Duration::from_secs(30) {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "model setup service did not accept the request",
            ));
        }
        if accepted && last_update.elapsed() > Duration::from_secs(120) {
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

fn read_service_messages(
    service: &mut UnixStream,
    buffer: &mut Vec<u8>,
) -> io::Result<(Vec<ServiceMessage>, bool)> {
    let mut closed = false;
    loop {
        let mut chunk = [0; 1024];
        match service.read(&mut chunk) {
            Ok(0) => {
                closed = true;
                break;
            }
            Ok(count) => {
                buffer.extend_from_slice(&chunk[..count]);
                if buffer.len() as u64 > MAX_MESSAGE_BYTES {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "model setup service response is too large",
                    ));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
            Err(error) => return Err(error),
        }
    }

    let mut messages = Vec::new();
    while let Some(end) = buffer.iter().position(|byte| *byte == b'\n') {
        let line: Vec<u8> = buffer.drain(..=end).collect();
        let message = serde_json::from_slice(&line[..line.len() - 1]).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid model setup service response: {error}"),
            )
        })?;
        messages.push(message);
    }
    Ok((messages, closed))
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
    if std::env::var_os(ACTIVATED_ENV).is_some() {
        return serve_activated_download();
    }
    download_request(&format!("{}-{}", std::process::id(), now_ms()))
}

fn serve_activated_download() -> io::Result<()> {
    let expected_socket = required_env_path(SOCKET_ENV)?;
    let listener = UnixListener::from(io::stdin().as_fd().try_clone_to_owned()?);
    let actual_socket = listener.local_addr()?;
    if actual_socket.as_pathname() != Some(expected_socket.as_path()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model setup received an unexpected activation socket",
        ));
    }
    let (mut client, _) = listener.accept()?;
    client.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut request = String::new();
    let count = BufReader::new(&client).take(128).read_line(&mut request)?;
    if count == 0 {
        return Ok(());
    }
    if request == "ping\n" {
        write_service_message(&mut client, "accepted", None)?;
        return write_service_message(&mut client, "complete", None);
    }
    let Some(request_id) = request
        .strip_suffix('\n')
        .and_then(|request| request.strip_prefix("ensure "))
        .filter(|request_id| {
            !request_id.is_empty()
                && request_id.len() <= 64
                && request_id
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'-')
        })
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid model setup request",
        ));
    };

    write_service_message(&mut client, "accepted", None)?;
    let result = download_request(request_id);
    let (state, error) = match &result {
        Ok(()) => ("complete", None),
        Err(error) => ("error", Some(error.to_string())),
    };
    let _ = write_service_message(&mut client, state, error);
    result
}

fn write_service_message(
    client: &mut UnixStream,
    state: &str,
    error: Option<String>,
) -> io::Result<()> {
    serde_json::to_writer(
        &mut *client,
        &ServiceMessage {
            state: state.into(),
            error,
        },
    )
    .map_err(io::Error::other)?;
    client.write_all(b"\n")?;
    client.flush()
}

fn download_request(request_id: &str) -> io::Result<()> {
    let models_dir = required_env_path("SINGSTONE_MODELS_DIR")?;
    let lock_path = required_env_path("SINGSTONE_MODELS_LOCK")?;
    fs::create_dir_all(&models_dir)?;
    let lock_file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(models_dir.join(LOCK_FILE))?;
    lock_file
        .try_lock()
        .map_err(|_| io::Error::other("another model setup process is already running"))?;

    let status_path = models_dir.join(STATUS_FILE);
    write_status(
        &status_path,
        &Status {
            request_id: request_id.into(),
            state: "starting".into(),
            current_model: "preparing".into(),
            downloaded_bytes: 0,
            total_bytes: 0,
            error: None,
            updated_unix_ms: now_ms(),
        },
    )?;
    let result = download_inner(&models_dir, &lock_path, &status_path, request_id);
    if let Err(error) = &result {
        let _ = write_status(
            &status_path,
            &Status {
                request_id: request_id.into(),
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

fn download_inner(
    models_dir: &Path,
    lock_path: &Path,
    status_path: &Path,
    request_id: &str,
) -> io::Result<()> {
    let lock = ModelsLock::load(lock_path)?;
    let total = lock.models.iter().map(ModelEntry::artifact_size).sum();
    download_all(&lock, models_dir, status_path, total, request_id)
}

fn download_all(
    lock: &ModelsLock,
    models_dir: &Path,
    status_path: &Path,
    total: u64,
    request_id: &str,
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
                download_artifact(entry, &artifact, status_path, completed, total, request_id)?;
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
            request_id: request_id.into(),
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
    request_id: &str,
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
                request_id: request_id.into(),
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
        .stdin(Stdio::null())
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
    use std::thread;

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
            request_id: "test".into(),
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
    fn recognizes_each_default_model_for_the_same_purpose() {
        let models_dir = PathBuf::from("/models");
        let make_entry = |name: &str, filename: &str| ModelEntry {
            name: name.into(),
            purpose: "transcription".into(),
            upstream: "fixture".into(),
            url: "file:///fixture".into(),
            revision: "1".into(),
            license: "MIT".into(),
            filename: filename.into(),
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            download_size: None,
            download_sha256: None,
            archive_member: None,
        };
        let lock = ModelsLock {
            models: vec![
                make_entry("swedish", "swedish.bin"),
                make_entry("multilingual", "multilingual.bin"),
            ],
        };
        let mut args = crate::cli::ProcessArgs::for_session("/session".into());
        args.whisper_model = Some(models_dir.join("multilingual.bin"));
        let command = Command::Process(args);

        let required = required_default_models(&command, &models_dir, &lock);

        assert_eq!(required.len(), 1);
        assert_eq!(required[0].name, "multilingual");
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
        download_all(
            &lock,
            &models_dir,
            &models_dir.join(STATUS_FILE),
            3,
            "test-request",
        )
        .expect("download model");
        assert_eq!(
            fs::read(models_dir.join("model.bin")).expect("read installed model"),
            b"abc"
        );
        assert!(!models_dir.join("model.bin.download.part").exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn requests_setup_through_the_activation_socket() {
        let root = fixture_dir();
        let socket = root.join("model-setup.sock");
        let listener = UnixListener::bind(&socket).expect("bind fixture socket");
        let server = thread::spawn(move || {
            let (mut client, _) = listener.accept().expect("accept request");
            let mut request = String::new();
            BufReader::new(&client)
                .read_line(&mut request)
                .expect("read request");
            assert_eq!(request, "ensure 123-456\n");
            write_service_message(&mut client, "accepted", None).expect("accept response");
            write_service_message(&mut client, "complete", None).expect("complete response");
        });

        let mut client = request_download(&socket, "123-456").expect("request setup");
        wait_for_download(&root, "123-456", &mut client).expect("wait for setup");
        server.join().expect("join fixture server");
        fs::remove_dir_all(root).expect("remove fixture");
    }
}
