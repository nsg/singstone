use serde_json::Value;
use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> io::Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "singstone-{label}-{}-{nonce}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn enabled() -> bool {
    if std::env::var("SINGSTONE_PW_TEST").is_err() {
        eprintln!("skipping PipeWire test; set SINGSTONE_PW_TEST=1 to enable");
        false
    } else {
        true
    }
}

fn recorder(output: &Path, extra: &[&str]) -> io::Result<Child> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_singstone"));
    command
        .arg("record")
        .arg("--output-dir")
        .arg(output)
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
}

fn wait_for_child(child: &mut Child, timeout: Duration) -> io::Result<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            child.kill()?;
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "child process timed out",
            ));
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn session_dir(output: &Path) -> PathBuf {
    fs::read_dir(output)
        .expect("read output directory")
        .map(|entry| entry.expect("read output entry").path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("session-"))
        })
        .expect("session directory")
}

fn wait_for_session(output: &Path) -> PathBuf {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(entries) = fs::read_dir(output) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.join("manifest.json").is_file() {
                    return path;
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "recorder did not create a session"
        );
        thread::sleep(Duration::from_millis(50));
    }
}

fn manifest(session: &Path) -> Value {
    serde_json::from_slice(&fs::read(session.join("manifest.json")).expect("read manifest"))
        .expect("parse manifest")
}

fn read_audio(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).expect("read audio");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn write_signal_fixture(path: &Path) -> io::Result<()> {
    const SAMPLE_RATE: u32 = 16_000;
    const SECONDS: u32 = 6;
    let sample_count = SAMPLE_RATE * SECONDS;
    let data_size = sample_count * 2;
    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    file.write_all(&(SAMPLE_RATE * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    let mut state = 0x6d2b_79f5u32;
    for index in 0..sample_count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let active = index % (SAMPLE_RATE / 2) < SAMPLE_RATE / 5;
        let sample = if active {
            ((state >> 16) as u16 as i16) / 2
        } else {
            0
        };
        file.write_all(&sample.to_le_bytes())?;
    }
    file.sync_all()
}

fn correlation_offset(left: &[f32], right: &[f32]) -> i64 {
    const STEP: usize = 8;
    const MAX_LAG: i64 = 100;
    let left: Vec<f32> = left.iter().step_by(STEP).copied().collect();
    let right: Vec<f32> = right.iter().step_by(STEP).copied().collect();
    let mut best = (f64::NEG_INFINITY, 0);
    for lag in -MAX_LAG..=MAX_LAG {
        let left_start = lag.max(0) as usize;
        let right_start = (-lag).max(0) as usize;
        let count = (left.len() - left_start).min(right.len() - right_start);
        let score = left[left_start..left_start + count]
            .iter()
            .zip(&right[right_start..right_start + count])
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>();
        if score > best.0 {
            best = (score, lag);
        }
    }
    best.1 * STEP as i64
}

#[test]
fn records_aligned_microphone_and_system_audio() {
    if !enabled() {
        return;
    }
    let output = TempDir::new("both").expect("create temp dir");
    let fixture = output.path().join("signal.wav");
    write_signal_fixture(&fixture).expect("write signal fixture");
    let mut record = recorder(
        output.path(),
        &[
            "--mic",
            "test-mic",
            "--system",
            "test-sink",
            "--duration",
            "12",
        ],
    )
    .expect("start recorder");
    wait_for_session(output.path());
    thread::sleep(Duration::from_secs(1));
    let mut mic_play = Command::new("pw-play")
        .args(["--target", "test-mic-in"])
        .arg(&fixture)
        .spawn()
        .expect("play microphone fixture");
    let mut system_play = Command::new("pw-play")
        .args(["--target", "test-sink"])
        .arg(&fixture)
        .spawn()
        .expect("play system fixture");
    assert!(record.wait().expect("wait for recorder").success());
    let mic_status = wait_for_child(&mut mic_play, Duration::from_secs(3));
    let system_status = wait_for_child(&mut system_play, Duration::from_secs(3));
    assert!(mic_status.expect("wait for mic playback").success());
    assert!(system_status.expect("wait for system playback").success());

    let session = session_dir(output.path());
    let manifest = manifest(&session);
    assert_eq!(manifest["state"], "stopped");
    let mic = read_audio(&session.join("audio/mic.f32le"));
    let system = read_audio(&session.join("audio/system.f32le"));
    let expected = 12 * 16_000;
    assert!(
        mic.len().abs_diff(expected) <= 8_000,
        "mic samples: {}",
        mic.len()
    );
    assert!(
        system.len().abs_diff(expected) <= 8_000,
        "system samples: {}",
        system.len()
    );
    assert!(
        mic.iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
            > 0.05
    );
    assert!(
        system
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
            > 0.05
    );
    let offset = correlation_offset(&mic, &system);
    eprintln!("mic/system correlation offset: {offset} samples");
    assert!(offset.abs() <= 800, "alignment offset: {offset} samples");
}

#[test]
fn records_system_only() {
    if !enabled() {
        return;
    }
    let output = TempDir::new("system").expect("create temp dir");
    let status = recorder(
        output.path(),
        &["--mic", "none", "--system", "test-sink", "--duration", "2"],
    )
    .expect("start recorder")
    .wait()
    .expect("wait for recorder");
    assert!(status.success());
    let session = session_dir(output.path());
    let manifest = manifest(&session);
    assert_eq!(manifest["state"], "stopped");
    assert_eq!(manifest["mic"]["enabled"], false);
    assert!(!session.join("audio/mic.f32le").exists());
    assert!(session.join("audio/system.f32le").is_file());
}

#[test]
fn copies_only_matching_screenshots_privately() {
    if !enabled() {
        return;
    }
    let output = TempDir::new("screens-output").expect("create output dir");
    let watched = TempDir::new("screens-input").expect("create watched dir");
    let mut record = recorder(
        output.path(),
        &[
            "--mic",
            "none",
            "--system",
            "test-sink",
            "--duration",
            "2",
            "--screenshots",
            watched.path().to_str().expect("UTF-8 watched path"),
            "--screenshot-ext",
            "png",
        ],
    )
    .expect("start recorder");
    wait_for_session(output.path());
    thread::sleep(Duration::from_millis(500));
    fs::write(watched.path().join("capture.PNG"), b"not decoded").expect("write PNG");
    fs::write(watched.path().join("ignore.txt"), b"ignored").expect("write text");
    assert!(record.wait().expect("wait for recorder").success());

    let session = session_dir(output.path());
    let lines =
        fs::read_to_string(session.join("screenshots.jsonl")).expect("read screenshot index");
    let entries: Vec<Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse screenshot entry"))
        .collect();
    assert_eq!(entries.len(), 1);
    let time_ms = entries[0]["time_ms"].as_u64().expect("integer timestamp");
    assert!(
        (100..2_500).contains(&time_ms),
        "screenshot time: {time_ms}"
    );
    let relative = entries[0]["file"].as_str().expect("screenshot path");
    let copied = session.join(relative);
    assert!(copied.is_file());
    assert_eq!(
        copied.metadata().expect("copied metadata").mode() & 0o777,
        0o600
    );
}

#[test]
fn sigkill_leaves_recording_manifest_and_raw_files() {
    if !enabled() {
        return;
    }
    let output = TempDir::new("crash").expect("create temp dir");
    let mut record = recorder(
        output.path(),
        &[
            "--mic",
            "test-mic",
            "--system",
            "test-sink",
            "--duration",
            "30",
        ],
    )
    .expect("start recorder");
    let session = wait_for_session(output.path());
    thread::sleep(Duration::from_secs(1));
    assert!(
        Command::new("kill")
            .args(["-KILL", &record.id().to_string()])
            .status()
            .expect("send SIGKILL")
            .success()
    );
    let status = record.wait().expect("reap recorder");
    assert!(!status.success());
    assert_eq!(manifest(&session)["state"], "recording");
    assert!(session.join("audio/mic.f32le").is_file());
    assert!(session.join("audio/system.f32le").is_file());
}
