//! Runs `singstone dictate` against the headless PipeWire test microphone.
//! Kept apart from the recording tests, which feed the same microphone in parallel.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

fn dictation_fixtures() -> Option<(PathBuf, PathBuf)> {
    std::env::var_os("SINGSTONE_PW_TEST")?;
    let models = PathBuf::from(std::env::var_os("SINGSTONE_TEST_MODELS")?);
    let samples = PathBuf::from(std::env::var_os("SINGSTONE_TEST_SAMPLES")?);
    Some((models.join("ggml-base.en.bin"), samples.join("jfk.wav")))
}

fn dictation(model: &Path) -> io::Result<Child> {
    Command::new(env!("CARGO_BIN_EXE_singstone"))
        .args(["dictate", "--mic", "test-mic", "--allow-unverified-models"])
        .arg("--whisper-model")
        .arg(model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
}

#[test]
fn dictation_prints_the_spoken_text_after_the_stop_line() {
    let Some((model, speech)) = dictation_fixtures() else {
        eprintln!(
            "skipped: SINGSTONE_PW_TEST, SINGSTONE_TEST_MODELS and SINGSTONE_TEST_SAMPLES are required"
        );
        return;
    };
    let mut dictate = dictation(&model).expect("start dictation");
    thread::sleep(Duration::from_secs(1));
    let played = Command::new("pw-play")
        .args(["--target", "test-mic-in"])
        .arg(&speech)
        .status()
        .expect("play speech");
    assert!(played.success());
    thread::sleep(Duration::from_millis(500));
    dictate
        .stdin
        .take()
        .expect("dictation stdin")
        .write_all(b"en\n")
        .expect("send stop line");
    let output = dictate.wait_with_output().expect("wait for dictation");
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).expect("UTF-8 text");
    assert!(
        text.to_lowercase().contains("ask not what your country"),
        "dictated text: {text:?}"
    );
}

#[test]
fn dictation_end_of_input_cancels_without_output() {
    let Some((model, _)) = dictation_fixtures() else {
        eprintln!(
            "skipped: SINGSTONE_PW_TEST, SINGSTONE_TEST_MODELS and SINGSTONE_TEST_SAMPLES are required"
        );
        return;
    };
    let mut dictate = dictation(&model).expect("start dictation");
    thread::sleep(Duration::from_secs(1));
    drop(dictate.stdin.take());
    let output = dictate.wait_with_output().expect("wait for dictation");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}
