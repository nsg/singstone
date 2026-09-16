//! Trusted model manifest (`models.lock`) and SHA-256 verification.
//!
//! Model files are untrusted, executable-like inputs to native parsers. Every
//! model passed on the command line must match an entry in the lock file
//! unless `--allow-unverified-models` is given.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    /// One of `transcription`, `diarization-segmentation`, `speaker-embedding`.
    pub purpose: String,
    pub upstream: String,
    pub url: String,
    pub revision: String,
    pub license: String,
    pub filename: String,
    pub size: u64,
    pub sha256: String,
    /// Size of the downloaded artifact. Defaults to `size` for direct files.
    #[serde(default)]
    pub download_size: Option<u64>,
    /// SHA-256 of the downloaded artifact. Defaults to `sha256` for direct files.
    #[serde(default)]
    pub download_sha256: Option<String>,
    /// Exact member to extract when the download is a tar archive.
    #[serde(default)]
    pub archive_member: Option<String>,
}

impl ModelEntry {
    pub fn artifact_size(&self) -> u64 {
        self.download_size.unwrap_or(self.size)
    }

    pub fn artifact_sha256(&self) -> &str {
        self.download_sha256.as_deref().unwrap_or(&self.sha256)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelsLock {
    pub models: Vec<ModelEntry>,
}

impl ModelsLock {
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        serde_json::from_slice(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Default lock location: `$XDG_CONFIG_HOME/singstone/models.lock`.
pub fn default_lock_path() -> PathBuf {
    xdg_dir("XDG_CONFIG_HOME", ".config")
        .join("singstone")
        .join("models.lock")
}

pub fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => {
            let home = std::env::var_os("HOME").unwrap_or_default();
            PathBuf::from(home).join(fallback)
        }
    }
}

/// Verify `model` against the lock file (explicit path, else default path).
/// With `allow_unverified`, a missing lock or entry only prints a warning.
pub fn verify_model(
    model: &Path,
    expected_purpose: &str,
    lock_path: Option<&Path>,
    allow_unverified: bool,
) -> io::Result<()> {
    let lock_path = lock_path
        .map(Path::to_path_buf)
        .unwrap_or_else(default_lock_path);
    let lock = match ModelsLock::load(&lock_path) {
        Ok(l) => l,
        Err(e) if e.kind() == io::ErrorKind::NotFound && allow_unverified => {
            eprintln!(
                "warning: no models.lock at {}; using unverified model {}",
                lock_path.display(),
                model.display()
            );
            return Ok(());
        }
        Err(e) => {
            return Err(io::Error::new(
                e.kind(),
                format!("cannot read models.lock {}: {e}", lock_path.display()),
            ));
        }
    };
    let actual_size = fs::metadata(model)?.len();
    let filename = model.file_name().and_then(|name| name.to_str());
    let named_entries = lock
        .models
        .iter()
        .filter(|entry| filename == Some(entry.filename.as_str()))
        .collect::<Vec<_>>();
    if !named_entries.is_empty() {
        let Some(entry) = named_entries
            .iter()
            .copied()
            .find(|entry| purpose_matches(&entry.purpose, expected_purpose))
        else {
            return verification_failure(
                allow_unverified,
                format!(
                    "model {} is listed for purpose {:?}, not {expected_purpose:?}",
                    model.display(),
                    named_entries
                        .iter()
                        .map(|entry| entry.purpose.as_str())
                        .collect::<Vec<_>>()
                ),
            );
        };
        if entry.size != actual_size {
            return verification_failure(
                allow_unverified,
                format!(
                    "model {} has size {actual_size}, but models.lock expects {} bytes for {}",
                    model.display(),
                    entry.size,
                    entry.name
                ),
            );
        }
    } else {
        let expected_sizes = lock
            .models
            .iter()
            .filter(|entry| purpose_matches(&entry.purpose, expected_purpose))
            .map(|entry| entry.size)
            .collect::<Vec<_>>();
        if !expected_sizes.contains(&actual_size) {
            return verification_failure(
                allow_unverified,
                format!(
                    "model {} has size {actual_size}, which is not listed for purpose {expected_purpose:?} in {}",
                    model.display(),
                    lock_path.display()
                ),
            );
        }
    }
    let digest = sha256_file(model)?;
    match lock.models.iter().find(|entry| {
        entry.size == actual_size
            && purpose_matches(&entry.purpose, expected_purpose)
            && entry.sha256.eq_ignore_ascii_case(&digest)
    }) {
        Some(entry) => {
            eprintln!("verified model {} ({})", entry.name, model.display());
            Ok(())
        }
        None if allow_unverified => {
            eprintln!(
                "warning: model {} (sha256 {digest}) is not in models.lock",
                model.display()
            );
            Ok(())
        }
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "model {} (sha256 {digest}) is not listed in {}; add it or pass --allow-unverified-models",
                model.display(),
                lock_path.display()
            ),
        )),
    }
}

fn purpose_matches(actual: &str, expected: &str) -> bool {
    actual == expected
}

fn verification_failure(allow_unverified: bool, message: String) -> io::Result<()> {
    if allow_unverified {
        eprintln!("warning: {message}");
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::DirBuilderExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "singstone-models-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .expect("create temp directory");
        let model = dir.join("model.bin");
        fs::write(&model, b"abc").expect("write model");
        let lock = ModelsLock {
            models: vec![ModelEntry {
                name: "fixture".into(),
                purpose: "transcription".into(),
                upstream: String::new(),
                url: String::new(),
                revision: String::new(),
                license: String::new(),
                filename: "model.bin".into(),
                size: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
                download_size: None,
                download_sha256: None,
                archive_member: None,
            }],
        };
        let lock_path = dir.join("models.lock");
        fs::write(
            &lock_path,
            serde_json::to_vec(&lock).expect("serialize lock"),
        )
        .expect("write lock");
        (model, lock_path)
    }

    #[test]
    fn verifies_size_hash_and_purpose() {
        let (model, lock) = fixture();
        verify_model(&model, "transcription", Some(&lock), false).expect("verify model");
        let error = verify_model(&model, "speaker-embedding", Some(&lock), false)
            .expect_err("reject wrong purpose");
        assert!(error.to_string().contains("not \"speaker-embedding\""));
        fs::remove_dir_all(model.parent().expect("fixture parent")).expect("remove fixture");
    }

    #[test]
    fn rejects_size_before_hash() {
        let (model, lock) = fixture();
        fs::write(&model, b"abcd").expect("change model size");
        let error = verify_model(&model, "transcription", Some(&lock), false)
            .expect_err("reject wrong size");
        assert!(error.to_string().contains("has size 4"));
        fs::remove_dir_all(model.parent().expect("fixture parent")).expect("remove fixture");
    }
}
