use crate::models;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelIdentity {
    pub name: String,
    pub sha256: String,
    pub dimension: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerRecord {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerDatabase {
    pub embedding_model: EmbeddingModelIdentity,
    pub speakers: BTreeMap<String, SpeakerRecord>,
}

impl SpeakerDatabase {
    pub fn empty(identity: EmbeddingModelIdentity) -> Self {
        Self {
            embedding_model: identity,
            speakers: BTreeMap::new(),
        }
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid speaker database {}: {error}", path.display()),
            )
        })
    }

    pub fn load_checked(path: &Path, expected: &EmbeddingModelIdentity) -> io::Result<Self> {
        let database = Self::load(path)?;
        database.validate_identity(expected)?;
        Ok(database)
    }

    pub fn validate_identity(&self, expected: &EmbeddingModelIdentity) -> io::Result<()> {
        if self
            .embedding_model
            .sha256
            .eq_ignore_ascii_case(&expected.sha256)
            && self.embedding_model.dimension == expected.dimension
        {
            return Ok(());
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "speaker database uses embedding model {} (sha256 {}, dimension {}), but configured model is {} (sha256 {}, dimension {}); re-enroll speakers with the configured model",
                self.embedding_model.name,
                self.embedding_model.sha256,
                self.embedding_model.dimension,
                expected.name,
                expected.sha256,
                expected.dimension
            ),
        ))
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
        let tmp = crate::format::jsonl::tmp_path(path);
        {
            let mut file = crate::session::create_private_file(&tmp)?;
            serde_json::to_writer_pretty(&mut file, self)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        fs::rename(tmp, path)
    }
}

pub fn identity(model: &Path, dimension: usize) -> io::Result<EmbeddingModelIdentity> {
    let name = model
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "embedding model filename is not valid UTF-8",
            )
        })?
        .to_string();
    Ok(EmbeddingModelIdentity {
        name,
        sha256: models::sha256_file(model)?,
        dimension,
    })
}

pub fn default_path() -> PathBuf {
    models::xdg_dir("XDG_DATA_HOME", ".local/share")
        .join("singstone")
        .join("speakers.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    fn identity_with(hash: &str, dimension: usize) -> EmbeddingModelIdentity {
        EmbeddingModelIdentity {
            name: "model".into(),
            sha256: hash.into(),
            dimension,
        }
    }

    #[test]
    fn rejects_fingerprint_and_dimension_mismatch() {
        let database = SpeakerDatabase::empty(identity_with("abc", 192));
        let hash_error = database
            .validate_identity(&identity_with("def", 192))
            .unwrap_err();
        assert!(hash_error.to_string().contains("re-enroll"));
        let dimension_error = database
            .validate_identity(&identity_with("abc", 256))
            .unwrap_err();
        assert!(dimension_error.to_string().contains("dimension 256"));
    }

    #[test]
    fn save_does_not_chmod_existing_parent() {
        let root =
            std::env::temp_dir().join(format!("singstone-speaker-db-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::DirBuilder::new()
            .mode(0o755)
            .create(&root)
            .expect("create parent");
        SpeakerDatabase::empty(identity_with("abc", 192))
            .save(&root.join("speakers.json"))
            .expect("save database");
        assert_eq!(
            fs::metadata(&root).expect("parent metadata").mode() & 0o777,
            0o755
        );
        fs::remove_dir_all(root).expect("remove temp directory");
    }
}
