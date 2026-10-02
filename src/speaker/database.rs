use crate::models;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelIdentity {
    pub name: String,
    pub sha256: String,
    pub dimension: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerRecord {
    pub embeddings: Vec<Vec<f32>>,
    #[serde(skip)]
    pub(crate) centroid: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerDatabase {
    pub format_version: u32,
    pub embedding_model: EmbeddingModelIdentity,
    pub speakers: BTreeMap<String, SpeakerRecord>,
}

#[derive(Deserialize)]
struct LegacySpeakerDatabase {
    embedding_model: EmbeddingModelIdentity,
    speakers: BTreeMap<String, SpeakerRecord>,
}

impl SpeakerDatabase {
    pub fn empty(identity: EmbeddingModelIdentity) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            embedding_model: identity,
            speakers: BTreeMap::new(),
        }
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid speaker database {}: {error}", path.display()),
            )
        })?;
        if value.get("format_version").is_none() {
            let legacy: LegacySpeakerDatabase = serde_json::from_value(value).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid speaker database {}: {error}", path.display()),
                )
            })?;
            let LegacySpeakerDatabase {
                embedding_model,
                speakers,
            } = legacy;
            drop(speakers);
            let backup = next_v1_backup_path(path)?;
            fs::rename(path, &backup)?;
            eprintln!(
                "speaker database {} used the old format; moved it to {} and started empty",
                path.display(),
                backup.display()
            );
            return Ok(Self::empty(embedding_model));
        }
        let mut database: Self = serde_json::from_value(value).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid speaker database {}: {error}", path.display()),
            )
        })?;
        if database.format_version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported speaker database format version {} in {}; expected {}",
                    database.format_version,
                    path.display(),
                    FORMAT_VERSION
                ),
            ));
        }
        database.validate_vectors(path)?;
        database.refresh_centroids();
        Ok(database)
    }

    pub fn load_checked(path: &Path, expected: &EmbeddingModelIdentity) -> io::Result<Self> {
        let mut database = Self::load(path)?;
        if database
            .speakers
            .values()
            .all(|speaker| speaker.embeddings.is_empty())
        {
            database.embedding_model = expected.clone();
            return Ok(database);
        }
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
                "speaker database uses embedding model {} (sha256 {}, dimension {}), but configured model is {} (sha256 {}, dimension {}); relearn speakers with the configured model",
                self.embedding_model.name,
                self.embedding_model.sha256,
                self.embedding_model.dimension,
                expected.name,
                expected.sha256,
                expected.dimension
            ),
        ))
    }

    pub fn forget_matching(&mut self, name: &str, embedding: &[f32], min_similarity: f32) -> usize {
        let Some(record) = self.speakers.get_mut(name) else {
            return 0;
        };
        let before = record.embeddings.len();
        record.embeddings.retain(|candidate| {
            !crate::speaker::embedding::cosine(candidate, embedding)
                .is_some_and(|similarity| similarity >= min_similarity)
        });
        record.centroid = crate::speaker::embedding::mean_normalized(&record.embeddings);
        before - record.embeddings.len()
    }

    pub fn refresh_centroids(&mut self) {
        for speaker in self.speakers.values_mut() {
            speaker.centroid = crate::speaker::embedding::mean_normalized(&speaker.embeddings);
        }
    }

    fn validate_vectors(&self, path: &Path) -> io::Result<()> {
        for (name, speaker) in &self.speakers {
            for (index, vector) in speaker.embeddings.iter().enumerate() {
                let norm_squared = vector
                    .iter()
                    .map(|value| f64::from(*value) * f64::from(*value))
                    .sum::<f64>();
                if vector.len() != self.embedding_model.dimension
                    || !norm_squared.is_finite()
                    || (norm_squared - 1.0).abs() > 1e-3
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "invalid embedding {index} for speaker {name:?} in {}: expected a unit vector of dimension {}",
                            path.display(),
                            self.embedding_model.dimension
                        ),
                    ));
                }
            }
        }
        Ok(())
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

fn next_v1_backup_path(path: &Path) -> io::Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "speaker database path has no file name",
        )
    })?;
    let mut suffix = 1usize;
    loop {
        let mut backup_name = file_name.to_os_string();
        backup_name.push(".v1.bak");
        if suffix > 1 {
            backup_name.push(format!(".{suffix}"));
        }
        let backup = path.with_file_name(backup_name);
        if !backup.try_exists()? {
            return Ok(backup);
        }
        suffix = suffix
            .checked_add(1)
            .ok_or_else(|| io::Error::other("too many speaker database backups"))?;
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
        assert!(hash_error.to_string().contains("relearn"));
        let dimension_error = database
            .validate_identity(&identity_with("abc", 256))
            .unwrap_err();
        assert!(dimension_error.to_string().contains("dimension 256"));
    }

    #[test]
    fn forgets_matching_embeddings_and_keeps_speaker_records() {
        let mut database = SpeakerDatabase::empty(identity_with("abc", 3));
        database.speakers.insert(
            "Alice".into(),
            SpeakerRecord {
                embeddings: vec![vec![0.9999, 0.014, 0.0], vec![0.0, 1.0, 0.0]],
                centroid: None,
            },
        );
        database.speakers.insert(
            "Bob".into(),
            SpeakerRecord {
                embeddings: vec![vec![1.0, 0.0, 0.0]],
                centroid: None,
            },
        );
        database.refresh_centroids();
        assert!(database.speakers["Alice"].centroid.is_some());
        assert!(database.speakers["Bob"].centroid.is_some());

        assert_eq!(
            database.forget_matching("Alice", &[1.0, 0.0, 0.0], 0.999),
            1
        );
        assert_eq!(
            database.speakers["Alice"].embeddings,
            vec![vec![0.0, 1.0, 0.0]]
        );
        assert_eq!(
            database.forget_matching("Unknown", &[1.0, 0.0, 0.0], 0.999),
            0
        );
        assert_eq!(database.forget_matching("Bob", &[1.0, 0.0, 0.0], 0.999), 1);
        assert!(database.speakers.contains_key("Bob"));
        assert!(database.speakers["Bob"].embeddings.is_empty());
        assert!(database.speakers["Bob"].centroid.is_none());
    }

    #[test]
    fn backs_up_unversioned_database_and_starts_empty() {
        let root =
            std::env::temp_dir().join(format!("singstone-speaker-db-v1-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        let first_backup = root.join("speakers.json.v1.bak");
        fs::write(&first_backup, b"existing backup").expect("write existing backup");
        let legacy = r#"{
  "embedding_model":{"name":"old","sha256":"abc","dimension":2},
  "speakers":{"Alice":{"embeddings":[[1.0,0.0]]}}
}"#;
        fs::write(&path, legacy).expect("write v1 database");

        let database = SpeakerDatabase::load(&path).expect("migrate v1 database");

        assert_eq!(database.format_version, FORMAT_VERSION);
        assert!(database.speakers.is_empty());
        assert!(!path.exists());
        assert_eq!(
            fs::read(root.join("speakers.json.v1.bak.2")).expect("read backup"),
            legacy.as_bytes()
        );
        assert_eq!(
            fs::read(first_backup).expect("read existing backup"),
            b"existing backup"
        );
        fs::remove_dir_all(root).expect("remove fixture directory");
    }

    #[test]
    fn saves_version_without_cache_and_loads_centroids() {
        let root =
            std::env::temp_dir().join(format!("singstone-speaker-db-v2-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        let mut database = SpeakerDatabase::empty(identity_with("abc", 2));
        database.speakers.insert(
            "Alice".into(),
            SpeakerRecord {
                embeddings: vec![vec![1.0, 0.0]],
                centroid: None,
            },
        );
        database.refresh_centroids();

        let value = serde_json::to_value(&database).expect("serialize database");
        database.save(&path).expect("save database");
        let loaded = SpeakerDatabase::load(&path).expect("load database");

        assert_eq!(value["format_version"], FORMAT_VERSION);
        assert!(value["speakers"]["Alice"].get("centroid").is_none());
        assert!(loaded.speakers["Alice"].centroid.is_some());
        fs::remove_dir_all(root).expect("remove fixture directory");
    }

    #[test]
    fn rejects_non_unit_and_wrong_dimension_v2_vectors() {
        let root = std::env::temp_dir().join(format!(
            "singstone-speaker-db-invalid-v2-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        fs::write(
            &path,
            r#"{
  "format_version":2,
  "embedding_model":{"name":"model","sha256":"abc","dimension":2},
  "speakers":{"Alice":{"embeddings":[[10.0,0.0]]}}
}"#,
        )
        .expect("write invalid database");
        assert!(
            SpeakerDatabase::load(&path)
                .expect_err("reject non-unit vector")
                .to_string()
                .contains("unit vector of dimension 2")
        );

        fs::write(
            &path,
            r#"{
  "format_version":2,
  "embedding_model":{"name":"model","sha256":"abc","dimension":2},
  "speakers":{"Alice":{"embeddings":[[1.0]]}}
}"#,
        )
        .expect("write wrong-dimension database");
        assert!(
            SpeakerDatabase::load(&path)
                .expect_err("reject wrong dimension")
                .to_string()
                .contains("unit vector of dimension 2")
        );
        fs::remove_dir_all(root).expect("remove fixture directory");
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
