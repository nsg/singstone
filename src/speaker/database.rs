use crate::models;
use crate::speaker::embedding::{cosine, mean_normalized};
use crate::types::AudioSource;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

pub const FORMAT_VERSION: u32 = 3;
/// Cosine above which two stored vectors count as the same sample.
pub const SAME_VECTOR_SIMILARITY: f32 = 0.999;

pub fn canonical_name_key(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn normalized_name(name: &str) -> String {
    name.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelIdentity {
    pub name: String,
    pub sha256: String,
    pub dimension: usize,
}

/// One learned voice sample. Samples imported from older formats have no
/// session and count as settled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// Unix seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learned_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<AudioSource>,
    /// Restored by the user, so tidying leaves it alone.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub kept_by_user: bool,
    pub embedding: Vec<f32>,
}

impl Sample {
    pub fn settled(embedding: Vec<f32>) -> Self {
        Self {
            learned_at: None,
            session: None,
            source: None,
            kept_by_user: false,
            embedding,
        }
    }

    pub fn learned(embedding: Vec<f32>, session: &str, source: AudioSource, at: u64) -> Self {
        Self {
            learned_at: Some(at),
            session: Some(session.to_owned()),
            source: Some(source),
            kept_by_user: false,
            embedding,
        }
    }
}

impl AsRef<[f32]> for Sample {
    fn as_ref(&self) -> &[f32] {
        &self.embedding
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum PruneReason {
    SoundsLike { rival: String },
    Isolated,
    User,
}

/// A sample taken out of recognition, kept for a while so it can be restored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrunedSample {
    #[serde(flatten)]
    pub reason: PruneReason,
    /// Unix seconds.
    pub pruned_at: u64,
    #[serde(flatten)]
    pub sample: Sample,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerRecord {
    /// Sessions this voice learned samples in, in the order they first did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sessions: Vec<String>,
    pub samples: Vec<Sample>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pruned: Vec<PrunedSample>,
    #[serde(skip)]
    pub(crate) centroid: Option<Vec<f32>>,
}

impl SpeakerRecord {
    pub fn settled(vectors: Vec<Vec<f32>>) -> Self {
        Self {
            samples: vectors.into_iter().map(Sample::settled).collect(),
            ..Self::default()
        }
    }

    pub fn holds(&self, vector: &[f32]) -> bool {
        self.samples
            .iter()
            .any(|sample| same_vector(&sample.embedding, vector))
    }

    /// Adds a sample unless an equal one is already stored. A removed copy
    /// of it is dropped, since learning it again is an explicit choice.
    pub fn learn(&mut self, sample: Sample) -> bool {
        if self.holds(&sample.embedding) {
            return false;
        }
        self.pruned
            .retain(|pruned| !same_vector(&pruned.sample.embedding, &sample.embedding));
        if let Some(session) = &sample.session
            && !self.sessions.contains(session)
        {
            self.sessions.push(session.clone());
        }
        self.samples.push(sample);
        true
    }
}

fn same_vector(left: &[f32], right: &[f32]) -> bool {
    cosine(left, right).is_some_and(|similarity| similarity >= SAME_VECTOR_SIMILARITY)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerDatabase {
    pub format_version: u32,
    pub embedding_model: EmbeddingModelIdentity,
    pub speakers: BTreeMap<String, SpeakerRecord>,
}

/// Exclusive use of a database file, released on drop.
pub struct DatabaseLock {
    _file: fs::File,
}

/// Formats 1 (unversioned) and 2 stored bare vectors.
#[derive(Deserialize)]
struct BareVectorDatabase {
    embedding_model: EmbeddingModelIdentity,
    speakers: BTreeMap<String, BareVectorRecord>,
}

#[derive(Deserialize)]
struct BareVectorRecord {
    embeddings: Vec<Vec<f32>>,
}

/// Why a database read from disk must be written back.
struct Rewrite {
    /// The older format version that was converted, if any.
    from: Option<u32>,
    dropped: usize,
    merged: usize,
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
        let (database, rewrite) = Self::read(path)?;
        if rewrite.is_none() {
            return Ok(database);
        }
        let lock = Self::lock(path)?;
        Self::load_locked(path, &lock)
    }

    /// Loads the database while the caller holds its lock.
    pub fn load_locked(path: &Path, _lock: &DatabaseLock) -> io::Result<Self> {
        let (database, rewrite) = Self::read(path)?;
        if let Some(rewrite) = rewrite {
            database.rewrite(path, rewrite)?;
        }
        Ok(database)
    }

    /// Waits for exclusive use of the database file. The GUI, its worker
    /// processes and the command line all change it, so each change loads
    /// and saves under this lock to keep the others' changes.
    pub fn lock(path: &Path) -> io::Result<DatabaseLock> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
        let mut name = path.as_os_str().to_owned();
        name.push(".lock");
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(PathBuf::from(name))?;
        file.lock()?;
        Ok(DatabaseLock { _file: file })
    }

    /// Loads, changes and saves the database under its lock.
    pub fn update<T>(path: &Path, change: impl FnOnce(&mut Self) -> T) -> io::Result<T> {
        let lock = Self::lock(path)?;
        let mut database = Self::load_locked(path, &lock)?;
        let result = change(&mut database);
        database.save(path)?;
        Ok(result)
    }

    /// Reads the database as `load` does, without converting the file.
    pub fn peek(path: &Path) -> io::Result<Self> {
        Self::read(path).map(|(database, _)| database)
    }

    fn read(path: &Path) -> io::Result<(Self, Option<Rewrite>)> {
        let bytes = fs::read(path)?;
        let invalid = |error: serde_json::Error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid speaker database {}: {error}", path.display()),
            )
        };
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(invalid)?;
        let version = match value.get("format_version") {
            None => 1,
            Some(version) => version.as_u64().unwrap_or(0),
        };
        let mut database = match version {
            1 | 2 => {
                let old: BareVectorDatabase = serde_json::from_value(value).map_err(invalid)?;
                Self {
                    format_version: FORMAT_VERSION,
                    embedding_model: old.embedding_model,
                    speakers: old
                        .speakers
                        .into_iter()
                        .map(|(name, record)| (name, SpeakerRecord::settled(record.embeddings)))
                        .collect(),
                }
            }
            3 => serde_json::from_value(value).map_err(invalid)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported speaker database format version {} in {}; expected {}",
                        value["format_version"],
                        path.display(),
                        FORMAT_VERSION
                    ),
                ));
            }
        };
        let dropped = if version == 1 {
            database.drop_invalid_vectors()
        } else {
            database.validate_vectors(path)?;
            0
        };
        let merged = database.merge_duplicate_names();
        database.refresh_centroids();
        let from = (version < FORMAT_VERSION as u64).then_some(version as u32);
        let rewrite = (from.is_some() || merged > 0).then_some(Rewrite {
            from,
            dropped,
            merged,
        });
        Ok((database, rewrite))
    }

    fn rewrite(&self, path: &Path, rewrite: Rewrite) -> io::Result<()> {
        let Some(version) = rewrite.from else {
            return self.save(path);
        };
        let backup = next_backup_path(path, version)?;
        fs::copy(path, &backup)?;
        self.save(path)?;
        let imported: usize = self
            .speakers
            .values()
            .map(|speaker| speaker.samples.len())
            .sum();
        eprintln!(
            "speaker database {} used format version {version}; backed it up to {} and imported {imported} voice sample(s) for {} speaker(s){}{}",
            path.display(),
            backup.display(),
            self.speakers.len(),
            if rewrite.dropped == 0 {
                String::new()
            } else {
                format!(", dropping {} invalid vector(s)", rewrite.dropped)
            },
            if rewrite.merged == 0 {
                String::new()
            } else {
                format!(", merged {} duplicate name(s)", rewrite.merged)
            }
        );
        Ok(())
    }

    /// Adopts `expected` while no samples are stored, else requires it.
    pub fn checked(mut self, expected: &EmbeddingModelIdentity) -> io::Result<Self> {
        if self
            .speakers
            .values()
            .all(|speaker| speaker.samples.is_empty())
        {
            self.embedding_model = expected.clone();
            return Ok(self);
        }
        self.validate_identity(expected)?;
        Ok(self)
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

    pub fn resolve_name(&self, typed: &str) -> String {
        let normalized = normalized_name(typed);
        let key = canonical_name_key(&normalized);
        self.speakers
            .keys()
            .find(|name| canonical_name_key(name) == key)
            .cloned()
            .unwrap_or(normalized)
    }

    /// Drops the samples, removed ones included, that match `embedding`;
    /// returns how many were in use.
    pub fn forget_matching(&mut self, name: &str, embedding: &[f32], min_similarity: f32) -> usize {
        let Some(record) = self.speakers.get_mut(name) else {
            return 0;
        };
        let matches = |candidate: &[f32]| {
            cosine(candidate, embedding).is_some_and(|similarity| similarity >= min_similarity)
        };
        let before = record.samples.len();
        record.samples.retain(|sample| !matches(&sample.embedding));
        record
            .pruned
            .retain(|pruned| !matches(&pruned.sample.embedding));
        record.centroid = mean_normalized(&record.samples);
        before - record.samples.len()
    }

    pub fn refresh_centroids(&mut self) {
        for speaker in self.speakers.values_mut() {
            speaker.centroid = mean_normalized(&speaker.samples);
        }
    }

    fn merge_duplicate_names(&mut self) -> usize {
        let mut groups = BTreeMap::<String, Vec<(String, SpeakerRecord)>>::new();
        for (name, record) in std::mem::take(&mut self.speakers) {
            groups
                .entry(canonical_name_key(&name))
                .or_default()
                .push((name, record));
        }

        let mut merged = 0;
        for records in groups.into_values() {
            let winner = records
                .iter()
                .enumerate()
                .fold(0, |winner, (index, (_, record))| {
                    if record.samples.len() > records[winner].1.samples.len() {
                        index
                    } else {
                        winner
                    }
                });
            let mut records = records;
            let (display_name, mut record) = records.remove(winner);
            merged += records.len();
            for (_, duplicate) in records {
                for session in duplicate.sessions {
                    if !record.sessions.contains(&session) {
                        record.sessions.push(session);
                    }
                }
                for sample in duplicate.samples {
                    if !record.holds(&sample.embedding) {
                        record.samples.push(sample);
                    }
                }
                record.pruned.extend(duplicate.pruned);
            }
            self.speakers.insert(display_name, record);
        }
        merged
    }

    fn drop_invalid_vectors(&mut self) -> usize {
        let dimension = self.embedding_model.dimension;
        let mut dropped = 0;
        for speaker in self.speakers.values_mut() {
            let before = speaker.samples.len();
            speaker
                .samples
                .retain(|sample| is_unit_vector(&sample.embedding, dimension));
            dropped += before - speaker.samples.len();
        }
        dropped
    }

    fn validate_vectors(&self, path: &Path) -> io::Result<()> {
        for (name, speaker) in &self.speakers {
            let vectors = speaker
                .samples
                .iter()
                .chain(speaker.pruned.iter().map(|pruned| &pruned.sample));
            for (index, sample) in vectors.enumerate() {
                if !is_unit_vector(&sample.embedding, self.embedding_model.dimension) {
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
            let mut writer = BufWriter::new(crate::session::create_private_file(&tmp)?);
            let mut serializer =
                serde_json::Serializer::with_formatter(&mut writer, SampleLines::default());
            self.serialize(&mut serializer)?;
            writer.write_all(b"\n")?;
            writer
                .into_inner()
                .map_err(|error| error.into_error())?
                .sync_all()?;
        }
        fs::rename(tmp, path)
    }
}

/// Pretty-prints the database, but writes each sample on a single line.
#[derive(Default)]
struct SampleLines {
    depth: usize,
    has_value: bool,
}

/// Nesting depth of a sample: database, speakers, record, list, sample.
const SAMPLE_DEPTH: usize = 5;

impl SampleLines {
    fn inline(&self) -> bool {
        self.depth >= SAMPLE_DEPTH
    }

    fn open<W: ?Sized + Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        self.depth += 1;
        self.has_value = false;
        writer.write_all(bracket)
    }

    fn close<W: ?Sized + Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        let inline = self.inline();
        self.depth -= 1;
        if self.has_value && !inline {
            self.newline(writer)?;
        }
        writer.write_all(bracket)
    }

    fn item<W: ?Sized + Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if !first {
            writer.write_all(b",")?;
        }
        if self.inline() {
            Ok(())
        } else {
            self.newline(writer)
        }
    }

    fn newline<W: ?Sized + Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b"\n")?;
        for _ in 0..self.depth {
            writer.write_all(b"  ")?;
        }
        Ok(())
    }
}

impl serde_json::ser::Formatter for SampleLines {
    fn begin_array<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b"[")
    }

    fn end_array<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"]")
    }

    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.item(writer, first)
    }

    fn end_array_value<W: ?Sized + Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b"{")
    }

    fn end_object<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"}")
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.item(writer, first)
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(if self.inline() { b":" } else { b": " })
    }

    fn end_object_value<W: ?Sized + Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }
}

fn is_unit_vector(vector: &[f32], dimension: usize) -> bool {
    let norm_squared = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    vector.len() == dimension && norm_squared.is_finite() && (norm_squared - 1.0).abs() <= 1e-3
}

fn next_backup_path(path: &Path, version: u32) -> io::Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "speaker database path has no file name",
        )
    })?;
    let mut suffix = 1usize;
    loop {
        let mut backup_name = file_name.to_os_string();
        backup_name.push(format!(".v{version}.bak"));
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

    fn vectors(record: &SpeakerRecord) -> Vec<Vec<f32>> {
        record
            .samples
            .iter()
            .map(|sample| sample.embedding.clone())
            .collect()
    }

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
            SpeakerRecord::settled(vec![vec![0.9999, 0.014, 0.0], vec![0.0, 1.0, 0.0]]),
        );
        database.speakers.insert(
            "Bob".into(),
            SpeakerRecord::settled(vec![vec![1.0, 0.0, 0.0]]),
        );
        database.refresh_centroids();
        assert!(database.speakers["Alice"].centroid.is_some());
        assert!(database.speakers["Bob"].centroid.is_some());

        assert_eq!(
            database.forget_matching("Alice", &[1.0, 0.0, 0.0], 0.999),
            1
        );
        assert_eq!(
            vectors(&database.speakers["Alice"]),
            vec![vec![0.0, 1.0, 0.0]]
        );
        assert_eq!(
            database.forget_matching("Unknown", &[1.0, 0.0, 0.0], 0.999),
            0
        );
        assert_eq!(database.forget_matching("Bob", &[1.0, 0.0, 0.0], 0.999), 1);
        assert!(database.speakers.contains_key("Bob"));
        assert!(database.speakers["Bob"].samples.is_empty());
        assert!(database.speakers["Bob"].centroid.is_none());
    }

    #[test]
    fn backs_up_unversioned_database_and_imports_its_vectors() {
        let root =
            std::env::temp_dir().join(format!("singstone-speaker-db-v1-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        let first_backup = root.join("speakers.json.v1.bak");
        fs::write(&first_backup, b"existing backup").expect("write existing backup");
        let legacy = r#"{
  "embedding_model":{"name":"old","sha256":"abc","dimension":2},
  "speakers":{"Alice":{"embeddings":[[1.0,0.0],[3.0,4.0]]},"Bob":{"embeddings":[[0.0,1.0]]}}
}"#;
        fs::write(&path, legacy).expect("write v1 database");

        let database = SpeakerDatabase::load(&path).expect("migrate v1 database");

        assert_eq!(database.format_version, FORMAT_VERSION);
        assert_eq!(vectors(&database.speakers["Alice"]), vec![vec![1.0, 0.0]]);
        assert_eq!(vectors(&database.speakers["Bob"]), vec![vec![0.0, 1.0]]);
        assert!(database.speakers["Alice"].centroid.is_some());
        let saved: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("read migrated database"))
                .expect("parse migrated database");
        assert_eq!(saved["format_version"], FORMAT_VERSION);
        assert_eq!(saved["speakers"]["Bob"]["samples"][0]["embedding"][1], 1.0);
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
    fn backs_up_v2_database_and_keeps_its_samples_as_settled() {
        let root =
            std::env::temp_dir().join(format!("singstone-speaker-db-v2-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        let v2 = r#"{
  "format_version":2,
  "embedding_model":{"name":"model","sha256":"abc","dimension":2},
  "speakers":{"Alice":{"embeddings":[[1.0,0.0],[0.6,0.8]]}}
}"#;
        fs::write(&path, v2).expect("write v2 database");

        let database = SpeakerDatabase::load(&path).expect("migrate v2 database");

        assert_eq!(
            database.speakers["Alice"].samples,
            vec![
                Sample::settled(vec![1.0, 0.0]),
                Sample::settled(vec![0.6, 0.8])
            ]
        );
        assert_eq!(
            fs::read(root.join("speakers.json.v2.bak")).expect("read backup"),
            v2.as_bytes()
        );
        let saved = fs::read_to_string(&path).expect("read migrated database");
        assert!(saved.contains("\n        {\"embedding\":[1.0,0.0]},\n"));
        let reloaded = SpeakerDatabase::load(&path).expect("load v3 database");
        assert_eq!(reloaded.format_version, 3);
        assert_eq!(
            reloaded.speakers["Alice"].samples,
            database.speakers["Alice"].samples
        );
        fs::remove_dir_all(root).expect("remove fixture directory");
    }

    #[test]
    fn load_merges_duplicate_names_with_larger_spelling_and_deduplicates_vectors() {
        let root = std::env::temp_dir().join(format!(
            "singstone-speaker-db-duplicate-names-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        fs::write(
            &path,
            r#"{
  "format_version":3,
  "embedding_model":{"name":"model","sha256":"abc","dimension":3},
  "speakers":{
    " Laura ":{"sessions":["b"],"samples":[{"session":"b","embedding":[0.0,0.0,1.0]}]},
    "Laura":{"sessions":["a"],"samples":[{"embedding":[1.0,0.0,0.0]},{"embedding":[0.0,1.0,0.0]}]},
    "laura":{"samples":[{"embedding":[1.0,0.0,0.0]}]}
  }
}"#,
        )
        .expect("write duplicate database");

        let database = SpeakerDatabase::load(&path).expect("load duplicate database");

        assert_eq!(database.speakers.len(), 1);
        assert_eq!(
            vectors(&database.speakers["Laura"]),
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0]
            ]
        );
        assert_eq!(database.speakers["Laura"].sessions, ["a", "b"]);
        let saved: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("read merged database"))
                .expect("parse merged database");
        assert_eq!(
            saved["speakers"]
                .as_object()
                .expect("speaker object")
                .keys()
                .collect::<Vec<_>>(),
            vec!["Laura"]
        );
        fs::remove_dir_all(root).expect("remove fixture directory");
    }

    #[test]
    fn resolve_name_reuses_existing_spelling() {
        let mut database = SpeakerDatabase::empty(identity_with("abc", 2));
        database
            .speakers
            .insert("Laura Mc Kay".into(), SpeakerRecord::default());

        assert_eq!(database.resolve_name("  laura\tMc  KAY "), "Laura Mc Kay");
    }

    #[test]
    fn saves_version_without_cache_and_loads_centroids() {
        let root =
            std::env::temp_dir().join(format!("singstone-speaker-db-v3-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).expect("create fixture directory");
        let path = root.join("speakers.json");
        let mut database = SpeakerDatabase::empty(identity_with("abc", 2));
        database
            .speakers
            .insert("Alice".into(), SpeakerRecord::settled(vec![vec![1.0, 0.0]]));
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
