use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Read every non-empty line of a JSONL file.
pub fn read_all<T: DeserializeOwned>(path: &Path) -> io::Result<Vec<T>> {
    let reader = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    for (n, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {}", path.display(), n + 1, e),
            )
        })?;
        out.push(value);
    }
    Ok(out)
}

/// Write all items to `path.tmp` and atomically rename over `path`.
pub fn write_all_atomic<T: Serialize>(path: &Path, items: &[T]) -> io::Result<()> {
    let tmp = tmp_path(path);
    {
        let mut w = BufWriter::new(crate::session::create_private_file(&tmp)?);
        for item in items {
            serde_json::to_writer(&mut w, item)?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&tmp, path)
}

pub fn tmp_path(path: &Path) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    s.into()
}

/// Append-only JSONL writer that flushes after every line so a crash loses
/// at most the line being written.
pub struct JsonlAppender {
    file: File,
}

impl JsonlAppender {
    pub fn create(path: &Path) -> io::Result<Self> {
        Ok(Self {
            file: crate::session::create_private_file(path)?,
        })
    }

    pub fn append<T: Serialize>(&mut self, item: &T) -> io::Result<()> {
        let mut line = serde_json::to_vec(item)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.file.flush()
    }
}
