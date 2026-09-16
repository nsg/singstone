use crate::format::jsonl;
use crate::models;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiConfig {
    pub meetings_dir: PathBuf,
    pub screenshots_dir: PathBuf,
    pub local_speaker: String,
}

impl Default for GuiConfig {
    fn default() -> Self {
        let home = std::env::var_os("SNAP_REAL_HOME")
            .or_else(|| std::env::var_os("HOME"))
            .map_or_else(|| PathBuf::from("."), PathBuf::from);
        Self {
            meetings_dir: std::env::var_os("SINGSTONE_MEETINGS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join("Meetings")),
            screenshots_dir: std::env::var_os("SINGSTONE_SCREENSHOTS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join("Pictures/Screenshots")),
            local_speaker: "Me".into(),
        }
    }
}

impl GuiConfig {
    pub fn load() -> io::Result<Self> {
        let path = path();
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid GUI settings {}: {error}", path.display()),
                )
            }),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error),
        }
    }

    pub fn save(&self) -> io::Result<()> {
        let path = path();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
        let temporary = jsonl::tmp_path(&path);
        {
            let mut file = crate::session::create_private_file(&temporary)?;
            serde_json::to_writer_pretty(&mut file, self)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        fs::rename(temporary, path)
    }
}

fn path() -> PathBuf {
    if let Some(common) = std::env::var_os("SNAP_USER_COMMON") {
        return PathBuf::from(common).join("gui.json");
    }
    models::xdg_dir("XDG_CONFIG_HOME", ".config")
        .join("singstone")
        .join("gui.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_have_a_nonempty_identity() {
        assert!(!GuiConfig::default().local_speaker.trim().is_empty());
    }
}
