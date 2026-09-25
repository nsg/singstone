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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_dir: Option<PathBuf>,
    pub swedish_transcription: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dark_mode: Option<bool>,
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
            context_dir: std::env::var_os("SINGSTONE_CONTEXT_DIR").map(PathBuf::from),
            swedish_transcription: true,
            dark_mode: None,
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
        let config = GuiConfig::default();
        assert!(!config.local_speaker.trim().is_empty());
        assert!(config.swedish_transcription);
        assert_eq!(config.dark_mode, None);
    }

    #[test]
    fn old_config_with_diarize_mic_still_loads() {
        let config: GuiConfig = serde_json::from_str(
            r#"{
                "meetings_dir": "/meetings",
                "screenshots_dir": "/screenshots",
                "local_speaker": "Alice",
                "diarize_mic": false
            }"#,
        )
        .expect("deserialize legacy GUI config");

        assert_eq!(config.context_dir, None);
        assert!(config.swedish_transcription);
        assert_eq!(config.dark_mode, None);
    }

    #[test]
    fn selected_theme_round_trips() {
        let config = GuiConfig {
            dark_mode: Some(true),
            ..GuiConfig::default()
        };
        let json = serde_json::to_string(&config).expect("serialize GUI config");
        let restored: GuiConfig = serde_json::from_str(&json).expect("deserialize GUI config");
        assert_eq!(restored.dark_mode, Some(true));
    }
}
