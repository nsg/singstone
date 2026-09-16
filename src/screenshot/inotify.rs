use crate::audio::pipewire::monotonic_ns;
use crate::format::jsonl::JsonlAppender;
use crate::types::ScreenshotEntry;
use rustix::event::{PollFd, PollFlags, poll};
use rustix::fs::inotify;
use rustix::time::Timespec;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io;
use std::mem::MaybeUninit;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub struct Watch {
    fd: rustix::fd::OwnedFd,
    input_dir: PathBuf,
    output_dir: PathBuf,
    index: JsonlAppender,
    extensions: Vec<String>,
    t0_ns: u64,
    filed: Arc<AtomicU64>,
}

impl Watch {
    pub fn new(
        input_dir: &Path,
        output_dir: PathBuf,
        index_path: &Path,
        extensions: Vec<String>,
        t0_ns: u64,
        filed: Arc<AtomicU64>,
    ) -> io::Result<Self> {
        let input_dir = input_dir.canonicalize()?;
        if !input_dir.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} is not a directory", input_dir.display()),
            ));
        }
        let fd = inotify::init(inotify::CreateFlags::NONBLOCK | inotify::CreateFlags::CLOEXEC)?;
        inotify::add_watch(
            &fd,
            &input_dir,
            inotify::WatchFlags::CLOSE_WRITE | inotify::WatchFlags::MOVED_TO,
        )?;
        Ok(Self {
            fd,
            input_dir,
            output_dir,
            index: JsonlAppender::create(index_path)?,
            extensions: extensions
                .into_iter()
                .map(|extension| extension.trim_start_matches('.').to_ascii_lowercase())
                .collect(),
            t0_ns,
            filed,
        })
    }

    pub fn run(mut self, stop: Arc<AtomicBool>) -> io::Result<()> {
        let timeout = Timespec {
            tv_sec: 0,
            tv_nsec: 200_000_000,
        };
        let mut buffer = [MaybeUninit::uninit(); 8_192];
        while !stop.load(Ordering::Acquire) {
            let mut fds = [PollFd::new(&self.fd, PollFlags::IN)];
            match poll(&mut fds, Some(&timeout)) {
                Ok(0) => continue,
                Ok(_) if !fds[0].revents().contains(PollFlags::IN) => continue,
                Ok(_) => {}
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => return Err(error.into()),
            }

            let events = {
                let mut reader = inotify::Reader::new(&self.fd, &mut buffer);
                let mut events = Vec::new();
                loop {
                    match reader.next() {
                        Ok(event) => {
                            let time_ms = monotonic_ns().saturating_sub(self.t0_ns) / 1_000_000;
                            if event.events().contains(inotify::ReadFlags::QUEUE_OVERFLOW) {
                                eprintln!("warning: screenshot inotify queue overflowed");
                                continue;
                            }
                            if let Some(name) = event.file_name() {
                                events
                                    .push((OsStr::from_bytes(name.to_bytes()).to_owned(), time_ms));
                            }
                        }
                        Err(rustix::io::Errno::AGAIN) => break,
                        Err(error) => return Err(error.into()),
                    }
                }
                events
            };
            for (name, time_ms) in events {
                if let Err(error) = self.handle_name(&name, time_ms) {
                    eprintln!("warning: screenshot {}: {error}", name.to_string_lossy());
                }
            }
        }
        Ok(())
    }

    fn handle_name(&mut self, name: &OsStr, time_ms: u64) -> io::Result<()> {
        let source = self.input_dir.join(name);
        if !extension_matches(&source, &self.extensions) {
            return Ok(());
        }
        if !fs::metadata(&source)?.is_file() {
            return Ok(());
        }
        let mut suffix = 0;
        let (destination, destination_name) = loop {
            let name = screenshot_file_name(time_ms, name, suffix);
            let path = self.output_dir.join(&name);
            if !path.try_exists()? {
                break (path, name);
            }
            suffix = suffix.checked_add(1).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "too many screenshot collisions",
                )
            })?;
        };

        let mut input = File::open(&source)?;
        let mut output = crate::session::create_private_file(&destination)?;
        io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        self.index.append(&ScreenshotEntry {
            time_ms,
            file: Path::new("screenshots")
                .join(destination_name)
                .to_string_lossy()
                .into_owned(),
            original: source.to_string_lossy().into_owned(),
        })?;
        self.filed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

pub fn extension_matches(path: &Path, extensions: &[String]) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|candidate| {
            extensions
                .iter()
                .any(|extension| candidate.eq_ignore_ascii_case(extension.trim_start_matches('.')))
        })
}

pub fn screenshot_file_name(time_ms: u64, original: &OsStr, suffix: u32) -> std::ffi::OsString {
    let mut bytes = format!("{time_ms:09}-").into_bytes();
    bytes.extend_from_slice(original.as_bytes());
    if suffix > 0 {
        bytes.extend_from_slice(format!("-{suffix}").as_bytes());
    }
    std::ffi::OsString::from_vec(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStrExt;

    #[test]
    fn extension_filter_is_case_insensitive() {
        let extensions = vec!["png".to_owned(), "jpeg".to_owned()];
        assert!(extension_matches(Path::new("shot.PNG"), &extensions));
        assert!(extension_matches(Path::new("shot.jpeg"), &extensions));
        assert!(!extension_matches(Path::new("shot.txt"), &extensions));
        assert!(!extension_matches(Path::new("shot"), &extensions));
    }

    #[test]
    fn screenshot_names_are_zero_padded() {
        assert_eq!(
            screenshot_file_name(123, OsStr::new("Shot.png"), 0).as_bytes(),
            b"000000123-Shot.png"
        );
        assert_eq!(
            screenshot_file_name(123, OsStr::new("Shot.png"), 2).as_bytes(),
            b"000000123-Shot.png-2"
        );
    }
}
