use crate::screenshot::inotify::Watch;
use crate::session::Session;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

pub struct WatcherHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<io::Result<()>>>,
}

impl WatcherHandle {
    pub fn start_with_counter(
        input_dir: &Path,
        session: &Session,
        extensions: Vec<String>,
        t0_ns: u64,
        filed: Arc<AtomicU64>,
    ) -> io::Result<Self> {
        let watch = Watch::new(
            input_dir,
            session.screenshots_dir(),
            &session.screenshots_index_path(),
            extensions,
            t0_ns,
            filed,
        )?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let join = thread::Builder::new()
            .name("singstone-screenshot-watcher".to_owned())
            .spawn(move || watch.run(thread_stop))?;
        Ok(Self {
            stop,
            join: Some(join),
        })
    }

    pub fn finish(&mut self) -> io::Result<()> {
        self.stop.store(true, Ordering::Release);
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        join.join()
            .map_err(|_| io::Error::other("screenshot watcher thread panicked"))?
    }
}
