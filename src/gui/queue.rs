use std::collections::VecDeque;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Running,
    Waiting { position: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Finished,
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedJob {
    pub path: PathBuf,
    pub error: String,
}

#[derive(Debug, Default)]
pub struct ProcessingQueue {
    running: Option<PathBuf>,
    waiting: VecDeque<PathBuf>,
    failed: Vec<FailedJob>,
    completed: usize,
    processed: usize,
    failed_count: usize,
}

impl ProcessingQueue {
    pub fn enqueue(&mut self, path: PathBuf) -> bool {
        if self.running.as_ref() == Some(&path) || self.waiting.contains(&path) {
            return false;
        }
        if self.is_idle() {
            self.completed = 0;
            self.processed = 0;
            self.failed_count = 0;
        }
        self.failed.retain(|job| job.path != path);
        self.waiting.push_back(path);
        true
    }

    pub fn start_next(&mut self) -> Option<PathBuf> {
        if self.running.is_some() {
            return None;
        }
        let path = self.waiting.pop_front()?;
        self.running = Some(path.clone());
        Some(path)
    }

    pub fn finish(&mut self, outcome: Outcome) -> Option<PathBuf> {
        let path = self.running.take()?;
        self.completed += 1;
        match outcome {
            Outcome::Finished => self.processed += 1,
            Outcome::Cancelled => {}
            Outcome::Failed(error) => {
                self.failed_count += 1;
                self.failed.push(FailedJob {
                    path: path.clone(),
                    error,
                });
            }
        }
        Some(path)
    }

    pub fn remove(&mut self, path: &Path) -> bool {
        let Some(index) = self.waiting.iter().position(|waiting| waiting == path) else {
            return false;
        };
        self.waiting.remove(index);
        true
    }

    pub fn state(&self, path: &Path) -> Option<JobState> {
        if self.running.as_deref() == Some(path) {
            return Some(JobState::Running);
        }
        self.waiting
            .iter()
            .position(|waiting| waiting == path)
            .map(|index| JobState::Waiting {
                position: index + if self.running.is_some() { 2 } else { 1 },
            })
    }

    pub fn batch_position_total(&self) -> Option<(usize, usize)> {
        (!self.is_idle()).then(|| {
            (
                self.completed + 1,
                self.completed + usize::from(self.running.is_some()) + self.waiting.len(),
            )
        })
    }

    pub fn is_idle(&self) -> bool {
        self.running.is_none() && self.waiting.is_empty()
    }

    pub fn clear_failed(&mut self) {
        self.failed.clear();
    }

    pub fn has_failed(&self, path: &Path) -> bool {
        self.failed.iter().any(|job| job.path == path)
    }

    pub fn remove_failed(&mut self, path: &Path) -> bool {
        let previous_len = self.failed.len();
        self.failed.retain(|job| job.path != path);
        self.failed.len() != previous_len
    }

    pub fn clear_waiting(&mut self) {
        self.waiting.clear();
    }

    pub fn running(&self) -> Option<&Path> {
        self.running.as_deref()
    }

    pub fn waiting(&self) -> &VecDeque<PathBuf> {
        &self.waiting
    }

    pub fn failed(&self) -> &[FailedJob] {
        &self.failed
    }

    pub fn processed_count(&self) -> usize {
        self.processed
    }

    pub fn failed_count(&self) -> usize {
        self.failed_count
    }

    pub fn completed_count(&self) -> usize {
        self.completed
    }

    pub fn outstanding_count(&self) -> usize {
        usize::from(self.running.is_some()) + self.waiting.len()
    }
}

pub fn ordinal(value: usize) -> String {
    let suffix = if matches!(value % 100, 11..=13) {
        "th"
    } else {
        match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    };
    format!("{value}{suffix}")
}

pub fn finished_banner_text(processed: usize, failed: usize) -> String {
    let mut text = format!(
        "Queue finished — {processed} session{} processed",
        if processed == 1 { "" } else { "s" }
    );
    if failed > 0 {
        text.push_str(&format!(", {failed} failed"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn enqueue_refuses_running_and_waiting_duplicates() {
        let mut queue = ProcessingQueue::default();
        assert!(queue.enqueue(path("one")));
        assert!(!queue.enqueue(path("one")));
        queue.start_next();
        assert!(!queue.enqueue(path("one")));
    }

    #[test]
    fn jobs_start_in_order_and_only_one_runs() {
        let mut queue = ProcessingQueue::default();
        queue.enqueue(path("one"));
        queue.enqueue(path("two"));
        assert_eq!(queue.start_next(), Some(path("one")));
        assert_eq!(queue.start_next(), None);
        queue.finish(Outcome::Finished);
        assert_eq!(queue.start_next(), Some(path("two")));
    }

    #[test]
    fn removing_waiting_job_renumbers_positions() {
        let mut queue = ProcessingQueue::default();
        for name in ["one", "two", "three"] {
            queue.enqueue(path(name));
        }
        queue.start_next();
        assert_eq!(
            queue.state(Path::new("three")),
            Some(JobState::Waiting { position: 3 })
        );
        assert!(queue.remove(Path::new("two")));
        assert_eq!(
            queue.state(Path::new("three")),
            Some(JobState::Waiting { position: 2 })
        );
    }

    #[test]
    fn batch_position_and_total_include_all_finish_outcomes() {
        let mut queue = ProcessingQueue::default();
        for name in ["one", "two", "three"] {
            queue.enqueue(path(name));
        }
        queue.start_next();
        assert_eq!(queue.batch_position_total(), Some((1, 3)));
        queue.finish(Outcome::Cancelled);
        queue.start_next();
        assert_eq!(queue.batch_position_total(), Some((2, 3)));
        queue.finish(Outcome::Failed("bad".into()));
        queue.start_next();
        assert_eq!(queue.batch_position_total(), Some((3, 3)));
    }

    #[test]
    fn failed_job_is_listed_and_retry_removes_it() {
        let mut queue = ProcessingQueue::default();
        queue.enqueue(path("one"));
        queue.start_next();
        queue.finish(Outcome::Failed("bad".into()));
        assert_eq!(queue.failed()[0].error, "bad");
        assert!(queue.has_failed(Path::new("one")));
        assert!(queue.enqueue(path("one")));
        assert!(queue.failed().is_empty());
        assert!(!queue.has_failed(Path::new("one")));
    }

    #[test]
    fn new_batch_resets_counters_but_keeps_failures() {
        let mut queue = ProcessingQueue::default();
        queue.enqueue(path("one"));
        queue.start_next();
        queue.finish(Outcome::Failed("bad".into()));
        assert!(queue.enqueue(path("two")));
        assert_eq!(queue.processed_count(), 0);
        assert_eq!(queue.failed_count(), 0);
        assert_eq!(queue.completed_count(), 0);
        assert_eq!(queue.failed().len(), 1);
    }

    #[test]
    fn ordinal_suffixes_handle_teens_and_larger_values() {
        let values = [1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111];
        let expected = [
            "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st", "111th",
        ];
        assert_eq!(values.map(ordinal), expected);
    }

    #[test]
    fn finished_banner_uses_singular_and_reports_failures() {
        assert_eq!(
            finished_banner_text(1, 0),
            "Queue finished — 1 session processed"
        );
        assert_eq!(
            finished_banner_text(3, 1),
            "Queue finished — 3 sessions processed, 1 failed"
        );
    }
}
