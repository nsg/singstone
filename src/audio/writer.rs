use crate::audio::capture::{Block, CaptureData, QUEUE_CAPACITY};
use crate::audio::timing::{Adjustment, Timeline};
use crate::format::jsonl::JsonlAppender;
use crate::session::Session;
use crate::types::{AudioSource, TimelineEvent, samples_to_ms};
use crossbeam_queue::ArrayQueue;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub struct WriterHandle {
    pub queue: Arc<ArrayQueue<Block>>,
    pub dropped: Arc<AtomicU64>,
    pub samples: Arc<AtomicU64>,
    pub format_ok: Arc<AtomicBool>,
    pub error: Arc<Mutex<Option<String>>>,
    pub level: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<io::Result<()>>>,
}

impl WriterHandle {
    pub fn start(session: &Session, source: AudioSource, t0_ns: u64) -> io::Result<Self> {
        let audio = BufWriter::new(crate::session::create_private_file(
            &session.audio_path(source),
        )?);
        let timeline = JsonlAppender::create(&session.timeline_path(source))?;
        let queue = Arc::new(ArrayQueue::new(QUEUE_CAPACITY));
        let dropped = Arc::new(AtomicU64::new(0));
        let samples = Arc::new(AtomicU64::new(0));
        let format_ok = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let level = Arc::new(AtomicU32::new(0.0f32.to_bits()));
        let stop = Arc::new(AtomicBool::new(false));

        let thread_queue = Arc::clone(&queue);
        let thread_dropped = Arc::clone(&dropped);
        let thread_samples = Arc::clone(&samples);
        let thread_error = Arc::clone(&error);
        let thread_stop = Arc::clone(&stop);
        let join = thread::Builder::new()
            .name(format!("singstone-{source}-writer"))
            .spawn(move || {
                let result = writer_loop(
                    audio,
                    timeline,
                    thread_queue,
                    thread_dropped,
                    thread_samples,
                    Arc::clone(&thread_error),
                    thread_stop,
                    t0_ns,
                );
                if let Err(error) = &result {
                    set_writer_error(&thread_error, error);
                }
                result
            })?;

        Ok(Self {
            queue,
            dropped,
            samples,
            format_ok,
            error,
            level,
            stop,
            join: Some(join),
        })
    }

    pub fn capture_data_with_level(&self, level: Arc<AtomicU32>) -> CaptureData {
        CaptureData::new(
            Arc::clone(&self.queue),
            Arc::clone(&self.dropped),
            Arc::clone(&self.format_ok),
            Arc::clone(&self.error),
            level,
        )
    }

    pub fn error_message(&self) -> Option<String> {
        self.error.lock().ok().and_then(|value| value.clone())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        self.stop.store(true, Ordering::Release);
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        join.join()
            .map_err(|_| io::Error::other("audio writer thread panicked"))?
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_loop(
    mut audio: BufWriter<File>,
    mut events: JsonlAppender,
    queue: Arc<ArrayQueue<Block>>,
    dropped: Arc<AtomicU64>,
    samples: Arc<AtomicU64>,
    error: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    t0_ns: u64,
) -> io::Result<()> {
    let mut timeline = Timeline::new(t0_ns);
    let mut last_dropped = 0;
    let mut last_seq = None;
    let mut error_logged = false;
    let mut last_flush = Instant::now();

    loop {
        if !error_logged && let Some(message) = error.lock().ok().and_then(|value| value.clone()) {
            events.append(&TimelineEvent::Error {
                sample: timeline.written(),
                time_ms: samples_to_ms(timeline.written()),
                message,
            })?;
            error_logged = true;
        }

        if let Some(block) = queue.pop() {
            let capture_ns =
                (block.now_ns as i128 - block.delay_ns as i128).clamp(0, u64::MAX as i128) as u64;
            let placement = timeline.place(capture_ns, block.len as u64);
            write_silence(&mut audio, placement.silence)?;

            let dropped_now = dropped.load(Ordering::Relaxed);
            let counter_drops = dropped_now.saturating_sub(last_dropped);
            let sequence_drops = last_seq
                .map(|previous: u64| block.seq.wrapping_sub(previous).saturating_sub(1))
                .unwrap_or(0);
            let new_drops = counter_drops.max(sequence_drops);
            let time_ms = samples_to_ms(match placement.adjustment {
                Adjustment::Start { sample }
                | Adjustment::Gap { sample, .. }
                | Adjustment::Overlap { sample, .. } => sample,
                Adjustment::None => timeline.written().saturating_sub(block.len as u64),
            });
            match placement.adjustment {
                Adjustment::Start { sample } => {
                    events.append(&TimelineEvent::Start { sample, time_ms })?;
                    if placement.requested_silence > placement.silence {
                        events.append(&TimelineEvent::Xrun {
                            sample: 0,
                            time_ms: 0,
                            missing_samples: placement.silence,
                        })?;
                    }
                }
                Adjustment::Gap {
                    sample,
                    missing_samples,
                } if new_drops == 0 || placement.requested_silence > placement.silence => {
                    events.append(&TimelineEvent::Xrun {
                        sample,
                        time_ms,
                        missing_samples,
                    })?;
                }
                Adjustment::Gap { .. } => {}
                Adjustment::Overlap {
                    sample,
                    extra_samples,
                } => {
                    events.append(&TimelineEvent::Overlap {
                        sample,
                        time_ms,
                        extra_samples,
                    })?;
                }
                Adjustment::None => {}
            }
            if placement.requested_silence > placement.silence {
                let sample = timeline
                    .written()
                    .saturating_sub(block.len as u64)
                    .saturating_sub(placement.silence);
                events.append(&TimelineEvent::ClockJump {
                    sample,
                    time_ms: samples_to_ms(sample),
                    requested_samples: placement.requested_silence,
                })?;
            }
            if new_drops > 0 {
                let sample = timeline
                    .written()
                    .saturating_sub(block.len as u64)
                    .saturating_sub(placement.silence);
                events.append(&TimelineEvent::Dropped {
                    sample,
                    time_ms: samples_to_ms(sample),
                    blocks: new_drops,
                })?;
            }
            last_dropped = dropped_now;
            last_seq = Some(block.seq);
            write_samples(&mut audio, &block.samples[..block.len as usize])?;
            samples.store(timeline.written(), Ordering::Relaxed);

            if last_flush.elapsed() >= Duration::from_secs(1) {
                audio.flush()?;
                last_flush = Instant::now();
            }
        } else if stop.load(Ordering::Acquire) {
            break;
        } else {
            if last_flush.elapsed() >= Duration::from_secs(1) {
                audio.flush()?;
                last_flush = Instant::now();
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    let dropped_now = dropped.load(Ordering::Relaxed);
    let tail_drops = dropped_now.saturating_sub(last_dropped);
    if tail_drops > 0 {
        let sample = timeline.written();
        let missing_samples =
            tail_drops.saturating_mul(crate::audio::capture::BLOCK_SAMPLES as u64);
        write_silence(&mut audio, missing_samples)?;
        timeline.append_silence(missing_samples);
        events.append(&TimelineEvent::Dropped {
            sample,
            time_ms: samples_to_ms(sample),
            blocks: tail_drops,
        })?;
    }
    let final_sample = timeline.written();
    events.append(&TimelineEvent::Stop {
        sample: final_sample,
        time_ms: samples_to_ms(final_sample),
    })?;
    audio.flush()?;
    audio.get_ref().sync_all()?;
    samples.store(final_sample, Ordering::Relaxed);
    Ok(())
}

fn set_writer_error(target: &Mutex<Option<String>>, error: &io::Error) {
    if let Ok(mut stored) = target.lock() {
        let message = format!("audio writer error: {error}");
        match stored.as_mut() {
            Some(existing) if !existing.contains(&message) => {
                existing.push_str("; ");
                existing.push_str(&message);
            }
            Some(_) => {}
            None => *stored = Some(message),
        }
    }
}

fn write_samples(writer: &mut BufWriter<File>, samples: &[f32]) -> io::Result<()> {
    let mut bytes = [0_u8; crate::audio::capture::BLOCK_SAMPLES * 4];
    for (sample, chunk) in samples.iter().zip(bytes.as_chunks_mut::<4>().0) {
        chunk.copy_from_slice(&sample.to_le_bytes());
    }
    writer.write_all(&bytes[..samples.len() * 4])
}

fn write_silence(writer: &mut BufWriter<File>, samples: u64) -> io::Result<()> {
    const ZEROES: [u8; 8_192] = [0; 8_192];
    let mut bytes = samples.saturating_mul(4);
    while bytes > 0 {
        let count = bytes.min(ZEROES.len() as u64) as usize;
        writer.write_all(&ZEROES[..count])?;
        bytes -= count as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::jsonl;
    use std::fs;
    use std::os::unix::fs::DirBuilderExt;

    #[test]
    fn timeline_jsonl_round_trip() {
        let dir = std::env::temp_dir().join(format!("singstone-writer-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .expect("create temp dir");
        let path = dir.join("timeline.jsonl");
        let expected = vec![
            TimelineEvent::Start {
                sample: 32_000,
                time_ms: 2_000,
            },
            TimelineEvent::Overlap {
                sample: 40_000,
                time_ms: 2_500,
                extra_samples: 5_000,
            },
            TimelineEvent::Stop {
                sample: 48_000,
                time_ms: 3_000,
            },
        ];
        let mut appender = JsonlAppender::create(&path).expect("create JSONL");
        for event in &expected {
            appender.append(event).expect("append event");
        }
        drop(appender);
        let actual: Vec<TimelineEvent> = jsonl::read_all(&path).expect("read JSONL");
        assert_eq!(actual, expected);
        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn clock_jump_is_capped_and_logged() {
        let dir = std::env::temp_dir().join(format!("singstone-clock-jump-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .expect("create temp dir");
        let session = Session::create(dir.join("session")).expect("create session");
        let mut writer = WriterHandle::start(&session, AudioSource::Mic, 0).expect("start writer");
        assert!(
            writer
                .queue
                .push(Block {
                    len: 1,
                    now_ns: u64::MAX,
                    ..Block::default()
                })
                .is_ok(),
            "queue block"
        );
        writer.finish().expect("finish writer");
        let events: Vec<TimelineEvent> =
            jsonl::read_all(&session.timeline_path(AudioSource::Mic)).expect("read timeline");
        assert!(events.iter().any(|event| matches!(
            event,
            TimelineEvent::Xrun {
                missing_samples: crate::audio::timing::MAX_GAP_SAMPLES,
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            TimelineEvent::ClockJump {
                requested_samples,
                ..
            } if *requested_samples > crate::audio::timing::MAX_GAP_SAMPLES
        )));
        assert_eq!(
            fs::metadata(session.audio_path(AudioSource::Mic))
                .expect("audio metadata")
                .len(),
            (crate::audio::timing::MAX_GAP_SAMPLES + 1) * 4
        );
        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn trailing_drops_are_silence_filled_before_stop() {
        let dir = std::env::temp_dir().join(format!("singstone-tail-drops-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .expect("create temp dir");
        let session = Session::create(dir.join("session")).expect("create session");
        let mut writer =
            WriterHandle::start(&session, AudioSource::System, 0).expect("start writer");
        writer.dropped.store(2, Ordering::Relaxed);
        writer.finish().expect("finish writer");
        let events: Vec<TimelineEvent> =
            jsonl::read_all(&session.timeline_path(AudioSource::System)).expect("read timeline");
        assert!(matches!(
            events.as_slice(),
            [
                TimelineEvent::Dropped { blocks: 2, .. },
                TimelineEvent::Stop { .. }
            ]
        ));
        assert_eq!(
            writer.samples.load(Ordering::Relaxed),
            2 * crate::audio::capture::BLOCK_SAMPLES as u64
        );
        fs::remove_dir_all(dir).expect("remove temp dir");
    }
}
