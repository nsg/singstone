use crossbeam_queue::ArrayQueue;
use pipewire as pw;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub const BLOCK_SAMPLES: usize = 2_048;
pub const QUEUE_CAPACITY: usize = 512;

#[derive(Clone)]
pub struct Block {
    pub samples: [f32; BLOCK_SAMPLES],
    pub len: u32,
    pub now_ns: u64,
    pub delay_ns: i64,
    pub seq: u64,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            samples: [0.0; BLOCK_SAMPLES],
            len: 0,
            now_ns: 0,
            delay_ns: 0,
            seq: 0,
        }
    }
}

pub struct CaptureData {
    pub queue: Arc<ArrayQueue<Block>>,
    pub dropped: Arc<AtomicU64>,
    pub format_ok: Arc<AtomicBool>,
    pub error: Arc<Mutex<Option<String>>>,
    /// Peak amplitude of the most recently captured buffer, encoded with
    /// `f32::to_bits` so the GUI can read it without touching the RT thread.
    pub level: Arc<AtomicU32>,
    next_seq: u64,
}

impl CaptureData {
    pub fn new(
        queue: Arc<ArrayQueue<Block>>,
        dropped: Arc<AtomicU64>,
        format_ok: Arc<AtomicBool>,
        error: Arc<Mutex<Option<String>>>,
        level: Arc<AtomicU32>,
    ) -> Self {
        Self {
            queue,
            dropped,
            format_ok,
            error,
            level,
            next_seq: 0,
        }
    }
}

pub fn listener(
    stream: &pw::stream::Stream,
    data: CaptureData,
) -> Result<pw::stream::StreamListener<CaptureData>, pw::Error> {
    stream
        .add_local_listener_with_user_data(data)
        .state_changed(|_, data, _, state| {
            if let pw::stream::StreamState::Error(message) = state {
                set_error(&data.error, format!("PipeWire stream error: {message}"));
                data.format_ok.store(false, Ordering::Release);
            }
        })
        .param_changed(|stream, data, id, param| {
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                return;
            };
            let mut info = pw::spa::param::audio::AudioInfoRaw::new();
            let valid = info.parse(param).is_ok()
                && info.format() == pw::spa::param::audio::AudioFormat::F32LE
                && info.rate() == crate::types::SAMPLE_RATE
                && info.channels() == 1;
            if valid {
                data.format_ok.store(true, Ordering::Release);
            } else {
                set_error(
                    &data.error,
                    format!(
                        "unsupported negotiated format {:?}/{} Hz/{} channels",
                        info.format(),
                        info.rate(),
                        info.channels()
                    ),
                );
                data.format_ok.store(false, Ordering::Release);
                let _ = stream.disconnect();
            }
        })
        .process(|stream, data| {
            let mut consumed_samples = 0u64;
            while let Some(mut buffer) = stream.dequeue_buffer() {
                if !data.format_ok.load(Ordering::Acquire) {
                    continue;
                }
                let Ok(time) = stream.time() else {
                    data.dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let rate = time.rate();
                if rate.denom == 0 {
                    data.dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let delay_ns = ((time.delay() as i128) * 1_000_000_000_i128 * (rate.num as i128)
                    / (rate.denom as i128))
                    .clamp(i64::MIN as i128, i64::MAX as i128)
                    as i64;
                let base_now = time.now().max(0) as u64;

                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    continue;
                }
                let data_plane = &mut datas[0];
                let (offset, size, stride, corrupted) = {
                    let chunk = data_plane.chunk();
                    (
                        chunk.offset() as usize,
                        chunk.size() as usize,
                        chunk.stride(),
                        chunk
                            .flags()
                            .contains(pw::spa::buffer::ChunkFlags::CORRUPTED),
                    )
                };
                let Some(bytes) = data_plane.data() else {
                    drop_blocks(data, 1);
                    continue;
                };
                let maxsize = bytes.len();
                if maxsize == 0 {
                    drop_blocks(data, 1);
                    continue;
                }
                let offset = offset % maxsize;
                let size = size.min(maxsize);
                let total_samples = size / size_of::<f32>();
                if corrupted || !matches!(stride, 0 | 4) || !size.is_multiple_of(size_of::<f32>()) {
                    let blocks = total_samples.div_ceil(BLOCK_SAMPLES).max(1) as u64;
                    drop_blocks(data, blocks);
                    consumed_samples = consumed_samples.saturating_add(total_samples as u64);
                    continue;
                }
                let mut sample_offset = 0;
                let mut peak = 0.0f32;
                while sample_offset < total_samples {
                    let count = (total_samples - sample_offset).min(BLOCK_SAMPLES);
                    let mut block = Block {
                        len: count as u32,
                        now_ns: base_now.saturating_add(
                            ((u128::from(consumed_samples) * 1_000_000_000)
                                / u128::from(crate::types::SAMPLE_RATE))
                            .min(u128::from(u64::MAX)) as u64,
                        ),
                        delay_ns,
                        seq: data.next_seq,
                        ..Block::default()
                    };
                    for (index, output) in block.samples[..count].iter_mut().enumerate() {
                        *output = wrapped_f32(bytes, offset, sample_offset + index);
                        peak = peak.max(output.abs());
                    }
                    data.next_seq = data.next_seq.wrapping_add(1);
                    if data.queue.push(block).is_err() {
                        data.dropped.fetch_add(1, Ordering::Relaxed);
                    }
                    sample_offset += count;
                    consumed_samples = consumed_samples.saturating_add(count as u64);
                }
                data.level
                    .store(peak.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
            }
        })
        .register()
}

fn wrapped_f32(bytes: &[u8], offset: usize, sample: usize) -> f32 {
    let begin = (offset + sample * size_of::<f32>()) % bytes.len();
    f32::from_le_bytes([
        bytes[begin],
        bytes[(begin + 1) % bytes.len()],
        bytes[(begin + 2) % bytes.len()],
        bytes[(begin + 3) % bytes.len()],
    ])
}

fn drop_blocks(data: &mut CaptureData, blocks: u64) {
    data.next_seq = data.next_seq.wrapping_add(blocks);
    data.dropped.fetch_add(blocks, Ordering::Relaxed);
}

fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
}

pub fn set_error(target: &Mutex<Option<String>>, message: String) {
    if let Ok(mut error) = target.lock()
        && error.is_none()
    {
        *error = Some(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_f32_across_wrapped_chunk_boundary() {
        let first = 1.25f32.to_le_bytes();
        let second = (-2.5f32).to_le_bytes();
        let mut bytes = [0u8; 8];
        bytes[6] = first[0];
        bytes[7] = first[1];
        bytes[0] = first[2];
        bytes[1] = first[3];
        bytes[2..6].copy_from_slice(&second);
        assert_eq!(wrapped_f32(&bytes, 6, 0), 1.25);
        assert_eq!(wrapped_f32(&bytes, 6, 1), -2.5);
    }
}
