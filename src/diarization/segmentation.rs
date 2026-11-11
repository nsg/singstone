//! Speaker segmentation: a pyannote-style powerset model run through ONNX
//! Runtime on overlapping ten-second chunks.

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SAMPLE_RATE: usize = 16_000;
const WINDOW_SAMPLES: usize = 160_000;
const STEP_DIVISOR: usize = 10;
pub const LOCAL_SPEAKERS: usize = 3;
const POWERSET_CLASSES: usize = 7;
const POWERSET_MAX_CLASSES: usize = 2;
const BATCH_CHUNKS: usize = 4;
/// Local speakers active in each powerset class, as bits.
const POWERSET: [u8; POWERSET_CLASSES] = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110];

/// Hard per-frame activity of each chunk's local speakers.
pub struct Segmentation {
    pub num_chunks: usize,
    pub num_frames: usize,
    pub window: usize,
    pub step: usize,
    pub rf_shift: usize,
    pub rf_size: usize,
    /// Bit `k` of `activity[chunk * num_frames + frame]` is local speaker `k`.
    pub activity: Vec<u8>,
}

impl Segmentation {
    pub fn frames(&self, chunk: usize) -> &[u8] {
        &self.activity[chunk * self.num_frames..(chunk + 1) * self.num_frames]
    }

    /// First sample of `frame` in `chunk`.
    pub fn frame_to_sample(&self, chunk: usize, frame: usize) -> usize {
        (frame as f64 / self.num_frames as f64 * self.window as f64) as usize + chunk * self.step
    }

    pub fn frame_seconds(&self) -> f64 {
        self.window as f64 / self.num_frames as f64 / SAMPLE_RATE as f64
    }
}

pub struct Segmenter {
    session: Session,
    window: usize,
    rf_shift: usize,
    rf_size: usize,
}

impl Segmenter {
    pub fn new(model: &Path, threads: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let environment = environment()?;
        let session = Session::builder(environment)?
            .with_intra_threads(threads.max(1))?
            .with_inter_threads(1)?
            .commit_from_file(model)?;
        let metadata = session.metadata()?;
        let value = |key: &str| -> Result<usize, String> {
            metadata
                .custom(key)
                .ok_or_else(|| format!("segmentation model metadata lacks {key}"))?
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("segmentation model metadata {key} is not a count"))
        };
        let expect = |key: &str, expected: usize| -> Result<(), String> {
            let actual = value(key)?;
            (actual == expected).then_some(()).ok_or_else(|| {
                format!("segmentation model metadata {key} is {actual}, expected {expected}")
            })
        };
        expect("window_size", WINDOW_SAMPLES)?;
        expect("num_speakers", LOCAL_SPEAKERS)?;
        expect("num_classes", POWERSET_CLASSES)?;
        expect("powerset_max_classes", POWERSET_MAX_CLASSES)?;
        if metadata.custom("sample_rate").is_some() {
            expect("sample_rate", SAMPLE_RATE)?;
        }
        let rf_shift = value("receptive_field_shift")?;
        let rf_size = value("receptive_field_size")?;
        if rf_shift == 0 || rf_size == 0 || rf_shift > WINDOW_SAMPLES || rf_size > WINDOW_SAMPLES {
            return Err(format!(
                "segmentation model receptive field {rf_size} with shift {rf_shift} is out of range"
            )
            .into());
        }
        drop(metadata);
        let names = |outlets: &[ort::value::Outlet]| {
            outlets
                .iter()
                .map(|outlet| outlet.name().to_owned())
                .collect::<Vec<_>>()
        };
        if names(session.inputs()) != ["x"] || names(session.outputs()) != ["y"] {
            return Err("segmentation model must have one input x and one output y".into());
        }
        Ok(Self {
            session,
            window: WINDOW_SAMPLES,
            rf_shift,
            rf_size,
        })
    }

    pub fn segment(&mut self, samples: &[f32]) -> Result<Segmentation, Box<dyn std::error::Error>> {
        let window = self.window;
        let step = window / STEP_DIVISOR;
        let num_chunks = chunk_count(samples.len(), window, step);
        let mut activity = Vec::new();
        let mut num_frames = 0;
        for first in (0..num_chunks).step_by(BATCH_CHUNKS) {
            let batch = BATCH_CHUNKS.min(num_chunks - first);
            let mut input = vec![0.0f32; batch * window];
            for (index, chunk) in input.chunks_mut(window).enumerate() {
                let start = ((first + index) * step).min(samples.len());
                let end = (start + window).min(samples.len());
                chunk[..end - start].copy_from_slice(&samples[start..end]);
            }
            let input = Tensor::from_array(([batch, 1, window], input))?;
            let outputs = self.session.run(ort::inputs!["x" => input])?;
            let (shape, scores) = outputs["y"].try_extract_tensor::<f32>()?;
            let frames = match **shape {
                [n, frames, classes]
                    if n == batch as i64 && frames > 0 && classes == POWERSET_CLASSES as i64 =>
                {
                    frames as usize
                }
                _ => return Err(format!("unexpected segmentation output shape {shape:?}").into()),
            };
            if num_frames != 0 && frames != num_frames {
                return Err("segmentation output frame count changed between batches".into());
            }
            num_frames = frames;
            activity.extend(
                scores
                    .as_chunks::<POWERSET_CLASSES>()
                    .0
                    .iter()
                    .map(|scores| decode_powerset(scores)),
            );
        }
        Ok(Segmentation {
            num_chunks,
            num_frames,
            window,
            step,
            rf_shift: self.rf_shift,
            rf_size: self.rf_size,
            activity,
        })
    }
}

fn chunk_count(samples: usize, window: usize, step: usize) -> usize {
    samples.saturating_sub(window).div_ceil(step) + 1
}

/// Active local speakers of the most likely powerset class.
fn decode_powerset(scores: &[f32]) -> u8 {
    let best = scores.iter().enumerate().fold(
        0,
        |best, (class, score)| {
            if *score > scores[best] { class } else { best }
        },
    );
    POWERSET[best]
}

/// The process-wide ONNX Runtime environment, bound to the library that
/// sherpa-onnx already loaded so one runtime serves both.
fn environment() -> Result<&'static Environment, String> {
    static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| {
            ort::init_from(loaded_runtime().unwrap_or_else(|| "libonnxruntime.so".into()))
                .map_err(|error| error.to_string())?
                .with_name("singstone")
                .with_telemetry(false)
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| format!("cannot load ONNX Runtime: {error}"))
}

/// The ONNX Runtime that sherpa-onnx linked against, from this process's
/// memory map. Finding the provider of `OrtGetApiBase` directly would need
/// `dlsym`/`dladdr`, which the crate's `forbid(unsafe_code)` rules out.
fn loaded_runtime() -> Option<PathBuf> {
    choose_runtime(&std::fs::read_to_string("/proc/self/maps").ok()?)
}

/// The only mapped `libonnxruntime.so*`, or with several, the one beside
/// `libsherpa-onnx-c-api.so`. `None` leaves the choice to the dynamic
/// loader, which resolves the bare soname to the copy sherpa-onnx loaded.
fn choose_runtime(maps: &str) -> Option<PathBuf> {
    let mapped = |prefix: &str| {
        maps.lines()
            .filter_map(|line| line.split_whitespace().nth(5))
            .map(Path::new)
            .filter(|path| {
                path.is_absolute()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(prefix))
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    let runtimes = mapped("libonnxruntime.so");
    if runtimes.len() == 1 {
        return runtimes.first().map(|path| path.to_path_buf());
    }
    let sherpa = mapped("libsherpa-onnx-c-api.so");
    let directory = sherpa.first()?.parent();
    let mut beside = runtimes
        .into_iter()
        .filter(|runtime| runtime.parent() == directory);
    match (beside.next(), beside.next()) {
        (Some(runtime), None) => Some(runtime.to_path_buf()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powerset_classes_decode_to_local_speakers() {
        let mut scores = [-9.0f32; POWERSET_CLASSES];
        scores[5] = -0.1;
        assert_eq!(decode_powerset(&scores), 0b101);
        assert_eq!(decode_powerset(&[0.0; POWERSET_CLASSES]), 0);
    }

    #[test]
    fn chunks_cover_the_audio_and_frames_map_to_samples() {
        assert_eq!(chunk_count(0, 160_000, 16_000), 1);
        assert_eq!(chunk_count(160_000, 160_000, 16_000), 1);
        assert_eq!(chunk_count(160_001, 160_000, 16_000), 2);
        assert_eq!(chunk_count(480_000, 160_000, 16_000), 21);
        let segmentation = Segmentation {
            num_chunks: 2,
            num_frames: 589,
            window: 160_000,
            step: 16_000,
            rf_shift: 270,
            rf_size: 991,
            activity: Vec::new(),
        };
        assert_eq!(segmentation.frame_to_sample(0, 1), 271);
        assert_eq!(segmentation.frame_to_sample(1, 589), 176_000);
    }

    #[test]
    fn runtime_is_the_one_sherpa_loaded() {
        let line = |path: &str| format!("7f00-7f10 r-xp 00000000 08:02 1 {path}\n");
        let one = line("/a/libonnxruntime.so.1") + &line("/a/libonnxruntime.so.1");
        assert_eq!(
            choose_runtime(&one),
            Some(PathBuf::from("/a/libonnxruntime.so.1"))
        );
        let two = line("/a/libonnxruntime.so") + &line("/b/libonnxruntime.so");
        assert_eq!(
            choose_runtime(&(two.clone() + &line("/b/libsherpa-onnx-c-api.so"))),
            Some(PathBuf::from("/b/libonnxruntime.so"))
        );
        assert_eq!(choose_runtime(&two), None);
    }
}
