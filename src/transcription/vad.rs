use crate::types::SAMPLE_RATE;

const FRAME_MS: usize = 20;
const FRAME_SAMPLES: usize = SAMPLE_RATE as usize * FRAME_MS / 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechRegion {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub threshold_db: f32,
    pub hangover_ms: usize,
    pub merge_gap_ms: usize,
    pub padding_ms: usize,
    pub min_speech_ms: usize,
    pub max_region_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold_db: -45.0,
            hangover_ms: 200,
            merge_gap_ms: 500,
            padding_ms: 300,
            min_speech_ms: 250,
            max_region_ms: 25 * 60 * 1000,
        }
    }
}

pub fn detect(samples: &[f32], config: VadConfig) -> Vec<SpeechRegion> {
    if samples.is_empty() {
        return Vec::new();
    }
    let speech = samples
        .chunks(FRAME_SAMPLES)
        .map(|frame| rms_dbfs(frame) > config.threshold_db)
        .collect::<Vec<_>>();
    let hangover_frames = config.hangover_ms.div_ceil(FRAME_MS);
    let mut raw = Vec::new();
    let mut start = None;
    let mut last_speech = 0;
    for (index, active) in speech.iter().copied().enumerate() {
        if active {
            start.get_or_insert(index);
            last_speech = index;
        } else if let Some(first) = start
            && index.saturating_sub(last_speech) > hangover_frames
        {
            raw.push((first, last_speech + 1));
            start = None;
        }
    }
    if let Some(first) = start {
        raw.push((first, last_speech + 1));
    }

    let min_frames = config.min_speech_ms.div_ceil(FRAME_MS);
    raw.retain(|(start, end)| end - start >= min_frames);
    let merge_frames = config.merge_gap_ms.div_ceil(FRAME_MS);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for region in raw {
        if let Some(previous) = merged.last_mut()
            && region.0.saturating_sub(previous.1) <= merge_frames
        {
            previous.1 = region.1;
            continue;
        }
        merged.push(region);
    }

    let pad = config.padding_ms * SAMPLE_RATE as usize / 1000;
    let mut regions = merged
        .into_iter()
        .map(|(start, end)| SpeechRegion {
            start: (start * FRAME_SAMPLES).saturating_sub(pad),
            end: (end * FRAME_SAMPLES + pad).min(samples.len()),
        })
        .collect::<Vec<_>>();
    merge_overlaps(&mut regions);
    split_long_regions(samples, &regions, config.max_region_ms)
}

fn rms_dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_square = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>()
        / samples.len() as f64;
    if mean_square <= 0.0 {
        f32::NEG_INFINITY
    } else {
        (20.0 * mean_square.sqrt().log10()) as f32
    }
}

fn merge_overlaps(regions: &mut Vec<SpeechRegion>) {
    let mut merged: Vec<SpeechRegion> = Vec::new();
    for region in regions.drain(..) {
        if let Some(previous) = merged.last_mut()
            && region.start <= previous.end
        {
            previous.end = previous.end.max(region.end);
            continue;
        }
        merged.push(region);
    }
    *regions = merged;
}

fn split_long_regions(
    samples: &[f32],
    regions: &[SpeechRegion],
    max_region_ms: usize,
) -> Vec<SpeechRegion> {
    let max_samples = max_region_ms * SAMPLE_RATE as usize / 1000;
    let mut output = Vec::new();
    for region in regions {
        let mut start = region.start;
        while region.end.saturating_sub(start) > max_samples {
            let ideal = start + max_samples;
            let radius = 30 * SAMPLE_RATE as usize;
            let search_start = ideal.saturating_sub(radius).max(start + FRAME_SAMPLES);
            let search_end = (ideal + radius).min(region.end - FRAME_SAMPLES);
            let split = quietest_frame(samples, search_start, search_end);
            output.push(SpeechRegion { start, end: split });
            start = split;
        }
        output.push(SpeechRegion {
            start,
            end: region.end,
        });
    }
    output
}

fn quietest_frame(samples: &[f32], start: usize, end: usize) -> usize {
    (start..end)
        .step_by(FRAME_SAMPLES)
        .min_by(|left, right| {
            rms_dbfs(&samples[*left..(*left + FRAME_SAMPLES).min(samples.len())]).total_cmp(
                &rms_dbfs(&samples[*right..(*right + FRAME_SAMPLES).min(samples.len())]),
            )
        })
        .unwrap_or(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(ms: usize, amplitude: f32) -> Vec<f32> {
        vec![amplitude; ms * SAMPLE_RATE as usize / 1000]
    }

    #[test]
    fn silence_has_no_regions() {
        assert!(detect(&tone(2_000, 0.0), VadConfig::default()).is_empty());
    }

    #[test]
    fn detects_pads_and_merges_speech() {
        let mut samples = tone(1_000, 0.0);
        samples.extend(tone(400, 0.1));
        samples.extend(tone(300, 0.0));
        samples.extend(tone(400, 0.1));
        samples.extend(tone(1_000, 0.0));
        let regions = detect(&samples, VadConfig::default());
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start, 700 * SAMPLE_RATE as usize / 1000);
        assert_eq!(regions[0].end, 2_400 * SAMPLE_RATE as usize / 1000);
    }

    #[test]
    fn drops_short_burst() {
        let mut samples = tone(500, 0.0);
        samples.extend(tone(200, 0.2));
        samples.extend(tone(500, 0.0));
        assert!(detect(&samples, VadConfig::default()).is_empty());
    }
}
