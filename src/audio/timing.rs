use crate::types::SAMPLE_RATE;

pub const DEFAULT_TOLERANCE: u64 = 480;
pub const MAX_GAP_SAMPLES: u64 = SAMPLE_RATE as u64 * 60;
const OVERLAP_EVENT_THRESHOLD: u64 = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjustment {
    Start { sample: u64 },
    Gap { sample: u64, missing_samples: u64 },
    Overlap { sample: u64, extra_samples: u64 },
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub silence: u64,
    pub requested_silence: u64,
    pub adjustment: Adjustment,
}

#[derive(Debug)]
pub struct Timeline {
    t0_ns: u64,
    written: u64,
    started: bool,
    tolerance: u64,
    pending_gap: bool,
}

impl Timeline {
    pub fn new(t0_ns: u64) -> Self {
        Self {
            t0_ns,
            written: 0,
            started: false,
            tolerance: DEFAULT_TOLERANCE,
            pending_gap: false,
        }
    }

    pub fn place(&mut self, capture_ns: u64, samples: u64) -> Placement {
        let elapsed_ns = u128::from(capture_ns.saturating_sub(self.t0_ns));
        let expected =
            (elapsed_ns * u128::from(SAMPLE_RATE) / 1_000_000_000).min(u128::from(u64::MAX)) as u64;

        let placement = if !self.started {
            self.started = true;
            let silence = expected.min(MAX_GAP_SAMPLES);
            Placement {
                silence,
                requested_silence: expected,
                adjustment: Adjustment::Start { sample: silence },
            }
        } else if expected > self.written.saturating_add(self.tolerance) {
            if self.pending_gap {
                self.pending_gap = false;
                let requested_silence = expected - self.written;
                let silence = requested_silence.min(MAX_GAP_SAMPLES);
                Placement {
                    silence,
                    requested_silence,
                    adjustment: Adjustment::Gap {
                        sample: self.written,
                        missing_samples: silence,
                    },
                }
            } else {
                self.pending_gap = true;
                Placement {
                    silence: 0,
                    requested_silence: 0,
                    adjustment: Adjustment::None,
                }
            }
        } else if self.written > expected.saturating_add(self.tolerance) {
            self.pending_gap = false;
            let extra_samples = self.written - expected;
            Placement {
                silence: 0,
                requested_silence: 0,
                adjustment: if extra_samples > OVERLAP_EVENT_THRESHOLD {
                    Adjustment::Overlap {
                        sample: self.written,
                        extra_samples,
                    }
                } else {
                    Adjustment::None
                },
            }
        } else {
            self.pending_gap = false;
            Placement {
                silence: 0,
                requested_silence: 0,
                adjustment: Adjustment::None,
            }
        };

        self.written = self
            .written
            .saturating_add(placement.silence)
            .saturating_add(samples);
        placement
    }

    pub fn written(&self) -> u64 {
        self.written
    }

    pub fn append_silence(&mut self, samples: u64) {
        self.written = self.written.saturating_add(samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns_for_samples(samples: u64) -> u64 {
        samples * 1_000_000_000 / SAMPLE_RATE as u64
    }

    #[test]
    fn contiguous_blocks_need_no_silence() {
        let mut timeline = Timeline::new(1_000_000_000);
        let first = timeline.place(1_000_000_000, 1_600);
        let second = timeline.place(1_000_000_000 + ns_for_samples(1_600), 1_600);
        assert_eq!(first.silence, 0);
        assert_eq!(
            second,
            Placement {
                silence: 0,
                requested_silence: 0,
                adjustment: Adjustment::None
            }
        );
        assert_eq!(timeline.written(), 3_200);
    }

    #[test]
    fn one_hundred_ms_hole_is_confirmed_by_next_block() {
        let mut timeline = Timeline::new(0);
        timeline.place(0, 1_600);
        let suspect = timeline.place(ns_for_samples(3_200), 1_600);
        assert_eq!(suspect.silence, 0);
        let placement = timeline.place(ns_for_samples(4_800), 1_600);
        assert_eq!(placement.silence, 1_600);
        assert_eq!(
            placement.adjustment,
            Adjustment::Gap {
                sample: 3_200,
                missing_samples: 1_600
            }
        );
    }

    #[test]
    fn one_off_timestamp_spike_does_not_insert_silence() {
        let mut timeline = Timeline::new(0);
        timeline.place(0, 1_600);
        assert_eq!(timeline.place(ns_for_samples(3_200), 1_600).silence, 0);
        assert_eq!(timeline.place(ns_for_samples(3_200), 1_600).silence, 0);
        assert_eq!(timeline.written(), 4_800);
    }

    #[test]
    fn late_start_is_aligned_to_session() {
        let mut timeline = Timeline::new(10);
        let placement = timeline.place(2_000_000_010, 1_000);
        assert_eq!(placement.silence, 32_000);
        assert_eq!(placement.requested_silence, 32_000);
        assert_eq!(placement.adjustment, Adjustment::Start { sample: 32_000 });
    }

    #[test]
    fn small_timestamp_jitter_is_ignored() {
        for jitter_ns in [-10_000_000_i64, 10_000_000] {
            let mut timeline = Timeline::new(1_000_000_000);
            timeline.place(1_000_000_000, 1_600);
            let expected = 1_100_000_000_i64 + jitter_ns;
            let placement = timeline.place(expected as u64, 1_600);
            assert_eq!(
                placement,
                Placement {
                    silence: 0,
                    requested_silence: 0,
                    adjustment: Adjustment::None
                }
            );
        }
    }

    #[test]
    fn huge_clock_jump_is_capped_without_u64_multiplication_overflow() {
        let mut timeline = Timeline::new(0);
        let placement = timeline.place(u64::MAX, 1);
        assert_eq!(placement.silence, MAX_GAP_SAMPLES);
        assert_eq!(
            placement.adjustment,
            Adjustment::Start {
                sample: MAX_GAP_SAMPLES
            }
        );
        assert!(placement.requested_silence > MAX_GAP_SAMPLES);
        assert_eq!(timeline.written(), MAX_GAP_SAMPLES + 1);
    }
}
