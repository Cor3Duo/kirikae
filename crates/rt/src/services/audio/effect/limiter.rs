//! Look-ahead limiter (DRC) used by the `audren` limiter effect.
//!
//! Layout (`LimiterParameter` in the Ryujinx audio renderer):
//!
//! ```text
//! +0x00 u8[6]  input channel indices
//! +0x06 u8[6]  output channel indices
//! +0x0C u16    maximum channel count
//! +0x0E u16    channel count
//! +0x10 i32    sample rate (Hz)
//! +0x14 i32    maximum look-ahead time (us)
//! +0x18 i32    attack time (us)
//! +0x1C i32    release time (us)
//! +0x20 i32    look-ahead time (us)
//! +0x24 f32    attack coefficient
//! +0x28 f32    release coefficient
//! +0x2C f32    threshold
//! +0x30 f32    input gain
//! +0x34 f32    output gain
//! +0x38 i32    minimum delay buffer samples
//! +0x3C i32    maximum delay buffer samples
//! +0x40 u8     client usage status
//! +0x41 bool   statistics enabled
//! +0x42 bool   statistics reset
//! ```
//!
//! The detector and the gain are one-pole averages; the output is the delayed
//! input multiplied by the smoothed gain, as in
//! `LimiterCommandVersion2.ProcessLimiter`.
//!
//! With statistics enabled, every processed channel records
//! `InputMax = max(|input * input_gain|)` and
//! `CompressionGainMin = min(gain)`. They are exported through
//! `EffectResultState.SpecificData` as `LimiterStatistics` (two `f32[6]`
//! arrays, 48 bytes), which `EffectOutStatusVersion2` places after the state
//! and its 15 reserved bytes. `StatisticsReset` clears both counters at the
//! start of a block.

use super::{MAX_CHANNELS, Route, byte_at, f32_at, finite, i32_at, u16_at};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LimiterParams {
    pub route: Route,
    pub sample_rate: u32,
    pub attack: f32,
    pub release: f32,
    pub threshold: f32,
    pub input_gain: f32,
    pub output_gain: f32,
    pub delay: usize,
    pub status: u8,
    pub statistics_enabled: bool,
    pub statistics_reset: bool,
}

impl Default for LimiterParams {
    fn default() -> LimiterParams {
        LimiterParams {
            route: Route::default(),
            sample_rate: 48_000,
            attack: 0.0,
            release: 0.0,
            threshold: 1.0,
            input_gain: 1.0,
            output_gain: 1.0,
            delay: 0,
            status: 0,
            statistics_enabled: false,
            statistics_reset: false,
        }
    }
}

impl LimiterParams {
    pub(crate) fn parse(d: &[u8]) -> LimiterParams {
        let channels = usize::from(u16_at(d, 0x0E));
        let delay_min = i32_at(d, 0x38).max(0) as usize;
        let delay_max = i32_at(d, 0x3C).max(0) as usize;
        let rate = i32_at(d, 0x10);
        LimiterParams {
            route: Route::parse(d).with_channels(channels),
            sample_rate: if (8000..=192_000).contains(&rate) {
                rate as u32
            } else {
                0
            },
            attack: finite(f32_at(d, 0x24), 0.0, 1.0),
            release: finite(f32_at(d, 0x28), 0.0, 1.0),
            threshold: finite(f32_at(d, 0x2C), 0.0001, 8.0),
            input_gain: finite(f32_at(d, 0x30), 0.0, 8.0),
            output_gain: finite(f32_at(d, 0x34), 0.0, 8.0),
            // The ring wraps at the minimum look-ahead count, so that is the
            // usable delay; keep the buffer size for validation only.
            delay: delay_min.min(delay_max).clamp(1, 8192),
            status: byte_at(d, 0x40),
            statistics_enabled: byte_at(d, 0x41) != 0,
            statistics_reset: byte_at(d, 0x42) != 0,
        }
    }

    pub(crate) fn routable(&self) -> bool {
        self.route.routable() && self.delay >= 1 && self.threshold > 0.0
    }
}

pub(crate) struct Limiter {
    params: LimiterParams,
    channels: usize,
    detector: [f32; MAX_CHANNELS],
    gain: [f32; MAX_CHANNELS],
    buffer: [Vec<f32>; MAX_CHANNELS],
    position: [usize; MAX_CHANNELS],
    input_max: [f32; MAX_CHANNELS],
    gain_min: [f32; MAX_CHANNELS],
}

impl Limiter {
    pub(super) fn new(p: LimiterParams, sample_rate: u32) -> Limiter {
        let mut limiter = Limiter {
            params: p,
            channels: p.route.channels,
            detector: [0.0; MAX_CHANNELS],
            gain: [1.0; MAX_CHANNELS],
            buffer: std::array::from_fn(|_| Vec::new()),
            position: [0; MAX_CHANNELS],
            input_max: [0.0; MAX_CHANNELS],
            gain_min: [1.0; MAX_CHANNELS],
        };
        limiter.allocate(sample_rate);
        limiter
    }

    fn allocate(&mut self, sample_rate: u32) {
        if self.params.sample_rate < 8000 {
            self.params.sample_rate = sample_rate.max(8000);
        }
        self.channels = self.params.route.channels.min(MAX_CHANNELS);
        let delay = self.params.delay.clamp(1, 8192);
        self.buffer = std::array::from_fn(|_| vec![0.0; delay]);
        self.position = [0; MAX_CHANNELS];
        self.detector = [0.0; MAX_CHANNELS];
        self.gain = [1.0; MAX_CHANNELS];
        self.reset_statistics();
    }

    fn reset_statistics(&mut self) {
        self.input_max = [0.0; MAX_CHANNELS];
        self.gain_min = [1.0; MAX_CHANNELS];
    }

    /// Whether `LimiterStatistics` should be reported for this slot.
    pub(crate) fn statistics_enabled(&self) -> bool {
        self.params.statistics_enabled
    }

    /// `LimiterStatistics`: per-channel `InputMax` and `CompressionGainMin`.
    pub(crate) fn statistics(&self) -> ([f32; MAX_CHANNELS], [f32; MAX_CHANNELS]) {
        (self.input_max, self.gain_min)
    }

    pub(super) fn configure(&mut self, p: LimiterParams, sample_rate: u32) {
        let rebuild =
            p.route.channels != self.params.route.channels || p.delay != self.params.delay;
        self.params = p;
        if rebuild {
            self.allocate(sample_rate);
        } else if self.params.sample_rate < 8000 {
            self.params.sample_rate = sample_rate.max(8000);
        }
    }

    pub(crate) fn process(&mut self, input: &[Vec<f32>], out: &mut [Vec<f32>], n: usize) {
        let ch = self.channels.min(input.len()).min(out.len());
        if ch == 0 || input[0].len() < n || out[0].len() < n {
            return;
        }
        let p = self.params;
        let ring = p.delay.clamp(1, 8192);
        if p.statistics_reset {
            self.reset_statistics();
        }
        for c in 0..ch {
            let buffer = &mut self.buffer[c];
            buffer.resize(ring, 0.0);
            let mut position = self.position[c].min(ring - 1);
            let mut detector = self.detector[c];
            let mut gain = self.gain[c];
            let mut input_max = self.input_max[c];
            let mut gain_min = self.gain_min[c];
            for s in 0..n {
                let raw = input[c][s];
                let x = raw / 32767.0 * p.input_gain;
                let level = x.abs();
                let alpha = if level > detector {
                    p.attack
                } else {
                    p.release
                };
                detector += alpha * (level - detector);
                let attenuation = if detector > p.threshold {
                    p.threshold / detector
                } else {
                    1.0
                };
                let alpha = if gain > attenuation {
                    p.attack
                } else {
                    p.release
                };
                gain += alpha * (attenuation - gain);
                let delayed = buffer[position];
                buffer[position] = x;
                position += 1;
                if position >= ring {
                    position = 0;
                }
                let y = delayed * gain * p.output_gain * 32767.0;
                out[c][s] = if y.is_finite() {
                    y.clamp(-32768.0, 32768.0)
                } else {
                    gain = 1.0;
                    detector = 0.0;
                    0.0
                };
                if p.statistics_enabled {
                    if level.is_finite() {
                        input_max = input_max.max(level);
                    }
                    gain_min = gain_min.min(gain);
                }
            }
            self.position[c] = position;
            self.detector[c] = detector;
            self.gain[c] = gain;
            self.input_max[c] = input_max;
            self.gain_min[c] = gain_min;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(channels: usize) -> Route {
        let mut r = Route {
            channels,
            ..Route::default()
        };
        for c in 0..channels {
            r.input[c] = c as u8;
            r.output[c] = c as u8;
        }
        r
    }

    fn params() -> LimiterParams {
        LimiterParams {
            route: route(1),
            sample_rate: 48_000,
            attack: 0.5,
            release: 0.01,
            threshold: 0.5,
            input_gain: 1.0,
            output_gain: 1.0,
            delay: 1,
            status: 2,
            statistics_enabled: false,
            statistics_reset: false,
        }
    }

    #[test]
    fn limiter_parameter_offsets_match_the_public_layout() {
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[6] = 0;
        d[0x0C..0x0E].copy_from_slice(&2u16.to_le_bytes());
        d[0x0E..0x10].copy_from_slice(&2u16.to_le_bytes());
        d[0x10..0x14].copy_from_slice(&48_000i32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&0.4f32.to_le_bytes());
        d[0x28..0x2C].copy_from_slice(&0.05f32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0.6f32.to_le_bytes());
        d[0x30..0x34].copy_from_slice(&1.25f32.to_le_bytes());
        d[0x34..0x38].copy_from_slice(&0.9f32.to_le_bytes());
        d[0x38..0x3C].copy_from_slice(&96i32.to_le_bytes());
        d[0x3C..0x40].copy_from_slice(&480i32.to_le_bytes());
        d[0x40] = 2;
        d[0x41] = 1;
        d[0x42] = 1;
        let p = LimiterParams::parse(&d);
        assert_eq!(p.route.channels, 2);
        assert_eq!(p.sample_rate, 48_000);
        assert!((p.threshold - 0.6).abs() < 1e-6);
        assert_eq!(p.delay, 96);
        assert_eq!(p.status, 2);
        assert!(p.statistics_enabled && p.statistics_reset);
        assert!(p.routable());
    }

    #[test]
    fn limiter_attenuates_a_hot_signal_and_keeps_silence() {
        let mut limiter = Limiter::new(params(), 48_000);
        let n = 480;
        let mut input = vec![vec![0.0f32; n]; MAX_CHANNELS];
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        input[0][..200].fill(28_000.0);
        limiter.process(&input, &mut out, n);
        let peak = out[0][..n].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.0, "the delayed signal must reach the output");
        assert!(
            peak < 28_000.0,
            "the limiter must attenuate above threshold: {peak}"
        );
        assert!(out[0][..n].iter().all(|v| v.is_finite()));

        let silence = vec![vec![0.0f32; n]; MAX_CHANNELS];
        for _ in 0..10 {
            limiter.process(&silence, &mut out, n);
            assert!(out[0][..n].iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn extreme_limiter_parameters_stay_finite() {
        let mut p = params();
        p.attack = 1.0;
        p.release = 1.0;
        p.threshold = 0.0001;
        p.input_gain = 8.0;
        p.output_gain = 8.0;
        p.route = route(6);
        p.delay = 8192;
        let mut limiter = Limiter::new(p, 48_000);
        let n = 240;
        let mut input = vec![vec![0.0f32; n]; MAX_CHANNELS];
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        for block in 0..10 {
            for channel in input.iter_mut().take(6) {
                channel.fill(if block == 0 { 40_000.0 } else { 0.0 });
            }
            limiter.process(&input, &mut out, n);
            for channel in out.iter().take(6) {
                assert!(channel[..n].iter().all(|v| v.is_finite()));
                assert!(channel[..n].iter().all(|v| v.abs() <= 32768.0));
            }
        }
    }

    #[test]
    fn limiter_statistics_track_max_input_and_min_gain() {
        let mut p = params();
        p.statistics_enabled = true;
        let mut limiter = Limiter::new(p, 48_000);
        let n = 480;
        let mut input = vec![vec![0.0f32; n]; MAX_CHANNELS];
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        input[0][..200].fill(28_000.0);
        limiter.process(&input, &mut out, n);
        let (input_max, gain_min) = limiter.statistics();
        let expected = 28_000.0 / 32767.0;
        assert!(
            (input_max[0] - expected).abs() < 1e-4,
            "max input {} vs {expected}",
            input_max[0]
        );
        assert!(gain_min[0] < 1.0, "the limiter must have reduced gain");
        assert_eq!(input_max[1], 0.0);
        assert_eq!(gain_min[1], 1.0);

        // The reset flag clears the counters at the start of the next block.
        let mut p = params();
        p.statistics_enabled = true;
        let mut limiter = Limiter::new(p, 48_000);
        input[0].fill(28_000.0);
        limiter.process(&input, &mut out, n);
        let before = limiter.statistics().1[0];
        let mut reset = params();
        reset.statistics_enabled = true;
        reset.statistics_reset = true;
        limiter.configure(reset, 48_000);
        let silence = vec![vec![0.0f32; n]; MAX_CHANNELS];
        limiter.process(&silence, &mut out, n);
        let (input_max, gain_min) = limiter.statistics();
        assert_eq!(input_max[0], 0.0, "reset must clear the input maximum");
        assert!(
            gain_min[0] >= before,
            "the reset block reports its own minimum"
        );
        assert!(gain_min[0] <= 1.0);
    }

    #[test]
    fn a_zero_length_delay_is_not_routable() {
        let mut p = params();
        p.delay = 0;
        assert!(!p.routable());
    }
}
