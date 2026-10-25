//! Feedback delay effect.
//!
//! Layout (`DelayParameter` in the Ryujinx audio renderer):
//!
//! ```text
//! +0x00 u8[6]  input channel indices
//! +0x06 u8[6]  output channel indices
//! +0x0C u16    maximum channel count
//! +0x0E u16    channel count
//! +0x10 u32    maximum delay time (ms)
//! +0x14 u32    delay time (ms)
//! +0x18 u32    sample rate (Q14 Hz)
//! +0x1C u32    input gain (Q14)
//! +0x20 u32    feedback gain (Q14)
//! +0x24 u32    output gain (Q14)
//! +0x28 u32    dry gain (Q14)
//! +0x2C u32    channel spread (Q14)
//! +0x30 u32    low pass amount (Q14)
//! +0x34 u8     client usage status
//! ```
//!
//! The DSP scales the input by 64, applies a channel matrix built from the
//! feedback and spread gains, low-passes the feedback value and scales the
//! result back down by 64. The matrix keeps every feedback row at the requested
//! gain, so the clamped gain is the only stability condition.

use super::line::{DelayLine, ms_samples};
use super::{MAX_CHANNELS, Route, gain, q14, u16_at, u32_at};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DelayParams {
    pub route: Route,
    pub delay_time_max: u32,
    pub delay_time: u32,
    pub sample_rate: u32,
    pub in_gain: f32,
    pub feedback_gain: f32,
    pub out_gain: f32,
    pub dry_gain: f32,
    pub channel_spread: f32,
    pub low_pass: f32,
    pub status: u8,
}

impl DelayParams {
    pub(crate) fn parse(d: &[u8]) -> DelayParams {
        let channels = usize::from(u16_at(d, 0x0E));
        let rate = q14(u32_at(d, 0x18) as i32);
        DelayParams {
            route: Route::parse(d).with_channels(channels),
            delay_time_max: u32_at(d, 0x10),
            delay_time: u32_at(d, 0x14),
            sample_rate: if rate.is_finite() && (1000.0..=192_000.0).contains(&rate) {
                rate as u32
            } else {
                0
            },
            in_gain: gain(u32_at(d, 0x1C) as i32),
            feedback_gain: gain(u32_at(d, 0x20) as i32),
            out_gain: gain(u32_at(d, 0x24) as i32),
            dry_gain: gain(u32_at(d, 0x28) as i32),
            channel_spread: q14(u32_at(d, 0x2C) as i32).clamp(0.0, 1.0),
            low_pass: q14(u32_at(d, 0x30) as i32).clamp(0.0, 1.0),
            status: super::byte_at(d, 0x34),
        }
    }

    pub(crate) fn routable(&self) -> bool {
        self.route.routable() && self.delay_time_max >= 1 && self.delay_time <= self.delay_time_max
    }
}

pub(crate) struct DelayEffect {
    params: DelayParams,
    rate: u32,
    lines: [DelayLine; MAX_CHANNELS],
    delay: [usize; MAX_CHANNELS],
    channels: usize,
    in_gain: f32,
    feedback: f32,
    base: f32,
    cross: f32,
    low_feedback: f32,
    low_base: f32,
    low_state: [f32; MAX_CHANNELS],
    wet: f32,
    dry: f32,
    target_wet: f32,
    target_dry: f32,
}

impl DelayEffect {
    pub(super) fn new(p: DelayParams, sample_rate: u32) -> DelayEffect {
        let mut d = DelayEffect {
            params: p,
            rate: 48,
            lines: std::array::from_fn(|_| DelayLine::new(2)),
            delay: [1; MAX_CHANNELS],
            channels: p.route.channels,
            in_gain: p.in_gain,
            feedback: 0.0,
            base: 0.0,
            cross: 0.0,
            low_feedback: 0.0,
            low_base: 1.0,
            low_state: [0.0; MAX_CHANNELS],
            wet: p.out_gain,
            dry: p.dry_gain,
            target_wet: p.out_gain,
            target_dry: p.dry_gain,
        };
        d.rate = rate_khz(p.sample_rate, sample_rate);
        d.allocate();
        d.retune();
        d
    }

    fn allocate(&mut self) {
        let max = ms_samples(self.rate, self.params.delay_time_max as f32).max(2);
        self.lines = std::array::from_fn(|_| DelayLine::new(max));
        self.channels = self.params.route.channels.min(MAX_CHANNELS);
        for (c, delay) in self.delay.iter_mut().enumerate() {
            *delay = ms_samples(self.rate, self.params.delay_time as f32).clamp(1, max - 1);
            if c >= self.channels {
                *delay = 1;
            }
        }
        self.low_state = [0.0; MAX_CHANNELS];
    }

    fn retune(&mut self) {
        let p = &self.params;
        self.feedback = (p.feedback_gain * 0.98).clamp(0.0, 0.999);
        self.in_gain = p.in_gain;
        self.base = (1.0 - p.channel_spread) * self.feedback;
        self.cross = if self.channels == 4 || self.channels == 6 {
            p.channel_spread * 0.5 * self.feedback
        } else {
            p.channel_spread * self.feedback
        };
        self.low_feedback = (0.95 * p.low_pass).clamp(0.0, 0.999);
        self.low_base = 1.0 - self.low_feedback;
        self.target_wet = p.out_gain;
        self.target_dry = p.dry_gain;
    }

    pub(super) fn configure(&mut self, p: DelayParams, sample_rate: u32) {
        let rate = rate_khz(p.sample_rate, sample_rate);
        let structural = p.delay_time_max != self.params.delay_time_max
            || p.route.channels != self.params.route.channels
            || rate != self.rate;
        self.params = p;
        self.rate = rate;
        if structural {
            self.allocate();
            self.wet = p.out_gain;
            self.dry = p.dry_gain;
        } else {
            let max = self.lines[0].len();
            for (c, delay) in self.delay.iter_mut().enumerate() {
                *delay = ms_samples(rate, p.delay_time as f32).clamp(1, max - 1);
                if c >= p.route.channels {
                    *delay = 1;
                }
            }
            self.channels = p.route.channels.min(MAX_CHANNELS);
        }
        self.retune();
    }

    fn feedback_row(&self, channel: usize, delayed: &[f32; MAX_CHANNELS]) -> f32 {
        let (base, cross) = (self.base, self.cross);
        match self.channels {
            1 => delayed[0] * self.feedback,
            2 => {
                if channel == 0 {
                    delayed[0] * base + delayed[1] * cross
                } else {
                    delayed[1] * base + delayed[0] * cross
                }
            }
            4 => match channel {
                0 => delayed[0] * base + (delayed[1] + delayed[2]) * cross,
                1 => delayed[1] * base + (delayed[0] + delayed[3]) * cross,
                2 => delayed[2] * base + (delayed[0] + delayed[3]) * cross,
                _ => delayed[3] * base + (delayed[1] + delayed[2]) * cross,
            },
            6 => match channel {
                0 => delayed[0] * base + (delayed[2] + delayed[4]) * cross,
                1 => delayed[1] * base + (delayed[2] + delayed[5]) * cross,
                2 => delayed[2] * base + (delayed[0] + delayed[1]) * cross,
                3 => delayed[3] * self.feedback,
                4 => delayed[4] * base + (delayed[0] + delayed[5]) * cross,
                _ => delayed[5] * base + (delayed[1] + delayed[4]) * cross,
            },
            _ => 0.0,
        }
    }

    pub(crate) fn process(&mut self, input: &[Vec<f32>], out: &mut [Vec<f32>], n: usize) {
        let ch = self.channels.min(input.len()).min(out.len());
        if ch == 0 || input[0].len() < n || out[0].len() < n {
            return;
        }
        let (dry_from, dry_to) = (self.dry, self.target_dry);
        let (wet_from, wet_to) = (self.wet, self.target_wet);
        let dry_step = (dry_to - dry_from) / n.max(1) as f32;
        let wet_step = (wet_to - wet_from) / n.max(1) as f32;
        let mut dry = dry_from;
        let mut wet = wet_from;
        for s in 0..n {
            let mut x = [0.0f32; MAX_CHANNELS];
            let mut delayed = [0.0f32; MAX_CHANNELS];
            for c in 0..ch {
                x[c] = input[c][s] * 64.0;
                delayed[c] = self.lines[c].read(self.delay[c]);
            }
            let mut next = [0.0f32; MAX_CHANNELS];
            for c in 0..ch {
                let feedback = self.feedback_row(c, &delayed);
                let value = x[c] * self.in_gain + feedback;
                let low = self.low_feedback * self.low_state[c] + value * self.low_base;
                let low = if low.is_finite() {
                    low.clamp(-1e6, 1e6)
                } else {
                    0.0
                };
                self.low_state[c] = low;
                next[c] = low;
            }
            for c in 0..ch {
                self.lines[c].write(next[c]);
                let y = (x[c] * dry + delayed[c] * wet) / 64.0;
                out[c][s] = if y.is_finite() {
                    y.clamp(-1e6, 1e6)
                } else {
                    self.lines[c].reset();
                    self.low_state[c] = 0.0;
                    0.0
                };
            }
            dry += dry_step;
            wet += wet_step;
        }
        self.dry = dry_to;
        self.wet = wet_to;
    }
}

/// Samples per millisecond from the effect block, falling back to the renderer
/// rate when the block leaves it empty.
fn rate_khz(parameter_hz: u32, renderer_hz: u32) -> u32 {
    let hz = if (8000..=192_000).contains(&parameter_hz) {
        parameter_hz
    } else {
        renderer_hz
    };
    (hz / 1000).clamp(1, 192)
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

    fn params(channels: usize) -> DelayParams {
        DelayParams {
            route: route(channels),
            delay_time_max: 20,
            delay_time: 2,
            sample_rate: 48_000,
            in_gain: 1.0,
            feedback_gain: 0.5,
            out_gain: 1.0,
            dry_gain: 0.0,
            channel_spread: 0.0,
            low_pass: 0.0,
            status: 2,
        }
    }

    fn scratch(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (
            vec![vec![0.0; n]; MAX_CHANNELS],
            vec![vec![0.0; n]; MAX_CHANNELS],
        )
    }

    #[test]
    fn delay_parameter_offsets_match_the_public_layout() {
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[6] = 0;
        d[0x0C..0x0E].copy_from_slice(&2u16.to_le_bytes());
        d[0x0E..0x10].copy_from_slice(&2u16.to_le_bytes());
        d[0x10..0x14].copy_from_slice(&50u32.to_le_bytes());
        d[0x14..0x18].copy_from_slice(&10u32.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&((48_000u32 << 14) as i32).to_le_bytes());
        for (k, v) in [1.0f32, 0.5, 0.75, 0.25, 0.0, 0.5].into_iter().enumerate() {
            d[0x1C + k * 4..0x20 + k * 4].copy_from_slice(&((v * 16384.0) as i32).to_le_bytes());
        }
        let p = DelayParams::parse(&d);
        assert_eq!(p.route.channels, 2);
        assert_eq!((p.delay_time_max, p.delay_time), (50, 10));
        assert_eq!(p.sample_rate, 48_000);
        assert!((p.feedback_gain - 0.5).abs() < 1e-6);
        assert!(p.routable());
    }

    #[test]
    fn impulse_is_delayed_by_the_requested_time() {
        let mut delay = DelayEffect::new(params(1), 48_000);
        let n = 480;
        let (mut input, mut out) = scratch(n);
        input[0][0] = 0.5;
        delay.process(&input, &mut out, n);
        // 2 ms at 48 kHz is 96 samples.
        assert_eq!(out[0][95], 0.0);
        assert!((out[0][96] - 0.5).abs() < 1e-4, "{}", out[0][96]);
        // The feedback copy is attenuated by 0.98 * 0.5.
        assert!((out[0][192] - 0.245).abs() < 1e-3, "{}", out[0][192]);
    }

    #[test]
    fn feedback_decays_and_stays_finite() {
        let mut d = params(2);
        d.feedback_gain = 1.0;
        d.channel_spread = 1.0;
        d.low_pass = 0.9;
        let mut delay = DelayEffect::new(d, 48_000);
        let n = 480;
        let (mut input, mut out) = scratch(n);
        input[0][0] = 0.5;
        input[1][0] = -0.5;
        let mut energies = Vec::new();
        for block in 0..20 {
            delay.process(&input, &mut out, n);
            input[0].fill(0.0);
            input[1].fill(0.0);
            if block >= 2 {
                let e: f64 = out
                    .iter()
                    .take(2)
                    .flat_map(|c| c[..n].iter())
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum();
                energies.push(e);
            }
            assert!(
                out.iter()
                    .take(2)
                    .all(|c| c[..n].iter().all(|v| v.is_finite()))
            );
            assert!(
                out.iter()
                    .take(2)
                    .all(|c| c[..n].iter().all(|v| v.abs() < 10.0))
            );
        }
        for pair in energies.windows(2) {
            assert!(
                pair[1] <= pair[0] * 1.01,
                "feedback must decay: {energies:?}"
            );
        }
    }

    #[test]
    fn extreme_delay_parameters_stay_finite() {
        let mut d = params(6);
        d.feedback_gain = 4.0;
        d.channel_spread = 1.0;
        d.low_pass = 1.0;
        d.in_gain = 4.0;
        d.delay_time_max = 50;
        d.delay_time = 50;
        let mut delay = DelayEffect::new(d, 48_000);
        let n = 240;
        let (mut input, mut out) = scratch(n);
        for block in 0..30 {
            for (c, channel) in input.iter_mut().take(6).enumerate() {
                for (s, v) in channel.iter_mut().enumerate() {
                    *v = if block == 0 && s == 0 {
                        (c as f32 + 1.0) * 0.1
                    } else {
                        0.0
                    };
                }
            }
            delay.process(&input, &mut out, n);
            for channel in out.iter().take(6) {
                for &v in &channel[..n] {
                    assert!(v.is_finite(), "non-finite delay output");
                    assert!(v.abs() < 1e6, "unbounded delay output {v}");
                }
            }
        }
    }

    #[test]
    fn unusable_block_is_not_routed() {
        let mut d = params(1);
        d.delay_time_max = 0;
        assert!(!d.routable());
        d.delay_time_max = 20;
        d.delay_time = 30;
        assert!(!d.routable());
        d.delay_time = 10;
        assert!(d.routable(), "zero gains are still a valid (muted) delay");
        d.route.channels = 3;
        assert!(!d.routable());
    }
}
