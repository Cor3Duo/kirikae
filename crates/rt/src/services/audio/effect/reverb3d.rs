//! I3DL2 3D reverb (effect type 5).
//!
//! Layout (`Reverb3dParameter` in the Ryujinx audio renderer):
//!
//! ```text
//! +0x00 u8[6]  input channel indices
//! +0x06 u8[6]  output channel indices
//! +0x0C u16    maximum channel count
//! +0x0E u16    channel count
//! +0x10 u32    reserved
//! +0x14 u32    sample rate (Hz)
//! +0x18 f32    room high-frequency gain (millibels)
//! +0x1C f32    high-frequency reference (Hz)
//! +0x20 f32    decay time (ms)
//! +0x24 f32    high-frequency decay ratio
//! +0x28 f32    room gain (millibels)
//! +0x2C f32    reflections gain (millibels)
//! +0x30 f32    reverb gain (millibels)
//! +0x34 f32    diffusion (percent)
//! +0x38 f32    reflection delay (s)
//! +0x3C f32    reverb delay time (s)
//! +0x40 f32    density (percent)
//! +0x44 f32    dry gain
//! ```
//!
//! The network is the four-line feedback delay network of Ryujinx's
//! `Reverb3dState`/`Reverb3dCommand` (MIT): a pre-delay feeding twenty early
//! reflection taps, a damped four-line late network and a fifth-channel comb.
//! Unlike the type 4 reverb, all gains and times here are plain IEEE floats.

use super::line::{DelayLine, ms_samples};
use super::{MAX_CHANNELS, Route, byte_at, f32_at, finite, u16_at, u32_at};

const FDN_MIN_MS: [f32; 4] = [5.0, 6.0, 13.0, 14.0];
const FDN_MAX_MS: [f32; 4] = [45.704, 82.782, 149.94, 271.58];
const DECAY1_MAX_MS: [f32; 4] = [17.0, 13.0, 9.0, 7.0];
const DECAY2_MAX_MS: [f32; 4] = [19.0, 11.0, 10.0, 6.0];
const PRE_MAX_MS: f32 = 400.0;
const COMB_MS: f32 = 5.0;

const EARLY_TIMES_S: [f32; 20] = [
    0.017_136, 0.059_154, 0.161_73, 0.390_19, 0.425_26, 0.455_41, 0.689_74, 0.745_91, 0.833_84,
    0.859_5, 0.0, 0.075_024, 0.168_79, 0.299_9, 0.337_44, 0.371_9, 0.599_01, 0.716_74, 0.817_86,
    0.851_66,
];
const EARLY_GAINS: [f32; 20] = [
    0.670_96, 0.610_27, 1.0, 0.356_80, 0.683_61, 0.659_78, 0.519_39, 0.247_12, 0.459_45, 0.450_21,
    0.641_96, 0.548_79, 0.929_25, 0.382_70, 0.728_67, 0.697_94, 0.5464, 0.245_63, 0.452_14,
    0.440_42,
];

const EARLY_MONO: [u8; 20] = [0; 20];
const INDEX_MONO: [u8; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];
const FEEDBACK_MONO: [i8; 1] = [0];

const EARLY_STEREO: [u8; 20] = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1];
const INDEX_STEREO: [u8; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];
const FEEDBACK_STEREO: [i8; 2] = [0, 1];

const EARLY_QUAD: [u8; 20] = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3];
const INDEX_QUAD: [u8; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];
const FEEDBACK_QUAD: [i8; 4] = [0, 1, 2, 3];

const EARLY_SURROUND: [u8; 40] = [
    4, 5, 0, 5, 0, 5, 1, 5, 1, 5, 1, 5, 1, 5, 2, 5, 2, 5, 2, 5, 1, 5, 1, 5, 1, 5, 0, 5, 0, 5, 0, 5,
    0, 5, 3, 5, 3, 5, 3, 5,
];
const INDEX_SURROUND: [u8; 40] = [
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14,
    14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
];
const FEEDBACK_SURROUND: [i8; 6] = [0, 1, 2, 3, -1, 3];

struct Mapping {
    early: &'static [u8],
    index: &'static [u8],
    feedback: &'static [i8],
}

const MAPPING_MONO: Mapping = Mapping {
    early: &EARLY_MONO,
    index: &INDEX_MONO,
    feedback: &FEEDBACK_MONO,
};
const MAPPING_STEREO: Mapping = Mapping {
    early: &EARLY_STEREO,
    index: &INDEX_STEREO,
    feedback: &FEEDBACK_STEREO,
};
const MAPPING_QUAD: Mapping = Mapping {
    early: &EARLY_QUAD,
    index: &INDEX_QUAD,
    feedback: &FEEDBACK_QUAD,
};
const MAPPING_SURROUND: Mapping = Mapping {
    early: &EARLY_SURROUND,
    index: &INDEX_SURROUND,
    feedback: &FEEDBACK_SURROUND,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Reverb3dParams {
    pub route: Route,
    pub sample_rate: u32,
    pub room_hf: f32,
    pub hf_reference: f32,
    pub decay_time: f32,
    pub hf_decay_ratio: f32,
    pub room_gain: f32,
    pub reflections_gain: f32,
    pub reverb_gain: f32,
    pub diffusion: f32,
    pub reflection_delay: f32,
    pub reverb_delay_time: f32,
    pub density: f32,
    pub dry_gain: f32,
    pub status: u8,
}

impl Reverb3dParams {
    pub(crate) fn parse(d: &[u8]) -> Reverb3dParams {
        let channels = usize::from(u16_at(d, 0x0E));
        let rate = u32_at(d, 0x14);
        Reverb3dParams {
            route: Route::parse(d).with_channels(channels),
            sample_rate: if (8000..=192_000).contains(&rate) {
                rate
            } else {
                0
            },
            room_hf: finite(f32_at(d, 0x18), -10_000.0, 10_000.0),
            hf_reference: finite(f32_at(d, 0x1C), 0.0, 24_000.0),
            decay_time: finite(f32_at(d, 0x20), 0.0, 120_000.0),
            hf_decay_ratio: finite(f32_at(d, 0x24), 0.01, 1.0),
            room_gain: finite(f32_at(d, 0x28), -10_000.0, 10_000.0),
            reflections_gain: finite(f32_at(d, 0x2C), -10_000.0, 10_000.0),
            reverb_gain: finite(f32_at(d, 0x30), -10_000.0, 10_000.0),
            diffusion: finite(f32_at(d, 0x34), 0.0, 100.0),
            reflection_delay: finite(f32_at(d, 0x38), 0.0, 1.0),
            reverb_delay_time: finite(f32_at(d, 0x3C), 0.0, 1.0),
            density: finite(f32_at(d, 0x40), 0.0, 100.0),
            dry_gain: finite(f32_at(d, 0x44), 0.0, 100.0),
            status: byte_at(d, 0x48),
        }
    }

    pub(crate) fn routable(&self) -> bool {
        self.route.routable()
            && (8000..=192_000).contains(&self.sample_rate)
            && self.decay_time > 0.0
            && self.hf_reference > 0.0
            && self.hf_decay_ratio > 0.0
    }
}

/// `FloatingPointHelper.Pow10` from the audio renderer: valid input range is
/// limited to avoid the huge exponents a malformed block could request.
fn pow10(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else if x <= -5.3 {
        0.0
    } else {
        10f32.powf(x)
    }
}

fn db_to_linear(millibels: f32) -> f32 {
    pow10(finite(millibels, -5000.0, 5000.0).min(5000.0) / 2000.0)
}

pub(crate) struct Reverb3d {
    params: Reverb3dParams,
    rate: u32,
    fdn: [DelayLine; 4],
    decay1: [DelayLine; 4],
    decay2: [DelayLine; 4],
    pre: DelayLine,
    comb: DelayLine,
    fdn_delay: [usize; 4],
    decay1_delay: [usize; 4],
    decay2_delay: [usize; 4],
    early_delay: [usize; 20],
    early_gain: [f32; 20],
    reflection_delay: usize,
    comb_delay: usize,
    decay_direct: [f32; 4],
    decay_current_fdn: [f32; 4],
    decay_current_out: [f32; 4],
    previous_decayed: [f32; 4],
    previous_pre: f32,
    previous_pre_gain: f32,
    target_pre_gain: f32,
    early_reflections_gain: f32,
    late_gain: f32,
    dry: f32,
    echo_decay: f32,
}

impl Reverb3d {
    pub(super) fn new(p: Reverb3dParams) -> Reverb3d {
        let mut r = Reverb3d {
            params: p,
            rate: (p.sample_rate / 1000).clamp(1, 192),
            fdn: std::array::from_fn(|i| {
                DelayLine::new(ms_samples((p.sample_rate / 1000).max(1), FDN_MAX_MS[i]))
            }),
            decay1: std::array::from_fn(|i| {
                DelayLine::new(ms_samples((p.sample_rate / 1000).max(1), DECAY1_MAX_MS[i]))
            }),
            decay2: std::array::from_fn(|i| {
                DelayLine::new(ms_samples((p.sample_rate / 1000).max(1), DECAY2_MAX_MS[i]))
            }),
            pre: DelayLine::new(ms_samples((p.sample_rate / 1000).max(1), PRE_MAX_MS)),
            comb: DelayLine::new(ms_samples((p.sample_rate / 1000).max(1), COMB_MS)),
            fdn_delay: [1; 4],
            decay1_delay: [1; 4],
            decay2_delay: [1; 4],
            early_delay: [1; 20],
            early_gain: [0.0; 20],
            reflection_delay: 1,
            comb_delay: 1,
            decay_direct: [0.0; 4],
            decay_current_fdn: [0.0; 4],
            decay_current_out: [0.0; 4],
            previous_decayed: [0.0; 4],
            previous_pre: 0.0,
            previous_pre_gain: 0.0,
            target_pre_gain: 1.0,
            early_reflections_gain: 1.0,
            late_gain: 1.0,
            dry: p.dry_gain,
            echo_decay: 0.0,
        };
        r.retune();
        r
    }

    pub(super) fn configure(&mut self, p: Reverb3dParams) {
        let rate = (p.sample_rate / 1000).clamp(1, 192);
        let rebuild = rate != self.rate || p.route.channels != self.params.route.channels;
        self.params = p;
        self.rate = rate;
        if rebuild {
            self.fdn = std::array::from_fn(|i| DelayLine::new(ms_samples(rate, FDN_MAX_MS[i])));
            self.decay1 =
                std::array::from_fn(|i| DelayLine::new(ms_samples(rate, DECAY1_MAX_MS[i])));
            self.decay2 =
                std::array::from_fn(|i| DelayLine::new(ms_samples(rate, DECAY2_MAX_MS[i])));
            self.pre = DelayLine::new(ms_samples(rate, PRE_MAX_MS));
            self.comb = DelayLine::new(ms_samples(rate, COMB_MS));
            self.reset();
        }
        self.retune();
    }

    fn reset(&mut self) {
        for line in self
            .fdn
            .iter_mut()
            .chain(self.decay1.iter_mut())
            .chain(self.decay2.iter_mut())
        {
            line.reset();
        }
        self.pre.reset();
        self.comb.reset();
        self.previous_decayed = [0.0; 4];
        self.previous_pre = 0.0;
    }

    fn retune(&mut self) {
        let p = self.params;
        let rate = self.rate.max(1);
        self.dry = p.dry_gain;
        self.previous_decayed = [0.0; 4];
        self.previous_pre = 0.0;

        self.early_reflections_gain = db_to_linear(p.room_gain + p.reflections_gain).min(100.0);
        self.late_gain = db_to_linear(p.room_gain + p.reverb_gain).min(100.0);

        let high_frequency_room_gain = db_to_linear(p.room_hf);
        if high_frequency_room_gain < 1.0 && p.hf_reference > 0.0 {
            let a = 1.0 - high_frequency_room_gain;
            let b = 2.0
                - 2.0
                    * high_frequency_room_gain
                    * (256.0 * p.hf_reference / p.sample_rate as f32)
                        .to_radians()
                        .cos();
            let c = (b * b - 4.0 * a * a).max(0.0).sqrt();
            let previous = ((b - c) / (2.0 * a.max(1e-6))).clamp(-1.0, 1.0);
            self.previous_pre_gain = previous;
            self.target_pre_gain = 1.0 - previous;
        } else {
            self.previous_pre_gain = 0.0;
            self.target_pre_gain = 1.0;
        }

        let seconds = (p.reflection_delay + p.reverb_delay_time).clamp(0.0, 1.0);
        self.reflection_delay =
            ms_samples(rate, seconds * 1000.0).min(self.pre.len().saturating_sub(1));
        self.comb_delay = (self.comb.len() - 1).max(1);
        self.echo_decay = (0.6 * p.diffusion * 0.01).clamp(0.0, 0.6);

        let sample_rate = p.sample_rate as f32;
        let decay_time = p.decay_time.max(0.001);
        let ratio = p.hf_decay_ratio.clamp(0.01, 1.0);
        let angle = (64.0 * p.hf_reference / sample_rate).to_radians();
        let sin = angle.sin().abs().max(1e-4);
        let tangent = angle.cos() / sin;

        for i in 0..4 {
            let delay = FDN_MIN_MS[i] + p.density / 100.0 * (FDN_MAX_MS[i] - FDN_MIN_MS[i]);
            self.fdn_delay[i] = ms_samples(rate, delay).clamp(1, self.fdn[i].len() - 1);
            self.decay1_delay[i] =
                ms_samples(rate, DECAY1_MAX_MS[i]).clamp(1, self.decay1[i].len() - 1);
            self.decay2_delay[i] =
                ms_samples(rate, DECAY2_MAX_MS[i]).clamp(1, self.decay2[i].len() - 1);
            let loop_samples =
                (self.fdn_delay[i] + self.decay1_delay[i] + self.decay2_delay[i]) as f32;
            let a = -60.0 * loop_samples / (decay_time * sample_rate);
            let b = a / ratio;
            let d = pow10((b - a) / 40.0);
            let e = pow10((b + a) / 40.0) * std::f32::consts::FRAC_1_SQRT_2;
            let denominator = (tangent + d).abs().max(1e-6);
            self.decay_direct[i] = (e * ((d * tangent) + 1.0) / denominator).clamp(-0.999, 0.999);
            self.decay_current_fdn[i] =
                (e * (1.0 - (d * tangent)) / denominator).clamp(-0.999, 0.999);
            self.decay_current_out[i] = ((tangent - d) / denominator).clamp(-0.999, 0.999);
            if !self.decay_direct[i].is_finite()
                || !self.decay_current_fdn[i].is_finite()
                || !self.decay_current_out[i].is_finite()
            {
                self.decay_direct[i] = 0.0;
                self.decay_current_fdn[i] = 0.0;
                self.decay_current_out[i] = 0.0;
            }
        }

        for i in 0..20 {
            let ms = p.reflection_delay * 1000.0
                + EARLY_TIMES_S[i] * 1000.0 * ((p.reverb_delay_time * 0.9998) + 0.02);
            self.early_delay[i] = ms_samples(rate, ms).min(self.pre.len() - 1);
            self.early_gain[i] = EARLY_GAINS[i];
        }
    }

    pub(crate) fn process(&mut self, input: &[Vec<f32>], out: &mut [Vec<f32>], n: usize) {
        let ch = self.params.route.channels.min(input.len()).min(out.len());
        if ch == 0 || input[0].len() < n || out[0].len() < n {
            return;
        }
        let mapping = match ch {
            1 => &MAPPING_MONO,
            2 => &MAPPING_STEREO,
            4 => &MAPPING_QUAD,
            6 => &MAPPING_SURROUND,
            _ => return,
        };
        for s in 0..n {
            let mut early = [0.0f32; MAX_CHANNELS];
            let tap_out = self.pre.read(self.reflection_delay + 1);
            for (&slot, &index) in mapping.early.iter().zip(mapping.index) {
                let index = index as usize;
                early[slot as usize] +=
                    self.pre.read(self.early_delay[index] + 1) * self.early_gain[index];
            }

            let mut x = [0.0f32; MAX_CHANNELS];
            let mut sum = 0.0f32;
            for c in 0..ch {
                x[c] = input[c][s];
                sum += x[c];
            }
            for value in early.iter_mut().take(ch) {
                *value *= self.early_reflections_gain;
            }

            self.previous_pre =
                sum * self.target_pre_gain + self.previous_pre * self.previous_pre_gain;
            if !self.previous_pre.is_finite() {
                self.previous_pre = 0.0;
            }
            self.pre.write(self.previous_pre);

            let mut feedback = [0.0f32; 4];
            for (i, slot) in feedback.iter_mut().enumerate() {
                let value = self.fdn[i].read(self.fdn_delay[i]);
                let output = value * self.decay_direct[i] + self.previous_decayed[i];
                self.previous_decayed[i] =
                    value * self.decay_current_fdn[i] + output * self.decay_current_out[i];
                *slot = if output.is_finite() { output } else { 0.0 };
            }
            let mixed = [
                feedback[2] + feedback[1],
                -feedback[0] - feedback[3],
                feedback[0] - feedback[3],
                feedback[1] - feedback[2],
            ];
            let mut values = [0.0f32; 4];
            for i in 0..4 {
                let value = tap_out * self.late_gain + mixed[i];
                let decayed = self.decay1[i].read(self.decay1_delay[i]);
                let processed = (value - self.echo_decay * decayed).clamp(-1e6, 1e6);
                self.decay1[i].write(if processed.is_finite() {
                    processed
                } else {
                    0.0
                });
                let first = decayed + processed * self.echo_decay;

                let decayed = self.decay2[i].read(self.decay2_delay[i]);
                let processed = (first + self.echo_decay * 0.9 * decayed).clamp(-1e6, 1e6);
                self.decay2[i].write(if processed.is_finite() {
                    processed
                } else {
                    0.0
                });
                let second = decayed - processed * self.echo_decay * 0.9;

                self.fdn[i].write(if second.is_finite() { second } else { 0.0 });
                values[i] = if second.is_finite() { second } else { 0.0 };
            }

            let mut y = [0.0f32; MAX_CHANNELS];
            for c in 0..ch {
                let Some(&feedback_index) = mapping.feedback.get(c) else {
                    y[c] = 0.0;
                    continue;
                };
                if feedback_index < 0 {
                    y[c] = 0.0;
                    continue;
                }
                y[c] = early[c] + values[feedback_index as usize] + x[c] * self.dry;
            }
            if ch == 1 {
                y[0] += values[1];
            }
            if ch == 6 {
                let delayed = self.comb.read(self.comb_delay);
                let centre = (values[2] - values[3]) * 0.5;
                self.comb
                    .write(if centre.is_finite() { centre } else { 0.0 });
                y[4] = early[4] + values[3] + delayed + x[4] * self.dry;
            }

            for c in 0..ch {
                out[c][s] = if y[c].is_finite() {
                    y[c].clamp(-1e6, 1e6)
                } else {
                    0.0
                };
            }
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

    fn params(channels: usize) -> Reverb3dParams {
        Reverb3dParams {
            route: route(channels),
            sample_rate: 48_000,
            room_hf: 0.0,
            hf_reference: 5000.0,
            decay_time: 500.0,
            hf_decay_ratio: 0.5,
            room_gain: -1000.0,
            reflections_gain: -1000.0,
            reverb_gain: -1000.0,
            diffusion: 100.0,
            reflection_delay: 0.01,
            reverb_delay_time: 0.05,
            density: 100.0,
            dry_gain: 1.0,
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
    fn reverb3d_parameter_offsets_match_the_public_layout() {
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[6] = 1;
        d[0x0C..0x0E].copy_from_slice(&2u16.to_le_bytes());
        d[0x0E..0x10].copy_from_slice(&2u16.to_le_bytes());
        d[0x14..0x18].copy_from_slice(&48_000u32.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&(-500.0f32).to_le_bytes());
        d[0x1C..0x20].copy_from_slice(&5000.0f32.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&800.0f32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&0.7f32.to_le_bytes());
        d[0x40..0x44].copy_from_slice(&50.0f32.to_le_bytes());
        d[0x44..0x48].copy_from_slice(&1.0f32.to_le_bytes());
        let p = Reverb3dParams::parse(&d);
        assert_eq!(p.route.channels, 2);
        assert_eq!(p.sample_rate, 48_000);
        assert!((p.decay_time - 800.0).abs() < 1e-6);
        assert!((p.density - 50.0).abs() < 1e-6);
        assert!(p.routable());
    }

    #[test]
    fn reverb3d_impulse_produces_a_finite_decaying_tail() {
        let n = 240;
        let mut p = params(2);
        p.dry_gain = 0.0;
        let mut reverb = Reverb3d::new(p);
        let (mut input, mut out) = scratch(n);
        let mut energies = Vec::new();
        for block in 0..160 {
            for channel in input.iter_mut().take(2) {
                channel.fill(0.0);
            }
            if block == 0 {
                input[0][0] = 1000.0;
            }
            reverb.process(&input, &mut out, n);
            for channel in out.iter().take(2) {
                assert!(channel[..n].iter().all(|v| v.is_finite()));
            }
            if block >= 10 {
                let e: f64 = out
                    .iter()
                    .take(2)
                    .flat_map(|c| c[..n].iter())
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum();
                energies.push(e);
            }
        }
        assert!(energies[0] > 1e-3, "the tail must be audible: {energies:?}");
        let windows: Vec<f64> = energies.chunks(16).map(|w| w.iter().sum()).collect();
        let first = windows[0];
        assert!(
            windows.iter().all(|&w| w <= first * 8.0),
            "the network must not run away: {windows:?}"
        );
        assert!(
            windows.last().copied().unwrap_or(1.0) < first * 0.5,
            "the tail must lose energy: {windows:?}"
        );
    }

    #[test]
    fn reverb3d_without_a_valid_rate_is_not_routable() {
        let mut p = params(2);
        p.sample_rate = 0;
        assert!(!p.routable());
        p.sample_rate = 44_100;
        assert!(p.routable());
        p.decay_time = 0.0;
        assert!(!p.routable());
    }

    #[test]
    fn reverb3d_is_silent_for_silent_input() {
        let n = 240;
        let mut reverb = Reverb3d::new(params(6));
        let (input, mut out) = scratch(n);
        for _ in 0..10 {
            reverb.process(&input, &mut out, n);
            for channel in out.iter().take(6) {
                assert!(channel[..n].iter().all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn extreme_reverb3d_parameters_stay_finite() {
        let mut p = params(6);
        p.room_hf = -6000.0;
        p.hf_reference = 24_000.0;
        p.decay_time = 120_000.0;
        p.hf_decay_ratio = 0.01;
        p.room_gain = 5000.0;
        p.reflections_gain = 5000.0;
        p.reverb_gain = 5000.0;
        p.diffusion = 100.0;
        p.reflection_delay = 1.0;
        p.reverb_delay_time = 1.0;
        p.dry_gain = 100.0;
        let mut reverb = Reverb3d::new(p);
        let n = 240;
        let (mut input, mut out) = scratch(n);
        for block in 0..40 {
            for (c, channel) in input.iter_mut().take(6).enumerate() {
                for (s, v) in channel.iter_mut().enumerate() {
                    *v = if block == 0 && s == 0 {
                        (c as f32 + 1.0) * 100.0
                    } else {
                        0.0
                    };
                }
            }
            reverb.process(&input, &mut out, n);
            for channel in out.iter().take(6) {
                for &v in &channel[..n] {
                    assert!(v.is_finite(), "non-finite reverb3d output");
                    assert!(v.abs() < 1e9, "unbounded reverb3d output {v}");
                }
            }
        }
    }
}
