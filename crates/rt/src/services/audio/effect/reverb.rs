//! Four-FDN reverb with early reflections and damped decay delays.
//!
//! The tuning constants and the network topology follow Ryujinx's MIT-licensed
//! audren reverb (`ReverbState`); LASRecomp's MIT `runtime/services/audio.c`
//! was used as a cross-reference. The decay envelope is derived directly from
//! the requested decay time so every feedback path stays below unity.

use super::line::{DelayLine, ms_samples};
use super::{MAX_CHANNELS, Route, byte_at, gain, i32_at, q14, trace_once, u16_at};

const MAX_FDN_MS: [f32; 4] = [53.953_25, 79.192_57, 116.238_77, 170.615_3];
const MAX_DECAY_MS: [f32; 4] = [7.0, 9.0, 13.0, 17.0];
const MAX_PRE_MS: f32 = 500.0;

const FDN_TIMES: [f32; 20] = [
    53.953_25, 79.192_57, 116.238_77, 130.615_3, // late mode 0
    53.953_25, 79.192_57, 116.238_77, 170.615_3, // late mode 1
    5.0, 10.0, 5.0, 10.0, // late mode 2
    47.03, 71.0, 103.0, 170.0, // late mode 3
    53.953_25, 79.192_57, 116.238_77, 170.615_3, // maximum lengths
];

const DECAY_TIMES: [f32; 20] = [
    7.0, 9.0, 13.0, 17.0, // late mode 0
    7.0, 9.0, 13.0, 17.0, // late mode 1
    1.0, 1.0, 1.0, 1.0, // late mode 2
    7.0, 7.0, 13.0, 9.0, // late mode 3
    7.0, 9.0, 13.0, 17.0, // maximum lengths
];

const EARLY_TIMES: [f32; 50] = [
    0.0, 3.5, 2.8, 3.9, 2.7, 13.4, 7.9, 8.4, 9.9, 12.0, // early mode 0
    0.0, 11.8, 5.5, 11.2, 10.4, 38.1, 22.2, 29.6, 21.2, 24.8, // early mode 1
    0.0, 41.5, 20.5, 41.3, 0.0, 29.5, 33.8, 45.2, 46.8, 0.0, // early mode 2
    33.1, 43.3, 22.8, 37.9, 14.9, 35.3, 17.9, 34.2, 0.0, 43.3, // early mode 3
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // early mode 4
];

const EARLY_GAINS: [f32; 50] = [
    0.70, 0.68, 0.70, 0.68, 0.70, 0.68, 0.70, 0.68, 0.68, 0.68, // early mode 0
    0.70, 0.68, 0.70, 0.68, 0.70, 0.68, 0.68, 0.68, 0.68, 0.68, // early mode 1
    0.50, 0.70, 0.70, 0.68, 0.50, 0.68, 0.68, 0.70, 0.68, 0.00, // early mode 2
    0.93, 0.92, 0.87, 0.86, 0.94, 0.81, 0.80, 0.77, 0.76, 0.65, // early mode 3
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // early mode 4
];

const PRE_TIMES: [f32; 5] = [12.5, 40.0, 50.0, 50.0, 0.0];

const OE1: [u8; 10] = [0; 10];
const OE2: [u8; 10] = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1];
const OE4: [u8; 10] = [0, 0, 1, 1, 0, 1, 2, 2, 3, 3];
const OE6: [u8; 20] = [0, 5, 0, 5, 1, 5, 1, 5, 4, 5, 4, 5, 2, 5, 2, 5, 3, 5, 3, 5];
const TAPS: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const TAPS6: [u8; 20] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
const OUT1: [u8; 4] = [0; 4];
const OUT2: [u8; 4] = [0, 0, 1, 1];
const OUT4: [u8; 4] = [0, 1, 2, 3];
const OUT6: [u8; 6] = [0, 1, 2, 3, 4, 5];
const MAP1: [i8; 4] = [0, 1, 2, 3];
const MAP2: [i8; 4] = [2, 0, 3, 1];
const MAP4: [i8; 4] = [0, 1, 2, 3];
const MAP6: [i8; 6] = [0, 1, 2, 3, -1, 3];

struct Mapping {
    out: &'static [u8],
    taps: &'static [u8],
    map: &'static [u8],
    feedback: &'static [i8],
}

const MAPPING_1: Mapping = Mapping {
    out: &OE1,
    taps: &TAPS,
    map: &OUT1,
    feedback: &MAP1,
};
const MAPPING_2: Mapping = Mapping {
    out: &OE2,
    taps: &TAPS,
    map: &OUT2,
    feedback: &MAP2,
};
const MAPPING_4: Mapping = Mapping {
    out: &OE4,
    taps: &TAPS,
    map: &OUT4,
    feedback: &MAP4,
};
const MAPPING_6: Mapping = Mapping {
    out: &OE6,
    taps: &TAPS6,
    map: &OUT6,
    feedback: &MAP6,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReverbParams {
    pub input: [u8; MAX_CHANNELS],
    pub output: [u8; MAX_CHANNELS],
    pub channel_max: usize,
    pub channels: usize,
    pub sample_rate: u32,
    pub early_mode: i32,
    pub early_gain: f32,
    pub pre_delay: f32,
    pub late_mode: i32,
    pub late_gain: f32,
    pub decay_time: f32,
    pub hf_decay_ratio: f32,
    pub coloration: f32,
    pub reverb_gain: f32,
    pub out_gain: f32,
    pub dry_gain: f32,
}

impl Default for ReverbParams {
    fn default() -> ReverbParams {
        ReverbParams {
            input: [0; MAX_CHANNELS],
            output: [0; MAX_CHANNELS],
            channel_max: 2,
            channels: 0,
            sample_rate: 48,
            early_mode: 0,
            early_gain: 0.0,
            pre_delay: 0.0,
            late_mode: 0,
            late_gain: 0.0,
            decay_time: 0.0,
            hf_decay_ratio: 0.0,
            coloration: 0.0,
            reverb_gain: 0.0,
            out_gain: 0.0,
            dry_gain: 0.0,
        }
    }
}

impl ReverbParams {
    pub(crate) fn parse(d: &[u8]) -> ReverbParams {
        let mut input = [0u8; MAX_CHANNELS];
        let mut output = [0u8; MAX_CHANNELS];
        for (k, v) in input.iter_mut().enumerate() {
            *v = byte_at(d, k);
        }
        for (k, v) in output.iter_mut().enumerate() {
            *v = byte_at(d, 6 + k);
        }
        let channel_max = usize::from(u16_at(d, 12));
        let channels = usize::from(u16_at(d, 14));
        ReverbParams {
            input,
            output,
            channel_max,
            channels,
            // The block stores the rate as Q14 Hz (Ryujinx `ReverbState`);
            // the engine works in samples per millisecond.
            sample_rate: {
                let hz = q14(i32_at(d, 0x10));
                if hz.is_finite() && (8000.0..=192_000.0).contains(&hz) {
                    (hz / 1000.0).round().clamp(1.0, 192.0) as u32
                } else {
                    48
                }
            },
            early_mode: i32_at(d, 0x14),
            early_gain: gain(i32_at(d, 0x18)),
            pre_delay: q14(i32_at(d, 0x1C)).clamp(0.0, MAX_PRE_MS),
            late_mode: i32_at(d, 0x20),
            late_gain: gain(i32_at(d, 0x24)),
            decay_time: q14(i32_at(d, 0x28)).clamp(0.0, 120_000.0),
            hf_decay_ratio: q14(i32_at(d, 0x2C)).clamp(0.0, 1.0),
            coloration: q14(i32_at(d, 0x30)).clamp(0.0, 1.0),
            reverb_gain: gain(i32_at(d, 0x34)),
            out_gain: gain(i32_at(d, 0x38)),
            dry_gain: gain(i32_at(d, 0x3C)),
        }
    }

    pub(super) fn routable(&self) -> bool {
        matches!(self.channels, 1 | 2 | 4 | 6)
            && matches!(self.channel_max, 1 | 2 | 4 | 6)
            && self.channels <= self.channel_max
    }

    pub(super) fn route(&self) -> Route {
        let mut route = Route {
            input: self.input,
            output: self.output,
            channels: self.channels,
        };
        for (k, slot) in route.input.iter_mut().enumerate() {
            if k >= self.channels {
                *slot = 0;
            }
        }
        for (k, slot) in route.output.iter_mut().enumerate() {
            if k >= self.channels {
                *slot = 0;
            }
        }
        route
    }
}

pub(crate) struct Reverb {
    params: ReverbParams,
    rate: u32,
    fdn: [DelayLine; 4],
    decay: [DelayLine; 4],
    pre: DelayLine,
    comb: DelayLine,
    fdn_delay: [usize; 4],
    decay_delay: [usize; 4],
    early_delay: [usize; 10],
    early_gain: [f32; 10],
    pre_delay: usize,
    comb_delay: usize,
    hf_direct: [f32; 4],
    hf_prev: [f32; 4],
    prev_fb: [f32; 4],
    decay_rate: f32,
    dry: f32,
    wet: f32,
    target_dry: f32,
    target_wet: f32,
}

impl Reverb {
    pub(super) fn new(p: ReverbParams) -> Reverb {
        let rate = p.sample_rate;
        let mut r = Reverb {
            params: p,
            rate,
            fdn: std::array::from_fn(|i| DelayLine::new(ms_samples(rate, MAX_FDN_MS[i]))),
            decay: std::array::from_fn(|i| DelayLine::new(ms_samples(rate, MAX_DECAY_MS[i]))),
            pre: DelayLine::new(ms_samples(rate, MAX_PRE_MS)),
            comb: DelayLine::new(ms_samples(rate, 5.0)),
            fdn_delay: [1; 4],
            decay_delay: [1; 4],
            early_delay: [1; 10],
            early_gain: [0.0; 10],
            pre_delay: 1,
            comb_delay: 1,
            hf_direct: [0.0; 4],
            hf_prev: [0.0; 4],
            prev_fb: [0.0; 4],
            decay_rate: 0.6,
            dry: p.dry_gain,
            wet: p.out_gain,
            target_dry: p.dry_gain,
            target_wet: p.out_gain,
        };
        r.retune();
        r
    }

    pub(super) fn configure(&mut self, p: ReverbParams, is_new: bool) {
        let changed = p != self.params;
        let rebuild = is_new || p.sample_rate != self.rate || p.channels != self.params.channels;
        self.params = p;
        if rebuild {
            if !is_new {
                trace_once(format!(
                    "audio: reverb rebuilt at {} samples/ms, {} channels",
                    p.sample_rate, p.channels
                ));
            }
            self.rate = p.sample_rate;
            self.fdn =
                std::array::from_fn(|i| DelayLine::new(ms_samples(p.sample_rate, MAX_FDN_MS[i])));
            self.decay =
                std::array::from_fn(|i| DelayLine::new(ms_samples(p.sample_rate, MAX_DECAY_MS[i])));
            self.pre = DelayLine::new(ms_samples(p.sample_rate, MAX_PRE_MS));
            self.comb = DelayLine::new(ms_samples(p.sample_rate, 5.0));
            self.reset();
            self.retune();
            self.dry = self.target_dry;
            self.wet = self.target_wet;
        } else if changed {
            self.retune();
        }
    }

    fn reset(&mut self) {
        for line in self.fdn.iter_mut().chain(self.decay.iter_mut()) {
            line.reset();
        }
        self.pre.reset();
        self.comb.reset();
        self.prev_fb = [0.0; 4];
    }

    fn retune(&mut self) {
        let p = &self.params;
        let rate = self.rate.max(1);
        let em = usize::try_from(p.early_mode)
            .ok()
            .filter(|&m| m < 5)
            .unwrap_or(4);
        let lm = usize::try_from(p.late_mode)
            .ok()
            .filter(|&m| m < 4)
            .unwrap_or(0);
        for i in 0..10 {
            let ms = EARLY_TIMES[em * 10 + i] + p.pre_delay;
            self.early_delay[i] = ms_samples(rate, ms).min(self.pre.len() - 1);
            self.early_gain[i] = EARLY_GAINS[em * 10 + i] * p.early_gain;
        }
        if p.channels == 2 {
            self.early_gain[4] *= 0.5;
            self.early_gain[5] *= 0.5;
        }
        self.pre_delay = ms_samples(rate, PRE_TIMES[em] + p.pre_delay).min(self.pre.len() - 1);
        self.comb_delay = (self.comb.len() - 1).max(1);

        for i in 0..4 {
            self.fdn_delay[i] = ms_samples(rate, FDN_TIMES[lm * 4 + i]).min(self.fdn[i].len() - 1);
            self.decay_delay[i] =
                ms_samples(rate, DECAY_TIMES[lm * 4 + i]).min(self.decay[i].len() - 1);
            // Per-loop gain for a -60 dB decay after `decay_time` milliseconds.
            let period = (self.decay_delay[i] + self.fdn_delay[i]) as f32 / rate as f32;
            let b = if p.decay_time > 0.0 {
                -3.0 * period / p.decay_time
            } else {
                -3.0
            };
            let g = 10f32.powf(b).clamp(0.0, 0.999);
            // One-pole shelf: DC gain `g`, high-frequency gain `g * ratio`, where
            // `ratio` shortens the tail above 1280/rate rad like `hf_decay_ratio`.
            let hf = p.hf_decay_ratio;
            let ratio = if hf < 0.995 && hf > 0.0 {
                10f32.powf(b * (1.0 / hf - 1.0)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let c = (1.0 - ratio) / (1.0 + ratio);
            let direct = g * (1.0 - c) * std::f32::consts::FRAC_1_SQRT_2;
            self.hf_direct[i] = if direct.is_finite() {
                direct.clamp(0.0, 0.999)
            } else {
                0.0
            };
            self.hf_prev[i] = if c.is_finite() {
                c.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        self.decay_rate = (0.6 * (1.0 - p.coloration)).clamp(0.0, 1.0);
        self.target_dry = p.dry_gain;
        self.target_wet = p.out_gain;
    }

    /// Process `n` samples of each input channel into the matching output
    /// channel. Input and output may alias the same mix buffer.
    pub(crate) fn process(&mut self, input: &[Vec<f32>], out: &mut [Vec<f32>], n: usize) {
        let ch = self.params.channels.min(input.len()).min(out.len());
        if ch == 0 || input[0].len() < n || out[0].len() < n {
            return;
        }
        let mapping = match ch {
            1 => &MAPPING_1,
            2 => &MAPPING_2,
            4 => &MAPPING_4,
            6 => &MAPPING_6,
            _ => return,
        };
        let reverb_gain = self.params.reverb_gain;
        let late_gain = self.params.late_gain;
        let (dry_from, dry_to) = (self.dry, self.target_dry);
        let (wet_from, wet_to) = (self.wet, self.target_wet);
        let dry_step = (dry_to - dry_from) / n.max(1) as f32;
        let wet_step = (wet_to - wet_from) / n.max(1) as f32;
        let mut dry = dry_from;
        let mut wet = wet_from;
        for s in 0..n {
            let mut ov = [0.0f32; MAX_CHANNELS];
            for (&slot, &tap) in mapping.out.iter().zip(mapping.taps) {
                let tap = tap as usize;
                ov[slot as usize] += self.pre.read(self.early_delay[tap]) * self.early_gain[tap];
            }
            if ch == 6 {
                ov[5] *= 0.2;
            }
            let mut cin = [0.0f32; MAX_CHANNELS];
            let mut pre = 0.0;
            for c in 0..ch {
                cin[c] = input[c][s] * 64.0;
                pre += cin[c] * reverb_gain;
            }
            self.pre.write(pre);
            let late = self.pre.read(self.pre_delay) * late_gain;

            let mut fo = [0.0f32; 4];
            // The feedback network index touches several per-line arrays plus the
            // output scratch, so an iterator over one of them is not clearer.
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                let delayed = self.fdn[i].read(self.fdn_delay[i]);
                fo[i] = delayed * self.hf_direct[i] + self.prev_fb[i] * self.hf_prev[i];
                self.prev_fb[i] = fo[i];
            }
            let fb = [fo[2] + fo[1], -fo[0] - fo[3], fo[0] - fo[3], fo[1] - fo[2]];
            for i in 0..4 {
                let old = self.decay[i].read(self.decay_delay[i]);
                let p = fb[i] + late - self.decay_rate * old;
                self.decay[i].write(if p.is_finite() { p } else { 0.0 });
                let v = old + p * self.decay_rate;
                self.fdn[i].write(if v.is_finite() { v } else { 0.0 });
                fo[i] = v;
            }
            for (&slot, &tap) in mapping.map.iter().zip(mapping.feedback) {
                if tap >= 0 {
                    ov[slot as usize] += fo[tap as usize];
                }
            }
            if ch == 6 {
                let v = (fo[2] - fo[3]) * 0.5;
                let old = self.comb.read(self.comb_delay);
                self.comb.write(if v.is_finite() { v } else { 0.0 });
                ov[4] += old;
            }

            let mut bad = false;
            for c in 0..ch {
                let y = (ov[c] * wet + cin[c] * dry) / 64.0;
                if y.is_finite() {
                    out[c][s] = y;
                } else {
                    out[c][s] = 0.0;
                    bad = true;
                }
            }
            if bad {
                self.reset();
            }
            dry += dry_step;
            wet += wet_step;
        }
        self.dry = dry_to;
        self.wet = wet_to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(channels: usize) -> ReverbParams {
        let mut p = ReverbParams {
            channels,
            channel_max: channels,
            sample_rate: 48,
            early_mode: 0,
            early_gain: 0.7,
            pre_delay: 0.0,
            late_mode: 0,
            late_gain: 1.0,
            decay_time: 500.0,
            hf_decay_ratio: 0.5,
            coloration: 0.1,
            reverb_gain: 1.0,
            out_gain: 1.0,
            dry_gain: 1.0,
            ..ReverbParams::default()
        };
        for c in 0..channels {
            p.input[c] = c as u8;
            p.output[c] = c as u8;
        }
        p
    }

    fn scratch(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (
            vec![vec![0.0; n]; MAX_CHANNELS],
            vec![vec![0.0; n]; MAX_CHANNELS],
        )
    }

    fn block_energy(s: &[f32]) -> f64 {
        s.iter().map(|&x| f64::from(x) * f64::from(x)).sum()
    }

    #[test]
    fn reverb_block_stores_the_rate_as_q14_hz() {
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[1] = 1;
        d[6] = 0;
        d[7] = 1;
        d[12..14].copy_from_slice(&2u16.to_le_bytes());
        d[14..16].copy_from_slice(&2u16.to_le_bytes());
        d[0x10..0x14].copy_from_slice(&((48_000i32 << 14) as u32).to_le_bytes());
        d[0x14..0x18].copy_from_slice(&0i32.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&((0.7f32 * 16384.0) as i32).to_le_bytes());
        d[0x1C..0x20].copy_from_slice(&0i32.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&0i32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&16384i32.to_le_bytes());
        d[0x28..0x2C].copy_from_slice(&((500.0f32 * 16384.0) as i32).to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&((0.5f32 * 16384.0) as i32).to_le_bytes());
        d[0x30..0x34].copy_from_slice(&((0.1f32 * 16384.0) as i32).to_le_bytes());
        for k in 0..3 {
            d[0x34 + k * 4..0x38 + k * 4].copy_from_slice(&16384i32.to_le_bytes());
        }
        let p = ReverbParams::parse(&d);
        assert_eq!(p.sample_rate, 48);
        assert_eq!(p.channels, 2);
        assert!(p.routable());
        assert_eq!(p.route().input[0..2], [0, 1]);
    }

    #[test]
    fn reverb_tail_energy_decreases_after_the_impulse() {
        let n = 240;
        let mut reverb = Reverb::new(params(2));
        let (mut input, mut out) = scratch(n);
        let mut energies = Vec::new();
        for block in 0..80 {
            for channel in input.iter_mut().take(2) {
                channel.fill(0.0);
            }
            if block == 0 {
                input[0][0] = 0.5;
            }
            reverb.process(&input, &mut out, n);
            for channel in out.iter().take(2) {
                assert!(channel[..n].iter().all(|v| v.is_finite()));
            }
            if block >= 12 {
                energies.push(block_energy(&out[0][..n]) + block_energy(&out[1][..n]));
            }
        }
        assert!(
            energies[0] > 1e-6,
            "the tail must still be audible: {energies:?}"
        );
        // The tail is a sum of sparse echoes, so single-block energy is not
        // monotonic; 40 ms windows are, because the envelope decays.
        let windows: Vec<f64> = energies.chunks(8).map(|w| w.iter().sum()).collect();
        for pair in windows.windows(2) {
            assert!(pair[1] <= pair[0] * 1.02, "tail must decay: {windows:?}");
        }
        assert!(
            windows.last().copied().unwrap_or(0.0) < windows[0] * 0.5,
            "tail must lose energy: {windows:?}"
        );
    }

    #[test]
    fn reverb_is_silent_for_silent_input() {
        let n = 240;
        let mut reverb = Reverb::new(params(2));
        let (input, mut out) = scratch(n);
        for _ in 0..10 {
            reverb.process(&input, &mut out, n);
            for channel in out.iter().take(2) {
                assert!(channel[..n].iter().all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn extreme_parameters_stay_finite_and_stable() {
        let n = 240;
        let mut p = params(6);
        p.decay_time = 120_000.0;
        p.hf_decay_ratio = 0.0;
        p.coloration = 0.0;
        p.early_gain = 4.0;
        p.pre_delay = 500.0;
        p.reverb_gain = 4.0;
        p.out_gain = 4.0;
        p.dry_gain = 4.0;
        let mut reverb = Reverb::new(p);
        let (mut input, mut out) = scratch(n);
        for block in 0..40 {
            for channel in input.iter_mut().take(6) {
                for (s, v) in channel.iter_mut().enumerate() {
                    *v = if block == 0 && s == 0 { 0.5 } else { 0.0 };
                }
            }
            reverb.process(&input, &mut out, n);
            for channel in out.iter().take(6) {
                for &v in &channel[..n] {
                    assert!(v.is_finite(), "non-finite reverb output");
                    assert!(v.abs() < 1e6, "unbounded reverb output {v}");
                }
            }
        }
    }
}
