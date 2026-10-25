//! Dynamic-range compressor used by the `audren` compressor effect (type 9).
//!
//! Layout (`CompressorParameter` in the Ryujinx audio renderer):
//!
//! ```text
//! +0x00 u8[6]  input channel indices
//! +0x06 u8[6]  output channel indices
//! +0x0C u16    maximum channel count (unused)
//! +0x0E u16    channel count
//! +0x10 i32    sample rate (Hz)
//! +0x14 f32    threshold (dB)
//! +0x18 f32    ratio (input:output above the threshold)
//! +0x1C i32    attack time (us)
//! +0x20 i32    release time (us)
//! +0x24 f32    input gain (linear)
//! +0x28 f32    attack coefficient (0 = derive from the time)
//! +0x2C f32    release coefficient (0 = derive from the time)
//! +0x30 f32    output gain (dB)
//! +0x34 u32    client usage status (not used by the DSP)
//! +0x38 bool   makeup gain enabled
//! +0x39 u8[2]  reserved
//! ```
//!
//! The block has no `EffectResultState` payload of its own: REV9 only defines
//! `LimiterStatistics`, so the compressor reports its state in
//! `EffectOutStatusVersion2` and leaves the specific-data area zero.
//!
//! The DSP is a per-channel feed-forward compressor. Each channel keeps its
//! own envelope detector and gain, so a loud channel does not duck the others:
//! the detector follows the absolute input with the attack/release
//! coefficients, the static curve maps the detector level in dB to
//! `(1/ratio - 1) * (level - threshold)` dB above the threshold, and the gain
//! is smoothed toward that target with the attack/release coefficients. The
//! SDK's precomputed coefficients are used when they are in `(0, 1]`; otherwise
//! they are derived from the attack/release times and the block's sample rate
//! (or the sink rate when the block does not carry one). Non-finite or
//! out-of-range parameters are clamped, and a non-finite sample resets that
//! channel instead of poisoning the rest of the stream.

use super::{MAX_CHANNELS, Route, byte_at, f32_at, finite, i32_at, u16_at};

/// Time constant convention: the coefficient reaches 90 % after the given
/// time (2.2 one-pole time constants).
const TIME_CONSTANTS: f32 = 2.2;
const MIN_LEVEL: f32 = 1.0e-10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompressorParams {
    pub route: Route,
    pub sample_rate: u32,
    pub threshold: f32,
    pub ratio: f32,
    pub attack_us: i32,
    pub release_us: i32,
    pub input_gain: f32,
    pub attack_coefficient: f32,
    pub release_coefficient: f32,
    pub output_gain: f32,
    pub makeup: bool,
}

impl Default for CompressorParams {
    fn default() -> CompressorParams {
        CompressorParams {
            route: Route::default(),
            sample_rate: 48_000,
            threshold: 0.0,
            ratio: 1.0,
            attack_us: 0,
            release_us: 0,
            input_gain: 1.0,
            attack_coefficient: 0.0,
            release_coefficient: 0.0,
            output_gain: 0.0,
            makeup: false,
        }
    }
}

impl CompressorParams {
    pub(crate) fn parse(d: &[u8]) -> CompressorParams {
        let channels = usize::from(u16_at(d, 0x0E));
        let rate = i32_at(d, 0x10);
        CompressorParams {
            route: Route::parse(d).with_channels(channels),
            sample_rate: if (8000..=192_000).contains(&rate) {
                rate as u32
            } else {
                0
            },
            // A ratio below 1 would expand instead of compress; keep a sane
            // ceiling so the static curve cannot produce an extreme slope.
            threshold: finite(f32_at(d, 0x14), -200.0, 24.0),
            ratio: finite(f32_at(d, 0x18), 1.0, 100.0),
            attack_us: i32_at(d, 0x1C).clamp(0, 1_000_000),
            release_us: i32_at(d, 0x20).clamp(0, 1_000_000),
            input_gain: finite(f32_at(d, 0x24), 0.0, 8.0),
            attack_coefficient: finite(f32_at(d, 0x28), 0.0, 1.0),
            release_coefficient: finite(f32_at(d, 0x2C), 0.0, 1.0),
            output_gain: finite(f32_at(d, 0x30), -60.0, 24.0),
            makeup: byte_at(d, 0x38) != 0,
        }
    }

    pub(crate) fn routable(&self) -> bool {
        self.route.routable()
            && self.ratio.is_finite()
            && self.ratio >= 1.0
            && self.threshold.is_finite()
            && self.input_gain.is_finite()
            && self.output_gain.is_finite()
    }

    /// Makeup gain in dB, as the reference renderer derives it: the curve is
    /// shifted so that a signal at half the threshold is restored, minus the
    /// fixed 3 dB offset the hardware applies.
    fn makeup_db(&self) -> f32 {
        if !self.makeup {
            return 0.0;
        }
        let db = self.threshold * 0.5 * (1.0 / self.ratio - 1.0) - 3.0;
        if db.is_finite() { db } else { 0.0 }
    }
}

fn time_coefficient(us: i32, rate: u32) -> f32 {
    if us <= 0 {
        return 1.0;
    }
    let seconds = us as f32 * 1.0e-6;
    let rate = rate.max(8000) as f32;
    (1.0 - (-TIME_CONSTANTS / (seconds * rate)).exp()).clamp(0.0, 1.0)
}

fn envelope_coefficient(precomputed: f32, us: i32, rate: u32) -> f32 {
    if precomputed.is_finite() && precomputed > 0.0 && precomputed <= 1.0 {
        precomputed
    } else {
        time_coefficient(us, rate)
    }
}

pub(crate) struct Compressor {
    params: CompressorParams,
    channels: usize,
    attack: f32,
    release: f32,
    output_linear: f32,
    detector: [f32; MAX_CHANNELS],
    gain: [f32; MAX_CHANNELS],
}

impl Compressor {
    pub(super) fn new(p: CompressorParams, sample_rate: u32) -> Compressor {
        let mut compressor = Compressor {
            params: p,
            channels: 0,
            attack: 1.0,
            release: 1.0,
            output_linear: 1.0,
            detector: [0.0; MAX_CHANNELS],
            gain: [1.0; MAX_CHANNELS],
        };
        compressor.allocate(sample_rate);
        compressor
    }

    fn allocate(&mut self, sample_rate: u32) {
        self.update_derived(sample_rate);
        self.detector = [0.0; MAX_CHANNELS];
        self.gain = [1.0; MAX_CHANNELS];
    }

    fn update_derived(&mut self, sample_rate: u32) {
        let rate = if self.params.sample_rate >= 8000 {
            self.params.sample_rate
        } else {
            sample_rate.max(8000)
        };
        self.channels = self.params.route.channels.min(MAX_CHANNELS);
        self.attack =
            envelope_coefficient(self.params.attack_coefficient, self.params.attack_us, rate);
        self.release = envelope_coefficient(
            self.params.release_coefficient,
            self.params.release_us,
            rate,
        );
        self.output_linear = self.output_linear();
    }

    fn output_linear(&self) -> f32 {
        let db = (self.params.output_gain + self.params.makeup_db()).clamp(-60.0, 24.0);
        10.0f32.powf(db / 20.0).clamp(0.0, 16.0)
    }

    pub(super) fn configure(&mut self, p: CompressorParams, sample_rate: u32) {
        let reset = p.route.channels != self.params.route.channels;
        self.params = p;
        self.update_derived(sample_rate);
        if reset {
            self.detector = [0.0; MAX_CHANNELS];
            self.gain = [1.0; MAX_CHANNELS];
        }
    }

    pub(crate) fn process(&mut self, input: &[Vec<f32>], out: &mut [Vec<f32>], n: usize) {
        let ch = self.channels.min(input.len()).min(out.len());
        if ch == 0 || (0..ch).any(|c| input[c].len() < n || out[c].len() < n) {
            return;
        }
        let threshold = self.params.threshold;
        let reduction = 1.0 / self.params.ratio - 1.0;
        let input_gain = self.params.input_gain;
        let output_linear = self.output_linear;
        let (attack, release) = (self.attack, self.release);
        for c in 0..ch {
            let mut detector = if self.detector[c].is_finite() {
                self.detector[c].max(0.0)
            } else {
                0.0
            };
            let mut gain = if self.gain[c].is_finite() {
                self.gain[c].clamp(0.0, 1.0)
            } else {
                1.0
            };
            for s in 0..n {
                let raw = input[c][s];
                let x = raw / 32767.0 * input_gain;
                let level = if x.is_finite() { x.abs().min(8.0) } else { 0.0 };
                let alpha = if level > detector { attack } else { release };
                detector += alpha * (level - detector);
                let db = 20.0 * detector.max(MIN_LEVEL).log10();
                let target_db = if db > threshold {
                    reduction * (db - threshold)
                } else {
                    0.0
                };
                let target = 10.0f32.powf(target_db / 20.0).clamp(0.0, 1.0);
                let alpha = if target < gain { attack } else { release };
                gain += alpha * (target - gain);
                let y = x * gain * output_linear * 32767.0;
                out[c][s] = if y.is_finite() {
                    y.clamp(-32768.0, 32768.0)
                } else {
                    gain = 1.0;
                    detector = 0.0;
                    0.0
                };
            }
            self.detector[c] = detector;
            self.gain[c] = gain;
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

    fn params() -> CompressorParams {
        CompressorParams {
            route: route(1),
            sample_rate: 48_000,
            threshold: -20.0,
            ratio: 4.0,
            attack_us: 0,
            release_us: 0,
            input_gain: 1.0,
            attack_coefficient: 1.0,
            release_coefficient: 1.0,
            output_gain: 0.0,
            makeup: false,
        }
    }

    fn block(level: f32, n: usize) -> Vec<Vec<f32>> {
        vec![vec![level * 32767.0; n]; MAX_CHANNELS]
    }

    #[test]
    fn compressor_parameter_offsets_match_the_public_layout() {
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[6] = 1;
        d[0x0C..0x0E].copy_from_slice(&2u16.to_le_bytes());
        d[0x0E..0x10].copy_from_slice(&2u16.to_le_bytes());
        d[0x10..0x14].copy_from_slice(&48_000i32.to_le_bytes());
        d[0x14..0x18].copy_from_slice(&(-18.0f32).to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&3.5f32.to_le_bytes());
        d[0x1C..0x20].copy_from_slice(&800i32.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&12_000i32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1.25f32.to_le_bytes());
        d[0x28..0x2C].copy_from_slice(&0.4f32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0.05f32.to_le_bytes());
        d[0x30..0x34].copy_from_slice(&(-2.0f32).to_le_bytes());
        d[0x38] = 1;
        let p = CompressorParams::parse(&d);
        assert_eq!(p.route.channels, 2);
        assert_eq!((p.route.input[0], p.route.output[0]), (0, 1));
        assert_eq!(p.sample_rate, 48_000);
        assert!((p.threshold + 18.0).abs() < 1e-6);
        assert!((p.ratio - 3.5).abs() < 1e-6);
        assert_eq!((p.attack_us, p.release_us), (800, 12_000));
        assert!((p.input_gain - 1.25).abs() < 1e-6);
        assert!((p.output_gain + 2.0).abs() < 1e-6);
        assert!(p.makeup);
        assert!(p.routable());
    }

    #[test]
    fn below_the_threshold_the_signal_passes_unity() {
        let mut compressor = Compressor::new(params(), 48_000);
        let n = 240;
        let input = block(0.05, n);
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        compressor.process(&input, &mut out, n);
        for (want, got) in input[0].iter().zip(&out[0]) {
            assert!((got - want).abs() < 1.0, "unity expected: {got} vs {want}");
        }
    }

    #[test]
    fn above_the_threshold_the_ratio_sets_the_gain() {
        let mut compressor = Compressor::new(params(), 48_000);
        let n = 512;
        let input = block(0.5, n);
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        compressor.process(&input, &mut out, n);
        // -6.02 dBFS against a -20 dB threshold at 4:1 is
        // (1/4 - 1) * 13.98 = -10.48 dB, i.e. 0.2992 linear.
        let expected = 0.5 * 0.2992 * 32767.0;
        let got = out[0][n - 1];
        assert!(
            (got - expected).abs() < 150.0,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn output_gain_is_in_decibels() {
        let mut compressor = Compressor::new(
            CompressorParams {
                output_gain: 6.0,
                ..params()
            },
            48_000,
        );
        let n = 240;
        let input = block(0.05, n);
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        compressor.process(&input, &mut out, n);
        let ratio = out[0][n - 1] / input[0][n - 1];
        let want = 10.0f32.powf(6.0 / 20.0);
        assert!((ratio - want).abs() < 0.01, "output gain {ratio} vs {want}");
    }

    #[test]
    fn makeup_gain_raises_the_compressed_output() {
        let n = 512;
        let input = block(0.5, n);
        let mut plain = Compressor::new(params(), 48_000);
        let mut made = Compressor::new(
            CompressorParams {
                makeup: true,
                ..params()
            },
            48_000,
        );
        let mut a = vec![vec![0.0f32; n]; MAX_CHANNELS];
        let mut b = vec![vec![0.0f32; n]; MAX_CHANNELS];
        plain.process(&input, &mut a, n);
        made.process(&input, &mut b, n);
        assert!(
            b[0][n - 1] > a[0][n - 1] * 1.2,
            "makeup gain must lift the output"
        );
        assert!(b[0][n - 1].is_finite());
    }

    #[test]
    fn channels_are_compressed_independently() {
        let mut compressor = Compressor::new(
            CompressorParams {
                route: route(2),
                ..params()
            },
            48_000,
        );
        let n = 512;
        let mut input = vec![vec![0.0f32; n]; MAX_CHANNELS];
        input[0].fill(0.5 * 32767.0);
        input[1].fill(0.05 * 32767.0);
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        compressor.process(&input, &mut out, n);
        assert!(
            out[0][n - 1] < input[0][n - 1] * 0.5,
            "hot channel compressed"
        );
        assert!(
            (out[1][n - 1] - input[1][n - 1]).abs() < 1.0,
            "quiet channel untouched"
        );
    }

    #[test]
    fn zero_times_fall_back_to_full_bandwidth_coefficients() {
        let p = CompressorParams {
            attack_coefficient: 0.0,
            release_coefficient: 0.0,
            attack_us: 0,
            release_us: 0,
            ..params()
        };
        let compressor = Compressor::new(p, 48_000);
        assert_eq!((compressor.attack, compressor.release), (1.0, 1.0));

        let p = CompressorParams {
            attack_coefficient: 0.0,
            release_coefficient: 0.0,
            attack_us: 10_000,
            release_us: 100_000,
            ..params()
        };
        let compressor = Compressor::new(p, 48_000);
        let want = 1.0 - (-2.2f32 / (0.01 * 48_000.0)).exp();
        assert!((compressor.attack - want).abs() < 1e-6);
        assert!(compressor.release < compressor.attack);
    }

    #[test]
    fn reconfiguring_the_curve_keeps_the_envelope() {
        let mut compressor = Compressor::new(params(), 48_000);
        let n = 240;
        let input = block(0.5, n);
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        compressor.process(&input, &mut out, n);
        let before = out[0][n - 1];
        compressor.configure(
            CompressorParams {
                threshold: -21.0,
                ..params()
            },
            48_000,
        );
        compressor.process(&input, &mut out, n);
        let after = out[0][0];
        assert!(
            (after - before).abs() < before * 0.12,
            "a curve update must not reset the gain: {before} -> {after}"
        );
    }

    #[test]
    fn extreme_compressor_parameters_stay_finite() {
        let p = CompressorParams {
            route: route(6),
            threshold: -100.0,
            ratio: 100.0,
            input_gain: 8.0,
            attack_coefficient: 1.0,
            release_coefficient: 1.0,
            output_gain: 24.0,
            makeup: true,
            ..params()
        };
        let mut compressor = Compressor::new(p, 48_000);
        let n = 240;
        let mut out = vec![vec![0.0f32; n]; MAX_CHANNELS];
        for block in 0..16 {
            let mut input = vec![vec![0.0f32; n]; MAX_CHANNELS];
            for channel in input.iter_mut().take(6) {
                channel.fill(if block % 2 == 0 { 32767.0 } else { -32767.0 });
            }
            compressor.process(&input, &mut out, n);
            for channel in out.iter().take(6) {
                assert!(channel[..n].iter().all(|v| v.is_finite()));
                assert!(channel[..n].iter().all(|v| v.abs() <= 32768.0));
            }
        }
        let silence = vec![vec![0.0f32; n]; MAX_CHANNELS];
        compressor.process(&silence, &mut out, n);
        assert!(out[0][..n].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_block_without_channels_is_not_routable() {
        let p = CompressorParams {
            route: route(0),
            ..params()
        };
        assert!(!p.routable());
        let p = CompressorParams {
            ratio: 0.0,
            ..params()
        };
        assert!(!p.routable());
    }
}
