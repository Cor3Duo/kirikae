//! Q14 biquad filter used by the `audren` biquad insert effect.
//!
//! Layout (`BiquadFilterEffectParameter` in the Ryujinx audio renderer):
//!
//! ```text
//! +0x00 u8[6]  input channel indices
//! +0x06 u8[6]  output channel indices
//! +0x0C i16[3] numerator   (b0, b1, b2, Q14)
//! +0x12 i16[2] denominator (a1, a2, Q14; a0 = 1)
//! +0x16 u8     channel count
//! +0x17 u8     client usage status
//! ```
//!
//! The DSP evaluates a transposed direct form II with `a1`/`a2` added (not
//! subtracted), matching `BiquadFilterHelper.ProcessBiquadFilter`.

use super::{Route, byte_at, i16_at, q14, trace_once};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BiquadParams {
    pub route: Route,
    pub b: [f32; 3],
    pub a: [f32; 2],
    /// False when the block was rejected as unstable; the effect is then
    /// bypassed instead of filtering with zeroed coefficients.
    pub valid: bool,
}

impl BiquadParams {
    pub(crate) fn parse(d: &[u8]) -> BiquadParams {
        let c = [
            i16_at(d, 0x0C),
            i16_at(d, 0x0E),
            i16_at(d, 0x10),
            i16_at(d, 0x12),
            i16_at(d, 0x14),
        ];
        let channels = usize::from(byte_at(d, 0x16));
        BiquadParams::from_q14(Route::parse(d).with_channels(channels), c)
    }

    fn from_q14(route: Route, c: [i16; 5]) -> BiquadParams {
        let p = BiquadParams {
            route,
            b: [q14(c[0].into()), q14(c[1].into()), q14(c[2].into())],
            a: [q14(c[3].into()), q14(c[4].into())],
            valid: true,
        };
        if route.routable() && !p.is_stable() {
            trace_once(format!(
                "audio: biquad rejected unstable coefficients {:?}",
                (p.b, p.a)
            ));
            return BiquadParams {
                route,
                valid: false,
                ..BiquadParams::default()
            };
        }
        p
    }

    pub(crate) fn routable(&self) -> bool {
        self.valid && self.route.routable()
    }

    /// Denominator is `1 - a1 z^-1 - a2 z^-2`; keep both poles inside the unit
    /// circle so a malformed descriptor cannot blow the mix up.
    fn is_stable(&self) -> bool {
        self.b.iter().chain(self.a.iter()).all(|v| v.is_finite())
            && self.a[1].abs() < 1.0
            && 1.0 - self.a[0] - self.a[1] > 0.0
            && 1.0 + self.a[0] - self.a[1] > 0.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
}

impl Biquad {
    pub(crate) fn set(&mut self, p: &BiquadParams) {
        self.b = p.b;
        self.a = p.a;
    }

    pub(crate) fn reset(&mut self) {
        self.z = [0.0; 2];
    }

    /// Transposed direct form II. A non-finite sample clears the state instead
    /// of poisoning the rest of the mix.
    #[inline]
    pub(crate) fn tick(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x + self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x + self.a[1] * y;
        if !y.is_finite() {
            self.reset();
            return 0.0;
        }
        for z in &mut self.z {
            if z.abs() < 1e-25 || !z.is_finite() {
                *z = 0.0;
            }
        }
        y
    }

    pub(crate) fn process(&mut self, s: &mut [f32]) {
        for x in s.iter_mut() {
            *x = self.tick(*x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route() -> Route {
        let mut r = Route {
            channels: 1,
            ..Route::default()
        };
        r.input[0] = 0;
        r.output[0] = 0;
        r
    }

    #[test]
    fn biquad_parameter_offsets_match_the_public_layout() {
        let q = |v: f32| (v * 16384.0) as i16;
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[6] = 1;
        d[0x0C..0x0E].copy_from_slice(&q(0.25).to_le_bytes());
        d[0x0E..0x10].copy_from_slice(&q(0.5).to_le_bytes());
        d[0x10..0x12].copy_from_slice(&q(0.25).to_le_bytes());
        d[0x12..0x14].copy_from_slice(&q(-0.4).to_le_bytes());
        d[0x14..0x16].copy_from_slice(&q(0.2).to_le_bytes());
        d[0x16] = 1;
        let p = BiquadParams::parse(&d);
        assert_eq!(p.route.channels, 1);
        assert!((p.b[1] - 0.5).abs() < 1e-4);
        assert!((p.a[0] + 0.4).abs() < 1e-4);
        assert_eq!(p.route.input[0], 0);
        assert_eq!(p.route.output[0], 1);
    }

    #[test]
    fn biquad_impulse_response_matches_direct_form() {
        // A stable second-order lowpass: b = 0.25, 0.5, 0.25; a1 = 0.4, a2 = -0.2.
        let q = |v: f32| (v * 16384.0) as i16;
        let p = BiquadParams::from_q14(route(), [q(0.25), q(0.5), q(0.25), q(0.4), q(-0.2)]);
        assert!(p.is_stable());

        let mut bq = Biquad::default();
        bq.set(&p);
        let impulse = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut got = impulse;
        bq.process(&mut got);

        // Independent direct form I reference with the documented denominator sign.
        let (b, a) = (p.b, p.a);
        let mut x1 = 0.0;
        let mut x2 = 0.0;
        let mut y1 = 0.0;
        let mut y2 = 0.0;
        let mut want = [0.0f32; 8];
        for (n, &x) in impulse.iter().enumerate() {
            let y = b[0] * x + b[1] * x1 + b[2] * x2 + a[0] * y1 + a[1] * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;
            want[n] = y;
        }
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{got:?} vs {want:?}");
        }
        assert!(
            want.iter().any(|v| v.abs() > 0.1),
            "filter must not be a no-op"
        );
    }

    #[test]
    fn unstable_biquad_is_rejected() {
        let q = |v: f32| (v * 16384.0) as i16;
        let p = BiquadParams::from_q14(route(), [q(1.0), 0, 0, q(1.5), 0]);
        assert_eq!(p.b, [0.0; 3]);
        assert_eq!(p.a, [0.0; 2]);
        assert!(!p.routable());
    }
}
