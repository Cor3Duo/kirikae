//! `audren` mix effects.
//!
//! An effect is described by an `EffectInParameterVersion1`/`Version2` block
//! (0xC0 bytes, version 2 only changes the outcome status): the first 0x20
//! bytes are the common descriptor and the remaining 0xA0 bytes are the
//! effect-specific parameter block.
//!
//! ```text
//! 0x00 u8   effect type
//! 0x01 u8   is new
//! 0x02 u8   enabled
//! 0x04 i32  destination mix
//! 0x08 u64  processing work buffer address
//! 0x10 u64  processing work buffer size
//! 0x18 u32  processing order inside the mix
//! 0x20      effect-specific parameters (0xA0 bytes)
//! ```
//!
//! The effect numbers and the specific parameter layouts follow the public
//! MIT-licensed Ryujinx audio renderer (`Common/EffectType.cs` and
//! `Parameter/Effect/*.cs`); Switchbrew's audio page documents the audren IPC
//! but not these blocks. Every layout used here is therefore recorded next to
//! its parser. Effects whose block is not public are left as a labelled
//! passthrough instead of being guessed.
//!
//! The DSP engines live in `aux_bus`, `biquad`, `compressor`, `delay`,
//! `limiter`, `line`, `reverb` and `reverb3d`; this module owns the descriptor
//! parsing and the per-mix processor lifecycle.

mod aux_bus;
mod biquad;
mod compressor;
mod delay;
mod limiter;
mod line;
mod reverb;
mod reverb3d;

pub(crate) use aux_bus::{AuxClock, AuxParams, process as process_aux};
pub(crate) use biquad::{Biquad, BiquadParams};
pub(crate) use compressor::{Compressor, CompressorParams};
pub(crate) use delay::{DelayEffect, DelayParams};
pub(crate) use limiter::{Limiter, LimiterParams};
pub(crate) use reverb::{Reverb, ReverbParams};
pub(crate) use reverb3d::{Reverb3d, Reverb3dParams};

pub(crate) const MAX_CHANNELS: usize = 6;
pub(crate) const MAX_MIXES: usize = 24;
pub(crate) const EFFECT_SIZE: usize = 0xC0;

/// `EffectResultState`: the REV9 per-effect specific data.
pub(crate) const RESULT_STATE_SIZE: usize = 0x80;
/// `EffectOutStatusVersion2` is `State` (4 bytes), 15 reserved bytes and then
/// `ResultState`, so the specific data starts at +0x13.
pub(crate) const RESULT_STATE_OFFSET: usize = 0x13;
/// `EffectOutStatusVersion2` exists from REV9 on.
pub(crate) const RESULT_STATE_REVISION: u32 = 9;

pub(crate) const KIND_INVALID: u8 = 0;
pub(crate) const KIND_BUFFER_MIX: u8 = 1;
pub(crate) const KIND_AUX_BUS: u8 = 2;
pub(crate) const KIND_DELAY: u8 = 3;
pub(crate) const KIND_REVERB: u8 = 4;
pub(crate) const KIND_REVERB3D: u8 = 5;
pub(crate) const KIND_BIQUAD: u8 = 6;
pub(crate) const KIND_LIMITER: u8 = 7;
pub(crate) const KIND_CAPTURE: u8 = 8;
pub(crate) const KIND_COMPRESSOR: u8 = 9;

const EFFECT_ENABLED: u32 = 3;
const EFFECT_DISABLED: u32 = 4;

fn byte_at(d: &[u8], o: usize) -> u8 {
    d.get(o).copied().unwrap_or(0)
}
fn u16_at(d: &[u8], o: usize) -> u16 {
    d.get(o..o + 2)
        .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
}
fn u32_at(d: &[u8], o: usize) -> u32 {
    d.get(o..o + 4)
        .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn i32_at(d: &[u8], o: usize) -> i32 {
    u32_at(d, o) as i32
}
fn i16_at(d: &[u8], o: usize) -> i16 {
    u16_at(d, o) as i16
}
fn u64_at(d: &[u8], o: usize) -> u64 {
    d.get(o..o + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
}
fn f32_at(d: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(d, o))
}

fn q14(v: i32) -> f32 {
    v as f32 / 16384.0
}

/// Q14 fixed point decoded to a linear gain, with non-finite values rejected.
fn gain(v: i32) -> f32 {
    let g = q14(v);
    if g.is_finite() {
        g.clamp(0.0, 4.0)
    } else {
        0.0
    }
}

fn finite(v: f32, min: f32, max: f32) -> f32 {
    if v.is_finite() {
        v.clamp(min, max)
    } else {
        min
    }
}

fn valid_channels(channels: usize) -> bool {
    matches!(channels, 1 | 2 | 4 | 6)
}

/// Input/output mix buffer indices shared by the effects that route one buffer
/// per channel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Route {
    pub input: [u8; MAX_CHANNELS],
    pub output: [u8; MAX_CHANNELS],
    pub channels: usize,
}

impl Route {
    fn parse(d: &[u8]) -> Route {
        let mut input = [0u8; MAX_CHANNELS];
        let mut output = [0u8; MAX_CHANNELS];
        for (k, v) in input.iter_mut().enumerate() {
            *v = byte_at(d, k);
        }
        for (k, v) in output.iter_mut().enumerate() {
            *v = byte_at(d, MAX_CHANNELS + k);
        }
        Route {
            input,
            output,
            channels: 0,
        }
    }

    fn with_channels(mut self, channels: usize) -> Route {
        self.channels = channels;
        self
    }

    /// Channels must be one of 1, 2, 4, 6 and every index must be a real mix
    /// buffer. The caller validates the indices against its buffer count.
    pub(crate) fn routable(&self) -> bool {
        valid_channels(self.channels)
    }
}

/// `BufferMixParameter`: 24 input/output pairs with a volume each.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BufferMixParams {
    pub input: [u8; MAX_MIXES],
    pub output: [u8; MAX_MIXES],
    pub volumes: [f32; MAX_MIXES],
    pub mixes: usize,
}

impl BufferMixParams {
    fn parse(d: &[u8]) -> BufferMixParams {
        let mut p = BufferMixParams::default();
        for k in 0..MAX_MIXES {
            p.input[k] = byte_at(d, k);
            p.output[k] = byte_at(d, MAX_MIXES + k);
            p.volumes[k] = finite(f32_at(d, 2 * MAX_MIXES + k * 4), -8.0, 8.0);
        }
        p.mixes = (u32_at(d, 2 * MAX_MIXES + MAX_MIXES * 4) as usize).min(MAX_MIXES);
        p
    }

    pub(crate) fn routable(&self) -> bool {
        self.mixes > 0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Specific {
    #[default]
    Invalid,
    BufferMix(BufferMixParams),
    Aux(AuxParams),
    Capture(AuxParams),
    Delay(DelayParams),
    Reverb(ReverbParams),
    Reverb3d(Reverb3dParams),
    Biquad(BiquadParams),
    Limiter(LimiterParams),
    Compressor(CompressorParams),
    /// A known effect whose public parameter block is not confirmable.
    Unimplemented(u8),
}

impl Specific {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Specific::Invalid => "invalid",
            Specific::BufferMix(_) => "buffer mix",
            Specific::Aux(_) => "auxiliary bus",
            Specific::Capture(_) => "capture buffer",
            Specific::Delay(_) => "delay",
            Specific::Reverb(_) => "reverb",
            Specific::Reverb3d(_) => "3d reverb",
            Specific::Biquad(_) => "biquad filter",
            Specific::Limiter(_) => "limiter",
            Specific::Compressor(_) => "compressor",
            Specific::Unimplemented(_) => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Descriptor {
    pub kind: u8,
    pub is_new: bool,
    pub enabled: bool,
    pub mix: i32,
    pub order: u32,
}

pub(crate) enum Processor {
    None,
    Delay(Box<DelayEffect>),
    Reverb(Box<Reverb>),
    Reverb3d(Box<Reverb3d>),
    Biquad {
        params: BiquadParams,
        state: Box<[Biquad; MAX_CHANNELS]>,
    },
    Limiter(Box<Limiter>),
    Compressor(Box<Compressor>),
}

/// The per-effect part of `EffectOutStatusVersion1`/`Version2`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OutStatus {
    pub state: u32,
    /// `EffectResultState.SpecificData`.
    pub result: [u8; RESULT_STATE_SIZE],
    pub result_valid: bool,
}

impl Default for OutStatus {
    fn default() -> OutStatus {
        OutStatus {
            state: 0,
            result: [0; RESULT_STATE_SIZE],
            result_valid: false,
        }
    }
}

impl Processor {
    pub(crate) fn new(desc: &Descriptor, specific: &Specific, sample_rate: u32) -> Processor {
        match specific {
            Specific::Biquad(p) if p.routable() => {
                let mut state = Box::new([Biquad::default(); MAX_CHANNELS]);
                for b in state.iter_mut() {
                    b.set(p);
                }
                Processor::Biquad { params: *p, state }
            }
            Specific::Delay(p) if p.routable() => {
                Processor::Delay(Box::new(DelayEffect::new(*p, sample_rate)))
            }
            Specific::Reverb(p) if p.routable() => Processor::Reverb(Box::new(Reverb::new(*p))),
            Specific::Reverb3d(p) if p.routable() => {
                Processor::Reverb3d(Box::new(Reverb3d::new(*p)))
            }
            Specific::Limiter(p) if p.routable() => {
                Processor::Limiter(Box::new(Limiter::new(*p, sample_rate)))
            }
            Specific::Compressor(p) if p.routable() => {
                Processor::Compressor(Box::new(Compressor::new(*p, sample_rate)))
            }
            Specific::BufferMix(_) | Specific::Aux(_) | Specific::Capture(_) => Processor::None,
            Specific::Compressor(_) => {
                trace_once("audio: compressor parameters are not usable; routing it dry".into());
                Processor::None
            }
            Specific::Unimplemented(kind) => {
                trace_once(format!(
                    "audio: effect type {kind} is not implemented; routing it dry"
                ));
                Processor::None
            }
            _ => {
                if desc.kind != KIND_INVALID
                    && !matches!(specific, Specific::Biquad(_) | Specific::Unimplemented(_))
                {
                    trace_once(format!(
                        "audio: effect type {} is passed through (parameters not usable)",
                        desc.kind
                    ));
                }
                Processor::None
            }
        }
    }

    /// Build the `EffectOutStatusVersion2` for this slot. Only the limiter has
    /// a documented `EffectResultState` payload (`LimiterStatistics`); the
    /// compressor and the rest report their state alone.
    pub(crate) fn out_status(&self, enabled: bool, kind: u8) -> OutStatus {
        let mut out = OutStatus {
            state: status(enabled, kind),
            ..OutStatus::default()
        };
        if let Processor::Limiter(limiter) = self
            && limiter.statistics_enabled()
        {
            let (input_max, gain_min) = limiter.statistics();
            for (c, value) in input_max.iter().enumerate() {
                let at = c * 4;
                out.result[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            for (c, value) in gain_min.iter().enumerate() {
                let at = 24 + c * 4;
                out.result[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            out.result_valid = true;
        }
        out
    }

    pub(crate) fn matches(&self, specific: &Specific) -> bool {
        match (self, specific) {
            (Processor::Biquad { .. }, Specific::Biquad(p)) => p.routable(),
            (Processor::Delay(_), Specific::Delay(p)) => p.routable(),
            (Processor::Reverb(_), Specific::Reverb(p)) => p.routable(),
            (Processor::Reverb3d(_), Specific::Reverb3d(p)) => p.routable(),
            (Processor::Limiter(_), Specific::Limiter(p)) => p.routable(),
            (Processor::Compressor(_), Specific::Compressor(p)) => p.routable(),
            (Processor::None, Specific::Invalid) => true,
            (Processor::None, Specific::BufferMix(p)) => p.routable(),
            (Processor::None, Specific::Aux(p)) => p.routable(),
            (Processor::None, Specific::Capture(p)) => p.routable(),
            (Processor::None, Specific::Biquad(p)) => !p.routable(),
            (Processor::None, Specific::Unimplemented(_)) => true,
            (Processor::None, Specific::Delay(p)) => !p.routable(),
            (Processor::None, Specific::Reverb(p)) => !p.routable(),
            (Processor::None, Specific::Reverb3d(p)) => !p.routable(),
            (Processor::None, Specific::Limiter(p)) => !p.routable(),
            (Processor::None, Specific::Compressor(p)) => !p.routable(),
            _ => false,
        }
    }

    pub(crate) fn reconfigure(&mut self, specific: &Specific, sample_rate: u32) {
        match (self, specific) {
            (Processor::Biquad { params, state }, Specific::Biquad(p)) => {
                if params != p {
                    *params = *p;
                    for b in state.iter_mut() {
                        b.set(p);
                    }
                }
            }
            (Processor::Delay(delay), Specific::Delay(p)) if p.routable() => {
                delay.configure(*p, sample_rate)
            }
            (Processor::Reverb(reverb), Specific::Reverb(p)) if p.routable() => {
                reverb.configure(*p, false)
            }
            (Processor::Reverb3d(reverb), Specific::Reverb3d(p)) if p.routable() => {
                reverb.configure(*p)
            }
            (Processor::Limiter(limiter), Specific::Limiter(p)) if p.routable() => {
                limiter.configure(*p, sample_rate)
            }
            (Processor::Compressor(compressor), Specific::Compressor(p)) if p.routable() => {
                compressor.configure(*p, sample_rate)
            }
            _ => {}
        }
    }
}

/// Parse one `EffectInParameterVersion1`/`Version2` block.
pub(crate) fn parse(d: &[u8]) -> (Descriptor, Specific) {
    let desc = Descriptor {
        kind: byte_at(d, 0),
        is_new: byte_at(d, 1) != 0,
        enabled: byte_at(d, 2) != 0,
        mix: i32_at(d, 4),
        order: u32_at(d, 0x18),
    };
    let specific_data = d.get(0x20..).unwrap_or_default();
    let specific = match desc.kind {
        KIND_BUFFER_MIX => Specific::BufferMix(BufferMixParams::parse(specific_data)),
        KIND_AUX_BUS => Specific::Aux(AuxParams::parse(specific_data, false)),
        KIND_DELAY => Specific::Delay(DelayParams::parse(specific_data)),
        KIND_REVERB => Specific::Reverb(ReverbParams::parse(specific_data)),
        KIND_REVERB3D => Specific::Reverb3d(Reverb3dParams::parse(specific_data)),
        KIND_BIQUAD => Specific::Biquad(BiquadParams::parse(specific_data)),
        KIND_LIMITER => Specific::Limiter(LimiterParams::parse(specific_data)),
        KIND_CAPTURE => Specific::Capture(AuxParams::parse(specific_data, true)),
        KIND_COMPRESSOR => Specific::Compressor(CompressorParams::parse(specific_data)),
        kind => Specific::Unimplemented(kind),
    };
    (desc, specific)
}

/// The channel route of an effect, if its parameter block has one.
pub(crate) fn route(specific: &Specific) -> Option<Route> {
    match specific {
        Specific::Aux(p) | Specific::Capture(p) => Some(p.route),
        Specific::Delay(p) => Some(p.route),
        Specific::Reverb(p) => Some(p.route()),
        Specific::Reverb3d(p) => Some(p.route),
        Specific::Biquad(p) => Some(p.route),
        Specific::Limiter(p) => Some(p.route),
        Specific::Compressor(p) => Some(p.route),
        _ => None,
    }
}

pub(crate) fn status(enabled: bool, kind: u8) -> u32 {
    if kind == KIND_INVALID {
        0
    } else if enabled {
        EFFECT_ENABLED
    } else {
        EFFECT_DISABLED
    }
}

/// Encode one effect's `EffectOutStatusVersion1`/`Version2` into the effects
/// section of the update output. REV9 and later append `EffectResultState`
/// after the 4-byte state and 15 reserved bytes.
pub(crate) fn write_out_status(
    out: &mut [u8],
    slot: usize,
    stride: usize,
    revision: u32,
    status: &OutStatus,
) {
    let start = slot * stride;
    if start + 4 > out.len() {
        return;
    }
    out[start..start + 4].copy_from_slice(&status.state.to_le_bytes());
    if !status.result_valid || revision < RESULT_STATE_REVISION || stride <= RESULT_STATE_OFFSET {
        return;
    }
    let at = start + RESULT_STATE_OFFSET;
    let len = (stride - RESULT_STATE_OFFSET)
        .min(RESULT_STATE_SIZE)
        .min(out.len() - at);
    out[at..at + len].copy_from_slice(&status.result[..len]);
}

pub(crate) fn trace_once(line: String) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap();
    if seen.get_or_insert_with(HashSet::new).insert(line.clone()) {
        eprintln!("{line}");
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Build a 0xC0 effect block with the common descriptor and specific data.
    pub(crate) fn block(kind: u8, enabled: bool, specific: &[u8]) -> Vec<u8> {
        let mut e = vec![0u8; EFFECT_SIZE];
        e[0] = kind;
        e[1] = 1;
        e[2] = enabled as u8;
        e[4..8].copy_from_slice(&0i32.to_le_bytes());
        e[0x18..0x1C].copy_from_slice(&0u32.to_le_bytes());
        e[0x20..0x20 + specific.len().min(EFFECT_SIZE - 0x20)]
            .copy_from_slice(&specific[..specific.len().min(EFFECT_SIZE - 0x20)]);
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::block;

    fn specific<const N: usize>(f: impl FnOnce(&mut [u8])) -> [u8; N] {
        let mut d = [0u8; N];
        f(&mut d);
        d
    }

    #[test]
    fn descriptor_offsets_match_the_sdk_layout() {
        let mut d = vec![0u8; EFFECT_SIZE];
        d[0] = KIND_REVERB;
        d[1] = 1;
        d[2] = 1;
        d[4..8].copy_from_slice(&3i32.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&7u32.to_le_bytes());
        let (desc, specific) = parse(&d);
        assert_eq!(desc.kind, KIND_REVERB);
        assert!(desc.is_new && desc.enabled);
        assert_eq!((desc.mix, desc.order), (3, 7));
        assert!(matches!(specific, Specific::Reverb(_)));
    }

    #[test]
    fn effect_numbers_follow_the_public_audio_renderer_enum() {
        // Ryujinx `Common/EffectType.cs`: 1 buffer mix, 2 auxiliary buffer,
        // 3 delay, 4 reverb, 5 reverb 3d, 6 biquad, 7 limiter, 8 capture,
        // 9 compressor.
        assert_eq!(KIND_BUFFER_MIX, 1);
        assert_eq!(KIND_AUX_BUS, 2);
        assert_eq!(KIND_DELAY, 3);
        assert_eq!(KIND_REVERB, 4);
        assert_eq!(KIND_REVERB3D, 5);
        assert_eq!(KIND_BIQUAD, 6);
        assert_eq!(KIND_LIMITER, 7);
        assert_eq!(KIND_CAPTURE, 8);
        assert_eq!(KIND_COMPRESSOR, 9);
    }

    #[test]
    fn buffer_mix_uses_the_documented_block() {
        let d = specific::<0xA0>(|d| {
            d[0] = 0;
            d[MAX_MIXES] = 1;
            d[2 * MAX_MIXES..2 * MAX_MIXES + 4].copy_from_slice(&0.5f32.to_le_bytes());
            d[2 * MAX_MIXES + MAX_MIXES * 4..2 * MAX_MIXES + MAX_MIXES * 4 + 4]
                .copy_from_slice(&1u32.to_le_bytes());
        });
        let (_, Specific::BufferMix(p)) = parse(&block(KIND_BUFFER_MIX, true, &d)) else {
            panic!("expected a buffer mix");
        };
        assert_eq!((p.input[0], p.output[0], p.mixes), (0, 1, 1));
        assert!((p.volumes[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unsupported_effect_keeps_a_precise_label() {
        let (_, specific) = parse(&block(10, true, &[]));
        assert_eq!(specific.label(), "unknown");
        assert_eq!(route(&specific), None);
        let (_, specific) = parse(&block(KIND_COMPRESSOR, true, &[]));
        assert_eq!(specific.label(), "compressor");
    }

    #[test]
    fn status_reports_enabled_and_disabled_slots() {
        assert_eq!(status(true, KIND_REVERB), EFFECT_ENABLED);
        assert_eq!(status(false, KIND_REVERB), EFFECT_DISABLED);
        assert_eq!(status(true, KIND_INVALID), 0);
    }

    #[test]
    fn version_two_writes_the_result_state_after_the_reserved_bytes() {
        let mut out = OutStatus {
            state: EFFECT_ENABLED,
            ..OutStatus::default()
        };
        out.result_valid = true;
        out.result[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        out.result[24..28].copy_from_slice(&0.25f32.to_le_bytes());
        let mut effects = vec![0u8; 0x93];
        write_out_status(&mut effects, 0, 0x93, RESULT_STATE_REVISION, &out);
        assert_eq!(u32_at(&effects, 0), EFFECT_ENABLED);
        assert!((f32_at(&effects, RESULT_STATE_OFFSET) - 0.5).abs() < 1e-6);
        assert!((f32_at(&effects, RESULT_STATE_OFFSET + 24) - 0.25).abs() < 1e-6);

        // REV8 only has the 4-byte state, so nothing is appended.
        let mut effects = vec![0u8; 4];
        write_out_status(&mut effects, 0, 4, 8, &out);
        assert_eq!(effects, EFFECT_ENABLED.to_le_bytes());

        // A slot too small for the specific data keeps the state and the rest
        // of the slot untouched.
        let mut effects = vec![0xFFu8; 8];
        write_out_status(&mut effects, 0, 8, RESULT_STATE_REVISION, &out);
        assert_eq!(u32_at(&effects, 0), EFFECT_ENABLED);
        assert!(effects[4..].iter().all(|&b| b == 0xFF));

        // A second slot starts at its own stride.
        let mut effects = vec![0u8; 0x93 * 2];
        write_out_status(&mut effects, 1, 0x93, RESULT_STATE_REVISION, &out);
        assert_eq!(u32_at(&effects, 0), 0);
        assert_eq!(u32_at(&effects, 0x93), EFFECT_ENABLED);
        assert!((f32_at(&effects, 0x93 + RESULT_STATE_OFFSET) - 0.5).abs() < 1e-6);
    }
}
