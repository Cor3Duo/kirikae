//! Auxiliary bus and capture buffer effects.
//!
//! Both share `AuxiliaryBufferParameter` (Ryujinx
//! `Parameter/Effect/AuxiliaryBufferParameter.cs`):
//!
//! ```text
//! +0x00 u8[24] input channel indices
//! +0x18 u8[24] output channel indices
//! +0x30 u32    channel count
//! +0x34 u32    sample rate (the DSP command does not resample for the user)
//! +0x38 u32    buffer storage size, in samples
//! +0x3C u32    maximum channel count (unused)
//! +0x40 u64    send buffer info address
//! +0x48 u64    send buffer storage address (unused)
//! +0x50 u64    return buffer info address
//! +0x58 u64    return buffer storage address (unused)
//! +0x60 u32    mix buffer sample size (unused)
//! +0x64 u32    total sample count (unused)
//! +0x68 u32    mix buffer sample count (unused)
//! ```
//!
//! The guest owns a ring buffer. Its info area is two 0x40-byte halves; the
//! DSP uses the second one and the samples start 0x80 bytes after the info
//! address (`AuxiliaryBufferHeader` in Ryujinx). The application consumes the
//! send ring and fills the return ring, so an auxiliary bus is only useful
//! when guest memory is available; this module exchanges it through
//! guest memory.
//!
//! A capture buffer is the same block with only the send side used.
//!
//! The block carries its own sample rate, but the DSP does not resample the
//! auxiliary bus. [`AuxClock`] makes that decision explicit at the API: a block
//! whose rate differs from the sink is still exchanged 1:1, [`AuxClock::exchange`]
//! returns the block size unchanged for both variants, and both sides advance
//! their pointers by `channels * sample_count` samples per command. The
//! application owns the producer/consumer clock; the runtime neither inserts a
//! resampler (which would change the sample count per command and desynchronise
//! the guest pointers) nor invents samples. On the console the two rates match;
//! when they do not, the mismatch is the application's, and this module keeps
//! the ring bounded and reports it once.

use super::{MAX_CHANNELS, MAX_MIXES, Route, trace_once, u32_at, u64_at};
use crate::services::audio::mixer::Mem;

const INFO_HALF: u64 = 0x40;
const DATA_OFFSET: u64 = 0x80;
const READ_OFFSET: u64 = 0x00;
const WRITE_OFFSET: u64 = 0x04;
const TOTAL_OFFSET: u64 = 0x0C;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct AuxParams {
    pub route: Route,
    pub sample_rate: u32,
    pub storage: usize,
    pub send_info: u64,
    pub return_info: u64,
    pub capture: bool,
}

impl AuxParams {
    pub(crate) fn parse(d: &[u8], capture: bool) -> AuxParams {
        // Input and output are two 24-byte arrays here, unlike the 6-byte
        // arrays of the other effects.
        let mut route = Route::parse(d);
        for k in 0..MAX_CHANNELS {
            route.output[k] = super::byte_at(d, MAX_MIXES + k);
        }
        let channels = (u32_at(d, 0x30) as usize).min(MAX_MIXES);
        AuxParams {
            route: route.with_channels(channels.min(MAX_CHANNELS)),
            sample_rate: u32_at(d, 0x34),
            storage: u32_at(d, 0x38) as usize,
            send_info: u64_at(d, 0x40),
            return_info: if capture { 0 } else { u64_at(d, 0x50) },
            capture,
        }
    }

    pub(crate) fn routable(&self) -> bool {
        self.route.routable()
            && self.storage > 0
            && self.send_info != 0
            && (self.capture || self.return_info != 0)
    }
}

/// How the auxiliary ring is clocked relative to the mix.
///
/// This is a decision, not an error: the DSP never resamples here. The variant
/// is computed once per processing pass and the exchange size is always the
/// renderer block size, so the guest pointers advance identically whatever the
/// declared rates are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuxClock {
    /// The block declares no rate, or it matches the sink.
    Matched,
    /// The block declares another rate; exchanged 1:1 anyway.
    Mismatched { effect: u32, sink: u32 },
}

impl AuxClock {
    pub(crate) fn of(params: &AuxParams, sink_rate: u32) -> AuxClock {
        if params.sample_rate == 0 || sink_rate == 0 || params.sample_rate == sink_rate {
            AuxClock::Matched
        } else {
            AuxClock::Mismatched {
                effect: params.sample_rate,
                sink: sink_rate,
            }
        }
    }

    /// Samples exchanged per command. The aux bus never resamples, so this is
    /// the renderer block size in both variants.
    pub(crate) fn exchange(self, block: usize) -> usize {
        match self {
            AuxClock::Matched | AuxClock::Mismatched { .. } => block,
        }
    }
}

fn read_u32(mem: &dyn Mem, addr: u64) -> Option<u32> {
    let b = mem.read(addr, 4)?;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

fn write_u32(mem: &dyn Mem, addr: u64, value: u32) -> bool {
    mem.write(addr, &value.to_le_bytes())
}

/// Read `samples` interleaved f32 samples from a wrapped i32 ring.
fn ring_read(
    mem: &dyn Mem,
    data: u64,
    storage: usize,
    start: usize,
    samples: &mut [f32],
) -> Option<()> {
    let mut at = start % storage;
    let mut done = 0;
    while done < samples.len() {
        let chunk = (storage - at).min(samples.len() - done);
        let bytes = mem.read(data + at as u64 * 4, chunk as u64 * 4)?;
        for (i, b) in bytes.as_chunks::<4>().0.iter().enumerate() {
            samples[done + i] = i32::from_le_bytes(*b) as f32;
        }
        at = 0;
        done += chunk;
    }
    Some(())
}

/// Write `samples` interleaved f32 samples to a wrapped i32 ring.
fn ring_write(mem: &dyn Mem, data: u64, storage: usize, start: usize, samples: &[f32]) -> bool {
    let mut at = start % storage;
    let mut done = 0;
    let mut bytes = Vec::new();
    while done < samples.len() {
        let chunk = (storage - at).min(samples.len() - done);
        bytes.clear();
        bytes.extend(
            samples[done..done + chunk]
                .iter()
                .flat_map(|v| (*v as i32).to_le_bytes()),
        );
        if !mem.write(data + at as u64 * 4, &bytes) {
            return false;
        }
        at = 0;
        done += chunk;
    }
    true
}

fn reset(mem: &dyn Mem, info: u64) {
    if info == 0 {
        return;
    }
    let dsp = info + INFO_HALF;
    write_u32(mem, dsp + READ_OFFSET, 0);
    write_u32(mem, dsp + TOTAL_OFFSET, 0);
}

fn send(
    mem: &dyn Mem,
    info: u64,
    storage: usize,
    channels: usize,
    block: &[Vec<f32>],
    n: usize,
) -> bool {
    let Some(write) = read_u32(mem, info + INFO_HALF + WRITE_OFFSET) else {
        return false;
    };
    let write = write as usize;
    if write > storage {
        return false;
    }
    for (c, channel) in block.iter().enumerate().take(channels) {
        let Some(samples) = channel.get(..n) else {
            return false;
        };
        if !ring_write(mem, info + DATA_OFFSET, storage, write + c * n, samples) {
            return false;
        }
    }
    let advanced = (write + channels * n) % storage;
    write_u32(mem, info + INFO_HALF + WRITE_OFFSET, advanced as u32)
}

fn receive(
    mem: &dyn Mem,
    info: u64,
    storage: usize,
    channels: usize,
    block: &mut [Vec<f32>],
    n: usize,
) -> bool {
    let Some(read) = read_u32(mem, info + INFO_HALF + READ_OFFSET) else {
        return false;
    };
    let read = read as usize;
    if read > storage {
        return false;
    }
    for (c, channel) in block.iter_mut().enumerate().take(channels) {
        let Some(samples) = channel.get_mut(..n) else {
            return false;
        };
        if ring_read(mem, info + DATA_OFFSET, storage, read + c * n, samples).is_none() {
            return false;
        }
    }
    let advanced = (read + channels * n) % storage;
    write_u32(mem, info + INFO_HALF + READ_OFFSET, advanced as u32)
}

fn passthrough(input: &[Vec<f32>], output: &mut [Vec<f32>], n: usize, channels: usize) {
    for c in 0..channels {
        if let (Some(src), Some(dst)) = (input[c].get(..n), output[c].get_mut(..n)) {
            dst.copy_from_slice(src);
        }
    }
}

/// Route one block through the auxiliary buffer. When the effect is disabled,
/// unusable, or the guest buffer cannot be reached, the input is copied to the
/// output so the mix keeps its dry signal.
pub(crate) fn process(
    params: &AuxParams,
    mem: &dyn Mem,
    input: &[Vec<f32>],
    output: &mut [Vec<f32>],
    block: usize,
    enabled: bool,
    clock: AuxClock,
) {
    let channels = params.route.channels.min(input.len()).min(output.len());
    if channels == 0 {
        return;
    }
    if let AuxClock::Mismatched { effect, sink } = clock {
        trace_once(format!(
            "audio: auxiliary bus is configured at {effect} Hz but the mix runs at {sink} Hz; exchanging the ring 1:1 without resampling"
        ));
    }
    let n = clock.exchange(block);
    if !enabled || !params.routable() {
        reset(mem, params.send_info);
        reset(mem, params.return_info);
        passthrough(input, output, n, channels);
        return;
    }

    let sent = send(mem, params.send_info, params.storage, channels, input, n);
    if params.capture {
        // A capture buffer only exposes the mix to the application; audio is
        // left untouched.
        if !sent {
            trace_once("audio: capture buffer could not write guest memory".into());
        }
        passthrough(input, output, n, channels);
        return;
    }

    let received = sent && receive(mem, params.return_info, params.storage, channels, output, n);
    if !received {
        trace_once(format!(
            "audio: auxiliary bus {:#x}/{:#x} is unreachable, routing the dry mix",
            params.send_info, params.return_info
        ));
        passthrough(input, output, n, channels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auxiliary_block_offsets() {
        let mut d = [0u8; 0xA0];
        d[0] = 0;
        d[6] = 1;
        d[0x30..0x34].copy_from_slice(&2u32.to_le_bytes());
        d[0x34..0x38].copy_from_slice(&48000u32.to_le_bytes());
        d[0x38..0x3C].copy_from_slice(&1024u32.to_le_bytes());
        d[0x40..0x48].copy_from_slice(&0x1000u64.to_le_bytes());
        d[0x50..0x58].copy_from_slice(&0x2000u64.to_le_bytes());
        let p = AuxParams::parse(&d, false);
        assert_eq!(p.route.channels, 2);
        assert_eq!(p.storage, 1024);
        assert_eq!((p.send_info, p.return_info), (0x1000, 0x2000));
        assert!(p.routable());
    }

    #[test]
    fn capture_buffer_has_no_return_side() {
        let mut d = [0u8; 0xA0];
        d[0x30..0x34].copy_from_slice(&2u32.to_le_bytes());
        d[0x38..0x3C].copy_from_slice(&512u32.to_le_bytes());
        d[0x40..0x48].copy_from_slice(&0x1000u64.to_le_bytes());
        let p = AuxParams::parse(&d, true);
        assert!(p.capture);
        assert!(p.routable());
        assert_eq!(p.route.input[0], 0);
    }

    #[test]
    fn the_exchange_size_is_the_block_size_whatever_the_rate() {
        let mut d = [0u8; 0xA0];
        d[0x30..0x34].copy_from_slice(&2u32.to_le_bytes());
        d[0x34..0x38].copy_from_slice(&44_100u32.to_le_bytes());
        d[0x38..0x3C].copy_from_slice(&1024u32.to_le_bytes());
        d[0x40..0x48].copy_from_slice(&0x1000u64.to_le_bytes());
        d[0x50..0x58].copy_from_slice(&0x2000u64.to_le_bytes());
        let p = AuxParams::parse(&d, false);

        let clock = AuxClock::of(&p, 48_000);
        assert_eq!(
            clock,
            AuxClock::Mismatched {
                effect: 44_100,
                sink: 48_000
            }
        );
        assert!(!matches!(clock, AuxClock::Matched));
        assert_eq!(clock.exchange(240), 240, "the block size never changes");
        assert_eq!(clock.exchange(1), 1);

        assert_eq!(AuxClock::of(&p, 44_100), AuxClock::Matched);
        assert_eq!(AuxClock::of(&p, 0), AuxClock::Matched);
        assert!(matches!(
            AuxClock::of(&p, 48_000),
            AuxClock::Mismatched { .. }
        ));

        // An unspecified effect rate is not a mismatch.
        d[0x34..0x38].fill(0);
        let p = AuxParams::parse(&d, false);
        assert_eq!(AuxClock::of(&p, 48_000), AuxClock::Matched);
    }
}
