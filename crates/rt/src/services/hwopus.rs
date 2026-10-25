//! `hwopus`/`IHardwareOpusDecoder`.
//!
//! Real decoding lives in [`crate::services::audio::opus`]; this module keeps
//! the Horizon command surface. The packet is an 8-byte big-endian header
//! (payload length and final range) followed by one Opus packet, and the
//! replies are the consumed byte count and the decoded sample count per
//! channel, as documented by Switchbrew and implemented by Ryujinx.
//!
//! Work-buffer queries (commands 1/3/5/7/8/9) report the libopus decoder-state
//! size for the requested layout, so the transfer memory the guest creates is
//! large enough for the decoder this runtime actually opens. An invalid packet
//! is a valid, empty decode: consumed bytes and zero samples, as documented in
//! [`crate::services::audio::opus`].

use std::sync::Arc;

use super::audio::opus::{Decoded, OpusDecoder, work_buffer_size, work_buffer_size_multistream};
use crate::cpu::Cpu;
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const MULTISTREAM_IN: usize = 0x118;

/// `OpusParametersEx`: sample rate, channel count, large-frame flag. The
/// large-frame flag only enlarges the guest's PCM buffer (120 ms instead of
/// 40 ms); the decoder state size does not change and this decoder already
/// accepts 120 ms packets.
fn parse_ex(data: &[u8]) -> (u32, u32, bool) {
    let word = |at: usize| {
        data.get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    (word(0), word(4), word(8) != 0)
}

/// The legacy queries pack sample rate and channel count in one u64.
fn packed_params(data: &[u8]) -> (u32, u32) {
    let word = |at: usize| {
        data.get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    (word(0), word(4))
}

/// `OpusMultiStreamParameters` (@0x10 map) and its `Ex` variant (@0x18 map):
/// sample rate, channel count, stream count, coupled-stream count, mapping.
fn parse_multistream(data: &[u8], ex: bool) -> (u32, u32, u32, u32, &[u8]) {
    let word = |at: usize| {
        data.get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let mapping_at = if ex { 0x18 } else { 0x10 };
    (
        word(0),
        word(4),
        word(8),
        word(0xC),
        data.get(mapping_at..).unwrap_or_default(),
    )
}

fn open(p: &Process, decoder: OpusDecoder, sample_rate: u32, channels: u32) -> Arc<dyn Service> {
    if let Some(error) = decoder.failure() {
        p.warn_once(format!(
            "hwopus: {} at {} Hz x{}; decoding silence",
            error.label(),
            sample_rate,
            channels
        ));
    }
    Arc::new(Decoder { decoder })
}

pub struct HwOpus;

impl Service for HwOpus {
    fn name(&self) -> String {
        "hwopus".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let read_input = |limit: usize| {
            req.in_buf(0)
                .and_then(|b| p.memory.read(b.addr, b.size.min(limit as u64)))
                .unwrap_or_default()
        };
        match req.cmd {
            0 => {
                let decoder = OpusDecoder::new(req.u32(0), req.u32(4));
                Reply::ok().object(open(p, decoder, req.u32(0), req.u32(4)))
            }
            4 => {
                // `OpenHardwareOpusDecoderEx` takes the parameter struct in an
                // input buffer instead of the packed u64 of command 0.
                let data = read_input(0x10);
                let (rate, channels, _large) = parse_ex(&data);
                let (rate, channels) = if rate != 0 {
                    (rate, channels)
                } else {
                    packed_params(&req.data)
                };
                let decoder = OpusDecoder::new(rate, channels);
                Reply::ok().object(open(p, decoder, rate, channels))
            }
            2 | 6 => {
                let ex = req.cmd == 6;
                let data = read_input(MULTISTREAM_IN);
                let (rate, channels, streams, coupled, mapping) = parse_multistream(&data, ex);
                let decoder =
                    OpusDecoder::new_multistream(rate, channels, streams, coupled, mapping);
                Reply::ok().object(open(p, decoder, rate, channels))
            }
            1 => {
                let (rate, channels) = packed_params(&req.data);
                Reply::ok().u32(work_buffer_size(rate, channels))
            }
            3 | 7 => {
                let data = read_input(MULTISTREAM_IN);
                let (rate, channels, streams, coupled, mapping) =
                    parse_multistream(&data, req.cmd == 7);
                Reply::ok().u32(work_buffer_size_multistream(
                    rate, channels, streams, coupled, mapping,
                ))
            }
            5 | 8 => {
                // Ex and ExEx differ in the input-buffer type and the extra
                // large-frame flag, not in the decoder state.
                let data = read_input(0x10);
                let (rate, channels, _large) = parse_ex(&data);
                let (rate, channels) = if rate != 0 {
                    (rate, channels)
                } else {
                    packed_params(&req.data)
                };
                Reply::ok().u32(work_buffer_size(rate, channels))
            }
            9 => {
                let data = read_input(MULTISTREAM_IN);
                let (rate, channels, streams, coupled, mapping) = parse_multistream(&data, true);
                Reply::ok().u32(work_buffer_size_multistream(
                    rate, channels, streams, coupled, mapping,
                ))
            }
            _ => unimplemented("hwopus", c, req),
        }
    }
}

struct Decoder {
    decoder: OpusDecoder,
}

/// The two/three output words every decode command returns. An invalid packet
/// is a successful empty decode: the consumed byte count and zero samples.
fn decode_reply(decoded: &Decoded, cmd: u32) -> Reply {
    let reply = Reply::ok().u32(decoded.consumed).u32(decoded.samples);
    if cmd >= 4 { reply.u64(0) } else { reply }
}

impl Service for Decoder {
    fn name(&self) -> String {
        "hwopus/IHardwareOpusDecoder".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0..=9 => {
                let input = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size.min(0x10000)))
                    .unwrap_or_default();
                if matches!(req.cmd, 1 | 3) {
                    // SetContext/SetContextForMultiStream: the context is opaque,
                    // but resetting the decoder matches the intent.
                    self.decoder.set_context();
                    return Reply::ok();
                }
                let reset = matches!(req.cmd, 6..=9) && req.data.first().is_some_and(|&b| b != 0);
                let decoded = self.decoder.decode(&input, reset);
                if let Some(error) = decoded.error {
                    p.warn_once(format!("hwopus: {}", error.label()));
                }
                if decoded.samples > 0
                    && let Some(out) = req.out_buf(0)
                {
                    let bytes: Vec<u8> = decoded
                        .pcm
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .take(out.size as usize)
                        .collect();
                    p.memory.write(out.addr, &bytes);
                }
                decode_reply(&decoded, req.cmd)
            }
            _ => unimplemented("hwopus/IHardwareOpusDecoder", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::audio::opus::DecodeError;

    #[test]
    fn multistream_open_keeps_the_error_label() {
        let decoder = OpusDecoder::new(48_000, 6);
        assert_eq!(
            decoder.failure(),
            Some(super::super::audio::opus::DecodeError::UnsupportedChannels)
        );
    }

    #[test]
    fn multistream_parameters_use_the_documented_offsets() {
        let mut d = vec![0u8; MULTISTREAM_IN];
        d[0..4].copy_from_slice(&48_000u32.to_le_bytes());
        d[4..8].copy_from_slice(&6u32.to_le_bytes());
        d[8..12].copy_from_slice(&4u32.to_le_bytes());
        d[12..16].copy_from_slice(&2u32.to_le_bytes());
        d[0x10..0x16].copy_from_slice(&[0, 4, 1, 2, 3, 5]);
        let (rate, channels, streams, coupled, mapping) = parse_multistream(&d, false);
        assert_eq!((rate, channels, streams, coupled), (48_000, 6, 4, 2));
        assert_eq!(&mapping[..6], &[0, 4, 1, 2, 3, 5]);
        let decoder = OpusDecoder::new_multistream(rate, channels, streams, coupled, mapping);
        assert_eq!(decoder.failure(), None);

        // The `Ex` variant shifts the map past the two extra flag words.
        let mut d = vec![0u8; MULTISTREAM_IN];
        d[0..4].copy_from_slice(&48_000u32.to_le_bytes());
        d[4..8].copy_from_slice(&6u32.to_le_bytes());
        d[8..12].copy_from_slice(&4u32.to_le_bytes());
        d[12..16].copy_from_slice(&2u32.to_le_bytes());
        d[0x18..0x1E].copy_from_slice(&[0, 4, 1, 2, 3, 5]);
        let (_, _, _, _, mapping) = parse_multistream(&d, true);
        assert_eq!(&mapping[..6], &[0, 4, 1, 2, 3, 5]);
    }

    #[test]
    fn ex_parameters_use_the_documented_offsets() {
        let mut d = vec![0u8; 0x10];
        d[0..4].copy_from_slice(&48_000u32.to_le_bytes());
        d[4..8].copy_from_slice(&6u32.to_le_bytes());
        d[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse_ex(&d), (48_000, 6, true));
        d[8..12].fill(0);
        assert_eq!(parse_ex(&d), (48_000, 6, false));

        let mut packed = vec![0u8; 8];
        packed[0..4].copy_from_slice(&48_000u32.to_le_bytes());
        packed[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(packed_params(&packed), (48_000, 2));
    }

    #[test]
    fn an_invalid_packet_is_a_zero_sample_success() {
        let decoder = OpusDecoder::new(48_000, 2);
        // A truncated header: three bytes are all the service can consume.
        let decoded = decoder.decode(&[0, 0, 0], false);
        assert_eq!(decoded.error, Some(DecodeError::InvalidLength));
        assert_eq!(decoded.samples, 0);
        assert!(decoded.pcm.is_empty());
        let reply = decode_reply(&decoded, 0);
        assert_eq!(reply.result, 0);
        assert_eq!(reply.data, [3, 0, 0, 0, 0, 0, 0, 0]);

        // The newer decode commands append the ignored performance word.
        let reply = decode_reply(&decoded, 8);
        assert_eq!(reply.result, 0);
        assert_eq!(reply.data.len(), 16);
        assert_eq!(&reply.data[8..], &[0u8; 8]);

        // A declared payload longer than the input consumes what is available.
        let decoded = decoder.decode(&[0, 0, 0, 8, 0, 0, 0, 0, 1], false);
        assert_eq!(decoded.consumed, 9);
        assert_eq!(decoded.samples, 0);
        assert!(decoded.pcm.is_empty());
    }
}
