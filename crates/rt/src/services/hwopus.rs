//! `hwopus`/`IHardwareOpusDecoder`.
//!
//! Real decoding lives in [`crate::services::audio::opus`]; this module keeps
//! the Horizon command surface. The packet is an 8-byte big-endian header
//! (payload length and final range) followed by one Opus packet, and the
//! replies are the consumed byte count and the decoded sample count per
//! channel, as documented by Switchbrew and implemented by Ryujinx.

use std::sync::Arc;

use super::audio::opus::OpusDecoder;
use crate::cpu::Cpu;
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const WORK_BUFFER_SIZE: u32 = 0x8000;
const MULTISTREAM_IN: usize = 0x118;

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
        match req.cmd {
            0 | 4 => {
                let decoder = OpusDecoder::new(req.u32(0), req.u32(4));
                Reply::ok().object(open(p, decoder, req.u32(0), req.u32(4)))
            }
            2 | 6 => {
                let ex = req.cmd == 6;
                let data = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size.min(MULTISTREAM_IN as u64)))
                    .unwrap_or_default();
                let (rate, channels, streams, coupled, mapping) = parse_multistream(&data, ex);
                let decoder =
                    OpusDecoder::new_multistream(rate, channels, streams, coupled, mapping);
                Reply::ok().object(open(p, decoder, rate, channels))
            }
            1 | 3 | 5 | 7 | 8 | 9 => Reply::ok().u32(WORK_BUFFER_SIZE),
            _ => unimplemented("hwopus", c, req),
        }
    }
}

struct Decoder {
    decoder: OpusDecoder,
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
                if let Some(out) = req.out_buf(0) {
                    let bytes: Vec<u8> = decoded
                        .pcm
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .take(out.size as usize)
                        .collect();
                    p.memory.write(out.addr, &bytes);
                }
                let reply = Reply::ok().u32(decoded.consumed).u32(decoded.samples);
                if req.cmd >= 4 { reply.u64(0) } else { reply }
            }
            _ => unimplemented("hwopus/IHardwareOpusDecoder", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
