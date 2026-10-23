use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const WORK_BUFFER_SIZE: u32 = 0x8000;

pub struct HwOpus;

impl Service for HwOpus {
    fn name(&self) -> String {
        "hwopus".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 | 4 => Reply::ok().object(Arc::new(Decoder { sample_rate: req.u32(0), channels: req.u32(4).max(1) })),
            2 | 6 => {
                let params = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size.min(8))).unwrap_or_default();
                let word = |at: usize| params.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
                Reply::ok().object(Arc::new(Decoder { sample_rate: word(0), channels: word(4).max(1) }))
            }
            1 | 3 | 5 | 7 | 8 | 9 => Reply::ok().u32(WORK_BUFFER_SIZE),
            _ => unimplemented("hwopus", c, req),
        }
    }
}

struct Decoder {
    sample_rate: u32,
    channels: u32,
}

impl Service for Decoder {
    fn name(&self) -> String {
        "hwopus/IHardwareOpusDecoder".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 | 2 | 4 | 5 | 6 | 7 => {
                let input = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size.min(0x10000))).unwrap_or_default();
                let (consumed, samples) = packet(&input, self.sample_rate);
                if let Some(out) = req.out_buf(0) {
                    let bytes = (samples as u64 * self.channels as u64 * 2).min(out.size);
                    p.memory.write(out.addr, &vec![0u8; bytes as usize]);
                }
                let reply = Reply::ok().u32(consumed).u32(samples);
                if req.cmd >= 4 { reply.u64(0) } else { reply }
            }
            1 | 3 => Reply::ok(),
            _ => unimplemented("hwopus/IHardwareOpusDecoder", c, req),
        }
    }
}

fn packet(data: &[u8], sample_rate: u32) -> (u32, u32) {
    let Some(header) = data.get(..8) else { return (data.len() as u32, 0) };
    let size = u32::from_be_bytes(header[..4].try_into().unwrap());
    let payload = &data[8..data.len().min(8 + size as usize)];
    (8 + payload.len() as u32, samples(payload, sample_rate))
}

fn samples(payload: &[u8], sample_rate: u32) -> u32 {
    let Some(&toc) = payload.first() else { return 0 };
    let config = toc >> 3;
    let units = match config {
        0..=11 => [4, 8, 16, 24][(config & 3) as usize],
        12..=15 => [4, 8][(config & 1) as usize],
        _ => [1, 2, 4, 8][(config & 3) as usize],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => payload.get(1).map_or(0, |&b| (b & 0x3f) as u32),
    };
    units * frames * sample_rate.max(8000) / 400
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_sizes() {
        assert_eq!(samples(&[0xf8], 48000), 960);
        assert_eq!(samples(&[0x19], 48000), 5760);
        assert_eq!(samples(&[0x93, 0x03], 24000), 720);
        let mut data = vec![0, 0, 0, 2, 0xaa, 0xbb, 0xcc, 0xdd, 0xf8, 0xff, 0x11];
        assert_eq!(packet(&data, 48000), (10, 960));
        data.truncate(4);
        assert_eq!(packet(&data, 48000), (4, 0));
    }
}
