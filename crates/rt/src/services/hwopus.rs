use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const WORK_BUFFER_SIZE: u32 = 0x8000;
const HEADER_SIZE: usize = 8;

const CODEC_MODULE: u32 = 111;
const fn codec_error(description: u32) -> u32 {
    (description << 9) | CODEC_MODULE
}

const INVALID_LENGTH: u32 = codec_error(3);
const OPUS_BAD_ARG: u32 = codec_error(130);
const OPUS_INVALID_PACKET: u32 = codec_error(133);
const INVALID_NUMBER_OF_STREAMS: u32 = codec_error(1000);
const INVALID_SAMPLE_RATE: u32 = codec_error(1001);
const INVALID_CHANNEL_COUNT: u32 = codec_error(1002);

struct Params {
    sample_rate: u32,
    channels: u32,
    streams: u32,
    stereo_streams: u32,
}

fn word(data: &[u8], at: usize) -> u32 {
    data.get(at..at + 4).and_then(|b| b.try_into().ok()).map_or(0, u32::from_le_bytes)
}

impl Params {
    fn from(data: &[u8]) -> Params {
        Params { sample_rate: word(data, 0), channels: word(data, 4), streams: word(data, 8), stereo_streams: word(data, 12) }
    }

    fn sample_rate_ok(&self) -> bool {
        matches!(self.sample_rate, 8000 | 12000 | 16000 | 24000 | 48000)
    }

    fn channels_ok(&self) -> bool {
        (1..=2).contains(&self.channels)
    }

    fn multi_channels_ok(&self) -> bool {
        (1..=255).contains(&self.channels)
    }

    fn streams_ok(&self) -> bool {
        let (streams, stereo, channels) = (self.streams as i32, self.stereo_streams as i32, self.channels as i32);
        streams > 0 && stereo >= 0 && stereo <= streams && streams.checked_add(stereo).is_some_and(|total| total <= channels)
    }

    fn check_rate_first(&self) -> Result<(), u32> {
        if !self.sample_rate_ok() {
            Err(INVALID_SAMPLE_RATE)
        } else if !self.channels_ok() {
            Err(INVALID_CHANNEL_COUNT)
        } else {
            Ok(())
        }
    }

    fn check_channels_first(&self) -> Result<(), u32> {
        if !self.channels_ok() {
            Err(INVALID_CHANNEL_COUNT)
        } else if !self.sample_rate_ok() {
            Err(INVALID_SAMPLE_RATE)
        } else {
            Ok(())
        }
    }

    fn check_multi(&self) -> Result<(), u32> {
        if !self.sample_rate_ok() {
            Err(INVALID_SAMPLE_RATE)
        } else if !self.multi_channels_ok() {
            Err(INVALID_CHANNEL_COUNT)
        } else if !self.streams_ok() {
            Err(INVALID_NUMBER_OF_STREAMS)
        } else {
            Ok(())
        }
    }

    fn check_multi_size(&self) -> Result<(), u32> {
        if !self.multi_channels_ok() {
            Err(INVALID_CHANNEL_COUNT)
        } else if !self.sample_rate_ok() || !self.streams_ok() {
            Err(INVALID_SAMPLE_RATE)
        } else {
            Ok(())
        }
    }

    fn open(self) -> Reply {
        Reply::ok().object(Arc::new(Decoder { sample_rate: self.sample_rate, channels: self.channels }))
    }
}

pub struct HwOpus;

impl Service for HwOpus {
    fn name(&self) -> String {
        "hwopus".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let buffer = || Params::from(&req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size.min(16))).unwrap_or_default());
        let reply = match req.cmd {
            0 | 4 => {
                let params = Params::from(&req.data);
                let checked = if req.cmd == 0 { params.check_rate_first() } else { params.check_channels_first() };
                checked.map(|()| params.open())
            }
            1 | 5 | 8 => Params::from(&req.data).check_channels_first().map(|()| Reply::ok().u32(WORK_BUFFER_SIZE)),
            2 | 6 => {
                let params = buffer();
                params.check_multi().map(|()| params.open())
            }
            3 | 7 | 9 => buffer().check_multi_size().map(|()| Reply::ok().u32(WORK_BUFFER_SIZE)),
            _ => return unimplemented("hwopus", c, req),
        };
        reply.unwrap_or_else(Reply::err)
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
            0 | 2 | 4 | 5 | 6 | 7 | 8 | 9 => {
                let input = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size.min(0x10000))).unwrap_or_default();
                let out = req.out_buf(0);
                let (consumed, samples) = match frame(&input, self.sample_rate, self.channels, out.map_or(0, |b| b.size)) {
                    Ok(decoded) => decoded,
                    Err(result) => return Reply::err(result),
                };
                if let Some(out) = out {
                    p.memory.write(out.addr, &vec![0u8; samples as usize * self.channels as usize * 2]);
                }
                let reply = Reply::ok().u32(consumed).u32(samples);
                if req.cmd >= 4 { reply.u64(0) } else { reply }
            }
            1 | 3 => Reply::ok(),
            _ => unimplemented("hwopus/IHardwareOpusDecoder", c, req),
        }
    }
}

fn frame(data: &[u8], sample_rate: u32, channels: u32, out_size: u64) -> Result<(u32, u32), u32> {
    let header = data.get(..HEADER_SIZE).ok_or(INVALID_LENGTH)?;
    let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let total = HEADER_SIZE.checked_add(size).filter(|&t| t <= data.len()).ok_or(INVALID_LENGTH)?;
    let samples = samples(&data[HEADER_SIZE..total], sample_rate)?;
    if u64::from(samples) * u64::from(channels) * 2 > out_size {
        return Err(INVALID_LENGTH);
    }
    Ok((total as u32, samples))
}

fn samples(payload: &[u8], sample_rate: u32) -> Result<u32, u32> {
    let &toc = payload.first().ok_or(OPUS_BAD_ARG)?;
    let config = toc >> 3;
    let units = match config {
        0..=11 => [4, 8, 16, 24][(config & 3) as usize],
        12..=15 => [4, 8][(config & 1) as usize],
        _ => [1, 2, 4, 8][(config & 3) as usize],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(payload.get(1).ok_or(OPUS_INVALID_PACKET)? & 0x3f),
    };
    let samples = units * frames * sample_rate / 400;
    if samples * 25 > sample_rate * 3 {
        Err(OPUS_INVALID_PACKET)
    } else if samples == 0 {
        Err(OPUS_BAD_ARG)
    } else {
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(sample_rate: u32, channels: u32, streams: u32, stereo_streams: u32) -> Params {
        Params { sample_rate, channels, streams, stereo_streams }
    }

    #[test]
    fn packet_sizes() {
        assert_eq!(samples(&[0xf8], 48000), Ok(960));
        assert_eq!(samples(&[0x19], 48000), Ok(5760));
        assert_eq!(samples(&[0x93, 0x03], 24000), Ok(720));
        let mut data = vec![0, 0, 0, 2, 0xaa, 0xbb, 0xcc, 0xdd, 0xf8, 0xff, 0x11];
        assert_eq!(frame(&data, 48000, 2, 960 * 4), Ok((10, 960)));
        data.truncate(4);
        assert_eq!(frame(&data, 48000, 2, 960 * 4), Err(INVALID_LENGTH));
    }

    #[test]
    fn bad_framing_is_refused_with_the_codec_results() {
        let packet = |size: u32, payload: &[u8]| [&size.to_be_bytes()[..], &[0; 4], payload].concat();
        assert_eq!(frame(&packet(4, &[0xf8]), 48000, 2, 0x10000), Err(INVALID_LENGTH), "payload shorter than the header says");
        assert_eq!(frame(&packet(0, &[]), 48000, 2, 0x10000), Err(OPUS_BAD_ARG), "no TOC byte");
        assert_eq!(frame(&packet(1, &[0xfb]), 48000, 2, 0x10000), Err(OPUS_INVALID_PACKET), "code 3 without its count byte");
        assert_eq!(frame(&packet(2, &[0xfb, 0x00]), 48000, 2, 0x10000), Err(OPUS_BAD_ARG), "code 3 with no frames");
        assert_eq!(frame(&packet(2, &[0xfb, 0x31]), 48000, 2, 0x10000), Err(OPUS_INVALID_PACKET), "more than 120 ms");
        assert_eq!(frame(&packet(1, &[0xf8]), 48000, 2, 960 * 4 - 1), Err(INVALID_LENGTH), "output too small");
        assert_eq!(frame(&packet(1, &[0xf8]), 48000, 2, 960 * 4), Ok((9, 960)));
    }

    #[test]
    fn decoder_parameters_are_checked_like_the_console() {
        assert_eq!(INVALID_SAMPLE_RATE, 0x7d26f, "2111-1001");
        assert_eq!(INVALID_CHANNEL_COUNT, 0x7d46f, "2111-1002");
        assert_eq!(INVALID_NUMBER_OF_STREAMS, 0x7d06f, "2111-1000");
        assert_eq!(INVALID_LENGTH, 0x66f, "2111-0003");
        assert_eq!(OPUS_INVALID_PACKET, 0x10a6f, "2111-0133");
        assert_eq!(OPUS_BAD_ARG, 0x1046f, "2111-0130");

        assert_eq!(params(48000, 2, 0, 0).check_rate_first(), Ok(()));
        assert_eq!(params(44100, 2, 0, 0).check_rate_first(), Err(INVALID_SAMPLE_RATE));
        assert_eq!(params(44100, 3, 0, 0).check_rate_first(), Err(INVALID_SAMPLE_RATE), "OpenHardwareOpusDecoder (0)");
        assert_eq!(params(44100, 3, 0, 0).check_channels_first(), Err(INVALID_CHANNEL_COUNT), "OpenHardwareOpusDecoderEx (4) and the sizes");
        assert_eq!(params(48000, 0, 0, 0).check_rate_first(), Err(INVALID_CHANNEL_COUNT));

        assert_eq!(params(48000, 6, 4, 2).check_multi(), Ok(()));
        assert_eq!(params(48000, 6, 0, 0).check_multi(), Err(INVALID_NUMBER_OF_STREAMS));
        assert_eq!(params(48000, 6, 4, 3).check_multi(), Err(INVALID_NUMBER_OF_STREAMS));
        assert_eq!(params(48000, 256, 4, 2).check_multi(), Err(INVALID_CHANNEL_COUNT));
        assert_eq!(params(48000, 6, 0x8000_0000, 0x8000_0000).check_multi(), Err(INVALID_NUMBER_OF_STREAMS), "no wrap-around");
        assert_eq!(params(48000, 6, 4, u32::MAX).check_multi(), Err(INVALID_NUMBER_OF_STREAMS), "negative stereo streams");
        assert_eq!(params(48000, 6, 0, 0).check_multi_size(), Err(INVALID_SAMPLE_RATE));
    }
}
