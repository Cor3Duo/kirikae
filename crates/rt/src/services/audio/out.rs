use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::output;
use crate::cpu::Cpu;
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const DEVICE_NAME: &[u8] = b"DeviceOut";
const NAME_SIZE: u64 = 0x100;
const SAMPLE_RATE: u32 = 48_000;
const PCM_INT16: u32 = 2;
const STARTED: u32 = 0;
const STOPPED: u32 = 1;
const BUFFER_INFO: u64 = 0x28;

fn write_name(c: &Cpu, req: &Request) {
    if let Some(out) = req.out_buf(0) {
        let mut name = vec![0u8; out.size.min(NAME_SIZE) as usize];
        let n = DEVICE_NAME.len().min(name.len().saturating_sub(1));
        name[..n].copy_from_slice(&DEVICE_NAME[..n]);
        c.process().memory.write(out.addr, &name);
    }
}

pub struct AudioOutManager;

impl Service for AudioOutManager {
    fn name(&self) -> String {
        "audout:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 | 2 => {
                write_name(c, req);
                Reply::ok().u32(1)
            }
            1 | 3 => {
                write_name(c, req);
                let channels = if req.u32(4) & 0xffff > 2 { 6 } else { 2 };
                let p = c.process();
                let out = Arc::new(AudioOut {
                    event: p.create_event(false),
                    channels,
                    state: Mutex::new(State::default()),
                });
                start_playback(p, out.clone());
                Reply::ok()
                    .u32(SAMPLE_RATE)
                    .u32(channels)
                    .u32(PCM_INT16)
                    .u32(STOPPED)
                    .object(out)
            }
            _ => unimplemented("audout:u", c, req),
        }
    }
}

#[derive(Default)]
struct State {
    started: bool,
    queued: VecDeque<Buffer>,
    released: VecDeque<u64>,
    played: u64,
    volume: f32,
}

impl State {
    /// `AudioOutState_Started` is 0 and `AudioOutState_Stopped` is 1 on Horizon.
    fn state(&self) -> u32 {
        if self.started { STARTED } else { STOPPED }
    }
}

#[derive(Clone, Copy)]
struct Buffer {
    tag: u64,
    samples: u64,
    size: u64,
}

impl Buffer {
    /// `AudioOutBuffer` is the 0x28-byte block `AppendAudioOutBuffer` sends: the
    /// PCM pointer at +0x08, its capacity at +0x10, the data size at +0x18 and
    /// the offset inside the buffer at +0x20.
    fn parse(tag: u64, info: &[u8]) -> Buffer {
        let word = |i: usize| {
            info.get(i * 8..i * 8 + 8)
                .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
        };
        Buffer {
            tag,
            samples: word(1) + word(4),
            size: word(3),
        }
    }
}

/// The tag the client gets back is the address of the `AudioOutBuffer` struct it
/// appended, because `GetAudioOutBufferDataPointer` dereferences it (+0x08 holds
/// the PCM pointer). The input buffer is that struct, so its address is the
/// contract; the raw parameter some clients also send is only redundant.
fn append_tag(req: &Request) -> u64 {
    req.in_buf(0).map_or(0, |b| b.addr)
}

/// Build the `GetReleasedAudioOutBuffers` reply: the count goes in the reply
/// body and the out buffer receives the 8-byte tags. Every entry of the out
/// buffer is written, zero-filled when fewer buffers were released, because the
/// SDK returns its first entry unconditionally and the guest would dereference
/// whatever stale stack value it found there.
///
/// A tag whose struct is no longer readable is dropped instead of handed to the
/// guest: the service only reports buffers it can still see in guest memory.
fn released_buffers(
    released: &mut VecDeque<u64>,
    out_size: u64,
    readable: impl Fn(u64) -> bool,
) -> (u32, Vec<u8>) {
    let capacity = (out_size / 8) as usize;
    let mut tags = Vec::with_capacity(out_size as usize);
    let mut count = 0;
    while count < capacity {
        let Some(tag) = released.pop_front() else {
            break;
        };
        if !readable(tag) {
            continue;
        }
        tags.extend_from_slice(&tag.to_le_bytes());
        count += 1;
    }
    tags.resize(out_size as usize, 0);
    (count as u32, tags)
}

struct AudioOut {
    event: u32,
    channels: u32,
    state: Mutex<State>,
}

impl Service for AudioOut {
    fn name(&self) -> String {
        "audout:u/IAudioOut".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let mut s = self.state.lock().unwrap();
        match req.cmd {
            0 => Reply::ok().u32(s.state()),
            1 => {
                s.started = true;
                Reply::ok()
            }
            2 => {
                s.started = false;
                Reply::ok()
            }
            3 | 7 => {
                let tag = append_tag(req);
                let Some(info) = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size.min(BUFFER_INFO)))
                else {
                    return Reply::ok();
                };
                if tag == 0 {
                    return Reply::ok();
                }
                s.queued.push_back(Buffer::parse(tag, &info));
                Reply::ok()
            }
            4 => Reply::ok().copy_handle(self.event),
            5 | 8 => {
                let Some(out) = req.out_buf(0) else {
                    return Reply::ok().u32(0);
                };
                let (count, tags) = released_buffers(&mut s.released, out.size, |tag| {
                    p.memory.read(tag, BUFFER_INFO).is_some()
                });
                p.memory.write(out.addr, &tags);
                Reply::ok().u32(count)
            }
            6 => {
                let tag = req.u64(0);
                Reply::ok().u32(s.queued.iter().any(|b| b.tag == tag) as u32)
            }
            9 => Reply::ok().u32(s.queued.len() as u32),
            10 => Reply::ok().u64(s.played),
            11 => {
                let flushed: Vec<u64> = s.queued.drain(..).map(|b| b.tag).collect();
                let any = !flushed.is_empty();
                s.released.extend(flushed);
                drop(s);
                if any {
                    p.set_event(self.event, true);
                }
                Reply::ok().u32(any as u32)
            }
            12 => {
                s.volume = f32::from_bits(req.u32(0));
                Reply::ok()
            }
            13 => Reply::ok().u32(s.volume.to_bits()),
            _ => unimplemented("audout:u/IAudioOut", c, req),
        }
    }
}

fn start_playback(p: &'static Process, out: Arc<AudioOut>) {
    let _ = std::thread::Builder::new()
        .name("kirikae-audio-out-service".into())
        .spawn(move || {
            out.state.lock().unwrap().volume = 1.0;
            let channels = out.channels as usize;
            let output = if std::env::var_os("KIRIKAE_MUTE").is_some_and(|v| v != "0") {
                None
            } else {
                output::Output::open(SAMPLE_RATE, channels).ok()
            };
            let mut next = Instant::now();
            let mut frames: Vec<output::Frame> = Vec::new();
            let stats = std::env::var_os("KIRIKAE_STATS").is_some();
            let mut next_stats = Instant::now() + Duration::from_secs(5);
            loop {
                if p.sched.exiting.load(Ordering::Relaxed) {
                    return;
                }
                let buffer = {
                    let s = out.state.lock().unwrap();
                    if s.started {
                        s.queued.front().copied()
                    } else {
                        None
                    }
                };
                let Some(b) = buffer else {
                    let position = match &output {
                        Some(o) => o.played_frames(),
                        None => out.state.lock().unwrap().played,
                    };
                    crate::host1x::clock::audio_position(
                        position,
                        SAMPLE_RATE,
                        false,
                        crate::host1x::clock::AudioSource::AudOut,
                    );
                    std::thread::sleep(Duration::from_millis(2));
                    next = Instant::now();
                    continue;
                };
                let pcm = p.memory.read(b.samples, b.size).unwrap_or_default();
                let volume = out.state.lock().unwrap().volume;
                let width = channels.clamp(1, output::MAX_CHANNELS);
                frames.clear();
                frames.extend(pcm.chunks_exact(2 * width).map(|f| {
                    let mut frame = [0.0f32; output::MAX_CHANNELS];
                    for (c, sample) in frame.iter_mut().enumerate().take(width) {
                        *sample =
                            i16::from_le_bytes([f[2 * c], f[2 * c + 1]]) as f32 / 32768.0 * volume;
                    }
                    frame
                }));
                if let Some(o) = &output {
                    o.push(&frames, width, output::DEFAULT_DOWNMIX);
                }
                next += Duration::from_secs_f64(frames.len() as f64 / f64::from(SAMPLE_RATE));
                let produced = out.state.lock().unwrap().played + frames.len() as u64;
                let position = output.as_ref().map_or(produced, |o| o.played_frames());
                crate::host1x::clock::audio_position(
                    position,
                    SAMPLE_RATE,
                    output.is_some(),
                    crate::host1x::clock::AudioSource::AudOut,
                );
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
                let mut s = out.state.lock().unwrap();
                if s.queued.front().is_some_and(|q| q.tag == b.tag) {
                    s.queued.pop_front();
                    s.released.push_back(b.tag);
                    s.played += frames.len() as u64;
                }
                drop(s);
                p.set_event(out.event, true);
                if stats && Instant::now() >= next_stats {
                    next_stats = Instant::now() + Duration::from_secs(5);
                    let (played, queued) = {
                        let s = out.state.lock().unwrap();
                        (s.played, output.as_ref().map_or(0, |o| o.queued()))
                    };
                    eprintln!(
                        "audio: audout {played} frames played, {queued} frames queued for the device ({:.1} ms lead)",
                        queued as f32 * 1000.0 / f64::from(SAMPLE_RATE) as f32
                    );
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::Buf;

    const BUFFER_SIZE: usize = 0x28;

    /// Flat guest memory for the contract tests: the same read/write shape the
    /// runtime gives the service, without a whole process.
    struct Ram(Mutex<Vec<u8>>);

    impl Ram {
        fn new(size: usize) -> Ram {
            Ram(Mutex::new(vec![0u8; size]))
        }

        fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
            let memory = self.0.lock().unwrap();
            let at = usize::try_from(addr).ok()?;
            memory.get(at..at + len as usize).map(<[u8]>::to_vec)
        }

        fn write(&self, addr: u64, bytes: &[u8]) -> bool {
            let mut memory = self.0.lock().unwrap();
            let Some(at) = usize::try_from(addr).ok() else {
                return false;
            };
            let Some(dst) = memory.get_mut(at..at + bytes.len()) else {
                return false;
            };
            dst.copy_from_slice(bytes);
            true
        }

        fn u64(&self, addr: u64) -> u64 {
            u64::from_le_bytes(self.read(addr, 8).unwrap().try_into().unwrap())
        }
    }

    fn audio_out_buffer(
        data: u64,
        buffer_size: u64,
        data_size: u64,
        data_offset: u64,
    ) -> [u8; BUFFER_SIZE] {
        let mut bytes = [0u8; BUFFER_SIZE];
        bytes[0x08..0x10].copy_from_slice(&data.to_le_bytes());
        bytes[0x10..0x18].copy_from_slice(&buffer_size.to_le_bytes());
        bytes[0x18..0x20].copy_from_slice(&data_size.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&data_offset.to_le_bytes());
        bytes
    }

    #[test]
    fn append_uses_the_buffer_struct_address_as_the_tag() {
        let req = Request {
            cmd: 7,
            x: vec![Buf {
                addr: 0x0010_4078_5480,
                size: BUFFER_INFO,
            }],
            ..Request::default()
        };
        assert_eq!(append_tag(&req), 0x0010_4078_5480);
        assert_eq!(append_tag(&Request::default()), 0);
    }

    #[test]
    fn the_audio_out_state_word_matches_horizon() {
        assert_eq!((STARTED, STOPPED), (0, 1));
        assert_eq!(State::default().state(), STOPPED);
        let mut state = State {
            started: true,
            ..State::default()
        };
        assert_eq!(state.state(), STARTED);
        state.started = false;
        assert_eq!(state.state(), STOPPED);
    }

    #[test]
    fn an_empty_release_writes_zeroes_to_every_entry() {
        // The SDK returns entry 0 of the array without looking at the count, so
        // leaving the entry untouched hands the guest a stale stack value.
        let mut released = VecDeque::new();
        let (count, bytes) = released_buffers(&mut released, 0x40, |_| true);
        assert_eq!(count, 0);
        assert_eq!(bytes, vec![0u8; 0x40]);
    }

    #[test]
    fn a_released_buffer_comes_back_as_its_struct_pointer() {
        let mut released = VecDeque::from([0x0010_4078_5480u64]);
        let (count, bytes) = released_buffers(&mut released, 0x40, |_| true);
        assert_eq!(count, 1);
        assert_eq!(&bytes[..8], &0x0010_4078_5480u64.to_le_bytes());
        assert!(
            bytes[8..].iter().all(|&b| b == 0),
            "unused entries are zero, not stale stack"
        );
    }

    #[test]
    fn an_unreadable_struct_is_not_handed_back() {
        let mut released = VecDeque::from([0x1000u64, 0x2000u64]);
        let (count, bytes) = released_buffers(&mut released, 0x10, |tag| tag == 0x2000);
        assert_eq!(count, 1);
        assert_eq!(&bytes[..8], &0x2000u64.to_le_bytes());
        assert_eq!(&bytes[8..], &[0u8; 8]);
    }

    #[test]
    fn the_reply_stops_at_the_out_buffer_capacity() {
        let mut released = VecDeque::from([1u64, 2, 3]);
        let (count, bytes) = released_buffers(&mut released, 0x10, |_| true);
        assert_eq!(count, 2);
        assert_eq!(released, VecDeque::from([3u64]));
        assert_eq!(bytes.len(), 0x10);
    }

    #[test]
    fn a_synthetic_guest_reads_the_data_pointer_through_the_released_pointer() {
        let guest = Ram::new(0x1_0000);
        let (object, pcm) = (0x1000u64, 0x8000u64);
        assert!(guest.write(object, &audio_out_buffer(pcm, 0x1000, 0x800, 0x10)));
        let info = guest.read(object, BUFFER_INFO).unwrap();

        let mut state = State {
            started: true,
            ..State::default()
        };
        state.queued.push_back(Buffer::parse(object, &info));
        let queued = state.queued.pop_front().unwrap();
        assert_eq!(
            (queued.samples, queued.size),
            (pcm + 0x10, 0x800),
            "the played range starts at buffer + data_offset and spans data_size"
        );
        state.released.push_back(queued.tag);

        let (count, reply) = released_buffers(&mut state.released, 0x40, |tag| {
            guest.read(tag, BUFFER_INFO).is_some()
        });
        assert_eq!(count, 1);
        let out = 0x2000;
        assert!(guest.write(out, &reply));

        // nn::audio::GetReleasedAudioOutBuffer returns entry 0 as an
        // AudioOutBuffer*, then GetAudioOutBufferDataPointer reads +0x08.
        let returned = guest.u64(out);
        assert_eq!(returned, object);
        assert_eq!(guest.u64(returned + 8), pcm);
    }
}
