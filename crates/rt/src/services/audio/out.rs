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
                let out = Arc::new(AudioOut { event: p.create_event(false), channels, state: Mutex::new(State::default()) });
                start_playback(p, out.clone());
                Reply::ok().u32(SAMPLE_RATE).u32(channels).u32(PCM_INT16).u32(STOPPED).object(out)
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

#[derive(Clone, Copy)]
struct Buffer {
    tag: u64,
    samples: u64,
    size: u64,
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
            0 => Reply::ok().u32(if s.started { STARTED } else { STOPPED }),
            1 => {
                s.started = true;
                Reply::ok()
            }
            2 => {
                s.started = false;
                Reply::ok()
            }
            3 | 7 => {
                let Some(info) = req.in_buf(0).and_then(|b| p.memory.read(b.addr, b.size.min(BUFFER_INFO))) else { return Reply::ok() };
                let word = |i: usize| info.get(i * 8..i * 8 + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
                s.queued.push_back(Buffer { tag: req.u64(0), samples: word(1) + word(4), size: word(3) });
                Reply::ok()
            }
            4 => Reply::ok().copy_handle(self.event),
            5 | 8 => {
                let Some(out) = req.out_buf(0) else { return Reply::ok().u32(0) };
                let n = ((out.size / 8) as usize).min(s.released.len());
                let tags: Vec<u8> = s.released.drain(..n).flat_map(|t| t.to_le_bytes()).collect();
                p.memory.write(out.addr, &tags);
                Reply::ok().u32(n as u32)
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
    let _ = std::thread::Builder::new().name("kirikae-audio-out-service".into()).spawn(move || {
        out.state.lock().unwrap().volume = 1.0;
        let output = if std::env::var_os("KIRIKAE_MUTE").is_some_and(|v| v != "0") { None } else { output::Output::open(SAMPLE_RATE).ok() };
        let channels = out.channels as usize;
        let mut next = Instant::now();
        let mut frames: Vec<[f32; 2]> = Vec::new();
        loop {
            if p.sched.exiting.load(Ordering::Relaxed) {
                return;
            }
            let buffer = {
                let s = out.state.lock().unwrap();
                if s.started { s.queued.front().copied() } else { None }
            };
            let Some(b) = buffer else {
                std::thread::sleep(Duration::from_millis(2));
                next = Instant::now();
                continue;
            };
            let pcm = p.memory.read(b.samples, b.size).unwrap_or_default();
            let volume = out.state.lock().unwrap().volume;
            frames.clear();
            frames.extend(pcm.chunks_exact(2 * channels).map(|f| {
                let s = |i: usize| i16::from_le_bytes([f[2 * i], f[2 * i + 1]]) as f32 / 32768.0 * volume;
                [s(0), s(1)]
            }));
            if let Some(o) = &output {
                o.push(&frames);
            }
            next += Duration::from_secs_f64(frames.len() as f64 / f64::from(SAMPLE_RATE));
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
            let mut s = out.state.lock().unwrap();
            if s.queued.front().is_some_and(|q| q.tag == b.tag) {
                s.queued.pop_front();
                s.released.push_back(b.tag);
                s.played += frames.len() as u64;
            }
            drop(s);
            p.set_event(out.event, true);
        }
    });
}
