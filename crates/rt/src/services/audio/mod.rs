mod dsp;
mod effect;
mod mixer;
pub(crate) mod opus;
mod out;
mod output;

pub use out::AudioOutManager;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::process::Process;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const HEADER_SIZE: usize = 0x40;
const MEMORY_POOL_IN: usize = 0x20;
const MEMORY_POOL_OUT: usize = 0x10;
const VOICE_OUT: usize = mixer::VOICE_OUT;
const SINK_OUT: usize = 0x20;
const PERFORMANCE_OUT: usize = 0x10;
const BEHAVIOR_OUT: usize = 0xb0;
const RENDER_INFO_OUT: usize = 0x10;
const WAVE_BUFFERS_PER_VOICE: u32 = 4;

const POOL_REQUEST_DETACH: u32 = 2;
const POOL_DETACHED: u32 = 3;
const POOL_REQUEST_ATTACH: u32 = 4;
const POOL_ATTACHED: u32 = 5;

#[derive(Clone, Copy, Debug, Default)]
struct Params {
    sample_rate: u32,
    sample_count: u32,
    mix_buffers: u32,
    sub_mixes: u32,
    voices: u32,
    sinks: u32,
    effects: u32,
    splitters: u32,
    splitter_destinations: u32,
    revision: u32,
}

impl Params {
    fn parse(d: &[u8]) -> Params {
        let u = |o: usize| {
            d.get(o..o + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        Params {
            sample_rate: u(0x00),
            sample_count: u(0x04),
            mix_buffers: u(0x08),
            sub_mixes: u(0x0c),
            voices: u(0x10),
            sinks: u(0x14),
            effects: u(0x18),
            splitters: u(0x24),
            splitter_destinations: u(0x28),
            revision: u(0x30),
        }
    }

    fn revision_number(&self) -> u32 {
        (self.revision >> 24).wrapping_sub(b'0' as u32)
    }

    fn sizes(&self) -> mixer::Sizes {
        mixer::Sizes {
            sample_rate: self.sample_rate,
            sample_count: self.sample_count as usize,
            mix_buffers: self.mix_buffers as usize,
            mixes: self.sub_mixes as usize + 1,
            voices: self.voices as usize,
            sinks: self.sinks as usize,
            splitters: self.splitters as usize,
            destinations: self.splitter_destinations as usize,
            revision: self.revision_number(),
        }
    }

    fn memory_pools(&self) -> u32 {
        self.effects + self.voices * WAVE_BUFFERS_PER_VOICE
    }

    fn fixed_output_size(&self) -> usize {
        let render_info = if self.revision_number() >= 5 {
            RENDER_INFO_OUT
        } else {
            0
        };
        HEADER_SIZE
            + self.memory_pools() as usize * MEMORY_POOL_OUT
            + self.voices as usize * VOICE_OUT
            + self.sinks as usize * SINK_OUT
            + PERFORMANCE_OUT
            + BEHAVIOR_OUT
            + render_info
    }
}

pub struct AudioRendererManager;

impl Service for AudioRendererManager {
    fn name(&self) -> String {
        "audren:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let params = Params::parse(&req.data);
                let p = c.process();
                p.log_ipc(format!("audren: open {params:?}"));
                let event = p.create_event(false);
                let state = Mutex::new(mixer::State::new(params.sizes()));
                let r = Arc::new(Renderer {
                    params,
                    event,
                    started: AtomicBool::new(false),
                    state,
                });
                start_ticks(p, r.clone());
                Reply::ok().object(r)
            }
            1 => Reply::ok().u64(0x10_0000),
            2 | 4 => Reply::ok().object(Arc::new(AudioDevice)),
            _ => unimplemented("audren:u", c, req),
        }
    }
}

struct Renderer {
    params: Params,
    event: u32,
    started: AtomicBool,
    state: Mutex<mixer::State>,
}

fn start_ticks(p: &'static Process, r: Arc<Renderer>) {
    let _ = std::thread::Builder::new()
        .name("kirikae-audio".into())
        .spawn(move || {
            let rate = r.params.sample_rate.max(1);
            let frame = r.params.sample_count.max(1);
            let period = std::time::Duration::from_secs_f64(f64::from(frame) / f64::from(rate));
            let muted = std::env::var_os("KIRIKAE_MUTE").is_some_and(|v| v != "0");
            let dump_path = std::env::var_os("KIRIKAE_AUDIO_DUMP");
            let (low, high) = (rate as usize / 25, rate as usize / 12);
            let mut samples: Vec<mixer::Frame> = Vec::new();
            let trace = std::env::var_os("KIRIKAE_TRACE_AUDIO").is_some();
            let mut rendered = 0u64;
            let mut next = std::time::Instant::now();
            let mut output: Option<output::Output> = None;
            let mut output_tried = false;
            let mut dump: Option<output::WavDump> = None;
            loop {
                next += period;
                let now = std::time::Instant::now();
                if next + period * 10 < now {
                    next = now;
                }
                std::thread::sleep(next.saturating_duration_since(now));
                if p.sched.exiting.load(Ordering::Relaxed) {
                    return;
                }
                if !r.started.load(Ordering::Relaxed) {
                    continue;
                }
                if !output_tried {
                    output_tried = true;
                    let (channels, _coefficients) = r.state.lock().unwrap().output();
                    if !muted {
                        match output::Output::open(rate, channels) {
                            Ok(o) => {
                                p.warn_once(format!(
                                    "audio: playing {} renderer channels on {}",
                                    channels, o.device
                                ));
                                output = Some(o);
                            }
                            Err(e) => {
                                p.warn_once(format!("audio: no sound output ({e})"));
                            }
                        }
                    }
                    dump = dump_path.as_ref().and_then(|path| {
                        match output::WavDump::create(path.as_ref(), rate, channels) {
                            Ok(d) => Some(d),
                            Err(e) => {
                                p.warn_once(format!(
                                    "audio: cannot write {}: {e}",
                                    path.to_string_lossy()
                                ));
                                None
                            }
                        }
                    });
                }
                let frames = match &output {
                    Some(o) if o.queued() < low => 2,
                    Some(o) if o.queued() > high => 0,
                    _ => 1,
                };
                if let Some(o) = &output
                    && trace
                    && rendered.is_multiple_of(2000)
                {
                    eprintln!(
                        "audio: {rendered} frames rendered, {} queued ({} device channels), {} device frames ran dry",
                        o.queued(),
                        o.device_channels,
                        o.dry_frames()
                    );
                }
                for _ in 0..frames {
                    rendered += 1;
                    let (channels, coefficients) = {
                        let mut state = r.state.lock().unwrap();
                        state.render(&p.memory, &mut samples);
                        state.output()
                    };
                    if let Some(o) = &output {
                        o.push(&samples, channels, coefficients);
                    }
                    if dump.as_mut().is_some_and(|d| d.write(&samples).is_err()) {
                        dump = None;
                    }
                    p.set_event(r.event, true);
                }
            }
        });
}

impl Renderer {
    fn update(&self, p: &Process, input: &[u8], out_size: usize) -> Vec<u8> {
        let u = |o: usize| {
            input
                .get(o..o + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        };
        let params = &self.params;
        let behavior_in = u(0x04);
        let pools_in = u(0x08);

        let part = |at: usize, size: usize| input.get(at..at + size).unwrap_or_default();
        let resources_at = HEADER_SIZE + behavior_in + pools_in;
        let voices_at = resources_at + u(0x10);
        let effects_at = voices_at + u(0x0c);
        let splitters_at = effects_at + u(0x14);
        let mut voice_status = vec![0; params.voices as usize * VOICE_OUT];
        let (elapsed_frames, effect_status) = {
            let mut state = self.state.lock().unwrap();
            state.note(|| {
                format!(
                    "audio: update sections {:x?}",
                    (0..9).map(|k| u(k * 4)).collect::<Vec<_>>()
                )
            });
            state.update_channel_resources(part(resources_at, u(0x10)));
            state.update_voices(part(voices_at, u(0x0c)), &mut voice_status);
            state.update_effects(part(effects_at, u(0x14)), params.effects as usize);
            let mixes_at = splitters_at
                + state.update_splitters(input.get(splitters_at..).unwrap_or_default());
            state.update_mixes(part(mixes_at, u(0x18)));
            state.update_sinks(part(mixes_at + u(0x18), u(0x1c)));
            state.note(|| {
                let voices = part(voices_at, u(0x0c));
                let used = voices
                    .chunks(0x170)
                    .filter(|v| v.get(9) == Some(&1))
                    .count();
                let touched = voices
                    .chunks(0x170)
                    .filter(|v| v.iter().any(|&b| b != 0))
                    .count();
                format!("audio: {used} voices in use, {touched} set")
            });
            (state.elapsed_frames, state.effect_status())
        };

        let mut out = vec![0u8; HEADER_SIZE];
        let section = |out: &mut Vec<u8>, bytes: Vec<u8>| -> u32 {
            out.extend_from_slice(&bytes);
            bytes.len() as u32
        };

        let mut pools = Vec::new();
        let pools_at = HEADER_SIZE + behavior_in;
        for i in 0..params.memory_pools() as usize {
            let at = pools_at + i * MEMORY_POOL_IN;
            let state = if i * MEMORY_POOL_IN < pools_in {
                u(at + 0x10) as u32
            } else {
                0
            };
            let state = match state {
                POOL_REQUEST_ATTACH => POOL_ATTACHED,
                POOL_REQUEST_DETACH => POOL_DETACHED,
                s => s,
            };
            let mut status = [0u8; MEMORY_POOL_OUT];
            status[..4].copy_from_slice(&state.to_le_bytes());
            pools.extend_from_slice(&status);
        }
        let pools_size = section(&mut out, pools);
        let voices_size = section(&mut out, voice_status);

        let fixed = params.fixed_output_size();
        let effects_bytes = if params.effects > 0 && out_size > fixed {
            let per = (out_size - fixed) / params.effects as usize;
            (per * params.effects as usize).min(out_size - fixed)
        } else {
            0
        };
        let mut effects = vec![0u8; effects_bytes];
        if params.effects > 0 && effects_bytes / params.effects as usize >= 4 {
            let per = effects_bytes / params.effects as usize;
            let revision = params.revision_number();
            for (i, status) in effect_status
                .iter()
                .enumerate()
                .take(params.effects as usize)
            {
                effect::write_out_status(&mut effects, i, per, revision, status);
            }
        }
        let effects_size = section(&mut out, effects);
        let sinks_size = section(&mut out, vec![0; params.sinks as usize * SINK_OUT]);
        let performance_size = section(&mut out, vec![0; PERFORMANCE_OUT]);
        let behavior_size = section(&mut out, vec![0; BEHAVIOR_OUT]);
        let mut render_info = vec![0; RENDER_INFO_OUT];
        render_info[..8].copy_from_slice(&elapsed_frames.to_le_bytes());
        let render_info_size = if params.revision_number() >= 5 {
            section(&mut out, render_info)
        } else {
            0
        };

        let total = out.len() as u32;
        let header = [
            params.revision,
            behavior_size,
            pools_size,
            voices_size,
            0,
            effects_size,
            0,
            sinks_size,
            performance_size,
            0,
            render_info_size,
            0,
            0,
            0,
            0,
            total,
        ];
        for (k, v) in header.iter().enumerate() {
            out[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        if out.len() != out_size {
            p.warn_once(format!(
                "audren: update output is {:#x} bytes, the game expects {out_size:#x}",
                out.len()
            ));
        }
        out
    }
}

impl Service for Renderer {
    fn name(&self) -> String {
        "audren:u/IAudioRenderer".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().u32(self.params.sample_rate),
            1 => Reply::ok().u32(self.params.sample_count),
            2 => Reply::ok().u32(self.params.mix_buffers),
            3 => Reply::ok().u32(if self.started.load(Ordering::Relaxed) {
                0
            } else {
                1
            }),
            4 | 10 => {
                let input = req
                    .in_buf(0)
                    .and_then(|b| p.memory.read(b.addr, b.size))
                    .unwrap_or_default();
                if let Some(b) = req.out_buf(0) {
                    let out = self.update(p, &input, b.size as usize);
                    p.memory
                        .write(b.addr, &out[..out.len().min(b.size as usize)]);
                }
                Reply::ok()
            }
            5 => {
                self.started.store(true, Ordering::Relaxed);
                Reply::ok()
            }
            6 => {
                self.started.store(false, Ordering::Relaxed);
                Reply::ok()
            }
            7 => Reply::ok().copy_handle(self.event),
            8 => Reply::ok(),
            9 => Reply::ok().u32(100),
            _ => unimplemented("audren:u/IAudioRenderer", c, req),
        }
    }
}

struct AudioDevice;

const DEVICE_NAME: &[u8] = b"AudioTvOutput";
const OUTPUT_DEVICE_NAME: &[u8] = b"Default";
const DEVICE_NAME_SIZE: usize = 0x100;

fn output_device_slots(size: Option<u64>) -> u32 {
    u32::from(size.is_some_and(|size| size >= DEVICE_NAME_SIZE as u64))
}

fn output_device_reply(size: Option<u64>) -> Reply {
    Reply::ok().u32(output_device_slots(size))
}

impl Service for AudioDevice {
    fn name(&self) -> String {
        "audren:u/IAudioDevice".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        let write_name = |req: &Request, name: &[u8]| {
            if let Some(b) = req.out_buf(0) {
                let mut buffer = [0u8; DEVICE_NAME_SIZE];
                buffer[..name.len()].copy_from_slice(name);
                p.memory
                    .write(b.addr, &buffer[..(b.size as usize).min(DEVICE_NAME_SIZE)]);
            }
        };
        match req.cmd {
            0 | 6 => {
                write_name(req, DEVICE_NAME);
                Reply::ok().u32(req.out_buf(0).map_or(0, |_| 1))
            }
            1 | 7 => Reply::ok(),
            2 | 8 => Reply::ok().u32(1.0f32.to_bits()),
            3 | 10 | 13 => {
                write_name(req, DEVICE_NAME);
                Reply::ok()
            }
            4 | 11 | 12 => Reply::ok().copy_handle(p.create_event(false)),
            5 => Reply::ok().u32(2),
            14 => {
                let size = req.out_buf(0).map(|b| b.size);
                if output_device_slots(size) > 0 {
                    write_name(req, OUTPUT_DEVICE_NAME);
                }
                output_device_reply(size)
            }
            _ => unimplemented("audren:u/IAudioDevice", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_one_default_output_device() {
        assert_eq!(OUTPUT_DEVICE_NAME, b"Default");
        assert!(OUTPUT_DEVICE_NAME.len() < DEVICE_NAME_SIZE);
    }

    #[test]
    fn output_device_reply_is_a_count() {
        assert_eq!(
            output_device_reply(Some(DEVICE_NAME_SIZE as u64)).data,
            1u32.to_le_bytes()
        );
        assert_eq!(
            output_device_reply(Some(DEVICE_NAME_SIZE as u64 * 2)).data,
            1u32.to_le_bytes()
        );
        assert_eq!(
            output_device_reply(Some(DEVICE_NAME_SIZE as u64 - 1)).data,
            0u32.to_le_bytes()
        );
        assert_eq!(output_device_reply(None).data, 0u32.to_le_bytes());
    }
}
