use super::dsp::{self, AdpcmContext, HISTORY};
use super::effect::{self, Descriptor, Processor, Specific};

pub const WAVE_BUFFERS: usize = 4;
pub const VOICE_IN: usize = 0x170;
pub const VOICE_OUT: usize = 0x10;
pub const CHANNEL_RESOURCE_IN: usize = 0x70;
pub const MIX_IN: usize = 0x930;
pub const SINK_IN: usize = 0x140;
const WAVE_BUFFER_IN: usize = 0x38;
const MAX_CHANNELS: usize = 6;
const MIX_BUFFERS_MAX: usize = 24;
const FINAL_MIX: usize = 0;
const UNUSED_MIX: i32 = i32::MAX;
const UNUSED_SPLITTER: u32 = u32::MAX;
pub use super::output::{DEFAULT_DOWNMIX, Frame, MAX_CHANNELS as MAX_OUTPUT_CHANNELS};

const PCM_INT16: u8 = 2;
const PCM_FLOAT: u8 = 5;
const ADPCM: u8 = 6;

const PLAYED_COUNT_RESET_ON_LOOP: u16 = 1;
const SKIP_SAMPLE_RATE_CONVERSION: u16 = 2;

const SINK_DEVICE: u8 = 1;
const SPLITTER_HEADER_MAGIC: u32 = u32::from_le_bytes(*b"SNDH");
const SPLITTER_MAGIC: u32 = u32::from_le_bytes(*b"SNDI");
const DESTINATION_MAGIC: u32 = u32::from_le_bytes(*b"SNDD");

/// Standard 5.1 to stereo fold-down (Ryujinx `DefaultSurroundToStereoCoefficients`):
/// front, centre, LFE, back.
pub trait Mem {
    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>>;
    fn write(&self, addr: u64, bytes: &[u8]) -> bool;
}

impl Mem for crate::mem::GuestMemory {
    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        crate::mem::GuestMemory::read(self, addr, len)
    }

    fn write(&self, addr: u64, bytes: &[u8]) -> bool {
        crate::mem::GuestMemory::write(self, addr, bytes)
    }
}

fn u16_at(d: &[u8], o: usize) -> u16 {
    d.get(o..o + 2)
        .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
}
fn u32_at(d: &[u8], o: usize) -> u32 {
    d.get(o..o + 4)
        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
}
fn i32_at(d: &[u8], o: usize) -> i32 {
    u32_at(d, o) as i32
}
fn u64_at(d: &[u8], o: usize) -> u64 {
    d.get(o..o + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
}
fn f32_at(d: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(d, o))
}
fn byte_at(d: &[u8], o: usize) -> u8 {
    d.get(o).copied().unwrap_or(0)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Play {
    Started,
    #[default]
    Stopped,
    Stopping,
    Paused,
}

#[derive(Clone, Copy, Debug, Default)]
struct WaveBuffer {
    addr: u64,
    size: u64,
    start: u32,
    end: u32,
    looping: bool,
    end_of_stream: bool,
    loop_count: i32,
    context: u64,
    loop_start: u32,
    loop_end: u32,
    pending: bool,
}

impl WaveBuffer {
    fn parse(w: &[u8]) -> WaveBuffer {
        WaveBuffer {
            addr: u64_at(w, 0x00),
            size: u64_at(w, 0x08),
            start: u32_at(w, 0x10),
            end: u32_at(w, 0x14),
            looping: byte_at(w, 0x18) != 0,
            end_of_stream: byte_at(w, 0x19) != 0,
            loop_count: i32_at(w, 0x1c),
            context: u64_at(w, 0x20),
            loop_start: u32_at(w, 0x30),
            loop_end: u32_at(w, 0x34),
            pending: true,
        }
    }

    fn offsets_valid(&self, format: u8) -> bool {
        let fits = |bytes: u64| bytes <= self.size;
        let (start, end) = (u64::from(self.start), u64::from(self.end));
        match format {
            PCM_INT16 => fits(start * 2) && fits(end * 2),
            PCM_FLOAT => fits(start * 4) && fits(end * 4),
            ADPCM => fits(dsp::adpcm_size(start)) && fits(dsp::adpcm_size(end)),
            _ => true,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Voice {
    in_use: bool,
    fresh: bool,
    play: Play,
    format: u8,
    sample_rate: u32,
    channels: usize,
    pitch: f32,
    volume: f32,
    previous_volume: f32,
    queued: u32,
    base_index: usize,
    flush: u32,
    decoding: u16,
    mix_id: i32,
    splitter_id: u32,
    channel_ids: [usize; MAX_CHANNELS],
    data_source: u64,
    buffers: [WaveBuffer; WAVE_BUFFERS],

    valid: [bool; WAVE_BUFFERS],
    index: usize,
    consumed: u32,
    offset: u32,
    played: u64,
    loops: i32,
    fraction: f32,
    adpcm: AdpcmContext,
    history: [[f32; HISTORY]; MAX_CHANNELS],
}

impl Voice {
    fn has_destination(&self) -> bool {
        self.mix_id != UNUSED_MIX || self.splitter_id != UNUSED_SPLITTER
    }

    fn set_play(&mut self, requested: u8) {
        self.play = match requested {
            0 => Play::Started,
            1 if self.play == Play::Stopped => Play::Stopped,
            1 => Play::Stopping,
            _ => Play::Paused,
        };
    }

    fn end_buffer(&mut self) {
        let end_of_stream = self.buffers[self.index].end_of_stream;
        self.valid[self.index] = false;
        self.index = (self.index + 1) % WAVE_BUFFERS;
        self.loops = 0;
        self.consumed = self.consumed.wrapping_add(1);
        if end_of_stream {
            self.played = 0;
        }
    }

    fn prepare(&mut self) -> bool {
        let mut slot = self.base_index;
        for _ in 0..std::mem::take(&mut self.flush) {
            self.buffers[slot].pending = false;
            self.valid[slot] = false;
            self.index = (self.index + 1) % WAVE_BUFFERS;
            self.consumed = self.consumed.wrapping_add(1);
            slot = (slot + 1) % WAVE_BUFFERS;
        }
        match self.play {
            Play::Started => {
                for (b, valid) in self.buffers.iter_mut().zip(&mut self.valid) {
                    if b.pending {
                        b.pending = false;
                        *valid = true;
                    }
                }
                self.valid.iter().any(|&v| v)
            }
            Play::Stopping => {
                for b in 0..WAVE_BUFFERS {
                    self.buffers[b].pending = false;
                    if self.valid[b] {
                        self.index = (self.index + 1) % WAVE_BUFFERS;
                        self.consumed = self.consumed.wrapping_add(1);
                    }
                    self.valid[b] = false;
                }
                self.offset = 0;
                self.played = 0;
                self.loops = 0;
                self.fraction = 0.0;
                self.adpcm = AdpcmContext::default();
                self.history = Default::default();
                self.play = Play::Stopped;
                false
            }
            Play::Stopped | Play::Paused => false,
        }
    }

    fn fetch(&mut self, mem: &dyn Mem, count: usize, src: &mut [Vec<f32>], buffers_v2: bool) {
        let mut coefficients = [0i16; 16];
        if self.format == ADPCM
            && let Some(d) = mem.read(self.data_source, 32)
        {
            for (k, c) in coefficients.iter_mut().enumerate() {
                *c = u16_at(&d, k * 2) as i16;
            }
        }
        let mut done = 0;
        for _ in 0..64 {
            if done >= count || !self.valid[self.index] {
                break;
            }
            let wb = self.buffers[self.index];
            if self.offset == 0
                && self.format == ADPCM
                && wb.context != 0
                && let Some(d) = mem.read(wb.context, 6)
            {
                self.adpcm = AdpcmContext::parse(&d);
            }
            let in_loop = buffers_v2
                && self.loops > 0
                && wb.loop_start != 0
                && wb.loop_end != 0
                && wb.loop_start <= wb.loop_end;
            let (start, end) = if in_loop {
                (wb.loop_start, wb.loop_end)
            } else {
                (wb.start, wb.end)
            };
            let total = end.saturating_sub(start);
            let want = (count - done).min(total.saturating_sub(self.offset) as usize);
            let Some(got) = self.decode(
                mem,
                &wb,
                u64::from(start + self.offset),
                want,
                &coefficients,
                src,
            ) else {
                self.offset = 0;
                self.end_buffer();
                continue;
            };
            done += got;
            self.offset += got as u32;
            self.played += got as u64;
            if self.offset >= total || got == 0 {
                self.offset = 0;
                if wb.looping {
                    self.loops += 1;
                    let limit = if buffers_v2 { wb.loop_count } else { -1 };
                    if limit >= 0 && (got == 0 || self.loops > limit) {
                        self.end_buffer();
                    }
                    if got == 0 {
                        break;
                    }
                    if self.decoding & PLAYED_COUNT_RESET_ON_LOOP != 0 {
                        self.played = 0;
                    }
                } else {
                    self.end_buffer();
                }
            }
        }
        for s in src.iter_mut().take(self.channels) {
            s.resize(HISTORY + count, 0.0);
        }
    }

    fn decode(
        &mut self,
        mem: &dyn Mem,
        wb: &WaveBuffer,
        pos: u64,
        want: usize,
        coefficients: &[i16; 16],
        src: &mut [Vec<f32>],
    ) -> Option<usize> {
        if want == 0 {
            return Some(0);
        }
        let channels = self.channels;
        match self.format {
            PCM_INT16 | PCM_FLOAT => {
                let width = if self.format == PCM_INT16 { 2 } else { 4 };
                let frame = width * channels as u64;
                let n = want.min((wb.size / frame).saturating_sub(pos) as usize);
                if n == 0 {
                    return None;
                }
                let bytes = mem.read(wb.addr + pos * frame, n as u64 * frame)?;
                for (c, s) in src.iter_mut().enumerate().take(channels) {
                    let w = width as usize;
                    let stride = frame as usize;
                    if width == 2 {
                        s.extend(
                            bytes
                                .chunks_exact(stride)
                                .map(|f| i16::from_le_bytes([f[c * w], f[c * w + 1]]) as f32),
                        );
                    } else {
                        s.extend(
                            bytes
                                .chunks_exact(stride)
                                .map(|f| f32_at(f, c * w) * 32768.0),
                        );
                    }
                }
                Some(n)
            }
            ADPCM => {
                let first = dsp::adpcm_nibble(pos) / 2;
                let last = dsp::adpcm_size(pos + want as u64).min(wb.size);
                if last <= first {
                    return None;
                }
                let bytes = mem.read(wb.addr + first, last - first)?;
                let n = dsp::decode_adpcm(
                    &bytes,
                    pos,
                    want,
                    coefficients,
                    &mut self.adpcm,
                    &mut src[0],
                );
                for s in src.iter_mut().take(channels).skip(1) {
                    s.resize(s.len() + n, 0.0);
                }
                (n > 0).then_some(n)
            }
            _ => {
                for s in src.iter_mut().take(channels) {
                    s.resize(s.len() + want, 0.0);
                }
                Some(want)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ChannelResource {
    mix: [f32; MIX_BUFFERS_MAX],
    previous: [f32; MIX_BUFFERS_MAX],
}

#[derive(Clone, Debug)]
struct Mix {
    used: bool,
    volume: f32,
    buffer_count: usize,
    destination: i32,
    splitter: u32,
    volumes: Vec<f32>,
    offset: usize,
}

impl Default for Mix {
    fn default() -> Mix {
        Mix {
            used: false,
            volume: 1.0,
            buffer_count: 0,
            destination: UNUSED_MIX,
            splitter: UNUSED_SPLITTER,
            volumes: vec![0.0; MIX_BUFFERS_MAX * MIX_BUFFERS_MAX],
            offset: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Sink {
    used: bool,
    device: bool,
    inputs: Vec<usize>,
    downmix: Option<[f32; 4]>,
}

#[derive(Clone, Debug, Default)]
struct Splitter {
    destinations: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Destination {
    used: bool,
    mix: i32,
    volumes: [f32; MIX_BUFFERS_MAX],
    previous: [f32; MIX_BUFFERS_MAX],
}

impl Default for Destination {
    fn default() -> Destination {
        Destination {
            used: false,
            mix: UNUSED_MIX,
            volumes: [0.0; MIX_BUFFERS_MAX],
            previous: [0.0; MIX_BUFFERS_MAX],
        }
    }
}

impl Destination {
    fn configured(&self) -> bool {
        self.used && self.mix != UNUSED_MIX && self.mix >= 0
    }
}

struct EffectSlot {
    kind: u8,
    enabled: bool,
    mix: i32,
    order: u32,
    specific: Specific,
    processor: Processor,
}

impl Default for EffectSlot {
    fn default() -> EffectSlot {
        EffectSlot {
            kind: 0,
            enabled: false,
            mix: UNUSED_MIX,
            order: 0,
            specific: Specific::Invalid,
            processor: Processor::None,
        }
    }
}

#[derive(Default)]
struct Effects {
    slots: Vec<EffectSlot>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Sizes {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub mix_buffers: usize,
    pub mixes: usize,
    pub voices: usize,
    pub sinks: usize,
    pub splitters: usize,
    pub destinations: usize,
    pub revision: u32,
}

pub struct State {
    sizes: Sizes,
    voices: Vec<Voice>,
    channels: Vec<ChannelResource>,
    mixes: Vec<Mix>,
    sinks: Vec<Sink>,
    splitters: Vec<Splitter>,
    destinations: Vec<Destination>,
    effects: Effects,
    buffers: Vec<f32>,
    effect_in: Vec<Vec<f32>>,
    effect_out: Vec<Vec<f32>>,
    mix_scratch: Vec<f32>,
    src: Vec<Vec<f32>>,
    voice_out: Vec<Vec<f32>>,
    pub elapsed_frames: u64,
    /// Renderer frames the host device callback has consumed, when a host
    /// output exists. The presentation clock follows this monotonic media
    /// position instead of the frames just mixed, so the queue ahead of the
    /// device is not counted as already heard. `None` falls back to the mix
    /// rate (muted, headless or no device).
    output_played: Option<u64>,
    trace: bool,
}

impl State {
    pub fn new(sizes: Sizes) -> State {
        let trace = std::env::var_os("KIRIKAE_TRACE_AUDIO").is_some();
        State {
            sizes,
            voices: vec![Voice::default(); sizes.voices],
            channels: vec![ChannelResource::default(); sizes.voices],
            mixes: vec![Mix::default(); sizes.mixes.max(1)],
            sinks: vec![Sink::default(); sizes.sinks],
            splitters: vec![Splitter::default(); sizes.splitters],
            destinations: vec![Destination::default(); sizes.destinations],
            effects: Effects::default(),
            buffers: Vec::new(),
            effect_in: vec![vec![0.0; sizes.sample_count]; effect::MAX_CHANNELS],
            effect_out: vec![vec![0.0; sizes.sample_count]; effect::MAX_CHANNELS],
            mix_scratch: vec![0.0; sizes.sample_count],
            src: vec![Vec::new(); MAX_CHANNELS],
            voice_out: vec![vec![0.0; sizes.sample_count]; MAX_CHANNELS],
            elapsed_frames: 0,
            output_played: None,
            trace,
        }
    }

    fn buffers_v2(&self) -> bool {
        self.sizes.revision >= 8
    }

    pub fn update_channel_resources(&mut self, d: &[u8]) {
        for (i, c) in self.channels.iter_mut().enumerate() {
            let Some(e) = d.get(i * CHANNEL_RESOURCE_IN..(i + 1) * CHANNEL_RESOURCE_IN) else {
                break;
            };
            for (j, m) in c.mix.iter_mut().enumerate() {
                *m = f32_at(e, 4 + j * 4);
            }
        }
    }

    pub fn update_voices(&mut self, d: &[u8], out: &mut [u8]) {
        let revision = self.sizes.revision;
        let channel_count = self.channels.len();
        for (i, voice) in self.voices.iter_mut().enumerate() {
            let was_in_use = std::mem::take(&mut voice.in_use);
            let Some(p) = d.get(i * VOICE_IN..(i + 1) * VOICE_IN) else {
                continue;
            };
            if byte_at(p, 0x09) == 0 {
                continue;
            }
            let is_new = byte_at(p, 0x08) != 0;
            if is_new {
                *voice = Voice {
                    fresh: true,
                    ..Voice::default()
                };
            }
            voice.in_use = true;
            voice.set_play(byte_at(p, 0x0a));
            voice.format = byte_at(p, 0x0b);
            voice.sample_rate = u32_at(p, 0x0c);
            voice.channels = (u32_at(p, 0x18) as usize).clamp(1, MAX_CHANNELS);
            voice.pitch = f32_at(p, 0x1c);
            voice.volume = f32_at(p, 0x20);
            voice.queued = u32_at(p, 0x3c);
            voice.base_index = u32_at(p, 0x40) as usize % WAVE_BUFFERS;
            voice.data_source = u64_at(p, 0x48);
            voice.mix_id = i32_at(p, 0x58);
            voice.splitter_id = if revision >= 2 {
                u32_at(p, 0x5c)
            } else {
                UNUSED_SPLITTER
            };
            for (c, id) in voice.channel_ids.iter_mut().enumerate() {
                *id =
                    (i32_at(p, 0x140 + c * 4).max(0) as usize).min(channel_count.saturating_sub(1));
            }
            if revision >= 5 {
                voice.flush += u32::from(byte_at(p, 0x159));
                voice.decoding = u16_at(p, 0x15c);
            }
            for b in 0..WAVE_BUFFERS {
                let w = &p[0x60 + b * WAVE_BUFFER_IN..0x60 + (b + 1) * WAVE_BUFFER_IN];
                if byte_at(w, 0x1a) == 0 {
                    let wb = WaveBuffer::parse(w);
                    if wb.offsets_valid(voice.format) {
                        voice.buffers[b] = wb;
                    }
                }
            }
            if self.trace && (is_new || !was_in_use) {
                eprintln!(
                    "audio: voice {i} new {is_new} play {:?} format {} {} Hz x{} pitch {} volume {} mix {} splitter {:#x} buffers {}",
                    voice.play,
                    voice.format,
                    voice.sample_rate,
                    voice.channels,
                    voice.pitch,
                    voice.volume,
                    voice.mix_id,
                    voice.splitter_id,
                    voice.queued
                );
                let buffers: Vec<String> = (0..WAVE_BUFFERS)
                    .map(|b| {
                        format!(
                            "{:#x}/{}",
                            u64_at(p, 0x60 + b * WAVE_BUFFER_IN),
                            byte_at(p, 0x60 + b * WAVE_BUFFER_IN + 0x1a)
                        )
                    })
                    .collect();
                eprintln!(
                    "audio:     id {} node {:#x} channels {:?} base {} buffers {buffers:?}",
                    i32_at(p, 0),
                    u32_at(p, 4),
                    &voice.channel_ids[..voice.channels],
                    voice.base_index
                );
            }
            let status = &mut out[i * VOICE_OUT..(i + 1) * VOICE_OUT];
            status.fill(0);
            if !voice.fresh {
                status[0..8].copy_from_slice(&voice.played.to_le_bytes());
                status[8..12].copy_from_slice(&voice.consumed.to_le_bytes());
            }
        }
    }

    pub fn update_splitters(&mut self, d: &[u8]) -> usize {
        if self.splitters.is_empty()
            || self.destinations.is_empty()
            || u32_at(d, 0) != SPLITTER_HEADER_MAGIC
        {
            return 0;
        }
        let splitter_count = u32_at(d, 4) as usize;
        let destination_count = u32_at(d, 8) as usize;
        let mut at = 0x20;
        for _ in 0..splitter_count {
            if u32_at(d, at) != SPLITTER_MAGIC {
                break;
            }
            let id = i32_at(d, at + 4);
            let n = (u32_at(d, at + 0xc) as usize).min(self.destinations.len());
            if let Some(s) = usize::try_from(id)
                .ok()
                .and_then(|id| self.splitters.get_mut(id))
            {
                s.destinations = (0..n)
                    .map(|k| i32_at(d, at + 0x10 + k * 4).max(0) as usize)
                    .collect();
            }
            let next = at + 0x10 + n * 4;
            at = if [SPLITTER_MAGIC, DESTINATION_MAGIC].contains(&u32_at(d, next)) {
                next
            } else {
                next + 0xc
            };
        }
        let stride = (0x70..=0x100)
            .step_by(0x10)
            .find(|&s| destination_count < 2 || u32_at(d, at + s) == DESTINATION_MAGIC)
            .unwrap_or(0x70);
        for _ in 0..destination_count {
            if u32_at(d, at) != DESTINATION_MAGIC {
                break;
            }
            let id = i32_at(d, at + 4);
            if let Some(dst) = usize::try_from(id)
                .ok()
                .and_then(|id| self.destinations.get_mut(id))
            {
                let used = byte_at(d, at + 0x6c) != 0;
                for (j, v) in dst.volumes.iter_mut().enumerate() {
                    *v = f32_at(d, at + 8 + j * 4);
                }
                if used && !dst.used {
                    dst.previous = dst.volumes;
                }
                dst.mix = i32_at(d, at + 0x68);
                dst.used = used;
            }
            at += stride;
        }
        at.next_multiple_of(0x10)
    }

    pub fn update_mixes(&mut self, d: &[u8]) {
        let dirty_only = self.sizes.revision >= 7;
        let (count, base) = if dirty_only {
            (u32_at(d, 4) as usize, 0x20)
        } else {
            (d.len() / MIX_IN, 0)
        };
        for i in 0..count {
            let Some(p) = d.get(base + i * MIX_IN..base + (i + 1) * MIX_IN) else {
                break;
            };
            let id = if dirty_only {
                i32_at(p, 0x10).max(0) as usize
            } else {
                i
            };
            let Some(mix) = self.mixes.get_mut(id) else {
                continue;
            };
            mix.used = byte_at(p, 0x0c) != 0;
            if mix.used {
                mix.volume = f32_at(p, 0x00);
                mix.buffer_count = (u32_at(p, 0x08) as usize).min(MIX_BUFFERS_MAX);
                for (k, v) in mix.volumes.iter_mut().enumerate() {
                    *v = f32_at(p, 0x24 + k * 4);
                }
                mix.destination = i32_at(p, 0x924);
                mix.splitter = u32_at(p, 0x928);
            }
        }
        let mut offset = 0;
        for mix in &mut self.mixes {
            mix.offset = offset;
            offset += mix.buffer_count;
        }
        if self.trace {
            for (id, m) in self.mixes.iter().enumerate().filter(|(_, m)| m.used) {
                let line = format!(
                    "audio: mix {id} buffers {}@{} volume {} -> mix {} splitter {:#x}",
                    m.buffer_count, m.offset, m.volume, m.destination, m.splitter
                );
                trace_once(line);
                let sends: Vec<String> = (0..m.buffer_count)
                    .flat_map(|i| (0..MIX_BUFFERS_MAX).map(move |j| (i, j)))
                    .filter(|&(i, j)| m.volumes[i * MIX_BUFFERS_MAX + j] != 0.0)
                    .map(|(i, j)| format!("{i}->{j}:{}", m.volumes[i * MIX_BUFFERS_MAX + j]))
                    .collect();
                trace_once(format!("audio:     mix {id} sends {sends:?}"));
            }
        }
    }

    pub fn note(&self, line: impl FnOnce() -> String) {
        if self.trace {
            trace_once(line());
        }
    }

    pub fn update_effects(&mut self, d: &[u8], count: usize) {
        if count == 0 || d.is_empty() {
            return;
        }
        let stride = if d.len() >= count { d.len() / count } else { 0 };
        if stride < 0x20 {
            self.note(|| {
                format!(
                    "audio: effects section is {} bytes for {count} effects, needs at least {:#x} each",
                    d.len(),
                    effect::EFFECT_SIZE
                )
            });
            return;
        }
        if self.effects.slots.len() != count {
            self.effects.slots = (0..count).map(|_| EffectSlot::default()).collect();
        }
        let sample_rate = self.sizes.sample_rate;
        for i in 0..count {
            let Some(e) = d.get(i * stride..(i + 1) * stride) else {
                break;
            };
            let (desc, specific) = effect::parse(e);
            let slot = &mut self.effects.slots[i];
            let rebuild =
                slot.kind != desc.kind || desc.is_new || !slot.processor.matches(&specific);
            slot.kind = desc.kind;
            slot.enabled = desc.enabled;
            slot.mix = desc.mix;
            slot.order = desc.order;
            slot.specific = specific;
            if rebuild {
                slot.processor = Processor::new(&desc, &specific, sample_rate);
            } else {
                slot.processor.reconfigure(&specific, sample_rate);
            }
            if self.trace && desc.kind != 0 {
                let line = describe_effect(i, &desc, &specific);
                trace_once(line);
            }
        }
    }

    pub fn effect_status(&self) -> Vec<effect::OutStatus> {
        self.effects
            .slots
            .iter()
            .map(|s| s.processor.out_status(s.enabled, s.kind))
            .collect()
    }

    pub fn update_sinks(&mut self, d: &[u8]) {
        for (i, sink) in self.sinks.iter_mut().enumerate() {
            let Some(p) = d.get(i * SINK_IN..(i + 1) * SINK_IN) else {
                break;
            };
            sink.used = byte_at(p, 1) != 0;
            sink.device = byte_at(p, 0) == SINK_DEVICE;
            if sink.device {
                let n = (u32_at(p, 0x120) as usize).min(MAX_OUTPUT_CHANNELS);
                sink.inputs = p[0x124..0x124 + n]
                    .iter()
                    .map(|&b| usize::from(b))
                    .collect();
                sink.downmix = (byte_at(p, 0x12b) != 0)
                    .then(|| std::array::from_fn(|k| f32_at(p, 0x12c + k * 4)));
                if self.trace {
                    trace_once(format!(
                        "audio: sink {i} device inputs {:?} downmix {:?}",
                        sink.inputs, sink.downmix
                    ));
                }
            }
        }
    }

    /// Report the frames the host device callback has consumed, or `None`
    /// when no host device is playing. The value is monotonic on the device
    /// side, so the presentation clock never sees the audio position move
    /// backwards when the queue depth changes.
    pub fn set_output_played(&mut self, frames: Option<u64>) {
        self.output_played = frames;
    }

    /// Media frames the presentation clock should follow: the device position
    /// when there is one, otherwise the mixer's own count (the historical
    /// behaviour for muted or headless runs). Split out so the arithmetic is
    /// testable without a clock.
    fn presentation_frames(&self, block: usize) -> u64 {
        self.output_played
            .unwrap_or_else(|| self.elapsed_frames.saturating_mul(block as u64))
    }

    pub fn render(&mut self, mem: &dyn Mem, out: &mut Vec<Frame>) {
        let n = self.sizes.sample_count;
        self.elapsed_frames += 1;
        crate::host1x::clock::audio_position(
            self.presentation_frames(n),
            self.sizes.sample_rate,
            true,
            crate::host1x::clock::AudioSource::AudRen,
        );
        if self.trace && self.elapsed_frames.is_multiple_of(400) {
            self.trace_voices(mem);
        }
        self.buffers.clear();
        self.buffers.resize(self.sizes.mix_buffers * n, 0.0);
        for v in 0..self.voices.len() {
            self.render_voice(v, mem);
        }
        self.render_mixes(mem);
        self.write_sink(out);
    }

    /// Channels and fold-down coefficients requested by the device sink.
    pub fn output(&self) -> (usize, [f32; 4]) {
        let sink = self.sinks.iter().find(|s| s.used && s.device);
        let channels = sink
            .map(|s| s.inputs.len().clamp(1, MAX_OUTPUT_CHANNELS))
            .unwrap_or(2);
        (
            channels,
            sink.and_then(|s| s.downmix).unwrap_or(DEFAULT_DOWNMIX),
        )
    }

    fn trace_voices(&self, mem: &dyn Mem) {
        for (i, v) in self.voices.iter().enumerate().filter(|(_, v)| v.in_use) {
            let b = &v.buffers[v.index];
            let coefficients = mem.read(v.data_source, 32).map(|d| {
                (0..16)
                    .map(|k| u16_at(&d, k * 2) as i16)
                    .collect::<Vec<_>>()
            });
            eprintln!(
                "audio: [{}] voice {i} {:?} volume {} queued {} valid {:?} consumed {} played {} buffer {:#x}+{:#x} {}..{} loop {} eos {}",
                self.elapsed_frames,
                v.play,
                v.volume,
                v.queued,
                v.valid,
                v.consumed,
                v.played,
                b.addr,
                b.size,
                b.start,
                b.end,
                b.looping,
                b.end_of_stream
            );
            eprintln!(
                "audio:     sends {:?} coefficients {coefficients:?} adpcm {:?}",
                &self.channels[v.channel_ids[0]].mix[..6],
                v.adpcm
            );
        }
    }

    fn render_voice(&mut self, v: usize, mem: &dyn Mem) {
        let n = self.sizes.sample_count;
        let rate = self.sizes.sample_rate.max(1);
        let buffers_v2 = self.buffers_v2();
        let voice = &mut self.voices[v];
        if !voice.in_use || voice.queued == 0 {
            return;
        }
        if std::mem::take(&mut voice.fresh) {
            voice.previous_volume = voice.volume;
            for &c in &voice.channel_ids[..voice.channels] {
                self.channels[c].previous = self.channels[c].mix;
            }
        }
        if !voice.prepare() {
            return;
        }

        let ratio = if voice.decoding & SKIP_SAMPLE_RATE_CONVERSION != 0 {
            1.0
        } else {
            voice.sample_rate as f32 / rate as f32 * voice.pitch
        };
        let ratio = if ratio.is_finite() {
            ratio.clamp(0.0, 16.0)
        } else {
            0.0
        };
        let needed = dsp::needed(voice.fraction, ratio, n);
        let channels = voice.channels;
        for (c, s) in self.src.iter_mut().enumerate().take(channels) {
            s.clear();
            s.extend_from_slice(&voice.history[c]);
        }
        voice.fetch(mem, needed, &mut self.src, buffers_v2);
        let mut fraction = voice.fraction;
        for c in 0..channels {
            let src = &self.src[c];
            fraction = dsp::resample(src, &mut self.voice_out[c][..n], ratio, voice.fraction);
            voice.history[c].copy_from_slice(&src[src.len() - HISTORY..]);
        }
        voice.fraction = fraction;

        if !voice.has_destination() {
            return;
        }
        let (from, to) = (voice.previous_volume, voice.volume);
        voice.previous_volume = to;
        for out in &mut self.voice_out[..channels] {
            ramp(&mut out[..n], from, to);
        }

        let voice = &self.voices[v];
        if voice.mix_id != UNUSED_MIX {
            let Some(mix) = usize::try_from(voice.mix_id)
                .ok()
                .and_then(|m| self.mixes.get(m))
            else {
                return;
            };
            let (offset, count) = (mix.offset, mix.buffer_count);
            for c in 0..channels {
                let resource = &mut self.channels[voice.channel_ids[c]];
                for j in 0..count {
                    mix_into(
                        &mut self.buffers,
                        offset + j,
                        n,
                        &self.voice_out[c][..n],
                        resource.previous[j],
                        resource.mix[j],
                    );
                }
                resource.previous = resource.mix;
            }
        } else if let Some(splitter) = self.splitters.get(voice.splitter_id as usize) {
            for c in 0..channels {
                for &d in splitter.destinations.iter().skip(c).step_by(channels) {
                    let Some(dst) = self.destinations.get_mut(d) else {
                        continue;
                    };
                    let Some(mix) = usize::try_from(dst.mix)
                        .ok()
                        .and_then(|m| self.mixes.get(m))
                    else {
                        continue;
                    };
                    if !dst.configured() {
                        continue;
                    }
                    for j in 0..mix.buffer_count {
                        mix_into(
                            &mut self.buffers,
                            mix.offset + j,
                            n,
                            &self.voice_out[c][..n],
                            dst.previous[j],
                            dst.volumes[j],
                        );
                    }
                    dst.previous = dst.volumes;
                }
            }
        }
    }

    fn distance(&self, id: usize) -> Option<usize> {
        let mut at = id;
        for hops in 0..=self.mixes.len() {
            if at == FINAL_MIX {
                return Some(hops);
            }
            let mix = &self.mixes[at];
            let next = if mix.destination != UNUSED_MIX {
                mix.destination
            } else {
                let splitter = self.splitters.get(mix.splitter as usize)?;
                splitter
                    .destinations
                    .iter()
                    .filter_map(|&d| self.destinations.get(d))
                    .find(|d| d.configured())?
                    .mix
            };
            at = usize::try_from(next)
                .ok()
                .filter(|&m| m < self.mixes.len())?;
        }
        None
    }

    fn apply_mix_effects(&mut self, index: usize, mem: &dyn Mem) {
        let n = self.sizes.sample_count;
        if n == 0 || self.effects.slots.is_empty() {
            return;
        }
        let Some(mix) = self.mixes.get(index) else {
            return;
        };
        let (offset, count) = (mix.offset, mix.buffer_count);
        if count == 0 {
            return;
        }
        let sink_rate = self.sizes.sample_rate;
        let State {
            effects,
            buffers,
            effect_in,
            effect_out,
            mix_scratch,
            ..
        } = self;
        let slots = effects.slots.as_mut_slice();
        let mix_id = index as i32;
        for order in 0..slots.len() {
            for slot in slots.iter_mut() {
                if slot.kind == 0 || slot.mix != mix_id || slot.order as usize != order {
                    continue;
                }
                match &slot.specific {
                    Specific::BufferMix(p) => {
                        if !slot.enabled || !p.routable() {
                            continue;
                        }
                        for k in 0..p.mixes {
                            let (source, target) = (p.input[k], p.output[k]);
                            let volume = p.volumes[k];
                            if volume == 0.0
                                || usize::from(source) >= count
                                || usize::from(target) >= count
                            {
                                continue;
                            }
                            let at = (offset + usize::from(source)) * n;
                            let Some(src) = buffers.get(at..at + n) else {
                                continue;
                            };
                            mix_scratch[..n].copy_from_slice(src);
                            let at = (offset + usize::from(target)) * n;
                            if let Some(dst) = buffers.get_mut(at..at + n) {
                                for (d, s) in dst.iter_mut().zip(&mix_scratch[..n]) {
                                    *d += *s * volume;
                                }
                            }
                        }
                        continue;
                    }
                    Specific::Aux(aux) | Specific::Capture(aux) => {
                        let channels = aux.route.channels.min(effect::MAX_CHANNELS);
                        if channels == 0 {
                            continue;
                        }
                        let sources = aux.route.input;
                        let sinks = aux.route.output;
                        if (0..channels).any(|c| {
                            usize::from(sources[c]) >= count || usize::from(sinks[c]) >= count
                        }) {
                            continue;
                        }
                        for c in 0..channels {
                            let at = (offset + usize::from(sources[c])) * n;
                            if let Some(src) = buffers.get(at..at + n) {
                                effect_in[c][..n].copy_from_slice(src);
                            }
                        }
                        effect::process_aux(
                            aux,
                            mem,
                            &effect_in[..channels],
                            &mut effect_out[..channels],
                            n,
                            slot.enabled,
                            effect::AuxClock::of(aux, sink_rate),
                        );
                        for c in 0..channels {
                            let at = (offset + usize::from(sinks[c])) * n;
                            if let Some(dst) = buffers.get_mut(at..at + n) {
                                dst.copy_from_slice(&effect_out[c][..n]);
                            }
                        }
                        continue;
                    }
                    _ => {}
                }
                let Some(route) = effect::route(&slot.specific) else {
                    continue;
                };
                let channels = route.channels.min(effect::MAX_CHANNELS);
                if channels == 0 {
                    continue;
                }
                if (0..channels).any(|c| {
                    usize::from(route.input[c]) >= count || usize::from(route.output[c]) >= count
                }) {
                    continue;
                }
                if !slot.enabled {
                    // A disabled effect routes its input to its output changed by
                    // nothing, as the DSP command generator does.
                    copy_route(buffers, offset, count, n, &route);
                    continue;
                }
                for (c, input) in effect_in.iter_mut().enumerate().take(channels) {
                    let at = (offset + usize::from(route.input[c])) * n;
                    if let Some(src) = buffers.get(at..at + n) {
                        input[..n].copy_from_slice(src);
                    }
                }
                match &mut slot.processor {
                    Processor::Delay(delay) => {
                        delay.process(&effect_in[..channels], &mut effect_out[..channels], n)
                    }
                    Processor::Reverb(reverb) => {
                        reverb.process(&effect_in[..channels], &mut effect_out[..channels], n)
                    }
                    Processor::Reverb3d(reverb) => {
                        reverb.process(&effect_in[..channels], &mut effect_out[..channels], n)
                    }
                    Processor::Biquad { state, .. } => {
                        for c in 0..channels {
                            effect_out[c][..n].copy_from_slice(&effect_in[c][..n]);
                            state[c].process(&mut effect_out[c][..n]);
                        }
                    }
                    Processor::Limiter(limiter) => {
                        limiter.process(&effect_in[..channels], &mut effect_out[..channels], n)
                    }
                    Processor::Compressor(compressor) => {
                        compressor.process(&effect_in[..channels], &mut effect_out[..channels], n)
                    }
                    Processor::None => {
                        // No usable processor: keep the route dry instead of
                        // muting the channel.
                        copy_route(buffers, offset, count, n, &route);
                        continue;
                    }
                }
                for (c, output) in effect_out.iter().enumerate().take(channels) {
                    let at = (offset + usize::from(route.output[c])) * n;
                    if let Some(dst) = buffers.get_mut(at..at + n) {
                        dst.copy_from_slice(&output[..n]);
                    }
                }
            }
        }
    }

    fn render_mixes(&mut self, mem: &dyn Mem) {
        let n = self.sizes.sample_count;
        let mut order: Vec<(usize, usize)> = (1..self.mixes.len())
            .filter(|&m| self.mixes[m].used)
            .filter_map(|m| self.distance(m).map(|d| (d, m)))
            .collect();
        order.sort_by(|a, b| b.cmp(a));
        let mut src = vec![0.0; n];
        for (_, m) in order {
            self.apply_mix_effects(m, mem);
            let mix = &self.mixes[m];
            if mix.destination != UNUSED_MIX {
                let Some(dst) = self.mixes.get(mix.destination as usize) else {
                    continue;
                };
                for i in 0..mix.buffer_count {
                    let at = (mix.offset + i) * n;
                    let Some(s) = self.buffers.get(at..at + n) else {
                        continue;
                    };
                    src.copy_from_slice(s);
                    for j in 0..dst.buffer_count {
                        let volume = mix.volume * mix.volumes[i * MIX_BUFFERS_MAX + j];
                        mix_into(&mut self.buffers, dst.offset + j, n, &src, volume, volume);
                    }
                }
            } else if let Some(splitter) = self.splitters.get(mix.splitter as usize) {
                for (k, &d) in splitter.destinations.iter().enumerate() {
                    let Some(dst) = self.destinations.get(d).filter(|d| d.configured()) else {
                        continue;
                    };
                    let Some(target) = self.mixes.get(dst.mix as usize) else {
                        continue;
                    };
                    let at = (mix.offset + k % mix.buffer_count.max(1)) * n;
                    let Some(s) = self.buffers.get(at..at + n) else {
                        continue;
                    };
                    src.copy_from_slice(s);
                    for j in 0..target.buffer_count {
                        let volume = mix.volume * dst.volumes[j];
                        mix_into(
                            &mut self.buffers,
                            target.offset + j,
                            n,
                            &src,
                            volume,
                            volume,
                        );
                    }
                }
            }
        }
        self.apply_mix_effects(FINAL_MIX, mem);
        let fin = &self.mixes[FINAL_MIX];
        if fin.volume != 1.0 {
            let range =
                fin.offset * n..((fin.offset + fin.buffer_count) * n).min(self.buffers.len());
            for s in &mut self.buffers[range] {
                *s *= fin.volume;
            }
        }
    }

    fn write_sink(&self, out: &mut Vec<Frame>) {
        let n = self.sizes.sample_count;
        out.clear();
        let fin = &self.mixes[FINAL_MIX];
        let Some(sink) = self.sinks.iter().find(|s| s.used && s.device) else {
            out.resize(n, [0.0; MAX_OUTPUT_CHANNELS]);
            return;
        };
        let count = sink.inputs.len().min(MAX_OUTPUT_CHANNELS);
        let channel = |k: usize| -> Option<&[f32]> {
            let b = fin.offset + *sink.inputs.get(k)?;
            self.buffers.get(b * n..(b + 1) * n)
        };
        let silent = vec![0.0; n];
        let ch: Vec<&[f32]> = (0..count).map(|k| channel(k).unwrap_or(&silent)).collect();
        let scale = 1.0 / 32768.0;
        for i in 0..n {
            let mut frame = [0.0f32; MAX_OUTPUT_CHANNELS];
            for (k, source) in ch.iter().enumerate().take(count) {
                frame[k] = (source[i] * scale).clamp(-1.0, 1.0);
            }
            out.push(frame);
        }
    }
}

fn copy_route(buffers: &mut [f32], offset: usize, count: usize, n: usize, route: &effect::Route) {
    let channels = route.channels.min(effect::MAX_CHANNELS);
    for c in 0..channels {
        let from = usize::from(route.input[c]);
        let to = usize::from(route.output[c]);
        if from >= count || to >= count || from == to {
            continue;
        }
        let src = (offset + from) * n..(offset + from + 1) * n;
        if src.end <= buffers.len() {
            buffers.copy_within(src, (offset + to) * n);
        }
    }
}

fn ramp(s: &mut [f32], from: f32, to: f32) {
    if from == 1.0 && to == 1.0 {
        return;
    }
    let step = (to - from) / s.len().max(1) as f32;
    let mut v = from;
    for x in s {
        *x *= v;
        v += step;
    }
}

fn mix_into(buffers: &mut [f32], buffer: usize, n: usize, src: &[f32], from: f32, to: f32) {
    if from == 0.0 && to == 0.0 {
        return;
    }
    let Some(dst) = buffers.get_mut(buffer * n..(buffer + 1) * n) else {
        return;
    };
    let step = (to - from) / n.max(1) as f32;
    let mut v = from;
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s * v;
        v += step;
    }
}

fn trace_once(line: String) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
        std::sync::Mutex::new(None);
    let mut seen = SEEN.lock().unwrap();
    if seen
        .get_or_insert_with(Default::default)
        .insert(line.clone())
    {
        eprintln!("{line}");
    }
}

fn describe_effect(index: usize, desc: &Descriptor, specific: &Specific) -> String {
    let detail = match specific {
        Specific::BufferMix(p) => {
            format!("buffer mix, {} routes", p.mixes)
        }
        Specific::Aux(p) | Specific::Capture(p) => format!(
            "{} {} channels {:.1} kHz storage {} send {:#x} return {:#x}",
            specific.label(),
            p.route.channels,
            p.sample_rate as f32 / 1000.0,
            p.storage,
            p.send_info,
            p.return_info
        ),
        Specific::Biquad(p) => format!(
            "biquad {} channels b {:?} a {:?}",
            p.route.channels, p.b, p.a
        ),
        Specific::Delay(p) => format!(
            "delay {}..{}ms feedback {:.3} spread {:.3} lowpass {:.3} wet {:.3} dry {:.3} {} channels",
            p.delay_time,
            p.delay_time_max,
            p.feedback_gain,
            p.channel_spread,
            p.low_pass,
            p.out_gain,
            p.dry_gain,
            p.route.channels
        ),
        Specific::Reverb(p) => format!(
            "reverb channels {} early {}/{} late {}/{} pre {:.2}ms decay {:.0}ms hf {:.3} colour {:.3} gains {:.2}/{:.2}/{:.2}",
            p.channels,
            p.early_mode,
            p.early_gain,
            p.late_mode,
            p.late_gain,
            p.pre_delay,
            p.decay_time,
            p.hf_decay_ratio,
            p.coloration,
            p.reverb_gain,
            p.out_gain,
            p.dry_gain
        ),
        Specific::Reverb3d(p) => format!(
            "3d reverb {} channels {:.1} kHz decay {:.0}ms hf {:.2} room {:.0}mB reflect {:.0}mB reverb {:.0}mB diffusion {:.0}% density {:.0}% dry {:.2}",
            p.route.channels,
            p.sample_rate as f32 / 1000.0,
            p.decay_time,
            p.hf_decay_ratio,
            p.room_gain,
            p.reflections_gain,
            p.reverb_gain,
            p.diffusion,
            p.density,
            p.dry_gain
        ),
        Specific::Limiter(p) => format!(
            "limiter {} channels {:.1} kHz threshold {:.3} attack {:.3} release {:.3} gains {:.2}/{:.2} delay {} samples statistics {}",
            p.route.channels,
            p.sample_rate as f32 / 1000.0,
            p.threshold,
            p.attack,
            p.release,
            p.input_gain,
            p.output_gain,
            p.delay,
            p.statistics_enabled
        ),
        Specific::Compressor(p) => format!(
            "compressor {} channels {:.1} kHz threshold {:.1} dB ratio {:.2} attack {}us release {}us gains {:.2} lin/{:.1} dB makeup {}",
            p.route.channels,
            p.sample_rate as f32 / 1000.0,
            p.threshold,
            p.ratio,
            p.attack_us,
            p.release_us,
            p.input_gain,
            p.output_gain,
            p.makeup
        ),
        Specific::Unimplemented(_) => format!("{}, passed through", specific.label()),
        Specific::Invalid => "invalid".to_string(),
    };
    format!(
        "audio: effect {index} type {} enabled {} mix {} order {} ({detail})",
        desc.kind, desc.enabled, desc.mix, desc.order
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct Fake(HashMap<u64, Vec<u8>>);

    impl Mem for Fake {
        fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
            self.0.iter().find_map(|(&base, bytes)| {
                let start = addr.checked_sub(base)? as usize;
                bytes.get(start..start + len as usize).map(<[u8]>::to_vec)
            })
        }

        fn write(&self, _addr: u64, _bytes: &[u8]) -> bool {
            false
        }
    }

    /// A flat, writable guest memory used by the auxiliary bus test.
    struct Ram(Mutex<Vec<u8>>);

    impl Ram {
        fn new(size: usize) -> Ram {
            Ram(Mutex::new(vec![0u8; size]))
        }
    }

    impl Mem for Ram {
        fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
            let memory = self.0.lock().unwrap();
            let at = addr as usize;
            memory.get(at..at + len as usize).map(<[u8]>::to_vec)
        }

        fn write(&self, addr: u64, bytes: &[u8]) -> bool {
            let mut memory = self.0.lock().unwrap();
            let at = addr as usize;
            let Some(dst) = memory.get_mut(at..at + bytes.len()) else {
                return false;
            };
            dst.copy_from_slice(bytes);
            true
        }
    }

    const BUFFER: u64 = 0x10_0000;

    fn sizes() -> Sizes {
        Sizes {
            sample_rate: 48000,
            sample_count: 240,
            mix_buffers: 2,
            mixes: 1,
            voices: 1,
            sinks: 1,
            revision: 11,
            ..Sizes::default()
        }
    }

    fn put_u32(d: &mut [u8], o: usize, v: u32) {
        d[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn graph(s: &mut State) {
        graph_with_buffers(s, 2);
    }

    fn graph_with_buffers(s: &mut State, buffers: usize) {
        let mut mixes = vec![0u8; 0x20 + MIX_IN];
        put_u32(&mut mixes, 4, 1);
        let m = &mut mixes[0x20..];
        put_u32(m, 0, 1.0f32.to_bits());
        put_u32(m, 8, buffers as u32);
        m[0x0c] = 1;
        put_u32(m, 0x924, UNUSED_MIX as u32);
        put_u32(m, 0x928, UNUSED_SPLITTER);
        s.update_mixes(&mixes);
        let mut sink = vec![0u8; SINK_IN];
        sink[0] = SINK_DEVICE;
        sink[1] = 1;
        put_u32(&mut sink, 0x120, 2);
        sink[0x124] = 0;
        sink[0x125] = 1;
        s.update_sinks(&sink);
        let mut resources = vec![0u8; CHANNEL_RESOURCE_IN];
        put_u32(&mut resources, 4, 0.5f32.to_bits());
        resources[0x64] = 1;
        s.update_channel_resources(&resources);
    }

    fn voice(new: bool, sent: bool, samples: u32, play: u8) -> Vec<u8> {
        let mut v = vec![0u8; VOICE_IN];
        v[0x08] = new as u8;
        v[0x09] = 1;
        v[0x0a] = play;
        v[0x0b] = PCM_INT16;
        put_u32(&mut v, 0x0c, 48000);
        put_u32(&mut v, 0x18, 1);
        put_u32(&mut v, 0x1c, 1.0f32.to_bits());
        put_u32(&mut v, 0x20, 1.0f32.to_bits());
        put_u32(&mut v, 0x3c, 1);
        put_u32(&mut v, 0x58, 0);
        put_u32(&mut v, 0x5c, UNUSED_SPLITTER);
        let w = &mut v[0x60..0x60 + WAVE_BUFFER_IN];
        w[0..8].copy_from_slice(&BUFFER.to_le_bytes());
        w[8..16].copy_from_slice(&(u64::from(samples) * 2).to_le_bytes());
        put_u32(w, 0x14, samples);
        w[0x1a] = sent as u8;
        for b in 1..WAVE_BUFFERS {
            v[0x60 + b * WAVE_BUFFER_IN + 0x1a] = 1;
        }
        v
    }

    fn status(out: &[u8]) -> (u64, u32) {
        (u64_at(out, 0), u32_at(out, 8))
    }

    fn effect_block(kind: u8, enabled: bool) -> Vec<u8> {
        let mut e = vec![0u8; effect::EFFECT_SIZE];
        e[0] = kind;
        e[1] = 1;
        e[2] = enabled as u8;
        put_u32(&mut e, 4, FINAL_MIX as u32);
        put_u32(&mut e, 0x18, 0);
        e
    }

    fn reverb_effect(enabled: bool) -> Vec<u8> {
        let mut e = effect_block(effect::KIND_REVERB, enabled);
        e[0x20] = 0;
        e[0x21] = 1;
        e[0x26] = 0;
        e[0x27] = 1;
        e[0x2C..0x2E].copy_from_slice(&2u16.to_le_bytes());
        e[0x2E..0x30].copy_from_slice(&2u16.to_le_bytes());
        put_u32(&mut e, 0x30, 48_000 << 14);
        for (k, v) in [
            0,
            (0.7 * 16384.0) as i32,
            0,
            0,
            16384,
            (500.0 * 16384.0) as i32,
            (0.5 * 16384.0) as i32,
            (0.1 * 16384.0) as i32,
            16384,
            16384,
            16384,
        ]
        .into_iter()
        .enumerate()
        {
            put_u32(&mut e, 0x34 + k * 4, v as u32);
        }
        e
    }

    fn buffer_mix_effect() -> Vec<u8> {
        let mut e = effect_block(effect::KIND_BUFFER_MIX, true);
        // Route buffer 0 into buffer 1 at half volume.
        e[0x20] = 0;
        e[0x20 + 24] = 1;
        put_u32(&mut e, 0x20 + 48, 0.5f32.to_bits());
        put_u32(&mut e, 0x20 + 48 + 24 * 4, 1);
        e
    }

    fn aux_effect(send: u64, ret: u64) -> Vec<u8> {
        let mut e = effect_block(effect::KIND_AUX_BUS, true);
        e[0x20] = 0;
        e[0x20 + 24] = 1;
        put_u32(&mut e, 0x20 + 0x30, 1);
        put_u32(&mut e, 0x20 + 0x34, 48_000);
        put_u32(&mut e, 0x20 + 0x38, 1024);
        e[0x20 + 0x40..0x20 + 0x48].copy_from_slice(&send.to_le_bytes());
        e[0x20 + 0x50..0x20 + 0x58].copy_from_slice(&ret.to_le_bytes());
        e
    }

    fn compressor_effect() -> Vec<u8> {
        let mut e = effect_block(effect::KIND_COMPRESSOR, true);
        e[0x20] = 0;
        e[0x20 + 6] = 1;
        e[0x20 + 0x0E..0x20 + 0x10].copy_from_slice(&1u16.to_le_bytes());
        put_u32(&mut e, 0x20 + 0x10, 48_000);
        e[0x20 + 0x14..0x20 + 0x18].copy_from_slice(&(-20.0f32).to_le_bytes());
        e[0x20 + 0x18..0x20 + 0x1C].copy_from_slice(&4.0f32.to_le_bytes());
        e[0x20 + 0x24..0x20 + 0x28].copy_from_slice(&1.0f32.to_le_bytes());
        e[0x20 + 0x28..0x20 + 0x2C].copy_from_slice(&1.0f32.to_le_bytes());
        e[0x20 + 0x2C..0x20 + 0x30].copy_from_slice(&1.0f32.to_le_bytes());
        e
    }

    fn limiter_effect(statistics: bool) -> Vec<u8> {
        let mut e = effect_block(effect::KIND_LIMITER, true);
        e[0x20] = 0;
        e[0x20 + 6] = 1;
        e[0x20 + 0x0E..0x20 + 0x10].copy_from_slice(&1u16.to_le_bytes());
        put_u32(&mut e, 0x20 + 0x10, 48_000);
        e[0x20 + 0x24..0x20 + 0x28].copy_from_slice(&0.5f32.to_le_bytes());
        e[0x20 + 0x28..0x20 + 0x2C].copy_from_slice(&0.01f32.to_le_bytes());
        e[0x20 + 0x2C..0x20 + 0x30].copy_from_slice(&0.5f32.to_le_bytes());
        e[0x20 + 0x30..0x20 + 0x34].copy_from_slice(&1.0f32.to_le_bytes());
        e[0x20 + 0x34..0x20 + 0x38].copy_from_slice(&1.0f32.to_le_bytes());
        put_u32(&mut e, 0x20 + 0x38, 1);
        put_u32(&mut e, 0x20 + 0x3C, 1);
        e[0x20 + 0x41] = statistics as u8;
        e
    }

    #[test]
    fn compressor_effect_compresses_the_final_mix() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        s.update_effects(&compressor_effect(), 1);
        s.buffers = vec![0.5 * 32767.0; 2 * n];
        let mem = Fake(HashMap::new());
        s.apply_mix_effects(FINAL_MIX, &mem);
        let expected = 0.5 * 0.2992 * 32767.0;
        assert!(
            (s.buffers[2 * n - 1] - expected).abs() < 200.0,
            "compressed sample {} vs {expected}",
            s.buffers[2 * n - 1]
        );
        let status = s.effect_status()[0];
        assert_eq!(status.state, effect::status(true, effect::KIND_COMPRESSOR));
        assert!(!status.result_valid, "the compressor has no result payload");
    }

    #[test]
    fn limiter_effect_reports_statistics_in_the_effect_status() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        s.update_effects(&limiter_effect(true), 1);
        s.buffers = vec![0.0; 2 * n];
        s.buffers[..n].fill(28_000.0);
        let mem = Fake(HashMap::new());
        s.apply_mix_effects(FINAL_MIX, &mem);
        let status = s.effect_status()[0];
        assert_eq!(status.state, effect::status(true, effect::KIND_LIMITER));
        assert!(status.result_valid);
        let input_max = f32::from_le_bytes(status.result[0..4].try_into().unwrap());
        let gain_min = f32::from_le_bytes(status.result[24..28].try_into().unwrap());
        assert!(
            (input_max - 28_000.0 / 32767.0).abs() < 1e-4,
            "input max {input_max}"
        );
        assert!(gain_min < 1.0, "compression gain min {gain_min}");
        assert_eq!(
            f32::from_le_bytes(status.result[4..8].try_into().unwrap()),
            0.0
        );

        // With statistics disabled the specific data is not advertised.
        s.update_effects(&limiter_effect(false), 1);
        assert!(!s.effect_status()[0].result_valid);
    }

    #[test]
    fn disabled_effect_is_bit_identical_passthrough() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        s.buffers = (0..2 * n)
            .map(|i| ((i as f32) * 0.37).sin() * 0.5)
            .collect();
        let before = s.buffers.clone();
        s.update_effects(&reverb_effect(false), 1);
        assert_eq!(
            s.effect_status()
                .iter()
                .map(|s| s.state)
                .collect::<Vec<_>>(),
            vec![effect::status(false, effect::KIND_REVERB)]
        );
        let mem = Fake(HashMap::new());
        s.apply_mix_effects(FINAL_MIX, &mem);
        assert_eq!(s.buffers, before);
    }

    #[test]
    fn enabled_reverb_effect_processes_the_final_mix() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        s.update_effects(&reverb_effect(true), 1);
        assert_eq!(
            s.effect_status()
                .iter()
                .map(|s| s.state)
                .collect::<Vec<_>>(),
            vec![effect::status(true, effect::KIND_REVERB)]
        );
        s.buffers = vec![0.0; 2 * n];
        s.buffers[0] = 1.0;
        s.buffers[n] = 1.0;
        let mem = Fake(HashMap::new());
        for _ in 0..4 {
            s.apply_mix_effects(FINAL_MIX, &mem);
            s.buffers[0] = 1.0;
            s.buffers[n] = 1.0;
        }
        assert!(s.buffers.iter().all(|v| v.is_finite()));
        assert!(
            s.buffers[1..n].iter().any(|v| v.abs() > 1e-6),
            "early reflections must reach samples that were silent"
        );
    }

    #[test]
    fn buffer_mix_effect_routes_with_volume() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        s.update_effects(&buffer_mix_effect(), 1);
        s.buffers = vec![0.0; 2 * n];
        s.buffers[..n].fill(2.0);
        s.buffers[n..].fill(1.0);
        let mem = Fake(HashMap::new());
        s.apply_mix_effects(FINAL_MIX, &mem);
        assert!(
            s.buffers[..n].iter().all(|&v| v == 2.0),
            "input is left alone"
        );
        assert!(
            s.buffers[n..].iter().all(|&v| (v - 2.0).abs() < 1e-6),
            "output must be the input times the route volume"
        );
    }

    #[test]
    fn aux_bus_exchanges_the_send_and_return_rings() {
        const SEND: u64 = 0x1000;
        const RETURN: u64 = 0x4000;
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        s.update_effects(&aux_effect(SEND, RETURN), 1);
        s.buffers = vec![0.0; 2 * n];
        for (i, v) in s.buffers[..n].iter_mut().enumerate() {
            *v = (i % 97) as f32 - 48.0;
        }
        let mem = Ram::new(0x10_000);
        // The application has already filled the return ring.
        let mut returned = Vec::new();
        for i in 0..n {
            returned.extend_from_slice(&((i as i32 * 3 - 7).to_le_bytes()));
        }
        assert!(mem.write(RETURN + 0x80, &returned));

        s.apply_mix_effects(FINAL_MIX, &mem);

        let returned = mem.read(RETURN + 0x80, n as u64 * 4).unwrap();
        for (i, sample) in returned.as_chunks::<4>().0.iter().enumerate() {
            let value = i32::from_le_bytes(*sample) as f32;
            assert!((s.buffers[n + i] - value).abs() < 1e-6);
        }
        let sent = mem.read(SEND + 0x80, n as u64 * 4).unwrap();
        for (i, sample) in sent.as_chunks::<4>().0.iter().enumerate() {
            let value = i32::from_le_bytes(*sample) as f32;
            assert!((value - s.buffers[i]).abs() < 1e-6);
        }
        let write = u32::from_le_bytes(mem.read(SEND + 0x44, 4).unwrap().try_into().unwrap());
        let read = u32::from_le_bytes(mem.read(RETURN + 0x44, 4).unwrap().try_into().unwrap());
        assert_eq!(write as usize, n);
        assert_eq!(read, 0, "the send ring owns its own DSP pointer");
        let read = u32::from_le_bytes(mem.read(RETURN + 0x40, 4).unwrap().try_into().unwrap());
        assert_eq!(read as usize, n);
    }

    #[test]
    fn aux_bus_with_a_mismatched_rate_is_exchanged_without_a_transient() {
        const SEND: u64 = 0x1000;
        const RETURN: u64 = 0x4000;
        const STORAGE: usize = 1024;
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        let mut e = aux_effect(SEND, RETURN);
        put_u32(&mut e, 0x20 + 0x34, 44_100);
        s.update_effects(&e, 1);
        let mem = Ram::new(0x10_000);
        let returned: Vec<u8> = (0..STORAGE)
            .flat_map(|i| ((i as i32 % 100) - 50).to_le_bytes())
            .collect();
        assert!(mem.write(RETURN + 0x80, &returned));

        for block in 0..5 {
            s.buffers = vec![0.0; 2 * n];
            for (i, v) in s.buffers[..n].iter_mut().enumerate() {
                *v = (i % 97) as f32 - 48.0;
            }
            s.apply_mix_effects(FINAL_MIX, &mem);
            for i in 0..n {
                let index = (block * n + i) % STORAGE;
                let want =
                    i32::from_le_bytes(returned[index * 4..index * 4 + 4].try_into().unwrap())
                        as f32;
                let got = s.buffers[n + i];
                assert!(got.is_finite());
                assert!((got - want).abs() < 1e-6, "block {block} sample {i}");
            }
        }
        let write = u32::from_le_bytes(mem.read(SEND + 0x44, 4).unwrap().try_into().unwrap());
        let read = u32::from_le_bytes(mem.read(RETURN + 0x40, 4).unwrap().try_into().unwrap());
        assert_eq!(write as usize, (5 * n) % STORAGE);
        assert_eq!(read as usize, (5 * n) % STORAGE);
    }

    #[test]
    fn a_44k1_aux_block_in_a_48k_mix_does_not_drift_or_click() {
        const SEND: u64 = 0x1000;
        const RETURN: u64 = 0x4000;
        const CHANNELS: usize = 2;
        let mut s = State::new(sizes());
        graph_with_buffers(&mut s, 3);
        let n = s.sizes.sample_count;
        // Four blocks fit in the ring, like a real application's storage.
        let storage = CHANNELS * n * 4;
        let mut e = aux_effect(SEND, RETURN);
        e[0x20 + 1] = 1; // channel 1 reads mix buffer 1
        e[0x20 + 24 + 1] = 2; // and writes mix buffer 2
        put_u32(&mut e, 0x20 + 0x30, CHANNELS as u32);
        put_u32(&mut e, 0x20 + 0x34, 44_100);
        put_u32(&mut e, 0x20 + 0x38, storage as u32);
        s.update_effects(&e, 1);

        // The application's return ring holds a continuous waveform. Its
        // layout is channel-major per block (`block, channel, sample`), the
        // same order the DSP reads, and it is periodic with the ring so the
        // wrap continues the waveform exactly as an application that keeps up
        // would have rewritten it.
        let period = (storage / CHANNELS) as f64;
        let amplitudes = [16_000.0f32, 8_000.0];
        let reference = |index: usize, channel: usize| {
            let block = index / n;
            let within = index % n;
            (std::f64::consts::TAU * (block * n + within) as f64 / period).sin() as f32
                * amplitudes[channel]
        };
        let mut ring = Vec::with_capacity(storage * 4);
        for index in 0..storage {
            let block = index / (CHANNELS * n);
            let at = index % (CHANNELS * n);
            let channel = at / n;
            let within = at % n;
            let value = reference(block * n + within, channel);
            ring.extend_from_slice(&(value as i32).to_le_bytes());
        }
        let mem = Ram::new(0x10_000);
        assert!(mem.write(RETURN + 0x80, &ring));

        let mut last: [Option<f32>; CHANNELS] = [None; CHANNELS];
        let mut max_step = [0.0f32; CHANNELS];
        for block in 0..7 {
            s.buffers = vec![0.0; 3 * n];
            let sent: Vec<f32> = (0..CHANNELS * n)
                .map(|i| ((block * CHANNELS * n + i) % 1000) as f32 - 500.0)
                .collect();
            s.buffers[..CHANNELS * n].copy_from_slice(&sent);
            s.apply_mix_effects(FINAL_MIX, &mem);

            for channel in 0..CHANNELS {
                for i in 0..n {
                    let want = reference(block * n + i, channel) as i32 as f32;
                    let got = s.buffers[(channel + 1) * n + i];
                    assert!(
                        got.is_finite(),
                        "block {block} channel {channel} sample {i}"
                    );
                    assert!(
                        (got - want).abs() < 1e-6,
                        "block {block} channel {channel} sample {i}: {got} vs {want}"
                    );
                    assert!(
                        got.abs() <= amplitudes[channel] + 1.0,
                        "sample out of the ring's range: {got}"
                    );
                    if let Some(previous) = last[channel] {
                        max_step[channel] = max_step[channel].max((got - previous).abs());
                    }
                    last[channel] = Some(got);
                }
            }

            // The send side is a 1:1 copy at the same advancing offset.
            let offset = (block * CHANNELS * n) % storage;
            for channel in 0..CHANNELS {
                let bytes = mem
                    .read(
                        SEND + 0x80 + ((offset + channel * n) * 4) as u64,
                        n as u64 * 4,
                    )
                    .unwrap();
                for (i, sample) in bytes.as_chunks::<4>().0.iter().enumerate() {
                    let value = i32::from_le_bytes(*sample) as f32;
                    assert!(
                        (value - sent[channel * n + i]).abs() < 1e-6,
                        "send block {block} channel {channel} sample {i}"
                    );
                }
            }
        }

        // No resampler, so the output never jumps more than the waveform's own
        // step: no click at a block boundary and no accumulated drift.
        for channel in 0..CHANNELS {
            let step = amplitudes[channel] as f64 * std::f64::consts::TAU / period;
            assert!(
                max_step[channel] as f64 <= step + 2.0,
                "channel {channel} jumps {} per sample, waveform step is {step}",
                max_step[channel]
            );
        }
        let advanced = (7 * CHANNELS * n) % storage;
        let write = u32::from_le_bytes(mem.read(SEND + 0x44, 4).unwrap().try_into().unwrap());
        let read = u32::from_le_bytes(mem.read(RETURN + 0x40, 4).unwrap().try_into().unwrap());
        assert_eq!(write as usize, advanced);
        assert_eq!(read as usize, advanced);
    }

    #[test]
    fn the_presentation_position_follows_the_device_when_it_exists() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        // No host output: the position is the historical mixing rate.
        s.elapsed_frames = 10;
        assert_eq!(s.presentation_frames(n), 10 * n as u64);
        // The device has consumed seven blocks; the three queued blocks have
        // not reached the speakers and are not part of the media position.
        s.set_output_played(Some(7 * n as u64));
        assert_eq!(s.presentation_frames(n), 7 * n as u64);
        // Muted, headless or no device: back to the mixing rate.
        s.set_output_played(None);
        assert_eq!(s.presentation_frames(n), 10 * n as u64);
        // A device that has consumed nothing reports zero, not a negative or
        // wrapped count.
        s.set_output_played(Some(0));
        assert_eq!(s.presentation_frames(n), 0);
    }

    #[test]
    fn disabled_effect_still_routes_its_input_to_its_output() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        let mut e = reverb_effect(false);
        e[0x26] = 1;
        e[0x27] = 0;
        s.update_effects(&e, 1);
        s.buffers = vec![0.0; 2 * n];
        s.buffers[..n].fill(3.0);
        let mem = Fake(HashMap::new());
        s.apply_mix_effects(FINAL_MIX, &mem);
        assert!(s.buffers[n..].iter().all(|&v| v == 3.0));
    }

    #[test]
    fn disabling_a_biquad_keeps_the_mix_routed() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let n = s.sizes.sample_count;
        let mut e = effect_block(effect::KIND_BIQUAD, false);
        e[0x20] = 0;
        e[0x26] = 1;
        e[0x36] = 1;
        s.update_effects(&e, 1);
        s.buffers = vec![0.0; 2 * n];
        s.buffers[..n].fill(-2.0);
        let mem = Fake(HashMap::new());
        s.apply_mix_effects(FINAL_MIX, &mem);
        assert!(s.buffers[n..].iter().all(|&v| v == -2.0));
    }

    #[test]
    fn six_channel_sink_keeps_every_channel() {
        let mut s = State::new(Sizes {
            mix_buffers: 6,
            ..sizes()
        });
        let n = s.sizes.sample_count;
        let mut mixes = vec![0u8; 0x20 + MIX_IN];
        put_u32(&mut mixes, 4, 1);
        let m = &mut mixes[0x20..];
        put_u32(m, 0, 1.0f32.to_bits());
        put_u32(m, 8, 6);
        m[0x0c] = 1;
        put_u32(m, 0x924, UNUSED_MIX as u32);
        put_u32(m, 0x928, UNUSED_SPLITTER);
        s.update_mixes(&mixes);
        let mut sink = vec![0u8; SINK_IN];
        sink[0] = SINK_DEVICE;
        sink[1] = 1;
        put_u32(&mut sink, 0x120, 6);
        for c in 0..6 {
            sink[0x124 + c] = c as u8;
        }
        s.update_sinks(&sink);
        s.buffers = (0..6 * n)
            .map(|i| ((i / n) as f32 + 1.0) * 1000.0)
            .collect();
        let mut out = Vec::new();
        s.write_sink(&mut out);
        assert_eq!(out.len(), n);
        for (c, sample) in out[10].iter().enumerate().take(6) {
            let want = ((c + 1) as f32 * 1000.0) / 32768.0;
            assert!((sample - want).abs() < 1e-6, "channel {c}");
        }
        assert_eq!(s.output().0, 6);
    }

    #[test]
    fn pcm_voice_plays_through_the_final_mix_and_reports_consumption() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let samples = 400;
        let pcm: Vec<u8> = (0..samples).flat_map(|_| 16384i16.to_le_bytes()).collect();
        let mem = Fake(HashMap::from([(BUFFER, pcm)]));
        let mut out = vec![0u8; VOICE_OUT];
        s.update_voices(&voice(true, false, samples, 0), &mut out);
        assert_eq!(status(&out), (0, 0));

        let mut frame = Vec::new();
        s.render(&mem, &mut frame);
        assert_eq!(frame.len(), 240);
        assert_eq!(frame[10][..2], [0.25, 0.0]);
        s.update_voices(&voice(false, true, samples, 0), &mut out);
        assert_eq!(status(&out), (240, 0));

        s.render(&mem, &mut frame);
        s.update_voices(&voice(false, true, samples, 0), &mut out);
        assert_eq!(status(&out), (400, 1));
        assert_eq!(frame[100][..2], [0.25, 0.0]);
        assert_eq!(frame[200], [0.0; MAX_OUTPUT_CHANNELS]);
    }

    #[test]
    fn stopping_a_voice_drops_its_queued_buffers() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let mem = Fake(HashMap::from([(BUFFER, vec![0; 4000])]));
        let mut out = vec![0u8; VOICE_OUT];
        s.update_voices(&voice(true, false, 2000, 0), &mut out);
        let mut frame = Vec::new();
        s.render(&mem, &mut frame);
        s.update_voices(&voice(false, true, 2000, 1), &mut out);
        assert_eq!(status(&out), (240, 0));
        s.render(&mem, &mut frame);
        s.update_voices(&voice(false, true, 2000, 1), &mut out);
        assert_eq!(status(&out), (0, 1));
    }

    #[test]
    fn paused_voices_keep_new_buffers_queued() {
        let mut s = State::new(sizes());
        graph(&mut s);
        let mem = Fake(HashMap::from([(BUFFER, vec![0; 400])]));
        let mut out = vec![0u8; VOICE_OUT];
        s.update_voices(&voice(true, false, 200, 2), &mut out);
        let mut frame = Vec::new();
        s.render(&mem, &mut frame);
        s.update_voices(&voice(false, true, 200, 2), &mut out);
        assert_eq!(status(&out), (0, 0));
        s.update_voices(&voice(false, true, 200, 0), &mut out);
        s.render(&mem, &mut frame);
        s.update_voices(&voice(false, true, 200, 0), &mut out);
        assert_eq!(status(&out), (200, 1));
    }
}
