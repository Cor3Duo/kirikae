//! Host audio output: queue, resampling, channel layouts and the measured
//! device lead.
//!
//! # Channel layouts
//!
//! The renderer's sink channels and the host device channels are matched by
//! name, not by index, through [`layout`]. The defined layouts are mono,
//! stereo, 3-channel (`FL FR FC`), quad (`FL FR BL BR`), 5.0 (`FL FR FC BL BR`),
//! 5.1 (`FL FR FC LFE BL BR`) and 7.1 (`FL FR FC LFE BL BR SL SR`). A host
//! device with seven channels keeps the 5.1 order plus the left surround; any
//! other count is clamped. A channel with no slot on the smaller layout is
//! folded into the front pair with the sink coefficients (front, centre, LFE,
//! back; the side pair uses the back coefficient because the sink parameter
//! carries four gains), so nothing is silently dropped. Extra device channels
//! stay silent.
//!
//! # Latency
//!
//! The queue depth ([`Output::lead_frames`]) is the measurable part of the
//! output latency: frames mixed and handed to the host stream but not consumed
//! by its callback yet. The presentation clock follows
//! [`Output::played_frames`], the monotonic count of frames the callback has
//! pulled (and therefore the media position reaching the speakers), so a frame
//! still in the queue is never reported as heard and the audio position cannot
//! move backwards when the queue depth changes. The host API's own buffer is
//! requested with `BufferSize::Default` and cpal exposes no way to query it, so
//! that part is not measurable here and is not compensated.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub const MAX_CHANNELS: usize = 8;
pub type Frame = [f32; MAX_CHANNELS];

/// Standard 5.1 fold-down: front, centre, LFE, back. Matches Ryujinx's
/// `DefaultSurroundToStereoCoefficients` and the `audren` sink default.
/// The back coefficient is also used for the side channels of a 7.1 layout,
/// because the sink parameter carries only these four gains.
pub const DEFAULT_DOWNMIX: [f32; 4] = [1.0, 0.707, 0.251, 0.707];

/// One named channel of a renderer/device layout.
///
/// The name, not the index, is what survives a mapping between layouts: a
/// quad renderer feeding a 5.1 device keeps its back channels on the back
/// speakers instead of shifting them into the centre and LFE slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    FrontLeft,
    FrontRight,
    FrontCenter,
    Lfe,
    BackLeft,
    BackRight,
    SideLeft,
    SideRight,
}

const MONO: [Channel; 1] = [Channel::FrontLeft];
const STEREO: [Channel; 2] = [Channel::FrontLeft, Channel::FrontRight];
const TRIO: [Channel; 3] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
];
/// Quad is `FL FR BL BR`; it is the only public convention for four channels.
const QUAD: [Channel; 4] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::BackLeft,
    Channel::BackRight,
];
/// Five channels without LFE: `FL FR FC BL BR`.
const FIVE: [Channel; 5] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
    Channel::BackLeft,
    Channel::BackRight,
];
/// The documented 5.1 order: `FL FR FC LFE BL BR`.
const FIVE_ONE: [Channel; 6] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
    Channel::Lfe,
    Channel::BackLeft,
    Channel::BackRight,
];
/// Seven channels are not a standard layout. Keep the 5.1 order and the left
/// surround, leaving the right surround silent, so the mapping stays defined.
const SEVEN: [Channel; 7] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
    Channel::Lfe,
    Channel::BackLeft,
    Channel::BackRight,
    Channel::SideLeft,
];
/// The documented 7.1 order: `FL FR FC LFE BL BR SL SR`.
const SEVEN_ONE: [Channel; 8] = [
    Channel::FrontLeft,
    Channel::FrontRight,
    Channel::FrontCenter,
    Channel::Lfe,
    Channel::BackLeft,
    Channel::BackRight,
    Channel::SideLeft,
    Channel::SideRight,
];

/// The channel map of a renderer or host device with `count` channels.
///
/// Only 1, 2, 3, 4, 5, 6 and 8 are standard; 7 is kept defined as 5.1 plus the
/// left surround so an unusual host configuration still has a deterministic
/// map. Counts outside 1..=8 clamp. Horizon's sink block does not document the
/// 4- and 5-channel orders, so quad (`FL FR BL BR`) and 5.0 (`FL FR FC BL BR`)
/// are recorded here as the convention this runtime uses.
pub fn layout(count: usize) -> &'static [Channel] {
    match count.clamp(1, MAX_CHANNELS) {
        1 => &MONO,
        2 => &STEREO,
        3 => &TRIO,
        4 => &QUAD,
        5 => &FIVE,
        6 => &FIVE_ONE,
        7 => &SEVEN,
        _ => &SEVEN_ONE,
    }
}

struct Shared {
    queue: Mutex<VecDeque<Frame>>,
    downmix: Mutex<[f32; 4]>,
    alive: AtomicBool,
    capacity: usize,
    dry: AtomicU64,
    source: AtomicU64,
    /// Renderer frames the host callback has consumed (pulled from the queue
    /// and interpolated into the device stream). This is the monotonic media
    /// position the presentation clock must follow: frames still queued ahead
    /// of the callback have not reached the speakers yet.
    played: AtomicU64,
}

pub struct Output {
    shared: Arc<Shared>,
    pub device: String,
    pub device_channels: usize,
}

impl Output {
    /// Open a host stream for `source_channels` interleaved renderer channels.
    /// A 6-channel device configuration is used when the renderer produces
    /// surround and the device advertises one; otherwise the stream folds the
    /// renderer channels down with the sink coefficients.
    pub fn open(rate: u32, source_channels: usize) -> Result<Output, String> {
        let source_channels = source_channels.clamp(1, MAX_CHANNELS);
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            downmix: Mutex::new(DEFAULT_DOWNMIX),
            alive: AtomicBool::new(true),
            capacity: rate as usize / 2,
            dry: AtomicU64::new(0),
            source: AtomicU64::new(source_channels as u64),
            played: AtomicU64::new(0),
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let s = shared.clone();
        std::thread::Builder::new()
            .name("kirikae-audio-out".into())
            .spawn(move || {
                let (stream, description, device_channels) =
                    match build(rate, s.clone(), source_channels) {
                        Ok(v) => {
                            let _ = tx.send(Ok((v.1.clone(), v.2)));
                            v
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            return;
                        }
                    };
                let _ = (description, device_channels);
                while s.alive.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                drop(stream);
            })
            .map_err(|e| e.to_string())?;
        let (device, device_channels) = rx.recv().map_err(|e| e.to_string())??;
        Ok(Output {
            shared,
            device,
            device_channels,
        })
    }

    pub fn queued(&self) -> usize {
        self.shared.queue.lock().unwrap().len()
    }

    pub fn dry_frames(&self) -> u64 {
        self.shared.dry.load(Ordering::Relaxed)
    }

    /// Renderer frames queued for the host stream but not consumed by its
    /// callback yet. This is the part of the device latency the runtime can
    /// measure: the host API's own internal buffer is not exposed by cpal and
    /// is requested as `BufferSize::Default`, so it is not included.
    pub fn lead_frames(&self) -> usize {
        self.queued()
    }

    /// [`Output::lead_frames`] expressed in milliseconds at `rate`.
    pub fn lead_ms(&self, rate: u32) -> f32 {
        self.lead_frames() as f32 * 1000.0 / rate.max(1) as f32
    }

    /// Renderer frames the device callback has consumed. Monotonic; it is the
    /// media position the presentation clock is fed.
    pub fn played_frames(&self) -> u64 {
        self.shared.played.load(Ordering::Relaxed)
    }

    pub fn push(&self, frames: &[Frame], channels: usize, downmix: [f32; 4]) {
        self.shared
            .source
            .store(channels.clamp(1, MAX_CHANNELS) as u64, Ordering::Relaxed);
        if downmix.iter().all(|v| v.is_finite()) {
            *self.shared.downmix.lock().unwrap() = downmix;
        }
        let mut q = self.shared.queue.lock().unwrap();
        q.extend(frames);
        let excess = q.len().saturating_sub(self.shared.capacity);
        q.drain(..excess);
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        self.shared.alive.store(false, Ordering::Relaxed);
    }
}

/// `(stream, description, device channel count)`
type Built = (cpal::Stream, String, usize);

/// Pick the host channel count for a renderer layout.
///
/// Preference order: the same count, then the smallest larger count (no
/// folding, the extra device channels stay silent), then the largest smaller
/// count above stereo (fold only the channels that do not fit), then `None`
/// so the host default (normally stereo) is used. Mono and stereo renderers
/// keep the default stereo device.
fn choose_channels(source: usize, counts: impl Iterator<Item = usize>) -> Option<usize> {
    let source = source.clamp(1, MAX_CHANNELS);
    if source <= 2 {
        return None;
    }
    let mut counts: Vec<usize> = counts.filter(|c| (1..=MAX_CHANNELS).contains(c)).collect();
    counts.sort_unstable();
    counts.dedup();
    counts
        .iter()
        .copied()
        .find(|&c| c == source)
        .or_else(|| counts.iter().copied().find(|&c| c > source))
        .or_else(|| counts.iter().rev().copied().find(|&c| c > 2))
}

fn build(rate: u32, shared: Arc<Shared>, source_channels: usize) -> Result<Built, String> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let default = device.default_output_config().map_err(|e| e.to_string())?;
    let format = default.sample_format();
    let mut supported = default;

    // Prefer a host stream whose channel count matches (or contains) the
    // renderer layout instead of folding everything to stereo.
    if source_channels > 2
        && let Ok(ranges) = device.supported_output_configs()
    {
        let ranges: Vec<_> = ranges.collect();
        let counts = ranges
            .iter()
            .filter(|r| r.sample_format() == format)
            .map(|r| usize::from(r.channels()));
        if let Some(chosen) = choose_channels(source_channels, counts) {
            let mut at_rate = None;
            let mut fallback = None;
            for range in ranges
                .iter()
                .filter(|r| r.sample_format() == format && usize::from(r.channels()) == chosen)
            {
                if (range.min_sample_rate()..=range.max_sample_rate()).contains(&rate) {
                    at_rate = Some(range.with_sample_rate(rate));
                    break;
                }
                if fallback.is_none() {
                    fallback = Some(range.with_sample_rate(range.max_sample_rate()));
                }
            }
            if let Some(range) = at_rate.or(fallback) {
                supported = range;
            }
        }
    }

    let mut config: cpal::StreamConfig = supported.config();
    if config.sample_rate != rate {
        let takes_rate = device.supported_output_configs().is_ok_and(|mut c| {
            c.any(|c| {
                c.channels() == config.channels
                    && c.sample_format() == format
                    && (c.min_sample_rate()..=c.max_sample_rate()).contains(&rate)
            })
        });
        if takes_rate {
            config.sample_rate = rate;
        }
    }
    let step = rate as f32 / config.sample_rate as f32;
    let device_channels = usize::from(config.channels).max(1);
    let name = device
        .description()
        .map(|d| d.to_string())
        .unwrap_or_else(|_| "default device".into());
    let description = format!(
        "{name}, {} Hz, {} device channels, {format}",
        config.sample_rate, config.channels
    );
    let stream = match format {
        cpal::SampleFormat::I16 => stream::<i16>(&device, &config, shared, step),
        cpal::SampleFormat::U16 => stream::<u16>(&device, &config, shared, step),
        cpal::SampleFormat::I32 => stream::<i32>(&device, &config, shared, step),
        _ => stream::<f32>(&device, &config, shared, step),
    }?;
    stream.play().map_err(|e| e.to_string())?;
    Ok((stream, description, device_channels))
}

/// Fold `source` renderer channels down to two channels.
///
/// The channel order comes from [`layout`]: 1 is mono, 2 is `FL FR`, 3 adds
/// `FC`, 4 is quad (`FL FR BL BR`), 5 is `FL FR FC BL BR`, 6 is 5.1
/// (`FL FR FC LFE BL BR`) and 8 is 7.1 (`FL FR FC LFE BL BR SL SR`). Centre,
/// LFE and the surrounds are folded with the sink coefficients (front, centre,
/// LFE, back); the side channels use the back coefficient because the sink
/// block only carries four gains. A stereo source is passed through instead of
/// scaled by the front gain, and the mono source is duplicated.
pub fn downmix(frame: &Frame, source: usize, coefficients: [f32; 4]) -> [f32; 2] {
    let source = source.clamp(1, MAX_CHANNELS);
    if source == 1 {
        return [frame[0]; 2];
    }
    if source == 2 {
        return [frame[0], frame[1]];
    }
    let k = coefficients;
    let mut left = 0.0;
    let mut right = 0.0;
    for (channel, &sample) in layout(source).iter().zip(frame) {
        fold(*channel, sample, k, &mut left, &mut right);
    }
    [left, right]
}

/// Add one source channel to the stereo pair, either directly when it has a
/// front slot or through the documented coefficient when it is folded.
fn fold(channel: Channel, sample: f32, k: [f32; 4], left: &mut f32, right: &mut f32) {
    match channel {
        Channel::FrontLeft => *left += sample * k[0],
        Channel::FrontRight => *right += sample * k[0],
        Channel::FrontCenter => {
            *left += sample * k[1];
            *right += sample * k[1];
        }
        Channel::Lfe => {
            *left += sample * k[2];
            *right += sample * k[2];
        }
        Channel::BackLeft | Channel::SideLeft => *left += sample * k[3],
        Channel::BackRight | Channel::SideRight => *right += sample * k[3],
    }
}

/// Map a renderer frame to a device frame.
///
/// Channels are matched by name: a quad renderer feeding a 5.1 device keeps its
/// back channels in the `BL BR` slots and leaves `FC LFE` silent; a 5.1
/// renderer feeding a quad device folds centre and LFE into the front pair with
/// the sink coefficients. A channel the device has no slot for is folded into
/// the front pair, never dropped. A mono renderer fills the front left/right
/// pair only, and a one-channel device receives the average of the stereo fold.
pub fn map_frame(frame: &Frame, source: usize, device: usize, coefficients: [f32; 4]) -> Frame {
    let source = source.clamp(1, MAX_CHANNELS);
    let device = device.clamp(1, MAX_CHANNELS);
    let mut out = [0.0f32; MAX_CHANNELS];
    if source == 1 {
        out[0] = frame[0];
        if device >= 2 {
            out[1] = frame[0];
        }
        return out;
    }
    if device == 1 {
        let [left, right] = downmix(frame, source, coefficients);
        out[0] = (left + right) * 0.5;
        return out;
    }
    if device == 2 {
        let [left, right] = downmix(frame, source, coefficients);
        out[0] = left;
        out[1] = right;
        return out;
    }
    let destination = layout(device);
    let mut folded = [0.0f32; 2];
    for (channel, &sample) in layout(source).iter().zip(frame) {
        match destination.iter().position(|slot| slot == channel) {
            Some(index) => out[index] += sample,
            None => {
                let (left, right) = folded.split_at_mut(1);
                fold(*channel, sample, coefficients, &mut left[0], &mut right[0]);
            }
        }
    }
    out[0] += folded[0];
    out[1] += folded[1];
    out
}

struct Resampler {
    position: f32,
    previous: Frame,
    next: Frame,
}

impl Resampler {
    fn next(
        &mut self,
        q: &mut VecDeque<Frame>,
        step: f32,
        dry: &mut u64,
        played: &mut u64,
    ) -> Frame {
        while self.position >= 1.0 {
            self.previous = self.next;
            self.next = match q.pop_front() {
                Some(frame) => {
                    *played += 1;
                    frame
                }
                None => {
                    *dry += 1;
                    [0.0; MAX_CHANNELS]
                }
            };
            self.position -= 1.0;
        }
        let t = self.position;
        self.position += step;
        std::array::from_fn(|c| self.previous[c] + (self.next[c] - self.previous[c]) * t)
    }
}

fn stream<T: cpal::SizedSample + cpal::FromSample<f32>>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<Shared>,
    step: f32,
) -> Result<cpal::Stream, String> {
    let device_channels = usize::from(config.channels).max(1);
    let mut resampler = Resampler {
        position: 1.0,
        previous: [0.0; MAX_CHANNELS],
        next: [0.0; MAX_CHANNELS],
    };
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                let mut q = shared.queue.lock().unwrap();
                let coefficients = *shared.downmix.lock().unwrap();
                let source = shared.source.load(Ordering::Relaxed) as usize;
                let mut dry = 0;
                let mut played = 0;
                for frame in data.chunks_mut(device_channels) {
                    let renderer = resampler.next(&mut q, step, &mut dry, &mut played);
                    let mapped = map_frame(&renderer, source, device_channels, coefficients);
                    for (k, sample) in frame.iter_mut().enumerate() {
                        *sample = T::from_sample(mapped.get(k).copied().unwrap_or(0.0));
                    }
                }
                shared.dry.fetch_add(dry, Ordering::Relaxed);
                shared.played.fetch_add(played, Ordering::Relaxed);
            },
            |e| eprintln!("kirikae: audio output: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

pub struct WavDump {
    file: std::io::BufWriter<std::fs::File>,
    rate: u32,
    channels: usize,
    frames: u32,
}

impl WavDump {
    pub fn create(path: &std::path::Path, rate: u32, channels: usize) -> std::io::Result<WavDump> {
        let mut dump = WavDump {
            file: std::io::BufWriter::new(std::fs::File::create(path)?),
            rate,
            channels: channels.clamp(1, MAX_CHANNELS),
            frames: 0,
        };
        dump.header()?;
        Ok(dump)
    }

    fn header(&mut self) -> std::io::Result<()> {
        use std::io::{Seek, Write};
        let width = self.channels as u16 * 2;
        let data = self.frames * u32::from(width);
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + data).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&(width / 2).to_le_bytes());
        h.extend_from_slice(&self.rate.to_le_bytes());
        h.extend_from_slice(&(self.rate * u32::from(width)).to_le_bytes());
        h.extend_from_slice(&width.to_le_bytes());
        h.extend_from_slice(&16u16.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&data.to_le_bytes());
        let end = self.file.stream_position()?;
        self.file.seek(std::io::SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.seek(std::io::SeekFrom::Start(end.max(44)))?;
        Ok(())
    }

    pub fn write(&mut self, frames: &[Frame]) -> std::io::Result<()> {
        use std::io::Write;
        for f in frames {
            for sample in f.iter().take(self.channels) {
                self.file
                    .write_all(&((sample * 32767.0) as i16).to_le_bytes())?;
            }
        }
        let before = self.frames;
        self.frames += frames.len() as u32;
        if before / self.rate != self.frames / self.rate {
            self.header()?;
            self.file.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_interpolates_and_plays_silence_when_dry() {
        let mut a = [0.0; MAX_CHANNELS];
        let mut b = [0.0; MAX_CHANNELS];
        a[..2].copy_from_slice(&[1.0, -1.0]);
        b[..2].copy_from_slice(&[3.0, -3.0]);
        let mut q: VecDeque<Frame> = [a, b].into_iter().collect();
        let mut r = Resampler {
            position: 1.0,
            previous: [0.0; MAX_CHANNELS],
            next: [0.0; MAX_CHANNELS],
        };
        let mut dry = 0;
        let mut played = 0;
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [0.0, 0.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [0.5, -0.5]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [1.0, -1.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [2.0, -2.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [3.0, -3.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [1.5, -1.5]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry, &mut played)[..2], [0.0, 0.0]);
        assert_eq!(dry, 2);
        assert_eq!(
            played, 2,
            "only frames that existed advance the media clock"
        );
    }

    #[test]
    fn the_layout_of_every_supported_channel_count_is_explicit() {
        assert_eq!(layout(1), &[Channel::FrontLeft]);
        assert_eq!(layout(2), &[Channel::FrontLeft, Channel::FrontRight]);
        assert_eq!(
            layout(3),
            &[
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::FrontCenter
            ]
        );
        // Quad is not a 5.1 prefix: its back channels start at index 2.
        assert_eq!(
            layout(4),
            &[
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::BackLeft,
                Channel::BackRight
            ]
        );
        assert_eq!(
            layout(5)[2..],
            [Channel::FrontCenter, Channel::BackLeft, Channel::BackRight]
        );
        assert_eq!(
            layout(6),
            &[
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::FrontCenter,
                Channel::Lfe,
                Channel::BackLeft,
                Channel::BackRight
            ]
        );
        assert_eq!(layout(7)[6], Channel::SideLeft);
        assert_eq!(
            layout(8)[6..],
            [Channel::SideLeft, Channel::SideRight],
            "8 channels are the documented 7.1 order"
        );
        assert_eq!(layout(0), layout(1));
        assert_eq!(layout(99), layout(8));
    }

    #[test]
    fn six_channel_downmix_uses_the_documented_coefficients() {
        let frame = [0.5, -0.5, 0.25, 0.125, 0.75, -0.75, 0.0, 0.0];
        let [left, right] = downmix(&frame, 6, DEFAULT_DOWNMIX);
        let k = DEFAULT_DOWNMIX;
        let want_left = frame[0] * k[0] + frame[2] * k[1] + frame[3] * k[2] + frame[4] * k[3];
        let want_right = frame[1] * k[0] + frame[2] * k[1] + frame[3] * k[2] + frame[5] * k[3];
        assert!((left - want_left).abs() < 1e-6, "{left} vs {want_left}");
        assert!((right - want_right).abs() < 1e-6, "{right} vs {want_right}");
        // Center and LFE are shared, the backs are separate.
        assert!((left - 1.238_375).abs() < 1e-4, "{left}");
        assert!((right + 0.822_125).abs() < 1e-4, "{right}");
    }

    #[test]
    fn quad_five_and_seven_one_fold_down_from_their_own_layouts() {
        // Quad: FL FR BL BR. The back pair is index 2/3, not 4/5.
        let mut quad = [0.0; MAX_CHANNELS];
        quad[..4].copy_from_slice(&[1.0, -1.0, 0.5, -0.5]);
        let [left, right] = downmix(&quad, 4, DEFAULT_DOWNMIX);
        assert!((left - (1.0 + 0.5 * 0.707)).abs() < 1e-6, "{left}");
        assert!((right + (1.0 + 0.5 * 0.707)).abs() < 1e-6, "{right}");

        // 5.0: FL FR FC BL BR.
        let mut five = [0.0; MAX_CHANNELS];
        five[..5].copy_from_slice(&[1.0, -1.0, 0.5, 0.25, -0.25]);
        let [left, right] = downmix(&five, 5, DEFAULT_DOWNMIX);
        let center = 0.5 * 0.707;
        assert!(
            (left - (1.0 + center + 0.25 * 0.707)).abs() < 1e-6,
            "{left}"
        );
        assert!(
            (right - (-1.0 + center - 0.25 * 0.707)).abs() < 1e-6,
            "{right}"
        );

        // 7.1: the side pair is folded with the back coefficient.
        let mut seven_one = [0.0; MAX_CHANNELS];
        seven_one.copy_from_slice(&[1.0, -1.0, 0.0, 0.0, 0.25, -0.25, 0.5, -0.5]);
        let [left, right] = downmix(&seven_one, 8, DEFAULT_DOWNMIX);
        assert!((left - (1.0 + 0.25 * 0.707 + 0.5 * 0.707)).abs() < 1e-6);
        assert!((right + (1.0 + 0.25 * 0.707 + 0.5 * 0.707)).abs() < 1e-6);

        // Mono and stereo are not scaled by the front gain.
        assert_eq!(downmix(&[0.25; 8], 1, DEFAULT_DOWNMIX), [0.25, 0.25]);
        assert_eq!(
            downmix(
                &[1.0, -1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                2,
                DEFAULT_DOWNMIX
            ),
            [1.0, -1.0]
        );
    }

    #[test]
    fn three_channel_fold_down_puts_the_centre_in_both_fronts() {
        let mut frame = [0.0; MAX_CHANNELS];
        frame[..3].copy_from_slice(&[0.5, -0.5, 1.0]);
        let [left, right] = downmix(&frame, 3, DEFAULT_DOWNMIX);
        assert!((left - (0.5 + 0.707)).abs() < 1e-6);
        assert!((right - (-0.5 + 0.707)).abs() < 1e-6);
    }

    #[test]
    fn a_six_channel_device_receives_every_channel() {
        let frame = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let six = map_frame(&frame, 6, 6, DEFAULT_DOWNMIX);
        assert_eq!(six[..6], frame[..6]);
        assert_eq!(six[6..], [0.0; 2]);
        let stereo = map_frame(&frame, 6, 2, DEFAULT_DOWNMIX);
        assert_eq!(stereo[..2], downmix(&frame, 6, DEFAULT_DOWNMIX));
        assert_eq!(stereo[2..], [0.0; 6]);
        let mono = map_frame(&frame, 6, 1, DEFAULT_DOWNMIX);
        let [left, right] = downmix(&frame, 6, DEFAULT_DOWNMIX);
        assert!((mono[0] - (left + right) * 0.5).abs() < 1e-6);
    }

    #[test]
    fn mappings_between_channel_counts_follow_the_channel_names() {
        // Quad into 5.1 keeps BL/BR in their own slots and leaves FC/LFE quiet.
        let mut quad = [0.0; MAX_CHANNELS];
        quad[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let five_one = map_frame(&quad, 4, 6, DEFAULT_DOWNMIX);
        assert_eq!(five_one[..6], [1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);

        // 5.1 into quad folds centre and LFE into the front pair.
        let mut five_one = [0.0; MAX_CHANNELS];
        five_one[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let quad = map_frame(&five_one, 6, 4, DEFAULT_DOWNMIX);
        let k = DEFAULT_DOWNMIX;
        assert!((quad[0] - (1.0 + 3.0 * k[1] + 4.0 * k[2])).abs() < 1e-6);
        assert!((quad[1] - (2.0 + 3.0 * k[1] + 4.0 * k[2])).abs() < 1e-6);
        assert_eq!(quad[2..4], [5.0, 6.0]);

        // 5.1 into 7.1 places six channels and leaves the surrounds silent.
        let seven_one = map_frame(&five_one, 6, 8, DEFAULT_DOWNMIX);
        assert_eq!(seven_one[..6], five_one[..6]);
        assert_eq!(seven_one[6..], [0.0, 0.0]);

        // 7.1 into 5.1 folds the side pair into the fronts with k[3].
        let mut seven_one = [0.0; MAX_CHANNELS];
        seven_one.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let folded = map_frame(&seven_one, 8, 6, DEFAULT_DOWNMIX);
        assert!((folded[0] - (1.0 + 7.0 * k[3])).abs() < 1e-6);
        assert!((folded[1] - (2.0 + 8.0 * k[3])).abs() < 1e-6);
        assert_eq!(folded[2..6], [3.0, 4.0, 5.0, 6.0]);

        // Stereo into 5.1 places the pair in front and keeps the rest quiet.
        let stereo = map_frame(
            &[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
            6,
            DEFAULT_DOWNMIX,
        );
        assert_eq!(stereo[..6], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);

        // A 3-channel device is FL FR FC, so a 5.1 source keeps its centre and
        // folds the back/LFE channels into the fronts.
        let trio = map_frame(&five_one, 6, 3, DEFAULT_DOWNMIX);
        assert!((trio[0] - (1.0 + 5.0 * k[3] + 4.0 * k[2])).abs() < 1e-6);
        assert!((trio[1] - (2.0 + 6.0 * k[3] + 4.0 * k[2])).abs() < 1e-6);
        assert!((trio[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn the_host_channel_count_is_chosen_explicitly() {
        // Exact matches win.
        assert_eq!(choose_channels(6, [2, 6, 8].into_iter()), Some(6));
        assert_eq!(choose_channels(4, [2, 4].into_iter()), Some(4));
        // No exact match: the smallest larger count, so nothing is folded.
        assert_eq!(choose_channels(4, [2, 6, 8].into_iter()), Some(6));
        assert_eq!(choose_channels(6, [2, 8].into_iter()), Some(8));
        // Only smaller surround counts: the largest one, so the least is folded.
        assert_eq!(choose_channels(8, [2, 4, 6].into_iter()), Some(6));
        // Stereo and mono renderers use the host default.
        assert_eq!(choose_channels(2, [2, 6].into_iter()), None);
        assert_eq!(choose_channels(1, [2, 6].into_iter()), None);
        assert_eq!(choose_channels(6, [2].into_iter()), None);
        assert_eq!(choose_channels(6, std::iter::empty()), None);
    }
}
