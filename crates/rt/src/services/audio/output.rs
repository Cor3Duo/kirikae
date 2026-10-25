use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub const MAX_CHANNELS: usize = 6;
pub type Frame = [f32; MAX_CHANNELS];

/// Standard 5.1 fold-down: front, centre, LFE, back. Matches Ryujinx's
/// `DefaultSurroundToStereoCoefficients` and the `audren` sink default.
pub const DEFAULT_DOWNMIX: [f32; 4] = [1.0, 0.707, 0.251, 0.707];

struct Shared {
    queue: Mutex<VecDeque<Frame>>,
    downmix: Mutex<[f32; 4]>,
    alive: AtomicBool,
    capacity: usize,
    dry: AtomicU64,
    source: AtomicU64,
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

fn build(rate: u32, shared: Arc<Shared>, source_channels: usize) -> Result<Built, String> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let default = device.default_output_config().map_err(|e| e.to_string())?;
    let format = default.sample_format();
    let mut supported = default;

    // Prefer a real 6-channel stream when the renderer produces surround.
    if source_channels >= MAX_CHANNELS
        && let Ok(ranges) = device.supported_output_configs()
    {
        let mut fallback = None;
        for range in ranges {
            if range.channels() != MAX_CHANNELS as u16 || range.sample_format() != format {
                continue;
            }
            if (range.min_sample_rate()..=range.max_sample_rate()).contains(&rate) {
                supported = range.with_sample_rate(rate);
                fallback = None;
                break;
            }
            if fallback.is_none() {
                fallback = Some(range.with_sample_rate(range.max_sample_rate()));
            }
        }
        if let Some(config) = fallback {
            supported = config;
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
/// The 6-channel layout is the documented 5.1 one (`FL FR FC LFE BL BR`); the
/// 4-channel layout is quad (`FL FR BL BR`), which is the only public
/// convention for it; 3 and 5 channels are prefix layouts.
pub fn downmix(frame: &Frame, source: usize, coefficients: [f32; 4]) -> [f32; 2] {
    let k = coefficients;
    let s = |i: usize| frame.get(i).copied().unwrap_or(0.0);
    match source.clamp(1, MAX_CHANNELS) {
        1 => [s(0); 2],
        2 => [s(0), s(1)],
        3 => [s(0) + s(2) * k[1], s(1) + s(2) * k[1]],
        4 => [s(0) + s(4) * k[3], s(1) + s(5) * k[3]],
        5 => [
            s(0) + s(2) * k[1] + s(4) * k[3],
            s(1) + s(2) * k[1] + s(5) * k[3],
        ],
        _ => [
            s(0) * k[0] + s(2) * k[1] + s(3) * k[2] + s(4) * k[3],
            s(1) * k[0] + s(2) * k[1] + s(3) * k[2] + s(5) * k[3],
        ],
    }
}

/// Map a renderer frame to a device frame. `frame[k]` is the sample for device
/// channel `k`; channels the source does not have stay silent.
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
    if device >= source {
        out[..source].copy_from_slice(&frame[..source]);
        return out;
    }
    let [left, right] = downmix(frame, source, coefficients);
    if device == 1 {
        out[0] = (left + right) * 0.5;
    } else {
        out[0] = left;
        out[1] = right;
    }
    out
}

struct Resampler {
    position: f32,
    previous: Frame,
    next: Frame,
}

impl Resampler {
    fn next(&mut self, q: &mut VecDeque<Frame>, step: f32, dry: &mut u64) -> Frame {
        while self.position >= 1.0 {
            self.previous = self.next;
            self.next = q.pop_front().unwrap_or_else(|| {
                *dry += 1;
                [0.0; MAX_CHANNELS]
            });
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
                for frame in data.chunks_mut(device_channels) {
                    let renderer = resampler.next(&mut q, step, &mut dry);
                    let mapped = map_frame(&renderer, source, device_channels, coefficients);
                    for (k, sample) in frame.iter_mut().enumerate() {
                        *sample = T::from_sample(mapped.get(k).copied().unwrap_or(0.0));
                    }
                }
                shared.dry.fetch_add(dry, Ordering::Relaxed);
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
        let mut q: VecDeque<Frame> = [
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, -3.0, 0.0, 0.0, 0.0, 0.0],
        ]
        .into_iter()
        .collect();
        let mut r = Resampler {
            position: 1.0,
            previous: [0.0; MAX_CHANNELS],
            next: [0.0; MAX_CHANNELS],
        };
        let mut dry = 0;
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [0.0, 0.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [0.5, -0.5]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [1.0, -1.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [2.0, -2.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [3.0, -3.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [1.5, -1.5]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry)[..2], [0.0, 0.0]);
        assert_eq!(dry, 2);
    }

    #[test]
    fn six_channel_downmix_uses_the_documented_coefficients() {
        let frame = [0.5, -0.5, 0.25, 0.125, 0.75, -0.75];
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
    fn four_and_two_channel_fold_down_is_explicit() {
        let quad = [1.0, -1.0, 0.0, 0.0, 0.5, -0.5];
        let [left, right] = downmix(&quad, 4, DEFAULT_DOWNMIX);
        assert!((left - (1.0 + 0.5 * 0.707)).abs() < 1e-6);
        assert!((right + (1.0 + 0.5 * 0.707)).abs() < 1e-6);
        let stereo = [1.0, -1.0, 0.9, 0.9, 0.9, 0.9];
        assert_eq!(downmix(&stereo, 2, DEFAULT_DOWNMIX), [1.0, -1.0]);
        assert_eq!(downmix(&[0.25; 6], 1, DEFAULT_DOWNMIX), [0.25, 0.25]);
    }

    #[test]
    fn a_six_channel_device_receives_every_channel() {
        let frame = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(map_frame(&frame, 6, 6, DEFAULT_DOWNMIX), frame);
        let stereo = map_frame(&frame, 6, 2, DEFAULT_DOWNMIX);
        assert_eq!(stereo[..2], downmix(&frame, 6, DEFAULT_DOWNMIX));
        assert_eq!(stereo[2..], [0.0; 4]);
        let mono = map_frame(&frame, 6, 1, DEFAULT_DOWNMIX);
        let [left, right] = downmix(&frame, 6, DEFAULT_DOWNMIX);
        assert!((mono[0] - (left + right) * 0.5).abs() < 1e-6);
    }
}
