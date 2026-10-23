use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

struct Shared {
    queue: Mutex<VecDeque<[f32; 2]>>,
    alive: AtomicBool,
    capacity: usize,
    dry: AtomicU64,
}

pub struct Output {
    shared: Arc<Shared>,
    pub device: String,
}

impl Output {
    pub fn open(rate: u32) -> Result<Output, String> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            alive: AtomicBool::new(true),
            capacity: rate as usize / 2,
            dry: AtomicU64::new(0),
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let s = shared.clone();
        std::thread::Builder::new()
            .name("kirikae-audio-out".into())
            .spawn(move || {
                let stream = match build(rate, s.clone()) {
                    Ok((stream, device)) => {
                        let _ = tx.send(Ok(device));
                        stream
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                while s.alive.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                drop(stream);
            })
            .map_err(|e| e.to_string())?;
        let device = rx.recv().map_err(|e| e.to_string())??;
        Ok(Output { shared, device })
    }

    pub fn queued(&self) -> usize {
        self.shared.queue.lock().unwrap().len()
    }

    pub fn dry_frames(&self) -> u64 {
        self.shared.dry.load(Ordering::Relaxed)
    }

    pub fn push(&self, frames: &[[f32; 2]]) {
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

fn build(rate: u32, shared: Arc<Shared>) -> Result<(cpal::Stream, String), String> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let format = supported.sample_format();
    let mut config: cpal::StreamConfig = supported.config();
    let takes_rate = device.supported_output_configs().is_ok_and(|mut c| {
        c.any(|c| c.channels() == config.channels && c.sample_format() == format && (c.min_sample_rate()..=c.max_sample_rate()).contains(&rate))
    });
    if takes_rate {
        config.sample_rate = rate;
    }
    let step = rate as f32 / config.sample_rate as f32;
    let name = device.description().map(|d| d.to_string()).unwrap_or_else(|_| "default device".into());
    let description = format!("{name}, {} Hz, {} channels, {format}", config.sample_rate, config.channels);
    let stream = match format {
        cpal::SampleFormat::I16 => stream::<i16>(&device, &config, shared, step),
        cpal::SampleFormat::U16 => stream::<u16>(&device, &config, shared, step),
        cpal::SampleFormat::I32 => stream::<i32>(&device, &config, shared, step),
        _ => stream::<f32>(&device, &config, shared, step),
    }?;
    stream.play().map_err(|e| e.to_string())?;
    Ok((stream, description))
}

struct Resampler {
    position: f32,
    previous: [f32; 2],
    next: [f32; 2],
}

impl Resampler {
    fn next(&mut self, q: &mut VecDeque<[f32; 2]>, step: f32, dry: &mut u64) -> [f32; 2] {
        while self.position >= 1.0 {
            self.previous = self.next;
            self.next = q.pop_front().unwrap_or_else(|| {
                *dry += 1;
                [0.0; 2]
            });
            self.position -= 1.0;
        }
        let t = self.position;
        self.position += step;
        [0, 1].map(|c| self.previous[c] + (self.next[c] - self.previous[c]) * t)
    }
}

fn stream<T: cpal::SizedSample + cpal::FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig, shared: Arc<Shared>, step: f32) -> Result<cpal::Stream, String> {
    let channels = usize::from(config.channels).max(1);
    let mut resampler = Resampler { position: 1.0, previous: [0.0; 2], next: [0.0; 2] };
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                let mut q = shared.queue.lock().unwrap();
                let mut dry = 0;
                for frame in data.chunks_mut(channels) {
                    let [l, r] = resampler.next(&mut q, step, &mut dry);
                    if let [only] = frame {
                        *only = T::from_sample((l + r) * 0.5);
                        continue;
                    }
                    for (k, s) in frame.iter_mut().enumerate() {
                        *s = T::from_sample(match k {
                            0 => l,
                            1 => r,
                            _ => 0.0,
                        });
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
    frames: u32,
}

impl WavDump {
    pub fn create(path: &std::path::Path, rate: u32) -> std::io::Result<WavDump> {
        let mut dump = WavDump { file: std::io::BufWriter::new(std::fs::File::create(path)?), rate, frames: 0 };
        dump.header()?;
        Ok(dump)
    }

    fn header(&mut self) -> std::io::Result<()> {
        use std::io::{Seek, Write};
        let data = self.frames * 4;
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + data).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&2u16.to_le_bytes());
        h.extend_from_slice(&self.rate.to_le_bytes());
        h.extend_from_slice(&(self.rate * 4).to_le_bytes());
        h.extend_from_slice(&4u16.to_le_bytes());
        h.extend_from_slice(&16u16.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&data.to_le_bytes());
        let end = self.file.stream_position()?;
        self.file.seek(std::io::SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.seek(std::io::SeekFrom::Start(end.max(44)))?;
        Ok(())
    }

    pub fn write(&mut self, frames: &[[f32; 2]]) -> std::io::Result<()> {
        use std::io::Write;
        for f in frames {
            for s in f {
                self.file.write_all(&((s * 32767.0) as i16).to_le_bytes())?;
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
        let mut q: VecDeque<[f32; 2]> = [[1.0, -1.0], [3.0, -3.0]].into_iter().collect();
        let mut r = Resampler { position: 1.0, previous: [0.0; 2], next: [0.0; 2] };
        let mut dry = 0;
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [0.0, 0.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [0.5, -0.5]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [1.0, -1.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [2.0, -2.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [3.0, -3.0]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [1.5, -1.5]);
        assert_eq!(r.next(&mut q, 0.5, &mut dry), [0.0, 0.0]);
        assert_eq!(dry, 2);
    }
}
