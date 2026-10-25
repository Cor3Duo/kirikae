//! Pixel-level frame capture for headless checks (`KIRIKAE_CAPTURE`).
//!
//! A black window is indistinguishable from a correct one in the logs, so the
//! runtime can write the frame it is about to present as a binary PPM. The
//! capture is opt-in and touches nothing when the variable is unset: the
//! renderer only reads the image back on the frame that was asked for.
//!
//! The swapchain path copies the swapchain image after the blit (what the
//! window receives); a run without a window copies the render target the guest
//! presents. If there is no such image at all, a black frame is written and the
//! reason is logged, so `tools/frame_diff.py` reports the same black screen the
//! player would see.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Capture {
    prefix: PathBuf,
    frame: u64,
    done: AtomicBool,
}

/// The configured capture, or `None` when `KIRIKAE_CAPTURE` is not set.
pub fn capture() -> Option<&'static Capture> {
    static CAPTURE: OnceLock<Option<Capture>> = OnceLock::new();
    CAPTURE
        .get_or_init(|| {
            let prefix = crate::env_var!("KIRIKAE_CAPTURE")?;
            let frame = crate::env_var!("KIRIKAE_CAPTURE_FRAME")
                .and_then(|v| v.to_str()?.parse().ok())
                .unwrap_or(60)
                .max(1);
            Some(Capture {
                prefix: PathBuf::from(prefix),
                frame,
                done: AtomicBool::new(false),
            })
        })
        .as_ref()
}

/// The capture when the frame being presented is the one it asks for.
///
/// Called before `display::FRAMES` counts the frame, so `KIRIKAE_CAPTURE_FRAME=60`
/// captures the 60th presented frame.
pub fn due() -> Option<&'static Capture> {
    let capture = capture()?;
    let presented = crate::display::FRAMES.load(Ordering::Relaxed);
    (presented.saturating_add(1) == capture.frame).then_some(capture)
}

impl Capture {
    /// The frame number asked for, 1-based.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Takes the single capture slot. Only the first caller on the target
    /// frame writes a file, even if several present paths run for it.
    pub fn claim(&self) -> bool {
        !self.done.swap(true, Ordering::Relaxed)
    }

    pub fn done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// `<prefix>_NNNNNN.ppm`, relative prefixes next to the executable.
    pub fn path(&self, frame: u64) -> PathBuf {
        let base = if self.prefix.is_absolute() {
            self.prefix.clone()
        } else {
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(|dir| dir.join(&self.prefix)))
                .unwrap_or_else(|| self.prefix.clone())
        };
        path_in(&base, frame)
    }

    /// Writes `pixels` (`0xRRGGBB`) and logs the metrics needed to tell a black
    /// frame from a dark one without opening the file.
    pub fn write(
        &self,
        frame: u64,
        width: u32,
        height: u32,
        pixels: &[u32],
    ) -> std::io::Result<PathBuf> {
        let path = self.path(frame);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, ppm(width, height, pixels))?;
        let summary = summarize(width, height, pixels);
        eprintln!(
            "kirikae: capture: frame {frame} {width}x{height} -> {} (mean RGB {:.1} {:.1} {:.1}, {:.2}% non-black)",
            path.display(),
            summary.mean[0],
            summary.mean[1],
            summary.mean[2],
            summary.nonblack,
        );
        Ok(path)
    }

    /// Writes a black frame; the caller logs why there was nothing to read.
    pub fn write_black(&self, frame: u64, width: u32, height: u32) -> std::io::Result<PathBuf> {
        let pixels = vec![0; (width.max(1) as usize) * (height.max(1) as usize)];
        self.write(frame, width.max(1), height.max(1), &pixels)
    }
}

/// Names a capture file next to a stem, stripping a `.ppm`/`.png` suffix so
/// `KIRIKAE_CAPTURE=frames/window.ppm` does not give `window.ppm_000060.ppm`.
fn path_in(base: &Path, frame: u64) -> PathBuf {
    let strip = base
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("ppm") || e.eq_ignore_ascii_case("png"));
    let stem = if strip {
        base.file_stem().map_or(base.as_os_str(), |s| s)
    } else {
        base.as_os_str()
    };
    let mut name = stem.to_os_string();
    name.push(format!("_{frame:06}.ppm"));
    base.with_file_name(name)
}

/// A binary PPM (P6) with the alpha channel dropped.
pub fn ppm(width: u32, height: u32, pixels: &[u32]) -> Vec<u8> {
    let (w, h) = (width.max(1), height.max(1));
    let n = (w as usize) * (h as usize);
    let mut out = Vec::with_capacity(n * 3 + 32);
    out.extend_from_slice(format!("P6\n{w} {h}\n255\n").as_bytes());
    let header = out.len();
    out.resize(header + n * 3, 0);
    for (i, &px) in pixels.iter().take(n).enumerate() {
        out[header + i * 3] = (px >> 16) as u8;
        out[header + i * 3 + 1] = (px >> 8) as u8;
        out[header + i * 3 + 2] = px as u8;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub mean: [f64; 3],
    /// Percentage of pixels with any channel above 8, the tool's default.
    pub nonblack: f64,
}

pub fn summarize(width: u32, height: u32, pixels: &[u32]) -> Summary {
    let n = (width.max(1) as usize) * (height.max(1) as usize);
    let mut sums = [0u64; 3];
    let mut nonblack = 0u64;
    for &px in pixels.iter().take(n) {
        let (r, g, b) = ((px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff);
        sums[0] += r as u64;
        sums[1] += g as u64;
        sums[2] += b as u64;
        if r.max(g).max(b) > 8 {
            nonblack += 1;
        }
    }
    let n = n.max(1) as f64;
    Summary {
        mean: [sums[0] as f64 / n, sums[1] as f64 / n, sums[2] as f64 / n],
        nonblack: 100.0 * nonblack as f64 / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixels(value: &[u32]) -> Vec<u32> {
        value.to_vec()
    }

    #[test]
    fn ppm_header_and_channel_order() {
        let data = ppm(2, 1, &pixels(&[0x11_22_33, 0xaa_bb_cc]));
        assert_eq!(&data[..11], b"P6\n2 1\n255\n");
        assert_eq!(&data[11..], &[0x11, 0x22, 0x33, 0xaa, 0xbb, 0xcc]);
    }

    #[test]
    fn ppm_pads_a_short_image_with_black() {
        let data = ppm(2, 2, &pixels(&[0x01_02_03]));
        assert_eq!(&data[..11], b"P6\n2 2\n255\n");
        assert_eq!(data.len(), 11 + 4 * 3);
        assert_eq!(&data[14..], &[0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// The frame that has just been presented selects the file name, the single
    /// claim prevents a second present path from overwriting it, and the
    /// relative prefix lands next to the executable.
    #[test]
    fn one_capture_per_configured_frame() {
        let capture = Capture {
            prefix: PathBuf::from("captures/window"),
            frame: 60,
            done: AtomicBool::new(false),
        };
        assert!(!capture.done());
        assert!(capture.claim());
        assert!(capture.done());
        assert!(!capture.claim());
        assert_eq!(capture.path(60).file_name().unwrap(), "window_000060.ppm");
        assert!(capture.path(60).is_absolute());
        assert!(
            capture
                .path(60)
                .to_string_lossy()
                .contains("captures/window_000060.ppm")
        );
    }

    #[test]
    fn a_ppm_suffix_is_not_doubled() {
        let base = Path::new("C:/tmp/cap.ppm");
        assert_eq!(path_in(base, 7), Path::new("C:/tmp/cap_000007.ppm"));
        let base = Path::new("C:/tmp/cap");
        assert_eq!(path_in(base, 7), Path::new("C:/tmp/cap_000007.ppm"));
    }

    /// A synthetic swapchain write: pixels in the form `decode_pixels` returns
    /// them, a P6 file out, read back as the same RGB values.
    #[test]
    fn a_synthetic_swapchain_image_is_written_as_ppm() {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "kirikae-capture-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut image = vec![0u32; 4 * 3];
        for (i, px) in image.iter_mut().enumerate() {
            *px = ((i as u32 * 13) & 0xff) << 16 | ((i as u32 * 7) & 0xff) << 8 | (i as u32 * 3);
        }
        let capture = Capture {
            prefix: dir.join("swapchain.png"),
            frame: 3,
            done: AtomicBool::new(false),
        };
        let path = capture.write(3, 4, 3, &image).unwrap();
        assert_eq!(path, dir.join("swapchain_000003.ppm"));
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..11], b"P6\n4 3\n255\n");
        let body = &bytes[11..];
        assert_eq!(body.len(), image.len() * 3);
        for (i, px) in image.iter().enumerate() {
            assert_eq!(
                &body[i * 3..i * 3 + 3],
                &[(px >> 16) as u8, (px >> 8) as u8, *px as u8]
            );
        }
        let summary = summarize(4, 3, &image);
        assert!(summary.nonblack > 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_black_frame_is_reported_as_such() {
        let summary = summarize(2, 2, &[0, 0, 0, 0]);
        assert_eq!(summary.mean, [0.0, 0.0, 0.0]);
        assert_eq!(summary.nonblack, 0.0);
        let summary = summarize(1, 1, &[0x00_00_09]);
        assert_eq!(summary.nonblack, 100.0);
    }
}
