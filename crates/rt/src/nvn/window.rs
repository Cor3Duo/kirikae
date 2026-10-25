//! NVN windows: the textures a title renders a frame into and presents.
//!
//! `nvnWindowBuilderSetTextures` names the swapchain's textures by guest address, `nvnWindowSetCrop` selects the
//! part shown and `nvnQueuePresentTexture(queue, window, index)` presents one of them (`docs/nvn/CONTRACT.md` §8).
//! The host records all three so a present can resolve the replayed image for that texture instead of leaving the
//! swapchain to the driver. The records are read by value at the call, never held as guest pointers.
//!
//! A present that names something the host cannot resolve is not dropped: the driver's own present still ran, so the
//! caller reports it as a fallback and the driver keeps the frame (`docs/nvn/CONTRACT.md` §3, "nothing skipped
//! silently").

/// A window's builder, as `nvnWindowBuilder*` filled it. The texture addresses are guest `NVNtexture*` values.
pub const MAX_TEXTURES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Builder {
    pub textures: [u64; MAX_TEXTURES],
    pub textures_count: u32,
    pub active: u32,
    pub present_interval: u32,
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            textures: [0; MAX_TEXTURES],
            textures_count: 0,
            active: 0,
            present_interval: 1,
        }
    }
}

impl Builder {
    /// `nvnWindowBuilderSetTextures(builder, count, const NVNtexture* const*)`, decoded to addresses.
    pub fn set_textures(&mut self, textures: &[u64]) {
        let count = textures.len().min(MAX_TEXTURES);
        self.textures = [0; MAX_TEXTURES];
        self.textures[..count].copy_from_slice(&textures[..count]);
        self.textures_count = count as u32;
        if self.active == 0 {
            self.active = self.textures_count;
        }
    }
}

/// The crop `nvnWindowSetCrop` selected, in texture pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// A rectangle with a non-zero extent: the source crop and the destination it is shown at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A live window: the builder values plus the mutable tuning the game applies every frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Window {
    pub textures: Vec<u64>,
    pub active: u32,
    pub present_interval: u32,
    pub crop: Option<Crop>,
}

impl Window {
    pub fn from_builder(builder: &Builder) -> Window {
        Window {
            textures: builder.textures[..(builder.textures_count as usize).min(MAX_TEXTURES)]
                .to_vec(),
            active: builder.active,
            present_interval: builder.present_interval,
            crop: None,
        }
    }

    /// The texture `index` names, by its guest address.
    pub fn texture(&self, index: u64) -> Option<u64> {
        self.textures
            .get(index as usize)
            .copied()
            .filter(|&t| t != 0)
    }

    /// The crop inside a `width` x `height` texture. No crop, a zero-sized one or one wholly outside the texture
    /// means the whole texture: a game that crops for handheld still presents its full frame when the crop is not
    /// usable, and a present is never dropped for it.
    pub fn crop_rect(&self, width: u32, height: u32) -> Rect {
        let full = Rect {
            x: 0,
            y: 0,
            width,
            height,
        };
        let Some(crop) = self.crop else {
            return full;
        };
        let x = crop.x.max(0) as u32;
        let y = crop.y.max(0) as u32;
        let width = crop.width.min(width.saturating_sub(x));
        let height = crop.height.min(height.saturating_sub(y));
        if width == 0 || height == 0 {
            return full;
        }
        Rect {
            x,
            y,
            width,
            height,
        }
    }
}

/// The swapchain target a recorded present resolves to. Everything is read under the host lock; the renderer only
/// needs the address, size and crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// Guest address of the window texture's storage (the image the replay rendered into).
    pub cpu: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub crop: Rect,
    /// The submit serial this present must wait for before the image is complete.
    pub serial: u64,
}

/// Why a present could not use the replayed frame. The driver's present is still the one that ran, so the frame is
/// never lost; the caller reports the reason once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    UnknownWindow,
    IndexOutsideWindow,
    UnknownTexture,
    NoHostAddress,
    NoReplayedSubmit,
}

impl Fallback {
    pub fn reason(self) -> &'static str {
        match self {
            Fallback::UnknownWindow => "the window is not in the host table",
            Fallback::IndexOutsideWindow => "the presented index is outside the window's textures",
            Fallback::UnknownTexture => "the window texture is not in the host table",
            Fallback::NoHostAddress => "the window texture has no host address",
            Fallback::NoReplayedSubmit => "no replayed submit since the last present",
        }
    }
}

/// Present counters, printed next to the recording and replay reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Presents the host resolved and handed to the swapchain.
    pub replayed: u64,
    /// Presents left to the driver because the target could not be resolved.
    pub fallback: u64,
    /// Replayed presents with no swapchain (headless or no window surface).
    pub no_swapchain: u64,
    /// Frames added to the global counter because the driver's own present did not count one.
    pub frames: u64,
}

/// Whether this present is the one that must advance the frame counter: the driver's present did not.
pub fn credit_frame(before: u64, after: u64) -> bool {
    after == before
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(textures: &[u64]) -> Window {
        let mut builder = Builder::default();
        builder.set_textures(textures);
        Window::from_builder(&builder)
    }

    #[test]
    fn a_builder_keeps_its_texture_order_and_defaults_active_to_all() {
        let mut builder = Builder::default();
        builder.set_textures(&[0x1000, 0x2000, 0x3000]);
        let window = Window::from_builder(&builder);
        assert_eq!(window.textures, [0x1000, 0x2000, 0x3000]);
        assert_eq!(window.active, 3);
        assert_eq!(window.present_interval, 1);
        assert_eq!(window.texture(1), Some(0x2000));
        assert_eq!(window.texture(3), None);
    }

    #[test]
    fn a_presented_index_resolves_its_texture() {
        let pair = window(&[0x1000, 0x2000]);
        assert_eq!(pair.texture(0), Some(0x1000));
        assert_eq!(pair.texture(1), Some(0x2000));
        assert_eq!(pair.texture(2), None);
        // A zero slot is not a texture, even when the array is that long.
        let gap = window(&[0x1000, 0, 0x3000]);
        assert_eq!(gap.texture(1), None);
        assert_eq!(gap.texture(2), Some(0x3000));
    }

    #[test]
    fn a_crop_is_clamped_to_the_texture_and_an_empty_one_means_all_of_it() {
        let mut w = window(&[0x1000]);
        assert_eq!(
            w.crop_rect(1920, 1080),
            Rect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080
            }
        );
        // Handheld: 1280x720 inside a 1920x1080 window texture, centred at 320,180.
        w.crop = Some(Crop {
            x: 320,
            y: 180,
            width: 1280,
            height: 720,
        });
        assert_eq!(
            w.crop_rect(1920, 1080),
            Rect {
                x: 320,
                y: 180,
                width: 1280,
                height: 720
            }
        );
        // A crop that reaches past the edge is cut at the edge, not dropped.
        w.crop = Some(Crop {
            x: 1800,
            y: 1000,
            width: 400,
            height: 400,
        });
        assert_eq!(
            w.crop_rect(1920, 1080),
            Rect {
                x: 1800,
                y: 1000,
                width: 120,
                height: 80
            }
        );
        // A crop wholly outside the texture, or a negative origin with no size, is the full frame.
        w.crop = Some(Crop {
            x: 2000,
            y: 0,
            width: 100,
            height: 100,
        });
        assert_eq!(w.crop_rect(1920, 1080).width, 1920);
        w.crop = Some(Crop {
            x: -8,
            y: -8,
            width: 0,
            height: 0,
        });
        assert_eq!(w.crop_rect(1920, 1080).width, 1920);
        assert_eq!(w.crop_rect(1920, 1080).height, 1080);
    }

    #[test]
    fn a_fallback_names_its_cause() {
        assert!(Fallback::UnknownWindow.reason().contains("window"));
        assert!(Fallback::UnknownTexture.reason().contains("texture"));
        assert!(Fallback::NoReplayedSubmit.reason().contains("submit"));
    }

    #[test]
    fn a_frame_is_credited_only_when_the_driver_did_not_count_it() {
        assert!(credit_frame(430, 430));
        assert!(!credit_frame(430, 431));
    }
}
