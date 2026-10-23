use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::process::Process;

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

pub static FRAMES: AtomicU64 = AtomicU64::new(0);
pub static PACED_NS: AtomicU64 = AtomicU64::new(0);

pub fn uncapped_fps(frames: u64, secs: f64, paced: u64) -> f64 {
    let busy = secs - paced as f64 / 1e9;
    if busy > 0.0 { frames as f64 / busy } else { 0.0 }
}
static LONGEST: AtomicU64 = AtomicU64::new(0);
static LAST_FRAME: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

fn note_frame() {
    let now = std::time::Instant::now();
    if let Some(last) = LAST_FRAME.lock().unwrap().replace(now) {
        LONGEST.fetch_max((now - last).as_nanos() as u64, Ordering::Relaxed);
        if now - last > std::time::Duration::from_millis(100) && std::env::var("KIRIKAE_STATS").is_ok_and(|v| v == "2") {
            let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            eprintln!("kirikae: hitch: frame {} took {:.0} ms, ending at {:.3}", FRAMES.load(Ordering::Relaxed), (now - last).as_secs_f64() * 1e3, unix.as_secs_f64());
        }
    }
}

pub fn take_longest_frame() -> std::time::Duration {
    std::time::Duration::from_nanos(LONGEST.swap(0, Ordering::Relaxed))
}

pub enum Shown {
    Frame(Frame),
    Gpu,
}

static PROXY: OnceLock<Option<Mutex<EventLoopProxy<Shown>>>> = OnceLock::new();

static WINDOW: OnceLock<(isize, isize)> = OnceLock::new();

pub fn headless() -> bool {
    crate::env_var!("KIRIKAE_HEADLESS").is_some()
}

pub fn window_handle() -> Option<(isize, isize)> {
    WINDOW.get().copied()
}

pub fn dump_due() -> bool {
    dumps(FRAMES.load(Ordering::Relaxed) + 1)
}

fn dumps(next: u64) -> bool {
    static EVERY: OnceLock<(u64, u64, u64)> = OnceLock::new();
    let (every, from, count) = *EVERY.get_or_init(|| {
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
        (num("KIRIKAE_FRAME_DUMP_EVERY").unwrap_or(60).max(1), num("KIRIKAE_FRAME_DUMP_FROM").unwrap_or(0), num("KIRIKAE_FRAME_DUMP_COUNT").unwrap_or(u64::MAX))
    });
    next >= from
        && (next - from) % every == if from == 0 { 1 % every } else { 0 }
        && (next - from) / every < count
        && crate::env_var!("KIRIKAE_FRAME_DUMP").is_some()
}

pub fn presented(p: &Process, dump: Option<&Frame>) {
    let n = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    note_frame();
    if n == 1 {
        p.warn_once("display: first frame presented (Vulkan swapchain)".to_string());
    }
    if let Some(frame) = dump {
        write_dump(p, n, frame);
    }
    crate::usage::frame(dump.map_or(0, |f| f.width), dump.map_or(0, |f| f.height), dump.map(|f| &f.pixels[..]));
    if let Some(Some(proxy)) = PROXY.get() {
        let _ = proxy.lock().unwrap().send_event(Shown::Gpu);
    }
}

fn write_dump(p: &Process, n: u64, frame: &Frame) {
    if dumps(n)
        && let Some(dir) = crate::env_var!("KIRIKAE_FRAME_DUMP")
    {
        let path = std::path::Path::new(&dir).join(format!("frame{n:06}.png"));
        if let Err(e) = write_png(&path, frame.width, frame.height, &frame.pixels) {
            p.warn_once(format!("display: cannot write {}: {e}", path.display()));
        }
    }
}

pub fn start(p: &'static Process, title: String) {
    PROXY.get_or_init(|| {
        if headless() {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("kirikae-display".into()).spawn(move || {
            let mut builder = EventLoop::<Shown>::with_user_event();
            #[cfg(windows)]
            winit::platform::windows::EventLoopBuilderExtWindows::with_any_thread(&mut builder, true);
            let event_loop = match builder.build() {
                Ok(l) => l,
                Err(e) => {
                    p.warn_once(format!("display: no window ({e})"));
                    let _ = tx.send(None);
                    return;
                }
            };
            let _ = tx.send(Some(event_loop.create_proxy()));
            let mut app = App { p, title, window: None, surface: None, frame: None, gpu: false, shown: 0, since: Instant::now(), counted: (0, 0) };
            if let Err(e) = event_loop.run_app(&mut app) {
                p.warn_once(format!("display: event loop failed ({e})"));
            }
        });
        if spawned.is_err() {
            return None;
        }
        rx.recv().ok().flatten().map(Mutex::new)
    });
}

pub fn present(p: &Process, frame: Frame) {
    let n = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    note_frame();
    if n == 1 {
        p.warn_once(format!("display: first frame presented ({}x{})", frame.width, frame.height));
    }
    write_dump(p, n, &frame);
    crate::usage::frame(frame.width, frame.height, Some(&frame.pixels));
    if let Some(Some(proxy)) = PROXY.get() {
        let _ = proxy.lock().unwrap().send_event(Shown::Frame(frame));
    }
}

pub(crate) fn write_png(path: &std::path::Path, width: u32, height: u32, pixels: &[u32]) -> std::io::Result<()> {
    let n = (width * height) as usize;
    let mut rgb = Vec::with_capacity(n * 3);
    for px in pixels.iter().take(n) {
        rgb.extend_from_slice(&[(px >> 16) as u8, (px >> 8) as u8, *px as u8]);
    }
    rgb.resize(n * 3, 0);
    std::fs::write(path, crate::usage::png_level(width, height, &rgb, 1))
}

struct App {
    p: &'static Process,
    title: String,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    frame: Option<Frame>,
    gpu: bool,
    shown: u64,
    since: Instant,
    counted: (u64, u64),
}

impl ApplicationHandler<Shown> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title(&self.title).with_inner_size(LogicalSize::new(1280.0, 720.0));
        let window = match el.create_window(attrs) {
            Ok(w) => Rc::new(w),
            Err(e) => {
                self.p.warn_once(format!("display: cannot create the window ({e})"));
                el.exit();
                return;
            }
        };
        #[cfg(windows)]
        if let Ok(h) = winit::raw_window_handle::HasWindowHandle::window_handle(&*window)
            && let winit::raw_window_handle::RawWindowHandle::Win32(h) = h.as_raw()
        {
            let _ = WINDOW.set((h.hwnd.get(), h.hinstance.map_or(0, |i| i.get())));
        }
        self.window = Some(window);
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, shown: Shown) {
        match shown {
            Shown::Frame(frame) if !self.gpu => {
                self.frame = Some(frame);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            Shown::Frame(_) => {}
            Shown::Gpu => {
                self.gpu = true;
                self.frame = None;
                self.shown += 1;
                self.update_title();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.p.terminate(0);
                el.exit();
            }
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                    crate::input::key(code, event.state.is_pressed());
                }
            }
            WindowEvent::Focused(false) => crate::input::release_all(),
            WindowEvent::Resized(_) => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
    fn draw(&mut self) {
        if self.gpu {
            return;
        }
        if self.surface.is_none()
            && let Some(window) = &self.window
        {
            match softbuffer::Context::new(window.clone()).and_then(|ctx| softbuffer::Surface::new(&ctx, window.clone())) {
                Ok(s) => self.surface = Some(s),
                Err(e) => self.p.warn_once(format!("display: no drawing surface ({e})")),
            }
        }
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else { return };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else { return };
        if surface.resize(w, h).is_err() {
            return;
        }
        let Ok(mut buf) = surface.buffer_mut() else { return };
        let (dw, dh) = (size.width as usize, size.height as usize);
        match &self.frame {
            Some(f) if f.width > 0 && f.height > 0 => {
                let (sw, sh) = (f.width as usize, f.height as usize);
                for y in 0..dh {
                    let row = &f.pixels[(y * sh / dh) * sw..][..sw];
                    let out = &mut buf[y * dw..][..dw];
                    for (x, px) in out.iter_mut().enumerate() {
                        *px = row[x * sw / dw];
                    }
                }
                self.shown += 1;
            }
            _ => buf.fill(0),
        }
        let _ = buf.present();
        self.update_title();
    }

    fn update_title(&mut self) {
        let Some(window) = &self.window else { return };
        let elapsed = self.since.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            let fps = FRAMES.load(Ordering::Relaxed);
            let mut title = format!("{} - {} frames - {:.0} fps", self.title, fps, self.shown as f64 / elapsed);
            let paced = PACED_NS.load(Ordering::Relaxed);
            let (frames0, paced0) = std::mem::replace(&mut self.counted, (fps, paced));
            if paced > paced0 {
                title += &format!(" (sem limite: {:.0} fps)", uncapped_fps(fps - frames0, elapsed, paced - paced0));
            }
            let gaps = crate::usage::new_events();
            if gaps > 0 {
                title += &format!(" - {gaps} {} de cobertura", if gaps == 1 { "novo item" } else { "novos itens" });
            }
            window.set_title(&title);
            self.shown = 0;
            self.since = Instant::now();
        }
    }
}
