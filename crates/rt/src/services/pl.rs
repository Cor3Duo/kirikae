use std::sync::Arc;

use crate::cpu::Cpu;
use crate::kernel::SharedMemory;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const SHARED_SIZE: u64 = 0x110_0000;
const FONT_TYPES: usize = 6;
const KEY: u32 = 0x4962_1806;
const MAGIC: u32 = 0x1802_9a7f;

struct Fonts {
    handle: u32,
    _memory: Arc<SharedMemory>,
    fonts: [(u32, u32); FONT_TYPES],
}

fn host_font() -> Option<Vec<u8>> {
    let mut candidates: Vec<std::path::PathBuf> = std::env::var_os("KIRIKAE_SHARED_FONT").map(Into::into).into_iter().collect();
    if let Some(windir) = std::env::var_os("WINDIR") {
        let fonts = std::path::Path::new(&windir).join("Fonts");
        candidates.extend(["segoeui.ttf", "arial.ttf"].map(|f| fonts.join(f)));
    }
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        .map(Into::into),
    );
    candidates.extend(fontconfig_sans());
    candidates.iter().find_map(|p| std::fs::read(p).ok())
}

fn fontconfig_sans() -> Option<std::path::PathBuf> {
    let out = std::process::Command::new("fc-match")
        .args(["--format=%{file}", "sans-serif:fontformat=TrueType"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let path = String::from_utf8(out.stdout).ok()?;
    (out.status.success() && path.ends_with(".ttf")).then(|| path.into())
}

fn fonts(c: &Cpu) -> Arc<Fonts> {
    let p = c.process();
    p.service_state("pl", || {
        let (memory, handle) = p.create_shared_memory(SHARED_SIZE);
        let font = host_font().unwrap_or_else(|| {
            p.warn_once("pl: no host TrueType font found; shared fonts are empty (set KIRIKAE_SHARED_FONT)".into());
            Vec::new()
        });
        let stride = (font.len() as u64 + 8).next_multiple_of(0x1000);
        let copies = if stride * FONT_TYPES as u64 <= SHARED_SIZE { FONT_TYPES } else { 1 };
        let size = font.len() as u32;
        for k in 0..copies as u64 {
            let at = k * stride;
            memory.write(&p.memory, at, &(MAGIC ^ KEY).to_le_bytes());
            memory.write(&p.memory, at + 4, &(size ^ KEY).to_be_bytes());
            memory.write(&p.memory, at + 8, &font);
        }
        let fonts = std::array::from_fn(|t| (((t % copies) as u64 * stride + 8) as u32, size));
        Fonts { handle, _memory: memory, fonts }
    })
}

pub struct Pl;

impl Service for Pl {
    fn name(&self) -> String {
        "pl:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let f = fonts(c);
        let font = |t: u32| f.fonts.get(t as usize).copied().unwrap_or((0, 0));
        match req.cmd {
            0 => Reply::ok(),
            1 => Reply::ok().u32(1),
            2 => Reply::ok().u32(font(req.u32(0)).1),
            3 => Reply::ok().u32(font(req.u32(0)).0),
            4 => Reply::ok().copy_handle(f.handle),
            5 => {
                let p = c.process();
                let order: Vec<u32> = (0..FONT_TYPES as u32).collect();
                let lists: [Vec<u32>; 3] = [order.clone(), order.iter().map(|&t| font(t).0).collect(), order.iter().map(|&t| font(t).1).collect()];
                for (i, list) in lists.iter().enumerate() {
                    if let Some(b) = req.out_buf(i) {
                        let bytes: Vec<u8> = list.iter().flat_map(|v| v.to_le_bytes()).collect();
                        p.memory.write(b.addr, &bytes[..bytes.len().min(b.size as usize)]);
                    }
                }
                Reply::ok().u8(1).u32(FONT_TYPES as u32)
            }
            _ => unimplemented("pl:u", c, req),
        }
    }
}
