use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use super::shadow::Draw;
use super::state::{Field, Kind, Learned};
use crate::gpu::threed::reg;

pub static ON: AtomicBool = AtomicBool::new(false);

const DISTINCT: usize = 64;

const ATTRIBUTES: usize = 16;

#[derive(Default)]
struct Coverage {
    prev: Vec<u32>,
    changes: Vec<u64>,
    values: Vec<BTreeSet<u32>>,
    draws: u64,
}

static COVERAGE: Mutex<Option<Coverage>> = Mutex::new(None);

pub fn observe(regs: &[u32]) {
    let mut guard = COVERAGE.lock().unwrap();
    let c = guard.get_or_insert_with(Coverage::default);
    if c.prev.len() != regs.len() {
        c.prev = regs.to_vec();
        c.changes = vec![0; regs.len()];
        c.values = vec![BTreeSet::new(); regs.len()];
    }
    c.draws += 1;
    for (i, (&now, prev)) in regs.iter().zip(c.prev.iter_mut()).enumerate() {
        if now != *prev {
            c.changes[i] += 1;
            *prev = now;
        }
        if c.values[i].len() < DISTINCT {
            c.values[i].insert(now);
        }
    }
}

pub fn report() {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    report_checks();
    let predicted = predicted_registers();
    let guard = COVERAGE.lock().unwrap();
    let Some(c) = guard.as_ref() else { return };
    let mut rows: Vec<(u64, usize)> = c.changes.iter().enumerate().filter(|(i, n)| **n > 0 && !predicted.contains(i)).map(|(i, n)| (*n, i)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let constant = c.values.iter().filter(|v| v.len() == 1).count();
    eprintln!(
        "kirikae:         nvn encode: {} draws, {} registers change between draws and are not predicted yet, {} keep one value",
        c.draws,
        rows.len(),
        constant
    );
    let line = |&(n, i): &(u64, usize)| {
        let v = &c.values[i];
        let sample: Vec<String> = v.iter().take(6).map(|x| format!("{x:#x}")).collect();
        format!("{i:#05x} changed {n} times, {}{} values: {}", v.len(), if v.len() >= DISTINCT { "+" } else { "" }, sample.join(" "))
    };
    for r in rows.iter().take(12) {
        eprintln!("kirikae:           {}", line(r));
    }
    if let Ok(exe) = std::env::current_exe() {
        let text: String = rows.iter().map(|r| line(r) + "\n").collect();
        let _ = std::fs::write(exe.with_extension("nvn-regs.txt"), text);
    }
}

pub type Prediction = Vec<(usize, u32, u32)>;

const MACRO_PROGRAM: usize = 0xd1d;

const MACRO_DEPTH_TARGET: usize = 0xd29;

pub fn predict(d: &Draw, learned: &Learned, regs: &[u32], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Prediction {
    predict_inner(d, learned, regs, read).0
}

pub fn predict_complete(d: &Draw, learned: &Learned, regs: &[u32], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> Result<Prediction, String> {
    let (p, missing) = predict_inner(d, learned, regs, read);
    match missing {
        None => Ok(p),
        Some(m) => Err(m),
    }
}

fn predict_inner(d: &Draw, learned: &Learned, regs: &[u32], read: &dyn Fn(u64, u64) -> Option<Vec<u8>>) -> (Prediction, Option<String>) {
    let mut missing = None;
    let mut out: Prediction = Vec::with_capacity(256);
    let mut put = |r: usize, mask: u32, value: u32| out.push((r, mask, value & mask));
    let mut viewport = [0i64; 4];
    let mut scissor = [0i64; 4];
    let mut cull = None;
    let fields: crate::gpu::fxhash::FastMap<Field, u64> = d.state.iter().copied().collect();
    let raw = |k: Kind, i: usize| fields.get(&Field(k, i as u8)).copied();
    let field = |k: Kind, i: usize| raw(k, i).and_then(|v| learned.usual(k, v));
    let (attribs, streams) = (raw(Kind::AttribCount, 0).map_or(32, |v| v as usize), raw(Kind::StreamCount, 0).map_or(16, |v| v as usize));
    let offset = field(Kind::PolygonOffsetEnables, 0).is_some_and(|r| r != 0);
    let stencil_test = field(Kind::StencilTest, 0) == Some(1);
    let depth_test = field(Kind::DepthTest, 0) == Some(1);
    let bound = |i: usize| d.surfaces.iter().any(|s| s.0 as usize == i);
    let blended = |i: usize| field(Kind::BlendEnable, i) == Some(1);
    let alpha = |i: usize| field(Kind::ChannelMask, i).is_some_and(|m| m & 8 != 0);
    let disabled = |i: usize| raw(Kind::AttribFormat, i) == Some(0);
    for i in 0..ATTRIBUTES {
        let constant = i >= attribs || disabled(i);
        put(0x458 + i, 0x40, (constant as u32) << 6);
    }
    for s in 0..16 {
        put(0x700 + 4 * s, 0x1000, ((s < streams) as u32) << 12);
    }
    for &(Field(kind, i), v) in &d.state {
        let i = i as usize;
        match kind {
            Kind::Viewport => viewport[i] = v as i64,
            Kind::Scissor => scissor[i] = v as i64,
            _ => {}
        }
        let stale = match kind {
            Kind::AttribFormat | Kind::AttribOffset | Kind::AttribStream => i >= attribs || disabled(i),
            Kind::StreamStride | Kind::StreamDivisor => i >= streams,
            _ => false,
        };
        if stale {
            continue;
        }
        let unused = match kind {
            Kind::ChannelMask | Kind::BlendEnable => !bound(i),
            Kind::DepthWrite | Kind::DepthFunc => !depth_test,
            Kind::StencilRef | Kind::StencilValueMask | Kind::StencilMask | Kind::StencilFunc | Kind::StencilFail | Kind::StencilDepthFail | Kind::StencilPass => !stencil_test,
            Kind::BlendEquation | Kind::BlendSrc | Kind::BlendDst => !bound(i) || !blended(i),
            Kind::BlendEquationAlpha | Kind::BlendSrcAlpha | Kind::BlendDstAlpha => !bound(i) || !blended(i) || !alpha(i),
            Kind::PolygonOffset => !offset,
            _ => false,
        };
        let Some(r) = learned.usual(kind, v) else {
            if unused {
                continue;
            }
            if !matches!(kind, Kind::Viewport | Kind::Scissor | Kind::AttribCount | Kind::StreamCount) {
                missing.get_or_insert_with(|| format!("{kind:?} {v:#x}"));
            }
            continue;
        };
        let r = r as u32;
        match kind {
            Kind::ChannelMask => put(0x680 + i, 0x1111, (r & 1) | (r & 2) << 3 | (r & 4) << 6 | (r & 8) << 9),
            Kind::BlendEnable => put(0x4d8 + i, 1, r),
            Kind::BlendEquation => put(0x781 + 8 * i, u32::MAX, r),
            Kind::BlendSrc => put(0x782 + 8 * i, u32::MAX, r),
            Kind::BlendDst => put(0x783 + 8 * i, u32::MAX, r),
            Kind::BlendEquationAlpha => put(0x784 + 8 * i, u32::MAX, r),
            Kind::BlendSrcAlpha => put(0x785 + 8 * i, u32::MAX, r),
            Kind::BlendDstAlpha => put(0x786 + 8 * i, u32::MAX, r),
            Kind::DepthTest => put(0x4b3, 1, r),
            Kind::DepthWrite => put(0x4ba, 1, r),
            Kind::DepthFunc => put(0x4c3, u32::MAX, r),
            Kind::StencilTest => put(0x4e0, 1, r),
            Kind::StencilFail => put([0x4e1, 0x566][i], u32::MAX, r),
            Kind::StencilDepthFail => put([0x4e2, 0x567][i], u32::MAX, r),
            Kind::StencilPass => put([0x4e3, 0x568][i], u32::MAX, r),
            Kind::StencilFunc => put([0x4e4, 0x569][i], u32::MAX, r),
            Kind::CullFace => cull = Some(r),
            Kind::FrontFace => put(0x647, u32::MAX, r),
            Kind::PolygonOffsetEnables => {
                for k in 0..3 {
                    put(0x370 + k, 1, r >> k);
                }
            }
            Kind::AttribFormat => put(0x458 + i, 0xffe0_0000, r << 21),
            Kind::AttribOffset => put(0x458 + i, 0x1f_ff80, r << 7),
            Kind::AttribStream => put(0x458 + i, 0x1f, r),
            Kind::StreamStride => put(0x700 + 4 * i, 0xfff, r),
            Kind::StreamDivisor => {
                put(0x620 + i, 1, (r != 0) as u32);
                if r != 0 {
                    put(0x703 + 4 * i, u32::MAX, r);
                }
            }
            Kind::BlendColor => put(0x4c7 + i, u32::MAX, r),
            Kind::PolygonOffset => put([0x56f, 0x55b, 0x61f][i], u32::MAX, r),
            Kind::StencilRef => put([0x4e5, 0x3d5][i], 0xff, r),
            Kind::StencilValueMask => put([0x4e6, 0x3d7][i], 0xff, r),
            Kind::StencilMask => put([0x4e7, 0x3d6][i], 0xff, r),
            Kind::DepthClamp => put(reg::VIEW_VOLUME_CLIP_CTRL, u32::MAX, r),
            Kind::RasterizerDiscard => put(0xdf, u32::MAX, r),
            Kind::PointSize => put(0x546, u32::MAX, r),
            Kind::LineWidth => put(0x4ec, u32::MAX, r),
            Kind::DepthBoundsEnable => put(0x66f, 1, r),
            Kind::DepthBoundsMin => put(0x3e7, u32::MAX, r),
            Kind::DepthBoundsMax => put(0x3e8, u32::MAX, r),
            Kind::Viewport | Kind::Scissor | Kind::TargetFormat | Kind::BlendRegister | Kind::AttribCount | Kind::StreamCount | Kind::DepthFormat => {}
        }
    }
    if let Some(c) = cull {
        put(0x646, 1, (c != 0) as u32);
        if c != 0 {
            put(0x648, u32::MAX, c);
        }
    }
    if viewport[2] != 0 {
        let (x, y, w, h) = (viewport[0] as f32, viewport[1] as f32, viewport[2] as f32, viewport[3] as f32);
        put(0x280, u32::MAX, (w / 2.0).to_bits());
        put(0x281, 0x7fff_ffff, (h / 2.0).to_bits());
        put(0x283, u32::MAX, (x + w / 2.0).to_bits());
        put(0x284, u32::MAX, (y + h / 2.0).to_bits());
    }
    if scissor[2] != 0 {
        let (x, y) = (scissor[0] as u32, scissor[1] as u32);
        put(0x381, u32::MAX, x | (x + scissor[2] as u32) << 16);
        put(0x382, u32::MAX, y | (y + scissor[3] as u32) << 16);
    }
    let address = |r: &mut dyn FnMut(usize, u32, u32), at: usize, a: u64| {
        r(at, 0xff, (a >> 32) as u32);
        r(at + 1, u32::MAX, a as u32);
    };
    for (pool, at) in d.pools.iter().zip([reg::TEX_HEADER_POOL, reg::TEX_SAMPLER_POOL]) {
        if *pool != 0 {
            address(&mut put, at, *pool);
        }
    }
    let colors = d.surfaces.iter().filter(|s| s.0 < 8).map(|s| s.0 as u32 + 1).max().unwrap_or(0);
    put(reg::CT_SELECT, u32::MAX, 0xfac_6880 | colors);
    put(0x54e, 1, d.surfaces.iter().any(|s| s.0 == 8) as u32);
    for k in (0..8).filter(|&k| !bound(k)) {
        put(reg::COLOR_TARGET + 16 * k + 4, u32::MAX, 0);
    }
    for &(k, s) in &d.surfaces {
        let format = learned.usual(if k < 8 { Kind::TargetFormat } else { Kind::DepthFormat }, s.format as u64);
        let memory = match s.pitch {
            Some(_) => 0x1000,
            None => s.block_height_log2 << 4,
        };
        let width = ((s.width * s.bytes).div_ceil(64) * 64).checked_div(s.bytes).unwrap_or(s.width);
        let width = if s.pitch.is_some() { s.width } else { width };
        if k < 8 {
            let base = reg::COLOR_TARGET + 16 * k as usize;
            address(&mut put, base, d.all_targets[k as usize]);
            put(base + 2, u32::MAX, s.pitch.unwrap_or(width));
            put(base + 3, 0x1_ffff, s.height);
            match format {
                Some(f) => put(base + 4, u32::MAX, f as u32),
                None => _ = missing.get_or_insert_with(|| format!("target format {:#x}", s.format)),
            }
            put(base + 5, 0x1070, memory);
        } else {
            put(0x48a, u32::MAX, width);
            put(0x48b, 0x1_ffff, s.height);
            match format {
                Some(f) => put(0x3fa, u32::MAX, f as u32),
                None => _ = missing.get_or_insert_with(|| format!("depth format {:#x}", s.format)),
            }
            put(0x3fb, 0x70, memory);
        }
    }
    if viewport[2] != 0 {
        for at in [0x300, 0x3fd] {
            put(at, u32::MAX, viewport[0] as u32 | (viewport[2] as u32) << 16);
            put(at + 1, u32::MAX, viewport[1] as u32 | (viewport[3] as u32) << 16);
        }
    }
    if d.targets[0] != 0 {
        let slot = (regs[reg::CT_SELECT] >> 4 & 7) as usize;
        address(&mut put, reg::COLOR_TARGET + 16 * slot, d.targets[0]);
    }
    if d.targets[1] != 0 {
        address(&mut put, reg::ZT_A, d.targets[1]);
    }
    for &(s, a) in &d.vertex {
        if a != 0 && (s as usize) < streams {
            address(&mut put, reg::VERTEX_STREAM + 4 * s as usize + 1, a);
        }
    }
    for &(s, size) in &d.vertex_sizes {
        let a = d.vertex.iter().find(|v| v.0 == s).map_or(0, |v| v.1);
        if a != 0 && size != 0 && (s as usize) < streams {
            address(&mut put, reg::VERTEX_STREAM_LIMIT + 2 * s as usize, a + size - 1);
        }
    }
    if d.indexed {
        address(&mut put, reg::INDEX_BUFFER, d.start);
        put(reg::INDEX_BUFFER + 4, 3, d.index_type as u32);
        put(reg::VB_ELEMENT_BASE, u32::MAX, d.base_vertex as u32);
        put(0x446, u32::MAX, d.base_vertex as u32);
    }
    let base = crate::gpu::threed::address(regs, reg::PROGRAM_REGION);
    let mut offsets = [0u32; 6];
    for &p in &d.programs {
        let Some(sph) = read(p, 4) else { continue };
        let ty = (u32::from_le_bytes(sph[..4].try_into().unwrap()) >> 10 & 0xf) as usize;
        if ty < 6 && p >= base {
            put(reg::PIPELINE + 16 * ty + 1, u32::MAX, (p - base) as u32);
            offsets[ty] = (p - base) as u32;
        }
    }
    if !d.programs.is_empty() {
        put(MACRO_PROGRAM, u32::MAX, d.program_id);
        put(MACRO_PROGRAM + 4, u32::MAX, d.program_id);
        for (s, o) in offsets.into_iter().enumerate() {
            put(MACRO_PROGRAM + 5 + s, u32::MAX, o);
        }
    }
    if d.targets[1] != 0 {
        put(MACRO_DEPTH_TARGET, u32::MAX, (d.targets[1] >> 8) as u32);
    }
    (out, missing)
}

#[derive(Default)]
struct Checks {
    draws: u64,
    skipped: u64,
    registers: std::collections::BTreeMap<usize, (u64, u64, String)>,
}

static CHECKS: Mutex<Option<Checks>> = Mutex::new(None);

pub fn check(prediction: Option<Prediction>, regs: &[u32]) {
    let mut guard = CHECKS.lock().unwrap();
    let c = guard.get_or_insert_with(Checks::default);
    let Some(p) = prediction else {
        c.skipped += 1;
        return;
    };
    c.draws += 1;
    for (r, mask, value) in p {
        let e = c.registers.entry(r).or_default();
        e.0 += 1;
        let seen = regs[r] & mask;
        if seen != value {
            e.1 += 1;
            if e.2.is_empty() {
                e.2 = format!("predicted {value:#x}, command processor {seen:#x} (mask {mask:#x})");
            }
        }
    }
}

fn report_checks() {
    let guard = CHECKS.lock().unwrap();
    let Some(c) = guard.as_ref() else { return };
    let wrong: Vec<String> = c.registers.iter().filter(|e| e.1.1 > 0).map(|(r, (n, bad, ex))| format!("{r:#05x} {bad}/{n} ({ex})")).collect();
    eprintln!(
        "kirikae:         nvn encode: {} draws predicted ({} not paired), {} registers predicted, {} with mismatches",
        c.draws,
        c.skipped,
        c.registers.len(),
        wrong.len()
    );
    for w in wrong.iter().take(20) {
        eprintln!("kirikae:           {w}");
    }
}

pub fn predicted_registers() -> BTreeSet<usize> {
    CHECKS.lock().unwrap().as_ref().map(|c| c.registers.keys().copied().collect()).unwrap_or_default()
}
