use super::NV_BAD_PARAMETER;
use super::NV_BAD_VALUE;
use super::Outcome;
use super::args::{Args, Env};
use crate::cpu::Cpu;
use crate::host1x::Host1x;

/// Header of `NVHOST_IOCTL_CHANNEL_SUBMIT`: `cmdbuf`, `reloc`, `reloc_shift` and
/// `syncpt_incr` entries are appended in that order.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct SubmitLayout {
    pub(super) buffers: usize,
    pub(super) relocs: usize,
    pub(super) incrs: usize,
    pub(super) fences: usize,
}

impl SubmitLayout {
    pub(super) fn parse(a: &Args) -> Result<SubmitLayout, u32> {
        let layout = SubmitLayout {
            buffers: a.u32(0) as usize,
            relocs: a.u32(4) as usize,
            incrs: a.u32(8) as usize,
            fences: a.u32(12) as usize,
        };
        let incr_at = 16 + 12 * layout.buffers + 20 * layout.relocs;
        let need = incr_at + 20 * layout.incrs + 4 * layout.fences;
        if a.len() < need {
            return Err(NV_BAD_PARAMETER);
        }
        Ok(layout)
    }
}

pub(super) fn legacy_submit(env: &Env, a: &mut Args) -> Outcome {
    let layout = match SubmitLayout::parse(a) {
        Ok(layout) => layout,
        Err(code) => return Outcome::Fail(code),
    };
    let reloc_at = 16 + 12 * layout.buffers;
    let incr_at = reloc_at + 20 * layout.relocs;
    let fence_at = incr_at + 20 * layout.incrs;
    let mut thresholds: Vec<(u32, u32)> = Vec::new();
    for i in 0..layout.incrs {
        let (id, count) = (a.u32(incr_at + 20 * i), a.u32(incr_at + 20 * i + 4));
        let before = thresholds
            .iter()
            .rev()
            .find(|t| t.0 == id)
            .map_or(env.gpu.syncpoint(id), |t| t.1);
        thresholds.push((id, before.wrapping_add(count)));
    }
    let mut entries = Vec::new();
    for i in 0..layout.buffers {
        let (handle, offset, words) = (
            a.u32(16 + 12 * i),
            a.u32(20 + 12 * i) as u64,
            a.u32(24 + 12 * i) as u64,
        );
        let Some(m) = env.gpu.nvmap(handle) else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let Some(base) = env.gpu.gpu_addresses_of(m.addr).first().copied() else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let va = base + offset;
        entries.push(gpfifo_entry(va, words));
    }
    for r in 0..layout.relocs {
        let (buffer, at, target, target_offset, shift) = (
            a.u32(reloc_at + 16 * r),
            a.u32(reloc_at + 16 * r + 4),
            a.u32(reloc_at + 16 * r + 8),
            a.u32(reloc_at + 16 * r + 12),
            a.u32(reloc_at + 16 * layout.relocs + 4 * r),
        );
        let Some(src) = env.gpu.nvmap(buffer) else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let Some(dst) = env.gpu.nvmap(target) else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let Some(base) = env.gpu.gpu_addresses_of(dst.addr).first().copied() else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let value = ((base + target_offset as u64) >> (shift & 31)) as u32;
        if !env.mem.write(src.addr + at as u64, &value.to_le_bytes()) {
            return Outcome::Fail(NV_BAD_VALUE);
        }
    }
    if let Err(e) = env.submit(&entries) {
        env.warn(format!("GPU: {e}"));
    }
    for &(id, threshold) in &thresholds {
        let behind = threshold.wrapping_sub(env.gpu.syncpoint(id));
        if (behind as i32) > 0 {
            env.gpu.increment(id, behind);
        }
    }
    for (i, &(_, threshold)) in thresholds.iter().take(layout.fences).enumerate() {
        a.set32(fence_at + 4 * i, threshold);
    }
    Outcome::Ok
}

pub(super) fn gpfifo_entry(va: u64, words: u64) -> u64 {
    (va & 0xffff_fffc) | (((va >> 32) & 0xff) << 32) | ((words & 0x1f_ffff) << 42)
}

fn gpfifo_entries(a: &Args, extra: &[u8]) -> Result<Vec<u64>, u32> {
    let count = a.u32(8) as usize;
    let needed = count * 8;
    let src = if !extra.is_empty() && extra.len() >= needed {
        &extra[..needed]
    } else {
        let inline = a.data().get(0x18..).ok_or(NV_BAD_PARAMETER)?;
        if inline.len() < needed {
            return Err(NV_BAD_PARAMETER);
        }
        &inline[..needed]
    };
    Ok(src
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect())
}

pub(super) fn submit_gpfifo(env: &Env, syncpoint: u32, a: &mut Args, extra: &[u8]) -> Outcome {
    let flags = a.u32(12);
    let entries = match gpfifo_entries(a, extra) {
        Ok(entries) => entries,
        Err(code) => return Outcome::Fail(code),
    };
    if flags & 1 != 0 {
        env.gpu.wait_idle();
    }
    if let Err(e) = env.submit(&entries) {
        env.warn(format!("GPU: {e}"));
    }
    // The fence is whatever the submitted pushbuffer left the syncpoint at:
    // the increments it carries are the ones the caller waits on.
    a.set32(16, syncpoint);
    a.set32(20, env.gpu.syncpoint(syncpoint));
    Outcome::Ok
}

pub(super) fn engine_submit(env: &Env, c: &Cpu, fd: u32, class: u32, a: &mut Args) -> Outcome {
    let layout = match SubmitLayout::parse(a) {
        Ok(layout) => layout,
        Err(code) => return Outcome::Fail(code),
    };
    let p = c.process();
    let host1x = c.process().service_state("host1x", Host1x::default);
    let incr_at = 16 + 12 * layout.buffers + 20 * layout.relocs;
    let fence_at = incr_at + 20 * layout.incrs;
    let mut thresholds: Vec<(u32, u32)> = Vec::new();
    for i in 0..layout.incrs {
        let (id, count) = (a.u32(incr_at + 20 * i), a.u32(incr_at + 20 * i + 4));
        let before = thresholds
            .iter()
            .rev()
            .find(|t| t.0 == id)
            .map_or(env.gpu.syncpoint(id), |t| t.1);
        thresholds.push((id, before.wrapping_add(count)));
    }
    if crate::host1x::trace() {
        let reloc_at = 16 + 12 * layout.buffers;
        let relocs: Vec<[u32; 5]> = (0..layout.relocs)
            .map(|i| [0, 4, 8, 12].map(|o| a.u32(reloc_at + 16 * i + o)))
            .enumerate()
            .map(|(i, r)| {
                [
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    a.u32(reloc_at + 16 * layout.relocs + 4 * i),
                ]
            })
            .collect();
        eprintln!(
            "nvdrv: class {class:#x} SUBMIT {0} buffer(s), relocs {relocs:x?}, increments {thresholds:?}",
            layout.buffers
        );
    }
    for i in 0..layout.buffers {
        let (handle, offset, words) = (a.u32(16 + 12 * i), a.u32(20 + 12 * i), a.u32(24 + 12 * i));
        let Some(m) = env.gpu.nvmap(handle).filter(|m| m.addr != 0) else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let Some(bytes) = p.memory.read(m.addr + offset as u64, words as u64 * 4) else {
            return Outcome::Fail(NV_BAD_VALUE);
        };
        let mut words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let reloc_at = 16 + 12 * layout.buffers;
        for r in 0..layout.relocs {
            let e = |o: usize| a.u32(reloc_at + 16 * r + o);
            let (buffer, at, target, target_offset, shift) = (
                e(0),
                e(4),
                e(8),
                e(12),
                a.u32(reloc_at + 16 * layout.relocs + 4 * r),
            );
            let Some(index) = at
                .checked_sub(offset)
                .map(|o| o as usize / 4)
                .filter(|&i| buffer == handle && i < words.len())
            else {
                continue;
            };
            if let Some(iova) = env
                .gpu
                .nvmap(target)
                .and_then(|m| host1x.map_buffer(target, m))
            {
                words[index] = (iova.wrapping_add(target_offset) as u64 >> (shift & 31)) as u32;
            }
        }
        let errors = crate::sched::without_gil(c, || {
            let _prof = crate::prof::enter(crate::prof::Cat::Video);
            host1x.submit(p, fd, class, &words)
        });
        for e in errors {
            p.warn_once(e);
        }
    }
    for &(id, threshold) in &thresholds {
        let behind = threshold.wrapping_sub(env.gpu.syncpoint(id));
        if (behind as i32) > 0 {
            env.gpu.increment(id, behind);
        }
    }
    for (i, &(_, threshold)) in thresholds.iter().take(layout.fences).enumerate() {
        a.set32(fence_at + 4 * i, threshold);
    }
    Outcome::Ok
}
