use crate::cpu::Cpu;
use crate::gpu::Gpu;
use crate::host1x::Host1x;
use crate::mem::GuestMemory;

use super::State;

/// Argument buffer of one ioctl, padded to the size the ioctl number declares.
/// Writes past the end are recorded instead of silently dropped so tests and
/// the caller can tell that a reply does not fit its documented struct.
pub(super) struct Args<'a> {
    pub(super) data: &'a mut [u8],
    pub(super) overrun: bool,
}

impl<'a> Args<'a> {
    pub(super) fn new(data: &'a mut [u8]) -> Args<'a> {
        Args {
            data,
            overrun: false,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.data.len()
    }

    pub(super) fn data(&self) -> &[u8] {
        self.data
    }

    pub(super) fn u8(&self, o: usize) -> u8 {
        self.data.get(o).copied().unwrap_or(0)
    }

    pub(super) fn u32(&self, o: usize) -> u32 {
        self.data
            .get(o..o + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub(super) fn u64(&self, o: usize) -> u64 {
        self.data
            .get(o..o + 8)
            .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    pub(super) fn bytes(&self, o: usize, n: usize) -> &[u8] {
        self.data.get(o..o + n).unwrap_or(&[])
    }

    pub(super) fn set16(&mut self, o: usize, v: u16) {
        self.set(o, &v.to_le_bytes());
    }

    pub(super) fn set32(&mut self, o: usize, v: u32) {
        self.set(o, &v.to_le_bytes());
    }

    pub(super) fn set64(&mut self, o: usize, v: u64) {
        self.set(o, &v.to_le_bytes());
    }

    pub(super) fn set(&mut self, o: usize, v: &[u8]) {
        match self.data.get_mut(o..o + v.len()) {
            Some(dst) => dst.copy_from_slice(v),
            None => self.overrun = true,
        }
    }

    pub(super) fn fill(&mut self, o: usize, n: usize, v: u8) {
        match self.data.get_mut(o..o + n) {
            Some(dst) => dst.fill(v),
            None => self.overrun = true,
        }
    }
}

/// Everything an ioctl may need besides its arguments. `cpu` is `None` in
/// tests that exercise the reply shapes without a running guest.
pub(super) struct Env<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) mem: &'a GuestMemory,
    pub(super) st: &'a State,
    pub(super) cpu: Option<&'a Cpu>,
    pub(super) host1x: Option<&'a Host1x>,
}

impl Env<'_> {
    pub(super) fn submit(&self, entries: &[u64]) -> Result<(), String> {
        let submit = || {
            let _p = crate::prof::enter(crate::prof::Cat::Gpu);
            self.gpu.submit(self.mem, entries)
        };
        match self.cpu {
            Some(c) => crate::sched::without_gil(c, submit),
            None => submit(),
        }
    }

    pub(super) fn warn(&self, msg: String) {
        if let Some(c) = self.cpu {
            c.process().warn_once(msg);
        }
    }
}
