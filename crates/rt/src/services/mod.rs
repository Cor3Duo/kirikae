use std::sync::Arc;

use crate::cpu::Cpu;

pub mod acc;
pub mod am;
pub mod csrng;
pub mod fs;
#[cfg(test)]
pub(crate) mod harness;
pub mod hid;
pub mod hwopus;
pub mod irs;
pub mod apm;
pub mod audio;
pub mod lm;
pub mod mii;
pub mod mm;
pub mod nfp;
pub mod nvdrv;
pub mod omm;
pub mod pctl;
pub mod pdm;
pub mod pl;
pub mod ro;
pub mod set;
pub mod sm;
pub mod stub;
pub mod time;
pub mod vi;
pub mod bt;
pub mod btm;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buf {
    pub addr: u64,
    pub size: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Request {
    pub cmd: u32,
    pub data: Vec<u8>,
    pub pid: Option<u64>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    pub x: Vec<Buf>,
    pub a: Vec<Buf>,
    pub b: Vec<Buf>,
    pub w: Vec<Buf>,
    pub c: Vec<Buf>,
    pub objects: Vec<u32>,
}

impl Request {
    pub fn u32(&self, off: usize) -> u32 {
        self.data.get(off..off + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn u64(&self, off: usize) -> u64 {
        self.data.get(off..off + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn out_buf(&self, i: usize) -> Option<Buf> {
        self.b.get(i).filter(|b| b.size > 0).or(self.c.get(i).filter(|b| b.size > 0)).copied()
    }

    pub fn in_buf(&self, i: usize) -> Option<Buf> {
        self.a.get(i).filter(|b| b.size > 0).or(self.x.get(i).filter(|b| b.size > 0)).copied()
    }
}

pub struct Reply {
    pub result: u32,
    pub data: Vec<u8>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    pub objects: Vec<Arc<dyn Service>>,
}

impl Reply {
    pub fn ok() -> Reply {
        Reply { result: 0, data: Vec::new(), copy_handles: Vec::new(), move_handles: Vec::new(), objects: Vec::new() }
    }

    pub fn err(result: u32) -> Reply {
        Reply { result, ..Reply::ok() }
    }

    pub fn u8(mut self, v: u8) -> Reply {
        self.data.push(v);
        self
    }

    pub fn u32(mut self, v: u32) -> Reply {
        self.align(4);
        self.data.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(mut self, v: u64) -> Reply {
        self.align(8);
        self.data.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn bytes(mut self, b: &[u8]) -> Reply {
        self.data.extend_from_slice(b);
        self
    }

    pub fn object(mut self, o: Arc<dyn Service>) -> Reply {
        self.objects.push(o);
        self
    }

    pub fn copy_handle(mut self, h: u32) -> Reply {
        self.copy_handles.push(h);
        self
    }

    pub fn move_handle(mut self, h: u32) -> Reply {
        self.move_handles.push(h);
        self
    }

    fn align(&mut self, to: usize) {
        while !self.data.len().is_multiple_of(to) {
            self.data.push(0);
        }
    }
}

pub trait Service: Send + Sync {
    fn name(&self) -> String;
    fn call(&self, c: &mut Cpu, req: &Request) -> Reply;
}

pub fn open(name: &str) -> Arc<dyn Service> {
    match name {
        "lm" => Arc::new(lm::Lm),
        "apm" | "apm:am" => Arc::new(apm::Apm),
        "appletOE" => Arc::new(am::AppletOe),
        "irs" => Arc::new(irs::Irs),
        "csrng" => Arc::new(csrng::Csrng),
        "omm" => Arc::new(omm::Omm),
        "pctl" => Arc::new(pctl::Pctl { name: "pctl" }),
        "pdm:qry" => Arc::new(pdm::PdmQry),
        "pctl:s" => Arc::new(pctl::Pctl { name: "pctl:s" }),
        "pctl:r" => Arc::new(pctl::Pctl { name: "pctl:r" }),
        "pctl:a" => Arc::new(pctl::Pctl { name: "pctl:a" }),
        "fsp-srv" => Arc::new(fs::FsProxy),
        "hid" => Arc::new(hid::Hid),
        "hwopus" => Arc::new(hwopus::HwOpus),
        "acc:u0" => Arc::new(acc::Account { name: "acc:u0" }),
        "acc:u1" => Arc::new(acc::Account { name: "acc:u1" }),
        "time:u" => Arc::new(time::Time { name: "time:u" }),
        "time:a" => Arc::new(time::Time { name: "time:a" }),
        "time:s" => Arc::new(time::Time { name: "time:s" }),
        "nvdrv" => Arc::new(nvdrv::Nvdrv { name: "nvdrv" }),
        "nvdrv:a" => Arc::new(nvdrv::Nvdrv { name: "nvdrv:a" }),
        "nvdrv:s" => Arc::new(nvdrv::Nvdrv { name: "nvdrv:s" }),
        "nvdrv:t" => Arc::new(nvdrv::Nvdrv { name: "nvdrv:t" }),
        "audren:u" => Arc::new(audio::AudioRendererManager),
        "audout:u" => Arc::new(audio::AudioOutManager),
        "pl:u" => Arc::new(pl::Pl),
        "ldr:ro" | "ro:1" => Arc::new(ro::Ro),
        "set" => Arc::new(set::Set),
        "mm:u" => Arc::new(mm::Mm),
        "mii:u" | "mii:e" => Arc::new(mii::Mii::new(name)),
        "nfp:user" => Arc::new(nfp::Nfp),
        "vi:m" => Arc::new(vi::Vi { name: "vi:m" }),
        "vi:s" => Arc::new(vi::Vi { name: "vi:s" }),
        "vi:u" => Arc::new(vi::Vi { name: "vi:u" }),
        "bt" => Arc::new(bt::Bt),
        "btm:u" => Arc::new(btm::BtmU),
        _ => Arc::new(stub::Stub::new(name)),
    }
}

pub mod result {
    pub const SM_NOT_REGISTERED: u32 = 0xe15;
    pub const CMIF_UNKNOWN_COMMAND: u32 = 0x1d60a;
}
