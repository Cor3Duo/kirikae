use std::sync::Arc;

use crate::cpu::Cpu;

pub mod acc;
pub mod am;
pub mod aoc;
pub mod apm;
pub mod audio;
pub mod bcat;
pub mod bsd;
pub mod csrng;
pub mod friend;
pub mod fs;
pub mod hid;
pub mod hwopus;
pub mod irs;
pub mod ldr;
pub mod lm;
pub mod mm;
pub mod nfp;
pub mod nifm;
pub mod nvdrv;
pub mod pctl;
pub mod pl;
pub mod prepo;
pub mod ro;
pub mod set;
pub mod sm;
pub mod ssl;
pub mod stub;
pub mod time;
pub mod vi;

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
    /// The message used TIPC (typed IPC), not CMIF. `sm:` switched to TIPC on
    /// 12.0.0 and serializes its arguments in a different order than the CMIF
    /// form of the same commands.
    pub tipc: bool,
}

impl Request {
    pub fn u32(&self, off: usize) -> u32 {
        self.data
            .get(off..off + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn u64(&self, off: usize) -> u64 {
        self.data
            .get(off..off + 8)
            .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    pub fn out_buf(&self, i: usize) -> Option<Buf> {
        self.b
            .get(i)
            .filter(|b| b.size > 0)
            .or(self.c.get(i).filter(|b| b.size > 0))
            .copied()
    }

    pub fn in_buf(&self, i: usize) -> Option<Buf> {
        self.a
            .get(i)
            .filter(|b| b.size > 0)
            .or(self.x.get(i).filter(|b| b.size > 0))
            .copied()
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
        Reply {
            result: 0,
            data: Vec::new(),
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            objects: Vec::new(),
        }
    }

    pub fn err(result: u32) -> Reply {
        Reply {
            result,
            ..Reply::ok()
        }
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

    /// Size of the pointer buffer this service prepares for
    /// `QueryPointerBufferSize` (CMIF manager command 3). The default is the
    /// pointer buffer the runtime gives every session; a service that packs
    /// transfers into it may advertise another size.
    fn pointer_buffer_size(&self) -> u16 {
        crate::ipc::POINTER_BUFFER_SIZE
    }
}

/// The object behind a service name the runtime itself serves.
///
/// `None` means the name is not in the runtime's service table. `sm:` uses this
/// to tell a real host service (or one of the runtime's explicit stubs) from a
/// name nobody registered; [`open`] still falls back to a stub for callers that
/// resolve names without going through `sm:`.
pub fn host_object(name: &str) -> Option<Arc<dyn Service>> {
    let service: Arc<dyn Service> = match name {
        "lm" => Arc::new(lm::Lm),
        "apm" | "apm:am" => Arc::new(apm::Apm),
        "appletOE" => Arc::new(am::AppletOe),
        "appletAE" => Arc::new(am::AppletAe),
        "aoc:u" => Arc::new(aoc::AddOnContentManager::new()),
        "bcat:u" => Arc::new(bcat::Bcat { name: "bcat:u" }),
        "bcat:a" => Arc::new(bcat::Bcat { name: "bcat:a" }),
        "bcat:m" => Arc::new(bcat::Bcat { name: "bcat:m" }),
        "bcat:s" => Arc::new(bcat::Bcat { name: "bcat:s" }),
        "friend:u" => Arc::new(friend::FriendServiceCreator),
        "pctl" => Arc::new(pctl::Pctl { name: "pctl" }),
        "pctl:s" => Arc::new(pctl::Pctl { name: "pctl:s" }),
        "pctl:r" => Arc::new(pctl::Pctl { name: "pctl:r" }),
        "pctl:a" => Arc::new(pctl::Pctl { name: "pctl:a" }),
        "fsp-srv" => Arc::new(fs::FsProxy),
        "hid" => Arc::new(hid::Hid),
        "hwopus" => Arc::new(hwopus::HwOpus),
        "csrng" => Arc::new(csrng::Csrng),
        "irs" => Arc::new(irs::Irs),
        "acc:u0" => Arc::new(acc::Account { name: "acc:u0" }),
        "acc:u1" => Arc::new(acc::Account { name: "acc:u1" }),
        "time:u" => Arc::new(time::Time { name: "time:u" }),
        "time:a" => Arc::new(time::Time { name: "time:a" }),
        "time:s" => Arc::new(time::Time { name: "time:s" }),
        "bsd:u" => Arc::new(bsd::Bsd::new("bsd:u")),
        "bsd:s" => Arc::new(bsd::Bsd::new("bsd:s")),
        "bsd:a" => Arc::new(bsd::Bsd::new("bsd:a")),
        "nifm:u" => Arc::new(nifm::Nifm::new("nifm:u")),
        "nifm:s" => Arc::new(nifm::Nifm::new("nifm:s")),
        "nifm:a" => Arc::new(nifm::Nifm::new("nifm:a")),
        "ssl" => Arc::new(ssl::Ssl::new("ssl")),
        "ssl:s" => Arc::new(ssl::Ssl::new("ssl:s")),
        "nvdrv" => Arc::new(nvdrv::Nvdrv { name: "nvdrv" }),
        "nvdrv:a" => Arc::new(nvdrv::Nvdrv { name: "nvdrv:a" }),
        "nvdrv:s" => Arc::new(nvdrv::Nvdrv { name: "nvdrv:s" }),
        "nvdrv:t" => Arc::new(nvdrv::Nvdrv { name: "nvdrv:t" }),
        "audren:u" => Arc::new(audio::AudioRendererManager),
        "audout:u" => Arc::new(audio::AudioOutManager),
        "pl:u" => Arc::new(pl::Pl),
        "prepo:u" => Arc::new(prepo::Prepo),
        "ldr:ro" | "ro:1" => Arc::new(ro::Ro),
        "ldr:dmnt" => Arc::new(ldr::LdrDmnt),
        "set" => Arc::new(set::Set),
        "mm:u" => Arc::new(mm::Mm),
        "nfp:user" => Arc::new(nfp::Nfp),
        "vi:m" => Arc::new(vi::Vi { name: "vi:m" }),
        "vi:s" => Arc::new(vi::Vi { name: "vi:s" }),
        "vi:u" => Arc::new(vi::Vi { name: "vi:u" }),
        // Names the runtime knows about but does not implement yet. They are
        // stubbed explicitly so a guest gets the same "service exists, command
        // unknown" answer it got before the service table became explicit.
        "ectx:aw" | "ectx:r" | "ectx:w" => Arc::new(stub::Stub::new(name)),
        "ngc:u" => Arc::new(stub::Stub::new(name)),
        _ => return None,
    };
    Some(service)
}

/// Whether the runtime serves `name` itself (implemented or an explicit stub).
pub fn host_served(name: &str) -> bool {
    host_object(name).is_some()
}

pub fn open(name: &str) -> Arc<dyn Service> {
    host_object(name).unwrap_or_else(|| Arc::new(stub::Stub::new(name)))
}

pub mod result {
    pub const SM_OUT_OF_PROCESSES: u32 = 0x215;
    pub const SM_NOT_INITIALIZED: u32 = 0x415;
    pub const SM_MAX_SESSIONS: u32 = 0x615;
    pub const SM_ALREADY_REGISTERED: u32 = 0x815;
    pub const SM_OUT_OF_SERVICES: u32 = 0xa15;
    pub const SM_INVALID_NAME: u32 = 0xc15;
    pub const SM_NOT_REGISTERED: u32 = 0xe15;
    pub const CMIF_UNKNOWN_COMMAND: u32 = 0x1d60a;
}

/// A process with guest memory and a live handle table, for tests that drive
/// the real IPC path from an entry point.
#[cfg(test)]
pub(crate) mod harness {
    use std::sync::Arc;

    use super::Service;
    use crate::abi::Entry;
    use crate::cpu::Cpu;
    use crate::kernel::{KObject, Session};
    use crate::process::{RunOptions, run_executable};

    pub const ENTRY: u64 = 0x1000_0000;
    pub const MESSAGE_SIZE: u64 = 0x100;

    /// Runs `entries` in a guest process. Every entry halts itself; a non-zero
    /// exit code fails the test.
    pub fn run(entries: Vec<Entry>) {
        let entries: &'static [Entry] = Box::leak(entries.into_boxed_slice());
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
            arch: kirikae_loader::Arch::A64,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        };
        let result = run_executable(
            &exe,
            entries,
            RunOptions {
                capture_output: true,
            },
        )
        .expect("the fake guest runs");
        assert_eq!(result.exit_code, 0, "the guest entry did not halt cleanly");
    }

    /// A session handle for `object` in the running process.
    pub fn session(c: &mut Cpu, object: Arc<dyn Service>) -> u32 {
        c.process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::Session(Session::new(object)))
    }

    /// Writes `msg` into the thread's HIPC buffer, sends it on `handle` and
    /// returns the reply. The request path is the one a guest uses:
    /// `svcSendSyncRequest` over the TLS message buffer.
    pub fn send(c: &mut Cpu, handle: u32, msg: &[u8]) -> Vec<u8> {
        let tls = c.tpidrro_el0;
        assert!(msg.len() as u64 <= MESSAGE_SIZE);
        assert!(c.process().memory.write(tls, msg));
        let result = crate::ipc::send_sync_request(c, handle, tls, MESSAGE_SIZE);
        assert_eq!(result, 0, "the IPC request failed");
        c.process()
            .memory
            .read(tls, MESSAGE_SIZE)
            .expect("the IPC buffer is mapped")
    }
}
