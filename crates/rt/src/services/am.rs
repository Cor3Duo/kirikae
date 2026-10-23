use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::apm::PERFORMANCE_MODE;
use crate::services::stub::{unimplemented, Stub};
use crate::services::{Reply, Request, Service};

const RESULT_NO_MESSAGES: u32 = 0x680;
const RESULT_NOT_AVAILABLE: u32 = 0x480;

const MESSAGE_FOCUS_STATE_CHANGED: u32 = 15;
const OPERATION_MODE: u8 = PERFORMANCE_MODE as u8;
const APPLET_RESOURCE_USER_ID: u64 = 0x60;
const LANGUAGE: &[u8; 5] = b"en-US";

pub struct AmState {
    messages: Mutex<VecDeque<u32>>,
    message_event: u32,
    user_popped: AtomicBool,
}

impl AmState {
    fn update_event(&self, c: &Cpu) {
        let pending = !self.messages.lock().unwrap().is_empty();
        c.process().set_event(self.message_event, pending);
    }
}

pub struct AppletOe;

impl Service for AppletOe {
    fn name(&self) -> String {
        "appletOE".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let state = Arc::new(AmState {
                    messages: Mutex::new(VecDeque::from([MESSAGE_FOCUS_STATE_CHANGED])),
                    message_event: c.process().create_event(true),
                    user_popped: AtomicBool::new(false),
                });
                Reply::ok().object(Arc::new(ApplicationProxy { state }))
            }
            _ => unimplemented("appletOE", c, req),
        }
    }
}

struct ApplicationProxy {
    state: Arc<AmState>,
}

impl Service for ApplicationProxy {
    fn name(&self) -> String {
        "am/IApplicationProxy".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let stub = |name: &str| -> Arc<dyn Service> { Arc::new(Stub::new(name)) };
        let obj: Arc<dyn Service> = match req.cmd {
            0 => Arc::new(CommonStateGetter { state: self.state.clone() }),
            1 => Arc::new(SelfController),
            2 => Arc::new(WindowController),
            3 => stub("am/IAudioController"),
            4 => stub("am/IDisplayController"),
            10 => stub("am/IProcessWindingController"),
            11 => Arc::new(LibraryAppletCreator),
            20 => Arc::new(ApplicationFunctions { state: self.state.clone() }),
            1000 => stub("am/IDebugFunctions"),
            _ => return unimplemented("am/IApplicationProxy", c, req),
        };
        Reply::ok().object(obj)
    }
}

struct CommonStateGetter {
    state: Arc<AmState>,
}

impl Service for CommonStateGetter {
    fn name(&self) -> String {
        "am/ICommonStateGetter".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().copy_handle(self.state.message_event),
            1 => {
                let msg = self.state.messages.lock().unwrap().pop_front();
                self.state.update_event(c);
                match msg {
                    Some(m) => Reply::ok().u32(m),
                    None => Reply::err(RESULT_NO_MESSAGES),
                }
            }
            5 => Reply::ok().u8(OPERATION_MODE),
            6 => Reply::ok().u32(PERFORMANCE_MODE),
            8 => Reply::ok().u8(0),
            9 => Reply::ok().u8(1),
            60 => Reply::ok().u32(1920).u32(1080),
            61 => Reply::ok().copy_handle(c.process().create_event(false)),
            66 => Reply::ok(),
            300 => Reply::ok().u32(1),
            _ => unimplemented("am/ICommonStateGetter", c, req),
        }
    }
}

struct SelfController;

impl Service for SelfController {
    fn name(&self) -> String {
        "am/ISelfController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            1 | 2 | 10..=19 | 50 | 62 | 65 | 68 | 69 | 72 | 100 => crate::services::stub::plain_ok(),
            9 => Reply::ok().copy_handle(c.process().create_event(true)),
            91 => Reply::ok().copy_handle(c.process().create_event(false)),
            40 => Reply::ok().u64(1),
            44 => Reply::ok().u64(1).u64(2),
            _ => unimplemented("am/ISelfController", c, req),
        }
    }
}

struct WindowController;

impl Service for WindowController {
    fn name(&self) -> String {
        "am/IWindowController".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            1 => Reply::ok().u64(APPLET_RESOURCE_USER_ID),
            10 => Reply::ok(),
            _ => unimplemented("am/IWindowController", c, req),
        }
    }
}

struct ApplicationFunctions {
    state: Arc<AmState>,
}

const LAUNCH_PRESELECTED_USER: u32 = 2;

fn preselected_user() -> Vec<u8> {
    let mut v = 0xc794_97cau32.to_le_bytes().to_vec();
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&crate::services::acc::USER_ID);
    v.resize(0x88, 0);
    v
}

pub struct Storage {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Storage {
    pub fn new(data: Vec<u8>) -> Self {
        Storage { data: Arc::new(Mutex::new(data)) }
    }
}

impl Service for Storage {
    fn name(&self) -> String {
        "am/IStorage".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(StorageAccessor { data: self.data.clone() })),
            _ => unimplemented("am/IStorage", c, req),
        }
    }
}

struct StorageAccessor {
    data: Arc<Mutex<Vec<u8>>>,
}

impl Service for StorageAccessor {
    fn name(&self) -> String {
        "am/IStorageAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(self.data.lock().unwrap().len() as u64),
            10 => {
                let mut data = self.data.lock().unwrap();
                let off = (req.u64(0) as usize).min(data.len());
                if let Some(b) = req.in_buf(0) {
                    let n = (b.size as usize).min(data.len() - off);
                    if let Some(bytes) = c.process().memory.read(b.addr, n as u64) {
                        data[off..off + n].copy_from_slice(&bytes);
                    }
                }
                Reply::ok()
            }
            11 => {
                let data = self.data.lock().unwrap();
                let off = (req.u64(0) as usize).min(data.len());
                if let Some(b) = req.out_buf(0) {
                    let n = (b.size as usize).min(data.len() - off);
                    c.process().memory.write(b.addr, &data[off..off + n]);
                }
                Reply::ok()
            }
            _ => unimplemented("am/IStorageAccessor", c, req),
        }
    }
}

const APPLET_CONTROLLER: u32 = 0xc;
const NPAD_HANDHELD: u32 = 0x20;

struct LibraryAppletCreator;

impl Service for LibraryAppletCreator {
    fn name(&self) -> String {
        "am/ILibraryAppletCreator".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let id = req.u32(0);
                let p = c.process();
                p.log_ipc(format!("am: library applet {id:#x}"));
                Reply::ok().object(Arc::new(LibraryAppletAccessor {
                    id,
                    state_event: p.create_event(false),
                    out_event: p.create_event(false),
                    output: Mutex::new(None),
                }))
            }
            10 => Reply::ok().object(Arc::new(Storage::new(vec![0; req.u64(0).min(1 << 24) as usize]))),
            _ => unimplemented("am/ILibraryAppletCreator", c, req),
        }
    }
}

struct LibraryAppletAccessor {
    id: u32,
    state_event: u32,
    out_event: u32,
    output: Mutex<Option<Vec<u8>>>,
}

impl LibraryAppletAccessor {
    fn result(&self) -> Vec<u8> {
        match self.id {
            APPLET_CONTROLLER => {
                let mut v = vec![1, 0, 0, 0];
                v.extend_from_slice(&NPAD_HANDHELD.to_le_bytes());
                v.extend_from_slice(&0u32.to_le_bytes());
                v
            }
            _ => Vec::new(),
        }
    }
}

impl Service for LibraryAppletAccessor {
    fn name(&self) -> String {
        "am/ILibraryAppletAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().copy_handle(self.state_event),
            1 => Reply::ok().u8(self.output.lock().unwrap().is_some() as u8 | 1),
            10 => {
                let out = self.result();
                if !out.is_empty() {
                    *self.output.lock().unwrap() = Some(out);
                    p.set_event(self.out_event, true);
                }
                p.set_event(self.state_event, true);
                Reply::ok()
            }
            20 | 25 | 30 | 50 | 60 | 100 | 150 => crate::services::stub::plain_ok(),
            101 => match self.output.lock().unwrap().take() {
                Some(data) => {
                    p.set_event(self.out_event, false);
                    Reply::ok().object(Arc::new(Storage::new(data)))
                }
                None => Reply::err(RESULT_NOT_AVAILABLE),
            },
            105 => Reply::ok().copy_handle(self.out_event),
            106 => Reply::ok().copy_handle(p.create_event(false)),
            110 => Reply::ok().u8(0),
            _ => unimplemented("am/ILibraryAppletAccessor", c, req),
        }
    }
}

fn language_code() -> u64 {
    let mut b = [0u8; 8];
    b[..LANGUAGE.len()].copy_from_slice(LANGUAGE);
    u64::from_le_bytes(b)
}

impl Service for ApplicationFunctions {
    fn name(&self) -> String {
        "am/IApplicationFunctions".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            1 => {
                if req.u32(0) == LAUNCH_PRESELECTED_USER && !self.state.user_popped.swap(true, Ordering::Relaxed) {
                    Reply::ok().object(Arc::new(Storage::new(preselected_user())))
                } else {
                    Reply::err(RESULT_NOT_AVAILABLE)
                }
            }
            20 => Reply::ok().u64(0),
            21 => Reply::ok().u64(language_code()),
            22 => {
                let r = req.u32(0);
                c.process().warn_once(format!("the application set its terminate result to {r:#x} (module {}, description {})", r & 0x1ff, r >> 9 & 0x1fff));
                Reply::ok()
            }
            66 | 67 | 90 | 100 | 170 => crate::services::stub::plain_ok(),
            23 => Reply::ok().bytes(b"1.0.0\0\0\0\0\0\0\0\0\0\0\0"),
            40 => Reply::ok().u8(1),
            50 => Reply::ok().bytes(&[0x44; 16]),
            130 | 140 | 150 => Reply::ok().copy_handle(c.process().create_event(false)),
            210 => Reply::ok().copy_handle(c.process().create_event(false)),
            _ => unimplemented("am/IApplicationFunctions", c, req),
        }
    }
}
