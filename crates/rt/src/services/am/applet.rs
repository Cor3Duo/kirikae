use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cpu::Cpu;
use crate::services::apm::PERFORMANCE_MODE;
use crate::services::stub::{Stub, unimplemented};
use crate::services::{Reply, Request, Service};

use super::library::LibraryAppletCreator;
use super::state::*;
use super::storage::Storage;

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
            0 => Arc::new(CommonStateGetter {
                state: self.state.clone(),
            }),
            1 => Arc::new(SelfController),
            2 => Arc::new(WindowController),
            3 => stub("am/IAudioController"),
            4 => stub("am/IDisplayController"),
            10 => stub("am/IProcessWindingController"),
            11 => Arc::new(LibraryAppletCreator),
            20 => Arc::new(ApplicationFunctions {
                state: self.state.clone(),
            }),
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
            1 | 2 | 10..=19 | 50 | 62 | 65 | 68 | 69 | 72 | 100 => {
                crate::services::stub::plain_ok()
            }
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

impl Service for ApplicationFunctions {
    fn name(&self) -> String {
        "am/IApplicationFunctions".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            1 => {
                if req.u32(0) == LAUNCH_PRESELECTED_USER
                    && !self.state.user_popped.swap(true, Ordering::Relaxed)
                {
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

fn language_code() -> u64 {
    let mut b = [0u8; 8];
    b[..LANGUAGE.len()].copy_from_slice(LANGUAGE);
    u64::from_le_bytes(b)
}
