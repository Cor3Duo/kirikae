use crate::cpu::Cpu;
use crate::services::{Reply, Request, Service, open};

pub struct Sm;

pub fn service_name(raw: u64) -> String {
    let bytes = raw.to_le_bytes();
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

impl Service for Sm {
    fn name(&self) -> String {
        "sm:".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok(),
            1 => {
                let name = service_name(req.u64(0));
                c.process()
                    .log_ipc(format!("sm: GetServiceHandle({name:?})"));
                Reply::ok().object(open(&name))
            }
            2 | 3 => Reply::ok(),
            _ => Reply::err(crate::services::result::CMIF_UNKNOWN_COMMAND),
        }
    }
}
