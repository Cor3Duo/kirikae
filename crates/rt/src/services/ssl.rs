use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

/// ssl (`nn::ssl::sf::ISslService`) and ssl:s (`ISslServiceForSystem`).
///
/// The runtime terminates TLS nowhere, so only the initialization handshake is
/// answered: `SetInterfaceVersion` carries no output, and its value is kept because the
/// SDK reads it back to decide which validation rules apply to later commands.
pub struct Ssl {
    name: &'static str,
    interface_version: AtomicU32,
}

impl Ssl {
    pub fn new(name: &'static str) -> Ssl {
        Ssl {
            name,
            interface_version: AtomicU32::new(0),
        }
    }

    fn set_interface_version(&self, req: &Request) -> Reply {
        self.interface_version.store(req.u32(0), Relaxed);
        Reply::ok()
    }

    #[cfg(test)]
    fn interface_version(&self) -> u32 {
        self.interface_version.load(Relaxed)
    }
}

impl Service for Ssl {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            5 => self.set_interface_version(req),
            _ => unimplemented(self.name, c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(data: &[u8]) -> Request {
        Request {
            data: data.to_vec(),
            ..Request::default()
        }
    }

    #[test]
    fn set_interface_version_has_no_output_and_is_kept() {
        let ssl = Ssl::new("ssl");
        assert_eq!(ssl.interface_version(), 0);

        let reply = ssl.set_interface_version(&request(&2u32.to_le_bytes()));
        assert_eq!(reply.result, 0);
        assert!(
            reply.data.is_empty(),
            "unexpected reply data {:x?}",
            reply.data
        );
        assert_eq!(ssl.interface_version(), 2);
    }

    #[test]
    fn a_truncated_version_reads_as_zero() {
        let ssl = Ssl::new("ssl");
        ssl.set_interface_version(&request(&[2, 0]));
        assert_eq!(ssl.interface_version(), 0);
    }
}
