use std::sync::Mutex;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

/// `LibraryConfigData` occupies the first 32 bytes of `RegisterClient`.
const CONFIG_SIZE: usize = 32;
/// The transfer-memory size follows the PID and the 32-byte configuration.
const TRANSFER_MEMORY_SIZE_OFFSET: usize = 40;

/// The `nn::socket::sf::LibraryConfigData` the SDK sends to `RegisterClient`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientConfig {
    pub version: u32,
    pub transfer_memory_size: u64,
}

impl ClientConfig {
    fn from_request(req: &Request) -> ClientConfig {
        ClientConfig {
            version: req.u32(0),
            transfer_memory_size: req.u64(TRANSFER_MEMORY_SIZE_OFFSET),
        }
    }
}

/// bsd:u/bsd:s/bsd:a (`nn::socket::sf::IClient`).
///
/// The runtime has no socket stack. `RegisterClient` and `StartMonitoring` are the two
/// commands `nn::socket::Initialize` issues before any socket exists, so they answer
/// with the documented success reply and keep the configuration. Every command that
/// would create or use a socket stays unimplemented and therefore visible in coverage.
pub struct Bsd {
    name: &'static str,
    config: Mutex<Option<ClientConfig>>,
}

impl Bsd {
    pub fn new(name: &'static str) -> Bsd {
        Bsd {
            name,
            config: Mutex::new(None),
        }
    }

    fn register_client(&self, req: &Request) -> Reply {
        if req.data.len() >= CONFIG_SIZE {
            *self.config.lock().unwrap() = Some(ClientConfig::from_request(req));
        }
        // RegisterClient returns one raw u32 on success.
        Reply::ok().u32(0)
    }

    fn start_monitoring(&self) -> Reply {
        Reply::ok()
    }

    #[cfg(test)]
    fn config(&self) -> Option<ClientConfig> {
        *self.config.lock().unwrap()
    }
}

impl Service for Bsd {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => self.register_client(req),
            1 => self.start_monitoring(),
            _ => unimplemented(self.name, c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(data: Vec<u8>) -> Request {
        Request {
            data,
            ..Request::default()
        }
    }

    fn register_data(version: u32, transfer_memory_size: u64) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&version.to_le_bytes());
        data.resize(32, 0);
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&transfer_memory_size.to_le_bytes());
        data
    }

    #[test]
    fn register_client_replies_with_one_success_word() {
        let bsd = Bsd::new("bsd:u");
        let reply = bsd.register_client(&request(register_data(2, 0x1e0000)));
        assert_eq!(reply.result, 0);
        assert_eq!(reply.data, 0u32.to_le_bytes());
        assert!(reply.copy_handles.is_empty() && reply.move_handles.is_empty());
    }

    #[test]
    fn register_client_keeps_the_configuration() {
        let bsd = Bsd::new("bsd:u");
        assert_eq!(bsd.config(), None);

        let reply = bsd.register_client(&request(register_data(2, 0x1e0000)));
        assert_eq!(reply.data.len(), 4);
        assert_eq!(
            bsd.config(),
            Some(ClientConfig {
                version: 2,
                transfer_memory_size: 0x1e0000
            })
        );
    }

    #[test]
    fn a_short_config_does_not_replace_the_stored_one() {
        let bsd = Bsd::new("bsd:u");
        bsd.register_client(&request(register_data(2, 0x1000)));
        let reply = bsd.register_client(&request(vec![1, 0, 0, 0]));
        assert_eq!(reply.data, 0u32.to_le_bytes());
        assert_eq!(
            bsd.config(),
            Some(ClientConfig {
                version: 2,
                transfer_memory_size: 0x1000
            })
        );
    }

    #[test]
    fn start_monitoring_has_no_output() {
        let bsd = Bsd::new("bsd:u");
        let reply = bsd.start_monitoring();
        assert_eq!(reply.result, 0);
        assert!(reply.data.is_empty());
        assert_eq!(bsd.name(), "bsd:u");
    }
}
