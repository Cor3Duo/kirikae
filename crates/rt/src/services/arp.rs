//! `arp:r` / `arp:w` — application-instance registration.
//!
//! Switchbrew documents both interfaces on its "Glue services" page.
//! `nn::arp::detail::IReader` asks who is running and where it came from; the
//! runtime runs one application in its own process, so the answers are the
//! running title and a single instance id. `GetApplicationControlProperty` and
//! `GetApplicationCertificate` need the NACP and the certificate chain, neither
//! of which the loader exposes, so they return `InvalidArgument` instead of
//! fabricated data.
//!
//! `IWriter` is used by the ARP sysmodule and by `ns` when an instance is
//! created or torn down, not by an application: every command is refused with
//! the CMIF unknown-command result after logging its name once.

use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// `nn::arp::ApplicationLaunchProperty` is 0x10 bytes.
const LAUNCH_PROPERTY_SIZE: usize = 0x10;
/// `nn::arp::ApplicationProcessProperty` is 0x10 bytes since 17.0.0 (1 before).
const PROCESS_PROPERTY_SIZE: usize = 0x10;
/// `nn::arp::ApplicationKind::Application`.
const APPLICATION_KIND: u8 = 0;
/// The runtime's machine model, shared with `am`'s `GetLaunchStorageInfoForDebug`:
/// the title is installed in built-in user storage.
const STORAGE_ID_BUILT_IN_USER: u8 = 4;
/// `ResultInvalidArgument`: the runtime has no data for this query.
const INVALID_ARGUMENT: u32 = 0x2EE202;

pub struct Arp {
    name: &'static str,
}

impl Arp {
    pub fn new(name: &'static str) -> Arp {
        Arp { name }
    }

    /// The running application's title id, or `InvalidArgument` when the process
    /// was started without one (a plain `.nro`).
    fn title(&self, c: &Cpu) -> Result<u64, Reply> {
        let title = c.process().title_id;
        if title == 0 {
            return Err(Reply::err(INVALID_ARGUMENT));
        }
        Ok(title)
    }
}

impl Service for Arp {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        // Each arm names the interface so the coverage report groups the
        // commands under `arp:r` instead of the service-name field.
        match (self.name, req.cmd) {
            // GetApplicationLaunchProperty (a ProcessId, or an instance id on
            // 10.0.0+) and GetPreomiaApplicationLaunchProperty.
            ("arp:r", 0) | ("arp:r", 9998) => match self.title(c) {
                Ok(title) => Reply::ok().bytes(&launch_property(title)),
                Err(reply) => reply,
            },
            // GetApplicationControlProperty: needs the NACP, which the loader
            // does not keep (the same reason `ns:am` refuses it).
            ("arp:r", 1) | ("arp:r", 9999) | ("arp:r", 7) => Reply::err(INVALID_ARGUMENT),
            // GetApplicationProcessProperty: the running application is program
            // index 0.
            ("arp:r", 2) => match self.title(c) {
                Ok(_) => Reply::ok().bytes(&process_property()),
                Err(reply) => reply,
            },
            // GetApplicationInstanceId / GetMicroApplicationInstanceId.
            ("arp:r", 3) | ("arp:r", 6) => match self.title(c) {
                Ok(_) => Reply::ok().u64(instance_id(c)),
                Err(reply) => reply,
            },
            // GetApplicationInstanceUnregistrationNotifier.
            ("arp:r", 4) => match self.title(c) {
                Ok(_) => Reply::ok().object(Arc::new(UnregistrationNotifier)),
                Err(reply) => reply,
            },
            // ListApplicationInstanceId: one instance, in the caller's buffer
            // and as the s32 count.
            ("arp:r", 5) => match self.title(c) {
                Ok(_) => {
                    let id = instance_id(c);
                    if let Some(b) = req.b.first() {
                        c.process().memory.write(b.addr, &id.to_le_bytes());
                    }
                    Reply::ok().u32(1)
                }
                Err(reply) => reply,
            },
            _ => unimplemented_named(self.name, arp_command_name(self.name, req.cmd), c, req),
        }
    }
}

/// `nn::arp::ApplicationLaunchProperty`: id, version, storage, patch storage,
/// kind and a reserved byte. The loader does not read a title version (the AM
/// display version is the runtime's fixed "1.0.0"), so the version stays 0.
fn launch_property(title: u64) -> [u8; LAUNCH_PROPERTY_SIZE] {
    let mut out = [0u8; LAUNCH_PROPERTY_SIZE];
    out[..8].copy_from_slice(&title.to_le_bytes());
    out[0xc] = STORAGE_ID_BUILT_IN_USER;
    out[0xd] = STORAGE_ID_BUILT_IN_USER;
    out[0xe] = APPLICATION_KIND;
    out
}

fn process_property() -> [u8; PROCESS_PROPERTY_SIZE] {
    // ProgramIndex 0 (the main program) and the rest reserved.
    [0u8; PROCESS_PROPERTY_SIZE]
}

fn instance_id(c: &Cpu) -> u64 {
    *c.process().service_state("arp.instance", || 1u64)
}

fn arp_command_name(service: &str, cmd: u32) -> &'static str {
    // A lookup table, not a `match`: the coverage report counts the arms of a
    // `match` on a command id as implemented commands.
    let table: &[(u32, &'static str)] = match service {
        "arp:r" => &[
            (0, "GetApplicationLaunchProperty"),
            (1, "GetApplicationControlProperty"),
            (2, "GetApplicationProcessProperty"),
            (3, "GetApplicationInstanceId"),
            (4, "GetApplicationInstanceUnregistrationNotifier"),
            (5, "ListApplicationInstanceId"),
            (6, "GetMicroApplicationInstanceId"),
            (7, "GetApplicationCertificate"),
            (9998, "GetPreomiaApplicationLaunchProperty"),
            (9999, "GetPreomiaApplicationControlProperty"),
        ],
        "arp:w" => &[
            (0, "AcquireRegistrar"),
            (1, "UnregisterApplicationInstance"),
            (2, "AcquireApplicationProcessPropertyUpdater"),
            (3, "AcquireApplicationCertificateUpdater"),
            (4, "RegisterApplicationControlProperty"),
        ],
        _ => &[],
    };
    table
        .iter()
        .find(|(id, _)| *id == cmd)
        .map_or("unknown", |(_, name)| name)
}

/// `nn::arp::detail::IUnregistrationNotifier`: the instance is never unregistered
/// while the process runs, so the event stays non-signalled.
struct UnregistrationNotifier;

impl Service for UnregistrationNotifier {
    fn name(&self) -> String {
        "arp/IUnregistrationNotifier".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let handle = *c
                    .process()
                    .service_state("arp.unregistration", || c.process().create_event(false));
                Reply::ok().copy_handle(handle)
            }
            _ => unimplemented_named("arp/IUnregistrationNotifier", "unknown", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::abi::Entry;
    use crate::cpu::Cpu;
    use crate::services::Service;
    use crate::services::harness;
    use crate::services::result::CMIF_UNKNOWN_COMMAND;

    const TITLE_ID: u64 = 0x0100_0000_0001_0000;

    #[test]
    fn the_reader_reports_the_running_application() {
        extern "C" fn probe(c: &mut Cpu) {
            let reader = super::Arp::new("arp:r");
            let property = reader.call(
                c,
                &crate::services::Request {
                    cmd: 0,
                    data: 1u64.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            let first = reader.call(
                c,
                &crate::services::Request {
                    cmd: 3,
                    ..Default::default()
                },
            );
            let second = reader.call(
                c,
                &crate::services::Request {
                    cmd: 6,
                    ..Default::default()
                },
            );
            let unknown = reader.call(
                c,
                &crate::services::Request {
                    cmd: 8,
                    ..Default::default()
                },
            );
            harness::record(
                "reader",
                &[
                    u64::from(property.result),
                    property.data.len() as u64,
                    u64::from_le_bytes(property.data[..8].try_into().unwrap()),
                    u64::from(property.data[0xc]),
                    u64::from(property.data[0xd]),
                    u64::from(property.data[0xe]),
                    u64::from_le_bytes(first.data[..8].try_into().unwrap()),
                    u64::from_le_bytes(second.data[..8].try_into().unwrap()),
                    u64::from(unknown.result),
                ],
            );
            c.halt();
        }
        harness::run_with_title(
            vec![Entry {
                addr: harness::ENTRY,
                func: probe,
            }],
            Some(TITLE_ID),
        );
        let seen = harness::observed("reader");
        assert_eq!(seen[0], 0);
        assert_eq!(seen[1], 0x10);
        assert_eq!(seen[2], TITLE_ID);
        assert_eq!(&seen[3..6], &[4, 4, 0]);
        assert_eq!(seen[6], 1);
        assert_eq!(
            seen[6], seen[7],
            "both instance-id commands return the same id"
        );
        assert_eq!(seen[8], u64::from(CMIF_UNKNOWN_COMMAND));
    }

    #[test]
    fn the_nacp_queries_are_refused_instead_of_faked() {
        extern "C" fn probe(c: &mut Cpu) {
            let reader = super::Arp::new("arp:r");
            let control = reader.call(
                c,
                &crate::services::Request {
                    cmd: 1,
                    ..Default::default()
                },
            );
            let certificate = reader.call(
                c,
                &crate::services::Request {
                    cmd: 7,
                    ..Default::default()
                },
            );
            harness::record(
                "nacp",
                &[u64::from(control.result), u64::from(certificate.result)],
            );
            c.halt();
        }
        harness::run_with_title(
            vec![Entry {
                addr: harness::ENTRY,
                func: probe,
            }],
            Some(TITLE_ID),
        );
        assert_eq!(
            harness::observed("nacp"),
            vec![u64::from(super::INVALID_ARGUMENT); 2]
        );
    }

    #[test]
    fn listing_instances_writes_the_id_and_returns_one() {
        extern "C" fn probe(c: &mut Cpu) {
            let out = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(out, 0x1000, crate::mem::state::NORMAL, crate::mem::perm::RW)
                .unwrap();
            let reply = super::Arp::new("arp:r").call(
                c,
                &crate::services::Request {
                    cmd: 5,
                    b: vec![crate::services::Buf {
                        addr: out,
                        size: 0x10,
                    }],
                    ..Default::default()
                },
            );
            let id = c.process().memory.read(out, 8).unwrap();
            harness::record(
                "list",
                &[
                    u64::from(reply.result),
                    u64::from(u32::from_le_bytes(reply.data[..4].try_into().unwrap())),
                    u64::from_le_bytes(id.as_slice().try_into().unwrap()),
                ],
            );
            c.halt();
        }
        harness::run_with_title(
            vec![Entry {
                addr: harness::ENTRY,
                func: probe,
            }],
            Some(TITLE_ID),
        );
        assert_eq!(harness::observed("list"), vec![0, 1, 1]);
    }

    #[test]
    fn the_unregistration_notifier_hands_out_one_event() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = super::Arp::new("arp:r").call(
                c,
                &crate::services::Request {
                    cmd: 4,
                    ..Default::default()
                },
            );
            let notifier = reply.objects.first().cloned().expect("notifier object");
            let first = notifier.call(
                c,
                &crate::services::Request {
                    cmd: 0,
                    ..Default::default()
                },
            );
            let second = notifier.call(
                c,
                &crate::services::Request {
                    cmd: 0,
                    ..Default::default()
                },
            );
            harness::record(
                "notifier",
                &[
                    u64::from(reply.result),
                    u64::from(first.copy_handles[0]),
                    u64::from(second.copy_handles[0]),
                ],
            );
            c.halt();
        }
        harness::run_with_title(
            vec![Entry {
                addr: harness::ENTRY,
                func: probe,
            }],
            Some(TITLE_ID),
        );
        let seen = harness::observed("notifier");
        assert_eq!(seen[0], 0);
        assert_eq!(seen[1], seen[2], "one event per process");
    }

    #[test]
    fn the_writer_is_refused_with_the_unknown_command_result() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = super::Arp::new("arp:w").call(
                c,
                &crate::services::Request {
                    cmd: 1,
                    ..Default::default()
                },
            );
            harness::record("writer", &[u64::from(reply.result)]);
            c.halt();
        }
        harness::run_with_title(
            vec![Entry {
                addr: harness::ENTRY,
                func: probe,
            }],
            Some(TITLE_ID),
        );
        assert_eq!(
            harness::observed("writer"),
            vec![u64::from(CMIF_UNKNOWN_COMMAND)]
        );
    }

    #[test]
    fn the_two_documented_arp_names_are_served() {
        // The Services API page lists `arp:r` and `arp:w`; there is no `arp:u`.
        for name in ["arp:r", "arp:w"] {
            assert_eq!(crate::services::open(name).name(), name);
        }
        assert!(!crate::services::host_served("arp:u"));
    }
}
