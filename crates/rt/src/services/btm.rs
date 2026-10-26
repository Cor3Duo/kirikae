use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const BTM_MODULE: u32 = 143;
const fn btm_error(description: u32) -> u32 {
    (description << 9) | BTM_MODULE
}

const BLUETOOTH_MODULE: u32 = 113;
const fn bluetooth_error(description: u32) -> u32 {
    (description << 9) | BLUETOOTH_MODULE
}

const LOW_LAYER_FAILURE: u32 = btm_error(201);
const DEVICE_ALREADY_DISCONNECTED: u32 = bluetooth_error(3202);

const BLE_ADVERTISE_PACKET_PARAMETER_SIZE: usize = 8;
const GATT_ATTRIBUTE_UUID_SIZE: usize = 0x14;

const NINTENDO_COMPANY_ID: u16 = 0x0553;

pub struct BtmU;

impl Service for BtmU {
    fn name(&self) -> String {
        "btm:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(std::sync::Arc::new(BtmUserCore)),
            _ => unimplemented_named("btm:u", "unknown", c, req),
        }
    }
}

pub struct BtmUserCore;

impl Service for BtmUserCore {
    fn name(&self) -> String {
        "btm:u/IBtmUserCore".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => acquire_event(c, "btm:u.ble_scan_event"),
            1 => scan_filter_parameter(c, req),
            2 => scan_filter_parameter_2(c, req),
            3 | 6 | 8 => Reply::ok(),
            4 | 7 | 9 => Reply::ok(),
            5 | 10 => Reply::ok().u8(0),
            17 => acquire_event(c, "btm:u.ble_connection_event"),
            18 => no_radio(c, "BleConnect"),
            19 => no_device(c, "BleDisconnect"),
            20 => Reply::ok().u8(0),
            21 => acquire_event(c, "btm:u.ble_pairing_event"),
            22..=24 => no_device(c, "pairing"),
            25 => Reply::ok().u8(0),
            26 => acquire_event(c, "btm:u.ble_service_discovery_event"),
            27 | 29 | 31 | 32 => Reply::ok().u8(0),
            28 | 30 => Reply::ok().u8(0),
            33 => acquire_event(c, "btm:u.ble_mtu_config_event"),
            34 => no_device(c, "ConfigureBleMtu"),
            35 => no_device(c, "GetBleMtu"),
            36 | 37 => no_radio(c, "GATT data path"),
            _ => unimplemented_named("btm:u/IBtmUserCore", "unknown", c, req),
        }
    }
}

fn acquire_event(c: &Cpu, key: &'static str) -> Reply {
    let handle = *c
        .process()
        .service_state(key, || c.process().create_event(false));
    Reply::ok().u8(1).copy_handle(handle)
}

fn scan_filter_parameter(c: &mut Cpu, req: &Request) -> Reply {
    let parameter_id = parameter_id(req);
    let pattern: [u8; 6] = match parameter_id {
        0x1 => [0x00, 0x01, 0x00, 0x00, 0x01, 0x00],
        0xFFFF => [0xAD, 0xDE, 0x00, 0xEF, 0xBE, 0x00],
        _ => {
            c.process().warn_once(format!(
                "btm:u: GetBleScanFilterParameter({parameter_id:#x}) names no documented filter"
            ));
            return Reply::err(LOW_LAYER_FAILURE);
        }
    };
    let mut out = [0u8; BLE_ADVERTISE_PACKET_PARAMETER_SIZE];
    out[..2].copy_from_slice(&NINTENDO_COMPANY_ID.to_le_bytes());
    out[2..].copy_from_slice(&pattern);
    Reply::ok().bytes(&out)
}

fn scan_filter_parameter_2(c: &mut Cpu, req: &Request) -> Reply {
    let parameter_id = parameter_id(req);
    if parameter_id != 0x2 {
        c.process().warn_once(format!(
            "btm:u: GetBleScanFilterParameter2({parameter_id:#x}) names no documented filter"
        ));
        return Reply::err(LOW_LAYER_FAILURE);
    }
    Reply::ok().bytes(&[0u8; GATT_ATTRIBUTE_UUID_SIZE])
}

fn parameter_id(req: &Request) -> u16 {
    req.data
        .get(..2)
        .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
}

fn no_radio(c: &Cpu, what: &str) -> Reply {
    c.process().warn_once(format!(
        "btm:u: {what} needs a Bluetooth radio; reporting ResultFailureLowLayerGeneral"
    ));
    Reply::err(LOW_LAYER_FAILURE)
}

fn no_device(c: &Cpu, what: &str) -> Reply {
    c.process().warn_once(format!(
        "btm:u: {what} names no connected device; reporting Device is already disconnected"
    ));
    Reply::err(DEVICE_ALREADY_DISCONNECTED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::kernel::KObject;
    use crate::services::harness;

    #[test]
    fn btm_u_is_opened_from_the_registry() {
        let opened = crate::services::harness::call_service("btm:u", None, Request { cmd: 0, ..Default::default() });
        assert_eq!((opened.result, opened.objects), (0, 1));
    }

    fn core(c: &mut Cpu) -> std::sync::Arc<dyn Service> {
        let created = BtmU.call(c, &Request::default());
        assert_eq!(created.result, 0);
        assert!(created.data.is_empty());
        let object = created.objects.first().expect("GetCore returns an object");
        assert_eq!(object.name(), "btm:u/IBtmUserCore");
        object.clone()
    }

    #[test]
    fn get_core_returns_the_documented_core_object() {
        extern "C" fn probe(c: &mut Cpu) {
            let created = BtmU.call(c, &Request::default());
            harness::record(
                "core",
                &[
                    created.result as u64,
                    created.objects.len() as u64,
                    created.data.len() as u64,
                    created.copy_handles.len() as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("core"),
            vec![0, 1, 0, 0],
            "GetCore succeeds with one object and no raw data"
        );
    }

    #[test]
    fn every_event_command_returns_a_bool_and_a_never_signalled_event() {
        extern "C" fn probe(c: &mut Cpu) {
            let core = core(c);
            let mut all = Vec::new();
            for cmd in [0u32, 17, 21, 26, 33] {
                let reply = core.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                let handle = reply.copy_handles.first().copied().unwrap_or(0);
                let signaled = match c.process().handles.lock().unwrap().get(handle) {
                    Some(KObject::Event(e)) => {
                        e.signaled.load(std::sync::atomic::Ordering::Relaxed)
                    }
                    _ => true,
                };
                all.extend([
                    reply.result as u64,
                    u64::from(reply.data.first().copied().unwrap_or(0)),
                    reply.data.len() as u64,
                    reply.copy_handles.len() as u64,
                    signaled as u64,
                ]);
            }
            harness::record("events", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let mut expected = Vec::new();
        for _ in 0..5 {
            expected.extend([0, 1, 1, 1, 0]);
        }
        assert_eq!(
            harness::observed("events"),
            expected,
            "each event command returns bool 1, a real handle and no signal"
        );
    }

    #[test]
    fn every_documented_query_returns_its_documented_width() {
        extern "C" fn probe(c: &mut Cpu) {
            let core = core(c);
            let mut all = Vec::new();
            let queries: [(u32, usize); 11] = [
                (5, 1),
                (10, 1),
                (20, 1),
                (25, 1),
                (27, 1),
                (28, 1),
                (29, 1),
                (30, 1),
                (31, 1),
                (32, 1),
                (2, GATT_ATTRIBUTE_UUID_SIZE),
            ];
            for (cmd, width) in queries {
                let reply = core.call(
                    c,
                    &Request {
                        cmd,
                        data: vec![2, 0, 0, 0],
                        ..Default::default()
                    },
                );
                all.extend([reply.result as u64, reply.data.len() as u64, width as u64]);
            }
            harness::record("queries", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let observed = harness::observed("queries");
        for (i, chunk) in observed.chunks(3).enumerate() {
            assert_eq!(chunk[0], 0, "query {i} succeeds");
            assert_eq!(chunk[1], chunk[2], "query {i} has its documented width");
        }
        assert_eq!(observed.len(), 11 * 3);
    }

    #[test]
    fn scan_filters_return_the_documented_parameters() {
        extern "C" fn probe(c: &mut Cpu) {
            let core = core(c);
            let first = core.call(
                c,
                &Request {
                    cmd: 1,
                    data: 0x1u32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            let second = core.call(
                c,
                &Request {
                    cmd: 1,
                    data: 0xFFFFu32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            let uuid = core.call(
                c,
                &Request {
                    cmd: 2,
                    data: 0x2u32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            harness::record("first", &[first.result as u64, first.data.len() as u64]);
            harness::record("second", &[second.result as u64, second.data.len() as u64]);
            harness::record(
                "uuid",
                &[
                    uuid.result as u64,
                    uuid.data.len() as u64,
                    uuid.data.iter().all(|&b| b == 0) as u64,
                ],
            );
            harness::record(
                "bytes",
                &[1u64]
                    .into_iter()
                    .chain(first.data.iter().map(|&b| u64::from(b)))
                    .chain(second.data.iter().map(|&b| u64::from(b)))
                    .collect::<Vec<_>>(),
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("first"),
            vec![0, BLE_ADVERTISE_PACKET_PARAMETER_SIZE as u64]
        );
        assert_eq!(
            harness::observed("second"),
            vec![0, BLE_ADVERTISE_PACKET_PARAMETER_SIZE as u64]
        );
        assert_eq!(
            harness::observed("uuid"),
            vec![0, GATT_ATTRIBUTE_UUID_SIZE as u64, 1]
        );
        assert_eq!(
            harness::observed("bytes"),
            vec![
                1, 0x53, 0x05, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x53, 0x05, 0xAD, 0xDE, 0x00,
                0xEF, 0xBE, 0x00
            ],
            "company id 0x0553 then the published pattern"
        );
    }

    #[test]
    fn operations_without_a_radio_or_device_report_the_published_errors() {
        extern "C" fn probe(c: &mut Cpu) {
            let core = core(c);
            let mut all = Vec::new();
            for cmd in [18u32, 36, 37] {
                let reply = core.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                all.extend([reply.result as u64, reply.data.len() as u64]);
            }
            for cmd in [19u32, 22, 23, 24, 34, 35] {
                let reply = core.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                all.extend([reply.result as u64, reply.data.len() as u64]);
            }
            let reply = core.call(
                c,
                &Request {
                    cmd: 1,
                    data: 0x1234u32.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            all.extend([reply.result as u64, reply.data.len() as u64]);
            harness::record("errors", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let observed = harness::observed("errors");
        let mut expected = Vec::new();
        for _ in 0..3 {
            expected.extend([LOW_LAYER_FAILURE as u64, 0]);
        }
        for _ in 0..6 {
            expected.extend([DEVICE_ALREADY_DISCONNECTED as u64, 0]);
        }
        expected.extend([LOW_LAYER_FAILURE as u64, 0]);
        assert_eq!(
            observed, expected,
            "no-radio, no-device and unknown-filter paths carry the published results and no data"
        );
        assert_eq!(LOW_LAYER_FAILURE, 0x1928F);
        assert_eq!(DEVICE_ALREADY_DISCONNECTED, 0x190471);
    }

    #[test]
    fn scan_start_stop_need_no_radio() {
        extern "C" fn probe(c: &mut Cpu) {
            let core = core(c);
            let mut all = Vec::new();
            for cmd in [3u32, 4, 6, 7, 8, 9] {
                let reply = core.call(
                    c,
                    &Request {
                        cmd,
                        ..Default::default()
                    },
                );
                all.extend([reply.result as u64, reply.data.len() as u64]);
            }
            harness::record("scans", &all);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("scans"),
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn an_undocumented_command_is_refused() {
        extern "C" fn probe(c: &mut Cpu) {
            let reply = core(c).call(
                c,
                &Request {
                    cmd: 38,
                    ..Default::default()
                },
            );
            harness::record("unknown", &[reply.result as u64, reply.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("unknown"),
            vec![crate::services::result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }
}
