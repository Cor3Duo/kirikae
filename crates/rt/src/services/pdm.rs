//! `pdm:qry` — play-log queries.
//!
//! The runtime keeps no play log yet, so every query answers with an empty range
//! and zeroed statistics for the asked application. The reply shapes follow
//! libnx's `pdm.h` (16.0.0+ layouts): `PdmPlayStatistics` 0x48, `PdmLastPlayTime`
//! 0x18, `PdmAppletEvent` 0x28 and `PdmAccountEvent` 0x38 bytes. Command numbers
//! are the ones `pdm.c` dispatches to; a game that only wants "how long did the
//! user play" gets a valid zero, not a malformed buffer.

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

const PLAY_STATISTICS_SIZE: usize = 0x48;
#[cfg(test)]
const LAST_PLAY_TIME_SIZE: usize = 0x18;
#[cfg(test)]
const APPLET_EVENT_SIZE: usize = 0x28;
#[cfg(test)]
const ACCOUNT_EVENT_SIZE: usize = 0x38;

pub struct PdmQry;

impl PdmQry {
    /// A `PdmPlayStatistics` with no launches: only `program_id` is meaningful.
    fn play_statistics(application_id: u64) -> [u8; PLAY_STATISTICS_SIZE] {
        let mut out = [0u8; PLAY_STATISTICS_SIZE];
        out[..8].copy_from_slice(&application_id.to_le_bytes());
        out
    }

    fn application_id(data: &[u8]) -> u64 {
        // 10.0.0+ prepends a flag byte and padding to the id.
        if data.len() >= 16 {
            u64::from_le_bytes(data[8..16].try_into().unwrap_or_default())
        } else {
            u64::from_le_bytes(data[..8.min(data.len())].try_into().unwrap_or_default())
        }
    }
}

impl Service for PdmQry {
    fn name(&self) -> String {
        "pdm:qry".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // QueryAppletEvent
            0 => Reply::ok().u32(0),
            // QueryPlayStatisticsByApplicationId
            4 => Reply::ok().bytes(&Self::play_statistics(Self::application_id(&req.data))),
            // QueryPlayStatisticsByApplicationIdAndUserAccountId
            5 => Reply::ok().bytes(&Self::play_statistics(Self::application_id(&req.data))),
            // QueryLastPlayTime: no entries.
            7 | 17 => Reply::ok().u32(0),
            // QueryPlayEvent
            8 => Reply::ok().u32(0),
            // GetAvailablePlayEventRange
            9 => Reply::ok().u32(0).u32(0).u32(0),
            // QueryAccountEvent
            10 => Reply::ok().u32(0),
            // QueryAccountPlayEvent
            11 => Reply::ok().u32(0),
            // GetAvailableAccountPlayEventRange
            12 => Reply::ok().u32(0).u32(0).u32(0),
            // QueryRecentlyPlayedApplication
            14 => Reply::ok().u32(0),
            // GetRecentlyPlayedApplicationUpdateEvent: never raised.
            15 => {
                let handle = *c
                    .process()
                    .service_state("pdm.recently-played", || c.process().create_event(false));
                Reply::ok().copy_handle(handle)
            }
            _ => unimplemented_named("pdm:qry", pdm_command_name(req.cmd), c, req),
        }
    }
}

fn pdm_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "QueryAppletEvent",
        4 => "QueryPlayStatisticsByApplicationId",
        5 => "QueryPlayStatisticsByApplicationIdAndUserAccountId",
        7 | 17 => "QueryLastPlayTime",
        8 => "QueryPlayEvent",
        9 => "GetAvailablePlayEventRange",
        10 => "QueryAccountEvent",
        11 => "QueryAccountPlayEvent",
        12 => "GetAvailableAccountPlayEventRange",
        14 => "QueryRecentlyPlayedApplication",
        15 => "GetRecentlyPlayedApplicationUpdateEvent",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::services::harness;

    #[test]
    fn query_replies_have_the_documented_shapes() {
        extern "C" fn probe(c: &mut Cpu) {
            let mut data = vec![1u8];
            data.extend_from_slice(&[0; 7]);
            data.extend_from_slice(&0x0100_0000_000A_0000u64.to_le_bytes());
            let stats = PdmQry.call(
                c,
                &Request {
                    cmd: 4,
                    data,
                    ..Default::default()
                },
            );
            harness::record(
                "stats",
                &[
                    stats.data.len() as u64,
                    u64::from_le_bytes(stats.data[..8].try_into().unwrap()),
                ],
            );
            let range = PdmQry.call(
                c,
                &Request {
                    cmd: 9,
                    ..Default::default()
                },
            );
            harness::record("range", &[range.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("stats"),
            vec![PLAY_STATISTICS_SIZE as u64, 0x0100_0000_000A_0000]
        );
        assert_eq!(harness::observed("range"), vec![12]);
    }

    #[test]
    fn the_statistics_reply_is_the_16_0_layout() {
        let stats = PdmQry::play_statistics(0x0100_0000_000A_0000);
        assert_eq!(stats.len(), PLAY_STATISTICS_SIZE);
        assert_eq!(
            u64::from_le_bytes(stats[..8].try_into().unwrap()),
            0x0100_0000_000A_0000
        );
        assert!(stats[8..].iter().all(|&b| b == 0));
    }

    #[test]
    fn the_current_event_layouts_are_documented_sizes() {
        assert_eq!(LAST_PLAY_TIME_SIZE, 0x18);
        assert_eq!(APPLET_EVENT_SIZE, 0x28);
        assert_eq!(ACCOUNT_EVENT_SIZE, 0x38);
    }
}
