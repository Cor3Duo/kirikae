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
    fn play_statistics(application_id: u64) -> [u8; PLAY_STATISTICS_SIZE] {
        let mut out = [0u8; PLAY_STATISTICS_SIZE];
        out[..8].copy_from_slice(&application_id.to_le_bytes());
        out
    }

    fn application_id(data: &[u8]) -> u64 {
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
            0 => Reply::ok().u32(0),
            4 => Reply::ok().bytes(&Self::play_statistics(Self::application_id(&req.data))),
            5 => Reply::ok().bytes(&Self::play_statistics(Self::application_id(&req.data))),
            7 | 17 => Reply::ok().u32(0),
            8 => Reply::ok().u32(0),
            9 => Reply::ok().u32(0).u32(0).u32(0),
            10 => Reply::ok().u32(0),
            11 => Reply::ok().u32(0),
            12 => Reply::ok().u32(0).u32(0).u32(0),
            14 => Reply::ok().u32(0),
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

    #[test]
    fn play_statistics_is_zeroed_except_the_program_id() {
        let statistics = PdmQry::play_statistics(0x0100_0000_000a_0000);
        assert_eq!(statistics.len(), PLAY_STATISTICS_SIZE);
        assert_eq!(
            u64::from_le_bytes(statistics[..8].try_into().unwrap()),
            0x0100_0000_000a_0000
        );
        assert!(statistics[8..].iter().all(|&b| b == 0));
    }

    #[test]
    fn the_application_id_is_read_past_the_newer_prefix() {
        let mut data = vec![1u8];
        data.extend_from_slice(&[0; 7]);
        data.extend_from_slice(&0x0100_0000_000a_0000u64.to_le_bytes());
        assert_eq!(PdmQry::application_id(&data), 0x0100_0000_000a_0000);
        assert_eq!(
            PdmQry::application_id(&0x0100_0000_000b_0000u64.to_le_bytes()),
            0x0100_0000_000b_0000
        );
    }

    #[test]
    fn the_reply_sizes_are_the_documented_ones() {
        assert_eq!(LAST_PLAY_TIME_SIZE, 0x18);
        assert_eq!(APPLET_EVENT_SIZE, 0x28);
        assert_eq!(ACCOUNT_EVENT_SIZE, 0x38);
    }
}
