use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const MODULE: u32 = 205;
const IRSENSOR_NOT_READY: u32 = (120 << 9) | MODULE;

const STATUS_MANAGER_SIZE: u64 = 0x8000;
const DEVICE_FORMAT_SIZE: usize = 0xE30;
const DEVICE_FORMAT_COUNT: usize = 9;
const DEVICE_STATUS: usize = 0x0;
const DEVICE_INTERNAL_STATUS: usize = 0x4;
const DEVICE_MODE: usize = 0x8;

const IR_CAMERA_UNCONNECTED: u32 = 2;
const IR_CAMERA_STOPPED: u32 = 0;
const IR_SENSOR_MODE_NONE: u32 = 0;

const LAST_PLAYER_ID: u32 = 7;
const HANDHELD_ID: u32 = 0x20;
const HANDHELD_INDEX: u32 = 8;

pub struct Irs;

impl Service for Irs {
    fn name(&self) -> String {
        "irs".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            302 => Reply::ok(),
            303 => Reply::ok(),
            304 => status_manager_handle(c.process()),
            311 => npad_ir_camera_handle(req.u32(0)),
            319 => Reply::ok(),
            _ => unimplemented("irs", c, req),
        }
    }
}

fn status_manager_handle(p: &'static crate::process::Process) -> Reply {
    let handle = *p.service_state("irs.status", || {
        let (memory, handle) = p.create_shared_memory(STATUS_MANAGER_SIZE);
        memory.write(&p.memory, 0, &status_manager());
        handle
    });
    status_manager_reply(handle)
}

fn status_manager_reply(handle: u32) -> Reply {
    Reply::ok().copy_handle(handle)
}

fn status_manager() -> Vec<u8> {
    let mut bytes = vec![0u8; STATUS_MANAGER_SIZE as usize];
    for device in 0..DEVICE_FORMAT_COUNT {
        let base = device * DEVICE_FORMAT_SIZE;
        bytes[base + DEVICE_STATUS..base + DEVICE_STATUS + 4]
            .copy_from_slice(&IR_CAMERA_UNCONNECTED.to_le_bytes());
        bytes[base + DEVICE_INTERNAL_STATUS..base + DEVICE_INTERNAL_STATUS + 4]
            .copy_from_slice(&IR_CAMERA_STOPPED.to_le_bytes());
        bytes[base + DEVICE_MODE..base + DEVICE_MODE + 4]
            .copy_from_slice(&IR_SENSOR_MODE_NONE.to_le_bytes());
    }
    bytes
}

fn camera_handle(npad: u32) -> Option<u32> {
    let player = match npad {
        0..=LAST_PLAYER_ID => npad,
        HANDHELD_ID => HANDHELD_INDEX,
        _ => return None,
    };
    Some(player)
}

fn npad_ir_camera_handle(npad: u32) -> Reply {
    match camera_handle(npad) {
        Some(handle) => Reply::ok().u32(handle),
        None => Reply::err(IRSENSOR_NOT_READY),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn status_manager_reports_every_device_unconnected() {
        let bytes = status_manager();
        assert_eq!(bytes.len(), STATUS_MANAGER_SIZE as usize);
        for device in 0..DEVICE_FORMAT_COUNT {
            let base = device * DEVICE_FORMAT_SIZE;
            assert_eq!(
                read_u32(&bytes, base + DEVICE_STATUS),
                IR_CAMERA_UNCONNECTED
            );
            assert_eq!(
                read_u32(&bytes, base + DEVICE_INTERNAL_STATUS),
                IR_CAMERA_STOPPED
            );
            assert_eq!(read_u32(&bytes, base + DEVICE_MODE), IR_SENSOR_MODE_NONE);
        }
        let aruid_start = DEVICE_FORMAT_COUNT * DEVICE_FORMAT_SIZE;
        assert_eq!(aruid_start, 0x7FB0);
        assert!(bytes[aruid_start..].iter().all(|&b| b == 0));
    }

    #[test]
    fn camera_handle_follows_the_npad_id() {
        for npad in 0..=LAST_PLAYER_ID {
            assert_eq!(camera_handle(npad), Some(npad));
        }
        assert_eq!(camera_handle(HANDHELD_ID), Some(HANDHELD_INDEX), "handheld does not collide with player 1");
        for npad in [8, 0x10, 0x1f, 0x21, 0x30, u32::MAX] {
            assert_eq!(camera_handle(npad), None);
        }
    }

    #[test]
    fn get_npad_ir_camera_handle_returns_a_u32_or_ir_sensor_not_ready() {
        let r = npad_ir_camera_handle(3);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, 3u32.to_le_bytes());
        assert!(r.copy_handles.is_empty() && r.move_handles.is_empty() && r.objects.is_empty());

        let r = npad_ir_camera_handle(HANDHELD_ID);
        assert_eq!(r.result, 0);
        assert_eq!(r.data, HANDHELD_INDEX.to_le_bytes());

        let r = npad_ir_camera_handle(0x3f);
        assert_eq!(r.result, IRSENSOR_NOT_READY);
        assert!(r.data.is_empty());
        assert_eq!(IRSENSOR_NOT_READY, 0xF0CD);
    }

    #[test]
    fn shared_memory_reply_is_a_single_copy_handle() {
        let r = status_manager_reply(7);
        assert_eq!(r.result, 0);
        assert_eq!(r.copy_handles, vec![7]);
        assert!(r.data.is_empty());
        assert!(r.move_handles.is_empty() && r.objects.is_empty());
    }
}
