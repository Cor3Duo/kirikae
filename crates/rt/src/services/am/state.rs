use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use crate::cpu::Cpu;
use crate::services::apm::PERFORMANCE_MODE;

pub(super) const RESULT_NO_MESSAGES: u32 = 0x680;
pub(super) const RESULT_NOT_AVAILABLE: u32 = 0x480;

pub(super) const MESSAGE_FOCUS_STATE_CHANGED: u32 = 15;
pub(super) const OPERATION_MODE: u8 = PERFORMANCE_MODE as u8;
pub(super) const APPLET_RESOURCE_USER_ID: u64 = 0x60;
pub(super) const LANGUAGE: &[u8; 5] = b"en-US";
pub struct AmState {
    pub(super) messages: Mutex<VecDeque<u32>>,
    pub(super) message_event: u32,
    pub(super) user_popped: AtomicBool,
}

impl AmState {
    pub(super) fn update_event(&self, c: &Cpu) {
        let pending = !self.messages.lock().unwrap().is_empty();
        c.process().set_event(self.message_event, pending);
    }
}
