use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::{Request, Service};
use crate::abi::Entry;
use crate::cpu::Cpu;
use crate::kernel::{KObject, Session};
use crate::process::{RunOptions, run_executable};

pub const ENTRY: u64 = 0x1000_0000;
pub const MESSAGE_SIZE: u64 = 0x100;

static RECORDED: Mutex<BTreeMap<&'static str, Vec<u64>>> = Mutex::new(BTreeMap::new());

pub fn record(key: &'static str, values: &[u64]) {
    assert!(
        key.split_once('/').is_some_and(|(module, name)| !module.is_empty() && !name.is_empty()),
        "harness keys are `module/name` so parallel tests never share one: {key}"
    );
    let previous = RECORDED.lock().unwrap().insert(key, values.to_vec());
    assert!(previous.is_none(), "{key} was recorded twice before being observed");
}

pub fn observed(key: &str) -> Vec<u64> {
    RECORDED.lock().unwrap().remove(key).unwrap_or_default()
}

pub fn run(entries: Vec<Entry>) {
    run_with_title(entries, None);
}

pub fn run_with_title(entries: Vec<Entry>, title_id: Option<u64>) {
    let entries: &'static [Entry] = Box::leak(entries.into_boxed_slice());
    let exe = kirikae_loader::Executable { entry: entries[0].addr, title_id, ..Default::default() };
    let result = run_executable(&exe, entries, RunOptions { capture_output: true }).expect("the fake guest runs");
    assert_eq!(result.exit_code, 0, "the guest entry did not halt cleanly");
}

pub struct Called {
    pub result: u32,
    pub data: Vec<u8>,
    pub objects: usize,
    pub copy_handles: usize,
}

static CALL_LOCK: Mutex<()> = Mutex::new(());
static CALL: Mutex<Option<(&'static str, Request)>> = Mutex::new(None);
static CALLED: Mutex<Option<Called>> = Mutex::new(None);

extern "C" fn call_entry(c: &mut Cpu) {
    let (name, req) = CALL.lock().unwrap().take().expect("a pending call");
    let reply = super::open(name).call(c, &req);
    *CALLED.lock().unwrap() = Some(Called {
        result: reply.result,
        data: reply.data.clone(),
        objects: reply.objects.len(),
        copy_handles: reply.copy_handles.len(),
    });
    c.halt();
}

pub fn call_service(name: &'static str, title_id: Option<u64>, req: Request) -> Called {
    let _one_at_a_time = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *CALL.lock().unwrap() = Some((name, req));
    run_with_title(vec![Entry { addr: ENTRY, func: call_entry }], title_id);
    CALLED.lock().unwrap().take().expect("the call ran")
}

pub fn session(c: &mut Cpu, object: Arc<dyn Service>) -> u32 {
    c.process().handles.lock().unwrap().insert(KObject::Session(Session::new(object)))
}

pub fn send(c: &mut Cpu, handle: u32, msg: &[u8]) -> Vec<u8> {
    let tls = c.tpidrro_el0;
    assert!(msg.len() as u64 <= MESSAGE_SIZE);
    assert!(c.process().memory.write(tls, msg));
    let result = crate::ipc::send_sync_request(c, handle, tls, MESSAGE_SIZE);
    assert_eq!(result, 0, "the IPC request failed");
    c.process().memory.read(tls, MESSAGE_SIZE).expect("the IPC buffer is mapped")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::TYPE_REQUEST;
    use crate::services::{Reply, Request};

    struct Echo;

    impl Service for Echo {
        fn name(&self) -> String {
            "echo".into()
        }

        fn call(&self, _c: &mut Cpu, req: &Request) -> Reply {
            Reply::ok().u32(req.cmd + 1)
        }
    }

    extern "C" fn echo_over_ipc(c: &mut Cpu) {
        let handle = session(c, Arc::new(Echo));
        let msg = crate::ipc::client::cmif(TYPE_REQUEST, 41, &[], None, false, &[]);
        let reply = send(c, handle, &msg);
        let (result, data, _) = crate::ipc::client::cmif_response(&reply, false);
        record("harness/echo", &[u64::from(result), u64::from(u32::from_le_bytes(data[..4].try_into().unwrap()))]);
        c.halt();
    }

    #[test]
    fn a_session_answers_a_request_sent_through_the_tls_buffer() {
        run(vec![Entry { addr: ENTRY, func: echo_over_ipc }]);
        assert_eq!(observed("harness/echo"), vec![0, 42]);
    }

    extern "C" fn title_of_the_process(c: &mut Cpu) {
        record("harness/title", &[c.process().title_id]);
        c.halt();
    }

    #[test]
    fn call_service_opens_the_service_by_its_registered_name() {
        let opened = call_service("lm", None, Request { cmd: 0, ..Default::default() });
        assert_eq!((opened.result, opened.objects, opened.copy_handles), (0, 1, 0));
        assert!(opened.data.is_empty());
    }

    #[test]
    #[should_panic(expected = "module/name")]
    fn a_key_without_a_module_prefix_is_refused() {
        record("event", &[1]);
    }

    #[test]
    fn observing_a_key_consumes_it() {
        record("harness/consumed", &[7]);
        assert_eq!(observed("harness/consumed"), vec![7]);
        assert!(observed("harness/consumed").is_empty());
    }

    #[test]
    fn the_fake_process_carries_the_given_title() {
        run_with_title(vec![Entry { addr: ENTRY, func: title_of_the_process }], Some(0x0100_0000_0000_0001));
        assert_eq!(observed("harness/title"), vec![0x0100_0000_0000_0001]);
    }
}
