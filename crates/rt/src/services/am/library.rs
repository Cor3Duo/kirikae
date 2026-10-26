use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::kernel::KObject;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

use super::state::RESULT_NOT_AVAILABLE;
use super::storage::{
    MAX_STORAGE_SIZE, RESULT_INVALID_PARAMETERS, RESULT_MEMORY_ALLOCATION_FAILED,
    RESULT_NULL_OBJECT, RESULT_OBJECT_INVALID, Storage,
};

const APPLET_CONTROLLER: u32 = 0xc;
const APPLET_DATA_ERASE: u32 = 0xd;
const APPLET_PLAYER_SELECT: u32 = 0x10;
const APPLET_SOFTWARE_KEYBOARD: u32 = 0x11;

const UI_RETURN_ARG_SIZE: usize = 0x18;
const DATA_ERASE_OUTPUT_SIZE: usize = 0x1000;

const NPAD_HANDHELD: u32 = 0x20;

const KEYBOARD_OUTPUT_SIZE: usize = 0x7d8;
const KEYBOARD_TEXT: &str = "Kirikae";

pub(super) struct LibraryAppletCreator;

impl LibraryAppletCreator {
    fn create_applet(&self, c: &mut Cpu, req: &Request) -> Reply {
        let id = req.u32(0);
        let mode = req.u32(4);
        let p = c.process();
        p.log_ipc(format!("am: library applet {id:#x} (mode {mode})"));
        Reply::ok().object(Arc::new(LibraryAppletAccessor::new(id, mode, p)))
    }

    fn create_storage(&self, req: &Request) -> Reply {
        let size = req.u64(0) as i64;
        if size <= 0 {
            return Reply::err(RESULT_OBJECT_INVALID);
        }
        let size = size as u64;
        if size > MAX_STORAGE_SIZE {
            return Reply::err(RESULT_MEMORY_ALLOCATION_FAILED);
        }
        Reply::ok().object(Arc::new(Storage::new(vec![0; size as usize])))
    }

    fn create_transfer_memory_storage(&self, c: &mut Cpu, req: &Request) -> Reply {
        let read_only = req.u64(0) & 1 == 0;
        let size = req.u64(8) as i64;
        if size <= 0 {
            return Reply::err(RESULT_OBJECT_INVALID);
        }
        let Some(handle) = req.copy_handles.first().copied() else {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        };
        let p = c.process();
        let Some((addr, tmem_size)) = p.handles.lock().unwrap().get(handle).and_then(|o| match o {
            KObject::TransferMemory { addr, size } => Some((addr, size)),
            _ => None,
        }) else {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        };
        if size as u64 > tmem_size {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        }
        Reply::ok().object(Arc::new(Storage::transfer_memory(addr, size as u64, read_only)))
    }

    fn create_handle_storage(&self, c: &mut Cpu, req: &Request) -> Reply {
        let size = req.u64(0) as i64;
        if size <= 0 {
            return Reply::err(RESULT_OBJECT_INVALID);
        }
        let Some(object) = req.copy_handles.first().and_then(|&h| c.process().handles.lock().unwrap().get(h)) else {
            return Reply::err(RESULT_INVALID_PARAMETERS);
        };
        Reply::ok().object(Arc::new(Storage::handle(size as u64, object)))
    }
}

impl Service for LibraryAppletCreator {
    fn name(&self) -> String {
        "am/ILibraryAppletCreator".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 | 3 => self.create_applet(c, req),
            1 => Reply::ok(),
            2 => Reply::ok().u8(0),
            10 => self.create_storage(req),
            11 => self.create_transfer_memory_storage(c, req),
            12 => self.create_handle_storage(c, req),
            _ => unimplemented("am/ILibraryAppletCreator", c, req),
        }
    }
}

struct LibraryAppletAccessor {
    id: u32,
    mode: u32,
    state_event: u32,
    out_event: u32,
    interactive_event: Mutex<Option<u32>>,
    output: Mutex<Option<Vec<u8>>>,
    completed: AtomicBool,
}

impl LibraryAppletAccessor {
    fn new(id: u32, mode: u32, p: &crate::process::Process) -> Self {
        LibraryAppletAccessor {
            id,
            mode,
            state_event: p.create_event(false),
            out_event: p.create_event(false),
            interactive_event: Mutex::new(None),
            output: Mutex::new(None),
            completed: AtomicBool::new(false),
        }
    }

    fn output(&self) -> Option<Vec<u8>> {
        match self.id {
            APPLET_CONTROLLER => {
                let mut v = vec![1, 0, 0, 0];
                v.extend_from_slice(&NPAD_HANDHELD.to_le_bytes());
                v.extend_from_slice(&0u32.to_le_bytes());
                Some(v)
            }
            APPLET_DATA_ERASE => Some(vec![0u8; DATA_ERASE_OUTPUT_SIZE]),
            APPLET_PLAYER_SELECT => {
                let mut out = vec![0u8; UI_RETURN_ARG_SIZE];
                out[8..].copy_from_slice(&crate::services::acc::USER_ID);
                Some(out)
            }
            APPLET_SOFTWARE_KEYBOARD => {
                let mut out = vec![0u8; KEYBOARD_OUTPUT_SIZE];
                for (i, unit) in KEYBOARD_TEXT.encode_utf16().enumerate() {
                    out[4 + i * 2..6 + i * 2].copy_from_slice(&unit.to_le_bytes());
                }
                Some(out)
            }
            _ => None,
        }
    }

    fn start(&self, c: &mut Cpu) -> Reply {
        if !self.completed.swap(true, Ordering::Relaxed) {
            let p = c.process();
            if let Some(bytes) = self.output() {
                *self.output.lock().unwrap() = Some(bytes);
                p.set_event(self.out_event, true);
            }
            p.set_event(self.state_event, true);
        }
        Reply::ok()
    }

    fn push_storage(&self, req: &Request) -> Reply {
        if req.objects.is_empty() {
            Reply::err(RESULT_NULL_OBJECT)
        } else {
            Reply::ok()
        }
    }

    fn interactive_event(&self, c: &mut Cpu) -> u32 {
        *self
            .interactive_event
            .lock()
            .unwrap()
            .get_or_insert_with(|| c.process().create_event(false))
    }
}

impl Service for LibraryAppletAccessor {
    fn name(&self) -> String {
        "am/ILibraryAppletAccessor".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().copy_handle(self.state_event),
            1 => Reply::ok().u8(self.completed.load(Ordering::Relaxed) as u8),
            10 => self.start(c),
            20 | 25 => {
                self.completed.store(true, Ordering::Relaxed);
                c.process().set_event(self.state_event, true);
                Reply::ok()
            }
            30 => Reply::ok(),
            50 | 60 => Reply::ok(),
            100 | 102 | 103 => self.push_storage(req),
            104 => Reply::err(RESULT_NOT_AVAILABLE),
            101 => match self.output.lock().unwrap().take() {
                Some(data) => {
                    c.process().set_event(self.out_event, false);
                    Reply::ok().object(Arc::new(Storage::new(data)))
                }
                None => Reply::err(RESULT_NOT_AVAILABLE),
            },
            105 => Reply::ok().copy_handle(self.out_event),
            106 => Reply::ok().copy_handle(self.interactive_event(c)),
            110 => Reply::ok().u8(0),
            120 => Reply::ok().u32(self.id).u32(self.mode),
            150 => Reply::ok(),
            _ => unimplemented("am/ILibraryAppletAccessor", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{Entry, GuestFn};
    use crate::mem::{perm, state};
    use crate::process::{RunOptions, run_executable};
    use crate::services::Buf;

    use super::super::storage::{RESULT_OPENED_AS_WRONG_TYPE, RESULT_OUT_OF_BOUNDS};

    static OBSERVED: Mutex<Vec<(&'static str, Vec<u64>)>> = Mutex::new(Vec::new());

    fn record(name: &'static str, values: &[u64]) {
        OBSERVED.lock().unwrap().push((name, values.to_vec()));
    }

    fn observed(name: &'static str) -> Vec<u64> {
        let mut all = OBSERVED.lock().unwrap();
        let i = all
            .iter()
            .position(|(n, _)| *n == name)
            .expect("the fake guest recorded nothing");
        all.remove(i).1
    }

    fn run_guest(f: GuestFn) {
        let entries: &'static [Entry] = Box::leak(
            vec![Entry {
                addr: 0x1000_0000,
                func: f,
            }]
            .into_boxed_slice(),
        );
        let exe = kirikae_loader::Executable {
            modules: Vec::new(),
            entry: entries[0].addr,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        };
        let result = run_executable(
            &exe,
            entries,
            RunOptions {
                capture_output: true,
            },
        )
        .expect("the fake guest runs");
        assert_eq!(result.exit_code, 0);
    }

    fn request(cmd: u32, data: Vec<u8>) -> Request {
        Request {
            cmd,
            data,
            ..Default::default()
        }
    }

    fn guest_buffer(c: &mut Cpu, addr: u64, size: u64) -> u64 {
        c.process()
            .memory
            .map_as(addr, size, state::NORMAL, perm::RW)
            .unwrap();
        addr
    }

    fn open(storage: &Arc<dyn Service>, c: &mut Cpu) -> Arc<dyn Service> {
        let reply = storage.call(c, &request(0, Vec::new()));
        assert_eq!(reply.result, 0);
        reply.objects.into_iter().next().expect("an accessor")
    }

    fn event_signaled(c: &mut Cpu, handle: u32) -> bool {
        matches!(
            c.process().handles.lock().unwrap().get(handle),
            Some(KObject::Event(e)) if e.signaled.load(Ordering::Relaxed)
        )
    }

    fn launch(c: &mut Cpu, id: u32) -> Arc<dyn Service> {
        let creator = LibraryAppletCreator;
        let mut start = id.to_le_bytes().to_vec();
        start.extend_from_slice(&0u32.to_le_bytes());
        let launched = creator.call(
            c,
            &Request {
                cmd: 0,
                data: start,
                ..Default::default()
            },
        );
        assert_eq!(launched.result, 0);
        launched
            .objects
            .first()
            .cloned()
            .expect("CreateLibraryApplet returns an accessor")
    }

    extern "C" fn storage_bounds(c: &mut Cpu) {
        let storage: Arc<dyn Service> = Arc::new(Storage::new(vec![0; 0x20]));
        let accessor = open(&storage, c);
        let src = guest_buffer(c, 0x1_1000_0000, 0x40);
        let dst = guest_buffer(c, 0x1_1000_1000, 0x40);
        let _ = c.process().memory.write(src, b"12345678");
        let _ = c.process().memory.write(dst, &[0xff; 0x40]);

        let size = accessor.call(c, &request(0, Vec::new()));
        let write = accessor.call(
            c,
            &Request {
                cmd: 10,
                data: 0x18u64.to_le_bytes().to_vec(),
                a: vec![Buf { addr: src, size: 8 }],
                ..Default::default()
            },
        );
        let read = accessor.call(
            c,
            &Request {
                cmd: 11,
                data: 0x18u64.to_le_bytes().to_vec(),
                b: vec![Buf { addr: dst, size: 8 }],
                ..Default::default()
            },
        );
        let stored = c.process().memory.read(dst, 0x10).unwrap();
        let write_across = accessor.call(
            c,
            &Request {
                cmd: 10,
                data: 0x18u64.to_le_bytes().to_vec(),
                a: vec![Buf { addr: src, size: 0x10 }],
                ..Default::default()
            },
        );
        let read_across = accessor.call(
            c,
            &Request {
                cmd: 11,
                data: 0x18u64.to_le_bytes().to_vec(),
                b: vec![Buf { addr: dst + 0x20, size: 0x10 }],
                ..Default::default()
            },
        );
        let read_into_input = accessor.call(
            c,
            &Request {
                cmd: 11,
                data: 0u64.to_le_bytes().to_vec(),
                x: vec![Buf { addr: dst + 0x20, size: 0x10 }],
                ..Default::default()
            },
        );
        let untouched = c.process().memory.read(dst + 0x20, 0x10).unwrap().iter().all(|b| *b == 0xff);
        let write_past = accessor.call(
            c,
            &Request {
                cmd: 10,
                data: 0x21u64.to_le_bytes().to_vec(),
                a: vec![Buf { addr: src, size: 8 }],
                ..Default::default()
            },
        );
        let read_past = accessor.call(
            c,
            &Request {
                cmd: 11,
                data: 0x21u64.to_le_bytes().to_vec(),
                b: vec![Buf { addr: dst, size: 8 }],
                ..Default::default()
            },
        );
        let write_at_end = accessor.call(
            c,
            &Request {
                cmd: 10,
                data: 0x20u64.to_le_bytes().to_vec(),
                a: vec![Buf { addr: src, size: 8 }],
                ..Default::default()
            },
        );
        let read_at_end = accessor.call(
            c,
            &Request {
                cmd: 11,
                data: 0x20u64.to_le_bytes().to_vec(),
                b: vec![Buf { addr: dst, size: 8 }],
                ..Default::default()
            },
        );

        record(
            "storage",
            &[
                u64::from(size.result),
                u64::from_le_bytes(size.data[..8].try_into().unwrap()),
                u64::from(write.result),
                u64::from(read.result),
                u64::from(stored[..8] == *b"12345678"),
                u64::from(stored[8..].iter().all(|b| *b == 0xff)),
                u64::from(write_past.result),
                u64::from(read_past.result),
                u64::from(write_at_end.result),
                u64::from(read_at_end.result),
                u64::from(write_across.result),
                u64::from(read_across.result),
                u64::from(read_into_input.result),
                u64::from(untouched),
            ],
        );
        c.halt();
    }

    #[test]
    fn storage_reads_and_writes_refuse_ranges_past_the_end() {
        run_guest(storage_bounds);
        let oob = u64::from(RESULT_OUT_OF_BOUNDS);
        assert_eq!(
            observed("storage"),
            vec![0, 0x20, 0, 0, 1, 1, oob, oob, oob, oob, oob, oob, 0, 1]
        );
    }

    extern "C" fn storage_read_only(c: &mut Cpu) {
        let storage: Arc<dyn Service> = Arc::new(Storage::transfer_memory(0, 0x10, true));
        let accessor = open(&storage, c);
        let src = guest_buffer(c, 0x1_1000_2000, 0x10);
        let _ = c.process().memory.write(src, b"abcd");
        let write = accessor.call(
            c,
            &Request {
                cmd: 10,
                data: 0u64.to_le_bytes().to_vec(),
                a: vec![Buf { addr: src, size: 4 }],
                ..Default::default()
            },
        );
        let wrong_open = storage.call(c, &request(1, Vec::new()));
        let handle_storage: Arc<dyn Service> =
            Arc::new(Storage::handle(0x100, KObject::TransferMemory { addr: 0, size: 0x100 }));
        let handle_open = handle_storage.call(c, &request(0, Vec::new()));
        record(
            "read_only",
            &[
                u64::from(write.result),
                u64::from(wrong_open.result),
                u64::from(handle_open.result),
            ],
        );
        c.halt();
    }

    #[test]
    fn a_read_only_storage_refuses_writes_and_storages_open_only_one_way() {
        run_guest(storage_read_only);
        assert_eq!(
            observed("read_only"),
            vec![
                u64::from(RESULT_OBJECT_INVALID),
                u64::from(RESULT_OPENED_AS_WRONG_TYPE),
                u64::from(RESULT_OPENED_AS_WRONG_TYPE),
            ]
        );
    }

    extern "C" fn transfer_memory_storage(c: &mut Cpu) {
        let addr = guest_buffer(c, 0x1_2000_0000, 0x1000);
        let _ = c.process().memory.write(addr, b"payload");
        let handle = c
            .process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::TransferMemory { addr, size: 0x1000 });
        let creator = LibraryAppletCreator;
        let mut data = 1u64.to_le_bytes().to_vec();
        data.extend_from_slice(&8u64.to_le_bytes());
        let reply = creator.call(
            c,
            &Request {
                cmd: 11,
                data,
                copy_handles: vec![handle],
                ..Default::default()
            },
        );
        let open_after = c.process().handles.lock().unwrap().get(handle).is_some();
        let storage = reply
            .objects
            .first()
            .cloned()
            .expect("CreateTransferMemoryStorage returns an IStorage");
        let _ = c.process().memory.write(addr, b"changed!");
        let accessor = open(&storage, c);
        let size = accessor.call(c, &request(0, Vec::new()));
        let out = guest_buffer(c, 0x1_2000_1000, 8);
        let read = accessor.call(
            c,
            &Request {
                cmd: 11,
                data: 0u64.to_le_bytes().to_vec(),
                b: vec![Buf { addr: out, size: 8 }],
                ..Default::default()
            },
        );
        let bytes = c.process().memory.read(out, 8).unwrap();
        let src = guest_buffer(c, 0x1_2000_2000, 4);
        let _ = c.process().memory.write(src, b"done");
        let write = accessor.call(
            c,
            &Request {
                cmd: 10,
                data: 0u64.to_le_bytes().to_vec(),
                a: vec![Buf { addr: src, size: 4 }],
                ..Default::default()
            },
        );
        let written = c.process().memory.read(addr, 4).unwrap();
        record(
            "tmem",
            &[
                u64::from(reply.result),
                reply.objects.len() as u64,
                u64::from(open_after),
                u64::from_le_bytes(size.data[..8].try_into().unwrap()),
                u64::from(read.result),
                u64::from(bytes == b"changed!".to_vec()),
                u64::from(write.result),
                u64::from(written == b"done".to_vec()),
            ],
        );
        c.halt();
    }

    #[test]
    fn a_transfer_memory_storage_maps_the_requested_bytes_and_leaves_the_handle_to_the_guest() {
        run_guest(transfer_memory_storage);
        assert_eq!(
            observed("tmem"),
            vec![0, 1, 1, 8, 0, 1, 0, 1],
            "the storage has the requested size, the guest keeps its handle and reads and writes go to the mapped memory"
        );
    }

    extern "C" fn create_storage_sizes(c: &mut Cpu) {
        let creator = LibraryAppletCreator;
        let storage = |c: &mut Cpu, size: u64| {
            creator.call(
                c,
                &Request {
                    cmd: 10,
                    data: size.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            )
        };
        let zero = storage(c, 0);
        let huge = storage(c, MAX_STORAGE_SIZE + 1);
        let ok = storage(c, 0x1000);
        let started = creator.call(c, &request(1, Vec::new()));
        let appointment = creator.call(c, &request(2, Vec::new()));
        record(
            "sizes",
            &[
                u64::from(zero.result),
                u64::from(huge.result),
                u64::from(ok.result),
                ok.objects.len() as u64,
                u64::from(started.result),
                u64::from(appointment.data.first().copied().unwrap_or(0xff)),
            ],
        );
        c.halt();
    }

    #[test]
    fn create_storage_rejects_invalid_sizes_and_caps_the_allocation() {
        run_guest(create_storage_sizes);
        assert_eq!(
            observed("sizes"),
            vec![
                u64::from(RESULT_OBJECT_INVALID),
                u64::from(RESULT_MEMORY_ALLOCATION_FAILED),
                0,
                1,
                0,
                0,
            ]
        );
    }

    extern "C" fn handle_storage(c: &mut Cpu) {
        let addr = guest_buffer(c, 0x1_2500_0000, 0x200);
        let handle = c
            .process()
            .handles
            .lock()
            .unwrap()
            .insert(KObject::TransferMemory { addr, size: 0x200 });
        let creator = LibraryAppletCreator;
        let reply = creator.call(
            c,
            &Request {
                cmd: 12,
                data: 0x200u64.to_le_bytes().to_vec(),
                copy_handles: vec![handle],
                ..Default::default()
            },
        );
        let storage = reply.objects.first().cloned().expect("an IStorage");
        let opened = storage.call(c, &request(0, Vec::new()));
        let transfer = storage.call(c, &request(1, Vec::new()));
        assert_eq!(transfer.result, 0);
        let accessor = transfer.objects.first().cloned().expect("an accessor");
        let size = accessor.call(c, &request(0, Vec::new()));
        let got = accessor.call(c, &request(1, Vec::new()));
        record(
            "handle",
            &[
                u64::from(reply.result),
                u64::from(opened.result),
                u64::from_le_bytes(size.data[..8].try_into().unwrap()),
                u64::from_le_bytes(got.data[..8].try_into().unwrap()),
                u64::from(got.copy_handles.first().copied() != Some(handle)),
                u64::from(matches!(
                    got.copy_handles.first().and_then(|&h| c.process().handles.lock().unwrap().get(h)),
                    Some(KObject::TransferMemory { addr: a, size: 0x200 }) if a == addr
                )),
            ],
        );
        c.halt();
    }

    #[test]
    fn a_handle_storage_only_opens_as_a_transfer_storage() {
        run_guest(handle_storage);
        assert_eq!(
            observed("handle"),
            vec![0, u64::from(RESULT_OPENED_AS_WRONG_TYPE), 0x200, 0x200, 1, 1]
        );
    }

    extern "C" fn keyboard_applet(c: &mut Cpu) {
        let accessor = launch(c, APPLET_SOFTWARE_KEYBOARD);
        let started = accessor.call(c, &request(10, Vec::new()));
        let event = accessor.call(c, &request(0, Vec::new()));
        let signaled = event_signaled(c, event.copy_handles[0]);
        let completed = accessor.call(c, &request(1, Vec::new()));
        let pushed = accessor.call(
            c,
            &Request {
                cmd: 100,
                objects: vec![9],
                ..Default::default()
            },
        );
        let extra = accessor.call(
            c,
            &Request {
                cmd: 102,
                objects: vec![9],
                ..Default::default()
            },
        );
        let null_push = accessor.call(c, &request(100, Vec::new()));
        let popped = accessor.call(c, &request(101, Vec::new()));
        let result = accessor.call(c, &request(30, Vec::new()));

        let out = guest_buffer(c, 0x1_3000_0000, KEYBOARD_OUTPUT_SIZE as u64);
        let mut keyboard = 0;
        let mut text = 0;
        let mut storage_size = 0;
        if let Some(storage) = popped.objects.first() {
            let reader = open(storage, c);
            let size = reader.call(c, &request(0, Vec::new()));
            storage_size = u64::from_le_bytes(size.data[..8].try_into().unwrap());
            let read = reader.call(
                c,
                &Request {
                    cmd: 11,
                    data: 0u64.to_le_bytes().to_vec(),
                    b: vec![Buf {
                        addr: out,
                        size: KEYBOARD_OUTPUT_SIZE as u64,
                    }],
                    ..Default::default()
                },
            );
            if read.result == 0 {
                let bytes = c
                    .process()
                    .memory
                    .read(out, KEYBOARD_OUTPUT_SIZE as u64)
                    .unwrap();
                keyboard = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64;
                let expected: Vec<u8> = KEYBOARD_TEXT
                    .encode_utf16()
                    .flat_map(|u| u.to_le_bytes())
                    .collect();
                text = u64::from(bytes[4..4 + expected.len()] == expected);
            }
        }
        record(
            "keyboard",
            &[
                u64::from(started.result),
                u64::from(signaled),
                u64::from(completed.data.first().copied().unwrap_or(0)),
                u64::from(pushed.result),
                u64::from(extra.result),
                u64::from(null_push.result),
                u64::from(popped.result),
                storage_size,
                keyboard,
                text,
                u64::from(result.result),
            ],
        );
        c.halt();
    }

    #[test]
    fn the_software_keyboard_launch_returns_a_real_output_storage() {
        run_guest(keyboard_applet);
        assert_eq!(
            observed("keyboard"),
            vec![
                0,
                1,
                1,
                0,
                0,
                u64::from(RESULT_NULL_OBJECT),
                0,
                0x7d8,
                0,
                1,
                0,
            ]
        );
    }

    extern "C" fn player_select_applet(c: &mut Cpu) {
        let accessor = launch(c, APPLET_PLAYER_SELECT);
        let state = accessor.call(c, &request(0, Vec::new()));
        let before = event_signaled(c, state.copy_handles[0]);
        let out_event = accessor.call(c, &request(105, Vec::new()));
        let out_before = event_signaled(c, out_event.copy_handles[0]);

        let started = accessor.call(c, &request(10, Vec::new()));
        let state_signaled = event_signaled(c, state.copy_handles[0]);
        let out_signaled = event_signaled(c, out_event.copy_handles[0]);
        let completed = accessor.call(c, &request(1, Vec::new()));
        let result = accessor.call(c, &request(30, Vec::new()));
        let info = accessor.call(c, &request(120, Vec::new()));
        let popped = accessor.call(c, &request(101, Vec::new()));
        let out_after_pop = event_signaled(c, out_event.copy_handles[0]);

        let out = guest_buffer(c, 0x1_4000_0000, UI_RETURN_ARG_SIZE as u64);
        let mut storage_size = 0;
        let mut output_result = 0;
        let mut uid_is_local = 0;
        if let Some(storage) = popped.objects.first() {
            let reader = open(storage, c);
            let size = reader.call(c, &request(0, Vec::new()));
            storage_size = u64::from_le_bytes(size.data[..8].try_into().unwrap());
            let read = reader.call(
                c,
                &Request {
                    cmd: 11,
                    data: 0u64.to_le_bytes().to_vec(),
                    b: vec![Buf {
                        addr: out,
                        size: UI_RETURN_ARG_SIZE as u64,
                    }],
                    ..Default::default()
                },
            );
            if read.result == 0 {
                let bytes = c
                    .process()
                    .memory
                    .read(out, UI_RETURN_ARG_SIZE as u64)
                    .unwrap();
                output_result = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                uid_is_local = u64::from(bytes[8..] == crate::services::acc::USER_ID[..]);
            }
        }
        let second_pop = accessor.call(c, &request(101, Vec::new()));

        record(
            "player_select",
            &[
                u64::from(before),
                u64::from(out_before),
                u64::from(started.result),
                u64::from(state_signaled),
                u64::from(out_signaled),
                u64::from(completed.data.first().copied().unwrap_or(0)),
                u64::from(result.result),
                info.data.len() as u64,
                u32::from_le_bytes(info.data[..4].try_into().unwrap()) as u64,
                u64::from(popped.result),
                u64::from(out_after_pop),
                storage_size,
                output_result,
                uid_is_local,
                u64::from(second_pop.result),
            ],
        );
        c.halt();
    }

    #[test]
    fn the_player_select_launch_returns_the_local_user_and_completes() {
        run_guest(player_select_applet);
        assert_eq!(
            observed("player_select"),
            vec![
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                8,
                u64::from(APPLET_PLAYER_SELECT),
                0,
                0,
                UI_RETURN_ARG_SIZE as u64,
                0,
                1,
                u64::from(RESULT_NOT_AVAILABLE),
            ]
        );
    }

    extern "C" fn data_erase_applet(c: &mut Cpu) {
        let accessor = launch(c, APPLET_DATA_ERASE);
        let state = accessor.call(c, &request(0, Vec::new()));
        let out_event = accessor.call(c, &request(105, Vec::new()));
        let out_before = event_signaled(c, out_event.copy_handles[0]);
        let started = accessor.call(c, &request(10, Vec::new()));
        let state_signaled = event_signaled(c, state.copy_handles[0]);
        let out_signaled = event_signaled(c, out_event.copy_handles[0]);
        let completed = accessor.call(c, &request(1, Vec::new()));
        let popped = accessor.call(c, &request(101, Vec::new()));
        let out_after_pop = event_signaled(c, out_event.copy_handles[0]);

        let out = guest_buffer(c, 0x1_5000_0000, 0x10);
        let mut storage_size = 0;
        let mut spaces_are_zero = 0;
        if let Some(storage) = popped.objects.first() {
            let reader = open(storage, c);
            let size = reader.call(c, &request(0, Vec::new()));
            storage_size = u64::from_le_bytes(size.data[..8].try_into().unwrap());
            let read = reader.call(
                c,
                &Request {
                    cmd: 11,
                    data: 0u64.to_le_bytes().to_vec(),
                    b: vec![Buf {
                        addr: out,
                        size: 0x10,
                    }],
                    ..Default::default()
                },
            );
            if read.result == 0 {
                let bytes = c.process().memory.read(out, 0x10).unwrap();
                spaces_are_zero = u64::from(bytes.iter().all(|b| *b == 0));
            }
        }
        let second_pop = accessor.call(c, &request(101, Vec::new()));

        record(
            "data_erase",
            &[
                u64::from(out_before),
                u64::from(started.result),
                u64::from(state_signaled),
                u64::from(out_signaled),
                u64::from(completed.data.first().copied().unwrap_or(0)),
                u64::from(popped.result),
                u64::from(out_after_pop),
                storage_size,
                spaces_are_zero,
                u64::from(second_pop.result),
            ],
        );
        c.halt();
    }

    #[test]
    fn the_data_erase_launch_returns_a_space_storage_and_completes() {
        run_guest(data_erase_applet);
        assert_eq!(
            observed("data_erase"),
            vec![
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                DATA_ERASE_OUTPUT_SIZE as u64,
                1,
                u64::from(RESULT_NOT_AVAILABLE),
            ]
        );
    }

    extern "C" fn unknown_applet(c: &mut Cpu) {
        let accessor = launch(c, 0x7f);
        let started = accessor.call(c, &request(10, Vec::new()));
        let completed = accessor.call(c, &request(1, Vec::new()));
        let popped = accessor.call(c, &request(101, Vec::new()));
        let out_event = accessor.call(c, &request(105, Vec::new()));
        record(
            "unknown",
            &[
                u64::from(started.result),
                u64::from(completed.data.first().copied().unwrap_or(0)),
                u64::from(popped.result),
                popped.objects.len() as u64,
                u64::from(event_signaled(c, out_event.copy_handles[0])),
            ],
        );
        c.halt();
    }

    #[test]
    fn an_unknown_applet_completes_but_has_no_output_storage() {
        run_guest(unknown_applet);
        assert_eq!(
            observed("unknown"),
            vec![0, 1, u64::from(RESULT_NOT_AVAILABLE), 0, 0]
        );
    }

    extern "C" fn interactive_channel(c: &mut Cpu) {
        let accessor = launch(c, APPLET_SOFTWARE_KEYBOARD);
        let _ = accessor.call(c, &request(10, Vec::new()));
        let interactive = accessor.call(c, &request(104, Vec::new()));
        let normal = accessor.call(c, &request(101, Vec::new()));
        record(
            "interactive",
            &[u64::from(interactive.result), u64::from(normal.result), normal.objects.len() as u64],
        );
        c.halt();
    }

    #[test]
    fn popping_interactive_data_leaves_the_normal_output_in_place() {
        run_guest(interactive_channel);
        assert_eq!(observed("interactive"), vec![u64::from(RESULT_NOT_AVAILABLE), 0, 1]);
    }
}
