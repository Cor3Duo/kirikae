//! `friend:u` — friends, presence and notifications.
//!
//! The runtime models a console with no Nintendo Network account: there are no
//! friends, no blocked users, no requests and no notifications. Every list
//! query answers "empty", every service object the SDK asks for is a real one,
//! and the notification queue's event exists but is never signalled. That is
//! the offline shape the SDK expects; a short or zero-filled reply that leaves
//! an object pointer null crashes the guest on the next virtual call.
//!
//! Command numbers and reply shapes follow the Switchbrew "Friend services"
//! page (`nn::friends::detail::ipc::IServiceCreator`, `IFriendService`,
//! `INotificationService`). Ryujinx has no `friend:u` implementation to compare
//! against (its `HOS/Services` tree has no `Friend` folder), so the wire sizes
//! were checked against the SDK proxies in the title's `sdk` module: the
//! `Out<int>` of a list query is the entry count, `GetUserPresenceView` writes
//! 0xe0 bytes and `LoadUserSetting` 0x800.

use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// `nn::friends::detail::ipc::IServiceCreator`. `friend:u`, `friend:v`,
/// `friend:m`, `friend:s` and `friend:a` all expose the same creator.
pub struct FriendServiceCreator;

impl FriendServiceCreator {
    /// Creator command 1: one `INotificationService` per call, all of them
    /// sharing the process' notification queue (one event, as on the console).
    fn create_notification_service(c: &Cpu) -> Reply {
        let p = c.process();
        let queue = p.service_state("friend:u/notifications", || Notifications {
            event: p.create_event(false),
        });
        Reply::ok().object(Arc::new(NotificationService { queue }))
    }
}

impl Service for FriendServiceCreator {
    fn name(&self) -> String {
        "friend:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(FriendService::new())),
            1 => Self::create_notification_service(c),
            2 => Reply::ok().object(Arc::new(DaemonSuspendSessionService)),
            _ => unimplemented_named("friend:u", creator_command_name(req.cmd), c, req),
        }
    }
}

/// `nn::friends::detail::ipc::IDaemonSuspendSessionService`: no commands, but
/// the object must exist because the creator returns a handle to it.
struct DaemonSuspendSessionService;

impl Service for DaemonSuspendSessionService {
    fn name(&self) -> String {
        "friend:u/IDaemonSuspendSessionService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        unimplemented_named("friend:u/IDaemonSuspendSessionService", "unknown", c, req)
    }
}

/// The notification queue of one user. An offline console has nothing to
/// deliver, so the event is never signalled and no notification is ever
/// pending; the handle is shared by every `INotificationService` of the
/// process, which is what `nn::friends::NotificationQueue` waits on.
struct Notifications {
    event: u32,
}

/// `nn::friends::detail::ipc::INotificationService`.
pub struct NotificationService {
    queue: Arc<Notifications>,
}

impl Service for NotificationService {
    fn name(&self) -> String {
        "friend:u/INotificationService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // GetEvent: the manual-clear event the queue waits on. With no
            // notifications it stays unsignalled (`Reply::copy_handle`, as the
            // page says this command returns an output handle).
            0 => Reply::ok().copy_handle(self.queue.event),
            // Clear: drops every pending notification. There are none.
            1 => Reply::ok(),
            // Pop: a `SizedNotificationInfo` (0x10 bytes on the wire). The
            // queue is empty, so the sized structure is zero: no notification
            // id and a zero payload length.
            2 => Reply::ok().bytes(&[0; SIZED_NOTIFICATION_INFO_SIZE]),
            _ => unimplemented_named(
                "friend:u/INotificationService",
                notification_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

const SIZED_NOTIFICATION_INFO_SIZE: usize = 0x10;
/// `nn::friends::UserPresenceView` as the SDK asks for it (`Out` size 0xe0).
const USER_PRESENCE_VIEW_SIZE: usize = 0xe0;
/// `nn::friends::UserSettingImpl` as the SDK asks for it (`Out` size 0x800).
const USER_SETTING_SIZE: usize = 0x800;

/// `nn::friends::detail::ipc::IFriendService`.
pub struct FriendService {
    /// The completion event is manual-clear: every request in this runtime is
    /// answered synchronously, so it is created signalled and stays that way.
    completion: Mutex<Option<u32>>,
}

impl FriendService {
    fn new() -> FriendService {
        FriendService {
            completion: Mutex::new(None),
        }
    }

    fn completion_event(&self, c: &Cpu) -> u32 {
        let mut slot = self.completion.lock().unwrap();
        *slot.get_or_insert_with(|| c.process().create_event(true))
    }

    /// Commands whose answer does not depend on the process. `None` means the
    /// command is not part of the offline model and the caller reports the
    /// CMIF unknown-command result.
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            // Cancel: the synchronous queries above have nothing to cancel.
            1 => Some(Reply::ok()),
            // GetFriendListIds / GetFriendList: `Out<int>` count, no entries.
            10100 | 10101 => Some(empty_list()),
            // UpdateFriendInfo: writes into the caller's OutArray; empty.
            10102 => Some(Reply::ok()),
            // GetFriendProfileImage: `Out<int>` bytes written to the buffer.
            10110 => Some(Reply::ok().u32(0)),
            // GetBlockedUserListIds: `Out<int>` count, no entries.
            10400 => Some(empty_list()),
            // GetProfileList: only an OutArray (the ids came in); empty.
            10500 => Some(Reply::ok()),
            // DeclareOpenOnlinePlaySession / DeclareCloseOnlinePlaySession /
            // UpdateUserPresence: no output.
            10600 | 10601 | 10610 => Some(Reply::ok()),
            // GetFriendCount / GetNewlyFriendCount.
            20100 | 20101 => Some(Reply::ok().u32(0)),
            // SyncFriendList / RequestSyncFriendList: a synchronous empty sync.
            20103 | 20104 => Some(Reply::ok()),
            // GetReceivedFriendRequestCount: (received, unread).
            20200 => Some(Reply::ok().u32(0).u32(0)),
            // GetFriendRequestList / GetFriendCandidateList.
            20201 | 20300 => Some(empty_list()),
            // GetBlockedUserList / SyncBlockedUserList.
            20400 => Some(empty_list()),
            20401 => Some(Reply::ok()),
            // GetProfileExtraList: only an OutArray, like GetProfileList.
            20500 => Some(Reply::ok()),
            // GetUserPresenceView: an all-zero view is "offline, no game".
            20600 => Some(Reply::ok().bytes(&[0; USER_PRESENCE_VIEW_SIZE])),
            // GetPlayHistoryList.
            20700 => Some(empty_list()),
            // LoadUserSetting: all-zero settings (nothing registered).
            20800 => Some(Reply::ok().bytes(&[0; USER_SETTING_SIZE])),
            // SyncUserSetting: saved.
            20801 => Some(Reply::ok()),
            // RequestListSummaryOverlayNotification: there is nothing to show.
            20900 => Some(Reply::ok()),
            // Flag edits and presence/friend-request changes on an offline
            // console: accepted, and there is no friend list to change.
            30100 | 30110 | 30120 | 30121 => Some(Reply::ok()),
            _ => None,
        }
    }
}

impl Service for FriendService {
    fn name(&self) -> String {
        "friend:u/IFriendService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        // GetCompletionEvent: a manual-clear event, already signalled because
        // every answer is immediate.
        if req.cmd == 0 {
            return Reply::ok().copy_handle(self.completion_event(c));
        }
        match self.command(req) {
            Some(r) => r,
            None => unimplemented_named(
                "friend:u/IFriendService",
                friend_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

/// A list query with no entries: the `Out<int>` count is zero and the SDK
/// leaves its OutArray untouched.
fn empty_list() -> Reply {
    Reply::ok().u32(0)
}

fn creator_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "CreateFriendService",
        1 => "CreateNotificationService",
        2 => "CreateDaemonSuspendSessionService",
        _ => "unknown",
    }
}

fn notification_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "GetEvent",
        1 => "Clear",
        2 => "Pop",
        _ => "unknown",
    }
}

fn friend_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "GetCompletionEvent",
        1 => "Cancel",
        10100 => "GetFriendListIds",
        10101 => "GetFriendList",
        10102 => "UpdateFriendInfo",
        10110 => "GetFriendProfileImage",
        10400 => "GetBlockedUserListIds",
        10500 => "GetProfileList",
        10600 => "DeclareOpenOnlinePlaySession",
        10601 => "DeclareCloseOnlinePlaySession",
        10610 => "UpdateUserPresence",
        20100 => "GetFriendCount",
        20101 => "GetNewlyFriendCount",
        20103 => "SyncFriendList",
        20104 => "RequestSyncFriendList",
        20200 => "GetReceivedFriendRequestCount",
        20201 => "GetFriendRequestList",
        20300 => "GetFriendCandidateList",
        20400 => "GetBlockedUserList",
        20401 => "SyncBlockedUserList",
        20500 => "GetProfileExtraList",
        20600 => "GetUserPresenceView",
        20700 => "GetPlayHistoryList",
        20800 => "LoadUserSetting",
        20801 => "SyncUserSetting",
        20900 => "RequestListSummaryOverlayNotification",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::ipc::{CONTROL_CONVERT_TO_DOMAIN, TYPE_CONTROL, TYPE_REQUEST};
    use crate::kernel::KObject;
    use crate::services::{Buf, harness, result};

    /// Builds a domain CMIF request whose HIPC message also carries an SDK
    /// pointer buffer and a copy handle, the descriptors the real
    /// `CreateNotificationService` path can add.
    fn domain_request(
        cmd: u32,
        object: u32,
        args: &[u8],
        pointer: Option<Buf>,
        copy: &[u32],
    ) -> Vec<u8> {
        let mut body = Vec::new();
        for w in [crate::ipc::SFCI, 1, cmd, 0] {
            body.extend_from_slice(&w.to_le_bytes());
        }
        body.extend_from_slice(args);

        let mut payload = Vec::new();
        payload.push(1u8); // domain send
        payload.push(0);
        payload.extend_from_slice(&(body.len() as u16).to_le_bytes());
        for w in [object, 0, 0] {
            payload.extend_from_slice(&w.to_le_bytes());
        }
        payload.extend_from_slice(&body);

        let special = pointer.is_some() || !copy.is_empty();
        let mut out = Vec::new();
        let nx = pointer.is_some() as u32;
        let h0 = TYPE_REQUEST as u32 | nx << 16;
        let h1 = (special as u32) << 31;
        out.extend_from_slice(&h0.to_le_bytes());
        out.extend_from_slice(&h1.to_le_bytes());
        if special {
            out.extend_from_slice(&((copy.len() as u32) << 1).to_le_bytes());
            for h in copy {
                out.extend_from_slice(&h.to_le_bytes());
            }
        }
        if let Some(buf) = pointer {
            for w in crate::ipc::encode_pointer(buf) {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        let data_start = out.len().next_multiple_of(16);
        let pad = (data_start - out.len()) / 4;
        out.resize(data_start, 0);
        out.extend_from_slice(&payload);
        let data_words = pad + payload.len().div_ceil(4);
        let h1 = data_words as u32 | (special as u32) << 31;
        out[4..8].copy_from_slice(&h1.to_le_bytes());
        out.resize(data_start + data_words * 4, 0);
        out
    }

    /// Splits a CMIF domain reply into its object ids and SFCO payload. The
    /// runtime puts the out-object ids after the CMIF data, so `data_len` is
    /// the service data length the command answers with.
    fn domain_reply(msg: &[u8], data_len: usize) -> (u32, Vec<u32>, u32, Vec<u8>) {
        let m = crate::ipc::parse_hipc(msg).expect("reply parses");
        let pad = m.data_offset.next_multiple_of(16) - m.data_offset;
        let p = &m.data[pad..];
        let count = u32::from_le_bytes(p[..4].try_into().unwrap());
        let result = u32::from_le_bytes(p[24..28].try_into().unwrap());
        let data = p[32..32 + data_len].to_vec();
        let ids_at = 32 + data_len.next_multiple_of(4);
        let ids = (0..count as usize)
            .map(|i| u32::from_le_bytes(p[ids_at + 4 * i..ids_at + 4 * i + 4].try_into().unwrap()))
            .collect();
        (count, ids, result, data)
    }

    fn convert_to_domain(c: &mut Cpu, handle: u32) {
        let msg = crate::ipc::client::cmif(
            TYPE_CONTROL,
            CONTROL_CONVERT_TO_DOMAIN,
            &[],
            None,
            false,
            &[],
        );
        let reply = harness::send(c, handle, &msg);
        let (result, _, _) = crate::ipc::client::cmif_response(&reply, false);
        assert_eq!(result, 0, "ConvertToDomain");
    }

    #[test]
    fn create_notification_service_answers_a_domain_object_and_no_data() {
        extern "C" fn probe(c: &mut Cpu) {
            let handle = harness::session(c, Arc::new(FriendServiceCreator));
            convert_to_domain(c, handle);
            let uid = 0x7073_6e64_0000_0001u64.to_le_bytes();
            let mut args = uid.to_vec();
            args.extend_from_slice(&0u64.to_le_bytes());
            let msg = domain_request(1, 1, &args, None, &[]);
            let reply = harness::send(c, handle, &msg);
            let (count, ids, result, data) = domain_reply(&reply, 0);
            harness::record(
                "create",
                &[
                    result as u64,
                    count as u64,
                    data.len() as u64,
                    ids.first().copied().unwrap_or(0) as u64,
                ],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let create = harness::observed("create");
        assert_eq!(
            create[0], 0,
            "CreateNotificationService returns success, not an unknown-command error"
        );
        assert_eq!(create[1], 1, "exactly one out object in the domain reply");
        assert_eq!(create[2], 0, "an Out<SharedPointer> has no raw reply data");
        assert_ne!(create[3], 0, "the object id must not be the null object");
    }

    #[test]
    fn the_notification_queue_has_a_real_event_and_empty_pops() {
        extern "C" fn probe(c: &mut Cpu) {
            let handle = harness::session(c, Arc::new(FriendServiceCreator));
            convert_to_domain(c, handle);
            let msg = domain_request(1, 1, &[0; 16], None, &[]);
            let reply = harness::send(c, handle, &msg);
            let (_, ids, _, _) = domain_reply(&reply, 0);
            let object = ids.first().copied().expect("notification service id");

            let msg = domain_request(0, object, &[], None, &[]);
            let reply = harness::send(c, handle, &msg);
            let m = crate::ipc::parse_hipc(&reply).unwrap();
            let event = m.copy_handles.first().copied().unwrap_or(0);
            harness::record(
                "event",
                &[
                    event as u64,
                    matches!(
                        c.process().handles.lock().unwrap().get(event),
                        Some(KObject::Event(_))
                    ) as u64,
                ],
            );

            let msg = domain_request(2, object, &[], None, &[]);
            let reply = harness::send(c, handle, &msg);
            let (_, _, result, data) = domain_reply(&reply, SIZED_NOTIFICATION_INFO_SIZE);
            harness::record("pop", &[result as u64, data.len() as u64]);

            let msg = domain_request(1, object, &[], None, &[]);
            let reply = harness::send(c, handle, &msg);
            let (_, _, result, data) = domain_reply(&reply, 0);
            harness::record("clear", &[result as u64, data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let event = harness::observed("event");
        assert_ne!(event[0], 0, "GetEvent returns a handle");
        assert_eq!(event[1], 1, "the handle names a real kernel event");
        assert_eq!(
            harness::observed("pop"),
            vec![0, SIZED_NOTIFICATION_INFO_SIZE as u64],
            "an empty queue pops a zero-sized notification, not an error"
        );
        assert_eq!(harness::observed("clear"), vec![0, 0]);
    }

    #[test]
    fn sdk_shaped_request_with_a_pointer_buffer_and_copy_handle() {
        extern "C" fn probe(c: &mut Cpu) {
            let handle = harness::session(c, Arc::new(FriendServiceCreator));
            convert_to_domain(c, handle);
            // The shape the SDK uses: a type-0x9 pointer buffer for the
            // arguments plus the copy handle it sends with the request.
            let pointer = Buf {
                addr: 0x1_0000_2000,
                size: 0x10,
            };
            let msg = domain_request(1, 1, &[0; 16], Some(pointer), &[0x20030]);
            let reply = harness::send(c, handle, &msg);
            let (count, ids, result, data) = domain_reply(&reply, 0);
            harness::record(
                "create",
                &[
                    result as u64,
                    count as u64,
                    data.len() as u64,
                    ids.first().copied().unwrap_or(0) as u64,
                ],
            );

            let object = ids.first().copied().unwrap_or(0);
            let msg = domain_request(0, object, &[], Some(pointer), &[0x20031]);
            let reply = harness::send(c, handle, &msg);
            let m = crate::ipc::parse_hipc(&reply).unwrap();
            harness::record("event", &[m.copy_handles.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let create = harness::observed("create");
        assert_eq!(create[0], 0);
        assert_eq!((create[1], create[2]), (1, 0));
        assert_ne!(create[3], 0);
        assert_eq!(harness::observed("event"), vec![1]);
    }

    #[test]
    fn unknown_commands_report_the_cmif_error() {
        extern "C" fn probe(c: &mut Cpu) {
            let r = FriendServiceCreator.call(
                c,
                &Request {
                    cmd: 3,
                    ..Default::default()
                },
            );
            harness::record("creator", &[r.result as u64, r.objects.len() as u64]);
            let r = FriendService::new().call(
                c,
                &Request {
                    cmd: 10103,
                    ..Default::default()
                },
            );
            harness::record("friend", &[r.result as u64, r.data.len() as u64]);
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        assert_eq!(
            harness::observed("creator"),
            vec![result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
        assert_eq!(
            harness::observed("friend"),
            vec![result::CMIF_UNKNOWN_COMMAND as u64, 0]
        );
    }

    #[test]
    fn list_queries_answer_an_empty_count() {
        for cmd in [10100, 10101, 10400, 20201, 20300, 20400, 20700] {
            let r = FriendService::new()
                .command(&Request {
                    cmd,
                    ..Default::default()
                })
                .expect("a list query");
            assert_eq!(r.result, 0);
            assert_eq!(r.data, 0u32.to_le_bytes(), "command {cmd}");
        }
        let requests = FriendService::new()
            .command(&Request {
                cmd: 20200,
                ..Default::default()
            })
            .expect("GetReceivedFriendRequestCount");
        assert_eq!(requests.data, [0; 8]);
        let presence = FriendService::new()
            .command(&Request {
                cmd: 20600,
                ..Default::default()
            })
            .expect("GetUserPresenceView");
        assert_eq!(presence.data.len(), USER_PRESENCE_VIEW_SIZE);
        assert!(presence.data.iter().all(|&b| b == 0));
        let setting = FriendService::new()
            .command(&Request {
                cmd: 20800,
                ..Default::default()
            })
            .expect("LoadUserSetting");
        assert_eq!(setting.data.len(), USER_SETTING_SIZE);
        assert!(setting.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn the_completion_event_is_a_signalled_copy() {
        extern "C" fn probe(c: &mut Cpu) {
            let service = FriendService::new();
            let r = service.call(
                c,
                &Request {
                    cmd: 0,
                    ..Default::default()
                },
            );
            let handle = r.copy_handles.first().copied().unwrap_or(0);
            let signaled = match c.process().handles.lock().unwrap().get(handle) {
                Some(KObject::Event(e)) => e.signaled.load(std::sync::atomic::Ordering::Relaxed),
                _ => false,
            };
            harness::record(
                "completion",
                &[handle as u64, signaled as u64, r.data.len() as u64],
            );
            // A second call reuses the same handle instead of leaking a new one.
            let again = service.call(
                c,
                &Request {
                    cmd: 0,
                    ..Default::default()
                },
            );
            harness::record(
                "same",
                &[again.copy_handles.first().copied().unwrap_or(0) as u64],
            );
            c.halt();
        }
        harness::run(vec![Entry {
            addr: harness::ENTRY,
            func: probe,
        }]);
        let completion = harness::observed("completion");
        assert_ne!(completion[0], 0);
        assert_eq!((completion[1], completion[2]), (1, 0));
        assert_eq!(harness::observed("same"), vec![completion[0]]);
    }
}
