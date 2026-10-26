use std::sync::{Arc, Mutex};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

pub struct FriendServiceCreator;

impl FriendServiceCreator {
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

struct DaemonSuspendSessionService;

impl Service for DaemonSuspendSessionService {
    fn name(&self) -> String {
        "friend:u/IDaemonSuspendSessionService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        unimplemented_named("friend:u/IDaemonSuspendSessionService", "unknown", c, req)
    }
}

struct Notifications {
    event: u32,
}

pub struct NotificationService {
    queue: Arc<Notifications>,
}

impl Service for NotificationService {
    fn name(&self) -> String {
        "friend:u/INotificationService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented_named(
                "friend:u/INotificationService",
                notification_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

impl NotificationService {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            0 => Some(Reply::ok().copy_handle(self.queue.event)),
            1 => Some(Reply::ok()),
            2 => Some(Reply::ok().bytes(&[0; SIZED_NOTIFICATION_INFO_SIZE])),
            _ => None,
        }
    }
}

const SIZED_NOTIFICATION_INFO_SIZE: usize = 0x10;
const USER_PRESENCE_VIEW_SIZE: usize = 0xe0;
const USER_SETTING_SIZE: usize = 0x800;

pub struct FriendService {
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

    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            1 => Some(Reply::ok()),
            10100 | 10101 => Some(empty_list()),
            10102 => Some(Reply::ok()),
            10110 => Some(Reply::ok().u32(0)),
            10400 => Some(empty_list()),
            10500 => Some(Reply::ok()),
            10600 | 10601 | 10610 => Some(Reply::ok()),
            20100 | 20101 => Some(Reply::ok().u32(0)),
            20103 | 20104 => Some(Reply::ok()),
            20200 => Some(Reply::ok().u32(0).u32(0)),
            20201 | 20300 => Some(empty_list()),
            20400 => Some(empty_list()),
            20401 => Some(Reply::ok()),
            20500 => Some(Reply::ok()),
            20600 => Some(Reply::ok().bytes(&[0; USER_PRESENCE_VIEW_SIZE])),
            20700 => Some(empty_list()),
            20800 => Some(Reply::ok().bytes(&[0; USER_SETTING_SIZE])),
            20801 => Some(Reply::ok()),
            20900 => Some(Reply::ok()),
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

    fn request(cmd: u32) -> Request {
        Request {
            cmd,
            ..Default::default()
        }
    }

    #[test]
    fn the_notification_queue_has_its_event_and_empty_pops() {
        let service = NotificationService {
            queue: Arc::new(Notifications { event: 0x1234 }),
        };
        let event = service.command(&request(0)).expect("GetEvent");
        assert_eq!((event.result, event.copy_handles), (0, vec![0x1234]));
        let clear = service.command(&request(1)).expect("Clear");
        assert_eq!((clear.result, clear.data.len()), (0, 0));
        let pop = service.command(&request(2)).expect("Pop");
        assert_eq!(pop.result, 0);
        assert_eq!(pop.data, [0; SIZED_NOTIFICATION_INFO_SIZE], "an empty queue pops a zeroed notification");
    }

    #[test]
    fn unknown_commands_are_left_to_the_cmif_error() {
        let service = NotificationService {
            queue: Arc::new(Notifications { event: 1 }),
        };
        assert!(service.command(&request(3)).is_none());
        assert!(FriendService::new().command(&request(10103)).is_none());
        assert_eq!(creator_command_name(3), "unknown");
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
}
