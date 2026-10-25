use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

// nn::friends::detail::ipc::IServiceCreator. friend:u command 0 is CreateFriendService
// (https://switchbrew.org/wiki/Friend_services); the created IFriendService only needs
// GetCompletionEvent/Cancel for the callers Kirikae has seen.
pub struct FriendServiceCreator;

impl FriendServiceCreator {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            0 => Some(Reply::ok().object(Arc::new(FriendService))),
            _ => None,
        }
    }
}

impl Service for FriendServiceCreator {
    fn name(&self) -> String {
        "friend:u".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented("friend:u", c, req),
        }
    }
}

struct FriendService;

impl Service for FriendService {
    fn name(&self) -> String {
        "friend:u/IFriendService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            // GetCompletionEvent returns a manual-clear event. There is no friend list to
            // fetch, so it is already signalled.
            0 => Reply::ok().copy_handle(c.process().create_event(true)),
            1 => Reply::ok(),
            _ => unimplemented("friend:u/IFriendService", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_a_friend_service_object() {
        let r = FriendServiceCreator
            .command(&Request {
                cmd: 0,
                ..Default::default()
            })
            .expect("CreateFriendService");
        assert!(r.data.is_empty());
        assert_eq!(r.objects.len(), 1);
        assert_eq!(r.objects[0].name(), "friend:u/IFriendService");
    }

    #[test]
    fn unknown_creator_commands_are_not_handled() {
        assert!(
            FriendServiceCreator
                .command(&Request {
                    cmd: 1,
                    ..Default::default()
                })
                .is_none()
        );
    }
}
