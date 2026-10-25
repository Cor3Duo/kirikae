use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

/// `nn::pctl::RestrictionSettings`: the rating age followed by the two
/// restriction flags (https://github.com/switchbrew/libnx, `PctlRestrictionSettings`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RestrictionSettings {
    rating_age: u8,
    sns_post_restriction: bool,
    free_communication_restriction: bool,
}

impl RestrictionSettings {
    fn from_bytes(data: &[u8]) -> RestrictionSettings {
        RestrictionSettings {
            rating_age: data.first().copied().unwrap_or(0),
            sns_post_restriction: data.get(1).is_some_and(|&b| b != 0),
            free_communication_restriction: data.get(2).is_some_and(|&b| b != 0),
        }
    }

    fn bytes(self) -> [u8; 3] {
        [
            self.rating_age,
            self.sns_post_restriction as u8,
            self.free_communication_restriction as u8,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EventKind {
    Synchronization = 0,
    PlayTimer = 1,
    Unlinked = 2,
    PinCode = 3,
    ExtendedPlayTimer = 4,
}

impl EventKind {
    const COUNT: usize = 5;
}

/// A `pctl/IParentalControlService` reply, including the commands whose only
/// output is a copy handle. The handle is created on first use, which needs a
/// `Cpu`, so the command table only names the event.
enum Command {
    Reply(Reply),
    Event(EventKind),
}

/// State of one `pctl/IParentalControlService` session.
pub struct PctlState {
    /// PID the factory received from the caller: `CreateService` sends a
    /// reserved u64 and the PID as a copy handle. Parental control would use it
    /// to look up the restricted user; Kirikae keeps it so the session is not
    /// anonymous.
    pid: Option<u64>,
    initialized: AtomicBool,
    /// No parental controls are configured, so the broad restriction switch stays
    /// off and the free-communication and stereo-vision permissions are granted.
    restriction_enabled: AtomicBool,
    restriction_temporary_unlocked: AtomicBool,
    restricted_system_settings_entered: AtomicBool,
    stereo_vision_restriction: AtomicBool,
    stereo_vision_confirmed: AtomicBool,
    all_features_disabled: AtomicBool,
    safety_level: AtomicU32,
    restricted_features: AtomicU32,
    default_rating_organization: AtomicU32,
    settings: Mutex<RestrictionSettings>,
    play_timer_started: AtomicBool,
    play_timer_alarm_disabled: AtomicBool,
    play_timer_spent: AtomicU64,
    settings_last_updated: AtomicU64,
    pin_code_length: AtomicU32,
    events: Mutex<[Option<u32>; EventKind::COUNT]>,
}

impl PctlState {
    pub fn new(pid: Option<u64>, initialized: bool) -> PctlState {
        PctlState {
            pid,
            initialized: AtomicBool::new(initialized),
            restriction_enabled: AtomicBool::new(false),
            restriction_temporary_unlocked: AtomicBool::new(false),
            restricted_system_settings_entered: AtomicBool::new(false),
            stereo_vision_restriction: AtomicBool::new(false),
            stereo_vision_confirmed: AtomicBool::new(false),
            all_features_disabled: AtomicBool::new(false),
            safety_level: AtomicU32::new(0),
            restricted_features: AtomicU32::new(0),
            default_rating_organization: AtomicU32::new(0),
            settings: Mutex::new(RestrictionSettings::default()),
            play_timer_started: AtomicBool::new(false),
            play_timer_alarm_disabled: AtomicBool::new(false),
            play_timer_spent: AtomicU64::new(0),
            settings_last_updated: AtomicU64::new(0),
            pin_code_length: AtomicU32::new(0),
            events: Mutex::new([None; EventKind::COUNT]),
        }
    }

    /// PID the session was created for, when the caller sent one.
    pub fn pid(&self) -> Option<u64> {
        self.pid
    }

    fn event(&self, c: &Cpu, kind: EventKind) -> u32 {
        *self.events.lock().unwrap()[kind as usize]
            .get_or_insert_with(|| c.process().create_event(false))
    }

    fn settings(&self) -> RestrictionSettings {
        *self.settings.lock().unwrap()
    }

    #[cfg(test)]
    fn started(&self) -> bool {
        self.play_timer_started.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn pin_code_length(&self) -> u32 {
        self.pin_code_length.load(Ordering::Relaxed)
    }
}

pub struct Pctl {
    pub name: &'static str,
}

impl Pctl {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            // CreateService / CreateServiceWithoutInitialize
            // (https://switchbrew.org/wiki/Parental_Control_services).
            // CreateService initializes the session itself; the without-initialize
            // variant waits for command 1.
            0 => Some(Reply::ok().object(Arc::new(ParentalControlService {
                state: Arc::new(PctlState::new(req.pid, true)),
            }))),
            1 => Some(Reply::ok().object(Arc::new(ParentalControlService {
                state: Arc::new(PctlState::new(req.pid, false)),
            }))),
            _ => None,
        }
    }
}

impl Service for Pctl {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented(self.name, c, req),
        }
    }
}

struct ParentalControlService {
    state: Arc<PctlState>,
}

impl ParentalControlService {
    fn command(&self, req: &Request) -> Option<Command> {
        let s = &self.state;
        use Command::{Event, Reply as Data};
        let r = match req.cmd {
            // Initialize takes no input and returns no output: libnx sends cmd 1
            // bare right after CreateService on 4.0.0+.
            1 => {
                s.initialized.store(true, Ordering::Relaxed);
                Data(self.ok())
            }
            // The application asks whether the running title may use free
            // communication, SNS posts, system settings, etc. With no
            // restrictions configured every permission is granted.
            1001 => Data(self.ok()),
            1002..=1005 => Data(self.ok()),
            1006 => Data(
                self.ok()
                    .u8(s.restriction_temporary_unlocked.load(Ordering::Relaxed) as u8),
            ),
            1007 => {
                s.restriction_temporary_unlocked
                    .store(false, Ordering::Relaxed);
                Data(self.ok())
            }
            1008 => {
                s.restricted_system_settings_entered
                    .store(true, Ordering::Relaxed);
                Data(self.ok())
            }
            1009 => {
                s.restricted_system_settings_entered
                    .store(false, Ordering::Relaxed);
                Data(self.ok())
            }
            1010 => Data(
                self.ok()
                    .u8(s.restricted_system_settings_entered.load(Ordering::Relaxed) as u8),
            ),
            1011 => {
                s.restricted_system_settings_entered
                    .store(false, Ordering::Relaxed);
                Data(self.ok())
            }
            1012 => Data(self.ok().u32(s.restricted_features.load(Ordering::Relaxed))),
            1013 => {
                s.stereo_vision_confirmed.store(true, Ordering::Relaxed);
                Data(self.ok())
            }
            1014..=1016 => Data(self.ok()),
            1017 => Data(self.ok()),
            // No restriction is configured, so free communication is available.
            1018 => Data(self.ok().u8(1)),
            1019 | 1020 | 1022 => Data(self.ok()),
            1031 => Data(
                self.ok()
                    .u8(s.restriction_enabled.load(Ordering::Relaxed) as u8),
            ),
            1032 => Data(self.ok().u32(s.safety_level.load(Ordering::Relaxed))),
            1033 => {
                s.safety_level.store(req.u32(0), Ordering::Relaxed);
                Data(self.ok())
            }
            1034 | 1035 => Data(self.ok().bytes(&s.settings().bytes())),
            1036 => {
                *s.settings.lock().unwrap() = RestrictionSettings::from_bytes(&req.data);
                Data(self.ok())
            }
            1037 => Data(
                self.ok()
                    .u32(s.default_rating_organization.load(Ordering::Relaxed)),
            ),
            1038 => {
                s.default_rating_organization
                    .store(req.u32(0), Ordering::Relaxed);
                Data(self.ok())
            }
            1039 => Data(self.ok().u32(0)),
            1042 | 1045 | 1046 | 1047 | 1048 | 1049 | 1050 | 1051 => Data(self.ok()),
            // DeleteSettings clears the custom restriction settings.
            1043 => {
                *s.settings.lock().unwrap() = RestrictionSettings::default();
                Data(self.ok())
            }
            // GetFreeCommunicationApplicationList: no title is ever on the list.
            1044 => Data(self.ok().u32(0)),
            1061 => Data(self.ok()),
            1062 => Data(
                self.ok()
                    .u8(s.stereo_vision_restriction.load(Ordering::Relaxed) as u8),
            ),
            1063 => {
                s.stereo_vision_restriction
                    .store(req.data.first().is_some_and(|&b| b != 0), Ordering::Relaxed);
                Data(self.ok())
            }
            1064 => {
                s.stereo_vision_confirmed.store(false, Ordering::Relaxed);
                Data(self.ok())
            }
            // VrMode is permitted unless it is restricted and the application
            // has not confirmed the permission yet.
            1065 => {
                let permitted = !s.stereo_vision_restriction.load(Ordering::Relaxed)
                    || s.stereo_vision_confirmed.load(Ordering::Relaxed);
                Data(self.ok().u8(permitted as u8))
            }
            1201 | 1202 => {
                s.restriction_temporary_unlocked
                    .store(true, Ordering::Relaxed);
                Data(self.ok())
            }
            1203 => {
                let length = req.in_buf(0).map_or(0, |b| b.size) as u32;
                s.pin_code_length.store(length, Ordering::Relaxed);
                Data(self.ok())
            }
            // No device pairing exists, so the inquiry code is the zeroed buffer and
            // no master key can validate against it.
            1204 => Data(self.ok().bytes(&[0; 0x20])),
            1205 => Data(self.ok().u8(0)),
            1206 => Data(self.ok().u32(s.pin_code_length.load(Ordering::Relaxed))),
            1207 => Event(EventKind::PinCode),
            1208 => Data(self.ok().u32(0)),
            1403 => Data(self.ok().u8(0)),
            1406 => Data(
                self.ok()
                    .u64(s.settings_last_updated.load(Ordering::Relaxed)),
            ),
            // PairingAccountInfoBase and PairingInfoBase are 0x10-byte structs.
            1411 => Data(self.ok().bytes(&[0; 0x10])),
            1421 => Data(self.ok().u32(0)),
            1424 => Data(self.ok().u32(0)),
            1432 => Event(EventKind::Synchronization),
            1451 => {
                s.play_timer_started.store(true, Ordering::Relaxed);
                Data(self.ok())
            }
            1452 => {
                s.play_timer_started.store(false, Ordering::Relaxed);
                Data(self.ok())
            }
            // No play timer is configured, so nothing is counted down or restricted.
            1453 => Data(self.ok().u8(0)),
            1454 => Data(self.ok().u64(0)),
            1455 => Data(self.ok().u8(0)),
            // nn::pctl::PlayTimerSettings is a 0x34-byte struct; all-zero means no
            // timer, no bedtime and no remaining time.
            1456 => Data(self.ok().bytes(&[0; 0x34])),
            1457 => Event(EventKind::PlayTimer),
            1458 => Data(
                self.ok()
                    .u8(s.play_timer_alarm_disabled.load(Ordering::Relaxed) as u8),
            ),
            1471 | 1472 | 1474 | 1501 | 1901 | 1902 | 1941 => Data(self.ok()),
            1473 => Event(EventKind::Unlinked),
            1475 => Event(EventKind::ExtendedPlayTimer),
            1951 | 1953 => {
                if req.cmd == 1953 {
                    s.play_timer_alarm_disabled
                        .store(req.data.first().is_some_and(|&b| b != 0), Ordering::Relaxed);
                }
                Data(self.ok())
            }
            1952 => Data(self.ok().u64(s.play_timer_spent.load(Ordering::Relaxed))),
            1601 => {
                // DisableAllFeatures/PostEnableAllFeatures return a bool; the runtime
                // reports that the requested switch was applied.
                s.all_features_disabled.store(true, Ordering::Relaxed);
                Data(self.ok().u8(1))
            }
            1602 => {
                s.all_features_disabled.store(false, Ordering::Relaxed);
                Data(self.ok().u8(1))
            }
            1603 => Data(
                self.ok()
                    .u8(s.all_features_disabled.load(Ordering::Relaxed) as u8)
                    .u8(0),
            ),
            _ => return None,
        };
        Some(r)
    }

    fn ok(&self) -> Reply {
        Reply::ok()
    }
}

impl Service for ParentalControlService {
    fn name(&self) -> String {
        "pctl/IParentalControlService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(Command::Reply(r)) => r,
            Some(Command::Event(kind)) => Reply::ok().copy_handle(self.state.event(c, kind)),
            None => unimplemented("pctl/IParentalControlService", c, req),
        }
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

    fn empty_ok(reply: &Reply) {
        assert_eq!(reply.result, 0);
        assert!(
            reply.data.is_empty(),
            "unexpected reply data {:x?}",
            reply.data
        );
    }

    fn service() -> (Arc<PctlState>, ParentalControlService) {
        let state = Arc::new(PctlState::new(Some(7), false));
        let service = ParentalControlService {
            state: state.clone(),
        };
        (state, service)
    }

    fn reply(service: &ParentalControlService, cmd: u32) -> Reply {
        match service.command(&request(cmd)) {
            Some(Command::Reply(r)) => r,
            _ => panic!("command {cmd} should return a data reply"),
        }
    }

    #[test]
    fn create_service_returns_a_parental_control_service() {
        let factory = Pctl { name: "pctl" };
        for cmd in [0, 1] {
            let reply = factory
                .command(&request(cmd))
                .expect("expected a parental control service object");
            assert_eq!(reply.result, 0);
            assert!(reply.data.is_empty());
            assert_eq!(reply.objects.len(), 1);
            assert_eq!(reply.objects[0].name(), "pctl/IParentalControlService");
        }
        assert!(factory.command(&request(2)).is_none());
        assert!(factory.command(&request(1001)).is_none());
    }

    #[test]
    fn create_service_initializes_but_the_without_initialize_variant_waits() {
        assert!(
            PctlState::new(None, true)
                .initialized
                .load(Ordering::Relaxed)
        );
        assert!(
            !PctlState::new(None, false)
                .initialized
                .load(Ordering::Relaxed)
        );
    }

    #[test]
    fn initialize_is_an_empty_success_that_marks_the_session() {
        let (state, service) = service();
        assert!(!state.initialized.load(Ordering::Relaxed));

        empty_ok(&reply(&service, 1));
        assert!(state.initialized.load(Ordering::Relaxed));
        assert_eq!(state.pid(), Some(7));
    }

    #[test]
    fn restrictions_are_off_by_default() {
        let (_, service) = service();
        assert_eq!(reply(&service, 1031).data, [0]);
        assert_eq!(reply(&service, 1032).data, 0u32.to_le_bytes());
        assert_eq!(reply(&service, 1035).data, [0, 0, 0]);
        assert_eq!(reply(&service, 1012).data, 0u32.to_le_bytes());
        assert_eq!(reply(&service, 1039).data, 0u32.to_le_bytes());
        assert_eq!(reply(&service, 1018).data, [1]);
        assert_eq!(reply(&service, 1403).data, [0]);
        assert_eq!(reply(&service, 1406).data, 0u64.to_le_bytes());
        assert_eq!(reply(&service, 1206).data, 0u32.to_le_bytes());
    }

    #[test]
    fn current_settings_round_trip_the_rating_age_and_flags() {
        let (_, service) = service();
        let mut req = request(1036);
        req.data = vec![12, 1, 0];
        empty_ok(
            &service
                .command(&req)
                .expect("set custom settings")
                .into_reply(),
        );

        assert_eq!(reply(&service, 1035).data, [12, 1, 0]);
        assert_eq!(reply(&service, 1034).data, [12, 1, 0]);
    }

    #[test]
    fn safety_level_and_rating_organization_keep_their_value() {
        let (state, service) = service();
        let mut req = request(1033);
        req.data = 3u32.to_le_bytes().to_vec();
        empty_ok(
            &service
                .command(&req)
                .expect("set safety level")
                .into_reply(),
        );
        assert_eq!(state.safety_level.load(Ordering::Relaxed), 3);
        assert_eq!(reply(&service, 1032).data, 3u32.to_le_bytes());

        let mut req = request(1038);
        req.data = 2u32.to_le_bytes().to_vec();
        empty_ok(
            &service
                .command(&req)
                .expect("set rating organization")
                .into_reply(),
        );
        assert_eq!(reply(&service, 1037).data, 2u32.to_le_bytes());
    }

    #[test]
    fn stereo_vision_permission_follows_the_restriction_and_confirmations() {
        let (_, service) = service();
        assert_eq!(reply(&service, 1065).data, [1]);
        assert_eq!(reply(&service, 1062).data, [0]);

        let mut req = request(1063);
        req.data = vec![1];
        empty_ok(
            &service
                .command(&req)
                .expect("set stereo restriction")
                .into_reply(),
        );
        assert_eq!(reply(&service, 1062).data, [1]);
        assert_eq!(reply(&service, 1065).data, [0]);

        empty_ok(&reply(&service, 1013));
        assert_eq!(reply(&service, 1065).data, [1]);
        empty_ok(&reply(&service, 1064));
        assert_eq!(reply(&service, 1065).data, [0]);
    }

    #[test]
    fn free_communication_and_temporary_unlock_follow_the_commands() {
        let (_, service) = service();
        empty_ok(&reply(&service, 1001));
        assert_eq!(reply(&service, 1018).data, [1]);
        empty_ok(&reply(&service, 1017));

        assert_eq!(reply(&service, 1006).data, [0]);
        empty_ok(&reply(&service, 1201));
        assert_eq!(reply(&service, 1006).data, [1]);
        empty_ok(&reply(&service, 1007));
        assert_eq!(reply(&service, 1006).data, [0]);
    }

    #[test]
    fn restricted_system_settings_state_is_entered_and_reverted() {
        let (_, service) = service();
        assert_eq!(reply(&service, 1010).data, [0]);
        empty_ok(&reply(&service, 1008));
        assert_eq!(reply(&service, 1010).data, [1]);
        empty_ok(&reply(&service, 1009));
        assert_eq!(reply(&service, 1010).data, [0]);
        empty_ok(&reply(&service, 1008));
        empty_ok(&reply(&service, 1011));
        assert_eq!(reply(&service, 1010).data, [0]);
    }

    #[test]
    fn play_timer_starts_and_stops_without_restricting_the_title() {
        let (state, service) = service();
        assert!(!state.started());
        assert_eq!(reply(&service, 1453).data, [0]);
        assert_eq!(reply(&service, 1454).data, 0u64.to_le_bytes());
        assert_eq!(reply(&service, 1455).data, [0]);

        empty_ok(&reply(&service, 1451));
        assert!(state.started());
        empty_ok(&reply(&service, 1452));
        assert!(!state.started());

        assert_eq!(reply(&service, 1952).data, 0u64.to_le_bytes());
        assert_eq!(reply(&service, 1458).data, [0]);
        let mut req = request(1953);
        req.data = vec![1];
        empty_ok(
            &service
                .command(&req)
                .expect("disable play timer alarm")
                .into_reply(),
        );
        assert_eq!(reply(&service, 1458).data, [1]);
    }

    #[test]
    fn set_pin_code_records_the_length() {
        let (state, service) = service();
        let mut req = request(1203);
        req.x.push(crate::services::Buf { addr: 0, size: 4 });
        empty_ok(&service.command(&req).expect("set pin code").into_reply());
        assert_eq!(state.pin_code_length(), 4);
        assert_eq!(reply(&service, 1206).data, 4u32.to_le_bytes());
    }

    #[test]
    fn all_feature_switch_is_recorded() {
        let (state, service) = service();
        assert_eq!(reply(&service, 1603).data, [0, 0]);
        assert_eq!(reply(&service, 1601).data, [1]);
        assert!(state.all_features_disabled.load(Ordering::Relaxed));
        assert_eq!(reply(&service, 1603).data, [1, 0]);
        assert_eq!(reply(&service, 1602).data, [1]);
        assert!(!state.all_features_disabled.load(Ordering::Relaxed));
        assert_eq!(reply(&service, 1603).data, [0, 0]);
    }

    #[test]
    fn event_commands_name_their_event_instead_of_a_data_reply() {
        let (_, service) = service();
        for (cmd, kind) in [
            (1207, EventKind::PinCode),
            (1432, EventKind::Synchronization),
            (1457, EventKind::PlayTimer),
            (1473, EventKind::Unlinked),
            (1475, EventKind::ExtendedPlayTimer),
        ] {
            match service.command(&request(cmd)) {
                Some(Command::Event(actual)) => assert_eq!(actual, kind, "command {cmd}"),
                _ => panic!("command {cmd} should return an event"),
            }
        }
    }

    #[test]
    fn pairing_and_play_timer_structs_have_the_documented_sizes() {
        let (_, service) = service();
        // nn::pctl::InquiryCode, PlayTimerSettings and PairingAccountInfoBase.
        assert_eq!(reply(&service, 1204).data, [0; 0x20]);
        assert_eq!(reply(&service, 1205).data, [0]);
        assert_eq!(reply(&service, 1208).data, 0u32.to_le_bytes());
        assert_eq!(reply(&service, 1411).data, [0; 0x10]);
        assert_eq!(reply(&service, 1421).data, 0u32.to_le_bytes());
        assert_eq!(reply(&service, 1424).data, 0u32.to_le_bytes());
        assert_eq!(reply(&service, 1456).data, [0; 0x34]);
    }

    #[test]
    fn unknown_commands_are_not_handled() {
        let (_, service) = service();
        for cmd in [0, 2, 1000, 1412, 2027] {
            assert!(service.command(&request(cmd)).is_none(), "command {cmd}");
        }
    }

    impl Command {
        fn into_reply(self) -> Reply {
            match self {
                Command::Reply(r) => r,
                Command::Event(_) => panic!("expected a data reply"),
            }
        }
    }
}
