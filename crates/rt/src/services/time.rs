use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cpu::Cpu;
use crate::kernel::SharedMemory;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const SHARED_SIZE: u64 = 0x1000;
const CLOCK_SOURCE: [u8; 16] = *b"kirikae-steady\0\0";
const LOCATION: &str = "UTC";

/// `nn::time::sf::ClockSnapshot` is 0xD0 bytes: two `SystemClockContext`s, the
/// user and network POSIX times, their calendar forms and additional info, the
/// steady time point, the location name, the automatic-correction flag, the
/// requested `nn::time::TimeType` and a reserved u16.
const CLOCK_SNAPSHOT_SIZE: usize = 0xd0;

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn steady_seconds(c: &Cpu) -> i64 {
    (c.process().ticks() / crate::horizon::TICK_HZ) as i64
}

fn time_point(c: &Cpu) -> Vec<u8> {
    let mut v = steady_seconds(c).to_le_bytes().to_vec();
    v.extend_from_slice(&CLOCK_SOURCE);
    v
}

fn system_context(c: &Cpu) -> Vec<u8> {
    let offset = now_unix() - steady_seconds(c);
    let mut v = offset.to_le_bytes().to_vec();
    v.extend(time_point(c));
    v
}

fn put_time_point(b: &mut [u8], offset: usize, seconds: i64) {
    b[offset..offset + 8].copy_from_slice(&seconds.to_le_bytes());
    b[offset + 8..offset + 0x18].copy_from_slice(&CLOCK_SOURCE);
}

/// `nn::time::sf::ClockSnapshot` as `GetClockSnapshot` returns it
/// (https://switchbrew.org/wiki/PCV_services#time:u,_time:a,_time:s, command 400;
/// layout from Ryujinx's `ClockSnapshot`):
///
/// | Offset | Field |
/// |---|---|
/// | 0x00 | User `SystemClockContext` |
/// | 0x20 | Network `SystemClockContext` |
/// | 0x40 | User POSIX time (s64) |
/// | 0x48 | Network POSIX time (s64) |
/// | 0x50 | User `CalendarTime` |
/// | 0x58 | Network `CalendarTime` |
/// | 0x60 | User `CalendarAdditionalInfo` |
/// | 0x78 | Network `CalendarAdditionalInfo` |
/// | 0x90 | `SteadyClockTimePoint` |
/// | 0xA8 | `LocationName` (0x24 bytes) |
/// | 0xCC | `IsAutomaticCorrectionEnabled` (bool) |
/// | 0xCD | Requested `TimeType` |
/// | 0xCE | Reserved u16 |
///
/// There is no network clock, so both times are the same user time and both
/// contexts point at the synthetic steady clock.
fn clock_snapshot(steady: i64, unix: i64, clock_type: u8) -> [u8; CLOCK_SNAPSHOT_SIZE] {
    let mut b = [0; CLOCK_SNAPSHOT_SIZE];
    let offset = unix - steady;
    for context in [0x00, 0x20] {
        b[context..context + 8].copy_from_slice(&offset.to_le_bytes());
        put_time_point(&mut b, context + 8, steady);
    }
    b[0x40..0x48].copy_from_slice(&unix.to_le_bytes());
    b[0x48..0x50].copy_from_slice(&unix.to_le_bytes());
    let cal = calendar(unix);
    b[0x50..0x58].copy_from_slice(&cal[..8]);
    b[0x58..0x60].copy_from_slice(&cal[..8]);
    b[0x60..0x78].copy_from_slice(&cal[8..0x20]);
    b[0x78..0x90].copy_from_slice(&cal[8..0x20]);
    put_time_point(&mut b, 0x90, steady);
    b[0xa8..0xcc].copy_from_slice(&location_name());
    // The runtime reports the automatic correction as disabled in command 100, so the
    // snapshot keeps the same answer.
    b[0xcc] = 0;
    b[0xcd] = clock_type;
    b
}

/// Where `GetClockSnapshot` has to put the snapshot: the type-0x1a receive
/// pointer buffer. The HIPC parser files those under `c`; `x`/`b` are accepted
/// too for callers that arrive through another descriptor form. A buffer
/// smaller than the snapshot is left untouched, like the other commands in this
/// service.
fn clock_snapshot_output(
    req: &Request,
    steady: i64,
    unix: i64,
) -> Option<(u64, [u8; CLOCK_SNAPSHOT_SIZE])> {
    let buf = req
        .c
        .first()
        .or_else(|| req.x.first())
        .or_else(|| req.b.first())?;
    (buf.size >= CLOCK_SNAPSHOT_SIZE as u64).then(|| {
        (
            buf.addr,
            clock_snapshot(steady, unix, req.data.first().copied().unwrap_or(0)),
        )
    })
}

fn shared(c: &Cpu) -> (Arc<SharedMemory>, u32) {
    let s = c.process().service_state("time.shared", || {
        let p = c.process();
        let (mem, handle) = p.create_shared_memory(SHARED_SIZE);
        let put = |offset: u64, padding: u64, value: &[u8]| {
            mem.write(&p.memory, offset, &1u32.to_le_bytes());
            mem.write(&p.memory, offset + 4 + padding + value.len() as u64, value);
        };
        let mut steady = 0u64.to_le_bytes().to_vec();
        steady.extend_from_slice(&CLOCK_SOURCE);
        put(0x00, 4, &steady);
        let ctx = system_context(c);
        put(0x38, 4, &ctx);
        put(0x80, 4, &ctx);
        put(0xc8, 0, &[0]);
        (mem, handle)
    });
    (s.0.clone(), s.1)
}

pub struct Time {
    pub name: &'static str,
}

impl Time {
    /// Commands whose reply does not need the process, kept apart from `call` so they
    /// can be tested without a guest CPU.
    fn stateless_command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            // IsStandardNetworkSystemClockAccuracySufficient: no network clock is
            // provided, so its accuracy is never sufficient. Output bool.
            200 => Some(Reply::ok().u8(0)),
            _ => None,
        }
    }
}

impl Service for Time {
    fn name(&self) -> String {
        self.name.into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        if let Some(r) = self.stateless_command(req) {
            return r;
        }
        match req.cmd {
            0 | 1 | 4 | 5 => Reply::ok().object(Arc::new(SystemClock)),
            2 => Reply::ok().object(Arc::new(SteadyClock)),
            3 => Reply::ok().object(Arc::new(TimeZone)),
            20 => Reply::ok().copy_handle(shared(c).1),
            100 => Reply::ok().u8(0),
            300 => {
                let base = req.u64(0) as i64 + req.u64(8) as i64;
                Reply::ok().u64(base as u64)
            }
            // 400 GetClockSnapshot(nn::time::TimeType) -> nn::time::sf::ClockSnapshot:
            // writes the 0xD0-byte snapshot to the receive pointer buffer, no raw
            // output. The buffer counts as a pointer list (type 0x1a) in the request.
            400 => {
                if let Some((addr, snapshot)) =
                    clock_snapshot_output(req, steady_seconds(c), now_unix())
                {
                    c.process().memory.write(addr, &snapshot);
                }
                Reply::ok()
            }
            500 | 501 => Reply::ok().u64(0),
            _ => unimplemented(self.name, c, req),
        }
    }
}

struct SystemClock;

impl Service for SystemClock {
    fn name(&self) -> String {
        "time/ISystemClock".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(now_unix() as u64),
            1 | 3 => Reply::ok(),
            2 => Reply::ok().bytes(&system_context(c)),
            4 => Reply::ok().copy_handle(c.process().create_event(false)),
            _ => unimplemented("time/ISystemClock", c, req),
        }
    }
}

struct SteadyClock;

impl Service for SteadyClock {
    fn name(&self) -> String {
        "time/ISteadyClock".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().bytes(&time_point(c)),
            2 | 200 => Reply::ok().u64(0),
            100 => Reply::ok().u64(now_unix() as u64),
            _ => unimplemented("time/ISteadyClock", c, req),
        }
    }
}

pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d)
}

pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn calendar(posix: i64) -> Vec<u8> {
    let days = posix.div_euclid(86_400);
    let secs = posix.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let mut v = Vec::with_capacity(0x20);
    v.extend_from_slice(&(y as i16).to_le_bytes());
    v.extend_from_slice(&[
        m as u8,
        d as u8,
        (secs / 3600) as u8,
        (secs / 60 % 60) as u8,
        (secs % 60) as u8,
        0,
    ]);
    let weekday = (days + 4).rem_euclid(7) as u32;
    let yday = (days - days_from_civil(y, 1, 1)) as u32;
    v.extend_from_slice(&weekday.to_le_bytes());
    v.extend_from_slice(&yday.to_le_bytes());
    v.extend_from_slice(b"UTC\0\0\0\0\0");
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0i32.to_le_bytes());
    v
}

fn location_name() -> [u8; 0x24] {
    let mut n = [0u8; 0x24];
    n[..LOCATION.len()].copy_from_slice(LOCATION.as_bytes());
    n
}

struct TimeZone;

impl Service for TimeZone {
    fn name(&self) -> String {
        "time/ITimeZoneService".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        let p = c.process();
        match req.cmd {
            0 => Reply::ok().bytes(&location_name()),
            1 => Reply::ok(),
            2 => Reply::ok().u32(1),
            3 => {
                if let Some(b) = req.b.first() {
                    p.memory.write(b.addr, &location_name());
                }
                Reply::ok().u32(1)
            }
            4 => {
                if let Some(b) = req.b.first() {
                    p.memory.write(b.addr, &vec![0u8; b.size as usize]);
                }
                Reply::ok()
            }
            5 => Reply::ok().bytes(b"2024a\0\0\0\0\0\0\0\0\0\0\0"),
            100 | 101 => Reply::ok().bytes(&calendar(req.u64(0) as i64)),
            201 | 202 => {
                let d = &req.data;
                let year = i16::from_le_bytes([d[0], d[1]]) as i64;
                let (mon, day, h, mi, s) = (
                    d[2] as u32,
                    d[3] as u32,
                    d[4] as i64,
                    d[5] as i64,
                    d[6] as i64,
                );
                let posix =
                    days_from_civil(year, mon.max(1), day.max(1)) * 86_400 + h * 3600 + mi * 60 + s;
                if let Some(out) = req.c.first().or(req.b.first()) {
                    p.memory.write(out.addr, &posix.to_le_bytes());
                }
                Reply::ok().u32(1)
            }
            _ => unimplemented("time/ITimeZoneService", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::Buf;

    fn snapshot_request(addr: u64, size: u64, clock_type: u8) -> Request {
        Request {
            cmd: 400,
            data: vec![clock_type],
            c: vec![Buf { addr, size }],
            ..Request::default()
        }
    }

    #[test]
    fn clock_snapshot_matches_the_documented_layout() {
        let steady = 1_000_000i64;
        let unix = 1_792_329_600i64;
        let s = clock_snapshot(steady, unix, 2);
        assert_eq!(s.len(), CLOCK_SNAPSHOT_SIZE);

        for context in [0x00, 0x20] {
            assert_eq!(
                i64::from_le_bytes(s[context..context + 8].try_into().unwrap()),
                unix - steady
            );
            assert_eq!(
                i64::from_le_bytes(s[context + 8..context + 0x10].try_into().unwrap()),
                steady
            );
            assert_eq!(&s[context + 0x10..context + 0x20], &CLOCK_SOURCE);
        }

        assert_eq!(
            i64::from_le_bytes(s[0x40..0x48].try_into().unwrap()),
            unix,
            "user time"
        );
        assert_eq!(
            i64::from_le_bytes(s[0x48..0x50].try_into().unwrap()),
            unix,
            "network time"
        );

        let cal = calendar(unix);
        assert_eq!(&s[0x50..0x58], &cal[..8], "user CalendarTime");
        assert_eq!(&s[0x58..0x60], &cal[..8], "network CalendarTime");
        assert_eq!(&s[0x60..0x78], &cal[8..0x20], "user CalendarAdditionalInfo");
        assert_eq!(
            &s[0x78..0x90],
            &cal[8..0x20],
            "network CalendarAdditionalInfo"
        );

        assert_eq!(
            i64::from_le_bytes(s[0x90..0x98].try_into().unwrap()),
            steady
        );
        assert_eq!(&s[0x98..0xa8], &CLOCK_SOURCE);

        assert_eq!(&s[0xa8..0xac], b"UTC\0");
        assert_eq!(s[0xcc], 0, "automatic correction is reported as disabled");
        assert_eq!(s[0xcd], 2, "the requested TimeType is stored");
        assert_eq!(&s[0xce..0xd0], &[0, 0]);
    }

    #[test]
    fn clock_snapshot_uses_the_receive_pointer_buffer() {
        let req = snapshot_request(0x2000, CLOCK_SNAPSHOT_SIZE as u64, 1);
        let (addr, snapshot) = clock_snapshot_output(&req, 10, 100).expect("a full buffer");
        assert_eq!(addr, 0x2000);
        assert_eq!(snapshot[0xcd], 1);
        assert_eq!(
            i64::from_le_bytes(snapshot[0x40..0x48].try_into().unwrap()),
            100
        );

        // A pointer descriptor of the other form is accepted too.
        let pointer = Request {
            cmd: 400,
            x: vec![Buf {
                addr: 0x2800,
                size: CLOCK_SNAPSHOT_SIZE as u64,
            }],
            ..Request::default()
        };
        assert_eq!(
            clock_snapshot_output(&pointer, 10, 100).map(|(a, _)| a),
            Some(0x2800)
        );

        // A type-0x6 map descriptor is accepted as well.
        let mapped = Request {
            cmd: 400,
            b: vec![Buf {
                addr: 0x3000,
                size: 0x200,
            }],
            ..Request::default()
        };
        assert_eq!(
            clock_snapshot_output(&mapped, 10, 100).map(|(a, _)| a),
            Some(0x3000)
        );

        // No buffer, or one too short for the 0xD0-byte struct, is left untouched.
        let bare = Request {
            cmd: 400,
            ..Request::default()
        };
        assert!(clock_snapshot_output(&bare, 10, 100).is_none());
        let short = snapshot_request(0x2000, CLOCK_SNAPSHOT_SIZE as u64 - 1, 0);
        assert!(clock_snapshot_output(&short, 10, 100).is_none());

        // No TimeType byte reads as UserSystemClock (0).
        let untyped = Request {
            cmd: 400,
            x: vec![Buf {
                addr: 0x2000,
                size: CLOCK_SNAPSHOT_SIZE as u64,
            }],
            ..Request::default()
        };
        assert_eq!(
            clock_snapshot_output(&untyped, 10, 100)
                .map(|(_, s)| s[0xcd])
                .unwrap(),
            0
        );
    }

    #[test]
    fn time_u_400_goes_through_the_per_process_path() {
        let time = Time { name: "time:u" };
        // GetClockSnapshot needs the process clocks, so it is not in the
        // stateless table; unknown commands stay unimplemented.
        for cmd in [400, 401, 402] {
            assert!(
                time.stateless_command(&Request {
                    cmd,
                    ..Request::default()
                })
                .is_none()
            );
        }
    }

    #[test]
    fn calendar_conversion() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(days_from_civil(2026, 10, 18), civil_days(2026, 10, 18));
        let c = calendar(1_792_329_600);
        assert_eq!(i16::from_le_bytes([c[0], c[1]]), 2026);
        assert_eq!((c[2], c[3]), (10, 18));
        assert_eq!(u32::from_le_bytes(c[8..12].try_into().unwrap()), 0);
    }

    fn civil_days(y: i64, m: u32, d: u32) -> i64 {
        (0..).find(|&n| civil_from_days(n) == (y, m, d)).unwrap()
    }

    #[test]
    fn time_u_200_reports_network_accuracy_as_one_false_byte() {
        let time = Time { name: "time:u" };
        let r = time
            .stateless_command(&Request {
                cmd: 200,
                ..Request::default()
            })
            .expect("IsStandardNetworkSystemClockAccuracySufficient");
        assert_eq!(r.result, 0);
        assert_eq!(r.data, vec![0]);
        assert!(r.objects.is_empty() && r.copy_handles.is_empty() && r.move_handles.is_empty());

        assert!(
            time.stateless_command(&Request {
                cmd: 201,
                ..Request::default()
            })
            .is_none()
        );
    }
}
