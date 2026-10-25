use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cpu::Cpu;
use crate::kernel::SharedMemory;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const SHARED_SIZE: u64 = 0x1000;
const CLOCK_SOURCE: [u8; 16] = *b"kirikae-steady\0\0";
const LOCATION: &str = "UTC";

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
