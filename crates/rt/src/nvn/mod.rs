pub mod encode;
pub mod resources;
pub mod shadow;
pub mod state;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::hle::{call_guest, guest_str};
use crate::process::Process;

#[derive(Default)]
struct Trace {
    functions: Mutex<Vec<(String, Arc<AtomicU64>)>>,
    handed: Mutex<HashMap<String, u64>>,
    real: Mutex<HashMap<String, u64>>,
    shadow: bool,
}

static TRACE: OnceLock<Arc<Trace>> = OnceLock::new();

pub fn install(p: &'static Process) {
    let shadow = match crate::env_var!("KIRIKAE_NVN").and_then(|v| v.to_str()) {
        Some("trace") => false,
        Some("shadow") => true,
        Some("encode") => {
            encode::ON.store(true, Ordering::Relaxed);
            true
        }
        _ => return,
    };
    let Some(boot) = p.symbol("nvnBootstrapLoader") else {
        p.warn_once("nvn: the game exports no nvnBootstrapLoader".into());
        return;
    };
    let trace = TRACE.get_or_init(|| Arc::new(Trace { shadow, ..Default::default() })).clone();
    shadow::ON.store(shadow, Ordering::Relaxed);
    p.hle.replace(
        boot,
        Arc::new(move |c| {
            let name = guest_str(c, c.x[0]);
            call_guest(c, boot, true);
            c.x[0] = wrap(p, &trace, &name, c.x[0]);
            c.pc = c.x[30];
        }),
    );
}

fn wrap(p: &'static Process, trace: &Arc<Trace>, name: &str, real: u64) -> u64 {
    if real == 0 {
        return 0;
    }
    if let Some(&at) = trace.handed.lock().unwrap().get(name) {
        return at;
    }
    trace.real.lock().unwrap().insert(name.to_string(), real);
    let at = if name == "nvnDeviceGetProcAddress" {
        let trace = trace.clone();
        p.hle.add(
            name,
            Arc::new(move |c| {
                let wanted = guest_str(c, c.x[1]);
                call_guest(c, real, false);
                c.x[0] = wrap(p, &trace, &wanted, c.x[0]);
                c.pc = c.x[30];
            }),
        )
    } else {
        let calls = Arc::new(AtomicU64::new(0));
        trace.functions.lock().unwrap().push((name.to_string(), calls.clone()));
        match shadow::hook(name).filter(|_| trace.shadow) {
            Some(hook) => {
                let trace = trace.clone();
                p.hle.add(
                    name,
                    Arc::new(move |c| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        shadow::run(hook, c, real, &|n| trace.real.lock().unwrap().get(n).copied());
                    }),
                )
            }
            None if trace.shadow && shadow::recorded_elsewhere(name) => {
                let name: &'static str = Box::leak(name.to_string().into_boxed_str());
                p.hle.add(
                    name,
                    Arc::new(move |c| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        shadow::unsupported(name, c.x[0]);
                        c.pc = real;
                    }),
                )
            }
            None => p.hle.add(
                name,
                Arc::new(move |c| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    c.pc = real;
                }),
            ),
        }
    };
    trace.handed.lock().unwrap().insert(name.to_string(), at);
    at
}

pub fn report(frames: u64) {
    let Some(trace) = TRACE.get() else { return };
    shadow::report();
    encode::report();
    static LAST: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let functions = trace.functions.lock().unwrap();
    let mut last = LAST.lock().unwrap();
    last.resize(functions.len(), 0);
    let mut rows: Vec<(u64, &str)> = functions
        .iter()
        .zip(last.iter_mut())
        .map(|((name, calls), prev)| {
            let now = calls.load(Ordering::Relaxed);
            (now - std::mem::replace(prev, now), name.as_str())
        })
        .collect();
    let total: u64 = rows.iter().map(|r| r.0).sum();
    let used = functions.iter().filter(|f| f.1.load(Ordering::Relaxed) > 0).count();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let per = |n: u64| n as f64 / frames.max(1) as f64;
    let top: Vec<String> = rows.iter().take(25).filter(|r| r.0 > 0).map(|(n, name)| format!("{} {:.0}", name.trim_start_matches("nvn"), per(*n))).collect();
    eprintln!("kirikae:         nvn: {:.0} calls per frame, {used} of {} functions used: {}", per(total), functions.len(), top.join(", "));
    if let Ok(exe) = std::env::current_exe() {
        let mut all: Vec<(u64, &str)> = functions.iter().map(|(name, calls)| (calls.load(Ordering::Relaxed), name.as_str())).collect();
        all.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
        let text: String = all.iter().map(|(n, name)| format!("{n:>12} {name}\n")).collect();
        let _ = std::fs::write(exe.with_extension("nvn.txt"), text);
    }
}
