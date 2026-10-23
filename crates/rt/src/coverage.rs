use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ON: AtomicBool = AtomicBool::new(false);
static SEEN: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

pub fn parse(text: &str) -> Vec<(u64, u64)> {
    text.lines()
        .filter_map(|l| {
            let l = l.split('#').next()?.trim();
            let mut f = l.split_whitespace();
            let addr = u64::from_str_radix(f.next()?.trim_start_matches("0x"), 16).ok()?;
            let count = f.next().and_then(|c| c.parse().ok()).unwrap_or(1);
            Some((addr, count))
        })
        .collect()
}

fn path() -> Option<PathBuf> {
    match std::env::var_os("KIRIKAE_PROFILE") {
        Some(v) if v == "0" => None,
        Some(v) => Some(PathBuf::from(v)),
        None => Some(std::env::current_exe().ok()?.with_extension("profile")),
    }
}

pub fn start() {
    let Some(path) = path() else { return };
    *SEEN.lock().unwrap() = Some(HashMap::new());
    ON.store(true, Ordering::Relaxed);
    let _ = std::thread::Builder::new().name("kirikae-profile".into()).spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let fresh: Vec<(u64, u64)> = match &mut *SEEN.lock().unwrap() {
                Some(m) => m.drain().collect(),
                None => return,
            };
            if fresh.is_empty() {
                continue;
            }
            let mut all: HashMap<u64, u64> = std::fs::read_to_string(&path).map(|t| parse(&t).into_iter().collect()).unwrap_or_default();
            for (a, n) in fresh {
                *all.entry(a).or_default() += n;
            }
            let mut rows: Vec<_> = all.into_iter().collect();
            rows.sort_unstable();
            let mut text = String::from("# kirikae: addresses the interpreter ran, for `kirikae build` (address, times entered)\n");
            for (a, n) in rows {
                text.push_str(&format!("{a:x} {n}\n"));
            }
            let tmp = path.with_extension("profile.tmp");
            if std::fs::File::create(&tmp).and_then(|mut f| f.write_all(text.as_bytes())).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    });
}

#[inline]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

pub fn record(addr: u64) {
    if let Some(m) = &mut *SEEN.lock().unwrap() {
        *m.entry(addr).or_default() += 1;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_what_it_writes() {
        let text = "# kirikae: comment\n83375c4 681856\n0x8339718 3 # note\n\nnot hex\n80356d0\n";
        assert_eq!(super::parse(text), vec![(0x83375c4, 681856), (0x8339718, 3), (0x80356d0, 1)]);
    }
}
