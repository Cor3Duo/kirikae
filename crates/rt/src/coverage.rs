use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const FLUSH_EVERY: Duration = Duration::from_secs(10);
const HEADER: &str =
    "# kirikae: addresses the interpreter ran, for `kirikae build` (address, times entered)\n";

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<Profile>> = Mutex::new(None);
static WRITER: Mutex<Option<Writer>> = Mutex::new(None);

struct Writer {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: std::thread::JoinHandle<()>,
}

struct Profile {
    path: PathBuf,
    seen: HashMap<u64, u64>,
}

impl Profile {
    fn new(path: PathBuf) -> Profile {
        Profile {
            path,
            seen: HashMap::new(),
        }
    }

    fn record(&mut self, addr: u64) {
        *self.seen.entry(addr).or_default() += 1;
    }

    /// Merges what was recorded into the profile, rewriting it in one step. Only forgets the
    /// recorded addresses once the file is on disk, so a failed write is retried by the next flush.
    fn flush(&mut self) -> std::io::Result<()> {
        if self.seen.is_empty() {
            return Ok(());
        }
        let mut all: HashMap<u64, u64> = std::fs::read_to_string(&self.path)
            .map(|t| parse(&t).into_iter().collect())
            .unwrap_or_default();
        for (&addr, &n) in &self.seen {
            *all.entry(addr).or_default() += n;
        }
        let mut rows: Vec<_> = all.into_iter().collect();
        rows.sort_unstable();
        let mut text = String::from(HEADER);
        for (a, n) in rows {
            text.push_str(&format!("{a:x} {n}\n"));
        }
        let tmp = self.path.with_extension("profile.tmp");
        std::fs::write(&tmp, text.as_bytes())?;
        std::fs::rename(&tmp, &self.path)?;
        self.seen.clear();
        Ok(())
    }
}

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
    if ON.load(Ordering::SeqCst) {
        return;
    }
    *STATE.lock().unwrap() = Some(Profile::new(path));
    ON.store(true, Ordering::SeqCst);
    let stop = Arc::new((Mutex::new(false), Condvar::new()));
    let flag = stop.clone();
    let spawned = std::thread::Builder::new()
        .name("kirikae-profile".into())
        .spawn(move || {
            let (lock, wake) = &*flag;
            loop {
                let stopped = lock.lock().unwrap();
                let (stopped, _) = wake
                    .wait_timeout_while(stopped, FLUSH_EVERY, |stop| !*stop)
                    .unwrap();
                if *stopped {
                    return;
                }
                drop(stopped);
                flush_pending();
            }
        });
    if let Ok(thread) = spawned {
        *WRITER.lock().unwrap() = Some(Writer { stop, thread });
    }
}

/// Stops the writer thread and writes every address recorded so far. Idempotent: the second call
/// does nothing, so calling it from more than one exit path cannot duplicate entries.
pub fn finish() {
    if !ON.swap(false, Ordering::SeqCst) {
        return;
    }
    if let Some(writer) = WRITER.lock().unwrap().take() {
        let (lock, wake) = &*writer.stop;
        *lock.lock().unwrap() = true;
        wake.notify_all();
        let _ = writer.thread.join();
    }
    flush_pending();
    *STATE.lock().unwrap() = None;
}

fn flush_pending() {
    let mut state = STATE.lock().unwrap();
    let Some(profile) = state.as_mut() else {
        return;
    };
    let path = profile.path.clone();
    let written = profile.flush();
    drop(state);
    if let Err(e) = written {
        eprintln!("kirikae: cannot write {}: {e}", path.display());
    }
}

#[inline]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

pub fn record(addr: u64) {
    if let Some(profile) = &mut *STATE.lock().unwrap() {
        profile.record(addr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kirikae-coverage-test-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn read(path: &std::path::Path) -> Vec<(u64, u64)> {
        parse(&std::fs::read_to_string(path).unwrap())
    }

    #[test]
    fn parses_what_it_writes() {
        let text = "# kirikae: comment\n83375c4 681856\n0x8339718 3 # note\n\nnot hex\n80356d0\n";
        assert_eq!(
            super::parse(text),
            vec![(0x83375c4, 681856), (0x8339718, 3), (0x80356d0, 1)]
        );
    }

    #[test]
    fn flush_keeps_previous_runs_and_is_idempotent() {
        let dir = temp("flush");
        let path = dir.join("game.profile");
        std::fs::write(&path, "# previous run\n100 2\n200 1\n").unwrap();

        let mut profile = Profile::new(path.clone());
        profile.record(0x100);
        profile.record(0x300);
        profile.record(0x300);
        profile.flush().unwrap();
        assert_eq!(read(&path), vec![(0x100, 3), (0x200, 1), (0x300, 2)]);
        assert!(!path.with_extension("profile.tmp").exists());

        profile.flush().unwrap();
        profile.flush().unwrap();
        assert_eq!(read(&path), vec![(0x100, 3), (0x200, 1), (0x300, 2)]);

        profile.record(0x200);
        profile.flush().unwrap();
        assert_eq!(read(&path), vec![(0x100, 3), (0x200, 2), (0x300, 2)]);
    }

    #[test]
    fn failed_write_keeps_the_addresses() {
        let dir = temp("failed");
        let path = dir.join("missing").join("game.profile");
        let mut profile = Profile::new(path);
        profile.record(0x100);
        assert!(profile.flush().is_err());
        assert_eq!(profile.seen, HashMap::from([(0x100, 1)]));
    }
}
