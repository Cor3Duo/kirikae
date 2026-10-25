use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::cpu::Cpu;

const FORMAT: u64 = 1;
const DEFAULT_EVENT_LIMIT: u32 = 200;
const SHOT_WIDTH: u32 = 480;
const FLUSH_EVERY: Duration = Duration::from_secs(10);
const ARG_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Section {
    Services,
    Svc,
    Ioctl,
    Shaders,
}

impl Section {
    const ALL: [Section; 4] = [
        Section::Services,
        Section::Svc,
        Section::Ioctl,
        Section::Shaders,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Section::Services => "services",
            Section::Svc => "svc",
            Section::Ioctl => "ioctl",
            Section::Shaders => "shaders",
        }
    }

    fn parse(s: &str) -> Option<Section> {
        Section::ALL.into_iter().find(|x| x.as_str() == s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Implemented,
    Stub,
    Missing,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Implemented => "implemented",
            Status::Stub => "stub",
            Status::Missing => "missing",
        }
    }

    fn parse(s: &str) -> Status {
        match s {
            "stub" => Status::Stub,
            "missing" => Status::Missing,
            _ => Status::Implemented,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Id {
    Num(u64),
    Name(String),
}

impl Id {
    fn to_json(&self) -> Value {
        match self {
            Id::Num(n) => json!(n),
            Id::Name(s) => json!(s),
        }
    }

    fn from_json(v: &Value) -> Option<Id> {
        v.as_u64()
            .map(Id::Num)
            .or_else(|| v.as_str().map(|s| Id::Name(s.to_string())))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct When {
    pub run: u32,
    pub seconds: f64,
    pub frame: u64,
    pub unix: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub count: u64,
    pub status: Status,
    pub note: String,
    pub first: When,
    pub event: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Context {
    pub backtrace: Vec<String>,
    pub args: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub n: u32,
    pub section: Section,
    pub interface: String,
    pub id: Id,
    pub status: Status,
    pub note: String,
    pub when: When,
    pub backtrace: Vec<String>,
    pub args: String,
    pub screenshot: Option<String>,
    pub saves: Option<String>,
}

type Items = BTreeMap<(Section, String, Id), Item>;

pub struct Recorder {
    path: PathBuf,
    events_dir: PathBuf,
    game: String,
    title_id: u64,
    run: u32,
    started: Instant,
    started_unix: u64,
    event_limit: u32,
    base_items: Items,
    base_runs: Vec<Value>,
    base_events: Vec<Value>,
    items: [HashMap<String, HashMap<Id, Item>>; 4],
    events: Vec<Event>,
    next_event: u32,
    want_shot: Vec<u32>,
    dirty: bool,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Recorder {
    pub fn open(path: PathBuf, title_id: u64) -> Recorder {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("game")
            .to_string();
        let stem = name
            .strip_suffix(".coverage.json")
            .or_else(|| name.rsplit_once('.').map(|(s, _)| s))
            .unwrap_or(&name)
            .to_string();
        let events_dir = path.with_file_name(format!("{stem}-coverage"));
        let mut r = Recorder {
            events_dir,
            game: stem,
            title_id,
            run: 0,
            started: Instant::now(),
            started_unix: unix_now(),
            event_limit: DEFAULT_EVENT_LIMIT,
            base_items: Items::new(),
            base_runs: Vec::new(),
            base_events: Vec::new(),
            items: Default::default(),
            events: Vec::new(),
            next_event: 1,
            want_shot: Vec::new(),
            dirty: true,
            path,
        };
        if let Ok(text) = std::fs::read_to_string(&r.path) {
            match serde_json::from_str::<Value>(&text) {
                Ok(v) if v["format"].as_u64() == Some(FORMAT) => r.load(&v),
                _ => {
                    let bad = r.path.with_extension("json.bad");
                    eprintln!(
                        "kirikae: {} is not a coverage file this kirikae reads; moved to {}",
                        r.path.display(),
                        bad.display()
                    );
                    let _ = std::fs::rename(&r.path, bad);
                }
            }
        }
        r
    }

    fn load(&mut self, v: &Value) {
        for it in v["items"].as_array().into_iter().flatten() {
            let (Some(section), Some(iface), Some(id)) = (
                it["section"].as_str().and_then(Section::parse),
                it["interface"].as_str(),
                Id::from_json(&it["id"]),
            ) else {
                continue;
            };
            let f = &it["first"];
            let item = Item {
                count: it["count"].as_u64().unwrap_or(0),
                status: Status::parse(it["status"].as_str().unwrap_or("")),
                note: it["note"].as_str().unwrap_or("").to_string(),
                first: When {
                    run: f["run"].as_u64().unwrap_or(0) as u32,
                    seconds: f["seconds"].as_f64().unwrap_or(0.0),
                    frame: f["frame"].as_u64().unwrap_or(0),
                    unix: f["unix"].as_u64().unwrap_or(0),
                },
                event: it["event"].as_u64().map(|n| n as u32),
            };
            self.base_items
                .insert((section, iface.to_string(), id), item);
        }
        self.base_runs = v["runs"].as_array().cloned().unwrap_or_default();
        self.base_events = v["events"].as_array().cloned().unwrap_or_default();
        self.run = self.base_runs.len() as u32;
        self.next_event = self
            .base_events
            .iter()
            .filter_map(|e| e["n"].as_u64())
            .max()
            .map_or(1, |n| n as u32 + 1);
    }

    pub fn set_event_limit(&mut self, limit: u32) {
        self.event_limit = limit;
    }

    pub fn events_dir(&self) -> &Path {
        &self.events_dir
    }

    pub fn new_events(&self) -> u32 {
        self.events.len() as u32
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        section: Section,
        iface: &str,
        id: Id,
        status: Status,
        note: &str,
        uses: u64,
        when: impl FnOnce() -> When,
        context: impl FnOnce() -> Context,
    ) -> Option<Event> {
        let map = &mut self.items[section as usize];
        if !map.contains_key(iface) {
            map.insert(iface.to_string(), HashMap::new());
        }
        let per = map.get_mut(iface).unwrap();
        self.dirty = true;
        let item = match per.get_mut(&id) {
            Some(item) => {
                item.count += uses;
                if status <= item.status {
                    return None;
                }
                item.status = status;
                item.note = note.to_string();
                item
            }
            None => per.entry(id.clone()).or_insert(Item {
                count: uses,
                status,
                note: note.to_string(),
                first: when(),
                event: None,
            }),
        };
        if status == Status::Implemented || item.event.is_some() {
            return None;
        }
        let key = (section, iface.to_string(), id);
        if self.base_items.get(&key).is_some_and(|b| b.event.is_some()) {
            return None;
        }
        if self.events.len() as u32 >= self.event_limit {
            return None;
        }
        let n = self.next_event;
        self.next_event += 1;
        item.event = Some(n);
        let first = item.first;
        let ctx = context();
        let e = Event {
            n,
            section,
            interface: key.1,
            id: key.2,
            status,
            note: note.to_string(),
            when: first,
            backtrace: ctx.backtrace,
            args: ctx.args,
            screenshot: None,
            saves: None,
        };
        self.events.push(e.clone());
        self.want_shot.push(n);
        self.dirty = true;
        Some(e)
    }

    pub fn event_dir(&self, n: u32) -> (String, PathBuf) {
        let name = self
            .events_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("coverage");
        (format!("{name}/{n}"), self.events_dir.join(n.to_string()))
    }

    fn event_mut(&mut self, n: u32) -> Option<&mut Event> {
        self.events.iter_mut().find(|e| e.n == n)
    }

    pub fn set_screenshot(&mut self, n: u32, rel: String) {
        if let Some(e) = self.event_mut(n) {
            e.screenshot = Some(rel);
            self.dirty = true;
        }
    }

    pub fn set_saves(&mut self, n: u32, rel: String) {
        if let Some(e) = self.event_mut(n) {
            e.saves = Some(rel);
            self.dirty = true;
        }
    }

    /// Records a draw whose shaders still use instructions with no
    /// translation. Counts one hit per draw so the cost of the gap is visible.
    fn draw_untranslated(
        &mut self,
        va: u64,
        unsupported: &[&'static str],
        partial: bool,
        when: When,
    ) {
        let note = if partial {
            "drawn without it (KIRIKAE_DRAW_PARTIAL)"
        } else {
            "draw skipped"
        };
        for &u in unsupported {
            self.record(
                Section::Shaders,
                SHADER_DRAWS,
                Id::Name(u.to_string()),
                Status::Missing,
                note,
                1,
                || when,
                || Context {
                    backtrace: Vec::new(),
                    args: format!("shader at gpu va {va:#x}"),
                },
            );
        }
    }

    pub fn merged(&self) -> Items {
        let mut all = self.base_items.clone();
        for (i, map) in self.items.iter().enumerate() {
            for (iface, per) in map {
                for (id, item) in per {
                    let key = (Section::ALL[i], iface.clone(), id.clone());
                    match all.get_mut(&key) {
                        Some(old) => {
                            old.count += item.count;
                            old.status = item.status;
                            old.note = item.note.clone();
                            old.event = old.event.or(item.event);
                        }
                        None => {
                            all.insert(key, item.clone());
                        }
                    }
                }
            }
        }
        all
    }

    pub fn to_json(&self, frames: u64) -> Value {
        let items: Vec<Value> = self
            .merged()
            .into_iter()
            .map(|((section, iface, id), it)| {
                let mut v = json!({
                    "section": section.as_str(),
                    "interface": iface,
                    "id": id.to_json(),
                    "count": it.count,
                    "status": it.status.as_str(),
                    "first": {"run": it.first.run, "seconds": round(it.first.seconds), "frame": it.first.frame, "unix": it.first.unix},
                });
                if !it.note.is_empty() {
                    v["note"] = json!(it.note);
                }
                if let Some(n) = it.event {
                    v["event"] = json!(n);
                }
                v
            })
            .collect();
        let mut runs = self.base_runs.clone();
        runs.push(json!({
            "started": self.started_unix,
            "seconds": round(self.started.elapsed().as_secs_f64()),
            "frames": frames,
            "events": self.events.len(),
        }));
        let mut events = self.base_events.clone();
        events.extend(self.events.iter().map(|e| {
            json!({
                "n": e.n,
                "section": e.section.as_str(),
                "interface": e.interface,
                "id": e.id.to_json(),
                "status": e.status.as_str(),
                "note": e.note,
                "run": e.when.run,
                "seconds": round(e.when.seconds),
                "frame": e.when.frame,
                "unix": e.when.unix,
                "backtrace": e.backtrace,
                "args": e.args,
                "screenshot": e.screenshot,
                "saves": e.saves,
            })
        }));
        json!({
            "format": FORMAT,
            "game": self.game,
            "title_id": format!("{:016x}", self.title_id),
            "updated": unix_now(),
            "runs": runs,
            "items": items,
            "events": events,
        })
    }

    pub fn write(&mut self, frames: u64) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(&self.to_json(frames))?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }
}

fn round(s: f64) -> f64 {
    (s * 1000.0).round() / 1000.0
}

pub fn hex(b: &[u8]) -> String {
    let mut s: String = b
        .iter()
        .take(ARG_BYTES)
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    if b.len() > ARG_BYTES {
        s += &format!(" … ({} bytes)", b.len());
    }
    s
}

pub fn backtrace(c: &Cpu) -> Vec<String> {
    let p = c.process();
    let mut out = vec![format!("lr {}", p.describe(c.x[30]))];
    let mut fp = c.x[29];
    for _ in 0..16 {
        let Some(rec) = p.memory.read(fp, 16) else {
            break;
        };
        let next = u64::from_le_bytes(rec[..8].try_into().unwrap());
        let lr = u64::from_le_bytes(rec[8..].try_into().unwrap());
        if lr == 0 {
            break;
        }
        out.push(p.describe(lr));
        if next <= fp {
            break;
        }
        fp = next;
    }
    out
}

pub fn scale(width: u32, height: u32, pixels: &[u32]) -> (u32, u32, Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    let dw = width.clamp(1, SHOT_WIDTH) as usize;
    let dh = ((h * dw).div_ceil(w.max(1))).max(1);
    let mut out = Vec::with_capacity(dw * dh * 3);
    for dy in 0..dh {
        let (y0, y1) = (dy * h / dh, ((dy + 1) * h / dh).max(dy * h / dh + 1).min(h));
        for dx in 0..dw {
            let (x0, x1) = (dx * w / dw, ((dx + 1) * w / dw).max(dx * w / dw + 1).min(w));
            let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
            for y in y0..y1 {
                for &px in &pixels[y * w + x0..y * w + x1] {
                    r += (px >> 16) & 0xff;
                    g += (px >> 8) & 0xff;
                    b += px & 0xff;
                    n += 1;
                }
            }
            let n = n.max(1);
            out.extend_from_slice(&[(r / n) as u8, (g / n) as u8, (b / n) as u8]);
        }
    }
    (dw as u32, dh as u32, out)
}

fn crc32(chunks: &[&[u8]]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *e = c;
        }
        t
    });
    let mut c = !0u32;
    for chunk in chunks {
        for &b in *chunk {
            c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
        }
    }
    !c
}

pub fn png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    png_level(width, height, rgb, 6)
}

pub fn png_level(width: u32, height: u32, rgb: &[u8], level: u8) -> Vec<u8> {
    let mut raw = Vec::with_capacity(rgb.len() + height as usize);
    for row in rgb.chunks(width as usize * 3) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        out.extend_from_slice(&crc32(&[kind, data]).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, level),
    );
    chunk(b"IEND", &[]);
    out
}

pub fn copy_tree(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(to)?;
    let mut bytes = 0;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            bytes += copy_tree(&entry.path(), &target)?;
        } else {
            bytes += std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(bytes)
}

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<Recorder>> = Mutex::new(None);
static WAKE: Condvar = Condvar::new();
static WANT_SHOT: AtomicBool = AtomicBool::new(false);
static NEW_EVENTS: AtomicU32 = AtomicU32::new(0);
static BUSY: AtomicU32 = AtomicU32::new(0);
static SVC_COUNT: [AtomicU64; 256] = [const { AtomicU64::new(0) }; 256];
static TITLE: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static CALL: Cell<(Status, &'static str)> = const { Cell::new((Status::Implemented, "")) };
}

fn path() -> Option<PathBuf> {
    match std::env::var_os("KIRIKAE_COVERAGE") {
        Some(v) if v == "0" => None,
        Some(v) => Some(PathBuf::from(v)),
        None => {
            let exe = std::env::current_exe().ok()?;
            let stem = exe.file_stem()?.to_str()?.to_string();
            Some(exe.with_file_name(format!("{stem}.coverage.json")))
        }
    }
}

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

fn frames() -> u64 {
    crate::display::FRAMES.load(Ordering::Relaxed)
}

pub fn start(title_id: u64) {
    let Some(path) = path() else { return };
    let mut r = Recorder::open(path, title_id);
    if let Some(n) = std::env::var("KIRIKAE_COVERAGE_EVENTS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        r.set_event_limit(n);
    }
    TITLE.store(title_id, Ordering::Relaxed);
    init_clock(&r);
    *STATE.lock().unwrap() = Some(r);
    ON.store(true, Ordering::Relaxed);
    let _ = std::thread::Builder::new()
        .name("kirikae-coverage".into())
        .spawn(|| {
            loop {
                let mut st = STATE.lock().unwrap();
                st = WAKE.wait_timeout(st, FLUSH_EVERY).unwrap().0;
                let Some(r) = st.as_mut() else { return };
                if r.dirty {
                    flush(r);
                }
            }
        });
}

fn flush(r: &mut Recorder) {
    for (imm, n) in SVC_COUNT.iter().enumerate() {
        let n = n.swap(0, Ordering::Relaxed);
        if n > 0
            && let Some(item) = r.items[Section::Svc as usize]
                .get_mut("SVCs")
                .and_then(|m| m.get_mut(&Id::Num(imm as u64)))
        {
            item.count += n;
        }
    }
    if let Err(e) = r.write(frames()) {
        eprintln!("kirikae: cannot write {}: {e}", r.path.display());
    }
}

pub fn finish() {
    if !on() {
        return;
    }
    let until = Instant::now() + Duration::from_secs(2);
    while BUSY.load(Ordering::Relaxed) > 0 && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(20));
    }
    if let Some(r) = STATE.lock().unwrap().as_mut() {
        flush(r);
        if r.new_events() > 0 {
            eprintln!(
                "kirikae: coverage: {} new item(s) not implemented in this run; see {} (python tools/coverage_report.py {})",
                r.new_events(),
                r.events_dir.display(),
                r.path.display()
            );
        }
    }
}

pub fn new_events() -> u32 {
    NEW_EVENTS.load(Ordering::Relaxed)
}

fn record(
    c: Option<&Cpu>,
    section: Section,
    iface: &str,
    id: Id,
    status: Status,
    note: &str,
    args: impl FnOnce() -> String,
) {
    let mut st = STATE.lock().unwrap();
    let Some(r) = st.as_mut() else { return };
    let frame = frames();
    let Some(e) = r.record(
        section,
        iface,
        id,
        status,
        note,
        1,
        || r_now(frame),
        || Context {
            backtrace: c.map(backtrace).unwrap_or_default(),
            args: args(),
        },
    ) else {
        return;
    };
    NEW_EVENTS.fetch_add(1, Ordering::Relaxed);
    WANT_SHOT.store(true, Ordering::Relaxed);
    let (rel, dir) = r.event_dir(e.n);
    let id = match &e.id {
        Id::Num(n) if section == Section::Services => n.to_string(),
        Id::Num(n) => format!("{n:#x}"),
        Id::Name(s) => s.clone(),
    };
    eprintln!(
        "kirikae: coverage: new {} item #{}: {} {id} ({}{}) at {:.1}s, frame {}",
        section.as_str(),
        e.n,
        e.interface,
        e.status.as_str(),
        if e.note.is_empty() {
            String::new()
        } else {
            format!(", {}", e.note)
        },
        e.when.seconds,
        e.when.frame
    );
    let saves = crate::services::fs::saves_root(TITLE.load(Ordering::Relaxed));
    let n = e.n;
    BUSY.fetch_add(1, Ordering::Relaxed);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("event.txt"), event_text(&e, &saves));
    let spawned = std::thread::Builder::new()
        .name("kirikae-coverage-saves".into())
        .spawn(move || {
            let copied = saves.is_dir() && copy_tree(&saves, &dir.join("saves")).is_ok();
            if copied && let Some(r) = STATE.lock().unwrap().as_mut() {
                r.set_saves(n, format!("{rel}/saves"));
            }
            BUSY.fetch_sub(1, Ordering::Relaxed);
            WAKE.notify_one();
        });
    if spawned.is_err() {
        BUSY.fetch_sub(1, Ordering::Relaxed);
    }
}

fn r_now(frame: u64) -> When {
    let started = START.get().copied().unwrap_or_else(Instant::now);
    When {
        run: RUN.load(Ordering::Relaxed),
        seconds: started.elapsed().as_secs_f64(),
        frame,
        unix: unix_now(),
    }
}

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
static RUN: AtomicU32 = AtomicU32::new(0);

fn event_text(e: &Event, saves: &Path) -> String {
    let mut s = format!(
        "{} {} {:?}: {}{}\nrun {}, {:.1} s after start, frame {}\nsaves copied from {}\nargs: {}\nbacktrace:\n",
        e.section.as_str(),
        e.interface,
        e.id,
        e.status.as_str(),
        if e.note.is_empty() {
            String::new()
        } else {
            format!(" ({})", e.note)
        },
        e.when.run,
        e.when.seconds,
        e.when.frame,
        saves.display(),
        e.args
    );
    for f in &e.backtrace {
        s += &format!("    {f}\n");
    }
    s
}

pub fn frame_wanted() -> bool {
    WANT_SHOT.load(Ordering::Relaxed)
}

pub fn frame(width: u32, height: u32, pixels: Option<&[u32]>) {
    let Some(pixels) = pixels.filter(|_| frame_wanted()) else {
        return;
    };
    if width == 0 || height == 0 || pixels.len() < (width * height) as usize {
        return;
    }
    let targets: Vec<(u32, String, PathBuf)> = {
        let mut st = STATE.lock().unwrap();
        let Some(r) = st.as_mut() else { return };
        WANT_SHOT.store(false, Ordering::Relaxed);
        std::mem::take(&mut r.want_shot)
            .into_iter()
            .map(|n| {
                let (rel, dir) = r.event_dir(n);
                (n, rel, dir)
            })
            .collect()
    };
    if targets.is_empty() {
        return;
    }
    let (w, h, rgb) = scale(width, height, pixels);
    BUSY.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new()
        .name("kirikae-coverage-shot".into())
        .spawn(move || {
            let data = png(w, h, &rgb);
            for (n, rel, dir) in targets {
                let _ = std::fs::create_dir_all(&dir);
                if std::fs::write(dir.join("frame.png"), &data).is_ok()
                    && let Some(r) = STATE.lock().unwrap().as_mut()
                {
                    r.set_screenshot(n, format!("{rel}/frame.png"));
                }
            }
            BUSY.fetch_sub(1, Ordering::Relaxed);
        });
    if spawned.is_err() {
        BUSY.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn begin_ipc() {
    if on() {
        CALL.with(|s| s.set((Status::Implemented, "")));
    }
}

pub fn mark_stub(note: &'static str) {
    if on() {
        CALL.with(|s| {
            let (status, old) = s.get();
            if status != Status::Stub || old.is_empty() || note.starts_with("service") {
                s.set((Status::Stub, note));
            }
        });
    }
}

pub fn end_ipc(c: &Cpu, iface: &str, req: &crate::services::Request) {
    if !on() {
        return;
    }
    let (status, note) = CALL.with(|s| s.get());
    record(
        Some(c),
        Section::Services,
        iface,
        Id::Num(req.cmd as u64),
        status,
        note,
        || {
            format!(
                "data: {}; buffers A={} B={} X={} C={} W={}, handles copy={} move={}",
                hex(&req.data),
                req.a.len(),
                req.b.len(),
                req.x.len(),
                req.c.len(),
                req.w.len(),
                req.copy_handles.len(),
                req.move_handles.len()
            )
        },
    );
}

pub fn command(c: &Cpu, iface: &str, cmd: u64, status: Status, note: &'static str, args: &[u8]) {
    if on() {
        record(
            Some(c),
            Section::Services,
            iface,
            Id::Num(cmd),
            status,
            note,
            || format!("data: {}", hex(args)),
        );
    }
}

#[inline]
pub fn svc(imm: u16) {
    if on() && SVC_COUNT[imm as usize & 0xff].fetch_add(1, Ordering::Relaxed) == 0 {
        let mut st = STATE.lock().unwrap();
        if let Some(r) = st.as_mut() {
            let frame = frames();
            if !r.items[Section::Svc as usize]
                .get("SVCs")
                .is_some_and(|m| m.contains_key(&Id::Num(imm as u64)))
            {
                r.record(
                    Section::Svc,
                    "SVCs",
                    Id::Num(imm as u64),
                    Status::Implemented,
                    "",
                    0,
                    || r_now(frame),
                    Context::default,
                );
            }
        }
    }
}

pub fn svc_missing(c: &Cpu, imm: u16) {
    if on() {
        record(
            Some(c),
            Section::Svc,
            "SVCs",
            Id::Num(imm as u64),
            Status::Missing,
            "not handled",
            || {
                format!(
                    "x0={:#x} x1={:#x} x2={:#x} x3={:#x} x4={:#x} x5={:#x}",
                    c.x[0], c.x[1], c.x[2], c.x[3], c.x[4], c.x[5]
                )
            },
        );
    }
}

pub fn ioctl(c: &Cpu, device: &str, cmd: u32, handled: bool, args: &[u8]) {
    if on() {
        let (status, note) = if handled {
            (Status::Implemented, "")
        } else {
            (Status::Missing, "not handled")
        };
        record(
            Some(c),
            Section::Ioctl,
            device,
            Id::Num(cmd as u64),
            status,
            note,
            || format!("cmd {cmd:#010x}, args: {}", hex(args)),
        );
    }
}

pub const SHADER_VARIANTS: &str = "maxwell (variantes)";

/// Coverage label for draws that could not run because a variant is missing.
pub const SHADER_DRAWS: &str = "maxwell (draws)";

/// Keeps a skipped (or partially drawn) draw visible in the coverage report,
/// with the variant that caused it.
pub fn draw_untranslated(va: u64, unsupported: &[&'static str], partial: bool) {
    if !on() {
        return;
    }
    let frame = frames();
    let mut st = STATE.lock().unwrap();
    if let Some(r) = st.as_mut() {
        r.draw_untranslated(va, unsupported, partial, r_now(frame));
    }
}

/// A mnemonic is missing if the translator reported it either bare or with a
/// precise sub-form label (`tex (bindless lod)`, `tld (multisample)`, ...).
fn is_missing(mnemonic: &str, unsupported: &[&'static str]) -> bool {
    unsupported.iter().any(|&u| {
        u == mnemonic
            || u.strip_prefix(mnemonic)
                .is_some_and(|rest| rest.starts_with(' '))
    })
}

pub fn shader(va: u64, mnemonics: &[(&'static str, u32)], unsupported: &[&'static str]) {
    if !on() {
        return;
    }
    let c = crate::fault::current_cpu();
    // SAFETY: the guest thread running on this host thread; only read.
    let c = c.and_then(|p| unsafe { p.as_ref() });
    for &(m, n) in mnemonics {
        let missing = is_missing(m, unsupported);
        let (status, note) = if missing {
            (Status::Missing, "no translation")
        } else {
            (Status::Implemented, "")
        };
        let mut st = STATE.lock().unwrap();
        let Some(r) = st.as_mut() else { return };
        if !missing {
            let frame = frames();
            r.record(
                Section::Shaders,
                "maxwell",
                Id::Name(m.to_string()),
                status,
                note,
                n as u64,
                || r_now(frame),
                Context::default,
            );
            continue;
        }
        drop(st);
        record(
            c,
            Section::Shaders,
            "maxwell",
            Id::Name(m.to_string()),
            status,
            note,
            || format!("shader at gpu va {va:#x}, {n} instruction(s)"),
        );
    }
    for &u in unsupported {
        if mnemonics.iter().any(|&(m, _)| m == u) {
            continue;
        }
        record(
            c,
            Section::Shaders,
            SHADER_VARIANTS,
            Id::Name(u.to_string()),
            Status::Missing,
            "left out of the program",
            || format!("shader at gpu va {va:#x}"),
        );
    }
}

fn init_clock(r: &Recorder) {
    let _ = START.set(r.started);
    RUN.store(r.run, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subform_labels_mark_the_mnemonic_missing() {
        assert!(is_missing("tex", &["tex"]));
        assert!(is_missing("tex", &["tex (bindless lod)"]));
        assert!(is_missing("tld", &["tld (multisample)"]));
        assert!(is_missing("tld4", &["tld4 (per-texel offsets)"]));
        assert!(!is_missing("tld4", &["tex (bindless)"]));
        assert!(!is_missing("tex", &["texs (unknown target)"]));
        assert!(!is_missing("tex", &["tld (multisample)"]));
        assert!(!is_missing("imad", &[]));
    }

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("kirikae-usage-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn when() -> When {
        When {
            run: 0,
            seconds: 1.5,
            frame: 10,
            unix: 1,
        }
    }

    fn ctx() -> Context {
        Context {
            backtrace: vec!["main+0x10".into()],
            args: "00 01".into(),
        }
    }

    #[test]
    fn counts_and_events() {
        let dir = temp("counts");
        let mut r = Recorder::open(dir.join("game.coverage.json"), 0x0100_0000_0000_1000);
        assert_eq!(r.events_dir(), dir.join("game-coverage"));
        assert!(
            r.record(
                Section::Services,
                "hid",
                Id::Num(1),
                Status::Implemented,
                "",
                1,
                when,
                ctx
            )
            .is_none()
        );
        assert!(
            r.record(
                Section::Services,
                "hid",
                Id::Num(1),
                Status::Implemented,
                "",
                1,
                when,
                ctx
            )
            .is_none()
        );
        let e = r
            .record(
                Section::Services,
                "hid",
                Id::Num(2),
                Status::Stub,
                "plain ok",
                1,
                when,
                ctx,
            )
            .unwrap();
        assert_eq!((e.n, e.backtrace.len(), e.when.frame), (1, 1, 10));
        assert!(
            r.record(
                Section::Services,
                "hid",
                Id::Num(2),
                Status::Stub,
                "plain ok",
                1,
                when,
                ctx
            )
            .is_none()
        );
        assert!(
            r.record(
                Section::Ioctl,
                "/dev/nvmap",
                Id::Num(0xc0080101),
                Status::Implemented,
                "",
                1,
                when,
                ctx
            )
            .is_none()
        );
        assert_eq!(
            r.record(
                Section::Ioctl,
                "/dev/nvmap",
                Id::Num(0xc0080101),
                Status::Missing,
                "x",
                1,
                when,
                ctx
            )
            .unwrap()
            .n,
            2
        );
        let all = r.merged();
        assert_eq!(all[&(Section::Services, "hid".into(), Id::Num(1))].count, 2);
        assert_eq!(
            all[&(Section::Services, "hid".into(), Id::Num(2))].event,
            Some(1)
        );
        assert_eq!(r.new_events(), 2);
    }

    #[test]
    fn runs_accumulate() {
        let dir = temp("runs");
        let path = dir.join("game.coverage.json");
        let mut r = Recorder::open(path.clone(), 1);
        r.record(
            Section::Svc,
            "SVCs",
            Id::Num(0x29),
            Status::Implemented,
            "",
            3,
            when,
            ctx,
        );
        r.record(
            Section::Shaders,
            "maxwell",
            Id::Name("fadd".into()),
            Status::Implemented,
            "",
            7,
            when,
            ctx,
        );
        r.record(
            Section::Svc,
            "SVCs",
            Id::Num(0x7f),
            Status::Missing,
            "not handled",
            1,
            when,
            ctx,
        )
        .unwrap();
        r.set_screenshot(1, "game-coverage/1/frame.png".into());
        r.write(100).unwrap();

        let mut r = Recorder::open(path.clone(), 1);
        assert_eq!(r.run, 1);
        assert!(
            r.record(
                Section::Svc,
                "SVCs",
                Id::Num(0x7f),
                Status::Missing,
                "not handled",
                1,
                when,
                ctx
            )
            .is_none()
        );
        let e = r
            .record(
                Section::Services,
                "acc:u0",
                Id::Num(60),
                Status::Stub,
                "unimplemented",
                1,
                when,
                ctx,
            )
            .unwrap();
        assert_eq!(e.n, 2);
        r.record(
            Section::Svc,
            "SVCs",
            Id::Num(0x29),
            Status::Implemented,
            "",
            2,
            when,
            ctx,
        );
        r.write(50).unwrap();

        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["runs"].as_array().unwrap().len(), 2);
        assert_eq!(v["events"].as_array().unwrap().len(), 2);
        assert_eq!(v["events"][0]["screenshot"], "game-coverage/1/frame.png");
        let items = v["items"].as_array().unwrap();
        let get_info = items
            .iter()
            .find(|i| i["section"] == "svc" && i["id"] == 0x29)
            .unwrap();
        assert_eq!(get_info["count"], 5);
        assert_eq!(get_info["first"]["run"], 0);
        let fadd = items.iter().find(|i| i["id"] == "fadd").unwrap();
        assert_eq!(
            (fadd["count"].as_u64(), fadd["status"].as_str()),
            (Some(7), Some("implemented"))
        );
        assert_eq!(
            items
                .iter()
                .find(|i| i["section"] == "svc" && i["id"] == 0x7f)
                .unwrap()["event"],
            1
        );
    }

    #[test]
    fn implemented_later_updates_status() {
        let dir = temp("status");
        let path = dir.join("g.coverage.json");
        let mut r = Recorder::open(path.clone(), 1);
        r.record(
            Section::Services,
            "hid",
            Id::Num(5),
            Status::Stub,
            "unimplemented",
            1,
            when,
            ctx,
        )
        .unwrap();
        r.write(0).unwrap();
        let mut r = Recorder::open(path.clone(), 1);
        r.record(
            Section::Services,
            "hid",
            Id::Num(5),
            Status::Implemented,
            "",
            1,
            when,
            ctx,
        );
        let item = &r.merged()[&(Section::Services, "hid".into(), Id::Num(5))];
        assert_eq!(
            (item.status, item.count, item.event),
            (Status::Implemented, 2, Some(1))
        );
    }

    #[test]
    fn untranslated_draws_are_recorded_per_draw() {
        let dir = temp("draws");
        let path = dir.join("g.coverage.json");
        let mut r = Recorder::open(path.clone(), 1);
        let key = (
            Section::Shaders,
            SHADER_DRAWS.to_string(),
            Id::Name("tex (bindless)".into()),
        );
        r.draw_untranslated(0x1000, &["tex (bindless)"], false, when());
        r.draw_untranslated(0x1000, &["tex (bindless)"], false, when());
        let item = &r.merged()[&key];
        assert_eq!(
            (item.count, item.status, item.event, item.note.as_str()),
            (2, Status::Missing, Some(1), "draw skipped")
        );
        r.draw_untranslated(0x1000, &["tex (bindless)"], true, when());
        assert_eq!(r.merged()[&key].count, 3);
        assert_eq!(r.new_events(), 1);
        r.write(1).unwrap();

        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let item = v["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["interface"] == SHADER_DRAWS)
            .unwrap();
        assert_eq!(item["id"], "tex (bindless)");
        assert_eq!(item["count"], 3);
        assert_eq!(item["status"], "missing");
        assert_eq!(item["note"], "draw skipped");
        assert_eq!(v["events"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn event_limit() {
        let dir = temp("limit");
        let mut r = Recorder::open(dir.join("g.coverage.json"), 1);
        r.set_event_limit(3);
        let made = (0..10)
            .filter(|&i| {
                r.record(
                    Section::Services,
                    "x",
                    Id::Num(i),
                    Status::Stub,
                    "",
                    1,
                    when,
                    ctx,
                )
                .is_some()
            })
            .count();
        assert_eq!(made, 3);
        assert_eq!(r.merged().len(), 10);
        r.write(0).unwrap();
        let mut r = Recorder::open(dir.join("g.coverage.json"), 1);
        assert_eq!(
            r.record(
                Section::Services,
                "x",
                Id::Num(5),
                Status::Stub,
                "",
                1,
                when,
                ctx
            )
            .unwrap()
            .n,
            4
        );
    }

    #[test]
    fn damaged_file_is_set_aside() {
        let dir = temp("damaged");
        let path = dir.join("g.coverage.json");
        std::fs::write(&path, "not json").unwrap();
        let mut r = Recorder::open(path.clone(), 1);
        assert!(dir.join("g.coverage.json.bad").exists());
        r.write(0).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn screenshot_png() {
        let px = [
            0xff0000, 0x00ff00, 0x0000ff, 0xffffff, 0xff0000, 0x00ff00, 0x0000ff, 0xffffff,
        ];
        let (w, h, rgb) = scale(4, 2, &px);
        assert_eq!((w, h, &rgb[..6]), (4, 2, &[255u8, 0, 0, 0, 255, 0][..]));
        let data = png(w, h, &rgb);
        assert_eq!(&data[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&data[12..16], b"IHDR");
        let len = u32::from_be_bytes(data[33..37].try_into().unwrap()) as usize;
        assert_eq!(&data[37..41], b"IDAT");
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&data[41..41 + len]).unwrap();
        assert_eq!(raw.len(), (1 + 4 * 3) * 2);
        assert_eq!(&raw[..4], &[0, 255, 0, 0]);
        assert_eq!(&data[data.len() - 4..], &0xae42_6082u32.to_be_bytes());
        let big = vec![0x808080u32; 1280 * 720];
        let (w, h, rgb) = scale(1280, 720, &big);
        assert_eq!((w, h, rgb[0]), (480, 270, 0x80));
    }

    #[test]
    fn saves_copy() {
        let dir = temp("saves");
        std::fs::create_dir_all(dir.join("src/user/sub")).unwrap();
        std::fs::write(dir.join("src/user/a.sav"), b"abc").unwrap();
        std::fs::write(dir.join("src/user/sub/b.sav"), b"de").unwrap();
        assert_eq!(copy_tree(&dir.join("src"), &dir.join("dst")).unwrap(), 5);
        assert_eq!(
            std::fs::read(dir.join("dst/user/sub/b.sav")).unwrap(),
            b"de"
        );
    }
}
