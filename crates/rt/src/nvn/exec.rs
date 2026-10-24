use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::resources::Surface;
use super::shadow::{Clear, Cmd, Draw, stage_group};
use super::state::{Field, Kind};

pub static ON: AtomicBool = AtomicBool::new(false);
pub static CHECK: AtomicBool = AtomicBool::new(false);

pub const CLEAR: u32 = 0xc1ea;

struct EndCheck {
    segments: u64,
    bindings: u64,
    regs: BTreeMap<usize, (u64, u32, u32)>,
}

static DIFFER: Mutex<EndCheck> = Mutex::new(EndCheck { segments: 0, bindings: 0, regs: BTreeMap::new() });

static REASONS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

pub(super) fn counted(why: Option<&str>) {
    match why {
        None => _ = EXECUTED.fetch_add(1, Ordering::Relaxed),
        Some(w) => {
            FALLBACK.fetch_add(1, Ordering::Relaxed);
            let key: String = w.chars().take(48).collect();
            *REASONS.lock().unwrap().entry(key).or_default() += 1;
        }
    }
}

type Bindings = [[Option<(u64, u32)>; crate::gpu::threed::CB_SLOTS]; 5];

static BINDINGS: Mutex<BTreeMap<(usize, usize), (u64, String)>> = Mutex::new(BTreeMap::new());

#[derive(Default)]
struct DrawCheck {
    expected: std::collections::VecDeque<(Vec<u32>, Bindings, crate::gpu::threed::Draw)>,
    draws: u64,
    params: u64,
    unpaired: u64,
    left: u64,
    regs: BTreeMap<usize, (u64, u32, u32)>,
    bindings: BTreeMap<(usize, usize), (u64, String)>,
}

static DRAWS: Mutex<Option<DrawCheck>> = Mutex::new(None);

fn noise(r: usize) -> bool {
    matches!(r, 0x8e0..=0x8f3 | 0x900..=0x928 | 0xd00..=0xd1c | 0xd2a..=0xdff | 0x35d | 0x35e | 0x5f7 | 0x5f8 | 0x585 | 0x586 | 0x6c0..=0x6c3)
}

static WRITES: Mutex<(u64, u64, u64)> = Mutex::new((0, 0, 0));

pub fn writes_checked(words: usize, differ: usize) {
    let mut w = WRITES.lock().unwrap();
    w.0 += 1;
    w.1 += words as u64;
    w.2 += differ as u64;
}

pub fn expect_draw(regs: Vec<u32>, cbs: Bindings, draw: crate::gpu::threed::Draw) {
    DRAWS.lock().unwrap().get_or_insert_with(DrawCheck::default).expected.push_back((regs, cbs, draw));
}

pub fn cp_draw(regs: &[u32], cbs: &Bindings, draw: &crate::gpu::threed::Draw) {
    let mut g = DRAWS.lock().unwrap();
    let c = g.get_or_insert_with(DrawCheck::default);
    let Some((sim, sim_cbs, sim_draw)) = c.expected.pop_front() else {
        c.unpaired += 1;
        return;
    };
    c.draws += 1;
    if format!("{sim_draw:?}") != format!("{draw:?}") {
        c.params += 1;
        if c.params <= 8 {
            eprintln!("kirikae: nvn exec draw check: host {sim_draw:?}, command processor {draw:?}");
        }
    }
    let tag = if draw.topology == CLEAR { 0x10000 } else { 0 };
    for (r, (&s, &a)) in sim.iter().zip(regs).enumerate() {
        if s != a && !noise(r) {
            c.regs.entry(r + tag).or_insert((0, s, a)).0 += 1;
        }
    }
    for gr in 0..5 {
        for s in 0..crate::gpu::threed::CB_SLOTS {
            if sim_cbs[gr][s] != cbs[gr][s] {
                c.bindings.entry((gr, s)).or_insert_with(|| (0, format!("{:x?} vs {:x?}", sim_cbs[gr][s], cbs[gr][s]))).0 += 1;
            }
        }
    }
}

pub fn segment_done() {
    let mut g = DRAWS.lock().unwrap();
    if let Some(c) = g.as_mut() {
        c.left += c.expected.len() as u64;
        c.expected.clear();
    }
}

pub fn checked(simulated: &[u32], actual: &[u32], sim: &Bindings, cp: &Bindings) {
    let mut d = DIFFER.lock().unwrap();
    d.segments += 1;
    if sim != cp {
        d.bindings += 1;
        let mut b = BINDINGS.lock().unwrap();
        for g in 0..5 {
            for s in 0..crate::gpu::threed::CB_SLOTS {
                if sim[g][s] != cp[g][s] {
                    let e = b.entry((g, s)).or_insert_with(|| (0, format!("{:x?} vs {:x?}", sim[g][s], cp[g][s])));
                    e.0 += 1;
                }
            }
        }
    }
    for (r, (&s, &a)) in simulated.iter().zip(actual).enumerate() {
        if s != a {
            let e = d.regs.entry(r).or_insert((0, s, a));
            e.0 += 1;
        }
    }
}

pub static EXECUTED: AtomicU64 = AtomicU64::new(0);
pub static FALLBACK: AtomicU64 = AtomicU64::new(0);

pub enum Step {
    Regs(Vec<(usize, u32, u32)>),
    Bind { group: usize, slot: usize, binding: Option<(u64, u32)> },
    Write { at: u64, words: Vec<u32> },
    Draw(crate::gpu::threed::Draw),
    Clear(u32),
}

#[derive(Default)]
pub(super) struct Tracker {
    pub(super) program: Arc<Vec<u64>>,
    pub(super) program_object: u64,
    program_id: u32,
    pub(super) targets: [u64; 9],
    surfaces: [Option<Surface>; 9],
    pub(super) uniforms: BTreeMap<(u8, u8), (u64, u64)>,
    vertex: BTreeMap<u8, u64>,
    vertex_sizes: BTreeMap<u8, u64>,
    textures: BTreeMap<(u8, u16), u64>,
    state: BTreeMap<Field, u64>,
    pools: [u64; 2],
    pub(super) storage: BTreeMap<(u8, u8), (u64, u64)>,
    pub(super) scratch: (u64, u64),
}

impl Tracker {
    pub(super) fn apply(&mut self, cmd: &Cmd) {
        match cmd {
            Cmd::Program(object, id, p) => (self.program_object, self.program_id, self.program) = (*object, *id, p.clone()),
            Cmd::State(fields) => self.state.extend(fields.iter().copied()),
            Cmd::Uniform { stage, index, range } => {
                if let Some(g) = stage_group(*stage) {
                    self.uniforms.insert((g, index + 3), (range.0, range.1.next_multiple_of(0x100).min(0x10000)));
                }
            }
            Cmd::Vertex { index, address, size } => {
                self.vertex.insert(*index, *address);
                self.vertex_sizes.insert(*index, *size);
            }
            Cmd::Texture { stage, index, handle } => {
                if let Some(g) = stage_group(*stage) {
                    self.textures.insert((g, *index), *handle);
                }
            }
            Cmd::Targets(t, s) => (self.targets, self.surfaces) = (*t, *s),
            Cmd::Pool { texture, address } => self.pools[usize::from(!*texture)] = *address,
            Cmd::Storage { stage, index, address, size } => {
                if let Some(g) = stage_group(*stage) {
                    self.storage.insert((g, *index), (*address, *size));
                }
            }
            Cmd::Scratch { address, size } => self.scratch = (*address, *size),
            _ => {}
        }
    }

    pub(super) fn draw(&self, d: &Draw) -> Draw {
        Draw {
            programs: self.program.to_vec(),
            targets: [self.targets[0], self.targets[8]],
            uniforms: self.uniforms.iter().map(|(k, v)| (*k, *v)).collect(),
            vertex: self.vertex.iter().map(|(k, v)| (*k, *v)).collect(),
            vertex_sizes: self.vertex_sizes.iter().map(|(k, v)| (*k, *v)).collect(),
            textures: self.textures.iter().map(|(k, v)| (*k, *v)).collect(),
            state: self.target_state(),
            surfaces: self.surfaces.iter().enumerate().filter_map(|(k, s)| s.map(|s| (k as u8, s))).collect(),
            all_targets: self.targets,
            pools: self.pools,
            program: self.program_object,
            program_id: self.program_id,
            ..d.clone()
        }
    }

    pub(super) fn layers(&self, cl: &Clear) -> u32 {
        let index = if cl.what & 0x3c != 0 { (cl.what >> 6 & 7) as usize } else { 8 };
        self.surfaces[index].map_or(1, |s| s.layers.max(1))
    }

    pub(super) fn clear(&self, cl: &Clear) -> Clear {
        let index = (cl.what >> 6 & 7) as usize;
        let target = if cl.what & 0x3c != 0 { self.targets[index] } else { self.targets[8] };
        Clear { target, ..cl.clone() }
    }

    fn target_state(&self) -> Vec<(Field, u64)> {
        let mut state: Vec<(Field, u64)> = self.state.iter().map(|(k, v)| (*k, *v)).collect();
        for (k, s) in self.surfaces.iter().enumerate() {
            if let Some(s) = s {
                state.push(if k < 8 { (Field(Kind::TargetFormat, k as u8), s.format as u64) } else { (Field(Kind::DepthFormat, 0), s.format as u64) });
            }
        }
        state
    }
}

pub fn report() {
    if !ON.load(Ordering::Relaxed) && !CHECK.load(Ordering::Relaxed) {
        return;
    }
    let paired = super::shadow::executed_report();
    let d = DIFFER.lock().unwrap();
    if d.segments > 0 {
        let mut regs: Vec<_> = d.regs.iter().collect();
        regs.sort_by_key(|r| std::cmp::Reverse(r.1.0));
        let regs: Vec<String> = regs.iter().take(40).map(|(r, (n, s, a))| format!("{r:#x} {n} ({s:#x} vs {a:#x})")).collect();
        eprintln!("kirikae:         nvn exec check: {} segments, {} with other constant buffer bindings; registers left different: {}", d.segments, d.bindings, regs.join(", "));
        let b = BINDINGS.lock().unwrap();
        let mut b: Vec<_> = b.iter().collect();
        b.sort_by_key(|x| std::cmp::Reverse(x.1.0));
        let b: Vec<String> = b.iter().take(16).map(|((g, s), (n, ex))| format!("group {g} slot {s}: {n} ({ex})")).collect();
        eprintln!("kirikae:         nvn exec check bindings: {}", b.join("; "));
    }
    if let Some(c) = DRAWS.lock().unwrap().as_ref() {
        let mut regs: Vec<_> = c.regs.iter().collect();
        regs.sort_by_key(|r| std::cmp::Reverse(r.1.0));
        let regs: Vec<String> = regs.iter().take(40).map(|(r, (n, s, a))| format!("{r:#x} {n} ({s:#x} vs {a:#x})")).collect();
        let mut b: Vec<_> = c.bindings.iter().collect();
        b.sort_by_key(|x| std::cmp::Reverse(x.1.0));
        let b: Vec<String> = b.iter().take(12).map(|((g, s), (n, ex))| format!("group {g} slot {s}: {n} ({ex})")).collect();
        eprintln!(
            "kirikae:         nvn exec draw check: {} draws, {} with other parameters, {} not simulated, {} simulated but not drawn; registers: {}; bindings: {}",
            c.draws,
            c.params,
            c.unpaired,
            c.left,
            regs.join(", "),
            b.join("; ")
        );
    }
    let w = *WRITES.lock().unwrap();
    eprintln!("kirikae:         nvn exec writes: {} segments, {} words, {} differ from the command processor", w.0, w.1, w.2);
    let r = REASONS.lock().unwrap();
    let mut r: Vec<_> = r.iter().collect();
    r.sort_by_key(|x| std::cmp::Reverse(*x.1));
    let r: Vec<String> = r.iter().take(16).map(|(k, n)| format!("{k} {n}")).collect();
    eprintln!("kirikae:         nvn exec fallback reasons: {}", r.join(", "));
    eprintln!(
        "kirikae:         nvn exec: {} pushbuffer segments run on the host{}, {} left to the command processor",
        EXECUTED.load(Ordering::Relaxed),
        paired,
        FALLBACK.load(Ordering::Relaxed)
    );
}
