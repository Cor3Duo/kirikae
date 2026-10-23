use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicI32};
use std::sync::{Arc, Mutex};

use crate::services::Service;

pub const PSEUDO_CURRENT_THREAD: u32 = 0xffff_8000;
pub const PSEUDO_CURRENT_PROCESS: u32 = 0xffff_8001;

pub enum SessionKind {
    Object(Arc<dyn Service>),
    Domain(Domain),
}

pub struct Domain {
    objects: BTreeMap<u32, Arc<dyn Service>>,
    next_id: u32,
}

impl Domain {
    pub fn add(&mut self, object: Arc<dyn Service>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.objects.insert(id, object);
        id
    }

    pub fn get(&self, id: u32) -> Option<Arc<dyn Service>> {
        self.objects.get(&id).cloned()
    }

    pub fn remove(&mut self, id: u32) -> bool {
        self.objects.remove(&id).is_some()
    }
}

pub struct Session {
    pub kind: Mutex<SessionKind>,
}

impl Session {
    pub fn new(object: Arc<dyn Service>) -> Arc<Session> {
        Arc::new(Session { kind: Mutex::new(SessionKind::Object(object)) })
    }

    pub fn convert_to_domain(&self) -> u32 {
        let mut kind = self.kind.lock().unwrap();
        if let SessionKind::Object(obj) = &*kind {
            let mut d = Domain { objects: BTreeMap::new(), next_id: 1 };
            d.add(obj.clone());
            *kind = SessionKind::Domain(d);
        }
        1
    }

    pub fn is_domain(&self) -> bool {
        matches!(*self.kind.lock().unwrap(), SessionKind::Domain(_))
    }

    pub fn object(&self) -> Option<Arc<dyn Service>> {
        match &*self.kind.lock().unwrap() {
            SessionKind::Object(o) => Some(o.clone()),
            SessionKind::Domain(d) => d.get(1),
        }
    }

    pub fn domain_object(&self, id: u32) -> Option<Arc<dyn Service>> {
        match &*self.kind.lock().unwrap() {
            SessionKind::Domain(d) => d.get(id),
            SessionKind::Object(o) if id == 1 => Some(o.clone()),
            SessionKind::Object(_) => None,
        }
    }

    pub fn domain_add(&self, object: Arc<dyn Service>) -> u32 {
        match &mut *self.kind.lock().unwrap() {
            SessionKind::Domain(d) => d.add(object),
            SessionKind::Object(_) => panic!("domain_add on a session that is not a domain"),
        }
    }

    pub fn domain_remove(&self, id: u32) -> bool {
        match &mut *self.kind.lock().unwrap() {
            SessionKind::Domain(d) => d.remove(id),
            SessionKind::Object(_) => false,
        }
    }
}

#[derive(Default)]
pub struct Event {
    pub signaled: AtomicBool,
}

pub struct SharedMemory {
    pub size: u64,
    state: Mutex<SharedState>,
}

struct SharedState {
    shadow: Vec<u8>,
    mapped_at: Option<u64>,
}

impl SharedMemory {
    pub fn new(size: u64) -> SharedMemory {
        SharedMemory { size, state: Mutex::new(SharedState { shadow: vec![0; size as usize], mapped_at: None }) }
    }

    pub fn write(&self, memory: &crate::mem::GuestMemory, offset: u64, bytes: &[u8]) {
        let mut st = self.state.lock().unwrap();
        match st.mapped_at {
            Some(addr) => {
                memory.write(addr + offset, bytes);
            }
            None => st.shadow[offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes),
        }
    }

    pub fn read(&self, memory: &crate::mem::GuestMemory, offset: u64, len: u64) -> Vec<u8> {
        let st = self.state.lock().unwrap();
        match st.mapped_at {
            Some(addr) => memory.read(addr + offset, len).unwrap_or_default(),
            None => st.shadow[offset as usize..(offset + len) as usize].to_vec(),
        }
    }

    pub fn map(&self, memory: &crate::mem::GuestMemory, addr: u64, perm: u32) -> bool {
        let mut st = self.state.lock().unwrap();
        if st.mapped_at.is_some() || memory.map_as(addr, self.size, crate::mem::state::SHARED, perm).is_err() {
            return false;
        }
        memory.write(addr, &st.shadow);
        st.mapped_at = Some(addr);
        true
    }

    pub fn unmap(&self, memory: &crate::mem::GuestMemory, addr: u64) -> bool {
        let mut st = self.state.lock().unwrap();
        if st.mapped_at != Some(addr) {
            return false;
        }
        st.shadow = memory.read(addr, self.size).unwrap_or_else(|| vec![0; self.size as usize]);
        st.mapped_at = None;
        memory.unmap(addr, self.size).is_ok()
    }
}

pub struct KThread {
    pub id: u64,
    pub handle: u32,
    pub priority: AtomicI32,
    pub core: AtomicI32,
    pub ideal_core: AtomicI32,
    pub affinity: std::sync::atomic::AtomicU64,
    pub inherited: AtomicI32,
    pub entry: u64,
    pub arg: u64,
    pub stack_top: u64,
    pub tls: u64,
    pub started: AtomicBool,
    pub exited: AtomicBool,
    pub cancelled: AtomicBool,
    pub last_pc: std::sync::atomic::AtomicU64,
    pub last_lr: std::sync::atomic::AtomicU64,
    pub paused: AtomicBool,
    pub context: Mutex<ThreadContext>,
}

impl KThread {
    pub fn effective_priority(&self) -> i32 {
        self.priority.load(std::sync::atomic::Ordering::Relaxed).min(self.inherited.load(std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Clone, Default)]
pub struct ThreadContext {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub psr: u32,
    pub v: [[u64; 2]; 32],
    pub fpcr: u32,
    pub fpsr: u32,
    pub tpidr: u64,
}

impl ThreadContext {
    pub const SIZE: usize = 0x320;

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::SIZE);
        for r in self.x.iter().chain([&self.sp, &self.pc]) {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out.extend_from_slice(&self.psr.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        for q in self.v.iter().flatten() {
            out.extend_from_slice(&q.to_le_bytes());
        }
        out.extend_from_slice(&self.fpcr.to_le_bytes());
        out.extend_from_slice(&self.fpsr.to_le_bytes());
        out.extend_from_slice(&self.tpidr.to_le_bytes());
        debug_assert_eq!(out.len(), Self::SIZE);
        out
    }
}

#[derive(Clone)]
pub enum KObject {
    Session(Arc<Session>),
    Event(Arc<Event>),
    Thread(Arc<KThread>),
    TransferMemory { addr: u64, size: u64 },
    SharedMemory(Arc<SharedMemory>),
}

pub struct HandleTable {
    objects: HashMap<u32, KObject>,
    next: u32,
}

impl Default for HandleTable {
    fn default() -> Self {
        HandleTable { objects: HashMap::new(), next: 0x0002_0001 }
    }
}

impl HandleTable {
    pub fn insert(&mut self, object: KObject) -> u32 {
        let h = self.next;
        self.next += 1;
        self.objects.insert(h, object);
        h
    }

    pub fn insert_at(&mut self, handle: u32, object: KObject) {
        self.objects.insert(handle, object);
    }

    pub fn reserve(&mut self) -> u32 {
        let h = self.next;
        self.next += 1;
        h
    }

    pub fn get(&self, handle: u32) -> Option<KObject> {
        self.objects.get(&handle).cloned()
    }

    pub fn close(&mut self, handle: u32) -> bool {
        self.objects.remove(&handle).is_some()
    }

    pub fn threads(&self) -> Vec<Arc<KThread>> {
        let mut v: Vec<Arc<KThread>> = self.objects.values().filter_map(|o| if let KObject::Thread(t) = o { Some(t.clone()) } else { None }).collect();
        v.sort_by_key(|t| t.id);
        v.dedup_by_key(|t| t.id);
        v
    }

    pub fn session(&self, handle: u32) -> Option<Arc<Session>> {
        match self.objects.get(&handle) {
            Some(KObject::Session(s)) => Some(s.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_context_layout() {
        let mut ctx = ThreadContext { sp: 0x5555, pc: 0x6666, psr: 0x6000_0000, fpcr: 7, fpsr: 8, tpidr: 0x9999, ..Default::default() };
        ctx.x[29] = 0x2929;
        ctx.x[30] = 0x3030;
        ctx.v[31] = [0x11, 0x22];
        let b = ctx.to_bytes();
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        assert_eq!(b.len(), ThreadContext::SIZE);
        assert_eq!((u64_at(0xe8), u64_at(0xf0), u64_at(0xf8), u64_at(0x100)), (0x2929, 0x3030, 0x5555, 0x6666));
        assert_eq!(u64_at(0x108), 0x6000_0000);
        assert_eq!((u64_at(0x300), u64_at(0x308)), (0x11, 0x22));
        assert_eq!((u64_at(0x310), u64_at(0x318)), (7 | 8 << 32, 0x9999));
    }
}
