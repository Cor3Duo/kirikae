use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::mem::GuestMemory;
use crate::services::Service;

pub const PSEUDO_CURRENT_THREAD: u32 = 0xffff_8000;
pub const PSEUDO_CURRENT_PROCESS: u32 = 0xffff_8001;

pub enum SessionKind {
    Object(Arc<dyn Service>),
    Domain(Domain),
    /// Client end of a session pair made by `svcCreateSession` on the guest.
    GuestClient(Arc<GuestSession>),
    /// Server end produced by `svcAcceptSession` over a queued client.
    GuestServer(Arc<GuestSession>),
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
        Arc::new(Session {
            kind: Mutex::new(SessionKind::Object(object)),
        })
    }

    /// The client end of a guest session pair (`svcCreateSession`).
    pub fn guest_client(link: Arc<GuestSession>) -> Arc<Session> {
        Arc::new(Session {
            kind: Mutex::new(SessionKind::GuestClient(link)),
        })
    }

    /// The server end (`svcAcceptSession`) over the same session pair.
    pub fn guest_server(link: Arc<GuestSession>) -> Arc<Session> {
        Arc::new(Session {
            kind: Mutex::new(SessionKind::GuestServer(link)),
        })
    }

    pub fn guest_client_link(&self) -> Option<Arc<GuestSession>> {
        match &*self.kind.lock().unwrap() {
            SessionKind::GuestClient(link) => Some(link.clone()),
            _ => None,
        }
    }

    pub fn guest_server_link(&self) -> Option<Arc<GuestSession>> {
        match &*self.kind.lock().unwrap() {
            SessionKind::GuestServer(link) => Some(link.clone()),
            _ => None,
        }
    }

    /// Whether this handle is one end of a guest session pair. Those are
    /// served by the rendezvous in [`GuestSession`], not by a host object.
    pub fn is_guest(&self) -> bool {
        matches!(
            &*self.kind.lock().unwrap(),
            SessionKind::GuestClient(_) | SessionKind::GuestServer(_)
        )
    }

    /// Marks the matching end of a guest session pair as closed.
    pub fn close_guest_end(&self) {
        match &*self.kind.lock().unwrap() {
            SessionKind::GuestClient(link) => link.close_client(),
            SessionKind::GuestServer(link) => link.close_server(),
            _ => {}
        }
    }

    pub fn convert_to_domain(&self) -> u32 {
        let mut kind = self.kind.lock().unwrap();
        if let SessionKind::Object(obj) = &*kind {
            let mut d = Domain {
                objects: BTreeMap::new(),
                next_id: 1,
            };
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
            SessionKind::GuestClient(_) | SessionKind::GuestServer(_) => None,
        }
    }

    pub fn domain_object(&self, id: u32) -> Option<Arc<dyn Service>> {
        match &*self.kind.lock().unwrap() {
            SessionKind::Domain(d) => d.get(id),
            SessionKind::Object(o) if id == 1 => Some(o.clone()),
            _ => None,
        }
    }

    pub fn domain_add(&self, object: Arc<dyn Service>) -> u32 {
        match &mut *self.kind.lock().unwrap() {
            SessionKind::Domain(d) => d.add(object),
            _ => panic!("domain_add on a session that is not a domain"),
        }
    }

    pub fn domain_remove(&self, id: u32) -> bool {
        match &mut *self.kind.lock().unwrap() {
            SessionKind::Domain(d) => d.remove(id),
            _ => false,
        }
    }
}

#[derive(Default)]
pub struct Event {
    pub signaled: AtomicBool,
}

/// One guest session pair (`svcCreateSession`) as the kernel models it.
///
/// `svcCreateSession` creates one `GuestSession` and two handles: the client end
/// ([`SessionKind::GuestClient`]) and a [`Port`] whose queue holds the pair
/// until `svcAcceptSession` turns it into a server session handle
/// ([`SessionKind::GuestServer`]). A request/reply exchange moves through this
/// record:
///
/// 1. the client queues a message with [`GuestSession::send`] and blocks;
/// 2. the server takes it with [`GuestSession::take_request`];
/// 3. the server posts the reply with [`GuestSession::post_reply`] and the
///    client takes it with [`GuestSession::take_reply`].
///
/// A host-implemented server plugs in at the same three methods: a host thread
/// would call `take_request`, build the reply and call `post_reply`, which is
/// exactly what the guest server does from `svcReplyAndReceiveLight`.
pub struct GuestSession {
    link: Mutex<GuestLink>,
    /// Wakes a host service blocked in [`GuestSession::take_request_blocking`].
    changed: std::sync::Condvar,
    /// Wakes the process scheduler. Both directions of the rendezvous go
    /// through it, so a host service that posts a reply from its own thread
    /// wakes a guest blocked in `svcSendSyncRequest`.
    wake: Box<dyn Fn() + Send + Sync>,
}

/// Why a client could not queue a request.
#[derive(Debug, PartialEq, Eq)]
pub enum SendError {
    /// A request is already pending or being served on this pair.
    Busy,
    /// The server end is closed, so nothing can ever reply.
    RemoteDead,
}

#[derive(Default)]
struct GuestLink {
    /// A request that the server has not taken yet.
    request: Option<Vec<u8>>,
    /// A reply the client has not taken yet.
    reply: Option<Vec<u8>>,
    client_open: bool,
    server_open: bool,
    /// The client thread is inside `svcSendSyncRequest` waiting for a reply.
    client_waiting: bool,
    /// The server took a request and has not posted the reply for it.
    in_service: bool,
}

impl GuestSession {
    /// A pair with no process behind it, for unit tests and objects that are
    /// never waited on. The kernel creates the real ones with [`Self::with_wake`].
    pub fn new() -> Arc<GuestSession> {
        Self::with_wake(|| {})
    }

    /// A pair whose state changes wake `wake` (normally `sched::wake`), so
    /// either end can be served by a host thread.
    pub fn with_wake(wake: impl Fn() + Send + Sync + 'static) -> Arc<GuestSession> {
        Arc::new(GuestSession {
            link: Mutex::new(GuestLink {
                client_open: true,
                server_open: true,
                ..GuestLink::default()
            }),
            changed: std::sync::Condvar::new(),
            wake: Box::new(wake),
        })
    }

    /// Queues a client request. The caller wakes the server and then waits for
    /// the reply. A reply left over from a cancelled wait is dropped: a request
    /// is only ever answered by the reply to that request.
    pub fn send(&self, bytes: Vec<u8>) -> Result<(), SendError> {
        let result = {
            let mut l = self.link.lock().unwrap();
            if !l.server_open {
                Err(SendError::RemoteDead)
            } else if l.request.is_some() || l.in_service {
                Err(SendError::Busy)
            } else {
                l.request = Some(bytes);
                l.reply = None;
                l.client_waiting = true;
                self.changed.notify_all();
                Ok(())
            }
        };
        if result.is_ok() {
            (self.wake)();
        }
        result
    }

    /// The next request, marking the pair as in service until the reply lands.
    pub fn take_request(&self) -> Option<Vec<u8>> {
        let mut l = self.link.lock().unwrap();
        let request = l.request.take()?;
        l.in_service = true;
        Some(request)
    }

    /// The next request, for a host service that serves the guest from its own
    /// thread. Blocks until a request arrives, the client closes, or `timeout`
    /// expires; `None` covers the last two. The guest server inside
    /// `svcReplyAndReceive` waits on the scheduler instead and uses
    /// [`GuestSession::take_request`].
    pub fn take_request_blocking(&self, timeout: std::time::Duration) -> Option<Vec<u8>> {
        let deadline = std::time::Instant::now() + timeout;
        let mut l = self.link.lock().unwrap();
        loop {
            if let Some(request) = l.request.take() {
                l.in_service = true;
                return Some(request);
            }
            if !l.client_open {
                return None;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            let (guard, _) = self
                .changed
                .wait_timeout(l, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            l = guard;
        }
    }

    /// Posts the reply for the request in service and wakes the waiting client.
    /// `false` when the client is gone, in which case the reply is dropped: a
    /// server does not fail because a client vanished.
    pub fn post_reply(&self, bytes: Vec<u8>) -> bool {
        let posted = {
            let mut l = self.link.lock().unwrap();
            l.in_service = false;
            if !l.client_open || !l.client_waiting {
                false
            } else {
                l.reply = Some(bytes);
                true
            }
        };
        if posted {
            // Not while holding the link lock: a waiter holds the scheduler
            // lock and takes this lock from its readiness check.
            (self.wake)();
        }
        posted
    }

    /// The reply a waiting client takes, or `None` while there is none. A
    /// readiness check may call this any number of times: only a reply clears
    /// the waiting flag, so a poll never cancels the wait.
    pub fn take_reply(&self) -> Option<Vec<u8>> {
        let mut l = self.link.lock().unwrap();
        let reply = l.reply.take()?;
        l.client_waiting = false;
        Some(reply)
    }

    /// The waiting client left without a reply (`svcCancelSynchronization`).
    /// A request the server has not taken is withdrawn; one in service keeps
    /// its reply slot closed.
    pub fn withdraw(&self) {
        let mut l = self.link.lock().unwrap();
        if !l.in_service {
            l.request = None;
        }
        l.client_waiting = false;
    }

    pub fn close_client(&self) {
        {
            let mut l = self.link.lock().unwrap();
            // A request the server has not taken yet dies with the client; one
            // already in service keeps its reply slot, which `post_reply` drops.
            if !l.in_service {
                l.request = None;
            }
            l.client_open = false;
            l.client_waiting = false;
            l.reply = None;
            self.changed.notify_all();
        }
        (self.wake)();
    }

    pub fn close_server(&self) {
        {
            let mut l = self.link.lock().unwrap();
            l.server_open = false;
            l.request = None;
        }
        (self.wake)();
    }

    pub fn client_open(&self) -> bool {
        self.link.lock().unwrap().client_open
    }

    pub fn server_open(&self) -> bool {
        self.link.lock().unwrap().server_open
    }

    pub fn has_request(&self) -> bool {
        self.link.lock().unwrap().request.is_some()
    }

    pub fn in_service(&self) -> bool {
        self.link.lock().unwrap().in_service
    }
}

/// A server port (`svcCreateSession`'s server handle). Kernel ports are
/// signalable while a client connection waits to be accepted, which is what
/// `svcWaitSynchronization` and the port entries of `svcReplyAndReceive` use.
pub struct Port {
    pub name: u64,
    pub is_light: bool,
    queue: Mutex<VecDeque<Arc<GuestSession>>>,
}

impl Port {
    pub fn new(name: u64, is_light: bool, first: Arc<GuestSession>) -> Arc<Port> {
        Arc::new(Port {
            name,
            is_light,
            queue: Mutex::new(VecDeque::from([first])),
        })
    }

    /// Pops the next queued client. `None` means no connection is waiting, the
    /// documented `0xf201` of `svcAcceptSession`.
    pub fn accept(&self) -> Option<Arc<GuestSession>> {
        self.queue.lock().unwrap().pop_front()
    }

    pub fn pending(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Closing the port drops the queued pairs, so a client blocked on one
    /// learns that no server can ever reply. The queue lock is released first:
    /// closing a pair wakes the scheduler, and a thread blocked on this port
    /// holds the scheduler lock while it looks at the queue.
    pub fn close(&self) {
        let queued: Vec<Arc<GuestSession>> = {
            let mut queue = self.queue.lock().unwrap();
            queue.drain(..).collect()
        };
        for link in queued {
            link.close_server();
        }
    }
}

/// A block of transfer memory (`svcCreateTransferMemory`). The kernel keeps the
/// pages where the creator left them (reprotected to the owner permission) and
/// `svcMapTransferMemory` aliases them at another address. The runtime keeps one
/// host view, so a mapping is a copy that is written back when it is unmapped;
/// the creator's pages stay the source of truth, which is why only one mapping
/// exists at a time, as the kernel guarantees.
pub struct TransferMemory {
    pub addr: u64,
    pub size: u64,
    /// The permission the creating process kept.
    pub perm: u32,
    mapped_at: Mutex<Option<u64>>,
}

impl TransferMemory {
    pub fn new(addr: u64, size: u64, perm: u32) -> Arc<TransferMemory> {
        Arc::new(TransferMemory {
            addr,
            size,
            perm,
            mapped_at: Mutex::new(None),
        })
    }

    /// `svcMapTransferMemory`. `false` when the block is already mapped or the
    /// destination cannot hold it.
    pub fn map(&self, memory: &GuestMemory, addr: u64, perm: u32) -> bool {
        let mut mapped_at = self.mapped_at.lock().unwrap();
        if mapped_at.is_some()
            || memory
                .map_as(addr, self.size, crate::mem::state::TRANSFER, perm)
                .is_err()
        {
            return false;
        }
        if !memory.copy(addr, self.addr, self.size) {
            let _ = memory.unmap(addr, self.size);
            return false;
        }
        *mapped_at = Some(addr);
        true
    }

    /// `svcUnmapTransferMemory`: writes the mapping back to the creator's pages
    /// and drops it. `false` when the block is not mapped at `addr`.
    pub fn unmap(&self, memory: &GuestMemory, addr: u64) -> bool {
        let mut mapped_at = self.mapped_at.lock().unwrap();
        if *mapped_at != Some(addr) {
            return false;
        }
        if !memory.copy(self.addr, addr, self.size) {
            return false;
        }
        *mapped_at = None;
        memory.unmap(addr, self.size).is_ok()
    }

    pub fn mapped_at(&self) -> Option<u64> {
        *self.mapped_at.lock().unwrap()
    }
}

/// Code memory (`svcCreateCodeMemory`), the JIT backing store. The kernel takes
/// the pages out of the process and leaves them `0xFF`-filled;
/// `svcControlCodeMemory` maps them writable, then executable, and the contents
/// move with the object the same way transfer memory's do.
pub struct CodeMemory {
    pub size: u64,
    state: Mutex<CodeState>,
}

struct CodeState {
    data: Vec<u8>,
    mapped_at: Option<(u64, u32)>,
}

impl CodeMemory {
    pub fn new(size: u64) -> Arc<CodeMemory> {
        Arc::new(CodeMemory {
            size,
            state: Mutex::new(CodeState {
                data: vec![0xff; size as usize],
                mapped_at: None,
            }),
        })
    }

    /// `CodeMemoryOperation_MapOwner`: the first mapping copies the contents
    /// out, a repeated mapping only changes the permission.
    pub fn map_owner(&self, memory: &GuestMemory, addr: u64, perm: u32) -> bool {
        let mut st = self.state.lock().unwrap();
        if let Some((at, current)) = st.mapped_at {
            if at != addr {
                return false;
            }
            if current == perm {
                return true;
            }
            let end = addr + self.size;
            if !memory.protect(
                addr,
                end,
                perm,
                &[
                    crate::mem::state::CODE_WRITABLE,
                    crate::mem::state::CODE_READONLY,
                ],
            ) {
                return false;
            }
            st.mapped_at = Some((addr, perm));
            return true;
        }
        let state = if perm & crate::mem::perm::RW == crate::mem::perm::RW {
            crate::mem::state::CODE_WRITABLE
        } else {
            crate::mem::state::CODE_READONLY
        };
        if memory.map_as(addr, self.size, state, perm).is_err() {
            return false;
        }
        if !memory.write(addr, &st.data) {
            let _ = memory.unmap(addr, self.size);
            return false;
        }
        st.mapped_at = Some((addr, perm));
        true
    }

    /// `CodeMemoryOperation_UnmapOwner`.
    pub fn unmap_owner(&self, memory: &GuestMemory, addr: u64) -> bool {
        let mut st = self.state.lock().unwrap();
        if st.mapped_at.map(|(at, _)| at) != Some(addr) {
            return false;
        }
        let Some(data) = memory.read(addr, self.size) else {
            return false;
        };
        if memory.unmap(addr, self.size).is_err() {
            return false;
        }
        st.data = data;
        st.mapped_at = None;
        true
    }

    pub fn mapped_at(&self) -> Option<(u64, u32)> {
        self.state.lock().unwrap().mapped_at
    }
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
        SharedMemory {
            size,
            state: Mutex::new(SharedState {
                shadow: vec![0; size as usize],
                mapped_at: None,
            }),
        }
    }

    pub fn write(&self, memory: &crate::mem::GuestMemory, offset: u64, bytes: &[u8]) {
        let mut st = self.state.lock().unwrap();
        match st.mapped_at {
            Some(addr) => {
                memory.write(addr + offset, bytes);
            }
            None => {
                st.shadow[offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes)
            }
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
        if st.mapped_at.is_some()
            || memory
                .map_as(addr, self.size, crate::mem::state::SHARED, perm)
                .is_err()
        {
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
        st.shadow = memory
            .read(addr, self.size)
            .unwrap_or_else(|| vec![0; self.size as usize]);
        st.mapped_at = None;
        memory.unmap(addr, self.size).is_ok()
    }
}

/// The resource limit the runtime models. Current values are derived from the
/// live process, so only the ceilings and the peaks live here.
pub struct ResourceLimit {
    values: [AtomicU64; ResourceLimit::COUNT],
    peak: Mutex<[u64; ResourceLimit::COUNT]>,
}

impl ResourceLimit {
    pub const COUNT: usize = 5;

    pub fn new(physical_memory_max: u64) -> ResourceLimit {
        // The ceilings of a regular application's NPDM: 64 threads, 0x100
        // events and transfer memories and 0x200 sessions.
        Self::with([physical_memory_max, 0x40, 0x100, 0x100, 0x200])
    }

    /// A limit as `svcCreateResourceLimit` leaves it: every LimitableResource
    /// starts at zero until the guest raises it.
    pub fn empty() -> ResourceLimit {
        Self::with([0; ResourceLimit::COUNT])
    }

    fn with(values: [u64; ResourceLimit::COUNT]) -> ResourceLimit {
        ResourceLimit {
            values: values.map(AtomicU64::new),
            peak: Mutex::new([0; ResourceLimit::COUNT]),
        }
    }

    /// The ceiling of `which`, or `None` when it is not a LimitableResource.
    pub fn limit(&self, which: u64) -> Option<u64> {
        let i = usize::try_from(which).ok().filter(|i| *i < Self::COUNT)?;
        Some(self.values[i].load(Ordering::Relaxed))
    }

    /// `svcSetResourceLimitLimitValue`. The kernel refuses a ceiling below the
    /// current usage, and raising or lowering it resets the peak to the
    /// current value. `None` means `which` is not a LimitableResource.
    pub fn set_limit(&self, which: u64, value: i64, current: u64) -> Option<bool> {
        let i = usize::try_from(which).ok().filter(|i| *i < Self::COUNT)?;
        let mut peak = self.peak.lock().unwrap();
        if current as i64 > value {
            return Some(false);
        }
        self.values[i].store(value as u64, Ordering::Relaxed);
        peak[i] = current;
        Some(true)
    }

    /// `svcGetResourceLimitPeakValue`. Peaks are sampled when the guest asks:
    /// the runtime derives current values from the live handle table instead
    /// of accounting every allocation.
    pub fn record_peak(&self, which: u64, current: u64) -> Option<u64> {
        let i = usize::try_from(which).ok().filter(|i| *i < Self::COUNT)?;
        let mut peak = self.peak.lock().unwrap();
        peak[i] = peak[i].max(current);
        Some(peak[i])
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
        self.priority
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(self.inherited.load(std::sync::atomic::Ordering::Relaxed))
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
    /// A server port from `svcCreateSession`.
    Port(Arc<Port>),
    Event(Arc<Event>),
    Thread(Arc<KThread>),
    TransferMemory(Arc<TransferMemory>),
    CodeMemory(Arc<CodeMemory>),
    SharedMemory(Arc<SharedMemory>),
    ResourceLimit(Arc<ResourceLimit>),
}

pub struct HandleTable {
    objects: HashMap<u32, KObject>,
    next: u32,
    resource_limit: Option<u32>,
}

impl Default for HandleTable {
    fn default() -> Self {
        HandleTable {
            objects: HashMap::new(),
            next: 0x0002_0001,
            resource_limit: None,
        }
    }
}

impl HandleTable {
    pub fn insert(&mut self, object: KObject) -> u32 {
        let h = self.next;
        self.next += 1;
        self.objects.insert(h, object);
        h
    }

    /// Handle of the process resource limit, created once and recreated if the
    /// guest closed it.
    pub fn resource_limit(&mut self, make: impl FnOnce() -> KObject) -> u32 {
        match self.resource_limit {
            Some(handle) if self.objects.contains_key(&handle) => handle,
            _ => {
                let handle = self.insert(make());
                self.resource_limit = Some(handle);
                handle
            }
        }
    }

    /// The handle of the process resource limit, if the guest obtained it.
    pub fn process_resource_limit(&self) -> Option<u32> {
        self.resource_limit
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

    pub fn objects(&self) -> impl Iterator<Item = &KObject> {
        self.objects.values()
    }

    pub fn threads(&self) -> Vec<Arc<KThread>> {
        let mut v: Vec<Arc<KThread>> = self
            .objects
            .values()
            .filter_map(|o| {
                if let KObject::Thread(t) = o {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
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

    pub fn port(&self, handle: u32) -> Option<Arc<Port>> {
        match self.objects.get(&handle) {
            Some(KObject::Port(p)) => Some(p.clone()),
            _ => None,
        }
    }

    pub fn transfer_memory(&self, handle: u32) -> Option<Arc<TransferMemory>> {
        match self.objects.get(&handle) {
            Some(KObject::TransferMemory(t)) => Some(t.clone()),
            _ => None,
        }
    }

    pub fn code_memory(&self, handle: u32) -> Option<Arc<CodeMemory>> {
        match self.objects.get(&handle) {
            Some(KObject::CodeMemory(m)) => Some(m.clone()),
            _ => None,
        }
    }

    /// Sessions the process holds, which is what the
    /// `LimitableResource_Sessions` ceiling accounts for: both ends of a session
    /// pair, including a server end still queued on its port until
    /// `svcAcceptSession` turns it into a session handle.
    pub fn sessions_in_use(&self) -> u64 {
        let mut used = 0;
        for object in self.objects.values() {
            match object {
                KObject::Session(_) => used += 1,
                KObject::Port(port) => used += port.pending() as u64,
                _ => {}
            }
        }
        used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_context_layout() {
        let mut ctx = ThreadContext {
            sp: 0x5555,
            pc: 0x6666,
            psr: 0x6000_0000,
            fpcr: 7,
            fpsr: 8,
            tpidr: 0x9999,
            ..Default::default()
        };
        ctx.x[29] = 0x2929;
        ctx.x[30] = 0x3030;
        ctx.v[31] = [0x11, 0x22];
        let b = ctx.to_bytes();
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        assert_eq!(b.len(), ThreadContext::SIZE);
        assert_eq!(
            (u64_at(0xe8), u64_at(0xf0), u64_at(0xf8), u64_at(0x100)),
            (0x2929, 0x3030, 0x5555, 0x6666)
        );
        assert_eq!(u64_at(0x108), 0x6000_0000);
        assert_eq!((u64_at(0x300), u64_at(0x308)), (0x11, 0x22));
        assert_eq!((u64_at(0x310), u64_at(0x318)), (7 | 8 << 32, 0x9999));
    }

    #[test]
    fn a_session_pair_moves_one_request_and_one_reply() {
        let link = GuestSession::new();
        assert!(link.send(b"hello".to_vec()).is_ok());
        assert_eq!(
            link.send(b"second".to_vec()),
            Err(SendError::Busy),
            "one request at a time"
        );
        assert_eq!(link.take_request().as_deref(), Some(&b"hello"[..]));
        assert!(link.in_service());
        assert_eq!(link.take_request(), None, "the request was taken once");
        assert!(link.post_reply(b"world".to_vec()));
        assert!(!link.in_service());
        assert_eq!(link.take_reply().as_deref(), Some(&b"world"[..]));
        assert_eq!(link.take_reply(), None, "one reply per request");
        assert!(link.send(b"again".to_vec()).is_ok(), "the pair is reusable");
    }

    #[test]
    fn a_closed_client_drops_the_late_reply() {
        let link = GuestSession::new();
        link.send(b"hello".to_vec()).unwrap();
        link.take_request().unwrap();
        link.close_client();
        assert!(link.send(b"more".to_vec()).is_err());
        assert!(
            !link.post_reply(b"world".to_vec()),
            "a server does not fail because the client vanished"
        );
        assert_eq!(link.take_reply(), None);
        assert!(!link.client_open());
    }

    #[test]
    fn polling_for_a_reply_does_not_cancel_the_wait() {
        let link = GuestSession::new();
        link.send(b"hello".to_vec()).unwrap();
        assert_eq!(
            link.take_reply(),
            None,
            "the readiness check runs before the reply exists"
        );
        assert_eq!(link.take_request().as_deref(), Some(&b"hello"[..]));
        assert!(
            link.post_reply(b"world".to_vec()),
            "a poll must not cancel the wait"
        );
        assert_eq!(link.take_reply().as_deref(), Some(&b"world"[..]));
    }

    #[test]
    fn a_withdrawn_request_is_not_served() {
        let link = GuestSession::new();
        link.send(b"hello".to_vec()).unwrap();
        link.withdraw();
        assert_eq!(link.take_request(), None, "the client stopped waiting");
        link.send(b"next".to_vec()).unwrap();
        link.take_request().unwrap();
        link.withdraw();
        assert!(
            !link.post_reply(b"late".to_vec()),
            "a client that stopped waiting gets no reply"
        );
        assert_eq!(link.take_reply(), None, "and no later request sees it");
    }

    #[test]
    fn a_host_service_serves_a_request_from_its_own_thread() {
        let link = GuestSession::new();
        let server = link.clone();
        let host = std::thread::spawn(move || {
            server.take_request_blocking(std::time::Duration::from_secs(5))
        });
        link.send(b"hello".to_vec()).unwrap();
        assert_eq!(host.join().unwrap().as_deref(), Some(&b"hello"[..]));
        assert!(link.in_service());
        assert!(link.post_reply(b"world".to_vec()));
        assert_eq!(link.take_reply().as_deref(), Some(&b"world"[..]));
    }

    #[test]
    fn a_closed_server_refuses_requests() {
        let link = GuestSession::new();
        link.close_server();
        assert_eq!(link.send(b"hello".to_vec()), Err(SendError::RemoteDead));
        assert!(!link.server_open());
    }

    #[test]
    fn a_closed_client_withdraws_a_queued_request() {
        let link = GuestSession::new();
        link.send(b"hello".to_vec()).unwrap();
        assert!(link.has_request());
        link.close_client();
        assert_eq!(
            link.take_request(),
            None,
            "the server must not serve a client that is gone"
        );
    }

    #[test]
    fn a_port_queues_and_drops_its_clients() {
        let first = GuestSession::new();
        let port = Port::new(0x3a6d73, false, first.clone());
        assert_eq!(port.pending(), 1);
        assert!(Arc::ptr_eq(&port.accept().unwrap(), &first));
        assert_eq!(port.pending(), 0);
        assert!(port.accept().is_none(), "no session waiting to be accepted");

        let second = GuestSession::new();
        let port = Port::new(0x3a6d73, false, second.clone());
        port.close();
        assert!(!second.server_open(), "a queued client loses its server");
        assert!(port.accept().is_none());
    }

    #[test]
    fn a_transfer_memory_aliases_the_creator_pages() {
        let memory = GuestMemory::new().unwrap();
        memory
            .map_as(
                0x2_0000_0000,
                0x2000,
                crate::mem::state::NORMAL,
                crate::mem::perm::RW,
            )
            .unwrap();
        assert!(memory.write(0x2_0000_0000, b"jit"));
        // svcCreateTransferMemory keeps the pages with the creator, retagged.
        assert!(memory.retag(
            0x2_0000_0000,
            0x2_0000_2000,
            crate::mem::state::TRANSFER,
            crate::mem::perm::NONE,
            &[crate::mem::state::NORMAL],
        ));
        let tmem = TransferMemory::new(0x2_0000_0000, 0x2000, crate::mem::perm::NONE);
        assert!(tmem.map(&memory, 0x2_0000_2000, crate::mem::perm::NONE));
        assert_eq!(
            memory.query(0x2_0000_2000).state,
            crate::mem::state::TRANSFER
        );
        assert_eq!(&memory.read(0x2_0000_2000, 3).unwrap(), b"jit");
        assert!(memory.write(0x2_0000_2000, b"NEW"));
        assert!(
            !tmem.map(&memory, 0x2_0000_4000, crate::mem::perm::NONE),
            "one mapping at a time"
        );
        assert!(tmem.unmap(&memory, 0x2_0000_2000));
        assert_eq!(
            &memory.read(0x2_0000_0000, 3).unwrap(),
            b"NEW",
            "the mapping writes back to the creator's pages"
        );
        assert_eq!(tmem.mapped_at(), None);
        assert!(tmem.map(&memory, 0x2_0000_2000, crate::mem::perm::NONE));
        assert_eq!(&memory.read(0x2_0000_2000, 3).unwrap(), b"NEW");
    }

    #[test]
    fn code_memory_maps_writable_then_executable() {
        let memory = GuestMemory::new().unwrap();
        memory
            .map_as(
                0x3_0000_0000,
                0x1000,
                crate::mem::state::HEAP,
                crate::mem::perm::RW,
            )
            .unwrap();
        let code = CodeMemory::new(0x1000);
        assert!(code.map_owner(&memory, 0x3_0000_1000, crate::mem::perm::RW));
        assert_eq!(
            memory.read(0x3_0000_1000, 4).unwrap(),
            vec![0xff; 4],
            "code memory starts as the kernel leaves it"
        );
        assert_eq!(
            memory.query(0x3_0000_1000).state,
            crate::mem::state::CODE_WRITABLE
        );
        assert!(memory.write(0x3_0000_1000, b"code"));
        assert!(code.map_owner(&memory, 0x3_0000_1000, crate::mem::perm::RX));
        assert_eq!(memory.query(0x3_0000_1000).perm, crate::mem::perm::RX);
        assert!(
            !code.map_owner(&memory, 0x3_0000_2000, crate::mem::perm::RW),
            "one mapping at a time"
        );
        assert!(code.unmap_owner(&memory, 0x3_0000_1000));
        assert!(!memory.is_mapped(0x3_0000_1000, 1));
        assert!(code.map_owner(&memory, 0x3_0000_2000, crate::mem::perm::R));
        assert_eq!(&memory.read(0x3_0000_2000, 4).unwrap(), b"code");
    }
}
