use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

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

/// `svcCreatePort`: the largest number of sessions a port accepts at once.
/// Horizon's `KPort` stores the ceiling in an `s32`, so `-1` means unlimited.
pub const MAX_PORT_SESSIONS: i32 = 0x3f;

/// Why a port refused a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectError {
    /// The port already holds `max_sessions` connections.
    Limit,
    /// The port was closed, so nobody can accept the connection.
    Closed,
}

/// A server port: `svcCreateSession`'s server handle, `svcCreatePort`'s server
/// end, or a port registered with `svcManageNamedPort`. Kernel ports are
/// signalable while a client connection waits to be accepted, which is what
/// `svcWaitSynchronization` and the port entries of `svcReplyAndReceive` use.
pub struct Port {
    /// The numeric tag `svcCreatePort` was given. A named port keeps zero here;
    /// its string name lives in the registry.
    pub name: u64,
    pub is_light: bool,
    /// Connections the port accepts at once; `-1` is unlimited. A port made by
    /// `svcCreateSession` carries exactly the pair it was made with, so its
    /// ceiling is one.
    max_sessions: i32,
    queue: Mutex<VecDeque<Arc<GuestSession>>>,
    /// Every pair connected through this port, so the ceiling counts the
    /// sessions the guest already accepted. Weak references: a port does not
    /// keep a session alive.
    connected: Mutex<Vec<Weak<GuestSession>>>,
    closed: AtomicBool,
    /// Wakes the process scheduler when a connection arrives or the port
    /// closes. Kept here (instead of taking it from a `Process`) so a pair made
    /// by `svcConnectToPort`/`svcConnectToNamedPort` blocks its client the same
    /// way `svcCreateSession`'s pair does.
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl Port {
    /// The server end of a `svcCreateSession` pair: the queue starts with the
    /// pair and the port accepts nothing else.
    pub fn new(name: u64, is_light: bool, first: Arc<GuestSession>) -> Arc<Port> {
        Arc::new(Port {
            name,
            is_light,
            max_sessions: 1,
            queue: Mutex::new(VecDeque::from([first.clone()])),
            connected: Mutex::new(vec![Arc::downgrade(&first)]),
            closed: AtomicBool::new(false),
            wake: Arc::new(|| {}),
        })
    }

    /// An empty port `svcCreatePort` or `svcManageNamedPort` made; connections
    /// arrive through [`Port::connect`].
    pub fn with_wake(
        name: u64,
        is_light: bool,
        max_sessions: i32,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Arc<Port> {
        Arc::new(Port {
            name,
            is_light,
            max_sessions,
            queue: Mutex::new(VecDeque::new()),
            connected: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
            wake: Arc::new(wake),
        })
    }

    /// One connection over the client end of this port. The pair is queued for
    /// `svcAcceptSession` and the server is woken.
    pub fn connect(&self) -> Result<Arc<GuestSession>, ConnectError> {
        if self.closed() {
            return Err(ConnectError::Closed);
        }
        let link = {
            let mut connected = self.connected.lock().unwrap();
            // A session stops counting when its server end is closed.
            connected.retain(|w| w.upgrade().is_some_and(|l| l.server_open()));
            if self.max_sessions >= 0 && connected.len() >= self.max_sessions as usize {
                return Err(ConnectError::Limit);
            }
            let wake = self.wake.clone();
            let link = GuestSession::with_wake(move || wake());
            connected.push(Arc::downgrade(&link));
            link
        };
        // Not while holding the connection lock or the queue lock: the wake
        // takes the scheduler lock, and a thread blocked on this port looks at
        // the queue while it holds that lock.
        let queued = {
            let mut queue = self.queue.lock().unwrap();
            // A close that started after the check above would otherwise drain
            // the queue without this pair and leave its client waiting forever.
            if self.closed() {
                false
            } else {
                queue.push_back(link.clone());
                true
            }
        };
        if !queued {
            link.close_server();
            return Err(ConnectError::Closed);
        }
        (self.wake)();
        Ok(link)
    }

    /// Whether a new connection would be accepted, which is when a client port
    /// is signalled.
    pub fn can_connect(&self) -> bool {
        if self.closed() {
            return false;
        }
        if self.max_sessions < 0 {
            return true;
        }
        let mut connected = self.connected.lock().unwrap();
        connected.retain(|w| w.upgrade().is_some_and(|l| l.server_open()));
        (connected.len() as i32) < self.max_sessions
    }

    /// Pops the next queued client. `None` means no connection is waiting, the
    /// documented `0xf201` of `svcAcceptSession`.
    pub fn accept(&self) -> Option<Arc<GuestSession>> {
        self.queue.lock().unwrap().pop_front()
    }

    pub fn pending(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    /// Closing the port drops the queued pairs, so a client blocked on one
    /// learns that no server can ever reply, and wakes a server waiting for a
    /// connection. The locks are released first: closing a pair wakes the
    /// scheduler, and a thread blocked on this port holds the scheduler lock
    /// while it looks at the queue.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        let queued: Vec<Arc<GuestSession>> = {
            let mut queue = self.queue.lock().unwrap();
            queue.drain(..).collect()
        };
        for link in queued {
            link.close_server();
        }
        (self.wake)();
    }
}

/// `svcCreatePort`'s client end. A client port is not a session: it can be given
/// to another process (or kept) and `svcConnectToPort` turns it into a client
/// session on the server port. It only then counts against
/// `LimitableResource_Sessions`.
pub struct ClientPort {
    port: Arc<Port>,
}

impl ClientPort {
    pub fn new(port: Arc<Port>) -> Arc<ClientPort> {
        Arc::new(ClientPort { port })
    }

    pub fn port(&self) -> &Arc<Port> {
        &self.port
    }

    /// Signalled while the server port can accept another connection.
    pub fn can_connect(&self) -> bool {
        self.port.can_connect()
    }
}

/// `svcManageNamedPort` refused a registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamedPortError {
    /// Another live port already owns the name.
    Taken,
    /// The runtime's named-port table is full.
    Full,
}

/// The live names the guest registered with `svcManageNamedPort`, which is the
/// guest-visible half of the `svcConnectToNamedPort` registry. `sm:` is
/// host-served and never enters this table.
///
/// The kernel keeps a name reserved while the server port exists, so the table
/// holds weak references: closing the port handle frees the name, exactly as
/// dropping the last `KPort` reference does on hardware.
static NAMED_PORTS: std::sync::OnceLock<Mutex<BTreeMap<String, Weak<Port>>>> =
    std::sync::OnceLock::new();

/// The runtime caps the guest named-port table at 0x40 live names. Horizon's
/// named-port table is a fixed list too; refusing past a bound keeps a guest
/// from growing an unbounded registry. `sm:` is host-served and never occupies
/// a slot here.
pub const MAX_NAMED_PORTS: usize = 0x40;

fn named_ports() -> &'static Mutex<BTreeMap<String, Weak<Port>>> {
    NAMED_PORTS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Drops the names whose port is gone and returns how many live ones remain.
fn live_named_ports(ports: &mut BTreeMap<String, Weak<Port>>) -> usize {
    ports.retain(|_, weak| weak.strong_count() > 0);
    ports.len()
}

/// Reserves `name` for `port`. `Err(Taken)` when a live port already owns it,
/// `Err(Full)` when the table has no free slot.
pub fn register_named_port(name: &str, port: &Arc<Port>) -> Result<(), NamedPortError> {
    let mut ports = named_ports().lock().unwrap();
    live_named_ports(&mut ports);
    if ports.contains_key(name) {
        return Err(NamedPortError::Taken);
    }
    if ports.len() >= MAX_NAMED_PORTS {
        return Err(NamedPortError::Full);
    }
    ports.insert(name.to_string(), Arc::downgrade(port));
    Ok(())
}

/// The live port registered under `name`, if any. A registration whose port is
/// gone is reaped on the way.
pub fn find_named_port(name: &str) -> Option<Arc<Port>> {
    let mut ports = named_ports().lock().unwrap();
    let weak = ports.get(name)?.clone();
    let port = weak.upgrade();
    if port.is_none() {
        ports.remove(name);
    }
    port
}

/// How many guest named ports are live (diagnostics and tests).
pub fn named_port_count() -> usize {
    let mut ports = named_ports().lock().unwrap();
    live_named_ports(&mut ports)
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

/// `MemoryPermission_DontCare`: `svcCreateSharedMemory` accepts it for the
/// remote side, and it lets the borrower map the block with any valid
/// permission.
pub const MEMORY_PERMISSION_DONT_CARE: u32 = 0x1000_0000;

/// Which side of a shared memory block a handle belongs to. Horizon keys the
/// mapping permission on the process: the creator maps with the block's
/// `owner_perm` and every other process maps with `remote_perm`. The runtime
/// runs one process, so the handle carries the role instead: the handle
/// `svcCreateSharedMemory` inserts is the creator's, and a handle the guest
/// received from another process (a host service, or a handle copy) is a
/// borrower's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedMemoryRole {
    Owner,
    Borrower,
}

/// Why `svcMapSharedMemory`/`svcUnmapSharedMemory` refused a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedMapError {
    /// The size is not the block's size.
    Size,
    /// The address is not page aligned or the range overflows.
    Address,
    /// The mapping permission is not the one the caller's side was created
    /// with, or is not a permission shared memory can have (no execute).
    Permission,
    /// The block is already mapped, the destination is taken, or the range was
    /// not mapped by this block.
    State,
}

/// A shared memory block (`svcCreateSharedMemory`, and the blocks the runtime's
/// own services share with the guest). The kernel object owns the pages; a
/// mapping commits guest pages and copies the block into them, and unmapping
/// copies them back, so the contents follow the object between mappings.
pub struct SharedMemory {
    pub size: u64,
    /// The permission the creating side maps with.
    owner_perm: u32,
    /// The permission the borrower maps with; `MemoryPermission_DontCare`
    /// accepts any valid one.
    remote_perm: u32,
    state: Mutex<SharedState>,
}

struct SharedState {
    shadow: Vec<u8>,
    mapped_at: Option<u64>,
}

impl SharedMemory {
    /// A block a host service shares with the guest: the service writes it with
    /// `RW` and the guest may map it with any valid permission, which is what
    /// `MemoryPermission_DontCare` means to the kernel.
    pub fn new(size: u64) -> SharedMemory {
        SharedMemory::with_permissions(size, crate::mem::perm::RW, MEMORY_PERMISSION_DONT_CARE)
    }

    pub fn with_permissions(size: u64, owner_perm: u32, remote_perm: u32) -> SharedMemory {
        SharedMemory {
            size,
            owner_perm,
            remote_perm,
            state: Mutex::new(SharedState {
                shadow: vec![0; size as usize],
                mapped_at: None,
            }),
        }
    }

    /// `svcCreateSharedMemory`: the shadow is a host allocation sized by the
    /// guest, so it is reserved explicitly instead of letting `vec![0; size]`
    /// abort the process when the host refuses it.
    pub fn try_new(size: u64, owner_perm: u32, remote_perm: u32) -> Option<SharedMemory> {
        let len = usize::try_from(size).ok()?;
        let mut shadow = Vec::new();
        shadow.try_reserve_exact(len).ok()?;
        shadow.resize(len, 0);
        Some(SharedMemory {
            size,
            owner_perm,
            remote_perm,
            state: Mutex::new(SharedState {
                shadow,
                mapped_at: None,
            }),
        })
    }

    pub fn owner_perm(&self) -> u32 {
        self.owner_perm
    }

    pub fn mapped_at(&self) -> Option<u64> {
        self.state.lock().unwrap().mapped_at
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

    /// `svcMapSharedMemory`. The size must be the block's size and the address
    /// page aligned and free; the permission must be the one the caller's side
    /// of the block was created with. The runtime keeps one mapping per block: a
    /// second mapping is refused until the first is unmapped.
    pub fn map(
        &self,
        memory: &crate::mem::GuestMemory,
        addr: u64,
        size: u64,
        perm: u32,
        role: SharedMemoryRole,
    ) -> Result<(), SharedMapError> {
        if size != self.size {
            return Err(SharedMapError::Size);
        }
        if !addr.is_multiple_of(crate::mem::PAGE)
            || addr
                .checked_add(size)
                .is_none_or(|end| end > crate::mem::ADDR_MASK + 1)
        {
            return Err(SharedMapError::Address);
        }
        // Shared memory is data; the kernel has no executable mapping for it.
        if !matches!(
            perm,
            crate::mem::perm::NONE | crate::mem::perm::R | crate::mem::perm::RW
        ) {
            return Err(SharedMapError::Permission);
        }
        let expected = match role {
            SharedMemoryRole::Owner => self.owner_perm,
            SharedMemoryRole::Borrower => self.remote_perm,
        };
        if expected != MEMORY_PERMISSION_DONT_CARE && perm != expected {
            return Err(SharedMapError::Permission);
        }
        let mut st = self.state.lock().unwrap();
        if st.mapped_at.is_some()
            || memory
                .map_as(addr, size, crate::mem::state::SHARED, perm)
                .is_err()
        {
            return Err(SharedMapError::State);
        }
        if !memory.write(addr, &st.shadow) {
            let _ = memory.unmap(addr, size);
            return Err(SharedMapError::State);
        }
        st.mapped_at = Some(addr);
        Ok(())
    }

    /// `svcUnmapSharedMemory`: copies the mapping back into the block and
    /// releases the guest pages.
    pub fn unmap(
        &self,
        memory: &crate::mem::GuestMemory,
        addr: u64,
        size: u64,
    ) -> Result<(), SharedMapError> {
        if size != self.size {
            return Err(SharedMapError::Size);
        }
        if !addr.is_multiple_of(crate::mem::PAGE) {
            return Err(SharedMapError::Address);
        }
        let mut st = self.state.lock().unwrap();
        if st.mapped_at != Some(addr) {
            return Err(SharedMapError::State);
        }
        let Some(bytes) = memory.read(addr, size) else {
            return Err(SharedMapError::State);
        };
        if memory.unmap(addr, size).is_err() {
            return Err(SharedMapError::State);
        }
        st.shadow = bytes;
        st.mapped_at = None;
        Ok(())
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
    /// A server port from `svcCreateSession`, `svcCreatePort` or
    /// `svcManageNamedPort`.
    Port(Arc<Port>),
    /// The client end of a `svcCreatePort` port, which `svcConnectToPort` turns
    /// into a client session.
    ClientPort(Arc<ClientPort>),
    Event(Arc<Event>),
    Thread(Arc<KThread>),
    TransferMemory(Arc<TransferMemory>),
    CodeMemory(Arc<CodeMemory>),
    SharedMemory(Arc<SharedMemory>),
    ResourceLimit(Arc<ResourceLimit>),
}

pub struct HandleTable {
    objects: HashMap<u32, KObject>,
    /// Role of each shared memory handle (see [`SharedMemoryRole`]).
    shared_roles: HashMap<u32, SharedMemoryRole>,
    next: u32,
    resource_limit: Option<u32>,
}

impl Default for HandleTable {
    fn default() -> Self {
        HandleTable {
            objects: HashMap::new(),
            shared_roles: HashMap::new(),
            next: 0x0002_0001,
            resource_limit: None,
        }
    }
}

impl HandleTable {
    pub fn insert(&mut self, object: KObject) -> u32 {
        let h = self.next;
        self.next += 1;
        // A shared memory handle the kernel did not insert from
        // `svcCreateSharedMemory` is a borrower's: the runtime's own services
        // create blocks for the guest, and a block copied from another process
        // arrives that way too. `insert_shared_memory` marks the creator's.
        if matches!(object, KObject::SharedMemory(_)) {
            self.shared_roles.insert(h, SharedMemoryRole::Borrower);
        }
        self.objects.insert(h, object);
        h
    }

    /// Inserts a shared memory handle with an explicit role.
    pub fn insert_shared_memory(
        &mut self,
        memory: Arc<SharedMemory>,
        role: SharedMemoryRole,
    ) -> u32 {
        let h = self.insert(KObject::SharedMemory(memory));
        self.shared_roles.insert(h, role);
        h
    }

    /// The block a shared memory handle refers to, with the role the handle
    /// plays in its ownership.
    pub fn shared_memory(&self, handle: u32) -> Option<(Arc<SharedMemory>, SharedMemoryRole)> {
        let Some(KObject::SharedMemory(m)) = self.objects.get(&handle) else {
            return None;
        };
        Some((
            m.clone(),
            self.shared_roles
                .get(&handle)
                .copied()
                .unwrap_or(SharedMemoryRole::Borrower),
        ))
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
        self.shared_roles.remove(&handle);
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

    pub fn client_port(&self, handle: u32) -> Option<Arc<ClientPort>> {
        match self.objects.get(&handle) {
            Some(KObject::ClientPort(p)) => Some(p.clone()),
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
    use std::sync::atomic::AtomicUsize;

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

    #[test]
    fn a_port_wakes_on_connect_and_stops_at_its_ceiling() {
        let woken = Arc::new(AtomicUsize::new(0));
        let counter = woken.clone();
        let port = Port::with_wake(7, false, 2, move || {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert!(port.can_connect());
        let first = port.connect().expect("the first connection fits");
        assert_eq!(
            woken.load(Ordering::Relaxed),
            1,
            "a connection wakes the waiting server"
        );
        let second = port.connect().expect("the second connection fits");
        assert_eq!(port.pending(), 2);
        assert_eq!(port.connect().err(), Some(ConnectError::Limit));
        assert!(!port.can_connect(), "the ceiling is reached");
        assert!(port.accept().is_some_and(|l| Arc::ptr_eq(&l, &first)));
        assert_eq!(
            port.connect().err(),
            Some(ConnectError::Limit),
            "an accepted session still counts against max_sessions"
        );
        first.close_server();
        let third = port
            .connect()
            .expect("closing the server end frees the slot");
        assert_eq!(port.pending(), 2);

        let before_close = woken.load(Ordering::Relaxed);
        port.close();
        assert!(port.closed());
        assert!(!second.server_open(), "a queued client loses its server");
        assert!(!third.server_open());
        assert!(port.accept().is_none());
        assert_eq!(port.connect().err(), Some(ConnectError::Closed));
        assert!(!port.can_connect());
        assert!(
            woken.load(Ordering::Relaxed) > before_close,
            "closing a port with waiters wakes the scheduler"
        );
    }

    #[test]
    fn named_ports_reserve_the_name_until_the_port_closes() {
        let one = Port::with_wake(0, false, 1, || {});
        let two = Port::with_wake(0, false, 1, || {});
        assert_eq!(register_named_port("ws9:kernel-one", &one), Ok(()));
        assert!(Arc::ptr_eq(
            &find_named_port("ws9:kernel-one").expect("the name resolves"),
            &one
        ));
        assert_eq!(
            register_named_port("ws9:kernel-one", &two),
            Err(NamedPortError::Taken)
        );
        assert_eq!(register_named_port("ws9:kernel-two", &two), Ok(()));
        assert!(named_port_count() >= 2, "both names are live");
        drop(one);
        assert!(
            find_named_port("ws9:kernel-one").is_none(),
            "the name dies with the port"
        );
        assert_eq!(
            register_named_port("ws9:kernel-one", &two),
            Ok(()),
            "the freed name can be taken again"
        );
        drop(two);
        assert!(find_named_port("ws9:kernel-one").is_none());
        assert!(find_named_port("ws9:kernel-two").is_none());
        assert!(find_named_port("ws9:nonexistent").is_none());
    }

    #[test]
    fn shared_memory_maps_by_role_and_keeps_its_contents() {
        let memory = GuestMemory::new().unwrap();
        let base = 0x2_0000_0000;
        let block =
            SharedMemory::with_permissions(0x2000, crate::mem::perm::RW, crate::mem::perm::R);
        assert_eq!(
            block.map(
                &memory,
                base,
                0x1000,
                crate::mem::perm::RW,
                SharedMemoryRole::Owner
            ),
            Err(SharedMapError::Size)
        );
        assert_eq!(
            block.map(
                &memory,
                base,
                0x2000,
                crate::mem::perm::R,
                SharedMemoryRole::Owner
            ),
            Err(SharedMapError::Permission),
            "the creator maps with its own permission"
        );
        assert_eq!(
            block.map(
                &memory,
                base,
                0x2000,
                crate::mem::perm::RW,
                SharedMemoryRole::Borrower
            ),
            Err(SharedMapError::Permission),
            "the borrower maps with the remote permission"
        );
        assert_eq!(
            block.map(
                &memory,
                base + 1,
                0x2000,
                crate::mem::perm::RW,
                SharedMemoryRole::Owner
            ),
            Err(SharedMapError::Address)
        );
        assert_eq!(
            block.map(
                &memory,
                base,
                0x2000,
                crate::mem::perm::RX,
                SharedMemoryRole::Owner
            ),
            Err(SharedMapError::Permission),
            "shared memory cannot be executable"
        );
        assert_eq!(
            block.map(
                &memory,
                base,
                0x2000,
                crate::mem::perm::RW,
                SharedMemoryRole::Owner
            ),
            Ok(())
        );
        assert_eq!(block.mapped_at(), Some(base));
        assert!(memory.write(base, b"shared"));
        assert_eq!(
            block.map(
                &memory,
                base + 0x2000,
                0x2000,
                crate::mem::perm::RW,
                SharedMemoryRole::Owner
            ),
            Err(SharedMapError::State),
            "one mapping at a time"
        );
        assert_eq!(
            block.unmap(&memory, base, 0x1000),
            Err(SharedMapError::Size)
        );
        assert_eq!(
            block.unmap(&memory, base + 0x2000, 0x2000),
            Err(SharedMapError::State)
        );
        assert_eq!(block.unmap(&memory, base, 0x2000), Ok(()));
        assert_eq!(&block.read(&memory, 0, 6)[..], b"shared");
        assert_eq!(
            block.map(
                &memory,
                base,
                0x2000,
                crate::mem::perm::R,
                SharedMemoryRole::Borrower
            ),
            Ok(())
        );
        assert_eq!(&memory.read(base, 6).unwrap()[..], b"shared");
        assert_eq!(block.unmap(&memory, base, 0x2000), Ok(()));

        // MemoryPermission_DontCare lets the borrower choose.
        let block = SharedMemory::new(0x1000);
        assert_eq!(block.owner_perm(), crate::mem::perm::RW);
        assert_eq!(
            block.map(
                &memory,
                base,
                0x1000,
                crate::mem::perm::NONE,
                SharedMemoryRole::Borrower
            ),
            Ok(())
        );
        assert_eq!(block.unmap(&memory, base, 0x1000), Ok(()));
    }
}
