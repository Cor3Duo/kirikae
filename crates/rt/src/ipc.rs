use std::sync::Arc;

use crate::cpu::Cpu;
use crate::kernel::{GuestSession, KObject, SendError, Session};
use crate::services::{Buf, Reply, Request, Service};

pub const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
pub const SFCO: u32 = u32::from_le_bytes(*b"SFCO");

pub const TYPE_CLOSE: u16 = 2;
pub const TYPE_REQUEST: u16 = 4;
pub const TYPE_CONTROL: u16 = 5;
pub const TYPE_REQUEST_CTX: u16 = 6;
pub const TYPE_CONTROL_CTX: u16 = 7;
pub const TYPE_TIPC_BASE: u16 = 16;

const DOMAIN_SEND: u8 = 1;
const DOMAIN_CLOSE: u8 = 2;

pub const RESULT_INVALID_HANDLE: u64 = 0xe401;
const RESULT_BAD_MESSAGE: u32 = 0x1a80a;
/// The pointer buffer every session advertises (`QueryPointerBufferSize`). A
/// type-0x9 buffer that does not fit is sent as a 0x5/0x6 map descriptor.
pub const POINTER_BUFFER_SIZE: u16 = 0x8000;

/// CMIF manager commands (`InvokeManagerMethod`).
pub const CONTROL_CONVERT_TO_DOMAIN: u32 = 0;
pub const CONTROL_COPY_FROM_DOMAIN: u32 = 1;
pub const CONTROL_CLONE_CURRENT_OBJECT: u32 = 2;
pub const CONTROL_QUERY_POINTER_BUFFER_SIZE: u32 = 3;
pub const CONTROL_CLONE_CURRENT_OBJECT_EX: u32 = 4;

#[derive(Clone, Debug, Default)]
pub struct Hipc {
    pub kind: u16,
    pub pid: Option<u64>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    pub x: Vec<Buf>,
    pub a: Vec<Buf>,
    pub b: Vec<Buf>,
    pub w: Vec<Buf>,
    pub c: Vec<Buf>,
    pub data_offset: usize,
    pub data: Vec<u8>,
    /// Words of raw data, used to resolve an inline receive list (mode 1).
    pub raw_words: usize,
    /// `ReceiveListCount` and `ReceiveListOffset` from `Header1`.
    pub receive_list_count: u8,
    pub receive_list_offset: usize,
}

impl Hipc {
    /// How many bytes of pointer buffer this message asks the server to
    /// provide: the sum of the type-0x9 descriptor sizes. The client refuses a
    /// request that exceeds [`POINTER_BUFFER_SIZE`] before sending it (0x11a0b).
    pub fn pointer_buffer_usage(&self) -> u64 {
        self.x.iter().map(|b| b.size).sum()
    }

    /// Where a mode-1 (inline) receive list lives, relative to the start of the
    /// HIPC message buffer. The address of the buffer itself is the caller's
    /// `tpidrro_el0`, which `parse_hipc` does not know.
    pub fn inline_receive_offset(&self) -> Option<u64> {
        (self.receive_list_count == 1)
            .then(|| ((self.raw_words * 4 + self.receive_list_offset) as u64).next_multiple_of(16))
    }
}

fn word(msg: &[u8], i: usize) -> Result<u32, &'static str> {
    msg.get(i * 4..i * 4 + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()))
        .ok_or("message is truncated")
}

/// Decodes a type-0x9 pointer descriptor: 32-bit low address, 32-to-35 address
/// bits in bits 12-15, 36-to-38 in bits 6-8 and a 16-bit size.
pub fn decode_pointer(words: [u32; 2]) -> Buf {
    let (d0, d1) = (words[0], words[1]);
    Buf {
        addr: d1 as u64 | (((d0 >> 12) & 0xf) as u64) << 32 | (((d0 >> 6) & 0x7) as u64) << 36,
        size: (d0 >> 16) as u64,
    }
}

pub fn encode_pointer(buf: Buf) -> [u32; 2] {
    let addr = buf.addr;
    let d0 = (((addr >> 32) as u32 & 0xf) << 12)
        | (((addr >> 36) as u32 & 0x7) << 6)
        | ((buf.size as u32 & 0xffff) << 16);
    [d0, addr as u32]
}

/// Decodes a map descriptor (send, receive or exchange): a 36-bit size and a
/// 39-bit address.
pub fn decode_map(words: [u32; 3]) -> Buf {
    let (d0, d1, d2) = (words[0], words[1], words[2]);
    Buf {
        addr: d1 as u64 | (((d2 >> 28) & 0xf) as u64) << 32 | (((d2 >> 2) & 0x7) as u64) << 36,
        size: d0 as u64 | (((d2 >> 24) & 0xf) as u64) << 32,
    }
}

pub fn encode_map(buf: Buf) -> [u32; 3] {
    let (addr, size) = (buf.addr, buf.size);
    let d2 = (((addr >> 36) as u32 & 0x7) << 2)
        | (((size >> 32) as u32 & 0xf) << 24)
        | (((addr >> 32) as u32 & 0xf) << 28);
    [size as u32, addr as u32, d2]
}

/// Decodes a receive-list descriptor: low address in word 0, address bits
/// 32-38 in bits 0-6 of word 1 and a 16-bit size in bits 16-31.
pub fn decode_receive_list(words: [u32; 2]) -> Buf {
    let (d0, d1) = (words[0], words[1]);
    Buf {
        addr: d0 as u64 | ((d1 & 0x7f) as u64) << 32,
        size: (d1 >> 16) as u64,
    }
}

pub fn encode_receive_list(buf: Buf) -> [u32; 2] {
    [
        buf.addr as u32,
        ((buf.addr >> 32) as u32 & 0x7f) | ((buf.size as u32 & 0xffff) << 16),
    ]
}

pub fn parse_hipc(msg: &[u8]) -> Result<Hipc, &'static str> {
    let (h0, h1) = (word(msg, 0)?, word(msg, 1)?);
    let mut m = Hipc {
        kind: h0 as u16,
        ..Hipc::default()
    };
    let (nx, na, nb, nw) = (
        (h0 >> 16) & 0xf,
        (h0 >> 20) & 0xf,
        (h0 >> 24) & 0xf,
        (h0 >> 28) & 0xf,
    );
    let ndata = (h1 & 0x3ff) as usize;
    let recv_mode = (h1 >> 10) & 0xf;
    let recv_offset = ((h1 >> 20) & 0x7ff) as usize;
    let mut i = 2;
    if h1 >> 31 == 1 {
        let sh = word(msg, i)?;
        i += 1;
        if sh & 1 == 1 {
            m.pid = Some(word(msg, i)? as u64 | (word(msg, i + 1)? as u64) << 32);
            i += 2;
        }
        for _ in 0..(sh >> 1) & 0xf {
            m.copy_handles.push(word(msg, i)?);
            i += 1;
        }
        for _ in 0..(sh >> 5) & 0xf {
            m.move_handles.push(word(msg, i)?);
            i += 1;
        }
    }
    for _ in 0..nx {
        let descriptor = [word(msg, i)?, word(msg, i + 1)?];
        m.x.push(decode_pointer(descriptor));
        i += 2;
    }
    for (count, list) in [(na, &mut m.a), (nb, &mut m.b), (nw, &mut m.w)] {
        for _ in 0..count {
            let descriptor = [word(msg, i)?, word(msg, i + 1)?, word(msg, i + 2)?];
            list.push(decode_map(descriptor));
            i += 3;
        }
    }
    m.data_offset = i * 4;
    m.raw_words = ndata;
    m.receive_list_count = recv_mode as u8;
    m.receive_list_offset = recv_offset;
    m.data = msg
        .get(i * 4..(i + ndata) * 4)
        .ok_or("data words run past the message")?
        .to_vec();
    // `ReceiveListCount` 1 is an inline buffer, not a descriptor array; a
    // caller that knows the message base resolves it with
    // `Hipc::inline_receive_offset`. Counts 2 and above come as descriptors.
    if recv_mode >= 2 {
        let count = if recv_mode == 2 {
            1
        } else {
            recv_mode as usize - 2
        };
        let mut j = if recv_offset != 0 {
            recv_offset
        } else {
            i + ndata
        };
        for _ in 0..count {
            let descriptor = [word(msg, j)?, word(msg, j + 1)?];
            m.c.push(decode_receive_list(descriptor));
            j += 2;
        }
    }
    Ok(m)
}

#[derive(Debug, PartialEq, Eq)]
pub struct Cmif {
    pub domain: Option<(u8, u32, Vec<u32>)>,
    pub cmd: u32,
    pub args: Vec<u8>,
}

fn u32_in(b: &[u8], off: usize) -> Result<u32, &'static str> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap_or_default()))
        .ok_or("CMIF payload is truncated")
}

pub fn parse_cmif(m: &Hipc, domain: bool) -> Result<Cmif, &'static str> {
    let pad = m.data_offset.next_multiple_of(16) - m.data_offset;
    let mut p = m.data.get(pad..).ok_or("CMIF payload is missing")?;
    let mut dom = None;
    if domain {
        let hdr = p.get(..16).ok_or("domain header is truncated")?;
        let (kind, nin, size) = (
            hdr[0],
            hdr[1] as usize,
            u16::from_le_bytes([hdr[2], hdr[3]]) as usize,
        );
        let object = u32_in(hdr, 4)?;
        if kind == DOMAIN_CLOSE {
            return Ok(Cmif {
                domain: Some((kind, object, Vec::new())),
                cmd: 0,
                args: Vec::new(),
            });
        }
        let objects = (0..nin)
            .map(|k| u32_in(p, 16 + size + 4 * k))
            .collect::<Result<Vec<_>, _>>()?;
        dom = Some((kind, object, objects));
        p = p.get(16..16 + size).ok_or("domain payload is truncated")?;
    }
    if u32_in(p, 0)? != SFCI {
        return Err("missing SFCI magic");
    }
    Ok(Cmif {
        domain: dom,
        cmd: u32_in(p, 8)?,
        args: p.get(16..).unwrap_or(&[]).to_vec(),
    })
}

/// A TIPC request: the command is the message type minus the 16 TIPC types, and
/// the raw data has no CMIF header in front of it.
#[derive(Debug, PartialEq, Eq)]
pub struct Tipc {
    pub cmd: u32,
    pub args: Vec<u8>,
}

pub fn parse_tipc(m: &Hipc) -> Result<Tipc, &'static str> {
    let kind = m.kind;
    if kind < TYPE_TIPC_BASE {
        return Err("not a TIPC message");
    }
    Ok(Tipc {
        cmd: (kind - TYPE_TIPC_BASE) as u32,
        args: m.data.clone(),
    })
}

fn push_words(out: &mut Vec<u8>, words: &[u32]) {
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

fn frame(copy: &[u32], moves: &[u32], data_words: usize) -> Vec<u8> {
    let special = !copy.is_empty() || !moves.is_empty();
    let mut out = Vec::with_capacity(0x100);
    push_words(&mut out, &[0, data_words as u32 | (special as u32) << 31]);
    if special {
        push_words(
            &mut out,
            &[(copy.len() as u32) << 1 | (moves.len() as u32) << 5],
        );
        push_words(&mut out, copy);
        push_words(&mut out, moves);
    }
    out
}

pub fn encode_cmif(
    result: u32,
    data: &[u8],
    copy: &[u32],
    moves: &[u32],
    domain_objects: Option<&[u32]>,
) -> Vec<u8> {
    let mut payload = Vec::new();
    if let Some(ids) = domain_objects {
        push_words(&mut payload, &[ids.len() as u32, 0, 0, 0]);
    }
    push_words(&mut payload, &[SFCO, 0, result, 0]);
    payload.extend_from_slice(data);
    while !payload.len().is_multiple_of(4) {
        payload.push(0);
    }
    if let Some(ids) = domain_objects {
        push_words(&mut payload, ids);
    }
    let data_words = (16 + payload.len()) / 4;
    let mut out = frame(copy, moves, data_words);
    let data_start = out.len();
    out.resize(data_start.next_multiple_of(16), 0);
    out.extend_from_slice(&payload);
    out.resize(data_start + data_words * 4, 0);
    out
}

pub fn encode_tipc(result: u32, data: &[u8], copy: &[u32], moves: &[u32]) -> Vec<u8> {
    let data_words = 1 + data.len().div_ceil(4);
    let mut out = frame(copy, moves, data_words);
    push_words(&mut out, &[result]);
    out.extend_from_slice(data);
    let end = out.len().next_multiple_of(4);
    out.resize(end, 0);
    out
}

fn request(m: &Hipc, cmd: u32, args: Vec<u8>, objects: Vec<u32>, tipc: bool) -> Request {
    Request {
        cmd,
        data: args,
        pid: m.pid,
        copy_handles: m.copy_handles.clone(),
        move_handles: m.move_handles.clone(),
        x: m.x.clone(),
        a: m.a.clone(),
        b: m.b.clone(),
        w: m.w.clone(),
        c: m.c.clone(),
        objects,
        tipc,
    }
}

/// A copy of the session's object table.
///
/// `CloneCurrentObject` (manager command 2) hands the same session to another
/// client of the process. Horizon's server object, and with it the domain map,
/// is shared: object ids created through one handle are valid through the
/// other, which is what the SDK relies on when it clones `fsp-srv` and then
/// opens storages. The runtime therefore shares the `Session` and lets the
/// handle references keep the object table alive; closing a handle drops only
/// that handle's reference, and the last one drops the objects.
fn clone_session(session: &Arc<Session>) -> Arc<Session> {
    session.clone()
}

fn new_session(c: &Cpu, object: Arc<dyn Service>) -> u32 {
    c.process()
        .handles
        .lock()
        .unwrap()
        .insert(KObject::Session(Session::new(object)))
}

fn call(
    c: &mut Cpu,
    object: &Arc<dyn Service>,
    req: &Request,
    domain: Option<&Arc<Session>>,
    kind: u16,
) -> (Reply, Vec<u32>) {
    let p = c.process();
    let name = object.name();
    if kind >= TYPE_TIPC_BASE {
        p.log_ipc(format!("{name} tipc type {kind} cmd {}", req.cmd));
    } else {
        p.log_ipc(format!("{name} cmd {}", req.cmd));
    }
    crate::usage::begin_ipc();
    let mut reply = object.call(c, req);
    crate::usage::end_ipc(c, &name, req);
    let objects = std::mem::take(&mut reply.objects);
    let mut ids = Vec::new();
    match domain {
        Some(session) => ids = objects.into_iter().map(|o| session.domain_add(o)).collect(),
        None => {
            let mut handles: Vec<u32> = objects.into_iter().map(|o| new_session(c, o)).collect();
            handles.append(&mut reply.move_handles);
            reply.move_handles = handles;
        }
    }
    (reply, ids)
}

fn control(c: &mut Cpu, session: &Arc<Session>, cmd: u32, args: &[u8], kind: u16) -> Reply {
    let p = c.process();
    p.log_ipc(format!("control cmd {cmd} (type {kind})"));
    match cmd {
        CONTROL_CONVERT_TO_DOMAIN => Reply::ok().u32(session.convert_to_domain()),
        CONTROL_COPY_FROM_DOMAIN => {
            let id = args
                .get(..4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap_or_default()));
            match session.domain_object(id) {
                Some(o) => Reply::ok().move_handle(new_session(c, o)),
                None => Reply::err(RESULT_BAD_MESSAGE),
            }
        }
        CONTROL_CLONE_CURRENT_OBJECT | CONTROL_CLONE_CURRENT_OBJECT_EX => {
            let h = p
                .handles
                .lock()
                .unwrap()
                .insert(KObject::Session(clone_session(session)));
            Reply::ok().move_handle(h)
        }
        CONTROL_QUERY_POINTER_BUFFER_SIZE => {
            let size = session
                .object()
                .map_or(POINTER_BUFFER_SIZE, |o| o.pointer_buffer_size());
            Reply::ok().bytes(&size.to_le_bytes())
        }
        _ => Reply::err(RESULT_BAD_MESSAGE),
    }
}

pub fn send_sync_request(c: &mut Cpu, handle: u32, addr: u64, size: u64) -> u64 {
    let _prof = crate::prof::enter(crate::prof::Cat::Ipc);
    let p = c.process();
    let Some(session) = p.handles.lock().unwrap().session(handle) else {
        p.warn_once(format!(
            "SendSyncRequest on invalid handle {handle:#x} from {}",
            p.describe(c.x[30])
        ));
        return RESULT_INVALID_HANDLE;
    };
    if session.guest_server_link().is_some() {
        p.warn_once(format!(
            "SendSyncRequest on the server end of a guest session from {}",
            p.describe(c.x[30])
        ));
        return RESULT_INVALID_HANDLE;
    }
    let Some(msg) = p.memory.read(addr, size) else {
        return crate::horizon::RESULT_INVALID_POINTER;
    };
    if let Some(link) = session.guest_client_link() {
        return send_to_guest_server(c, &link, addr, size, msg);
    }
    let response = match respond(c, handle, &session, &msg) {
        Ok(r) => r,
        Err(why) => {
            p.warn_once(format!(
                "malformed IPC message ({why}) from {}",
                p.describe(c.x[30])
            ));
            encode_cmif(RESULT_BAD_MESSAGE, &[], &[], &[], None)
        }
    };
    if response.len() as u64 > size {
        p.warn_once(format!(
            "IPC response of {} bytes does not fit the {size}-byte message buffer",
            response.len()
        ));
        return RESULT_INVALID_HANDLE;
    }
    p.memory.write(addr, &response);
    0
}

/// `svcSendSyncRequest` on the client end of a guest session pair. The request
/// goes to the rendezvous ([`GuestSession`]) and the caller blocks until the
/// server replies, the server end is closed (`0xf601`) or the thread is
/// cancelled (`0xec01`). A host server uses the same three calls
/// (`take_request`, `post_reply`) from its own thread.
fn send_to_guest_server(
    c: &mut Cpu,
    link: &GuestSession,
    addr: u64,
    size: u64,
    msg: Vec<u8>,
) -> u64 {
    use crate::horizon::{
        RESULT_BUSY, RESULT_CANCELLED, RESULT_INVALID_POINTER, RESULT_REMOTE_DEAD,
    };
    use std::sync::atomic::Ordering;

    let p = c.process();
    match link.send(msg) {
        Ok(()) => {}
        Err(SendError::Busy) => return RESULT_BUSY,
        Err(SendError::RemoteDead) => return RESULT_REMOTE_DEAD,
    }
    let me = c.thread.clone();
    let what = || {
        format!(
            "SendSyncRequest on a guest session from {}",
            p.describe(c.x[30])
        )
    };
    let ended = crate::sched::block(c, -1, what, |_| {
        if me.cancelled.swap(false, Ordering::Relaxed) {
            return Some(Err(RESULT_CANCELLED));
        }
        if let Some(reply) = link.take_reply() {
            return Some(Ok(reply));
        }
        if !link.server_open() {
            return Some(Err(RESULT_REMOTE_DEAD));
        }
        None
    });
    match ended {
        Some(Ok(reply)) => {
            if reply.len() as u64 > size {
                p.warn_once(format!(
                    "a guest server replied with {} bytes into a {size}-byte buffer",
                    reply.len()
                ));
                return RESULT_INVALID_POINTER;
            }
            if p.memory.write(addr, &reply) {
                0
            } else {
                RESULT_INVALID_POINTER
            }
        }
        Some(Err(result)) => {
            link.withdraw();
            result
        }
        None => {
            link.withdraw();
            RESULT_CANCELLED
        }
    }
}

fn respond(
    c: &mut Cpu,
    handle: u32,
    session: &Arc<Session>,
    msg: &[u8],
) -> Result<Vec<u8>, &'static str> {
    let m = parse_hipc(msg)?;
    match m.kind {
        TYPE_CLOSE => {
            // The client released the session. Dropping the handle drops the
            // session's object table; closing the guest end wakes a peer that
            // is blocked in a request/reply exchange.
            c.process().handles.lock().unwrap().close(handle);
            if session.is_guest() {
                session.close_guest_end();
            }
            Ok(encode_cmif(0, &[], &[], &[], None))
        }
        k if k >= TYPE_TIPC_BASE => {
            let object = session.object().ok_or("session has no object")?;
            let tipc = parse_tipc(&m)?;
            let req = request(&m, tipc.cmd, tipc.args, Vec::new(), true);
            let (r, _) = call(c, &object, &req, None, k);
            Ok(encode_tipc(
                r.result,
                &r.data,
                &r.copy_handles,
                &r.move_handles,
            ))
        }
        TYPE_CONTROL | TYPE_CONTROL_CTX => {
            let cmif = parse_cmif(&m, false)?;
            let r = control(c, session, cmif.cmd, &cmif.args, m.kind);
            Ok(encode_cmif(
                r.result,
                &r.data,
                &r.copy_handles,
                &r.move_handles,
                None,
            ))
        }
        TYPE_REQUEST | TYPE_REQUEST_CTX => {
            let domain = session.is_domain();
            let cmif = parse_cmif(&m, domain)?;
            match cmif.domain {
                None => {
                    let object = session.object().ok_or("session has no object")?;
                    let (r, _) = call(
                        c,
                        &object,
                        &request(&m, cmif.cmd, cmif.args, Vec::new(), false),
                        None,
                        m.kind,
                    );
                    Ok(encode_cmif(
                        r.result,
                        &r.data,
                        &r.copy_handles,
                        &r.move_handles,
                        None,
                    ))
                }
                Some((DOMAIN_CLOSE, id, _)) => {
                    session.domain_remove(id);
                    Ok(encode_cmif(0, &[], &[], &[], Some(&[])))
                }
                Some((DOMAIN_SEND, id, objects)) => {
                    let Some(object) = session.domain_object(id) else {
                        c.process()
                            .warn_once(format!("domain request to unknown object {id}"));
                        return Ok(encode_cmif(RESULT_BAD_MESSAGE, &[], &[], &[], Some(&[])));
                    };
                    let (r, ids) = call(
                        c,
                        &object,
                        &request(&m, cmif.cmd, cmif.args, objects, false),
                        Some(session),
                        m.kind,
                    );
                    Ok(encode_cmif(
                        r.result,
                        &r.data,
                        &r.copy_handles,
                        &r.move_handles,
                        Some(&ids),
                    ))
                }
                Some(_) => Err("unknown domain message type"),
            }
        }
        _ => Err("unsupported message type"),
    }
}

#[cfg(test)]
pub mod client {
    use super::*;

    pub fn cmif(
        kind: u16,
        cmd: u32,
        args: &[u8],
        domain_object: Option<u32>,
        pid: bool,
        copy: &[u32],
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        let mut body = Vec::new();
        push_words(&mut body, &[SFCI, 1, cmd, 0]);
        body.extend_from_slice(args);
        if let Some(id) = domain_object {
            payload.push(DOMAIN_SEND);
            payload.push(0);
            payload.extend_from_slice(&(body.len() as u16).to_le_bytes());
            push_words(&mut payload, &[id, 0, 0]);
        }
        payload.extend_from_slice(&body);
        let data_words = (16 + payload.len()).div_ceil(4);
        let special = pid || !copy.is_empty();
        let mut out = Vec::new();
        push_words(
            &mut out,
            &[kind as u32, data_words as u32 | (special as u32) << 31],
        );
        if special {
            push_words(&mut out, &[pid as u32 | (copy.len() as u32) << 1]);
            if pid {
                push_words(&mut out, &[0, 0]);
            }
            push_words(&mut out, copy);
        }
        let start = out.len();
        out.resize(start.next_multiple_of(16), 0);
        out.extend_from_slice(&payload);
        out.resize(start + data_words * 4, 0);
        out
    }

    pub fn cmif_response(msg: &[u8], domain: bool) -> (u32, Vec<u8>, Hipc) {
        let m = parse_hipc(msg).unwrap();
        let pad = m.data_offset.next_multiple_of(16) - m.data_offset;
        let mut p = &m.data[pad..];
        if domain {
            p = &p[16..];
        }
        assert_eq!(u32::from_le_bytes(p[0..4].try_into().unwrap()), SFCO);
        let result = u32::from_le_bytes(p[8..12].try_into().unwrap());
        (result, p[16..].to_vec(), m)
    }

    /// A TIPC request: message type 16 + `cmd` and the raw payload right after
    /// the header (no CMIF structure).
    pub fn tipc(cmd: u32, args: &[u8], copy: &[u32]) -> Vec<u8> {
        let data_words = args.len().div_ceil(4);
        let special = !copy.is_empty();
        let mut out = Vec::new();
        push_words(
            &mut out,
            &[
                TYPE_TIPC_BASE as u32 + cmd,
                data_words as u32 | (special as u32) << 31,
            ],
        );
        if special {
            push_words(&mut out, &[(copy.len() as u32) << 1]);
            push_words(&mut out, copy);
        }
        out.extend_from_slice(args);
        out.resize(out.len().next_multiple_of(4), 0);
        out
    }

    pub fn tipc_response(msg: &[u8]) -> (u32, Vec<u8>, Hipc) {
        let m = parse_hipc(msg).unwrap();
        let result = m
            .data
            .get(..4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .unwrap();
        (result, m.data[4..].to_vec(), m)
    }

    /// A domain release (request kind 2) for `object`.
    pub fn domain_close(object: u32) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.push(DOMAIN_CLOSE);
        payload.push(0);
        payload.extend_from_slice(&0u16.to_le_bytes());
        push_words(&mut payload, &[object, 0, 0]);
        let data_words = (8 + payload.len()) / 4;
        let mut out = Vec::new();
        push_words(&mut out, &[TYPE_REQUEST as u32, data_words as u32]);
        // The CMIF payload starts on the next 16-byte boundary.
        out.resize(16, 0);
        out.extend_from_slice(&payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::client;
    use super::*;

    #[test]
    fn request_round_trip() {
        let msg = client::cmif(
            TYPE_REQUEST,
            1,
            &0x3a6d73u64.to_le_bytes(),
            None,
            true,
            &[0x1234],
        );
        let m = parse_hipc(&msg).unwrap();
        assert_eq!(m.kind, TYPE_REQUEST);
        assert_eq!(m.pid, Some(0));
        assert_eq!(m.copy_handles, vec![0x1234]);
        let cmif = parse_cmif(&m, false).unwrap();
        assert_eq!(cmif.cmd, 1);
        assert_eq!(&cmif.args[..8], &0x3a6d73u64.to_le_bytes());
    }

    #[test]
    fn domain_request() {
        let msg = client::cmif(TYPE_REQUEST, 7, &[1, 2, 3, 4], Some(5), false, &[]);
        let cmif = parse_cmif(&parse_hipc(&msg).unwrap(), true).unwrap();
        assert_eq!(cmif.domain, Some((DOMAIN_SEND, 5, vec![])));
        assert_eq!((cmif.cmd, &cmif.args[..4]), (7, &[1u8, 2, 3, 4][..]));
    }

    #[test]
    fn responses() {
        let r = encode_cmif(0, &42u32.to_le_bytes(), &[], &[0x20005], None);
        let (result, data, m) = client::cmif_response(&r, false);
        assert_eq!((result, &data[..4]), (0, &42u32.to_le_bytes()[..]));
        assert_eq!(m.move_handles, vec![0x20005]);

        let r = encode_cmif(0xe15, &[], &[], &[], Some(&[3]));
        let (result, data, _) = client::cmif_response(&r, true);
        assert_eq!(result, 0xe15);
        assert_eq!(&data[..4], &3u32.to_le_bytes());

        let r = encode_tipc(0, &[], &[], &[0x20009]);
        let m = parse_hipc(&r).unwrap();
        assert_eq!(
            (m.move_handles.clone(), &m.data[..4]),
            (vec![0x20009], &[0u8; 4][..])
        );
    }

    #[test]
    fn buffer_descriptors() {
        let x0 = 0x40u32 << 16 | 1 << 12;
        let a2 = 1u32 << 28;
        let words = [0x0011_0004u32, 0, x0, 0x2345_6780, 0x100, 0x8000_0000, a2];
        let msg: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = parse_hipc(&msg).unwrap();
        assert_eq!(
            m.x,
            vec![Buf {
                addr: 0x1_2345_6780,
                size: 0x40
            }]
        );
        assert_eq!(
            m.a,
            vec![Buf {
                addr: 0x1_8000_0000,
                size: 0x100
            }]
        );
    }

    #[test]
    fn transfer_descriptors_round_trip_sizes_and_high_bits() {
        // Every descriptor keeps its full address and size, including the bits
        // above 32 that HIPC packs into the tag words.
        for buf in [
            Buf { addr: 0, size: 0 },
            Buf {
                addr: 0xffff_ffff,
                size: 0xffff,
            },
            Buf {
                addr: 0x7f_1234_5678,
                size: 0xf_ffff_ffff,
            },
            Buf {
                addr: 0x1_0000_0000,
                size: 0x1_0000_0000,
            },
        ] {
            assert_eq!(decode_map(encode_map(buf)), buf, "map {buf:?}");
            let pointer = Buf {
                addr: buf.addr,
                size: buf.size & 0xffff,
            };
            assert_eq!(
                decode_pointer(encode_pointer(pointer)),
                pointer,
                "pointer {pointer:?}"
            );
            let receive = Buf {
                addr: buf.addr & 0x7f_ffff_ffff,
                size: buf.size & 0xffff,
            };
            assert_eq!(
                decode_receive_list(encode_receive_list(receive)),
                receive,
                "receive list {receive:?}"
            );
        }

        // A pointer descriptor is aligned to 4 bytes and its size field is 16
        // bits, so a larger transfer must use a map descriptor.
        let [d0, d1] = encode_pointer(Buf {
            addr: 0x1_2345_6780,
            size: 0x40,
        });
        assert_eq!(((d0 >> 12) & 0xf, (d0 >> 6) & 0x7), (1, 0));
        assert_eq!(d1, 0x2345_6780);
        let [d0, d1, d2] = encode_map(Buf {
            addr: 0x1_2345_6780,
            size: 0x1_0000_0100,
        });
        assert_eq!(d0, 0x100);
        assert_eq!((d1, (d2 >> 24) & 0xf), (0x2345_6780, 1));
    }

    #[test]
    fn a_message_carries_pointer_map_and_receive_list_descriptors() {
        let x = Buf {
            addr: 0x1_0000_2000,
            size: 0x40,
        };
        let a = Buf {
            addr: 0x2_0000_3000,
            size: 0x100,
        };
        let b = Buf {
            addr: 0x3_0000_4000,
            size: 0x200,
        };
        let w = Buf {
            addr: 0x4_0000_5000,
            size: 0x300,
        };
        let c = Buf {
            addr: 0x5_0000_6000,
            size: 0x80,
        };
        let mut words = vec![
            TYPE_REQUEST as u32 | 1 << 16 | 1 << 20 | 1 << 24 | 1 << 28,
            4 | 2 << 10,
        ];
        // raw data is 4 words; the receive list descriptor follows it.
        let recv_at = words.len() + 2 + 3 + 3 + 3 + 4;
        words[1] |= (recv_at as u32) << 20;
        words.extend(encode_pointer(x));
        words.extend(encode_map(a));
        words.extend(encode_map(b));
        words.extend(encode_map(w));
        words.extend([0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444]);
        words.extend(encode_receive_list(c));
        let msg: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = parse_hipc(&msg).unwrap();
        assert_eq!((m.x.as_slice(), m.a.as_slice()), (&[x][..], &[a][..]));
        assert_eq!((m.b.as_slice(), m.w.as_slice()), (&[b][..], &[w][..]));
        assert_eq!(m.c.as_slice(), &[c]);
        assert_eq!(m.pointer_buffer_usage(), 0x40);
        assert_eq!(
            m.data,
            words[recv_at - 4..recv_at]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>()
        );
        assert_eq!(m.receive_list_count, 2);
        assert_eq!(m.inline_receive_offset(), None);
    }

    #[test]
    fn an_inline_receive_list_is_an_offset_not_a_descriptor() {
        let mut words = vec![TYPE_REQUEST as u32, 2 | 1 << 10 | 3 << 20];
        words.extend([0xaaaa_aaaa, 0xbbbb_bbbb]);
        let msg: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = parse_hipc(&msg).unwrap();
        assert!(m.c.is_empty());
        // raw is two words (8 bytes), plus the receive-list offset, rounded up.
        assert_eq!(m.inline_receive_offset(), Some(16));
    }

    #[test]
    fn tipc_request_and_reply() {
        let msg = client::tipc(7, &[1, 2, 3, 4, 5], &[0x2001]);
        let m = parse_hipc(&msg).unwrap();
        assert_eq!(m.kind, TYPE_TIPC_BASE + 7);
        let tipc = parse_tipc(&m).unwrap();
        assert_eq!(tipc.cmd, 7);
        assert_eq!(&tipc.args[..5], &[1, 2, 3, 4, 5]);
        assert_eq!(m.copy_handles, vec![0x2001]);

        let reply = encode_tipc(0xe15, &42u32.to_le_bytes(), &[], &[]);
        let (result, data, _) = client::tipc_response(&reply);
        assert_eq!((result, &data[..4]), (0xe15, &42u32.to_le_bytes()[..]));
        assert!(parse_tipc(&parse_hipc(&reply).unwrap()).is_err());
    }

    // The guest half of the round-trip tests: a handle to a host object, real
    // requests written to the TLS buffer and replies parsed from it.

    static OBSERVED: std::sync::Mutex<Vec<(&'static str, Vec<u64>)>> =
        std::sync::Mutex::new(Vec::new());
    static CHILD: std::sync::Mutex<Option<std::sync::Weak<dyn Service>>> =
        std::sync::Mutex::new(None);
    static GUEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn record(name: &'static str, values: &[u64]) {
        OBSERVED.lock().unwrap().push((name, values.to_vec()));
    }

    fn observed(name: &'static str) -> Vec<u64> {
        let mut all = OBSERVED.lock().unwrap();
        let i = all
            .iter()
            .position(|(n, _)| *n == name)
            .expect("the guest recorded nothing under that name");
        all.remove(i).1
    }

    struct Root;

    impl Service for Root {
        fn name(&self) -> String {
            "test/root".into()
        }

        fn pointer_buffer_size(&self) -> u16 {
            0x1234
        }

        fn call(&self, _c: &mut Cpu, req: &Request) -> Reply {
            match req.cmd {
                10 => {
                    let child: Arc<dyn Service> = Arc::new(Child);
                    *CHILD.lock().unwrap() = Some(Arc::downgrade(&child));
                    Reply::ok().object(child)
                }
                11 => Reply::ok().u32(0x2a),
                12 => Reply::ok().u32(0x99),
                _ => Reply::err(RESULT_BAD_MESSAGE),
            }
        }
    }

    struct Child;

    impl Service for Child {
        fn name(&self) -> String {
            "test/child".into()
        }

        fn call(&self, _c: &mut Cpu, req: &Request) -> Reply {
            match req.cmd {
                11 => Reply::ok().u32(0x2b),
                _ => Reply::err(RESULT_BAD_MESSAGE),
            }
        }
    }

    fn send_cmif(c: &mut Cpu, handle: u32, cmd: u32, args: &[u8]) -> (u32, Vec<u8>, Hipc) {
        let msg = client::cmif(TYPE_REQUEST, cmd, args, None, false, &[]);
        let reply = crate::services::harness::send(c, handle, &msg);
        client::cmif_response(&reply, false)
    }

    fn send_domain(
        c: &mut Cpu,
        handle: u32,
        cmd: u32,
        object: u32,
        args: &[u8],
    ) -> (u32, Vec<u8>, Hipc) {
        let msg = client::cmif(TYPE_REQUEST, cmd, args, Some(object), false, &[]);
        let reply = crate::services::harness::send(c, handle, &msg);
        client::cmif_response(&reply, true)
    }

    fn send_control(c: &mut Cpu, handle: u32, cmd: u32, args: &[u8]) -> (u32, Vec<u8>, Hipc) {
        let msg = client::cmif(TYPE_CONTROL, cmd, args, None, false, &[]);
        let reply = crate::services::harness::send(c, handle, &msg);
        client::cmif_response(&reply, false)
    }

    fn u32_at(data: &[u8], at: usize) -> u32 {
        data.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default()))
            .unwrap_or(u32::MAX)
    }

    extern "C" fn domain_round_trip(c: &mut Cpu) {
        let object: Arc<dyn Service> = Arc::new(Root);
        let root = Arc::downgrade(&object);
        let handle = crate::services::harness::session(c, object);

        // A legacy CMIF request to the object, before it becomes a domain.
        let (result, data, _) = send_cmif(c, handle, 12, &[]);
        record("legacy", &[result as u64, u32_at(&data, 0) as u64]);

        // QueryPointerBufferSize is manager command 3.
        let (result, data, _) = send_control(c, handle, CONTROL_QUERY_POINTER_BUFFER_SIZE, &[]);
        record(
            "pointer_buffer",
            &[result as u64, u16::from_le_bytes([data[0], data[1]]) as u64],
        );

        // Convert to a domain: the root object is id 1.
        let (result, data, _) = send_control(c, handle, CONTROL_CONVERT_TO_DOMAIN, &[]);
        record("convert", &[result as u64, u32_at(&data, 0) as u64]);

        // A domain request that creates an object returns its new id.
        let (result, data, _) = send_domain(c, handle, 10, 1, &[]);
        let child_id = u32_at(&data, 0);
        record("create", &[result as u64, child_id as u64]);
        let child = CHILD
            .lock()
            .unwrap()
            .clone()
            .expect("the child was created");

        // The new object answers through the same session.
        let (result, data, _) = send_domain(c, handle, 11, child_id, &[]);
        record("child_call", &[result as u64, u32_at(&data, 0) as u64]);

        // CopyFromCurrentDomain returns a handle to the same object.
        let (result, _, hipc) =
            send_control(c, handle, CONTROL_COPY_FROM_DOMAIN, &1u32.to_le_bytes());
        let copy = hipc.move_handles.first().copied().unwrap_or(0);
        record(
            "copy",
            &[result as u64, hipc.move_handles.len() as u64, copy as u64],
        );

        // Releasing a domain object drops its reference.
        let msg = client::domain_close(child_id);
        let reply = crate::services::harness::send(c, handle, &msg);
        let (result, _, _) = client::cmif_response(&reply, true);
        record(
            "release",
            &[result as u64, child.upgrade().is_some() as u64],
        );
        let (result, _, _) = send_domain(c, handle, 11, child_id, &[]);
        record("released_is_gone", &[result as u64]);

        // CloneCurrentObject hands out another handle to the same session, so
        // an object created through one handle is visible through the other.
        let (result, _, hipc) = send_control(c, handle, CONTROL_CLONE_CURRENT_OBJECT, &[]);
        let clone = hipc.move_handles.first().copied().unwrap_or(0);
        record("clone", &[result as u64, clone as u64]);
        let (result, data, _) = send_domain(c, clone, 10, 1, &[]);
        let cloned_child = u32_at(&data, 0);
        record("clone_create", &[result as u64, cloned_child as u64]);
        let clone_child = CHILD
            .lock()
            .unwrap()
            .clone()
            .expect("the child was created");
        let (result, data, _) = send_domain(c, handle, 11, cloned_child, &[]);
        record("shared_object", &[result as u64, u32_at(&data, 0) as u64]);

        // Closing the original handle leaves the clone's table and root alive;
        // closing the clone drops what only it held.
        c.process().handles.lock().unwrap().close(handle);
        record(
            "close_original",
            &[
                root.upgrade().is_some() as u64,
                clone_child.upgrade().is_some() as u64,
            ],
        );
        c.process().handles.lock().unwrap().close(clone);
        record(
            "close_clone",
            &[
                root.upgrade().is_some() as u64,
                clone_child.upgrade().is_some() as u64,
            ],
        );
        // The copy from CopyFromCurrentDomain still holds the root object; the
        // clone's own object went with the clone.
        c.process().handles.lock().unwrap().close(copy);
        record(
            "close_copy",
            &[
                root.upgrade().is_some() as u64,
                clone_child.upgrade().is_some() as u64,
            ],
        );
        c.halt();
    }

    #[test]
    fn guest_messages_round_trip_domains_and_close_their_objects() {
        let _guard = GUEST_LOCK.lock().unwrap();
        crate::services::harness::run(vec![crate::abi::Entry {
            addr: crate::services::harness::ENTRY,
            func: domain_round_trip,
        }]);
        assert_eq!(observed("legacy"), vec![0, 0x99]);
        assert_eq!(
            observed("pointer_buffer"),
            vec![0, 0x1234],
            "QueryPointerBufferSize returns the object's advertised size"
        );
        assert_eq!(observed("convert"), vec![0, 1]);
        let create = observed("create");
        assert_eq!(create[0], 0);
        assert_eq!(create[1], 2);
        assert_eq!(observed("child_call"), vec![0, 0x2b]);
        assert_eq!(observed("copy")[0..2], [0, 1]);
        let clone = observed("clone");
        assert_eq!(clone[0], 0);
        assert_ne!(clone[1], 0);
        let clone_create = observed("clone_create");
        assert_eq!(
            (clone_create[0], clone_create[1]),
            (0, 3),
            "the shared table's next id follows the released object"
        );
        assert_eq!(
            observed("shared_object"),
            vec![0, 0x2b],
            "a cloned handle sees objects created through the other handle"
        );
        let release = observed("release");
        assert_eq!(release[0], 0);
        assert_eq!(release[1], 0, "releasing the id dropped the object");
        assert_eq!(observed("released_is_gone")[0], RESULT_BAD_MESSAGE as u64);
        assert_eq!(observed("close_original"), vec![1, 1]);
        assert_eq!(observed("close_clone"), vec![1, 0]);
        assert_eq!(observed("close_copy"), vec![0, 0]);
    }

    extern "C" fn legacy_and_tipc(c: &mut Cpu) {
        let handle = crate::services::harness::session(c, Arc::new(Root));
        let (result, data, _) = send_cmif(c, handle, 12, &[]);
        record("legacy", &[result as u64, u32_at(&data, 0) as u64]);

        let msg = client::tipc(12, &[], &[]);
        let reply = crate::services::harness::send(c, handle, &msg);
        let (result, data, _) = client::tipc_response(&reply);
        record("tipc", &[result as u64, u32_at(&data, 0) as u64]);
        c.halt();
    }

    #[test]
    fn a_legacy_cmif_request_and_a_tipc_request_both_reach_the_object() {
        let _guard = GUEST_LOCK.lock().unwrap();
        crate::services::harness::run(vec![crate::abi::Entry {
            addr: crate::services::harness::ENTRY,
            func: legacy_and_tipc,
        }]);
        assert_eq!(observed("legacy"), vec![0, 0x99]);
        assert_eq!(observed("tipc"), vec![0, 0x99]);
    }
}
