use std::sync::Arc;

use crate::cpu::Cpu;
use crate::kernel::{KObject, Session};
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
const POINTER_BUFFER_SIZE: u16 = 0x8000;

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
}

fn word(msg: &[u8], i: usize) -> Result<u32, &'static str> {
    msg.get(i * 4..i * 4 + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or("message is truncated")
}

pub fn parse_hipc(msg: &[u8]) -> Result<Hipc, &'static str> {
    let (h0, h1) = (word(msg, 0)?, word(msg, 1)?);
    let mut m = Hipc { kind: h0 as u16, ..Hipc::default() };
    let (nx, na, nb, nw) = ((h0 >> 16) & 0xf, (h0 >> 20) & 0xf, (h0 >> 24) & 0xf, (h0 >> 28) & 0xf);
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
        let (d0, d1) = (word(msg, i)?, word(msg, i + 1)?);
        let addr = d1 as u64 | (((d0 >> 12) & 0xf) as u64) << 32 | (((d0 >> 6) & 0x7) as u64) << 36;
        m.x.push(Buf { addr, size: (d0 >> 16) as u64 });
        i += 2;
    }
    for (count, list) in [(na, &mut m.a), (nb, &mut m.b), (nw, &mut m.w)] {
        for _ in 0..count {
            let (d0, d1, d2) = (word(msg, i)?, word(msg, i + 1)?, word(msg, i + 2)?);
            let addr = d1 as u64 | (((d2 >> 28) & 0xf) as u64) << 32 | (((d2 >> 2) & 0x7) as u64) << 36;
            let size = d0 as u64 | (((d2 >> 24) & 0xf) as u64) << 32;
            list.push(Buf { addr, size });
            i += 3;
        }
    }
    m.data_offset = i * 4;
    m.data = msg.get(i * 4..(i + ndata) * 4).ok_or("data words run past the message")?.to_vec();
    if recv_mode >= 2 {
        let count = if recv_mode == 2 { 1 } else { recv_mode as usize - 2 };
        let mut j = if recv_offset != 0 { recv_offset } else { i + ndata };
        for _ in 0..count {
            let (e0, e1) = (word(msg, j)?, word(msg, j + 1)?);
            m.c.push(Buf { addr: e0 as u64 | ((e1 & 0xffff) as u64) << 32, size: (e1 >> 16) as u64 });
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
    b.get(off..off + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or("CMIF payload is truncated")
}

pub fn parse_cmif(m: &Hipc, domain: bool) -> Result<Cmif, &'static str> {
    let pad = m.data_offset.next_multiple_of(16) - m.data_offset;
    let mut p = m.data.get(pad..).ok_or("CMIF payload is missing")?;
    let mut dom = None;
    if domain {
        let hdr = p.get(..16).ok_or("domain header is truncated")?;
        let (kind, nin, size) = (hdr[0], hdr[1] as usize, u16::from_le_bytes([hdr[2], hdr[3]]) as usize);
        let object = u32_in(hdr, 4)?;
        if kind == DOMAIN_CLOSE {
            return Ok(Cmif { domain: Some((kind, object, Vec::new())), cmd: 0, args: Vec::new() });
        }
        let objects = (0..nin).map(|k| u32_in(p, 16 + size + 4 * k)).collect::<Result<Vec<_>, _>>()?;
        dom = Some((kind, object, objects));
        p = p.get(16..16 + size).ok_or("domain payload is truncated")?;
    }
    if u32_in(p, 0)? != SFCI {
        return Err("missing SFCI magic");
    }
    Ok(Cmif { domain: dom, cmd: u32_in(p, 8)?, args: p.get(16..).unwrap_or(&[]).to_vec() })
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
        push_words(&mut out, &[(copy.len() as u32) << 1 | (moves.len() as u32) << 5]);
        push_words(&mut out, copy);
        push_words(&mut out, moves);
    }
    out
}

pub fn encode_cmif(result: u32, data: &[u8], copy: &[u32], moves: &[u32], domain_objects: Option<&[u32]>) -> Vec<u8> {
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

fn request(m: &Hipc, cmd: u32, args: Vec<u8>, objects: Vec<u32>) -> Request {
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
    }
}

fn new_session(c: &Cpu, object: Arc<dyn Service>) -> u32 {
    c.process().handles.lock().unwrap().insert(KObject::Session(Session::new(object)))
}

fn call(c: &mut Cpu, object: &Arc<dyn Service>, req: &Request, domain: Option<&Arc<Session>>) -> (Reply, Vec<u32>) {
    let p = c.process();
    let name = object.name();
    p.log_ipc(format!("{name} cmd {}", req.cmd));
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

fn control(c: &mut Cpu, session: &Arc<Session>, cmd: u32, args: &[u8]) -> Reply {
    let p = c.process();
    p.log_ipc(format!("control cmd {cmd}"));
    match cmd {
        0 => Reply::ok().u32(session.convert_to_domain()),
        1 => {
            let id = args.get(..4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
            match session.domain_object(id) {
                Some(o) => Reply::ok().move_handle(new_session(c, o)),
                None => Reply::err(RESULT_BAD_MESSAGE),
            }
        }
        2 | 4 => {
            let h = p.handles.lock().unwrap().insert(KObject::Session(session.clone()));
            Reply::ok().move_handle(h)
        }
        3 => Reply::ok().bytes(&POINTER_BUFFER_SIZE.to_le_bytes()),
        _ => Reply::err(RESULT_BAD_MESSAGE),
    }
}

pub fn send_sync_request(c: &mut Cpu, handle: u32, addr: u64, size: u64) -> u64 {
    let _prof = crate::prof::enter(crate::prof::Cat::Ipc);
    let p = c.process();
    let Some(session) = p.handles.lock().unwrap().session(handle) else {
        p.warn_once(format!("SendSyncRequest on invalid handle {handle:#x} from {}", p.describe(c.x[30])));
        return RESULT_INVALID_HANDLE;
    };
    let Some(msg) = p.memory.read(addr, size) else { return RESULT_INVALID_HANDLE };
    let response = match respond(c, handle, &session, &msg) {
        Ok(r) => r,
        Err(why) => {
            p.warn_once(format!("malformed IPC message ({why}) from {}", p.describe(c.x[30])));
            encode_cmif(RESULT_BAD_MESSAGE, &[], &[], &[], None)
        }
    };
    if response.len() as u64 > size {
        p.warn_once(format!("IPC response of {} bytes does not fit the {size}-byte message buffer", response.len()));
        return RESULT_INVALID_HANDLE;
    }
    p.memory.write(addr, &response);
    0
}

fn respond(c: &mut Cpu, handle: u32, session: &Arc<Session>, msg: &[u8]) -> Result<Vec<u8>, &'static str> {
    let m = parse_hipc(msg)?;
    match m.kind {
        TYPE_CLOSE => {
            c.process().handles.lock().unwrap().close(handle);
            Ok(encode_cmif(0, &[], &[], &[], None))
        }
        k if k >= TYPE_TIPC_BASE => {
            let object = session.object().ok_or("session has no object")?;
            let req = request(&m, (k - TYPE_TIPC_BASE) as u32, m.data.clone(), Vec::new());
            let (r, _) = call(c, &object, &req, None);
            Ok(encode_tipc(r.result, &r.data, &r.copy_handles, &r.move_handles))
        }
        TYPE_CONTROL | TYPE_CONTROL_CTX => {
            let cmif = parse_cmif(&m, false)?;
            let r = control(c, session, cmif.cmd, &cmif.args);
            Ok(encode_cmif(r.result, &r.data, &r.copy_handles, &r.move_handles, None))
        }
        TYPE_REQUEST | TYPE_REQUEST_CTX => {
            let domain = session.is_domain();
            let cmif = parse_cmif(&m, domain)?;
            match cmif.domain {
                None => {
                    let object = session.object().ok_or("session has no object")?;
                    let (r, _) = call(c, &object, &request(&m, cmif.cmd, cmif.args, Vec::new()), None);
                    Ok(encode_cmif(r.result, &r.data, &r.copy_handles, &r.move_handles, None))
                }
                Some((DOMAIN_CLOSE, id, _)) => {
                    session.domain_remove(id);
                    Ok(encode_cmif(0, &[], &[], &[], Some(&[])))
                }
                Some((DOMAIN_SEND, id, objects)) => {
                    let Some(object) = session.domain_object(id) else {
                        c.process().warn_once(format!("domain request to unknown object {id}"));
                        return Ok(encode_cmif(RESULT_BAD_MESSAGE, &[], &[], &[], Some(&[])));
                    };
                    let (r, ids) = call(c, &object, &request(&m, cmif.cmd, cmif.args, objects), Some(session));
                    Ok(encode_cmif(r.result, &r.data, &r.copy_handles, &r.move_handles, Some(&ids)))
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

    pub fn cmif(kind: u16, cmd: u32, args: &[u8], domain_object: Option<u32>, pid: bool, copy: &[u32]) -> Vec<u8> {
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
        push_words(&mut out, &[kind as u32, data_words as u32 | (special as u32) << 31]);
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
}

#[cfg(test)]
mod tests {
    use super::client;
    use super::*;

    #[test]
    fn request_round_trip() {
        let msg = client::cmif(TYPE_REQUEST, 1, &0x3a6d73u64.to_le_bytes(), None, true, &[0x1234]);
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
        assert_eq!((m.move_handles.clone(), &m.data[..4]), (vec![0x20009], &[0u8; 4][..]));
    }

    #[test]
    fn buffer_descriptors() {
        let x0 = 0x40u32 << 16 | 1 << 12;
        let a2 = 1u32 << 28;
        let words = [0x0011_0004u32, 0, x0, 0x2345_6780, 0x100, 0x8000_0000, a2];
        let msg: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = parse_hipc(&msg).unwrap();
        assert_eq!(m.x, vec![Buf { addr: 0x1_2345_6780, size: 0x40 }]);
        assert_eq!(m.a, vec![Buf { addr: 0x1_8000_0000, size: 0x100 }]);
    }
}
