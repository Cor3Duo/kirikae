use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

pub struct Lm;

impl Service for Lm {
    fn name(&self) -> String {
        "lm".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().object(Arc::new(Logger)),
            _ => unimplemented("lm", c, req),
        }
    }
}

pub struct Logger;

const CHUNK_MESSAGE: u8 = 2;
const CHUNK_LINE: u8 = 3;
const CHUNK_FILE: u8 = 4;
const CHUNK_FUNCTION: u8 = 5;

fn uleb128(b: &[u8], i: &mut usize) -> Option<usize> {
    let (mut v, mut shift) = (0usize, 0);
    loop {
        let byte = *b.get(*i)?;
        *i += 1;
        v |= ((byte & 0x7f) as usize) << shift;
        if byte & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
    }
}

pub fn format_packet(packet: &[u8]) -> Option<String> {
    let severity = *packet.get(0x11)?;
    let (mut message, mut file, mut func, mut line) =
        (String::new(), String::new(), String::new(), None);
    let mut i = 0x18;
    while i < packet.len() {
        let id = packet[i];
        i += 1;
        let len = uleb128(packet, &mut i)?;
        let data = packet.get(i..i + len)?;
        i += len;
        let text = || {
            String::from_utf8_lossy(data)
                .trim_end_matches('\0')
                .to_string()
        };
        match id {
            CHUNK_MESSAGE => message.push_str(&text()),
            CHUNK_FILE => file = text(),
            CHUNK_FUNCTION => func = text(),
            CHUNK_LINE if len >= 4 => {
                line = Some(u32::from_le_bytes(data[..4].try_into().unwrap()))
            }
            _ => {}
        }
    }
    if message.is_empty() {
        return None;
    }
    let level = ["trace", "info", "warn", "error", "fatal"]
        .get(severity as usize)
        .copied()
        .unwrap_or("log");
    let at = match (file.is_empty(), line) {
        (false, Some(l)) => format!(" ({file}:{l} {func})"),
        _ => String::new(),
    };
    Some(format!("[guest {level}] {}{at}", message.trim_end()))
}

impl Service for Logger {
    fn name(&self) -> String {
        "lm/ILogger".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => {
                let p = c.process();
                if let Some(buf) = req.in_buf(0)
                    && let Some(bytes) = p.memory.read(buf.addr, buf.size)
                    && let Some(text) = format_packet(&bytes)
                {
                    eprintln!("{text}");
                }
                Reply::ok()
            }
            1 => Reply::ok(),
            _ => unimplemented("lm/ILogger", c, req),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_a_packet() {
        let mut p = vec![0u8; 0x18];
        p[0x11] = 2;
        p.extend_from_slice(&[CHUNK_MESSAGE, 5]);
        p.extend_from_slice(b"hello");
        p.extend_from_slice(&[CHUNK_LINE, 4, 7, 0, 0, 0, CHUNK_FILE, 3]);
        p.extend_from_slice(b"a.c");
        assert_eq!(format_packet(&p).unwrap(), "[guest warn] hello (a.c:7 )");
    }
}
