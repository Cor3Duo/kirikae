use crate::cpu::Cpu;
use crate::services::stub::unimplemented;
use crate::services::{Reply, Request, Service};

const LANGUAGES: [&str; 18] = [
    "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW", "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR",
];
const OLD_LANGUAGE_COUNT: usize = 15;
const SYSTEM_LANGUAGE: &str = "en-US";
const REGION_USA: u32 = 1;

pub fn language_code(tag: &str) -> u64 {
    let mut b = [0u8; 8];
    b[..tag.len()].copy_from_slice(tag.as_bytes());
    u64::from_le_bytes(b)
}

pub struct Set;

impl Service for Set {
    fn name(&self) -> String {
        "set".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u64(language_code(SYSTEM_LANGUAGE)),
            1 | 5 => {
                let all = if req.cmd == 1 { OLD_LANGUAGE_COUNT } else { LANGUAGES.len() };
                let Some(out) = req.out_buf(0) else { return Reply::ok().u32(0) };
                let n = all.min(out.size as usize / 8);
                let codes: Vec<u8> = LANGUAGES[..n].iter().flat_map(|l| language_code(l).to_le_bytes()).collect();
                c.process().memory.write(out.addr, &codes);
                Reply::ok().u32(n as u32)
            }
            2 => {
                match LANGUAGES.get(req.u32(0) as usize) {
                    Some(l) => Reply::ok().u64(language_code(l)),
                    None => Reply::err(RESULT_INVALID_LANGUAGE),
                }
            }
            3 => Reply::ok().u32(OLD_LANGUAGE_COUNT as u32),
            4 => Reply::ok().u32(REGION_USA),
            6 => Reply::ok().u32(LANGUAGES.len() as u32),
            8 => Reply::ok().u8(0),
            _ => unimplemented("set", c, req),
        }
    }
}

const RESULT_INVALID_LANGUAGE: u32 = (625 << 9) | 105;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_codes_are_little_endian_tags() {
        assert_eq!(language_code("ja"), 0x616a);
        assert_eq!(language_code("en-US").to_le_bytes(), *b"en-US\0\0\0");
        assert_eq!(LANGUAGES[1], SYSTEM_LANGUAGE);
    }
}
