pub mod cnmt;
pub mod dynamic;
pub mod lz4;
pub mod nacp;
pub mod nca;
pub mod nro;
pub mod nso;
pub mod package;
pub mod pfs0;
pub mod romfs;

pub use package::{open, Executable, Module, RomFs, Source};

use std::fmt;
use std::ops::Range;

pub const LOAD_BASE: u64 = 0x0800_0000;

#[derive(Debug)]
pub enum Error {
    TooSmall,
    BadMagic,
    Malformed(&'static str),
    Encrypted,
    Unsupported(&'static str),
    Io(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TooSmall => write!(f, "file is too small"),
            Error::BadMagic => write!(f, "not a recognised Switch executable"),
            Error::Malformed(what) => write!(f, "malformed executable: {what}"),
            Error::Encrypted => write!(f, "this file is encrypted; only decrypted dumps are supported (.dnsp, an extracted ExeFS folder, NSO or NRO)"),
            Error::Unsupported(what) => write!(f, "not supported: {what}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug)]
pub struct Image {
    pub base: u64,
    pub memory: Vec<u8>,
    pub text: Range<u64>,
    pub ro: Range<u64>,
    pub data: Range<u64>,
    pub bss: Range<u64>,
    pub entry: u64,
}

impl Image {
    pub fn end(&self) -> u64 {
        self.base + self.memory.len() as u64
    }

    pub fn is_code(&self, addr: u64) -> bool {
        self.text.contains(&addr) && addr.is_multiple_of(4)
    }

    pub fn word(&self, addr: u64) -> Option<u32> {
        if !self.is_code(addr) {
            return None;
        }
        let off = (addr - self.base) as usize;
        let bytes = self.memory.get(off..off + 4)?;
        Some(u32::from_le_bytes(bytes.try_into().unwrap()))
    }
}

pub fn load(file: &[u8], base: u64) -> Result<Image, Error> {
    if nro::is_nro(file) {
        return nro::load(file, base);
    }
    if nso::is_nso(file) {
        return nso::load(file, base);
    }
    Err(Error::BadMagic)
}

pub(crate) fn align_up(v: u64, align: u64) -> u64 {
    v.div_ceil(align) * align
}
