use std::io::{Read, Seek, SeekFrom};

use crate::Error;

pub const CONTENT_PROGRAM: u8 = 0;
pub const FS_ROMFS: u8 = 0;
pub const FS_PFS0: u8 = 1;
const MEDIA: u64 = 0x200;
pub const HEADER_SIZE: usize = 0xc00;
pub const CRYPTO_NONE: u8 = 1;
pub const CRYPTO_XTS: u8 = 2;
pub const CRYPTO_CTR: u8 = 3;
pub const CRYPTO_BKTR: u8 = 4;
pub const CRYPTO_CTR_SKIP_HASH: u8 = 5;
pub const CRYPTO_BKTR_SKIP_HASH: u8 = 6;
const HASH_SHA256: u8 = 2;
const HASH_IVFC: u8 = 3;

#[derive(Clone, Debug)]
pub struct Section {
    pub index: usize,
    pub fs_type: u8,
    pub hash_type: u8,
    pub crypto: u8,
    pub start: u64,
    pub end: u64,
    pub nonce: u64,
    pub data_offset: u64,
    pub data_size: u64,
    pub compression: Option<(u64, u64)>,
}

#[derive(Clone, Debug)]
pub struct Nca {
    pub magic: [u8; 4],
    pub base: u64,
    pub decrypted_dump: bool,
    pub content_type: u8,
    pub title_id: u64,
    pub sections: Vec<Section>,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn plain_magic(m: &[u8]) -> bool {
    matches!(m, b"DNCA" | b"NCA3" | b"NCA2")
}

pub fn parse<R: Read + Seek>(r: &mut R, base: u64) -> Result<Nca, Error> {
    r.seek(SeekFrom::Start(base))?;
    let mut h = vec![0u8; HEADER_SIZE];
    r.read_exact(&mut h)?;
    if !plain_magic(&h[0x200..0x204]) {
        return Err(Error::Encrypted);
    }
    let magic: [u8; 4] = h[0x200..0x204].try_into().unwrap();
    let decrypted_dump = &magic == b"DNCA";

    let mut sections = Vec::new();
    for i in 0..4 {
        let start = u32_at(&h, 0x240 + i * 0x10) as u64 * MEDIA;
        let end = u32_at(&h, 0x244 + i * 0x10) as u64 * MEDIA;
        if end == 0 {
            continue;
        }
        let fs = &h[0x400 + i * 0x200..0x600 + i * 0x200];
        let (fs_type, hash_type, crypto) = (fs[2], fs[3], fs[4]);
        let (offset, size) = match hash_type {
            HASH_SHA256 => {
                let layers = u32_at(fs, 0x2c) as usize;
                if layers == 0 || layers > 5 {
                    return Err(Error::Malformed("NCA SHA-256 layer count"));
                }
                let at = 0x30 + (layers - 1) * 0x10;
                (u64_at(fs, at), u64_at(fs, at + 8))
            }
            HASH_IVFC => {
                let max_layers = u32_at(fs, 0x14) as usize;
                if !(2..=7).contains(&max_layers) {
                    return Err(Error::Malformed("NCA IVFC level count"));
                }
                let at = 0x18 + (max_layers - 2) * 0x18;
                (u64_at(fs, at), u64_at(fs, at + 8))
            }
            _ => (0, end - start),
        };
        let compression =
            (&fs[0x188..0x18c] == b"BKTR").then(|| (u64_at(fs, 0x178), u64_at(fs, 0x180)));
        sections.push(Section {
            index: i,
            fs_type,
            hash_type,
            crypto,
            start: base + start,
            end: base + end,
            nonce: u64_at(fs, 0x140),
            data_offset: base + start + offset,
            data_size: size,
            compression,
        });
    }
    Ok(Nca {
        magic,
        base,
        decrypted_dump,
        content_type: h[0x205],
        title_id: u64_at(&h, 0x210),
        sections,
    })
}

impl Nca {
    pub fn check_plaintext(&self, s: &Section) -> Result<(), Error> {
        if self.decrypted_dump || s.crypto == CRYPTO_NONE {
            return Ok(());
        }
        match s.crypto {
            CRYPTO_BKTR | CRYPTO_BKTR_SKIP_HASH => Err(Error::Unsupported(
                "this section is an update patch (BKTR); updates are not supported yet",
            )),
            _ => Err(Error::Encrypted),
        }
    }
}
