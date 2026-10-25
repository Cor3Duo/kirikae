//! `.eh_frame_hdr` and FDE parsing.
//!
//! The header, its binary search table and the DWARF pointer encodings follow
//! the Itanium C++ ABI unwind tables. The algorithm is modelled on LASRecomp's
//! `recomp/ehframe.py` (MIT, Copyright (c) LASRecomp contributors).

use std::fmt;
use std::ops::Range;

use kirikae_loader::{Image, dynamic};

const DW_EH_PE_ABSPTR: u8 = 0x00;
const DW_EH_PE_UDATA4: u8 = 0x03;
const DW_EH_PE_UDATA8: u8 = 0x04;
const DW_EH_PE_SDATA4: u8 = 0x0b;
const DW_EH_PE_SDATA8: u8 = 0x0c;
const DW_EH_PE_PCREL: u8 = 0x10;
const DW_EH_PE_DATAREL: u8 = 0x30;
const DW_EH_PE_OMIT: u8 = 0xff;

const MAX_ENTRIES: u64 = 1 << 22;
const MAX_AUGMENTATION: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The module has no MOD0 or no `.eh_frame_hdr`.
    Missing,
    /// The MOD0 range points outside the module image.
    OutOfBounds,
    /// A field, an entry or an FDE runs past the end of the data.
    Truncated,
    /// Only version 1 of the header is defined.
    Version(u8),
    /// A DW_EH_PE format or application this parser does not implement.
    Encoding(u8),
    /// The entry count is too large to be plausible.
    TooManyEntries(u64),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Missing => write!(f, "module has no .eh_frame_hdr"),
            Error::OutOfBounds => write!(f, ".eh_frame_hdr lies outside the module"),
            Error::Truncated => write!(f, ".eh_frame_hdr is truncated"),
            Error::Version(v) => write!(f, "unsupported .eh_frame_hdr version {v}"),
            Error::Encoding(e) => write!(f, "unsupported DWARF pointer encoding {e:#x}"),
            Error::TooManyEntries(n) => write!(f, "implausible .eh_frame_hdr entry count {n}"),
        }
    }
}

impl std::error::Error for Error {}

/// One function in the header's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Function {
    pub start: u64,
    /// `address_range` from the matching FDE, when its CIE reports the encoding.
    pub size: Option<u64>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    base: u64,
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], base: u64) -> Reader<'a> {
        Reader { bytes, base, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or(Error::Truncated)?;
        let bytes = self.bytes.get(self.at..end).ok_or(Error::Truncated)?;
        self.at = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| Error::Truncated)?,
        ))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| Error::Truncated)?,
        ))
    }

    fn uleb(&mut self) -> Result<u64, Error> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift >= 64 {
                return Err(Error::Truncated);
            }
            value |= u64::from(byte & 0x7f) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
    }

    fn sleb(&mut self) -> Result<i64, Error> {
        let mut value = 0i64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift >= 64 {
                return Err(Error::Truncated);
            }
            value |= i64::from(byte & 0x7f) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                if shift < 64 && byte & 0x40 != 0 {
                    value |= -1i64 << shift;
                }
                return Ok(value);
            }
        }
    }

    /// Reads one DW_EH_PE encoded pointer. `pcrel` is relative to the field
    /// itself, `datarel` to the start of the buffer (for the header, the start
    /// of `.eh_frame_hdr`).
    fn encoded(&mut self, enc: u8) -> Result<u64, Error> {
        if enc == DW_EH_PE_OMIT {
            return Err(Error::Encoding(enc));
        }
        let field = self.base + self.at as u64;
        let raw = match enc & 0x0f {
            DW_EH_PE_ABSPTR => self.u64()?,
            DW_EH_PE_UDATA4 => u64::from(self.u32()?),
            DW_EH_PE_SDATA4 => self.u32()? as i32 as i64 as u64,
            DW_EH_PE_UDATA8 | DW_EH_PE_SDATA8 => self.u64()?,
            _ => return Err(Error::Encoding(enc)),
        };
        let base = match enc & 0x70 {
            0 => return Ok(raw),
            DW_EH_PE_PCREL => field,
            DW_EH_PE_DATAREL => self.base,
            _ => return Err(Error::Encoding(enc)),
        };
        Ok(raw.wrapping_add(base))
    }
}

/// Function bounds for a module, or [`Error::Missing`] when it has no header.
pub fn functions(image: &Image) -> Result<Vec<Function>, Error> {
    let section = dynamic::eh_frame_hdr(image).ok_or(Error::Missing)?;
    parse_section(image, section)
}

/// Parses the header at a guest address range. Entries outside the module text
/// are dropped; the result is sorted by start address and deduplicated.
pub fn parse_section(image: &Image, section: Range<u64>) -> Result<Vec<Function>, Error> {
    if section.start < image.base || section.end < section.start || section.end > image.end() {
        return Err(Error::OutOfBounds);
    }
    let start = (section.start - image.base) as usize;
    let end = (section.end - image.base) as usize;
    let bytes = image.memory.get(start..end).ok_or(Error::OutOfBounds)?;
    let mut r = Reader::new(bytes, section.start);

    let version = r.u8()?;
    if version != 1 {
        return Err(Error::Version(version));
    }
    let ptr_enc = r.u8()?;
    let count_enc = r.u8()?;
    let table_enc = r.u8()?;
    let eh_frame = if ptr_enc == DW_EH_PE_OMIT {
        None
    } else {
        Some(r.encoded(ptr_enc)?)
    };
    let count = r.encoded(count_enc)?;
    if count > MAX_ENTRIES {
        return Err(Error::TooManyEntries(count));
    }

    let mut out = Vec::new();
    for _ in 0..count {
        let start = r.encoded(table_enc)?;
        let fde = r.encoded(table_enc)?;
        if !image.is_code(start) {
            continue;
        }
        let size = if eh_frame.is_none_or(|frame| fde >= frame) {
            fde_size(image, fde, start)
        } else {
            None
        };
        out.push(Function { start, size });
    }
    out.sort_unstable_by_key(|f| f.start);
    out.dedup_by_key(|f| f.start);
    Ok(out)
}

/// Reads the FDE at `fde` and returns its `address_range`, but only when it is
/// consistent with the module text.
fn fde_size(image: &Image, fde: u64, start: u64) -> Option<u64> {
    let at = usize::try_from(fde.checked_sub(image.base)?).ok()?;
    let mem = image.memory.get(at..)?;
    let mut r = Reader::new(mem, fde);
    let len = r.u32().ok()?;
    if len == u32::MAX {
        return None;
    }
    let body_at = r.at;
    let body_end = body_at.checked_add(usize::try_from(len).ok()?)?;
    let body = mem.get(body_at..body_end)?;
    let mut body = Reader::new(body, fde + body_at as u64);

    let cie_ptr = body.u32().ok()?;
    if cie_ptr == 0 {
        return None;
    }
    let enc = cie_encoding(image, body.base.checked_sub(u64::from(cie_ptr))?)?;
    body.encoded(enc).ok()?;
    let size = body.encoded(enc & 0x0f).ok()?;
    (size != 0 && start.checked_add(size)? <= image.text.end).then_some(size)
}

/// Returns the `R` augmentation encoding of the CIE at `cie`.
fn cie_encoding(image: &Image, cie: u64) -> Option<u8> {
    let at = usize::try_from(cie.checked_sub(image.base)?).ok()?;
    let mem = image.memory.get(at..)?;
    let mut r = Reader::new(mem, cie);
    let len = r.u32().ok()?;
    if len == u32::MAX {
        return None;
    }
    let body_at = r.at;
    let body = mem.get(body_at..body_at.checked_add(usize::try_from(len).ok()?)?)?;
    let mut r = Reader::new(body, cie + body_at as u64);

    if r.u32().ok()? != 0 {
        return None;
    }
    r.u8().ok()?;
    let mut aug = [0u8; MAX_AUGMENTATION];
    let mut n = 0;
    loop {
        let byte = r.u8().ok()?;
        if byte == 0 {
            break;
        }
        *aug.get_mut(n)? = byte;
        n += 1;
    }
    let aug = aug.get(..n)?;
    if aug.first() != Some(&b'z') {
        return None;
    }
    r.uleb().ok()?;
    r.sleb().ok()?;
    r.uleb().ok()?;
    let aug_len = r.uleb().ok()?;
    let data = r.at;
    let end = data.checked_add(usize::try_from(aug_len).ok()?)?;
    if end > body.len() {
        return None;
    }
    // "zPLR" and friends put a personality pointer and an LSDA encoding
    // before the R byte; without reading those we cannot find it.
    (aug == b"zR" && aug_len >= 1)
        .then(|| body.get(data).copied())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kirikae_a64::asm;
    use kirikae_loader::LOAD_BASE;

    const MOD0: usize = 0x100;
    const HDR: usize = 0x140;
    const CODE: usize = 0x800;
    const HDR_ADDR: u64 = LOAD_BASE + HDR as u64;

    struct Header {
        bytes: Vec<u8>,
        base: u64,
    }

    impl Header {
        fn new(version: u8, ptr_enc: u8, count_enc: u8, table_enc: u8) -> Header {
            Header {
                bytes: vec![version, ptr_enc, count_enc, table_enc],
                base: HDR_ADDR,
            }
        }

        fn enc(&mut self, enc: u8, value: u64) {
            let field = self.base + self.bytes.len() as u64;
            let raw = match enc & 0x70 {
                DW_EH_PE_PCREL => value.wrapping_sub(field),
                DW_EH_PE_DATAREL => value.wrapping_sub(self.base),
                _ => value,
            };
            match enc & 0x0f {
                DW_EH_PE_UDATA4 | DW_EH_PE_SDATA4 => {
                    self.bytes.extend_from_slice(&(raw as u32).to_le_bytes())
                }
                _ => self.bytes.extend_from_slice(&raw.to_le_bytes()),
            }
        }
    }

    fn image_with(header: &[u8]) -> Image {
        let base = LOAD_BASE;
        let mut memory = vec![0u8; 0x1000];
        memory[4..8].copy_from_slice(&(MOD0 as u32).to_le_bytes());
        for code in [CODE, CODE + 0x40] {
            memory[code..code + 4].copy_from_slice(&asm::ret().to_le_bytes());
        }
        let rel = |addr: usize| (addr as i64 - MOD0 as i64) as i32 as u32;
        let mod0 = [
            u32::from_le_bytes(*b"MOD0"),
            0,
            rel(0x1000),
            rel(0x1000),
            rel(HDR),
            rel(HDR + header.len()),
            0,
        ];
        for (i, word) in mod0.iter().enumerate() {
            memory[MOD0 + 4 * i..MOD0 + 4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        memory[HDR..HDR + header.len()].copy_from_slice(header);
        Image {
            base,
            memory,
            text: base + CODE as u64..base + 0x1000,
            ro: base..base + CODE as u64,
            data: base + 0x1000..base + 0x1000,
            bss: base + 0x1000..base + 0x1000,
            entry: base + CODE as u64,
        }
    }

    fn two_entries(table_enc: u8) -> Header {
        let mut h = Header::new(
            1,
            DW_EH_PE_SDATA4 | DW_EH_PE_PCREL,
            DW_EH_PE_UDATA4,
            table_enc,
        );
        h.enc(DW_EH_PE_SDATA4 | DW_EH_PE_PCREL, HDR_ADDR + 0x80);
        h.enc(DW_EH_PE_UDATA4, 2);
        for start in [LOAD_BASE + CODE as u64, LOAD_BASE + CODE as u64 + 0x40] {
            h.enc(table_enc, start);
            h.enc(table_enc, HDR_ADDR + 0x80);
        }
        h
    }

    #[test]
    fn udata4_pcrel_table_yields_starts() {
        let h = two_entries(DW_EH_PE_UDATA4 | DW_EH_PE_PCREL);
        let image = image_with(&h.bytes);
        assert_eq!(
            functions(&image).unwrap(),
            [
                Function {
                    start: LOAD_BASE + CODE as u64,
                    size: None
                },
                Function {
                    start: LOAD_BASE + CODE as u64 + 0x40,
                    size: None
                },
            ]
        );
    }

    #[test]
    fn wide_and_absolute_encodings() {
        for enc in [
            DW_EH_PE_UDATA8 | DW_EH_PE_PCREL,
            DW_EH_PE_SDATA8 | DW_EH_PE_DATAREL,
            DW_EH_PE_ABSPTR,
        ] {
            let image = image_with(&two_entries(enc).bytes);
            let starts: Vec<u64> = functions(&image).unwrap().iter().map(|f| f.start).collect();
            assert_eq!(
                starts,
                [LOAD_BASE + CODE as u64, LOAD_BASE + CODE as u64 + 0x40],
                "{enc:#x}"
            );
        }
    }

    #[test]
    fn sdata4_datarel_table_yields_starts() {
        let h = two_entries(DW_EH_PE_SDATA4 | DW_EH_PE_DATAREL);
        let image = image_with(&h.bytes);
        let starts: Vec<u64> = functions(&image).unwrap().iter().map(|f| f.start).collect();
        assert_eq!(
            starts,
            [LOAD_BASE + CODE as u64, LOAD_BASE + CODE as u64 + 0x40]
        );
    }

    #[test]
    fn fde_supplies_the_size() {
        let (cie, fde) = (0x200, 0x240);
        let mut h = Header::new(
            1,
            DW_EH_PE_SDATA4 | DW_EH_PE_PCREL,
            DW_EH_PE_UDATA4,
            DW_EH_PE_UDATA4 | DW_EH_PE_DATAREL,
        );
        h.enc(DW_EH_PE_SDATA4 | DW_EH_PE_PCREL, LOAD_BASE + cie as u64);
        h.enc(DW_EH_PE_UDATA4, 1);
        h.enc(DW_EH_PE_UDATA4 | DW_EH_PE_DATAREL, LOAD_BASE + CODE as u64);
        h.enc(DW_EH_PE_UDATA4 | DW_EH_PE_DATAREL, LOAD_BASE + fde as u64);
        let mut image = image_with(&h.bytes);

        let enc = DW_EH_PE_SDATA4 | DW_EH_PE_PCREL;
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.push(1);
        body.extend_from_slice(b"zR\0");
        body.extend_from_slice(&[1, 1, 30, 1, enc]);
        image.memory[cie..cie + 4].copy_from_slice(&(body.len() as u32).to_le_bytes());
        image.memory[cie + 4..cie + 4 + body.len()].copy_from_slice(&body);

        let mut body = Vec::new();
        body.extend_from_slice(
            &((LOAD_BASE + fde as u64 + 4 - (LOAD_BASE + cie as u64)) as u32).to_le_bytes(),
        );
        let start = LOAD_BASE + CODE as u64;
        let field = LOAD_BASE + fde as u64 + 4;
        body.extend_from_slice(&((start as i64 - field as i64) as i32).to_le_bytes());
        body.extend_from_slice(&0x20u32.to_le_bytes());
        image.memory[fde..fde + 4].copy_from_slice(&(body.len() as u32).to_le_bytes());
        image.memory[fde + 4..fde + 4 + body.len()].copy_from_slice(&body);

        assert_eq!(
            functions(&image).unwrap(),
            [Function {
                start,
                size: Some(0x20)
            }]
        );
    }

    #[test]
    fn truncated_table_is_an_error() {
        let mut h = two_entries(DW_EH_PE_UDATA4 | DW_EH_PE_PCREL);
        h.bytes.truncate(h.bytes.len() - 4);
        let image = image_with(&h.bytes);
        assert_eq!(functions(&image), Err(Error::Truncated));
    }

    #[test]
    fn malformed_inputs_are_errors() {
        let mut h = two_entries(DW_EH_PE_UDATA4 | DW_EH_PE_PCREL);
        h.bytes[0] = 2;
        assert_eq!(functions(&image_with(&h.bytes)), Err(Error::Version(2)));

        let h = two_entries(0x99);
        assert_eq!(functions(&image_with(&h.bytes)), Err(Error::Encoding(0x99)));

        let mut image = image_with(&two_entries(DW_EH_PE_UDATA4 | DW_EH_PE_PCREL).bytes);
        for i in 0x10..0x18 {
            image.memory[MOD0 + i] = 0;
        }
        assert_eq!(functions(&image), Err(Error::Missing));
    }

    #[test]
    fn fde_starts_seed_discovery() {
        let h = two_entries(DW_EH_PE_UDATA4 | DW_EH_PE_PCREL);
        let image = image_with(&h.bytes);
        let prog = crate::discover(&image);
        assert_eq!(
            prog.entries.into_iter().collect::<Vec<_>>(),
            [LOAD_BASE + CODE as u64, LOAD_BASE + CODE as u64 + 0x40]
        );
    }
}
