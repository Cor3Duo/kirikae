use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

use crate::{Error, Image, LOAD_BASE, align_up, nca, nro, nso, pfs0};

#[derive(Clone, Debug)]
pub struct Module {
    pub name: String,
    pub image: Image,
}

#[derive(Clone, Debug)]
pub struct Executable {
    pub modules: Vec<Module>,
    pub entry: u64,
    pub title_id: Option<u64>,
    pub main_stack_size: Option<u64>,
    pub system_resource_size: u64,
    pub romfs: Option<RomFs>,
}

pub use crate::romfs::RomFs;

impl Executable {
    pub fn module_at(&self, addr: u64) -> Option<&Module> {
        self.modules
            .iter()
            .find(|m| (m.image.base..m.image.end()).contains(&addr))
    }
}

pub type Files = Vec<(String, Vec<u8>)>;

const LOAD_ORDER: [&str; 13] = [
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];

pub fn from_exefs(
    files: &[(String, Vec<u8>)],
    title_id: Option<u64>,
    romfs: Option<RomFs>,
) -> Result<Executable, Error> {
    let mut modules = Vec::new();
    let mut base = LOAD_BASE;
    for name in LOAD_ORDER {
        let Some((_, bytes)) = files.iter().find(|(n, _)| n == name) else {
            continue;
        };
        let image = nso::load(bytes, base)?;
        base = align_up(image.end(), 0x1000);
        modules.push(Module {
            name: name.to_string(),
            image,
        });
    }
    let entry = modules
        .first()
        .ok_or(Error::Malformed("ExeFS has no modules"))?
        .image
        .entry;
    let npdm = files
        .iter()
        .find(|(n, _)| n == "main.npdm")
        .map(|(_, b)| b.as_slice());
    let aarch32 = npdm.is_some_and(npdm_is_aarch32)
        || modules
            .iter()
            .find(|m| m.name == "main")
            .is_some_and(|m| looks_aarch32(&m.image));
    if aarch32 {
        return Err(Error::Unsupported(
            "this title's main module is AArch32 (32-bit); kirikae only recompiles AArch64",
        ));
    }
    let main_stack_size = npdm.and_then(|b| npdm_u32(b, 0x1c));
    let system_resource_size = npdm.and_then(|b| npdm_u32(b, 0x14)).unwrap_or(0);
    Ok(Executable {
        modules,
        entry,
        title_id,
        main_stack_size,
        system_resource_size,
        romfs,
    })
}

/// META.Flags0 bit 0 is `Is64BitInstruction`: MK8D 1.0 and NSMBUDX have 0x04 (32-bit),
/// while AArch64 titles have 0x07. The ACID flags do not indicate the architecture.
fn npdm_is_aarch32(npdm: &[u8]) -> bool {
    npdm_u32(npdm, 0x0c).is_some_and(|flags| flags != 0 && flags & 1 == 0)
}

/// Fallback for a missing or zero META.Flags0: sample the code. AArch32
/// modules are full of `bx lr` / `push {r11, lr}` and have no AArch64 frame prologues.
fn looks_aarch32(image: &Image) -> bool {
    let start = (image.text.start - image.base) as usize;
    let end = (image.text.end - image.base) as usize;
    let Some(text) = image.memory.get(start..end) else {
        return false;
    };
    let count = |pattern: [u8; 4]| text.windows(4).filter(|w| *w == pattern).count();
    let a32 = count([0x1e, 0xff, 0x2f, 0xe1]) + count([0x00, 0x48, 0x2d, 0xe9]);
    let a64 = count([0xfd, 0x7b, 0xbf, 0xa9]) + count([0xc0, 0x03, 0x5f, 0xd6]);
    a32 > 64 && a64 * 8 < a32
}

fn npdm_title(npdm: &[u8]) -> Option<u64> {
    let at = npdm_u32(npdm, 0x70)? as usize; // aci0_offset
    let aci = npdm.get(at..at + 0x18)?;
    (&aci[..4] == b"ACI0")
        .then(|| u64::from_le_bytes(aci[0x10..0x18].try_into().unwrap()))
        .filter(|&t| t != 0)
}

fn romfs_beside(base: &Path) -> Result<Option<RomFs>, Error> {
    let image = base.join("romfs.bin");
    let folder = base.join("romfs");
    if !image.is_file() && folder.is_dir() {
        eprintln!(
            "kirikae: building {} from {} (once; delete it to rebuild)",
            image.display(),
            folder.display()
        );
        crate::romfs::build(&folder, &image)?;
    }
    if !image.is_file() {
        return Ok(None);
    }
    let size = std::fs::metadata(&image)?.len();
    Ok(Some(RomFs {
        path: std::path::absolute(&image).unwrap_or(image),
        offset: 0,
        size,
        compression: None,
    }))
}

fn npdm_u32(npdm: &[u8], at: usize) -> Option<u64> {
    if npdm.len() < 0x80 || &npdm[..4] != b"META" {
        return None;
    }
    Some(u32::from_le_bytes(npdm[at..at + 4].try_into().unwrap()) as u64)
}

fn single(name: &str, image: Image) -> Executable {
    let entry = image.entry;
    Executable {
        modules: vec![Module {
            name: name.to_string(),
            image,
        }],
        entry,
        title_id: None,
        main_stack_size: None,
        system_resource_size: 0,
        romfs: None,
    }
}

pub fn exefs_from_nsp(path: &Path) -> Result<(Files, u64, Option<RomFs>), Error> {
    let mut r = BufReader::new(File::open(path)?);
    let entries = pfs0::parse(&mut r, 0)?;
    for entry in &entries {
        if entry.name.ends_with(".ncz") {
            return Err(Error::Encrypted);
        }
        if !entry.name.ends_with(".nca") || entry.name.contains(".cnmt.") {
            continue;
        }
        let nca = nca::parse(&mut r, entry.offset)?;
        if let Some(found) = program(&mut r, nca, path)? {
            return Ok(found);
        }
    }
    Err(Error::Malformed("no program NCA with an ExeFS"))
}

fn program<R: Read + Seek>(
    r: &mut R,
    nca: nca::Nca,
    path: &Path,
) -> Result<Option<(Files, u64, Option<RomFs>)>, Error> {
    if nca.content_type != nca::CONTENT_PROGRAM {
        return Ok(None);
    }
    let romfs = match nca.sections.iter().find(|s| s.fs_type == nca::FS_ROMFS) {
        Some(s) => {
            nca.check_plaintext(s)?;
            Some(RomFs {
                path: std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
                offset: s.data_offset,
                size: s.data_size,
                compression: s.compression.map(|(off, len)| (s.data_offset + off, len)),
            })
        }
        None => None,
    };
    for s in nca.sections.iter().filter(|s| s.fs_type == nca::FS_PFS0) {
        nca.check_plaintext(s)?;
        let Ok(files) = pfs0::parse(&mut *r, s.data_offset) else {
            continue;
        };
        if !files.iter().any(|f| f.name == "main") {
            continue;
        }
        let mut out = Vec::new();
        for f in &files {
            out.push((f.name.clone(), pfs0::read_entry(&mut *r, f)?));
        }
        return Ok(Some((out, nca.title_id, romfs)));
    }
    Ok(None)
}

#[derive(Clone, Debug)]
pub enum Source {
    Single(Vec<u8>),
    ExeFs(Files, Option<u64>, Option<RomFs>),
}

pub fn read_source(path: &Path) -> Result<Source, Error> {
    if path.is_dir() {
        let (exefs, base) = if path.join("main").is_file() {
            (path.to_path_buf(), path.parent().map(Path::to_path_buf))
        } else {
            (path.join("exefs"), Some(path.to_path_buf()))
        };
        if !exefs.join("main").is_file() {
            return Err(Error::Malformed(
                "the folder has no ExeFS: expected `main` in it or in an `exefs` folder",
            ));
        }
        let mut files = Vec::new();
        for name in LOAD_ORDER.iter().copied().chain(["main.npdm"]) {
            if let Ok(bytes) = std::fs::read(exefs.join(name)) {
                files.push((name.to_string(), bytes));
            }
        }
        let title = files
            .iter()
            .find(|(n, _)| n == "main.npdm")
            .and_then(|(_, b)| npdm_title(b));
        let romfs = match base {
            Some(base) => romfs_beside(&base)?,
            None => None,
        };
        return Ok(Source::ExeFs(files, title, romfs));
    }
    let mut magic = [0u8; 0x14];
    File::open(path)?
        .read_exact(&mut magic)
        .map_err(|_| Error::TooSmall)?;
    if &magic[..4] == b"PFS0" {
        let (files, title, romfs) = exefs_from_nsp(path)?;
        return Ok(Source::ExeFs(files, Some(title), romfs));
    }
    Ok(Source::Single(std::fs::read(path)?))
}

pub fn from_source(source: &Source) -> Result<Executable, Error> {
    match source {
        Source::ExeFs(files, title, romfs) => from_exefs(files, *title, romfs.clone()),
        Source::Single(bytes) if nro::is_nro(bytes) => {
            Ok(single("main", nro::load(bytes, LOAD_BASE)?))
        }
        Source::Single(bytes) if nso::is_nso(bytes) => {
            Ok(single("main", nso::load(bytes, LOAD_BASE)?))
        }
        Source::Single(_) => Err(Error::BadMagic),
    }
}

pub fn open(path: &Path) -> Result<Executable, Error> {
    from_source(&read_source(path)?)
}

const PACK_MAGIC: &[u8; 8] = b"KRKPKG02";

pub fn pack(source: &Source) -> Vec<u8> {
    type Parts<'a> = (u8, u64, Vec<(&'a str, &'a [u8])>, Option<&'a RomFs>);
    let (kind, title, files, romfs): Parts = match source {
        Source::Single(b) => (0, 0, vec![("main", b.as_slice())], None),
        Source::ExeFs(f, t, r) => (
            1,
            t.unwrap_or(0),
            f.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect(),
            r.as_ref(),
        ),
    };
    let mut out = PACK_MAGIC.to_vec();
    out.push(kind);
    out.extend_from_slice(&title.to_le_bytes());
    out.extend_from_slice(&(files.len() as u32).to_le_bytes());
    for (name, bytes) in files {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    match romfs {
        Some(r) => {
            let p = r.path.to_string_lossy();
            out.extend_from_slice(&(p.len() as u16).to_le_bytes());
            out.extend_from_slice(p.as_bytes());
            out.extend_from_slice(&r.offset.to_le_bytes());
            out.extend_from_slice(&r.size.to_le_bytes());
            let (co, cl) = r.compression.unwrap_or((0, 0));
            out.extend_from_slice(&co.to_le_bytes());
            out.extend_from_slice(&cl.to_le_bytes());
        }
        None => out.extend_from_slice(&0u16.to_le_bytes()),
    }
    out
}

pub fn unpack_source(blob: &[u8]) -> Result<Source, Error> {
    let bad = || Error::Malformed("bad embedded package");
    let take = |at: &mut usize, n: usize| -> Result<&[u8], Error> {
        let s = blob.get(*at..*at + n).ok_or_else(bad)?;
        *at += n;
        Ok(s)
    };
    let mut at = 0;
    if take(&mut at, 8)? != PACK_MAGIC {
        return Err(bad());
    }
    let kind = take(&mut at, 1)?[0];
    let title = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap());
    let count = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap());
    let mut files = Vec::new();
    for _ in 0..count {
        let len = u16::from_le_bytes(take(&mut at, 2)?.try_into().unwrap()) as usize;
        let name = String::from_utf8_lossy(take(&mut at, len)?).into_owned();
        let size = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap()) as usize;
        files.push((name, take(&mut at, size)?.to_vec()));
    }
    let path_len = u16::from_le_bytes(take(&mut at, 2)?.try_into().unwrap()) as usize;
    let romfs = if path_len == 0 {
        None
    } else {
        let path = PathBuf::from(String::from_utf8_lossy(take(&mut at, path_len)?).into_owned());
        let offset = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap());
        let size = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap());
        let co = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap());
        let cl = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap());
        Some(RomFs {
            path,
            offset,
            size,
            compression: (cl != 0).then_some((co, cl)),
        })
    };
    match kind {
        0 => Ok(Source::Single(files.pop().ok_or_else(bad)?.1)),
        1 => Ok(Source::ExeFs(files, (title != 0).then_some(title), romfs)),
        _ => Err(bad()),
    }
}

pub fn unpack(blob: &[u8]) -> Result<Executable, Error> {
    from_source(&unpack_source(blob)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npdm_with_flags(flags: u32) -> Vec<u8> {
        let mut npdm = vec![0u8; 0x80 + 0x210];
        npdm[..4].copy_from_slice(b"META");
        npdm[0x78..0x7C].copy_from_slice(&0x80u32.to_le_bytes());
        npdm[0x80 + 0x200..0x80 + 0x204].copy_from_slice(b"ACID");
        npdm[0x80 + 0x20C..0x80 + 0x210].copy_from_slice(&flags.to_le_bytes());
        npdm
    }

    #[test]
    fn reads_the_title_from_aci0() {
        let mut npdm = vec![0u8; 0x100];
        npdm[..4].copy_from_slice(b"META");
        npdm[0x70..0x74].copy_from_slice(&0x80u32.to_le_bytes());
        npdm[0x80..0x84].copy_from_slice(b"ACI0");
        npdm[0x90..0x98].copy_from_slice(&0x0100_0000_0001_0000u64.to_le_bytes());
        assert_eq!(npdm_title(&npdm), Some(0x0100_0000_0001_0000));
    }

    #[test]
    fn reads_is64bit_from_meta_flags0() {
        let mut npdm = vec![0u8; 0x100];
        npdm[..4].copy_from_slice(b"META");
        npdm[0x0c..0x10].copy_from_slice(&0x04u32.to_le_bytes());
        assert!(npdm_is_aarch32(&npdm));
        npdm[0x0c..0x10].copy_from_slice(&0x07u32.to_le_bytes());
        assert!(!npdm_is_aarch32(&npdm));
        npdm[0x0c..0x10].copy_from_slice(&0u32.to_le_bytes());
        assert!(!npdm_is_aarch32(&npdm));
    }

    #[test]
    fn acid_flags_do_not_decide_the_architecture() {
        assert!(!npdm_is_aarch32(&npdm_with_flags(0)));
        assert!(!npdm_is_aarch32(&npdm_with_flags(u32::MAX)));
    }

    #[test]
    fn detects_aarch32_from_the_code_sample() {
        let mut memory = vec![0u8; 0x1000];
        for i in 0..100 {
            memory[i * 4..i * 4 + 4].copy_from_slice(&[0x1e, 0xff, 0x2f, 0xe1]);
        }
        let a32 = Image {
            base: 0,
            memory,
            text: 0..0x1000,
            ro: 0x1000..0x1000,
            data: 0x1000..0x1000,
            bss: 0x1000..0x1000,
            entry: 0,
        };
        assert!(looks_aarch32(&a32));

        let mut memory = vec![0u8; 0x1000];
        for i in 0..100 {
            memory[i * 8..i * 8 + 4].copy_from_slice(&[0xfd, 0x7b, 0xbf, 0xa9]);
        }
        let a64 = Image {
            memory,
            ..a32.clone()
        };
        assert!(!looks_aarch32(&a64));
    }
}
