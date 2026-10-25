use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

use crate::{Error, Image, LOAD_BASE, align_up, cnmt, nacp, nca, nro, nso, pfs0};

#[derive(Clone, Debug)]
pub struct Module {
    pub name: String,
    pub image: Image,
    /// The `ModuleId` from the NSO header: the build id the SDK reports to
    /// `nn::ro`. Homebrew modules and synthetic test images may leave it zero.
    pub module_id: [u8; 32],
}

/// Instruction set of the program's code modules.
///
/// A Switch program is entirely A32 or entirely A64; `META.Flags0` bit 0
/// (`Is64BitInstruction`) plus a code sample of `main` decide it in
/// [`from_exefs`]. The interpreter needs the tag so it can decode the words it
/// fetches; the recompiler and its backends still only accept A64.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Arch {
    #[default]
    A64,
    A32,
}

impl Arch {
    pub const fn name(self) -> &'static str {
        match self {
            Arch::A64 => "AArch64",
            Arch::A32 => "AArch32",
        }
    }

    pub const fn is_32_bit(self) -> bool {
        matches!(self, Arch::A32)
    }
}

#[derive(Clone, Debug)]
pub struct Executable {
    pub modules: Vec<Module>,
    pub entry: u64,
    pub arch: Arch,
    pub title_id: Option<u64>,
    pub main_stack_size: Option<u64>,
    pub system_resource_size: u64,
    pub romfs: Option<RomFs>,
    /// `control.nacp` from the Control content, when the dump carries one.
    pub nacp: Option<nacp::Nacp>,
    /// `PackagedContentMetaHeader::Version` of the content meta that supplied
    /// the program, or 0 when the dump has no CNMT.
    pub title_version: u32,
}

/// The extra ExeFS entries `read_source` carries for the runtime. They are
/// metadata, not modules: `from_exefs` consumes them and never maps them.
pub const NACP_FILE: &str = "control.nacp";
pub const CNMT_FILE: &str = "content.cnmt";

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
        let module_id = nso::Header::parse(bytes)?.module_id;
        base = align_up(image.end(), 0x1000);
        modules.push(Module {
            name: name.to_string(),
            image,
            module_id,
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
    let arch = if npdm.is_some_and(npdm_is_aarch32)
        || modules
            .iter()
            .find(|m| m.name == "main")
            .is_some_and(|m| looks_aarch32(&m.image))
    {
        Arch::A32
    } else {
        Arch::A64
    };
    let main_stack_size = npdm.and_then(|b| npdm_u32(b, 0x1c));
    let system_resource_size = npdm.and_then(|b| npdm_u32(b, 0x14)).unwrap_or(0);
    let nacp = files
        .iter()
        .find(|(n, _)| n == NACP_FILE)
        .and_then(|(_, b)| match nacp::Nacp::parse(b) {
            Ok(nacp) => Some(nacp),
            Err(e) => {
                eprintln!("kirikae: the dump's {NACP_FILE} cannot be read ({e}); control data is unavailable");
                None
            }
        });
    let title_version = files
        .iter()
        .find(|(n, _)| n == CNMT_FILE)
        .and_then(|(_, b)| cnmt::Cnmt::parse(b).ok())
        .map_or(0, |meta| meta.version);
    Ok(Executable {
        modules,
        entry,
        arch,
        title_id,
        main_stack_size,
        system_resource_size,
        romfs,
        nacp,
        title_version,
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
            module_id: [0; 32],
        }],
        entry,
        arch: Arch::A64,
        title_id: None,
        main_stack_size: None,
        system_resource_size: 0,
        romfs: None,
        nacp: None,
        title_version: 0,
    }
}

pub fn exefs_from_nsp(path: &Path) -> Result<(Files, u64, Option<RomFs>), Error> {
    let mut r = BufReader::new(File::open(path)?);
    let entries = pfs0::parse(&mut r, 0)?;
    for entry in &entries {
        if entry.name.ends_with(".ncz") {
            return Err(Error::Encrypted);
        }
    }
    // The content metas say which program NCA belongs to which title version
    // and where the Control content is. Highest version first; at the same
    // version the Application meta (type 1) wins over a Patch (type 2), because
    // an update's RomFS is BKTR and cannot be read yet.
    let mut metas = Vec::new();
    for entry in entries.iter().filter(|e| e.name.ends_with(".cnmt.nca")) {
        let Ok(nca) = nca::parse(&mut r, entry.offset) else {
            continue;
        };
        if nca.content_type != nca::CONTENT_META {
            continue;
        }
        let Some(bytes) = meta_file(&mut r, &nca) else {
            continue;
        };
        let Ok(meta) = cnmt::Cnmt::parse(&bytes) else {
            continue;
        };
        metas.push((meta, bytes));
    }
    metas.sort_by(|a, b| {
        b.0.version
            .cmp(&a.0.version)
            .then(a.0.meta_type.cmp(&b.0.meta_type))
    });
    for (meta, bytes) in &metas {
        let Some(content) = meta.content(cnmt::CONTENT_PROGRAM) else {
            continue;
        };
        let Some(entry) = find_content(&entries, &content.id) else {
            continue;
        };
        let Ok(nca) = nca::parse(&mut r, entry.offset) else {
            continue;
        };
        if let Ok(Some((files, title, romfs))) = program(&mut r, nca, path) {
            let nacp = control_file(&mut r, &entries, Some(meta));
            return Ok(with_metadata(files, title, romfs, Some(bytes), nacp));
        }
    }
    // A package without a usable CNMT (or with only an encrypted update) keeps
    // the old content scan.
    for entry in &entries {
        if !entry.name.ends_with(".nca") || entry.name.contains(".cnmt.") {
            continue;
        }
        let nca = nca::parse(&mut r, entry.offset)?;
        if let Some(found) = program(&mut r, nca, path)? {
            let nacp = control_file(&mut r, &entries, None);
            let bytes = metas.first().map(|(_, b)| b);
            return Ok(with_metadata(found.0, found.1, found.2, bytes, nacp));
        }
    }
    Err(Error::Malformed("no program NCA with an ExeFS"))
}

/// Adds the metadata the runtime answers `ns`/`arp`/`am` from to the ExeFS
/// file list. `pack`/`unpack` carry them through unchanged and
/// [`from_exefs`] consumes them.
fn with_metadata(
    mut files: Files,
    title: u64,
    romfs: Option<RomFs>,
    cnmt: Option<&Vec<u8>>,
    nacp: Option<Vec<u8>>,
) -> (Files, u64, Option<RomFs>) {
    if let Some(bytes) = cnmt {
        files.push((CNMT_FILE.to_string(), bytes.clone()));
    }
    if let Some(bytes) = nacp {
        files.push((NACP_FILE.to_string(), bytes));
    }
    (files, title, romfs)
}

/// The `.cnmt` file inside a Meta NCA's PFS0 section.
fn meta_file<R: Read + Seek>(r: &mut R, nca: &nca::Nca) -> Option<Vec<u8>> {
    for section in nca.sections.iter().filter(|s| s.fs_type == nca::FS_PFS0) {
        if nca.check_plaintext(section).is_err() {
            continue;
        }
        let files = pfs0::parse(&mut *r, section.data_offset).ok()?;
        if let Some(entry) = files.iter().find(|f| f.name.ends_with(".cnmt")) {
            return pfs0::read_entry(&mut *r, entry).ok();
        }
    }
    None
}

/// The `control.nacp` of the Control content, from the selected meta when it
/// lists one, otherwise from the first Control NCA in the package.
fn control_file<R: Read + Seek>(
    r: &mut R,
    entries: &[pfs0::Entry],
    meta: Option<&cnmt::Cnmt>,
) -> Option<Vec<u8>> {
    let candidates = meta
        .and_then(|m| m.content(cnmt::CONTENT_CONTROL))
        .and_then(|c| find_content(entries, &c.id));
    for entry in candidates.into_iter().chain(
        entries
            .iter()
            .filter(|e| e.name.ends_with(".nca") && !e.name.contains(".cnmt.")),
    ) {
        let Ok(nca) = nca::parse(&mut *r, entry.offset) else {
            continue;
        };
        if nca.content_type != nca::CONTENT_CONTROL {
            continue;
        }
        for section in nca.sections.iter().filter(|s| s.fs_type == nca::FS_PFS0) {
            if nca.check_plaintext(section).is_err() {
                continue;
            }
            let Ok(files) = pfs0::parse(&mut *r, section.data_offset) else {
                continue;
            };
            if let Some(file) = files.iter().find(|f| f.name == NACP_FILE) {
                return pfs0::read_entry(&mut *r, file).ok();
            }
        }
        break;
    }
    None
}

/// The package entry whose name is the content id, `{id}.nca`.
fn find_content<'a>(entries: &'a [pfs0::Entry], id: &[u8; 16]) -> Option<&'a pfs0::Entry> {
    let hex = cnmt::content_id_hex(id);
    entries
        .iter()
        .find(|e| e.name.eq_ignore_ascii_case(&format!("{hex}.nca")))
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
        // An extracted dump may keep the control data next to the ExeFS or in
        // the dump root. The NSP path puts the same two entries in `files` from
        // the Control and Meta NCAs.
        for name in [NACP_FILE, CNMT_FILE] {
            let found = [
                exefs.join(name),
                base.as_ref().map(|b| b.join(name)).unwrap_or_default(),
            ]
            .into_iter()
            .find(|p| p.is_file());
            if let Some(path) = found {
                files.push((name.to_string(), std::fs::read(path)?));
            }
        }
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

/// Loads a program for recompilation. A32 titles are refused here because the
/// A64 lifter and both backends cannot see their code; the interpreter path
/// uses [`open`] and keeps the architecture tag instead.
pub fn from_source(source: &Source) -> Result<Executable, Error> {
    let exe = from_source_any(source)?;
    if exe.arch.is_32_bit() {
        return Err(Error::Unsupported(
            "this title's main module is AArch32 (32-bit); kirikae only recompiles AArch64",
        ));
    }
    Ok(exe)
}

/// [`from_source`] without the architecture check: tools and the interpreter
/// accept A32 programs and read [`Executable::arch`] to decide how to run them.
pub fn from_source_any(source: &Source) -> Result<Executable, Error> {
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
    from_source_any(&read_source(path)?)
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

    /// A minimal uncompressed NSO whose text is `code` and whose data segment
    /// holds nothing. Enough for `nso::load` and the architecture tag.
    fn nso_with(code: &[u32]) -> Vec<u8> {
        let text_size = (code.len() * 4) as u32;
        let data_mem = text_size;
        let data_size = 4u32;
        let file_text = 0x100u32;
        let file_data = file_text + text_size;
        let mut file = vec![0u8; (file_data + data_size) as usize];
        file[..4].copy_from_slice(b"NSO0");
        let mut segment = |index: usize, file_offset: u32, mem_offset: u32, size: u32| {
            let at = 0x10 + index * 0x10;
            file[at..at + 4].copy_from_slice(&file_offset.to_le_bytes());
            file[at + 4..at + 8].copy_from_slice(&mem_offset.to_le_bytes());
            file[at + 8..at + 12].copy_from_slice(&size.to_le_bytes());
        };
        segment(0, file_text, 0, text_size);
        segment(1, file_data, data_mem, 0);
        segment(2, file_data, data_mem, data_size);
        for (i, word) in code.iter().enumerate() {
            let at = file_text as usize + i * 4;
            file[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        file
    }

    /// A minimal META header with only `Flags0` set; enough for the
    /// architecture tag (`npdm_title` returns `None` without an ACI0).
    fn npdm_with_meta_flags(flags: u32) -> Vec<u8> {
        let mut npdm = vec![0u8; 0x80];
        npdm[..4].copy_from_slice(b"META");
        npdm[0x0c..0x10].copy_from_slice(&flags.to_le_bytes());
        npdm[0x78..0x7C].copy_from_slice(&0x80u32.to_le_bytes());
        npdm
    }

    #[test]
    fn tags_aarch32_programs_but_only_runs_them_through_the_interpreter() {
        let files: Vec<(String, Vec<u8>)> = vec![
            ("main".into(), nso_with(&[0xE12F_FF1E])),
            ("main.npdm".into(), npdm_with_meta_flags(0x04)),
        ];
        let exe = from_exefs(&files, None, None).expect("A32 loads a tagged program");
        assert_eq!(exe.arch, Arch::A32);
        assert!(exe.arch.is_32_bit());
        assert_eq!(exe.arch.name(), "AArch32");
        assert_eq!(exe.modules.len(), 1);

        let source = Source::ExeFs(files, None, None);
        assert!(matches!(
            from_source(&source),
            Err(Error::Unsupported(
                "this title's main module is AArch32 (32-bit); kirikae only recompiles AArch64"
            ))
        ));
        assert_eq!(
            from_source_any(&source)
                .expect("the interpreter path loads it")
                .arch,
            Arch::A32
        );
    }

    #[test]
    fn tags_aarch64_programs_by_default() {
        let files: Vec<(String, Vec<u8>)> = vec![
            ("main".into(), nso_with(&[0xD65F_03C0])),
            ("main.npdm".into(), npdm_with_meta_flags(0x07)),
        ];
        let exe = from_source(&Source::ExeFs(files, None, None)).expect("A64 builds");
        assert_eq!(exe.arch, Arch::A64);
        assert_eq!(exe.arch.name(), "AArch64");
    }

    const APP_ID: u64 = 0x0100_0000_0000_0000;

    /// A PFS0 with `files` in order, for the contents of an NCA section.
    fn pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut table = Vec::new();
        let mut strings = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, bytes) in files {
            table.extend((data.len() as u64).to_le_bytes());
            table.extend((bytes.len() as u64).to_le_bytes());
            table.extend((strings.len() as u32).to_le_bytes());
            table.extend(0u32.to_le_bytes());
            strings.extend(name.as_bytes());
            strings.push(0);
            data.extend_from_slice(bytes);
        }
        let mut out = b"PFS0".to_vec();
        out.extend((files.len() as u32).to_le_bytes());
        out.extend((strings.len() as u32).to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(table);
        out.extend(strings);
        out.extend(data);
        out
    }

    /// A decrypted NCA ("DNCA") with one plaintext PFS0 section at 0x1000 and
    /// no hash layers.
    fn nca(content_type: u8, title: u64, contents: &[(&str, &[u8])]) -> Vec<u8> {
        let section = pfs0(contents);
        let start = 0x1000usize;
        let end = (start + section.len()).next_multiple_of(0x200);
        let mut out = vec![0u8; end];
        out[0x200..0x204].copy_from_slice(b"DNCA");
        out[0x205] = content_type;
        out[0x210..0x218].copy_from_slice(&title.to_le_bytes());
        out[0x240..0x244].copy_from_slice(&((start / 0x200) as u32).to_le_bytes());
        out[0x244..0x248].copy_from_slice(&((end / 0x200) as u32).to_le_bytes());
        let fs = 0x400;
        out[fs + 2] = nca::FS_PFS0;
        out[fs + 3] = 2; // HierarchicalSha256Hash
        out[fs + 4] = nca::CRYPTO_NONE;
        out[fs + 0x2c..fs + 0x30].copy_from_slice(&1u32.to_le_bytes());
        out[fs + 0x30..fs + 0x38].copy_from_slice(&0u64.to_le_bytes());
        out[fs + 0x38..fs + 0x40].copy_from_slice(&(section.len() as u64).to_le_bytes());
        out[start..start + section.len()].copy_from_slice(&section);
        out
    }

    fn cnmt(meta_type: u8, version: u32, application: u64, contents: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; 0x20];
        out[..8].copy_from_slice(&(APP_ID | 0x800).to_le_bytes());
        out[8..12].copy_from_slice(&version.to_le_bytes());
        out[0xc] = meta_type;
        out[0xe..0x10].copy_from_slice(&0x10u16.to_le_bytes());
        out[0x10..0x12].copy_from_slice(&(contents.len() as u16 / 0x38).to_le_bytes());
        let mut ext = vec![0u8; 0x10];
        ext[..8].copy_from_slice(&application.to_le_bytes());
        out.extend(ext);
        out.extend(contents);
        out
    }

    fn content_entry(id: u8, content_type: u8) -> Vec<u8> {
        let mut entry = vec![0u8; 0x38];
        entry[0x20..0x30].fill(id);
        entry[0x36] = content_type;
        entry
    }

    fn nacp_bytes(version: &str) -> Vec<u8> {
        let mut bytes = vec![0u8; crate::nacp::NACP_SIZE];
        bytes[0x3060..0x3060 + version.len()].copy_from_slice(version.as_bytes());
        bytes
    }

    fn write_package(bytes: &[u8]) -> PathBuf {
        let unique = format!(
            "kirikae-package-{}-{}.nsp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::write(&path, bytes).expect("the synthetic package is writable");
        path
    }

    #[test]
    fn a_package_selects_the_newest_content_meta_and_reads_its_control_nacp() {
        let base_id = 0x11u8;
        let update_id = 0x22u8;
        let control_id = 0x33u8;
        let base = nca(
            nca::CONTENT_PROGRAM,
            APP_ID,
            &[
                ("main", &nso_with(&[0x1111_0000, 0xD65F_03C0])),
                ("main.npdm", &npdm_with_meta_flags(0x07)),
            ],
        );
        let update = nca(
            nca::CONTENT_PROGRAM,
            APP_ID,
            &[
                ("main", &nso_with(&[0x2222_0000, 0xD65F_03C0])),
                ("main.npdm", &npdm_with_meta_flags(0x07)),
            ],
        );
        let control = nca(
            nca::CONTENT_CONTROL,
            APP_ID,
            &[("control.nacp", &nacp_bytes("2.3.4"))],
        );
        let base_meta = cnmt(1, 0, APP_ID, &content_entry(base_id, 0));
        let update_contents = [content_entry(update_id, 0), content_entry(control_id, 2)].concat();
        let update_meta = cnmt(2, 0x2_0000, APP_ID, &update_contents);
        let base_meta_nca = nca(nca::CONTENT_META, APP_ID, &[("base.cnmt", &base_meta)]);
        let update_meta_nca = nca(nca::CONTENT_META, APP_ID, &[("update.cnmt", &update_meta)]);
        // The update is listed *after* the base NCA: file order alone would
        // pick the base, the content meta picks version 0x20000.
        let base_name = format!("{base_id:02x}").repeat(16) + ".nca";
        let update_name = format!("{update_id:02x}").repeat(16) + ".nca";
        let control_name = format!("{control_id:02x}").repeat(16) + ".nca";
        let base_meta_name = "aa".repeat(16) + ".cnmt.nca";
        let update_meta_name = "bb".repeat(16) + ".cnmt.nca";
        let entries: Vec<(&str, &[u8])> = vec![
            (base_name.as_str(), base.as_slice()),
            (base_meta_name.as_str(), base_meta_nca.as_slice()),
            (update_name.as_str(), update.as_slice()),
            (update_meta_name.as_str(), update_meta_nca.as_slice()),
            (control_name.as_str(), control.as_slice()),
        ];
        let package = pfs0(&entries);
        let path = write_package(&package);
        let result = exefs_from_nsp(&path);
        let _ = std::fs::remove_file(&path);
        let (files, title, romfs) = result.expect("the synthetic package loads");
        assert_eq!(title, APP_ID);
        let exe = from_exefs(&files, Some(title), romfs).expect("the selected program loads");
        assert_eq!(exe.title_version, 0x2_0000, "the CNMT version is carried");
        assert_eq!(
            exe.nacp.as_ref().map(|n| n.display_version.as_str()),
            Some("2.3.4")
        );
        let main = exe.modules.iter().find(|m| m.name == "main").unwrap();
        assert!(
            main.image
                .memory
                .windows(4)
                .any(|w| w == 0x2222_0000u32.to_le_bytes()),
            "the update's program NCA supplied the code"
        );
        assert!(
            !main
                .image
                .memory
                .windows(4)
                .any(|w| w == 0x1111_0000u32.to_le_bytes()),
            "the base program was not selected"
        );
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
