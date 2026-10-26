use crate::Error;

pub const HEADER_SIZE: usize = 0x20;
pub const CONTENT_INFO_SIZE: usize = 0x38;

pub const META_APPLICATION: u8 = 0x80;
pub const META_PATCH: u8 = 0x81;
pub const META_ADD_ON_CONTENT: u8 = 0x82;
pub const META_DELTA: u8 = 0x83;
pub const META_DATA_PATCH: u8 = 0x84;

pub const CONTENT_META: u8 = 0;
pub const CONTENT_PROGRAM: u8 = 1;
pub const CONTENT_DATA: u8 = 2;
pub const CONTENT_CONTROL: u8 = 3;
pub const CONTENT_HTML_DOCUMENT: u8 = 4;
pub const CONTENT_LEGAL_INFORMATION: u8 = 5;
pub const CONTENT_DELTA_FRAGMENT: u8 = 6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentInfo {
    pub id: [u8; 16],
    pub content_type: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cnmt {
    pub id: u64,
    pub version: u32,
    pub meta_type: u8,
    pub application_id: Option<u64>,
    pub patch_id: Option<u64>,
    pub required_system_version: Option<u32>,
    pub contents: Vec<ContentInfo>,
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap_or_default())
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default())
}

impl Cnmt {
    pub fn parse(bytes: &[u8]) -> Result<Cnmt, Error> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::Malformed(
                "CNMT is shorter than its 0x20-byte header",
            ));
        }
        let id = u64_at(bytes, 0);
        let version = u32_at(bytes, 8);
        let meta_type = bytes[0xc];
        let extended = u16_at(bytes, 0xe) as usize;
        let content_count = u16_at(bytes, 0x10) as usize;
        let (application_id, patch_id, required_system_version) = match meta_type {
            META_APPLICATION => {
                let ext = bytes
                    .get(HEADER_SIZE..HEADER_SIZE + extended.max(0xc))
                    .ok_or(Error::Malformed("CNMT application header is truncated"))?;
                (Some(id), Some(u64_at(ext, 0)), Some(u32_at(ext, 8)))
            }
            META_PATCH => {
                let ext = bytes
                    .get(HEADER_SIZE..HEADER_SIZE + extended.max(0xc))
                    .ok_or(Error::Malformed("CNMT patch header is truncated"))?;
                (Some(u64_at(ext, 0)), None, Some(u32_at(ext, 8)))
            }
            META_ADD_ON_CONTENT => {
                let ext = bytes
                    .get(HEADER_SIZE..HEADER_SIZE + extended.max(0xc))
                    .ok_or(Error::Malformed("CNMT add-on-content header is truncated"))?;
                (Some(u64_at(ext, 0)), None, None)
            }
            _ => (None, None, None),
        };
        let entries_at = HEADER_SIZE + extended;
        let entries_len = content_count
            .checked_mul(CONTENT_INFO_SIZE)
            .ok_or(Error::Malformed("CNMT content count overflows"))?;
        let entries = bytes
            .get(entries_at..entries_at + entries_len)
            .ok_or(Error::Malformed("CNMT content entries run past the file"))?;
        let mut contents = Vec::with_capacity(content_count);
        for entry in entries.as_chunks::<CONTENT_INFO_SIZE>().0 {
            let mut content_id = [0u8; 16];
            content_id.copy_from_slice(&entry[0x20..0x30]);
            contents.push(ContentInfo {
                id: content_id,
                content_type: entry[0x36],
            });
        }
        Ok(Cnmt {
            id,
            version,
            meta_type,
            application_id,
            patch_id,
            required_system_version,
            contents,
        })
    }

    pub fn content(&self, content_type: u8) -> Option<&ContentInfo> {
        self.contents
            .iter()
            .find(|c| c.content_type == content_type)
    }
}

pub fn content_id_hex(id: &[u8; 16]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(32);
    for byte in id {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cnmt(meta_type: u8, version: u32, ext: &[u8], contents: &[(u8, u8)]) -> Vec<u8> {
        let mut header = vec![0u8; HEADER_SIZE];
        header[..8].copy_from_slice(&0x0100_0000_0000_0800u64.to_le_bytes());
        header[8..12].copy_from_slice(&version.to_le_bytes());
        header[0xc] = meta_type;
        header[0xe..0x10].copy_from_slice(&(ext.len() as u16).to_le_bytes());
        header[0x10..0x12].copy_from_slice(&(contents.len() as u16).to_le_bytes());
        let mut out = header;
        out.extend_from_slice(ext);
        for (i, (id, content_type)) in contents.iter().enumerate() {
            let mut entry = vec![0u8; CONTENT_INFO_SIZE];
            entry[0..0x20].fill(0x55);
            entry[0x20..0x30].fill(*id);
            entry[0x30..0x36].copy_from_slice(&(0x1000u64 * (i as u64 + 1)).to_le_bytes()[..6]);
            entry[0x36] = *content_type;
            entry[0x37] = i as u8;
            out.extend(entry);
        }
        out
    }

    #[test]
    fn reads_the_patch_meta_and_its_program_content() {
        let mut ext = vec![0u8; 0x10];
        ext[..8].copy_from_slice(&0x0100_0000_0000_0000u64.to_le_bytes());
        ext[8..12].copy_from_slice(&0x0008_0000u32.to_le_bytes());
        let bytes = cnmt(
            META_PATCH,
            0x1_0000,
            &ext,
            &[(0x11, CONTENT_PROGRAM), (0x22, CONTENT_CONTROL)],
        );
        let meta = Cnmt::parse(&bytes).expect("a patch CNMT parses");
        assert_eq!(meta.id, 0x0100_0000_0000_0800);
        assert_eq!(meta.version, 0x1_0000);
        assert_eq!(meta.meta_type, META_PATCH);
        assert_eq!(meta.application_id, Some(0x0100_0000_0000_0000));
        assert_eq!(meta.required_system_version, Some(0x0008_0000));
        assert_eq!(meta.contents.len(), 2);
        assert_eq!(meta.content(CONTENT_PROGRAM).map(|c| c.id[0]), Some(0x11));
        assert_eq!(
            content_id_hex(&meta.content(CONTENT_PROGRAM).unwrap().id),
            "11111111111111111111111111111111"
        );
    }

    #[test]
    fn an_application_meta_is_its_own_application_and_names_its_patch() {
        let mut ext = vec![0u8; 0xc];
        ext[..8].copy_from_slice(&0x0100_0000_0000_0800u64.to_le_bytes());
        ext[8..12].copy_from_slice(&0u32.to_le_bytes());
        let mut bytes = cnmt(META_APPLICATION, 0, &ext, &[(0x33, CONTENT_PROGRAM)]);
        bytes[..8].copy_from_slice(&0x0100_0000_0000_0000u64.to_le_bytes());
        let meta = Cnmt::parse(&bytes).unwrap();
        assert_eq!(meta.application_id, Some(0x0100_0000_0000_0000));
        assert_eq!(meta.patch_id, Some(0x0100_0000_0000_0800));
        assert_eq!(meta.version, 0);
        assert_eq!(meta.content(CONTENT_PROGRAM).unwrap().id, [0x33; 16]);
    }

    #[test]
    fn the_type_values_are_the_documented_ones() {
        assert_eq!((META_APPLICATION, META_PATCH, META_ADD_ON_CONTENT), (0x80, 0x81, 0x82));
        assert_eq!((CONTENT_META, CONTENT_PROGRAM, CONTENT_DATA, CONTENT_CONTROL), (0, 1, 2, 3));
    }

    #[test]
    fn a_truncated_meta_is_refused() {
        assert!(matches!(
            Cnmt::parse(&[0u8; 8]),
            Err(Error::Malformed(
                "CNMT is shorter than its 0x20-byte header"
            ))
        ));
        let mut bytes = cnmt(META_APPLICATION, 0, &[0u8; 0xc], &[(1, CONTENT_PROGRAM)]);
        bytes.truncate(HEADER_SIZE + 0xc + 0x10);
        assert!(matches!(
            Cnmt::parse(&bytes),
            Err(Error::Malformed("CNMT content entries run past the file"))
        ));
    }
}
