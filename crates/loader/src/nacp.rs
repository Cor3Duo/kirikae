//! `control.nacp` — the `nn::ns::ApplicationControlProperty` stored in the
//! Control content (NCA content type 2).
//!
//! Offsets and language order follow Switchbrew's NACP page. The file is a
//! fixed 0x4000-byte struct; only the fields the runtime can answer from a
//! decrypted dump are decoded, and the raw bytes are kept so a service can hand
//! the whole struct to the guest unchanged. Nothing here decrypts, verifies a
//! signature or touches a key.

use crate::Error;

pub const NACP_SIZE: usize = 0x4000;
/// Number of `ApplicationTitle` language entries in the 0x3000-byte table.
pub const LANGUAGE_COUNT: usize = 16;
const LANGUAGE_ENTRY_SIZE: usize = 0x300;
const NAME_SIZE: usize = 0x200;
const PUBLISHER_SIZE: usize = 0x100;

const ISBN: usize = 0x3000;
const DISPLAY_VERSION: usize = 0x3060;
const ADD_ON_CONTENT_BASE_ID: usize = 0x3070;
const SAVE_DATA_OWNER_ID: usize = 0x3078;
const USER_ACCOUNT_SAVE_DATA_SIZE: usize = 0x3080;
const USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE: usize = 0x3088;
const DEVICE_SAVE_DATA_SIZE: usize = 0x3090;
const DEVICE_SAVE_DATA_JOURNAL_SIZE: usize = 0x3098;

/// The decoded fields of `control.nacp`. `raw` is the complete 0x4000-byte
/// struct, which is what `GetApplicationControlData` and the two
/// `GetApplicationControlProperty` commands return.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nacp {
    pub raw: Vec<u8>,
    pub display_version: String,
    pub add_on_content_base_id: u64,
    pub save_data_owner_id: u64,
    pub user_account_save_data_size: u64,
    pub user_account_save_data_journal_size: u64,
    pub device_save_data_size: u64,
    pub device_save_data_journal_size: u64,
    /// Bit per language entry that has at least one non-empty string, which is
    /// what `GetApplicationDesiredLanguage` takes as its input.
    pub supported_language_flag: u32,
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default())
}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn language_entry(language: usize) -> Option<usize> {
    (language < LANGUAGE_COUNT).then_some(language * LANGUAGE_ENTRY_SIZE)
}

impl Nacp {
    pub fn parse(bytes: &[u8]) -> Result<Nacp, Error> {
        if bytes.len() < NACP_SIZE {
            return Err(Error::Malformed(
                "control.nacp is shorter than 0x4000 bytes",
            ));
        }
        let language_flag = (0..LANGUAGE_COUNT)
            .filter(|i| {
                let at = language_entry(*i).unwrap_or(0);
                let name = &bytes[at..at + NAME_SIZE];
                let publisher = &bytes[at + NAME_SIZE..at + LANGUAGE_ENTRY_SIZE];
                name.iter().any(|&b| b != 0) || publisher.iter().any(|&b| b != 0)
            })
            .fold(0u32, |mask, i| mask | 1 << i);
        Ok(Nacp {
            raw: bytes[..NACP_SIZE].to_vec(),
            display_version: c_string(&bytes[DISPLAY_VERSION..DISPLAY_VERSION + 0x10]),
            add_on_content_base_id: u64_at(bytes, ADD_ON_CONTENT_BASE_ID),
            save_data_owner_id: u64_at(bytes, SAVE_DATA_OWNER_ID),
            user_account_save_data_size: u64_at(bytes, USER_ACCOUNT_SAVE_DATA_SIZE),
            user_account_save_data_journal_size: u64_at(bytes, USER_ACCOUNT_SAVE_DATA_JOURNAL_SIZE),
            device_save_data_size: u64_at(bytes, DEVICE_SAVE_DATA_SIZE),
            device_save_data_journal_size: u64_at(bytes, DEVICE_SAVE_DATA_JOURNAL_SIZE),
            supported_language_flag: language_flag,
        })
    }

    pub fn isbn(&self) -> String {
        c_string(&self.raw[ISBN..ISBN + 0x25])
    }

    /// The application name in language entry `language`, or `""` when the
    /// entry is out of range or empty.
    pub fn name(&self, language: usize) -> String {
        let Some(at) = language_entry(language) else {
            return String::new();
        };
        c_string(&self.raw[at..at + NAME_SIZE])
    }

    /// The publisher in language entry `language`.
    pub fn publisher(&self, language: usize) -> String {
        let Some(at) = language_entry(language) else {
            return String::new();
        };
        c_string(&self.raw[at + NAME_SIZE..at + NAME_SIZE + PUBLISHER_SIZE])
    }

    /// The raw language entry (name and publisher) the guest expects in a
    /// `NacpLanguageEntry`.
    pub fn language_entry(&self, language: usize) -> &[u8] {
        match language_entry(language) {
            Some(at) => &self.raw[at..at + LANGUAGE_ENTRY_SIZE],
            None => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> Vec<u8> {
        let mut bytes = vec![0u8; NACP_SIZE];
        for language in [0usize, 1, 2] {
            let at = language * LANGUAGE_ENTRY_SIZE;
            let name = match language {
                0 => "Sample",
                1 => "Sample (UK)",
                _ => "サンプル",
            };
            bytes[at..at + name.len()].copy_from_slice(name.as_bytes());
            bytes[at + NAME_SIZE..at + NAME_SIZE + 4].copy_from_slice(b"Pub!");
        }
        bytes[DISPLAY_VERSION..DISPLAY_VERSION + 5].copy_from_slice(b"1.2.3");
        bytes[ADD_ON_CONTENT_BASE_ID..ADD_ON_CONTENT_BASE_ID + 8]
            .copy_from_slice(&0x0100_0000_1000_0000u64.to_le_bytes());
        bytes[SAVE_DATA_OWNER_ID..SAVE_DATA_OWNER_ID + 8]
            .copy_from_slice(&0x0100_0000_0001_0000u64.to_le_bytes());
        bytes[USER_ACCOUNT_SAVE_DATA_SIZE..USER_ACCOUNT_SAVE_DATA_SIZE + 8]
            .copy_from_slice(&0x20_0000u64.to_le_bytes());
        bytes[DEVICE_SAVE_DATA_JOURNAL_SIZE..DEVICE_SAVE_DATA_JOURNAL_SIZE + 8]
            .copy_from_slice(&0x10_0000u64.to_le_bytes());
        bytes
    }

    #[test]
    fn reads_the_documented_nacp_fields() {
        let bytes = synthetic();
        let nacp = Nacp::parse(&bytes).expect("a full control.nacp parses");
        assert_eq!(nacp.display_version, "1.2.3");
        assert_eq!(nacp.add_on_content_base_id, 0x0100_0000_1000_0000);
        assert_eq!(nacp.save_data_owner_id, 0x0100_0000_0001_0000);
        assert_eq!(nacp.user_account_save_data_size, 0x20_0000);
        assert_eq!(nacp.device_save_data_journal_size, 0x10_0000);
        assert_eq!(nacp.name(0), "Sample");
        assert_eq!(nacp.name(1), "Sample (UK)");
        assert_eq!(nacp.publisher(2), "Pub!");
        assert_eq!(nacp.name(LANGUAGE_COUNT), "");
        assert_eq!(
            nacp.supported_language_flag, 0b111,
            "the three non-empty entries"
        );
        assert_eq!(nacp.raw.len(), NACP_SIZE);
        assert_eq!(&nacp.raw[..6], b"Sample");
    }

    #[test]
    fn a_short_nacp_is_refused() {
        assert!(matches!(
            Nacp::parse(&[0u8; 0x100]),
            Err(Error::Malformed(
                "control.nacp is shorter than 0x4000 bytes"
            ))
        ));
    }
}
