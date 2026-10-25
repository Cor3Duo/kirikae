//! `ns:am` / `ns:am2` — the application launch and record queries.
//!
//! The runtime runs one application in its own process, so the honest answer to
//! "which applications are installed / running" is a one-entry list for the
//! current title. Queries that need the content archive (`GetApplicationControlData`,
//! content paths, update tasks) return a documented error instead of fabricated
//! data: the loader does not expose other titles' NCAs.
//!
//! Command numbers and layouts follow Switchbrew's "NS services" page and
//! libnx's `ns.h`.

use std::sync::Arc;

use crate::cpu::Cpu;
use crate::services::stub::unimplemented_named;
use crate::services::{Reply, Request, Service};

/// `NsApplicationRecord` is 0x18 bytes.
const APPLICATION_RECORD_SIZE: usize = 0x18;
/// `ResultInvalidArgument` (module 1): the request names something this runtime
/// cannot provide, such as another title's content archive.
const INVALID_ARGUMENT: u32 = 0x2EE202;
/// `nn::ns` "language code not found", documented by Switchbrew
/// (`ConvertLanguageCodeToApplicationLanguage`).
const LANGUAGE_CODE_NOT_FOUND: u32 = 0x25810;

/// NACP language indices, in `control.nacp` order.
const LANG_AMERICAN_ENGLISH: u8 = 0;
const LANG_BRITISH_ENGLISH: u8 = 1;
const LANG_JAPANESE: u8 = 2;
const LANG_FRENCH: u8 = 3;
const LANG_GERMAN: u8 = 4;
const LANG_LATIN_AMERICAN_SPANISH: u8 = 5;
const LANG_SPANISH: u8 = 6;
const LANG_ITALIAN: u8 = 7;
const LANG_DUTCH: u8 = 8;
const LANG_CANADIAN_FRENCH: u8 = 9;
const LANG_PORTUGUESE: u8 = 10;
const LANG_RUSSIAN: u8 = 11;
const LANG_KOREAN: u8 = 12;
const LANG_TRADITIONAL_CHINESE: u8 = 13;
const LANG_SIMPLIFIED_CHINESE: u8 = 14;
const LANG_BRAZILIAN_PORTUGUESE: u8 = 15;

pub struct Ns {
    name: String,
}

impl Ns {
    pub fn new(name: &str) -> Ns {
        Ns {
            name: name.to_string(),
        }
    }
}

impl Service for Ns {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        // ns:am predates the getter interface and is the manager itself.
        if self.name == "ns:am" {
            return manager(c, req);
        }
        match req.cmd {
            // 7996 GetApplicationManagerInterface. The other getters expose
            // interfaces (download tasks, e-commerce, factory reset) whose state
            // this runtime does not keep; they are refused by name.
            7996 => Reply::ok().object(Arc::new(ApplicationManager {
                title: c.process().title_id,
            })),
            _ => unimplemented_named(&self.name, getter_command_name(req.cmd), c, req),
        }
    }
}

/// `nn::ns::detail::IApplicationManagerInterface`.
struct ApplicationManager {
    title: u64,
}

impl ApplicationManager {
    fn reply(&self, c: &mut Cpu, req: &Request) -> Option<Reply> {
        let p = c.process();
        match req.cmd {
            // ListApplicationRecord: one record, the running application, at
            // entry_offset 0; past it the list is empty (the SDK loops until the
            // count is zero).
            0 => {
                if req.u32(0) != 0 {
                    return Some(Reply::ok().u32(0));
                }
                let mut record = [0u8; APPLICATION_RECORD_SIZE];
                record[..8].copy_from_slice(&self.title.to_le_bytes());
                // last_event = NsShellEvent_None, attributes = 0, last_updated = 0.
                if let Some(b) = req.b.first() {
                    let n = (b.size as usize).min(APPLICATION_RECORD_SIZE);
                    p.memory.write(b.addr, &record[..n]);
                }
                Some(Reply::ok().u32(1))
            }
            // GenerateApplicationRecordCount
            1 => Some(Reply::ok().u32(1)),
            // GetApplicationDesiredLanguage: the first non-empty language entry.
            55 => Some(Reply::ok().u8(desired_language(req.data.first().copied().unwrap_or(0)))),
            // CheckApplicationLaunchVersion: success only for the running title.
            38 => Some(if req.u64(0) == self.title {
                Reply::ok()
            } else {
                Reply::err(INVALID_ARGUMENT)
            }),
            // ConvertLanguageCodeToApplicationLanguage: the index goes through
            // the caller's output pointer; the reply only carries the result.
            60 => Some(match convert_language_code(req.u64(0)) {
                Ok(index) => {
                    if let Some(b) = req.x.first().or_else(|| req.b.first()) {
                        p.memory.write(b.addr, &[index]);
                    }
                    Reply::ok()
                }
                Err(code) => Reply::err(code),
            }),
            // GetApplicationControlData: the runtime has no NACP/icon cache for
            // arbitrary applications.
            400 => Some(Reply::err(INVALID_ARGUMENT)),
            // HasApplicationRecord (5.1.0+)
            910 => Some(Reply::ok().u8(u8::from(req.u64(0) == self.title))),
            // NeedsUpdateVulnerability
            1200 => Some(Reply::ok().u8(0)),
            // IsAnyApplicationEntityInstalled
            1300 => Some(Reply::ok().u8(u8::from(req.u64(0) == self.title))),
            // CountApplicationContentMeta
            600 => Some(Reply::ok().u32(u32::from(req.u64(0) == self.title))),
            _ => None,
        }
    }
}

fn manager(c: &mut Cpu, req: &Request) -> Reply {
    let manager = ApplicationManager {
        title: c.process().title_id,
    };
    match manager.reply(c, req) {
        Some(reply) => reply,
        None => unimplemented_named("ns:am", manager_command_name(req.cmd), c, req),
    }
}

impl Service for ApplicationManager {
    fn name(&self) -> String {
        "ns/IApplicationManagerInterface".into()
    }

    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.reply(c, req) {
            Some(reply) => reply,
            None => unimplemented_named(
                "ns/IApplicationManagerInterface",
                manager_command_name(req.cmd),
                c,
                req,
            ),
        }
    }
}

/// The first language bit set in a `control.nacp` language bitmask. Zero (the
/// American English entry) when the mask is empty, which is what the SDK's own
/// helper does for a title with no localized strings.
fn desired_language(mask: u8) -> u8 {
    (0..8).find(|bit| mask & (1 << bit) != 0).unwrap_or(0)
}

/// Converts the u64-packed language code the SDK sends (little-endian ASCII,
/// e.g. `0x53552d6e65` = "en-US") to a NACP language index.
fn convert_language_code(code: u64) -> Result<u8, u32> {
    let bytes = code.to_le_bytes();
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
    let text = std::str::from_utf8(&bytes[..len]).map_err(|_| LANGUAGE_CODE_NOT_FOUND)?;
    let index = match text.to_ascii_lowercase().as_str() {
        "en-us" => LANG_AMERICAN_ENGLISH,
        "en-gb" => LANG_BRITISH_ENGLISH,
        "ja" | "ja-jp" => LANG_JAPANESE,
        "fr" | "fr-fr" => LANG_FRENCH,
        "de" | "de-de" => LANG_GERMAN,
        "es-419" => LANG_LATIN_AMERICAN_SPANISH,
        "es" | "es-es" => LANG_SPANISH,
        "it" | "it-it" => LANG_ITALIAN,
        "nl" | "nl-nl" => LANG_DUTCH,
        "fr-ca" => LANG_CANADIAN_FRENCH,
        "pt" | "pt-pt" => LANG_PORTUGUESE,
        "ru" | "ru-ru" => LANG_RUSSIAN,
        "ko" | "ko-kr" => LANG_KOREAN,
        "zh-tw" | "zh-hant" => LANG_TRADITIONAL_CHINESE,
        "zh-cn" | "zh-hans" => LANG_SIMPLIFIED_CHINESE,
        "pt-br" => LANG_BRAZILIAN_PORTUGUESE,
        _ => return Err(LANGUAGE_CODE_NOT_FOUND),
    };
    Ok(index)
}

fn getter_command_name(cmd: u32) -> &'static str {
    match cmd {
        7988 => "GetDynamicRightsInterface",
        7989 => "GetReadOnlyApplicationControlDataInterface",
        7991 => "GetReadOnlyApplicationRecordInterface",
        7992 => "GetECommerceInterface",
        7993 => "GetApplicationVersionInterface",
        7994 => "GetFactoryResetInterface",
        7995 => "GetAccountProxyInterface",
        7996 => "GetApplicationManagerInterface",
        7997 => "GetDownloadTaskInterface",
        7998 => "GetContentManagementInterface",
        7999 => "GetDocumentInterface",
        _ => "unknown",
    }
}

fn manager_command_name(cmd: u32) -> &'static str {
    match cmd {
        0 => "ListApplicationRecord",
        1 => "GenerateApplicationRecordCount",
        2 => "GetApplicationRecordUpdateSystemEvent",
        3 => "GetApplicationViewDeprecated",
        4 => "DeleteApplicationEntity",
        5 => "DeleteApplicationCompletely",
        8 => "IsApplicationEntityMovable",
        9 => "MoveApplicationEntity",
        11 => "CalculateApplicationOccupiedSize",
        16 => "PushApplicationRecord",
        17 => "ListApplicationRecordContentMeta",
        19 => "LaunchApplicationOld",
        21 => "GetApplicationContentPath",
        22 => "TerminateApplication",
        23 => "ResolveApplicationContentPath",
        26 => "BeginInstallApplication",
        27 => "DeleteApplicationRecord",
        30 => "RequestApplicationUpdateInfo",
        38 => "CheckApplicationLaunchVersion",
        43 => "CheckSdCardMountStatus",
        47 => "GetTotalSpaceSize",
        48 => "GetFreeSpaceSize",
        55 => "GetApplicationDesiredLanguage",
        56 => "SetApplicationTerminateResult",
        57 => "ClearApplicationTerminateResult",
        60 => "ConvertLanguageCodeToApplicationLanguage",
        91 => "DeprecatedLaunchApplication",
        94 => "LaunchApplication",
        95 => "GetApplicationLaunchInfo",
        96 => "AcquireApplicationLaunchInfo",
        97 => "GetMainApplicationProgramIndexByApplicationLaunchInfo",
        400 => "GetApplicationControlData",
        600 => "CountApplicationContentMeta",
        601 => "ListApplicationContentMetaStatus",
        910 => "HasApplicationRecord",
        1200 => "NeedsUpdateVulnerability",
        1300 => "IsAnyApplicationEntityInstalled",
        1701 => "GetApplicationView",
        1704 => "GetApplicationViewWithPromotionInfo",
        1800 => "IsNotificationSetupCompleted",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Entry;
    use crate::services::harness;

    fn coded(text: &str) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[..text.len()].copy_from_slice(text.as_bytes());
        u64::from_le_bytes(bytes)
    }

    #[test]
    fn manager_replies_match_the_documented_values() {
        const TITLE: u64 = 0x0100_0000_0009_0000;

        extern "C" fn probe(c: &mut Cpu) {
            let call = |c: &mut Cpu, data: Vec<u8>| {
                let mut data = data;
                data.resize(16, 0);
                Ns::new("ns:am").call(
                    c,
                    &Request {
                        cmd: 55,
                        data,
                        ..Default::default()
                    },
                )
            };
            let language = call(c, vec![0b1000]);
            harness::record("language", &[u64::from(language.data[0])]);

            // The language index is written through the output pointer.
            let out = 0x1_2000_0000u64;
            c.process()
                .memory
                .map_as(out, 0x1000, crate::mem::state::NORMAL, crate::mem::perm::RW)
                .unwrap();
            c.process().memory.write(out, &[0xaa]);
            let mut data = coded("en-GB").to_le_bytes().to_vec();
            data.resize(16, 0);
            let index = Ns::new("ns:am").call(
                c,
                &Request {
                    cmd: 60,
                    data,
                    x: vec![crate::services::Buf { addr: out, size: 1 }],
                    ..Default::default()
                },
            );
            let written = c.process().memory.read(out, 1).unwrap()[0];
            harness::record(
                "code",
                &[
                    u64::from(index.result),
                    index.data.len() as u64,
                    u64::from(written),
                ],
            );

            let launch = Ns::new("ns:am").call(
                c,
                &Request {
                    cmd: 38,
                    data: TITLE.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            let other = Ns::new("ns:am").call(
                c,
                &Request {
                    cmd: 38,
                    data: 0x0100_0000_000a_0000u64.to_le_bytes().to_vec(),
                    ..Default::default()
                },
            );
            harness::record(
                "launch",
                &[u64::from(launch.result), u64::from(other.result)],
            );

            // The record list ends after the single entry, so an offset of 1
            // reports zero entries instead of repeating it.
            let record = || Request {
                cmd: 0,
                b: vec![crate::services::Buf {
                    addr: out,
                    size: 0x18,
                }],
                ..Default::default()
            };
            let first = Ns::new("ns:am").call(c, &record());
            let mut past = record();
            past.data = 1u32.to_le_bytes().to_vec();
            let past = Ns::new("ns:am").call(c, &past);
            let id = c.process().memory.read(out, 8).unwrap();
            harness::record(
                "list",
                &[
                    u64::from(first.result),
                    u64::from(u32::from_le_bytes(first.data[..4].try_into().unwrap())),
                    u64::from_le_bytes(id.as_slice().try_into().unwrap()),
                    u64::from(u32::from_le_bytes(past.data[..4].try_into().unwrap())),
                ],
            );

            let update = Ns::new("ns:am").call(
                c,
                &Request {
                    cmd: 1200,
                    data: vec![0; 8],
                    ..Default::default()
                },
            );
            harness::record("update", &[u64::from(update.data[0])]);
            c.halt();
        }
        harness::run_with_title(
            vec![Entry {
                addr: harness::ENTRY,
                func: probe,
            }],
            Some(TITLE),
        );
        assert_eq!(harness::observed("language"), vec![3]);
        assert_eq!(harness::observed("code"), vec![0, 0, 1]);
        assert_eq!(
            harness::observed("launch"),
            vec![0, u64::from(INVALID_ARGUMENT)]
        );
        assert_eq!(harness::observed("list"), vec![0, 1, TITLE, 0]);
        assert_eq!(harness::observed("update"), vec![0]);
    }

    #[test]
    fn language_codes_map_to_the_nacp_order() {
        let code = |text: &str| {
            let mut bytes = [0u8; 8];
            bytes[..text.len()].copy_from_slice(text.as_bytes());
            u64::from_le_bytes(bytes)
        };
        assert_eq!(convert_language_code(code("en-US")), Ok(0));
        assert_eq!(convert_language_code(code("en-GB")), Ok(1));
        assert_eq!(convert_language_code(code("ja")), Ok(2));
        assert_eq!(convert_language_code(code("fr-CA")), Ok(9));
        assert_eq!(convert_language_code(code("es-419")), Ok(5));
        assert_eq!(convert_language_code(code("zh-TW")), Ok(13));
        assert_eq!(convert_language_code(code("pt-BR")), Ok(15));
        assert_eq!(
            convert_language_code(code("xx")),
            Err(LANGUAGE_CODE_NOT_FOUND)
        );
    }

    #[test]
    fn the_desired_language_is_the_first_non_empty_entry() {
        assert_eq!(desired_language(0), 0);
        assert_eq!(desired_language(1 << 0), 0);
        assert_eq!(desired_language(1 << 3), 3);
        assert_eq!(desired_language(0b1000_1000), 3);
        assert_eq!(desired_language(1 << 7), 7);
    }

    #[test]
    fn an_application_record_is_the_documented_size() {
        let mut record = [0u8; APPLICATION_RECORD_SIZE];
        record[..8].copy_from_slice(&0x0100_0000_0009_0000u64.to_le_bytes());
        assert_eq!(record.len(), 0x18);
        assert_eq!(
            u64::from_le_bytes(record[..8].try_into().unwrap()),
            0x0100_0000_0009_0000
        );
    }
}
