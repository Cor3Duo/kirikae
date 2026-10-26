use std::fmt;
use std::str::FromStr;

use kirikae_loader::Executable;

use crate::cpu::Cpu;
use crate::services::{Reply, Request};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
    pub micro: u8,
}

impl Version {
    pub const fn new(major: u8, minor: u8, micro: u8) -> Version {
        Version {
            major,
            minor,
            micro,
        }
    }

    pub fn parse(text: &str) -> Option<Version> {
        let mut parts = text.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next().map_or(Some(0), |p| p.parse().ok())?;
        let micro = parts.next().map_or(Some(0), |p| p.parse().ok())?;
        parts.next().is_none().then_some(Version {
            major,
            minor,
            micro,
        })
    }

    pub fn from_system_version(value: u32) -> Version {
        Version {
            major: (value >> 26) as u8,
            minor: ((value >> 20) & 0x3f) as u8,
            micro: ((value >> 16) & 0x0f) as u8,
        }
    }
}

pub const V1_0_0: Version = Version::new(1, 0, 0);
pub const V2_0_0: Version = Version::new(2, 0, 0);
pub const V2_3_0: Version = Version::new(2, 3, 0);
pub const V3_0_0: Version = Version::new(3, 0, 0);
pub const V4_0_0: Version = Version::new(4, 0, 0);
pub const V5_0_0: Version = Version::new(5, 0, 0);
pub const V6_0_0: Version = Version::new(6, 0, 0);
pub const V8_0_0: Version = Version::new(8, 0, 0);
pub const V9_0_0: Version = Version::new(9, 0, 0);
pub const V9_1_0: Version = Version::new(9, 1, 0);
pub const V10_0_0: Version = Version::new(10, 0, 0);
pub const V11_0_0: Version = Version::new(11, 0, 0);
pub const V15_0_0: Version = Version::new(15, 0, 0);
pub const V21_0_0: Version = Version::new(21, 0, 0);
pub const V22_0_0: Version = Version::new(22, 0, 0);

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.micro)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SdkVersion {
    #[default]
    Unknown,
    Known(Version),
}

impl SdkVersion {
    pub const fn known(major: u8, minor: u8, micro: u8) -> SdkVersion {
        SdkVersion::Known(Version::new(major, minor, micro))
    }

    pub fn is_unknown(self) -> bool {
        matches!(self, SdkVersion::Unknown)
    }

    pub fn at_least(self, version: Version) -> bool {
        match self {
            SdkVersion::Unknown => true,
            SdkVersion::Known(v) => v >= version,
        }
    }

    pub fn predates(self, version: Version) -> bool {
        matches!(self, SdkVersion::Known(v) if v < version)
    }
}

impl fmt::Display for SdkVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SdkVersion::Unknown => f.write_str("unknown"),
            SdkVersion::Known(v) => v.fmt(f),
        }
    }
}

impl FromStr for SdkVersion {
    type Err = String;

    fn from_str(text: &str) -> Result<SdkVersion, String> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("unknown") {
            return Ok(SdkVersion::Unknown);
        }
        Version::parse(text).map(SdkVersion::Known).ok_or_else(|| {
            format!("`{text}` is not an SDK version (expected 17, 17.5, 17.5.4 or unknown)")
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    SdkModule,
    Environment,
    Unknown,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::SdkModule => "sdk module strings",
            Source::Environment => "KIRIKAE_SDK_VERSION",
            Source::Unknown => "none, permissive",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Detected {
    pub version: SdkVersion,
    pub source: Source,
}

pub fn resolve(exe: &Executable) -> Detected {
    if let Ok(text) = std::env::var("KIRIKAE_SDK_VERSION") {
        match text.parse::<SdkVersion>() {
            Ok(version) => {
                return Detected {
                    version,
                    source: Source::Environment,
                };
            }
            Err(e) => eprintln!("kirikae: ignoring KIRIKAE_SDK_VERSION: {e}"),
        }
    }
    match detect(exe) {
        Some(version) => Detected {
            version: SdkVersion::Known(version),
            source: Source::SdkModule,
        },
        None => Detected {
            version: SdkVersion::Unknown,
            source: Source::Unknown,
        },
    }
}

pub fn detect(exe: &Executable) -> Option<Version> {
    if let Some(module) = exe.modules.iter().find(|m| m.name == "sdk")
        && let Some(version) = best_in(&module.image.memory)
    {
        return Some(version);
    }
    exe.modules.iter().find_map(|m| best_in(&m.image.memory))
}

fn best_in(bytes: &[u8]) -> Option<Version> {
    const SUFFIX: &[u8] = b"-Release";
    let mut best: Option<(u32, Version)> = None;
    let mut at = 0;
    while let Some(found) = find(&bytes[at..], SUFFIX) {
        let end = at + found;
        let mut version_start = end;
        while version_start > 0
            && (bytes[version_start - 1].is_ascii_digit() || bytes[version_start - 1] == b'_')
        {
            version_start -= 1;
        }
        if version_start < end && version_start > 0 && bytes[version_start - 1] == b'-' {
            let library_end = version_start - 1;
            let mut library_start = library_end;
            while library_start > 0
                && (bytes[library_start - 1].is_ascii_alphanumeric()
                    || bytes[library_start - 1] == b'_')
            {
                library_start -= 1;
            }
            let library = &bytes[library_start..library_end];
            if let Some(rank) = library_rank(library)
                && let Some(version) = parse_underscored(&bytes[version_start..end])
                && best.as_ref().is_none_or(|(best_rank, best_version)| {
                    rank < *best_rank || (rank == *best_rank && version > *best_version)
                })
            {
                best = Some((rank, version));
            }
        }
        at = end + SUFFIX.len();
    }
    best.map(|(_, version)| version)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn library_rank(library: &[u8]) -> Option<u32> {
    let lower = library.to_ascii_lowercase();
    let rank = if lower.ends_with(b"_nnsdk") {
        0
    } else if lower.ends_with(b"_libz") {
        1
    } else if lower.ends_with(b"_libcurl") {
        2
    } else if lower.ends_with(b"_movie") {
        3
    } else if lower.ends_with(b"_gfx") {
        4
    } else if lower.ends_with(b"_nvn") {
        5
    } else if lower.ends_with(b"_npln") {
        return None;
    } else if lower.starts_with(b"sdk_") || lower.starts_with(b"nintendosdk_") {
        6
    } else {
        return None;
    };
    Some(rank)
}

fn parse_underscored(token: &[u8]) -> Option<Version> {
    let text = std::str::from_utf8(token).ok()?;
    let mut parts = text.split('_');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let micro = parts.next()?.parse().ok()?;
    parts.next().is_none().then_some(Version {
        major,
        minor,
        micro,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct CommandRange {
    pub cmd: u32,
    pub name: &'static str,
    pub min: Version,
    pub max: Option<Version>,
}

impl CommandRange {
    pub const fn since(cmd: u32, name: &'static str, min: Version) -> CommandRange {
        CommandRange {
            cmd,
            name,
            min,
            max: None,
        }
    }

    pub const fn between(cmd: u32, name: &'static str, min: Version, max: Version) -> CommandRange {
        CommandRange {
            cmd,
            name,
            min,
            max: Some(max),
        }
    }

    pub fn allows(&self, version: Version) -> bool {
        version >= self.min && self.max.is_none_or(|max| version <= max)
    }

    fn range_label(&self) -> String {
        match self.max {
            Some(max) => format!("{}..{max}", self.min),
            None => format!("{}+", self.min),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    Refuse,
    Infer(Version),
}

pub fn verdict(version: SdkVersion, command: &CommandRange) -> Verdict {
    match version {
        SdkVersion::Unknown => Verdict::Infer(command.min),
        SdkVersion::Known(version) if command.allows(version) => Verdict::Allow,
        SdkVersion::Known(_) => Verdict::Refuse,
    }
}

pub fn gate(c: &Cpu, service: &str, commands: &[CommandRange], req: &Request) -> Option<Reply> {
    let command = commands.iter().find(|command| command.cmd == req.cmd)?;
    let process = c.process();
    let version = process.sdk_version();
    match verdict(version, command) {
        Verdict::Allow => None,
        Verdict::Infer(min) => {
            process.infer_sdk_version(min, service, command.name);
            None
        }
        Verdict::Refuse => {
            process.warn_once(format!(
                "{service} command {} ({}) exists in SDK {} but this title is {version}; answering the unknown-command error",
                req.cmd,
                command.name,
                command.range_label()
            ));
            Some(Reply::err(crate::services::result::CMIF_UNKNOWN_COMMAND))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kirikae_loader::{Image, Module};

    fn executable_with(text: &str) -> Executable {
        let mut memory = vec![0u8; 0x1000];
        memory[..text.len()].copy_from_slice(text.as_bytes());
        let base = kirikae_loader::LOAD_BASE;
        Executable {
            modules: vec![Module {
                name: "sdk".into(),
                image: Image {
                    base,
                    memory,
                    text: base..base + 0x1000,
                    ro: base + 0x1000..base + 0x1000,
                    data: base + 0x1000..base + 0x1000,
                    bss: base + 0x1000..base + 0x1000,
                    entry: base,
                },
            }],
            entry: base,
            title_id: None,
            main_stack_size: None,
            system_resource_size: 0,
            romfs: None,
        }
    }

    #[test]
    fn parses_the_cli_and_dump_forms() {
        assert_eq!("17".parse::<SdkVersion>(), Ok(SdkVersion::known(17, 0, 0)));
        assert_eq!(
            "17.5".parse::<SdkVersion>(),
            Ok(SdkVersion::known(17, 5, 0))
        );
        assert_eq!(
            "17.5.4".parse::<SdkVersion>(),
            Ok(SdkVersion::known(17, 5, 4))
        );
        assert_eq!(" unknown ".parse::<SdkVersion>(), Ok(SdkVersion::Unknown));
        assert!("17.x".parse::<SdkVersion>().is_err());
        assert!("17.5.4.1".parse::<SdkVersion>().is_err());
        assert!("".parse::<SdkVersion>().is_err());
    }

    #[test]
    fn orders_generations_and_prints_them() {
        let versions = [
            SdkVersion::known(0, 12, 16),
            SdkVersion::known(3, 5, 1),
            SdkVersion::known(13, 3, 6),
            SdkVersion::known(17, 5, 4),
            SdkVersion::known(20, 5, 17),
            SdkVersion::known(21, 0, 0),
        ];
        assert!(versions.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(versions[3].to_string(), "17.5.4");
        assert_eq!(SdkVersion::Unknown.to_string(), "unknown");
        assert!(SdkVersion::Unknown < versions[0]);
    }

    #[test]
    fn unknown_is_permissive_and_known_versions_are_not() {
        let command = CommandRange::since(10106, "SaveReport", Version::new(21, 0, 0));
        assert_eq!(
            verdict(SdkVersion::known(17, 5, 4), &command),
            Verdict::Refuse
        );
        assert_eq!(
            verdict(SdkVersion::known(21, 0, 0), &command),
            Verdict::Allow
        );
        assert_eq!(
            verdict(SdkVersion::Unknown, &command),
            Verdict::Infer(Version::new(21, 0, 0))
        );
        assert!(SdkVersion::Unknown.at_least(Version::new(21, 0, 0)));
        assert!(!SdkVersion::known(17, 5, 4).at_least(Version::new(21, 0, 0)));
    }

    #[test]
    fn only_a_known_older_title_predates_a_version() {
        assert!(SdkVersion::known(0, 12, 14).predates(V2_0_0));
        assert!(!SdkVersion::known(2, 0, 0).predates(V2_0_0));
        assert!(!SdkVersion::known(17, 5, 4).predates(V2_0_0));
        assert!(
            !SdkVersion::Unknown.predates(V2_0_0),
            "an undetected title keeps the kernel the runtime reports"
        );
    }

    #[test]
    fn reads_the_sdk_library_version_string() {
        assert_eq!(
            detect(&executable_with(
                "SDK MW+Nintendo\u{9}SDK_libz-17_5_4-Release"
            )),
            Some(Version::new(17, 5, 4))
        );
        assert_eq!(
            detect(&executable_with("Sdk_nnSdk-0_12_16-Release")),
            Some(Version::new(0, 12, 16))
        );
        assert_eq!(detect(&executable_with("SDK_NPLN-1_2_0-Release")), None);
        assert_eq!(detect(&executable_with("no version here")), None);
    }

    #[test]
    fn the_system_version_encoding_decodes() {
        assert_eq!(
            Version::from_system_version(0x2c00_0000),
            Version::new(11, 0, 0)
        );
        assert_eq!(
            Version::from_system_version(0x4c00_0000),
            Version::new(19, 0, 0)
        );
    }
}
