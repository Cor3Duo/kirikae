//! The `.profile` file the interpreter writes and the next build compiles.
//!
//! The runtime records every address where the interpreter had to take over and rewrites the file
//! every ten seconds (`kirikae_rt::coverage`). A build reads it and turns those addresses into
//! entries, which is what makes interpretation disappear after one play session. The discovery
//! loop runs that cycle a few times, so the merge has to be a union: a second run of the same
//! game must keep what the first one found and only add what is new.
//!
//! The file carries no title id (the runtime writes one header and the addresses), so the build
//! tags it once it has read it: `<profile>.title` holds the title that owns the addresses. A
//! profile tagged with another title is rejected instead of compiling addresses that belong to a
//! different image.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use kirikae_rt::coverage;

/// Header written by both the runtime and this module.
pub const HEADER: &str =
    "# kirikae: addresses the interpreter ran, for `kirikae build` (address, times entered)\n";

/// Addresses the interpreter ran, with how often each one was entered.
///
/// Ascending by address, so the file is stable across merges and two profiles of the same game
/// can be compared with a plain diff.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    counts: BTreeMap<u64, u64>,
}

impl Profile {
    pub fn parse(text: &str) -> Profile {
        Profile {
            counts: coverage::parse(text).into_iter().collect(),
        }
    }

    /// Read a profile; a missing file is an empty profile, which is what the first build of a
    /// game has.
    pub fn read(path: &Path) -> Result<Profile, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Profile::parse(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Profile::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    pub fn addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.counts.keys().copied()
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Add every address of `other`, summing the counts. Returns how many addresses are new to
    /// this profile.
    pub fn merge(&mut self, other: &Profile) -> usize {
        let before = self.counts.len();
        for (&addr, &count) in &other.counts {
            *self.counts.entry(addr).or_default() += count;
        }
        self.counts.len() - before
    }

    /// Every address of `self` that `previous` does not have.
    pub fn added_since(&self, previous: &Profile) -> Vec<u64> {
        self.counts
            .keys()
            .filter(|a| !previous.counts.contains_key(a))
            .copied()
            .collect()
    }

    pub fn text(&self) -> String {
        let mut text = String::from(HEADER);
        for (&addr, &count) in &self.counts {
            let _ = writeln!(text, "{addr:x} {count}");
        }
        text
    }

    /// Rewrite the profile in one step: a reader never sees half a merge.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("profile.tmp");
        std::fs::write(&tmp, self.text()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Where the title tag of `profile` lives: `game.profile.title` next to it.
pub fn title_path(profile: &Path) -> PathBuf {
    let mut name: OsString = profile
        .file_name()
        .map_or_else(|| OsString::from("game.profile"), OsString::from);
    name.push(".title");
    profile.with_file_name(name)
}

fn parse_title(text: &str) -> Option<u64> {
    text.lines()
        .find_map(|l| l.strip_prefix("title "))
        .and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
}

pub fn read_title(profile: &Path) -> Option<u64> {
    std::fs::read_to_string(title_path(profile))
        .ok()
        .and_then(|text| parse_title(&text))
}

/// Tag `profile` with the title that wrote it.
pub fn write_title(profile: &Path, title: u64) -> Result<(), String> {
    let path = title_path(profile);
    std::fs::write(&path, format!("title {title:016x}\n"))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Reject a profile that belongs to another title.
///
/// The rule the README states ("only use a profile with the same game and version") is only
/// checkable when the profile is tagged and the executable knows its title id; an untagged
/// profile is accepted, because a profile from a hand-run game without a build in between has no
/// tag yet.
pub fn check_title(profile: &Path, title: Option<u64>) -> Result<(), String> {
    let (Some(owner), Some(title)) = (read_title(profile), title) else {
        return Ok(());
    };
    if owner == title {
        return Ok(());
    }
    Err(format!(
        "{} belongs to title {owner:016x}, not {title:016x}: use the profile of the same game",
        profile.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kirikae-recomp-profile-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn merge_is_a_union_in_address_order() {
        let mut a = Profile::parse("# one\n300 1\n100 4\n");
        let b = Profile::parse("200 2\n100 1\n");
        assert_eq!(a.merge(&b), 1, "only 200 is new");
        assert_eq!(
            a.text(),
            format!("{HEADER}100 5\n200 2\n300 1\n"),
            "ascending, deduplicated, counts added"
        );
        assert_eq!(a.len(), 3);
        assert_eq!(a.addresses().collect::<Vec<_>>(), vec![0x100, 0x200, 0x300]);

        assert_eq!(a.merge(&b), 0, "a second merge adds nothing");
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn added_since_lists_only_the_new_addresses() {
        let previous = Profile::parse("100 1\n200 9\n");
        let now = Profile::parse("100 1\n300 1\n");
        assert_eq!(now.added_since(&previous), vec![0x300]);
        assert!(now.added_since(&now).is_empty());
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = temp("roundtrip");
        let path = dir.join("game.profile");
        let mut p = Profile::parse("10 1\n8 2\n");
        p.merge(&Profile::parse("18 3\n"));
        p.write(&path).unwrap();
        assert_eq!(Profile::read(&path).unwrap(), p);
        assert!(!path.with_extension("profile.tmp").exists());
        assert!(
            Profile::read(&dir.join("missing.profile"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_profile_from_another_title_is_rejected() {
        let dir = temp("title");
        let path = dir.join("game.profile");
        Profile::parse("100 1\n").write(&path).unwrap();

        assert_eq!(
            check_title(&path, Some(0x0100_0000_0000_0001)),
            Ok(()),
            "untagged"
        );
        write_title(&path, 0x0100_0000_0000_0001).unwrap();
        assert_eq!(read_title(&path), Some(0x0100_0000_0000_0001));
        assert_eq!(check_title(&path, Some(0x0100_0000_0000_0001)), Ok(()));
        assert_eq!(
            check_title(&path, None),
            Ok(()),
            "a test NRO has no title id"
        );
        let err = check_title(&path, Some(0x0100_0000_0000_0002)).unwrap_err();
        assert!(err.contains("0100000000000002"), "{err}");
        assert!(err.contains("same game"), "{err}");
        assert_eq!(title_path(&path), dir.join("game.profile.title"));
    }
}
