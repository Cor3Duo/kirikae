use std::sync::OnceLock;

use crate::process::Process;

pub struct GameHooks {
    pub pace_interval: fn(&Process, u32) -> u32,
    pub presented: fn(&Process),
}

static HOOKS: OnceLock<GameHooks> = OnceLock::new();

pub fn install(hooks: GameHooks) {
    let _ = HOOKS.set(hooks);
}

pub fn pace_interval(p: &Process, asked: u32) -> u32 {
    HOOKS.get().map_or(asked, |h| (h.pace_interval)(p, asked))
}

pub fn presented(p: &Process) {
    if let Some(h) = HOOKS.get() {
        (h.presented)(p);
    }
}
