use std::collections::HashSet;
use std::sync::Mutex;

use winit::keyboard::KeyCode;

pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const STICK_L: u64 = 1 << 4;
    pub const STICK_R: u64 = 1 << 5;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
    pub const STICK_L_LEFT: u64 = 1 << 16;
    pub const STICK_R_LEFT: u64 = 1 << 20;
}

pub const STICK_MAX: i32 = 0x7fff;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PadState {
    pub buttons: u64,
    pub left: (i32, i32),
    pub right: (i32, i32),
}

enum Control {
    Button(u64),
    Left(i32, i32),
    Right(i32, i32),
}

fn control(code: KeyCode) -> Option<Control> {
    use Control::*;
    use KeyCode as K;
    Some(match code {
        K::KeyW | K::ArrowUp => Left(0, 1),
        K::KeyS | K::ArrowDown => Left(0, -1),
        K::KeyA | K::ArrowLeft => Left(-1, 0),
        K::KeyD | K::ArrowRight => Left(1, 0),
        K::KeyI => Right(0, 1),
        K::KeyK => Right(0, -1),
        K::KeyJ => Right(-1, 0),
        K::KeyL => Right(1, 0),
        K::KeyZ | K::Space => Button(button::A),
        K::KeyX => Button(button::B),
        K::KeyC => Button(button::X),
        K::KeyV => Button(button::Y),
        K::KeyE => Button(button::L),
        K::KeyU => Button(button::R),
        K::KeyQ => Button(button::ZL),
        K::KeyO => Button(button::ZR),
        K::Enter | K::Equal | K::NumpadAdd => Button(button::PLUS),
        K::Backspace | K::Minus | K::NumpadSubtract => Button(button::MINUS),
        K::KeyT => Button(button::UP),
        K::KeyF => Button(button::LEFT),
        K::KeyG => Button(button::DOWN),
        K::KeyH => Button(button::RIGHT),
        K::ShiftLeft => Button(button::STICK_L),
        K::KeyM => Button(button::STICK_R),
        _ => return None,
    })
}

static KEYS: Mutex<Option<HashSet<KeyCode>>> = Mutex::new(None);

pub fn key(code: KeyCode, pressed: bool) {
    let mut keys = KEYS.lock().unwrap();
    let keys = keys.get_or_insert_with(HashSet::new);
    if pressed {
        keys.insert(code);
    } else {
        keys.remove(&code);
    }
}

pub fn release_all() {
    if let Some(keys) = KEYS.lock().unwrap().as_mut() {
        keys.clear();
    }
}

fn key_stick(x: i32, y: i32) -> (i32, i32) {
    let (x, y) = (x.signum(), y.signum());
    let scale = if x != 0 && y != 0 { (STICK_MAX as f32 * std::f32::consts::FRAC_1_SQRT_2) as i32 } else { STICK_MAX };
    (x * scale, y * scale)
}

fn keyboard() -> PadState {
    let keys = KEYS.lock().unwrap();
    let (mut buttons, mut l, mut r) = (0, (0, 0), (0, 0));
    for c in keys.iter().flatten().filter_map(|&k| control(k)) {
        match c {
            Control::Button(b) => buttons |= b,
            Control::Left(x, y) => l = (l.0 + x, l.1 + y),
            Control::Right(x, y) => r = (r.0 + x, r.1 + y),
        }
    }
    PadState { buttons, left: key_stick(l.0, l.1), right: key_stick(r.0, r.1) }
}

fn stick_bits(s: (i32, i32), first: u64) -> u64 {
    const T: i32 = STICK_MAX / 2;
    let mut b = 0;
    if s.0 < -T {
        b |= first;
    }
    if s.1 > T {
        b |= first << 1;
    }
    if s.0 > T {
        b |= first << 2;
    }
    if s.1 < -T {
        b |= first << 3;
    }
    b
}

#[derive(Debug, PartialEq)]
struct Step {
    from: f64,
    to: f64,
    pad: PadState,
}

fn parse_script(s: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for item in s.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let (time, controls) = item.split_once(':').ok_or_else(|| format!("missing ':' in {item:?}"))?;
        let num = |t: &str| t.trim().parse::<f64>().map_err(|_| format!("bad time in {item:?}"));
        let (from, to) = match time.split_once('-') {
            Some((a, b)) => (num(a)?, num(b)?),
            None => (num(time)?, num(time)? + 0.2),
        };
        let mut pad = PadState::default();
        for c in controls.split('+') {
            let c = c.trim().to_ascii_uppercase();
            let stick = |name: &str| match name {
                "UP" => Some((0, STICK_MAX)),
                "DOWN" => Some((0, -STICK_MAX)),
                "LEFT" => Some((-STICK_MAX, 0)),
                "RIGHT" => Some((STICK_MAX, 0)),
                _ => None,
            };
            let b = match c.as_str() {
                "A" => button::A,
                "B" => button::B,
                "X" => button::X,
                "Y" => button::Y,
                "L" => button::L,
                "R" => button::R,
                "ZL" => button::ZL,
                "ZR" => button::ZR,
                "PLUS" => button::PLUS,
                "MINUS" => button::MINUS,
                "UP" => button::UP,
                "DOWN" => button::DOWN,
                "LEFT" => button::LEFT,
                "RIGHT" => button::RIGHT,
                "LSTICK" => button::STICK_L,
                "RSTICK" => button::STICK_R,
                _ => 0,
            };
            if b != 0 {
                pad.buttons |= b;
            } else if let Some(v) = c.strip_prefix('L').and_then(stick) {
                pad.left = v;
            } else if let Some(v) = c.strip_prefix('R').and_then(stick) {
                pad.right = v;
            } else {
                return Err(format!("unknown control {c:?}"));
            }
        }
        steps.push(Step { from, to, pad });
    }
    Ok(steps)
}

fn scripted() -> Option<PadState> {
    static SCRIPT: std::sync::OnceLock<(std::time::Instant, Vec<Step>)> = std::sync::OnceLock::new();
    let (start, steps) = SCRIPT.get_or_init(|| {
        let steps = match std::env::var("KIRIKAE_INPUT").map(|s| parse_script(&s)) {
            Ok(Ok(steps)) => steps,
            Ok(Err(e)) => {
                eprintln!("kirikae: KIRIKAE_INPUT ignored: {e}");
                Vec::new()
            }
            Err(_) => Vec::new(),
        };
        (std::time::Instant::now(), steps)
    });
    let t = start.elapsed().as_secs_f64();
    steps.iter().filter(|s| (s.from..s.to).contains(&t)).map(|s| s.pad).reduce(|a, b| PadState {
        buttons: a.buttons | b.buttons,
        left: if b.left != (0, 0) { b.left } else { a.left },
        right: if b.right != (0, 0) { b.right } else { a.right },
    })
}

pub fn state() -> PadState {
    let mut s = keyboard();
    for g in [gamepad::poll(), scripted()].into_iter().flatten() {
        s.buttons |= g.buttons;
        let pick = |a: i32, b: i32| if b.abs() > a.abs() { b } else { a };
        s.left = (pick(s.left.0, g.left.0), pick(s.left.1, g.left.1));
        s.right = (pick(s.right.0, g.right.0), pick(s.right.1, g.right.1));
    }
    s.buttons |= stick_bits(s.left, button::STICK_L_LEFT) | stick_bits(s.right, button::STICK_R_LEFT);
    s
}

#[cfg(windows)]
mod gamepad {
    use std::ffi::{c_char, c_void};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use super::{button, PadState, STICK_MAX};

    #[repr(C)]
    #[derive(Default)]
    struct XInputState {
        packet: u32,
        buttons: u16,
        left_trigger: u8,
        right_trigger: u8,
        lx: i16,
        ly: i16,
        rx: i16,
        ry: i16,
    }

    type GetState = unsafe extern "system" fn(u32, *mut XInputState) -> u32;

    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    struct Pads {
        get_state: Option<GetState>,
        connected: [bool; 4],
        last_scan: Option<Instant>,
    }

    static PADS: Mutex<Option<Pads>> = Mutex::new(None);

    fn load() -> Option<GetState> {
        for dll in [c"xinput1_4.dll", c"xinput1_3.dll", c"xinput9_1_0.dll"] {
            // SAFETY: NUL-terminated names; XInputGetState has this signature
            // in every XInput version.
            unsafe {
                let module = LoadLibraryA(dll.as_ptr());
                if module.is_null() {
                    continue;
                }
                let f = GetProcAddress(module, c"XInputGetState".as_ptr());
                if !f.is_null() {
                    return Some(std::mem::transmute::<*mut c_void, GetState>(f));
                }
            }
        }
        None
    }

    fn axis(v: i16, dead: i32) -> i32 {
        let v = (v as i32).clamp(-STICK_MAX, STICK_MAX);
        if v.abs() < dead { 0 } else { v }
    }

    fn convert(s: &XInputState) -> PadState {
        const MAP: [(u16, u64); 14] = [
            (0x0001, button::UP),
            (0x0002, button::DOWN),
            (0x0004, button::LEFT),
            (0x0008, button::RIGHT),
            (0x0010, button::PLUS),
            (0x0020, button::MINUS),
            (0x0040, button::STICK_L),
            (0x0080, button::STICK_R),
            (0x0100, button::L),
            (0x0200, button::R),
            (0x1000, button::B),
            (0x2000, button::A),
            (0x4000, button::Y),
            (0x8000, button::X),
        ];
        let mut buttons = MAP.iter().filter(|(x, _)| s.buttons & x != 0).fold(0, |b, (_, n)| b | n);
        if s.left_trigger > 30 {
            buttons |= button::ZL;
        }
        if s.right_trigger > 30 {
            buttons |= button::ZR;
        }
        PadState { buttons, left: (axis(s.lx, 7849), axis(s.ly, 7849)), right: (axis(s.rx, 8689), axis(s.ry, 8689)) }
    }

    pub fn poll() -> Option<PadState> {
        let mut pads = PADS.lock().unwrap();
        let pads = pads.get_or_insert_with(|| Pads { get_state: load(), connected: [false; 4], last_scan: None });
        let get_state = pads.get_state?;
        let scan = pads.last_scan.is_none_or(|t| t.elapsed() > Duration::from_secs(2));
        if scan {
            pads.last_scan = Some(Instant::now());
        }
        let mut found = None;
        for i in 0..4 {
            if !pads.connected[i] && !scan {
                continue;
            }
            let mut st = XInputState::default();
            // SAFETY: `st` is a valid XINPUT_STATE for the call to fill.
            pads.connected[i] = unsafe { get_state(i as u32, &mut st) } == 0;
            if pads.connected[i] && found.is_none() {
                found = Some(convert(&st));
            }
        }
        found
    }
}

#[cfg(not(windows))]
mod gamepad {
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use gilrs::{Axis, Button, Gilrs};

    use super::{button, PadState, STICK_MAX};

    static STATE: Mutex<Option<PadState>> = Mutex::new(None);

    fn axis(v: f32, dead: f32) -> i32 {
        if v.abs() < dead { 0 } else { (v.clamp(-1.0, 1.0) * STICK_MAX as f32) as i32 }
    }

    fn read(gilrs: &Gilrs) -> Option<PadState> {
        const MAP: [(Button, u64); 16] = [
            (Button::DPadUp, button::UP),
            (Button::DPadDown, button::DOWN),
            (Button::DPadLeft, button::LEFT),
            (Button::DPadRight, button::RIGHT),
            (Button::Start, button::PLUS),
            (Button::Select, button::MINUS),
            (Button::LeftThumb, button::STICK_L),
            (Button::RightThumb, button::STICK_R),
            (Button::LeftTrigger, button::L),
            (Button::RightTrigger, button::R),
            (Button::LeftTrigger2, button::ZL),
            (Button::RightTrigger2, button::ZR),
            (Button::South, button::B),
            (Button::East, button::A),
            (Button::West, button::Y),
            (Button::North, button::X),
        ];
        let (_, pad) = gilrs.gamepads().find(|(_, g)| g.is_connected())?;
        let buttons = MAP.iter().filter(|(b, _)| pad.is_pressed(*b)).fold(0, |acc, (_, n)| acc | n);
        let stick = |x: Axis, y: Axis| (axis(pad.value(x), 0.24), axis(pad.value(y), 0.24));
        Some(PadState { buttons, left: stick(Axis::LeftStickX, Axis::LeftStickY), right: stick(Axis::RightStickX, Axis::RightStickY) })
    }

    fn run() {
        let mut gilrs = match Gilrs::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("kirikae: gamepads unavailable ({e})");
                return;
            }
        };
        loop {
            while gilrs.next_event_blocking(Some(Duration::from_millis(8))).is_some() {}
            *STATE.lock().unwrap() = read(&gilrs);
        }
    }

    pub fn poll() -> Option<PadState> {
        static STARTED: OnceLock<()> = OnceLock::new();
        STARTED.get_or_init(|| {
            let _ = std::thread::Builder::new().name("kirikae-gamepad".into()).spawn(run);
        });
        *STATE.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_drive_buttons_and_sticks() {
        key(KeyCode::KeyZ, true);
        key(KeyCode::KeyW, true);
        key(KeyCode::KeyD, true);
        key(KeyCode::KeyL, true);
        let s = keyboard();
        assert_eq!(s.buttons, button::A);
        assert!(s.left.0 > 20000 && s.left.0 == s.left.1, "diagonal: {:?}", s.left);
        assert_eq!(s.right, (STICK_MAX, 0));
        assert_eq!(stick_bits(s.right, button::STICK_R_LEFT), button::STICK_R_LEFT << 2);
        key(KeyCode::Space, true);
        key(KeyCode::KeyZ, false);
        assert_eq!(keyboard().buttons, button::A);
        release_all();
        assert_eq!(keyboard(), PadState::default());
    }

    #[test]
    fn parses_input_scripts() {
        let s = parse_script("40:A, 45-47:lup+B+RRIGHT").unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].from, s[0].to, s[0].pad.buttons), (40.0, 40.2, button::A));
        assert_eq!((s[1].from, s[1].to), (45.0, 47.0));
        assert_eq!(s[1].pad, PadState { buttons: button::B, left: (0, STICK_MAX), right: (STICK_MAX, 0) });
        assert!(parse_script("4:JUMP").is_err());
        assert!(parse_script("A").is_err());
    }
}
