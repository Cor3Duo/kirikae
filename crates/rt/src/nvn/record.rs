//! Command records in the guest's own command memory.
//!
//! The recorder appends one fixed 16-byte-aligned record per command at the game's write pointer and keeps the
//! decoded values a record names in a side blob at the far end of the same range. A list's handle is the guest
//! address of its first record, so a list recorded later over the same bytes retires the older handle by address
//! rather than by object identity (`docs/nvn/CONTRACT.md` §3). No record keeps a pointer to a builder, an array or
//! a state object: everything a replay needs is copied at record time.
//!
//! Record layout, little endian, [`RECORD_SIZE`] bytes:
//!
//! | offset | size | field |
//! | --- | --- | --- |
//! | 0x00 | 4 | op |
//! | 0x04 | 2 | size (always [`RECORD_SIZE`]) |
//! | 0x06 | 2 | flags |
//! | 0x08 | 4 | count (handles, state entries or inline records) |
//! | 0x0c | 4 | blob offset from the list handle, 0 when there is no blob |
//! | 0x10 | 4 | blob length in bytes |
//! | 0x14 | 4 | reserved, zero |
//! | 0x18 | 72 | 9 u64 words |
//! | 0x60 | 16 | 4 f32 |
//! | 0x70 | 16 | 4 i32 |
//!
//! Records grow up from `handle`; blobs grow down from the list's blob top, which is the end of the span or below
//! the blobs of the lists already recorded there. A list uses at most the space between its first record and that
//! top. The guest write counter is advanced by the bytes we write, so the game's own usage queries and allocator
//! see the same numbers the driver would have produced.
//!
//! `CopyCommands` is unconfirmed (`docs/nvn/CONTRACT.md` §3, §10): we model it as a copy of the callee's command
//! records into the caller's list at record time, re-basing every blob, so overwriting the callee's memory cannot
//! change the copy. `CallCommands` stores the callee handles instead. Nesting stops at [`MAX_NESTING`].
//!
//! Recording is enabled by `KIRIKAE_NVN=record`. In that mode the host takes the recorded procs over and the driver
//! does not run them, because both would write the same command memory; the driver's own `BeginRecording` and
//! `EndRecording` still run around the recording so its header, callback and allocator state stay consistent.
//! `nvnQueueSubmitCommands` is taken over for a submit whose lists are all recorded here, and `replay.rs` walks them
//! into Vulkan in submission order. A submit that names a list the host did not record stays with the driver, whose
//! pushbuffer the GPU path renders: the two command streams are never mixed.
//!
//! The host keeps a copy of every blob it writes (`Recorder::blob`): the driver and the next lists reuse that command
//! memory while a submit waits to be replayed, so a queued submit reads the copy, not the bytes as they are now.
//!
//! **Implicit lists.** A driver keeps a fresh list after every `EndRecording` and many titles record most of their
//! per-frame lists that way, without asking the game to `BeginRecording` again. Those commands cannot be recorded into
//! the game's command memory: the driver is writing its own pushbuffer there, and two writers in one range corrupt
//! both streams. So a list the host opens without `BeginRecording` is recorded into a **host arena**: records and
//! blobs live in this process, keyed at `EndRecording` by the handle the driver returned, and the driver keeps its
//! memory and its stream to itself. A submit that names such a handle resolves to the arena list by that key.
//! `KIRIKAE_NVN=trace` and `shadow` are not affected: they keep the driver as their oracle and never write here.
//!
//! **Chaining.** A pass can straddle an `EndRecording`: the render targets and binds end in one chunk and the draw
//! that uses them is recorded into the next. When the driver closes a chunk under a pointer above the GPU space, the
//! host keeps it and makes it the prefix of the next list closed under a usable handle ([`Recorder::chain`]). A
//! replay walks the prefix first, in recording order, so the draw sees the state that precedes it. Nothing is copied
//! and the driver's memory is never written. A chunk goes when the list that carries it does, and a submit's arena
//! lists are retired once its replay is done — both keep the arena bounded (`docs/nvn/CONTRACT.md` §3).

use std::collections::BTreeMap;

use super::error::Error;
use super::state::Kind;

/// Every record is this long; a replay walks a list by adding it to the record address.
pub const RECORD_SIZE: u64 = 0x80;
const RECORD_BYTES: usize = RECORD_SIZE as usize;

/// Recursion limit for `CallCommands`/`CopyCommands`, the same depth the replay will use (`CONTRACT.md` §4).
pub const MAX_NESTING: u32 = 8;

/// Handles one `CallCommands`/`CopyCommands` call may name. The API takes a count and an array, so this is a bound
/// on guest data rather than an API constant.
pub const MAX_HANDLES: usize = 4096;

/// Most bytes one `CopyCommands` source may hold.
pub const MAX_COPY_BYTES: u64 = 1 << 20;

/// Bytes copied from a state object when the host tables have no decoded value for it.
///
/// TODO(4.6): NVN state struct sizes are not established in this tree; this is the bound we copy, not a claimed
/// `sizeof`. The SDK driver's getters remain the authority, and a later milestone should read each field from it
/// instead of copying a struct whose size we do not know.
pub const RAW_STATE_BYTES: u64 = 0x100;

/// A list the host opens without `BeginRecording` is recorded here, not in the game's command memory. Its storage
/// address is a synthetic value in a range the guest cannot name, so a driver handle and an arena storage address
/// never collide. Each slot leaves [`ARENA_CAPACITY`] bytes of room, well above the largest per-frame list seen.
pub const ARENA_BASE: u64 = 0x4000_0000_0000_0000;
/// Distance between two arena storage addresses; larger than [`ARENA_CAPACITY`] so their blobs cannot overlap.
pub const ARENA_SLOT: u64 = 0x100_0000;
/// Arena slots in rotation. A list that gets a slot back is retired first, the way re-recording over a handle
/// retires it; the count only caps host memory, it is not an API constant.
pub const ARENA_SLOTS: u64 = 4096;
/// Records plus blobs one arena list may hold.
pub const ARENA_CAPACITY: u64 = 0x80_0000;

/// NVN GPU addresses are 40-bit (`docs/nvn/CONTRACT.md` §2), so a command-list handle a submit or a `CallCommands`
/// can name is below this. Some `EndRecording` calls also report a driver-internal pointer with tag bits above the
/// GPU space, and no game call can name that value.
pub const HANDLE_LIMIT: u64 = 1 << 40;

/// Whether `value` can be a command-list handle the game names.
pub fn is_handle(value: u64) -> bool {
    value != 0 && value < HANDLE_LIMIT
}

/// Set when `KIRIKAE_NVN=record`: the only mode that takes command procs over and writes records.
pub static ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The record carries a blob; `blob`/`blob_len` are set.
pub const FLAG_BLOB: u16 = 1 << 0;
/// The record names other lists (`CallCommands`) or inlines them (`CopyCommands`).
pub const FLAG_NESTED: u16 = 1 << 1;
/// `count` records follow this one and belong to it (`CopyCommands`).
pub const FLAG_INLINE: u16 = 1 << 2;
/// The blob is a raw state-object copy, not decoded [`StateValue`]s.
pub const FLAG_RAW: u16 = 1 << 3;
/// The state object had neither a decoded value nor readable bytes; only its address is known.
pub const FLAG_UNRESOLVED: u16 = 1 << 4;

pub const OFF_OP: u64 = 0x00;
pub const OFF_SIZE: u64 = 0x04;
pub const OFF_FLAGS: u64 = 0x06;
pub const OFF_COUNT: u64 = 0x08;
pub const OFF_BLOB: u64 = 0x0c;
pub const OFF_BLOB_LEN: u64 = 0x10;
pub const OFF_WORDS: u64 = 0x18;
pub const OFF_FLOATS: u64 = 0x60;
pub const OFF_INTS: u64 = 0x70;

/// Guest command-buffer header fields the write pointer lives in (`shadow.rs:446-458`).
pub mod header {
    /// Write pointer: the next free byte. The driver this tree runs advances this one.
    pub const WRITE: u64 = 0x00;
    /// The field `shadow.rs:446` reads as the write pointer. This driver keeps the span end here.
    pub const WRITE_SHADOW: u64 = 0x18;
    /// Guest address the used count is relative to.
    pub const BASE: u64 = 0x50;
    /// GPU address of the same byte.
    pub const GPU_BASE: u64 = 0x60;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Op {
    Begin = 0x01,
    End = 0x02,
    SetRenderTargets = 0x10,
    ClearColor = 0x11,
    ClearDepthStencil = 0x12,
    DiscardColor = 0x13,
    BindProgram = 0x20,
    BindBlendState = 0x21,
    BindColorState = 0x22,
    BindChannelMaskState = 0x23,
    BindDepthStencilState = 0x24,
    BindPolygonState = 0x25,
    BindVertexAttribState = 0x26,
    BindVertexStreamState = 0x27,
    BindTexture = 0x28,
    BindUniformBuffer = 0x29,
    BindStorageBuffer = 0x2e,
    SetSamplerPool = 0x2a,
    SetTexturePool = 0x2b,
    BindVertexBuffer = 0x2c,
    SetViewport = 0x2d,
    DrawArrays = 0x30,
    DrawElements = 0x31,
    DispatchCompute = 0x32,
    Barrier = 0x40,
    CallCommands = 0x50,
    CopyCommands = 0x51,
}

impl Op {
    pub fn code(self) -> u32 {
        self as u32
    }

    pub fn from_code(code: u32) -> Option<Op> {
        use Op::*;
        Some(match code {
            0x01 => Begin,
            0x02 => End,
            0x10 => SetRenderTargets,
            0x11 => ClearColor,
            0x12 => ClearDepthStencil,
            0x13 => DiscardColor,
            0x20 => BindProgram,
            0x21 => BindBlendState,
            0x22 => BindColorState,
            0x23 => BindChannelMaskState,
            0x24 => BindDepthStencilState,
            0x25 => BindPolygonState,
            0x26 => BindVertexAttribState,
            0x27 => BindVertexStreamState,
            0x28 => BindTexture,
            0x29 => BindUniformBuffer,
            0x2e => BindStorageBuffer,
            0x2a => SetSamplerPool,
            0x2b => SetTexturePool,
            0x2c => BindVertexBuffer,
            0x2d => SetViewport,
            0x30 => DrawArrays,
            0x31 => DrawElements,
            0x32 => DispatchCompute,
            0x40 => Barrier,
            0x50 => CallCommands,
            0x51 => CopyCommands,
            _ => return None,
        })
    }
}

/// One decoded `nvn*State` field. [`StateValue::BYTES`] per entry in the record's blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateValue {
    pub kind: u16,
    pub index: u8,
    pub value: u64,
}

impl StateValue {
    pub const BYTES: u64 = 16;

    fn encode(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.value.to_le_bytes());
        out[8..10].copy_from_slice(&self.kind.to_le_bytes());
        out[10] = self.index;
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<StateValue> {
        let value = u64::from_le_bytes(bytes.get(0..8)?.try_into().ok()?);
        let kind = u16::from_le_bytes(bytes.get(8..10)?.try_into().ok()?);
        let index = *bytes.get(10)?;
        Some(StateValue { kind, index, value })
    }
}

/// Stable wire tag for a decoded state field. Kept as an explicit match so reordering [`Kind`] never changes the
/// meaning of bytes already recorded in guest memory.
pub fn kind_code(kind: Kind) -> u16 {
    use Kind::*;
    match kind {
        ChannelMask => 1,
        BlendEnable => 2,
        BlendEquation => 3,
        BlendSrc => 4,
        BlendDst => 5,
        BlendEquationAlpha => 6,
        BlendSrcAlpha => 7,
        BlendDstAlpha => 8,
        DepthTest => 9,
        DepthWrite => 10,
        DepthFunc => 11,
        StencilTest => 12,
        StencilFunc => 13,
        StencilFail => 14,
        StencilDepthFail => 15,
        StencilPass => 16,
        CullFace => 17,
        FrontFace => 18,
        PolygonOffsetEnables => 19,
        AttribFormat => 20,
        AttribOffset => 21,
        AttribStream => 22,
        StreamStride => 23,
        StreamDivisor => 24,
        BlendColor => 25,
        Viewport => 26,
        Scissor => 27,
        TargetFormat => 28,
        BlendRegister => 29,
        AttribCount => 30,
        StreamCount => 31,
        DepthFormat => 32,
        PolygonOffset => 33,
        StencilRef => 34,
        StencilValueMask => 35,
        StencilMask => 36,
        DepthClamp => 37,
        RasterizerDiscard => 38,
        PointSize => 39,
        LineWidth => 40,
        DepthBoundsEnable => 41,
        DepthBoundsMin => 42,
        DepthBoundsMax => 43,
    }
}

pub fn kind_from_code(code: u16) -> Option<Kind> {
    use Kind::*;
    Some(match code {
        1 => ChannelMask,
        2 => BlendEnable,
        3 => BlendEquation,
        4 => BlendSrc,
        5 => BlendDst,
        6 => BlendEquationAlpha,
        7 => BlendSrcAlpha,
        8 => BlendDstAlpha,
        9 => DepthTest,
        10 => DepthWrite,
        11 => DepthFunc,
        12 => StencilTest,
        13 => StencilFunc,
        14 => StencilFail,
        15 => StencilDepthFail,
        16 => StencilPass,
        17 => CullFace,
        18 => FrontFace,
        19 => PolygonOffsetEnables,
        20 => AttribFormat,
        21 => AttribOffset,
        22 => AttribStream,
        23 => StreamStride,
        24 => StreamDivisor,
        25 => BlendColor,
        26 => Viewport,
        27 => Scissor,
        28 => TargetFormat,
        29 => BlendRegister,
        30 => AttribCount,
        31 => StreamCount,
        32 => DepthFormat,
        33 => PolygonOffset,
        34 => StencilRef,
        35 => StencilValueMask,
        36 => StencilMask,
        37 => DepthClamp,
        38 => RasterizerDiscard,
        39 => PointSize,
        40 => LineWidth,
        41 => DepthBoundsEnable,
        42 => DepthBoundsMin,
        43 => DepthBoundsMax,
        _ => return None,
    })
}

/// Guest memory as the recorder sees it: readable and writable spans, no allocation.
pub trait Memory {
    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>>;
    fn write(&self, addr: u64, bytes: &[u8]) -> bool;
}

impl Memory for crate::mem::GuestMemory {
    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        crate::mem::GuestMemory::read(self, addr, len)
    }

    fn write(&self, addr: u64, bytes: &[u8]) -> bool {
        crate::mem::GuestMemory::write(self, addr, bytes)
    }
}

fn header_word(mem: &dyn Memory, cmdbuf: u64, offset: u64) -> Option<u64> {
    let at = cmdbuf.checked_add(offset)?;
    let bytes = mem.read(at, 8)?;
    Some(u64::from_le_bytes(bytes.as_slice().try_into().ok()?))
}

/// Reads one header word; the recorder and the host both place records through these fields.
pub fn word(mem: &dyn Memory, cmdbuf: u64, offset: u64) -> Option<u64> {
    header_word(mem, cmdbuf, offset)
}

/// Writes one header word (the write pointer, or the driver's view of it around its own `EndRecording`).
pub fn set_word(mem: &dyn Memory, cmdbuf: u64, offset: u64, value: u64) -> bool {
    match cmdbuf.checked_add(offset) {
        Some(at) => mem.write(at, &value.to_le_bytes()),
        None => false,
    }
}

/// The three header words the game's write pointer is made of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header {
    /// Next free byte (`header::WRITE`). A pointer in the same address space as [`Header::base`].
    pub write: u64,
    /// The older field [`header::WRITE_SHADOW`]; tried when [`Header::write`] does not resolve.
    pub write_shadow: u64,
    /// Start of the command memory, in the driver's own address space.
    pub base: u64,
    /// GPU address of the same start. The driver's GPU space is not the host's synthetic one.
    pub gpu_base: u64,
}

pub fn header(mem: &dyn Memory, cmdbuf: u64) -> Option<Header> {
    Some(Header {
        write: header_word(mem, cmdbuf, header::WRITE)?,
        write_shadow: header_word(mem, cmdbuf, header::WRITE_SHADOW)?,
        base: header_word(mem, cmdbuf, header::BASE)?,
        gpu_base: header_word(mem, cmdbuf, header::GPU_BASE)?,
    })
}

/// GPU address shadow's field names: `gpu_base + write - base` (`shadow.rs:450-458`). This is the address the
/// command processor sees, not one of the host's synthetic pool ranges.
pub fn position(mem: &dyn Memory, cmdbuf: u64) -> Option<u64> {
    let header = header(mem, cmdbuf)?;
    Some(
        header
            .gpu_base
            .wrapping_add(header.write_shadow.wrapping_sub(header.base)),
    )
}

/// A command-memory span the game handed over, resolved to guest memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub guest: u64,
    pub size: u64,
}

impl Span {
    fn end(&self) -> u64 {
        self.guest.saturating_add(self.size)
    }
}

/// Where the next record goes, and which header word says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub guest: u64,
    /// Bytes left in the span that holds [`Placement::guest`].
    pub capacity: u64,
    /// Header offset to advance after writing; [`header::WRITE`] or [`header::WRITE_SHADOW`].
    pub field: u64,
}

/// Where the game's write pointer names inside `spans`.
///
/// `write` and `base` are pointers in the same address space (guest, or the driver's GPU space), so their
/// difference is the byte offset into the memory `AddCommandMemory` registered, and the guest address is that
/// span's guest start plus the offset. This is why the driver's GPU address never has to match one of the host's
/// synthetic pool ranges. A driver that keeps the GPU base in [`header::GPU_BASE`] is covered by the second
/// candidate, and an absolute guest `write` with a zero base by the third.
///
/// Both [`header::WRITE`] and [`header::WRITE_SHADOW`] are tried, because the field shadow reads is not the write
/// pointer in every driver build: one title leaves the span end there and advances `+0x00` instead.
pub fn placement(mem: &dyn Memory, cmdbuf: u64, spans: &[Span]) -> Result<Placement, Error> {
    if spans.is_empty() {
        return Err(Error::NoCommandMemory);
    }
    let header = header(mem, cmdbuf).ok_or(Error::Unmapped)?;
    for field in [header::WRITE, header::WRITE_SHADOW] {
        let write = match field {
            header::WRITE => header.write,
            _ => header.write_shadow,
        };
        if header.base != 0
            && let Some((guest, capacity)) = at_offset(spans, write.wrapping_sub(header.base))
        {
            return Ok(Placement {
                guest,
                capacity,
                field,
            });
        }
        if header.gpu_base != 0
            && let Some((guest, capacity)) = at_offset(spans, write.wrapping_sub(header.gpu_base))
        {
            return Ok(Placement {
                guest,
                capacity,
                field,
            });
        }
        if let Some((guest, capacity)) = containing(spans, write) {
            return Ok(Placement {
                guest,
                capacity,
                field,
            });
        }
    }
    Err(Error::WritePointerOutsideMemory)
}

fn at_offset(spans: &[Span], used: u64) -> Option<(u64, u64)> {
    let mut remaining = used;
    for span in spans {
        if remaining < span.size {
            return Some((span.guest + remaining, span.size - remaining));
        }
        remaining -= span.size;
    }
    None
}

fn containing(spans: &[Span], guest: u64) -> Option<(u64, u64)> {
    spans.iter().find_map(|span| {
        (span.guest <= guest && guest < span.end()).then(|| (guest, span.end() - guest))
    })
}

/// Advances the guest's used-byte counter so its own accounting agrees with the records we wrote. `field` is the
/// header offset [`placement`] resolved from; only that word moves.
pub fn advance(mem: &dyn Memory, cmdbuf: u64, field: u64, bytes: u64) -> bool {
    let Some(write) = header_word(mem, cmdbuf, field) else {
        return false;
    };
    let Some(next) = write.checked_add(bytes) else {
        return false;
    };
    let Some(at) = cmdbuf.checked_add(field) else {
        return false;
    };
    mem.write(at, &next.to_le_bytes())
}

/// One fixed-size command record. `blob`/`blob_len` are relative to the list handle and are set by [`Recorder`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub op: Op,
    pub flags: u16,
    pub count: u32,
    pub blob: u32,
    pub blob_len: u32,
    pub words: [u64; 9],
    pub floats: [f32; 4],
    pub ints: [i32; 4],
}

impl Record {
    pub fn new(op: Op) -> Record {
        Record {
            op,
            flags: 0,
            count: 0,
            blob: 0,
            blob_len: 0,
            words: [0; 9],
            floats: [0.0; 4],
            ints: [0; 4],
        }
    }

    pub fn set_word(&mut self, index: usize, value: u64) {
        if let Some(slot) = self.words.get_mut(index) {
            *slot = value;
        }
    }

    pub fn set_float(&mut self, index: usize, value: f32) {
        if let Some(slot) = self.floats.get_mut(index) {
            *slot = value;
        }
    }

    pub fn set_int(&mut self, index: usize, value: i32) {
        if let Some(slot) = self.ints.get_mut(index) {
            *slot = value;
        }
    }

    pub fn encode(&self) -> [u8; RECORD_BYTES] {
        let mut out = [0u8; RECORD_BYTES];
        out[OFF_OP as usize..OFF_OP as usize + 4].copy_from_slice(&self.op.code().to_le_bytes());
        out[OFF_SIZE as usize..OFF_SIZE as usize + 2]
            .copy_from_slice(&(RECORD_SIZE as u16).to_le_bytes());
        out[OFF_FLAGS as usize..OFF_FLAGS as usize + 2].copy_from_slice(&self.flags.to_le_bytes());
        out[OFF_COUNT as usize..OFF_COUNT as usize + 4].copy_from_slice(&self.count.to_le_bytes());
        out[OFF_BLOB as usize..OFF_BLOB as usize + 4].copy_from_slice(&self.blob.to_le_bytes());
        out[OFF_BLOB_LEN as usize..OFF_BLOB_LEN as usize + 4]
            .copy_from_slice(&self.blob_len.to_le_bytes());
        for (index, word) in self.words.iter().enumerate() {
            let at = OFF_WORDS as usize + 8 * index;
            out[at..at + 8].copy_from_slice(&word.to_le_bytes());
        }
        for (index, float) in self.floats.iter().enumerate() {
            let at = OFF_FLOATS as usize + 4 * index;
            out[at..at + 4].copy_from_slice(&float.to_bits().to_le_bytes());
        }
        for (index, int) in self.ints.iter().enumerate() {
            let at = OFF_INTS as usize + 4 * index;
            out[at..at + 4].copy_from_slice(&int.to_le_bytes());
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Record> {
        if bytes.len() < RECORD_BYTES {
            return None;
        }
        let op = Op::from_code(u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?))?;
        let size = u16::from_le_bytes(bytes.get(4..6)?.try_into().ok()?);
        if u64::from(size) != RECORD_SIZE {
            return None;
        }
        let flags = u16::from_le_bytes(bytes.get(6..8)?.try_into().ok()?);
        let count = u32::from_le_bytes(bytes.get(8..12)?.try_into().ok()?);
        let blob = u32::from_le_bytes(bytes.get(12..16)?.try_into().ok()?);
        let blob_len = u32::from_le_bytes(bytes.get(16..20)?.try_into().ok()?);
        let mut words = [0u64; 9];
        for (index, word) in words.iter_mut().enumerate() {
            let at = OFF_WORDS as usize + 8 * index;
            *word = u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?);
        }
        let mut floats = [0.0f32; 4];
        for (index, float) in floats.iter_mut().enumerate() {
            let at = OFF_FLOATS as usize + 4 * index;
            *float = f32::from_bits(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
        }
        let mut ints = [0i32; 4];
        for (index, int) in ints.iter_mut().enumerate() {
            let at = OFF_INTS as usize + 4 * index;
            *int = i32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?);
        }
        Some(Record {
            op,
            flags,
            count,
            blob,
            blob_len,
            words,
            floats,
            ints,
        })
    }
}

/// What a list has used so far, and what the recorder has written in total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub lists: u64,
    pub implicit: u64,
    pub records: u64,
    pub bytes: u64,
    pub blobs: u64,
    pub blob_bytes: u64,
}

impl Usage {
    pub fn total_bytes(&self) -> u64 {
        self.bytes + self.blob_bytes
    }
}

/// A live list: the bytes it reserved, what it used, and how deep it nests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct List {
    /// The handle the game submits and a `CallCommands` names. For a list recorded in guest memory this is the
    /// address of its first record; for an arena list it is the handle the driver's `EndRecording` returned.
    pub key: u64,
    /// Where this list's records were placed. Same as [`List::key`] for a guest list, the arena storage address for
    /// an implicit one; record and blob offsets are relative to it.
    pub handle: u64,
    /// The list lives in [`Recorder`]'s arena, not in the game's command memory.
    pub arena: bool,
    /// The recorder generation this list was opened at. A host that holds a key across records compares it, so a
    /// recycled arena slot is not mistaken for the list it held.
    pub generation: u64,
    /// The arena records themselves, one [`RECORD_SIZE`] per command. Empty for a guest list.
    pub data: Vec<u8>,
    pub cmdbuf: u64,
    pub capacity: u64,
    pub records: u32,
    /// Draw records in this list, so the report can tell a captured draw from one that was recorded only.
    pub draws: u32,
    pub bytes: u64,
    pub blobs: u32,
    pub blob_bytes: u64,
    /// Absolute address its blobs grow down from: the span end, or the lowest blob start of the lists already
    /// recorded in the same command memory.
    pub blob_top: u64,
    /// Absolute address of each blob this list wrote, so its host copy can be retired with the list.
    pub blob_addrs: Vec<u64>,
    /// Lists whose records precede this one in the same command stream, in recording order. An implicit chunk the
    /// driver closed under an internal pointer is one of these: the list that follows it continues its state, so a
    /// replay walks it first and does not copy its records (`docs/nvn/CONTRACT.md` §3).
    pub chained: Vec<u64>,
    /// Nesting below this list; 0 when it names no other list and continues none.
    pub depth: u32,
}

/// The recorder's view of a frame, small enough to keep in the host snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub live: u64,
    pub lists: u64,
    /// Lists the host opened without `BeginRecording` and recorded in its own arena.
    pub implicit: u64,
    pub records: u64,
    pub bytes: u64,
    pub blob_bytes: u64,
    pub submits: u64,
    pub rejected: u64,
    /// Draw records the recorder took in, whether or not anything ever submitted them.
    pub recorded_draws: u64,
    /// Clear records, same rule.
    pub recorded_clears: u64,
    /// Chain links a live list still carries, and arena lists retired once their submit was replayed.
    pub chained: u64,
    pub retired: u64,
}

#[derive(Default)]
pub struct Recorder {
    active: BTreeMap<u64, u64>,
    lists: BTreeMap<u64, List>,
    /// The bytes of every live blob, by absolute address. The command memory is the record of truth for the game,
    /// but the driver and the next lists reuse that memory while a submit is queued; the host keeps the copy a
    /// replay reads so a queued submit sees what the game recorded, not what was written over it.
    blobs: BTreeMap<u64, Vec<u8>>,
    /// Arena storage address -> the key its list is filed under, so a reused slot retires the list that had it.
    arena_owners: BTreeMap<u64, u64>,
    /// Next arena slot to hand out.
    arena_next: u64,
    /// Live lists that continue each chunk, by the chunk's key. The slot rotation skips a chunk a live list still
    /// names, so a chain never loses a rule it replays.
    chain_refs: BTreeMap<u64, u32>,
    /// Chunks the host is holding for a handle that has not arrived yet. The rotation skips them: recycling one
    /// would make the host chain a different list than the one it recorded.
    protected: std::collections::HashSet<u64>,
    usage: Usage,
    submits: u64,
    rejected: u64,
    recorded_draws: u64,
    recorded_clears: u64,
    retired: u64,
    generation: u64,
    last: Option<u64>,
}

impl Recorder {
    /// Starts a list at `base` (the current write pointer) with `capacity` bytes available. Any list whose used
    /// bytes that range walks over is retired, so its handle stops resolving. The driver names the start of the
    /// memory it handed over, which is not necessarily 8-byte aligned (`AddCommandMemory` may be called at a +4
    /// offset), so records are aligned relative to the handle rather than to an absolute boundary.
    pub fn begin(
        &mut self,
        mem: &dyn Memory,
        cmdbuf: u64,
        base: u64,
        capacity: u64,
    ) -> Result<u64, Error> {
        if capacity < 3 * RECORD_SIZE {
            return Err(Error::OutOfRange);
        }
        self.retire_at(base);
        self.generation += 1;
        let generation = self.generation;
        let list = List {
            key: base,
            handle: base,
            generation,
            cmdbuf,
            capacity,
            blob_top: base.checked_add(capacity).ok_or(Error::Overflow)?,
            ..List::default()
        };
        self.lists.insert(base, list);
        self.active.insert(cmdbuf, base);
        let mut record = Record::new(Op::Begin);
        record.set_word(0, cmdbuf);
        record.set_word(1, generation);
        record.set_word(2, capacity);
        if let Err(error) = self.append(mem, cmdbuf, record, None, RECORD_SIZE) {
            self.abort(cmdbuf);
            return Err(error);
        }
        Ok(base)
    }

    /// Opens a list the driver started on its own, in the host arena. Nothing is written to the game's command
    /// memory: the driver is the only writer of its pushbuffer (`docs/nvn/CONTRACT.md` §3). The key is the storage
    /// address until the caller refiles the closed list under the handle the driver returned.
    pub fn begin_arena(&mut self, mem: &dyn Memory, cmdbuf: u64) -> Result<u64, Error> {
        // Never take back the slot of a list that is still being recorded, still waiting for its handle, or still
        // continued by another list: the driver cannot reuse memory it is writing into or that a live list names.
        // The held slots are few, so a short scan finds a free one.
        let mut storage = None;
        for _ in 0..ARENA_SLOTS {
            let candidate = ARENA_BASE
                .checked_add((self.arena_next % ARENA_SLOTS) * ARENA_SLOT)
                .ok_or(Error::Overflow)?;
            self.arena_next = self.arena_next.wrapping_add(1);
            if !self.slot_in_use(candidate) {
                storage = Some(candidate);
                break;
            }
        }
        // Every slot is still reachable: the caller reports it and that list stays with the driver instead of the
        // rotation dropping a list an open buffer or a live chain still names.
        let storage = storage.ok_or(Error::Full)?;
        // A slot that comes around again wins over whatever list still held it, the same rule guest memory has.
        if let Some(old) = self.arena_owners.insert(storage, storage) {
            self.forget(old);
        }
        self.generation += 1;
        let generation = self.generation;
        let list = List {
            key: storage,
            handle: storage,
            arena: true,
            generation,
            cmdbuf,
            capacity: ARENA_CAPACITY,
            blob_top: storage.checked_add(ARENA_CAPACITY).ok_or(Error::Overflow)?,
            ..List::default()
        };
        self.lists.insert(storage, list);
        self.active.insert(cmdbuf, storage);
        let mut record = Record::new(Op::Begin);
        record.set_word(0, cmdbuf);
        record.set_word(1, generation);
        record.set_word(2, ARENA_CAPACITY);
        // Word 3 marks an implicit list, so a snapshot says where its records came from.
        record.set_word(3, 1);
        if let Err(error) = self.append(mem, cmdbuf, record, None, RECORD_SIZE) {
            self.abort(cmdbuf);
            return Err(error);
        }
        Ok(storage)
    }

    /// Closes the list, writes its `End` record and returns the handle the game will submit.
    pub fn end(&mut self, mem: &dyn Memory, cmdbuf: u64) -> Result<u64, Error> {
        let handle = *self.active.get(&cmdbuf).ok_or(Error::NotRecording)?;
        let list = self
            .lists
            .get(&handle)
            .cloned()
            .ok_or(Error::NotRecording)?;
        let mut record = Record::new(Op::End);
        record.set_word(0, u64::from(list.records));
        record.set_word(1, list.bytes);
        record.set_word(2, list.blob_bytes);
        self.append(mem, cmdbuf, record, None, 0)?;
        self.active.remove(&cmdbuf);
        self.usage.lists += 1;
        self.last = Some(handle);
        Ok(handle)
    }

    /// Closes an implicit list and keeps it filed under its storage key. The caller files it under the handle the
    /// driver reported with [`Recorder::refile`], or drops it when that report is a driver-internal pointer no game
    /// call can name.
    pub fn end_arena(&mut self, mem: &dyn Memory, cmdbuf: u64) -> Result<u64, Error> {
        let storage = *self.active.get(&cmdbuf).ok_or(Error::NotRecording)?;
        let list = self
            .lists
            .get(&storage)
            .cloned()
            .ok_or(Error::NotRecording)?;
        let mut record = Record::new(Op::End);
        record.set_word(0, u64::from(list.records));
        record.set_word(1, list.bytes);
        record.set_word(2, list.blob_bytes);
        let result = self.append(mem, cmdbuf, record, None, 0);
        self.active.remove(&cmdbuf);
        if let Err(error) = result {
            self.forget(storage);
            return Err(error);
        }
        self.usage.lists += 1;
        self.usage.implicit += 1;
        self.last = Some(storage);
        Ok(storage)
    }

    /// Files a closed arena list under `key`, the handle a submit or a `CallCommands` will name. Whatever list the
    /// key held before is retired: the newer list wins (`CONTRACT.md` §3, "Discard").
    pub fn refile(&mut self, key: u64, new_key: u64) -> bool {
        if key == new_key {
            return self.lists.contains_key(&key);
        }
        let Some(mut list) = self.lists.remove(&key) else {
            return false;
        };
        if new_key != list.handle {
            self.forget(new_key);
        }
        list.key = new_key;
        self.arena_owners.insert(list.handle, new_key);
        self.lists.insert(new_key, list);
        self.last = Some(new_key);
        true
    }

    /// Drops a list by key (a superseded arena list that was never filed under a usable handle).
    pub fn drop_list(&mut self, key: u64) {
        self.forget(key);
    }

    /// Marks a chunk the host is keeping until the handle that continues it arrives.
    pub fn pin(&mut self, key: u64) {
        if self.lists.contains_key(&key) {
            self.protected.insert(key);
        }
    }

    /// Releases a chunk the host consumed (chained, refiled or dropped).
    pub fn unpin(&mut self, key: u64) {
        self.protected.remove(&key);
    }

    /// Whether an arena slot is still reachable: the list being recorded on some buffer, a chunk a live list
    /// continues, or one the host holds for a handle that has not arrived. Any other slot may be handed out again,
    /// the way guest command memory is reused.
    fn slot_in_use(&self, key: u64) -> bool {
        self.protected.contains(&key)
            || self.chain_refs.get(&key).is_some_and(|count| *count > 0)
            || self.active.values().any(|value| *value == key)
    }

    /// The generation of a live list, so a key held across records can be checked against a recycled slot.
    pub fn generation(&self, key: u64) -> Option<u64> {
        self.lists.get(&key).map(|list| list.generation)
    }

    /// Attaches `predecessors` to `key`, in recording order: a replay walks their records before the list's own,
    /// the way one command stream continues across the driver's internal `EndRecording`. Nothing is copied: the
    /// chunks keep their own records and blobs, and the list names them until it is retired.
    ///
    /// A chain is a nesting level of the same limit `CallCommands` uses (`docs/nvn/CONTRACT.md` §4), and a list
    /// cannot continue itself.
    pub fn chain(&mut self, key: u64, predecessors: &[u64]) -> Result<(), Error> {
        if predecessors.is_empty() {
            return Ok(());
        }
        if predecessors.contains(&key) {
            self.rejected += 1;
            return Err(Error::Nesting);
        }
        let list = self.lists.get(&key).ok_or(Error::UnknownList)?;
        let mut depth = list.depth;
        for &predecessor in predecessors {
            let Some(source) = self.lists.get(&predecessor) else {
                self.rejected += 1;
                return Err(Error::UnknownList);
            };
            depth = depth.max(1 + source.depth);
        }
        if depth > MAX_NESTING {
            self.rejected += 1;
            return Err(Error::Nesting);
        }
        let Some(list) = self.lists.get_mut(&key) else {
            return Err(Error::UnknownList);
        };
        for &predecessor in predecessors {
            if !list.chained.contains(&predecessor) {
                list.chained.push(predecessor);
                *self.chain_refs.entry(predecessor).or_default() += 1;
            }
        }
        list.depth = depth;
        Ok(())
    }

    /// Retires the arena lists of a replayed submit. The submit's snapshot owns their records while it is queued,
    /// and once its replay is done no queued replay can name them (`CONTRACT.md` §7); a later submit that names one
    /// of them is a handle the host no longer records, which stays with the driver. Guest lists are the game's own
    /// command memory and are never retired here.
    pub fn retire_replayed(&mut self, keys: &[u64]) -> u64 {
        let mut retired = 0;
        for &key in keys {
            let arena = self.lists.get(&key).is_some_and(|list| list.arena);
            if arena && !self.protected.contains(&key) {
                let before = self.lists.len();
                self.forget(key);
                retired += (before - self.lists.len()) as u64;
            }
        }
        self.retired += retired;
        retired
    }

    pub fn emit(&mut self, mem: &dyn Memory, cmdbuf: u64, record: Record) -> Result<u64, Error> {
        self.append(mem, cmdbuf, record, None, RECORD_SIZE)
    }

    /// Appends a record and copies `blob` into the list, with the record's blob offset pointing at the copy.
    pub fn emit_blob(
        &mut self,
        mem: &dyn Memory,
        cmdbuf: u64,
        record: Record,
        blob: &[u8],
    ) -> Result<u64, Error> {
        self.append(mem, cmdbuf, record, Some(blob), RECORD_SIZE)
    }

    /// Appends a record whose blob is the decoded state of the object at `address` (`words[0]`).
    pub fn emit_state(
        &mut self,
        mem: &dyn Memory,
        cmdbuf: u64,
        op: Op,
        address: u64,
        values: &[StateValue],
    ) -> Result<u64, Error> {
        let mut record = Record::new(op);
        record.set_word(0, address);
        record.count = values.len() as u32;
        let mut blob = Vec::with_capacity(values.len() * StateValue::BYTES as usize);
        for value in values {
            blob.extend_from_slice(&value.encode());
        }
        self.append(mem, cmdbuf, record, Some(&blob), RECORD_SIZE)
    }

    /// Appends a raw copy of a state object the host tables did not decode.
    pub fn emit_raw_state(
        &mut self,
        mem: &dyn Memory,
        cmdbuf: u64,
        op: Op,
        address: u64,
    ) -> Result<u64, Error> {
        let bytes = mem.read(address, RAW_STATE_BYTES);
        let mut record = Record::new(op);
        record.set_word(0, address);
        match bytes.filter(|b| !b.is_empty()) {
            Some(bytes) => {
                record.flags |= FLAG_RAW;
                record.count = bytes.len() as u32;
                self.append(mem, cmdbuf, record, Some(&bytes), RECORD_SIZE)
            }
            None => {
                record.flags |= FLAG_UNRESOLVED;
                self.append(mem, cmdbuf, record, None, RECORD_SIZE)
            }
        }
    }

    /// `CallCommands`: the record names the callee handles; their records stay where they are.
    pub fn call(&mut self, mem: &dyn Memory, cmdbuf: u64, handles: &[u64]) -> Result<u64, Error> {
        self.nested(mem, cmdbuf, Op::CallCommands, handles)
    }

    /// `CopyCommands`: the callee's command records are copied into this list now, so a later overwrite of the
    /// callee cannot change them (assumption: the real semantics are unconfirmed, `CONTRACT.md` §10).
    pub fn copy(&mut self, mem: &dyn Memory, cmdbuf: u64, handles: &[u64]) -> Result<u64, Error> {
        if handles.is_empty() || handles.len() > MAX_HANDLES {
            self.rejected += 1;
            return Err(Error::OutOfRange);
        }
        let caller = *self.active.get(&cmdbuf).ok_or(Error::NotRecording)?;
        let mut depth = self.lists.get(&caller).ok_or(Error::NotRecording)?.depth;
        let mut last = 0;
        for &handle in handles {
            let Some(source) = self.lists.get(&handle).cloned() else {
                self.rejected += 1;
                return Err(Error::UnknownList);
            };
            depth = depth.max(1 + source.depth);
            if depth > MAX_NESTING {
                self.rejected += 1;
                return Err(Error::Nesting);
            }
            last = self.copy_one(mem, cmdbuf, &source)?;
            if let Some(list) = self.lists.get_mut(&caller) {
                list.depth = list.depth.max(depth);
            }
        }
        Ok(last)
    }

    /// Notes a submitted batch, for the report. `replay::submit` queues the actual job.
    pub fn submit(&mut self, handles: &[u64]) -> u64 {
        self.submits += 1;
        handles.len() as u64
    }

    /// `nvnCommandBufferInitialize`/`Finalize`: the buffer object is set up again, but lists it already ended stay
    /// valid until their bytes are recorded over — the game keeps calling them while the buffer is reused. Only a
    /// list that was still recording is dropped here.
    pub fn initialize(&mut self, cmdbuf: u64) {
        self.abort(cmdbuf);
    }

    pub fn is_recording(&self, cmdbuf: u64) -> bool {
        self.active.contains_key(&cmdbuf)
    }

    pub fn handle(&self, cmdbuf: u64) -> Option<u64> {
        self.active.get(&cmdbuf).copied()
    }

    pub fn list(&self, handle: u64) -> Option<&List> {
        self.lists.get(&handle)
    }

    pub fn live(&self) -> usize {
        self.lists.len()
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    pub fn last_handle(&self) -> Option<u64> {
        self.last
    }

    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// The recorder's generation: every `begin` bumps it, so it stamps the end of a recording.
    pub fn mark(&self) -> u64 {
        self.generation
    }

    pub fn report(&self) -> Report {
        Report {
            live: self.lists.len() as u64,
            lists: self.usage.lists,
            implicit: self.usage.implicit,
            records: self.usage.records,
            bytes: self.usage.bytes,
            blob_bytes: self.usage.blob_bytes,
            submits: self.submits,
            rejected: self.rejected,
            recorded_draws: self.recorded_draws,
            recorded_clears: self.recorded_clears,
            chained: self
                .lists
                .values()
                .map(|list| list.chained.len() as u64)
                .sum(),
            retired: self.retired,
        }
    }

    fn append(
        &mut self,
        mem: &dyn Memory,
        cmdbuf: u64,
        mut record: Record,
        blob: Option<&[u8]>,
        reserve: u64,
    ) -> Result<u64, Error> {
        let handle = *self.active.get(&cmdbuf).ok_or(Error::NotRecording)?;
        let list = self.lists.get_mut(&handle).ok_or(Error::NotRecording)?;
        let has_blob = blob.is_some_and(|bytes| !bytes.is_empty());
        let mut blob_len = 0u64;
        let mut blob_at = list.blob_top;
        if let Some(bytes) = blob.filter(|bytes| !bytes.is_empty()) {
            blob_len = bytes.len() as u64;
            let reserved = align8(blob_len)
                .and_then(|len| list.blob_bytes.checked_add(len))
                .ok_or(Error::Overflow)?;
            let top = list.blob_top.checked_sub(reserved).ok_or(Error::Full)?;
            // A list with no room left puts its blob below its own first record: that is a full list, not a bad
            // address range, and the caller reports it as such.
            blob_at = top & !7u64;
            let offset = blob_at.checked_sub(handle).ok_or(Error::Full)?;
            if offset > u64::from(u32::MAX) || blob_len > u64::from(u32::MAX) {
                return Err(Error::Full);
            }
            record.blob = offset as u32;
            record.blob_len = blob_len as u32;
            record.flags |= FLAG_BLOB;
        }
        let records_end = list.bytes.checked_add(RECORD_SIZE).ok_or(Error::Overflow)?;
        let reserved_blobs = align8(blob_len)
            .and_then(|len| list.blob_bytes.checked_add(len))
            .ok_or(Error::Overflow)?;
        // Everything this record and its blob need has to stay below the blob area and leave the end record room.
        let blob_room = list
            .blob_top
            .checked_sub(reserved_blobs)
            .ok_or(Error::Full)?;
        if handle
            .checked_add(records_end)
            .and_then(|end| end.checked_add(reserve))
            .is_none_or(|end| end > blob_room)
        {
            return Err(Error::Full);
        }
        let encoded = record.encode();
        if list.arena {
            // The arena is ours: records and blobs stay in this process and the driver's memory is not touched.
            list.data.extend_from_slice(&encoded);
        } else {
            if has_blob && !mem.write(blob_at, blob.unwrap_or_default()) {
                return Err(Error::Unmapped);
            }
            let at = handle.checked_add(list.bytes).ok_or(Error::Overflow)?;
            if !mem.write(at, &encoded) {
                return Err(Error::Unmapped);
            }
        }
        let at = handle.checked_add(list.bytes).ok_or(Error::Overflow)?;
        list.records += 1;
        list.bytes = records_end;
        if has_blob {
            list.blobs += 1;
            list.blob_bytes = reserved_blobs;
            list.blob_addrs.push(blob_at);
            self.blobs
                .insert(blob_at, blob.unwrap_or_default().to_vec());
        }
        self.usage.records += 1;
        self.usage.bytes += RECORD_SIZE;
        match record.op {
            Op::DrawArrays | Op::DrawElements => {
                self.recorded_draws += 1;
                list.draws += 1;
            }
            Op::ClearColor | Op::ClearDepthStencil => self.recorded_clears += 1,
            _ => {}
        }
        if has_blob {
            self.usage.blobs += 1;
            self.usage.blob_bytes += blob_len;
        }
        Ok(at)
    }

    fn nested(
        &mut self,
        mem: &dyn Memory,
        cmdbuf: u64,
        op: Op,
        handles: &[u64],
    ) -> Result<u64, Error> {
        if handles.is_empty() || handles.len() > MAX_HANDLES {
            self.rejected += 1;
            return Err(Error::OutOfRange);
        }
        let caller = *self.active.get(&cmdbuf).ok_or(Error::NotRecording)?;
        let mut depth = 0;
        for &handle in handles {
            let Some(source) = self.lists.get(&handle) else {
                self.rejected += 1;
                return Err(Error::UnknownList);
            };
            depth = depth.max(1 + source.depth);
        }
        let list = self.lists.get(&caller).ok_or(Error::NotRecording)?;
        depth = depth.max(list.depth);
        if depth > MAX_NESTING {
            self.rejected += 1;
            return Err(Error::Nesting);
        }
        let mut blob = Vec::with_capacity(handles.len() * 8);
        for handle in handles {
            blob.extend_from_slice(&handle.to_le_bytes());
        }
        let mut record = Record::new(op);
        record.flags |= FLAG_NESTED;
        record.count = handles.len() as u32;
        record.set_word(0, handles[0]);
        let at = self.append(mem, cmdbuf, record, Some(&blob), RECORD_SIZE)?;
        if let Some(list) = self.lists.get_mut(&caller) {
            list.depth = depth;
        }
        Ok(at)
    }

    /// Copies one list's command records (everything between its `Begin` and `End`) into the caller.
    fn copy_one(&mut self, mem: &dyn Memory, cmdbuf: u64, source: &List) -> Result<u64, Error> {
        if source.bytes == 0 || source.bytes > MAX_COPY_BYTES {
            self.rejected += 1;
            return Err(Error::OutOfRange);
        }
        let bytes = if source.arena {
            source
                .data
                .get(..source.bytes as usize)
                .map(<[u8]>::to_vec)
                .ok_or(Error::Corrupt)?
        } else {
            mem.read(source.handle, source.bytes)
                .ok_or(Error::Unmapped)?
        };
        let mut records = Vec::new();
        let mut blob_total = 0u64;
        for chunk in bytes.as_chunks::<RECORD_BYTES>().0 {
            let record = Record::decode(chunk).ok_or(Error::Corrupt)?;
            let blob = if record.flags & FLAG_BLOB != 0 {
                let at = source
                    .handle
                    .checked_add(u64::from(record.blob))
                    .ok_or(Error::Overflow)?;
                let data = if source.arena {
                    self.blobs.get(&at).cloned().ok_or(Error::Unmapped)?
                } else {
                    mem.read(at, u64::from(record.blob_len))
                        .ok_or(Error::Unmapped)?
                };
                blob_total = align8(data.len() as u64)
                    .and_then(|len| blob_total.checked_add(len))
                    .ok_or(Error::Overflow)?;
                Some(data)
            } else {
                None
            };
            records.push((record, blob));
        }
        if records.len() < 2
            || records.first().map(|(r, _)| r.op) != Some(Op::Begin)
            || records.last().map(|(r, _)| r.op) != Some(Op::End)
        {
            self.rejected += 1;
            return Err(Error::Corrupt);
        }
        let inner = &records[1..records.len() - 1];
        let caller = *self.active.get(&cmdbuf).ok_or(Error::NotRecording)?;
        let list = self.lists.get(&caller).ok_or(Error::NotRecording)?;
        let needed = (inner.len() as u64 + 1)
            .checked_mul(RECORD_SIZE)
            .and_then(|bytes| bytes.checked_add(blob_total))
            .and_then(|bytes| bytes.checked_add(RECORD_SIZE))
            .ok_or(Error::Overflow)?;
        if list
            .bytes
            .checked_add(needed)
            .is_none_or(|used| used > list.capacity)
        {
            self.rejected += 1;
            return Err(Error::Full);
        }
        let mut marker = Record::new(Op::CopyCommands);
        marker.flags |= FLAG_NESTED | FLAG_INLINE;
        marker.count = inner.len() as u32;
        marker.set_word(0, source.handle);
        marker.set_word(1, source.bytes);
        let mut at = self.append(mem, cmdbuf, marker, None, RECORD_SIZE)?;
        let inner = 1..records.len() - 1;
        for (record, blob) in records.drain(inner) {
            at = self.append(mem, cmdbuf, record, blob.as_deref(), RECORD_SIZE)?;
        }
        Ok(at)
    }

    /// Drops the list `cmdbuf` is recording, and everything it wrote.
    pub fn abort(&mut self, cmdbuf: u64) {
        if let Some(handle) = self.active.remove(&cmdbuf) {
            self.forget(handle);
        }
    }

    /// Drops a list and the host copy of its blobs. A chunk the list continues goes with it, unless the host still
    /// holds it or another live list names it: a chain is only reachable through the list that carries it.
    fn forget(&mut self, key: u64) -> Option<List> {
        let list = self.lists.remove(&key)?;
        self.protected.remove(&key);
        // Whatever named this list stops naming it: a chain is only as long as the list that carries it. Every
        // reference is dropped here, so a recycled arena slot can never be reached through an older chain.
        for other in self.lists.values_mut() {
            if other.chained.contains(&key) {
                other.chained.retain(|chained| *chained != key);
                if let Some(count) = self.chain_refs.get_mut(&key) {
                    *count = count.saturating_sub(1);
                }
            }
        }
        self.chain_refs.remove(&key);
        for at in &list.blob_addrs {
            self.blobs.remove(at);
        }
        if list.arena {
            self.arena_owners.remove(&list.handle);
        }
        for chained in &list.chained {
            let last = self
                .chain_refs
                .get_mut(chained)
                .map(|count| {
                    *count = count.saturating_sub(1);
                    *count == 0
                })
                .unwrap_or(false);
            if last && !self.protected.contains(chained) {
                self.forget(*chained);
            }
        }
        Some(list)
    }

    /// The host copy of a blob a list recorded, by its offset from the list's handle.
    pub fn blob(&self, key: u64, offset: u32) -> Option<&[u8]> {
        let list = self.lists.get(&key)?;
        self.blobs
            .get(&list.handle.checked_add(u64::from(offset))?)
            .map(Vec::as_slice)
    }

    /// The records of a list as they were recorded, for a submit to snapshot: the arena copy when the host owns
    /// them, the game's command memory otherwise.
    pub fn capture_bytes(&self, mem: &dyn Memory, key: u64, bytes: u64) -> Option<Vec<u8>> {
        let list = self.lists.get(&key)?;
        let len = bytes.min(list.bytes) as usize;
        if list.arena {
            list.data.get(..len).map(<[u8]>::to_vec)
        } else {
            mem.read(list.handle, len as u64)
        }
    }

    /// Whether the list being recorded for `cmdbuf` lives in the host arena.
    pub fn is_arena(&self, cmdbuf: u64) -> bool {
        self.active
            .get(&cmdbuf)
            .and_then(|key| self.lists.get(key))
            .is_some_and(|list| list.arena)
    }

    /// Retires every live list whose used bytes the new list's first record lands on, so a caller that resolves the
    /// write pointer before opening a list sees the room the new one will actually have. A list recorded over the
    /// same command memory wins, and the older handle stops resolving (`CONTRACT.md` §3, "Discard"). Lists recorded
    /// sequentially after an older one do not overlap it, and both stay live. Arena lists are keyed in a synthetic
    /// address range of their own and are never retired by a guest write.
    pub fn retire_at(&mut self, start: u64) -> usize {
        let first = start.saturating_add(RECORD_SIZE);
        let dead: Vec<u64> = self
            .lists
            .iter()
            .filter(|(_, list)| {
                if list.arena {
                    return false;
                }
                let used_until = list
                    .handle
                    .saturating_add(list.bytes)
                    .saturating_add(list.blob_bytes);
                list.handle < first && start < used_until
            })
            .map(|(key, _)| *key)
            .collect();
        for key in &dead {
            self.forget(*key);
        }
        self.active.retain(|_, key| !dead.contains(key));
        dead.len()
    }

    /// Lowest blob start among the lists recorded in `[span_start, span_end)`, so a new list's blobs go below the
    /// ones already there instead of over them.
    pub fn blob_floor(&self, span_start: u64, span_end: u64) -> u64 {
        self.lists
            .values()
            .filter(|list| {
                !list.arena
                    && list.handle >= span_start
                    && list.handle < span_end
                    && list.blob_bytes > 0
            })
            .map(|list| list.blob_top.saturating_sub(list.blob_bytes))
            .min()
            .unwrap_or(span_end)
    }
}

fn align8(bytes: u64) -> Option<u64> {
    bytes.checked_add(7).map(|bytes| bytes & !7)
}

#[cfg(test)]
pub(crate) struct FakeMemory {
    base: u64,
    data: std::cell::RefCell<Vec<u8>>,
}

#[cfg(test)]
impl FakeMemory {
    pub(crate) fn new(base: u64, len: u64) -> FakeMemory {
        FakeMemory {
            base,
            data: std::cell::RefCell::new(vec![0; len as usize]),
        }
    }

    pub(crate) fn poke(&self, at: u64, bytes: &[u8]) {
        assert!(self.write(at, bytes), "poke outside the fake memory");
    }

    pub(crate) fn read_back(&self, at: u64, len: u64) -> Vec<u8> {
        self.read(at, len).expect("read inside the fake memory")
    }
}

#[cfg(test)]
impl Memory for FakeMemory {
    fn read(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        let start = addr.checked_sub(self.base)? as usize;
        let end = start.checked_add(len as usize)?;
        self.data.borrow().get(start..end).map(<[u8]>::to_vec)
    }

    fn write(&self, addr: u64, bytes: &[u8]) -> bool {
        let Some(start) = addr.checked_sub(self.base).map(|at| at as usize) else {
            return false;
        };
        let Some(end) = start.checked_add(bytes.len()) else {
            return false;
        };
        let mut data = self.data.borrow_mut();
        match data.get_mut(start..end) {
            Some(slot) => {
                slot.copy_from_slice(bytes);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x1_0000_0000;
    const CMD: u64 = 0x4000;
    const OTHER: u64 = 0x5000;
    const CAPACITY: u64 = 0x1000;

    fn memory() -> FakeMemory {
        FakeMemory::new(BASE, 0x4000)
    }

    fn records(mem: &FakeMemory, handle: u64, count: usize) -> Vec<Record> {
        (0..count)
            .map(|index| {
                let at = handle + index as u64 * RECORD_SIZE;
                Record::decode(&mem.read_back(at, RECORD_SIZE)).expect("record")
            })
            .collect()
    }

    fn blob(mem: &FakeMemory, handle: u64, record: &Record) -> Vec<u8> {
        mem.read_back(handle + u64::from(record.blob), u64::from(record.blob_len))
    }

    fn scripted(mem: &FakeMemory) -> (Recorder, u64) {
        let mut recorder = Recorder::default();
        recorder
            .begin(mem, CMD, BASE, CAPACITY)
            .expect("begin recording");
        let mut blend = Record::new(Op::BindBlendState);
        blend.set_word(0, 0x7000);
        recorder.emit(mem, CMD, blend).expect("bind blend state");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 12);
        draw.set_word(1, 3);
        draw.set_int(2, 1);
        recorder.emit(mem, CMD, draw).expect("draw");
        let end = recorder.end(mem, CMD).expect("end recording");
        (recorder, end)
    }

    #[test]
    fn records_sit_at_fixed_offsets_and_the_handle_is_the_first_one() {
        let mem = memory();
        let (recorder, handle) = scripted(&mem);
        assert_eq!(handle, BASE);
        assert_eq!(recorder.last_handle(), Some(BASE));
        let list = recorder.list(handle).expect("list is live");
        assert_eq!(
            (list.records, list.bytes, list.blob_bytes),
            (4, 4 * 0x80, 0)
        );

        let (begin, bind, draw, end) = {
            let all = records(&mem, handle, 4);
            (all[0], all[1], all[2], all[3])
        };
        assert_eq!(begin.op, Op::Begin);
        assert_eq!(begin.words[0], CMD);
        assert_eq!(begin.words[2], CAPACITY);

        let bytes = mem.read_back(handle, RECORD_SIZE);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            Op::Begin.code()
        );
        assert_eq!(
            u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            RECORD_SIZE as u16
        );
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            begin.count
        );

        assert_eq!(bind.op, Op::BindBlendState);
        assert_eq!(bind.words[0], 0x7000);
        assert_eq!(draw.op, Op::DrawArrays);
        assert_eq!(
            (draw.ints[0], draw.words[0], draw.words[1], draw.ints[2]),
            (4, 12, 3, 1)
        );
        assert_eq!(draw.floats, [0.0; 4]);
        assert_eq!(end.op, Op::End);
        assert_eq!((end.words[0], end.words[1]), (3, 3 * 0x80));

        let usage = recorder.usage();
        assert_eq!(
            (usage.lists, usage.records, usage.bytes),
            (1, 4, 4 * RECORD_SIZE)
        );
    }

    #[test]
    fn a_list_may_start_on_a_four_byte_aligned_handle() {
        let mem = FakeMemory::new(BASE, 0x1000);
        let mut recorder = Recorder::default();
        recorder
            .begin(&mem, CMD, BASE + 4, 0x400)
            .expect("the driver may add memory at a +4 offset");
        let mut record = Record::new(Op::Barrier);
        record.set_word(0, 1);
        recorder.emit(&mem, CMD, record).expect("record");
        let handle = recorder.end(&mem, CMD).expect("end");
        assert_eq!(handle, BASE + 4);
        assert_eq!(recorder.list(handle).expect("live").records, 3);
        let decoded =
            Record::decode(&mem.read_back(BASE + 4 + RECORD_SIZE, RECORD_SIZE)).expect("record");
        assert_eq!(decoded.words[0], 1);
    }

    #[test]
    fn sequential_lists_in_one_span_stay_live_and_keep_their_blobs() {
        let mem = FakeMemory::new(BASE, 0x2000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, 0x1000).expect("first");
        let one = [StateValue {
            kind: kind_code(Kind::BlendSrc),
            index: 0,
            value: 7,
        }];
        recorder
            .emit_state(&mem, CMD, Op::BindBlendState, 0x7000, &one)
            .expect("first state");
        let first = recorder.end(&mem, CMD).expect("end first");
        let used = {
            let list = recorder.list(first).expect("first live");
            list.bytes + list.blob_bytes
        };

        let floor = recorder.blob_floor(BASE, BASE + 0x1000);
        recorder
            .begin(&mem, CMD, BASE + used, floor - (BASE + used))
            .expect("second");
        assert!(
            recorder.list(first).is_some(),
            "sequential lists do not overlap"
        );
        let two = [StateValue {
            kind: kind_code(Kind::BlendDst),
            index: 0,
            value: 9,
        }];
        recorder
            .emit_state(&mem, CMD, Op::BindBlendState, 0x7000, &two)
            .expect("second state");
        let second = recorder.end(&mem, CMD).expect("end second");
        assert_ne!(first, second);
        assert_eq!(recorder.live(), 2);

        let first_record =
            Record::decode(&mem.read_back(first + RECORD_SIZE, RECORD_SIZE)).unwrap();
        assert_eq!(
            StateValue::decode(&blob(&mem, first, &first_record)),
            Some(one[0])
        );
        let second_record =
            Record::decode(&mem.read_back(second + RECORD_SIZE, RECORD_SIZE)).unwrap();
        assert_eq!(
            StateValue::decode(&blob(&mem, second, &second_record)),
            Some(two[0])
        );

        // Recording over the first list's bytes retires it and frees its blob space for the next list.
        recorder
            .begin(&mem, CMD, BASE, 0x1000 - used)
            .expect("re-record the first bytes");
        assert_eq!(recorder.list(first).expect("new list").records, 1);
        assert_eq!(
            recorder.live(),
            2,
            "the first list was retired, the second kept"
        );
        assert!(recorder.list(second).is_some());
        assert_eq!(
            recorder.blob_floor(BASE, BASE + 0x1000),
            floor - StateValue::BYTES,
            "the retired list's blob space is free; the second list's is the lowest in use"
        );
    }

    #[test]
    fn a_later_list_over_the_same_bytes_retires_the_older_handle() {
        let mem = memory();
        let (mut recorder, old) = scripted(&mem);
        let new = recorder
            .begin(&mem, CMD, BASE, CAPACITY)
            .expect("re-record the same bytes");
        assert_eq!(new, old);
        assert!(recorder.list(old).is_some(), "the re-recorded list is live");
        assert_eq!(recorder.live(), 1);

        let _ = recorder.end(&mem, CMD).expect("end the new list");
        let handle = recorder.last_handle().expect("handle");
        assert_eq!(handle, old);
        // A list one record into the old range is enough: addresses, not object identity, decide.
        let moved = recorder
            .begin(&mem, CMD, BASE + RECORD_SIZE, CAPACITY)
            .expect("overlap the old range");
        assert_eq!(moved, BASE + RECORD_SIZE);
        assert!(recorder.list(old).is_none(), "the overlapped list is gone");
    }

    #[test]
    fn calls_name_the_callee_and_copies_inline_its_records() {
        let mem = memory();
        let mut recorder = Recorder::default();
        recorder
            .begin(&mem, OTHER, BASE + 0x2000, 0x800)
            .expect("callee");
        let mut bind = Record::new(Op::BindTexture);
        bind.set_int(0, 1);
        bind.set_int(1, 2);
        bind.set_word(0, 0xdead_beef);
        recorder.emit(&mem, OTHER, bind).expect("bind texture");
        let mut state = Record::new(Op::BindPolygonState);
        state.set_word(0, 0x7100);
        recorder.emit(&mem, OTHER, state).expect("bind polygon");
        let callee = recorder.end(&mem, OTHER).expect("end callee");
        assert_eq!(callee, BASE + 0x2000);
        assert_eq!(recorder.list(callee).expect("callee live").records, 4);

        let caller = recorder.begin(&mem, CMD, BASE, CAPACITY).expect("caller");
        recorder.call(&mem, CMD, &[callee]).expect("call");
        let copied = recorder.copy(&mem, CMD, &[callee]).expect("copy");
        assert!(copied > caller);
        let handle = recorder.end(&mem, CMD).expect("end caller");

        let list = recorder.list(handle).expect("caller live");
        assert_eq!(list.depth, 1);
        let all = records(&mem, handle, list.records as usize);
        let call = all[1];
        assert_eq!(call.op, Op::CallCommands);
        assert_eq!(call.count, 1);
        assert!(call.flags & FLAG_NESTED != 0 && call.flags & FLAG_INLINE == 0);
        let handles = blob(&mem, handle, &call);
        assert_eq!(handles.len(), 8);
        assert_eq!(
            u64::from_le_bytes(handles.as_slice().try_into().unwrap()),
            callee
        );

        // The copy marker is followed by the callee's two commands, not by its Begin/End framing.
        let marker = all[2];
        assert_eq!(marker.op, Op::CopyCommands);
        assert_eq!(marker.count, 2);
        assert!(marker.flags & FLAG_INLINE != 0);
        assert_eq!(marker.words[0], callee);
        assert_eq!(all[3].op, Op::BindTexture);
        assert_eq!(all[3].words[0], 0xdead_beef);
        assert_eq!(all[4].op, Op::BindPolygonState);
        assert_eq!(all[5].op, Op::End);

        // Overwriting the callee's bytes leaves the copy untouched.
        let copied_before: Vec<(u8, u64)> = all[3..5]
            .iter()
            .map(|record| (record.op.code() as u8, record.words[0]))
            .collect();
        mem.poke(BASE + 0x2000, &vec![0xff; 0x800]);
        let all = records(&mem, handle, list.records as usize);
        let copied_after: Vec<(u8, u64)> = all[3..5]
            .iter()
            .map(|record| (record.op.code() as u8, record.words[0]))
            .collect();
        assert_eq!(copied_before, copied_after);
    }

    #[test]
    fn nesting_stops_at_the_replay_limit() {
        let mem = FakeMemory::new(BASE, 0x3_0000);
        let mut recorder = Recorder::default();
        let mut deepest = None;
        for level in 0..MAX_NESTING {
            let base = BASE + u64::from(level) * 0x2000;
            recorder
                .begin(&mem, CMD, base, 0x1000)
                .expect("list per level");
            if let Some(callee) = deepest {
                recorder.call(&mem, CMD, &[callee]).expect("nested call");
            }
            deepest = Some(recorder.end(&mem, CMD).expect("end level"));
        }
        let depth = MAX_NESTING - 1;
        assert_eq!(
            recorder
                .list(deepest.expect("deepest"))
                .expect("live")
                .depth,
            depth
        );

        // One more level is still inside the limit.
        let base = BASE + u64::from(MAX_NESTING) * 0x2000;
        recorder
            .begin(&mem, CMD, base, 0x1000)
            .expect("caller at the limit");
        recorder
            .call(&mem, CMD, &[deepest.expect("deepest")])
            .expect("call at the limit");
        let at_limit = recorder.end(&mem, CMD).expect("end at the limit");
        assert_eq!(recorder.list(at_limit).expect("live").depth, MAX_NESTING);

        // The next level would recurse deeper than the replay will.
        let base = BASE + u64::from(MAX_NESTING + 1) * 0x2000;
        recorder.begin(&mem, CMD, base, 0x1000).expect("caller");
        assert_eq!(recorder.call(&mem, CMD, &[at_limit]), Err(Error::Nesting));
        assert_eq!(recorder.rejected(), 1);
        assert_eq!(recorder.copy(&mem, CMD, &[at_limit]), Err(Error::Nesting));
        assert_eq!(recorder.rejected(), 2);
        assert_eq!(recorder.list(at_limit).unwrap().depth, MAX_NESTING);
    }

    #[test]
    fn state_and_arrays_are_copied_by_value_into_the_blob() {
        let mem = memory();
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, CAPACITY).expect("begin");
        let object = BASE + 0x3000;

        // A state object the game reuses right after the call; the record keeps the old values.
        mem.poke(object, &[0xaa; 0x40]);
        let values = [
            StateValue {
                kind: kind_code(Kind::BlendEnable),
                index: 0,
                value: 1,
            },
            StateValue {
                kind: kind_code(Kind::BlendSrc),
                index: 0,
                value: 4,
            },
        ];
        recorder
            .emit_state(&mem, CMD, Op::BindBlendState, object, &values)
            .expect("state record");
        mem.poke(object, &[0xbb; 0x40]);

        // A raw fallback for an object whose struct the host never decoded.
        let raw_object = BASE + 0x3800;
        mem.poke(raw_object, &[0xcd; 0x100]);
        recorder
            .emit_raw_state(&mem, CMD, Op::BindDepthStencilState, raw_object)
            .expect("raw record");
        mem.poke(raw_object, &[0xee; 0x100]);

        let handle = recorder.end(&mem, CMD).expect("end");
        let all = records(&mem, handle, 4);
        let state = all[1];
        assert_eq!(state.op, Op::BindBlendState);
        assert_eq!(state.words[0], object);
        assert_eq!(state.count, 2);
        assert!(state.flags & FLAG_BLOB != 0);
        assert!(
            state.blob >= RECORD_SIZE as u32,
            "blobs grow below the records"
        );
        let bytes = blob(&mem, handle, &state);
        assert_eq!(bytes.len(), 2 * StateValue::BYTES as usize);
        assert_eq!(StateValue::decode(&bytes[0..16]), Some(values[0]));
        assert_eq!(StateValue::decode(&bytes[16..32]), Some(values[1]));

        let raw = all[2];
        assert!(raw.flags & FLAG_RAW != 0 && raw.flags & FLAG_UNRESOLVED == 0);
        assert_eq!(raw.count, RAW_STATE_BYTES as u32);
        assert_eq!(
            blob(&mem, handle, &raw),
            vec![0xcd; RAW_STATE_BYTES as usize]
        );

        // Unreadable objects still leave an address and are marked unresolved.
        recorder.begin(&mem, CMD, BASE, CAPACITY).expect("begin");
        recorder
            .emit_raw_state(&mem, CMD, Op::BindPolygonState, 0x9000_0000)
            .expect("unresolved record");
        let handle = recorder.end(&mem, CMD).expect("end");
        let record = records(&mem, handle, 2)[1];
        assert!(record.flags & FLAG_UNRESOLVED != 0);
        assert_eq!(record.words[0], 0x9000_0000);
    }

    #[test]
    fn a_full_list_refuses_the_record_instead_of_overwriting_its_blobs() {
        let mem = memory();
        let mut recorder = Recorder::default();
        recorder
            .begin(&mem, CMD, BASE, 3 * RECORD_SIZE)
            .expect("begin");
        let mut record = Record::new(Op::Barrier);
        record.set_word(0, 0xff);
        recorder.emit(&mem, CMD, record).expect("one command fits");
        // Begin + one command leave exactly the end record's slot; a second command does not fit.
        assert_eq!(recorder.emit(&mem, CMD, record), Err(Error::Full));
        let handle = recorder.end(&mem, CMD).expect("end always fits");
        let all = records(&mem, handle, 3);
        assert_eq!(
            (all[0].op, all[1].op, all[2].op),
            (Op::Begin, Op::Barrier, Op::End)
        );
        assert_eq!(all[1].words[0], 0xff);
        assert_eq!(all[2].words[0], 2);
    }

    #[test]
    fn a_command_without_a_list_is_rejected() {
        let mem = memory();
        let mut recorder = Recorder::default();
        assert_eq!(
            recorder.emit(&mem, CMD, Record::new(Op::Barrier)),
            Err(Error::NotRecording)
        );
        assert_eq!(recorder.end(&mem, CMD), Err(Error::NotRecording));
        assert!(!recorder.is_recording(CMD));
    }

    #[test]
    fn the_write_pointer_moves_by_the_bytes_written() {
        let mem = FakeMemory::new(0x8000, 0x100);
        let cmdbuf = 0x8000;
        mem.poke(cmdbuf + header::BASE, &0x1000u64.to_le_bytes());
        mem.poke(cmdbuf + header::GPU_BASE, &0x1_0000_0000u64.to_le_bytes());
        mem.poke(cmdbuf + header::WRITE, &0x1200u64.to_le_bytes());
        mem.poke(cmdbuf + header::WRITE_SHADOW, &0x4000u64.to_le_bytes());
        assert!(advance(&mem, cmdbuf, header::WRITE, RECORD_SIZE));
        let word = |offset: u64| {
            let bytes = mem.read_back(cmdbuf + offset, 8);
            u64::from_le_bytes(bytes.as_slice().try_into().unwrap())
        };
        assert_eq!(word(header::WRITE), 0x1200 + RECORD_SIZE);
        assert_eq!(
            word(header::WRITE_SHADOW),
            0x4000,
            "only the field the list resolved from moves"
        );
        assert!(advance(&mem, cmdbuf, header::WRITE_SHADOW, RECORD_SIZE));
        assert_eq!(word(header::WRITE_SHADOW), 0x4000 + RECORD_SIZE);
        assert_eq!(
            position(&mem, cmdbuf),
            Some(0x1_0000_0000 + 0x4000 + RECORD_SIZE - 0x1000),
            "shadow's formula uses its own field and the base"
        );
        // The counter ends before it can overflow; nothing wraps.
        mem.poke(cmdbuf + header::WRITE, &u64::MAX.to_le_bytes());
        assert!(!advance(&mem, cmdbuf, header::WRITE, 1));
        assert_eq!(position(&mem, 0x9000), None);
    }

    #[test]
    fn the_write_pointer_is_mapped_by_offset_not_by_the_drivers_gpu_address() {
        const GUEST: u64 = 0x1_0000_0000;
        const DRIVER_GPU: u64 = 0x8000_0000;
        let mem = FakeMemory::new(GUEST, 0x4000);
        let cmdbuf = GUEST + 0x100;
        let span = Span {
            guest: GUEST + 0x2000,
            size: 0x2000,
        };
        let spans = [span];
        let header = |write: u64, write_shadow: u64, base: u64, gpu_base: u64| {
            mem.poke(cmdbuf + header::WRITE, &write.to_le_bytes());
            mem.poke(cmdbuf + header::WRITE_SHADOW, &write_shadow.to_le_bytes());
            mem.poke(cmdbuf + header::BASE, &base.to_le_bytes());
            mem.poke(cmdbuf + header::GPU_BASE, &gpu_base.to_le_bytes());
        };
        let expected = Placement {
            guest: span.guest + 0x800,
            capacity: span.size - 0x800,
            field: header::WRITE,
        };

        // The driver advances +0x00 and leaves the span end at +0x18, which is the field shadow reads.
        header(span.guest + 0x800, span.end(), span.guest, DRIVER_GPU);
        assert_eq!(placement(&mem, cmdbuf, &spans), Ok(expected));
        assert_eq!(
            position(&mem, cmdbuf),
            Some(DRIVER_GPU + span.size),
            "shadow's view of this driver is the span end, not the write pointer"
        );

        // The driver keeps its own GPU pointers and no guest base.
        header(DRIVER_GPU + 0x800, 0, DRIVER_GPU, DRIVER_GPU);
        assert_eq!(placement(&mem, cmdbuf, &spans), Ok(expected));

        // Guest base, GPU write pointer: the second candidate is the offset from the GPU base.
        header(DRIVER_GPU + 0x800, 0, span.guest, DRIVER_GPU);
        assert_eq!(placement(&mem, cmdbuf, &spans), Ok(expected));

        // No base at all: the write pointer is an absolute guest address.
        header(span.guest + 0x800, 0, 0, 0);
        assert_eq!(placement(&mem, cmdbuf, &spans), Ok(expected));

        // A driver that keeps the write pointer at +0x18 instead: the fallback field resolves, and the recorder
        // advances that word.
        header(0, span.guest + 0x400, span.guest, DRIVER_GPU);
        assert_eq!(
            placement(&mem, cmdbuf, &spans),
            Ok(Placement {
                guest: span.guest + 0x400,
                capacity: span.size - 0x400,
                field: header::WRITE_SHADOW,
            })
        );

        // Two spans: the offset walks past the first one.
        let two = [
            Span {
                guest: GUEST + 0x2000,
                size: 0x1000,
            },
            Span {
                guest: GUEST + 0x3000,
                size: 0x1000,
            },
        ];
        header(GUEST + 0x2000 + 0x1080, 0, GUEST + 0x2000, 0);
        assert_eq!(
            placement(&mem, cmdbuf, &two),
            Ok(Placement {
                guest: GUEST + 0x3080,
                capacity: 0x1000 - 0x80,
                field: header::WRITE,
            })
        );

        // A pointer outside every span is a precise failure, not a silent wrap.
        header(GUEST + 0x1000, GUEST + 0x1000, GUEST + 0x2000, 0);
        assert_eq!(
            placement(&mem, cmdbuf, &spans),
            Err(Error::WritePointerOutsideMemory)
        );
        header(span.guest, span.end(), GUEST + 0x2000, 0);
        assert_eq!(
            placement(&mem, 0x9000, &spans),
            Err(Error::Unmapped),
            "the header itself is unreadable"
        );
        assert_eq!(placement(&mem, cmdbuf, &[]), Err(Error::NoCommandMemory));
    }

    #[test]
    fn every_state_field_has_a_stable_wire_tag() {
        use Kind::*;
        let kinds = [
            ChannelMask,
            BlendEnable,
            BlendEquation,
            BlendSrc,
            BlendDst,
            BlendEquationAlpha,
            BlendSrcAlpha,
            BlendDstAlpha,
            DepthTest,
            DepthWrite,
            DepthFunc,
            StencilTest,
            StencilFunc,
            StencilFail,
            StencilDepthFail,
            StencilPass,
            CullFace,
            FrontFace,
            PolygonOffsetEnables,
            AttribFormat,
            AttribOffset,
            AttribStream,
            StreamStride,
            StreamDivisor,
            BlendColor,
            Viewport,
            Scissor,
            TargetFormat,
            BlendRegister,
            AttribCount,
            StreamCount,
            DepthFormat,
            PolygonOffset,
            StencilRef,
            StencilValueMask,
            StencilMask,
            DepthClamp,
            RasterizerDiscard,
            PointSize,
            LineWidth,
            DepthBoundsEnable,
            DepthBoundsMin,
            DepthBoundsMax,
        ];
        let mut codes = std::collections::BTreeSet::new();
        for kind in kinds {
            let code = kind_code(kind);
            assert!(codes.insert(code), "{kind:?} shares code {code}");
            assert_eq!(kind_from_code(code), Some(kind));
        }
        assert_eq!(kind_from_code(0), None);
        assert_eq!(kind_from_code(999), None);
    }

    #[test]
    fn records_round_trip_every_field() {
        let mut record = Record::new(Op::DrawElements);
        record.flags = FLAG_BLOB | FLAG_NESTED;
        record.count = 7;
        record.blob = 0x180;
        record.blob_len = 0x20;
        record.set_word(8, u64::MAX);
        record.set_float(3, -1.5);
        record.set_int(3, -7);
        let bytes = record.encode();
        assert_eq!(Record::decode(&bytes), Some(record));
        assert_eq!(
            u32::from_le_bytes(
                bytes[OFF_BLOB as usize..OFF_BLOB as usize + 4]
                    .try_into()
                    .unwrap()
            ),
            0x180
        );
        assert_eq!(Record::decode(&bytes[..0x40]), None);

        let mut truncated = bytes;
        truncated[0..4].copy_from_slice(&0xffff_u32.to_le_bytes());
        assert_eq!(Record::decode(&truncated), None);
        let mut wrong_size = bytes;
        wrong_size[4..6].copy_from_slice(&0x40u16.to_le_bytes());
        assert_eq!(Record::decode(&wrong_size), None);
    }

    #[test]
    fn initialize_aborts_only_the_open_list() {
        let mem = memory();
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, CAPACITY).expect("first");
        let ended = recorder.end(&mem, CMD).expect("end first");
        recorder
            .begin(&mem, OTHER, BASE + 0x1000, 0x800)
            .expect("second buffer");
        recorder.initialize(CMD);
        assert!(!recorder.is_recording(CMD));
        assert!(
            recorder.is_recording(OTHER),
            "another command buffer keeps its open list"
        );
        assert!(
            recorder.list(ended).is_some(),
            "an ended list stays valid until its bytes are reused"
        );
        assert_eq!(recorder.live(), 2, "the ended list and the other open one");
    }

    #[test]
    fn an_implicit_list_is_recorded_in_the_host_arena() {
        let mem = memory();
        let mut recorder = Recorder::default();
        let storage = recorder.begin_arena(&mem, CMD).expect("begin");
        assert!(storage >= ARENA_BASE);
        assert!(recorder.is_arena(CMD), "the open list lives in the arena");
        assert!(recorder.list(storage).is_some_and(|list| list.arena));

        let mut state = Record::new(Op::BindBlendState);
        state.set_word(0, 0x7000);
        let blob = [0xa5u8; 24];
        recorder
            .emit_blob(&mem, CMD, state, &blob)
            .expect("state with a blob");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 12);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");

        let key = recorder.end_arena(&mem, CMD).expect("end");
        assert_eq!(key, storage);
        let list = recorder.list(key).cloned().expect("live");
        assert_eq!(
            (list.records, list.draws, list.bytes),
            (4, 1, 4 * RECORD_SIZE)
        );
        assert_eq!(recorder.usage().implicit, 1);

        // The driver is the only writer of its command memory: the host never wrote the buffer.
        assert!(
            mem.read_back(BASE, CAPACITY).iter().all(|byte| *byte == 0),
            "an implicit list must not touch guest command memory"
        );

        // The submit reads the arena copy, with the blob at its offset from the list.
        let captured = recorder
            .capture_bytes(&mem, key, list.bytes)
            .expect("capture");
        let ops: Vec<Op> = captured
            .as_chunks::<RECORD_BYTES>()
            .0
            .iter()
            .filter_map(|chunk| Record::decode(chunk).map(|record| record.op))
            .collect();
        assert_eq!(
            ops,
            [Op::Begin, Op::BindBlendState, Op::DrawArrays, Op::End]
        );
        let state =
            Record::decode(&captured[RECORD_BYTES..2 * RECORD_BYTES]).expect("state record");
        assert_eq!(recorder.blob(key, state.blob), Some(blob.as_slice()));
    }

    #[test]
    fn an_arena_list_is_filed_under_the_handle_the_driver_reported() {
        let mem = memory();
        let mut recorder = Recorder::default();
        let storage = recorder.begin_arena(&mem, CMD).expect("begin");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let key = recorder.end_arena(&mem, CMD).expect("end");

        // `EndRecording` first reports an internal pointer, then the GPU handle the game can name.
        assert!(!is_handle(1 << 40), "a tagged pointer is not a game handle");
        assert!(is_handle(0x2117_fc208));
        assert!(recorder.refile(key, 0x2117_fc208), "the handle resolves");
        assert!(recorder.list(key).is_none(), "the storage key is gone");
        let list = recorder.list(0x2117_fc208).expect("filed");
        assert_eq!(
            (list.key, list.handle, list.arena, list.draws),
            (0x2117_fc208, storage, true, 1)
        );

        // A list that takes the arena slot back retires the one filed there, like re-recording over memory.
        recorder.arena_next = ARENA_SLOTS;
        let reused = recorder.begin_arena(&mem, CMD).expect("reuse");
        assert_eq!(reused, storage, "the slot comes around again");
        recorder.end_arena(&mem, CMD).expect("end");
        assert!(
            recorder.list(0x2117_fc208).is_none(),
            "the newer list wins over the older handle"
        );
    }

    #[test]
    fn chained_chunks_replay_before_their_list_and_go_when_it_does() {
        let mem = memory();
        let mut recorder = Recorder::default();
        // Two chunks the driver ended under its internal pointer, in recording order.
        recorder.begin_arena(&mem, CMD).expect("first chunk");
        let mut program = Record::new(Op::BindProgram);
        program.set_word(0, 0x9000);
        recorder.emit(&mem, CMD, program).expect("program");
        let first = recorder.end_arena(&mem, CMD).expect("end first");
        recorder.begin_arena(&mem, CMD).expect("second chunk");
        let mut texture = Record::new(Op::BindTexture);
        texture.set_word(0, 0x0020_2007);
        recorder.emit(&mem, CMD, texture).expect("texture");
        let second = recorder.end_arena(&mem, CMD).expect("end second");

        // The list the next usable handle names continues both.
        recorder.begin_arena(&mem, CMD).expect("successor");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let successor = recorder.end_arena(&mem, CMD).expect("end successor");

        recorder.chain(successor, &[first, second]).expect("chain");
        let list = recorder.list(successor).expect("successor live");
        assert_eq!(list.chained, [first, second], "recording order is kept");
        assert_eq!(list.depth, 1, "a chain is one nesting level");
        assert_eq!(list.draws, 1);

        assert!(recorder.refile(successor, 0x2117_fc208));
        assert_eq!(recorder.report().chained, 2);
        // The chunks are reachable only through the list that continues them.
        assert_eq!(recorder.retire_replayed(&[0x2117_fc208]), 3);
        assert_eq!(recorder.live(), 0);
        assert_eq!(recorder.report().retired, 3);
        assert_eq!(recorder.retire_replayed(&[0x2117_fc208]), 0, "idempotent");
    }

    #[test]
    fn a_chain_refuses_a_missing_chunk_and_an_unknown_list() {
        let mem = memory();
        let mut recorder = Recorder::default();
        recorder.begin_arena(&mem, CMD).expect("chunk");
        recorder
            .emit(&mem, CMD, Record::new(Op::Barrier))
            .expect("bind");
        let chunk = recorder.end_arena(&mem, CMD).expect("end");
        recorder.begin_arena(&mem, CMD).expect("successor");
        let successor = recorder.end_arena(&mem, CMD).expect("end successor");

        assert_eq!(
            recorder.chain(successor, &[0xdead_0000]),
            Err(Error::UnknownList)
        );
        assert!(recorder.list(successor).expect("live").chained.is_empty());
        assert_eq!(recorder.chain(successor, &[successor]), Err(Error::Nesting));
        assert_eq!(recorder.chain(successor, &[chunk]), Ok(()));
        assert_eq!(recorder.list(successor).expect("live").chained, [chunk]);
    }

    #[test]
    fn retiring_a_replayed_submit_drops_only_arena_lists() {
        let mem = memory();
        let mut recorder = Recorder::default();
        recorder
            .begin(&mem, CMD, BASE, CAPACITY)
            .expect("guest list");
        let guest = recorder.end(&mem, CMD).expect("end guest");
        recorder.begin_arena(&mem, CMD).expect("arena list");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let arena = recorder.end_arena(&mem, CMD).expect("end arena");

        assert_eq!(
            recorder.retire_replayed(&[guest]),
            0,
            "the game's command memory is not ours to retire"
        );
        assert!(recorder.list(guest).is_some());
        assert_eq!(recorder.retire_replayed(&[arena]), 1);
        assert!(recorder.list(arena).is_none());
        assert_eq!(recorder.report().retired, 1);
    }
}
