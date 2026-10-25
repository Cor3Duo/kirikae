//! Replay of recorded NVN command lists into Vulkan.
//!
//! Recording (`record.rs`) leaves one fixed record per command in the game's command memory and hands back the
//! address as the list handle. This module walks those handles, decodes the records into a live [`State`], and feeds
//! them to a [`Sink`] in submission order. The Vulkan sink (`gpu/vulkan/nvn.rs`) turns that state into render passes,
//! clears and draws; the tests here use a recording sink, so the walker, call recursion, inline copies and the clear
//! selection are exercised without a device.
//!
//! Shape (`docs/nvn/CONTRACT.md` §4):
//!
//! - **One worker, FIFO.** A single replay thread takes one submit at a time, in the order the game submitted them.
//!   Submits are at most one behind the game, so a present never sees a frame two submits old.
//! - **Snapshot at submit.** A list's records are copied when `nvnQueueSubmitCommands` runs, together with every list
//!   it can reach through `CallCommands`. Re-recording the same bytes afterwards cannot change a submit already
//!   queued (`nvn_replay.cpp:14-36`), and a handle that a list names resolves through that snapshot only (`MAX_DEPTH`).
//! - **Inline copies.** `CopyCommands` inlines the callee's records at record time (`record.rs`), so the walker
//!   processes them where they sit; it never re-reads the callee.
//! - **Chained chunks.** A list can continue a chunk the driver closed under an internal pointer: the walker replays
//!   that chunk first, in recording order, so the state it set is the state of the records that follow
//!   (`record::List::chained`). This is what a pass split across an `EndRecording` needs.
//! - **State across submits.** NVN state belongs to the command buffer, not to one submit: a frame sets its targets,
//!   program and binds at the end of one list and the draw is the first command of the next submit on the same
//!   buffer. The worker keeps one [`State`] per command buffer between submits and `nvnCommandBufferInitialize`
//!   resets it (`reset_state`).
//! - **Nothing is skipped silently.** A record the sink cannot replay yet is counted once with its reason and the
//!   walk continues, so the log says what is missing instead of a draw disappearing.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use super::record::{self, Op, RECORD_SIZE, Record, StateValue};
use super::state::Kind;
use crate::process::Process;

/// Recursion limit for `CallCommands`, the depth the recorder also stops at.
pub const MAX_DEPTH: usize = record::MAX_NESTING as usize;

/// How many submits may be queued ahead of the replay thread. One keeps the frame a present shows in order; the
/// game's own fences would allow more, and milestone 4.7 can raise it.
pub const MAX_PENDING: u64 = 1;

/// A colour or depth attachment as the record names it: the texture object and its view, both guest addresses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ViewRef {
    pub texture: u64,
    pub view: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Targets {
    pub colors: Vec<Option<ViewRef>>,
    pub depth: Option<ViewRef>,
}

/// What a `SetRenderTargets` view selected. 0 means "all of the texture's".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewInfo {
    pub base_level: u32,
    pub levels: u32,
    pub min_layer: u32,
    pub layers: u32,
    pub format: u32,
}

/// A texture object resolved at submit time, so the replay never has to lock the host registry mid-frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureInfo {
    pub host: u64,
    pub gpu: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
    pub layers: u32,
    pub samples: u32,
    pub stride: u64,
    pub block_height_log2: u32,
    pub target: u32,
    pub swizzle: [u8; 4],
}

/// One stage of a bound program: the address of its SPH (the code block plus the control's `program_start`), whether
/// it is the fragment stage, and where the block keeps `c[1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    /// GPU address the Maxwell translation starts at: the 0x50-byte SPH.
    pub code: u64,
    pub fragment: bool,
    /// The block's own constant data (`c[1]`) as an absolute GPU address and size.
    pub constants: Option<(u64, u32)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    pub stages: Vec<Stage>,
}

/// One list's records and the blobs they name, copied when the submit was queued. Blobs sit at the far end of the
/// list's command memory, outside the record range, so they are carried next to it (`record.rs` layout).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ListSnapshot {
    pub handle: u64,
    /// The guest command buffer the list was recorded on. NVN keeps one state per command buffer, so two roots
    /// from different buffers must not share the render targets or binds of the buffer replayed before them.
    pub command_buffer: u64,
    pub bytes: Vec<u8>,
    /// `(offset from the handle, bytes)` for each blob a record names.
    pub blobs: Vec<(u32, Vec<u8>)>,
    /// Lists replayed before this one, in recording order. An implicit chunk the driver closed under an internal
    /// pointer is one of these: the records that follow continue its state (`record.rs`, "Implicit lists").
    pub chained: Vec<u64>,
}

impl ListSnapshot {
    pub fn records(&self) -> usize {
        self.bytes.len() / RECORD_SIZE as usize
    }

    /// The blob a record names, from this snapshot.
    pub fn blob(&self, record: &Record) -> Option<&[u8]> {
        if record.flags & record::FLAG_BLOB == 0 || record.blob_len == 0 {
            return None;
        }
        self.blobs
            .iter()
            .find(|(at, _)| *at == record.blob)
            .map(|(_, data)| data.as_slice())
    }

    /// Copies a live list: its records from wherever the recorder put them (the game's command memory, or the host
    /// arena for a list opened implicitly) and every blob from the recorder's copy (the driver reuses the command
    /// memory those blobs sit in while a submit waits to be replayed).
    pub fn capture(
        mem: &dyn super::record::Memory,
        recorder: &super::record::Recorder,
        handle: u64,
        bytes: u64,
        command_buffer: u64,
    ) -> Option<ListSnapshot> {
        let records = recorder.capture_bytes(mem, handle, bytes)?;
        let chained = recorder
            .list(handle)
            .map(|list| list.chained.clone())
            .unwrap_or_default();
        let mut blobs = Vec::new();
        for chunk in records.as_chunks::<{ RECORD_SIZE as usize }>().0 {
            let Some(record) = Record::decode(chunk) else {
                continue;
            };
            if record.flags & record::FLAG_BLOB == 0 || record.blob_len == 0 {
                continue;
            }
            if let Some(data) = recorder.blob(handle, record.blob)
                && data.len() >= record.blob_len as usize
            {
                blobs.push((record.blob, data[..record.blob_len as usize].to_vec()));
            }
        }
        Some(ListSnapshot {
            handle,
            command_buffer,
            bytes: records,
            blobs,
            chained,
        })
    }
}

/// Everything one `nvnQueueSubmitCommands` queued. Built under the recording lock, replayed on the worker.
#[derive(Clone, Debug, Default)]
pub struct Submit {
    pub roots: Vec<Arc<ListSnapshot>>,
    /// Every list a root can reach, by handle, so a `CallCommands` never reads the callee's current bytes.
    pub callees: HashMap<u64, Arc<ListSnapshot>>,
    pub textures: HashMap<u64, TextureInfo>,
    pub views: HashMap<u64, ViewInfo>,
    pub programs: HashMap<u64, Program>,
    /// Guest address of the texture and sampler descriptor tables the submit's pools name.
    pub texture_pool: u64,
    pub sampler_pool: u64,
    /// Window textures the host knows about, by guest `NVNtexture*` address. A `SetRenderTargets` that names one of
    /// these is the pass a title composes its frame into (`docs/nvn/CONTRACT.md` §8); the walker reports it so the
    /// sink can remember the image it must compose from when the title never issues the draw itself.
    pub windows: HashSet<u64>,
    /// The recorder's generation at the end of recording (`docs/nvn/CONTRACT.md` §7).
    pub mark: u64,
}

/// Vertex attribute state as `BindVertexAttribState` decoded it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Attrib {
    pub format: u32,
    pub offset: u32,
    pub stream: u32,
}

/// Vertex stream state as `BindVertexStreamState` decoded it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stream {
    pub stride: u32,
    pub divisor: u32,
}

/// The render state a draw uses, with NVN's own defaults (`state.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Render {
    pub channel_mask: [u8; 8],
    pub blend_enable: [bool; 8],
    pub blend_target: u8,
    pub blend_src: u32,
    pub blend_dst: u32,
    pub blend_src_alpha: u32,
    pub blend_dst_alpha: u32,
    pub blend_equation: u32,
    pub blend_equation_alpha: u32,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_func: u32,
    pub stencil_test: bool,
    /// Per face `[func, fail, depth_fail, pass]`.
    pub stencil: [[u32; 4]; 2],
    pub cull: u32,
    pub front: u32,
}

impl Default for Render {
    fn default() -> Render {
        Render {
            channel_mask: [0xf; 8],
            blend_enable: [false; 8],
            blend_target: 0,
            blend_src: 2,
            blend_dst: 1,
            blend_src_alpha: 2,
            blend_dst_alpha: 1,
            blend_equation: 1,
            blend_equation_alpha: 1,
            depth_test: false,
            depth_write: true,
            depth_func: 2,
            stencil_test: false,
            stencil: [[8, 1, 1, 1]; 2],
            cull: 0,
            front: 1,
        }
    }
}

/// The walker's live state. A sink sees it at each draw, exactly as the records left it.
#[derive(Clone, Default)]
pub struct State {
    pub targets: Targets,
    pub program: u64,
    /// `c[3 + n]` per stage (0 vertex, 1 fragment): the GPU address and size `BindUniformBuffer` named.
    pub cbufs: [[Option<(u64, u64)>; 16]; 2],
    /// Storage buffers per stage (0 vertex, 1 fragment): the GPU address and size `BindStorageBuffer` named.
    pub storage: [[Option<(u64, u64)>; 16]; 2],
    /// Texture handle per sampler binding, per stage.
    pub textures: [[u32; 32]; 2],
    pub attribs: [Option<Attrib>; 32],
    pub streams: [Option<Stream>; 16],
    /// Vertex stream address and size `BindVertexBuffer` named.
    pub buffers: [Option<(u64, u64)>; 16],
    /// `SetViewport`: x, y, width, height, near, far.
    pub viewport: Option<[f32; 6]>,
    pub render: Render,
}

/// One draw command, as recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexCall {
    pub address: u64,
    /// 0 u8, 1 u16, 2 u32.
    pub kind: u32,
    pub base_vertex: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawCall {
    pub topology: u32,
    pub first: u32,
    pub count: u32,
    pub instances: u32,
    pub base_instance: u32,
    pub index: Option<IndexCall>,
}

/// What the walker does with a decoded command. The Vulkan renderer implements it; tests record it.
pub trait Sink {
    fn targets(&mut self, targets: &Targets) -> Result<(), String>;
    fn clear_color(&mut self, index: u32, color: [f32; 4], mask: u32) -> Result<(), String>;
    fn clear_depth_stencil(
        &mut self,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) -> Result<(), String>;
    fn discard_color(&mut self, index: u32) -> Result<(), String> {
        let _ = index;
        Ok(())
    }
    fn draw(&mut self, state: &State, draw: &DrawCall) -> Result<(), String>;
    /// A `SetRenderTargets` named a window texture, in the shape a title that lets the driver compose leaves it:
    /// `source` is the last colour target the submit drew into, before the window target was set. A title that draws
    /// into the window itself needs no composition; the sink decides from the records, not from a guess.
    fn window_target(&mut self, window: u64, source: Option<u64>) {
        let _ = (window, source);
    }
    /// A texture binding outside a draw. A title that composes its window through a full-screen draw records the
    /// texture that draw would sample right after setting the window target; the sink uses it as the composition
    /// source when the draw is missing.
    fn texture_bound(&mut self, stage: usize, index: usize, handle: u32) {
        let _ = (stage, index, handle);
    }
    fn dispatch(&mut self, grid: [u32; 3]) -> Result<(), String> {
        let _ = grid;
        Ok(())
    }
    fn barrier(&mut self, flags: u64) {
        let _ = flags;
    }
    fn begin(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Per-submit counters, added to the global report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub submits: u64,
    pub lists: u64,
    pub records: u64,
    pub calls: u64,
    pub copies: u64,
    /// Chunks replayed as the prefix of a list that continues them.
    pub chained: u64,
    pub draws: u64,
    pub draw_arrays: u64,
    pub draw_elements: u64,
    pub clears: u64,
    pub dispatches: u64,
    pub skipped: u64,
    pub skipped_depth: u64,
    pub reasons: BTreeMap<String, u64>,
}

impl Stats {
    fn skip(&mut self, reason: &str) {
        self.skipped += 1;
        *self.reasons.entry(reason.to_string()).or_default() += 1;
    }
}

/// The walker. Constructed per submit from the queued snapshots.
pub struct Walker {
    pub state: State,
    /// The state of every command buffer replayed so far, and the one `state` currently belongs to.
    states: HashMap<u64, State>,
    command_buffer: Option<u64>,
    snapshots: HashMap<u64, Arc<ListSnapshot>>,
    pub stats: Stats,
    /// Reasons already logged, so a handle called every frame is one line.
    logged: HashSet<String>,
    /// Lists currently being walked, so a malformed chain or call cannot recurse forever.
    walking: HashSet<u64>,
    /// The window textures this submit knows, copied so the record walk does not borrow the submit.
    submit_windows: HashSet<u64>,
    /// The last colour target a draw rendered into in this submit. When a `SetRenderTargets` names a window texture,
    /// that image is what the title's composition would have sampled if it had issued the final draw.
    last_drawn: Option<u64>,
    pub log: Vec<String>,
}

impl Walker {
    pub fn new(submit: &Submit) -> Walker {
        Walker {
            state: State::default(),
            states: HashMap::new(),
            command_buffer: None,
            snapshots: submit.callees.clone(),
            stats: Stats::default(),
            logged: HashSet::new(),
            walking: HashSet::new(),
            submit_windows: submit.windows.clone(),
            last_drawn: None,
            log: Vec::new(),
        }
    }

    /// Walks the submit's roots in order.
    pub fn run(&mut self, submit: &Submit, sink: &mut dyn Sink) {
        if let Err(why) = sink.begin() {
            self.skip(&why);
            return;
        }
        for list in &submit.roots {
            self.snapshots.insert(list.handle, Arc::clone(list));
        }
        let roots: Vec<Arc<ListSnapshot>> = submit.roots.clone();
        for list in roots {
            // A root starts on its own command buffer's state. A callee does not: `CallCommands` appends its
            // commands to the caller's stream and the caller's state is the one the GPU runs them with.
            self.switch(list.command_buffer);
            if let Err(why) = self.walk(&list, 0, sink) {
                self.skip(&why);
            }
        }
        if let Some(command_buffer) = self.command_buffer {
            self.states.insert(command_buffer, self.state.clone());
        }
        if let Err(why) = sink.finish() {
            self.skip(&why);
        }
    }

    /// Makes `command_buffer`'s state current, saving the one it replaces. A buffer seen for the first time starts
    /// at NVN's defaults.
    fn switch(&mut self, command_buffer: u64) {
        if self.command_buffer == Some(command_buffer) {
            return;
        }
        if let Some(active) = self.command_buffer {
            self.states.insert(active, self.state.clone());
        }
        self.state = self
            .states
            .get(&command_buffer)
            .cloned()
            .unwrap_or_default();
        self.command_buffer = Some(command_buffer);
    }

    /// The guest texture a colour attachment index names, if that target exists.
    fn colour_target(&self, index: usize) -> Option<u64> {
        self.state
            .targets
            .colors
            .get(index)
            .copied()
            .flatten()
            .map(|color| color.texture)
    }

    fn walk(
        &mut self,
        list: &ListSnapshot,
        depth: usize,
        sink: &mut dyn Sink,
    ) -> Result<(), String> {
        if !self.walking.insert(list.handle) {
            return Err(format!("list {:#x} is replayed inside itself", list.handle));
        }
        self.stats.lists += 1;
        // The chunks this list continues were recorded earlier on the same command buffer and their state carries
        // into these records. Replay them first, in recording order, as part of the same stream.
        for handle in &list.chained {
            if depth >= MAX_DEPTH {
                self.skip(format!(
                    "chained chunk {handle:#x} nested deeper than {MAX_DEPTH}"
                ));
                continue;
            }
            let Some(chained) = self.snapshots.get(handle).cloned() else {
                self.skip(format!("chained chunk {handle:#x} has no snapshot"));
                continue;
            };
            self.stats.chained += 1;
            if let Err(why) = self.walk(&chained, depth + 1, sink) {
                self.skip(format!("chained chunk {handle:#x}: {why}"));
            }
        }
        let result = self.run_records(list, 0, list.records(), depth, sink);
        self.walking.remove(&list.handle);
        result
    }

    /// Runs the records in `[start, end)`. An inline copy is a nested range of records in the same list; a call is a
    /// walk of the callee's snapshot.
    fn run_records(
        &mut self,
        list: &ListSnapshot,
        start: usize,
        end: usize,
        depth: usize,
        sink: &mut dyn Sink,
    ) -> Result<(), String> {
        let mut at = start;
        while at < end {
            let record = match Record::decode(&list.bytes[at * RECORD_SIZE as usize..]) {
                Some(record) => record,
                None => return Err("record does not decode".into()),
            };
            at += 1;
            self.stats.records += 1;
            match record.op {
                Op::End => return Ok(()),
                Op::Begin => {}
                Op::SetRenderTargets => {
                    let targets = self.targets(list, &record)?;
                    self.state.targets = targets.clone();
                    if let Err(why) = sink.targets(&targets) {
                        self.skip(format!("targets: {why}"));
                    }
                    // A window target is the composition the title leaves to the driver: report the image it would
                    // have sampled so the sink can blit it when the title never draws into the window itself.
                    for color in targets.colors.iter().flatten() {
                        if self.submit_windows.contains(&color.texture) {
                            sink.window_target(color.texture, self.last_drawn);
                        }
                    }
                }
                Op::ClearColor => {
                    let index = record.ints[0].max(0) as u32;
                    let mask = record.ints[1] as u32;
                    self.stats.clears += 1;
                    if let Err(why) = sink.clear_color(index, record.floats, mask) {
                        self.skip(format!("clear colour: {why}"));
                    }
                }
                Op::ClearDepthStencil => {
                    let depth = (record.ints[1] != 0).then_some(record.floats[0]);
                    let stencil = (record.ints[2] != 0).then_some(record.ints[0]);
                    self.stats.clears += 1;
                    if let Err(why) = sink.clear_depth_stencil(depth, stencil) {
                        self.skip(format!("clear depth/stencil: {why}"));
                    }
                }
                Op::DiscardColor => {
                    let index = record.ints[0].max(0) as u32;
                    let _ = sink.discard_color(index);
                }
                Op::BindProgram => self.state.program = record.words[0],
                Op::BindBlendState
                | Op::BindColorState
                | Op::BindChannelMaskState
                | Op::BindDepthStencilState
                | Op::BindPolygonState => {
                    self.bind_render(list, &record)?;
                }
                Op::BindVertexAttribState => self.bind_attribs(list, &record)?,
                Op::BindVertexStreamState => self.bind_streams(list, &record)?,
                Op::BindTexture => {
                    let stage = record.ints[0].clamp(0, 1) as usize;
                    let index = record.ints[1].clamp(0, 31) as usize;
                    let handle = record.words[0] as u32;
                    self.state.textures[stage][index] = handle;
                    sink.texture_bound(stage, index, handle);
                }
                Op::BindUniformBuffer => {
                    let stage = record.ints[0].clamp(0, 1) as usize;
                    let index = record.ints[1];
                    if (0..16).contains(&index) {
                        self.state.cbufs[stage][index as usize] =
                            Some((record.words[0], record.words[1]));
                    } else {
                        self.skip(format!("uniform block {index} outside c[3..19]"));
                    }
                }
                Op::BindStorageBuffer => {
                    let stage = record.ints[0].clamp(0, 1) as usize;
                    let index = record.ints[1];
                    if (0..16).contains(&index) {
                        self.state.storage[stage][index as usize] =
                            Some((record.words[0], record.words[1]));
                    } else {
                        self.skip(format!("storage buffer {index} outside 0..16"));
                    }
                }
                Op::SetSamplerPool => {}
                Op::SetTexturePool => {}
                Op::BindVertexBuffer => {
                    let stream = record.ints[0];
                    if (0..16).contains(&stream) {
                        self.state.buffers[stream as usize] =
                            Some((record.words[0], record.words[1]));
                    } else {
                        self.skip(format!("vertex stream {stream} outside 0..16"));
                    }
                }
                Op::SetViewport => {
                    let mut viewport = [0.0f32; 6];
                    viewport[..4].copy_from_slice(&record.floats);
                    viewport[4] = f32::from_bits(record.words[0] as u32);
                    viewport[5] = f32::from_bits(record.words[1] as u32);
                    self.state.viewport = Some(viewport);
                }
                Op::DrawArrays => {
                    self.stats.draws += 1;
                    self.stats.draw_arrays += 1;
                    let draw = DrawCall {
                        topology: record.ints[0].max(0) as u32,
                        first: record.words[0] as u32,
                        count: record.words[1] as u32,
                        instances: record.ints[2].max(0) as u32,
                        base_instance: record.ints[1].max(0) as u32,
                        index: None,
                    };
                    if let Err(why) = sink.draw(&self.state, &draw) {
                        self.skip(&why);
                    } else {
                        self.last_drawn = self.colour_target(0);
                    }
                }
                Op::DrawElements => {
                    self.stats.draws += 1;
                    self.stats.draw_elements += 1;
                    let draw = DrawCall {
                        topology: record.ints[0].max(0) as u32,
                        first: 0,
                        count: record.words[1] as u32,
                        instances: record.ints[3].max(0) as u32,
                        base_instance: record.words[2] as u32,
                        index: Some(IndexCall {
                            address: record.words[0],
                            kind: record.ints[1].clamp(0, 2) as u32,
                            base_vertex: record.ints[2],
                        }),
                    };
                    if let Err(why) = sink.draw(&self.state, &draw) {
                        self.skip(&why);
                    } else {
                        self.last_drawn = self.colour_target(0);
                    }
                }
                Op::DispatchCompute => {
                    self.stats.dispatches += 1;
                    let grid = [
                        record.words[0] as u32,
                        record.words[1] as u32,
                        record.words[2] as u32,
                    ];
                    if let Err(why) = sink.dispatch(grid) {
                        self.skip(&why);
                    }
                }
                Op::Barrier => sink.barrier(record.words[0]),
                Op::CallCommands => {
                    self.stats.calls += 1;
                    self.call(list, &record, depth, sink)?;
                }
                Op::CopyCommands => {
                    self.stats.copies += 1;
                    if record.flags & record::FLAG_INLINE != 0 {
                        // The copied records follow the marker in this list; walk them in place.
                        let inner = at;
                        let inner_end = (at + record.count as usize).min(end);
                        self.run_records(list, inner, inner_end, depth, sink)?;
                        at = inner_end;
                    } else {
                        self.call(list, &record, depth, sink)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Counts one skipped record and logs its reason once per submit.
    fn skip(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        Stats::skip(&mut self.stats, &reason);
        if self.logged.insert(reason.clone()) {
            self.log.push(format!("nvn replay: skipped: {reason}"));
        }
    }

    fn call(
        &mut self,
        list: &ListSnapshot,
        record: &Record,
        depth: usize,
        sink: &mut dyn Sink,
    ) -> Result<(), String> {
        if depth >= MAX_DEPTH {
            return Err(format!("CallCommands nested deeper than {MAX_DEPTH}"));
        }
        let handles = self.blob(list, record)?;
        for handle in handles.as_chunks::<8>().0 {
            let handle = u64::from_le_bytes(*handle);
            let Some(callee) = self.snapshots.get(&handle).cloned() else {
                self.skip(format!("call to {handle:#x}, which has no snapshot"));
                continue;
            };
            self.walk(&callee, depth + 1, sink)?;
        }
        Ok(())
    }

    fn blob(&mut self, list: &ListSnapshot, record: &Record) -> Result<Vec<u8>, String> {
        if record.flags & record::FLAG_BLOB == 0 || record.blob_len == 0 {
            return Ok(Vec::new());
        }
        if let Some((_, data)) = list.blobs.iter().find(|(at, _)| *at == record.blob) {
            return Ok(data.clone());
        }
        Err(format!(
            "record blob at +{:#x} is not in the snapshot",
            record.blob
        ))
    }

    fn values(&mut self, list: &ListSnapshot, record: &Record) -> Result<Vec<StateValue>, String> {
        if record.flags & record::FLAG_UNRESOLVED != 0 {
            self.skip(format!(
                "{:?} of an object the host never saw is not decoded",
                record.op
            ));
            return Ok(Vec::new());
        }
        if record.flags & record::FLAG_RAW != 0 {
            self.skip(format!(
                "{:?} carries a raw state struct, which the replay cannot decode yet",
                record.op
            ));
            return Ok(Vec::new());
        }
        let blob = self.blob(list, record)?;
        Ok(blob
            .as_chunks::<16>()
            .0
            .iter()
            .filter_map(|chunk| StateValue::decode(chunk))
            .collect())
    }

    fn targets(&mut self, list: &ListSnapshot, record: &Record) -> Result<Targets, String> {
        let blob = self.blob(list, record)?;
        let words: Vec<u64> = blob
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w))
            .collect();
        let count = record.count.min(8) as usize;
        if words.len() < 2 * count + 2 {
            return Err("SetRenderTargets blob is short".into());
        }
        let mut colors = Vec::with_capacity(count);
        for index in 0..count {
            let texture = words[2 * index];
            let view = words[2 * index + 1];
            colors.push((texture != 0).then_some(ViewRef { texture, view }));
        }
        while colors.last().is_some_and(Option::is_none) {
            colors.pop();
        }
        let depth = (words[2 * count] != 0).then_some(ViewRef {
            texture: words[2 * count],
            view: words[2 * count + 1],
        });
        Ok(Targets { colors, depth })
    }

    fn bind_render(&mut self, list: &ListSnapshot, record: &Record) -> Result<(), String> {
        for value in self.values(list, record)? {
            let Some(kind) = record::kind_from_code(value.kind) else {
                self.skip("state field with an unknown tag");
                continue;
            };
            apply(&mut self.state.render, kind, value.index, value.value);
        }
        Ok(())
    }

    fn bind_attribs(&mut self, list: &ListSnapshot, record: &Record) -> Result<(), String> {
        let values = self.values(list, record)?;
        let count = values
            .iter()
            .find(|v| v.kind == record::kind_code(Kind::AttribCount))
            .map_or(0, |v| v.value as usize)
            .min(32);
        for index in 0..count {
            self.state.attribs[index] = Some(Attrib::default());
        }
        for value in values {
            let Some(kind) = record::kind_from_code(value.kind) else {
                continue;
            };
            let Some(slot) = self.state.attribs.get_mut(value.index as usize) else {
                continue;
            };
            let slot = slot.get_or_insert_with(Attrib::default);
            match kind {
                Kind::AttribFormat => slot.format = value.value as u32,
                Kind::AttribOffset => slot.offset = value.value as u32,
                Kind::AttribStream => slot.stream = value.value as u32,
                Kind::AttribCount => {}
                _ => {}
            }
        }
        Ok(())
    }

    fn bind_streams(&mut self, list: &ListSnapshot, record: &Record) -> Result<(), String> {
        let values = self.values(list, record)?;
        let count = values
            .iter()
            .find(|v| v.kind == record::kind_code(Kind::StreamCount))
            .map_or(0, |v| v.value as usize)
            .min(16);
        for index in 0..count {
            self.state.streams[index] = Some(Stream::default());
        }
        for value in values {
            let Some(kind) = record::kind_from_code(value.kind) else {
                continue;
            };
            let Some(slot) = self.state.streams.get_mut(value.index as usize) else {
                continue;
            };
            let slot = slot.get_or_insert_with(Stream::default);
            match kind {
                Kind::StreamStride => slot.stride = value.value as u32,
                Kind::StreamDivisor => slot.divisor = value.value as u32,
                _ => {}
            }
        }
        Ok(())
    }
}

fn apply(render: &mut Render, kind: Kind, index: u8, value: u64) {
    let target = index as usize;
    match kind {
        Kind::ChannelMask => {
            if let Some(slot) = render.channel_mask.get_mut(target) {
                *slot = value as u8 & 0xf;
            }
        }
        Kind::BlendEnable => {
            if let Some(slot) = render.blend_enable.get_mut(target) {
                *slot = value != 0;
            }
            render.blend_target = index;
        }
        Kind::BlendEquation => render.blend_equation = value as u32,
        Kind::BlendSrc => render.blend_src = value as u32,
        Kind::BlendDst => render.blend_dst = value as u32,
        Kind::BlendEquationAlpha => render.blend_equation_alpha = value as u32,
        Kind::BlendSrcAlpha => render.blend_src_alpha = value as u32,
        Kind::BlendDstAlpha => render.blend_dst_alpha = value as u32,
        Kind::DepthTest => render.depth_test = value != 0,
        Kind::DepthWrite => render.depth_write = value != 0,
        Kind::DepthFunc => render.depth_func = value as u32,
        Kind::StencilTest => render.stencil_test = value != 0,
        Kind::StencilFunc => {
            if let Some(face) = render.stencil.get_mut(target) {
                face[0] = value as u32;
            }
        }
        Kind::StencilFail => {
            if let Some(face) = render.stencil.get_mut(target) {
                face[1] = value as u32;
            }
        }
        Kind::StencilDepthFail => {
            if let Some(face) = render.stencil.get_mut(target) {
                face[2] = value as u32;
            }
        }
        Kind::StencilPass => {
            if let Some(face) = render.stencil.get_mut(target) {
                face[3] = value as u32;
            }
        }
        Kind::CullFace => render.cull = value as u32,
        Kind::FrontFace => render.front = value as u32,
        _ => {}
    }
}

/// Replays one submit against a sink. Used by the worker and by the tests.
pub fn replay(submit: &Submit, sink: &mut dyn Sink) -> Stats {
    let mut walker = Walker::new(submit);
    walker.run(submit, sink);
    for line in &walker.log {
        eprintln!("{line}");
    }
    walker.stats
}

/// The state of every command buffer the replay worker has seen. NVN state belongs to the command buffer and
/// persists across submissions: a frame sets its render targets and program at the end of one list and the draw that
/// uses them is the first command of the next submit on the same buffer (`docs/nvn/CONTRACT.md` §7). One replay
/// worker, one state table.
static STATES: Mutex<Option<HashMap<u64, State>>> = Mutex::new(None);

/// Replays one submit on the worker, continuing each command buffer from where its previous submit left it.
pub fn replay_resumed(submit: &Submit, sink: &mut dyn Sink) -> Stats {
    let mut walker = Walker::new(submit);
    if let Ok(mut states) = STATES.lock() {
        walker.states = states.take().unwrap_or_default();
    }
    walker.run(submit, sink);
    let stats = walker.stats.clone();
    if let Ok(mut states) = STATES.lock() {
        *states = Some(walker.states.clone());
    }
    for line in &walker.log {
        eprintln!("{line}");
    }
    stats
}

/// `nvnCommandBufferInitialize`/`Finalize`: the buffer's recorded state goes with it.
pub fn reset_state(command_buffer: u64) {
    if let Ok(mut states) = STATES.lock()
        && let Some(states) = states.as_mut()
    {
        states.remove(&command_buffer);
    }
}

// ---- the worker ----------------------------------------------------------------------------------------------------

enum Job {
    Submit(Box<Submit>),
}

struct Queue {
    jobs: VecDeque<(u64, Job)>,
    enqueued: u64,
    done: u64,
    /// The process the worker replays for; set with the first submit.
    process: Option<&'static Process>,
    started: bool,
}

fn queue() -> &'static (Mutex<Queue>, Condvar) {
    static QUEUE: OnceLock<(Mutex<Queue>, Condvar)> = OnceLock::new();
    QUEUE.get_or_init(|| {
        (
            Mutex::new(Queue {
                jobs: VecDeque::new(),
                enqueued: 0,
                done: 0,
                process: None,
                started: false,
            }),
            Condvar::new(),
        )
    })
}

fn lock() -> std::sync::MutexGuard<'static, Queue> {
    queue()
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wait_timeout(
    guard: std::sync::MutexGuard<'static, Queue>,
) -> std::sync::MutexGuard<'static, Queue> {
    match queue()
        .1
        .wait_timeout(guard, std::time::Duration::from_millis(50))
    {
        Ok((guard, _)) => guard,
        Err(error) => error.into_inner().0,
    }
}

fn wait_here(
    guard: std::sync::MutexGuard<'static, Queue>,
) -> std::sync::MutexGuard<'static, Queue> {
    match queue().1.wait(guard) {
        Ok(guard) => guard,
        Err(error) => error.into_inner(),
    }
}

/// Whether replay is switched on. `KIRIKAE_NVN=record` turns recording on; replay follows it.
pub static ACTIVE: AtomicBool = AtomicBool::new(false);

static TOTALS: Mutex<Option<Stats>> = Mutex::new(None);

/// Queues a submit and returns its serial. Blocks only while [`MAX_PENDING`] submits are still in flight, which keeps
/// the submission order and the presented frame one submit behind the game at most.
pub fn submit(p: &'static Process, submit: Submit) -> u64 {
    if !ACTIVE.load(Ordering::Relaxed) {
        return 0;
    }
    let mut guard = lock();
    while guard.started && guard.enqueued.saturating_sub(guard.done) >= MAX_PENDING {
        guard = wait_timeout(guard);
    }
    guard.enqueued += 1;
    let serial = guard.enqueued;
    guard
        .jobs
        .push_back((serial, Job::Submit(Box::new(submit))));
    guard.process = Some(p);
    if !guard.started {
        guard.started = true;
        let _ = std::thread::Builder::new()
            .name("kirikae-nvn-replay".into())
            .spawn(worker);
    }
    drop(guard);
    queue().1.notify_all();
    serial
}

fn worker() {
    loop {
        let (serial, job) = {
            let mut guard = lock();
            let job = loop {
                if let Some(job) = guard.jobs.pop_front() {
                    break Some(job);
                }
                guard = wait_here(guard);
            };
            match job {
                Some(job) => job,
                None => return,
            }
        };
        let Job::Submit(submit) = job;
        let p = lock().process;
        if let Some(p) = p {
            let stats = crate::gpu::vulkan::replay_submit(p, &submit);
            stats_add(stats);
        }
        let mut guard = lock();
        guard.done = guard.done.max(serial);
        drop(guard);
        queue().1.notify_all();
    }
}

fn stats_add(stats: Stats) {
    let mut totals = TOTALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let total = totals.get_or_insert_with(Stats::default);
    total.submits += 1;
    total.lists += stats.lists;
    total.records += stats.records;
    total.calls += stats.calls;
    total.copies += stats.copies;
    total.chained += stats.chained;
    total.draws += stats.draws;
    total.draw_arrays += stats.draw_arrays;
    total.draw_elements += stats.draw_elements;
    total.clears += stats.clears;
    total.dispatches += stats.dispatches;
    total.skipped += stats.skipped;
    total.skipped_depth += stats.skipped_depth;
    for (reason, count) in stats.reasons {
        *total.reasons.entry(reason).or_default() += count;
    }
}

/// Waits until the submit named by `serial` has been replayed. 0 waits for everything queued so far.
pub fn wait(serial: u64) {
    let mut guard = lock();
    let wanted = if serial == 0 { guard.enqueued } else { serial };
    while guard.done < wanted {
        guard = wait_timeout(guard);
    }
}

/// Waits for every submit queued so far: `nvnQueueFinish`, pool invalidation and shutdown use this.
pub fn finish() {
    wait(0);
}

/// Prints the replay totals when they changed, in the shape the other NVN reports use.
pub fn report(frames: u64) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let total = TOTALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .unwrap_or_default();
    if total.submits == 0 {
        return;
    }
    let per = |n: u64| n as f64 / frames.max(1) as f64;
    eprintln!(
        "kirikae:         nvn replay: {} submits, {} lists, {} records, {} calls, {} copies, {} chained; {} draws ({} arrays, {} elements), {} clears, {} dispatches; {} skipped",
        total.submits,
        total.lists,
        total.records,
        total.calls,
        total.copies,
        total.chained,
        total.draws,
        total.draw_arrays,
        total.draw_elements,
        total.clears,
        total.dispatches,
        total.skipped
    );
    eprintln!(
        "kirikae:         nvn replay per frame: {:.0} lists, {:.0} records, {:.0} draws, {:.0} clears, {} skipped",
        per(total.lists),
        per(total.records),
        per(total.draws),
        per(total.clears),
        total.skipped
    );
    let mut reasons: Vec<(&String, &u64)> = total.reasons.iter().collect();
    reasons.sort_by_key(|r| std::cmp::Reverse(*r.1));
    for (reason, count) in reasons.iter().take(12) {
        eprintln!("kirikae:           {count} skipped: {reason}");
    }
}

/// Logs the first occurrence of each distinct reason once, as the rest of NVN does.
pub fn report_totals() -> Stats {
    TOTALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::super::record::{FakeMemory, Recorder};
    use super::*;

    const BASE: u64 = 0x1_0000_0000;
    const CMD: u64 = 0x4000;
    const OTHER: u64 = 0x5000;

    #[derive(Default)]
    struct Recording {
        events: Vec<String>,
        states: Vec<State>,
        targets_seen: Vec<Targets>,
        clear_colors: Vec<(u32, [f32; 4], u32)>,
        clears: Vec<(Option<f32>, Option<i32>)>,
        draws: u32,
        window_targets: Vec<(u64, Option<u64>)>,
        texture_binds: Vec<(usize, usize, u32)>,
    }

    impl Sink for Recording {
        fn targets(&mut self, targets: &Targets) -> Result<(), String> {
            self.events
                .push(format!("targets {}", targets.colors.len()));
            self.targets_seen.push(targets.clone());
            Ok(())
        }
        fn clear_color(&mut self, index: u32, color: [f32; 4], mask: u32) -> Result<(), String> {
            self.clear_colors.push((index, color, mask));
            self.events.push(format!("clear color {index}"));
            Ok(())
        }
        fn clear_depth_stencil(
            &mut self,
            depth: Option<f32>,
            stencil: Option<i32>,
        ) -> Result<(), String> {
            self.clears.push((depth, stencil));
            self.events.push("clear depth".into());
            Ok(())
        }
        fn draw(&mut self, state: &State, draw: &DrawCall) -> Result<(), String> {
            self.draws += 1;
            self.states.push(state.clone());
            self.events.push(format!(
                "draw {} {} {}",
                draw.topology,
                draw.count,
                draw.index.is_some()
            ));
            Ok(())
        }
        fn barrier(&mut self, _: u64) {
            self.events.push("barrier".into());
        }
        fn window_target(&mut self, window: u64, source: Option<u64>) {
            self.window_targets.push((window, source));
        }
        fn texture_bound(&mut self, stage: usize, index: usize, handle: u32) {
            self.texture_binds.push((stage, index, handle));
        }
    }

    fn snapshot(mem: &FakeMemory, recorder: &Recorder, handle: u64) -> Arc<ListSnapshot> {
        snapshot_on(mem, recorder, handle, 0)
    }

    fn snapshot_on(
        mem: &FakeMemory,
        recorder: &Recorder,
        handle: u64,
        command_buffer: u64,
    ) -> Arc<ListSnapshot> {
        let list = recorder.list(handle).expect("list is live");
        Arc::new(
            ListSnapshot::capture(mem, recorder, handle, list.bytes, command_buffer)
                .expect("snapshot"),
        )
    }

    /// Builds a callee (clear + draw) and a caller (targets + call + copy + draw), the shape of a frame.
    fn scripted() -> (FakeMemory, Recorder, u64, u64) {
        let mem = FakeMemory::new(BASE, 0x8000);
        let mut recorder = Recorder::default();
        recorder
            .begin(&mem, OTHER, BASE + 0x4000, 0x1000)
            .expect("callee");
        let mut clear = Record::new(Op::ClearColor);
        clear.set_int(0, 0);
        clear.set_int(1, 0xf);
        clear.set_float(0, 0.25);
        clear.set_float(3, 1.0);
        recorder.emit(&mem, OTHER, clear).expect("clear");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 1);
        draw.set_word(1, 3);
        recorder.emit(&mem, OTHER, draw).expect("callee draw");
        let callee = recorder.end(&mem, OTHER).expect("end callee");

        recorder.begin(&mem, CMD, BASE, 0x800).expect("caller");
        let targets = BASE + 0x3000;
        // One colour pair (texture, view) and the depth pair, the blob `SetRenderTargets` stores.
        mem.poke(
            targets,
            &[
                0x1111u64.to_le_bytes(),
                0x2222u64.to_le_bytes(),
                0x3333u64.to_le_bytes(),
                0x4444u64.to_le_bytes(),
            ]
            .concat(),
        );
        let mut record = Record::new(Op::SetRenderTargets);
        record.count = 1;
        recorder
            .emit_blob(&mem, CMD, record, &mem.read_back(targets, 32))
            .expect("targets");
        recorder.call(&mem, CMD, &[callee]).expect("call");
        recorder.copy(&mem, CMD, &[callee]).expect("copy");
        let mut draw = Record::new(Op::DrawElements);
        draw.set_int(0, 4);
        draw.set_int(1, 2);
        draw.set_int(2, -1);
        draw.set_word(0, 0x5555);
        draw.set_word(1, 6);
        recorder.emit(&mem, CMD, draw).expect("caller draw");
        let caller = recorder.end(&mem, CMD).expect("end caller");
        (mem, recorder, caller, callee)
    }

    #[test]
    fn calls_and_inline_copies_replay_in_order() {
        let (mem, recorder, caller, callee) = scripted();
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, caller)],
            callees: HashMap::from([(callee, snapshot(&mem, &recorder, callee))]),
            ..Submit::default()
        };
        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(
            sink.events,
            [
                "targets 1",
                "clear color 0",
                "draw 4 3 false",
                "clear color 0",
                "draw 4 3 false",
                "draw 4 6 true",
            ]
        );
        assert_eq!(sink.draws, 3);
        assert_eq!(sink.clear_colors.len(), 2);
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.copies, 1);
        assert_eq!(stats.records, 12);
        assert_eq!(stats.skipped, 0);
        assert_eq!(sink.states[2].program, 0);
    }

    #[test]
    fn a_call_without_a_snapshot_is_reported_and_the_list_continues() {
        let (mem, recorder, caller, _callee) = scripted();
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, caller)],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        // The caller's own draw and the copy's inlined draw still ran; only the call was lost.
        assert_eq!(sink.draws, 2);
        assert!(stats.skipped >= 1);
        assert!(
            stats
                .reasons
                .keys()
                .any(|reason| reason.contains("no snapshot"))
        );
    }

    #[test]
    fn clear_selection_keeps_the_mask_and_the_value() {
        let mem = FakeMemory::new(BASE, 0x2000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, 0x800).expect("list");
        let mut clear = Record::new(Op::ClearColor);
        clear.set_int(0, 1);
        clear.set_int(1, 0x5);
        clear.set_float(0, 1.0);
        clear.set_float(1, 0.5);
        clear.set_float(2, 0.25);
        clear.set_float(3, 1.0);
        recorder.emit(&mem, CMD, clear).expect("clear");
        let mut depth = Record::new(Op::ClearDepthStencil);
        depth.set_float(0, 0.75);
        depth.set_int(0, 7);
        depth.set_int(1, 1);
        depth.set_int(2, 1);
        recorder.emit(&mem, CMD, depth).expect("depth");
        let handle = recorder.end(&mem, CMD).expect("end");
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        replay(&submit, &mut sink);
        assert_eq!(sink.clear_colors, [(1, [1.0, 0.5, 0.25, 1.0], 0x5)]);
        assert_eq!(sink.clears, [(Some(0.75), Some(7))]);
    }

    #[test]
    fn binds_decode_by_value_and_land_in_the_draw_state() {
        let mem = FakeMemory::new(BASE, 0x2000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, 0x800).expect("list");
        let values = [
            StateValue {
                kind: record::kind_code(Kind::ChannelMask),
                index: 0,
                value: 0x3,
            },
            StateValue {
                kind: record::kind_code(Kind::BlendEnable),
                index: 0,
                value: 1,
            },
            StateValue {
                kind: record::kind_code(Kind::BlendSrc),
                index: 0,
                value: 5,
            },
            StateValue {
                kind: record::kind_code(Kind::DepthTest),
                index: 0,
                value: 1,
            },
            StateValue {
                kind: record::kind_code(Kind::DepthFunc),
                index: 0,
                value: 4,
            },
        ];
        recorder
            .emit_state(&mem, CMD, Op::BindBlendState, 0x7000, &values)
            .expect("state");
        let attribs = [
            StateValue {
                kind: record::kind_code(Kind::AttribCount),
                index: 0,
                value: 2,
            },
            StateValue {
                kind: record::kind_code(Kind::AttribFormat),
                index: 0,
                value: 0x2e,
            },
            StateValue {
                kind: record::kind_code(Kind::AttribOffset),
                index: 0,
                value: 8,
            },
            StateValue {
                kind: record::kind_code(Kind::AttribStream),
                index: 0,
                value: 1,
            },
        ];
        recorder
            .emit_state(&mem, CMD, Op::BindVertexAttribState, 0x8000, &attribs)
            .expect("attribs");
        let mut texture = Record::new(Op::BindTexture);
        texture.set_int(0, 1);
        texture.set_int(1, 3);
        texture.set_word(0, 0x0020_2007);
        recorder.emit(&mem, CMD, texture).expect("texture");
        let mut uniform = Record::new(Op::BindUniformBuffer);
        uniform.set_int(0, 1);
        uniform.set_int(1, 2);
        uniform.set_word(0, 0x9000);
        uniform.set_word(1, 0x100);
        recorder.emit(&mem, CMD, uniform).expect("uniform");
        let mut storage = Record::new(Op::BindStorageBuffer);
        storage.set_int(0, 1);
        storage.set_int(1, 4);
        storage.set_word(0, 0xa000);
        storage.set_word(1, 0x200);
        recorder.emit(&mem, CMD, storage).expect("storage");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 0);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let handle = recorder.end(&mem, CMD).expect("end");
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        replay(&submit, &mut sink);
        let state = &sink.states[0];
        assert_eq!(state.render.channel_mask[0], 0x3);
        assert!(state.render.blend_enable[0]);
        assert_eq!(state.render.blend_src, 5);
        assert!(state.render.depth_test);
        assert_eq!(state.render.depth_func, 4);
        assert_eq!(
            state.attribs[0],
            Some(Attrib {
                format: 0x2e,
                offset: 8,
                stream: 1
            })
        );
        assert_eq!(state.attribs[1], Some(Attrib::default()));
        assert_eq!(state.attribs[2], None);
        assert_eq!(state.textures[1][3], 0x0020_2007);
        assert_eq!(state.cbufs[1][2], Some((0x9000, 0x100)));
        assert_eq!(state.storage[1][4], Some((0xa000, 0x200)));
        assert_eq!(state.storage[0][4], None);
    }

    #[test]
    fn targets_keep_the_pairs_the_record_snapshotted() {
        let mem = FakeMemory::new(BASE, 0x2000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, 0x800).expect("list");
        let blob: Vec<u8> = [0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let mut record = Record::new(Op::SetRenderTargets);
        record.count = 2;
        recorder
            .emit_blob(&mem, CMD, record, &blob)
            .expect("targets");
        let handle = recorder.end(&mem, CMD).expect("end");
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        replay(&submit, &mut sink);
        assert_eq!(
            sink.targets_seen,
            [Targets {
                colors: vec![
                    Some(ViewRef {
                        texture: 0x11,
                        view: 0x22
                    }),
                    Some(ViewRef {
                        texture: 0x33,
                        view: 0x44
                    }),
                ],
                depth: Some(ViewRef {
                    texture: 0x55,
                    view: 0x66
                }),
            }]
        );
    }

    #[test]
    fn a_nested_copy_stays_inside_its_own_range() {
        let mem = FakeMemory::new(BASE, 0x8000);
        let mut recorder = Recorder::default();
        recorder
            .begin(&mem, OTHER, BASE + 0x4000, 0x1000)
            .expect("callee");
        let mut clear = Record::new(Op::ClearColor);
        clear.set_int(0, 2);
        recorder.emit(&mem, OTHER, clear).expect("clear");
        let callee = recorder.end(&mem, OTHER).expect("end callee");

        recorder.begin(&mem, CMD, BASE, 0x800).expect("caller");
        recorder.copy(&mem, CMD, &[callee]).expect("copy");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 0);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("own draw");
        let caller = recorder.end(&mem, CMD).expect("end caller");

        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, caller)],
            callees: HashMap::from([(callee, snapshot(&mem, &recorder, callee))]),
            ..Submit::default()
        };
        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(sink.events, ["clear color 2", "draw 4 3 false"]);
        assert_eq!(stats.copies, 1);
        assert_eq!(sink.draws, 1);
    }

    #[test]
    fn an_implicit_list_snapshots_into_a_submit_and_replays_its_commands() {
        let mem = FakeMemory::new(BASE, 0x8000);
        let mut recorder = Recorder::default();
        // The driver recorded this list on its own; the host captured it in the arena, keyed by the handle the
        // driver reported (`record.rs`, "Implicit lists").
        let storage = recorder.begin_arena(&mem, CMD).expect("begin");
        let mut clear = Record::new(Op::ClearColor);
        clear.set_int(0, 0);
        clear.set_int(1, 0xf);
        clear.set_float(0, 0.5);
        recorder.emit(&mem, CMD, clear).expect("clear");
        let targets = BASE + 0x3000;
        mem.poke(
            targets,
            &[
                0x1111u64.to_le_bytes(),
                0x2222u64.to_le_bytes(),
                0x3333u64.to_le_bytes(),
                0x4444u64.to_le_bytes(),
            ]
            .concat(),
        );
        let mut targets_record = Record::new(Op::SetRenderTargets);
        targets_record.count = 1;
        recorder
            .emit_blob(&mem, CMD, targets_record, &mem.read_back(targets, 32))
            .expect("targets");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 0);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let key = recorder.end_arena(&mem, CMD).expect("end");
        assert_eq!(key, storage);

        // `EndRecording`'s second report names the GPU handle; the list is filed under it and the game submits it.
        let handle = 0x2117_fc208u64;
        assert!(recorder.refile(key, handle));
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            ..Submit::default()
        };
        // The snapshot owns the records and blobs; dropping the arena list cannot change a queued submit.
        recorder.drop_list(handle);
        assert!(recorder.list(handle).is_none());

        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(stats.records, 5);
        assert_eq!(stats.calls, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(sink.draws, 1);
        assert_eq!(
            sink.events,
            ["clear color 0", "targets 1", "draw 4 3 false"]
        );
        assert_eq!(
            sink.targets_seen[0].colors[0].map(|c| c.texture),
            Some(0x1111)
        );
    }

    #[test]
    fn a_draw_in_a_later_chunk_sees_the_chunks_it_continues() {
        let mem = FakeMemory::new(BASE, 0x8000);
        let mut recorder = Recorder::default();
        // The chunk the driver ended under its internal pointer: a clear, then the program and targets the window
        // pass set up.
        recorder.begin_arena(&mem, CMD).expect("prefix");
        let mut clear = Record::new(Op::ClearColor);
        clear.set_int(0, 0);
        clear.set_int(1, 0xf);
        clear.set_float(3, 1.0);
        recorder.emit(&mem, CMD, clear).expect("clear");
        let blob: Vec<u8> = [0x1111u64, 0x2222, 0x3333, 0x4444]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let mut targets = Record::new(Op::SetRenderTargets);
        targets.count = 1;
        recorder
            .emit_blob(&mem, CMD, targets, &blob)
            .expect("targets");
        let mut program = Record::new(Op::BindProgram);
        program.set_word(0, 0x9000);
        recorder.emit(&mem, CMD, program).expect("program");
        let prefix = recorder.end_arena(&mem, CMD).expect("end prefix");

        // The game handle's list only draws: without the chunk it continues, it has no targets and no program.
        recorder.begin_arena(&mem, CMD).expect("successor");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 0);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let successor = recorder.end_arena(&mem, CMD).expect("end successor");

        recorder.chain(successor, &[prefix]).expect("chain");
        let handle = 0x2117_fc208u64;
        assert!(recorder.refile(successor, handle));
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            callees: HashMap::from([(prefix, snapshot(&mem, &recorder, prefix))]),
            ..Submit::default()
        };
        assert_eq!(submit.roots[0].chained, [prefix]);

        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(stats.chained, 1);
        assert_eq!(stats.skipped, 0);
        assert_eq!(sink.draws, 1);
        assert_eq!(
            sink.events,
            ["clear color 0", "targets 1", "draw 4 3 false"],
            "the chunk replays before the records it continues"
        );
        assert_eq!(sink.states[0].program, 0x9000);
        assert_eq!(
            sink.states[0].targets.colors[0].map(|c| c.texture),
            Some(0x1111)
        );
    }

    #[test]
    fn a_chained_chunk_without_a_snapshot_is_reported_and_the_list_still_runs() {
        let (mem, recorder, _targets, draw) = targets_then_draw();
        let mut root = (*snapshot_on(&mem, &recorder, draw, CMD)).clone();
        root.chained = vec![0xdead_0000];
        let submit = Submit {
            roots: vec![Arc::new(root)],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(stats.chained, 0);
        assert_eq!(stats.skipped, 1);
        assert!(
            stats
                .reasons
                .keys()
                .any(|reason| reason.contains("chained chunk")),
            "{:?}",
            stats.reasons
        );
        assert_eq!(sink.draws, 1, "the list itself is still replayed");
    }

    #[test]
    fn a_command_buffer_state_persists_across_submits_and_is_reset() {
        let mem = FakeMemory::new(BASE, 0x8000);
        let mut recorder = Recorder::default();
        // The setup list: targets and program, no draw.
        recorder.begin(&mem, CMD, BASE, 0x800).expect("setup");
        let blob: Vec<u8> = [0x1111u64, 0x2222, 0, 0]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let mut targets = Record::new(Op::SetRenderTargets);
        targets.count = 1;
        recorder
            .emit_blob(&mem, CMD, targets, &blob)
            .expect("targets");
        let mut program = Record::new(Op::BindProgram);
        program.set_word(0, 0x9000);
        recorder.emit(&mem, CMD, program).expect("program");
        let setup = recorder.end(&mem, CMD).expect("end setup");

        // The draw is the next submit's first command, on the same buffer.
        recorder
            .begin(&mem, CMD, BASE + 0x1000, 0x800)
            .expect("draw");
        let mut draw = Record::new(Op::DrawElements);
        draw.set_int(0, 4);
        draw.set_int(1, 2);
        draw.set_word(0, 0x5555);
        draw.set_word(1, 6);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let drawn = recorder.end(&mem, CMD).expect("end draw");

        let setup = Submit {
            roots: vec![snapshot_on(&mem, &recorder, setup, CMD)],
            ..Submit::default()
        };
        let drawn = Submit {
            roots: vec![snapshot_on(&mem, &recorder, drawn, CMD)],
            ..Submit::default()
        };
        reset_state(CMD);
        let mut sink = Recording::default();
        assert_eq!(replay_resumed(&setup, &mut sink).draws, 0);

        let mut sink = Recording::default();
        let stats = replay_resumed(&drawn, &mut sink);
        assert_eq!(stats.skipped, 0, "the state of the buffer is continued");
        assert_eq!(sink.draws, 1);
        assert_eq!(sink.states[0].program, 0x9000);
        assert_eq!(
            sink.states[0].targets.colors[0].map(|color| color.texture),
            Some(0x1111)
        );

        // Initializing the buffer forgets its state: the draw starts at NVN's defaults again.
        reset_state(CMD);
        let mut sink = Recording::default();
        replay_resumed(&drawn, &mut sink);
        assert_eq!(sink.states[0].program, 0);
        assert!(sink.states[0].targets.colors.is_empty());
        reset_state(CMD);
    }

    /// A list that only sets targets, and one that only draws, the way SMBW splits a command buffer around
    /// `EndRecording`.
    fn targets_then_draw() -> (FakeMemory, Recorder, u64, u64) {
        let mem = FakeMemory::new(BASE, 0x8000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, OTHER, BASE, 0x800).expect("targets");
        let blob: Vec<u8> = [0x11u64, 0x22, 0x33, 0x44]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let mut record = Record::new(Op::SetRenderTargets);
        record.count = 1;
        recorder
            .emit_blob(&mem, OTHER, record, &blob)
            .expect("targets");
        let targets = recorder.end(&mem, OTHER).expect("end targets");

        recorder
            .begin(&mem, CMD, BASE + 0x2000, 0x800)
            .expect("draw");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 0);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("draw");
        let draw = recorder.end(&mem, CMD).expect("end draw");
        (mem, recorder, targets, draw)
    }

    #[test]
    fn a_root_of_another_command_buffer_does_not_inherit_its_targets() {
        let (mem, recorder, targets, draw) = targets_then_draw();
        let submit = Submit {
            roots: vec![
                snapshot_on(&mem, &recorder, targets, OTHER),
                snapshot_on(&mem, &recorder, draw, CMD),
            ],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        replay(&submit, &mut sink);
        assert_eq!(sink.draws, 1);
        assert!(
            sink.states[0].targets.colors.is_empty(),
            "the second buffer did not inherit the first one's targets"
        );
    }

    #[test]
    fn a_window_target_without_a_draw_reports_the_last_drawn_target() {
        let mem = FakeMemory::new(BASE, 0x2000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, 0x800).expect("list");
        let pairs = |texture: u64| {
            [texture, 0u64, 0, 0]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<u8>>()
        };
        let mut targets = Record::new(Op::SetRenderTargets);
        targets.count = 1;
        recorder
            .emit_blob(&mem, CMD, targets, &pairs(0x1111))
            .expect("scene targets");
        let mut draw = Record::new(Op::DrawArrays);
        draw.set_int(0, 4);
        draw.set_word(0, 0);
        draw.set_word(1, 3);
        recorder.emit(&mem, CMD, draw).expect("scene draw");
        // The title sets the window target and records the composition state, but never issues its draw.
        let mut targets = Record::new(Op::SetRenderTargets);
        targets.count = 1;
        recorder
            .emit_blob(&mem, CMD, targets, &pairs(0x2222))
            .expect("window targets");
        let mut texture = Record::new(Op::BindTexture);
        texture.set_int(0, 1);
        texture.set_int(1, 0);
        texture.set_word(0, 0xdead);
        recorder
            .emit(&mem, CMD, texture)
            .expect("composition texture");
        let handle = recorder.end(&mem, CMD).expect("end");
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            windows: HashSet::from([0x2222]),
            ..Submit::default()
        };
        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(sink.draws, 1);
        assert_eq!(stats.skipped, 0);
        assert_eq!(sink.window_targets, [(0x2222, Some(0x1111))]);
        assert_eq!(sink.texture_binds, [(1, 0, 0xdead)]);
    }

    #[test]
    fn a_window_target_before_any_draw_has_no_composition_source() {
        let mem = FakeMemory::new(BASE, 0x2000);
        let mut recorder = Recorder::default();
        recorder.begin(&mem, CMD, BASE, 0x800).expect("list");
        let blob: Vec<u8> = [0x2222u64, 0, 0, 0]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let mut targets = Record::new(Op::SetRenderTargets);
        targets.count = 1;
        recorder
            .emit_blob(&mem, CMD, targets, &blob)
            .expect("window targets");
        let handle = recorder.end(&mem, CMD).expect("end");
        let submit = Submit {
            roots: vec![snapshot(&mem, &recorder, handle)],
            windows: HashSet::from([0x2222]),
            ..Submit::default()
        };
        let mut sink = Recording::default();
        replay(&submit, &mut sink);
        assert_eq!(sink.window_targets, [(0x2222, None)]);
    }

    #[test]
    fn a_root_of_the_same_command_buffer_continues_its_state() {
        let (mem, recorder, targets, draw) = targets_then_draw();
        let submit = Submit {
            roots: vec![
                snapshot_on(&mem, &recorder, targets, CMD),
                snapshot_on(&mem, &recorder, draw, CMD),
            ],
            ..Submit::default()
        };
        let mut sink = Recording::default();
        let stats = replay(&submit, &mut sink);
        assert_eq!(stats.skipped, 0);
        assert_eq!(sink.draws, 1);
        assert_eq!(
            sink.states[0].targets.colors[0].map(|c| c.texture),
            Some(0x11)
        );
    }
}
