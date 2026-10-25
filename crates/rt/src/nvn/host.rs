//! Host side of the NVN object procs.
//!
//! `proc` maps an `nvn*` name to the object or command work the host does, and `apply` runs it from the HLE thunk
//! around the recompiled SDK driver. The driver stays the authority on builder struct layouts, and 4.3 records
//! commands with the values it and the host tables expose. A proc this module does not know keeps its existing
//! tracing/real-driver path.
//!
//! Recording only runs with `KIRIKAE_NVN=record` (see `record.rs`); everything else here is milestone 4.1/4.2.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use super::buffers::Buffer;
use super::error::Error;
use super::objects::{Counts, Object, Registry};
use super::pools::{Mapping, Pool};
use super::record;
use super::replay;
use super::samplers::{self, Sampler};
use super::state;
use super::textures::{Texture, TextureBuilder, TextureView};
use super::va::{Allocator, Ranges};
use super::window::{self, Window};
use crate::cpu::Cpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proc {
    PoolBuilderDefaults,
    PoolBuilderStorage,
    PoolBuilderFlags,
    PoolInitialize,
    PoolFinalize,
    PoolMapVirtual,
    BufferBuilderDefaults,
    BufferBuilderStorage,
    BufferInitialize,
    BufferFinalize,
    Texture(TextureProc),
    TextureInitialize,
    TextureFinalize,
    View(ViewProc),
    SamplerBuilder(samplers::Field),
    SamplerInitialize,
    SamplerFinalize,
    Program(ProgramProc),
    TexturePool,
    SamplerPool,
    Handle(HandleProc),
    State(state::Setter),
    Command(CommandProc),
    Window(WindowProc),
}

/// Program objects: the stage list a `BindProgram` names (`nvn_program.cpp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramProc {
    /// `NVNboolean (NVNprogram*, NVNdevice*)`.
    Initialize,
    Finalize,
    /// `(program, count, const NVNshaderData*)`, 0x10 bytes per entry.
    Shaders,
    /// The Ext form, 0x18 bytes per entry.
    ShadersExt,
}

/// This title takes its texture handles from the device, not from a pool registration call, so the host answers the
/// handle procs itself: they only pack two ids (`nvn_objects.cpp:640-644`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleProc {
    Texture,
    TexelFetch,
    Image,
}

/// A descriptor pool's memory range: the driver writes the TIC/TSC entries there, so the replay reads them back
/// through the pool's GPU address (`nvn_objects.cpp`, `DescriptorPoolInitialize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorPool {
    pub pool: u64,
    pub offset: u64,
    pub count: u32,
}

/// A command buffer proc the host records into the guest's command memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandProc {
    /// Lifecycle: resets the recorder, then runs the driver so it can set up the header we write through.
    Initialize,
    /// Lifecycle: the command memory is handed back, so its handles stop resolving. Kept on the driver path.
    Finalize,
    AddCommandMemory,
    BeginRecording,
    EndRecording,
    SetRenderTargets,
    ClearColor,
    ClearDepthStencil,
    DiscardColor,
    BindProgram,
    BindBlendState,
    BindColorState,
    BindChannelMaskState,
    BindDepthStencilState,
    BindPolygonState,
    BindVertexAttribState,
    BindVertexStreamState,
    BindTexture,
    BindUniformBuffer,
    BindStorageBuffer,
    SetSamplerPool,
    SetTexturePool,
    BindVertexBuffer,
    SetViewport,
    DrawArrays,
    DrawArraysInstanced,
    DrawElements,
    DrawElementsBaseVertex,
    DrawElementsInstanced,
    DispatchCompute,
    Barrier,
    CallCommands,
    CopyCommands,
    /// `nvnQueueSubmitCommands`: recorded and taken over, because the handles are ours. Replay is 4.4.
    SubmitCommands,
    /// Command-memory accounting. The driver's own header counters are stale once we record, so the host answers
    /// these from the spans and the write pointer it advanced.
    GetCommandMemorySize,
    GetCommandMemoryUsed,
    GetCommandMemoryFree,
}

/// A window proc: the swapchain a title presents from (`docs/nvn/CONTRACT.md` §8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowProc {
    BuilderDefaults,
    /// `(builder, count, const NVNtexture* const*)`: the addresses are read at the call.
    BuilderTextures,
    BuilderActiveTextures,
    BuilderPresentInterval,
    Initialize,
    Finalize,
    /// `(window, int x, int y, int w, int h)`.
    SetCrop,
    SetPresentInterval,
    SetNumActiveTextures,
    /// `nvnQueuePresentTexture(queue, window, index)`: the driver still presents; the host resolves the replayed
    /// image after it, so a target it cannot resolve stays with the driver.
    QueuePresent,
    /// `nvnWindowPresentTexture(window, index)`, the queue-less form some SDKs expose.
    PresentTexture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureProc {
    Defaults,
    Flags,
    Target,
    Width,
    Height,
    Depth,
    Size1D,
    Size2D,
    Size3D,
    Levels,
    Format,
    Samples,
    Stride,
    Storage,
    PackagedLayout,
    Swizzle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewProc {
    Defaults,
    Levels,
    Layers,
    Format,
    Target,
}

/// The host's meaning for `name`, or `None` to leave the proc on the existing path.
pub fn proc(name: &str) -> Option<Proc> {
    use Proc::*;
    Some(match name.strip_prefix("nvn")? {
        "MemoryPoolBuilderSetDefaults" => PoolBuilderDefaults,
        "MemoryPoolBuilderSetStorage" => PoolBuilderStorage,
        "MemoryPoolBuilderSetFlags" => PoolBuilderFlags,
        "MemoryPoolInitialize" => PoolInitialize,
        "MemoryPoolFinalize" => PoolFinalize,
        "MemoryPoolMapVirtual" => PoolMapVirtual,
        "BufferBuilderSetDefaults" => BufferBuilderDefaults,
        "BufferBuilderSetStorage" => BufferBuilderStorage,
        "BufferInitialize" => BufferInitialize,
        "BufferFinalize" => BufferFinalize,
        "TextureBuilderSetDefaults" => Texture(TextureProc::Defaults),
        "TextureBuilderSetFlags" => Texture(TextureProc::Flags),
        "TextureBuilderSetTarget" => Texture(TextureProc::Target),
        "TextureBuilderSetWidth" => Texture(TextureProc::Width),
        "TextureBuilderSetHeight" => Texture(TextureProc::Height),
        "TextureBuilderSetDepth" => Texture(TextureProc::Depth),
        "TextureBuilderSetSize1D" => Texture(TextureProc::Size1D),
        "TextureBuilderSetSize2D" => Texture(TextureProc::Size2D),
        "TextureBuilderSetSize3D" => Texture(TextureProc::Size3D),
        "TextureBuilderSetLevels" => Texture(TextureProc::Levels),
        "TextureBuilderSetFormat" => Texture(TextureProc::Format),
        "TextureBuilderSetSamples" => Texture(TextureProc::Samples),
        "TextureBuilderSetStride" => Texture(TextureProc::Stride),
        "TextureBuilderSetStorage" => Texture(TextureProc::Storage),
        "TextureBuilderSetPackagedTextureLayout" => Texture(TextureProc::PackagedLayout),
        "TextureBuilderSetSwizzle" => Texture(TextureProc::Swizzle),
        "TextureInitialize" => TextureInitialize,
        "TextureFinalize" => TextureFinalize,
        "TextureViewSetDefaults" => View(ViewProc::Defaults),
        "TextureViewSetLevels" => View(ViewProc::Levels),
        "TextureViewSetLayers" => View(ViewProc::Layers),
        "TextureViewSetFormat" => View(ViewProc::Format),
        "TextureViewSetTarget" => View(ViewProc::Target),
        "SamplerBuilderSetDefaults" => SamplerBuilder(samplers::Field::Defaults),
        "SamplerBuilderSetMinMagFilter" => SamplerBuilder(samplers::Field::MinMagFilter),
        "SamplerBuilderSetWrapMode" => SamplerBuilder(samplers::Field::WrapMode),
        "SamplerBuilderSetLodClamp" => SamplerBuilder(samplers::Field::LodClamp),
        "SamplerBuilderSetLodBias" => SamplerBuilder(samplers::Field::LodBias),
        "SamplerBuilderSetCompare" => SamplerBuilder(samplers::Field::Compare),
        "SamplerBuilderSetMaxAnisotropy" => SamplerBuilder(samplers::Field::MaxAnisotropy),
        "SamplerBuilderSetReductionFilter" => SamplerBuilder(samplers::Field::ReductionFilter),
        "SamplerBuilderSetBorderColor" | "SamplerBuilderSetBorderColour" => {
            SamplerBuilder(samplers::Field::BorderColor)
        }
        "SamplerInitialize" => SamplerInitialize,
        "SamplerFinalize" => SamplerFinalize,
        "ProgramInitialize" => Program(ProgramProc::Initialize),
        "ProgramFinalize" => Program(ProgramProc::Finalize),
        "ProgramSetShaders" => Program(ProgramProc::Shaders),
        "ProgramSetShadersExt" => Program(ProgramProc::ShadersExt),
        "TexturePoolInitialize" => TexturePool,
        "SamplerPoolInitialize" => SamplerPool,
        "DeviceGetTextureHandle" => Handle(HandleProc::Texture),
        "DeviceGetTexelFetchHandle" => Handle(HandleProc::TexelFetch),
        "DeviceGetImageHandle" => Handle(HandleProc::Image),
        "CommandBufferInitialize" => Command(CommandProc::Initialize),
        "CommandBufferFinalize" => Command(CommandProc::Finalize),
        "CommandBufferAddCommandMemory" => Command(CommandProc::AddCommandMemory),
        "CommandBufferBeginRecording" => Command(CommandProc::BeginRecording),
        "CommandBufferEndRecording" => Command(CommandProc::EndRecording),
        "CommandBufferSetRenderTargets" => Command(CommandProc::SetRenderTargets),
        "CommandBufferClearColor" => Command(CommandProc::ClearColor),
        "CommandBufferClearDepthStencil" => Command(CommandProc::ClearDepthStencil),
        "CommandBufferDiscardColor" => Command(CommandProc::DiscardColor),
        "CommandBufferBindProgram" => Command(CommandProc::BindProgram),
        "CommandBufferBindBlendState" => Command(CommandProc::BindBlendState),
        "CommandBufferBindColorState" => Command(CommandProc::BindColorState),
        "CommandBufferBindChannelMaskState" => Command(CommandProc::BindChannelMaskState),
        "CommandBufferBindDepthStencilState" => Command(CommandProc::BindDepthStencilState),
        "CommandBufferBindPolygonState" => Command(CommandProc::BindPolygonState),
        "CommandBufferBindVertexAttribState" => Command(CommandProc::BindVertexAttribState),
        "CommandBufferBindVertexStreamState" => Command(CommandProc::BindVertexStreamState),
        "CommandBufferBindTexture" => Command(CommandProc::BindTexture),
        "CommandBufferBindUniformBuffer" => Command(CommandProc::BindUniformBuffer),
        "CommandBufferBindStorageBuffer" => Command(CommandProc::BindStorageBuffer),
        "CommandBufferSetSamplerPool" => Command(CommandProc::SetSamplerPool),
        "CommandBufferSetTexturePool" => Command(CommandProc::SetTexturePool),
        "CommandBufferBindVertexBuffer" => Command(CommandProc::BindVertexBuffer),
        "CommandBufferSetViewport" => Command(CommandProc::SetViewport),
        "CommandBufferDrawArrays" => Command(CommandProc::DrawArrays),
        "CommandBufferDrawArraysInstanced" => Command(CommandProc::DrawArraysInstanced),
        "CommandBufferDrawElements" => Command(CommandProc::DrawElements),
        "CommandBufferDrawElementsBaseVertex" => Command(CommandProc::DrawElementsBaseVertex),
        "CommandBufferDrawElementsInstanced" => Command(CommandProc::DrawElementsInstanced),
        "CommandBufferDispatchCompute" => Command(CommandProc::DispatchCompute),
        "CommandBufferBarrier" => Command(CommandProc::Barrier),
        "CommandBufferCallCommands" => Command(CommandProc::CallCommands),
        "CommandBufferCopyCommands" => Command(CommandProc::CopyCommands),
        "QueueSubmitCommands" => Command(CommandProc::SubmitCommands),
        "WindowBuilderSetDefaults" => Window(WindowProc::BuilderDefaults),
        "WindowBuilderSetTextures" => Window(WindowProc::BuilderTextures),
        "WindowBuilderSetNumActiveTextures" => Window(WindowProc::BuilderActiveTextures),
        "WindowBuilderSetPresentInterval" => Window(WindowProc::BuilderPresentInterval),
        "WindowInitialize" => Window(WindowProc::Initialize),
        "WindowFinalize" => Window(WindowProc::Finalize),
        "WindowSetCrop" => Window(WindowProc::SetCrop),
        "WindowSetPresentInterval" => Window(WindowProc::SetPresentInterval),
        "WindowSetNumActiveTextures" => Window(WindowProc::SetNumActiveTextures),
        "QueuePresentTexture" => Window(WindowProc::QueuePresent),
        "WindowPresentTexture" => Window(WindowProc::PresentTexture),
        "CommandBufferGetCommandMemorySize" => Command(CommandProc::GetCommandMemorySize),
        "CommandBufferGetCommandMemoryUsed" => Command(CommandProc::GetCommandMemoryUsed),
        "CommandBufferGetCommandMemoryFree" => Command(CommandProc::GetCommandMemoryFree),
        _ => return state::setter(name).map(Proc::State),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct PoolBuilder {
    storage: u64,
    size: u64,
    flags: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct BufferBuilder {
    pool: u64,
    offset: u64,
    size: u64,
}

/// One `nvnCommandBufferAddCommandMemory` call, kept as the pool range it named so a virtual pool's mappings can
/// still arrive before the first `BeginRecording`.
#[derive(Clone, Copy, Debug)]
struct CommandMemory {
    pool: u64,
    offset: u64,
    size: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Snapshot {
    counts: Counts,
    failures: u64,
    record: record::Report,
    decoded: u64,
    copied: u64,
    unresolved: u64,
    presents: window::Counters,
}

#[derive(Default)]
struct Host {
    objects: Registry,
    va: Allocator,
    ranges: Ranges,
    pool_builders: HashMap<u64, PoolBuilder>,
    buffer_builders: HashMap<u64, BufferBuilder>,
    texture_builders: HashMap<u64, TextureBuilder>,
    sampler_builders: HashMap<u64, Sampler>,
    /// Programs by guest address: the stage code addresses and constant-data ranges `nvnProgramSetShaders` handed in.
    programs: HashMap<u64, replay::Program>,
    /// Descriptor pools by guest address: the memory range the driver writes TIC/TSC entries into.
    texture_pools: HashMap<u64, DescriptorPool>,
    sampler_pools: HashMap<u64, DescriptorPool>,
    /// The process the recorder serves, captured with the first recorded command: `SubmitCommands` rates the replay
    /// for it. `None` in tests, which never replay.
    process: Option<&'static crate::process::Process>,
    /// The setters' decoded values, so a bind can copy state by value at record time.
    state: state::Objects,
    /// Window builders by guest address, and the windows they initialize.
    window_builders: HashMap<u64, window::Builder>,
    windows: HashMap<u64, Window>,
    /// The submit each window last presented, so a present never re-shows a frame from before the one it follows.
    presented: HashMap<u64, u64>,
    /// Serial of the last `nvnQueueSubmitCommands` the host replayed.
    last_submit: u64,
    presents: window::Counters,
    record: record::Recorder,
    /// Command memory the game handed each command buffer, resolved through its pool at `BeginRecording`.
    command_memory: HashMap<u64, Vec<CommandMemory>>,
    /// Header offset the write pointer of each recording command buffer came from.
    write_field: HashMap<u64, u64>,
    /// Command buffers whose implicit list broke while recording: their commands stay on the driver path until the
    /// next `EndRecording`, so a partial arena list is never submitted as if it were whole.
    implicit_dead: std::collections::HashSet<u64>,
    /// Per command buffer, the arena chunks that just closed under the driver's internal pointer, in recording
    /// order, with the recorder generation they were opened at. The next list the buffer closes under a game handle
    /// continues them, so they are replayed before it (`record.rs`, "Implicit lists").
    pending: HashMap<u64, Vec<(u64, u64)>>,
    /// Arena lists each queued submit captured, by serial, retired once that submit has been replayed.
    retiring: VecDeque<(u64, Vec<u64>)>,
    /// Diagnostics: submits the host left on the driver path because a handle it named was not recorded.
    submits_driver: u64,
    record_diagnosed: bool,
    seen_failures: std::collections::HashSet<String>,
    decoded: u64,
    copied: u64,
    unresolved: u64,
    failures: u64,
    last: Snapshot,
}

fn host() -> &'static Mutex<Host> {
    static HOST: OnceLock<Mutex<Host>> = OnceLock::new();
    HOST.get_or_init(|| Mutex::new(Host::default()))
}

fn lock() -> std::sync::MutexGuard<'static, Host> {
    host()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Deferred objects pile up until a replay consumes them (4.3/4.7); with none yet, bound the list.
const DEFERRED_LIMIT: usize = 4096;

/// Runs the host's meaning of `proc`. Returns `true` when the host took the call over and the recompiled driver
/// must not run it: recording writes the guest command memory itself, so the driver cannot also write its pushbuffer
/// there (`docs/nvn/CONTRACT.md` §3).
///
/// `BeginRecording`/`EndRecording` run the real driver around the recording, outside the host lock: its own header,
/// callback and allocator state must stay consistent with what the game sees, and a guest call can yield.
pub fn apply(proc: Proc, c: &mut Cpu, real: u64) -> bool {
    if record::ON.load(Ordering::Relaxed) {
        match proc {
            Proc::Command(CommandProc::BeginRecording) => {
                let args = c.x;
                crate::hle::call_guest(c, real, false);
                // The driver's Begin already ran: never let the thunk run it a second time. If the host could not
                // open a list, the following commands fall through to the driver, which is recording now.
                c.x = args;
                {
                    let mut host = lock();
                    let _ = host.command(CommandProc::BeginRecording, c);
                }
                c.x[0] = 0;
                c.pc = c.x[30];
                return true;
            }
            Proc::Command(CommandProc::EndRecording) => {
                let args = c.x;
                let cmdbuf = c.x[0];
                let open = { lock().open_kind(cmdbuf) };
                match open {
                    Open::Guest => {
                        let handle = {
                            let mut host = lock();
                            match host.command(CommandProc::EndRecording, c) {
                                Outcome::Host(handle) => Some(handle),
                                Outcome::Driver => None,
                            }
                        };
                        let Some(handle) = handle else {
                            return false;
                        };
                        let memory = &c.process().memory;
                        // Hand the driver an empty list: it must not walk our records as its own command stream.
                        let field = {
                            let host = lock();
                            host.write_field
                                .get(&cmdbuf)
                                .copied()
                                .unwrap_or(record::header::WRITE)
                        };
                        let base = record::word(memory, cmdbuf, record::header::BASE);
                        let saved = record::word(memory, cmdbuf, field);
                        if let Some(base) = base {
                            record::set_word(memory, cmdbuf, field, base);
                        }
                        c.x = args;
                        crate::hle::call_guest(c, real, false);
                        if let Some(saved) = saved {
                            record::set_word(memory, cmdbuf, field, saved);
                        }
                        c.x[0] = handle;
                        c.pc = c.x[30];
                        return true;
                    }
                    Open::Arena | Open::None => {
                        // The driver ends its own list and returns the handle; the arena list is filed under it.
                        // Nothing in the driver's header is touched: its stream stays exactly as it wrote it.
                        if open == Open::None
                            && !{
                                let host = lock();
                                host.command_memory.contains_key(&cmdbuf)
                            }
                        {
                            return false;
                        }
                        crate::hle::call_guest(c, real, false);
                        let handle = c.x[0];
                        let memory = &c.process().memory;
                        {
                            let mut host = lock();
                            host.finish_implicit(memory, cmdbuf, handle, open == Open::Arena);
                        }
                        c.x = args;
                        c.x[0] = handle;
                        c.pc = c.x[30];
                        return true;
                    }
                }
            }
            Proc::Window(WindowProc::QueuePresent) => {
                return present(c, real, 1, 2);
            }
            Proc::Window(WindowProc::PresentTexture) => {
                return present(c, real, 0, 1);
            }
            _ => {}
        }
    }
    // The procs added for replay only observe, and only in record mode: `trace`, `shadow` and `encode` keep the
    // driver as their authority and must see exactly the calls they saw before.
    if !record::ON.load(Ordering::Relaxed)
        && matches!(
            proc,
            Proc::Program(_)
                | Proc::TexturePool
                | Proc::SamplerPool
                | Proc::Handle(_)
                | Proc::Window(_)
        )
    {
        return false;
    }
    let mut host = lock();
    if host.objects.deferred() >= DEFERRED_LIMIT {
        host.objects.drain();
    }
    match proc {
        Proc::PoolBuilderDefaults => {
            host.pool_builders.insert(c.x[0], PoolBuilder::default());
        }
        Proc::PoolBuilderStorage => {
            let builder = host.pool_builders.entry(c.x[0]).or_default();
            builder.storage = c.x[1];
            builder.size = c.x[2];
        }
        Proc::PoolBuilderFlags => {
            host.pool_builders.entry(c.x[0]).or_default().flags = c.x[1] as u32
        }
        Proc::PoolInitialize => host.pool_initialize(c.x[0], c.x[1]),
        Proc::PoolFinalize => host.pool_finalize(c.x[0]),
        Proc::PoolMapVirtual => host.pool_map_virtual(c),
        Proc::BufferBuilderDefaults => {
            host.buffer_builders
                .insert(c.x[0], BufferBuilder::default());
        }
        Proc::BufferBuilderStorage => {
            host.buffer_builders.insert(
                c.x[0],
                BufferBuilder {
                    pool: c.x[1],
                    offset: c.x[2],
                    size: c.x[3],
                },
            );
        }
        Proc::BufferInitialize => host.buffer_initialize(c.x[0], c.x[1]),
        Proc::BufferFinalize => {
            host.objects.destroy(c.x[0]);
        }
        Proc::Texture(field) => host.texture_builder(field, c),
        Proc::TextureInitialize => {
            host.texture_initialize(c.x[0], c.x[1], |address| read_u64(c, address))
        }
        Proc::TextureFinalize => {
            host.objects.destroy(c.x[0]);
        }
        Proc::View(field) => host.view(field, c),
        Proc::SamplerBuilder(field) => host.sampler_builder(field, c),
        Proc::SamplerInitialize => host.sampler_initialize(c.x[0], c.x[1]),
        Proc::SamplerFinalize => {
            host.objects.destroy(c.x[0]);
        }
        Proc::Program(program) => host.program(program, c),
        Proc::TexturePool | Proc::SamplerPool => host.descriptor_pool(proc, c),
        Proc::Handle(kind) => {
            c.x[0] = match kind {
                HandleProc::Texture => (c.x[1] & 0xfffff) | ((c.x[2] & 0xfff) << 20),
                HandleProc::TexelFetch => c.x[1] & 0xfffff,
                HandleProc::Image => (c.x[1] & 0xfffff) | ((c.x[2] & 0x3f) << 20),
            };
            return true;
        }
        Proc::State(setter) => host.state.set(setter, &c.x),
        Proc::Window(which) => host.window(which, c),
        Proc::Command(command) => {
            let outcome = host.command(command, c);
            if let Outcome::Host(value) = outcome {
                c.x[0] = value;
            }
            return outcome.is_host();
        }
    }
    false
}

/// `nvnQueuePresentTexture`/`nvnWindowPresentTexture`: run the driver's present, then put the replayed image on the
/// swapchain when the window names a texture the host recorded.
///
/// The driver call is never skipped. Its fences, its vsync clock and its native-window queue belong to it, and a
/// target the host cannot resolve has to stay there or the frame would be lost (`docs/nvn/CONTRACT.md` §8). After it
/// returns, the replay's submit for this frame has been queued, so waiting for its serial makes the swapchain order
/// equal the submit order. The frame counter is the same one `trace` uses: when the driver's own present did not
/// advance it, the host credits this present so `KIRIKAE_STATS` shows the frame in record mode too.
fn present(c: &mut Cpu, real: u64, window_arg: usize, index_arg: usize) -> bool {
    let args = c.x;
    let window = args[window_arg];
    let index = args[index_arg];
    let before = crate::display::FRAMES.load(Ordering::Relaxed);
    let target = { lock().present_target(window, index) };
    // The capture belongs to the replayed frame. Reserve its slot before the driver's own present runs, or that path
    // would capture the stale swapchain the replay is about to replace. Only a present the host can resolve reserves
    // it; an unresolved one leaves the capture to the driver, as before.
    let capture = crate::gpu::capture::capture()
        .filter(|capture| before.saturating_add(1) == capture.frame());
    let capture = match (&target, capture) {
        (Ok(_), Some(capture)) if !capture.done() => {
            capture.claim();
            Some(capture)
        }
        _ => None,
    };
    if real != 0 {
        crate::hle::call_guest(c, real, false);
    }
    let p = c.process();
    let mut credit = false;
    match target {
        Ok(target) => {
            replay::wait(target.serial);
            // The submit's snapshot owns the records it replayed, so the arena lists it captured go now instead of
            // waiting for the slot rotation.
            lock().retire_replayed(target.serial);
            let shown = p
                .gpu
                .with_renderer(|renderer| {
                    renderer.nvn_present(
                        target.cpu,
                        target.width,
                        target.height,
                        Some(target.crop),
                        before,
                        capture,
                    )
                })
                .unwrap_or(crate::gpu::vulkan::Presented::NoSwapchain);
            lock().note_present(window, target.serial, shown);
            credit = shown != crate::gpu::vulkan::Presented::NoImage;
        }
        Err(fallback) => {
            // A present this host cannot resolve still leaves nothing queued for long: the arena lists of the
            // submits it names are retired once their replay is done, as on the replayed path.
            let serial = { lock().last_submit };
            if serial != 0 {
                replay::wait(serial);
                lock().retire_replayed(serial);
            }
            lock().note_fallback(fallback);
        }
    }
    if credit {
        let after = crate::display::FRAMES.load(Ordering::Relaxed);
        if window::credit_frame(before, after) {
            crate::display::presented(p, None);
            lock().presents.frames += 1;
        }
    }
    c.pc = c.x[30];
    true
}

/// What `cmdbuf` is recording when `EndRecording` arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Open {
    /// A list the game bracketed with `BeginRecording`, recorded into its own command memory.
    Guest,
    /// A list the driver started on its own, recorded in the host arena.
    Arena,
    /// No recorded command since the last list ended; the driver still makes an (empty) one.
    None,
}

/// Whether the host recorded the proc, and the value the proc returns when it did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// Leave the proc on its existing tracing/real-driver path.
    Driver,
    /// The host took the proc over; the driver must not run and the proc returns this value.
    Host(u64),
}

impl Outcome {
    fn is_host(self) -> bool {
        matches!(self, Outcome::Host(_))
    }
}

/// Prints the host table when its shape changes, so a trace run shows the objects it built.
pub fn report() {
    let mut host = lock();
    let snapshot = Snapshot {
        counts: host.objects.counts(),
        failures: host.failures,
        record: host.record.report(),
        decoded: host.decoded,
        copied: host.copied,
        unresolved: host.unresolved,
        presents: host.presents,
    };
    if snapshot == host.last {
        return;
    }
    let Counts {
        pools,
        buffers,
        textures,
        views,
        samplers,
    } = snapshot.counts;
    eprintln!(
        "kirikae:         nvn host: {pools} pools, {buffers} buffers, {textures} textures, {views} views, {samplers} samplers; {} deferred, {} rejected",
        host.objects.deferred(),
        snapshot.failures
    );
    let record = snapshot.record;
    if record.records > 0
        || record.lists > 0
        || record.implicit > 0
        || snapshot.decoded + snapshot.copied > 0
    {
        eprintln!(
            "kirikae:         nvn record: {} lists live, {} recorded ({} implicit, in the host arena), {} records, {} bytes ({} blob bytes); {} chained links live, {} retired after replay; {} submits ({} stayed with the driver), {} rejected; {} draws and {} clears recorded; {} binds decoded, {} structs copied, {} unresolved",
            record.live,
            record.lists,
            record.implicit,
            record.records,
            record.bytes + record.blob_bytes,
            record.blob_bytes,
            record.chained,
            record.retired,
            record.submits,
            host.submits_driver,
            record.rejected,
            record.recorded_draws,
            record.recorded_clears,
            snapshot.decoded,
            snapshot.copied,
            snapshot.unresolved
        );
    }
    let presents = snapshot.presents;
    if presents.replayed + presents.fallback + presents.no_swapchain > 0 {
        eprintln!(
            "kirikae:         nvn present: {} windows, {} presents ({} from the replay, {} composed from the last target, {} stayed with the driver, {} without a swapchain); {} frames counted here",
            host.windows.len(),
            presents.replayed + presents.fallback + presents.no_swapchain,
            presents.replayed,
            window::COMPOSED.load(Ordering::Relaxed),
            presents.fallback,
            presents.no_swapchain,
            presents.frames
        );
    }
    host.last = snapshot;
}

impl Host {
    /// A window builder, window or their per-frame tuning. All of it is read by value at the call; the driver still
    /// runs every one of these procs, because the window struct it allocated is its own.
    fn window(&mut self, which: WindowProc, c: &Cpu) {
        use WindowProc::*;
        let address = c.x[0];
        match which {
            BuilderDefaults => {
                self.window_builders
                    .insert(address, window::Builder::default());
            }
            BuilderTextures => {
                let count = c.x[1].min(window::MAX_TEXTURES as u64);
                let textures = read_words(&c.process().memory, c.x[2], count);
                if textures.len() != count as usize {
                    self.fail(format!(
                        "nvnWindowBuilderSetTextures: {count} texture pointers at {:#x} are unreadable",
                        c.x[2]
                    ));
                    return;
                }
                self.window_builders
                    .entry(address)
                    .or_default()
                    .set_textures(&textures);
            }
            BuilderActiveTextures => {
                self.window_builders.entry(address).or_default().active = c.x[1] as u32;
            }
            BuilderPresentInterval => {
                self.window_builders
                    .entry(address)
                    .or_default()
                    .present_interval = c.x[1] as u32;
            }
            Initialize => {
                // `(window, builder)`: the textures live on the builder, not the window address.
                let builder = self
                    .window_builders
                    .get(&c.x[1])
                    .copied()
                    .unwrap_or_default();
                self.windows.insert(address, Window::from_builder(&builder));
            }
            Finalize => {
                self.windows.remove(&address);
                self.presented.remove(&address);
            }
            SetCrop => {
                if let Some(window) = self.windows.get_mut(&address) {
                    window.crop = Some(window::Crop {
                        x: c.x[1] as u32 as i32,
                        y: c.x[2] as u32 as i32,
                        width: c.x[3] as u32,
                        height: c.x[4] as u32,
                    });
                }
            }
            SetPresentInterval => {
                if let Some(window) = self.windows.get_mut(&address) {
                    window.present_interval = c.x[1] as u32;
                }
            }
            SetNumActiveTextures => {
                if let Some(window) = self.windows.get_mut(&address) {
                    window.active = c.x[1] as u32;
                }
            }
            QueuePresent | PresentTexture => {}
        }
    }

    /// The guest textures every live window names, once each, in window order. The replay needs them even when the
    /// title never draws into the window: the present composes the frame from the last drawn target.
    fn window_textures(&self) -> Vec<u64> {
        let mut textures = Vec::new();
        for window in self.windows.values() {
            for &texture in &window.textures {
                if texture != 0
                    && !textures.contains(&texture)
                    && matches!(self.objects.get(texture), Some(Object::Texture(_)))
                {
                    textures.push(texture);
                }
            }
        }
        textures
    }

    /// The window texture a present names, as the renderer needs it. This only reads the host tables; the caller
    /// waits for the replay and takes the renderer lock after dropping it.
    fn present_target(&self, address: u64, index: u64) -> Result<window::Target, window::Fallback> {
        let Some(window) = self.windows.get(&address) else {
            return Err(window::Fallback::UnknownWindow);
        };
        let Some(texture) = window.texture(index) else {
            return Err(window::Fallback::IndexOutsideWindow);
        };
        let Some(texture) = self.objects.get(texture).and_then(|object| match object {
            Object::Texture(texture) => Some(texture),
            _ => None,
        }) else {
            return Err(window::Fallback::UnknownTexture);
        };
        if texture.host == 0 {
            return Err(window::Fallback::NoHostAddress);
        }
        if self.last_submit == 0
            || self
                .presented
                .get(&address)
                .is_some_and(|&serial| serial >= self.last_submit)
        {
            return Err(window::Fallback::NoReplayedSubmit);
        }
        Ok(window::Target {
            cpu: texture.host,
            format: texture.format,
            width: texture.width,
            height: texture.height,
            crop: window.crop_rect(texture.width, texture.height),
            serial: self.last_submit,
        })
    }

    fn note_present(&mut self, window: u64, serial: u64, shown: crate::gpu::vulkan::Presented) {
        use crate::gpu::vulkan::Presented;
        match shown {
            Presented::Shown => {
                self.presents.replayed += 1;
                self.presented.insert(window, serial);
            }
            Presented::NoSwapchain => {
                self.presents.no_swapchain += 1;
                self.presented.insert(window, serial);
            }
            Presented::NoImage => {
                self.presents.fallback += 1;
                self.fail_keyed(
                    "present-no-image",
                    "nvn present: the replay has no image for the window texture; the driver keeps this frame"
                        .to_string(),
                );
            }
        }
    }

    fn note_fallback(&mut self, fallback: window::Fallback) {
        self.presents.fallback += 1;
        let key = match fallback {
            window::Fallback::UnknownWindow => "present-window",
            window::Fallback::IndexOutsideWindow => "present-index",
            window::Fallback::UnknownTexture => "present-texture",
            window::Fallback::NoHostAddress => "present-address",
            window::Fallback::NoReplayedSubmit => "present-submit",
        };
        self.fail_keyed(
            key,
            format!(
                "nvn present: {}; the driver keeps this frame",
                fallback.reason()
            ),
        );
    }

    fn fail(&mut self, what: String) {
        self.fail_keyed(&what.clone(), what);
    }

    /// Counts one distinct cause, not one occurrence: a handle the game calls every frame is still one
    /// unresolvable object.
    fn fail_keyed(&mut self, key: &str, what: String) {
        if !self.seen_failures.insert(key.to_string()) {
            return;
        }
        self.failures += 1;
        if self.seen_failures.len() <= 32 {
            eprintln!("kirikae: nvn host: {what}");
        }
    }

    fn pool_initialize(&mut self, address: u64, builder: u64) {
        let builder = self
            .pool_builders
            .get(&builder)
            .copied()
            .unwrap_or_default();
        if let Some(old) = self.objects.pool(address)
            && old.gpu != 0
        {
            self.ranges.unregister(old.gpu);
        }
        let mut gpu = match self.va.allocate(builder.size) {
            Ok(gpu) => gpu,
            Err(error) => {
                self.fail(format!(
                    "memory pool {address:#x}: {} bytes: {error:?}",
                    builder.size
                ));
                0
            }
        };
        if gpu != 0
            && let Err(error) = self.ranges.register(gpu, builder.size, address)
        {
            self.fail(format!(
                "memory pool {address:#x}: GPU range {gpu:#x}+{:#x}: {error:?}",
                builder.size
            ));
            gpu = 0;
        }
        let pool = Pool {
            storage: builder.storage,
            size: builder.size,
            flags: builder.flags,
            gpu,
            ..Pool::default()
        };
        self.objects.make(address, Object::Pool(pool));
    }

    fn pool_finalize(&mut self, address: u64) {
        if let Some(pool) = self.objects.pool(address)
            && pool.gpu != 0
        {
            self.ranges.unregister(pool.gpu);
        }
        self.objects.destroy(address);
        // A command memory range into this pool would resolve again if a new pool object lands on the same address,
        // and the recorder would write its records into whatever that memory became. Forget the ranges instead.
        let mut dropped = Vec::new();
        for (cmdbuf, ranges) in self.command_memory.iter_mut() {
            if ranges.iter().any(|range| range.pool == address) {
                dropped.push(*cmdbuf);
            }
            ranges.retain(|range| range.pool != address);
        }
        self.command_memory.retain(|_, ranges| !ranges.is_empty());
        for cmdbuf in dropped {
            self.record.initialize(cmdbuf);
            self.write_field.remove(&cmdbuf);
        }
    }

    fn pool_map_virtual(&mut self, c: &Cpu) {
        let address = c.x[0];
        let count = c.x[1].min(4096);
        let requests = c.x[2];
        let Some(mut pool) = self.objects.pool(address) else {
            self.fail(format!(
                "nvnMemoryPoolMapVirtual: pool {address:#x} is not in the host table"
            ));
            return;
        };
        if !pool.is_virtual() {
            self.fail(format!(
                "nvnMemoryPoolMapVirtual: pool {address:#x} owns its storage"
            ));
            return;
        }
        let mut changed = false;
        for index in 0..count {
            let Some(bytes) = c
                .process()
                .memory
                .read(requests.wrapping_add(0x28 * index), 0x28)
            else {
                self.fail(format!(
                    "nvnMemoryPoolMapVirtual: request {index} at {requests:#x} is unreadable"
                ));
                continue;
            };
            let physical = word(&bytes, 0);
            let physical_offset = word(&bytes, 8);
            let virtual_offset = word(&bytes, 16);
            let size = word(&bytes, 24);
            if physical == 0 {
                changed |= pool.unmap_virtual(virtual_offset);
                continue;
            }
            let Some(physical_pool) = self.objects.pool(physical) else {
                self.fail(format!(
                    "nvnMemoryPoolMapVirtual: physical pool {physical:#x} is not in the host table"
                ));
                continue;
            };
            if physical_pool.is_virtual() {
                self.fail(format!(
                    "nvnMemoryPoolMapVirtual: physical pool {physical:#x} owns no storage"
                ));
                continue;
            }
            let mapping = Mapping {
                physical,
                physical_offset,
                size,
            };
            match physical_pool.contains(physical_offset, size).and_then(|()| pool.map_virtual(virtual_offset, mapping)) {
                Ok(()) => changed = true,
                Err(error) => self.fail(format!("nvnMemoryPoolMapVirtual: {virtual_offset:#x}+{size:#x} -> {physical:#x}+{physical_offset:#x}: {error:?}")),
            }
        }
        if changed {
            self.objects.make(address, Object::Pool(pool));
        }
    }

    fn buffer_initialize(&mut self, address: u64, builder: u64) {
        let builder = self
            .buffer_builders
            .get(&builder)
            .copied()
            .unwrap_or_default();
        let Some(pool) = self.objects.pool(builder.pool) else {
            self.fail(format!(
                "buffer {address:#x}: memory pool {:#x} is not in the host table",
                builder.pool
            ));
            return;
        };
        let host = match self
            .objects
            .pool_host(builder.pool, builder.offset, builder.size)
        {
            Ok(host) => host,
            Err(error) => {
                self.fail(format!(
                    "buffer {address:#x}: {:#x}+{:#x} in pool {:#x}: {error:?}",
                    builder.offset, builder.size, builder.pool
                ));
                return;
            }
        };
        match Buffer::create(builder.pool, &pool, builder.offset, builder.size, host) {
            Ok(buffer) => {
                self.objects.make(address, Object::Buffer(buffer));
            }
            Err(error) => self.fail(format!("buffer {address:#x}: {error:?}")),
        }
    }

    fn texture_builder(&mut self, field: TextureProc, c: &Cpu) {
        use TextureProc::*;
        let address = c.x[0];
        if field == Defaults {
            self.texture_builders
                .insert(address, TextureBuilder::default());
            return;
        }
        let builder = self.texture_builders.entry(address).or_default();
        let value = |index: usize| c.x.get(index).copied().unwrap_or(0);
        match field {
            Defaults => {}
            Flags => builder.flags = value(1) as u32,
            Target => builder.target = value(1) as u32,
            Width => builder.width = value(1) as u32,
            Height => builder.height = value(1) as u32,
            Depth => builder.depth = value(1) as u32,
            Size1D => (builder.width, builder.height, builder.depth) = (value(1) as u32, 1, 1),
            Size2D => {
                (builder.width, builder.height, builder.depth) =
                    (value(1) as u32, value(2) as u32, 1)
            }
            Size3D => {
                (builder.width, builder.height, builder.depth) =
                    (value(1) as u32, value(2) as u32, value(3) as u32)
            }
            Levels => builder.levels = value(1) as u32,
            Format => builder.format = value(1) as u32,
            Samples => builder.samples = value(1) as u32,
            Stride => builder.stride = value(1),
            Storage => (builder.pool, builder.offset) = (value(1), value(2)),
            PackagedLayout => builder.packaged = value(1),
            Swizzle => {
                builder.swizzle = [
                    value(1) as u8,
                    value(2) as u8,
                    value(3) as u8,
                    value(4) as u8,
                ];
            }
        }
    }

    fn texture_initialize(
        &mut self,
        address: u64,
        builder_address: u64,
        read: impl Fn(u64) -> Option<u64>,
    ) {
        let mut builder = self
            .texture_builders
            .get(&builder_address)
            .copied()
            .unwrap_or_default();
        builder.packaged = match builder.packaged {
            0 => 0,
            raw => read(raw).unwrap_or(raw),
        };
        let layout = match builder.layout() {
            Ok(layout) => layout,
            Err(error) => {
                self.fail(format!(
                    "texture {address:#x}: builder {builder_address:#x}: {error:?}"
                ));
                return;
            }
        };
        let Some(pool) = self.objects.pool(builder.pool) else {
            self.fail(format!(
                "texture {address:#x}: memory pool {:#x} is not in the host table",
                builder.pool
            ));
            return;
        };
        let host = match self
            .objects
            .pool_host(builder.pool, builder.offset, layout.size)
        {
            Ok(host) => host,
            Err(error) => {
                let mappings = if pool.is_virtual() {
                    format!(", {} virtual mappings", pool.mappings.len())
                } else {
                    String::new()
                };
                self.fail(format!(
                    "texture {address:#x}: {:#x}+{:#x} in pool {:#x} ({} bytes{mappings}): {error:?}; it stays out of the host table",
                    builder.offset, layout.size, builder.pool, pool.size
                ));
                return;
            }
        };
        match Texture::create(builder.pool, &pool, &builder, host) {
            Ok(texture) => {
                self.objects.make(address, Object::Texture(texture));
            }
            Err(error) => self.fail(format!("texture {address:#x}: {error:?}")),
        }
    }

    fn view(&mut self, field: ViewProc, c: &Cpu) {
        use ViewProc::*;
        let address = c.x[0];
        let mut view = match self.objects.get(address) {
            Some(Object::View(view)) => view,
            _ => TextureView::default(),
        };
        let value = |index: usize| c.x.get(index).copied().unwrap_or(0) as u32;
        match field {
            Defaults => view = TextureView::default(),
            Levels => (view.base_level, view.levels) = (value(1), value(2)),
            Layers => (view.min_layer, view.layers) = (value(1), value(2)),
            Format => view.format = value(1),
            Target => view.target = value(1),
        }
        self.objects.make(address, Object::View(view));
    }

    fn sampler_builder(&mut self, field: samplers::Field, c: &Cpu) {
        let address = c.x[0];
        let mut sampler = self
            .sampler_builders
            .get(&address)
            .copied()
            .unwrap_or_default();
        let floats: [f32; 4] = std::array::from_fn(|index| f32::from_bits(c.v[index][0] as u32));
        let border = if field == samplers::Field::BorderColor {
            read_border(c, c.x[1])
        } else {
            [0.0; 4]
        };
        sampler.apply(field, &c.x, &floats, border);
        self.sampler_builders.insert(address, sampler);
    }

    fn sampler_initialize(&mut self, address: u64, builder: u64) {
        let sampler = self
            .sampler_builders
            .get(&builder)
            .copied()
            .unwrap_or_default();
        self.objects.make(address, Object::Sampler(sampler));
    }

    /// A program object and the stages `nvnProgramSetShaders` handed in. The host keeps the stage code addresses
    /// and the control section's constant-data range; the translation happens on the first draw that binds it.
    fn program(&mut self, which: ProgramProc, c: &Cpu) {
        use ProgramProc::*;
        match which {
            Initialize => {
                self.programs.insert(c.x[0], replay::Program::default());
            }
            Finalize => {
                self.programs.remove(&c.x[0]);
            }
            Shaders | ShadersExt => {
                let stride = if which == ShadersExt { 0x18 } else { 0x10 };
                let count = c.x[1].min(8);
                let mut program = replay::Program::default();
                for index in 0..count {
                    let entry = c.x[2].wrapping_add(index * stride);
                    let Some(bytes) = c.process().memory.read(entry, 16) else {
                        self.fail(format!(
                            "nvnProgramSetShaders: stage {index} entry at {entry:#x} is unreadable"
                        ));
                        continue;
                    };
                    let code = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or_default());
                    let control = u64::from_le_bytes(bytes[8..16].try_into().unwrap_or_default());
                    match stage(c, code, control) {
                        Some(stage) => program.stages.push(stage),
                        None => self.fail(format!(
                            "program {:#x}: stage {index} code {code:#x} control {control:#x} has no readable control section",
                            c.x[0]
                        )),
                    }
                }
                self.programs.insert(c.x[0], program);
            }
        }
    }

    /// `nvnTexturePoolInitialize`/`nvnSamplerPoolInitialize`: `(pool, memory pool, offset, count)`. The driver writes
    /// the descriptors into that memory, so the replay only needs the range's guest address.
    fn descriptor_pool(&mut self, which: Proc, c: &Cpu) {
        let entry = DescriptorPool {
            pool: c.x[1],
            offset: c.x[2],
            count: c.x[3].min(u32::MAX as u64) as u32,
        };
        match which {
            Proc::TexturePool => self.texture_pools.insert(c.x[0], entry),
            _ => self.sampler_pools.insert(c.x[0], entry),
        };
    }

    /// Builds the replay job for `nvnQueueSubmitCommands`: snapshots every list it can reach, resolves the render
    /// targets' textures and the pools, and queues it. A handle the host never recorded is reported and left out.
    fn submit_lists(
        &mut self,
        mem: &dyn record::Memory,
        handles: &[u64],
        p: &'static crate::process::Process,
    ) -> u64 {
        let mut submit = replay::Submit {
            mark: self.record.mark(),
            ..replay::Submit::default()
        };
        let mut roots: Vec<u64> = Vec::new();
        for &handle in handles {
            if self.record.list(handle).is_some() && !roots.contains(&handle) {
                roots.push(handle);
            } else if self.record.list(handle).is_none() {
                self.fail_keyed(
                    "submit-handle",
                    format!(
                        "nvnQueueSubmitCommands: handle {handle:#x} was not recorded by the host"
                    ),
                );
            }
        }
        let mut textures = std::collections::BTreeSet::new();
        let mut views = std::collections::BTreeSet::new();
        let mut programs = std::collections::BTreeSet::new();
        let mut texture_pool = 0;
        let mut sampler_pool = 0;
        let mut seen = std::collections::HashSet::new();
        let mut want = roots.clone();
        let mut arena = Vec::new();
        let mut snapshots: std::collections::HashMap<u64, std::sync::Arc<replay::ListSnapshot>> =
            std::collections::HashMap::new();
        while let Some(handle) = want.pop() {
            if !seen.insert(handle) {
                continue;
            }
            let Some(list) = self.record.list(handle).cloned() else {
                continue;
            };
            if list.arena {
                arena.push(handle);
            }
            let Some(snapshot) =
                replay::ListSnapshot::capture(mem, &self.record, handle, list.bytes, list.cmdbuf)
            else {
                self.fail_keyed(
                    "submit-list-unmapped",
                    format!("nvnQueueSubmitCommands: list {handle:#x} is unreadable"),
                );
                continue;
            };
            // A list that continues a chunk the driver ended under its internal pointer needs that chunk in the
            // snapshot too: the walker replays it before this list's records (`record::List::chained`).
            want.extend(list.chained.iter().copied());
            let bytes = snapshot.bytes.clone();
            let mut callers = Vec::new();
            for chunk in bytes.as_chunks::<{ record::RECORD_SIZE as usize }>().0 {
                let Some(record) = record::Record::decode(chunk) else {
                    continue;
                };
                match record.op {
                    record::Op::CallCommands if record.flags & record::FLAG_INLINE == 0 => {
                        if let Some(blob) = snapshot.blob(&record) {
                            for word in blob.as_chunks::<8>().0 {
                                callers.push(u64::from_le_bytes(*word));
                            }
                        }
                    }
                    record::Op::SetRenderTargets => {
                        if let Some(blob) = snapshot.blob(&record) {
                            let words: Vec<u64> = blob
                                .as_chunks::<8>()
                                .0
                                .iter()
                                .map(|w| u64::from_le_bytes(*w))
                                .collect();
                            let count = record.count.min(8) as usize;
                            let mut pair = |index: usize| {
                                if let (Some(&texture), Some(&view)) =
                                    (words.get(index), words.get(index + 1))
                                {
                                    if texture != 0 {
                                        textures.insert(texture);
                                    }
                                    if view != 0 {
                                        views.insert(view);
                                    }
                                }
                            };
                            for index in 0..count {
                                pair(2 * index);
                            }
                            pair(2 * count);
                        }
                    }
                    record::Op::BindProgram => {
                        programs.insert(record.words[0]);
                    }
                    record::Op::SetTexturePool => texture_pool = record.words[0],
                    record::Op::SetSamplerPool => sampler_pool = record.words[0],
                    _ => {}
                }
            }
            want.extend(callers);
            snapshots.insert(handle, std::sync::Arc::new(snapshot));
        }
        for handle in &roots {
            if let Some(snapshot) = snapshots.get(handle) {
                submit.roots.push(std::sync::Arc::clone(snapshot));
            }
        }
        submit.callees = snapshots;
        // Render-target textures and views are resolved now, at the submit's mark.
        for address in textures {
            match self.objects.get(address) {
                Some(Object::Texture(texture)) => {
                    submit.textures.insert(address, texture_info(&texture));
                }
                other => self.fail_keyed(
                    "submit-target",
                    format!(
                        "nvnQueueSubmitCommands: render target {address:#x} is {}",
                        other.as_ref().map_or("not in the host table", Object::kind)
                    ),
                ),
            }
        }
        // A window texture is where the title composes the presented frame. It may never be a recorded draw target
        // on this title (the driver performs the composition), so resolve it even when the records only set it as a
        // target and never draw into it (`docs/nvn/CONTRACT.md` §8).
        for address in self.window_textures() {
            if let Some(Object::Texture(texture)) = self.objects.get(address) {
                submit.windows.insert(address);
                submit
                    .textures
                    .entry(address)
                    .or_insert_with(|| texture_info(&texture));
            }
        }
        for address in views {
            if let Some(Object::View(view)) = self.objects.get(address) {
                submit.views.insert(
                    address,
                    replay::ViewInfo {
                        base_level: view.base_level,
                        levels: view.levels,
                        min_layer: view.min_layer,
                        layers: view.layers,
                        format: view.format,
                    },
                );
            }
        }
        for address in programs {
            if let Some(program) = self.programs.get(&address) {
                submit.programs.insert(address, program.clone());
            }
        }
        submit.texture_pool = self
            .texture_pools
            .get(&texture_pool)
            .and_then(|pool| self.descriptor_table(pool))
            .unwrap_or(0);
        submit.sampler_pool = self
            .sampler_pools
            .get(&sampler_pool)
            .and_then(|pool| self.descriptor_table(pool))
            .unwrap_or(0);
        let serial = replay::submit(p, submit);
        if serial != 0 && !arena.is_empty() {
            self.retiring.push_back((serial, arena));
        }
        serial
    }

    /// Guest address of a descriptor pool's table. The driver writes TIC/TSC entries into the memory pool the game
    /// handed `nvnTexturePoolInitialize`, so the replay reads them at their guest address: the pool's GPU address is
    /// ours (`va::Allocator`), and it need not coincide with the mapping the driver gave the same memory in nvdrv.
    fn descriptor_table(&self, descriptor: &DescriptorPool) -> Option<u64> {
        let size = u64::from(descriptor.count).max(1).checked_mul(32)?;
        self.objects
            .pool_host(descriptor.pool, descriptor.offset, size)
            .ok()
    }

    /// Records a command buffer proc. `Outcome::Driver` keeps the existing path, so a run without
    /// `KIRIKAE_NVN=record` never writes guest memory.
    fn command(&mut self, command: CommandProc, c: &mut Cpu) -> Outcome {
        if !record::ON.load(Ordering::Relaxed) {
            return Outcome::Driver;
        }
        self.process = Some(c.process());
        let memory = &c.process().memory;
        match self.recording(command, memory, &c.x, &c.v) {
            Ok(Outcome::Host(value)) => {
                c.x[0] = value;
                Outcome::Host(value)
            }
            Ok(outcome) => outcome,
            Err(error) => {
                // One cause per kind: a list that hits the end of its command memory is one problem, not one per
                // command name.
                let key = match error {
                    Error::Full => "record-full",
                    Error::Unmapped => "record-unmapped",
                    Error::WritePointerOutsideMemory => "record-write-pointer",
                    Error::NoCommandMemory => "record-no-memory",
                    Error::Corrupt => "record-corrupt",
                    _ => "record-other",
                };
                self.fail_keyed(key, format!("nvn record: {command:?}: {error:?}"));
                // An implicit list that broke is dropped: the driver still runs the call and the whole submit
                // stays on its path, which is reported. Half a list must never be replayed as a whole one.
                if self.record.is_arena(c.x[0]) || self.implicit_dead.contains(&c.x[0]) {
                    self.record.abort(c.x[0]);
                    self.implicit_dead.insert(c.x[0]);
                    return Outcome::Driver;
                }
                // A list the host opened must not fall back to the driver: the driver would write its own
                // pushbuffer over the records we already stored.
                if self.record.is_recording(c.x[0]) {
                    Outcome::Host(0)
                } else {
                    Outcome::Driver
                }
            }
        }
    }

    fn recording(
        &mut self,
        command: CommandProc,
        mem: &dyn record::Memory,
        a: &[u64; 31],
        v: &[[u64; 2]; 32],
    ) -> Result<Outcome, Error> {
        use CommandProc::*;
        let cmdbuf = a[0];
        let before = self.record.usage();
        match command {
            Initialize | Finalize => {
                self.record.initialize(cmdbuf);
                self.command_memory.remove(&cmdbuf);
                self.write_field.remove(&cmdbuf);
                self.implicit_dead.remove(&cmdbuf);
                replay::reset_state(cmdbuf);
                // Chunks waiting for a handle stay in `pending` (the driver may still report one on this buffer),
                // but they release their arena slot: a buffer that is set up again or torn down must not pin chunks
                // forever. A recycled slot is caught by the generation check when the handle finally arrives.
                if let Some(pending) = self.pending.get(&cmdbuf).cloned() {
                    for (key, _) in pending {
                        self.record.unpin(key);
                    }
                }

                // The driver owns the header and its timestamp/callback state; we only place records through it.
                return Ok(Outcome::Driver);
            }
            AddCommandMemory => {
                self.add_command_memory(a)?;
                return Ok(Outcome::Driver);
            }
            SubmitCommands => {
                let handles = read_words(mem, a[2], a[1]);
                self.record.submit(&handles);
                // A submit that names any list the host did not record stays with the driver: its pushbuffer is a
                // valid command stream for the GPU path, and replaying the host's records in its place would drop
                // whatever the driver recorded (this title's per-frame lists are all like that). Mixing both in one
                // submit would interleave two streams, so the whole call goes to the driver and is reported.
                let unknown = handles
                    .iter()
                    .filter(|handle| self.record.list(**handle).is_none())
                    .count();
                if unknown > 0 || handles.is_empty() {
                    self.submits_driver += 1;
                    self.fail_keyed(
                        "submit-driver",
                        format!(
                            "nvnQueueSubmitCommands: {unknown} of {} lists were driver-recorded; the driver submits this call",
                            handles.len()
                        ),
                    );
                    return Ok(Outcome::Driver);
                }
                if let Some(p) = self.process {
                    self.last_submit = self.submit_lists(mem, &handles, p);
                }
                return Ok(Outcome::Host(0));
            }
            GetCommandMemorySize | GetCommandMemoryUsed | GetCommandMemoryFree => {
                let Some((size, used, free)) = self.command_usage(mem, cmdbuf) else {
                    return Ok(Outcome::Driver);
                };
                let value = match command {
                    GetCommandMemorySize => size,
                    GetCommandMemoryUsed => used,
                    _ => free,
                };
                return Ok(Outcome::Host(value));
            }
            _ => {}
        }
        // A list the game brackets with `BeginRecording` is recorded into its command memory. A list the driver
        // starts on its own after `EndRecording` (this title's per-frame draw lists) is recorded into the host arena
        // instead, so the driver keeps its pushbuffer and its header to itself: two writers in one command memory
        // corrupt both streams. The driver still runs every one of those commands, because its stream is its own.
        let mut implicit = self.record.is_arena(cmdbuf);
        if self.record.is_recording(cmdbuf) {
            if command == BeginRecording && implicit {
                // The game begins a list while an implicit one is open: the driver starts a new list too, so the
                // implicit capture ends here (its handle is never returned to the game) and the rest is recorded
                // into the command memory like any bracketed list.
                self.record.abort(cmdbuf);
                self.open_list(mem, cmdbuf)?;
                implicit = false;
            }
        } else {
            match command {
                BeginRecording if self.command_memory.contains_key(&cmdbuf) => {
                    self.open_list(mem, cmdbuf)?
                }
                _ if self.command_memory.contains_key(&cmdbuf)
                    && !self.implicit_dead.contains(&cmdbuf) =>
                {
                    self.open_implicit(mem, cmdbuf)?;
                    implicit = true;
                }
                _ => return Ok(Outcome::Driver),
            }
        }
        let value = match command {
            Initialize | Finalize | AddCommandMemory | SubmitCommands => 0,
            GetCommandMemorySize | GetCommandMemoryUsed | GetCommandMemoryFree => 0,
            BeginRecording => 0,
            EndRecording => self.record.end(mem, cmdbuf)?,
            SetRenderTargets => {
                self.set_render_targets(mem, a)?;
                0
            }
            ClearColor => {
                self.clear_color(mem, a)?;
                0
            }
            ClearDepthStencil => {
                self.clear_depth_stencil(mem, a, v)?;
                0
            }
            DiscardColor => {
                let mut record = record::Record::new(record::Op::DiscardColor);
                record.set_int(0, a[1] as u32 as i32);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            BindProgram => {
                let mut record = record::Record::new(record::Op::BindProgram);
                record.set_word(0, a[1]);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            BindBlendState => {
                let values = self.state.blend(a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindBlendState, a[1], values)?;
                0
            }
            BindColorState => {
                let values = self.state.color(a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindColorState, a[1], values)?;
                0
            }
            BindChannelMaskState => {
                let values = self.state.channel_mask(a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindChannelMaskState, a[1], values)?;
                0
            }
            BindDepthStencilState => {
                let values = self.state.depth_stencil(a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindDepthStencilState, a[1], values)?;
                0
            }
            BindPolygonState => {
                let values = self.state.polygon(a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindPolygonState, a[1], values)?;
                0
            }
            BindVertexAttribState => {
                let values = self.state.attribs(a[2], a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindVertexAttribState, a[2], values)?;
                0
            }
            BindVertexStreamState => {
                let values = self.state.streams(a[2], a[1]);
                self.bind_state(mem, cmdbuf, record::Op::BindVertexStreamState, a[2], values)?;
                0
            }
            BindTexture => {
                let mut record = record::Record::new(record::Op::BindTexture);
                let handle = a[3];
                record.set_word(0, handle);
                record.set_int(0, a[1] as u32 as i32);
                record.set_int(1, a[2] as u32 as i32);
                // The split (20-bit texture id, 12-bit sampler id) is a decode rule, not an API guarantee
                // (`docs/nvn/CONTRACT.md` §2); keep both halves next to the raw handle.
                record.set_int(2, (handle & 0xfffff) as i32);
                record.set_int(3, ((handle >> 20) & 0xfff) as i32);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            BindUniformBuffer => {
                let mut record = record::Record::new(record::Op::BindUniformBuffer);
                record.set_word(0, a[3]);
                record.set_word(1, a[4]);
                record.set_int(0, a[1] as u32 as i32);
                record.set_int(1, a[2] as u32 as i32);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            BindStorageBuffer => {
                // `nvnCommandBufferBindStorageBuffer(cmd, stage, index, GPU address, size)`.
                let mut record = record::Record::new(record::Op::BindStorageBuffer);
                record.set_word(0, a[3]);
                record.set_word(1, a[4]);
                record.set_int(0, a[1] as u32 as i32);
                record.set_int(1, a[2] as u32 as i32);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            SetSamplerPool => {
                let mut record = record::Record::new(record::Op::SetSamplerPool);
                record.set_word(0, a[1]);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            SetTexturePool => {
                let mut record = record::Record::new(record::Op::SetTexturePool);
                record.set_word(0, a[1]);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            BindVertexBuffer => {
                // `nvnCommandBufferBindVertexBuffer(cmd, index, address, size)`.
                let mut record = record::Record::new(record::Op::BindVertexBuffer);
                record.set_int(0, a[1] as u32 as i32);
                record.set_word(0, a[2]);
                record.set_word(1, a[3]);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            SetViewport => {
                // `nvnCommandBufferSetViewport(cmd, float x, float y, float w, float h, float near, float far)`.
                let mut record = record::Record::new(record::Op::SetViewport);
                for (index, slot) in record.floats.iter_mut().enumerate() {
                    *slot = f32::from_bits(v.get(index).map_or(0, |r| r[0]) as u32);
                }
                record.set_word(0, v.get(4).map_or(0, |r| r[0]));
                record.set_word(1, v.get(5).map_or(0, |r| r[0]));
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            DrawArrays | DrawArraysInstanced => {
                let instanced = command == DrawArraysInstanced;
                let mut record = record::Record::new(record::Op::DrawArrays);
                record.set_int(0, a[1] as u32 as i32);
                record.set_word(0, a[2]);
                record.set_word(1, a[3]);
                record.set_int(1, if instanced { a[4] as u32 as i32 } else { 0 });
                record.set_int(2, if instanced { a[5] as u32 as i32 } else { 1 });
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            DrawElements | DrawElementsBaseVertex | DrawElementsInstanced => {
                let base_vertex = matches!(command, DrawElementsBaseVertex | DrawElementsInstanced);
                let instanced = command == DrawElementsInstanced;
                let mut record = record::Record::new(record::Op::DrawElements);
                record.set_int(0, a[1] as u32 as i32);
                record.set_int(1, a[2] as u32 as i32);
                record.set_word(0, a[4]);
                record.set_word(1, a[3]);
                record.set_int(2, if base_vertex { a[5] as u32 as i32 } else { 0 });
                record.set_int(3, if instanced { a[6] as u32 as i32 } else { 0 });
                record.set_word(2, if instanced { a[7] } else { 1 });
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            DispatchCompute => {
                let mut record = record::Record::new(record::Op::DispatchCompute);
                record.set_word(0, a[1]);
                record.set_word(1, a[2]);
                record.set_word(2, a[3]);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            Barrier => {
                let mut record = record::Record::new(record::Op::Barrier);
                record.set_word(0, a[1]);
                self.record.emit(mem, cmdbuf, record)?;
                0
            }
            CallCommands => {
                let handles = read_words(mem, a[2], a[1]);
                self.call_known(mem, cmdbuf, &handles, "CallCommands")?;
                0
            }
            CopyCommands => {
                let handles = read_words(mem, a[2], a[1]);
                self.call_known(mem, cmdbuf, &handles, "CopyCommands")?;
                0
            }
        };
        // The game's own accounting reads this counter, so move it by everything the command consumed. An implicit
        // list wrote nothing to that memory (the driver is its only writer), so its counter is none of our business.
        if implicit {
            return Ok(Outcome::Driver);
        }
        let after = self.record.usage();
        let written =
            (after.bytes + after.blob_bytes).saturating_sub(before.bytes + before.blob_bytes);
        let field = self
            .write_field
            .get(&cmdbuf)
            .copied()
            .unwrap_or(record::header::WRITE);
        if written > 0 && !record::advance(mem, cmdbuf, field, written) {
            return Err(Error::Unmapped);
        }
        Ok(Outcome::Host(value))
    }

    /// Opens a list at the write pointer the header names. Called by `BeginRecording` and by the first recorded
    /// command of a buffer the driver records without one.
    fn open_list(&mut self, mem: &dyn record::Memory, cmdbuf: u64) -> Result<(), Error> {
        let spans = self.command_spans(cmdbuf)?;
        let placement = match record::placement(mem, cmdbuf, &spans) {
            Ok(placement) => placement,
            Err(error) => {
                self.note_command_memory(cmdbuf, mem);
                return Err(error);
            }
        };
        // The lists recorded at these bytes are over: this one replaces them. Retiring them first makes the blob
        // floor below the room the new list really has, instead of one list's worth short.
        self.record.retire_at(placement.guest);
        // A span can hold several lists one after another; this one's blobs go below the ones still live there.
        let (span_start, span_end) = spans
            .iter()
            .find(|span| placement.guest >= span.guest && placement.guest < span.guest + span.size)
            .map(|span| (span.guest, span.guest + span.size))
            .ok_or(Error::WritePointerOutsideMemory)?;
        let floor = self.record.blob_floor(span_start, span_end);
        let capacity = floor.saturating_sub(placement.guest);
        self.write_field.insert(cmdbuf, placement.field);
        self.record.begin(mem, cmdbuf, placement.guest, capacity)?;
        Ok(())
    }

    /// Opens an implicit list: a command the driver records into its own pushbuffer, which the host captures in its
    /// arena. The game's command memory is never touched here (`record.rs`, "Implicit lists").
    fn open_implicit(&mut self, mem: &dyn record::Memory, cmdbuf: u64) -> Result<(), Error> {
        match self.record.begin_arena(mem, cmdbuf) {
            Ok(_) => Ok(()),
            Err(error) => {
                self.implicit_dead.insert(cmdbuf);
                Err(error)
            }
        }
    }

    /// Which list `EndRecording` closes for `cmdbuf`.
    fn open_kind(&self, cmdbuf: u64) -> Open {
        if self.record.is_arena(cmdbuf) {
            Open::Arena
        } else if self.record.is_recording(cmdbuf) {
            Open::Guest
        } else {
            Open::None
        }
    }

    /// Files the implicit list `cmdbuf` recorded under the handle the driver reported. `opened` is false when no
    /// command the host records arrived since the last list ended.
    ///
    /// Some `EndRecording` calls return the driver's internal pointer, tagged above the 40-bit GPU space; no game
    /// call can name that value, so its records are kept until a later `EndRecording` on the same buffer reports a
    /// GPU handle, and that list then continues them (`record::Recorder::chain`). Keeping the chunks is what makes
    /// the window pass work: the driver ends its render-target and bind chunk under the internal pointer and records
    /// the draw that follows into the next one, and the draw is only correct with the state that precedes it. A
    /// buffer whose list had no recorded command gets an empty list, so a submit that names its handle still
    /// resolves instead of falling back to the driver.
    fn finish_implicit(
        &mut self,
        memory: &dyn record::Memory,
        cmdbuf: u64,
        handle: u64,
        opened: bool,
    ) {
        self.implicit_dead.remove(&cmdbuf);
        let closed = if opened {
            match self.record.end_arena(memory, cmdbuf) {
                Ok(key) => Some(key),
                Err(error) => {
                    self.fail_keyed(
                        "implicit-end",
                        format!("nvn record: implicit list at {handle:#x}: {error:?}"),
                    );
                    None
                }
            }
        } else {
            None
        };
        if !record::is_handle(handle) {
            // Not a handle any game call can name: keep the chunk, in recording order, for the list that follows.
            if let Some(key) = closed
                && let Some(generation) = self.record.generation(key)
            {
                self.record.pin(key);
                self.pending
                    .entry(cmdbuf)
                    .or_default()
                    .push((key, generation));
            }
            return;
        }
        let waiting = self.pending.remove(&cmdbuf).unwrap_or_default();
        for (key, _) in &waiting {
            self.record.unpin(*key);
        }
        // A slot the arena handed out again is not the chunk the host kept: it is dropped, not chained.
        let pending: Vec<u64> = waiting
            .iter()
            .filter_map(|(key, generation)| {
                (self.record.generation(*key) == Some(*generation)).then_some(*key)
            })
            .collect();
        if pending.len() != waiting.len() {
            self.fail_keyed(
                "implicit-recycled",
                format!(
                    "nvn record: implicit list at {handle:#x}: {} of {} waiting chunk(s) took their arena slot back",
                    waiting.len() - pending.len(),
                    waiting.len()
                ),
            );
        }
        let key = match (closed, pending.is_empty()) {
            (Some(key), true) => key,
            (Some(key), false) => {
                self.chain_chunks(key, &pending, handle);
                key
            }
            (None, false) => {
                // The handle names the last chunk re-reported by the driver; the ones before it are its prefix.
                let (successor, prefix) = pending.split_last().expect("pending is not empty");
                if !prefix.is_empty() {
                    self.chain_chunks(*successor, prefix, handle);
                }
                *successor
            }
            (None, true) => match self
                .record
                .begin_arena(memory, cmdbuf)
                .and_then(|_| self.record.end_arena(memory, cmdbuf))
            {
                Ok(key) => key,
                Err(error) => {
                    self.fail_keyed(
                        "implicit-empty",
                        format!("nvn record: empty implicit list at {handle:#x}: {error:?}"),
                    );
                    return;
                }
            },
        };
        if !self.record.refile(key, handle) {
            self.fail_keyed(
                "implicit-refile",
                format!("nvn record: implicit list at {handle:#x} could not be filed"),
            );
        }
    }

    /// Makes `pending` the prefix of `key`, in recording order. A chain that cannot be formed (a chunk already
    /// retired, or the replay's nesting limit) is reported and the chunks are dropped: the list is still filed, so
    /// the submit resolves instead of becoming a driver submit, which is the fallback this path had before.
    fn chain_chunks(&mut self, key: u64, pending: &[u64], handle: u64) -> bool {
        match self.record.chain(key, pending) {
            Ok(()) => true,
            Err(error) => {
                for old in pending {
                    self.record.drop_list(*old);
                }
                self.fail_keyed(
                    "implicit-chain",
                    format!(
                        "nvn record: implicit list at {handle:#x}: {error:?}; its {} chunk(s) are dropped",
                        pending.len()
                    ),
                );
                false
            }
        }
    }

    /// Retires the arena lists a submit captured, once its replay is done. The queued snapshot owns their records
    /// while it runs (`CONTRACT.md` §7), and the arena space goes back to the slot rotation instead of growing with
    /// the frame count. One submit of slack: this title's frame N+1 records `CallCommands` that name frame N's lists,
    /// and those calls resolve when N+1 is submitted, a little after N's replay. A later submit that still names one
    /// of them is a handle the host no longer records, which stays with the driver.
    fn retire_replayed(&mut self, serial: u64) {
        const GRACE: u64 = 1;
        while let Some((queued, keys)) = self.retiring.front() {
            if queued.saturating_add(GRACE) > serial {
                return;
            }
            let keys = keys.clone();
            self.retiring.pop_front();
            self.record.retire_replayed(&keys);
        }
    }

    /// `nvnCommandBufferAddCommandMemory`: remember the pool range so `BeginRecording` can resolve it later. The
    /// driver still runs, because the write pointer and the overflow callbacks are its fields.
    ///
    /// The range is not resolved here: a virtual pool may not have its mappings yet, and the guest address is only
    /// needed when a list actually starts (`command_spans`).
    fn add_command_memory(&mut self, a: &[u64; 31]) -> Result<(), Error> {
        if a[3] == 0 {
            return Err(Error::ZeroSize);
        }
        let ranges = self.command_memory.entry(a[0]).or_default();
        if ranges.len() >= 8 {
            ranges.remove(0);
        }
        ranges.push(CommandMemory {
            pool: a[1],
            offset: a[2],
            size: a[3],
        });
        Ok(())
    }

    /// Guest spans of the command memory this command buffer registered, in the order the driver walks them.
    fn command_spans(&self, cmdbuf: u64) -> Result<Vec<record::Span>, Error> {
        let ranges = self
            .command_memory
            .get(&cmdbuf)
            .ok_or(Error::NoCommandMemory)?;
        let mut spans = Vec::with_capacity(ranges.len());
        let mut first = None;
        for range in ranges {
            match self.objects.pool_host(range.pool, range.offset, range.size) {
                Ok(guest) => spans.push(record::Span {
                    guest,
                    size: range.size,
                }),
                Err(error) => first = first.or(Some(error)),
            }
        }
        if spans.is_empty() {
            return Err(first.unwrap_or(Error::NoCommandMemory));
        }
        Ok(spans)
    }

    /// `(size, used, free)` from the command memory this buffer registered: the game asks for these, and the
    /// driver's own counters stop moving once the host records instead of it.
    fn command_usage(&self, mem: &dyn record::Memory, cmdbuf: u64) -> Option<(u64, u64, u64)> {
        let spans = self.command_spans(cmdbuf).ok()?;
        let size: u64 = spans.iter().map(|span| span.size).sum();
        let placement = record::placement(mem, cmdbuf, &spans).ok()?;
        let used = placement
            .guest
            .saturating_sub(spans.first()?.guest)
            .min(size);
        Some((size, used, size - used))
    }

    /// Records a `CallCommands`/`CopyCommands` with the handles this recorder holds, and names the ones it does
    /// not: a handle the driver recorded before the host started recording is genuinely unresolvable here.
    fn call_known(
        &mut self,
        mem: &dyn record::Memory,
        cmdbuf: u64,
        handles: &[u64],
        what: &str,
    ) -> Result<(), Error> {
        let (known, unknown): (Vec<u64>, Vec<u64>) = handles
            .iter()
            .partition(|handle| self.record.list(**handle).is_some());
        if !unknown.is_empty() {
            let first: Vec<String> = unknown
                .iter()
                .take(4)
                .map(|handle| format!("{handle:#x}"))
                .collect();
            self.fail_keyed(
                if what == "CopyCommands" {
                    "copy-handles"
                } else {
                    "call-handles"
                },
                format!(
                    "nvn record: {what}: {} handles were recorded by the driver, not the host (first {})",
                    unknown.len(),
                    first.join(", ")
                ),
            );
        }
        if known.is_empty() {
            return Ok(());
        }
        if what == "CopyCommands" {
            self.record.copy(mem, cmdbuf, &known)?;
        } else {
            self.record.call(mem, cmdbuf, &known)?;
        }
        Ok(())
    }

    /// One bounded line with the header words and spans, so a write pointer that does not resolve tells us what the
    /// driver actually stores instead of a bare `Unmapped`.
    fn note_command_memory(&mut self, cmdbuf: u64, mem: &dyn record::Memory) {
        if self.record_diagnosed {
            return;
        }
        self.record_diagnosed = true;
        let header = record::header(mem, cmdbuf);
        let spans: Vec<String> = self
            .command_memory
            .get(&cmdbuf)
            .map(|ranges| {
                ranges
                    .iter()
                    .map(|range| {
                        let guest = self.objects.pool_host(range.pool, range.offset, range.size);
                        match guest {
                            Ok(guest) => format!("{guest:#x}+{:#x}", range.size),
                            Err(error) => format!("{error:?}"),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        match header {
            Some(header) => eprintln!(
                "kirikae: nvn record: command buffer {cmdbuf:#x}: write {:#x} (shadow field {:#x}), base {:#x}, gpu base {:#x}, command memory [{}]",
                header.write,
                header.write_shadow,
                header.base,
                header.gpu_base,
                spans.join(", ")
            ),
            None => eprintln!(
                "kirikae: nvn record: command buffer {cmdbuf:#x}: the header is unreadable; command memory [{}]",
                spans.join(", ")
            ),
        }
    }

    fn set_render_targets(
        &mut self,
        mem: &dyn record::Memory,
        a: &[u64; 31],
    ) -> Result<u64, Error> {
        let count = a[1].min(8);
        let mut targets = Vec::with_capacity(2 * count as usize + 2);
        for index in 0..count {
            let texture = if a[2] == 0 {
                0
            } else {
                read_word(mem, a[2] + 8 * index)?
            };
            let view = if a[2] == 0 || a[3] == 0 {
                0
            } else {
                read_word(mem, a[3] + 8 * index)?
            };
            targets.push(texture);
            targets.push(view);
        }
        targets.push(a[4]);
        targets.push(a[5]);
        let mut record = record::Record::new(record::Op::SetRenderTargets);
        record.count = count as u32;
        record.set_word(0, a[2]);
        record.set_word(1, a[3]);
        record.set_word(2, a[4]);
        record.set_word(3, a[5]);
        let mut blob = Vec::with_capacity(targets.len() * 8);
        for target in targets {
            blob.extend_from_slice(&target.to_le_bytes());
        }
        self.record.emit_blob(mem, a[0], record, &blob)
    }

    /// Clears keep the target index and the channel mask; the pass decision waits for replay (`CONTRACT.md` §3).
    fn clear_color(&mut self, mem: &dyn record::Memory, a: &[u64; 31]) -> Result<u64, Error> {
        let bytes = mem.read(a[2], 16).ok_or(Error::Unmapped)?;
        let mut record = record::Record::new(record::Op::ClearColor);
        for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            record.set_float(index, f32::from_bits(u32::from_le_bytes(*chunk)));
        }
        record.set_int(0, a[1] as u32 as i32);
        record.set_int(1, a[3] as u32 as i32);
        self.record.emit(mem, a[0], record)
    }

    fn clear_depth_stencil(
        &mut self,
        mem: &dyn record::Memory,
        a: &[u64; 31],
        v: &[[u64; 2]; 32],
    ) -> Result<u64, Error> {
        // Switchbrew: (commandBuffer, boolean depth, float depthValue, boolean stencil, int stencilValue).
        // `shadow.rs:716-732` reads the stencil value from x2 instead; both are kept so a replay comparison can
        // decide which one the driver actually passes (ints[3] is the shadow reading).
        let depth = a[1] as u8 != 0;
        let stencil = a[3] as u8 != 0;
        let mut record = record::Record::new(record::Op::ClearDepthStencil);
        record.set_float(0, f32::from_bits(v[0][0] as u32));
        record.set_int(0, a[4] as u32 as i32);
        record.set_int(1, i32::from(depth));
        record.set_int(2, i32::from(stencil));
        record.set_int(3, a[2] as u32 as i32);
        self.record.emit(mem, a[0], record)
    }

    /// Copies a state object's decoded fields into the record. A bind of an object no setter filled falls back to a
    /// raw struct copy, and to its address alone when even that is unreadable.
    fn bind_state(
        &mut self,
        mem: &dyn record::Memory,
        cmdbuf: u64,
        op: record::Op,
        address: u64,
        values: Vec<(state::Field, u64)>,
    ) -> Result<u64, Error> {
        if values.is_empty() {
            if mem.read(address, record::RAW_STATE_BYTES).is_none() {
                self.unresolved += 1;
            }
            self.copied += 1;
            return self.record.emit_raw_state(mem, cmdbuf, op, address);
        }
        self.decoded += 1;
        let values: Vec<record::StateValue> = values
            .iter()
            .map(|(field, value)| record::StateValue {
                kind: record::kind_code(field.0),
                index: field.1,
                value: *value,
            })
            .collect();
        self.record.emit_state(mem, cmdbuf, op, address, &values)
    }
}

/// The stage an `NVNshaderData` names: code block plus its control section (`nvn_shaders.h`). Graphics stages say
/// which they are in the SPH; only vertex and fragment exist in a graphics program.
fn stage(c: &Cpu, code: u64, control: u64) -> Option<replay::Stage> {
    let p = c.process();
    let magic = p.memory.read(control, 4)?;
    if u32::from_le_bytes(magic.as_slice().try_into().ok()?) != 0x9876_1234 {
        return None;
    }
    let words = p.memory.read(control.checked_add(0x6f8)?, 0x14)?;
    let read = |index: usize| {
        u32::from_le_bytes(
            words[index * 4..index * 4 + 4]
                .try_into()
                .unwrap_or_default(),
        )
    };
    let const_size = read(1);
    let const_offset = read(2);
    let program_start = read(4);
    let cpu = p
        .gpu
        .translate(code.checked_add(u64::from(program_start))?)?;
    let sph = p.memory.read(cpu, 4)?;
    let sph = u32::from_le_bytes(sph.try_into().ok()?);
    Some(replay::Stage {
        // The Maxwell translation starts at the SPH, which sits `program_start` bytes into the code block
        // (`docs/nvn/TRANSLATOR.md`), not at the block itself.
        code: code.checked_add(u64::from(program_start))?,
        fragment: (sph >> 10) & 0xf == 5,
        // `c[1]` is located from the block base, not from the SPH.
        constants: (const_size > 0)
            .then(|| {
                code.checked_add(u64::from(const_offset))
                    .map(|at| (at, const_size))
            })
            .flatten(),
    })
}

fn texture_info(texture: &Texture) -> replay::TextureInfo {
    replay::TextureInfo {
        host: texture.host,
        gpu: texture.gpu,
        format: texture.format,
        width: texture.width,
        height: texture.height,
        depth: texture.depth,
        levels: texture.levels,
        layers: texture.layers,
        samples: texture.samples,
        stride: texture.stride,
        block_height_log2: texture.block_height_log2,
        target: texture.target,
        swizzle: texture.swizzle,
    }
}

fn word(bytes: &[u8], offset: usize) -> u64 {
    bytes
        .get(offset..offset + 8)
        .and_then(|slice| <[u8; 8]>::try_from(slice).ok())
        .map_or(0, u64::from_le_bytes)
}

fn read_word(mem: &dyn record::Memory, address: u64) -> Result<u64, Error> {
    let bytes = mem.read(address, 8).ok_or(Error::Unmapped)?;
    <[u8; 8]>::try_from(bytes.as_slice())
        .map(u64::from_le_bytes)
        .map_err(|_| Error::Corrupt)
}

/// Reads a guest array a command proc names. A short or unmapped array yields nothing; each caller decides whether
/// that is an error.
fn read_words(mem: &dyn record::Memory, address: u64, count: u64) -> Vec<u64> {
    let count = count.min(record::MAX_HANDLES as u64);
    if count == 0 {
        return Vec::new();
    }
    let Some(bytes) = mem.read(address, 8 * count) else {
        return Vec::new();
    };
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| u64::from_le_bytes(*word))
        .collect()
}

/// The builder's packaged-layout argument points at the driver's value; a non-zero exponent 1..=5 is the BNTX block
/// height the texture was tiled with (`nvn_objects.cpp:303-315`).
fn read_u64(c: &Cpu, address: u64) -> Option<u64> {
    c.process()
        .memory
        .read(address, 8)
        .and_then(|bytes| <[u8; 8]>::try_from(bytes.as_slice()).ok())
        .map(u64::from_le_bytes)
}

fn read_border(c: &Cpu, address: u64) -> [f32; 4] {
    let Some(bytes) = c.process().memory.read(address, 16) else {
        return [0.0; 4];
    };
    let mut border = [0.0; 4];
    for (slot, chunk) in border.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *slot = f32::from_bits(u32::from_le_bytes(*chunk));
    }
    border
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_hosted_proc_has_a_host_meaning() {
        for name in [
            "nvnMemoryPoolBuilderSetDefaults",
            "nvnMemoryPoolBuilderSetStorage",
            "nvnMemoryPoolBuilderSetFlags",
            "nvnMemoryPoolInitialize",
            "nvnMemoryPoolFinalize",
            "nvnMemoryPoolMapVirtual",
            "nvnBufferBuilderSetDefaults",
            "nvnBufferBuilderSetStorage",
            "nvnBufferInitialize",
            "nvnBufferFinalize",
            "nvnTextureBuilderSetDefaults",
            "nvnTextureBuilderSetStorage",
            "nvnTextureBuilderSetPackagedTextureLayout",
            "nvnTextureBuilderSetSwizzle",
            "nvnTextureInitialize",
            "nvnTextureFinalize",
            "nvnTextureViewSetDefaults",
            "nvnTextureViewSetLevels",
            "nvnTextureViewSetLayers",
            "nvnTextureViewSetFormat",
            "nvnTextureViewSetTarget",
            "nvnSamplerBuilderSetDefaults",
            "nvnSamplerBuilderSetMinMagFilter",
            "nvnSamplerBuilderSetWrapMode",
            "nvnSamplerBuilderSetLodClamp",
            "nvnSamplerBuilderSetLodBias",
            "nvnSamplerBuilderSetCompare",
            "nvnSamplerBuilderSetMaxAnisotropy",
            "nvnSamplerBuilderSetReductionFilter",
            "nvnSamplerBuilderSetBorderColor",
            "nvnSamplerBuilderSetBorderColour",
            "nvnSamplerInitialize",
            "nvnSamplerFinalize",
            "nvnWindowBuilderSetDefaults",
            "nvnWindowBuilderSetTextures",
            "nvnWindowBuilderSetNumActiveTextures",
            "nvnWindowBuilderSetPresentInterval",
            "nvnWindowInitialize",
            "nvnWindowFinalize",
            "nvnWindowSetCrop",
            "nvnWindowSetPresentInterval",
            "nvnWindowSetNumActiveTextures",
            "nvnQueuePresentTexture",
            "nvnWindowPresentTexture",
        ] {
            assert!(proc(name).is_some(), "{name} is not hosted");
        }
        assert_eq!(proc("nvnDeviceInitialize"), None);
        assert_eq!(proc("nvnCommandBufferReportCounter"), None);
        assert_eq!(proc("nvnMemoryPoolGetBufferAddress"), None);
    }

    #[test]
    fn every_recorded_command_has_a_host_meaning() {
        for name in [
            "nvnCommandBufferInitialize",
            "nvnCommandBufferFinalize",
            "nvnCommandBufferAddCommandMemory",
            "nvnCommandBufferBeginRecording",
            "nvnCommandBufferEndRecording",
            "nvnCommandBufferSetRenderTargets",
            "nvnCommandBufferClearColor",
            "nvnCommandBufferClearDepthStencil",
            "nvnCommandBufferDiscardColor",
            "nvnCommandBufferBindProgram",
            "nvnCommandBufferBindBlendState",
            "nvnCommandBufferBindColorState",
            "nvnCommandBufferBindChannelMaskState",
            "nvnCommandBufferBindDepthStencilState",
            "nvnCommandBufferBindPolygonState",
            "nvnCommandBufferBindVertexAttribState",
            "nvnCommandBufferBindVertexStreamState",
            "nvnCommandBufferBindTexture",
            "nvnCommandBufferBindUniformBuffer",
            "nvnCommandBufferBindStorageBuffer",
            "nvnCommandBufferSetSamplerPool",
            "nvnCommandBufferDrawArrays",
            "nvnCommandBufferDrawArraysInstanced",
            "nvnCommandBufferDrawElements",
            "nvnCommandBufferDrawElementsBaseVertex",
            "nvnCommandBufferDrawElementsInstanced",
            "nvnCommandBufferDispatchCompute",
            "nvnCommandBufferBarrier",
            "nvnCommandBufferCallCommands",
            "nvnCommandBufferCopyCommands",
            "nvnQueueSubmitCommands",
        ] {
            assert!(
                matches!(proc(name), Some(Proc::Command(_))),
                "{name} is not recorded"
            );
        }
        // The state setters feed the tables a bind decodes, so the record copies values, not structs.
        assert!(matches!(
            proc("nvnBlendStateSetDefaults"),
            Some(Proc::State(state::Setter::BlendDefaults))
        ));
    }

    #[test]
    fn a_scripted_sequence_lands_in_guest_command_memory() {
        use super::record::{FLAG_BLOB, FakeMemory, Op, Record, StateValue};

        const GUEST: u64 = 0x1_0000_0000;
        const GPU: u64 = 0x2_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const POOL: u64 = 0x1000;
        const SIZE: u64 = 0x8000;
        const MEMORY: u64 = GUEST + 0x1000;

        let mem = FakeMemory::new(GUEST, SIZE);
        let mut host = Host::default();
        host.objects.make(
            POOL,
            Object::Pool(Pool {
                storage: GUEST,
                size: SIZE,
                gpu: GPU,
                ..Pool::default()
            }),
        );
        host.ranges.register(GPU, SIZE, POOL).expect("GPU range");

        // What the driver leaves in the command buffer header: the write pointer one page into the memory. Its GPU
        // base is the driver's own address space, deliberately not the host's synthetic pool range.
        mem.poke(CMD + record::header::BASE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::GPU_BASE,
            &0x8000_0000u64.to_le_bytes(),
        );
        mem.poke(
            CMD + record::header::WRITE,
            &(MEMORY + 0x1000).to_le_bytes(),
        );
        // This driver keeps the span end in the field shadow reads.
        mem.poke(
            CMD + record::header::WRITE_SHADOW,
            &(MEMORY + 0x6000).to_le_bytes(),
        );

        let v = [[0u64; 2]; 32];
        // A command with no command memory has nowhere to go and stays on the driver path.
        let mut a = [0u64; 31];
        a[0] = CMD;
        assert_eq!(
            host.recording(CommandProc::DrawArrays, &mem, &a, &v),
            Ok(Outcome::Driver)
        );
        (a[0], a[1], a[2], a[3]) = (CMD, POOL, 0x1000, 0x6000);
        assert_eq!(
            host.recording(CommandProc::AddCommandMemory, &mem, &a, &v),
            Ok(Outcome::Driver)
        );

        let mut a = [0u64; 31];
        a[0] = CMD;
        assert_eq!(
            host.recording(CommandProc::BeginRecording, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );
        let handle = host.record.handle(CMD).expect("recording");
        assert_eq!(
            handle,
            MEMORY + 0x1000,
            "handle is the first record's address"
        );

        // State setters fill the host tables; the bind copies the decoded values.
        let state_object = GUEST + 0x3000;
        let mut setter = [0u64; 31];
        setter[0] = state_object;
        host.state.set(state::Setter::BlendDefaults, &setter);
        let mut a = [0u64; 31];
        (a[0], a[1]) = (CMD, state_object);
        assert_eq!(
            host.recording(CommandProc::BindBlendState, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        // Render targets: the arrays are snapshotted at record time.
        let textures = GUEST + 0x7800;
        let views = GUEST + 0x7810;
        mem.poke(
            textures,
            &[0x1111u64.to_le_bytes(), 0x2222u64.to_le_bytes()].concat(),
        );
        mem.poke(
            views,
            &[0x3333u64.to_le_bytes(), 0x4444u64.to_le_bytes()].concat(),
        );
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3], a[4], a[5]) = (CMD, 2, textures, views, 0x5555, 0x6666);
        assert_eq!(
            host.recording(CommandProc::SetRenderTargets, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        // A colour clear keeps its value, target index and channel mask.
        let color = GUEST + 0x7820;
        mem.poke(
            color,
            &[
                1.0f32.to_bits(),
                0.5f32.to_bits(),
                0.25f32.to_bits(),
                1.0f32.to_bits(),
            ]
            .map(u32::to_le_bytes)
            .concat(),
        );
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 1, color, 0x5);
        assert_eq!(
            host.recording(CommandProc::ClearColor, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        // A depth/stencil clear: the depth value is the float argument.
        let mut v = [[0u64; 2]; 32];
        v[0][0] = u64::from(0.75f32.to_bits());
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3], a[4]) = (CMD, 1, 0xdead, 1, 7);
        assert_eq!(
            host.recording(CommandProc::ClearDepthStencil, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 1, 3, 0x0020_2007);
        assert_eq!(
            host.recording(CommandProc::BindTexture, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]) = (CMD, 0, 1, 42, 0x8000_0000, 3, 2, 5);
        assert_eq!(
            host.recording(CommandProc::DrawElementsInstanced, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 4, 2, 1);
        assert_eq!(
            host.recording(CommandProc::DispatchCompute, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        let mut a = [0u64; 31];
        (a[0], a[1]) = (CMD, 0xff);
        assert_eq!(
            host.recording(CommandProc::Barrier, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );

        let mut a = [0u64; 31];
        a[0] = CMD;
        assert_eq!(
            host.recording(CommandProc::EndRecording, &mem, &a, &v),
            Ok(Outcome::Host(handle))
        );

        let list = host.record.list(handle).expect("list is live");
        assert_eq!(list.records, 10);
        assert_eq!(list.bytes, 10 * record::RECORD_SIZE);
        let mut records = Vec::new();
        for index in 0..list.records {
            let at = handle + u64::from(index) * record::RECORD_SIZE;
            records.push(Record::decode(&mem.read_back(at, record::RECORD_SIZE)).expect("record"));
        }
        let ops: Vec<Op> = records.iter().map(|record| record.op).collect();
        assert_eq!(
            ops,
            [
                Op::Begin,
                Op::BindBlendState,
                Op::SetRenderTargets,
                Op::ClearColor,
                Op::ClearDepthStencil,
                Op::BindTexture,
                Op::DrawElements,
                Op::DispatchCompute,
                Op::Barrier,
                Op::End,
            ]
        );
        assert_eq!((records[6].ints[0], records[6].ints[1]), (0, 1));
        assert_eq!(records[6].words[0], 0x8000_0000);
        assert_eq!((records[6].words[1], records[6].words[2]), (42, 5));
        assert_eq!(
            (
                records[6].ints[2],
                records[6].ints[3],
                records[4].ints[0],
                records[4].ints[1],
                records[4].ints[2],
                records[4].ints[3]
            ),
            (3, 2, 7, 1, 1, 0xdead)
        );
        assert_eq!(records[3].floats, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!((records[3].ints[0], records[3].ints[1]), (1, 0x5));
        assert_eq!(records[4].floats[0], 0.75);

        // The texture handle keeps both halves of the descriptor.
        assert_eq!(records[5].words[0], 0x0020_2007);
        assert_eq!((records[5].ints[0], records[5].ints[1]), (1, 3));
        assert_eq!((records[5].ints[2], records[5].ints[3]), (0x2007, 2));

        // Render targets are stored by value: texture/view pairs, then the depth pair.
        let targets = &records[2];
        assert_eq!((targets.count, targets.flags & FLAG_BLOB), (2, FLAG_BLOB));
        let blob = mem.read_back(
            handle + u64::from(targets.blob),
            u64::from(targets.blob_len),
        );
        let words: Vec<u64> = blob
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .collect();
        assert_eq!(words, [0x1111, 0x3333, 0x2222, 0x4444, 0x5555, 0x6666]);

        // State is decoded by value, not a pointer to the object.
        let blend = &records[1];
        assert_eq!(blend.op, Op::BindBlendState);
        assert_eq!((blend.words[0], blend.count), (state_object, 6));
        let blob = mem.read_back(handle + u64::from(blend.blob), u64::from(blend.blob_len));
        let values: Vec<StateValue> = blob
            .as_chunks::<16>()
            .0
            .iter()
            .filter_map(|chunk| StateValue::decode(chunk))
            .collect();
        assert_eq!(values.len(), 6);
        assert!(values.iter().any(|value| {
            value.kind == record::kind_code(state::Kind::BlendSrc) && value.value == 2
        }));

        assert_eq!(records[9].op, Op::End);
        assert_eq!(records[9].words[0], 9);

        // The guest's own counter moved by everything the list consumed, blobs included.
        let usage = host.record.usage();
        let write = mem.read_back(CMD + record::header::WRITE, 8);
        let write = u64::from_le_bytes(write.as_slice().try_into().unwrap());
        assert_eq!(write, MEMORY + 0x1000 + usage.bytes + usage.blob_bytes);
        let limit = mem.read_back(CMD + record::header::WRITE_SHADOW, 8);
        assert_eq!(
            u64::from_le_bytes(limit.as_slice().try_into().unwrap()),
            MEMORY + 0x6000,
            "the span end is not touched"
        );
        assert_eq!(usage.lists, 1);
        assert_eq!(usage.records, 10);
        assert_eq!(host.decoded, 1);
        assert_eq!(host.failures, 0);
        assert_eq!(host.record.report().live, 1);

        // The game's command-memory queries come from the same spans and write pointer.
        let mut a = [0u64; 31];
        a[0] = CMD;
        let v = [[0u64; 2]; 32];
        let used = match host.recording(CommandProc::GetCommandMemoryUsed, &mem, &a, &v) {
            Ok(Outcome::Host(used)) => used,
            other => panic!("used: {other:?}"),
        };
        assert_eq!(used, write - MEMORY);
        assert_eq!(
            host.recording(CommandProc::GetCommandMemorySize, &mem, &a, &v),
            Ok(Outcome::Host(0x6000))
        );
        assert_eq!(
            host.recording(CommandProc::GetCommandMemoryFree, &mem, &a, &v),
            Ok(Outcome::Host(0x6000 - used))
        );
    }

    #[test]
    fn a_bind_without_decoded_state_copies_the_struct() {
        use super::record::{FLAG_RAW, FakeMemory, Op, Record};

        const GUEST: u64 = 0x1_0000_0000;
        const GPU: u64 = 0x2_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const MEMORY: u64 = GUEST + 0x1000;

        let mem = FakeMemory::new(GUEST, 0x8000);
        let mut host = Host::default();
        host.objects.make(
            0x1000,
            Object::Pool(Pool {
                storage: GUEST,
                size: 0x8000,
                gpu: GPU,
                ..Pool::default()
            }),
        );
        host.ranges
            .register(GPU, 0x8000, 0x1000)
            .expect("GPU range");
        mem.poke(CMD + record::header::BASE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::GPU_BASE,
            &0x8000_0000u64.to_le_bytes(),
        );
        mem.poke(
            CMD + record::header::WRITE,
            &(MEMORY + 0x1000).to_le_bytes(),
        );
        // This driver keeps the span end in the field shadow reads.
        mem.poke(
            CMD + record::header::WRITE_SHADOW,
            &(MEMORY + 0x6000).to_le_bytes(),
        );

        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 0x1000, 0x1000, 0x6000);
        host.recording(CommandProc::AddCommandMemory, &mem, &a, &[[0u64; 2]; 32])
            .expect("command memory");
        let mut a = [0u64; 31];
        a[0] = CMD;
        host.recording(CommandProc::BeginRecording, &mem, &a, &[[0u64; 2]; 32])
            .expect("begin");
        let handle = host.record.handle(CMD).expect("recording");

        // The polygon state object was filled with setters this host never saw, so its bytes are copied.
        let object = GUEST + 0x3000;
        mem.poke(object, &[0x5a; record::RAW_STATE_BYTES as usize]);
        let mut a = [0u64; 31];
        (a[0], a[1]) = (CMD, object);
        host.recording(CommandProc::BindPolygonState, &mem, &a, &[[0u64; 2]; 32])
            .expect("bind polygon");
        let mut a = [0u64; 31];
        a[0] = CMD;
        host.recording(CommandProc::EndRecording, &mem, &a, &[[0u64; 2]; 32])
            .expect("end");

        let at = handle + record::RECORD_SIZE;
        let record = Record::decode(&mem.read_back(at, record::RECORD_SIZE)).expect("record");
        assert_eq!(record.op, Op::BindPolygonState);
        assert_ne!(record.flags & FLAG_RAW, 0);
        let blob = mem.read_back(handle + u64::from(record.blob), u64::from(record.blob_len));
        assert_eq!(blob, vec![0x5a; record::RAW_STATE_BYTES as usize]);
        assert_eq!((host.decoded, host.copied), (0, 1));
    }

    #[test]
    fn a_submit_of_a_driver_recorded_list_stays_with_the_driver() {
        use super::record::FakeMemory;

        const GUEST: u64 = 0x1_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const MEMORY: u64 = GUEST + 0x1000;

        let mem = FakeMemory::new(GUEST, 0x8000);
        let mut host = Host::default();
        host.objects.make(
            0x1000,
            Object::Pool(Pool {
                storage: GUEST,
                size: 0x8000,
                gpu: 0x2_0000_0000,
                ..Pool::default()
            }),
        );
        mem.poke(CMD + record::header::BASE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::GPU_BASE,
            &0x8000_0000u64.to_le_bytes(),
        );
        mem.poke(CMD + record::header::WRITE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::WRITE_SHADOW,
            &(MEMORY + 0x6000).to_le_bytes(),
        );
        let v = [[0u64; 2]; 32];
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 0x1000, 0x1000, 0x6000);
        host.recording(CommandProc::AddCommandMemory, &mem, &a, &v)
            .expect("command memory");
        let mut a = [0u64; 31];
        a[0] = CMD;
        host.recording(CommandProc::BeginRecording, &mem, &a, &v)
            .expect("begin");
        let mut a = [0u64; 31];
        a[0] = CMD;
        let handle = match host.recording(CommandProc::EndRecording, &mem, &a, &v) {
            Ok(Outcome::Host(handle)) => handle,
            other => panic!("end: {other:?}"),
        };

        // A submit that names a driver-recorded list keeps its pushbuffer: the driver submits it and the GPU path
        // renders it, so nothing is silently dropped.
        let driver = GUEST + 0x5000;
        let handles = GUEST + 0x3000;
        mem.poke(handles, &driver.to_le_bytes());
        let mut a = [0u64; 31];
        (a[0], a[1], a[2]) = (CMD, 1, handles);
        assert_eq!(
            host.recording(CommandProc::SubmitCommands, &mem, &a, &v),
            Ok(Outcome::Driver)
        );

        // The host-recorded list is submitted for replay instead.
        mem.poke(handles, &handle.to_le_bytes());
        assert_eq!(
            host.recording(CommandProc::SubmitCommands, &mem, &a, &v),
            Ok(Outcome::Host(0))
        );
    }

    #[test]
    fn a_call_to_a_driver_recorded_handle_is_one_reject_not_one_per_frame() {
        use super::record::FakeMemory;

        const GUEST: u64 = 0x1_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const MEMORY: u64 = GUEST + 0x1000;

        let mem = FakeMemory::new(GUEST, 0x8000);
        let mut host = Host::default();
        host.objects.make(
            0x1000,
            Object::Pool(Pool {
                storage: GUEST,
                size: 0x8000,
                gpu: 0x2_0000_0000,
                ..Pool::default()
            }),
        );
        mem.poke(CMD + record::header::BASE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::GPU_BASE,
            &0x8000_0000u64.to_le_bytes(),
        );
        mem.poke(CMD + record::header::WRITE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::WRITE_SHADOW,
            &(MEMORY + 0x6000).to_le_bytes(),
        );
        let v = [[0u64; 2]; 32];
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 0x1000, 0x1000, 0x6000);
        host.recording(CommandProc::AddCommandMemory, &mem, &a, &v)
            .expect("command memory");
        let mut a = [0u64; 31];
        a[0] = CMD;
        host.recording(CommandProc::BeginRecording, &mem, &a, &v)
            .expect("begin");

        // A handle the driver recorded (the game's implicit lists) is unresolvable, and it is reported once even
        // though the game calls it every frame.
        let handles = GUEST + 0x3000;
        mem.poke(handles, &(GUEST + 0x5000u64).to_le_bytes());
        let mut a = [0u64; 31];
        (a[0], a[1], a[2]) = (CMD, 1, handles);
        for _ in 0..3 {
            assert_eq!(
                host.recording(CommandProc::CallCommands, &mem, &a, &v),
                Ok(Outcome::Host(0))
            );
        }
        assert_eq!(host.failures, 1);
        // The list itself stays recordable.
        let mut a = [0u64; 31];
        a[0] = CMD;
        assert!(matches!(
            host.recording(CommandProc::EndRecording, &mem, &a, &v),
            Ok(Outcome::Host(_))
        ));
    }

    #[test]
    fn a_command_without_begin_is_captured_in_the_arena() {
        use super::record::FakeMemory;

        const GUEST: u64 = 0x1_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const MEMORY: u64 = GUEST + 0x1000;

        let mem = FakeMemory::new(GUEST, 0x8000);
        let mut host = Host::default();
        host.objects.make(
            0x1000,
            Object::Pool(Pool {
                storage: GUEST,
                size: 0x8000,
                gpu: 0x2_0000_0000,
                ..Pool::default()
            }),
        );
        mem.poke(CMD + record::header::BASE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::GPU_BASE,
            &0x8000_0000u64.to_le_bytes(),
        );
        mem.poke(CMD + record::header::WRITE, &MEMORY.to_le_bytes());
        mem.poke(
            CMD + record::header::WRITE_SHADOW,
            &(MEMORY + 0x6000).to_le_bytes(),
        );
        let v = [[0u64; 2]; 32];
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, 0x1000, 0x1000, 0x6000);
        host.recording(CommandProc::AddCommandMemory, &mem, &a, &v)
            .expect("command memory");

        // The driver keeps recording into a fresh list after every EndRecording without asking the game to begin
        // again. The host cannot write those bytes: two writers in one command memory corrupt both streams. It
        // captures the command in its own arena instead and still returns `Driver`, so the driver writes its own
        // pushbuffer exactly as before (see `docs/nvn/CONTRACT.md` §3).
        let mut a = [0u64; 31];
        (a[0], a[1]) = (CMD, 0x7100);
        assert_eq!(
            host.recording(CommandProc::BindPolygonState, &mem, &a, &v),
            Ok(Outcome::Driver)
        );
        assert!(
            host.record.is_arena(CMD),
            "the implicit list lives in the host arena"
        );
        assert_eq!(
            host.record.usage().records,
            2,
            "Begin and the bind, both in the arena"
        );
        assert!(
            mem.read_back(MEMORY, 0x400).iter().all(|byte| *byte == 0),
            "the driver's command memory is untouched"
        );
        assert_eq!(host.failures, 0);
    }

    #[test]
    fn initialize_fills_the_host_table() {
        const STORAGE: u64 = 0x8_0000_0000;
        let mut host = Host::default();
        let pool = 0x1000;
        host.pool_builders.insert(
            0x2000,
            PoolBuilder {
                storage: STORAGE,
                size: 0x10_0000,
                flags: 0,
            },
        );
        host.pool_initialize(pool, 0x2000);
        let gpu = match host.objects.get(pool) {
            Some(Object::Pool(pool)) => {
                assert_eq!(
                    (pool.storage, pool.size, pool.flags),
                    (STORAGE, 0x10_0000, 0)
                );
                assert_eq!(pool.gpu % crate::nvn::va::ALIGN, 0);
                pool.gpu
            }
            other => panic!("expected a pool, got {other:?}"),
        };
        assert_eq!(host.ranges.lookup(gpu, 0x10_0000), Some((pool, 0)));

        host.buffer_builders.insert(
            0x3000,
            BufferBuilder {
                pool,
                offset: 0x2000,
                size: 0x4000,
            },
        );
        host.buffer_initialize(0x3100, 0x3000);
        match host.objects.get(0x3100) {
            Some(Object::Buffer(buffer)) => {
                assert_eq!(
                    (buffer.host, buffer.gpu, buffer.size),
                    (STORAGE + 0x2000, gpu + 0x2000, 0x4000)
                );
            }
            other => panic!("expected a buffer, got {other:?}"),
        }

        host.texture_builders.insert(
            0x4000,
            TextureBuilder {
                format: 0x25,
                width: 64,
                height: 64,
                levels: 1,
                pool,
                offset: 0x8000,
                ..TextureBuilder::default()
            },
        );
        host.texture_initialize(0x4100, 0x4000, |_| None);
        match host.objects.get(0x4100) {
            Some(Object::Texture(texture)) => {
                assert_eq!(
                    (texture.host, texture.gpu, texture.storage_size),
                    (STORAGE + 0x8000, gpu + 0x8000, 0x4000)
                );
                assert_eq!(texture.block_height_log2, 3);
            }
            other => panic!("expected a texture, got {other:?}"),
        }

        host.sampler_builders.insert(
            0x5000,
            Sampler {
                min_filter: 1,
                ..Sampler::default()
            },
        );
        host.sampler_initialize(0x5100, 0x5000);
        assert!(matches!(
            host.objects.get(0x5100),
            Some(Object::Sampler(sampler)) if sampler.min_filter == 1
        ));

        let counts = host.objects.counts();
        assert_eq!(
            (
                counts.pools,
                counts.buffers,
                counts.textures,
                counts.samplers
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(host.failures, 0);

        host.pool_finalize(pool);
        assert!(host.objects.get(pool).is_none());
        assert!(host.ranges.is_empty());
    }

    #[test]
    fn a_command_without_begin_recording_is_captured_in_the_arena() {
        use super::record::{ARENA_BASE, FakeMemory};

        const GUEST: u64 = 0x1_0000_0000;
        const GPU: u64 = 0x2_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const POOL: u64 = 0x1000;
        const SIZE: u64 = 0x8000;
        const MEMORY: u64 = GUEST + 0x1000;

        let mem = FakeMemory::new(GUEST, SIZE);
        let mut host = Host::default();
        host.objects.make(
            POOL,
            Object::Pool(Pool {
                storage: GUEST,
                size: SIZE,
                gpu: GPU,
                ..Pool::default()
            }),
        );
        host.ranges.register(GPU, SIZE, POOL).expect("GPU range");

        let v = [[0u64; 2]; 32];
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, POOL, 0x1000, 0x6000);
        assert_eq!(
            host.recording(CommandProc::AddCommandMemory, &mem, &a, &v),
            Ok(Outcome::Driver)
        );

        // No `BeginRecording`: the driver records this command into its own pushbuffer, so the host captures it in
        // its arena and leaves the game's command memory alone.
        let mut draw = [0u64; 31];
        (draw[0], draw[1], draw[2], draw[3]) = (CMD, 4, MEMORY, 3);
        assert_eq!(
            host.recording(CommandProc::DrawArrays, &mem, &draw, &v),
            Ok(Outcome::Driver)
        );
        assert!(
            host.record.is_arena(CMD),
            "the implicit list is in the arena"
        );
        let storage = host.record.handle(CMD).expect("a list is open");
        assert!(storage >= ARENA_BASE);
        assert!(
            mem.read_back(MEMORY, 0x400).iter().all(|byte| *byte == 0),
            "the driver's command memory was not written"
        );

        // `EndRecording` reports the GPU handle; the list resolves under it, ready for `nvnQueueSubmitCommands`.
        host.finish_implicit(&mem, CMD, 0x20_0000_0000, true);
        let list = host
            .record
            .list(0x20_0000_0000)
            .expect("the handle resolves");
        assert_eq!(
            (list.key, list.handle, list.arena, list.draws),
            (0x20_0000_0000, storage, true, 1)
        );
        assert!(host.record.list(storage).is_none(), "filed, not duplicated");
        assert_eq!(host.record.usage().implicit, 1);
        assert!(!host.record.is_recording(CMD));
    }

    /// Adds command memory to a fresh host and returns its fake memory, the setup every implicit-list test needs.
    fn implicit_setup() -> (super::record::FakeMemory, Host) {
        use super::record::FakeMemory;

        const GUEST: u64 = 0x1_0000_0000;
        const GPU: u64 = 0x2_0000_0000;
        const CMD: u64 = GUEST + 0x100;
        const POOL: u64 = 0x1000;
        const SIZE: u64 = 0x8000;

        let mem = FakeMemory::new(GUEST, SIZE);
        let mut host = Host::default();
        host.objects.make(
            POOL,
            Object::Pool(Pool {
                storage: GUEST,
                size: SIZE,
                gpu: GPU,
                ..Pool::default()
            }),
        );
        host.ranges.register(GPU, SIZE, POOL).expect("GPU range");
        let v = [[0u64; 2]; 32];
        let mut a = [0u64; 31];
        (a[0], a[1], a[2], a[3]) = (CMD, POOL, 0x1000, 0x6000);
        assert_eq!(
            host.recording(CommandProc::AddCommandMemory, &mem, &a, &v),
            Ok(Outcome::Driver)
        );
        (mem, host)
    }

    #[test]
    fn a_chunk_closed_under_the_internal_pointer_is_chained_into_the_next_handle() {
        const CMD: u64 = 0x1_0000_0100;
        let (mem, mut host) = implicit_setup();
        let v = [[0u64; 2]; 32];

        // The driver ends the window pass's bind chunk under its internal pointer: no game call can name it.
        let mut program = [0u64; 31];
        (program[0], program[1]) = (CMD, 0x7100);
        assert_eq!(
            host.recording(CommandProc::BindProgram, &mem, &program, &v),
            Ok(Outcome::Driver)
        );
        host.finish_implicit(&mem, CMD, 1 << 40, true);
        let pending = host.pending.get(&CMD).cloned().expect("chunk kept");
        assert_eq!(pending.len(), 1);
        assert_eq!(
            host.record.generation(pending[0].0),
            Some(pending[0].1),
            "the host remembers the generation of the chunk it keeps"
        );
        assert!(host.record.list(pending[0].0).is_some());

        // The draw that follows is recorded into the next chunk, which ends under a handle the game submits.
        let mut draw = [0u64; 31];
        (draw[0], draw[1], draw[2], draw[3]) = (CMD, 4, 0x1_0000_1000, 3);
        assert_eq!(
            host.recording(CommandProc::DrawArrays, &mem, &draw, &v),
            Ok(Outcome::Driver)
        );
        host.finish_implicit(&mem, CMD, 0x20_0000_0000, true);

        let list = host
            .record
            .list(0x20_0000_0000)
            .expect("the handle resolves");
        assert_eq!(
            list.chained,
            [pending[0].0],
            "the bind chunk precedes the draw"
        );
        assert_eq!(list.draws, 1);
        let chunk = host.record.list(pending[0].0).expect("the chunk is kept");
        assert_eq!((chunk.records, chunk.arena), (3, true));
        assert!(!host.pending.contains_key(&CMD), "the chunk was consumed");
        assert_eq!(host.failures, 0);
    }

    #[test]
    fn a_chunk_that_cannot_be_chained_is_reported_and_the_list_still_files() {
        const CMD: u64 = 0x1_0000_0100;
        let (mem, mut host) = implicit_setup();
        let v = [[0u64; 2]; 32];

        let mut program = [0u64; 31];
        (program[0], program[1]) = (CMD, 0x7100);
        host.recording(CommandProc::BindProgram, &mem, &program, &v)
            .expect("bind");
        host.finish_implicit(&mem, CMD, 1 << 40, true);
        let pending = host.pending.remove(&CMD).expect("chunk kept");
        // The chunk was retired before the handle arrived (a slot rotation, or a buffer being finalized).
        for (key, _) in &pending {
            host.record.drop_list(*key);
        }
        host.pending.insert(CMD, pending);

        let mut draw = [0u64; 31];
        (draw[0], draw[1], draw[2], draw[3]) = (CMD, 4, 0x1_0000_1000, 3);
        host.recording(CommandProc::DrawArrays, &mem, &draw, &v)
            .expect("draw");
        host.finish_implicit(&mem, CMD, 0x20_0000_0000, true);

        let list = host
            .record
            .list(0x20_0000_0000)
            .expect("the fallback still files the list");
        assert!(list.chained.is_empty());
        assert_eq!(list.draws, 1);
        assert_eq!(host.failures, 1);
        assert!(
            host.seen_failures.contains("implicit-recycled"),
            "the chunk that cannot be chained is reported: {:?}",
            host.seen_failures
        );
        assert_eq!(host.record.usage().implicit, 2);
    }

    fn window_texture(host: u64, format: u32, width: u32, height: u32) -> Texture {
        Texture {
            host,
            gpu: 0x2_0000_0000,
            pool: 0x1000,
            offset: 0,
            flags: 0,
            target: 1,
            format,
            width,
            height,
            depth: 1,
            layers: 1,
            levels: 1,
            samples: 1,
            stride: 0,
            packaged: 0,
            swizzle: super::super::textures::DEFAULT_SWIZZLE,
            storage_size: u64::from(width) * u64::from(height) * 4,
            alignment: 0x100,
            block_height_log2: 4,
        }
    }

    /// A window at `address` whose textures are `textures`, the way `WindowBuilderSetTextures` plus `WindowInitialize`
    /// leaves it.
    fn add_window(host: &mut Host, address: u64, textures: &[u64]) {
        let mut builder = window::Builder::default();
        builder.set_textures(textures);
        host.windows.insert(address, Window::from_builder(&builder));
    }

    #[test]
    fn a_present_resolves_the_window_texture_and_its_crop() {
        use super::super::window::Crop;
        let mut host = Host::default();
        add_window(&mut host, 0x1000, &[0x2000]);
        host.objects.make(
            0x2000,
            Object::Texture(window_texture(0x1_0000, 0x1d, 1920, 1080)),
        );
        host.windows.get_mut(&0x1000).unwrap().crop = Some(Crop {
            x: 320,
            y: 180,
            width: 1280,
            height: 720,
        });
        host.last_submit = 7;
        let target = host.present_target(0x1000, 0).expect("the target resolves");
        assert_eq!(target.cpu, 0x1_0000);
        assert_eq!(target.format, 0x1d);
        assert_eq!((target.width, target.height), (1920, 1080));
        assert_eq!(
            target.crop,
            window::Rect {
                x: 320,
                y: 180,
                width: 1280,
                height: 720
            }
        );
        assert_eq!(target.serial, 7);
    }

    #[test]
    fn a_present_of_something_unrecorded_stays_with_the_driver() {
        use super::super::window::Fallback;
        let mut host = Host::default();
        assert_eq!(host.present_target(0x9999, 0), Err(Fallback::UnknownWindow));
        add_window(&mut host, 0x1000, &[0x2000]);
        host.last_submit = 7;
        // The window names 0x2000, but no such texture reached the host table.
        assert_eq!(
            host.present_target(0x1000, 0),
            Err(Fallback::UnknownTexture)
        );
        assert_eq!(
            host.present_target(0x1000, 1),
            Err(Fallback::IndexOutsideWindow)
        );
        // Each fallback is reported once per cause, and the present is counted as the driver's.
        host.note_fallback(Fallback::UnknownTexture);
        host.note_fallback(Fallback::UnknownTexture);
        assert_eq!(host.presents.fallback, 2);
        assert_eq!(host.failures, 1);
    }

    #[test]
    fn a_present_waits_for_a_replayed_submit_and_is_not_shown_twice() {
        use super::super::window::Fallback;
        use crate::gpu::vulkan::Presented;
        let mut host = Host::default();
        add_window(&mut host, 0x1000, &[0x2000]);
        host.objects.make(
            0x2000,
            Object::Texture(window_texture(0x1_0000, 0x1d, 1280, 720)),
        );
        // Nothing has been replayed yet: the driver's frame is the only one.
        assert_eq!(
            host.present_target(0x1000, 0),
            Err(Fallback::NoReplayedSubmit)
        );
        host.last_submit = 1;
        assert!(host.present_target(0x1000, 0).is_ok());
        host.note_present(0x1000, 1, Presented::Shown);
        assert_eq!(host.presents.replayed, 1);
        assert_eq!(host.presents.fallback, 0);
        // A second present of the same submit has no frame of its own; the driver keeps it.
        assert_eq!(
            host.present_target(0x1000, 0),
            Err(Fallback::NoReplayedSubmit)
        );
        // A replayed present without a swapchain is still a frame, credited through the same counter.
        host.last_submit = 2;
        host.note_present(0x1000, 2, Presented::NoSwapchain);
        assert_eq!(host.presents.no_swapchain, 1);
        assert_eq!(host.presented.get(&0x1000), Some(&2));
        // No image for the texture: the driver keeps the frame.
        host.last_submit = 3;
        host.note_present(0x1000, 3, Presented::NoImage);
        assert_eq!(host.presents.fallback, 1);
        assert_eq!(host.presented.get(&0x1000), Some(&2));
    }

    #[test]
    fn a_submit_knows_every_window_texture_even_without_a_draw() {
        let mut host = Host::default();
        add_window(&mut host, 0x1000, &[0x2000, 0x3000, 0x2000]);
        host.objects.make(
            0x2000,
            Object::Texture(window_texture(0x1_0000, 0x1d, 1920, 1080)),
        );
        host.objects.make(
            0x3000,
            Object::Texture(window_texture(0x2_0000, 0x1d, 1920, 1080)),
        );
        // A texture the table does not know is not reported: the submit resolves textures, not addresses.
        add_window(&mut host, 0x4000, &[0x9000]);
        assert_eq!(host.window_textures(), [0x2000, 0x3000]);
    }
}
