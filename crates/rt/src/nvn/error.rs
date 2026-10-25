//! Failures of the NVN host object model.

/// What an NVN proc can hand the host object model that it cannot represent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A size was zero where the NVN API requires storage.
    ZeroSize,
    /// An address or byte count does not fit in the modeled address space.
    Overflow,
    /// A range does not fit in the pool, texture or mapping it belongs to.
    OutOfRange,
    /// Two GPU virtual-address ranges, or two virtual-pool mappings, overlap.
    Overlap,
    /// A virtual-pool offset has no mapping, or the mapping does not cover the range.
    Unmapped,
    /// `nvnMemoryPoolMapVirtual` was called on a pool that owns its storage.
    NotVirtual,
    /// The pool a buffer, texture or mapping names is not in the registry.
    UnknownPool,
    /// The NVN format has no block layout, so no storage size can be computed.
    UnknownFormat,
    /// A user list names a handle this recorder does not hold; it was overwritten or never recorded.
    UnknownList,
    /// A list would recurse deeper than the replay's nesting limit.
    Nesting,
    /// The command memory the game handed over has no room for the record and its blob.
    Full,
    /// A command was recorded without an open list.
    NotRecording,
    /// `nvnCommandBufferBeginRecording` had no command memory registered for the buffer.
    NoCommandMemory,
    /// The write pointer the command buffer header holds does not land inside any command memory span.
    WritePointerOutsideMemory,
    /// A record read back from guest memory does not decode.
    Corrupt,
}
