use crate::gpu::{CLASS_2D, CLASS_3D, CLASS_COMPUTE, CLASS_DMA, CLASS_INLINE};

use super::Device;

pub(super) fn characteristics() -> Vec<u8> {
    let mut v = Vec::with_capacity(0xa0);
    let u32s = |xs: &[u32], v: &mut Vec<u8>| {
        xs.iter()
            .for_each(|x| v.extend_from_slice(&x.to_le_bytes()))
    };
    u32s(&[0x120, 0xb, 0xa1, 1], &mut v);
    v.extend_from_slice(&0x40000u64.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes());
    u32s(
        &[
            2, 0x20, 0x20000, 0x20000, 0x1b, 0x30000, 1, 0x503, 0x503, 0x80,
        ],
        &mut v,
    );
    v.extend_from_slice(&0x28u16.to_le_bytes());
    v.extend_from_slice(&0u16.to_le_bytes());
    u32s(
        &[
            0x55,
            CLASS_2D,
            CLASS_3D,
            CLASS_COMPUTE,
            0xb06f,
            CLASS_INLINE,
            CLASS_DMA,
            1,
            0,
            2,
            1,
            0,
            1,
            0x21d70,
            0,
        ],
        &mut v,
    );
    v.extend_from_slice(&0x6230326d67u64.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes());
    v.resize(0xa0, 0);
    v
}

/// Documented name of an ioctl, used to say what is missing instead of
/// answering with a bare zero.
pub(super) fn ioctl_name(dev: &Device, cmd: u32) -> &'static str {
    let nr = (cmd & 0xff) as usize;
    let group = (cmd >> 8) & 0xff;
    match dev {
        Device::Nvmap => match nr {
            0x01 => "NVMAP_IOC_CREATE",
            0x02 => "NVMAP_IOC_CLAIM",
            0x03 => "NVMAP_IOC_FROM_ID",
            0x04 => "NVMAP_IOC_ALLOC",
            0x05 => "NVMAP_IOC_FREE",
            0x06 => "NVMAP_IOC_MMAP",
            0x07 => "NVMAP_IOC_WRITE",
            0x08 => "NVMAP_IOC_READ",
            0x09 => "NVMAP_IOC_PARAM",
            0x0a => "NVMAP_IOC_PIN_MULT",
            0x0b => "NVMAP_IOC_UNPIN_MULT",
            0x0c => "NVMAP_IOC_CACHE",
            0x0d => "NVMAP_IOC_GET_IVC_ID",
            0x0e => "NVMAP_IOC_GET_ID",
            0x0f => "NVMAP_IOC_FROM_IVC_ID",
            0x10 => "NVMAP_IOC_SET_ALLOCATION_TAG_LABEL",
            0x11 => "NVMAP_IOC_RESERVE",
            0x12 => "NVMAP_IOC_EXPORT_FOR_ARUID",
            0x13 => "NVMAP_IOC_IS_OWNED_BY_ARUID",
            0x14 => "NVMAP_IOC_REMOVE_EXPORT_FOR_ARUID",
            _ => "unknown nvmap ioctl",
        },
        Device::AsGpu => match nr {
            0x01 => "NVGPU_AS_IOCTL_BIND_CHANNEL",
            0x02 => "NVGPU_AS_IOCTL_ALLOC_SPACE",
            0x03 => "NVGPU_AS_IOCTL_FREE_SPACE",
            0x04 => "NVGPU_AS_IOCTL_MAP_BUFFER",
            0x05 => "NVGPU_AS_IOCTL_UNMAP_BUFFER",
            0x06 => "NVGPU_AS_IOCTL_MAP_BUFFER_EX",
            0x07 => "NVGPU_AS_IOCTL_ALLOC_AS",
            0x08 => "NVGPU_AS_IOCTL_GET_VA_REGIONS",
            0x09 => "NVGPU_AS_IOCTL_ALLOC_AS_EX",
            0x0a => "NVGPU_AS_IOCTL_MAP_BUFFER_EX2",
            0x14 => "NVGPU_AS_IOCTL_REMAP",
            _ => "unknown nvhost-as-gpu ioctl",
        },
        Device::Ctrl => match nr {
            0x14 => "NVHOST_IOCTL_CTRL_SYNCPT_READ",
            0x15 => "NVHOST_IOCTL_CTRL_SYNCPT_INCR",
            0x16 => "NVHOST_IOCTL_CTRL_SYNCPT_WAIT",
            0x17 => "NVHOST_IOCTL_CTRL_MODULE_MUTEX",
            0x18 => "NVHOST_IOCTL_CTRL_MODULE_REGRDWR",
            0x19 => "NVHOST_IOCTL_CTRL_SYNCPT_WAITEX",
            0x1a => "NVHOST_IOCTL_CTRL_SYNCPT_READ_MAX",
            0x1b => "NVHOST_IOCTL_CTRL_GET_CONFIG",
            0x1c => "NVHOST_IOCTL_CTRL_SYNCPT_CLEAR_EVENT_WAIT",
            0x1d => "NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT",
            0x1e => "NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT_EX",
            0x1f => "NVHOST_IOCTL_CTRL_SYNCPT_ALLOC_EVENT",
            0x20 => "NVHOST_IOCTL_CTRL_SYNCPT_FREE_EVENT",
            0x21 => "NVHOST_IOCTL_CTRL_SYNCPT_FREE_EVENT_BATCH",
            0x22 => "NVHOST_IOCTL_CTRL_SYNCPT_GET_SHIFT",
            _ => "unknown nvhost-ctrl ioctl",
        },
        Device::CtrlGpu => match nr {
            0x01 => "NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE",
            0x02 => "NVGPU_GPU_IOCTL_ZCULL_GET_INFO",
            0x03 => "NVGPU_GPU_IOCTL_ZBC_SET_TABLE",
            0x04 => "NVGPU_GPU_IOCTL_ZBC_QUERY_TABLE",
            0x05 => "NVGPU_GPU_IOCTL_GET_CHARACTERISTICS",
            0x06 => "NVGPU_GPU_IOCTL_GET_TPC_MASKS",
            0x07 => "NVGPU_GPU_IOCTL_FLUSH_L2",
            0x0d => "NVGPU_GPU_IOCTL_INVAL_ICACHE",
            0x0e => "NVGPU_GPU_IOCTL_SET_MMU_DEBUG_MODE",
            0x0f => "NVGPU_GPU_IOCTL_SET_SM_DEBUG_MODE",
            0x10 => "NVGPU_GPU_IOCTL_WAIT_FOR_PAUSE",
            0x11 => "NVGPU_GPU_IOCTL_GET_TPC_EXCEPTION_EN_STATUS",
            0x12 => "NVGPU_GPU_IOCTL_NUM_VSMS",
            0x13 => "NVGPU_GPU_IOCTL_VSMS_MAPPING",
            0x14 => "NVGPU_GPU_IOCTL_ZBC_GET_ACTIVE_SLOT_MASK",
            0x15 => "NVGPU_GPU_IOCTL_PMU_GET_GPU_LOAD",
            0x16 => "NVGPU_GPU_IOCTL_SET_CG_CONTROLS",
            0x17 => "NVGPU_GPU_IOCTL_GET_CG_CONTROLS",
            0x18 => "NVGPU_GPU_IOCTL_SET_PG_CONTROLS",
            0x19 => "NVGPU_GPU_IOCTL_GET_PG_CONTROLS",
            0x1a => "NVGPU_GPU_IOCTL_PMU_GET_ELPG_RESIDENCY_GATING",
            0x1b => "NVGPU_GPU_IOCTL_GET_ERROR_CHANNEL_USER_DATA",
            0x1c => "NVGPU_GPU_IOCTL_GET_GPU_TIME",
            0x1d => "NVGPU_GPU_IOCTL_GET_CPU_TIME_CORRELATION_INFO",
            _ => "unknown nvhost-ctrl-gpu ioctl",
        },
        Device::Channel { .. } => match (group, nr) {
            (0x00, 0x01) => "NVHOST_IOCTL_CHANNEL_SUBMIT",
            (0x00, 0x02) => "NVHOST_IOCTL_CHANNEL_GET_SYNCPOINT",
            (0x00, 0x03) => "NVHOST_IOCTL_CHANNEL_GET_WAITBASE",
            (0x00, 0x04) => "NVHOST_IOCTL_CHANNEL_GET_MODMUTEX",
            (0x00, 0x07) => "NVHOST_IOCTL_CHANNEL_SET_SUBMIT_TIMEOUT",
            (0x00, 0x08) => "NVHOST_IOCTL_CHANNEL_SET_CLK_RATE",
            (0x00, 0x09) => "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER",
            (0x00, 0x0a) => "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER",
            (0x00, 0x13) => "NVHOST_IOCTL_CHANNEL_SET_TIMEOUT_EX",
            (0x00, 0x14) => "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE",
            (0x00, 0x23) => "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE",
            (0x00, 0x24) => "NVHOST_IOCTL_CHANNEL_SUBMIT_EX",
            (0x00, 0x25) => "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER_EX",
            (0x00, 0x26) => "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER_EX",
            (0x47, 0x14) => "NVGPU_IOCTL_CHANNEL_SET_USER_DATA",
            (0x47, 0x15) => "NVGPU_IOCTL_CHANNEL_GET_USER_DATA",
            (0x48, 0x01) => "NVGPU_IOCTL_CHANNEL_SET_NVMAP_FD",
            (0x48, 0x03) => "NVGPU_IOCTL_CHANNEL_SET_TIMEOUT",
            (0x48, 0x05) => "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO",
            (0x48, 0x06) => "NVGPU_IOCTL_CHANNEL_WAIT",
            (0x48, 0x07) => "NVGPU_IOCTL_CHANNEL_CYCLE_STATS",
            (0x48, 0x08) => "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO",
            (0x48, 0x09) => "NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX",
            (0x48, 0x0a) => "NVHOST_IOCTL_CHANNEL_FREE_OBJ_CTX",
            (0x48, 0x0b) => "NVGPU_IOCTL_CHANNEL_ZCULL_BIND",
            (0x48, 0x0c) => "NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER",
            (0x48, 0x0d) => "NVGPU_IOCTL_CHANNEL_SET_PRIORITY",
            (0x48, 0x0e) => "NVGPU_IOCTL_CHANNEL_ENABLE",
            (0x48, 0x0f) => "NVGPU_IOCTL_CHANNEL_DISABLE",
            (0x48, 0x10) => "NVGPU_IOCTL_CHANNEL_PREEMPT",
            (0x48, 0x11) => "NVGPU_IOCTL_CHANNEL_FORCE_RESET",
            (0x48, 0x12) => "NVGPU_IOCTL_CHANNEL_EVENT_ID_CONTROL",
            (0x48, 0x13) => "NVGPU_IOCTL_CHANNEL_CYCLE_STATS_SNAPSHOT",
            (0x48, 0x16) => "NVGPU_IOCTL_CHANNEL_GET_ERROR_INFO",
            (0x48, 0x17) => "NVGPU_IOCTL_CHANNEL_GET_ERROR_NOTIFICATION",
            (0x48, 0x18) => "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX",
            (0x48, 0x19) => "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO_RETRY",
            (0x48, 0x1a) => "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX2",
            (0x48, 0x1b) => "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2",
            (0x48, 0x1c) => "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2_RETRY",
            (0x48, 0x1d) => "NVGPU_IOCTL_CHANNEL_SET_TIMESLICE",
            _ => "unknown nvhost channel ioctl",
        },
        Device::Engine { .. } => match nr {
            0x01 => "NVHOST_IOCTL_CHANNEL_SUBMIT",
            0x02 => "NVHOST_IOCTL_CHANNEL_GET_SYNCPOINT",
            0x03 => "NVHOST_IOCTL_CHANNEL_GET_WAITBASE",
            0x04 => "NVHOST_IOCTL_CHANNEL_GET_MODMUTEX",
            0x07 => "NVHOST_IOCTL_CHANNEL_SET_SUBMIT_TIMEOUT",
            0x08 => "NVHOST_IOCTL_CHANNEL_SET_CLK_RATE",
            0x09 => "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER",
            0x0a => "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER",
            0x13 => "NVHOST_IOCTL_CHANNEL_SET_TIMEOUT_EX",
            0x23 => "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE",
            0x24 => "NVHOST_IOCTL_CHANNEL_SUBMIT_EX",
            0x25 => "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER_EX",
            0x26 => "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER_EX",
            _ => "unknown engine channel ioctl",
        },
    }
}
