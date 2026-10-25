use super::*;
use crate::gpu::Gpu;
use crate::mem::GuestMemory;

fn fixture() -> (Gpu, GuestMemory, State) {
    (
        Gpu::default(),
        GuestMemory::new().unwrap(),
        State::default(),
    )
}

fn env<'a>(gpu: &'a Gpu, mem: &'a GuestMemory, st: &'a State) -> Env<'a> {
    Env {
        gpu,
        mem,
        st,
        cpu: None,
        host1x: None,
    }
}

fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

fn get32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn get64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn run(env: &Env, dev: &Device, cmd: u32, args: &mut [u8], extra: &[u8]) -> (u32, Vec<u8>) {
    let mut extra_out = Vec::new();
    let (outcome, overrun) = {
        let mut a = Args::new(&mut *args);
        let outcome = dispatch(env, 1, dev, cmd, &mut a, extra, &mut extra_out);
        (outcome, a.overrun)
    };
    assert!(
        !overrun,
        "ioctl {cmd:#010x} wrote past its {}-byte argument struct",
        args.len()
    );
    let code = match outcome {
        Outcome::Ok => NV_SUCCESS,
        Outcome::Fail(code) => code,
        Outcome::Unimplemented => NV_NOT_IMPLEMENTED,
    };
    (code, extra_out)
}

struct Row {
    dev: Device,
    cmd: u32,
    size: u32,
    name: &'static str,
    /// Engine submissions need a live guest process; their header is covered by
    /// `submit_layout_rejects_short_buffers`.
    needs_process: bool,
}

/// Every ioctl the driver answers, with the size its ioctl number declares.
fn table() -> Vec<Row> {
    let nvmap = |cmd, size, name| Row {
        dev: Device::Nvmap,
        cmd,
        size,
        name,
        needs_process: false,
    };
    let as_gpu = |cmd, size, name| Row {
        dev: Device::AsGpu,
        cmd,
        size,
        name,
        needs_process: false,
    };
    let ctrl = |cmd, size, name| Row {
        dev: Device::Ctrl,
        cmd,
        size,
        name,
        needs_process: false,
    };
    let ctrl_gpu = |cmd, size, name| Row {
        dev: Device::CtrlGpu,
        cmd,
        size,
        name,
        needs_process: false,
    };
    let channel = |cmd, size, name| Row {
        dev: Device::Channel { syncpoint: 16 },
        cmd,
        size,
        name,
        needs_process: false,
    };
    let engine = |cmd, size, name, needs_process| Row {
        dev: Device::Engine { class: CLASS_NVDEC },
        cmd,
        size,
        name,
        needs_process,
    };
    vec![
        nvmap(0xc008_0101, 8, "NVMAP_IOC_CREATE"),
        nvmap(0x0000_0102, 0, "NVMAP_IOC_CLAIM"),
        nvmap(0xc008_0103, 8, "NVMAP_IOC_FROM_ID"),
        nvmap(0xc020_0104, 32, "NVMAP_IOC_ALLOC"),
        nvmap(0xc018_0105, 24, "NVMAP_IOC_FREE"),
        nvmap(0xc028_0106, 40, "NVMAP_IOC_MMAP"),
        nvmap(0xc028_0107, 40, "NVMAP_IOC_WRITE"),
        nvmap(0xc028_0108, 40, "NVMAP_IOC_READ"),
        nvmap(0xc00c_0109, 12, "NVMAP_IOC_PARAM"),
        nvmap(0xc010_010a, 16, "NVMAP_IOC_PIN_MULT"),
        nvmap(0xc010_010b, 16, "NVMAP_IOC_UNPIN_MULT"),
        nvmap(0xc008_010c, 8, "NVMAP_IOC_CACHE"),
        nvmap(0xc004_010d, 4, "NVMAP_IOC_GET_IVC_ID"),
        nvmap(0xc008_010e, 8, "NVMAP_IOC_GET_ID"),
        nvmap(0xc004_010f, 4, "NVMAP_IOC_FROM_IVC_ID"),
        nvmap(0x4004_0110, 4, "NVMAP_IOC_SET_ALLOCATION_TAG_LABEL"),
        nvmap(0x0000_0111, 0, "NVMAP_IOC_RESERVE"),
        nvmap(0x4010_0112, 16, "NVMAP_IOC_EXPORT_FOR_ARUID"),
        nvmap(0x4010_0113, 16, "NVMAP_IOC_IS_OWNED_BY_ARUID"),
        nvmap(0x4010_0114, 16, "NVMAP_IOC_REMOVE_EXPORT_FOR_ARUID"),
        as_gpu(0x4004_4101, 4, "NVGPU_AS_IOCTL_BIND_CHANNEL"),
        as_gpu(0xc018_4102, 24, "NVGPU_AS_IOCTL_ALLOC_SPACE"),
        as_gpu(0xc010_4103, 16, "NVGPU_AS_IOCTL_FREE_SPACE"),
        as_gpu(0xc018_4104, 24, "NVGPU_AS_IOCTL_MAP_BUFFER"),
        as_gpu(0xc008_4105, 8, "NVGPU_AS_IOCTL_UNMAP_BUFFER"),
        as_gpu(0xc028_4106, 40, "NVGPU_AS_IOCTL_MAP_BUFFER_EX"),
        as_gpu(0x4010_4107, 16, "NVGPU_AS_IOCTL_ALLOC_AS"),
        as_gpu(0xc040_4108, 64, "NVGPU_AS_IOCTL_GET_VA_REGIONS"),
        as_gpu(0x4028_4109, 40, "NVGPU_AS_IOCTL_ALLOC_AS_EX"),
        as_gpu(0xc038_410a, 56, "NVGPU_AS_IOCTL_MAP_BUFFER_EX2"),
        as_gpu(0xc014_4114, 20, "NVGPU_AS_IOCTL_REMAP"),
        ctrl(0xc008_0014, 8, "NVHOST_IOCTL_CTRL_SYNCPT_READ"),
        ctrl(0x4004_0015, 4, "NVHOST_IOCTL_CTRL_SYNCPT_INCR"),
        ctrl(0xc00c_0016, 12, "NVHOST_IOCTL_CTRL_SYNCPT_WAIT"),
        ctrl(0x4008_0017, 8, "NVHOST_IOCTL_CTRL_MODULE_MUTEX"),
        ctrl(0xc018_0018, 24, "NVHOST_IOCTL_CTRL_MODULE_REGRDWR"),
        ctrl(0xc010_0019, 16, "NVHOST_IOCTL_CTRL_SYNCPT_WAITEX"),
        ctrl(0xc008_001a, 8, "NVHOST_IOCTL_CTRL_SYNCPT_READ_MAX"),
        ctrl(0xc183_001b, 387, "NVHOST_IOCTL_CTRL_GET_CONFIG"),
        ctrl(0xc004_001c, 4, "NVHOST_IOCTL_CTRL_SYNCPT_CLEAR_EVENT_WAIT"),
        ctrl(0xc010_001d, 16, "NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT"),
        ctrl(0xc010_001e, 16, "NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT_EX"),
        ctrl(0xc004_001f, 4, "NVHOST_IOCTL_CTRL_SYNCPT_ALLOC_EVENT"),
        ctrl(0xc004_0020, 4, "NVHOST_IOCTL_CTRL_SYNCPT_FREE_EVENT"),
        ctrl(0x4008_0021, 8, "NVHOST_IOCTL_CTRL_SYNCPT_FREE_EVENT_BATCH"),
        ctrl(0xc004_0022, 4, "NVHOST_IOCTL_CTRL_SYNCPT_GET_SHIFT"),
        ctrl_gpu(0x8004_4701, 4, "NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE"),
        ctrl_gpu(0x8028_4702, 40, "NVGPU_GPU_IOCTL_ZCULL_GET_INFO"),
        ctrl_gpu(0x402c_4703, 44, "NVGPU_GPU_IOCTL_ZBC_SET_TABLE"),
        ctrl_gpu(0xc034_4704, 52, "NVGPU_GPU_IOCTL_ZBC_QUERY_TABLE"),
        ctrl_gpu(0xc0b0_4705, 176, "NVGPU_GPU_IOCTL_GET_CHARACTERISTICS"),
        ctrl_gpu(0xc018_4706, 24, "NVGPU_GPU_IOCTL_GET_TPC_MASKS"),
        ctrl_gpu(0x4008_4707, 8, "NVGPU_GPU_IOCTL_FLUSH_L2"),
        ctrl_gpu(0x4008_470d, 8, "NVGPU_GPU_IOCTL_INVAL_ICACHE"),
        ctrl_gpu(0x4008_470e, 8, "NVGPU_GPU_IOCTL_SET_MMU_DEBUG_MODE"),
        ctrl_gpu(0x4010_470f, 16, "NVGPU_GPU_IOCTL_SET_SM_DEBUG_MODE"),
        ctrl_gpu(0xc030_4710, 48, "NVGPU_GPU_IOCTL_WAIT_FOR_PAUSE"),
        ctrl_gpu(0xc008_4710, 8, "NVGPU_GPU_IOCTL_WAIT_FOR_PAUSE (old)"),
        ctrl_gpu(
            0x8008_4711,
            8,
            "NVGPU_GPU_IOCTL_GET_TPC_EXCEPTION_EN_STATUS",
        ),
        ctrl_gpu(0x8008_4712, 8, "NVGPU_GPU_IOCTL_NUM_VSMS"),
        ctrl_gpu(0xc004_4713, 4, "NVGPU_GPU_IOCTL_VSMS_MAPPING"),
        ctrl_gpu(0x8008_4714, 8, "NVGPU_GPU_IOCTL_ZBC_GET_ACTIVE_SLOT_MASK"),
        ctrl_gpu(0x8004_4715, 4, "NVGPU_GPU_IOCTL_PMU_GET_GPU_LOAD"),
        ctrl_gpu(0x4008_4716, 8, "NVGPU_GPU_IOCTL_SET_CG_CONTROLS"),
        ctrl_gpu(0xc008_4717, 8, "NVGPU_GPU_IOCTL_GET_CG_CONTROLS"),
        ctrl_gpu(0x4008_4718, 8, "NVGPU_GPU_IOCTL_SET_PG_CONTROLS"),
        ctrl_gpu(0xc008_4719, 8, "NVGPU_GPU_IOCTL_GET_PG_CONTROLS"),
        ctrl_gpu(
            0x8018_471a,
            24,
            "NVGPU_GPU_IOCTL_PMU_GET_ELPG_RESIDENCY_GATING",
        ),
        ctrl_gpu(
            0xc008_471b,
            8,
            "NVGPU_GPU_IOCTL_GET_ERROR_CHANNEL_USER_DATA",
        ),
        ctrl_gpu(0xc010_471c, 16, "NVGPU_GPU_IOCTL_GET_GPU_TIME"),
        ctrl_gpu(
            0xc108_471d,
            264,
            "NVGPU_GPU_IOCTL_GET_CPU_TIME_CORRELATION_INFO",
        ),
        channel(0xc010_0001, 16, "NVHOST_IOCTL_CHANNEL_SUBMIT"),
        channel(0xc008_0002, 8, "NVHOST_IOCTL_CHANNEL_GET_SYNCPOINT"),
        channel(0xc008_0003, 8, "NVHOST_IOCTL_CHANNEL_GET_WAITBASE"),
        channel(0xc008_0004, 8, "NVHOST_IOCTL_CHANNEL_GET_MODMUTEX"),
        channel(0x4004_0007, 4, "NVHOST_IOCTL_CHANNEL_SET_SUBMIT_TIMEOUT"),
        channel(0x4008_0008, 8, "NVHOST_IOCTL_CHANNEL_SET_CLK_RATE"),
        channel(0xc014_0009, 20, "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER"),
        channel(0xc014_000a, 20, "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER"),
        channel(0x0000_0013, 0, "NVHOST_IOCTL_CHANNEL_SET_TIMEOUT_EX"),
        channel(0xc008_0014, 8, "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE"),
        channel(0xc008_0023, 8, "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE"),
        channel(0xc010_0024, 16, "NVHOST_IOCTL_CHANNEL_SUBMIT_EX"),
        channel(0xc014_0025, 20, "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER_EX"),
        channel(0xc014_0026, 20, "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER_EX"),
        channel(0x4004_4801, 4, "NVGPU_IOCTL_CHANNEL_SET_NVMAP_FD"),
        channel(0x4004_4803, 4, "NVGPU_IOCTL_CHANNEL_SET_TIMEOUT"),
        channel(0x4008_4805, 8, "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO"),
        channel(0x4018_4806, 24, "NVGPU_IOCTL_CHANNEL_WAIT"),
        channel(0xc004_4807, 4, "NVGPU_IOCTL_CHANNEL_CYCLE_STATS"),
        channel(0xc020_4808, 32, "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO"),
        channel(0xc010_4809, 16, "NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX"),
        channel(0x4008_480a, 8, "NVHOST_IOCTL_CHANNEL_FREE_OBJ_CTX"),
        channel(0xc010_480b, 16, "NVGPU_IOCTL_CHANNEL_ZCULL_BIND"),
        channel(0xc018_480c, 24, "NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER"),
        channel(0x4004_480d, 4, "NVGPU_IOCTL_CHANNEL_SET_PRIORITY"),
        channel(0x0000_480e, 0, "NVGPU_IOCTL_CHANNEL_ENABLE"),
        channel(0x0000_480f, 0, "NVGPU_IOCTL_CHANNEL_DISABLE"),
        channel(0x0000_4810, 0, "NVGPU_IOCTL_CHANNEL_PREEMPT"),
        channel(0x0000_4811, 0, "NVGPU_IOCTL_CHANNEL_FORCE_RESET"),
        channel(0x4008_4812, 8, "NVGPU_IOCTL_CHANNEL_EVENT_ID_CONTROL"),
        channel(0xc010_4813, 16, "NVGPU_IOCTL_CHANNEL_CYCLE_STATS_SNAPSHOT"),
        channel(0x4008_4714, 8, "NVGPU_IOCTL_CHANNEL_SET_USER_DATA"),
        channel(0x8008_4715, 8, "NVGPU_IOCTL_CHANNEL_GET_USER_DATA"),
        channel(0x8080_4816, 128, "NVGPU_IOCTL_CHANNEL_GET_ERROR_INFO"),
        channel(
            0xc010_4817,
            16,
            "NVGPU_IOCTL_CHANNEL_GET_ERROR_NOTIFICATION",
        ),
        channel(0x4020_4818, 32, "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX"),
        channel(0xc018_4819, 24, "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO_RETRY"),
        channel(0xc020_481a, 32, "NVGPU_IOCTL_CHANNEL_ALLOC_GPFIFO_EX2"),
        channel(0xc018_481b, 24, "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2"),
        channel(0xc018_481c, 24, "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2_RETRY"),
        channel(0xc004_481d, 4, "NVGPU_IOCTL_CHANNEL_SET_TIMESLICE"),
        engine(0xc010_0001, 16, "NVHOST_IOCTL_CHANNEL_SUBMIT", true),
        engine(0xc008_0002, 8, "NVHOST_IOCTL_CHANNEL_GET_SYNCPOINT", false),
        engine(0xc008_0003, 8, "NVHOST_IOCTL_CHANNEL_GET_WAITBASE", false),
        engine(0xc008_0004, 8, "NVHOST_IOCTL_CHANNEL_GET_MODMUTEX", false),
        engine(
            0x4004_0007,
            4,
            "NVHOST_IOCTL_CHANNEL_SET_SUBMIT_TIMEOUT",
            false,
        ),
        engine(0x4008_0008, 8, "NVHOST_IOCTL_CHANNEL_SET_CLK_RATE", false),
        engine(
            0xc014_0009,
            20,
            "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER",
            false,
        ),
        engine(
            0xc014_000a,
            20,
            "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER",
            false,
        ),
        engine(0x0000_0013, 0, "NVHOST_IOCTL_CHANNEL_SET_TIMEOUT_EX", false),
        engine(0xc008_0023, 8, "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE", false),
        engine(0xc010_0024, 16, "NVHOST_IOCTL_CHANNEL_SUBMIT_EX", true),
        engine(
            0xc014_0025,
            20,
            "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER_EX",
            false,
        ),
        engine(
            0xc014_0026,
            20,
            "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER_EX",
            false,
        ),
    ]
}

#[test]
fn every_ioctl_keeps_the_size_its_number_declares() {
    for row in table() {
        let declared = (row.cmd >> 16) & 0x3fff;
        assert_eq!(
            declared, row.size,
            "{} ({:#010x}) declares {declared} bytes, the table says {}",
            row.name, row.cmd, row.size
        );
    }
}

#[test]
fn every_implemented_ioctl_answers_without_leaving_its_buffer() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let mut answered = 0;
    for row in table() {
        let mut args = vec![0u8; row.size as usize];
        let mut extra = Vec::new();
        let mut extra_out = Vec::new();
        let (outcome, overrun) = {
            let mut a = Args::new(&mut args);
            if row.cmd == 0xc018_481b || row.cmd == 0xc018_481c {
                put32(&mut *a.data, 8, 1);
                extra = gpfifo_entry(0, 0).to_le_bytes().to_vec();
            }
            let outcome = dispatch(&env, 1, &row.dev, row.cmd, &mut a, &extra, &mut extra_out);
            (outcome, a.overrun)
        };
        assert!(
            !overrun,
            "{} ({:#010x}) wrote past its {}-byte struct",
            row.name, row.cmd, row.size
        );
        if row.needs_process {
            continue;
        }
        assert_ne!(
            outcome,
            Outcome::Unimplemented,
            "{} ({:#010x}) is in the table but not dispatched",
            row.name,
            row.cmd
        );
        answered += 1;
    }
    assert!(answered > 120, "only {answered} ioctls exercised");
}

#[test]
fn nvmap_object_lifecycle_and_params() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);

    let mut args = [0u8; 8];
    put32(&mut args, 0, 0x4000);
    assert_eq!(run(&env, &Device::Nvmap, 0xc008_0101, &mut args, &[]).0, 0);
    let handle = get32(&args, 4);
    assert!(gpu.nvmap(handle).is_some());

    let mut args = [0u8; 32];
    put32(&mut args, 0, handle);
    put32(&mut args, 12, 0x2000);
    put64(&mut args, 24, 0x8000_0000);
    assert_eq!(run(&env, &Device::Nvmap, 0xc020_0104, &mut args, &[]).0, 0);
    assert_eq!(get32(&args, 12), 0x2000);
    assert_eq!(gpu.nvmap(handle).unwrap().addr, 0x8000_0000);

    let mut args = [0u8; 12];
    put32(&mut args, 0, handle);
    put32(&mut args, 4, 1);
    assert_eq!(run(&env, &Device::Nvmap, 0xc00c_0109, &mut args, &[]).0, 0);
    assert_eq!(get32(&args, 8), 0x4000);

    put32(&mut args, 4, 3);
    assert_eq!(
        run(&env, &Device::Nvmap, 0xc00c_0109, &mut args, &[]).0,
        NV_BAD_VALUE
    );

    let mut args = [0u8; 24];
    put32(&mut args, 0, handle);
    assert_eq!(run(&env, &Device::Nvmap, 0xc018_0105, &mut args, &[]).0, 0);
    assert_eq!(get64(&args, 8), 0x8000_0000);
    assert_eq!(get32(&args, 16), 0x4000);
    assert_eq!(
        run(&env, &Device::Nvmap, 0xc018_0105, &mut args, &[]).0,
        NV_BAD_VALUE
    );
}

#[test]
fn nvmap_reports_not_supported_where_the_console_does() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    for cmd in [
        0x0000_0102, // CLAIM
        0xc028_0106, // MMAP
        0xc028_0107, // WRITE
        0xc028_0108, // READ
        0xc010_010a, // PIN_MULT
        0xc010_010b, // UNPIN_MULT
        0xc008_010c, // CACHE
        0xc004_010d, // GET_IVC_ID
        0xc004_010f, // FROM_IVC_ID
        0x4004_0110, // SET_ALLOCATION_TAG_LABEL
        0x0000_0111, // RESERVE
    ] {
        let size = ((cmd >> 16) & 0x3fff) as usize;
        let mut args = vec![0u8; size];
        assert_eq!(
            run(&env, &Device::Nvmap, cmd, &mut args, &[]).0,
            NV_NOT_SUPPORTED,
            "{cmd:#010x}"
        );
    }
}

#[test]
fn nvmap_tracks_aruid_bindings() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let handle = gpu.nvmap_create(0x1000);

    let mut args = [0u8; 16];
    put64(&mut args, 0, 0x1234);
    put32(&mut args, 8, handle);
    assert_eq!(run(&env, &Device::Nvmap, 0x4010_0112, &mut args, &[]).0, 0);
    assert_eq!(run(&env, &Device::Nvmap, 0x4010_0113, &mut args, &[]).0, 0);

    put64(&mut args, 0, 0x9999);
    assert_eq!(
        run(&env, &Device::Nvmap, 0x4010_0113, &mut args, &[]).0,
        NV_BAD_VALUE
    );
    put64(&mut args, 0, 0x1234);
    assert_eq!(run(&env, &Device::Nvmap, 0x4010_0114, &mut args, &[]).0, 0);
    assert_eq!(
        run(&env, &Device::Nvmap, 0x4010_0113, &mut args, &[]).0,
        NV_BAD_VALUE
    );
}

#[test]
fn as_gpu_reserves_maps_and_frees_space() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let buffer = gpu.nvmap_create(0x8000);
    assert!(gpu.nvmap_alloc(buffer, 0x5000_0000, 0x1000, 0));

    let mut args = [0u8; 24];
    put32(&mut args, 0, 4);
    put32(&mut args, 4, PAGE_SMALL as u32);
    assert_eq!(run(&env, &Device::AsGpu, 0xc018_4102, &mut args, &[]).0, 0);
    let space = get64(&args, 16);
    assert_ne!(space, 0);

    let mut args = [0u8; 40];
    put32(&mut args, 8, buffer);
    put32(&mut args, 12, PAGE_SMALL as u32);
    put64(&mut args, 16, 0x4000);
    put64(&mut args, 24, 0x2000);
    assert_eq!(run(&env, &Device::AsGpu, 0xc028_4106, &mut args, &[]).0, 0);
    let va = get64(&args, 32);
    assert_eq!(gpu.translate(va), Some(0x5000_4000));

    // A mapping that runs past the nvmap is accepted: NVN rounds small buffers
    // up to the 64 KiB compression page before asking for it.
    put64(&mut args, 24, 0x8000);
    assert_eq!(run(&env, &Device::AsGpu, 0xc028_4106, &mut args, &[]).0, 0);

    let mut args = [0u8; 8];
    put64(&mut args, 0, va);
    assert_eq!(run(&env, &Device::AsGpu, 0xc008_4105, &mut args, &[]).0, 0);
    assert!(gpu.translate(va).is_none());
    put64(&mut args, 0, u64::MAX);
    assert_eq!(run(&env, &Device::AsGpu, 0xc008_4105, &mut args, &[]).0, 0);

    let mut args = [0u8; 16];
    put64(&mut args, 0, space);
    put32(&mut args, 8, 4);
    put32(&mut args, 12, PAGE_SMALL as u32);
    assert_eq!(run(&env, &Device::AsGpu, 0xc010_4103, &mut args, &[]).0, 0);
    assert_eq!(
        run(&env, &Device::AsGpu, 0xc010_4103, &mut args, &[]).0,
        NV_BAD_VALUE
    );
}

#[test]
fn as_gpu_reports_its_va_regions_and_rejects_bad_pages() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let mut args = [0u8; 64];
    let (code, _) = run(&env, &Device::AsGpu, 0xc040_4108, &mut args, &[]);
    assert_eq!(code, 0);
    assert_eq!(get32(&args, 8), 0x30);
    assert_eq!(get64(&args, 16), VA_REGIONS[0].0);
    assert_eq!(get32(&args, 24), VA_REGIONS[0].1);
    assert_eq!(get64(&args, 32), VA_REGIONS[0].2);
    assert_eq!(get64(&args, 40), VA_REGIONS[1].0);
    assert_eq!(get32(&args, 48), VA_REGIONS[1].1);
    assert_eq!(get64(&args, 56), VA_REGIONS[1].2);

    // The console accepts the address space hints without a reply, even when
    // the guest leaves the page size at zero.
    let mut args = [0u8; 16];
    assert_eq!(run(&env, &Device::AsGpu, 0x4010_4107, &mut args, &[]).0, 0);
    let mut args = [0u8; 40];
    assert_eq!(run(&env, &Device::AsGpu, 0x4028_4109, &mut args, &[]).0, 0);
    let mut args = [0u8; 24];
    assert_eq!(
        run(&env, &Device::AsGpu, 0xc018_4102, &mut args, &[]).0,
        NV_BAD_PARAMETER
    );
}

#[test]
fn ctrl_reads_increments_and_times_out_syncpoints() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);

    let mut args = [0u8; 4];
    put32(&mut args, 0, 5);
    assert_eq!(run(&env, &Device::Ctrl, 0x4004_0015, &mut args, &[]).0, 0);

    let mut args = [0u8; 8];
    put32(&mut args, 0, 5);
    assert_eq!(run(&env, &Device::Ctrl, 0xc008_0014, &mut args, &[]).0, 0);
    assert_eq!(get32(&args, 4), 1);

    let mut args = [0u8; 12];
    put32(&mut args, 0, 5);
    put32(&mut args, 4, 1);
    assert_eq!(run(&env, &Device::Ctrl, 0xc00c_0016, &mut args, &[]).0, 0);
    put32(&mut args, 4, 9);
    assert_eq!(
        run(&env, &Device::Ctrl, 0xc00c_0016, &mut args, &[]).0,
        NV_TIMEOUT
    );

    let mut args = [0u8; 16];
    put32(&mut args, 0, 5);
    put32(&mut args, 4, 9);
    assert_eq!(
        run(&env, &Device::Ctrl, 0xc010_0019, &mut args, &[]).0,
        NV_TIMEOUT
    );
    assert_eq!(get32(&args, 12), 1);

    let mut args = [0u8; 4];
    put32(&mut args, 0, 0x60);
    assert_eq!(
        run(&env, &Device::Ctrl, 0xc004_001f, &mut args, &[]).0,
        NV_BAD_VALUE
    );

    let mut args = [0u8; 4];
    assert_eq!(run(&env, &Device::Ctrl, 0xc004_0022, &mut args, &[]).0, 0);
    assert_eq!(get32(&args, 0), 0);
}

#[test]
fn ctrl_reports_not_supported_for_config_and_register_access() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    for cmd in [0xc018_0018, 0xc183_001b] {
        let size = ((cmd >> 16) & 0x3fff) as usize;
        let mut args = vec![0u8; size];
        assert_eq!(
            run(&env, &Device::Ctrl, cmd, &mut args, &[]).0,
            NV_NOT_SUPPORTED
        );
    }
}

#[test]
fn ctrl_gpu_keeps_the_zbc_table_it_is_given() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);

    let mut args = [0u8; 44];
    for i in 0..8u32 {
        put32(&mut args, 4 * i as usize, i + 1);
    }
    put32(&mut args, 32, 0x1111);
    put32(&mut args, 36, 0x2222);
    put32(&mut args, 40, 1);
    assert_eq!(
        run(&env, &Device::CtrlGpu, 0x402c_4703, &mut args, &[]).0,
        0
    );

    let mut args = [0u8; 52];
    assert_eq!(
        run(&env, &Device::CtrlGpu, 0xc034_4704, &mut args, &[]).0,
        0
    );
    assert_eq!(get32(&args, 0), 1);
    assert_eq!(get32(&args, 28), 8);
    assert_eq!(get32(&args, 32), 0x1111);
    assert_eq!(get32(&args, 36), 1);
    assert_eq!(get32(&args, 40), 0x2222);
    assert_eq!(get32(&args, 44), 1);

    put32(&mut args, 40, 3);
    assert_eq!(
        run(&env, &Device::CtrlGpu, 0x402c_4703, &mut args, &[]).0,
        NV_BAD_VALUE
    );
}

#[test]
fn ctrl_gpu_answers_identity_and_clock_queries() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);

    let mut args = [0u8; 176];
    let (code, _) = run(&env, &Device::CtrlGpu, 0xc0b0_4705, &mut args, &[]);
    assert_eq!(code, 0);
    assert_eq!(get64(&args, 0), 0xa0);
    assert_eq!(get32(&args, 16), 0x120);

    let mut args = [0u8; 8];
    assert_eq!(
        run(&env, &Device::CtrlGpu, 0x8008_4712, &mut args, &[]).0,
        0
    );
    assert_eq!(get32(&args, 0), 2);
    assert_eq!(get32(&args, 4), 0);

    let mut args = [0u8; 4];
    assert_eq!(
        run(&env, &Device::CtrlGpu, 0xc004_4713, &mut args, &[]).0,
        0
    );
    assert_eq!(args, [0, 0, 0, 1]);

    let mut args = [0u8; 16];
    assert_eq!(
        run(&env, &Device::CtrlGpu, 0xc010_471c, &mut args, &[]).0,
        0
    );
    assert_eq!(get64(&args, 8), 0);
}

#[test]
fn channel_exposes_its_syncpoint_and_user_data() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let sp = gpu.alloc_syncpoint();
    let dev = Device::Channel { syncpoint: sp };
    gpu.increment(sp, 7);

    let mut args = [0u8; 8];
    assert_eq!(run(&env, &dev, 0xc008_0002, &mut args, &[]).0, 0);
    assert_eq!(get32(&args, 4), 7);

    let mut args = [0u8; 8];
    put64(&mut args, 0, 0xdead_beef);
    assert_eq!(run(&env, &dev, 0x4008_4714, &mut args, &[]).0, 0);
    let mut args = [0u8; 8];
    assert_eq!(run(&env, &dev, 0x8008_4715, &mut args, &[]).0, 0);
    assert_eq!(get64(&args, 0), 0xdead_beef);

    let mut args = [0u8; 32];
    put32(&mut args, 0, 0x100);
    put32(&mut args, 4, 4);
    assert_eq!(run(&env, &dev, 0xc020_481a, &mut args, &[]).0, 0);
    assert_eq!(get32(&args, 12), sp);
    assert_eq!(get32(&args, 16), 7);

    let mut args = [0u8; 16];
    assert_eq!(run(&env, &dev, 0xc010_4817, &mut args, &[]).0, 0);
    assert_eq!(&args[0xc..0xe], &[0, 0]);
    assert_eq!(&args[0xe..0x10], &[0xff, 0xff]);
}

fn mapped_buffer(gpu: &Gpu, mem: &GuestMemory, words: &[u32]) -> u32 {
    mem.map(0x1000_0000, 0x1000).unwrap();
    gpu.map(0x5_0000_0000, 0x1000_0000, 0x1000);
    let handle = gpu.nvmap_create(0x1000);
    assert!(gpu.nvmap_alloc(handle, 0x1000_0000, 0x1000, 0));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(mem.write(0x1000_0000, &bytes));
    handle
}

#[test]
fn channel_submits_gpfifo_entries_inline_and_through_ioctl2() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let sp = gpu.alloc_syncpoint();
    let dev = Device::Channel { syncpoint: sp };
    let increment = [(1u32 << 29) | (1 << 16) | 0x1d, (sp << 8) | 1];
    let _handle = mapped_buffer(&gpu, &mem, &increment);
    let entry = gpfifo_entry(0x5_0000_0000, increment.len() as u64);

    let mut args = vec![0u8; 0x20];
    put32(&mut args, 8, 1);
    put64(&mut args, 0x18, entry);
    assert_eq!(run(&env, &dev, 0xc020_4808, &mut args, &[]).0, 0);
    assert_eq!(gpu.syncpoint(sp), 1);
    assert_eq!(get32(&args, 16), sp);
    assert_eq!(get32(&args, 20), 1);

    let mut args = vec![0u8; 0x18];
    put32(&mut args, 8, 1);
    assert_eq!(
        run(&env, &dev, 0xc018_481b, &mut args, &entry.to_le_bytes()).0,
        0
    );
    assert_eq!(gpu.syncpoint(sp), 2);
    assert_eq!(get32(&args, 20), 2);
}

#[test]
fn channel_rejects_a_gpfifo_that_does_not_fit() {
    let (gpu, mem, st) = fixture();
    let env = env(&gpu, &mem, &st);
    let dev = Device::Channel { syncpoint: 16 };
    let mut args = vec![0u8; 0x18];
    put32(&mut args, 8, 3);
    assert_eq!(
        run(&env, &dev, 0xc018_4808, &mut args, &[]).0,
        NV_BAD_PARAMETER
    );
}

#[test]
fn submit_layout_rejects_short_buffers() {
    let mut args = [0u8; 16];
    put32(&mut args, 0, 2);
    assert_eq!(
        SubmitLayout::parse(&Args::new(&mut args)),
        Err(NV_BAD_PARAMETER)
    );
    put32(&mut args, 0, 0);
    let layout = SubmitLayout::parse(&Args::new(&mut args)).unwrap();
    assert_eq!(
        (layout.buffers, layout.relocs, layout.incrs, layout.fences),
        (0, 0, 0, 0)
    );
}

#[test]
fn out_only_ioctls_take_their_argument_buffer_from_the_reply() {
    // ZCULL_GET_CTX_SIZE is `_IOR(0, 1, u32)`: the guest sends no input buffer.
    assert!(args_supplied(0x8004_4701, 0, 4));
    assert!(!args_supplied(0x8004_4701, 0, 0));
    // NVMAP_IOC_CREATE is inout: the input buffer carries the struct.
    assert!(args_supplied(0xc008_0101, 8, 8));
    assert!(!args_supplied(0xc008_0101, 0, 8));
    // A zero-size ioctl needs no buffer at all.
    assert!(args_supplied(0x0000_0013, 0, 0));
}

#[test]
fn names_say_what_is_missing() {
    assert_eq!(ioctl_name(&Device::Nvmap, 0xc028_0107), "NVMAP_IOC_WRITE");
    assert_eq!(
        ioctl_name(&Device::Channel { syncpoint: 16 }, 0xc018_481b),
        "NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2"
    );
    assert_eq!(
        ioctl_name(&Device::Engine { class: CLASS_NVDEC }, 0xc008_0023),
        "NVHOST_IOCTL_CHANNEL_GET_CLK_RATE"
    );
}
