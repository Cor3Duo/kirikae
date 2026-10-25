use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kirikae_a64::{Control, Insn, Stmt, lift};
use kirikae_loader::{LOAD_BASE, load};
use kirikae_recomp::project::{Backend, BuildOptions, build};
use kirikae_recomp::rust_backend::emit_program;
use kirikae_recomp::{Image, discover};

const USAGE: &str = "\
kirikae: recompile Nintendo Switch executables into native programs

usage:
  kirikae build <file> [-o <dir>] [--name <name>]       recompile into a native executable
        [--backend cranelift|rust] [-j <jobs>]
        [--fast-regs]                                keep guest registers in variables
                                                     (experimental; or KIRIKAE_FAST_REGS=1)
        [--plugin <crate dir>]                       link a crate whose install() runs first
        [--split <dlls>]                             put the recompiled code in that many DLLs
                                                     (default: only when one image would not fit)
        [--profile <file>]                           also compile the addresses a run had to
                                                     interpret (default: <dir>/<name>.profile,
                                                     written by the executable)
        [--profile-loop <runs>]                      build, run the result headless for
                                                     --run-seconds with a profile, rebuild with
                                                     what it interpreted, and repeat up to <runs>
                                                     times (stop when the count stops falling or a
                                                     run adds nothing); it starts the game, so it
                                                     needs this flag explicitly
        [--run-seconds <s>]                          bound for each --profile-loop run (default
                                                     60, at least 10: the stats window)
  kirikae run <file>                                    run with the interpreter only
  kirikae info <file>                                   modules, segments and decoder coverage
                                                     (NRO, NSO, ExeFS folder, DNSP)
  kirikae emit <file.nro> [-o <main.rs>]                print the generated Rust
  kirikae testgen <out.nro>                             write the built-in test program
  kirikae dev simd-cases <out> <count> [seed]           random SIMD/FP cases for tools/simd_oracle.py
  kirikae dev simd-check <cases> <results>              compare the oracle's results with kirikae
  kirikae dev extract <dump> <dir>                   write <dir>/exefs and <dir>/romfs
  kirikae dev module-dump <file> <module> <out>         write a module's loaded image
  kirikae dev syms <file> [pattern]                     list exported functions

Only use executables dumped from your own console.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("build") => cmd_build(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("info") => cmd_info(&args[1..]),
        Some("emit") => cmd_emit(&args[1..]),
        Some("testgen") => cmd_testgen(&args[1..]),
        Some("dev") => cmd_dev(&args[1..]),
        Some("-h" | "--help" | "help") | None => {
            println!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        Some(other) => Err(format!("unknown command `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("kirikae: {e}");
            ExitCode::FAILURE
        }
    }
}

const SWITCHES: &[&str] = &["--fast-regs"];

struct Args {
    positional: Vec<String>,
    options: Vec<(String, String)>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut out = Args {
        positional: Vec::new(),
        options: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if SWITCHES.contains(&a.as_str()) {
            out.options.push((a.clone(), String::new()));
        } else if a.starts_with('-') {
            let value = it.next().ok_or_else(|| format!("{a} needs a value"))?;
            out.options.push((a.clone(), value.clone()));
        } else {
            out.positional.push(a.clone());
        }
    }
    Ok(out)
}

impl Args {
    fn input(&self) -> Result<&str, String> {
        self.positional
            .first()
            .map(String::as_str)
            .ok_or_else(|| format!("missing input file\n\n{USAGE}"))
    }

    fn option(&self, names: &[&str]) -> Option<&str> {
        self.options
            .iter()
            .find(|(k, _)| names.contains(&k.as_str()))
            .map(|(_, v)| v.as_str())
    }
}

fn read(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

fn load_image(file: &[u8]) -> Result<Image, String> {
    load(file, LOAD_BASE).map_err(|e| e.to_string())
}

fn stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "game".into())
}

fn cmd_build(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let input = a.input()?;
    let source =
        kirikae_loader::package::read_source(Path::new(input)).map_err(|e| e.to_string())?;
    let name = a
        .option(&["--name"])
        .map(String::from)
        .unwrap_or_else(|| stem(input));
    let mut opts = BuildOptions::new(
        PathBuf::from(a.option(&["-o", "--out"]).unwrap_or(".")),
        &name,
    );
    opts.backend = match a.option(&["--backend"]) {
        None | Some("cranelift") => Backend::Cranelift,
        Some("rust") => Backend::Rust,
        Some(other) => return Err(format!("unknown backend `{other}` (cranelift or rust)")),
    };
    if let Some(j) = a.option(&["-j", "--jobs"]) {
        opts.jobs = j.parse().map_err(|_| format!("bad job count `{j}`"))?;
    }
    opts.profile = a.option(&["--profile"]).map(PathBuf::from);
    opts.plugins = a
        .options
        .iter()
        .filter(|(k, _)| k == "--plugin")
        .map(|(_, v)| PathBuf::from(v))
        .collect();
    opts.fast_regs = a.option(&["--fast-regs"]).is_some()
        || std::env::var("KIRIKAE_FAST_REGS").is_ok_and(|v| v == "1");
    opts.split = match a.option(&["--split"]) {
        Some(n) => Some(
            n.parse()
                .map_err(|_| format!("bad DLL count `{n}` (expected 2 or more)"))?,
        ),
        None => None,
    };
    if opts.fast_regs && opts.backend != Backend::Cranelift {
        return Err("--fast-regs needs the Cranelift backend".into());
    }
    let runs = profile_loop_arguments(args)?;
    if runs.is_some() && opts.backend != Backend::Cranelift {
        return Err("--profile-loop needs the Cranelift backend".into());
    }
    if let Some((runs, run_seconds)) = runs {
        return profile_loop(&source, &mut opts, runs, run_seconds);
    }
    let started = std::time::Instant::now();
    let report = build(&source, &opts)?;
    let dlls = match report.dlls {
        0 | 1 => String::new(),
        n => format!(" and {n} code DLLs"),
    };
    println!(
        "kirikae: recompiled {} functions ({} of {} instructions reachable) into {}{dlls} in {:.0}s",
        report.functions,
        report.reachable,
        report.total,
        report.executable.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

/// Parse and validate `--profile-loop <runs>` with its `--run-seconds` bound.
///
/// `None` means no loop was asked for: the game is never started without that flag.
fn profile_loop_arguments(args: &[String]) -> Result<Option<(usize, f64)>, String> {
    let a = parse(args)?;
    let runs = match a.option(&["--profile-loop"]) {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| format!("bad profile-loop count `{v}`"))?,
        None => return Ok(None),
    };
    if runs == 0 {
        return Err("--profile-loop needs at least one run".into());
    }
    let run_seconds = match a.option(&["--run-seconds"]) {
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| format!("bad run bound `{v}`"))?,
        None => 60.0,
    };
    if run_seconds < 10.0 {
        return Err("--run-seconds must be at least 10: the stats window is 10 s".into());
    }
    Ok(Some((runs, run_seconds)))
}

/// What one bounded headless run measured.
struct RunOutcome {
    /// Interpreted instructions, from the last stats line (a run shorter than one window has no
    /// stats line, so this stays `None`).
    interpreted: Option<u64>,
    frames: Option<u64>,
    /// Seconds from spawn until the first presented frame, seen as the frame dump the run was
    /// asked to write.
    first_frame: Option<f64>,
    exit: Option<i32>,
    log: PathBuf,
}

/// The last number written before `marker`.
fn number_before(text: &str, marker: &str) -> Option<u64> {
    let head = &text[..text.rfind(marker)?];
    head.split(|c: char| !c.is_ascii_digit())
        .next_back()
        .filter(|s| !s.is_empty())?
        .parse()
        .ok()
}

/// Read the counters out of the runtime's log.
///
/// The periodic line is `... [  10s] N frames (..) ... M interpreted`; a clean exit adds
/// `... N recompiled function calls, M interpreted instructions, exit code X`. The Vulkan
/// summary also says "0 frames through the swapchain", which is why the frame marker includes
/// the parenthesis.
fn parse_run(log: &str) -> (Option<u64>, Option<u64>) {
    (
        number_before(log, " interpreted"),
        number_before(log, " frames ("),
    )
}

/// Run the built executable headless for at most `seconds`, with a profile and a first-frame
/// dump, and read back what it recorded.
///
/// The runtime flushes the profile every ten seconds and the stats line is written at the same
/// interval, so a shorter bound would measure nothing; `profile_loop_arguments` rejects that.
/// Killing the process at the bound loses the last few seconds of recorded addresses, which is why
/// the profile is only evidence for the next build and never trusted for the bounds of this one.
fn run_headless(
    exe: &Path,
    profile: &Path,
    seconds: f64,
    dir: &Path,
) -> Result<RunOutcome, String> {
    let frames = dir.join("frames");
    let _ = std::fs::remove_dir_all(&frames);
    std::fs::create_dir_all(&frames).map_err(|e| format!("{}: {e}", frames.display()))?;
    let log_path = dir.join("run.log");
    let log =
        std::fs::File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let err = log
        .try_clone()
        .map_err(|e| format!("{}: {e}", log_path.display()))?;
    let mut child = std::process::Command::new(exe)
        .current_dir(exe.parent().unwrap_or_else(|| Path::new(".")))
        // The run is what turns a profile into a rebuild, so it is always explicit: the caller
        // asked for `--profile-loop`.
        .env("KIRIKAE_HEADLESS", "1")
        .env("KIRIKAE_MUTE", "1")
        .env("KIRIKAE_STATS", "1")
        .env("KIRIKAE_PROFILE", profile)
        .env("KIRIKAE_FRAME_DUMP", &frames)
        .env("KIRIKAE_FRAME_DUMP_EVERY", "1")
        .env("KIRIKAE_FRAME_DUMP_FROM", "1")
        .env("KIRIKAE_FRAME_DUMP_COUNT", "1")
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("{}: {e}", exe.display()))?;

    // `display::write_dump` names the frame it was asked for `frameNNNNNN.png`; asking for the
    // first one and watching for it is the only clock the runtime exposes for the first frame.
    let frame = frames.join("frame000001.png");
    let started = std::time::Instant::now();
    let mut first_frame = None;
    let mut exit = None;
    loop {
        if first_frame.is_none() && frame.is_file() {
            first_frame = Some(started.elapsed().as_secs_f64());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                exit = status.code();
                break;
            }
            Ok(None) => {}
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{}: {e}", exe.display()));
            }
        }
        if started.elapsed().as_secs_f64() >= seconds {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let timed_out = exit.is_none();
    if timed_out {
        let _ = child.kill();
        let _ = child.wait();
        if first_frame.is_none() && frame.is_file() {
            first_frame = Some(started.elapsed().as_secs_f64());
        }
    }
    let text = std::fs::read_to_string(&log_path).unwrap_or_default();
    let (interpreted, frames) = parse_run(&text);
    Ok(RunOutcome {
        interpreted,
        frames,
        first_frame,
        exit,
        log: log_path,
    })
}

/// Build, run, rebuild with what the run interpreted, until the count stops falling.
///
/// The profile is the same file the runtime writes, so every run adds to it and every build reads
/// the union: nothing is thrown away between iterations, and the loop stops as soon as a build
/// has no new address to compile or the interpreted count does not fall.
fn profile_loop(
    source: &kirikae_loader::package::Source,
    opts: &mut BuildOptions,
    runs: usize,
    run_seconds: f64,
) -> Result<ExitCode, String> {
    let out_dir = std::path::absolute(&opts.out_dir).map_err(|e| e.to_string())?;
    let name = kirikae_recomp::project::package_name(&opts.name);
    let profile = std::path::absolute(
        opts.profile
            .clone()
            .unwrap_or_else(|| out_dir.join(format!("{name}.profile"))),
    )
    .map_err(|e| e.to_string())?;
    let loop_dir = out_dir.join(format!("{name}-profile-loop"));
    // The first iteration has nothing to compile yet; an empty profile keeps `--profile` honest
    // (an explicit file that must exist) and lets the runtime merge the first run into it.
    if !profile.exists() {
        kirikae_recomp::profile::Profile::default().write(&profile)?;
    }
    opts.profile = Some(profile.clone());

    println!(
        "kirikae: profile loop of {runs} run(s) on {} ({run_seconds}s each), profile {}",
        out_dir.display(),
        profile.display()
    );
    let mut previous: Option<u64> = None;
    let mut first: Option<(Option<u64>, Option<f64>)> = None;
    let mut last: (Option<u64>, Option<f64>) = (None, None);
    let mut stop = String::from("the run budget ended");
    let mut used = 0;
    let mut last_report = None;
    let mut build_total = 0.0;
    for i in 1..=runs {
        let before = kirikae_recomp::profile::Profile::read(&profile)?;
        let started = std::time::Instant::now();
        let report = build(source, opts)?;
        let build_seconds = started.elapsed().as_secs_f64();
        build_total += build_seconds;
        let executable = report.executable.clone();
        last_report = Some(report);
        let dir = loop_dir.join(format!("run-{i}"));
        let outcome = run_headless(&executable, &profile, run_seconds, &dir)?;
        let after = kirikae_recomp::profile::Profile::read(&profile)?;
        let new = after.added_since(&before);
        used = i;
        println!(
            "kirikae: profile loop {i}/{runs}: compiled {} addresses, new {}, interpreted {}, build {build_seconds:.1}s, first frame {}{}",
            before.len(),
            new.len(),
            outcome
                .interpreted
                .map_or("-".to_string(), |n| n.to_string()),
            outcome
                .first_frame
                .map_or("-".to_string(), |s| format!("{s:.1}s")),
            match outcome.exit {
                Some(code) => format!(
                    " (exited {code}, {} frames; {})",
                    outcome.frames.unwrap_or(0),
                    outcome.log.display()
                ),
                None => format!(
                    " (timed out at {run_seconds}s, {} frames; {})",
                    outcome.frames.unwrap_or(0),
                    outcome.log.display()
                ),
            }
        );
        if first.is_none() {
            first = Some((outcome.interpreted, outcome.first_frame));
        }
        last = (outcome.interpreted, outcome.first_frame);
        if let Some(reason) = loop_stop(previous, outcome.interpreted, new.len()) {
            stop = reason;
            break;
        }
        previous = outcome.interpreted.or(previous);
    }
    let final_profile = kirikae_recomp::profile::Profile::read(&profile)?;
    let (first_interpreted, first_frame) = first.unwrap_or((None, None));
    if let Some(report) = last_report {
        let dlls = match report.dlls {
            0 | 1 => String::new(),
            n => format!(" and {n} code DLLs"),
        };
        println!(
            "kirikae: recompiled {} functions ({} of {} instructions reachable) into {}{dlls} in {build_total:.0}s",
            report.functions,
            report.instructions,
            report.total,
            report.executable.display(),
        );
    }
    println!(
        "kirikae: profile loop stopped after {used} of {runs} run(s) ({stop}): interpreted {}, addresses {}, first frame {}",
        pair(first_interpreted, last.0),
        final_profile.len(),
        pair(
            first_frame.map(|s| format!("{s:.1}s")),
            last.1.map(|s| format!("{s:.1}s")),
        ),
    );
    Ok(ExitCode::SUCCESS)
}

/// `a -> b`, with `-` for a number the run did not measure.
fn pair<T: std::fmt::Display>(first: Option<T>, last: Option<T>) -> String {
    match (first, last) {
        (None, None) => "-".to_string(),
        (first, last) => format!(
            "{} -> {}",
            first.map_or_else(|| "-".to_string(), |v| v.to_string()),
            last.map_or_else(|| "-".to_string(), |v| v.to_string()),
        ),
    }
}

/// Why the loop stops after a run, if it stops.
///
/// The count only compares when both runs measured one (a crash before the first stats line does
/// not), and a build that has nothing new to compile cannot do better than the last one.
fn loop_stop(previous: Option<u64>, now: Option<u64>, new_addresses: usize) -> Option<String> {
    if let (Some(prev), Some(now)) = (previous, now)
        && now >= prev
    {
        return Some(format!(
            "the interpreted count stopped falling ({prev} -> {now})"
        ));
    }
    (new_addresses == 0).then(|| "no new addresses".to_string())
}

fn cmd_run(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let exe = kirikae_loader::open(Path::new(a.input()?)).map_err(|e| e.to_string())?;
    if exe.arch.is_32_bit() {
        eprintln!(
            "kirikae: interpreting {} (ARM mode with VFP and the NEON slice; Thumb, banked modes and the remaining coprocessor forms are not implemented)",
            exe.arch.name()
        );
    }
    let r = kirikae_rt::run_executable(
        &exe,
        &[],
        kirikae_rt::RunOptions {
            capture_output: false,
        },
    )?;
    if std::env::var_os("KIRIKAE_STATS").is_some() {
        eprintln!(
            "kirikae: {} interpreted instructions, exit code {}",
            r.interpreted, r.exit_code
        );
    }
    Ok(ExitCode::from(r.exit_code as u8))
}

fn cmd_info(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let exe = kirikae_loader::open(Path::new(a.input()?)).map_err(|e| e.to_string())?;
    if let Some(t) = exe.title_id {
        println!("title  {t:016x}");
    }
    if let Some(s) = exe.main_stack_size {
        println!("stack  {s:#x} (main thread)");
    }
    if exe.system_resource_size != 0 {
        println!(
            "sysres {:#x} (svcMapPhysicalMemory allowed)",
            exe.system_resource_size
        );
    }
    if let Some(r) = &exe.romfs {
        match kirikae_loader::romfs::Reader::open(r) {
            Ok(rd) => println!(
                "romfs  {:#x} bytes{} at {:#x} in {}",
                rd.size(),
                if r.compression.is_some() {
                    " (compressed)"
                } else {
                    ""
                },
                r.offset,
                r.path.display()
            ),
            Err(e) => println!("romfs  unreadable: {e}"),
        }
    }
    println!(
        "entry  {:#x}
",
        exe.entry
    );
    let mut total = Coverage::default();
    for m in &exe.modules {
        let image = &m.image;
        println!("{} @ {:#x}..{:#x}", m.name, image.base, image.end());
        for (name, r) in [
            ("text", &image.text),
            ("ro", &image.ro),
            ("data", &image.data),
            ("bss", &image.bss),
        ] {
            println!(
                "  {name:<5} {:#x}..{:#x} ({:#x} bytes)",
                r.start,
                r.end,
                r.end - r.start
            );
        }
        let cov = coverage(image);
        println!(
            "  decoder: {} words, {:.2}% lifted in IR, {:.2}% SIMD/FP via helper, {:.2}% SIMD/FP unsupported, {:.2}% other",
            cov.words,
            cov.pct(cov.lifted),
            cov.pct(cov.simd_helper),
            cov.pct(cov.simd_failed),
            cov.pct(cov.other)
        );
        if exe.modules.len() == 1 {
            let prog = discover(image);
            println!(
                "
{} functions reachable from the entry:",
                prog.functions.len()
            );
            for f in &prog.functions {
                println!(
                    "  {:#x}  {} blocks, {} instructions",
                    f.entry,
                    f.blocks.len(),
                    f.insn_count()
                );
            }
        }
        total.add(&cov);
    }
    if exe.modules.len() > 1 {
        println!(
            "
all code: {} words, {:.2}% lifted in IR, {:.2}% SIMD/FP via helper, {:.2}% SIMD/FP unsupported, {:.2}% other",
            total.words,
            total.pct(total.lifted),
            total.pct(total.simd_helper),
            total.pct(total.simd_failed),
            total.pct(total.other)
        );
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Default)]
struct Coverage {
    words: u64,
    lifted: u64,
    simd_helper: u64,
    simd_failed: u64,
    other: u64,
}

impl Coverage {
    fn pct(&self, n: u64) -> f64 {
        100.0 * n as f64 / self.words.max(1) as f64
    }

    fn add(&mut self, o: &Coverage) {
        self.words += o.words;
        self.lifted += o.lifted;
        self.simd_helper += o.simd_helper;
        self.simd_failed += o.simd_failed;
        self.other += o.other;
    }
}

fn is_simd_fp(w: u32) -> bool {
    let op0 = (w >> 25) & 0xf;
    op0 & 0b0111 == 0b0111 || (op0 & 0b0101 == 0b0100 && w & (1 << 26) != 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// The decoder produced IR the interpreter and the backends know how to run.
    Lifted,
    /// A SIMD/FP word that only exists as a call into the `a64` runtime helper.
    SimdHelper,
    /// A SIMD/FP word the decoder does not know, so running it faults the guest.
    SimdUnsupported,
    /// Anything else the decoder does not know.
    Other,
}

fn classify(insn: &Insn) -> Class {
    if insn.stmts.iter().any(|s| matches!(s, Stmt::Simd(_))) {
        Class::SimdHelper
    } else if insn.control() != Control::Undefined {
        Class::Lifted
    } else if is_simd_fp(insn.word) {
        Class::SimdUnsupported
    } else {
        Class::Other
    }
}

fn coverage(image: &Image) -> Coverage {
    let mut c = Coverage::default();
    let mut addr = image.text.start;
    while addr < image.text.end {
        let insn = lift(addr, image.word(addr).unwrap_or(0));
        c.words += 1;
        match classify(&insn) {
            Class::Lifted => c.lifted += 1,
            Class::SimdHelper => c.simd_helper += 1,
            Class::SimdUnsupported => c.simd_failed += 1,
            Class::Other => c.other += 1,
        }
        addr += 4;
    }
    c
}

fn cmd_emit(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let image = load_image(&read(a.input()?)?)?;
    let source = emit_program(&discover(&image), "image.bin");
    match a.option(&["-o", "--out"]) {
        Some(path) => std::fs::write(path, source).map_err(|e| format!("{path}: {e}"))?,
        None => print!("{source}"),
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_testgen(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let path = a.input()?;
    std::fs::write(path, kirikae_testkit::hello_nro()).map_err(|e| format!("{path}: {e}"))?;
    println!("kirikae: wrote {path}");
    Ok(ExitCode::SUCCESS)
}

fn cmd_dev(args: &[String]) -> Result<ExitCode, String> {
    use kirikae_a64::simd::oracle;
    match args.first().map(String::as_str) {
        Some("simd-cases") => {
            let out = args.get(1).ok_or("missing output file")?;
            let n: usize = args.get(2).ok_or("missing count")?.parse().map_err(|_| "bad count")?;
            let seed: u64 = args.get(3).map_or(Ok(1), |s| s.parse()).map_err(|_| "bad seed")?;
            std::fs::write(out, oracle::generate(n, seed)).map_err(|e| format!("{out}: {e}"))?;
            Ok(ExitCode::SUCCESS)
        }
        Some("simd-check") => {
            let cases = read(args.get(1).ok_or("missing cases file")?)?;
            let results = read(args.get(2).ok_or("missing results file")?)?;
            let r = oracle::check(&cases, &results);
            println!("compared {} cases: {} match, {} differ", r.compared, r.matched, r.mismatches.len());
            println!("{} encodings accepted here but undefined on the A57", r.over_accepting.len());
            let mut by_group = std::collections::BTreeMap::<String, Vec<(u32, String)>>::new();
            for (w, d) in &r.mismatches {
                by_group.entry(oracle::group(*w)).or_default().push((*w, d.clone()));
            }
            let mut groups: Vec<_> = by_group.into_iter().collect();
            groups.sort_by_key(|g| std::cmp::Reverse(g.1.len()));
            for (g, list) in groups.iter().take(40) {
                println!("  mismatch {g}: {} (e.g. {:#010x}: {})", list.len(), list[0].0, list[0].1);
            }
            let mut over = std::collections::BTreeMap::<String, (usize, u32)>::new();
            for w in &r.over_accepting {
                let e = over.entry(oracle::group(*w)).or_insert((0, *w));
                e.0 += 1;
            }
            let mut over: Vec<_> = over.into_iter().collect();
            over.sort_by_key(|g| std::cmp::Reverse(g.1.0));
            for (g, (n, w)) in over.iter().take(15) {
                println!("  over-accepting {g}: {n} (e.g. {w:#010x})");
            }
            let total_missing: usize = r.missing.iter().map(|m| m.1).sum();
            println!("{total_missing} encodings the A57 runs that kirikae does not (top groups):");
            for (g, n, w) in r.missing.iter().take(25) {
                println!("  missing {g}: {n} (e.g. {w:#010x})");
            }
            Ok(if r.mismatches.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE })
        }
        Some("romfs-ls") => {
            let exe = kirikae_loader::open(Path::new(args.get(1).ok_or("missing input")?)).map_err(|e| e.to_string())?;
            let romfs = exe.romfs.ok_or("this executable has no RomFS")?;
            let mut r = kirikae_loader::romfs::Reader::open(&romfs).map_err(|e| e.to_string())?;
            let limit: usize = args.get(2).map_or(Ok(30), |s| s.parse()).map_err(|_| "bad count")?;
            let all = kirikae_loader::romfs::list(&mut r, usize::MAX).map_err(|e| e.to_string())?;
            let total: u64 = all.iter().map(|f| f.size).sum();
            println!("{} files, {:.2} GiB", all.len(), total as f64 / (1u64 << 30) as f64);
            for f in all.iter().take(limit) {
                let mut magic = [0u8; 8];
                r.read_at(f.offset, &mut magic).map_err(|e| e.to_string())?;
                let shown: String = magic.iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect();
                println!("  {:>12}  {shown}  {}", f.size, f.path);
            }
            Ok(ExitCode::SUCCESS)
        }
        Some("romfs-find") => {
            let exe = kirikae_loader::open(Path::new(args.get(1).ok_or("missing input")?)).map_err(|e| e.to_string())?;
            let romfs = exe.romfs.ok_or("this executable has no RomFS")?;
            let mut r = kirikae_loader::romfs::Reader::open(&romfs).map_err(|e| e.to_string())?;
            for path in &args[2..] {
                match kirikae_loader::romfs::lookup(&mut r, path).map_err(|e| e.to_string())? {
                    Some(f) => println!("{:>12} at {:#x}  {}", f.size, f.offset, f.path),
                    None => println!("{:>12}  {path}", "not found"),
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Some("extract") => {
            let input = Path::new(args.get(1).ok_or("missing input")?);
            let out = Path::new(args.get(2).ok_or("missing output folder")?);
            let source = kirikae_loader::package::read_source(input).map_err(|e| e.to_string())?;
            let kirikae_loader::Source::ExeFs(files, _, romfs) = source else { return Err("not a game dump with an ExeFS".into()) };
            let exefs = out.join("exefs");
            std::fs::create_dir_all(&exefs).map_err(|e| format!("{}: {e}", exefs.display()))?;
            for (name, bytes) in &files {
                std::fs::write(exefs.join(name), bytes).map_err(|e| format!("{name}: {e}"))?;
            }
            let mut count = 0;
            if let Some(romfs) = romfs {
                let mut r = kirikae_loader::romfs::Reader::open(&romfs).map_err(|e| e.to_string())?;
                for f in kirikae_loader::romfs::list(&mut r, usize::MAX).map_err(|e| e.to_string())? {
                    let dest = out.join("romfs").join(f.path.trim_start_matches('/'));
                    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
                    let mut w = std::io::BufWriter::new(std::fs::File::create(&dest).map_err(|e| format!("{}: {e}", dest.display()))?);
                    let mut buf = vec![0u8; 1 << 20];
                    let mut done = 0;
                    while done < f.size {
                        let n = (f.size - done).min(buf.len() as u64) as usize;
                        r.read_at(f.offset + done, &mut buf[..n]).map_err(|e| e.to_string())?;
                        std::io::Write::write_all(&mut w, &buf[..n]).map_err(|e| e.to_string())?;
                        done += n as u64;
                    }
                    count += 1;
                }
            }
            println!("kirikae: wrote {} ExeFS files and {count} RomFS files to {}", files.len(), out.display());
            Ok(ExitCode::SUCCESS)
        }
        Some("module-dump") => {
            let exe = kirikae_loader::open(Path::new(args.get(1).ok_or("missing input")?)).map_err(|e| e.to_string())?;
            let name = args.get(2).ok_or("missing module name")?;
            let out = args.get(3).ok_or("missing output file")?;
            let m = exe.modules.iter().find(|m| &m.name == name).ok_or_else(|| {
                let names: Vec<&str> = exe.modules.iter().map(|m| m.name.as_str()).collect();
                format!("no module `{name}` (modules: {})", names.join(", "))
            })?;
            std::fs::write(out, &m.image.memory).map_err(|e| format!("{out}: {e}"))?;
            println!("{name}: base {:#x}, {} bytes, text {:#x?}", m.image.base, m.image.memory.len(), m.image.text);
            Ok(ExitCode::SUCCESS)
        }
        Some("syms") => {
            let exe = kirikae_loader::open(Path::new(args.get(1).ok_or("missing input")?)).map_err(|e| e.to_string())?;
            let pattern = args.get(2).map_or("", String::as_str);
            for m in &exe.modules {
                for (addr, name) in kirikae_loader::dynamic::function_symbols(&m.image) {
                    if name.contains(pattern) {
                        println!("{}+{:#x}  {name}", m.name, addr - m.image.base);
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        _ => Err("dev commands: simd-cases, simd-check, romfs-ls, romfs-find, extract, module-dump, syms".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_counters_out_of_a_run_log() {
        let text = "kirikae: [  10s] 120 frames (12.0 fps, 60.0 without the limit, longest 8300 ms), GPU: 1 submits, 2 draws, 0 clears, 0 copies, 0 dispatches; 55 native calls, 2713811 interpreted\n\
                    kirikae:         vulkan: 0.0s of GPU work, 0 frames through the swapchain, 0 batches\n\
                    kirikae: [  20s] 240 frames (12.0 fps, 60.0 without the limit, longest 1300 ms), GPU: 1 submits, 2 draws, 0 clears, 0 copies, 0 dispatches; 60 native calls, 3400000 interpreted\n";
        assert_eq!(
            parse_run(text),
            (Some(3_400_000), Some(240)),
            "the Vulkan summary's '0 frames through the swapchain' is not the counter"
        );
        let exit =
            "kirikae: 900 recompiled function calls, 12 interpreted instructions, exit code 0\n";
        assert_eq!(parse_run(exit), (Some(12), None));
        assert_eq!(parse_run("no stats yet\n"), (None, None));
    }

    #[test]
    fn the_loop_stops_when_the_count_stops_falling() {
        assert_eq!(
            loop_stop(None, Some(100), 3),
            None,
            "first run cannot stop it"
        );
        assert_eq!(loop_stop(Some(100), Some(90), 3), None, "still falling");
        assert_eq!(
            loop_stop(Some(100), Some(100), 3).unwrap(),
            "the interpreted count stopped falling (100 -> 100)"
        );
        assert_eq!(
            loop_stop(Some(100), Some(140), 0).unwrap(),
            "the interpreted count stopped falling (100 -> 140)"
        );
        assert_eq!(
            loop_stop(Some(100), None, 0).unwrap(),
            "no new addresses",
            "a crash before the stats line still stops when nothing is new"
        );
        assert_eq!(loop_stop(None, None, 2), None);
    }

    #[test]
    fn pairs_read_as_before_and_after() {
        assert_eq!(pair(Some(10), Some(0)), "10 -> 0");
        assert_eq!(pair(Some(10), None), "10 -> -");
        assert_eq!(pair(None, Some(5)), "- -> 5");
        assert_eq!(pair::<u64>(None, None), "-");
    }

    #[test]
    fn a_profile_loop_needs_a_positive_count_and_a_stats_window() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            profile_loop_arguments(&args(&["--profile-loop", "0"])).unwrap_err(),
            "--profile-loop needs at least one run"
        );
        assert!(
            profile_loop_arguments(&args(&["--profile-loop", "2", "--run-seconds", "5"]))
                .unwrap_err()
                .contains("at least 10")
        );
        assert_eq!(
            profile_loop_arguments(&args(&["--profile-loop", "3"])).unwrap(),
            Some((3, 60.0))
        );
        assert_eq!(
            profile_loop_arguments(&args(&["--profile-loop", "2", "--run-seconds", "90"])).unwrap(),
            Some((2, 90.0))
        );
        assert_eq!(profile_loop_arguments(&args(&["-o", "out"])).unwrap(), None);
    }

    #[test]
    fn classifies_ir_helper_unsupported_and_other() {
        let scalar = lift(0, 0x8b02_0020); // add x0, x1, x2
        assert_eq!(classify(&scalar), Class::Lifted);

        let fp_ir = lift(0, 0x1e22_2820); // fadd s0, s1, s2, lifted to IR like any other op
        assert!(is_simd_fp(fp_ir.word));
        assert_eq!(classify(&fp_ir), Class::Lifted);

        let neon_ir = lift(0, 0x4e22_8420); // add v0.4s, v1.4s, v2.4s, expanded to lane IR
        assert!(is_simd_fp(neon_ir.word));
        assert_eq!(classify(&neon_ir), Class::Lifted);

        let helper = lift(0, 0x4e02_0020); // tbl v0.16b, {v1.16b}, v2.16b, run by the a64 helper
        assert_eq!(helper.stmts, vec![Stmt::Simd(0x4e02_0020)]);
        assert_eq!(classify(&helper), Class::SimdHelper);

        let unsupported = lift(0, 0x1e00_0000); // unallocated SIMD/FP encoding
        assert_eq!(unsupported.control(), Control::Undefined);
        assert!(is_simd_fp(unsupported.word));
        assert_eq!(classify(&unsupported), Class::SimdUnsupported);

        let other = lift(0, 0x0000_0000);
        assert_eq!(classify(&other), Class::Other);
    }

    #[test]
    fn coverage_percentages_add_up() {
        let one = Coverage {
            words: 4,
            lifted: 2,
            simd_helper: 1,
            simd_failed: 0,
            other: 1,
        };
        let mut c = Coverage::default();
        c.add(&one);
        c.add(&one);
        assert_eq!(
            (
                c.words,
                c.pct(c.lifted),
                c.pct(c.simd_helper),
                c.pct(c.simd_failed),
                c.pct(c.other)
            ),
            (8, 50.0, 25.0, 0.0, 25.0)
        );
    }
}
