use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kirikae_a64::{lift, Control};
use kirikae_loader::{load, LOAD_BASE};
use kirikae_recomp::project::{build, Backend, BuildOptions};
use kirikae_recomp::rust_backend::emit_program;
use kirikae_recomp::{discover, Image};

const USAGE: &str = "\
kirikae: recompile Nintendo Switch executables into native programs

usage:
  kirikae build <file> [-o <dir>] [--name <name>]       recompile into a native executable
        [--backend cranelift|rust] [-j <jobs>]
        [--fast-regs]                                keep guest registers in variables
                                                     (experimental; or KIRIKAE_FAST_REGS=1)
        [--plugin <crate dir>]                       link a crate whose install() runs first
        [--profile <file>]                           also compile the addresses a run had to
                                                     interpret (default: <dir>/<name>.profile,
                                                     written by the executable)
        [--no-profile]                               ignore any profile, including the default
                                                     one; fails with --profile
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

const SWITCHES: &[&str] = &["--fast-regs", "--no-profile"];

struct Args {
    positional: Vec<String>,
    options: Vec<(String, String)>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut out = Args { positional: Vec::new(), options: Vec::new() };
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
        self.positional.first().map(String::as_str).ok_or_else(|| format!("missing input file\n\n{USAGE}"))
    }

    fn option(&self, names: &[&str]) -> Option<&str> {
        self.options.iter().find(|(k, _)| names.contains(&k.as_str())).map(|(_, v)| v.as_str())
    }
}

fn read(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

fn load_image(file: &[u8]) -> Result<Image, String> {
    load(file, LOAD_BASE).map_err(|e| e.to_string())
}

fn stem(path: &str) -> String {
    Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "game".into())
}

fn cmd_build(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let input = a.input()?;
    let source = kirikae_loader::package::read_source(Path::new(input)).map_err(|e| e.to_string())?;
    let name = a.option(&["--name"]).map(String::from).unwrap_or_else(|| stem(input));
    let mut opts = BuildOptions::new(PathBuf::from(a.option(&["-o", "--out"]).unwrap_or(".")), &name);
    opts.backend = match a.option(&["--backend"]) {
        None | Some("cranelift") => Backend::Cranelift,
        Some("rust") => Backend::Rust,
        Some(other) => return Err(format!("unknown backend `{other}` (cranelift or rust)")),
    };
    if let Some(j) = a.option(&["-j", "--jobs"]) {
        opts.jobs = j.parse().map_err(|_| format!("bad job count `{j}`"))?;
    }
    (opts.profile, opts.no_profile) = profile_options(&a)?;
    opts.plugins = a.options.iter().filter(|(k, _)| k == "--plugin").map(|(_, v)| PathBuf::from(v)).collect();
    opts.fast_regs = a.option(&["--fast-regs"]).is_some() || std::env::var("KIRIKAE_FAST_REGS").is_ok_and(|v| v == "1");
    if opts.fast_regs && opts.backend != Backend::Cranelift {
        return Err("--fast-regs needs the Cranelift backend".into());
    }
    let started = std::time::Instant::now();
    let report = build(&source, &opts)?;
    println!(
        "kirikae: recompiled {} functions ({} of {} instructions reachable) into {} in {:.0}s",
        report.functions,
        report.reachable,
        report.total,
        report.executable.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(ExitCode::SUCCESS)
}

fn profile_options(a: &Args) -> Result<(Option<PathBuf>, bool), String> {
    let profile = a.option(&["--profile"]).map(PathBuf::from);
    let no_profile = a.option(&["--no-profile"]).is_some();
    if no_profile && profile.is_some() {
        return Err("--no-profile cannot be combined with --profile".into());
    }
    Ok((profile, no_profile))
}

fn cmd_run(args: &[String]) -> Result<ExitCode, String> {
    let a = parse(args)?;
    let exe = kirikae_loader::open(Path::new(a.input()?)).map_err(|e| e.to_string())?;
    let r = kirikae_rt::run_executable(&exe, &[], kirikae_rt::RunOptions { capture_output: false })?;
    if std::env::var_os("KIRIKAE_STATS").is_some() {
        eprintln!("kirikae: {} interpreted instructions, exit code {}", r.interpreted, r.exit_code);
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
        println!("sysres {:#x} (svcMapPhysicalMemory allowed)", exe.system_resource_size);
    }
    if let Some(r) = &exe.romfs {
        match kirikae_loader::romfs::Reader::open(r) {
            Ok(rd) => println!(
                "romfs  {:#x} bytes{} at {:#x} in {}",
                rd.size(),
                if r.compression.is_some() { " (compressed)" } else { "" },
                r.offset,
                r.path.display()
            ),
            Err(e) => println!("romfs  unreadable: {e}"),
        }
    }
    println!("entry  {:#x}
", exe.entry);
    let mut total = Coverage::default();
    for m in &exe.modules {
        let image = &m.image;
        println!("{} @ {:#x}..{:#x}", m.name, image.base, image.end());
        for (name, r) in [("text", &image.text), ("ro", &image.ro), ("data", &image.data), ("bss", &image.bss)] {
            println!("  {name:<5} {:#x}..{:#x} ({:#x} bytes)", r.start, r.end, r.end - r.start);
        }
        let cov = coverage(image);
        println!(
            "  decoder: {} words, {:.2}% lifted, {:.2}% SIMD/FP (not lifted yet), {:.2}% other",
            cov.words,
            cov.pct(cov.lifted),
            cov.pct(cov.simd),
            cov.pct(cov.other)
        );
        if exe.modules.len() == 1 {
            let prog = discover(image);
            println!("
{} functions reachable from the entry:", prog.functions.len());
            for f in &prog.functions {
                println!("  {:#x}  {} blocks, {} instructions", f.entry, f.blocks.len(), f.insn_count());
            }
        }
        total.add(&cov);
    }
    if exe.modules.len() > 1 {
        println!(
            "
all code: {} words, {:.2}% lifted, {:.2}% SIMD/FP, {:.2}% other",
            total.words,
            total.pct(total.lifted),
            total.pct(total.simd),
            total.pct(total.other)
        );
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Default)]
struct Coverage {
    words: u64,
    lifted: u64,
    simd: u64,
    other: u64,
}

impl Coverage {
    fn pct(&self, n: u64) -> f64 {
        100.0 * n as f64 / self.words.max(1) as f64
    }

    fn add(&mut self, o: &Coverage) {
        self.words += o.words;
        self.lifted += o.lifted;
        self.simd += o.simd;
        self.other += o.other;
    }
}

fn is_simd_fp(w: u32) -> bool {
    let op0 = (w >> 25) & 0xf;
    op0 & 0b0111 == 0b0111 || (op0 & 0b0101 == 0b0100 && w & (1 << 26) != 0)
}

fn coverage(image: &Image) -> Coverage {
    let mut c = Coverage::default();
    let mut addr = image.text.start;
    while addr < image.text.end {
        let w = image.word(addr).unwrap_or(0);
        c.words += 1;
        if lift(addr, w).control() != Control::Undefined {
            c.lifted += 1;
        } else if is_simd_fp(w) {
            c.simd += 1;
        } else {
            c.other += 1;
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

    fn parse_args(list: &[&str]) -> Args {
        let args: Vec<String> = list.iter().map(|s| (*s).to_string()).collect();
        parse(&args).unwrap()
    }

    #[test]
    fn no_profile_is_a_switch() {
        let a = parse_args(&["game.nro", "--no-profile"]);
        assert_eq!(a.positional, vec!["game.nro"]);
        assert_eq!(profile_options(&a).unwrap(), (None, true));
    }

    #[test]
    fn no_profile_rejects_profile_flags() {
        let a = parse_args(&["game.nro", "--no-profile", "--profile", "p"]);
        assert!(profile_options(&a).unwrap_err().contains("--profile"));
    }
}
