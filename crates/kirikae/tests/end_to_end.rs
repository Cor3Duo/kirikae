use std::path::PathBuf;
use std::process::{Command, Output};

use kirikae_loader::Source;
use kirikae_recomp::project::{build, Backend, BuildOptions, BuildReport};
use kirikae_testkit::{hello_nro, parallel_threads_nro, threads_nro, HELLO_EXPECTED, THREADS_EXPECTED};

fn out_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kirikae-e2e-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_and_run(file: &[u8], name: &str, backend: Backend, fast_regs: bool) -> (BuildReport, Output) {
    let mut opts = BuildOptions::new(out_dir(name), name);
    opts.backend = backend;
    opts.fast_regs = fast_regs;
    let report = build(&Source::Single(file.to_vec()), &opts).unwrap();
    let run = run_built(&report, &[]);
    (report, run)
}

fn run_built(report: &BuildReport, env: &[(&str, &str)]) -> Output {
    let run = Command::new(&report.executable).env("KIRIKAE_STATS", "1").envs(env.iter().copied()).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    run
}

fn interpreted(file: &[u8]) -> String {
    let r = kirikae_rt::run(file, &[], kirikae_rt::RunOptions { capture_output: true }).unwrap();
    String::from_utf8_lossy(&r.output).into_owned()
}

fn check_hello(backend: Backend, fast_regs: bool, name: &str) {
    let file = hello_nro();
    assert_eq!(interpreted(&file), HELLO_EXPECTED);
    let (report, run) = build_and_run(&file, name, backend, fast_regs);
    assert_eq!(report.functions, 5);
    assert_eq!(String::from_utf8_lossy(&run.stdout), HELLO_EXPECTED);
    let stats = String::from_utf8_lossy(&run.stderr);
    assert!(stats.contains(" 0 interpreted instructions"), "{stats}");
    let _ = std::fs::remove_dir_all(report.executable.parent().unwrap());
}

fn check_threads(backend: Backend, fast_regs: bool, name: &str) {
    let file = threads_nro();
    assert_eq!(interpreted(&file), THREADS_EXPECTED);
    let (report, run) = build_and_run(&file, name, backend, fast_regs);
    assert_eq!(report.functions, 3);
    assert_eq!(String::from_utf8_lossy(&run.stdout), THREADS_EXPECTED);
    let stats = String::from_utf8_lossy(&run.stderr);
    assert!(stats.contains(" 0 interpreted instructions"), "{stats}");
    let _ = std::fs::remove_dir_all(report.executable.parent().unwrap());
}

fn check_parallel_threads(backend: Backend, fast_regs: bool, name: &str) {
    let file = parallel_threads_nro();
    assert_eq!(interpreted(&file), THREADS_EXPECTED);
    let (report, run) = build_and_run(&file, name, backend, fast_regs);
    assert_eq!(String::from_utf8_lossy(&run.stdout), THREADS_EXPECTED);
    for _ in 0..5 {
        let run = run_built(&report, &[("KIRIKAE_PARALLEL", "1")]);
        assert_eq!(String::from_utf8_lossy(&run.stdout), THREADS_EXPECTED);
    }
    let _ = std::fs::remove_dir_all(report.executable.parent().unwrap());
}

#[test]
fn rust_backend_parallel_threads() {
    check_parallel_threads(Backend::Rust, false, "pthreads-rust");
}

#[test]
fn cranelift_parallel_threads() {
    check_parallel_threads(Backend::Cranelift, true, "pthreads-clif-fast");
}

#[test]
fn rust_backend_hello() {
    check_hello(Backend::Rust, false, "hello-rust");
}

#[test]
fn rust_backend_threads() {
    check_threads(Backend::Rust, false, "threads-rust");
}

#[test]
fn cranelift_backend_hello() {
    check_hello(Backend::Cranelift, false, "hello-clif");
}

#[test]
fn cranelift_backend_threads() {
    check_threads(Backend::Cranelift, false, "threads-clif");
}

#[test]
fn cranelift_fast_regs_hello() {
    check_hello(Backend::Cranelift, true, "hello-clif-fast");
}

#[test]
fn cranelift_fast_regs_threads() {
    check_threads(Backend::Cranelift, true, "threads-clif-fast");
}
