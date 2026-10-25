use std::path::PathBuf;

fn temp(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("kirikae-coverage-it-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn finish_writes_and_a_second_call_does_not_duplicate() {
    let dir = temp("finish");
    let path = dir.join("game.profile");
    // SAFETY: this test binary only builds one test, so no other thread reads the environment
    // while it is changed here; the profile path is read once by coverage::start below.
    unsafe { std::env::set_var("KIRIKAE_PROFILE", &path) };

    kirikae_rt::coverage::start();
    assert!(kirikae_rt::coverage::on());
    for addr in [0x100, 0x100, 0x200, 0x300] {
        kirikae_rt::coverage::record(addr);
    }
    kirikae_rt::coverage::finish();
    assert!(!kirikae_rt::coverage::on());

    let first = kirikae_rt::coverage::parse(&std::fs::read_to_string(&path).unwrap());
    assert_eq!(first, vec![(0x100, 2), (0x200, 1), (0x300, 1)]);

    kirikae_rt::coverage::finish();
    let second = kirikae_rt::coverage::parse(&std::fs::read_to_string(&path).unwrap());
    assert_eq!(second, first);
}
