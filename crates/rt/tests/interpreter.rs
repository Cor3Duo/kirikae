use kirikae_rt::{RunOptions, run};
use kirikae_testkit::{HELLO_EXPECTED, THREADS_EXPECTED, deadlock_nro, hello_nro, threads_nro};

fn capture() -> RunOptions {
    RunOptions {
        capture_output: true,
    }
}

#[test]
fn interprets_hello() {
    let r = run(&hello_nro(), &[], capture()).unwrap();
    assert_eq!(String::from_utf8_lossy(&r.output), HELLO_EXPECTED);
    assert_eq!(r.exit_code, 0);
    assert_eq!(r.native_calls, 0);
    assert!(r.interpreted > 100);
}

#[test]
fn threads_share_a_mutex() {
    for _ in 0..5 {
        let r = run(&threads_nro(), &[], capture()).unwrap();
        assert_eq!(String::from_utf8_lossy(&r.output), THREADS_EXPECTED);
        assert_eq!(r.exit_code, 0);
    }
}

#[test]
fn reports_deadlock() {
    let r = run(&deadlock_nro(), &[], capture()).unwrap();
    assert_eq!(r.exit_code, 5);
}
