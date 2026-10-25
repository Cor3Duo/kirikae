use kirikae_a64::simd::exec;
use kirikae_a64::simd::oracle::{State, mem_pattern};

fn state(fpcr: u64) -> State {
    State {
        x: [0; 31],
        sp: 0,
        v: [0; 32],
        nzcv: 0,
        fpcr,
        fpsr: 0,
        mem: mem_pattern(),
    }
}

const FZ: u64 = 1 << 24;
const DN: u64 = 1 << 25;

#[test]
fn frsqrte_even_biased_exponent() {
    let mut s = state(0);
    s.v[29] = 0x2707_f2cf_f1c2_72af;
    assert!(exec(&mut s, 0x2ea1_dba1));
    assert_eq!(s.v[1] >> 32 & 0xffff_ffff, 0x4bb0_0000);
}

#[test]
fn frsqrts_negates_before_nan_processing() {
    let mut s = state(FZ);
    s.v[0] = 0x7fc0_0000_2d22_ea5f;
    s.v[10] = 0x0000_0001_43d5_0000;
    assert!(exec(&mut s, 0x0eaa_fc0d));
    assert_eq!(s.v[13], 0xffc0_0000_3fc0_0000);
}

#[test]
fn fz_flushes_by_the_unrounded_result() {
    let mut s = state(FZ | DN);
    s.v[22] = 0x1825_3dc6_0000_0001;
    s.v[7] = 0xf96c_655a_0080_0000;
    s.v[23] = 0x0080_0000_d590_81f1;
    assert!(exec(&mut s, 0x0f87_52d7));
    assert_eq!(s.v[23] >> 32, 0);
}

#[test]
fn rbit_works_on_bytes() {
    let mut s = state(0);
    s.v[14] = 0x8000_0000_0000_0000;
    assert!(exec(&mut s, 0x2e60_59d8));
    assert_eq!(s.v[24], 0x0100_0000_0000_0000);
}

#[test]
fn sqshl_saturates_large_shifts() {
    let mut s = state(0);
    s.v[4] = (0x8000_0000_0000_0001u128) << 64;
    s.v[3] = (0x40u128) << 64;
    assert!(exec(&mut s, 0x4ee3_4c97));
    assert_eq!(s.v[23] >> 64, 0x8000_0000_0000_0000);
}
