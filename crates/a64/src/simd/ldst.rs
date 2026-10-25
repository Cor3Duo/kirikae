use super::{Env, bits, lane, set_lane, sized};

pub fn exec<E: Env + ?Sized>(e: &mut E, w: u32) -> bool {
    if w & 0xbfbf_0000 == 0x0c00_0000 || w & 0xbfa0_0000 == 0x0c80_0000 {
        return multiple(e, w);
    }
    if w & 0xbf9f_0000 == 0x0d00_0000 || w & 0xbf80_0000 == 0x0d80_0000 {
        return single(e, w);
    }
    false
}

fn base<E: Env + ?Sized>(e: &E, rn: u32) -> u64 {
    if rn == 31 { e.sp() } else { e.x(rn) }
}

fn writeback<E: Env + ?Sized>(e: &mut E, rn: u32, rm: u32, addr: u64, imm: u64) {
    let off = if rm == 31 { imm } else { e.x(rm) };
    let v = addr.wrapping_add(off);
    if rn == 31 {
        e.set_sp(v)
    } else {
        e.set_x(rn, v)
    }
}

fn multiple<E: Env + ?Sized>(e: &mut E, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let load = bits(w, 22, 1) == 1;
    let post = bits(w, 23, 1) == 1;
    let (rm, opcode, size, rn, rt) = (
        bits(w, 16, 5),
        bits(w, 12, 4),
        bits(w, 10, 2),
        bits(w, 5, 5),
        bits(w, 0, 5),
    );
    let (rpt, selem) = match opcode {
        0b0000 => (1, 4),
        0b0010 => (4, 1),
        0b0100 => (1, 3),
        0b0110 => (3, 1),
        0b0111 => (1, 1),
        0b1000 => (1, 2),
        0b1010 => (2, 1),
        _ => return false,
    };
    if size == 3 && q == 0 && selem != 1 {
        return false;
    }
    let es = 8 << size;
    let ebytes = (es / 8) as u8;
    let datasize = 64 << q;
    let elems = datasize / es;
    let addr = base(e, rn);
    let mut regs: [u128; 4] =
        std::array::from_fn(|k| if load { 0 } else { e.v((rt + k as u32) % 32) });
    let mut offs = 0u64;
    for r in 0..rpt {
        for i in 0..elems {
            for s in 0..selem {
                let reg = (r + s) as usize;
                let a = addr.wrapping_add(offs);
                if load {
                    let x = e.load(a, ebytes);
                    set_lane(&mut regs[reg], es, i, x);
                } else {
                    e.store(a, ebytes, lane(regs[reg], es, i));
                }
                offs += ebytes as u64;
            }
        }
    }
    if load {
        for (k, reg) in regs.iter().enumerate().take((rpt * selem) as usize) {
            e.set_v((rt + k as u32) % 32, sized(*reg, datasize));
        }
    }
    if post {
        writeback(e, rn, rm, addr, offs);
    }
    true
}

fn single<E: Env + ?Sized>(e: &mut E, w: u32) -> bool {
    let q = bits(w, 30, 1);
    let load = bits(w, 22, 1) == 1;
    let post = bits(w, 23, 1) == 1;
    let r = bits(w, 21, 1);
    let (rm, opcode, s, size, rn, rt) = (
        bits(w, 16, 5),
        bits(w, 13, 3),
        bits(w, 12, 1),
        bits(w, 10, 2),
        bits(w, 5, 5),
        bits(w, 0, 5),
    );
    let selem = ((opcode & 1) << 1 | r) + 1;
    let scale = opcode >> 1;
    let (es, index, replicate) = match scale {
        0 => (8, q << 3 | s << 2 | size, false),
        1 if size & 1 == 0 => (16, q << 2 | s << 1 | (size >> 1), false),
        2 if size == 0 => (32, q << 1 | s, false),
        2 if size == 1 && s == 0 => (64, q, false),
        3 if load && s == 0 => (8 << size, 0, true),
        _ => return false,
    };
    let ebytes = (es / 8) as u8;
    let addr = base(e, rn);
    let mut offs = 0u64;
    for k in 0..selem {
        let t = (rt + k) % 32;
        let a = addr.wrapping_add(offs);
        if replicate {
            let x = e.load(a, ebytes);
            let datasize = 64 << q;
            let mut v = 0u128;
            for i in 0..datasize / es {
                set_lane(&mut v, es, i, x);
            }
            e.set_v(t, v);
        } else if load {
            let x = e.load(a, ebytes);
            let mut v = e.v(t);
            set_lane(&mut v, es, index, x);
            e.set_v(t, v);
        } else {
            let x = lane(e.v(t), es, index);
            e.store(a, ebytes, x);
        }
        offs += ebytes as u64;
    }
    if post {
        writeback(e, rn, rm, addr, offs);
    }
    true
}
