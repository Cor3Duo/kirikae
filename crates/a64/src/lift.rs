use crate::bits::{decode_bit_masks, sext};
use crate::ir::*;
use crate::ops::Op;

pub fn lift(pc: u64, word: u32) -> Insn {
    let stmts = lift_stmts(pc, word).unwrap_or_else(|| vec![Stmt::Undefined(word)]);
    Insn { addr: pc, word, stmts }
}

fn lift_stmts(pc: u64, w: u32) -> Option<Vec<Stmt>> {
    let stmts = match (w >> 25) & 0xf {
        0b1000 | 0b1001 => dp_imm(pc, w),
        0b1010 | 0b1011 => branch_sys(pc, w),
        0b0100 | 0b0110 | 0b1100 | 0b1110 => load_store(pc, w),
        0b0101 | 0b1101 => dp_reg(w),
        _ => None,
    };
    stmts.or_else(|| crate::lift_fp::lift(w)).or_else(|| crate::simd::supported(w).then(|| vec![Stmt::Simd(w)]))
}

fn field(w: u32, lo: u32, bits: u32) -> u32 {
    (w >> lo) & ((1 << bits) - 1)
}

fn rz(n: u32) -> Expr {
    if n == 31 { c(0) } else { Expr::Reg(n as u8) }
}

fn rsp(n: u32) -> Expr {
    if n == 31 { Expr::Sp } else { Expr::Reg(n as u8) }
}

fn wz(out: &mut Vec<Stmt>, n: u32, e: Expr) {
    if n != 31 {
        out.push(Stmt::SetReg(n as u8, e));
    }
}

fn wsp(out: &mut Vec<Stmt>, n: u32, e: Expr) {
    out.push(if n == 31 { Stmt::SetSp(e) } else { Stmt::SetReg(n as u8, e) });
}

fn width(e: Expr, sf: bool) -> Expr {
    if sf { e } else { op1(Op::Zext32, e) }
}

fn shift_op(kind: u32, sf: bool) -> Op {
    match (kind, sf) {
        (0, true) => Op::Lsl64,
        (1, true) => Op::Lsr64,
        (2, true) => Op::Asr64,
        (3, true) => Op::Ror64,
        (0, false) => Op::Lsl32,
        (1, false) => Op::Lsr32,
        (2, false) => Op::Asr32,
        _ => Op::Ror32,
    }
}

fn shifted(e: Expr, kind: u32, amount: u32, sf: bool) -> Expr {
    if amount == 0 { e } else { op2(shift_op(kind, sf), e, c(amount as u64)) }
}

fn extended(e: Expr, option: u32, shift: u32) -> Expr {
    let e = match option {
        0 => op1(Op::Zext8, e),
        1 => op1(Op::Zext16, e),
        2 => op1(Op::Zext32, e),
        4 => op1(Op::Sext8, e),
        5 => op1(Op::Sext16, e),
        6 => op1(Op::Sext32, e),
        _ => e,
    };
    if shift == 0 { e } else { op2(Op::Lsl64, e, c(shift as u64)) }
}

fn add_flags(sf: bool, a: Expr, b: Expr, carry: Expr) -> Expr {
    op3(if sf { Op::FlagsAdd64 } else { Op::FlagsAdd32 }, a, b, carry)
}

fn logic_flags(sf: bool, r: Expr) -> Expr {
    op1(if sf { Op::FlagsLogic64 } else { Op::FlagsLogic32 }, r)
}

fn addsub(sf: bool, sub: bool, set_flags: bool, a: Expr, b: Expr, rd: u32, dest_sp: bool) -> Vec<Stmt> {
    let mut out = vec![Stmt::Let(0, a), Stmt::Let(1, b)];
    let res = width(op2(if sub { Op::Sub } else { Op::Add }, t(0), t(1)), sf);
    if set_flags {
        let (b2, carry) = if sub { (op1(Op::Not, t(1)), c(1)) } else { (t(1), c(0)) };
        out.push(Stmt::SetNzcv(add_flags(sf, t(0), b2, carry)));
    }
    if dest_sp && !set_flags { wsp(&mut out, rd, res) } else { wz(&mut out, rd, res) }
    out
}

fn dp_imm(pc: u64, w: u32) -> Option<Vec<Stmt>> {
    let rd = field(w, 0, 5);
    let rn = field(w, 5, 5);
    let sf = w >> 31 == 1;
    let mut out = Vec::new();

    if field(w, 24, 5) == 0b10000 {
        let imm = sext(((field(w, 5, 19) << 2) | field(w, 29, 2)) as u64, 21);
        let value = if w >> 31 == 0 { pc.wrapping_add(imm) } else { (pc & !0xfff).wrapping_add(imm << 12) };
        wz(&mut out, rd, c(value));
        return Some(out);
    }

    match field(w, 23, 6) {
        0b100010 => {
            let sub = field(w, 30, 1) == 1;
            let s = field(w, 29, 1) == 1;
            let imm = (field(w, 10, 12) as u64) << (12 * field(w, 22, 1));
            Some(addsub(sf, sub, s, rsp(rn), c(imm), rd, true))
        }
        0b100100 => {
            let n = field(w, 22, 1);
            if !sf && n == 1 {
                return None;
            }
            let (imm, _) = decode_bit_masks(n, field(w, 10, 6), field(w, 16, 6), true, if sf { 64 } else { 32 })?;
            let opc = field(w, 29, 2);
            let o = [Op::And, Op::Or, Op::Xor, Op::And][opc as usize];
            out.push(Stmt::Let(0, width(op2(o, rz(rn), c(imm)), sf)));
            if opc == 3 {
                out.push(Stmt::SetNzcv(logic_flags(sf, t(0))));
                wz(&mut out, rd, t(0));
            } else {
                wsp(&mut out, rd, t(0));
            }
            Some(out)
        }
        0b100101 => {
            let opc = field(w, 29, 2);
            let hw = field(w, 21, 2);
            if opc == 1 || (!sf && hw >= 2) {
                return None;
            }
            let shift = hw * 16;
            let imm = (field(w, 5, 16) as u64) << shift;
            let mask = if sf { u64::MAX } else { 0xffff_ffff };
            let value = match opc {
                0 => c(!imm & mask),
                2 => c(imm),
                _ => width(op2(Op::Or, op2(Op::And, rz(rd), c(!(0xffffu64 << shift))), c(imm)), sf),
            };
            wz(&mut out, rd, value);
            Some(out)
        }
        0b100110 => {
            let opc = field(w, 29, 2);
            let n = field(w, 22, 1);
            let immr = field(w, 16, 6);
            let imms = field(w, 10, 6);
            if opc == 3 || (sf as u32) != n || (!sf && (immr >= 32 || imms >= 32)) {
                return None;
            }
            let (wmask, tmask) = decode_bit_masks(n, imms, immr, false, if sf { 64 } else { 32 })?;
            let (r, s) = (c(immr as u64), c(imms as u64));
            let value = match (opc, sf) {
                (0, true) => op(Op::Sbfm64, vec![rz(rn), r, s, c(wmask), c(tmask)]),
                (0, false) => op(Op::Sbfm32, vec![rz(rn), r, s, c(wmask), c(tmask)]),
                (1, true) => op(Op::Bfm64, vec![rz(rd), rz(rn), r, c(wmask), c(tmask)]),
                (1, false) => op(Op::Bfm32, vec![rz(rd), rz(rn), r, c(wmask), c(tmask)]),
                (_, true) => op(Op::Ubfm64, vec![rz(rn), r, c(wmask), c(tmask)]),
                (_, false) => op(Op::Ubfm32, vec![rz(rn), r, c(wmask), c(tmask)]),
            };
            wz(&mut out, rd, value);
            Some(out)
        }
        0b100111 => {
            let n = field(w, 22, 1);
            let lsb = field(w, 10, 6);
            if field(w, 29, 2) != 0 || field(w, 21, 1) != 0 || (sf as u32) != n || (!sf && lsb >= 32) {
                return None;
            }
            let rm = field(w, 16, 5);
            let o = if sf { Op::Extr64 } else { Op::Extr32 };
            wz(&mut out, rd, op3(o, rz(rn), rz(rm), c(lsb as u64)));
            Some(out)
        }
        _ => None,
    }
}

fn sysreg(w: u32) -> Option<SysReg> {
    Some(match (w | (1 << 21)) & !0x1f {
        0xD53B_D040 => SysReg::TpidrEl0,
        0xD53B_D060 => SysReg::TpidrroEl0,
        0xD53B_E040 => SysReg::CntvctEl0,
        0xD53B_E020 => SysReg::CntpctEl0,
        0xD53B_E000 => SysReg::CntfrqEl0,
        0xD53B_4400 => SysReg::Fpcr,
        0xD53B_4420 => SysReg::Fpsr,
        0xD53B_00E0 => SysReg::DczidEl0,
        0xD53B_0020 => SysReg::CtrEl0,
        _ => return None,
    })
}

fn branch_sys(pc: u64, w: u32) -> Option<Vec<Stmt>> {
    let mut out = Vec::new();

    if field(w, 26, 5) == 0b00101 {
        let target = pc.wrapping_add(sext((field(w, 0, 26) as u64) << 2, 28));
        if w >> 31 == 1 {
            out.push(Stmt::SetReg(30, c(pc + 4)));
            out.push(Stmt::Call(target));
        } else {
            out.push(Stmt::Jump(target));
        }
        return Some(out);
    }
    if field(w, 25, 6) == 0b011010 {
        let sf = w >> 31 == 1;
        let target = pc.wrapping_add(sext((field(w, 5, 19) as u64) << 2, 21));
        let value = width(rz(field(w, 0, 5)), sf);
        let o = if field(w, 24, 1) == 1 { Op::Ne } else { Op::Eq };
        out.push(Stmt::JumpIf(op2(o, value, c(0)), target));
        return Some(out);
    }
    if field(w, 25, 6) == 0b011011 {
        let bit = (field(w, 31, 1) << 5) | field(w, 19, 5);
        let target = pc.wrapping_add(sext((field(w, 5, 14) as u64) << 2, 16));
        let tested = op2(Op::Bit, rz(field(w, 0, 5)), c(bit as u64));
        let cond = if field(w, 24, 1) == 1 { tested } else { op2(Op::Eq, tested, c(0)) };
        out.push(Stmt::JumpIf(cond, target));
        return Some(out);
    }
    if w & 0xFF00_0010 == 0x5400_0000 {
        let target = pc.wrapping_add(sext((field(w, 5, 19) as u64) << 2, 21));
        let cc = field(w, 0, 4);
        if cc >= 14 {
            out.push(Stmt::Jump(target));
        } else {
            out.push(Stmt::JumpIf(op2(Op::Cond, Expr::Nzcv, c(cc as u64)), target));
        }
        return Some(out);
    }
    if w & 0xFFE0_001F == 0xD400_0001 {
        out.push(Stmt::Svc(field(w, 5, 16) as u16));
        return Some(out);
    }
    let rn = field(w, 5, 5);
    match w & 0xFFFF_FC1F {
        0xD61F_0000 => {
            out.push(Stmt::JumpIndirect(rz(rn)));
            return Some(out);
        }
        0xD63F_0000 => {
            out.push(Stmt::Let(0, rz(rn)));
            out.push(Stmt::SetReg(30, c(pc + 4)));
            out.push(Stmt::CallIndirect(t(0)));
            return Some(out);
        }
        0xD65F_0000 => {
            out.push(Stmt::Return(rz(rn)));
            return Some(out);
        }
        _ => {}
    }
    if w & 0xFFFF_F01F == 0xD503_201F {
        return Some(out);
    }
    if w & 0xFFFF_F01F == 0xD503_301F {
        let (crm, op2) = (field(w, 8, 4), field(w, 5, 3));
        match op2 {
            0b010 => out.push(Stmt::Monitor { addr: c(u64::MAX), lo: c(0), hi: c(0) }),
            0b100 | 0b101 if crm & 3 == 3 => out.push(Stmt::Fence),
            _ => {}
        }
        return Some(out);
    }
    let rt = field(w, 0, 5);
    if w & !0x1f == 0xD50B_7420 {
        out.push(Stmt::Let(0, op2(Op::And, rz(rt), c(!63))));
        for i in 0..8u64 {
            out.push(Stmt::Store { size: 8, addr: op2(Op::Add, t(0), c(i * 8)), value: c(0) });
        }
        return Some(out);
    }
    if w & 0xFFF8_0000 == 0xD508_0000 {
        return Some(out);
    }
    if w & !0x1f == 0xD53B_4200 {
        wz(&mut out, rt, Expr::Nzcv);
        return Some(out);
    }
    if w & !0x1f == 0xD51B_4200 {
        out.push(Stmt::SetNzcv(op2(Op::And, rz(rt), c(0xf000_0000))));
        return Some(out);
    }
    if w & 0xFFF0_0000 == 0xD530_0000 {
        wz(&mut out, rt, Expr::SysReg(sysreg(w)?));
        return Some(out);
    }
    if w & 0xFFF0_0000 == 0xD510_0000 {
        out.push(Stmt::SetSysReg(sysreg(w)?, rz(rt)));
        return Some(out);
    }
    None
}

enum Access {
    Store,
    Load,
    LoadSigned64,
    LoadSigned32,
    Prefetch,
}

fn access(size: u32, opc: u32) -> Option<Access> {
    Some(match opc {
        0 => Access::Store,
        1 => Access::Load,
        2 if size == 3 => Access::Prefetch,
        2 => Access::LoadSigned64,
        3 if size < 2 => Access::LoadSigned32,
        _ => return None,
    })
}

fn sign_extend_op(bytes: u8) -> Op {
    match bytes {
        1 => Op::Sext8,
        2 => Op::Sext16,
        _ => Op::Sext32,
    }
}

fn loaded(kind: &Access, bytes: u8, addr: Expr) -> Expr {
    let raw = load(bytes, addr);
    match kind {
        Access::LoadSigned64 => op1(sign_extend_op(bytes), raw),
        Access::LoadSigned32 => op1(Op::Zext32, op1(sign_extend_op(bytes), raw)),
        _ => raw,
    }
}

fn single(kind: Access, bytes: u8, rt: u32, rn: u32, addr: Expr, writeback: Option<Expr>) -> Vec<Stmt> {
    let mut out = vec![Stmt::Let(0, addr)];
    match kind {
        Access::Prefetch => {}
        Access::Store => {
            out.push(Stmt::Let(1, rz(rt)));
            out.push(Stmt::Store { size: bytes, addr: t(0), value: t(1) });
        }
        ref k => out.push(Stmt::Let(1, loaded(k, bytes, t(0)))),
    }
    if let Some(wb) = writeback {
        wsp(&mut out, rn, wb);
    }
    if !matches!(kind, Access::Store | Access::Prefetch) {
        wz(&mut out, rt, t(1));
    }
    out
}

fn load_store(pc: u64, w: u32) -> Option<Vec<Stmt>> {
    if field(w, 26, 1) == 1 {
        return load_store_simd(pc, w);
    }
    let rt = field(w, 0, 5);
    let rn = field(w, 5, 5);
    let size = field(w, 30, 2);
    let bytes = 1u8 << size;

    if field(w, 24, 6) == 0b001000 {
        let (o2, l, o1) = (field(w, 23, 1), field(w, 22, 1), field(w, 21, 1));
        let rs = field(w, 16, 5) as u8;
        let rt2 = field(w, 10, 5);
        let mut out = vec![Stmt::Let(0, rsp(rn))];
        match (o2, o1) {
            (0, 0) => {
                if l == 1 {
                    out.push(Stmt::Let(1, load(bytes, t(0))));
                    out.push(Stmt::Monitor { addr: t(0), lo: t(1), hi: c(0) });
                    wz(&mut out, rt, t(1));
                } else {
                    out.push(Stmt::StoreExclusive { size: bytes, addr: t(0), lo: rz(rt), hi: c(0), status: rs });
                }
            }
            (1, 0) => {
                if l == 1 {
                    out.push(Stmt::Let(1, load(bytes, t(0))));
                    wz(&mut out, rt, t(1));
                } else {
                    out.push(Stmt::Store { size: bytes, addr: t(0), value: rz(rt) });
                    out.push(Stmt::Fence);
                }
            }
            (0, 1) if size == 2 => {
                if l == 1 {
                    out.push(Stmt::Let(1, load(8, t(0))));
                    out.push(Stmt::Monitor { addr: t(0), lo: t(1), hi: c(0) });
                    wz(&mut out, rt, op1(Op::Zext32, t(1)));
                    wz(&mut out, rt2, op2(Op::Lsr64, t(1), c(32)));
                } else {
                    let pair = op2(Op::Or, op1(Op::Zext32, rz(rt)), op2(Op::Lsl64, rz(rt2), c(32)));
                    out.push(Stmt::StoreExclusive { size: 8, addr: t(0), lo: pair, hi: c(0), status: rs });
                }
            }
            (0, 1) if size == 3 => {
                if l == 1 {
                    out.push(Stmt::Let(1, load(8, t(0))));
                    out.push(Stmt::Let(2, load(8, op2(Op::Add, t(0), c(8)))));
                    out.push(Stmt::Monitor { addr: t(0), lo: t(1), hi: t(2) });
                    wz(&mut out, rt, t(1));
                    wz(&mut out, rt2, t(2));
                } else {
                    out.push(Stmt::StoreExclusive { size: 16, addr: t(0), lo: rz(rt), hi: rz(rt2), status: rs });
                }
            }
            _ => return None,
        }
        return Some(out);
    }

    if field(w, 24, 6) == 0b011000 {
        let addr = pc.wrapping_add(sext((field(w, 5, 19) as u64) << 2, 21));
        let mut out = Vec::new();
        let value = match size {
            0 => load(4, c(addr)),
            1 => load(8, c(addr)),
            2 => op1(Op::Sext32, load(4, c(addr))),
            _ => return Some(out),
        };
        wz(&mut out, rt, value);
        return Some(out);
    }

    if field(w, 27, 3) == 0b101 {
        let idx = field(w, 23, 2);
        let l = field(w, 22, 1) == 1;
        let opc = field(w, 30, 2);
        let rt2 = field(w, 10, 5);
        let (bytes, signed) = match (opc, l) {
            (0, _) => (4u8, false),
            (1, true) => (4u8, true),
            (2, _) => (8u8, false),
            _ => return None,
        };
        let offset = sext(field(w, 15, 7) as u64, 7).wrapping_mul(bytes as u64);
        let base = rsp(rn);
        let addr = if idx == 1 { base.clone() } else { op2(Op::Add, base.clone(), c(offset)) };
        let writeback = matches!(idx, 1 | 3).then(|| op2(Op::Add, base, c(offset)));
        let second = op2(Op::Add, t(0), c(bytes as u64));
        let mut out = vec![Stmt::Let(0, addr)];
        if l {
            let get = |a: Expr| if signed { op1(Op::Sext32, load(4, a)) } else { load(bytes, a) };
            out.push(Stmt::Let(1, get(t(0))));
            out.push(Stmt::Let(2, get(second)));
            if let Some(wb) = writeback {
                wsp(&mut out, rn, wb);
            }
            wz(&mut out, rt, t(1));
            wz(&mut out, rt2, t(2));
        } else {
            out.push(Stmt::Let(1, rz(rt)));
            out.push(Stmt::Let(2, rz(rt2)));
            out.push(Stmt::Store { size: bytes, addr: t(0), value: t(1) });
            out.push(Stmt::Store { size: bytes, addr: second, value: t(2) });
            if let Some(wb) = writeback {
                wsp(&mut out, rn, wb);
            }
        }
        return Some(out);
    }

    let opc = field(w, 22, 2);

    if field(w, 24, 6) == 0b111001 {
        let kind = access(size, opc)?;
        let offset = (field(w, 10, 12) as u64) << size;
        return Some(single(kind, bytes, rt, rn, op2(Op::Add, rsp(rn), c(offset)), None));
    }

    if field(w, 24, 6) == 0b111000 {
        let kind = access(size, opc)?;
        if field(w, 21, 1) == 0 {
            let imm = sext(field(w, 12, 9) as u64, 9);
            let base = rsp(rn);
            let with_imm = op2(Op::Add, base.clone(), c(imm));
            let (addr, writeback) = match field(w, 10, 2) {
                1 => (base, Some(with_imm)),
                3 => (with_imm.clone(), Some(with_imm)),
                _ => (with_imm, None),
            };
            return Some(single(kind, bytes, rt, rn, addr, writeback));
        }
        if field(w, 10, 2) == 0b10 {
            let option = field(w, 13, 3);
            if option & 0b010 == 0 {
                return None;
            }
            let shift = if field(w, 12, 1) == 1 { size } else { 0 };
            let offset = extended(rz(field(w, 16, 5)), option, shift);
            return Some(single(kind, bytes, rt, rn, op2(Op::Add, rsp(rn), offset), None));
        }
        return None;
    }
    None
}

fn load_v(out: &mut Vec<Stmt>, rt: u32, bytes: u8, addr: Expr) {
    if bytes == 16 {
        out.push(Stmt::SetV { reg: rt as u8, hi: false, value: load(8, addr.clone()) });
        out.push(Stmt::SetV { reg: rt as u8, hi: true, value: load(8, op2(Op::Add, addr, c(8))) });
    } else {
        out.push(Stmt::SetV { reg: rt as u8, hi: false, value: load(bytes, addr) });
        out.push(Stmt::SetV { reg: rt as u8, hi: true, value: c(0) });
    }
}

fn store_v(out: &mut Vec<Stmt>, rt: u32, bytes: u8, addr: Expr) {
    if bytes == 16 {
        out.push(Stmt::Store { size: 8, addr: addr.clone(), value: vlo(rt) });
        out.push(Stmt::Store { size: 8, addr: op2(Op::Add, addr, c(8)), value: vhi(rt) });
    } else {
        out.push(Stmt::Store { size: bytes, addr, value: vlo(rt) });
    }
}

fn load_store_simd(pc: u64, w: u32) -> Option<Vec<Stmt>> {
    let rt = field(w, 0, 5);
    let rn = field(w, 5, 5);
    let size = field(w, 30, 2);

    if field(w, 24, 6) == 0b011100 {
        if size == 3 {
            return None;
        }
        let bytes = 4u8 << size;
        let addr = pc.wrapping_add(sext((field(w, 5, 19) as u64) << 2, 21));
        let mut out = Vec::new();
        load_v(&mut out, rt, bytes, c(addr));
        return Some(out);
    }

    if field(w, 27, 3) == 0b101 {
        let opc = field(w, 30, 2);
        if opc == 3 {
            return None;
        }
        let bytes = 4u8 << opc;
        let idx = field(w, 23, 2);
        let l = field(w, 22, 1) == 1;
        let rt2 = field(w, 10, 5);
        let offset = sext(field(w, 15, 7) as u64, 7).wrapping_mul(bytes as u64);
        let base = rsp(rn);
        let addr = if idx == 1 { base.clone() } else { op2(Op::Add, base.clone(), c(offset)) };
        let writeback = matches!(idx, 1 | 3).then(|| op2(Op::Add, base, c(offset)));
        let second = op2(Op::Add, t(0), c(bytes as u64));
        let mut out = vec![Stmt::Let(0, addr)];
        if l {
            load_v(&mut out, rt, bytes, t(0));
            load_v(&mut out, rt2, bytes, second);
        } else {
            store_v(&mut out, rt, bytes, t(0));
            store_v(&mut out, rt2, bytes, second);
        }
        if let Some(wb) = writeback {
            wsp(&mut out, rn, wb);
        }
        return Some(out);
    }

    let opc = field(w, 22, 2);
    let is_load = opc & 1 == 1;
    let bytes = match (size, opc >> 1) {
        (s, 0) => 1u8 << s,
        (0, 1) => 16,
        _ => return None,
    };
    let shift = bytes.trailing_zeros();
    let (addr, writeback) = if field(w, 24, 6) == 0b111101 {
        (op2(Op::Add, rsp(rn), c((field(w, 10, 12) as u64) << shift)), None)
    } else if field(w, 24, 6) == 0b111100 && field(w, 21, 1) == 0 {
        let imm = sext(field(w, 12, 9) as u64, 9);
        let base = rsp(rn);
        let with_imm = op2(Op::Add, base.clone(), c(imm));
        match field(w, 10, 2) {
            1 => (base, Some(with_imm)),
            3 => (with_imm.clone(), Some(with_imm)),
            0 => (with_imm, None),
            _ => return None,
        }
    } else if field(w, 24, 6) == 0b111100 && field(w, 10, 2) == 0b10 {
        let option = field(w, 13, 3);
        if option & 0b010 == 0 {
            return None;
        }
        let s = if field(w, 12, 1) == 1 { shift } else { 0 };
        (op2(Op::Add, rsp(rn), extended(rz(field(w, 16, 5)), option, s)), None)
    } else {
        return None;
    };
    let mut out = vec![Stmt::Let(0, addr)];
    if is_load {
        load_v(&mut out, rt, bytes, t(0));
    } else {
        store_v(&mut out, rt, bytes, t(0));
    }
    if let Some(wb) = writeback {
        wsp(&mut out, rn, wb);
    }
    Some(out)
}

fn dp_reg(w: u32) -> Option<Vec<Stmt>> {
    let sf = w >> 31 == 1;
    let rd = field(w, 0, 5);
    let rn = field(w, 5, 5);
    let rm = field(w, 16, 5);
    let mut out = Vec::new();

    match field(w, 24, 5) {
        0b01010 => {
            let imm6 = field(w, 10, 6);
            if !sf && imm6 >= 32 {
                return None;
            }
            let mut b = shifted(rz(rm), field(w, 22, 2), imm6, sf);
            if field(w, 21, 1) == 1 {
                b = op1(Op::Not, b);
            }
            let opc = field(w, 29, 2);
            let o = [Op::And, Op::Or, Op::Xor, Op::And][opc as usize];
            out.push(Stmt::Let(0, width(op2(o, rz(rn), b), sf)));
            if opc == 3 {
                out.push(Stmt::SetNzcv(logic_flags(sf, t(0))));
            }
            wz(&mut out, rd, t(0));
            return Some(out);
        }
        0b01011 => {
            let sub = field(w, 30, 1) == 1;
            let s = field(w, 29, 1) == 1;
            if field(w, 21, 1) == 0 {
                let kind = field(w, 22, 2);
                let imm6 = field(w, 10, 6);
                if kind == 3 || (!sf && imm6 >= 32) {
                    return None;
                }
                return Some(addsub(sf, sub, s, rz(rn), shifted(rz(rm), kind, imm6, sf), rd, false));
            }
            let imm3 = field(w, 10, 3);
            if field(w, 22, 2) != 0 || imm3 > 4 {
                return None;
            }
            let b = extended(rz(rm), field(w, 13, 3), imm3);
            return Some(addsub(sf, sub, s, rsp(rn), b, rd, true));
        }
        0b11011 => {
            if field(w, 29, 2) != 0 {
                return None;
            }
            let o0 = field(w, 15, 1) == 1;
            let ra = rz(field(w, 10, 5));
            let (a, b) = (rz(rn), rz(rm));
            let value = match (field(w, 21, 3), o0, sf) {
                (0b000, _, _) => {
                    let prod = op2(Op::Mul, a, b);
                    width(op2(if o0 { Op::Sub } else { Op::Add }, ra, prod), sf)
                }
                (0b001, _, true) => {
                    let prod = op2(Op::Mul, op1(Op::Sext32, a), op1(Op::Sext32, b));
                    op2(if o0 { Op::Sub } else { Op::Add }, ra, prod)
                }
                (0b101, _, true) => {
                    let prod = op2(Op::Mul, op1(Op::Zext32, a), op1(Op::Zext32, b));
                    op2(if o0 { Op::Sub } else { Op::Add }, ra, prod)
                }
                (0b010, false, true) => op2(Op::Smulh, a, b),
                (0b110, false, true) => op2(Op::Umulh, a, b),
                _ => return None,
            };
            wz(&mut out, rd, value);
            return Some(out);
        }
        _ => {}
    }

    match field(w, 21, 8) {
        0b1101_0000 => {
            if field(w, 10, 6) != 0 {
                return None;
            }
            let sub = field(w, 30, 1) == 1;
            let s = field(w, 29, 1) == 1;
            out.push(Stmt::Let(0, rz(rn)));
            out.push(Stmt::Let(1, if sub { op1(Op::Not, rz(rm)) } else { rz(rm) }));
            out.push(Stmt::Let(2, op1(Op::Carry, Expr::Nzcv)));
            if s {
                out.push(Stmt::SetNzcv(add_flags(sf, t(0), t(1), t(2))));
            }
            wz(&mut out, rd, width(op2(Op::Add, op2(Op::Add, t(0), t(1)), t(2)), sf));
            Some(out)
        }
        0b1101_0010 => {
            if field(w, 29, 1) != 1 || field(w, 10, 1) != 0 || field(w, 4, 1) != 0 {
                return None;
            }
            let sub = field(w, 30, 1) == 1;
            let b = if field(w, 11, 1) == 1 { c(rm as u64) } else { rz(rm) };
            let (b, carry) = if sub { (op1(Op::Not, b), c(1)) } else { (b, c(0)) };
            let cc = field(w, 12, 4) as u64;
            let fallback = c((field(w, 0, 4) as u64) << 28);
            let flags = add_flags(sf, rz(rn), b, carry);
            out.push(Stmt::SetNzcv(op3(Op::Select, op2(Op::Cond, Expr::Nzcv, c(cc)), flags, fallback)));
            Some(out)
        }
        0b1101_0100 => {
            let op2bits = field(w, 10, 2);
            if field(w, 29, 1) != 0 || op2bits > 1 {
                return None;
            }
            let b = rz(rm);
            let b = match (field(w, 30, 1), op2bits) {
                (0, 0) => b,
                (0, _) => op2(Op::Add, b, c(1)),
                (_, 0) => op1(Op::Not, b),
                _ => op2(Op::Sub, c(0), b),
            };
            let cc = c(field(w, 12, 4) as u64);
            wz(&mut out, rd, width(op3(Op::Select, op2(Op::Cond, Expr::Nzcv, cc), rz(rn), b), sf));
            Some(out)
        }
        0b1101_0110 => {
            if field(w, 29, 1) != 0 {
                return None;
            }
            let opcode = field(w, 10, 6);
            let value = if field(w, 30, 1) == 0 {
                let o = match (opcode, sf) {
                    (2, true) => Op::Udiv64,
                    (2, false) => Op::Udiv32,
                    (3, true) => Op::Sdiv64,
                    (3, false) => Op::Sdiv32,
                    (8..=11, _) => shift_op(opcode - 8, sf),
                    (0x10..=0x17, _) if (opcode & 3 == 3) == sf => {
                        let o = if opcode & 4 == 0 { Op::Crc32 } else { Op::Crc32c };
                        wz(&mut out, rd, op3(o, rz(rn), rz(rm), c((opcode & 3) as u64)));
                        return Some(out);
                    }
                    _ => return None,
                };
                op2(o, rz(rn), rz(rm))
            } else {
                if rm != 0 {
                    return None;
                }
                let o = match (opcode, sf) {
                    (0, true) => Op::Rbit64,
                    (0, false) => Op::Rbit32,
                    (1, true) => Op::Rev16x4,
                    (1, false) => Op::Rev16x2,
                    (2, true) => Op::Rev32x2,
                    (2, false) => Op::Rev32,
                    (3, true) => Op::Rev64,
                    (4, true) => Op::Clz64,
                    (4, false) => Op::Clz32,
                    (5, true) => Op::Cls64,
                    (5, false) => Op::Cls32,
                    _ => return None,
                };
                op1(o, rz(rn))
            };
            wz(&mut out, rd, value);
            Some(out)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(word: u32) -> Control {
        lift(0x1000, word).control()
    }

    fn defined(word: u32) -> bool {
        !matches!(lift(0x1000, word).control(), Control::Undefined)
    }

    #[test]
    fn known_words_decode() {
        for w in [
            0x9acc4d0c,
            0x1ac24020,
            0x1ac25820,
            0xd503201f,
            0x91000400,
            0xa9bf7bfd,
            0xa8c17bfd,
            0x910003fd,
            0xaa0103e0,
            0xf1000421,
            0x9ac20820,
            0x9b028c20,
            0x381ffc45,
            0x90000000,
            0x9a820020,
            0xd37df020,
            0xf9400420,
            0xb860792a,
            0x92401c20,
            0x320003e0,
            0xd2800000,
            0xd53bd060,
            0xc85f7c20,
            0xc8027c20,
            0xd5033bbf,
            0x93407c00,
            0xdac01000,
            0x1a9f17e0,
            0xadbf1fe6,
            0xacc11fe6,
            0x3dc00420,
            0xfd000fe8,
            0xbc404420,
        ] {
            assert!(defined(w), "{w:#010x} should decode");
        }
    }

    #[test]
    fn control_flow() {
        assert_eq!(ctl(0xd65f03c0), Control::Return);
        assert_eq!(ctl(0x94000004), Control::Call(0x1010));
        assert_eq!(ctl(0x17ffffff), Control::Jump(0xffc));
        assert_eq!(ctl(0x54000041), Control::JumpIf(0x1008));
        assert_eq!(ctl(0xb4000040), Control::JumpIf(0x1008));
        assert_eq!(ctl(0xd61f0120), Control::JumpIndirect);
        assert_eq!(ctl(0xd63f0120), Control::CallIndirect);
        assert_eq!(ctl(0x00000000), Control::Undefined);
    }

    #[test]
    fn exclusives_and_barriers() {
        let stmts = |w| lift(0, w).stmts;
        assert!(matches!(stmts(0xc85f7c20)[..], [Stmt::Let(0, _), Stmt::Let(1, _), Stmt::Monitor { .. }, Stmt::SetReg(0, _)]));
        assert!(matches!(stmts(0xc8027c20)[..], [Stmt::Let(0, _), Stmt::StoreExclusive { size: 8, status: 2, .. }]));
        assert!(matches!(stmts(0x880afe93)[..], [_, Stmt::StoreExclusive { size: 4, status: 10, .. }]));
        assert!(matches!(stmts(0x889ffc20)[..], [_, Stmt::Store { size: 4, .. }, Stmt::Fence]));
        assert!(matches!(stmts(0xc8220c20)[..], [_, Stmt::StoreExclusive { size: 16, status: 2, .. }]));
        assert_eq!(stmts(0xd5033bbf), vec![Stmt::Fence]);
        assert_eq!(stmts(0xd5033f9f), vec![Stmt::Fence]);
        assert_eq!(stmts(0xd50339bf), vec![]);
        assert_eq!(stmts(0xd5033abf), vec![]);
        assert_eq!(stmts(0xd5033fdf), vec![]);
        assert_eq!(stmts(0xd503305f), vec![Stmt::Monitor { addr: c(u64::MAX), lo: c(0), hi: c(0) }]);
    }

    #[test]
    fn svc_immediate() {
        assert_eq!(lift(0, 0xd40004e1).stmts, vec![Stmt::Svc(0x27)]);
    }
}
