use crate::ops::Op;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SysReg {
    TpidrEl0,
    TpidrroEl0,
    CntvctEl0,
    CntpctEl0,
    CntfrqEl0,
    Fpcr,
    Fpsr,
    DczidEl0,
    CtrEl0,
}

impl SysReg {
    pub const ALL: &'static [SysReg] = &[
        SysReg::TpidrEl0,
        SysReg::TpidrroEl0,
        SysReg::CntvctEl0,
        SysReg::CntpctEl0,
        SysReg::CntfrqEl0,
        SysReg::Fpcr,
        SysReg::Fpsr,
        SysReg::DczidEl0,
        SysReg::CtrEl0,
    ];
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Const(u64),
    Reg(u8),
    Sp,
    Nzcv,
    Tmp(u8),
    Op(Op, Vec<Expr>),
    Load { size: u8, addr: Box<Expr> },
    SysReg(SysReg),
    V { reg: u8, hi: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    Let(u8, Expr),
    SetReg(u8, Expr),
    SetSp(Expr),
    SetNzcv(Expr),
    SetSysReg(SysReg, Expr),
    SetV {
        reg: u8,
        hi: bool,
        value: Expr,
    },
    Store {
        size: u8,
        addr: Expr,
        value: Expr,
    },
    Monitor {
        addr: Expr,
        lo: Expr,
        hi: Expr,
    },
    StoreExclusive {
        size: u8,
        addr: Expr,
        lo: Expr,
        hi: Expr,
        status: u8,
    },
    Fence,
    Svc(u16),
    Simd(u32),
    Jump(u64),
    JumpIf(Expr, u64),
    JumpIndirect(Expr),
    Call(u64),
    CallIndirect(Expr),
    Return(Expr),
    Undefined(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Fallthrough,
    Jump(u64),
    JumpIf(u64),
    JumpIndirect,
    Call(u64),
    CallIndirect,
    Return,
    Undefined,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Insn {
    pub addr: u64,
    pub word: u32,
    pub stmts: Vec<Stmt>,
}

impl Insn {
    pub fn control(&self) -> Control {
        match self.stmts.last() {
            Some(Stmt::Jump(t)) => Control::Jump(*t),
            Some(Stmt::JumpIf(_, t)) => Control::JumpIf(*t),
            Some(Stmt::JumpIndirect(_)) => Control::JumpIndirect,
            Some(Stmt::Call(t)) => Control::Call(*t),
            Some(Stmt::CallIndirect(_)) => Control::CallIndirect,
            Some(Stmt::Return(_)) => Control::Return,
            Some(Stmt::Undefined(_)) => Control::Undefined,
            _ => Control::Fallthrough,
        }
    }

    pub fn body(&self) -> &[Stmt] {
        match self.control() {
            Control::Fallthrough => &self.stmts,
            _ => &self.stmts[..self.stmts.len() - 1],
        }
    }

    pub fn next(&self) -> u64 {
        self.addr.wrapping_add(4)
    }
}

pub fn c(v: u64) -> Expr {
    Expr::Const(v)
}

pub fn t(n: u8) -> Expr {
    Expr::Tmp(n)
}

pub fn op(o: Op, args: Vec<Expr>) -> Expr {
    debug_assert_eq!(args.len(), o.arity(), "{o:?}");
    Expr::Op(o, args)
}

pub fn op1(o: Op, a: Expr) -> Expr {
    op(o, vec![a])
}

pub fn op2(o: Op, a: Expr, b: Expr) -> Expr {
    op(o, vec![a, b])
}

pub fn op3(o: Op, a: Expr, b: Expr, cc: Expr) -> Expr {
    op(o, vec![a, b, cc])
}

pub fn vlo(reg: u32) -> Expr {
    Expr::V {
        reg: reg as u8,
        hi: false,
    }
}

pub fn vhi(reg: u32) -> Expr {
    Expr::V {
        reg: reg as u8,
        hi: true,
    }
}

pub fn load(size: u8, addr: Expr) -> Expr {
    Expr::Load {
        size,
        addr: Box::new(addr),
    }
}
