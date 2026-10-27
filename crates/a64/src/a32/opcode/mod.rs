mod names;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ElemStyle {
    Hidden,
    Plain,
    Int,
    Signed,
    Unsigned,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Opcode {
    And,
    Eor,
    Sub,
    Rsb,
    Add,
    Adc,
    Sbc,
    Rsc,
    Tst,
    Teq,
    Cmp,
    Cmn,
    Orr,
    Mov,
    Bic,
    Mvn,
    Movw,
    Movt,
    B,
    Bl,
    Bx,
    Blx,
    Bkpt,
    Udf,
    Svc,
    Ldr,
    Str,
    Ldm,
    Stm,
    Mul,
    Mla,
    Mls,
    Umull,
    Umlal,
    Smull,
    Smlal,
    Smlabb,
    Smlabt,
    Smlatb,
    Smlatt,
    Smlawb,
    Smlawt,
    Smulbb,
    Smulbt,
    Smultb,
    Smultt,
    Smulwb,
    Smulwt,
    Smlalbb,
    Smlalbt,
    Smlaltb,
    Smlaltt,
    Smuad,
    Smuadx,
    Smusd,
    Smusdx,
    Smlad,
    Smladx,
    Smlsd,
    Smlsdx,
    Smlald,
    Smlaldx,
    Smlsld,
    Smlsldx,
    Smmul,
    Smmulr,
    Smmla,
    Smmlar,
    Smmls,
    Smmlsr,
    Ldrh,
    Strh,
    Lda,
    Ldab,
    Ldah,
    Stl,
    Stlb,
    Stlh,
    Ldrsb,
    Ldrsh,
    Ldrd,
    Strd,
    Ldrex,
    Strex,
    Ldrexb,
    Strexb,
    Ldrexh,
    Strexh,
    Ldrexd,
    Strexd,
    Ldaex,
    Stlex,
    Ldaexb,
    Stlexb,
    Ldaexh,
    Stlexh,
    Ldaexd,
    Stlexd,
    Uxtb,
    Uxth,
    Sxtb,
    Sxth,
    Uxtab,
    Uxtah,
    Sxtab,
    Sxtah,
    Uxtb16,
    Sxtb16,
    Uxtab16,
    Sxtab16,
    Rev,
    Rev16,
    Revsh,
    Rbit,
    Clz,
    Pkhbt,
    Pkhtb,
    Sel,
    Ssat,
    Usat,
    Ssat16,
    Usat16,
    Bfc,
    Bfi,
    Sbfx,
    Ubfx,
    Sadd16,
    Sasx,
    Ssax,
    Ssub16,
    Sadd8,
    Ssub8,
    Qadd16,
    Qasx,
    Qsax,
    Qsub16,
    Qadd8,
    Qsub8,
    Shadd16,
    Shasx,
    Shsax,
    Shsub16,
    Shadd8,
    Shsub8,
    Uadd16,
    Uasx,
    Usax,
    Usub16,
    Uadd8,
    Usub8,
    Uqadd16,
    Uqasx,
    Uqsax,
    Uqsub16,
    Uqadd8,
    Uqsub8,
    Uhadd16,
    Uhasx,
    Uhsax,
    Uhsub16,
    Uhadd8,
    Uhsub8,
    Udiv,
    Sdiv,
    Qadd,
    Qsub,
    Qdadd,
    Qdsub,
    Mrs,
    Msr,
    Nop,
    Yield,
    Wfe,
    Wfi,
    Sev,
    Sevl,
    Dsb,
    Dmb,
    Isb,
    Clrex,
    Sb,
    Pld,
    Pli,
    Vldr,
    Vstr,
    Vldm,
    Vstm,
    Vpush,
    Vpop,
    Vmov,
    VmovLane,
    VmovLaneS,
    VmovLaneU,
    Vdup,
    Vmrs,
    Vmsr,
    Mrc,
    Mcr,
    Mrrc,
    Vmla,
    Vmls,
    Vnmla,
    Vnmls,
    Vmul,
    Vnmul,
    Vadd,
    Vsub,
    Vdiv,
    Vfnma,
    Vfnms,
    Vfma,
    Vfms,
    Vabs,
    Vneg,
    Vsqrt,
    Vcmp,
    Vcmpe,
    Vrintr,
    Vrintz,
    Vrintx,
    Vcvt,
    VcvtR,
    Vld1,
    Vst1,
    Vld2,
    Vst2,
    Vld3,
    Vst3,
    Vld4,
    Vst4,
    Vext,
    Vand,
    Vorr,
    Veor,
    Vbic,
    Vorn,
    VaddF,
    VsubF,
    VmulF,
    VmlaF,
    VmlsF,
    VmlaI,
    VmlsI,
    VmaxF,
    VminF,
    Vbsl,
    Vmax,
    VmaxU,
    Vmin,
    VminU,
    Vcgt,
    VcgtU,
    Vcge,
    VcgeU,
    Vceq,
    VceqF,
    VcgeF,
    Vtst,
    VabsF,
    VnegF,
    Vrev16,
    Vrev32,
    Vrev64,
    Vmvn,
    VmovImm,
    VorrImm,
    VmvnImm,
    VbicImm,
}

impl Opcode {
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Opcode::VmovImm => "vmov",
            Opcode::VmovLane | Opcode::VmovLaneS | Opcode::VmovLaneU => "vmov",
            Opcode::VorrImm => "vorr",
            Opcode::VmvnImm => "vmvn",
            Opcode::VbicImm => "vbic",
            Opcode::VaddF => "vadd",
            Opcode::VsubF => "vsub",
            Opcode::VmulF => "vmul",
            Opcode::VmlaF => "vmla",
            Opcode::VmlsF => "vmls",
            Opcode::VmlaI => "vmla",
            Opcode::VmlsI => "vmls",
            Opcode::VmaxF => "vmax",
            Opcode::VminF => "vmin",
            Opcode::VmaxU => "vmax",
            Opcode::VminU => "vmin",
            Opcode::VcgtU => "vcgt",
            Opcode::VcgeU => "vcge",
            Opcode::VceqF => "vceq",
            Opcode::VcgeF => "vcge",
            Opcode::VabsF => "vabs",
            Opcode::VnegF => "vneg",
            other => other.name(),
        }
    }
    pub const fn is_flag_test(self) -> bool {
        matches!(self, Opcode::Tst | Opcode::Teq | Opcode::Cmp | Opcode::Cmn)
    }

    pub const fn is_data_processing(self) -> bool {
        matches!(
            self,
            Opcode::And
                | Opcode::Eor
                | Opcode::Sub
                | Opcode::Rsb
                | Opcode::Add
                | Opcode::Adc
                | Opcode::Sbc
                | Opcode::Rsc
                | Opcode::Tst
                | Opcode::Teq
                | Opcode::Cmp
                | Opcode::Cmn
                | Opcode::Orr
                | Opcode::Mov
                | Opcode::Bic
                | Opcode::Mvn
                | Opcode::Movw
                | Opcode::Movt
        )
    }

    pub const fn takes_precision_suffix(self) -> bool {
        matches!(
            self,
            Opcode::Vmla
                | Opcode::Vmls
                | Opcode::Vnmla
                | Opcode::Vnmls
                | Opcode::Vmul
                | Opcode::Vnmul
                | Opcode::Vadd
                | Opcode::Vsub
                | Opcode::Vdiv
                | Opcode::Vfnma
                | Opcode::Vfnms
                | Opcode::Vfma
                | Opcode::Vfms
                | Opcode::Vabs
                | Opcode::Vneg
                | Opcode::Vsqrt
                | Opcode::Vcmp
                | Opcode::Vcmpe
                | Opcode::Vrintr
                | Opcode::Vrintz
                | Opcode::Vrintx
        )
    }

    const fn elem_style(self) -> ElemStyle {
        match self {
            Opcode::Vld1
            | Opcode::Vst1
            | Opcode::Vld2
            | Opcode::Vst2
            | Opcode::Vld3
            | Opcode::Vst3
            | Opcode::Vld4
            | Opcode::Vst4
            | Opcode::Vext
            | Opcode::Vdup
            | Opcode::Vtst
            | Opcode::VmovLane
            | Opcode::Vrev16
            | Opcode::Vrev32
            | Opcode::Vrev64 => ElemStyle::Plain,
            Opcode::Vadd
            | Opcode::Vsub
            | Opcode::Vmul
            | Opcode::Vceq
            | Opcode::VmlaI
            | Opcode::VmlsI
            | Opcode::VmovImm
            | Opcode::VorrImm
            | Opcode::VmvnImm
            | Opcode::VbicImm => ElemStyle::Int,
            Opcode::Vcgt
            | Opcode::Vcge
            | Opcode::Vmax
            | Opcode::Vmin
            | Opcode::Vabs
            | Opcode::Vneg
            | Opcode::VmovLaneS => ElemStyle::Signed,
            Opcode::VcgtU
            | Opcode::VcgeU
            | Opcode::VmaxU
            | Opcode::VminU
            | Opcode::VmovLaneU => ElemStyle::Unsigned,
            Opcode::VaddF
            | Opcode::VsubF
            | Opcode::VmulF
            | Opcode::VmlaF
            | Opcode::VmlsF
            | Opcode::Vfma
            | Opcode::Vfms
            | Opcode::VmaxF
            | Opcode::VminF
            | Opcode::VceqF
            | Opcode::VcgeF
            | Opcode::VabsF
            | Opcode::VnegF => ElemStyle::Float,
            _ => ElemStyle::Hidden,
        }
    }

    fn vector_suffix(self, shape: super::ir::VecShape) -> Option<&'static str> {
        Some(match self.elem_style() {
            ElemStyle::Hidden => return None,
            ElemStyle::Float => "f32",
            ElemStyle::Plain => match shape.esize {
                8 => "8",
                16 => "16",
                32 => "32",
                _ => "64",
            },
            style => {
                let prefix = match style {
                    ElemStyle::Int => "i",
                    ElemStyle::Signed => "s",
                    _ => "u",
                };
                match (prefix, shape.esize) {
                    ("i", 8) => "i8",
                    ("i", 16) => "i16",
                    ("i", 32) => "i32",
                    ("i", _) => "i64",
                    ("s", 8) => "s8",
                    ("s", 16) => "s16",
                    ("s", 32) => "s32",
                    ("s", _) => "s64",
                    (_, 8) => "u8",
                    (_, 16) => "u16",
                    (_, 32) => "u32",
                    _ => "u64",
                }
            }
        })
    }

    pub(crate) fn mnemonic_suffix(self, insn: &super::ir::Insn) -> Option<&'static str> {
        if let Some(shape) = insn.shape {
            return self.vector_suffix(shape);
        }
        if matches!(self, Opcode::Vcvt | Opcode::VcvtR)
            && let Some(super::ir::Operand::Convert(convert)) = insn.operands.get(2)
        {
            return Some(convert.suffix());
        }
        if matches!(self, Opcode::Vmov) {
            let fp = |operand: &super::ir::Operand| {
                matches!(
                    operand,
                    super::ir::Operand::S(_)
                        | super::ir::Operand::D(_)
                        | super::ir::Operand::ImmFloat32(_)
                        | super::ir::Operand::ImmFloat64(_)
                )
            };
            if !insn.operands.iter().all(fp) {
                return None;
            }
        } else if !self.takes_precision_suffix() {
            return None;
        }
        match insn.operands.first()? {
            super::ir::Operand::S(_) => Some("f32"),
            super::ir::Operand::D(_) => Some("f64"),
            _ => None,
        }
    }

    pub const fn takes_s_suffix(self) -> bool {
        matches!(
            self,
            Opcode::And
                | Opcode::Eor
                | Opcode::Sub
                | Opcode::Rsb
                | Opcode::Add
                | Opcode::Adc
                | Opcode::Sbc
                | Opcode::Rsc
                | Opcode::Orr
                | Opcode::Mov
                | Opcode::Bic
                | Opcode::Mvn
                | Opcode::Mul
                | Opcode::Mla
                | Opcode::Umull
                | Opcode::Umlal
                | Opcode::Smull
                | Opcode::Smlal
                | Opcode::Smlabb
                | Opcode::Smlabt
                | Opcode::Smlatb
                | Opcode::Smlatt
                | Opcode::Smlawb
                | Opcode::Smlawt
                | Opcode::Smulbb
                | Opcode::Smulbt
                | Opcode::Smultb
                | Opcode::Smultt
                | Opcode::Smulwb
                | Opcode::Smulwt
                | Opcode::Smlalbb
                | Opcode::Smlalbt
                | Opcode::Smlaltb
                | Opcode::Smlaltt
        )
    }

    pub const fn from_data_processing(field: u32) -> Opcode {
        match field & 0xf {
            0 => Opcode::And,
            1 => Opcode::Eor,
            2 => Opcode::Sub,
            3 => Opcode::Rsb,
            4 => Opcode::Add,
            5 => Opcode::Adc,
            6 => Opcode::Sbc,
            7 => Opcode::Rsc,
            8 => Opcode::Tst,
            9 => Opcode::Teq,
            10 => Opcode::Cmp,
            11 => Opcode::Cmn,
            12 => Opcode::Orr,
            13 => Opcode::Mov,
            14 => Opcode::Bic,
            _ => Opcode::Mvn,
        }
    }

    pub const fn parallel(group: u32, selector: u32) -> Option<Opcode> {
        let index = match selector & 0xf {
            0b0001 => 0,
            0b0011 => 1,
            0b0101 => 2,
            0b0111 => 3,
            0b1001 => 4,
            0b1111 => 5,
            _ => return None,
        };
        let table: [Opcode; 6] = match group & 0x1f {
            0b00001 => [
                Opcode::Sadd16,
                Opcode::Sasx,
                Opcode::Ssax,
                Opcode::Ssub16,
                Opcode::Sadd8,
                Opcode::Ssub8,
            ],
            0b00010 => [
                Opcode::Qadd16,
                Opcode::Qasx,
                Opcode::Qsax,
                Opcode::Qsub16,
                Opcode::Qadd8,
                Opcode::Qsub8,
            ],
            0b00011 => [
                Opcode::Shadd16,
                Opcode::Shasx,
                Opcode::Shsax,
                Opcode::Shsub16,
                Opcode::Shadd8,
                Opcode::Shsub8,
            ],
            0b00101 => [
                Opcode::Uadd16,
                Opcode::Uasx,
                Opcode::Usax,
                Opcode::Usub16,
                Opcode::Uadd8,
                Opcode::Usub8,
            ],
            0b00110 => [
                Opcode::Uqadd16,
                Opcode::Uqasx,
                Opcode::Uqsax,
                Opcode::Uqsub16,
                Opcode::Uqadd8,
                Opcode::Uqsub8,
            ],
            0b00111 => [
                Opcode::Uhadd16,
                Opcode::Uhasx,
                Opcode::Uhsax,
                Opcode::Uhsub16,
                Opcode::Uhadd8,
                Opcode::Uhsub8,
            ],
            _ => return None,
        };
        Some(table[index])
    }
}
