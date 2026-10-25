//! Module-wide SPIR-V checks shared by the translator tests, covering the rules
//! `spirv-val` reports beyond what naga validates.

use rspirv::binary::Disassemble;
use rspirv::dr::{Loader, Operand};
use rspirv::spirv::Op;

/// The structural rules a whole module must follow: every function-local
/// variable is part of the entry block's leading variables, and no id carries
/// the same decoration twice.
pub fn check_structure(spirv: &[u32]) {
    let mut loader = Loader::new();
    rspirv::binary::parse_words(spirv, &mut loader).unwrap();
    let module = loader.module();
    for function in &module.functions {
        for (k, block) in function.blocks.iter().enumerate() {
            let mut body = k > 0;
            for inst in &block.instructions {
                if inst.class.opcode == Op::Variable {
                    assert!(
                        !body,
                        "OpVariable after the entry block's leading variables:\n{}",
                        module.disassemble()
                    );
                } else {
                    body = true;
                }
            }
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let (Operand::IdRef(id), Operand::Decoration(kind)) =
            (&inst.operands[0], &inst.operands[1])
        else {
            continue;
        };
        assert!(
            seen.insert((*id, format!("{kind:?}"))),
            "duplicate decoration {kind:?} on %{id}:\n{}",
            module.disassemble()
        );
    }
}
