import struct
import sys

from unicorn import Uc, UcError, UC_ARCH_ARM64, UC_MODE_ARM
from unicorn.arm64_const import *

CODE = 0x1000
MEM_BASE = 0x10_0000
MEM_SIZE = 0x1_0000
STATE = 8 + 8 + 31 * 8 + 8 + 32 * 16
PATTERN = bytes(((i * 167 + 13) & 0xFF) for i in range(MEM_SIZE))

X = [getattr(__import__("unicorn.arm64_const", fromlist=["x"]), f"UC_ARM64_REG_X{i}") for i in range(29)]
X += [UC_ARM64_REG_X29, UC_ARM64_REG_X30]
Q = [getattr(__import__("unicorn.arm64_const", fromlist=["x"]), f"UC_ARM64_REG_Q{i}") for i in range(32)]

def make_uc():
    uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    uc.ctl_set_cpu_model(UC_CPU_ARM64_A57)
    uc.reg_write(UC_ARM64_REG_CPACR_EL1, 0x300000)
    uc.mem_map(CODE, 0x1000)
    uc.mem_map(MEM_BASE, MEM_SIZE)
    return uc

def main(cases_path, results_path):
    data = open(cases_path, "rb").read()
    uc = make_uc()
    out = bytearray()
    case = 4 + STATE
    for k in range(0, len(data) - case + 1, case):
        word = struct.unpack_from("<I", data, k)[0]
        s = struct.unpack_from("<" + "Q" * (STATE // 8), data, k + 4)
        fpcr, nzcv = s[0], s[1]
        xs, sp, vs = s[2:33], s[33], s[34:]
        is_ldst = (word >> 25) & 0b111 != 0b111
        if is_ldst:
            uc.mem_write(MEM_BASE, PATTERN)
        uc.mem_write(CODE, struct.pack("<I", word))
        for i in range(31):
            uc.reg_write(X[i], xs[i])
        uc.reg_write(UC_ARM64_REG_SP, sp)
        for i in range(32):
            uc.reg_write(Q[i], vs[2 * i] | (vs[2 * i + 1] << 64))
        uc.reg_write(UC_ARM64_REG_NZCV, nzcv)
        uc.reg_write(UC_ARM64_REG_FPCR, fpcr)
        uc.reg_write(UC_ARM64_REG_FPSR, 0)
        status = 1
        try:
            uc.emu_start(CODE, CODE + 4, count=1)
            if uc.reg_read(UC_ARM64_REG_PC) != CODE + 4:
                status = 0
        except UcError:
            status = 0
            uc = make_uc()
        out.append(status)
        if status:
            st = [fpcr, uc.reg_read(UC_ARM64_REG_NZCV)]
            st += [uc.reg_read(X[i]) for i in range(31)]
            st.append(uc.reg_read(UC_ARM64_REG_SP))
            for i in range(32):
                q = uc.reg_read(Q[i])
                st += [q & (2**64 - 1), q >> 64]
            out += struct.pack("<" + "Q" * len(st), *st)
            out += struct.pack("<Q", uc.reg_read(UC_ARM64_REG_FPSR))
            changes = []
            if is_ldst:
                mem = uc.mem_read(MEM_BASE, MEM_SIZE)
                changes = [(i, mem[i]) for i in range(MEM_SIZE) if mem[i] != PATTERN[i]]
            out += struct.pack("<I", len(changes))
            for off, b in changes:
                out += struct.pack("<IB", off, b)
        else:
            out += bytes(STATE + 8) + struct.pack("<I", 0)
    open(results_path, "wb").write(out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
