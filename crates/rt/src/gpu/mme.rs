pub trait Host {
    fn write(&mut self, method: u32, value: u32);
    fn read(&mut self, method: u32) -> u32;
}

const STEP_LIMIT: usize = 1 << 20;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Undefined { pc: usize, word: u32 },
    OutOfBounds(usize),
    TooLong,
}

struct Machine<'a, H: Host> {
    regs: [u32; 8],
    carry: bool,
    method: u32,
    increment: u32,
    params: std::slice::Iter<'a, u32>,
    host: &'a mut H,
}

impl<H: Host> Machine<'_, H> {
    fn set(&mut self, r: u32, v: u32) {
        if r != 0 {
            self.regs[r as usize] = v;
        }
    }

    fn param(&mut self) -> u32 {
        self.params.next().copied().unwrap_or(0)
    }

    fn set_method(&mut self, v: u32) {
        self.method = v & 0xfff;
        self.increment = (v >> 12) & 0x3f;
    }

    fn send(&mut self, v: u32) {
        self.host.write(self.method, v);
        self.method = (self.method + self.increment) & 0xfff;
    }

    fn step(&mut self, pc: usize, w: u32) -> Result<(), Error> {
        let (dst, a, b) = (
            (w >> 8) & 7,
            self.regs[((w >> 11) & 7) as usize],
            self.regs[((w >> 14) & 7) as usize],
        );
        let imm = ((w as i32) >> 14) as u32;
        let (src_pos, size, dst_pos) = ((w >> 17) & 31, (w >> 22) & 31, (w >> 27) & 31);
        let mask = ((1u64 << size) - 1) as u32;
        let result = match w & 7 {
            0 => {
                let (r, c) = match (w >> 17) & 31 {
                    0 => add(a, b, false),
                    1 => add(a, b, self.carry),
                    2 => add(a, !b, true),
                    3 => add(a, !b, self.carry),
                    8 => (a ^ b, self.carry),
                    9 => (a | b, self.carry),
                    10 => (a & b, self.carry),
                    11 => (a & !b, self.carry),
                    12 => (!(a & b), self.carry),
                    _ => return Err(Error::Undefined { pc, word: w }),
                };
                self.carry = c;
                r
            }
            1 => a.wrapping_add(imm),
            2 => (a & !(mask << dst_pos)) | (((b >> src_pos) & mask) << dst_pos),
            3 => ((b >> (a & 31)) & mask) << dst_pos,
            4 => ((b >> src_pos) & mask) << (a & 31),
            5 => {
                let v = self.host.read(a.wrapping_add(imm) & 0xfff);
                self.set(dst, v);
                return Ok(());
            }
            _ => return Err(Error::Undefined { pc, word: w }),
        };
        match (w >> 4) & 7 {
            0 => {
                let p = self.param();
                self.set(dst, p);
            }
            1 => self.set(dst, result),
            2 => {
                self.set_method(result);
                self.set(dst, result);
            }
            3 => {
                self.send(result);
                let p = self.param();
                self.set(dst, p);
            }
            4 => {
                self.send(result);
                self.set(dst, result);
            }
            5 => {
                self.set_method(result);
                let p = self.param();
                self.set(dst, p);
            }
            6 => {
                self.set_method(result);
                self.set(dst, result);
                let p = self.param();
                self.send(p);
            }
            _ => {
                self.set_method(result);
                self.send((result >> 12) & 0x3f);
                self.set(dst, result);
            }
        }
        Ok(())
    }
}

fn add(a: u32, b: u32, carry: bool) -> (u32, bool) {
    let r = a as u64 + b as u64 + carry as u64;
    (r as u32, r >> 32 != 0)
}

pub fn run(code: &[u32], start: usize, params: &[u32], host: &mut impl Host) -> Result<(), Error> {
    let mut m = Machine {
        regs: [0; 8],
        carry: false,
        method: 0,
        increment: 0,
        params: params.iter(),
        host,
    };
    m.regs[1] = m.param();
    let mut pc = start;
    let mut branch: Option<usize> = None;
    let mut exit = false;
    for _ in 0..STEP_LIMIT {
        let w = code.get(pc).copied().ok_or(Error::OutOfBounds(pc))?;
        let mut next = pc + 1;
        let (mut new_branch, mut new_exit) = (None, false);
        if w & 7 == 7 {
            let a = m.regs[((w >> 11) & 7) as usize];
            let taken = (a == 0) != (w & 0x10 != 0);
            let target = pc.wrapping_add_signed(((w as i32) >> 14) as isize);
            match (taken, w & 0x20 != 0) {
                (true, true) => next = target,
                (true, false) => new_branch = Some(target),
                (false, _) => new_exit = w & 0x80 != 0,
            }
        } else {
            m.step(pc, w)?;
            new_exit = w & 0x80 != 0;
        }
        if let Some(target) = branch.take() {
            next = target;
        }
        if exit {
            return Ok(());
        }
        (branch, exit, pc) = (new_branch, new_exit, next);
    }
    Err(Error::TooLong)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Log {
        writes: Vec<(u32, u32)>,
        regs: std::collections::HashMap<u32, u32>,
    }

    impl Host for Log {
        fn write(&mut self, method: u32, value: u32) {
            self.writes.push((method, value));
        }
        fn read(&mut self, method: u32) -> u32 {
            self.regs.get(&method).copied().unwrap_or(0)
        }
    }

    const PARM: u32 = 0 << 4;
    const MOV: u32 = 1 << 4;
    const MADDR: u32 = 2 << 4;
    const SEND: u32 = 4 << 4;
    const EXIT: u32 = 0x80;

    fn alu(op: u32, asg: u32, d: u32, a: u32, b: u32) -> u32 {
        asg | (d << 8) | (a << 11) | (b << 14) | (op << 17)
    }
    fn addi(asg: u32, d: u32, a: u32, imm: i32) -> u32 {
        1 | asg | (d << 8) | (a << 11) | ((imm as u32) << 14)
    }
    fn braz(a: u32, off: i32, annul: bool) -> u32 {
        7 | ((annul as u32) << 5) | (a << 11) | ((off as u32) << 14)
    }
    fn branz(a: u32, off: i32, annul: bool) -> u32 {
        braz(a, off, annul) | 0x10
    }

    #[test]
    fn sends_parameters_to_consecutive_methods() {
        let code = [
            addi(MADDR, 0, 1, 0),
            addi(PARM, 2, 0, 0),
            addi(SEND, 0, 2, 0),
            addi(PARM, 2, 0, 0),
            addi(SEND, 0, 2, 0) | EXIT,
            addi(MOV, 3, 0, 7),
        ];
        let mut log = Log::default();
        run(&code, 0, &[0x1000 | 0x200, 0xaa, 0xbb], &mut log).unwrap();
        assert_eq!(log.writes, vec![(0x200, 0xaa), (0x201, 0xbb)]);
    }

    #[test]
    fn loop_with_delay_slot_and_carry() {
        let code = [
            addi(MADDR, 0, 0, 0x300),
            addi(MOV, 2, 0, 5),
            addi(SEND, 2, 2, 1),
            addi(MOV, 1, 1, -1),
            branz(1, -2, false),
            addi(MOV, 4, 0, 1),
            alu(0, MOV, 5, 0, 0) | EXIT,
            0,
        ];
        let mut log = Log::default();
        run(&code, 0, &[3], &mut log).unwrap();
        assert_eq!(log.writes, vec![(0x300, 6), (0x300, 7), (0x300, 8)]);
        let code = [
            addi(MOV, 2, 0, -1),
            addi(MOV, 3, 0, 1),
            alu(0, MOV, 4, 2, 3),
            alu(1, MOV, 5, 0, 0),
            addi(MADDR, 0, 0, 0x10),
            addi(SEND, 0, 5, 0) | EXIT,
            0,
        ];
        let mut log = Log::default();
        run(&code, 0, &[0], &mut log).unwrap();
        assert_eq!(log.writes, vec![(0x10, 1)]);
    }

    #[test]
    fn exit_in_the_delay_slot_of_a_taken_branch() {
        let code = [
            addi(MADDR, 0, 0, 0x30),
            addi(SEND, 0, 0, 1),
            braz(0, -1, false),
            addi(MOV, 2, 0, 0) | EXIT,
            0,
        ];
        let mut log = Log::default();
        run(&code, 0, &[0], &mut log).unwrap();
        assert_eq!(log.writes, vec![(0x30, 1), (0x30, 1)]);
    }

    #[test]
    fn bitfields_reads_and_annulled_branches() {
        let mut log = Log::default();
        log.regs.insert(0x123, 0xabcd_1234);
        let code = [
            5 | MOV | (2 << 8) | (0x123 << 14),
            4 | MOV | (3 << 8) | (2 << 14) | (8 << 17) | (8 << 22),
            braz(0, 3, true),
            addi(MOV, 3, 0, 0x7777),
            0,
            addi(MADDR, 0, 0, 0x20),
            addi(SEND, 0, 3, 0) | EXIT,
            0,
        ];
        run(&code, 0, &[0], &mut log).unwrap();
        assert_eq!(log.writes, vec![(0x20, 0x12)]);
    }
}
