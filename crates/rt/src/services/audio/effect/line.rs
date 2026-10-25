//! Shared delay-line primitives for the delay and reverb effects.

pub(super) struct DelayLine {
    buf: Vec<f32>,
    at: usize,
}

impl DelayLine {
    pub(super) fn new(max: usize) -> DelayLine {
        DelayLine {
            buf: vec![0.0; max.max(2)],
            at: 0,
        }
    }

    pub(super) fn reset(&mut self) {
        self.buf.fill(0.0);
        self.at = 0;
    }

    pub(super) fn len(&self) -> usize {
        self.buf.len()
    }

    /// Value written `delay` writes ago; `1` is the most recent write.
    #[inline]
    pub(super) fn read(&self, delay: usize) -> f32 {
        let len = self.buf.len();
        let d = delay.clamp(1, len - 1);
        self.buf[(self.at + len - d) % len]
    }

    #[inline]
    pub(super) fn write(&mut self, v: f32) {
        self.buf[self.at] = v;
        self.at += 1;
        if self.at == self.buf.len() {
            self.at = 0;
        }
    }
}

pub(super) fn ms_samples(rate: u32, ms: f32) -> usize {
    let n = (rate as f32 * ms).round();
    if n.is_finite() {
        (n as usize).max(2)
    } else {
        2
    }
}
