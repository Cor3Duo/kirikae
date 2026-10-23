pub const ADPCM_SAMPLES_PER_FRAME: u64 = 14;
const ADPCM_NIBBLES_PER_FRAME: u64 = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdpcmContext {
    pub pred_scale: u16,
    pub history0: i16,
    pub history1: i16,
}

impl AdpcmContext {
    pub fn parse(d: &[u8]) -> AdpcmContext {
        let h = |o: usize| d.get(o..o + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]]));
        AdpcmContext { pred_scale: h(0), history0: h(2) as i16, history1: h(4) as i16 }
    }
}

pub fn adpcm_nibble(sample: u64) -> u64 {
    let in_frame = sample % ADPCM_SAMPLES_PER_FRAME;
    let header = if in_frame != 0 { 2 } else { 0 };
    sample / ADPCM_SAMPLES_PER_FRAME * ADPCM_NIBBLES_PER_FRAME + in_frame + header
}

pub fn adpcm_size(samples: u64) -> u64 {
    adpcm_nibble(samples).div_ceil(2)
}

pub fn decode_adpcm(data: &[u8], start: u64, count: usize, coefficients: &[i16; 16], ctx: &mut AdpcmContext, out: &mut Vec<f32>) -> usize {
    let first = adpcm_nibble(start);
    let mut nibble = first - (first & !1);
    let mut scale = u32::from(ctx.pred_scale & 0xf);
    let mut index = usize::from((ctx.pred_scale >> 4) & 7);
    let (mut h0, mut h1) = (i64::from(ctx.history0), i64::from(ctx.history1));
    let mut done = 0;
    let mut abs = first;
    while done < count {
        if abs.is_multiple_of(ADPCM_NIBBLES_PER_FRAME) {
            let Some(&header) = data.get((nibble / 2) as usize) else { break };
            ctx.pred_scale = u16::from(header);
            scale = u32::from(header & 0xf);
            index = usize::from((header >> 4) & 7);
            nibble += 2;
            abs += 2;
        }
        let Some(&byte) = data.get((nibble / 2) as usize) else { break };
        let value = if nibble & 1 == 0 { (byte as i8) >> 4 } else { ((byte << 4) as i8) >> 4 };
        let prediction = i64::from(coefficients[index * 2]) * h0 + i64::from(coefficients[index * 2 + 1]) * h1;
        let sample = (((i64::from(value) << 11) << scale) + prediction + 1024) >> 11;
        let sample = sample.clamp(i16::MIN.into(), i16::MAX.into());
        h1 = h0;
        h0 = sample;
        out.push(sample as f32);
        nibble += 1;
        abs += 1;
        done += 1;
    }
    ctx.history0 = h0 as i16;
    ctx.history1 = h1 as i16;
    done
}

pub const HISTORY: usize = 3;

pub fn resample(input: &[f32], out: &mut [f32], ratio: f32, fraction: f32) -> f32 {
    let last = input.len().saturating_sub(4);
    for (k, o) in out.iter_mut().enumerate() {
        let pos = fraction + ratio * k as f32;
        let i = (pos as usize).min(last);
        let t = (pos - i as f32).min(1.0);
        let p0 = input[i];
        let p1 = input[i + 1];
        let p2 = input[i + 2];
        let p3 = input[i + 3];
        *o = p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)));
    }
    let end = fraction + ratio * out.len() as f32;
    end - end.floor()
}

pub fn needed(fraction: f32, ratio: f32, count: usize) -> usize {
    (fraction + ratio * count as f32) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_stays_within_the_samples_it_needs() {
        for &ratio in &[0.6667f32, 1.0884, 0.9977, 1.3333, 0.3] {
            let mut fraction = 0.0;
            let mut out = vec![0.0; 240];
            for _ in 0..2000 {
                let input = vec![0.5; HISTORY + needed(fraction, ratio, out.len())];
                fraction = resample(&input, &mut out, ratio, fraction);
                assert!((0.0..1.0).contains(&fraction));
            }
        }
    }

    #[test]
    fn nibble_positions() {
        assert_eq!(adpcm_nibble(0), 0);
        assert_eq!(adpcm_nibble(1), 3);
        assert_eq!(adpcm_nibble(13), 15);
        assert_eq!(adpcm_nibble(14), 16);
        assert_eq!(adpcm_nibble(15), 19);
        assert_eq!(adpcm_size(14), 8);
        assert_eq!(adpcm_size(28), 16);
    }

    fn frame(header: u8, nibbles: [i8; 14]) -> Vec<u8> {
        let mut f = vec![header];
        for pair in nibbles.chunks(2) {
            f.push(((pair[0] as u8 & 0xf) << 4) | (pair[1] as u8 & 0xf));
        }
        f
    }

    #[test]
    fn adpcm_decodes_scaled_nibbles_with_prediction() {
        let mut coefficients = [0i16; 16];
        coefficients[2] = 2048;
        let mut data = frame(0x04, [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]);
        data.extend(frame(0x10, [1; 14]));
        let mut ctx = AdpcmContext::default();
        let mut out = Vec::new();
        assert_eq!(decode_adpcm(&data, 0, 28, &coefficients, &mut ctx, &mut out), 28);
        assert_eq!(&out[..4], &[16.0, 0.0, 0.0, -16.0]);
        assert_eq!(out[13], 112.0);
        assert_eq!(out[14], 113.0);
        assert_eq!(out[27], 126.0);
        assert_eq!(ctx, AdpcmContext { pred_scale: 0x10, history0: 126, history1: 125 });

        let mut ctx2 = AdpcmContext { pred_scale: 0x10, history0: 113, history1: 112 };
        let mut tail = Vec::new();
        let first = adpcm_nibble(15) & !1;
        assert_eq!(decode_adpcm(&data[(first / 2) as usize..], 15, 13, &coefficients, &mut ctx2, &mut tail), 13);
        assert_eq!(tail, out[15..].to_vec());
    }

    #[test]
    fn resampling_at_unit_ratio_delays_by_one_sample() {
        let input = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0; 4];
        let f = resample(&input, &mut out, 1.0, 0.0);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(f, 0.0);
        assert_eq!(needed(0.0, 1.0, 4), 4);
    }

    #[test]
    fn resampling_halves_and_doubles() {
        let ramp: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut out = [0.0; 8];
        let f = resample(&ramp, &mut out, 2.0, 0.0);
        assert_eq!(out, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]);
        assert_eq!(f, 0.0);
        let f = resample(&ramp, &mut out, 0.5, 0.0);
        assert_eq!(out, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]);
        assert_eq!(f, 0.0);
        assert_eq!(needed(0.0, 0.5, 8), 4);
    }
}
