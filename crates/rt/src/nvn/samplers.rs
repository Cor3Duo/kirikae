//! NVN sampler state, decoded from the builder's setters.
//!
//! The enums are the public NVN ones and stay raw: the Vulkan mapping belongs to milestone 4.6. Defaults match the
//! real sampler builder: nearest filtering, clamp wrapping, no compare, LOD clamp 0..1000, bias 0, anisotropy 1,
//! transparent black border (`nvn_sampler_objects.cpp` decodes the same fields through the driver's getters).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub wrap: [u32; 3],
    pub compare_mode: u32,
    pub compare_func: u32,
    pub reduction: u32,
    pub lod_min: f32,
    pub lod_max: f32,
    pub lod_bias: f32,
    pub max_anisotropy: f32,
    pub border: [f32; 4],
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler {
            min_filter: 0,
            mag_filter: 0,
            wrap: [0; 3],
            compare_mode: 0,
            compare_func: 0,
            reduction: 0,
            lod_min: 0.0,
            lod_max: 1000.0,
            lod_bias: 0.0,
            max_anisotropy: 1.0,
            border: [0.0; 4],
        }
    }
}

/// One `nvnSamplerBuilderSet*` proc. Integer arguments arrive in `x`, floats in the vector registers (`v`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Defaults,
    MinMagFilter,
    WrapMode,
    LodClamp,
    LodBias,
    Compare,
    MaxAnisotropy,
    ReductionFilter,
    BorderColor,
}

impl Sampler {
    /// Applies a builder setter. `border` is the four floats `nvnSamplerBuilderSetBorderColor` points at.
    pub fn apply(&mut self, field: Field, x: &[u64], v: &[f32], border: [f32; 4]) {
        match field {
            Field::Defaults => *self = Sampler::default(),
            Field::MinMagFilter => {
                self.min_filter = x[1] as u32;
                self.mag_filter = x[2] as u32;
            }
            Field::WrapMode => self.wrap = [x[1] as u32, x[2] as u32, x[3] as u32],
            Field::LodClamp => {
                self.lod_min = v[0];
                self.lod_max = v[1];
            }
            Field::LodBias => self.lod_bias = v[0],
            Field::Compare => {
                self.compare_mode = x[1] as u32;
                self.compare_func = x[2] as u32;
            }
            Field::MaxAnisotropy => self.max_anisotropy = v[0],
            Field::ReductionFilter => self.reduction = x[1] as u32,
            Field::BorderColor => self.border = border,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_builder() {
        let s = Sampler::default();
        assert_eq!((s.min_filter, s.mag_filter), (0, 0));
        assert_eq!(s.wrap, [0; 3]);
        assert_eq!((s.lod_min, s.lod_max, s.lod_bias), (0.0, 1000.0, 0.0));
        assert_eq!(s.max_anisotropy, 1.0);
        assert_eq!(s.border, [0.0; 4]);
    }

    #[test]
    fn setters_decode_integer_and_float_arguments() {
        let mut s = Sampler::default();
        s.apply(Field::MinMagFilter, &[0x100, 5, 1, 0], &[0.0; 4], [0.0; 4]);
        s.apply(Field::WrapMode, &[0x100, 1, 2, 5, 0], &[0.0; 4], [0.0; 4]);
        s.apply(
            Field::LodClamp,
            &[0x100, 0, 0, 0],
            &[-1000.0, 12.5, 0.0, 0.0],
            [0.0; 4],
        );
        s.apply(
            Field::LodBias,
            &[0x100, 0, 0, 0],
            &[-0.25, 0.0, 0.0, 0.0],
            [0.0; 4],
        );
        s.apply(Field::Compare, &[0x100, 1, 7, 0], &[0.0; 4], [0.0; 4]);
        s.apply(
            Field::MaxAnisotropy,
            &[0x100, 0, 0, 0],
            &[8.0, 0.0, 0.0, 0.0],
            [0.0; 4],
        );
        s.apply(
            Field::ReductionFilter,
            &[0x100, 3, 0, 0],
            &[0.0; 4],
            [0.0; 4],
        );
        s.apply(
            Field::BorderColor,
            &[0x100, 0, 0, 0],
            &[0.0; 4],
            [1.0, 0.5, 0.25, 1.0],
        );
        assert_eq!((s.min_filter, s.mag_filter), (5, 1));
        assert_eq!(s.wrap, [1, 2, 5]);
        assert_eq!((s.lod_min, s.lod_max), (-1000.0, 12.5));
        assert_eq!(s.lod_bias, -0.25);
        assert_eq!((s.compare_mode, s.compare_func), (1, 7));
        assert_eq!(s.max_anisotropy, 8.0);
        assert_eq!(s.reduction, 3);
        assert_eq!(s.border, [1.0, 0.5, 0.25, 1.0]);

        s.apply(Field::Defaults, &[], &[], [0.0; 4]);
        assert_eq!(s, Sampler::default());
    }
}
