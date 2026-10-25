use crate::gpu::fxhash::FastMap as HashMap;

use crate::gpu::threed::reg;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Field(pub Kind, pub u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    ChannelMask,
    BlendEnable,
    BlendEquation,
    BlendSrc,
    BlendDst,
    BlendEquationAlpha,
    BlendSrcAlpha,
    BlendDstAlpha,
    DepthTest,
    DepthWrite,
    DepthFunc,
    StencilTest,
    StencilFunc,
    StencilFail,
    StencilDepthFail,
    StencilPass,
    CullFace,
    FrontFace,
    PolygonOffsetEnables,
    AttribFormat,
    AttribOffset,
    AttribStream,
    StreamStride,
    StreamDivisor,
    BlendColor,
    Viewport,
    Scissor,
    TargetFormat,
    BlendRegister,
    AttribCount,
    StreamCount,
    DepthFormat,
    PolygonOffset,
    StencilRef,
    StencilValueMask,
    StencilMask,
    DepthClamp,
    RasterizerDiscard,
    PointSize,
    LineWidth,
    DepthBoundsEnable,
    DepthBoundsMin,
    DepthBoundsMax,
}

#[derive(Default)]
pub struct Objects {
    channel_mask: HashMap<u64, [u8; 8]>,
    color: HashMap<u64, u8>,
    blend: HashMap<u64, [u8; 7]>,
    depth_stencil: HashMap<u64, DepthStencil>,
    polygon: HashMap<u64, [u8; 4]>,
    attrib: HashMap<u64, [u64; 3]>,
    stream: HashMap<u64, [u64; 2]>,
    pub unknown: u64,
}

#[derive(Clone, Copy)]
struct DepthStencil {
    test: u8,
    write: u8,
    func: u8,
    stencil: u8,
    faces: [[u8; 4]; 2],
}

const DEPTH_STENCIL_DEFAULTS: DepthStencil = DepthStencil {
    test: 0,
    write: 1,
    func: 2,
    stencil: 0,
    faces: [[8, 1, 1, 1]; 2],
};

pub const ATTRIB_SIZE: u64 = 4;
pub const STREAM_SIZE: u64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setter {
    ChannelMaskDefaults,
    ChannelMask,
    ColorDefaults,
    BlendEnable,
    BlendDefaults,
    BlendTarget,
    BlendFunc,
    BlendEquation,
    DepthStencilDefaults,
    DepthTest,
    DepthWrite,
    DepthFunc,
    StencilTest,
    StencilFunc,
    StencilOp,
    PolygonDefaults,
    CullFace,
    FrontFace,
    PolygonMode,
    PolygonOffsetEnables,
    AttribDefaults,
    AttribFormat,
    AttribStream,
    StreamDefaults,
    StreamStride,
    StreamDivisor,
}

pub fn setter(name: &str) -> Option<Setter> {
    use Setter::*;
    Some(match name.strip_prefix("nvn")? {
        "ChannelMaskStateSetDefaults" => ChannelMaskDefaults,
        "ChannelMaskStateSetChannelMask" => ChannelMask,
        "ColorStateSetDefaults" => ColorDefaults,
        "ColorStateSetBlendEnable" => BlendEnable,
        "BlendStateSetDefaults" => BlendDefaults,
        "BlendStateSetBlendTarget" => BlendTarget,
        "BlendStateSetBlendFunc" => BlendFunc,
        "BlendStateSetBlendEquation" => BlendEquation,
        "DepthStencilStateSetDefaults" => DepthStencilDefaults,
        "DepthStencilStateSetDepthTestEnable" => DepthTest,
        "DepthStencilStateSetDepthWriteEnable" => DepthWrite,
        "DepthStencilStateSetDepthFunc" => DepthFunc,
        "DepthStencilStateSetStencilTestEnable" => StencilTest,
        "DepthStencilStateSetStencilFunc" => StencilFunc,
        "DepthStencilStateSetStencilOp" => StencilOp,
        "PolygonStateSetDefaults" => PolygonDefaults,
        "PolygonStateSetCullFace" => CullFace,
        "PolygonStateSetFrontFace" => FrontFace,
        "PolygonStateSetPolygonMode" => PolygonMode,
        "PolygonStateSetPolygonOffsetEnables" => PolygonOffsetEnables,
        "VertexAttribStateSetDefaults" => AttribDefaults,
        "VertexAttribStateSetFormat" => AttribFormat,
        "VertexAttribStateSetStreamIndex" => AttribStream,
        "VertexStreamStateSetDefaults" => StreamDefaults,
        "VertexStreamStateSetStride" => StreamStride,
        "VertexStreamStateSetDivisor" => StreamDivisor,
        _ => return None,
    })
}

impl Objects {
    pub fn set(&mut self, s: Setter, a: &[u64; 31]) {
        use Setter::*;
        let o = a[0];
        let b = |v: u64| v as u8;
        let on = |v: u64| v as u8 != 0;
        match s {
            ChannelMaskDefaults => {
                self.channel_mask.insert(o, [0xf; 8]);
            }
            ChannelMask => {
                let m = self.channel_mask.entry(o).or_insert([0xf; 8]);
                if let Some(t) = m.get_mut(a[1] as u32 as usize) {
                    *t = on(a[2]) as u8
                        | (on(a[3]) as u8) << 1
                        | (on(a[4]) as u8) << 2
                        | (on(a[5]) as u8) << 3;
                }
            }
            ColorDefaults => {
                self.color.insert(o, 0);
            }
            BlendEnable => {
                let m = self.color.entry(o).or_insert(0);
                let bit = 1u8.checked_shl(a[1] as u32).unwrap_or(0);
                *m = if on(a[2]) { *m | bit } else { *m & !bit };
            }
            BlendDefaults => {
                self.blend.insert(o, [0, 2, 1, 2, 1, 1, 1]);
            }
            BlendTarget | BlendFunc | BlendEquation => {
                let v = self.blend.entry(o).or_insert([0, 2, 1, 2, 1, 1, 1]);
                match s {
                    BlendTarget => v[0] = b(a[1]),
                    BlendFunc => v[1..5].copy_from_slice(&[b(a[1]), b(a[2]), b(a[3]), b(a[4])]),
                    _ => v[5..7].copy_from_slice(&[b(a[1]), b(a[2])]),
                }
            }
            DepthStencilDefaults => {
                self.depth_stencil.insert(o, DEPTH_STENCIL_DEFAULTS);
            }
            DepthTest | DepthWrite | DepthFunc | StencilTest | StencilFunc | StencilOp => {
                let d = self
                    .depth_stencil
                    .entry(o)
                    .or_insert(DEPTH_STENCIL_DEFAULTS);
                match s {
                    DepthTest => d.test = b(a[1]),
                    DepthWrite => d.write = b(a[1]),
                    DepthFunc => d.func = b(a[1]),
                    StencilTest => d.stencil = b(a[1]),
                    StencilFunc => {
                        for f in 0..2 {
                            if a[1] >> f & 1 == 1 {
                                d.faces[f][0] = b(a[2]);
                            }
                        }
                    }
                    _ => {
                        for f in 0..2 {
                            if a[1] >> f & 1 == 1 {
                                d.faces[f][1..4].copy_from_slice(&[b(a[2]), b(a[3]), b(a[4])]);
                            }
                        }
                    }
                }
            }
            PolygonDefaults => {
                self.polygon.insert(o, [0, 1, 2, 0]);
            }
            CullFace | FrontFace | PolygonMode | PolygonOffsetEnables => {
                let p = self.polygon.entry(o).or_insert([0, 1, 2, 0]);
                let i = match s {
                    CullFace => 0,
                    FrontFace => 1,
                    PolygonMode => 2,
                    _ => 3,
                };
                p[i] = b(a[1]);
            }
            AttribDefaults => {
                self.attrib.insert(o, [0; 3]);
            }
            AttribFormat => {
                let v = self.attrib.entry(o).or_insert([0; 3]);
                v[0] = a[1] as u32 as u64;
                v[1] = a[2] as u32 as u64;
            }
            AttribStream => self.attrib.entry(o).or_insert([0; 3])[2] = a[1] as u32 as u64,
            StreamDefaults => {
                self.stream.insert(o, [0; 2]);
            }
            StreamStride => self.stream.entry(o).or_insert([0; 2])[0] = a[1] as u32 as u64,
            StreamDivisor => self.stream.entry(o).or_insert([0; 2])[1] = a[1] as u32 as u64,
        }
    }

    pub fn channel_mask(&mut self, o: u64) -> Vec<(Field, u64)> {
        let Some(m) = self.channel_mask.get(&o).copied() else {
            return self.missed();
        };
        (0..8)
            .map(|t| (Field(Kind::ChannelMask, t), m[t as usize] as u64))
            .collect()
    }

    pub fn color(&mut self, o: u64) -> Vec<(Field, u64)> {
        let Some(m) = self.color.get(&o).copied() else {
            return self.missed();
        };
        (0..8)
            .map(|t| (Field(Kind::BlendEnable, t), (m >> t & 1) as u64))
            .collect()
    }

    pub fn blend(&mut self, o: u64) -> Vec<(Field, u64)> {
        let Some(v) = self.blend.get(&o).copied() else {
            return self.missed();
        };
        let t = v[0];
        [
            (Kind::BlendSrc, v[1]),
            (Kind::BlendDst, v[2]),
            (Kind::BlendSrcAlpha, v[3]),
            (Kind::BlendDstAlpha, v[4]),
            (Kind::BlendEquation, v[5]),
            (Kind::BlendEquationAlpha, v[6]),
        ]
        .into_iter()
        .map(|(k, x)| (Field(k, t), x as u64))
        .collect()
    }

    pub fn depth_stencil(&mut self, o: u64) -> Vec<(Field, u64)> {
        let Some(d) = self.depth_stencil.get(&o).copied() else {
            return self.missed();
        };
        let mut out = vec![
            (Field(Kind::DepthTest, 0), d.test as u64),
            (Field(Kind::DepthWrite, 0), d.write as u64),
            (Field(Kind::DepthFunc, 0), d.func as u64),
            (Field(Kind::StencilTest, 0), d.stencil as u64),
        ];
        for f in 0..2 {
            let x = d.faces[f];
            out.extend(
                [
                    (Kind::StencilFunc, x[0]),
                    (Kind::StencilFail, x[1]),
                    (Kind::StencilDepthFail, x[2]),
                    (Kind::StencilPass, x[3]),
                ]
                .map(|(k, v)| (Field(k, f as u8), v as u64)),
            );
        }
        out
    }

    pub fn polygon(&mut self, o: u64) -> Vec<(Field, u64)> {
        let Some(p) = self.polygon.get(&o).copied() else {
            return self.missed();
        };
        vec![
            (Field(Kind::CullFace, 0), p[0] as u64),
            (Field(Kind::FrontFace, 0), p[1] as u64),
            (Field(Kind::PolygonOffsetEnables, 0), p[3] as u64),
        ]
    }

    pub fn attribs(&mut self, base: u64, count: u64) -> Vec<(Field, u64)> {
        let mut out = vec![(Field(Kind::AttribCount, 0), count.min(32))];
        for i in 0..count.min(32) {
            match self.attrib.get(&(base + i * ATTRIB_SIZE)).copied() {
                Some(v) => out.extend(
                    [
                        (Kind::AttribFormat, v[0]),
                        (Kind::AttribOffset, v[1]),
                        (Kind::AttribStream, v[2]),
                    ]
                    .map(|(k, x)| (Field(k, i as u8), x)),
                ),
                None => self.unknown += 1,
            }
        }
        out
    }

    pub fn streams(&mut self, base: u64, count: u64) -> Vec<(Field, u64)> {
        let mut out = vec![(Field(Kind::StreamCount, 0), count.min(16))];
        for i in 0..count.min(16) {
            match self.stream.get(&(base + i * STREAM_SIZE)).copied() {
                Some(v) => out.extend(
                    [(Kind::StreamStride, v[0]), (Kind::StreamDivisor, v[1])]
                        .map(|(k, x)| (Field(k, i as u8), x)),
                ),
                None => self.unknown += 1,
            }
        }
        out
    }

    fn missed(&mut self) -> Vec<(Field, u64)> {
        self.unknown += 1;
        Vec::new()
    }
}

pub fn from_registers(regs: &[u32]) -> Vec<(Field, u64)> {
    let mut out = Vec::with_capacity(256);
    let mut put = |k: Kind, i: u8, v: u64| out.push((Field(k, i), v));
    let per_target = regs[0x4b9] & 1 == 1;
    for (i, r) in [0x4b9]
        .into_iter()
        .chain(0x4cf..0x4d7)
        .chain(0x780..0x787)
        .enumerate()
    {
        put(Kind::BlendRegister, i as u8, regs[r] as u64);
    }
    let ct = regs[reg::CT_SELECT];
    for k in 0..8u8 {
        let rt = reg::COLOR_TARGET + 16 * (ct >> (4 + 3 * k as u32) & 7) as usize;
        if k as u32 >= (ct & 0xf).min(8) || regs[rt + 4] == 0 {
            continue;
        }
        put(Kind::TargetFormat, k, regs[rt + 4] as u64);
        let m = regs[0x680 + k as usize];
        let mask = m & 1 | (m >> 3) & 2 | (m >> 6) & 4 | (m >> 9) & 8;
        put(Kind::ChannelMask, k, mask as u64);
        let enabled = regs[0x4d8 + k as usize] & 1;
        put(Kind::BlendEnable, k, enabled as u64);
        if enabled == 0 {
            continue;
        }
        let r = if per_target {
            let b = 0x780 + 8 * k as usize;
            [
                regs[b],
                regs[b + 1],
                regs[b + 2],
                regs[b + 3],
                regs[b + 4],
                regs[b + 5],
                regs[b + 6],
            ]
        } else {
            [
                regs[0x4cf],
                regs[0x4d0],
                regs[0x4d1],
                regs[0x4d2],
                regs[0x4d3],
                regs[0x4d4],
                regs[0x4d6],
            ]
        };
        let (aop, asrc, adst) = if per_target || r[0] & 1 == 1 {
            (r[4], r[5], r[6])
        } else {
            (r[1], r[2], r[3])
        };
        put(Kind::BlendEquation, k, r[1] as u64);
        put(Kind::BlendSrc, k, r[2] as u64);
        put(Kind::BlendDst, k, r[3] as u64);
        if mask & 8 != 0 {
            put(Kind::BlendEquationAlpha, k, aop as u64);
            put(Kind::BlendSrcAlpha, k, asrc as u64);
            put(Kind::BlendDstAlpha, k, adst as u64);
        }
    }
    if regs[0x54e] & 1 == 1 {
        put(Kind::DepthFormat, 0, regs[0x3fa] as u64);
    }
    let depth = regs[0x4b3] & 1;
    put(Kind::DepthTest, 0, depth as u64);
    if depth == 1 {
        put(Kind::DepthWrite, 0, (regs[0x4ba] & 1) as u64);
        put(Kind::DepthFunc, 0, regs[0x4c3] as u64);
    }
    let stencil = regs[0x4e0] & 1;
    put(Kind::StencilTest, 0, stencil as u64);
    let front = [regs[0x4e1], regs[0x4e2], regs[0x4e3], regs[0x4e4]];
    let back = if regs[0x565] & 1 == 1 {
        [regs[0x566], regs[0x567], regs[0x568], regs[0x569]]
    } else {
        front
    };
    let two_sided = regs[0x565] & 1 == 1;
    for (f, values) in [
        [0x4e5, 0x4e6, 0x4e7],
        if two_sided {
            [0x3d5, 0x3d7, 0x3d6]
        } else {
            [0x4e5, 0x4e6, 0x4e7]
        },
    ]
    .into_iter()
    .enumerate()
    .filter(|_| stencil == 1)
    {
        put(Kind::StencilRef, f as u8, (regs[values[0]] & 0xff) as u64);
        put(
            Kind::StencilValueMask,
            f as u8,
            (regs[values[1]] & 0xff) as u64,
        );
        put(Kind::StencilMask, f as u8, (regs[values[2]] & 0xff) as u64);
    }
    put(Kind::DepthClamp, 0, regs[reg::VIEW_VOLUME_CLIP_CTRL] as u64);
    put(Kind::RasterizerDiscard, 0, regs[0xdf] as u64);
    put(Kind::PointSize, 0, regs[0x546] as u64);
    put(Kind::LineWidth, 0, regs[0x4ec] as u64);
    put(Kind::DepthBoundsEnable, 0, (regs[0x66f] & 1) as u64);
    if regs[0x66f] & 1 == 1 {
        put(Kind::DepthBoundsMin, 0, regs[0x3e7] as u64);
        put(Kind::DepthBoundsMax, 0, regs[0x3e8] as u64);
    }
    for (f, s) in [front, back]
        .into_iter()
        .enumerate()
        .filter(|_| stencil == 1)
    {
        put(Kind::StencilFunc, f as u8, s[3] as u64);
        put(Kind::StencilFail, f as u8, s[0] as u64);
        put(Kind::StencilDepthFail, f as u8, s[1] as u64);
        put(Kind::StencilPass, f as u8, s[2] as u64);
    }
    put(
        Kind::CullFace,
        0,
        if regs[0x646] & 1 == 1 {
            regs[0x648] as u64
        } else {
            0
        },
    );
    put(Kind::FrontFace, 0, regs[0x647] as u64);
    put(
        Kind::PolygonOffsetEnables,
        0,
        ((regs[0x370] & 1) | (regs[0x371] & 1) << 1 | (regs[0x372] & 1) << 2) as u64,
    );
    if regs[0x370..0x373].iter().any(|r| r & 1 == 1) {
        for (i, r) in [0x56f, 0x55b, 0x61f].into_iter().enumerate() {
            put(Kind::PolygonOffset, i as u8, regs[r] as u64);
        }
    }
    for i in 0..32u8 {
        let a = regs[reg::VERTEX_ATTRIBUTE + i as usize];
        if a >> 6 & 1 == 1 {
            continue;
        }
        put(Kind::AttribFormat, i, (a >> 21) as u64);
        put(Kind::AttribOffset, i, (a >> 7 & 0x3fff) as u64);
        put(Kind::AttribStream, i, (a & 0x1f) as u64);
    }
    for s in 0..16u8 {
        let base = reg::VERTEX_STREAM + 4 * s as usize;
        if regs[base] >> 12 & 1 == 0 {
            continue;
        }
        put(Kind::StreamStride, s, (regs[base] & 0xfff) as u64);
        let instanced = regs[0x620 + s as usize] & 1 == 1;
        put(
            Kind::StreamDivisor,
            s,
            if instanced { regs[base + 3] as u64 } else { 0 },
        );
    }
    for i in 0..4u8 {
        put(Kind::BlendColor, i, regs[0x4c7 + i as usize] as u64);
    }
    let f = |r: usize| f32::from_bits(regs[r]);
    let (sx, sy, ox, oy) = (
        f(reg::VIEWPORT_SCALE),
        f(reg::VIEWPORT_SCALE + 1),
        f(reg::VIEWPORT_SCALE + 3),
        f(reg::VIEWPORT_SCALE + 4),
    );
    for (i, v) in [ox - sx, oy - sy.abs(), 2.0 * sx, 2.0 * sy.abs()]
        .into_iter()
        .enumerate()
    {
        put(Kind::Viewport, i as u8, v.round() as i64 as u64);
    }
    let (h, v) = (regs[reg::SCISSOR + 1], regs[reg::SCISSOR + 2]);
    let scissor = if regs[reg::SCISSOR] & 1 == 1 {
        [
            h & 0xffff,
            v & 0xffff,
            (h >> 16).saturating_sub(h & 0xffff),
            (v >> 16).saturating_sub(v & 0xffff),
        ]
    } else {
        [0, 0, u32::MAX, u32::MAX]
    };
    for (i, x) in scissor.into_iter().enumerate() {
        put(Kind::Scissor, i as u8, x as u64);
    }
    out
}

#[derive(Default)]
struct Seen {
    counts: Vec<(u64, u64)>,
    usual: u64,
}

#[derive(Default)]
pub struct Learned {
    tables: HashMap<(Kind, u64), Seen>,
    pub generation: u64,
    pub conflicts: u64,
    pub checked: u64,
    pub examples: HashMap<(Kind, u64, u64), String>,
}

impl Learned {
    pub fn usual(&self, kind: Kind, value: u64) -> Option<u64> {
        self.tables.get(&(kind, value)).map(|s| s.usual)
    }

    pub fn add(
        &mut self,
        model: &[(Field, u64)],
        regs: &[(Field, u64)],
    ) -> Vec<(Field, u64, u64, u64)> {
        let regs: HashMap<Field, u64> = regs.iter().copied().collect();
        let mut bad = Vec::new();
        for &(f, v) in model {
            let Some(&r) = regs.get(&f) else { continue };
            self.checked += 1;
            let seen = self.tables.entry((f.0, v)).or_default();
            if !seen.counts.is_empty() && seen.usual != r {
                self.conflicts += 1;
                bad.push((f, v, seen.usual, r));
            }
            let n = match seen.counts.iter_mut().find(|c| c.0 == r) {
                Some(c) => {
                    c.1 += 1;
                    c.1
                }
                None => {
                    seen.counts.push((r, 1));
                    1
                }
            };
            let best = seen
                .counts
                .iter()
                .find(|c| c.0 == seen.usual)
                .map_or(0, |c| c.1);
            if (n > best || seen.counts.len() == 1) && seen.usual != r {
                seen.usual = r;
                self.generation += 1;
            } else if seen.counts.len() == 1 && n == 1 {
                self.generation += 1;
            }
        }
        bad
    }

    pub fn odd_examples(&self) -> String {
        let mut out = String::new();
        for ((k, v), seen) in &self.tables {
            for &(r, n) in seen.counts.iter().filter(|c| c.0 != seen.usual) {
                if let Some(e) = self.examples.get(&(*k, *v, r)) {
                    out += &format!("kirikae:           odd {k:?}: NVN {v:#x} gave {r:#x} {n} times (usually {:#x}): {e}
", seen.usual);
                }
            }
        }
        out
    }

    pub fn report(&self) -> String {
        let mut kinds: std::collections::BTreeMap<Kind, Vec<(u64, &Seen)>> =
            std::collections::BTreeMap::new();
        for (&(k, v), seen) in &self.tables {
            kinds.entry(k).or_default().push((v, seen));
        }
        let mut out = String::new();
        for (k, mut rows) in kinds {
            rows.sort_by_key(|r| r.0);
            let shown: Vec<String> = rows
                .iter()
                .take(12)
                .map(|(v, seen)| {
                    let mut rs = seen.counts.clone();
                    rs.sort_by_key(|r| std::cmp::Reverse(r.1));
                    let r: Vec<String> = rs.iter().map(|(r, n)| format!("{r:#x}x{n}")).collect();
                    format!("{v:#x}->{}", r.join("|"))
                })
                .collect();
            out += &format!(
                "kirikae:           {k:?}: {}{}
",
                shown.join(" "),
                if rows.len() > 12 {
                    format!(" (+{} values)", rows.len() - 12)
                } else {
                    String::new()
                }
            );
        }
        out
    }
}
