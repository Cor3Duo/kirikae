pub mod cfg;
pub mod decode;
pub mod opcodes;
pub mod sph;
pub mod translate;

pub use translate::{
    Env, ImageUse, Kind, NoEnv, Options, Shader, Stage, TexDim, TextureUse, translate,
};

pub fn mnemonics() -> Vec<&'static str> {
    let mut names: Vec<&str> = opcodes::OPCODES.iter().map(|o| o.name).collect();
    names.sort_unstable();
    names.dedup();
    names
}
