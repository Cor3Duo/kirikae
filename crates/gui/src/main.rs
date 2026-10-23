#![windows_subsystem = "windows"]

fn main() -> Result<(), String> {
    kirikae_gui::run(kirikae_gui::Options::default())
}
