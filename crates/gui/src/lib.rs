use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

use eframe::egui;

pub type ConvertFn = fn(&Path, &Path, &mut dyn FnMut(u64, u64)) -> Result<PathBuf, String>;

#[derive(Clone, Copy)]
pub struct Converter {
    pub label: &'static str,
    pub extensions: &'static [&'static str],
    pub convert: ConvertFn,
}

#[derive(Default)]
pub struct Options {
    pub converters: Vec<Converter>,
    pub plugins: Vec<PathBuf>,
}

const INPUTS: &[&str] = &["dnsp", "nso", "nro"];

pub fn run(options: Options) -> Result<(), String> {
    let viewport = egui::ViewportBuilder::default()
        .with_title("Kirikae")
        .with_inner_size([780.0, 600.0])
        .with_min_inner_size([520.0, 420.0])
        .with_drag_and_drop(true);
    let native = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native("Kirikae", native, Box::new(|_| Ok(Box::new(App::new(options))))).map_err(|e| e.to_string())
}

#[derive(Clone, PartialEq)]
enum Status {
    Waiting,
    Running,
    Done(Option<PathBuf>),
    Failed(String),
}

struct Item {
    path: PathBuf,
    status: Status,
}

enum Msg {
    Log(String),
    Status(usize, Status),
    Finished,
}

struct App {
    options: Options,
    items: Vec<Item>,
    out_dir: String,
    fast_regs: bool,
    log: Vec<String>,
    rx: Option<Receiver<Msg>>,
    cargo: Option<String>,
}

fn default_out_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(|h| PathBuf::from(h).join("Documents").join("Kirikae"))
        .unwrap_or_else(|| PathBuf::from("Kirikae"))
}

fn cargo_version() -> Option<String> {
    let out = Command::new("cargo").arg("--version").stdin(Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn cli() -> PathBuf {
    let name = format!("kirikae{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join(&name)))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(name))
}

fn extension(p: &Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

impl App {
    fn new(options: Options) -> App {
        App {
            options,
            items: Vec::new(),
            out_dir: default_out_dir().display().to_string(),
            fast_regs: false,
            log: Vec::new(),
            rx: None,
            cargo: cargo_version(),
        }
    }

    fn busy(&self) -> bool {
        self.rx.is_some()
    }

    fn accepts(&self, p: &Path) -> bool {
        if p.is_dir() {
            return p.join("main").is_file() || p.join("exefs").join("main").is_file();
        }
        let ext = extension(p);
        INPUTS.contains(&ext.as_str()) || self.options.converters.iter().any(|c| c.extensions.contains(&ext.as_str()))
    }

    fn add(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for p in paths {
            if !self.accepts(&p) {
                self.log.push(format!("not a supported game file: {}", p.display()));
                continue;
            }
            if !self.items.iter().any(|i| i.path == p) {
                self.items.push(Item { path: p, status: Status::Waiting });
            }
        }
    }

    fn extensions(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = INPUTS.to_vec();
        for c in &self.options.converters {
            v.extend_from_slice(c.extensions);
        }
        v
    }

    fn start(&mut self) {
        let (tx, rx) = channel();
        self.rx = Some(rx);
        for i in &mut self.items {
            if !matches!(i.status, Status::Done(_)) {
                i.status = Status::Waiting;
            }
        }
        let jobs: Vec<(usize, PathBuf)> =
            self.items.iter().enumerate().filter(|(_, i)| i.status == Status::Waiting).map(|(k, i)| (k, i.path.clone())).collect();
        let out_dir = PathBuf::from(self.out_dir.trim());
        let converters = self.options.converters.clone();
        let plugins = self.options.plugins.clone();
        let fast_regs = self.fast_regs;
        std::thread::spawn(move || {
            for (k, path) in jobs {
                let _ = tx.send(Msg::Status(k, Status::Running));
                let status = match build_one(&path, &out_dir, &converters, &plugins, fast_regs, &tx) {
                    Ok(exe) => Status::Done(exe),
                    Err(e) => {
                        let _ = tx.send(Msg::Log(format!("error: {e}")));
                        Status::Failed(e)
                    }
                };
                let _ = tx.send(Msg::Status(k, status));
            }
            let _ = tx.send(Msg::Finished);
        });
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut finished = false;
        while let Ok(m) = rx.try_recv() {
            match m {
                Msg::Log(line) => self.log.push(line),
                Msg::Status(k, s) => {
                    if let Some(i) = self.items.get_mut(k) {
                        i.status = s;
                    }
                }
                Msg::Finished => finished = true,
            }
        }
        if self.log.len() > 4000 {
            self.log.drain(..self.log.len() - 4000);
        }
        if finished {
            self.rx = None;
        }
    }
}

fn build_one(path: &Path, out_dir: &Path, converters: &[Converter], plugins: &[PathBuf], fast_regs: bool, tx: &Sender<Msg>) -> Result<Option<PathBuf>, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "game".into());
    let mut input = path.to_path_buf();
    let ext = extension(path);
    if let Some(c) = converters.iter().find(|c| c.extensions.contains(&ext.as_str())) {
        let _ = tx.send(Msg::Log(format!("{}: preparing {}", c.label, path.display())));
        let mut last = u64::MAX;
        let mut progress = |done: u64, total: u64| {
            let pct = done * 100 / total.max(1);
            if pct / 10 != last / 10 || last == u64::MAX {
                last = pct;
                let _ = tx.send(Msg::Log(format!("{}: {pct}%", c.label)));
            }
        };
        input = (c.convert)(path, out_dir, &mut progress)?;
    }
    let mut cmd = Command::new(cli());
    cmd.arg("build").arg(&input).arg("-o").arg(out_dir).arg("--name").arg(&name);
    if fast_regs {
        cmd.arg("--fast-regs");
    }
    for p in plugins {
        cmd.arg("--plugin").arg(p);
    }
    let _ = tx.send(Msg::Log(format!("building {name}")));
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| format!("could not start {}: {e}", cli().display()))?;
    let forward = |r: Box<dyn Read + Send>, tx: Sender<Msg>| {
        std::thread::spawn(move || {
            let mut last = None;
            for line in BufReader::new(r).lines().map_while(Result::ok) {
                let _ = tx.send(Msg::Log(line.clone()));
                last = Some(line);
            }
            last
        })
    };
    let out = forward(Box::new(child.stdout.take().unwrap()), tx.clone());
    let err = forward(Box::new(child.stderr.take().unwrap()), tx.clone());
    let status = child.wait().map_err(|e| e.to_string())?;
    let summary = out.join().ok().flatten();
    let _ = err.join();
    if !status.success() {
        return Err(format!("the build of {name} failed ({status})"));
    }
    let exe = summary.and_then(|l| {
        let (_, rest) = l.split_once(" into ")?;
        let (p, _) = rest.rsplit_once(" in ")?;
        Some(PathBuf::from(p))
    });
    Ok(exe)
}

fn open(path: &Path) {
    let _ = if cfg!(windows) {
        Command::new("explorer").arg(path).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(path).spawn()
    } else {
        Command::new("xdg-open").arg(path).spawn()
    };
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if self.busy() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() && !self.busy() {
            self.add(dropped);
        }

        egui::TopBottomPanel::top("title").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.heading("Kirikae");
            ui.label("Recompile Switch executables into native programs.");
            ui.add_space(4.0);
        });
        egui::TopBottomPanel::bottom("legal").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.small("Only use games dumped from your own console. Recompiled programs contain the game's code: do not share them.");
            ui.add_space(2.0);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.cargo.is_none() {
                ui.colored_label(egui::Color32::from_rgb(220, 120, 40), "Rust (cargo) was not found: install it from https://rustup.rs to build games.");
                ui.add_space(6.0);
            }
            let busy = self.busy();
            ui.strong("Game files");
            ui.horizontal(|ui| {
                if ui.add_enabled(!busy, egui::Button::new("Add files...")).clicked() {
                    let ext = self.extensions();
                    if let Some(files) = rfd::FileDialog::new().add_filter("Game files", &ext).pick_files() {
                        self.add(files);
                    }
                }
                if ui.add_enabled(!busy, egui::Button::new("Add game folder...")).clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    self.add([dir]);
                }
                if ui.add_enabled(!busy && !self.items.is_empty(), egui::Button::new("Clear")).clicked() {
                    self.items.clear();
                }
            });
            let accepted = self.extensions().iter().map(|e| format!(".{e}")).collect::<Vec<_>>().join(", ");
            ui.weak(format!("Drop files here: {accepted}, or a game folder (exefs/ and romfs/)."));
            let mut remove = None;
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_min_height(60.0);
                ui.set_width(ui.available_width());
                if self.items.is_empty() {
                    ui.weak("No files yet.");
                }
                for (k, item) in self.items.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let (text, color) = match &item.status {
                            Status::Waiting => ("waiting", ui.visuals().weak_text_color()),
                            Status::Running => ("building...", egui::Color32::from_rgb(80, 150, 230)),
                            Status::Done(_) => ("done", egui::Color32::from_rgb(70, 170, 90)),
                            Status::Failed(_) => ("failed", egui::Color32::from_rgb(210, 70, 60)),
                        };
                        ui.colored_label(color, text);
                        ui.label(item.path.display().to_string());
                        if let Status::Done(Some(exe)) = &item.status
                            && ui.small_button("Show").clicked()
                        {
                            open(exe.parent().unwrap_or(exe));
                        }
                        if !busy && ui.small_button("x").clicked() {
                            remove = Some(k);
                        }
                    });
                }
            });
            if let Some(k) = remove {
                self.items.remove(k);
            }
            ui.add_space(8.0);
            ui.strong("Output folder");
            ui.horizontal(|ui| {
                ui.add_enabled(!busy, egui::TextEdit::singleline(&mut self.out_dir).desired_width(ui.available_width() - 90.0));
                if ui.add_enabled(!busy, egui::Button::new("Choose...")).clicked()
                    && let Some(dir) = rfd::FileDialog::new().set_directory(&self.out_dir).pick_folder()
                {
                    self.out_dir = dir.display().to_string();
                }
            });
            ui.add_enabled(!busy, egui::Checkbox::new(&mut self.fast_regs, "Keep guest registers in variables (experimental)"));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let ready = !busy && self.cargo.is_some() && self.items.iter().any(|i| !matches!(i.status, Status::Done(_))) && !self.out_dir.trim().is_empty();
                if ui.add_enabled(ready, egui::Button::new(egui::RichText::new("Recompile").strong()).min_size(egui::vec2(120.0, 28.0))).clicked() {
                    self.start();
                }
                if busy {
                    ui.spinner();
                }
                if ui.button("Open output folder").clicked() {
                    let _ = std::fs::create_dir_all(self.out_dir.trim());
                    open(Path::new(self.out_dir.trim()));
                }
            });
            ui.add_space(8.0);
            ui.strong("Log");
            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    for line in &self.log {
                        ui.label(egui::RichText::new(line).monospace().size(11.5));
                    }
                });
            });
        });
    }
}
