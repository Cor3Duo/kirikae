import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]["package"]["version"]
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    subprocess.run(["cargo", "build", "--release", "-p", "kirikae", "-p", "kirikae-gui"], cwd=ROOT, check=True)
    name = f"kirikae-{version}-windows-x64"
    out = ROOT / "dist" / name
    if out.exists():
        shutil.rmtree(out)
    runtime = out / "runtime"
    runtime.mkdir(parents=True)
    for exe in ["kirikae.exe", "kirikae-gui.exe"]:
        shutil.copy2(target / "release" / exe, out / exe)
    for f in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(ROOT / f, runtime / f)
    ignore = shutil.ignore_patterns("target", "*-build")
    shutil.copytree(ROOT / "crates", runtime / "crates", ignore=ignore)
    for f in ["README.md", "LEGAL.md", "LICENSE-MIT", "LICENSE-APACHE"]:
        shutil.copy2(ROOT / f, out / f)
    archive = shutil.make_archive(str(ROOT / "dist" / name), "zip", root_dir=out.parent, base_dir=name)
    print(archive)

if __name__ == "__main__":
    sys.exit(main())
