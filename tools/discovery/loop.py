#!/usr/bin/env python3
"""Mechanical discovery cycle: build, run scripted scenarios, collect, triage.

This is the deterministic half of the discovery loop. It never invents changes and
never needs an AI: run it in CI, on a schedule, or before handing artifacts to the
analysis layer.

Config (JSON):

{
  "kirikae": "target/release/kirikae.exe",
  "games": [
    {
      "name": "smbw",
      "source": "myswitch/extracted",
      "out": "myswitch/build",
      "build_args": ["--profile"],
      "scenarios": [
        {"name": "boot", "seconds": 60, "input": null},
        {"name": "menu", "seconds": 120, "input": "1:PLUS,5:Z", "env": {"KIRIKAE_MUTE": "1"}}
      ]
    }
  ]
}

Usage:
    python tools/discovery/loop.py --config discovery.json
    python tools/discovery/loop.py --config discovery.json --game smbw --scenario boot
    python tools/discovery/loop.py --config discovery.json --skip-build --skip-triage

Everything written by a cycle lives outside the repository, under
$KIRIKAE_DISCOVERY_DIR (default %LOCALAPPDATA%/kirikae-discovery).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def discovery_root() -> Path:
    env = os.environ.get("KIRIKAE_DISCOVERY_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share"
    return Path(base) / "kirikae-discovery"


def kirikae_bin(cfg: dict) -> Path:
    env = os.environ.get("KIRIKAE_BIN")
    candidate = Path(env) if env else ROOT / cfg.get("kirikae", "target/release/kirikae.exe")
    if not candidate.is_file():
        raise SystemExit(f"kirikae binary not found: {candidate} (set KIRIKAE_BIN)")
    return candidate


def run(cmd: list[str], cwd: Path, timeout: float | None, env: dict | None = None, log: Path | None = None) -> dict:
    started = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            errors="replace",
        )
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code = "timeout"
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    if log is not None:
        log.write_text(f"$ {' '.join(cmd)}\n\n{out}\n{err}", encoding="utf-8", errors="replace")
    return {"seconds": round(time.time() - started, 2), "exit": code, "stdout": out, "stderr": err}


def out_stem(game: dict) -> str:
    """kirikae names the executable and profile after the source, not the output folder."""
    src = Path(game["source"])
    return src.stem if src.suffix else src.name


def build_game(game: dict, kirikae: Path, run_dir: Path) -> dict:
    out = Path(game["out"])
    args = [str(kirikae), "build", str(Path(game["source"])), "-o", str(out)]
    profile = out / f"{out_stem(game)}.profile"
    for a in game.get("build_args", []):
        args.append(str(profile) if a == "--profile" else a)
    return run(args, ROOT, game.get("build_timeout", 1800), log=run_dir / "build.log")


def scenario_env(game: dict, scenario: dict, run_dir: Path) -> dict:
    env = {
        "KIRIKAE_HEADLESS": "1",
        "KIRIKAE_COVERAGE": str(run_dir / "coverage.json"),
        "KIRIKAE_PROFILE": str(run_dir / "game.profile"),
        "KIRIKAE_STATS": "1",
        "KIRIKAE_SAVES": str(run_dir / "saves"),
        "KIRIKAE_MUTE": "1",
    }
    if scenario.get("input"):
        env["KIRIKAE_INPUT"] = scenario["input"]
    env.update({k: str(v) for k, v in scenario.get("env", {}).items()})
    return env


def cycle(cfg: dict, only_game: str | None, only_scenario: str | None, skip_build: bool, do_triage: bool) -> Path:
    kirikae = kirikae_bin(cfg)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    root = discovery_root()
    manifest = {"schema": 1, "run_id": run_id, "started": int(time.time()), "games": {}}

    for game in cfg["games"]:
        if only_game and game["name"] != only_game:
            continue
        run_dir = root / game["name"] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        g = {"build": None, "scenarios": {}}
        manifest["games"][game["name"]] = g

        if not skip_build:
            print(f"[{game['name']}] build …", flush=True)
            g["build"] = {k: v for k, v in build_game(game, kirikae, run_dir).items() if k in ("seconds", "exit")}
            print(f"[{game['name']}] build exit={g['build']['exit']} in {g['build']['seconds']}s", flush=True)
            if g["build"]["exit"] != 0:
                continue

        exe = Path(game["out"]) / f"{out_stem(game)}.exe"
        for scenario in game.get("scenarios", [{"name": "boot", "seconds": 60}]):
            if only_scenario and scenario["name"] != only_scenario:
                continue
            sdir = run_dir / "run" / scenario["name"]
            sdir.mkdir(parents=True, exist_ok=True)
            print(f"[{game['name']}/{scenario['name']}] run {scenario['seconds']}s …", flush=True)
            env = scenario_env(game, scenario, sdir)
            r = run([str(exe)], exe.parent, scenario["seconds"], env=env, log=sdir / "stderr.log")
            r.pop("stdout", None)
            r.pop("stderr", None)
            g["scenarios"][scenario["name"]] = r
            print(f"[{game['name']}/{scenario['name']}] exit={r['exit']} in {r['seconds']}s", flush=True)

        if do_triage:
            coverage = [str(p) for p in run_dir.glob("run/*/coverage.json")]
            logs = [str(p) for p in run_dir.glob("run/*/stderr.log")]
            cmd = [sys.executable, str(ROOT / "tools/discovery/triage.py"), "--game", game["name"], "--run-id", run_id]
            for c in coverage:
                cmd += ["--coverage", c]
            for l in logs:
                cmd += ["--log", l]
            print(f"[{game['name']}] triage …", flush=True)
            result = run(cmd, ROOT, 600)
            print(result.get("stdout", ""), end="")
            if result.get("stderr"):
                print(result["stderr"], file=sys.stderr, end="")

    manifest["finished"] = int(time.time())
    (root / "last-run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {root / 'last-run.json'}")
    return root / "last-run.json"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--game")
    ap.add_argument("--scenario")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-triage", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cycle(cfg, args.game, args.scenario, args.skip_build, not args.skip_triage)


if __name__ == "__main__":
    main(sys.argv[1:])
