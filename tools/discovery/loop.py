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
      "name": "title-010015100b514000",
      "source": "path/to/decrypted/dump",
      "out": "out/title-010015100b514000",
      "build_args": ["--profile"],
      "scenarios": [
        {"name": "boot", "seconds": 60, "input": null},
        {
          "name": "menu", "seconds": 120, "input": "1:PLUS,5:A",
          "capture_every": 20, "capture_fps": 30
        }
      ]
    }
  ]
}

A scenario with `capture_every` writes every Nth presented frame with the runtime's
`KIRIKAE_FRAME_DUMP` (N = `capture_every` seconds × `capture_fps`, default 30 fps),
classifies each PNG with `tools/frame_diff.py` and writes `captures.json` next to the
run. `capture_every_frames` does the same with a frame interval instead of a wall-clock
one, for titles whose frame rate collapses under gameplay; those captures carry the
frame number and no second label. `capture_frame` additionally asks `KIRIKAE_CAPTURE`
for one PPM at that frame. Neither changes how the game runs; both are evidence of what
reached the screen.

`runner` is an optional command prefix for the executable, as in `batch.py`; the tests
use it to run a Python script that ends in `.exe` without a real build.

Usage:
    python tools/discovery/loop.py --config discovery.json
    python tools/discovery/loop.py --config discovery.json --game title-010015100b514000 --scenario boot
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
    if scenario.get("capture_every") or scenario.get("capture_every_frames"):
        if scenario.get("capture_every_frames"):
            schedule = capture_frames_schedule(
                float(scenario["seconds"]),
                int(scenario.get("capture_fps", 30)),
                int(scenario["capture_every_frames"]),
                scenario.get("capture_count"),
            )
        else:
            schedule = capture_schedule(
                float(scenario["seconds"]),
                float(scenario["capture_every"]),
                int(scenario.get("capture_fps", 30)),
                scenario.get("capture_count"),
            )
        env["KIRIKAE_FRAME_DUMP"] = str(run_dir / "frames")
        env["KIRIKAE_FRAME_DUMP_EVERY"] = str(schedule["every_frames"])
        env["KIRIKAE_FRAME_DUMP_FROM"] = str(schedule["from_frame"])
        env["KIRIKAE_FRAME_DUMP_COUNT"] = str(schedule["count"])
    if scenario.get("capture_frame"):
        env["KIRIKAE_CAPTURE"] = str(run_dir / "capture")
        env["KIRIKAE_CAPTURE_FRAME"] = str(int(scenario["capture_frame"]))
    env.update({k: str(v) for k, v in scenario.get("env", {}).items()})
    return env


def capture_schedule(seconds: float, every: float, fps: int, count: int | None = None) -> dict:
    """Frames the runtime should dump so a run leaves one capture every `every` seconds.

    `display::dumps` keeps a frame when `(frame - from) % every_frames == 0` and fewer
    than `count` such frames were kept. Starting at `from = every_frames` skips the
    first frame (which is normally a black clear) and the run bound decides how many
    of the scheduled frames are ever reached.
    """
    if every <= 0:
        raise ValueError("capture_every must be positive")
    if seconds <= 0:
        raise ValueError("scenario seconds must be positive")
    fps = max(1, int(fps))
    every_frames = max(1, round(every * fps))
    scheduled = max(1, int(seconds // every))
    if count is not None:
        scheduled = max(1, int(count))
    return {"every_frames": every_frames, "from_frame": every_frames, "count": scheduled}


def capture_frames_schedule(
    seconds: float, fps: int, every_frames: int, count: int | None = None
) -> dict:
    """Frame-indexed schedule for titles whose frame rate is not known in advance.

    A game that falls to 0.5 fps never reaches a wall-clock schedule, so the frames
    themselves are the clock: every `every_frames` presented frames, starting at the
    first multiple, up to the approximate frame budget of the run.
    """
    if seconds <= 0:
        raise ValueError("scenario seconds must be positive")
    step = max(1, int(every_frames))
    scheduled = max(1, int(seconds * max(1, int(fps)) // step))
    if count is not None:
        scheduled = max(1, int(count))
    return {"every_frames": step, "from_frame": step, "count": scheduled}


def frame_diff_module():
    """`tools/frame_diff.py` as a module; it is the single frame classifier."""
    tools = Path(__file__).resolve().parents[1]
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import frame_diff

    return frame_diff


def classify_frames(frames_dir: Path, fps: float) -> list[dict]:
    """Measure every dump in `frames_dir`; empty when the run wrote none."""
    if not frames_dir.is_dir():
        return []
    try:
        module = frame_diff_module()
    except ImportError as e:  # noqa: BLE001 - evidence missing, not a crash
        return [{"error": str(e)}]
    out = []
    for path in sorted(frames_dir.glob("frame*.png")):
        try:
            m = path.stem.removeprefix("frame").lstrip("0") or "0"
            frame = int(m)
        except ValueError:
            continue
        try:
            stats = module.analysis(module.load(path))
        except Exception as e:  # noqa: BLE001 - a bad frame is a result, not a crash
            out.append({"file": path.name, "error": str(e)})
            continue
        out.append(
            {
                "file": path.name,
                "frame": frame,
                "seconds": round(frame / fps, 1) if fps > 0 else None,
                "mean": [round(v, 1) for v in stats["mean"]],
                "luma": round(stats["luma"], 2),
                "nonblack": round(stats["nonblack"], 2),
                "distinct": stats["distinct"],
                "top_color": f"#{stats['top_color']:06x}",
                "top_share": round(stats["top_share"], 1),
                "black": stats["nonblack"] < 0.1,
            }
        )
    return out


def write_captures(sdir: Path, captures: list[dict], fps: float) -> None:
    doc = {"schema": 1, "fps": fps, "frames": captures}
    (sdir / "captures.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


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
            frames_dir = sdir / "frames"
            if frames_dir.is_dir():
                shutil.rmtree(frames_dir)
            if scenario.get("capture_every") or scenario.get("capture_every_frames"):
                frames_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{game['name']}/{scenario['name']}] run {scenario['seconds']}s …", flush=True)
            env = scenario_env(game, scenario, sdir)
            cmd = [*(str(x) for x in cfg.get("runner", [])), str(exe)]
            r = run(cmd, exe.parent, scenario["seconds"], env=env, log=sdir / "stderr.log")
            r.pop("stdout", None)
            r.pop("stderr", None)
            if scenario.get("capture_every") or scenario.get("capture_every_frames"):
                fps = 0 if scenario.get("capture_every_frames") else int(scenario.get("capture_fps", 30))
                captures = classify_frames(frames_dir, fps)
                write_captures(sdir, captures, fps)
                r["captures"] = captures
                print(
                    f"[{game['name']}/{scenario['name']}] {len(captures)} capture(s), "
                    f"{sum(1 for c in captures if c.get('black'))} black",
                    flush=True,
                )
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
