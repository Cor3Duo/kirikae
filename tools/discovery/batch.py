#!/usr/bin/env python3
"""Run many games through kirikae and keep a compatibility matrix.

This is the scalable half of the discovery loop: a queue over a folder of already
extracted games (or, later, dumps), a compile pass, a bounded boot pass, and a
JSON/Markdown compatibility matrix. Everything stays outside the repository.

Config (JSON):
{
  "kirikae": "target/release/kirikae.exe",
  "games_root": "D:/games/extracted",
  "work": "%LOCALAPPDATA%/kirikae-discovery/batch",
  "boot_seconds": 20,
  "build_timeout": 1200,
  "nvn": "trace",
  "runner": ["C:/Python/python.exe"]
}

`runner` is an optional command prefix for the game executable. Real builds need
none; the tests use it to run a Python script from a file with an `.exe` suffix.

`--profile-loop` hands the build to `kirikae build --profile-loop`, which alternates
build, bounded headless run and rebuild until the interpreted count stops falling. The
flag is opt-in because it starts the game; the per-run bound (`--profile-seconds`,
default `boot_seconds` clamped to at least 10) and the total bound
(`profile_loop_timeout`, default one build timeout plus its bound per run) keep it
bounded. The per-iteration numbers are recorded as `profile_loop` in `state.jsonl` and
`compat.json`, and summarized in the matrix's Profile column.

`--capture` answers the other half of "does it run": did anything reach the screen? The
boot sets `KIRIKAE_CAPTURE=<work>/<slug>/frame` and `KIRIKAE_CAPTURE_FRAME` (default
120), then classifies the written PPM with `tools/frame_diff.py` (mean RGB, percentage
of non-black pixels, distinct colours, dominant colour). The record carries `capture`,
`frames_10s` and `fps_10s`, the matrix grows Frame/Visibility columns, and a frame with
under 0.1 % non-black pixels is `black`, anything else `non-black`. A game that never
reaches the requested frame keeps its boot status (`booted`/`timeout`) and records that
no frame was written; the capture never changes how the runtime presents. `--recapture`
is the companion of `--capture` for a second look: it ignores the recorded status and
reuses the executable already in `<work>/<slug>/out`, so another frame can be measured
without a rebuild (it never compiles anything).

Statuses:
  compiled       built, boot not requested
  booted         ran and produced frames (a headless boot normally hits the bound)
  non-black      a captured frame has visible content (the screen is not blank)
  black          a captured frame is black: it booted but nothing reached the screen
  timeout        the bound expired before the first frame
  crash          the process exited before the first frame, or failed to start
  aarch32        the title is ARM32, which the recompiler does not handle yet
  compile_error  the build failed, or no executable was available

Usage:
    python tools/discovery/batch.py --config batch.json
    python tools/discovery/batch.py --config batch.json --limit 10
    python tools/discovery/batch.py --config batch.json --only <dump>,<dump>
    python tools/discovery/batch.py --config batch.json --resume
    python tools/discovery/batch.py --config batch.json --skip-build --boot-seconds 20 --nvn trace
    python tools/discovery/batch.py --config batch.json --compile-only
    python tools/discovery/batch.py --config batch.json --profile-loop --profile-seconds 30
    python tools/discovery/batch.py --config batch.json --capture --boot-seconds 40
    python tools/discovery/batch.py --config batch.json --capture --capture-frame 30 --recapture
    python tools/discovery/batch.py --config batch.json --json

`--json` writes `<work>/compat.summary.json` (schema 1) next to the matrix:

    {"schema": 1, "generated": 172..., "games": [
      {"slug", "title_id", "status", "functions", "instructions",
       "frames", "fps", "gaps", "top_gaps", "note", "capture"}
    ]}

`capture` is present only for a game measured with `--capture`.

The full per-game record (exit code, draws, clears, errors, timings, `profile_loop`) stays
in `compat.json` and `state.jsonl`. `top_gaps` is a list of `"<interface> <id>"`
strings ordered by observation count.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = 1
CAPTURE_DEFAULT_FRAME = 120
CAPTURE_BLACK_BELOW = 0.1

STATS_FRAMES = re.compile(r"\[\s*\d+s\]\s+(\d+)\s+frames\s*\(\s*([\d.]+)\s+fps")
STATS_GPU = re.compile(
    r"GPU:\s*(\d+)\s+submits,\s*(\d+)\s+draws,\s*(\d+)\s+clears,\s*(\d+)\s+copies,\s*(\d+)\s+dispatches"
)
STATS_CALLS = re.compile(r"(\d+)\s+native calls,\s*(\d+)\s+interpreted")
PROFILE_LOOP = re.compile(
    r"profile loop (\d+)/(\d+): compiled (\d+) addresses, new (\d+), "
    r"interpreted (\d+|-), build ([\d.]+)s, first frame ([\d.]+s|-)"
)
PROFILE_LOOP_DONE = re.compile(
    r"profile loop stopped after (\d+) of (\d+) run\(s\) \((.*)\): "
    r"interpreted (\d+|-) -> (\d+|-), addresses (\d+), first frame (\S+) -> (\S+)"
)
ERROR_WORDS = re.compile(r"error|failed|panic|abort|svcBreak|unimplemented|missing|deadlock", re.I)
STATS_NOISE = ("kirikae: [", "kirikae:         ")
STATS_WINDOW = re.compile(r"\[\s*(\d+)s\]")


def work_root(cfg: dict) -> Path:
    raw = cfg.get("work") or os.path.join(os.environ.get("LOCALAPPDATA", str(Path.home())), "kirikae-discovery", "batch")
    return Path(os.path.expandvars(raw))


def kirikae_bin(cfg: dict) -> Path:
    env = os.environ.get("KIRIKAE_BIN")
    path = Path(env) if env else ROOT / cfg.get("kirikae", "target/release/kirikae.exe")
    if not path.is_file():
        raise SystemExit(f"kirikae binary not found: {path} (set KIRIKAE_BIN)")
    return path


def is_game_folder(path: Path) -> bool:
    return (path / "exefs" / "main").is_file() or (path / "main").is_file()


def discover(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"games_root is not a folder: {root}")
    skip = {"exefs", "romfs", "control", "manual"}
    out = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in skip and is_game_folder(p)]
    if is_game_folder(root):
        out.insert(0, root)
    return out


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "game"


def load_state(path: Path) -> dict[str, dict]:
    state = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                prev = state.get(rec["game"])
                if prev is not None:
                    for key in ("functions", "instructions", "reachable_total", "build_seconds", "title_id"):
                        if key not in rec and key in prev:
                            rec[key] = prev[key]
                state[rec["game"]] = rec
    return state


def append_state(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_status(status: str | None) -> str:
    return "booted" if status == "boots" else (status or "pending")


def parse_build(text: str) -> dict:
    m = re.search(r"recompiled (\d+) functions \((\d+) of (\d+) instructions", text)
    if not m:
        return {}
    return {"functions": int(m.group(1)), "instructions": int(m.group(2)), "reachable_total": int(m.group(3))}


def parse_stats(text: str) -> dict:
    """Last periodic stats line: frames, fps, draws, clears and interpreter count."""
    for line in reversed(text.splitlines()):
        if rec := stats_from_line(line):
            return rec
    return {}


def stats_from_line(line: str) -> dict:
    m = STATS_FRAMES.search(line)
    if not m:
        return {}
    rec = {"frames": int(m.group(1)), "fps": float(m.group(2))}
    if g := STATS_GPU.search(line):
        rec.update(
            {
                "submits": int(g.group(1)),
                "draws": int(g.group(2)),
                "clears": int(g.group(3)),
                "copies": int(g.group(4)),
                "dispatches": int(g.group(5)),
            }
        )
    if c := STATS_CALLS.search(line):
        rec["native_calls"] = int(c.group(1))
        rec["interpreted"] = int(c.group(2))
    return rec


def parse_stats_at(text: str, seconds: int = 10) -> dict:
    """The stats line of one window; the visibility matrix compares frames@10s."""
    for line in text.splitlines():
        m = STATS_WINDOW.search(line)
        if m and int(m.group(1)) == seconds:
            if rec := stats_from_line(line):
                return rec
    return {}


def parse_frames(text: str) -> dict:
    frames = parse_stats(text)
    return {k: frames[k] for k in ("frames", "fps") if k in frames}


def _count(raw: str) -> int | None:
    return None if raw == "-" else int(raw)


def _seconds(raw: str) -> float | None:
    return None if raw == "-" else float(raw.removesuffix("s"))


def parse_profile_loop(text: str) -> dict:
    """What `kirikae build --profile-loop` reported, or `{}` when it did not run.

    The per-iteration lines carry the numbers of that iteration; the closing line carries the
    final profile size and the first-to-last comparison, so both are read when present.
    """
    iterations = []
    for line in text.splitlines():
        m = PROFILE_LOOP.search(line)
        if not m:
            continue
        iterations.append(
            {
                "iteration": int(m.group(1)),
                "of": int(m.group(2)),
                "compiled": int(m.group(3)),
                "new": int(m.group(4)),
                "interpreted": _count(m.group(5)),
                "build_seconds": float(m.group(6)),
                "first_frame": _seconds(m.group(7)),
            }
        )
    if not iterations:
        return {}
    done = None
    for line in reversed(text.splitlines()):
        done = PROFILE_LOOP_DONE.search(line)
        if done:
            break
    rec = {
        "runs": len(iterations),
        "new": sum(it["new"] for it in iterations),
        "iterations": iterations,
    }
    first, last = iterations[0], iterations[-1]
    rec.update(
        {
            "interpreted": first["interpreted"],
            "interpreted_after": last["interpreted"],
            "first_frame": first["first_frame"],
            "first_frame_after": last["first_frame"],
        }
    )
    if done:
        rec["addresses"] = int(done.group(6))
        rec["stop"] = done.group(3)
        rec["runs"] = int(done.group(1))
    else:
        rec["addresses"] = last["compiled"] + last["new"]
        rec["stop"] = ""
    return rec


def profile_loop_text(rec: dict) -> str:
    """One matrix cell for the loop, or an empty one when it did not run."""
    if not rec:
        return ""
    interpreted = rec.get("interpreted")
    after = rec.get("interpreted_after")
    first = rec.get("first_frame")
    last = rec.get("first_frame_after")
    parts = [f"{rec.get('runs', '?')}it"]
    if interpreted is not None or after is not None:
        before = "-" if interpreted is None else interpreted
        parts.append(f"{before}→{'-' if after is None else after} interp")
    if first is not None or last is not None:
        parts.append(f"{'-' if first is None else f'{first:g}'}→{'-' if last is None else f'{last:g}'}s")
    return ", ".join(parts)


def coverage_gaps(data: dict) -> tuple[int | None, list[str]]:
    items = data.get("items")
    if not isinstance(items, list):
        return None, []
    gaps = [it for it in items if it.get("status") != "implemented"]
    gaps.sort(key=lambda it: (-int(it.get("count", 1) or 0), str(it.get("interface")), str(it.get("id"))))
    top = [f"{it.get('interface', '?')} {it.get('id')}" for it in gaps[:5]]
    return len(gaps), top


def frames_from_coverage(data: dict) -> dict:
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        return {}
    last = runs[-1]
    frames = last.get("frames")
    if not isinstance(frames, int):
        return {}
    out = {"frames": frames}
    seconds = last.get("seconds")
    if isinstance(seconds, (int, float)) and seconds > 0:
        out["fps"] = round(frames / float(seconds), 1)
    return out


def last_error_lines(text: str, limit: int = 3) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if ERROR_WORDS.search(line)]
    if not errors:
        errors = [line for line in lines if not line.startswith(STATS_NOISE)]
    return [line[:300] for line in errors[-limit:]]


def error_hint(errors: list[str]) -> str | None:
    for line in reversed(errors):
        if ERROR_WORDS.search(line):
            return line
    return None


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and its children; Windows must not leave an orphan."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run(cmd: list[str], cwd: Path, timeout: float, env: dict | None = None) -> dict:
    """Run with a hard wall-clock bound; on timeout the whole process tree dies."""
    started = time.time()
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except OSError as e:
        return {"exit": "error", "stdout": "", "stderr": str(e), "seconds": round(time.time() - started, 2)}

    try:
        out, err = proc.communicate(timeout=timeout)
        return {"exit": proc.returncode, "stdout": out or "", "stderr": err or "", "seconds": round(time.time() - started, 2)}
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return {"exit": "timeout", "stdout": out or "", "stderr": err or "", "seconds": round(time.time() - started, 2)}


def out_stem(game: Path) -> str:
    return game.name


def is_complete(rec: dict, boot: bool, resume: bool) -> bool:
    status = normalize_status(rec.get("status"))
    done = {"booted", "black", "non-black"} if boot else {"compiled", "booted", "black", "non-black"}
    if resume:
        done |= {"aarch32", "compile_error", "crash", "timeout"}
    return status in done


def can_reuse(prior: dict, exe: Path, any_status: bool) -> bool:
    """A build on disk is reusable when the previous run compiled it."""
    if not exe.is_file():
        return False
    status = normalize_status(prior.get("status"))
    if any_status:
        return status not in ("aarch32", "compile_error") or bool(prior.get("build_seconds"))
    return status in ("compiled", "booted", "black", "non-black")


def capture_prefix(gdir: Path) -> Path:
    """The stem the runtime appends `_NNNNNN.ppm` to for `KIRIKAE_CAPTURE`."""
    return gdir / "frame"


def capture_file(gdir: Path, frame: int) -> Path:
    return gdir / f"frame_{frame:06}.ppm"


def frame_diff_module():
    """`tools/frame_diff.py` as a module; it is the single frame classifier."""
    tools = Path(__file__).resolve().parents[1]
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import frame_diff

    return frame_diff


def classify_frame(path: Path) -> dict:
    """Measure one captured frame with `frame_diff`: is there anything on screen?"""
    try:
        module = frame_diff_module()
        image = module.load(path)
        stats = module.analysis(image)
    except Exception as e:
        return {"error": str(e)}
    return {
        "file": path.name,
        "width": image.width,
        "height": image.height,
        "mean": [round(v, 1) for v in stats["mean"]],
        "luma": round(stats["luma"], 2),
        "nonblack": round(stats["nonblack"], 2),
        "distinct": stats["distinct"],
        "top_color": f"#{stats['top_color']:06x}",
        "top_share": round(stats["top_share"], 1),
        "black": stats["nonblack"] < CAPTURE_BLACK_BELOW,
    }


def capture_cell(cap: dict) -> tuple[str, str]:
    """(Frame, Visibility) cells for the matrix; empty when nothing was measured."""
    if not cap:
        return "", ""
    if cap.get("error"):
        return "unreadable", str(cap["error"])[:60]
    if cap.get("missing"):
        return "", "no frame written"
    frame = f"{cap['width']}x{cap['height']}, mean " + "/".join(f"{v:g}" for v in cap["mean"])
    visible = (
        f"{cap['nonblack']:.2f}% non-black, {cap['distinct']} colours, "
        f"{cap['top_color']} {cap['top_share']:.1f}%"
    )
    return frame, visible


def boot_game(
    game: Path,
    cfg: dict,
    exe: Path,
    gdir: Path,
    seconds: float,
    nvn: str | None,
    capture: bool = False,
    capture_frame: int = CAPTURE_DEFAULT_FRAME,
) -> dict:
    coverage = gdir / "coverage.json"
    profile = gdir / "game.profile"
    boot_log = gdir / "boot.log"
    try:
        coverage.unlink()
    except OSError:
        pass
    gdir.mkdir(parents=True, exist_ok=True)
    env = {
        "KIRIKAE_HEADLESS": "1",
        "KIRIKAE_MUTE": "1",
        "KIRIKAE_STATS": "1",
        "KIRIKAE_COVERAGE": str(coverage),
        "KIRIKAE_PROFILE": str(profile),
        "KIRIKAE_SAVES": str(gdir / "saves"),
    }
    if nvn:
        env["KIRIKAE_NVN"] = nvn
    if capture:
        for stale in gdir.glob("frame_*.ppm"):
            try:
                stale.unlink()
            except OSError:
                pass
        env["KIRIKAE_CAPTURE"] = str(capture_prefix(gdir))
        env["KIRIKAE_CAPTURE_FRAME"] = str(capture_frame)
    cmd = [*(str(x) for x in cfg.get("runner", [])), str(exe)]
    print(f"[{game.name}] boot {seconds:g}s …", flush=True)
    result = run(cmd, exe.parent, seconds, env)
    boot_log.write_text(
        f"$ {' '.join(cmd)}\n\n{result['stdout']}\n{result['stderr']}", encoding="utf-8", errors="replace"
    )
    rec = {
        "boot_seconds": result["seconds"],
        "exit": result["exit"],
        "timed_out": result["exit"] == "timeout",
    }
    rec.update(parse_stats(result["stdout"] + result["stderr"]))
    window = parse_stats_at(result["stdout"] + result["stderr"], 10)
    if window:
        rec["frames_10s"] = window["frames"]
        rec["fps_10s"] = window["fps"]
        for key in ("draws", "clears", "submits"):
            if key in window:
                rec[f"{key}_10s"] = window[key]

    if coverage.is_file():
        try:
            data = json.loads(coverage.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            rec["coverage_error"] = str(e)
        else:
            rec["title_id"] = data.get("title_id")
            gaps, top = coverage_gaps(data)
            rec["gaps"] = gaps
            rec["top_gaps"] = top
            if "frames" not in rec:
                rec.update(frames_from_coverage(data))

    errors = last_error_lines(result["stderr"] or result["stdout"])
    if errors:
        rec["errors"] = errors

    frames = int(rec.get("frames") or 0)
    if result["exit"] == "timeout":
        if frames > 0:
            rec["status"] = "booted"
            rec["note"] = f"ran to the {seconds:g}s bound"
        else:
            rec["status"] = "timeout"
            rec["note"] = f"no frames before the {seconds:g}s bound"
            if hint := error_hint(errors):
                rec["note"] += f": {hint[:120]}"
    elif result["exit"] == 0:
        if frames > 0:
            rec["status"] = "booted"
            rec["note"] = "exited 0"
        else:
            rec["status"] = "crash"
            rec["note"] = "exited 0 before the first frame"
    else:
        rec["status"] = "crash"
        rec["note"] = f"exit {result['exit']}"
        if hint := error_hint(errors):
            rec["note"] += f": {hint[:120]}"

    if capture:
        rec["capture_frame"] = capture_frame
        path = capture_file(gdir, capture_frame)
        if path.is_file():
            cap = classify_frame(path)
            rec["capture"] = cap
            if cap.get("error"):
                rec["note"] = f"capture unreadable: {cap['error']}"[:200]
            else:
                rec["status"] = "black" if cap["black"] else "non-black"
                was = rec.get("note") or ""
                seen = "black" if cap["black"] else "visible"
                rec["note"] = f"frame {capture_frame} {seen}" + (f"; {was}" if was else "")
        else:
            rec["capture"] = {"missing": True, "frame": capture_frame}
            extra = f"no frame {capture_frame} written"
            rec["note"] = f"{rec['note']}; {extra}" if rec.get("note") else extra
    return rec


def one_game(
    game: Path,
    cfg: dict,
    kirikae: Path,
    work: Path,
    boot: bool,
    skip_build: bool,
    resume: bool,
    boot_seconds: float,
    nvn: str | None,
    state: dict,
    loop: dict | None = None,
    capture: bool = False,
    capture_frame: int = CAPTURE_DEFAULT_FRAME,
    reuse_any: bool = False,
) -> dict:
    slug = slugify(game.name)
    gdir = work / slug
    out = gdir / "out"
    exe = out / f"{out_stem(game)}.exe"
    gdir.mkdir(parents=True, exist_ok=True)
    rec = {"game": game.name, "slug": slug, "source": str(game), "started": int(time.time()), "status": "pending"}

    prior = state.get(game.name, {})
    prior_status = normalize_status(prior.get("status"))
    reuse = can_reuse(prior, exe, reuse_any)
    skip_build_this = skip_build or (resume and reuse) or reuse_any

    if loop and skip_build_this:
        print(f"[{game.name}] profile loop skipped: reusing the existing build", flush=True)

    if not skip_build_this:
        print(f"[{game.name}] build …", flush=True)
        cmd = [str(kirikae), "build", str(game), "-o", str(out)]
        timeout = cfg.get("build_timeout", 1200)
        if loop:
            cmd += ["--profile-loop", str(loop["runs"]), "--run-seconds", f"{loop['seconds']:g}"]
            timeout = cfg.get(
                "profile_loop_timeout",
                timeout * loop["runs"] + (loop["seconds"] + 60) * loop["runs"],
            )
        r = run(cmd, ROOT, timeout)
        (gdir / "build.log").write_text(
            f"$ {' '.join(cmd)}\n\n{r['stdout']}\n{r['stderr']}", encoding="utf-8", errors="replace"
        )
        rec["build_seconds"] = r["seconds"]
        text = (r["stderr"] or "") + (r["stdout"] or "")
        rec.update(parse_build(text))
        if loop:
            if loop_rec := parse_profile_loop(text):
                rec["profile_loop"] = loop_rec
        if r["exit"] != 0:
            if "AArch32" in text:
                rec["status"] = "aarch32"
                rec["note"] = "32-bit title (WS12): the recompiler only handles AArch64"
            else:
                rec["status"] = "compile_error"
                rec["note"] = text.strip().splitlines()[-1][:200] if text.strip() else f"build exit {r['exit']}"
        else:
            rec["status"] = "compiled"
            rec["note"] = ""
    elif not exe.is_file():
        if prior_status in ("aarch32", "compile_error"):
            rec["status"] = prior_status
            rec["note"] = prior.get("note") or f"no executable at {exe}"
        else:
            rec["status"] = "compile_error"
            rec["note"] = f"no executable at {exe}; build it without --skip-build/--recapture"
    else:
        rec["status"] = "compiled"
        rec["note"] = ""
        rec["reused_build"] = True
        for key in ("functions", "instructions", "reachable_total", "build_seconds", "title_id"):
            if key in prior:
                rec[key] = prior[key]

    if boot and rec["status"] == "compiled":
        if not exe.is_file():
            rec["status"] = "compile_error"
            rec["note"] = f"no executable at {exe}; build it without --skip-build/--recapture"
        else:
            rec.update(boot_game(game, cfg, exe, gdir, boot_seconds, nvn, capture, capture_frame))

    rec["finished"] = int(time.time())
    print(f"[{game.name}] {rec['status']}" + (f" ({rec['note']})" if rec.get("note") else ""), flush=True)
    append_state(work / "state.jsonl", rec)
    return rec


def matrix_rows(state: dict) -> list[dict]:
    rank = {
        "non-black": 0,
        "black": 1,
        "booted": 2,
        "timeout": 3,
        "compiled": 4,
        "crash": 5,
        "compile_error": 6,
        "aarch32": 7,
    }
    rows = list(state.values())
    for r in rows:
        r["status"] = normalize_status(r.get("status"))
    return sorted(rows, key=lambda r: (rank.get(r.get("status"), 9), r["game"]))


def fps_text(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else ""


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_matrix(work: Path, state: dict) -> None:
    rows = matrix_rows(state)
    capture = any(isinstance(r.get("capture"), dict) for r in rows)
    lines = [
        "# Kirikae compatibility matrix",
        "",
        "Generated by `tools/discovery/batch.py`. Game data stays outside the repository.",
        "",
    ]
    if capture:
        lines += [
            "| Game | Status | Frames@10s | FPS | Draws | Frame | Visibility | Gaps | Note |",
            "|---|---|---:|---:|---:|---|---|---:|---|",
        ]
        for r in rows:
            frame, visible = capture_cell(r.get("capture") or {})
            note = md_cell(r.get("note") or "")[:80]
            frames = r.get("frames_10s", r.get("frames", ""))
            fps = fps_text(r.get("fps_10s", r.get("fps")))
            draws = r.get("draws_10s", r.get("draws", ""))
            lines.append(
                f"| {r['game']} | {r['status']} | {frames} | {fps} | {draws} | "
                f"{md_cell(frame)} | {md_cell(visible)} | {r.get('gaps', '')} | {note} |"
            )
    else:
        lines += [
            "| Game | Status | Functions | Instructions | Frames | FPS | Gaps | Top gaps | Note | Profile |",
            "|---|---|---:|---|---:|---:|---:|---|---|---|",
        ]
        for r in rows:
            note = md_cell(r.get("note") or "")[:80]
            top = r.get("top_gaps") or r.get("gap_top") or []
            top_text = md_cell(", ".join(str(t) for t in top[:3]))[:70]
            profile = md_cell(profile_loop_text(r.get("profile_loop") or {}))
            lines.append(
                f"| {r['game']} | {r['status']} | {r.get('functions', '')} | "
                f"{r.get('instructions', '')}/{r.get('reachable_total', '')} | {r.get('frames', '')} | "
                f"{fps_text(r.get('fps'))} | {r.get('gaps', '')} | {top_text} | {note} | {profile} |"
            )
    (work / "compat.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (work / "compat.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summary_rows(state: dict) -> list[dict]:
    out = []
    for r in matrix_rows(state):
        entry = {
            "slug": r.get("slug") or slugify(r["game"]),
            "title_id": r.get("title_id"),
            "status": r["status"],
            "functions": r.get("functions"),
            "instructions": r.get("instructions"),
            "frames": r.get("frames"),
            "fps": r.get("fps"),
            "gaps": r.get("gaps"),
            "top_gaps": r.get("top_gaps") or r.get("gap_top") or [],
            "note": r.get("note") or "",
        }
        if isinstance(r.get("capture"), dict):
            entry["capture"] = r["capture"]
        out.append(entry)
    return out


def write_summary(work: Path, state: dict) -> Path:
    path = work / "compat.summary.json"
    doc = {"schema": SUMMARY_SCHEMA, "generated": int(time.time()), "games": summary_rows(state)}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def select_games(games: list[Path], only: str | None, game: str | None) -> list[Path]:
    if game:
        games = [g for g in games if g.name == game]
    if only:
        wanted = {s.strip().lower() for s in only.split(",") if s.strip()}
        by_slug = {slugify(g.name).lower(): g for g in games}
        by_name = {g.name.lower(): g for g in games}
        selected = []
        missing = []
        for key in sorted(wanted):
            match = by_slug.get(key) or by_name.get(key)
            if match is None:
                missing.append(key)
            elif match not in selected:
                selected.append(match)
        if missing:
            known = ", ".join(sorted(by_slug))
            raise SystemExit(f"unknown game(s): {', '.join(missing)}; known: {known}")
        games = selected
    return games


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending games")
    ap.add_argument("--game", help="only this game folder name")
    ap.add_argument("--only", help="comma-separated slugs or folder names")
    ap.add_argument("--compile-only", action="store_true", help="build, do not boot")
    ap.add_argument("--skip-build", action="store_true", help="boot the executable already in <work>/<slug>/out")
    ap.add_argument("--resume", action="store_true", help="skip finished games and reuse existing builds")
    ap.add_argument(
        "--recapture",
        action="store_true",
        help="measure again from the build already on disk, ignoring the recorded status",
    )
    ap.add_argument("--force", action="store_true", help="redo even if the state says it is finished")
    ap.add_argument("--boot-seconds", type=float, default=None, help="boot bound (default: config or 20)")
    ap.add_argument("--nvn", nargs="?", const="trace", default=None, help="set KIRIKAE_NVN (default trace when given)")
    ap.add_argument("--profile-loop", action="store_true", help="build with kirikae's profile loop (starts each game)")
    ap.add_argument("--capture", action="store_true", help="capture one presented frame and classify it as black or visible")
    ap.add_argument(
        "--capture-frame",
        type=int,
        default=CAPTURE_DEFAULT_FRAME,
        help=f"which presented frame to capture (default {CAPTURE_DEFAULT_FRAME})",
    )
    ap.add_argument("--profile-runs", type=int, default=None, help="profile-loop iterations (default: config or 3)")
    ap.add_argument(
        "--profile-seconds",
        type=float,
        default=None,
        help="bound for each profile-loop run (default: config or boot_seconds, at least 10)",
    )
    ap.add_argument("--json", action="store_true", help="write the compact compat.summary.json")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.profile_loop and args.skip_build:
        raise SystemExit("--profile-loop builds and runs; it cannot be combined with --skip-build")
    if args.profile_loop and args.compile_only:
        raise SystemExit("--profile-loop needs runs; it cannot be combined with --compile-only")
    if args.capture and args.compile_only:
        raise SystemExit("--capture needs a boot run; it cannot be combined with --compile-only")
    if args.recapture and args.compile_only:
        raise SystemExit("--recapture reuses builds and never compiles; drop --compile-only")
    if args.capture_frame < 1:
        raise SystemExit("--capture-frame must be at least 1")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    loop = None
    boot_seconds = args.boot_seconds if args.boot_seconds is not None else cfg.get("boot_seconds", 20)
    if args.profile_loop:
        runs = args.profile_runs if args.profile_runs is not None else cfg.get("profile_loop_runs", 3)
        seconds = args.profile_seconds if args.profile_seconds is not None else cfg.get("profile_loop_seconds")
        if seconds is None:
            seconds = max(10.0, float(boot_seconds))
        if runs < 1:
            raise SystemExit("--profile-runs must be at least 1")
        if seconds < 10:
            raise SystemExit("--profile-seconds must be at least 10: the stats window is 10 s")
        loop = {"runs": int(runs), "seconds": float(seconds)}

    kirikae = kirikae_bin(cfg)
    games_root = Path(os.path.expandvars(cfg["games_root"]))
    work = work_root(cfg)
    work.mkdir(parents=True, exist_ok=True)
    state = load_state(work / "state.jsonl")
    if boot_seconds <= 0:
        raise SystemExit("--boot-seconds must be positive")
    nvn = args.nvn if args.nvn is not None else cfg.get("nvn")

    games = select_games(discover(games_root), args.only, args.game)
    if not games:
        raise SystemExit(f"no game folders under {games_root}")
    print(f"{len(games)} game(s) under {games_root}; work: {work}", flush=True)

    boot = not args.compile_only
    done = 0
    for game in games:
        prior = state.get(game.name, {})
        if not args.force and not args.recapture and is_complete(prior, boot, args.resume):
            print(f"[{game.name}] skip (already {normalize_status(prior.get('status'))})", flush=True)
            continue
        try:
            one_game(
                game,
                cfg,
                kirikae,
                work,
                boot,
                args.skip_build,
                (args.resume or args.recapture) and not args.force,
                boot_seconds,
                nvn,
                state,
                loop,
                args.capture,
                args.capture_frame,
                args.recapture,
            )
        except Exception as e:
            rec = {
                "game": game.name,
                "slug": slugify(game.name),
                "source": str(game),
                "started": int(time.time()),
                "finished": int(time.time()),
                "status": "crash",
                "note": f"batch error: {e}",
            }
            append_state(work / "state.jsonl", rec)
            print(f"[{game.name}] crash (batch error: {e})", flush=True)
        state = load_state(work / "state.jsonl")
        done += 1
        if args.limit and done >= args.limit:
            break

    write_matrix(work, state)
    summary = write_summary(work, state) if args.json else None
    statuses = [normalize_status(r.get("status")) for r in state.values()]
    boots = sum(1 for s in statuses if s in ("booted", "non-black", "black"))
    compiled = sum(1 for s in statuses if s in ("compiled", "booted", "non-black", "black"))
    visible = sum(1 for s in statuses if s == "non-black")
    black = sum(1 for s in statuses if s == "black")
    print(f"\nmatrix: {work / 'compat.md'}  ({visible} visible, {black} black, {boots} boot, {compiled} compile, {len(state)} total)")
    if summary:
        print(f"summary: {summary}")


if __name__ == "__main__":
    main(sys.argv[1:])
