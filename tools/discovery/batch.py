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

Statuses:
  compiled       built, boot not requested
  booted         ran and produced frames (a headless boot normally hits the bound)
  timeout        the bound expired before the first frame
  crash          the process exited before the first frame, or failed to start
  aarch32        the title is ARM32, which the recompiler does not handle yet
  compile_error  the build failed, or no executable was available

Usage:
    python tools/discovery/batch.py --config batch.json
    python tools/discovery/batch.py --config batch.json --limit 10
    python tools/discovery/batch.py --config batch.json --only super-mario-odyssey,pokemon-sword
    python tools/discovery/batch.py --config batch.json --resume
    python tools/discovery/batch.py --config batch.json --skip-build --boot-seconds 20 --nvn trace
    python tools/discovery/batch.py --config batch.json --compile-only
    python tools/discovery/batch.py --config batch.json --json

`--json` writes `<work>/compat.summary.json` (schema 1) next to the matrix:

    {"schema": 1, "generated": 172..., "games": [
      {"slug", "title_id", "status", "functions", "instructions",
       "frames", "fps", "gaps", "top_gaps", "note"}
    ]}

The full per-game record (exit code, draws, clears, errors, timings) stays in
`compat.json` and `state.jsonl`. `top_gaps` is a list of `"<interface> <id>"`
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

STATS_FRAMES = re.compile(r"\[\s*\d+s\]\s+(\d+)\s+frames\s*\(\s*([\d.]+)\s+fps")
STATS_GPU = re.compile(
    r"GPU:\s*(\d+)\s+submits,\s*(\d+)\s+draws,\s*(\d+)\s+clears,\s*(\d+)\s+copies,\s*(\d+)\s+dispatches"
)
STATS_CALLS = re.compile(r"(\d+)\s+native calls,\s*(\d+)\s+interpreted")
ERROR_WORDS = re.compile(r"error|failed|panic|abort|svcBreak|unimplemented|missing|deadlock", re.I)
STATS_NOISE = ("kirikae: [", "kirikae:         ")


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
                    # The log is append-only: a boot-only record must not hide the build metrics.
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
        m = STATS_FRAMES.search(line)
        if not m:
            continue
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
    return {}


def parse_frames(text: str) -> dict:
    frames = parse_stats(text)
    return {k: frames[k] for k in ("frames", "fps") if k in frames}


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
    done = {"booted"} if boot else {"compiled", "booted"}
    if resume:
        done |= {"aarch32", "compile_error", "crash", "timeout"}
    return status in done


def boot_game(game: Path, cfg: dict, exe: Path, gdir: Path, seconds: float, nvn: str | None) -> dict:
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
) -> dict:
    slug = slugify(game.name)
    gdir = work / slug
    out = gdir / "out"
    exe = out / f"{out_stem(game)}.exe"
    gdir.mkdir(parents=True, exist_ok=True)
    rec = {"game": game.name, "slug": slug, "source": str(game), "started": int(time.time()), "status": "pending"}

    prior = state.get(game.name, {})
    prior_status = normalize_status(prior.get("status"))
    reuse = exe.is_file() and prior_status in ("compiled", "booted")
    skip_build_this = skip_build or (resume and reuse)

    if not skip_build_this:
        print(f"[{game.name}] build …", flush=True)
        r = run([str(kirikae), "build", str(game), "-o", str(out)], ROOT, cfg.get("build_timeout", 1200))
        (gdir / "build.log").write_text(
            f"$ {kirikae} build {game}\n\n{r['stdout']}\n{r['stderr']}", encoding="utf-8", errors="replace"
        )
        rec["build_seconds"] = r["seconds"]
        rec.update(parse_build(r["stdout"] + r["stderr"]))
        if r["exit"] != 0:
            text = (r["stderr"] or "") + (r["stdout"] or "")
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
            rec["note"] = f"no executable at {exe}; build it without --skip-build"
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
            rec["note"] = f"no executable at {exe}; build it without --skip-build"
        else:
            rec.update(boot_game(game, cfg, exe, gdir, boot_seconds, nvn))

    rec["finished"] = int(time.time())
    print(f"[{game.name}] {rec['status']}" + (f" ({rec['note']})" if rec.get("note") else ""), flush=True)
    append_state(work / "state.jsonl", rec)
    return rec


def matrix_rows(state: dict) -> list[dict]:
    rank = {"booted": 0, "timeout": 1, "compiled": 2, "crash": 3, "compile_error": 4, "aarch32": 5}
    rows = list(state.values())
    for r in rows:
        r["status"] = normalize_status(r.get("status"))
    return sorted(rows, key=lambda r: (rank.get(r.get("status"), 9), r["game"]))


def write_matrix(work: Path, state: dict) -> None:
    rows = matrix_rows(state)
    lines = [
        "# Kirikae compatibility matrix",
        "",
        "Generated by `tools/discovery/batch.py`. Game data stays outside the repository.",
        "",
        "| Game | Status | Functions | Instructions | Frames | FPS | Gaps | Top gaps | Note |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        note = str(r.get("note") or "").replace("|", "\\|")[:80]
        top = r.get("top_gaps") or r.get("gap_top") or []
        top_text = ", ".join(str(t) for t in top[:3]).replace("|", "\\|")[:70]
        fps = r.get("fps")
        fps_text = f"{fps:.1f}" if isinstance(fps, (int, float)) else ""
        lines.append(
            f"| {r['game']} | {r['status']} | {r.get('functions', '')} | "
            f"{r.get('instructions', '')}/{r.get('reachable_total', '')} | {r.get('frames', '')} | "
            f"{fps_text} | {r.get('gaps', '')} | {top_text} | {note} |"
        )
    (work / "compat.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (work / "compat.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summary_rows(state: dict) -> list[dict]:
    out = []
    for r in matrix_rows(state):
        out.append(
            {
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
        )
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
    ap.add_argument("--force", action="store_true", help="redo even if the state says it is finished")
    ap.add_argument("--boot-seconds", type=float, default=None, help="boot bound (default: config or 20)")
    ap.add_argument("--nvn", nargs="?", const="trace", default=None, help="set KIRIKAE_NVN (default trace when given)")
    ap.add_argument("--json", action="store_true", help="write the compact compat.summary.json")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    kirikae = kirikae_bin(cfg)
    games_root = Path(os.path.expandvars(cfg["games_root"]))
    work = work_root(cfg)
    work.mkdir(parents=True, exist_ok=True)
    state = load_state(work / "state.jsonl")
    boot_seconds = args.boot_seconds if args.boot_seconds is not None else cfg.get("boot_seconds", 20)
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
        if not args.force and is_complete(prior, boot, args.resume):
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
                args.resume and not args.force,
                boot_seconds,
                nvn,
                state,
            )
        except Exception as e:  # noqa: BLE001 - one bad game must not stop the batch
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
    boots = sum(1 for r in state.values() if normalize_status(r.get("status")) == "booted")
    compiled = sum(1 for r in state.values() if normalize_status(r.get("status")) in ("compiled", "booted"))
    print(f"\nmatrix: {work / 'compat.md'}  ({boots} boot, {compiled} compile, {len(state)} total)")
    if summary:
        print(f"summary: {summary}")


if __name__ == "__main__":
    main(sys.argv[1:])
