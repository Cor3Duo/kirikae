#!/usr/bin/env python3
"""Tests for tools/discovery/batch.py (standard library only).

    python tools/discovery/test_batch.py
    python -m unittest discover -s tools/discovery -p "test_*.py"

The boot tests use a fake `kirikae` launcher and a fake game executable. No game
data is involved: the "games" are empty folders and the executables are Python
scripts written under a temporary directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch

GAME_SCRIPT = r'''
import json, os, sys, time

mode = os.environ.get("FAKE_MODE", "success")
pid_file = os.environ.get("FAKE_PID_FILE")
if pid_file:
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
nvn_file = os.environ.get("FAKE_NVN_FILE")
if nvn_file:
    with open(nvn_file, "w", encoding="utf-8") as f:
        f.write(os.environ.get("KIRIKAE_NVN", ""))
if mode == "timeout":
    time.sleep(120)
    sys.exit(0)
if mode == "crash":
    print("kirikae: guest crashed: svcBreak at 0x1234", file=sys.stderr)
    sys.exit(3)
print("kirikae: [  10s] 120 frames (30.0 fps, 60.0 without the limit, longest 33 ms), "
      "GPU: 4 submits, 9 draws, 2 clears, 0 copies, 0 dispatches; 55 native calls, 3 interpreted")
coverage = os.environ.get("KIRIKAE_COVERAGE")
if coverage:
    data = {
        "format": 1,
        "game": "fake",
        "title_id": "0100000000000001",
        "items": [
            {"section": "services", "interface": "prepo:u", "id": 10101, "count": 9, "status": "missing"},
            {"section": "svc", "interface": "SVCs", "id": 45, "count": 3, "status": "stub"},
            {"section": "ioctl", "interface": "nvdrv", "id": "ZBCSetTable", "count": 7, "status": "implemented"},
        ],
        "runs": [{"frames": 120, "seconds": 10.0}],
    }
    os.makedirs(os.path.dirname(coverage), exist_ok=True)
    with open(coverage, "w", encoding="utf-8") as f:
        json.dump(data, f)
sys.exit(0)
'''

FAKE_KIRIKAE = r'''
import os, sys
from pathlib import Path

args = sys.argv[1:]
if not args or args[0] != "build":
    print("fake kirikae: only `build` is supported", file=sys.stderr)
    sys.exit(2)
log = os.environ.get("FAKE_BUILD_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write(" ".join(args) + "\n")
mode = os.environ.get("FAKE_MODE", "success")
if mode == "compile_error":
    print("error: cargo build failed (exit code: 101)", file=sys.stderr)
    sys.exit(1)
source = Path(args[1])
out = Path(args[args.index("-o") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / (source.name + ".exe")).write_text(os.environ["FAKE_GAME_SCRIPT"], encoding="utf-8")
print("kirikae: recompiled 3 functions (30 of 40 instructions reachable) into " + str(out))
'''


def make_launcher(tmp: Path, script: Path) -> Path:
    if os.name == "nt":
        launcher = tmp / "kirikae.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp / "kirikae"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, errors="replace"
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class BatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.games = self.root / "games"
        for name in ("game-a", "game-b"):
            (self.games / name / "exefs").mkdir(parents=True)
            (self.games / name / "exefs" / "main").write_bytes(b"")
        self.work = self.root / "work"
        self.fake_script = self.root / "fake_kirikae.py"
        self.fake_script.write_text(FAKE_KIRIKAE, encoding="utf-8")
        self.kirikae = make_launcher(self.root, self.fake_script)
        self.game_script = self.root / "fake_game.py"
        self.game_script.write_text(GAME_SCRIPT, encoding="utf-8")
        self.build_log = self.root / "builds.log"
        self.config = self.root / "batch.json"
        self.config.write_text(
            json.dumps(
                {
                    "kirikae": str(self.kirikae),
                    "games_root": str(self.games),
                    "work": str(self.work),
                    "build_timeout": 60,
                    "runner": [sys.executable],
                }
            ),
            encoding="utf-8",
        )
        self.saved_env = dict(os.environ)
        os.environ["KIRIKAE_BIN"] = str(self.kirikae)
        os.environ["FAKE_MODE"] = "success"
        os.environ["FAKE_BUILD_LOG"] = str(self.build_log)
        os.environ["FAKE_GAME_SCRIPT"] = GAME_SCRIPT
        os.environ.pop("FAKE_PID_FILE", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved_env)
        self.tmp.cleanup()

    def run_batch(self, *extra: str) -> None:
        batch.main(["--config", str(self.config), *extra])

    def state(self) -> dict[str, dict]:
        return batch.load_state(self.work / "state.jsonl")

    def test_success_boot_and_json_summary(self) -> None:
        nvn_file = self.root / "nvn"
        os.environ["FAKE_NVN_FILE"] = str(nvn_file)
        self.run_batch("--only", "game-a", "--boot-seconds", "2", "--nvn", "trace", "--json")

        self.assertEqual(nvn_file.read_text(encoding="utf-8"), "trace")
        state = self.state()
        self.assertEqual(list(state), ["game-a"])
        rec = state["game-a"]
        self.assertEqual(rec["status"], "booted")
        self.assertEqual(rec["functions"], 3)
        self.assertEqual(rec["instructions"], 30)
        self.assertEqual(rec["frames"], 120)
        self.assertEqual(rec["fps"], 30.0)
        self.assertEqual(rec["draws"], 9)
        self.assertEqual(rec["clears"], 2)
        self.assertEqual(rec["gaps"], 2)
        self.assertEqual(rec["top_gaps"], ["prepo:u 10101", "SVCs 45"])
        self.assertEqual(rec["title_id"], "0100000000000001")
        self.assertEqual(rec["exit"], 0)

        matrix = (self.work / "compat.md").read_text(encoding="utf-8")
        self.assertIn(
            "| Game | Status | Functions | Instructions | Frames | FPS | Gaps | Top gaps | Note |", matrix
        )
        self.assertIn("| game-a | booted | 3 | 30/40 | 120 | 30.0 | 2 | prepo:u 10101, SVCs 45 |", matrix)

        summary = json.loads((self.work / "compat.summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema"], batch.SUMMARY_SCHEMA)
        self.assertEqual(len(summary["games"]), 1)
        entry = summary["games"][0]
        self.assertEqual(
            set(entry),
            {"slug", "title_id", "status", "functions", "instructions", "frames", "fps", "gaps", "top_gaps", "note"},
        )
        self.assertEqual(entry["slug"], "game-a")
        self.assertEqual(entry["status"], "booted")
        self.assertEqual(entry["top_gaps"], ["prepo:u 10101", "SVCs 45"])

    def test_timeout_kills_the_process_and_continues(self) -> None:
        pid_file = self.root / "pid"
        os.environ["FAKE_MODE"] = "timeout"
        os.environ["FAKE_PID_FILE"] = str(pid_file)
        started = time.time()
        self.run_batch("--boot-seconds", "1")

        self.assertLess(time.time() - started, 30)
        state = self.state()
        self.assertEqual(state["game-a"]["status"], "timeout")
        self.assertEqual(state["game-a"]["exit"], "timeout")
        self.assertEqual(state["game-b"]["status"], "timeout")
        pid = int(pid_file.read_text(encoding="utf-8"))
        self.assertFalse(pid_alive(pid), "the boot timeout left an orphan process")

    def test_compile_error_is_reported(self) -> None:
        os.environ["FAKE_MODE"] = "compile_error"
        self.run_batch("--only", "game-a")

        rec = self.state()["game-a"]
        self.assertEqual(rec["status"], "compile_error")
        self.assertIn("cargo build failed", rec["note"])

    def test_crash_is_reported(self) -> None:
        os.environ["FAKE_MODE"] = "crash"
        self.run_batch("--only", "game-a", "--boot-seconds", "2")

        rec = self.state()["game-a"]
        self.assertEqual(rec["status"], "crash")

    def test_resume_does_not_retry_finished_failures(self) -> None:
        os.environ["FAKE_MODE"] = "compile_error"
        self.run_batch("--only", "game-a")
        self.assertEqual(self.state()["game-a"]["status"], "compile_error")
        self.assertEqual(len(self.build_log.read_text(encoding="utf-8").splitlines()), 1)

        os.environ["FAKE_MODE"] = "success"
        self.run_batch("--only", "game-a", "--resume")
        self.assertEqual(self.state()["game-a"]["status"], "compile_error")
        self.assertEqual(len(self.build_log.read_text(encoding="utf-8").splitlines()), 1)

        self.run_batch("--only", "game-a", "--boot-seconds", "2")
        self.assertEqual(self.state()["game-a"]["status"], "booted")
        self.assertEqual(len(self.build_log.read_text(encoding="utf-8").splitlines()), 2)

    def test_resume_reuses_the_build_and_keeps_metrics(self) -> None:
        self.run_batch("--only", "game-a", "--compile-only")
        rec = self.state()["game-a"]
        self.assertEqual(rec["status"], "compiled")
        self.assertEqual(rec["functions"], 3)
        self.assertEqual(len(self.build_log.read_text(encoding="utf-8").splitlines()), 1)

        self.run_batch("--only", "game-a", "--resume", "--boot-seconds", "2")
        rec = self.state()["game-a"]
        self.assertEqual(rec["status"], "booted")
        self.assertEqual(rec["functions"], 3)
        self.assertEqual(rec["instructions"], 30)
        self.assertTrue(rec.get("reused_build"))
        self.assertEqual(len(self.build_log.read_text(encoding="utf-8").splitlines()), 1)

    def test_unknown_only_slug_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_batch("--only", "nope")



class ParserTest(unittest.TestCase):
    STATS = (
        "kirikae: [  10s] 300 frames (30.0 fps, 60.1 without the limit, longest 33 ms), "
        "GPU: 12 submits, 34 draws, 5 clears, 2 copies, 1 dispatches; 1000 native calls, 7 interpreted\n"
    )

    def test_parse_stats_takes_the_last_line(self) -> None:
        text = self.STATS + self.STATS.replace("300 frames", "600 frames").replace("34 draws", "40 draws")
        rec = batch.parse_stats(text)
        self.assertEqual(rec["frames"], 600)
        self.assertEqual(rec["fps"], 30.0)
        self.assertEqual(rec["submits"], 12)
        self.assertEqual(rec["draws"], 40)
        self.assertEqual(rec["clears"], 5)
        self.assertEqual(rec["dispatches"], 1)
        self.assertEqual(rec["native_calls"], 1000)
        self.assertEqual(rec["interpreted"], 7)

    def test_parse_stats_without_gpu(self) -> None:
        rec = batch.parse_stats("kirikae: [  10s] 42 frames (4.2 fps, 4.2 without the limit)")
        self.assertEqual(rec, {"frames": 42, "fps": 4.2})

    def test_parse_build(self) -> None:
        rec = batch.parse_build("kirikae: recompiled 114500 functions (4846472 of 4921233 instructions reachable)")
        self.assertEqual(rec, {"functions": 114500, "instructions": 4846472, "reachable_total": 4921233})
        self.assertEqual(batch.parse_build("nothing here"), {})

    def test_coverage_gaps_sorts_by_count(self) -> None:
        data = {
            "items": [
                {"section": "svc", "interface": "SVCs", "id": 45, "count": 3, "status": "stub"},
                {"section": "services", "interface": "prepo:u", "id": 10101, "count": 9, "status": "missing"},
                {"section": "ioctl", "interface": "nvdrv", "id": "X", "count": 7, "status": "implemented"},
            ]
        }
        count, top = batch.coverage_gaps(data)
        self.assertEqual(count, 2)
        self.assertEqual(top, ["prepo:u 10101", "SVCs 45"])

    def test_frames_from_coverage(self) -> None:
        self.assertEqual(batch.frames_from_coverage({"runs": [{"frames": 120, "seconds": 10.0}]}), {"frames": 120, "fps": 12.0})
        self.assertEqual(batch.frames_from_coverage({"runs": []}), {})

    def test_load_state_carries_build_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.jsonl"
            path.write_text(
                '{"game": "g", "status": "compiled", "functions": 3, "instructions": 30, "reachable_total": 40}\n'
                '{"game": "g", "status": "booted", "frames": 120}\n',
                encoding="utf-8",
            )
            rec = batch.load_state(path)["g"]
            self.assertEqual(rec["status"], "booted")
            self.assertEqual(rec["functions"], 3)
            self.assertEqual(rec["instructions"], 30)
            self.assertEqual(rec["reachable_total"], 40)

    def test_last_error_lines_prefers_errors(self) -> None:
        text = "kirikae: [  10s] 1 frames (0.1 fps)\nkirikae: all good\nkirikae: svcBreak at 0x1\n"
        self.assertEqual(batch.last_error_lines(text), ["kirikae: svcBreak at 0x1"])



if __name__ == "__main__":
    unittest.main(verbosity=2)
