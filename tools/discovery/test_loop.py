#!/usr/bin/env python3
"""Tests for tools/discovery/loop.py (standard library only).

    python tools/discovery/test_loop.py
    python -m unittest discover -s tools/discovery -p "test_*.py"

The cycle test uses the same idea as `test_batch.py`: a Python script with an
`.exe` suffix is the "game" and `runner: [sys.executable]` runs it. It writes the
frame dumps the schedule asks for, so `loop.py` classifies real files with
`tools/frame_diff.py`. No game data is involved.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loop  # noqa: E402

FAKE_GAME = r'''
import json, os, struct, sys, zlib

def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

def png(path, color):
    # 2x2 RGB PNG so frame_diff reads a real image. The parent folder is created by
    # loop.py: this is the regression check for a frame-indexed schedule.
    raw = b"".join(b"\x00" + bytes(color) * 2 for _ in range(2))
    data = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "env.json"), "w", encoding="utf-8") as f:
    json.dump({k: v for k, v in os.environ.items()
               if k.startswith("KIRIKAE_") and k != "KIRIKAE_BIN"}, f)

folder = os.environ.get("KIRIKAE_FRAME_DUMP")
if folder:
    every = int(os.environ.get("KIRIKAE_FRAME_DUMP_EVERY", "60"))
    start = int(os.environ.get("KIRIKAE_FRAME_DUMP_FROM", "0"))
    count = int(os.environ.get("KIRIKAE_FRAME_DUMP_COUNT", "1"))
    for i in range(count):
        color = (0, 0, 0) if i == 0 else (40 * i, 10 * i, 200 - 20 * i)
        png(os.path.join(folder, f"frame{start + i * every:06}.png"), color)

prefix = os.environ.get("KIRIKAE_CAPTURE")
if prefix:
    frame = int(os.environ.get("KIRIKAE_CAPTURE_FRAME", "1"))
    path = f"{prefix}_{frame:06}.ppm"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"P6\n1 1\n255\n" + bytes([10, 20, 30]))

coverage = os.environ.get("KIRIKAE_COVERAGE")
if coverage:
    os.makedirs(os.path.dirname(coverage), exist_ok=True)
    with open(coverage, "w", encoding="utf-8") as f:
        json.dump({"items": [], "runs": [{"frames": 60, "seconds": 1.0}]}, f)
print("kirikae: [  10s] 60 frames (30.0 fps, 30.0 without the limit); 1 native calls, 0 interpreted")
'''


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "game"
        (self.source / "exefs").mkdir(parents=True)
        (self.source / "exefs" / "main").write_bytes(b"")
        self.out = self.root / "out"
        self.out.mkdir()
        (self.out / "game.exe").write_text(FAKE_GAME, encoding="utf-8")
        self.work = self.root / "discovery"
        self.saved_env = dict(os.environ)
        os.environ["KIRIKAE_DISCOVERY_DIR"] = str(self.work)
        self.kirikae = self.root / "kirikae"
        self.kirikae.write_text("", encoding="utf-8")
        os.environ["KIRIKAE_BIN"] = str(self.kirikae)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.saved_env)
        self.tmp.cleanup()

    def config(self, scenario: dict, runner: bool = True) -> dict:
        cfg = {
            "kirikae": str(self.root / "kirikae"),
            "games": [
                {
                    "name": "game",
                    "source": str(self.source),
                    "out": str(self.out),
                    "scenarios": [scenario],
                }
            ],
        }
        if runner:
            cfg["runner"] = [sys.executable]
        return cfg

    def test_schedule_starts_after_one_interval(self) -> None:
        self.assertEqual(
            loop.capture_schedule(125, 20, 30),
            {"every_frames": 600, "from_frame": 600, "count": 6},
        )
        self.assertEqual(loop.capture_schedule(5, 20, 30)["count"], 1)
        self.assertEqual(loop.capture_schedule(10, 0.5, 30)["every_frames"], 15)
        with self.assertRaises(ValueError):
            loop.capture_schedule(10, 0, 30)
        with self.assertRaises(ValueError):
            loop.capture_schedule(0, 10, 30)

    def test_frame_schedule_does_not_depend_on_fps(self) -> None:
        self.assertEqual(
            loop.capture_frames_schedule(160, 30, 60),
            {"every_frames": 60, "from_frame": 60, "count": 80},
        )
        self.assertEqual(loop.capture_frames_schedule(4, 30, 60)["count"], 2)
        self.assertEqual(
            loop.capture_frames_schedule(4, 30, 60, 7)["count"], 7
        )
        with self.assertRaises(ValueError):
            loop.capture_frames_schedule(0, 30, 60)

    def test_scenario_env_maps_the_schedule(self) -> None:
        run_dir = self.root / "run"
        env = loop.scenario_env(
            {"name": "game"},
            {"name": "play", "seconds": 100, "capture_every": 20, "capture_fps": 30},
            run_dir,
        )
        self.assertEqual(env["KIRIKAE_FRAME_DUMP"], str(run_dir / "frames"))
        self.assertEqual(env["KIRIKAE_FRAME_DUMP_EVERY"], "600")
        self.assertEqual(env["KIRIKAE_FRAME_DUMP_FROM"], "600")
        self.assertEqual(env["KIRIKAE_FRAME_DUMP_COUNT"], "5")
        self.assertNotIn("KIRIKAE_CAPTURE", env)

    def test_cycle_classifies_every_dump(self) -> None:
        scenario = {
            "name": "play",
            "seconds": 3,
            "capture_every": 1,
            "capture_fps": 10,
            "capture_frame": 30,
            "input": "1:A",
        }
        manifest_path = loop.cycle(self.config(scenario), None, None, True, False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run = self.work / "game" / manifest["run_id"] / "run" / "play"

        env = json.loads((self.out / "env.json").read_text(encoding="utf-8"))
        self.assertEqual(env["KIRIKAE_INPUT"], "1:A")
        self.assertEqual(env["KIRIKAE_FRAME_DUMP_EVERY"], "10")
        self.assertEqual(env["KIRIKAE_CAPTURE_FRAME"], "30")
        self.assertTrue((run / "capture_000030.ppm").is_file())

        captures = json.loads((run / "captures.json").read_text(encoding="utf-8"))
        self.assertEqual(captures["schema"], 1)
        self.assertEqual(captures["fps"], 10)
        self.assertEqual([c["frame"] for c in captures["frames"]], [10, 20, 30])
        self.assertEqual([c["seconds"] for c in captures["frames"]], [1.0, 2.0, 3.0])
        self.assertTrue(captures["frames"][0]["black"])
        self.assertFalse(captures["frames"][1]["black"])

        recorded = manifest["games"]["game"]["scenarios"]["play"]["captures"]
        self.assertEqual(len(recorded), 3)
        self.assertEqual(recorded[2]["top_color"], "#5014a0")

    def test_cycle_records_frame_indexed_captures_without_seconds(self) -> None:
        scenario = {
            "name": "play",
            "seconds": 1,
            "capture_every_frames": 5,
            "capture_fps": 10,
        }
        manifest_path = loop.cycle(self.config(scenario), None, None, True, False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run = self.work / "game" / manifest["run_id"] / "run" / "play"
        captures = json.loads((run / "captures.json").read_text(encoding="utf-8"))
        self.assertEqual(captures["fps"], 0)
        self.assertEqual([c["frame"] for c in captures["frames"]], [5, 10])
        self.assertEqual([c["seconds"] for c in captures["frames"]], [None, None])

    def test_without_capture_every_no_dump_variables(self) -> None:
        scenario = {"name": "play", "seconds": 2, "input": "1:A"}
        manifest_path = loop.cycle(self.config(scenario), None, None, True, False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run = self.work / "game" / manifest["run_id"] / "run" / "play"
        env = json.loads((self.out / "env.json").read_text(encoding="utf-8"))
        self.assertNotIn("KIRIKAE_FRAME_DUMP", env)
        self.assertNotIn("captures", manifest["games"]["game"]["scenarios"]["play"])
        self.assertFalse((run / "captures.json").exists())


if __name__ == "__main__":
    unittest.main()
