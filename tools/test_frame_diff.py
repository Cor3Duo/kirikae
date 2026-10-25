#!/usr/bin/env python3
"""Tests for tools/frame_diff.py.

    python -m unittest discover -s tools -p "test_frame_diff.py"
"""

import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_diff


def write_ppm(path, width, height, pixels):
    body = bytearray()
    for px in pixels:
        body.extend(((px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + body)


def write_png(path, width, height, pixels):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            px = pixels[y * width + x]
            raw.extend(((px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF))
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def gradient(width, height):
    return [
        ((x * 7) & 0xFF) << 16 | ((y * 5) & 0xFF) << 8 | ((x + y) & 0xFF)
        for y in range(height)
        for x in range(width)
    ]


class ReadingTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_ppm_round_trip(self):
        path = self.root / "frame.ppm"
        write_ppm(path, 2, 2, [0x112233, 0x445566, 0x778899, 0xAABBCC])
        image = frame_diff.load(path)
        self.assertEqual((image.width, image.height), (2, 2))
        self.assertEqual(image.pixels, [0x112233, 0x445566, 0x778899, 0xAABBCC])

    def test_ascii_ppm_round_trip(self):
        path = self.root / "ascii.ppm"
        path.write_bytes(b"P3\n2 1\n255\n255 0 0  0 128 255\n")
        image = frame_diff.load(path)
        self.assertEqual(image.pixels, [0xFF0000, 0x0080FF])

    def test_png_round_trip(self):
        path = self.root / "frame.png"
        pixels = [0x112233, 0x445566, 0x778899, 0xAABBCC]
        write_png(path, 2, 2, pixels)
        image = frame_diff.load(path)
        self.assertEqual((image.width, image.height), (2, 2))
        self.assertEqual(image.pixels, pixels)

    def test_garbage_is_rejected(self):
        path = self.root / "bad.bin"
        path.write_bytes(b"not an image")
        with self.assertRaises(frame_diff.ImageError):
            frame_diff.load(path)


class AnalysisTest(unittest.TestCase):
    def test_black_frame_is_reported(self):
        image = frame_diff.Image(4, 4, [0] * 16, 4, "black")
        stats = frame_diff.analysis(image)
        self.assertEqual(stats["mean"], [0.0, 0.0, 0.0])
        self.assertEqual(stats["luma"], 0.0)
        self.assertEqual(stats["nonblack"], 0.0)
        self.assertEqual(stats["black"], 100.0)
        self.assertEqual(stats["top_share"], 100.0)

    def test_a_dark_but_lit_frame_is_not_black(self):
        image = frame_diff.Image(2, 1, [0x000000, 0x000009], 4, "dark")
        stats = frame_diff.analysis(image)
        self.assertEqual(stats["nonblack"], 50.0)
        self.assertEqual(stats["black"], 50.0)

    def test_histogram_matches_the_pixels(self):
        pixels = [0x000000, 0xFFFFFF, 0xFF0000, 0x000000]
        image = frame_diff.Image(2, 2, pixels, 4, "mixed")
        stats = frame_diff.analysis(image, bins=16)
        self.assertEqual(sum(stats["histogram"][0]), 4)
        self.assertEqual(stats["histogram"][0][0], 2)
        self.assertEqual(stats["histogram"][0][15], 2)
        self.assertEqual(stats["histogram"][1][15], 1)
        self.assertEqual(stats["histogram"][2][15], 1)

    def test_identical_frames_are_identical(self):
        pixels = gradient(8, 6)
        a = frame_diff.Image(8, 6, pixels, 4, "a")
        b = frame_diff.Image(8, 6, pixels, 4, "b")
        diff = frame_diff.compare(a, b)
        self.assertTrue(diff["identical"])
        self.assertEqual(diff["mae"], 0.0)
        self.assertEqual(diff["changed"], 0.0)

    def test_a_black_frame_compared_with_a_white_one(self):
        black = frame_diff.Image(4, 4, [0] * 16, 4, "black")
        white = frame_diff.Image(4, 4, [0xFFFFFF] * 16, 4, "white")
        diff = frame_diff.compare(black, white)
        self.assertFalse(diff["identical"])
        self.assertEqual(diff["mae"], 255.0)
        self.assertEqual(diff["changed"], 100.0)
        self.assertEqual(diff["max_delta"], 255)

    def test_a_shifted_frame_differs_and_alignment_finds_it(self):
        width, height = 12, 6
        base = [0] * (width * height)
        for y in range(2, 4):
            for x in range(3, 6):
                base[y * width + x] = 0xFFFFFF
        shifted = [0] * (width * height)
        for y in range(height):
            for x in range(width):
                if x + 1 < width:
                    shifted[y * width + x + 1] = base[y * width + x]
        a = frame_diff.Image(width, height, base, 4, "a")
        b = frame_diff.Image(width, height, shifted, 4, "b")
        diff = frame_diff.compare(a, b)
        self.assertFalse(diff["identical"])
        self.assertGreater(diff["mae"], 0)
        self.assertGreater(diff["changed"], 0)
        best, dx, dy = frame_diff.aligned_mae(a, b, radius=2, stride=1)
        self.assertEqual((dx, dy), (1, 0))
        self.assertEqual(best, 0.0)

    def test_different_sizes_do_not_compare(self):
        a = frame_diff.Image(2, 2, [0] * 4, 4, "a")
        b = frame_diff.Image(3, 2, [0] * 6, 4, "b")
        self.assertIsNone(frame_diff.compare(a, b))


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.tool = Path(__file__).resolve().with_name("frame_diff.py")

    def tearDown(self):
        self.dir.cleanup()

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, str(self.tool), *map(str, args)],
            capture_output=True,
            text=True,
        )

    def test_one_file_reports_and_exits_zero(self):
        path = self.root / "one.ppm"
        write_ppm(path, 3, 2, gradient(3, 2))
        result = self.run_tool(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("non-black", result.stdout)

    def test_identical_files_exit_zero(self):
        a = self.root / "a.ppm"
        write_ppm(a, 4, 4, gradient(4, 4))
        b = self.root / "b.ppm"
        write_ppm(b, 4, 4, gradient(4, 4))
        result = self.run_tool(a, b)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("identical", result.stdout)

    def test_different_files_exit_one_and_a_black_one_is_called_black(self):
        a = self.root / "black.ppm"
        write_ppm(a, 4, 4, [0] * 16)
        b = self.root / "white.ppm"
        write_ppm(b, 4, 4, [0xFFFFFF] * 16)
        result = self.run_tool(a, b)
        self.assertEqual(result.returncode, 1)
        self.assertIn("BLACK FRAME", result.stdout)
        self.assertIn("different", result.stdout)

    def test_a_missing_file_is_an_error(self):
        result = self.run_tool(self.root / "nope.ppm")
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
