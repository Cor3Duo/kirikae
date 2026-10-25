#!/usr/bin/env python3
"""Measure captured frames so a black (or wrong) frame is visible without eyes.

Reads the formats the runtime writes: binary PPM (P6, `KIRIKAE_CAPTURE`) and
8-bit PNG (the `KIRIKAE_FRAME_DUMP` / `KIRIKAE_PRESENT_DUMP` dumps). Standard
library only; no image package is needed.

    python tools/frame_diff.py frame_000061.ppm
    python tools/frame_diff.py reference.png captured.ppm
    python tools/frame_diff.py --align reference.png captured.ppm

With one file it prints the metrics of that frame. With two or more, it also
reports the mean absolute error of every file against the first one. Exit
status is 1 when a comparison finds different images, 0 otherwise.
"""

import argparse
import json
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


class ImageError(Exception):
    pass


class Image:
    """`0xRRGGBB` pixels plus the source colour channels they came from."""

    def __init__(self, width, height, pixels, channels, name):
        self.width = width
        self.height = height
        self.pixels = pixels
        self.channels = channels
        self.name = name

    def rgb(self):
        for px in self.pixels:
            yield (px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF

    def alpha(self):
        if self.channels not in (2, 4):
            return None
        return [(px >> 24) & 0xFF for px in self.pixels]


def _png_pixels(data, name):
    if not data.startswith(PNG_SIGNATURE):
        raise ImageError(f"{name}: not a PNG file")
    pos = len(PNG_SIGNATURE)
    header = None
    idat = []
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        payload = data[pos + 8:pos + 8 + length]
        if len(payload) != length:
            raise ImageError(f"{name}: truncated {ctype.decode('ascii', 'replace')} chunk")
        pos += 12 + length
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif ctype == b"IDAT":
            idat.append(payload)
        elif ctype == b"IEND":
            break
    if header is None:
        raise ImageError(f"{name}: no IHDR chunk")
    width, height, depth, colour, compression, filtering, interlace = header
    if compression != 0 or filtering != 0:
        raise ImageError(f"{name}: unsupported PNG compression or filter method")
    if interlace != 0:
        raise ImageError(f"{name}: interlaced PNG is not supported")
    channels = CHANNELS.get(colour)
    if channels is None:
        raise ImageError(f"{name}: PNG colour type {colour} is not supported")
    if depth not in (8, 16):
        raise ImageError(f"{name}: {depth}-bit PNG is not supported")
    if not idat:
        raise ImageError(f"{name}: no IDAT chunk")
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as e:
        raise ImageError(f"{name}: bad PNG data ({e})") from e
    step = channels * (depth // 8)
    stride = width * step
    if len(raw) < (stride + 1) * height:
        raise ImageError(f"{name}: PNG has {len(raw)} bytes, expected {(stride + 1) * height}")
    out = bytearray(stride * height)
    prev = bytearray(stride)
    at = 0
    for y in range(height):
        kind = raw[at]
        line = bytearray(raw[at + 1:at + 1 + stride])
        at += 1 + stride
        if kind == 0:
            pass
        elif kind == 1:
            for i in range(step, stride):
                line[i] = (line[i] + line[i - step]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                left = line[i - step] if i >= step else 0
                up = prev[i]
                upleft = prev[i - step] if i >= step else 0
                p = left + up - upleft
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - upleft)
                predict = left if pa <= pb and pa <= pc else (up if pb <= pc else upleft)
                line[i] = (line[i] + predict) & 0xFF
        else:
            raise ImageError(f"{name}: unknown PNG filter {kind} on row {y}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    pixels = []
    for i in range(0, stride * height, step):
        if channels == 1:
            g = out[i]
            pixels.append((g << 16) | (g << 8) | g)
        elif channels == 2:
            g, a = out[i], out[i + 1]
            pixels.append((a << 24) | (g << 16) | (g << 8) | g)
        elif channels == 3:
            pixels.append((out[i] << 16) | (out[i + 1] << 8) | out[i + 2])
        else:
            r, g, b, a = out[i], out[i + 1], out[i + 2], out[i + 3]
            pixels.append((a << 24) | (r << 16) | (g << 8) | b)
    return width, height, pixels, channels


def _ppm_pixels(data, name):
    if not data.startswith(b"P6") and not data.startswith(b"P3"):
        raise ImageError(f"{name}: not a PPM file (expected P6 or P3)")
    ascii_ppm = data.startswith(b"P3")
    tokens = []
    pos = 2
    while len(tokens) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while pos < len(data) and data[pos] != 0x0A:
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        if start == pos:
            raise ImageError(f"{name}: truncated PPM header")
        tokens.append(data[start:pos])
    try:
        width, height, maxval = (int(t) for t in tokens)
    except ValueError as e:
        raise ImageError(f"{name}: bad PPM header") from e
    if width <= 0 or height <= 0 or maxval <= 0 or maxval > 65535:
        raise ImageError(f"{name}: bad PPM size {width}x{height} maxval {maxval}")
    if ascii_ppm:
        values = [int(v) for v in re_split(data[pos:])]
        scale = 255.0 / maxval
        need = width * height * 3
        if len(values) < need:
            raise ImageError(f"{name}: {len(values)} PPM samples, expected {need}")
        pixels = []
        for i in range(0, need, 3):
            r, g, b = (min(255, int(v * scale + 0.5)) for v in values[i:i + 3])
            pixels.append((r << 16) | (g << 8) | b)
        return width, height, pixels, 4
    bytes_per = 1 if maxval < 256 else 2
    pos += 1
    need = width * height * 3 * bytes_per
    body = data[pos:pos + need]
    if len(body) < need:
        raise ImageError(f"{name}: {len(body)} PPM bytes, expected {need}")
    if bytes_per == 2:
        samples = struct.unpack(f">{need // 2}H", body)
        scale = 255.0 / maxval
        values = [min(255, int(v * scale + 0.5)) for v in samples]
    else:
        values = body
        if maxval != 255:
            scale = 255.0 / maxval
            values = [min(255, int(v * scale + 0.5)) for v in body]
    pixels = []
    for i in range(0, width * height * 3, 3):
        pixels.append((values[i] << 16) | (values[i + 1] << 8) | values[i + 2])
    return width, height, pixels, 4


def re_split(data):
    return [v for v in data.split() if v and not v.startswith(b"#")]


def load(path):
    data = Path(path).read_bytes()
    name = str(path)
    if data.startswith(PNG_SIGNATURE):
        width, height, pixels, channels = _png_pixels(data, name)
    else:
        width, height, pixels, channels = _ppm_pixels(data, name)
    return Image(width, height, pixels, channels, name)


def histogram_from(counter, bins=16):
    step = 256 // bins
    hist = [[0] * bins for _ in range(3)]
    for color, count in counter.items():
        hist[0][min(bins - 1, ((color >> 16) & 0xFF) // step)] += count
        hist[1][min(bins - 1, ((color >> 8) & 0xFF) // step)] += count
        hist[2][min(bins - 1, (color & 0xFF) // step)] += count
    return hist


def analysis(image, threshold=8, bins=16):
    total = image.width * image.height
    counter = Counter(image.pixels)
    sums = [0, 0, 0]
    nonblack = 0
    black = 0
    for color, count in counter.items():
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        sums[0] += r * count
        sums[1] += g * count
        sums[2] += b * count
        if max(r, g, b) > threshold:
            nonblack += count
        if r == 0 and g == 0 and b == 0:
            black += count
    mean = [s / total for s in sums]
    luma = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
    top_color, top_count = counter.most_common(1)[0]
    return {
        "width": image.width,
        "height": image.height,
        "channels": image.channels,
        "mean": mean,
        "luma": luma,
        "nonblack": 100.0 * nonblack / total,
        "black": 100.0 * black / total,
        "distinct": len(counter),
        "top_color": top_color,
        "top_share": 100.0 * top_count / total,
        "histogram": histogram_from(counter, bins),
    }


def compare(a, b):
    if (a.width, a.height) != (b.width, b.height):
        return None
    n = len(a.pixels)
    sums = [0, 0, 0]
    changed = 0
    biggest = 0
    for pa, pb in zip(a.pixels, b.pixels):
        deltas = []
        for shift in (16, 8, 0):
            d = abs(((pa >> shift) & 0xFF) - ((pb >> shift) & 0xFF))
            sums[0 if shift == 16 else 1 if shift == 8 else 2] += d
            deltas.append(d)
        m = max(deltas)
        if m:
            changed += 1
            biggest = max(biggest, m)
    return {
        "mae": sum(sums) / (3 * n),
        "mae_channels": [s / n for s in sums],
        "changed": 100.0 * changed / n,
        "max_delta": biggest,
        "identical": changed == 0,
    }


def aligned_mae(a, b, radius=8, stride=4):
    """Best MAE over small integer shifts of `b`, to separate a real change from a lag."""
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            total = 0
            count = 0
            for y in range(0, a.height, stride):
                sy = y + dy
                if not 0 <= sy < b.height:
                    continue
                for x in range(0, a.width, stride):
                    sx = x + dx
                    if not 0 <= sx < b.width:
                        continue
                    pa = a.pixels[y * a.width + x]
                    pb = b.pixels[sy * b.width + sx]
                    total += (
                        abs(((pa >> 16) & 0xFF) - ((pb >> 16) & 0xFF))
                        + abs(((pa >> 8) & 0xFF) - ((pb >> 8) & 0xFF))
                        + abs((pa & 0xFF) - (pb & 0xFF))
                    )
                    count += 1
            if count == 0:
                continue
            score = total / (3 * count)
            if best is None or score < best[0]:
                best = (score, dx, dy)
    return best


def print_analysis(image, result):
    print(f"{Path(image.name).name}: {image.width}x{image.height}, "
          f"{result['mean'][0]:.1f} {result['mean'][1]:.1f} {result['mean'][2]:.1f} mean, "
          f"luma {result['luma']:.1f}")
    print(f"  non-black {result['nonblack']:.2f}% (any channel > threshold), "
          f"exactly black {result['black']:.2f}%, {result['distinct']} distinct colours")
    top = result["top_color"]
    print(f"  dominant colour #{top:06x} covers {result['top_share']:.2f}% of the frame")
    if result["nonblack"] < 0.1:
        print("  BLACK FRAME")
    elif result["luma"] < 1.0:
        print("  nearly black: the image has structure but almost no light")
    for name, channel in zip("RGB", result["histogram"]):
        total = image.width * image.height
        cells = " ".join(f"{c * 100.0 / total:5.1f}" for c in channel)
        print(f"  {name} 0-255 in 16 bins (%): {cells}")


def print_compare(a, b, result):
    print(f"{Path(a.name).name} vs {Path(b.name).name}:")
    if result is None:
        print(f"  DIFFERENT SIZE: {a.width}x{a.height} vs {b.width}x{b.height}")
        return
    print(f"  MAE {result['mae']:.2f}/255 "
          f"(R {result['mae_channels'][0]:.2f} G {result['mae_channels'][1]:.2f} "
          f"B {result['mae_channels'][2]:.2f}), "
          f"{result['changed']:.2f}% of pixels differ, largest delta {result['max_delta']}")
    print(f"  {'identical' if result['identical'] else 'different'}")


def main():
    ap = argparse.ArgumentParser(
        description="Report whether captured frames are black and how far apart two frames are.")
    ap.add_argument("files", nargs="+", help="PPM (P6/P3) or 8-bit PNG frames")
    ap.add_argument("--threshold", type=int, default=8,
                    help="a pixel is non-black when any channel is above this (default 8)")
    ap.add_argument("--bins", type=int, default=16, help="histogram bins per channel (default 16)")
    ap.add_argument("--align", action="store_true",
                    help="also search small shifts for the closest alignment of each pair")
    ap.add_argument("--all-pairs", action="store_true",
                    help="compare every file with the first even when there are many (slow)")
    ap.add_argument("--json", action="store_true", help="print a JSON report instead")
    args = ap.parse_args()
    if args.bins < 1 or args.bins > 256:
        ap.error("--bins must be between 1 and 256")

    images = []
    errors = 0
    for path in args.files:
        try:
            images.append(load(path))
        except (OSError, ImageError) as e:
            print(f"error: {e}", file=sys.stderr)
            errors += 1
    if not images:
        return 2

    report = []
    for image in images:
        result = analysis(image, args.threshold)
        report.append({"file": image.name, "stats": result})

    pairs = []
    if len(images) > 1 and (len(images) <= 4 or args.all_pairs):
        first = images[0]
        for other in images[1:]:
            result = compare(first, other)
            entry = {"a": first.name, "b": other.name, "diff": result}
            if args.align:
                entry["align"] = aligned_mae(first, other)
            pairs.append(entry)

    if args.json:
        print(json.dumps({"frames": report, "pairs": pairs}, indent=2))
    else:
        for (image, item) in zip(images, report):
            print_analysis(image, item["stats"])
        for entry in pairs:
            print()
            print_compare(images[0], next(i for i in images if i.name == entry["b"]), entry["diff"])
            if "align" in entry and entry["align"] is not None:
                score, dx, dy = entry["align"]
                print(f"  best alignment: shift ({dx},{dy}) left/up gives MAE {score:.2f}")
        if errors:
            print(f"{errors} file(s) could not be read", file=sys.stderr)

    if any(p["diff"] is None or not p["diff"]["identical"] for p in pairs):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
