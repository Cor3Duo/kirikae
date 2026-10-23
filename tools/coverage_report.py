#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress

def main():
    ap = argparse.ArgumentParser(description='What a played game used that kirikae does not implement yet.')
    ap.add_argument("files", nargs="+", help="coverage files, or folders holding them")
    ap.add_argument("--all", action="store_true", help="also list what is implemented")
    ap.add_argument("--details", type=int, default=5, metavar="N", help="backtrace and arguments of the N most used gaps (default 5)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    files = progress.load_coverage(args.files)
    if not files:
        sys.exit("no coverage file found")
    report = progress.build_report()
    items = progress.coverage_items(files)
    progress.annotate_coverage(report, items)
    cov = report["coverage"]

    for path, data in files:
        runs = data.get("runs", [])
        played = sum(r.get("seconds", 0) for r in runs)
        frames = sum(r.get("frames", 0) for r in runs)
        print(f'{data["game"]} ({data.get("title_id", "?")}): {len(runs)} run(s), {progress.fmt_time(played)} played, {frames} frames')
        print(f"  {path}")
    used = {}
    for (section, _, _) in items:
        used[section] = used.get(section, 0) + 1
    print("used: " + ", ".join(f"{n} {s}" for s, n in sorted(used.items())))
    print()

    gaps = cov["gaps"]
    if not gaps:
        print("Nothing missing: everything the game used is implemented.")
    else:
        print(f"Missing or stub ({len(gaps)}), most used first:")
        print(f'  {"count":>8}  {"status":<8} item')
        for g in gaps:
            name = f' {g["name"]}' if g["name"] else ""
            note = f' ({g["note"]})' if g["note"] else ""
            first = g["first"] or {}
            ev = g["event"]
            where = f'first use: {first.get("game", "")} {progress.fmt_time(first.get("seconds"))}, frame {first.get("frame", "?")}'
            print(f'  {g["count"]:>8}  {g["status"]:<8} {g["interface"]} {progress.fmt_id(g["section"], g["id"])}{name}{note}')
            print(f"            {where}" + (f'  event #{ev["n"]}' if ev else ""))
            if ev:
                if ev["screenshot"]:
                    print(f'            image:  {ev["screenshot"]}')
                if ev["saves"]:
                    print(f'            saves:  {ev["saves"]}')
                elif not ev["screenshot"]:
                    print(f'            folder: {ev["folder"]}')
    if cov["resolved"]:
        print()
        print("Implemented after being recorded as stub/missing:")
        for r in cov["resolved"]:
            print(f'  {r["interface"]} {progress.fmt_id(r["section"], r["id"])} (was {r["was"]}, {r["count"]} times)')

    detailed = [g for g in gaps if g["event"] and (g["event"]["backtrace"] or g["event"]["args"])][: args.details]
    if detailed:
        print()
        print("Details of the most used:")
        for g in detailed:
            ev = g["event"]
            print(f'  #{ev["n"]} {g["interface"]} {progress.fmt_id(g["section"], g["id"])}')
            if ev["args"]:
                print(f'      args: {ev["args"]}')
            for f in ev["backtrace"][:8]:
                print(f"      {f}")

    if args.all:
        print()
        print("Implemented and used:")
        for (section, iface, cid), e in sorted(items.items(), key=lambda kv: -kv[1]["count"]):
            if not any(g["section"] == section and g["interface"] == iface and g["id"] == cid for g in gaps):
                print(f'  {e["count"]:>8}  {iface} {progress.fmt_id(section, cid)}')

if __name__ == "__main__":
    main()
