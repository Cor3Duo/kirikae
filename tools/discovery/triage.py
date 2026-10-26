#!/usr/bin/env python3
"""Turn run artifacts (coverage JSON + logs) into deduplicated work items.

Mechanical stage of the discovery loop: no AI, no network. It normalizes what a game
used and what failed, fingerprints it, routes each item to the source file that most
likely implements it, and merges everything into a persistent items.jsonl.

    python tools/discovery/triage.py --game title-0100000000000001 \
        --coverage path/to/game.coverage.json --log path/to/stderr.log

Outputs, under $KIRIKAE_DISCOVERY_DIR (default: %LOCALAPPDATA%/kirikae-discovery):
    <game>/items.jsonl    one JSON object per work item (merged across runs)
    <game>/summary.md     human-readable delta and priority table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SERVICE_FILES = {
    "am": "crates/rt/src/services/am.rs",
    "acc": "crates/rt/src/services/acc.rs",
    "aoc": "crates/rt/src/services/aoc.rs",
    "friend": "crates/rt/src/services/friend.rs",
    "prepo": "crates/rt/src/services/prepo.rs",
    "fs": "crates/rt/src/services/fs.rs",
    "hid": "crates/rt/src/services/hid.rs",
    "vi": "crates/rt/src/services/vi.rs",
    "nvdrv": "crates/rt/src/services/nvdrv.rs",
    "pctl": "crates/rt/src/services/pctl.rs",
    "time": "crates/rt/src/services/time.rs",
    "set": "crates/rt/src/services/set.rs",
    "pl": "crates/rt/src/services/pl.rs",
    "nfp": "crates/rt/src/services/nfp.rs",
    "ectx": "crates/rt/src/services/ectx.rs",
    "hwopus": "crates/rt/src/services/hwopus.rs",
    "audren": "crates/rt/src/services/audio/mod.rs",
    "audout": "crates/rt/src/services/audio/mod.rs",
    "sm": "crates/rt/src/services/sm.rs",
    "mm": "crates/rt/src/services/mm.rs",
    "apm": "crates/rt/src/services/apm.rs",
    "lm": "crates/rt/src/services/lm.rs",
    "ldr": "crates/rt/src/services/ro.rs",
}

SECTION_FILES = {
    "svc": ["crates/rt/src/horizon.rs"],
    "ioctl": ["crates/rt/src/services/nvdrv.rs"],
    "shader": ["crates/maxwell/src/translate.rs", "crates/maxwell/src/decode.rs"],
}

CRASH_PATTERNS = [
    (re.compile(r"svcBreak\(reason=(?P<reason>[^,]+).*?at (?P<where>\S+)"), "svcBreak"),
    (re.compile(r"terminate result to (?P<reason>0x[0-9a-fA-F]+)"), "terminate"),
    (re.compile(r"unimplemented (?:SVC|service|command|ioctl)[^:]*:?\s*(?P<where>.*)"), "unimplemented"),
    (re.compile(r"missing (?P<where>.+)"), "missing"),
    (re.compile(r"deadlock|blocked_forever", re.I), "deadlock"),
]


def discovery_root() -> Path:
    env = os.environ.get("KIRIKAE_DISCOVERY_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share"
    return Path(base) / "kirikae-discovery"


def fingerprint(section: str, interface: str, key: object, kind: str) -> str:
    raw = f"{kind}|{section}|{interface}|{key}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def files_for(section: str, interface: str) -> list[str]:
    if section == "services":
        family = re.split(r"[/:]", interface, maxsplit=1)[0].lower()
        if family in SERVICE_FILES:
            return [SERVICE_FILES[family]]
        return [f"crates/rt/src/services/{family}.rs"]
    return SECTION_FILES.get(section, [])


def priority_for(section: str, status: str, count: int, game_count: int = 1) -> str:
    if section == "crash":
        return "P0"
    if status in ("missing", "stub") and game_count >= 5:
        return "P0"
    if status in ("missing", "stub"):
        return "P1"
    return "P2"


def item_id(section: str, interface: str, key: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", interface).strip("-")[:24]
    return f"{section}:{slug}:{key}"


def load_items(path: Path) -> dict[str, dict]:
    items: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                it = json.loads(line)
                items[it["fingerprint"]] = it
    return items


def implemented_keys(path: Path) -> set[tuple[str, str, object]]:
    """Commands the game used and the runtime reports as implemented in this run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (it.get("section", "?"), it.get("interface", "?"), it.get("id"))
        for it in data.get("items", [])
        if it.get("status") == "implemented"
    }


def from_coverage(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for it in data.get("items", []):
        status = it.get("status", "missing")
        if status == "implemented":
            continue
        section = it.get("section", "?")
        interface = it.get("interface", "?")
        key = it.get("id")
        out.append(
            {
                "kind": "service_command" if section == "services" else section,
                "section": section,
                "interface": interface,
                "key": key,
                "status": status,
                "count": it.get("count", 1),
                "first": it.get("first", {}),
                "note": it.get("note"),
                "unit": data.get("game"),
                "game": data.get("game"),
                "title_id": data.get("title_id"),
                "evidence": [f"{path}#items/{interface}/{key}"],
            }
        )
    return out


def from_logs(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for n, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("ipc:"):
                continue
            for rx, kind in CRASH_PATTERNS:
                m = rx.search(stripped)
                if not m:
                    continue
                where = (m.groupdict().get("where") or "").strip()
                reason = (m.groupdict().get("reason") or "").strip()
                key = f"{kind}|{reason}|{where[:80]}"
                out.append(
                    {
                        "kind": "log_event",
                        "section": "crash" if kind in ("svcBreak", "deadlock", "abort", "terminate") else "log",
                        "interface": kind,
                        "key": key,
                        "status": "missing",
                        "count": 1,
                        "first": {},
                        "note": stripped[:240],
                        "game": None,
                        "evidence": [f"{path}#L{n}"],
                    }
                )
                break
    return out


def merge(items: dict[str, dict], fresh: list[dict], run_id: str) -> tuple[dict[str, dict], int]:
    added = 0
    for it in fresh:
        fp = fingerprint(it["section"], it["interface"], it["key"], it["kind"])
        prev = items.get(fp)
        if prev is None:
            added += 1
            it.update(
                {
                    "id": item_id(it["section"], it["interface"], it["key"]),
                    "fingerprint": fp,
                    "priority": priority_for(it["section"], it["status"], it["count"], 1),
                    "games": [it["game"]] if it.get("game") else [],
                    "game_count": 1 if it.get("game") else 0,
                    "files_hint": files_for(it["section"], it["interface"]),
                    "state": "new",
                    "attempts": 0,
                    "runs": [run_id],
                    "first_seen": int(time.time()),
                    "last_seen": int(time.time()),
                }
            )
            items[fp] = it
        else:
            prev["count"] = prev.get("count", 0) + it.get("count", 1)
            prev["last_seen"] = int(time.time())
            prev["status"] = it["status"]
            prev["note"] = it.get("note") or prev.get("note")
            prev["files_hint"] = files_for(prev["section"], prev["interface"])
            if run_id not in prev.setdefault("runs", []):
                prev["runs"].append(run_id)
            game = it.get("game")
            if game and game not in prev.setdefault("games", []):
                prev["games"].append(game)
            prev["game_count"] = len(prev.get("games", []))
            elevated = priority_for(prev["section"], prev["status"], prev["count"], prev["game_count"])
            if {elevated: 0, "P0": 1, "P1": 2, "P2": 3}[elevated] < {elevated: 0, "P0": 1, "P1": 2, "P2": 3}[prev["priority"]]:
                prev["priority"] = elevated
            for ev in it.get("evidence", []):
                if ev not in prev.setdefault("evidence", []):
                    prev["evidence"].append(ev)
    return items, added


def summary(items: dict[str, dict], added: int, run_id: str, resolved: int = 0) -> str:
    rows = sorted(items.values(), key=lambda i: (i["priority"], -i.get("game_count", 0), -i.get("count", 0)))
    lines = [
        f"# Discovery summary — run `{run_id}`",
        "",
        f"- work items: **{len(items)}** ({added} new, {resolved} verified this run)",
        f"- by priority: " + ", ".join(f"{p}: {sum(1 for i in rows if i['priority'] == p)}" for p in ("P0", "P1", "P2")),
        "",
        "| prio | games | count | section | item | state | file |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for i in rows[:60]:
        key = str(i["key"]).replace("|", "\\|")[:70]
        files = ", ".join(f"`{f}`" for f in i.get("files_hint", [])[:2]) or "—"
        lines.append(
            f"| {i['priority']} | {i.get('game_count', 0)} | {i.get('count', 0)} | {i['section']} "
            f"| {i['interface']} `{key}` | {i['state']} | {files} |"
        )
    if len(rows) > 60:
        lines.append(f"\n… {len(rows) - 60} more in items.jsonl")
    lines.append("")
    lines.append("Next: pick P0 items, reproduce with the same scenario, and create one branch per item.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True)
    ap.add_argument("--coverage", type=Path, action="append", default=[])
    ap.add_argument("--log", type=Path, action="append", default=[])
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out = args.out or discovery_root() / args.game
    out.mkdir(parents=True, exist_ok=True)
    db = out / "items.jsonl"

    fresh: list[dict] = []
    implemented: set[tuple[str, str, object]] = set()
    for path in args.coverage:
        fresh += from_coverage(path)
        implemented |= implemented_keys(path)
    fresh += from_logs(args.log)

    items = load_items(db)
    items, added = merge(items, fresh, args.run_id)
    resolved = 0
    for it in items.values():
        key = (it["section"], it["interface"], it["key"])
        if key in implemented and it["state"] != "verified":
            it["state"] = "verified"
            it["verified_by"] = args.run_id
            it["note"] = (it.get("note") or "") + f"  [verified by run {args.run_id}]"
            resolved += 1

    for it in items.values():
        it["files_hint"] = files_for(it["section"], it["interface"])

    with db.open("w", encoding="utf-8") as f:
        for it in sorted(items.values(), key=lambda i: i["fingerprint"]):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    text = summary(items, added, args.run_id, resolved)
    (out / "summary.md").write_text(text, encoding="utf-8")

    print(text)
    print(f"\nitems: {db}")
    print(f"summary: {out / 'summary.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
