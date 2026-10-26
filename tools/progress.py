#!/usr/bin/env python3

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from progress_scan import (
    ALIASES,
    NV_MAGIC,
    REFERENCE,
    REF_NAMES,
    ROOT,
    RUNTIME_ALIASES,
    SECTIONS,
    family,
    read_reference,
    scan_code,
    shader_groups,
)

from progress_render import (
    pct,
    render_history,
    render_html,
    render_svg,
    update_history,
)

def build_report():
    code, ref = scan_code(), read_reference()
    sections = []
    for sid, title, unit in SECTIONS:
        if sid == "shaders":
            groups = shader_groups()
            done = sum(len(g["implemented"]) for g in groups)
            total = sum(g["total"] for g in groups)
            sections.append({"id": sid, "title": title, "unit": unit, "implemented": done, "total": total, "groups": groups})
            continue
        keys = sorted({k for k in list(code) + list(ref) if k[0] == sid}, key=lambda k: (family(*k), k[1]))
        groups = []
        for key in keys:
            arms = code.get(key, {})
            r = ref.get(key)
            documented = r["ids"] if r and r["found"] and r["ids"] else None
            if sid == "ioctl" and documented:
                magic = NV_MAGIC.get(key[1])
                served = {cid: cid & 0xFF for cid in documented if (cid >> 8) & 0xFF == magic}
                impl = {cid: documented[cid] for cid, nr in served.items() if nr in arms}
                trivial = [cid for cid in impl if arms[cid & 0xFF]]
                extra = sorted(nr for nr in arms if nr not in served.values())
            elif documented:
                impl = {cid: name for cid, name in documented.items() if cid in arms}
                trivial = [cid for cid in impl if arms[cid]]
                extra = sorted(c for c in arms if c not in documented)
            else:
                impl, trivial, extra = dict.fromkeys(arms, ""), [c for c, t in arms.items() if t], []
            total = documented if documented is not None else dict.fromkeys(arms, "")
            groups.append(
                {
                    "name": key[1],
                    "family": family(*key),
                    "official": r["official"] if r else "",
                    "url": r["url"] if r else "",
                    "referenced": documented is not None,
                    "implemented": sorted(impl),
                    "constant": sorted(trivial),
                    "missing": [{"id": c, "name": n} for c, n in sorted(total.items()) if c not in impl],
                    "undocumented": extra,
                    "total": len(total),
                }
            )
        done = sum(len(g["implemented"]) for g in groups)
        total = sum(g["total"] for g in groups)
        sections.append({"id": sid, "title": title, "unit": unit, "implemented": done, "total": total, "groups": groups})
    return {"generated": datetime.date.today().isoformat(), "sections": sections}

STATUS_RANK = {"implemented": 0, "stub": 1, "missing": 2}
SHADER_VARIANTS = "maxwell (variantes)"
def find_coverage(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out += sorted(p.glob("*.coverage.json")) + sorted(p.glob("*/*.coverage.json"))
        elif p.exists():
            out.append(p)
        else:
            print(f"note: {raw} does not exist", file=sys.stderr)
    seen, unique = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique

def load_coverage(paths):
    files = []
    for path in find_coverage(paths):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"note: cannot read {path}: {e}", file=sys.stderr)
            continue
        if data.get("format") != 1:
            print(f"note: {path} is not a coverage file", file=sys.stderr)
            continue
        files.append((path, data))
    return files

def coverage_items(files):
    items = {}
    for path, data in files:
        events = {e["n"]: e for e in data.get("events", [])}
        for it in data.get("items", []):
            key = (it["section"], it["interface"], it["id"])
            ev = events.get(it.get("event"))
            entry = items.setdefault(
                key, {"count": 0, "status": "implemented", "note": "", "games": [], "first": None, "event": None}
            )
            entry["count"] += it.get("count", 0)
            if STATUS_RANK.get(it["status"], 0) >= STATUS_RANK[entry["status"]]:
                entry["status"], entry["note"] = it["status"], it.get("note", "")
            if data["game"] not in entry["games"]:
                entry["games"].append(data["game"])
            if entry["first"] is None:
                entry["first"] = {**it.get("first", {}), "game": data["game"]}
            if ev and entry["event"] is None:
                base = path.parent
                entry["event"] = {
                    "n": ev["n"],
                    "game": data["game"],
                    "seconds": ev.get("seconds"),
                    "frame": ev.get("frame"),
                    "screenshot": str(base / ev["screenshot"]) if ev.get("screenshot") else None,
                    "saves": str(base / ev["saves"]) if ev.get("saves") else None,
                    "folder": str(base / f'{data["game"]}-coverage' / str(ev["n"])),
                    "backtrace": ev.get("backtrace", []),
                    "args": ev.get("args", ""),
                }
    return items

def variant_mnemonics(variant):
    head = re.split(r"\s*\(|\s+from\s+", variant)[0].strip()
    return [m for m in head.split("/") if re.fullmatch(r"[a-z0-9_.]+", m)]

def report_name(section, iface, groups):
    if (section, iface) in groups:
        return iface
    return ALIASES.get(iface) or RUNTIME_ALIASES.get(iface) or iface

def code_implements(section, g, cid):
    if g is None:
        return None
    impl, const = set(g["implemented"]), set(g["constant"])
    if section == "ioctl":
        hits = [c for c in impl if (c & 0xFFFF) == (cid & 0xFFFF)]
        if not hits:
            return None if (cid & 0xFF) in g["undocumented"] else False
        return not all(c in const for c in hits)
    if cid in impl or cid in g["undocumented"]:
        return cid not in const
    return False

def item_status(section, iface, cid, e, g, shader_impl, variants):
    status, note = e["status"], e["note"]
    if section == "shaders":
        if iface == SHADER_VARIANTS:
            return status, note
        if cid not in shader_impl:
            return "missing", note or "no translation"
        if variants.get(cid):
            return "stub", "variantes faltando: " + ", ".join(variants[cid])
        return "implemented", ""
    now = code_implements(section, g, cid)
    if status != "implemented" and now:
        return "implemented", "implementado depois do registro"
    return status, note

def annotate_coverage(report, items):
    groups = {(s["id"], g["name"]): g for s in report["sections"] for g in s["groups"]}
    shader_groups = [g for s in report["sections"] if s["id"] == "shaders" for g in s["groups"]]
    shader_impl = {m for g in shader_groups for m in g["implemented"]}
    variants = {}
    for (section, iface, cid) in items:
        if section == "shaders" and iface == SHADER_VARIANTS:
            for m in variant_mnemonics(str(cid)):
                variants.setdefault(m, []).append(cid)

    gaps, resolved = [], []
    cells = {}
    for (section, iface, cid), e in items.items():
        name_in_report = report_name(section, iface, groups)
        g = groups.get((section, name_in_report)) if section != "shaders" else None
        status, note = item_status(section, iface, cid, e, g, shader_impl, variants)
        if e["status"] != "implemented" and status == "implemented":
            resolved.append({"section": section, "interface": iface, "id": cid, "was": e["status"], "count": e["count"]})
        kind = "used_done" if status == "implemented" else "used_missing"
        if section == "shaders" and iface != SHADER_VARIANTS:
            for sg in shader_groups:
                if cid in sg["implemented"] or any(m["id"] == cid for m in sg["missing"]):
                    cells.setdefault(("shaders", sg["name"]), {})[cid] = kind
        elif g is not None:
            if section == "ioctl":
                ids = [c for c in g["implemented"] + [m["id"] for m in g["missing"]] if (c & 0xFFFF) == (cid & 0xFFFF)]
                for c in ids:
                    cells.setdefault((section, g["name"]), {})[c] = kind
            else:
                cells.setdefault((section, g["name"]), {})[cid] = kind
        if status == "implemented":
            continue
        name = REF_NAMES.get((section, name_in_report, cid), "")
        if section == "ioctl" and not name:
            name = next(
                (n for (s2, i2, c2), n in REF_NAMES.items() if s2 == "ioctl" and i2 == iface and (c2 & 0xFFFF) == (cid & 0xFFFF)),
                "",
            )
        gaps.append(
            {
                "section": section,
                "interface": iface,
                "id": cid,
                "name": name,
                "status": status,
                "note": note,
                **{k: e[k] for k in ("count", "games", "first", "event")},
            }
        )
    for s in report["sections"]:
        for g in s["groups"]:
            used = cells.get((s["id"], g["name"]), {})
            g["used"] = sorted((c for c, k in used.items() if k == "used_done"), key=str)
            g["used_missing"] = sorted((c for c, k in used.items() if k == "used_missing"), key=str)
    gaps.sort(key=lambda x: (-x["count"], x["section"], x["interface"], str(x["id"])))
    report["coverage"] = {"used": len(items), "gaps": gaps, "resolved": resolved}

def main():
    ap = argparse.ArgumentParser(description="Progress report for kirikae: what the runtime implements of the Switch's system interface, against what is documented.")
    ap.add_argument("--out", help="output folder (default docs/progress, or docs/progress-games with --coverage)")
    ap.add_argument(
        "--coverage",
        nargs="+",
        action="extend",
        default=[],
        metavar="FILE_OR_DIR",
        help="<name>.coverage.json files written by recompiled games, or folders holding them",
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    out = Path(args.out or ROOT / ("docs/progress-games" if args.coverage else "docs/progress"))
    out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    if args.coverage:
        files = load_coverage(args.coverage)
        if not files:
            sys.exit("no coverage file found")
        report["games"] = sorted({d["game"] for _, d in files})
        annotate_coverage(report, coverage_items(files))
    (out / "progress.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "progress.svg").write_text(render_svg(report), encoding="utf-8")
    (out / "index.html").write_text(render_html(report), encoding="utf-8")
    badges = out / "badges"
    badges.mkdir(exist_ok=True)
    labels = {"services": "services", "svc": "SVCs", "ioctl": "nvdrv ioctls", "shaders": "shader ops"}
    for s in report["sections"]:
        ratio = s["implemented"] / s["total"] if s["total"] else 0
        color = "2ea043" if ratio >= 0.5 else "d29922" if ratio >= 0.3 else "db6d28"
        badge = {"schemaVersion": 1, "label": labels.get(s["id"], s["id"]), "message": f'{s["implemented"]}/{s["total"]} ({pct(s["implemented"], s["total"])})', "color": color}
        (badges / f'{s["id"]}.json').write_text(json.dumps(badge) + "\n", encoding="utf-8")
    history = update_history(out / "history.json", report)
    (out / "history.html").write_text(render_history(history), encoding="utf-8")
    for s in report["sections"]:
        print(f'{s["title"]}: {s["implemented"]}/{s["total"]} ({pct(s["implemented"], s["total"])})')
    if "coverage" in report:
        cov = report["coverage"]
        print(f'{", ".join(report["games"])}: {cov["used"]} items used, {len(cov["gaps"])} stub/missing')
    if not REFERENCE.exists():
        print(f"note: {REFERENCE.relative_to(ROOT)} is missing; totals only count handled commands")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
