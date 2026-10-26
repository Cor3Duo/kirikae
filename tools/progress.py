#!/usr/bin/env python3

import argparse
import datetime
import json
import re
import sys
import tempfile
from pathlib import Path

from progress_scan import (
    ALIASES,
    NV_MAGIC,
    REFERENCE,
    REF_NAMES,
    ROOT,
    RUNTIME_ALIASES,
    SECTIONS,
    SHADER_FAMILIES,
    SHADER_STAGES,
    SILENT_STUBS,
    arms_in,
    family,
    is_trivial,
    read_reference,
    scan_code,
    service_registry,
    shader_groups,
)

from progress_render import (
    COUNTED_SECTIONS,
    implemented_real,
    pct,
    render_history,
    render_html,
    render_svg,
    update_history,
)

def shader_family(name):
    for fam in ("atomics", "texture", "control"):
        if name in SHADER_FAMILIES[fam].split():
            return fam
    return "alu"

def shader_breakdown(groups):
    """Per-stage and per-family views over the same mnemonic scan as the shader section."""
    names = {m["id"] for g in groups for m in g["missing"]} | {n for g in groups for n in g["implemented"]}
    handled = {n for g in groups for n in g["implemented"]}
    hinted = {n for ns in SHADER_STAGES.values() for n in ns}
    shared = names - hinted
    stages = []
    for stage, specific in SHADER_STAGES.items():
        specific = set(specific) & names
        done = (shared | specific) & handled
        stages.append(
            {
                "name": stage,
                "family": "stage",
                "implemented": len(done),
                "total": len(names),
                "specific": [len(specific & handled), len(specific)],
            }
        )
    families = []
    for fam in ("texture", "alu", "control", "atomics"):
        members = {n for n in names if shader_family(n) == fam}
        done = members & handled
        families.append(
            {
                "name": fam,
                "family": "family",
                "implemented": len(done),
                "total": len(members),
                "missing": [{"id": n, "name": ""} for n in sorted(members - handled)],
            }
        )
    return stages, families

def build_report():
    code, ref = scan_code(), read_reference()
    sections, shader = [], []
    for sid, title, unit in SECTIONS:
        if sid == "shaders":
            shader = shader_groups()
            done = sum(len(g["implemented"]) for g in shader)
            total = sum(g["total"] for g in shader)
            sections.append({"id": sid, "title": title, "unit": unit, "implemented": done, "total": total, "groups": shader})
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
                impl, trivial, extra = {}, [], sorted(arms)
            total = documented if documented is not None else {}
            silent = key[1].split("/")[0].split(":")[0] in SILENT_STUBS
            if silent:
                trivial = list(impl)
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
                    "silent": silent,
                }
            )
        done = sum(len(g["implemented"]) for g in groups)
        total = sum(g["total"] for g in groups)
        silent = sum(len(g["implemented"]) for g in groups if g.get("silent"))
        sections.append(
            {
                "id": sid,
                "title": title,
                "unit": unit,
                "implemented": done,
                "silent": silent,
                "total": total,
                "groups": groups,
            }
        )
    for section in sections:
        section["real"] = implemented_real(section)
    counted = [s for s in sections if s["id"] in COUNTED_SECTIONS]
    report = {
        "generated": datetime.date.today().isoformat(),
        "sections": sections,
        "totals": {
            "implemented": sum(s["implemented"] for s in counted),
            "silent": sum(s.get("silent", 0) for s in counted),
            "real": sum(s["real"] for s in counted),
            "total": sum(s["total"] for s in counted),
        },
    }
    report["shader_stages"], report["shader_families"] = shader_breakdown(shader)
    return report

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

def self_test():
    """Synthetic sources for every parser, plus a determinism check of the real report."""
    checks, failures = 0, []

    def check(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(what)

    def synthetic(services):
        """Scan a throwaway `crates/rt/src/services` tree; no horizon.rs, so services only."""
        with tempfile.TemporaryDirectory(prefix="kirikae-progress-") as tmp:
            root = Path(tmp)
            for rel, text in services.items():
                path = root / "services" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return scan_code(root / "services", root / "horizon.rs")

    svc = """match req.cmd {
        0 => Reply::ok(),
        1 | 2 => self.real(),
        ( "svc/IFoo", 3 ) => Reply::ok().u32(1),
        0x10..=0x12 => self.real(),
        _ => crate::services::stub::plain_ok(),
    }"""
    found = arms_in(svc, r"match req\.cmd\s*\{", {})
    check(found.get(None) == {0: True, 1: False, 2: False, 16: False, 17: False, 18: False}, "service arms")
    check(found.get("svc/IFoo") == {3: True}, "service arms with an interface name")
    check(is_trivial("Reply::ok().u32(1)") and not is_trivial("self.real()"), "trivial bodies")

    groups = shader_groups()
    names = {m["id"] for g in groups for m in g["missing"]} | {n for g in groups for n in g["implemented"]}
    check(names, "shader mnemonics")
    for fam in SHADER_FAMILIES:
        check({n for n in names if shader_family(n) == fam}, f"shader family {fam}")
    seen = [shader_family(n) for n in names]
    check(seen.count("alu") + seen.count("atomics") + seen.count("texture") + seen.count("control") == len(names), "shader families partition the table")
    for stage, hint in SHADER_STAGES.items():
        check(set(hint) <= names, f"stage hint {stage}")
    stages, families = shader_breakdown(groups)
    check(sum(f["total"] for f in families) == len(names), "family totals")
    check(all(s["total"] == len(names) for s in stages), "stage totals")

    registry = """pub fn host_object(name: &str) -> Option<Arc<dyn Service>> {
        let service: Arc<dyn Service> = match name {
            "split" | "split:u" => Arc::new(split::Split),
            "tabled" => Arc::new(tabled::Tabled { name: "tabled" }),
            _ => return None,
        };
        Some(service)
    }"""
    check(service_registry(registry).get("Split") == ["split", "split:u"], "service registry names")
    check(service_registry(registry).get("Tabled") == ["tabled"], "service registry reads the struct")
    check(service_registry("pub fn open(name: &str) { }") == {}, "service registry tolerates a missing table")

    split_call = """impl Service for Split {
    fn name(&self) -> String { "split".into() }
    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok(),
            1 | 2 => self.real(),
            _ => unimplemented("split", c, req),
        }
    }
}"""
    tabled_call = """impl Service for Tabled {
    fn name(&self) -> String { self.name.into() }
    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented(self.name, c, req),
        }
    }
}"""
    tabled_command = """impl Tabled {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            5 => Reply::ok(),
            6 => self.real(),
            _ => None,
        }
    }
}"""
    one_file = synthetic({"mod.rs": registry, "split.rs": split_call, "tabled.rs": tabled_call + tabled_command})
    split_module = synthetic(
        {
            "mod.rs": registry,
            "split/mod.rs": split_call.replace("match req.cmd", "match table::dispatch(req.cmd)"),
            "split/table.rs": "pub(super) fn dispatch(cmd: u32) -> Option<Reply> {\n"
            "    match cmd {\n"
            "        0 => Reply::ok(),\n"
            "        1 | 2 => self.real(),\n"
            "        _ => None,\n"
            "    }\n"
            "}",
            "tabled.rs": tabled_call,
            "tabled/command.rs": tabled_command,
        }
    )
    want = {("services", "split"): {0: True, 1: False, 2: False}, ("services", "tabled"): {5: True, 6: False}}
    check(one_file == want, "single-file service arms")
    check(split_module == want, "a split service counts like a single file")

    ioctl_tree = {
        "mod.rs": registry.replace('"split" | "split:u" => Arc::new(split::Split),', '"nvdrv" => Arc::new(nvdrv::Nvdrv),')
        .replace('"tabled" => Arc::new(tabled::Tabled { name: "tabled" }),', ""),
        "nvdrv.rs": """impl Service for Nvdrv {
    fn name(&self) -> String { self.name.into() }
    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok(),
            _ => unimplemented(self.name, c, req),
        }
    }
}""",
        "nvdrv/ioctls.rs": """fn nvmap(env: &Env, nr: u32, a: &mut Args) -> Outcome {
    match nr {
        1 => self.real(),
        2 | 3 => Outcome::Ok,
        _ => Outcome::Unimplemented,
    }
}""",
    }
    ioctls = synthetic(ioctl_tree)
    check(set(ioctls.get(("ioctl", "/dev/nvmap")) or {}) == {1, 2, 3}, "ioctl arms in a split nvdrv folder")
    check(ioctls.get(("services", "nvdrv")) == {0: True}, "split nvdrv service arms")

    folder_tree = synthetic(
        {
            "mod.rs": """pub fn host_object(name: &str) -> Option<Arc<dyn Service>> {
        let service: Arc<dyn Service> = match name {
            "folder:u" => Arc::new(folder::Folder::new("folder:u")),
            "folder:s" => Arc::new(folder::Folder::new("folder:s")),
            _ => return None,
        };
        Some(service)
    }""",
            "folder/mod.rs": """impl Service for Folder {
    fn name(&self) -> String { self.name.into() }
    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match self.command(req) {
            Some(r) => r,
            None => unimplemented(self.name, c, req),
        }
    }
}""",
            "folder/command.rs": """impl Folder {
    fn command(&self, req: &Request) -> Option<Reply> {
        match req.cmd {
            0 => Reply::ok(),
            1 | 2 => self.real(),
            _ => None,
        }
    }
}""",
        }
    )
    arms_want = {0: True, 1: False, 2: False}
    check(
        folder_tree == {("services", "folder:u"): arms_want, ("services", "folder:s"): arms_want},
        "a folder service registered under two names counts for both",
    )

    errno_tree = synthetic(
        {
            "code.rs": """impl Service for Code {
    fn name(&self) -> String { "code".into() }
    fn call(&self, c: &mut Cpu, req: &Request) -> Reply {
        match req.cmd {
            0 => Reply::ok().u32(self.host_errno(req.u32(0))),
            _ => unimplemented("code", c, req),
        }
    }
}

impl Code {
    fn host_errno(&self, code: u32) -> u32 {
        match code {
            10004 => 4,
            10009 => 9,
            _ => 0,
        }
    }
}""",
        }
    )
    check(sorted(errno_tree.get(("services", "code"), {})) == [0], "a host errno table is not a command dispatch")

    with tempfile.TemporaryDirectory(prefix="kirikae-progress-") as tmp:
        reference = Path(tmp) / "horizon.txt"
        reference.write_text(
            "[service folder:u]\n"
            "# nn::test::IFolder — https://switchbrew.org/wiki/NS_services\n"
            "# Retrieved 2026-10-25.\n"
            "0 Open\n"
            "1 Read\n"
            "[service folder:s]\n"
            "# not found in the sources\n",
            encoding="utf-8",
        )
        parsed = read_reference(reference)
    check(
        parsed[("services", "folder:u")]["ids"] == {0: "Open", 1: "Read"},
        "a reference block is read back with its ids",
    )
    check(
        parsed[("services", "folder:u")]["url"] == "https://switchbrew.org/wiki/NS_services"
        and "https://" not in parsed[("services", "folder:u")]["official"],
        "a reference block keeps the url out of the official name",
    )
    check(parsed[("services", "folder:s")]["found"] is False, "a block with no source stays uncounted")
    check(
        {cid for cid in parsed[("services", "folder:u")]["ids"] if cid in folder_tree[("services", "folder:s")]} == {0, 1},
        "the reference block and the folder dispatch meet on the same ids",
    )

    report = build_report()
    service_groups = {g["name"]: g for g in report["sections"][0]["groups"]}
    for name in ("omm", "omm/ICradleFirmwareUpdater", "pdm:qry", "hwopus", "nfp:user"):
        check(service_groups.get(name, {}).get("referenced", False), f"{name} is counted against its reference")
    for name in ("bsd:u", "ns:am2"):
        check(name not in service_groups, f"{name} is not served, so it is not counted")
    bsd = service_groups.get("bsd:u") or {}
    check(10004 not in bsd.get("undocumented", []), "winsock errno codes stay out of the bsd command list")
    check(
        service_groups.get("nfp:user", {}).get("silent") and service_groups.get("hwopus", {}).get("silent"),
        "silent stubs stay flagged after gaining a reference",
    )
    check(report.get("shader_stages") and report.get("shader_families"), "shader breakdown in the report")

    with tempfile.TemporaryDirectory(prefix="kirikae-progress-") as tmp:
        stored = json.loads((write_outputs(Path(tmp) / "progress", report) / "progress.json").read_text(encoding="utf-8"))
    check([s["id"] for s in stored["sections"]] == [s["id"] for s in report["sections"]], "json keeps every section")
    check(summary_lines(stored) == summary_lines(report), "console summary agrees with progress.json")
    check(
        stored["totals"]["real"] == stored["totals"]["implemented"] - stored["totals"]["silent"],
        "totals subtract the silent stubs",
    )
    check(
        stored["totals"]["real"]
        == sum(s["real"] for s in stored["sections"] if s["id"] in COUNTED_SECTIONS),
        "totals add up the counted sections",
    )
    for section in stored["sections"]:
        check(section["real"] == section["implemented"] - section.get("silent", 0), f'{section["id"]} reports both counts')
    services_line = summary_lines(report)[0]
    services = report["sections"][0]
    check(
        str(services["real"]) in services_line
        and (services.get("silent", 0) == 0 or "handled" in services_line),
        "console prints counted services and the handled count when silent stubs exist",
    )

    first = json.dumps(report, sort_keys=True)
    second = json.dumps(build_report(), sort_keys=True)
    check(first == second, "report is deterministic")
    check(render_svg(build_report()) == render_svg(report), "svg is deterministic")

    if failures:
        for f in failures:
            print(f"self-test: {f}", file=sys.stderr)
        print(f"self-test: {len(failures)} of {checks} checks failed", file=sys.stderr)
        return 1
    print(f"self-test: {checks} checks passed")
    return 0

def summary_lines(report):
    """The console summary, read from the same fields that progress.json stores.

    A silent stub is handled (the service answers it) but not counted as implemented; the line
    says both numbers instead of printing one and leaving the other to the JSON reader.
    """
    lines = []
    for section in report["sections"]:
        done, total = section["real"], section["total"]
        if section.get("silent"):
            lines.append(
                f'{section["title"]}: {done}/{total} counted ({pct(done, total)}), '
                f'{section["implemented"]} handled, {section["silent"]} silent stubs excluded'
            )
        else:
            lines.append(f'{section["title"]}: {done}/{total} ({pct(done, total)})')
    for stage in report.get("shader_stages", []):
        lines.append(f'shaders {stage["name"]}: {stage["implemented"]}/{stage["total"]} ({pct(stage["implemented"], stage["total"])})')
    totals = report.get("totals")
    if totals:
        lines.append(
            f'total: {totals["real"]}/{totals["total"]} counted ({pct(totals["real"], totals["total"])}), '
            f'{totals["implemented"]} handled, {totals["silent"]} silent stubs'
        )
    if "coverage" in report:
        cov = report["coverage"]
        lines.append(f'{", ".join(report["games"])}: {cov["used"]} items used, {len(cov["gaps"])} stub/missing')
    return lines


def write_outputs(out, report):
    """progress.json, the treemap, the HTML dashboard, the badges and the daily history."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "progress.svg").write_text(render_svg(report), encoding="utf-8")
    (out / "index.html").write_text(render_html(report), encoding="utf-8")
    badges = out / "badges"
    badges.mkdir(exist_ok=True)
    labels = {"services": "services", "svc": "SVCs", "ioctl": "nvdrv ioctls", "shaders": "shader ops"}
    for s in report["sections"]:
        done = implemented_real(s)
        ratio = done / s["total"] if s["total"] else 0
        color = "2ea043" if ratio >= 0.5 else "d29922" if ratio >= 0.3 else "db6d28"
        badge = {"schemaVersion": 1, "label": labels.get(s["id"], s["id"]), "message": f'{done}/{s["total"]} ({pct(done, s["total"])})', "color": color}
        (badges / f'{s["id"]}.json').write_text(json.dumps(badge) + "\n", encoding="utf-8")
    history = update_history(out / "history.json", report)
    (out / "history.html").write_text(render_history(history), encoding="utf-8")
    return out


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
    ap.add_argument("--self-test", action="store_true", help="check the parsers against synthetic sources and exit")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    if args.self_test:
        sys.exit(self_test())
    out = Path(args.out or ROOT / ("docs/progress-games" if args.coverage else "docs/progress"))
    report = build_report()
    if args.coverage:
        files = load_coverage(args.coverage)
        if not files:
            sys.exit("no coverage file found")
        report["games"] = sorted({d["game"] for _, d in files})
        annotate_coverage(report, coverage_items(files))
    write_outputs(out, report)
    for line in summary_lines(report):
        print(line)
    if not REFERENCE.exists():
        print(f"note: {REFERENCE.relative_to(ROOT)} is missing; totals only count handled commands")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
