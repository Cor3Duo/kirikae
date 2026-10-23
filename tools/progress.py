#!/usr/bin/env python3

import argparse
import datetime
import os
import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "crates/rt/src/services"
HORIZON = ROOT / "crates/rt/src/horizon.rs"
REFERENCE = ROOT / "tools/reference/horizon.txt"
MAXWELL_OPCODES = ROOT / "crates/maxwell/src/opcodes.rs"
MAXWELL_TRANSLATE = ROOT / "crates/maxwell/src/translate.rs"

SHADER_CLASSES = {
    "floating point": "fadd fadd32i fmul fmul32i ffma ffma32i fmnmx fset fsetp fcmp fchk fswzadd mufu rro dadd dmul dfma dmnmx dset dsetp",
    "half precision": "hadd2 hadd2_32i hmul2 hmul2_32i hfma2 hfma2_32i hset2 hsetp2",
    "integer": "iadd iadd32i iadd3 iscadd iscadd32i imad imad32i imadsp imul imul32i imnmx icmp iset isetp xmad lop lop32i lop3 shl shr shf lea bfe bfi flo popc sel prmt vabsdiff vmad vmnmx vset vsetp vadd vshl vshr",
    "conversion": "f2f f2i i2f i2i",
    "registers and predicates": "mov mov32i s2r p2r r2p psetp pset csetp cset vote shfl r2b b2r",
    "memory": "ld ldc ldg lds ldl st stg sts stl atom atoms atomcas red cctl cctll ald ast ipa out al2p isberd pixld",
    "textures and surfaces": "tex texs tld tlds tld4 tld4s txq txd txa tmml tmmls suld sust sured suatom",
    "control flow": "bra brx ssy sync pbk brk cont pcnt exit kil bar membar depbar nop cal jcal ret jmp jmx pret sam rtt bpt preexit ide ret",
}

NV_DEVICES = {
    "nvmap": "/dev/nvmap",
    "as_gpu": "/dev/nvhost-as-gpu",
    "ctrl": "/dev/nvhost-ctrl",
    "ctrl_gpu": "/dev/nvhost-ctrl-gpu",
    "channel": "/dev/nvhost-gpu",
    "engine": ("/dev/nvhost-nvdec", "/dev/nvhost-vic"),
}
NV_MAGIC = {
    "/dev/nvmap": 0x01,
    "/dev/nvhost-as-gpu": 0x41,
    "/dev/nvhost-ctrl": 0x00,
    "/dev/nvhost-ctrl-gpu": 0x47,
    "/dev/nvhost-gpu": 0x48,
    "/dev/nvhost-nvdec": 0x00,
    "/dev/nvhost-vic": 0x00,
}
ALIASES = {"fs/IStorage(RomFS)": "fs/IStorage", "fs/IFileSystem(RomFS)": "fs/IFileSystem", "fs/IFile(RomFS)": "fs/IFile"}
NV_SHORT = {
    "/dev/nvmap": "nvmap",
    "/dev/nvhost-as-gpu": "as-gpu",
    "/dev/nvhost-ctrl": "ctrl",
    "/dev/nvhost-ctrl-gpu": "ctrl-gpu",
    "/dev/nvhost-gpu": "channel",
    "/dev/nvhost-nvdec": "nvdec",
    "/dev/nvhost-vic": "vic",
}
FAMILY = {"appletOE": "am", "fsp-srv": "fs"}
RUNTIME_ALIASES = {}

def strip_comments(src):
    out, i, n = [], 0, len(src)
    char_lit = re.compile(r"b?'(\\.|\\u\{[0-9a-fA-F]+\}|[^\\'\n])'")
    while i < n:
        c = src[i]
        if c == '"' or (c == "b" and src.startswith('b"', i)):
            j = i + (2 if c == "b" else 1)
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
        elif src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif c == "'" or (c == "b" and src.startswith("b'", i)):
            m = char_lit.match(src, i)
            if m:
                out.append(m.group(0))
                i = m.end()
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)

def skip_string(src, i):
    if src[i] == '"':
        j = i + 1
        while src[j] != '"':
            j += 2 if src[j] == "\\" else 1
        return j + 1
    m = re.compile(r"'(\\.|[^\\'\n])'").match(src, i)
    return m.end() if m else i

def matching(src, i):
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack, j = [], i
    while j < len(src):
        c = src[j]
        if c in "\"'":
            k = skip_string(src, j)
            if k != j:
                j = k
                continue
        if c in pairs:
            stack.append(pairs[c])
        elif c in "})]":
            stack.pop()
            if not stack:
                return j
        j += 1
    raise ValueError("unbalanced brackets")

def match_arms(src, brace):
    arms, i, end = [], brace + 1, matching(src, brace)
    while True:
        while i < end and src[i] in " \t\r\n,":
            i += 1
        if i >= end:
            return arms
        j, depth = i, 0
        while not (depth == 0 and src.startswith("=>", j)):
            if src[j] in "\"'":
                k = skip_string(src, j)
                if k != j:
                    j = k
                    continue
            depth += src[j] in "([{"
            depth -= src[j] in ")]}"
            j += 1
        pattern = src[i:j].strip()
        j += 2
        while src[j] in " \t\r\n":
            j += 1
        if src[j] == "{":
            k = matching(src, j)
            body, i = src[j : k + 1], k + 1
        else:
            k, depth = j, 0
            while k < end and not (depth == 0 and src[k] == ","):
                if src[k] in "\"'":
                    m = skip_string(src, k)
                    if m != k:
                        k = m
                        continue
                depth += src[k] in "([{"
                depth -= src[k] in ")]}"
                k += 1
            body, i = src[j:k], k + 1
        arms.append((pattern, body.strip()))

def consts(src):
    return {m.group(1): int(m.group(2).replace("_", ""), 0) for m in re.finditer(r"const (\w+): \w+ = (0x[0-9a-fA-F_]+|\d[\d_]*);", src)}

def pattern_ids(pattern, constants):
    pattern = re.split(r"\s+if\s+", pattern)[0]
    out = []
    for alt in split_top(pattern, "|"):
        alt = alt.strip()
        tup = re.fullmatch(r'\(\s*"([^"]+)"\s*,\s*(.+?)\s*\)', alt)
        iface = None
        if tup:
            iface, alt = tup.group(1), tup.group(2)
        rng = re.fullmatch(r"(\w+)\s*\.\.=\s*(\w+)", alt)
        if rng:
            lo, hi = (value(x, constants) for x in rng.groups())
            if lo is not None and hi is not None:
                out += [(iface, v) for v in range(lo, hi + 1)]
            continue
        v = value(alt, constants)
        if v is not None:
            out.append((iface, v))
    return out

def split_top(s, sep):
    parts, depth, cur = [], 0, ""
    for c in s:
        depth += c in "(["
        depth -= c in ")]"
        if c == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    return parts + [cur]

def value(tok, constants):
    tok = tok.strip()
    if re.fullmatch(r"0x[0-9a-fA-F_]+|\d[\d_]*", tok):
        return int(tok.replace("_", ""), 0)
    return constants.get(tok)

TRIVIAL = [
    re.compile(r"Reply::ok\(\)(\.(u8|u32|u64)\((0x[0-9a-fA-F_]+|\d[\d_]*)\))*"),
    re.compile(r"(crate::services::)?(stub::)?plain_ok\(\)"),
    re.compile(r"ok\(c\)"),
    re.compile(r"NV_SUCCESS"),
    re.compile(r"\{\s*\}"),
]

def is_trivial(body):
    return any(t.fullmatch(body.strip()) for t in TRIVIAL)

def arms_in(src, regex, constants):
    found = {}
    for m in re.finditer(regex, src):
        for pattern, body in match_arms(src, m.end() - 1):
            for iface, cid in pattern_ids(pattern, constants):
                d = found.setdefault(iface, {})
                d[cid] = d.get(cid, True) and is_trivial(body)
    return found

def fn_body(src, name):
    m = re.search(r"\bfn " + re.escape(name) + r"\s*\(", src)
    if not m:
        return None
    brace = src.index("{", matching(src, m.end() - 1))
    return src[brace : matching(src, brace) + 1]

def scan_code():
    found = {}
    mod = strip_comments((SERVICES / "mod.rs").read_text(encoding="utf-8"))
    names_of = {}
    for line in fn_body(mod, "open").splitlines():
        if "=>" in line and "Arc::new" in line:
            pat, expr = line.split("=>", 1)
            m = re.search(r"Arc::new\((?:\w+::)*(\w+)", expr)
            if m:
                names_of.setdefault(m.group(1), []).extend(re.findall(r'"([^"]+)"', pat))

    for names in names_of.values():
        for n in names:
            RUNTIME_ALIASES.setdefault(n, names[0])

    for path in sorted(SERVICES.glob("*.rs")):
        src = strip_comments(path.read_text(encoding="utf-8"))
        constants = consts(src)
        for m in re.finditer(r"impl Service for (\w+)\s*\{", src):
            struct = m.group(1)
            block = src[m.end() - 1 : matching(src, m.end() - 1) + 1]
            lit = re.search(r'fn name\(&self\) -> String \{\s*"([^"]+)"\.into\(\)', block)
            if lit:
                default = lit.group(1)
            elif struct in names_of:
                default = names_of[struct][0]
            else:
                continue
            for iface, ids in arms_in(block, r"match (?:req\.cmd|\(self\.name, req\.cmd\))\s*\{", constants).items():
                found.setdefault(("services", iface or default), {}).update(ids)
        for stub in re.findall(r'Stub::new\("([^"]+)"\)', src):
            found.setdefault(("services", stub), {})
        if path.name == "vi.rs":
            body = fn_body(src, "transact")
            for ids in arms_in(body, r"match code\s*\{", constants).values():
                found.setdefault(("services", "vi/IGraphicBufferProducer"), {}).update(ids)
        if path.name == "nvdrv.rs":
            for fn, devs in NV_DEVICES.items():
                body = fn_body(src, fn)
                for ids in arms_in(body, r"match nr\s*\{", constants).values():
                    for dev in [devs] if isinstance(devs, str) else devs:
                        found.setdefault(("ioctl", dev), {}).update(ids)

    src = strip_comments(HORIZON.read_text(encoding="utf-8"))
    body = fn_body(src, "dispatch")
    for ids in arms_in(body, r"match imm\s*\{", consts(src)).values():
        found.setdefault(("svc", "SVCs"), {}).update(ids)
    return {(sec, ALIASES.get(name, name)): ids for (sec, name), ids in found.items()}

def read_reference():
    ref, cur = {}, None
    if not REFERENCE.exists():
        return ref
    for raw in REFERENCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        head = re.fullmatch(r"\[(service|svc|ioctl)\s*(.*)\]", line)
        if head:
            kind, name = head.groups()
            key = {"service": ("services", name), "svc": ("svc", "SVCs"), "ioctl": ("ioctl", name)}[kind]
            cur = ref.setdefault(key, {"official": "", "url": "", "ids": {}, "found": True})
            continue
        if cur is None:
            continue
        if line.startswith("#"):
            text = line.lstrip("# ").strip()
            if "not found" in text:
                cur["found"] = False
            elif not cur["official"]:
                url = re.search(r"https?://\S+", text)
                cur["url"] = url.group(0) if url else ""
                cur["official"] = re.sub(r"\s*[—-]+\s*https?://\S+", "", text).strip()
            continue
        parts = line.split(None, 1)
        try:
            cid = int(parts[0].replace("?", "0"), 0)
        except ValueError:
            continue
        name = parts[1] if len(parts) > 1 else ""
        cur["ids"][cid] = f"{name} ({parts[0]})" if "?" in parts[0] else name
        REF_NAMES[(key[0], key[1], cid)] = name
    return ref

SECTIONS = [
    ("services", "Horizon services", "IPC commands"),
    ("svc", "Kernel SVCs", "SVCs"),
    ("ioctl", "GPU driver (nvdrv)", "ioctls"),
    ("shaders", "Maxwell shader instructions", "mnemonics"),
]

def shader_groups():
    if not MAXWELL_OPCODES.exists() or not MAXWELL_TRANSLATE.exists():
        return []
    names = sorted(set(re.findall(r'name: "([a-z0-9_.]+)"', MAXWELL_OPCODES.read_text(encoding="utf-8"))))
    handled = set()
    for line in strip_comments(MAXWELL_TRANSLATE.read_text(encoding="utf-8")).splitlines():
        if "=>" in line:
            handled.update(re.findall(r'"([a-z0-9_.]+)"', line.split("=>", 1)[0]))
    class_of = {n: c for c, ns in SHADER_CLASSES.items() for n in ns.split()}
    groups = {}
    for n in names:
        groups.setdefault(class_of.get(n, "outras"), []).append(n)
    out = []
    for cls in list(SHADER_CLASSES) + ["outras"]:
        members = groups.get(cls)
        if not members:
            continue
        done = [n for n in members if n in handled]
        out.append(
            {
                "name": cls,
                "family": cls,
                "official": "",
                "url": "",
                "referenced": True,
                "implemented": done,
                "constant": [],
                "missing": [{"id": n, "name": ""} for n in members if n not in handled],
                "undocumented": [],
                "total": len(members),
            }
        )
    return out

def family(section, iface):
    if section == "svc":
        return "SVC"
    if section == "ioctl":
        return NV_SHORT.get(iface, iface)
    base = iface.split("/")[0]
    base = FAMILY.get(base, base)
    return base.split(":")[0] or base

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
REF_NAMES = {}

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

def fmt_id(section, cid):
    return hex(cid) if section in ("svc", "ioctl") and isinstance(cid, int) else str(cid)

def fmt_time(seconds):
    if seconds is None:
        return "?"
    m, s = divmod(float(seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"

COLORS = {
    "done": "#2ea043",
    "constant": "#9be9a8",
    "missing": "#6e7681",
    "used_done": "#0f5323",
    "used_missing": "#f85149",
    "bg": "#0d1117",
    "text": "#f0f6fc",
}

def squarify(items, x, y, w, h):
    rects, items = {}, [it for it in items if it[1] > 0]

    def worst(row, side):
        s = sum(a for _, a in row)
        return max(max(side * side * a / (s * s), s * s / (side * side * a)) for _, a in row)

    while items:
        side = min(w, h)
        row = [items[0]]
        for it in items[1:]:
            if worst(row + [it], side) <= worst(row, side):
                row.append(it)
            else:
                break
        items = items[len(row) :]
        s = sum(a for _, a in row)
        if w >= h:
            cw, yy = s / h, y
            for k, a in row:
                rects[k] = (x, yy, cw, a / cw)
                yy += a / cw
            x, w = x + cw, w - cw
        else:
            rh, xx = s / w, x
            for k, a in row:
                rects[k] = (xx, y, a / rh, rh)
                xx += a / rh
            y, h = y + rh, h - rh
    return rects

def cell_grid(n, w, h):
    if n == 0 or w <= 0 or h <= 0:
        return 0, 0
    s = math.sqrt(w * h / n)
    while s > 1 and math.floor(w / s) * math.floor(h / s) < n:
        s -= 0.25
    return s, max(1, math.floor(w / s))

def treemap_panel(section, x0, y0, w, h, out):
    pct = 100 * section["implemented"] / section["total"] if section["total"] else 0
    out.append(
        f'<text x="{x0 + 4}" y="{y0 + 17}" font-size="16" fill="{COLORS["text"]}">'
        f'{html.escape(section["title"])}: {pct:.1f}% ({section["implemented"]}/{section["total"]})</text>'
    )
    top = y0 + 26
    families = {}
    for g in section["groups"]:
        families.setdefault(g["family"], []).append(g)
    items = sorted(((f, sum(g["total"] for g in gs)) for f, gs in families.items()), key=lambda t: -t[1])
    scale = (w * (h - 26)) / max(1, sum(a for _, a in items))
    rects = squarify([(f, a * scale) for f, a in items], x0, top, w, h - 26)
    for fam, (rx, ry, rw, rh) in rects.items():
        gs = families[fam]
        cells = []
        order = {"used_done": 0, "done": 1, "constant": 2, "used_missing": 3, "missing": 4}
        for g in gs:
            const, used, used_missing = set(g["constant"]), set(g.get("used", [])), set(g.get("used_missing", []))
            for c in g["implemented"]:
                cells.append("used_missing" if c in used_missing else "used_done" if c in used else "constant" if c in const else "done")
            cells += ["used_missing" if m["id"] in used_missing else "missing" for m in g["missing"]]
        cells.sort(key=lambda c: order[c])
        strip = 18 if rh > 44 and rw > 26 else 0
        inner = (rx + 2, ry + 2 + strip, rw - 4, rh - 4 - strip)
        out.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" fill="{COLORS["bg"]}"/>')
        s, cols = cell_grid(len(cells), inner[2], inner[3])
        if s >= 1:
            out.append(f'<g stroke="{COLORS["bg"]}" stroke-width="{0.6 if s > 4 else 0.3}">')
            for i, kind in enumerate(cells):
                cx, cy = inner[0] + (i % cols) * s, inner[1] + (i // cols) * s
                out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{s:.1f}" height="{s:.1f}" fill="{COLORS[kind]}"/>')
            out.append("</g>")
        if rw >= 7.5 * len(fam) + 8 and rh >= 18:
            label = html.escape(fam)
            out.append(
                f'<text x="{rx + 5:.1f}" y="{ry + 15:.1f}" font-size="12" font-weight="bold" fill="{COLORS["text"]}" '
                f'stroke="{COLORS["bg"]}" stroke-width="3" paint-order="stroke">{label}</text>'
            )

def render_svg(report):
    services = report["sections"][0]
    system = [s for s in report["sections"] if s["id"] in ("svc", "ioctl")]
    kernel = {
        "title": "Kernel and GPU",
        "implemented": sum(s["implemented"] for s in system),
        "total": sum(s["total"] for s in system),
        "groups": [g for s in system for g in s["groups"]],
    }
    shaders = next((s for s in report["sections"] if s["id"] == "shaders"), None)
    W, H = 1000, 340
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
    ]
    if shaders and shaders["total"]:
        treemap_panel(services, 0, 0, 500, H - 22, out)
        treemap_panel(kernel, 510, 0, 235, H - 22, out)
        treemap_panel({**shaders, "title": "Maxwell shaders"}, 755, 0, 245, H - 22, out)
    else:
        treemap_panel(services, 0, 0, 640, H - 22, out)
        treemap_panel(kernel, 650, 0, 350, H - 22, out)
    lx = 4
    legend = [("done", "implemented"), ("constant", "returns success/constants only"), ("missing", "missing")]
    if "coverage" in report:
        legend = [("used_done", "used by the game"), ("used_missing", "used, stub/missing")] + legend
    for kind, text in legend:
        out.append(f'<rect x="{lx}" y="{H - 15}" width="10" height="10" fill="{COLORS[kind]}"/>')
        out.append(f'<text x="{lx + 14}" y="{H - 6}" font-size="11" fill="{COLORS["text"]}">{text}</text>')
        lx += 26 + 5.6 * len(text)
    out.append(f'<text x="{W - 4}" y="{H - 6}" font-size="11" fill="{COLORS["missing"]}" text-anchor="end">kirikae · {report["generated"]}</text>')
    out.append("</svg>")
    return "\n".join(out)

STYLE = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--bar:#e6e9ec;--done:#2ea043;--const:#9be9a8}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#f0f6fc;--muted:#9198a1;--line:#3d444d;--bar:#262c36}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1000px;margin:auto;padding:16px}
img{max-width:100%;height:auto;border-radius:6px}a{color:#4493f8}
h2{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
td.n,th.n{text-align:right;white-space:nowrap}tr.fam td{font-weight:600;padding-top:12px;border-bottom:none}
.bar{display:inline-block;width:90px;height:8px;background:var(--bar);border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:6px}
.bar i{display:block;height:100%;background:var(--done)}
details summary{cursor:pointer;color:var(--muted)}details ul{columns:2;margin:4px 0;padding-left:18px;font-size:13px}
code{font-size:13px}.muted{color:var(--muted)}
@media (max-width:640px){details ul{columns:1}td:nth-child(2),th:nth-child(2){display:none}}
"""

def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "—"

def section_table(section):
    rows = [
        f'<h2>{html.escape(section["title"])}</h2>',
        f'<p class="muted">{section["implemented"]} of {section["total"]} documented {section["unit"]} ({pct(section["implemented"], section["total"])}).</p>',
        '<table><tr><th>Interface</th><th>Official name</th><th class="n">Done</th><th class="n">Constant</th><th class="n">Total</th><th class="n">%</th></tr>',
    ]
    fam = None
    hexa = section["id"] in ("svc", "ioctl")
    for g in section["groups"]:
        if section["id"] == "services" and g["family"] != fam:
            fam = g["family"]
            rows.append(f'<tr class="fam"><td colspan="6">{html.escape(fam)}</td></tr>')
        done, total = len(g["implemented"]), g["total"]
        official = html.escape(g["official"])
        if g["url"]:
            official = f'<a href="{html.escape(g["url"])}">{official or "switchbrew"}</a>'
        width = 100 * done / total if total else 0
        name = f'<code>{html.escape(g["name"])}</code>'
        missing = ""
        if g["missing"]:
            items = "".join(f'<li><code>{(hex(m["id"]) if hexa else m["id"])}</code> {html.escape(m["name"])}</li>' for m in g["missing"])
            missing = f'<details><summary>{len(g["missing"])} missing</summary><ul>{items}</ul></details>'
        if section["id"] == "shaders":
            pass
        elif not g["referenced"]:
            missing += '<div class="muted">no reference list: total = handled commands</div>'
        if g["undocumented"]:
            ids = ", ".join(hex(c) if hexa else str(c) for c in g["undocumented"])
            missing += f'<div class="muted">handled but undocumented: {ids}</div>'
        if g.get("used") or g.get("used_missing"):
            fmt = lambda c: hex(c) if hexa and isinstance(c, int) else str(c)
            used = ", ".join(fmt(c) for c in g.get("used", []) + g.get("used_missing", []))
            missing += f'<div class="muted">used by games: {used}</div>'
            if g.get("used_missing"):
                missing += f'<div style="color:#f85149">used, stub/missing: {", ".join(fmt(c) for c in g["used_missing"])}</div>'
        rows.append(
            f'<tr><td>{name}{missing}</td><td>{official}</td><td class="n">{done}</td><td class="n">{len(g["constant"])}</td>'
            f'<td class="n">{total}</td><td class="n"><span class="bar"><i style="width:{width:.0f}%"></i></span>{pct(done, total)}</td></tr>'
        )
    rows.append("</table>")
    return "\n".join(rows)

def coverage_section(report):
    cov = report["coverage"]
    games = ", ".join(report.get("games", []))
    rows = [
        "<h2>Used by the game and missing</h2>",
        f'<p class="muted">{cov["used"]} commands, SVCs, ioctls and instructions used ({html.escape(games)}); '
        f'{len(cov["gaps"])} of them are still stubs or missing, most used first. Each event keeps the moment '
        "of the first use: the next frame and a copy of the game's saves at that point "
        "(to go back there, load the latest save the game wrote before it).</p>",
        '<table><tr><th>Item</th><th>Status</th><th class="n">Count</th><th>First use</th><th>Event</th></tr>',
    ]
    for gap in cov["gaps"]:
        item = f'<code>{html.escape(gap["interface"])} {html.escape(fmt_id(gap["section"], gap["id"]))}</code>'
        if gap["name"]:
            item += f" {html.escape(gap['name'])}"
        status = html.escape(gap["status"] + (f" ({gap['note']})" if gap["note"] else ""))
        first = gap["first"] or {}
        when = f'{html.escape(first.get("game", ""))} {fmt_time(first.get("seconds"))}, frame {first.get("frame", "?")}'
        ev = gap["event"]
        link = ""
        if ev:
            parts = [f'#{ev["n"]}']
            if ev["screenshot"]:
                uri = Path(ev["screenshot"]).resolve().as_uri()
                parts.append(f'<a href="{uri}"><img src="{uri}" alt="frame" style="width:120px;vertical-align:middle"></a>')
            if ev["saves"]:
                parts.append(f'<a href="{Path(ev["saves"]).resolve().as_uri()}">saves</a>')
            link = " ".join(parts)
        rows.append(f'<tr><td>{item}</td><td>{status}</td><td class="n">{gap["count"]}</td><td>{when}</td><td>{link}</td></tr>')
    rows.append("</table>")
    if cov["resolved"]:
        done = ", ".join(f'<code>{html.escape(r["interface"])} {html.escape(fmt_id(r["section"], r["id"]))}</code>' for r in cov["resolved"])
        rows.append(f'<p class="muted">Implemented after being recorded as stub/missing: {done}.</p>')
    return "\n".join(rows)

def render_html(report):
    total_done = sum(s["implemented"] for s in report["sections"])
    total_all = sum(s["total"] for s in report["sections"])
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>kirikae progress</title><style>{STYLE}</style></head><body>",
        "<h1>Kirikae progress</h1>",
        '<img src="progress.svg" alt="Treemap: each square is a command, SVC, ioctl or shader instruction; green = implemented, gray = missing">',
        f'<p>Generated by <code>tools/progress.py</code> on {report["generated"]}. Raw numbers: <a href="progress.json">progress.json</a>. '
        f'<a href="history.html">History</a> (one point a day). In total: {total_done} of {total_all} ({pct(total_done, total_all)}).</p>',
    ]
    if "coverage" in report:
        parts.append(coverage_section(report))
    parts += [section_table(s) for s in report["sections"]]
    parts.append(
        "<h2>How it is counted</h2><p>A command counts as implemented when the runtime has its own arm for it in the "
        "<code>match</code> of the service (<code>crates/rt/src/services</code>), the SVC (<code>horizon.rs</code>) or the device "
        "ioctl (<code>services/nvdrv.rs</code>); anything that falls to the default arm goes to the stub, which answers with zeros "
        "and warns. The <em>Constant</em> column sets apart implemented commands whose body only returns success or fixed values. "
        "Ioctls are compared by the low byte of their number, which is how the devices dispatch them.</p>"
        "<p>The totals are the commands documented on <a href=\"https://switchbrew.org\">Switchbrew</a> "
        "(<code>tools/reference/horizon.txt</code>, IDs and names only) for the interfaces Kirikae exposes or games ask for, "
        "not every system service. A new interface joins the count when it joins the reference.</p>"
        "<p>Shader instructions count the mnemonics of the Maxwell opcode table generated from envytools "
        "(<code>crates/maxwell/src/opcodes.rs</code>) that have their own arm in the SPIR-V translator "
        "(<code>crates/maxwell/src/translate.rs</code>). A mnemonic counts whole even when only some of its "
        "variants are translated.</p></body></html>"
    )
    return "\n".join(parts)

def update_history(path, report):
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    point = {"date": report["generated"], **{s["id"]: [s["implemented"], s["total"]] for s in report["sections"]}}
    history = [p for p in history if p["date"] != point["date"]] + [point]
    history.sort(key=lambda p: p["date"])
    path.write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    return history

def render_history(history):
    W, H, pad = 900, 300, 40
    colors = {"services": "#2ea043", "svc": "#4493f8", "ioctl": "#d29922", "shaders": "#a371f7"}
    names = {sid: title for sid, title, _ in SECTIONS}
    n = len(history)
    xs = [pad + (W - 2 * pad) * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Progress per day">']
    for p in (0, 25, 50, 75, 100):
        y = H - pad - (H - 2 * pad) * p / 100
        svg.append(f'<line x1="{pad}" x2="{W - pad}" y1="{y}" y2="{y}" stroke="currentColor" stroke-opacity=".15"/>')
        svg.append(f'<text x="{pad - 6}" y="{y + 4}" font-size="11" text-anchor="end" fill="currentColor">{p}%</text>')
    for sid, color in colors.items():
        pts = [(x, H - pad - (H - 2 * pad) * (p[sid][0] / p[sid][1] if p.get(sid) and p[sid][1] else 0)) for x, p in zip(xs, history)]
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        svg += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>' for x, y in pts]
    for x, p in zip(xs, history):
        if n <= 12 or p is history[-1] or p is history[0]:
            svg.append(f'<text x="{x:.1f}" y="{H - pad + 16}" font-size="11" text-anchor="middle" fill="currentColor">{p["date"]}</text>')
    svg.append("</svg>")
    legend = " ".join(f'<span style="color:{c}">■</span> {html.escape(names[s])}' for s, c in colors.items())
    rows = "".join(
        f'<tr><td>{p["date"]}</td>' + "".join(f'<td class="n">{p[s][0]}/{p[s][1]}</td>' if p.get(s) else "<td></td>" for s in colors) + "</tr>"
        for p in reversed(history)
    )
    heads = "".join(f'<th class="n">{html.escape(names[s])}</th>' for s in colors)
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>kirikae history</title><style>{STYLE}</style></head><body><h1>Progress history</h1>"
        f'<p><a href="index.html">← Progress</a></p><p>{legend}</p>{"".join(svg)}'
        f"<table><tr><th>Day</th>{heads}</tr>{rows}</table></body></html>"
    )

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
