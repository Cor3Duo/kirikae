#!/usr/bin/env python3

import argparse
import datetime
import os
import html
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "crates/rt/src/services"
HORIZON = ROOT / "crates/rt/src/horizon.rs"
REFERENCE = ROOT / "tools/reference/horizon.txt"
MAXWELL_OPCODES = ROOT / "crates/maxwell/src/opcodes.rs"
MAXWELL_TRANSLATE = ROOT / "crates/maxwell/src/translate.rs"
NVN_DIR = ROOT / "crates/rt/src/nvn"
# The batch matrix lives outside the repository (see tools/discovery/batch.py) and is read from
# --compat, KIRIKAE_COMPAT or the discovery work folder, in that order. It is only included when
# the file exists, so a fresh clone and a CI run without the artifact regenerate the same dashboard.
STATUS_MD = ROOT / "docs/plan/STATUS.md"

# NVN host table: `host::proc` takes the call over, `state::setter` feeds the state it records,
# while `shadow::hook` and `resources::call` only observe or build resources on the driver path.
NVN_HOST = NVN_DIR / "host.rs"
NVN_STATE = NVN_DIR / "state.rs"
NVN_SHADOW = NVN_DIR / "shadow.rs"
NVN_RESOURCES = NVN_DIR / "resources.rs"

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

# The Maxwell opcode table has no stage column (envytools' gm107 table lists encodings only), so
# the stage split uses the instructions a stage alone encodes. Everything not listed here is
# shared and counts in every stage; the denominator is always the whole table.
SHADER_STAGES = {
    "vertex": (),
    "fragment": ("ipa", "pixld", "kil", "out"),
    "compute": ("bar", "membar", "cctl", "cctll", "cctlt"),
}

# Coarse families over the same mnemonics. Atomics are checked first so the surface atomics do
# not also land in textures; control keeps the barrier ops; the rest is ALU.
SHADER_FAMILIES = {
    "atomics": "atom atoms atomcas red sured suatom",
    "texture": "tex texs tld tlds tld4 tld4s txq txd txa tmml tmmls suld sust isberd",
    "control": "bra brx ssy sync pbk brk cont pcnt exit kil bar membar depbar nop cal jcal ret jmp jmx pret sam rtt bpt preexit ide",
}

NVN_FAMILIES = ["pools", "buffers", "textures", "samplers", "programs", "device", "commands", "state", "other"]
NVN_FAMILY_LABELS = {
    "pools": "memory, texture and sampler pools",
    "buffers": "buffers",
    "textures": "textures and views",
    "samplers": "samplers",
    "programs": "programs and shader data",
    "device": "device handles",
    "commands": "command buffers and queues",
    "state": "state setters",
    "other": "other procs",
}
# Procs whose host arm only computes a return value (the device handle getters pack two ids).
# They are implemented, but shown as constant like the services' fixed replies.
NVN_CONSTANT_PREFIX = "Handle("

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

# Services that answer every command but do not do the real work (silence, zeros or
# fixed replies). Their commands stay "handled" in the data, but they are not counted
# as implemented in the badges or the dashboard: the light-green cell means "returns
# success/constants only". Keep this list in sync with the source and say why.
SILENT_STUBS = {
    "hwopus": "decodes packet headers but writes silence",
    "nfp": "fake amiibo service, no NFC device",
}

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

SERVICE_DISPATCH = r"match (?:req\.cmd|\(self\.name, req\.cmd\)|cmd|code)\s*\{"
# A call in a `call` body: `self.method(`, `module::fn(` or `fn(`. Definitions (`fn call(`) and
# paths that start at a type (`Self::`, `Arc::new`) are not callees of the service.
CALLEE = re.compile(r"(?<!fn )(?<![\w.])(?:(self)\.(\w+)|([a-z_]\w*)::(\w+)|([a-z_]\w*))\s*\(")


def command_arms(scope, sources, consts_of, struct):
    """Arms of an inherent `fn command` used by services whose `call` delegates to it.

    `scope` is the module scope of the file that holds the `impl Service`; a split service may
    keep the inherent impl in a sibling file, so the lookup is not limited to one file.
    """
    return struct_method_arms(scope, sources, consts_of, struct, "command")


def method_body(scope, sources, struct, method):
    """`(file, body)` of `impl <struct> { fn <method> }`, wherever the module split put it."""
    for path in scope:
        src = sources[path]
        for m in re.finditer(r"impl\s+" + re.escape(struct) + r"\s*\{", src):
            block = src[m.end() - 1 : matching(src, m.end() - 1) + 1]
            body = fn_body(block, method)
            if body:
                return path, body
    return None


def struct_method_arms(scope, sources, consts_of, struct, method):
    """Arms of `impl <struct> { fn <method> }`, wherever the module split put that impl."""
    found = method_body(scope, sources, struct, method)
    if not found:
        return {}
    path, body = found
    return arms_in(body, SERVICE_DISPATCH, consts_of[path])


def module_scope(path, sources):
    """Where a service's helpers may live, in resolution order.

    `foo.rs` keeps its submodules in `foo/`, and `foo/mod.rs` keeps them next to itself; a service
    split from one file into a folder is therefore still one module.
    """
    folder = path.parent / path.stem
    scope = [path]
    scope += sorted(p for p in sources if p.parent == folder and p != path)
    scope += sorted(p for p in sources if p.parent == path.parent and p != path and p.parent != folder)
    return scope


def module_source(path, name, sources):
    """The file of module `name` as seen from `path`: a sibling, or inside its module folder."""
    for cand in (
        path.parent / f"{name}.rs",
        path.parent / name / "mod.rs",
        path.parent / path.stem / f"{name}.rs",
        path.parent / path.stem / name / "mod.rs",
    ):
        if cand in sources:
            return cand
    return None


def local_scope(path, sources):
    """The file and the folder of its submodules: where an unqualified free function lives."""
    folder = path.parent / path.stem
    return [path] + sorted(p for p in sources if p.parent == folder and p != path)


def free_fn_body(scope, sources, name):
    """`(file, body)` of `fn <name>` in scope, for a `call` that delegates to a free function."""
    for path in scope:
        body = fn_body(sources[path], name)
        if body:
            return path, body
    return None


def callee_bodies(block, path, sources, struct):
    """`(name, file, body)` of every helper the block calls, resolved inside the service's module."""
    scope = module_scope(path, sources)
    local = local_scope(path, sources)
    out = []
    for m in CALLEE.finditer(block):
        self_, method, module, qualified, bare = m.groups()
        if self_:
            found = method_body(scope, sources, struct, method)
        elif module:
            target = module_source(path, module, sources)
            found = free_fn_body([target], sources, qualified) if target else None
        else:
            found = free_fn_body(local, sources, bare)
        if found:
            name = method or qualified or bare or ""
            out.append((name, found[0], found[1]))
    return out


def delegated_arms(block, path, sources, consts_of, struct, depth=3):
    """Arms of the helpers a `call` delegates to, when the dispatch lives in another function.

    Handles `self.object(req.cmd)`, `module::open(req.cmd, ..)` and `host_command(.., req, ..)`,
    including a second hop such as `object` falling through to `applet_manager`. Every helper
    that carries a command table contributes its arms; unrelated helper calls resolve to
    functions without one and add nothing.
    """
    merged = {}
    if depth <= 0:
        return merged
    for name, target, body in callee_bodies(block, path, sources, struct):
        # A `*_command_name` helper is a diagnostic name table, not a dispatch: counting its
        # arms as implemented inflates every service that names its unknown commands.
        if name.endswith("_command_name") or name.endswith("_name"):
            continue
        merge_arms(merged, arms_in(body, SERVICE_DISPATCH, consts_of[target]))
        merge_arms(merged, delegated_arms(body, target, sources, consts_of, struct, depth - 1))
    return merged


def merge_arms(dst, src):
    for iface, ids in src.items():
        dst.setdefault(iface, {}).update(ids)
    return dst


def fn_body(src, name):
    m = re.search(r"\bfn " + re.escape(name) + r"\s*\(", src)
    if not m:
        return None
    brace = src.index("{", matching(src, m.end() - 1))
    return src[brace : matching(src, brace) + 1]

def arc_struct(arm):
    """The service struct of an `Arc::new(path::Type { .. })` arm, ignoring `::new`."""
    m = re.match(r"\s*Arc::new\(\s*([A-Za-z_][\w:]*)", arm)
    if not m:
        return None
    segments = [s for s in m.group(1).split("::") if s and s[0].isupper()]
    return segments[-1] if segments else None


def service_registry(src):
    """`Struct -> [service names]` from the registry that `services::open` delegates to.

    The registry function was renamed from `open` to `host_object`; resolving only the old name
    silently dropped every service whose `name()` returns a field instead of a literal.
    """
    body = fn_body(src, "host_object") or fn_body(src, "open")
    if not body:
        return {}
    m = re.search(r"match\s+name\s*\{", body)
    if not m:
        return {}
    names_of = {}
    for pattern, arm in match_arms(body, m.end() - 1):
        struct = arc_struct(arm)
        if not struct:
            continue
        for name in re.findall(r'"([^"]+)"', pattern.split(" if ")[0]):
            names_of.setdefault(struct, []).append(name)
    return names_of


def nvdrv_scope(services, sources):
    """`nvdrv.rs` and every file of its `nvdrv/` folder: the ioctl arms may move either way."""
    return [p for p in sorted(sources) if p == services / "nvdrv.rs" or p.parent == services / "nvdrv"]


def scan_code(services=None, horizon=None):
    services = Path(services or SERVICES)
    horizon = Path(horizon or HORIZON)
    sources = {p: strip_comments(p.read_text(encoding="utf-8")) for p in sorted(services.rglob("*.rs"))}
    consts_of = {p: consts(src) for p, src in sources.items()}
    RUNTIME_ALIASES.clear()
    found = {}
    names_of = service_registry(sources.get(services / "mod.rs", ""))

    for names in names_of.values():
        for n in names:
            RUNTIME_ALIASES.setdefault(n, names[0])

    for path in sorted(sources):
        src = sources[path]
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
            scope = module_scope(path, sources)
            arms = arms_in(block, r"match (?:req\.cmd|\(self\.name, req\.cmd\))\s*\{", consts_of[path])
            merge_arms(arms, command_arms(scope, sources, consts_of, struct))
            merge_arms(arms, delegated_arms(block, path, sources, consts_of, struct))
            for iface, ids in arms.items():
                found.setdefault(("services", iface or default), {}).update(ids)
        for stub in re.findall(r'Stub::new\("([^"]+)"\)', src):
            found.setdefault(("services", stub), {})
        if path.name == "vi.rs":
            body = fn_body(src, "transact")
            if body:
                for ids in arms_in(body, r"match code\s*\{", consts_of[path]).values():
                    found.setdefault(("services", "vi/IGraphicBufferProducer"), {}).update(ids)

    ioctls = nvdrv_scope(services, sources)
    for fn, devs in NV_DEVICES.items():
        for path in ioctls:
            body = fn_body(sources[path], fn)
            if not body:
                continue
            for ids in arms_in(body, r"match nr\s*\{", consts_of[path]).values():
                for dev in [devs] if isinstance(devs, str) else devs:
                    found.setdefault(("ioctl", dev), {}).update(ids)

    if horizon.exists():
        src = strip_comments(horizon.read_text(encoding="utf-8"))
        body = fn_body(src, "dispatch")
        if body:
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
    ("nvn", "NVN host (WS4)", "NVN procs"),
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

def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""

def named_arms(src, fn_name):
    """Quoted `"Name" => body` arms of a dispatcher's `match name`, with the `nvn` prefix restored."""
    body = fn_body(strip_comments(src), fn_name)
    if not body:
        return {}
    m = re.search(r"match name(?:\.strip_prefix\(\"nvn\"\)\?)?\s*\{", body)
    if not m:
        return {}
    out = {}
    for pattern, arm in match_arms(body, m.end() - 1):
        if pattern.strip() == "_":
            continue
        for name in re.findall(r'"([A-Za-z0-9_]+)"', pattern):
            out[name if name.startswith("nvn") else "nvn" + name] = arm
    return out

def nvn_family(name):
    n = name[3:] if name.startswith("nvn") else name
    if n.startswith(("MemoryPool", "TexturePool", "SamplerPool")):
        return "pools"
    if n.startswith("Buffer"):
        return "buffers"
    if n.startswith("Sampler"):
        return "samplers"
    if n.startswith("Texture"):
        return "textures"
    if n.startswith("Program"):
        return "programs"
    if n.startswith("Device"):
        return "device"
    if n.startswith(("CommandBuffer", "Queue")):
        return "commands"
    if "State" in n:
        return "state"
    return "other"

def nvn_section():
    """What the NVN host takes over, against every `nvn*` proc the nvn modules name."""
    sources = [NVN_HOST, NVN_STATE, NVN_SHADOW, NVN_RESOURCES]
    if not all(p.is_file() for p in sources):
        return None
    host = named_arms(read_text(NVN_HOST), "proc")
    state = named_arms(read_text(NVN_STATE), "setter")
    hooks = named_arms(read_text(NVN_SHADOW), "hook")
    resources = named_arms(read_text(NVN_RESOURCES), "call")
    m = re.search(r"const NOT_RECORDED: \[&str; \d+\] = \[(.*?)\];", strip_comments(read_text(NVN_SHADOW)), re.S)
    elsewhere = set(re.findall(r'"(nvn[A-Za-z0-9_]+)"', m.group(1))) if m else set()
    hosted = set(host) | set(state)
    constants = {n for n, arm in host.items() if arm.startswith(NVN_CONSTANT_PREFIX)}
    known = hosted | set(hooks) | set(resources) | elsewhere
    groups = []
    for fam in NVN_FAMILIES:
        names = sorted(n for n in known if nvn_family(n) == fam)
        if not names:
            continue
        groups.append(
            {
                "name": fam,
                "family": fam,
                "official": NVN_FAMILY_LABELS.get(fam, fam),
                "url": "",
                "referenced": False,
                "implemented": [n for n in names if n in hosted],
                "constant": sorted(n for n in names if n in constants),
                "missing": [{"id": n, "name": ""} for n in names if n not in hosted],
                "undocumented": [],
                "total": len(names),
            }
        )
    return {
        "id": "nvn",
        "title": "NVN host (WS4)",
        "unit": "NVN procs",
        "implemented": len(hosted),
        "constant": len(constants),
        "silent": 0,
        "total": len(known),
        "referenced": False,
        "groups": groups,
    }

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

def split_md_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cur, i = [], "", 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if line[i] == "|":
            cells.append(cur.strip())
            cur = ""
        else:
            cur += line[i]
        i += 1
    cells.append(cur.strip())
    return cells

def compat_record(rec):
    instructions = rec.get("instructions", "")
    if instructions != "" and rec.get("reachable_total") not in (None, ""):
        instructions = f'{instructions}/{rec["reachable_total"]}'
    return {
        "game": str(rec.get("game", "")),
        "status": str(rec.get("status", "")),
        "functions": rec.get("functions", ""),
        "instructions": instructions,
        "frames": rec.get("frames", ""),
        "fps": rec.get("fps", ""),
        "gaps": rec.get("gaps", ""),
        "note": str(rec.get("note") or ""),
    }

def compat_json_rows(text):
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("games", [])
    return [compat_record(r) for r in data if isinstance(r, dict) and r.get("game")]

def compat_markdown_rows(text):
    header, rows = None, []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = split_md_row(line)
        if header is None:
            if cells and cells[0].strip().lower() == "game":
                header = [c.strip().lower() for c in cells]
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rec = dict(zip(header, cells))
        rows.append(compat_record(rec))
    return rows

def load_compat(path):
    """The batch matrix (compat.json or compat.md), or None when there is none."""
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        found = [p / n for n in ("compat.json", "compat.md") if (p / n).is_file()]
        if not found:
            return None
        p = found[0]
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8")
        rows = compat_json_rows(text) if p.suffix.lower() == ".json" else compat_markdown_rows(text)
    except (OSError, ValueError) as e:
        print(f"note: cannot read {p}: {e}", file=sys.stderr)
        return None
    if not rows:
        return None
    boots = sum(r["status"] == "boots" for r in rows)
    compiled = sum(r["status"] in ("boots", "compiled") for r in rows)
    return {
        "source": p.name,
        "counts": {"games": len(rows), "boots": boots, "compiled": compiled},
        "games": rows,
    }


def discovery_root():
    """The artifact folder, with the same default as tools/discovery/loop.py."""
    env = os.environ.get("KIRIKAE_DISCOVERY_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share"
    return Path(base) / "kirikae-discovery"


def resolve_compat(explicit=None):
    """The batch matrix if it exists: --compat, KIRIKAE_COMPAT, then the discovery work folder.

    `tools/discovery/batch.py` writes `compat.json`/`compat.md` under `<work>/`; the CI entry
    point passes it only when the artifact is there and the section is skipped otherwise.
    """
    for candidate in (explicit, os.environ.get("KIRIKAE_COMPAT"), str(discovery_root() / "batch")):
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_dir() and any((p / n).is_file() for n in ("compat.json", "compat.md")):
            return p
        if p.is_file():
            return p
    return None

WORKSTREAM_STATUS = {"hecho": "done", "resuelto": "done", "confirmado": "confirmed", "parcial": "partial"}

def workstream_rows():
    """The workstream table of docs/plan/STATUS.md, one row per WS with its status counts."""
    if not STATUS_MD.is_file():
        return []
    by_ws = {}
    for line in read_text(STATUS_MD).splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = split_md_row(line)
        if len(cells) < 3 or not re.fullmatch(r"WS\d+", cells[0].strip()):
            continue
        ws, item, status = cells[0].strip(), cells[1], cells[2]
        clean = lambda s: re.sub(r"[`*]", "", s).strip()
        state = WORKSTREAM_STATUS.get(clean(status).lower().split(" (")[0], "pending")
        entry = by_ws.setdefault(
            ws, {"ws": ws, "items": 0, "done": 0, "partial": 0, "pending": 0, "confirmed": 0, "state": "done"}
        )
        entry["items"] += 1
        entry[state] += 1
        rank = {"done": 0, "confirmed": 1, "partial": 2, "pending": 3}
        if rank[state] > rank[entry["state"]]:
            entry["state"] = state
        entry.setdefault("latest", clean(item))
    order = sorted(by_ws, key=lambda w: int(w[2:]))
    return [by_ws[w] for w in order]

def report_compat(report):
    compat = report.get("compat")
    if not compat:
        return []
    rows = [
        "<h2>Compatibility matrix</h2>",
        f'<p class="muted">{compat["counts"]["games"]} titles from <code>{html.escape(compat["source"])}</code>: '
        f'{compat["counts"]["boots"]} boot to frames, {compat["counts"]["compiled"]} compile. '
        "The matrix is produced by <code>tools/discovery/batch.py</code> and stays outside the repository.</p>",
        '<table><tr><th>Game</th><th>Status</th><th class="n">Functions</th><th class="n">Instructions</th>'
        '<th class="n">Frames</th><th class="n">FPS</th><th class="n">Gaps</th><th>Note</th></tr>',
    ]
    for g in compat["games"]:
        cls = {"boots": "done", "compiled": "", "crash": "missing", "compile_error": "missing", "aarch32": "missing"}.get(g["status"], "")
        note = html.escape(str(g["note"]))
        note = note.replace("|", "&#124;")
        rows.append(
            f'<tr><td>{html.escape(g["game"])}</td><td class="{cls}">{html.escape(g["status"])}</td>'
            f'<td class="n">{html.escape(str(g["functions"]))}</td><td class="n">{html.escape(str(g["instructions"]))}</td>'
            f'<td class="n">{html.escape(str(g["frames"]))}</td><td class="n">{html.escape(str(g["fps"]))}</td>'
            f'<td class="n">{html.escape(str(g["gaps"]))}</td><td>{note}</td></tr>'
        )
    rows.append("</table>")
    return rows

def report_workstreams(report):
    rows = report.get("workstreams") or []
    if not rows:
        return []
    done = sum(r["done"] for r in rows)
    total = sum(r["items"] for r in rows)
    out = [
        "<h2>Workstreams</h2>",
        f'<p class="muted">{done} of {total} tracked items done, from '
        '<a href="../plan/STATUS.md"><code>docs/plan/STATUS.md</code></a>.</p>',
        '<table><tr><th>WS</th><th class="n">Items</th><th class="n">Done</th><th class="n">Partial</th>'
        '<th class="n">Open</th><th>Status</th><th>Last item</th></tr>',
    ]
    for w in rows:
        open_items = w["pending"] + w.get("confirmed", 0)
        width = 100 * (w["done"] + 0.5 * w["partial"]) / w["items"] if w["items"] else 0
        out.append(
            f'<tr><td>{html.escape(w["ws"])}</td><td class="n">{w["items"]}</td><td class="n">{w["done"]}</td>'
            f'<td class="n">{w["partial"]}</td><td class="n">{open_items}</td>'
            f'<td><span class="bar"><i style="width:{width:.0f}%"></i></span>{html.escape(w["state"])}</td>'
            f'<td>{html.escape(w.get("latest", ""))}</td></tr>'
        )
    out.append("</table>")
    return out

def report_shader_breakdown(report):
    stages, families = report.get("shader_stages") or [], report.get("shader_families") or []
    if not stages and not families:
        return []
    out = ['<h2>Shaders by stage and family</h2>']
    for title, rows, extra in (
        ("Stage", stages, "Shared mnemonics count in every stage; a stage-only mnemonic counts in its stage."),
        ("Family", families, "Atomics first, then textures, control flow and the rest as ALU."),
    ):
        out.append(f'<h3>{title}</h3><p class="muted">{extra}</p>')
        out.append('<table><tr><th>Name</th><th class="n">Done</th><th class="n">Total</th><th class="n">%</th></tr>')
        for r in rows:
            width = 100 * r["implemented"] / r["total"] if r["total"] else 0
            label = r["name"]
            if r.get("specific", [0, 0])[1]:
                label += f' (specific {r["specific"][0]}/{r["specific"][1]})'
            out.append(
                f'<tr><td>{html.escape(label)}</td><td class="n">{r["implemented"]}</td>'
                f'<td class="n">{r["total"]}</td><td class="n"><span class="bar"><i style="width:{width:.0f}%"></i></span>'
                f'{pct(r["implemented"], r["total"])}</td></tr>'
            )
        out.append("</table>")
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
    sections, shader = [], []
    for sid, title, unit in SECTIONS:
        if sid == "shaders":
            shader = shader_groups()
            done = sum(len(g["implemented"]) for g in shader)
            total = sum(g["total"] for g in shader)
            sections.append({"id": sid, "title": title, "unit": unit, "implemented": done, "total": total, "groups": shader})
            continue
        if sid == "nvn":
            section = nvn_section()
            if section:
                sections.append(section)
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
                # No reference list for this group: its arms are handled but cannot be called
                # implemented against a documented surface, so they are reported separately
                # instead of counting themselves as 100 %.
                impl, trivial, extra = {}, [], sorted(arms)
            total = documented if documented is not None else {}
            silent = key[1].split("/")[0].split(":")[0] in SILENT_STUBS
            if silent:
                trivial = list(impl)  # handled, but only silence / fixed values
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
    # `implemented` counts every handled command; `real` subtracts the silent stubs, which is
    # what the badges, the treemap and the console count. Both live in progress.json so a reader
    # never has to guess which of the two a number means.
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
    report["workstreams"] = workstream_rows()
    return report

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
    done = implemented_real(section)
    pct = 100 * done / section["total"] if section["total"] else 0
    out.append(
        f'<text x="{x0 + 4}" y="{y0 + 17}" font-size="16" fill="{COLORS["text"]}">'
        f'{html.escape(section["title"])}: {pct:.1f}% ({done}/{section["total"]})</text>'
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
        "silent": sum(s.get("silent", 0) for s in system),
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
td.done{color:#2ea043;font-weight:600}td.missing{color:#f85149}
@media (max-width:640px){details ul{columns:1}td:nth-child(2),th:nth-child(2){display:none}}
"""

def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "—"

def implemented_real(section):
    """Commands that do the real work: silent stubs are handled but count as stubs."""
    return section["implemented"] - section.get("silent", 0)

def section_table(section):
    done_section = implemented_real(section)
    silent_note = ""
    if section.get("silent"):
        silent_note = (
            f' {section["silent"]} more are handled by a silent stub (it answers but returns no real data) '
            "and are not counted as implemented."
        )
    if section["id"] == "nvn":
        what = "NVN procs named by the runtime"
        note = (
            ' No external NVN reference list: the total is the surface these modules name, and a proc counts as '
            "implemented when the host takes it over."
        )
    else:
        what = f'documented {section["unit"]}'
        note = ""
    rows = [
        f'<h2>{html.escape(section["title"])}</h2>',
        f'<p class="muted">{done_section} of {section["total"]} {what} ({pct(done_section, section["total"])}).{silent_note}{note}</p>',
        f'<table><tr><th>Interface</th><th>{"Group" if section["id"] == "nvn" else "Official name"}</th>'
        '<th class="n">Done</th><th class="n">Constant</th><th class="n">Total</th><th class="n">%</th></tr>',
    ]
    fam = None
    hexa = section["id"] in ("svc", "ioctl")
    for g in section["groups"]:
        if section["id"] == "services" and g["family"] != fam:
            fam = g["family"]
            rows.append(f'<tr class="fam"><td colspan="6">{html.escape(fam)}</td></tr>')
        silent = g.get("silent", 0)
        done, total = len(g["implemented"]) - silent, g["total"]
        official = html.escape(g["official"])
        if g["url"]:
            official = f'<a href="{html.escape(g["url"])}">{official or "switchbrew"}</a>'
        width = 100 * done / total if total else 0
        name = f'<code>{html.escape(g["name"])}</code>'
        if silent:
            name += f'<div class="muted">silent stub: {html.escape(SILENT_STUBS.get(g["name"].split("/")[0].split(":")[0], "handled, no real data"))}</div>'
        missing = ""
        if g["missing"]:
            items = "".join(f'<li><code>{(hex(m["id"]) if hexa else m["id"])}</code> {html.escape(m["name"])}</li>' for m in g["missing"])
            missing = f'<details><summary>{len(g["missing"])} missing</summary><ul>{items}</ul></details>'
        if section["id"] == "shaders":
            pass
        elif section["id"] == "nvn":
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

COUNTED_SECTIONS = ("services", "svc", "ioctl", "shaders")

def render_html(report):
    counted = [s for s in report["sections"] if s["id"] in COUNTED_SECTIONS]
    total_done = sum(implemented_real(s) for s in counted)
    total_all = sum(s["total"] for s in counted)
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>kirikae progress</title><style>{STYLE}</style></head><body>",
        "<h1>Kirikae progress</h1>",
        '<img src="progress.svg" alt="Treemap: each square is a command, SVC, ioctl or shader instruction; green = implemented, gray = missing">',
        f'<p>Generated by <code>tools/progress.py</code> on {report["generated"]}. Raw numbers: <a href="progress.json">progress.json</a>. '
        f'<a href="history.html">History</a> (one point a day). In total: {total_done} of {total_all} ({pct(total_done, total_all)}).</p>',
    ]
    parts += report_workstreams(report)
    if "coverage" in report:
        parts.append(coverage_section(report))
    for s in report["sections"]:
        parts.append(section_table(s))
        if s["id"] == "shaders":
            parts += report_shader_breakdown(report)
    parts += report_compat(report)
    parts.append(
        "<h2>How it is counted</h2><p>A command counts as implemented when the runtime has its own arm for it in the "
        "<code>match</code> of the service (<code>crates/rt/src/services/**/*.rs</code>), the SVC (<code>horizon.rs</code>) or the "
        "device ioctl (<code>services/nvdrv.rs</code> and <code>services/nvdrv/*.rs</code>); anything that falls to the default arm "
        "goes to the stub, which answers with zeros and warns. A service split into a folder counts like a single file: the "
        "scanner follows the inherent <code>impl</code> and the free function a <code>call</code> delegates to across the module. "
        "The <em>Constant</em> column sets apart implemented commands whose body only returns success or fixed values. "
        "Ioctls are compared by the low byte of their number, which is how the devices dispatch them. "
        "Services marked as a <em>silent stub</em> (for example <code>hwopus</code> or <code>nfp</code>) answer every command "
        "but return zeros or fixed values; they are listed as light green and are not counted as implemented. Those commands "
        "are handled but excluded from the counted total, and <code>progress.json</code> keeps both numbers "
        "(<code>implemented</code>, <code>silent</code> and <code>real</code>) in every section and in its <code>totals</code>.</p>"
        "<p>The totals are the commands documented on <a href=\"https://switchbrew.org\">Switchbrew</a> "
        "(<code>tools/reference/horizon.txt</code>, IDs and names only) for the interfaces Kirikae exposes or games ask for, "
        "not every system service. A new interface joins the count when it joins the reference.</p>"
        "<p>NVN procs are the <code>nvn*</code> names in <code>crates/rt/src/nvn/*.rs</code>. An NVN proc is "
        "<em>implemented</em> when <code>host::proc</code> takes it over (object, state or recording work) or "
        "<code>state::setter</code> decodes it; it is <em>missing</em> when only <code>shadow::hook</code>, "
        "<code>resources::call</code> or the not-recorded list names it, which leaves it on the recompiled driver path. "
        "The device handle getters are <em>constant</em>: their arm only packs two ids. There is no external NVN "
        "reference list yet, so the total is the surface these modules name, not the whole NVN API.</p>"
        "<p>Shader instructions count the mnemonics of the Maxwell opcode table generated from envytools "
        "(<code>crates/maxwell/src/opcodes.rs</code>) that have their own arm in the SPIR-V translator "
        "(<code>crates/maxwell/src/translate.rs</code>). A mnemonic counts whole even when only some of its "
        "variants are translated. The opcode table carries no stage column, so the stage table in "
        "<code>tools/progress.py</code> lists the mnemonics only one stage encodes (fragment: <code>ipa</code>, "
        "<code>pixld</code>, <code>kil</code>, <code>out</code>; compute: <code>bar</code>, <code>membar</code>, "
        "<code>cctl</code>, <code>cctll</code>, <code>cctlt</code>); the rest count in every stage. Families are the "
        "same mnemonics grouped as atomics, textures, control flow and ALU.</p>"
        "<p>The compatibility matrix is optional: <code>--compat</code> (or <code>KIRIKAE_COMPAT</code>) points at the "
        "<code>compat.json</code>/<code>compat.md</code> that <code>tools/discovery/batch.py</code> writes outside the "
        "repository, and the section is skipped when the file is absent.</p></body></html>"
    )
    return "\n".join(parts)

def update_history(path, report):
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    point = {"date": report["generated"], **{s["id"]: [implemented_real(s), s["total"]] for s in report["sections"]}}
    history = [p for p in history if p["date"] != point["date"]] + [point]
    history.sort(key=lambda p: p["date"])
    path.write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    return history

def render_history(history):
    W, H, pad = 900, 300, 40
    colors = {"services": "#2ea043", "svc": "#4493f8", "ioctl": "#d29922", "nvn": "#db6d28", "shaders": "#a371f7"}
    names = {sid: title for sid, title, _ in SECTIONS}
    n = len(history)
    xs = [pad + (W - 2 * pad) * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Progress per day">']
    for p in (0, 25, 50, 75, 100):
        y = H - pad - (H - 2 * pad) * p / 100
        svg.append(f'<line x1="{pad}" x2="{W - pad}" y1="{y}" y2="{y}" stroke="currentColor" stroke-opacity=".15"/>')
        svg.append(f'<text x="{pad - 6}" y="{y + 4}" font-size="11" text-anchor="end" fill="currentColor">{p}%</text>')
    for sid, color in colors.items():
        pts = [
            (xs[i], H - pad - (H - 2 * pad) * (p[sid][0] / p[sid][1] if p[sid][1] else 0))
            for i, p in enumerate(history)
            if p.get(sid)
        ]
        if len(pts) > 1:
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

    nvn = """pub fn proc(name: &str) -> Option<Proc> {
        Some(match name.strip_prefix("nvn")? {
            "BufferInitialize" => BufferInitialize,
            "DeviceGetTextureHandle" => Handle(HandleProc::Texture),
            "SamplerBuilderSetBorderColor" | "SamplerBuilderSetBorderColour" => SamplerBuilder(Field::BorderColor),
            _ => return state::setter(name).map(Proc::State),
        })
    }"""
    arms = named_arms(nvn, "proc")
    check(
        set(arms) == {
            "nvnBufferInitialize",
            "nvnDeviceGetTextureHandle",
            "nvnSamplerBuilderSetBorderColor",
            "nvnSamplerBuilderSetBorderColour",
        },
        "nvn proc arms",
    )
    check({n for n, a in arms.items() if a.startswith(NVN_CONSTANT_PREFIX)} == {"nvnDeviceGetTextureHandle"}, "nvn constant arms")
    state = """pub fn setter(name: &str) -> Option<Setter> {
        Some(match name.strip_prefix("nvn")? {
            "ChannelMaskStateSetDefaults" => ChannelMaskDefaults,
            _ => return None,
        })
    }"""
    check(set(named_arms(state, "setter")) == {"nvnChannelMaskStateSetDefaults"}, "nvn state setters")
    check(nvn_family("nvnTexturePoolRegisterTexture") == "pools", "nvn family")
    check(nvn_family("nvnCommandBufferDrawArrays") == "commands", "nvn family commands")

    record = {
        "game": "Demo",
        "status": "boots",
        "functions": 12,
        "instructions": 34,
        "reachable_total": 56,
        "frames": 90,
        "fps": 29.9,
        "gaps": 3,
        "note": "ok",
    }
    rows = compat_json_rows(json.dumps([record]))
    check(rows == [compat_record(record)] and rows[0]["instructions"] == "34/56", "compat json")
    md = "\n".join(
        [
            "# Kirikae compatibility matrix",
            "",
            "| Game | Status | Functions | Instructions | Frames | FPS | Gaps | Note |",
            "|---|---|---:|---|---:|---:|---:|---|",
            "| Demo | boots | 12 | 34/56 | 90 | 29.9 | 3 | ok |",
        ]
    )
    check(compat_markdown_rows(md) == [{k: str(v) for k, v in r.items()} for r in rows], "compat markdown")

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
            # `foo.rs` keeps its inherent impl in `foo/command.rs` and its dispatch table in
            # `foo/table.rs`; the module is still one service.
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

    report = build_report()
    nvn = next((s for s in report["sections"] if s["id"] == "nvn"), None)
    check(nvn is not None and nvn["total"] > 0, "nvn section")
    if nvn:
        check(nvn["implemented"] + sum(len(g["missing"]) for g in nvn["groups"]) == nvn["total"], "nvn partition")
    check(report.get("shader_stages") and report.get("shader_families"), "shader breakdown in the report")
    check(report.get("workstreams"), "workstreams in the report")

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

    with tempfile.TemporaryDirectory(prefix="kirikae-progress-") as tmp:
        (Path(tmp) / "compat.json").write_text(json.dumps([record]), encoding="utf-8")
        resolved = resolve_compat(tmp)
        check(resolved is not None and load_compat(resolved)["counts"]["games"] == 1, "compat artifact is resolved when it exists")
    check("[skip ci]" in PROGRESS_COMMIT, "ci commit cannot retrigger the workflow")

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
    if report.get("workstreams"):
        done = sum(w["done"] for w in report["workstreams"])
        total = sum(w["items"] for w in report["workstreams"])
        lines.append(f"workstreams: {done}/{total} tracked items done")
    compat = report.get("compat")
    if compat:
        lines.append(
            f'compatibility: {compat["counts"]["games"]} titles '
            f'({compat["counts"]["boots"]} boot, {compat["counts"]["compiled"]} compile)'
        )
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
    labels = {"services": "services", "svc": "SVCs", "ioctl": "nvdrv ioctls", "nvn": "NVN host", "shaders": "shader ops"}
    for s in report["sections"]:
        done = implemented_real(s)
        ratio = done / s["total"] if s["total"] else 0
        color = "2ea043" if ratio >= 0.5 else "d29922" if ratio >= 0.3 else "db6d28"
        badge = {"schemaVersion": 1, "label": labels.get(s["id"], s["id"]), "message": f'{done}/{s["total"]} ({pct(done, s["total"])})', "color": color}
        (badges / f'{s["id"]}.json').write_text(json.dumps(badge) + "\n", encoding="utf-8")
    history = update_history(out / "history.json", report)
    (out / "history.html").write_text(render_history(history), encoding="utf-8")
    return out


PROGRESS_COMMIT = "chore(progress): regenerate the dashboard [skip ci]"
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


def git(*args):
    return subprocess.run(["git", *args], cwd=str(ROOT), text=True, capture_output=True)


def ci(out, dry_run=False, push=False, compat_path=None):
    """The entry point of .github/workflows/progress.yml.

    Regenerates the dashboard, commits it when it changed and (with --push) pushes the commit.
    --dry-run writes the files and reports what would be committed, without touching git.
    """
    report = build_report()
    compat = load_compat(resolve_compat(compat_path))
    if compat:
        report["compat"] = compat
    out = write_outputs(out, report)
    lines = summary_lines(report)
    for line in lines:
        print(line)
    print(f"wrote {out}")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as f:
                f.write("### Progress dashboard\n\n```\n" + "\n".join(lines) + "\n```\n")
        except OSError as e:
            print(f"note: cannot write the step summary: {e}", file=sys.stderr)

    try:
        rel = out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        print(f"ci: {out} is outside the repository; nothing to commit")
        return 0
    status = git("status", "--porcelain", "--", rel)
    if status.returncode != 0:
        print(f"ci: git status failed: {status.stderr.strip()}", file=sys.stderr)
        return status.returncode
    changed = [line for line in status.stdout.splitlines() if line.strip()]
    if not changed:
        print(f"ci: {rel} is up to date")
        return 0
    for line in changed:
        print(f"ci: {line}")
    if dry_run:
        print(f'ci: dry run, would commit {len(changed)} changed file(s) as "{PROGRESS_COMMIT}"')
        return 0
    add = git("add", "--", rel)
    if add.returncode != 0:
        print(f"ci: git add failed: {add.stderr.strip()}", file=sys.stderr)
        return add.returncode
    # The workflow documents why this identity can write: `permissions: contents: write`.
    commit = git(
        "-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}",
        "commit", "-m", PROGRESS_COMMIT, "--", rel,
    )
    if commit.returncode != 0:
        print(f"ci: git commit failed: {commit.stderr.strip()}", file=sys.stderr)
        return commit.returncode
    print(f"ci: committed {len(changed)} file(s)")
    if push:
        pushed = git("push")
        if pushed.returncode != 0:
            # actions/checkout can leave a detached HEAD; push the commit to its branch then.
            branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            if branch and branch != "HEAD":
                pushed = git("push", "origin", f"HEAD:{branch}")
        if pushed.returncode != 0:
            print(f"ci: git push failed: {pushed.stderr.strip()}", file=sys.stderr)
            return pushed.returncode
        print("ci: pushed")
    return 0


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
    ap.add_argument(
        "--compat",
        metavar="FILE_OR_DIR",
        help="batch compatibility matrix (compat.json or compat.md), or the folder holding it; "
        "defaults to KIRIKAE_COMPAT or the discovery loop's work folder",
    )
    ap.add_argument("--self-test", action="store_true", help="check the parsers against synthetic sources and exit")
    ap.add_argument(
        "--ci",
        action="store_true",
        help="workflow entry point: regenerate docs/progress and commit it when it changed",
    )
    ap.add_argument("--dry-run", action="store_true", help="with --ci: regenerate and report, but do not commit or push")
    ap.add_argument("--push", action="store_true", help="with --ci: push the commit to the current branch")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    if args.self_test:
        sys.exit(self_test())
    if args.dry_run or args.push:
        if not args.ci:
            ap.error("--dry-run and --push are only valid with --ci")
    if args.ci and args.coverage:
        ap.error("--ci regenerates docs/progress; it does not take --coverage")
    out = Path(args.out or ROOT / ("docs/progress-games" if args.coverage else "docs/progress"))
    if args.ci:
        sys.exit(ci(out, dry_run=args.dry_run, push=args.push, compat_path=args.compat))
    report = build_report()
    compat = load_compat(resolve_compat(args.compat))
    if compat:
        report["compat"] = compat
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
