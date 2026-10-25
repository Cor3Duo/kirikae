import re
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

SHADER_STAGES = {
    "vertex": (),
    "fragment": ("ipa", "pixld", "kil", "out"),
    "compute": ("bar", "membar", "cctl", "cctll", "cctlt"),
}

SHADER_FAMILIES = {
    "atomics": "atom atoms atomcas red sured suatom",
    "texture": "tex texs tld tlds tld4 tld4s txq txd txa tmml tmmls suld sust isberd",
    "control": "bra brx ssy sync pbk brk cont pcnt exit kil bar membar depbar nop cal jcal ret jmp jmx pret sam rtt bpt preexit ide",
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

SERVICE_DISPATCH = r"match (?:req\.cmd|\(self\.name, req\.cmd\)|cmd)\s*\{"
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
            defaults = [lit.group(1)] if lit else names_of.get(struct, [])
            if not defaults:
                continue
            scope = module_scope(path, sources)
            arms = arms_in(block, SERVICE_DISPATCH, consts_of[path])
            merge_arms(arms, command_arms(scope, sources, consts_of, struct))
            merge_arms(arms, delegated_arms(block, path, sources, consts_of, struct))
            for iface, ids in arms.items():
                for default in defaults if iface is None else [iface]:
                    found.setdefault(("services", default), {}).update(ids)
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

def read_reference(path=None):
    ref, cur = {}, None
    REF_NAMES.clear()
    path = Path(path or REFERENCE)
    if not path.exists():
        return ref
    for raw in path.read_text(encoding="utf-8").splitlines():
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

def read_text(path):
    return path.read_text(encoding="utf-8") if path.is_file() else ""

def family(section, iface):
    if section == "svc":
        return "SVC"
    if section == "ioctl":
        return NV_SHORT.get(iface, iface)
    base = iface.split("/")[0]
    base = FAMILY.get(base, base)
    return base.split(":")[0] or base

REF_NAMES = {}
