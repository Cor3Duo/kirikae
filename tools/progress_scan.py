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

REF_NAMES = {}
