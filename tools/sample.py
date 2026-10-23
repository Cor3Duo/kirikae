import ctypes, ctypes.wintypes as W, sys, time, json
pid = int(sys.argv[1]); secs = float(sys.argv[2]); out = open(sys.argv[3], "w")
hz = float(sys.argv[4]) if len(sys.argv) > 4 else 300
depth = int(sys.argv[5]) if len(sys.argv) > 5 else 24
prefix = sys.argv[6] if len(sys.argv) > 6 else ""
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
dbg = ctypes.WinDLL("dbghelp", use_last_error=True)
k32.OpenProcess.restype = W.HANDLE; k32.OpenThread.restype = W.HANDLE; k32.CreateToolhelp32Snapshot.restype = W.HANDLE
class TE(ctypes.Structure):
    _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("tid", W.DWORD), ("pid", W.DWORD), ("base", W.LONG), ("delta", W.LONG), ("flags", W.DWORD)]
class LINE(ctypes.Structure):
    _fields_ = [("size", W.DWORD), ("key", ctypes.c_void_p), ("line", W.DWORD), ("file", ctypes.c_char_p), ("addr", ctypes.c_uint64)]
hp = k32.OpenProcess(0x1F0FFF, False, pid)
dbg.SymSetOptions(0x2 | 0x10)
dbg.SymInitialize.argtypes = [W.HANDLE, ctypes.c_char_p, W.BOOL]
t0 = time.time(); assert dbg.SymInitialize(hp, None, True); print(f"symbols loaded in {time.time()-t0:.1f}s", flush=True)
k32.GetThreadDescription.argtypes = [W.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
dbg.SymFromAddr.argtypes = [W.HANDLE, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
dbg.SymGetLineFromAddr64.argtypes = [W.HANDLE, ctypes.c_uint64, ctypes.POINTER(W.DWORD), ctypes.POINTER(LINE)]
def threads():
    snap = k32.CreateToolhelp32Snapshot(4, 0); te = TE(); te.size = ctypes.sizeof(TE); res = []
    ok = k32.Thread32First(W.HANDLE(snap), ctypes.byref(te))
    while ok:
        if te.pid == pid: res.append(te.tid)
        ok = k32.Thread32Next(W.HANDLE(snap), ctypes.byref(te))
    k32.CloseHandle(W.HANDLE(snap)); return res
cache = {}
def name(a):
    if a in cache: return cache[a]
    buf = ctypes.create_string_buffer(88 + 512)
    ctypes.c_uint32.from_buffer(buf, 0).value = 88; ctypes.c_uint32.from_buffer(buf, 80).value = 500
    disp = ctypes.c_uint64()
    r = buf.raw[84:84 + ctypes.c_uint32.from_buffer(buf, 76).value].decode(errors="replace") if dbg.SymFromAddr(hp, a, ctypes.byref(disp), buf) else f"?{a>>16:x}"
    ln = LINE(); ln.size = ctypes.sizeof(LINE); d = W.DWORD()
    if dbg.SymGetLineFromAddr64(hp, a, ctypes.byref(d), ctypes.byref(ln)) and ln.file:
        f = ln.file.decode(errors="replace").replace("\\", "/")
        r += "@" + "/".join(f.split("/")[-2:]) + f":{ln.line}"
    cache[a] = r; return r
dbg.StackWalkEx.argtypes = [W.DWORD, W.HANDLE, W.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, W.DWORD]
dbg.SymFromInlineContext.argtypes = [W.HANDLE, ctypes.c_uint64, W.DWORD, ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
dbg.SymGetLineFromInlineContext.argtypes = [W.HANDLE, ctypes.c_uint64, W.DWORD, ctypes.c_uint64, ctypes.POINTER(W.DWORD), ctypes.POINTER(LINE)]
icache = {}
def iname(a, ictx):
    k = (a, ictx)
    if k in icache: return icache[k]
    buf = ctypes.create_string_buffer(88 + 512)
    ctypes.c_uint32.from_buffer(buf, 0).value = 88; ctypes.c_uint32.from_buffer(buf, 80).value = 500
    disp = ctypes.c_uint64()
    r = buf.raw[84:84 + ctypes.c_uint32.from_buffer(buf, 76).value].decode(errors="replace") if dbg.SymFromInlineContext(hp, a, ictx, ctypes.byref(disp), buf) else f"?{a>>16:x}"
    ln = LINE(); ln.size = ctypes.sizeof(LINE); d = W.DWORD()
    if dbg.SymGetLineFromInlineContext(hp, a, ictx, 0, ctypes.byref(d), ctypes.byref(ln)) and ln.file:
        f = ln.file.decode(errors="replace").replace("\\", "/")
        r += "@" + "/".join(f.split("/")[-2:]) + f":{ln.line}"
    icache[k] = r; return r
fta = ctypes.cast(dbg.SymFunctionTableAccess64, ctypes.c_void_p).value
gmb = ctypes.cast(dbg.SymGetModuleBase64, ctypes.c_void_p).value
handles = {}; names = {}
def th(tid):
    if tid not in handles:
        h = k32.OpenThread(0x1FFFFF, False, tid); handles[tid] = h
        p = ctypes.c_wchar_p()
        names[tid] = p.value if k32.GetThreadDescription(W.HANDLE(h), ctypes.byref(p)) == 0 and p.value else str(tid)
    return handles[tid]
raw = ctypes.create_string_buffer(1232 + 16); ctx = ctypes.addressof(raw) + ((-ctypes.addressof(raw)) % 16)
sf = ctypes.create_string_buffer(1024)
WAITS = ("WaitOnAddress", "NtWaitFor", "NtDelayExecution", "ZwWaitFor", "NtRemoveIoCompletion", "NtUserMsgWaitForMultipleObjectsEx", "RtlSleepConditionVariable", "NtAlertThreadByThreadId", "ZwAlertThreadByThreadId", "NtWaitForAlertByThreadId", "ZwWaitForAlertByThreadId", "NtUserGetMessage")
n = 0; end = time.time() + secs; period = 1 / hz; tids = []; refreshed = 0
while time.time() < end:
    t1 = time.time()
    if t1 - refreshed > 2:
        allt = threads(); tids = [t for t in allt if (th(t), names[t].startswith(prefix))[1]]
        if not refreshed: print(len(allt), "threads", sorted(set(names[t] for t in allt))[:12], flush=True)
        refreshed = t1
    for tid in tids:
        h = th(tid)
        if k32.SuspendThread(W.HANDLE(h)) == 0xFFFFFFFF: continue
        ctypes.c_uint32.from_address(ctx + 0x30).value = 0x10000B
        if not k32.GetThreadContext(W.HANDLE(h), ctypes.c_void_p(ctx)):
            k32.ResumeThread(W.HANDLE(h)); continue
        rip = ctypes.c_uint64.from_address(ctx + 0xF8).value
        frames = []
        top = name(rip)
        waiting = any(w in top for w in WAITS)
        if not waiting:
            ctypes.memset(sf, 0, 1024)
            for o, v in ((0, rip), (32, ctypes.c_uint64.from_address(ctx + 0xA0).value), (48, ctypes.c_uint64.from_address(ctx + 0x98).value)):
                ctypes.c_uint64.from_buffer(sf, o).value = v; ctypes.c_uint32.from_buffer(sf, o + 12).value = 3
            ctypes.c_uint32.from_buffer(sf, 264).value = 272
            for _ in range(depth):
                if not dbg.StackWalkEx(0x8664, hp, h, sf, ctypes.c_void_p(ctx), None, ctypes.c_void_p(fta), ctypes.c_void_p(gmb), None, 0): break
                pc = ctypes.c_uint64.from_buffer(sf, 0).value
                if pc == 0: break
                frames.append((pc, ctypes.c_uint32.from_buffer(sf, 268).value))
        k32.ResumeThread(W.HANDLE(h))
        if waiting:
            out.write(json.dumps({"t": t1, "th": names[tid], "f": ["<wait>"]}) + "\n"); continue
        if not frames: continue
        n += 1
        out.write(json.dumps({"t": t1, "th": names[tid], "f": [iname(a, c) for a, c in frames]}) + "\n")
    dt = time.time() - t1
    if dt < period: time.sleep(period - dt)
out.close()
print(f"{n} busy samples")
