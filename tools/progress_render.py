import html
import json
import math
from pathlib import Path

from progress_scan import (
    SECTIONS,
    SILENT_STUBS,
    family,
)

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
    what = f'documented {section["unit"]}'
    note = ""
    rows = [
        f'<h2>{html.escape(section["title"])}</h2>',
        f'<p class="muted">{done_section} of {section["total"]} {what} ({pct(done_section, section["total"])}).{silent_note}{note}</p>',
        f'<table><tr><th>Interface</th><th>Official name</th>'
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
    if "coverage" in report:
        parts.append(coverage_section(report))
    for s in report["sections"]:
        parts.append(section_table(s))
        if s["id"] == "shaders":
            parts += report_shader_breakdown(report)
    parts.append(
        "<h2>How it is counted</h2><p>A command counts as implemented when the runtime has its own arm for it in the "
        "<code>match</code> of the service (<code>crates/rt/src/services/**/*.rs</code>), the SVC (<code>horizon.rs</code>) or the "
        "device ioctl (<code>services/nvdrv.rs</code> and <code>services/nvdrv/*.rs</code>); anything that falls to the default arm "
        "goes to the stub, which answers with zeros and warns. A service split into a folder counts like a single file: the "
        "scanner follows the inherent <code>impl</code> and the free function a <code>call</code> delegates to across the module. "
        "A struct the registry constructs under several names (<code>bsd:u</code>, <code>bsd:s</code>, <code>bsd:a</code>) is "
        "one dispatch, so each name is counted against its own reference table with the same arms. "
        "The <em>Constant</em> column sets apart implemented commands whose body only returns success or fixed values. "
        "Ioctls are compared by the low byte of their number, which is how the devices dispatch them. "
        "Services marked as a <em>silent stub</em> (for example <code>hwopus</code> or <code>nfp</code>) answer every command "
        "but return zeros or fixed values; they are listed as light green and are not counted as implemented. Those commands "
        "are handled but excluded from the counted total, and <code>progress.json</code> keeps both numbers "
        "(<code>implemented</code>, <code>silent</code> and <code>real</code>) in every section and in its <code>totals</code>.</p>"
        "<p>The totals are the commands documented on <a href=\"https://switchbrew.org\">Switchbrew</a> "
        "(<code>tools/reference/horizon.txt</code>, IDs and names only) for the interfaces Kirikae exposes or games ask for, "
        "not every system service. A new interface joins the count when it joins the reference.</p>"
        "<p>Shader instructions count the mnemonics of the Maxwell opcode table generated from envytools "
        "(<code>crates/maxwell/src/opcodes.rs</code>) that have their own arm in the SPIR-V translator "
        "(<code>crates/maxwell/src/translate.rs</code>). A mnemonic counts whole even when only some of its "
        "variants are translated. The opcode table carries no stage column, so the stage table in "
        "<code>tools/progress.py</code> lists the mnemonics only one stage encodes (fragment: <code>ipa</code>, "
        "<code>pixld</code>, <code>kil</code>, <code>out</code>; compute: <code>bar</code>, <code>membar</code>, "
        "<code>cctl</code>, <code>cctll</code>, <code>cctlt</code>); the rest count in every stage. Families are the "
        "same mnemonics grouped as atomics, textures, control flow and ALU.</p></body></html>"
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
