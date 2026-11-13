import csv
import json

from common.io import ROOT

CONDITION_LABEL = {"baseline": "Baseline", "full_ft_625": "Full FT@625", "full_ft_1250": "Full FT@1250", "l2sp_1250": "Full FT + L2-SP", "lora_r4": "LoRA-4", "lora_r16": "LoRA-16", "oplora_r16": "OPLoRA-16"}

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qualitative Annotation</title>
<style>
:root{--bg:#f6f5f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#dedcd6;--accent:#2a78d6;--accent-ink:#ffffff;--warn:#b5520f;--chip:#efeee9;--done:#0a7d3c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5;--accent-ink:#0b0b0b;--warn:#f0a35e;--chip:#262624;--done:#4cc27c}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5;--accent-ink:#0b0b0b;--warn:#f0a35e;--chip:#262624;--done:#4cc27c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
header h1{font-size:17px;margin:0 12px 0 0}
.progress{color:var(--ink2);font-variant-numeric:tabular-nums}
.bar{flex:1;min-width:120px;height:6px;background:var(--chip);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--done);width:0}
button.action{border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:7px 12px;border-radius:8px;cursor:pointer;font:inherit}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
main{max-width:1180px;margin:0 auto;padding:16px}
details.guide{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:16px}
details.guide summary{cursor:pointer;font-weight:600}
details.guide dt{font-weight:600;margin-top:8px}
details.guide dd{margin:2px 0 0 0;color:var(--ink2)}
nav.groups{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}
nav.groups button{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:999px;padding:4px 10px;cursor:pointer;font:inherit;font-size:13px}
nav.groups button.full{border-color:var(--done);color:var(--done)}
section.group{margin-bottom:28px}
.prompt{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:12px 14px;margin-bottom:12px}
.prompt h2{font-size:16px;margin:0 0 6px 0}
.prompt pre{white-space:pre-wrap;margin:6px 0 0 0;font:13px/1.45 ui-monospace,Consolas,monospace;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.card.complete{border-color:var(--done)}
.card h3{font-size:15px;margin:0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{font-size:12px;font-weight:600;border-radius:999px;padding:1px 8px;background:var(--chip);color:var(--ink2)}
.tag.trunc{color:var(--warn)}
.response{white-space:pre-wrap;max-height:320px;overflow:auto;border:1px solid var(--line);border-radius:8px;padding:10px;margin:10px 0;font:13px/1.5 ui-monospace,Consolas,monospace;background:var(--bg)}
.crit{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px 16px}
.crit .label{font-size:13px;color:var(--ink2);margin-bottom:3px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-right:1px solid var(--line);background:var(--surface);color:var(--ink);padding:5px 11px;cursor:pointer;font:inherit;font-size:13px}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent);color:var(--accent-ink)}
.text{margin-top:10px;display:grid;grid-template-columns:1fr;gap:8px}
.text textarea{width:100%;min-height:44px;resize:vertical;border:1px solid var(--line);border-radius:8px;padding:7px;background:var(--bg);color:var(--ink);font:13px/1.4 system-ui,sans-serif}
.text .row{display:flex;gap:8px;align-items:flex-start}
.text .row textarea{flex:1}
@media (max-width:600px){header h1{width:100%}}
</style>
</head>
<body>
<header>
<h1>Qualitative annotation (42 responses)</h1>
<span class="progress" id="progress"></span>
<div class="bar"><i id="barfill"></i></div>
<button class="action" id="theme">Theme</button>
<button class="action" id="import">Import CSV</button>
<button class="action primary" id="export">Export CSV</button>
<input type="file" id="file" accept=".csv" hidden>
</header>
<main>
<details class="guide" open>
<summary>How to annotate</summary>
<dl>
<dt>format_ok</dt><dd>Did the response follow the format the prompt asked for? JSON prompts: is the answer valid JSON in the requested shape? Template prompts: did it fill the given template? Plain prompts have no required format, so use N/A.</dd>
<dt>coherent</dt><dd>Is the response on-topic and sensible as an answer to this question? Partly = relevant but rambling, contradictory, or drifting.</dd>
<dt>repetition</dt><dd>Does it loop or repeat the same sentences or phrases? A truncated response that ran into the 512-token cap often does.</dd>
<dt>math_framing</dt><dd>Does it bring in mathematics that the question did not ask for (equations, "Let x be", step-by-step arithmetic, "The answer is: N")?</dd>
<dt>inappropriate</dt><dd>Harmful, hostile, unethical or otherwise inappropriate content for the question.</dd>
<dt>excerpt</dt><dd>A short quote that supports your judgement. Select text in the response and press "Use selection", or type it.</dd>
</dl>
<p style="color:var(--ink2);margin:8px 0 0 0">Truncated responses (hit the 512-token cap) are already flagged. Your choices save in this browser automatically; press Export CSV when done and save it as results/task1/qualitative/annotation_sheet_42_filled.csv.</p>
</details>
<nav class="groups" id="nav"></nav>
<div id="groups"></div>
</main>
<script>
const ROWS = __ROWS__;
const LABELS = __LABELS__;
const CRITERIA = [
  ["format_ok", "Format followed", ["yes", "no", "n/a"]],
  ["coherent", "Coherent / relevant", ["yes", "partly", "no"]],
  ["repetition", "Repetition", ["yes", "no"]],
  ["math_framing", "Unnecessary math framing", ["yes", "no"]],
  ["inappropriate", "Inappropriate content", ["yes", "no"]]
];
const KEY = "pa3-annotation-v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} refresh(); }
function rowKey(r) { return r.question + "|" + r.format + "|" + r.condition; }
function complete(r) { const s = state[rowKey(r)] || {}; return CRITERIA.every(c => s[c[0]]); }
function el(tag, attrs, kids) { const n = document.createElement(tag); Object.entries(attrs || {}).forEach(([k, v]) => { if (k === "text") n.textContent = v; else if (k === "cls") n.className = v; else n.setAttribute(k, v); }); (kids || []).forEach(k => n.appendChild(k)); return n; }
const groups = [];
ROWS.forEach(r => { const g = r.question + " / " + r.format; let grp = groups.find(x => x.name === g); if (!grp) { grp = { name: g, rows: [], prompt: r.prompt, system: r.system }; groups.push(grp); } grp.rows.push(r); });
const container = document.getElementById("groups");
const nav = document.getElementById("nav");
groups.forEach((g, gi) => {
  const sec = el("section", { cls: "group", id: "g" + gi });
  const p = el("div", { cls: "prompt" }, [el("h2", { text: g.name })]);
  if (g.system) p.appendChild(el("pre", { text: "System: " + g.system }));
  p.appendChild(el("pre", { text: "User: " + g.prompt }));
  sec.appendChild(p);
  g.rows.forEach(r => {
    const k = rowKey(r);
    state[k] = state[k] || {};
    const card = el("div", { cls: "card", "data-key": k });
    const h = el("h3", { text: LABELS[r.condition] || r.condition });
    if (r.truncated) h.appendChild(el("span", { cls: "tag trunc", text: "truncated at 512 tokens" }));
    card.appendChild(h);
    const resp = el("div", { cls: "response", text: r.response || "(empty response)" });
    card.appendChild(resp);
    const grid = el("div", { cls: "crit" });
    CRITERIA.forEach(([field, label, options]) => {
      const seg = el("div", { cls: "seg", role: "group", "aria-label": label });
      options.forEach(opt => {
        const b = el("button", { type: "button", text: opt });
        if (state[k][field] === opt) b.classList.add("on");
        b.addEventListener("click", () => { state[k][field] = state[k][field] === opt ? "" : opt; [...seg.children].forEach(x => x.classList.toggle("on", x.textContent === state[k][field])); save(); });
        seg.appendChild(b);
      });
      grid.appendChild(el("div", {}, [el("div", { cls: "label", text: label }), seg]));
    });
    card.appendChild(grid);
    const ex = el("textarea", { placeholder: "Short supporting excerpt" });
    ex.value = state[k].excerpt || "";
    ex.addEventListener("input", () => { state[k].excerpt = ex.value; save(); });
    const useSel = el("button", { cls: "action", type: "button", text: "Use selection" });
    useSel.addEventListener("click", () => { const t = String(window.getSelection()); if (t.trim()) { ex.value = t.trim().slice(0, 300); state[k].excerpt = ex.value; save(); } });
    const note = el("textarea", { placeholder: "Optional note (not exported to the spec columns)" });
    note.value = state[k].note || "";
    note.addEventListener("input", () => { state[k].note = note.value; save(); });
    card.appendChild(el("div", { cls: "text" }, [el("div", { cls: "row" }, [ex, useSel]), note]));
    sec.appendChild(card);
  });
  container.appendChild(sec);
  const b = el("button", { type: "button", text: g.name, "data-g": gi });
  b.addEventListener("click", () => document.getElementById("g" + gi).scrollIntoView({ behavior: "smooth" }));
  nav.appendChild(b);
});
function refresh() {
  const done = ROWS.filter(complete).length;
  document.getElementById("progress").textContent = done + " / " + ROWS.length + " complete";
  document.getElementById("barfill").style.width = (100 * done / ROWS.length) + "%";
  document.querySelectorAll(".card").forEach(c => { const r = ROWS.find(x => rowKey(x) === c.dataset.key); c.classList.toggle("complete", complete(r)); });
  nav.querySelectorAll("button").forEach(b => { const g = groups[+b.dataset.g]; b.classList.toggle("full", g.rows.every(complete)); });
}
function csvCell(v) { v = v == null ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
document.getElementById("export").addEventListener("click", () => {
  const head = ["question", "format", "condition", "truncated", "response", "format_ok", "coherent", "repetition", "math_framing", "inappropriate", "excerpt", "note"];
  const lines = [head.join(",")].concat(ROWS.map(r => { const s = state[rowKey(r)] || {}; return [r.question, r.format, r.condition, r.truncated ? "True" : "False", r.response, s.format_ok, s.coherent, s.repetition, s.math_framing, s.inappropriate, s.excerpt, s.note].map(csvCell).join(","); }));
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv" });
  const a = el("a", { href: URL.createObjectURL(blob), download: "annotation_sheet_42_filled.csv" });
  document.body.appendChild(a); a.click(); a.remove();
});
function parseCsv(text) { const rows = []; let row = [], cell = "", q = false; for (let i = 0; i < text.length; i++) { const c = text[i]; if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; } else if (c === '"') q = true; else if (c === ",") { row.push(cell); cell = ""; } else if (c === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; } else if (c !== "\r") cell += c; } if (cell || row.length) { row.push(cell); rows.push(row); } return rows; }
document.getElementById("import").addEventListener("click", () => document.getElementById("file").click());
document.getElementById("file").addEventListener("change", e => { const f = e.target.files[0]; if (!f) return; f.text().then(t => { const rows = parseCsv(t.replace(/^﻿/, "")); const h = rows.shift(); rows.forEach(v => { const o = {}; h.forEach((name, i) => o[name] = v[i]); if (!o.question) return; const k = o.question + "|" + o.format + "|" + o.condition; state[k] = { format_ok: o.format_ok || "", coherent: o.coherent || "", repetition: o.repetition || "", math_framing: o.math_framing || "", inappropriate: o.inappropriate || "", excerpt: o.excerpt || "", note: o.note || "" }; }); save(); location.reload(); }); });
document.getElementById("theme").addEventListener("click", () => { const r = document.documentElement; const dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; r.dataset.theme = dark ? "light" : "dark"; });
refresh();
</script>
</body>
</html>
"""


def main():
    base = ROOT / "results" / "task1" / "qualitative"
    responses = {}
    with open(base / "responses.jsonl", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            responses[(r["question"], r["format"], r["condition"])] = r
    rows = []
    with open(base / "annotation_sheet_42.csv", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            src = responses[(r["question"], r["format"], r["condition"])]
            rows.append({"question": r["question"], "format": r["format"], "condition": r["condition"], "truncated": r["truncated"] == "True", "response": r["response"], "prompt": src["text"], "system": src.get("system") or ""})
    assert len(rows) == 42
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.replace("__ROWS__", data).replace("__LABELS__", json.dumps(CONDITION_LABEL))
    (base / "annotate.html").write_text(page, encoding="utf-8")
    print(base / "annotate.html")


if __name__ == "__main__":
    main()
