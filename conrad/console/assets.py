"""Inline CSS and JS for the static console page. No external resource is ever referenced.

The script only reads embedded JSON and toggles presentation. It has no network access (the page CSP is
``default-src 'none'``), no storage use and no way to issue a request or command.
"""

from __future__ import annotations

CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; "
    "connect-src 'none'; form-action 'none'; base-uri 'none'"
)

CSS = """
:root{--bg:#f6f7f8;--panel:#fff;--ink:#1d232a;--muted:#5b6570;--line:#d5dade;--ok:#1f7a3a;--warn:#9a6700;
--bad:#b42318;--obs:#0e7490;--inf:#6d28d9;--unk:#6b7280;--pred:#334155;--truth:#8a1c7c}
@media (prefers-color-scheme:dark){:root{--bg:#12161a;--panel:#1a2026;--ink:#e6e9ec;--muted:#9aa4ae;
--line:#2f3841;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--obs:#22d3ee;--inf:#a78bfa;--unk:#9ca3af;
--pred:#cbd5e1;--truth:#f0abfc}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
code,.mono,td.id{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;overflow-wrap:anywhere}
.banner{padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--line);font-weight:600}
.banner small{display:block;font-weight:400;color:var(--muted)}
.layout{display:grid;grid-template-columns:200px minmax(0,1fr) 320px;gap:0;min-height:100vh}
nav{position:sticky;top:0;align-self:start;padding:12px;max-height:100vh;overflow:auto}
nav a{display:block;color:var(--ink);text-decoration:none;padding:3px 6px;border-radius:4px}
nav a:hover{background:var(--line)}
main{padding:12px 16px;min-width:0}
aside{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:12px;
border-left:1px solid var(--line);background:var(--panel)}
section{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin:0 0 14px}
h2{font-size:16px;margin:4px 0 8px}h3{font-size:14px;margin:10px 0 6px}
table{border-collapse:collapse;width:100%;margin:4px 0 8px}
th,td{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.scroll{max-height:360px;overflow:auto}
.missing{color:var(--muted);font-style:italic}
.badge{display:inline-block;padding:0 6px;border:1.5px solid currentColor;border-radius:4px;font-size:12px;
font-weight:600;white-space:nowrap}
.ks-OBSERVED{color:var(--obs)}.ks-INFERRED{color:var(--inf)}.ks-UNKNOWN{color:var(--unk)}
.ks-PREDICTED{color:var(--pred);border-style:dashed}.ks-MIXED{color:var(--warn)}
.out-ACCEPTED{color:var(--ok)}.out-REJECTED{color:var(--bad)}.out-SENT_NO_ACK{color:var(--warn)}
.sev-WARNING{color:var(--warn)}.sev-ERROR,.sev-CRITICAL{color:var(--bad)}
.truth{border:2px dashed var(--truth)}.truth h2{color:var(--truth)}
.truth-label{color:var(--truth);font-weight:700}
.chart{width:100%;max-width:720px;height:auto;display:block}
.axis,.lane{stroke:var(--line);stroke-width:1}.tick,.axlabel{fill:var(--muted);font-size:10px}
.series{fill:none;stroke-width:2}.dot{stroke:none}
.c-ua{stroke:var(--warn);fill:var(--warn)}.c-ue{stroke:var(--inf);fill:var(--inf)}
.c-uc{stroke:var(--bad);fill:var(--bad)}.c-uo{stroke:var(--obs);fill:var(--obs)}
.c-est{stroke:var(--obs);fill:var(--obs)}.c-truth{stroke:var(--truth);fill:var(--truth)}
.c-traj{stroke:var(--muted);fill:var(--muted)}
.evtick{stroke:var(--ink);stroke-width:1}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted)}
.legend-item svg{vertical-align:middle;margin-right:4px}
.future{opacity:.35}
.tree div{white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.tree .obs{color:var(--obs);font-weight:700}
.controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.controls input[type=range]{flex:1;min-width:200px}
button,select{font:inherit;padding:2px 8px}
pre{white-space:pre-wrap;word-break:break-all;font-size:12px;margin:0}
.error-page{max-width:900px;margin:40px auto;padding:16px;border:2px solid var(--bad);border-radius:6px;
background:var(--panel)}
@media (max-width:1100px){.layout{grid-template-columns:1fr}nav,aside{position:static;max-height:none}}
"""

JS = r"""
(function(){
  "use strict";
  var ev = JSON.parse(document.getElementById("events-data").textContent);
  var prov = JSON.parse(document.getElementById("prov-data").textContent);
  var scrub = document.getElementById("scrub");
  var label = document.getElementById("scrub-label");
  var insp = document.getElementById("inspector-body");
  var timer = null;
  function show(i){
    if(!ev.length){ label.textContent = "no events present"; return; }
    i = Math.max(0, Math.min(ev.length - 1, Number(i)));
    scrub.value = String(i);
    var e = ev[i];
    label.textContent = "seq " + e[0] + " / " + (ev.length - 1) + " : " + e[1] + " (" + e[2] + ")";
    var nodes = document.querySelectorAll("[data-seq]");
    for (var k = 0; k < nodes.length; k++){
      var after = Number(nodes[k].getAttribute("data-seq")) > e[0];
      nodes[k].classList.toggle("future", after);
      nodes[k].title = after ? "after the replay cursor" : "";
    }
    insp.textContent = JSON.stringify({sequence:e[0], event_type:e[1], module:e[2], severity:e[3],
      availability:e[4], trace_id:e[5], measurement_time_ns:e[6], payload:e[7]}, null, 2);
  }
  function step(d){ show(Number(scrub.value) + d); }
  if (scrub){
    scrub.addEventListener("input", function(){ show(scrub.value); });
    document.getElementById("b-first").onclick = function(){ show(0); };
    document.getElementById("b-prev").onclick = function(){ step(-1); };
    document.getElementById("b-next").onclick = function(){ step(1); };
    document.getElementById("b-last").onclick = function(){ show(ev.length - 1); };
    document.getElementById("b-play").onclick = function(){
      if (timer){ clearInterval(timer); timer = null; this.textContent = "Play"; return; }
      this.textContent = "Pause";
      var self = this;
      timer = setInterval(function(){
        if (Number(scrub.value) >= ev.length - 1){ clearInterval(timer); timer = null; self.textContent = "Play"; }
        else { step(1); }
      }, 120);
    };
    show(ev.length - 1);
  }
  var sel = document.getElementById("prov-root");
  var tree = document.getElementById("prov-tree");
  function line(depth, text, cls){
    var d = document.createElement("div");
    d.textContent = new Array(depth + 1).join("  ") + (depth ? "└ " : "") + text;
    if (cls) d.className = cls;
    tree.appendChild(d);
  }
  function walk(rid, depth, seen){
    var n = prov.nodes[rid];
    if (!n){ line(depth, "MISSING record " + rid, "sev-ERROR"); return; }
    if (seen[rid]){ line(depth, "(already shown) " + n.t + " " + rid); return; }
    seen[rid] = true;
    line(depth, n.t + " " + n.op + " [" + n.m + "] record " + rid);
    if (n.t === "DIRECT_OBSERVATION"){
      n.s.forEach(function(s){ if (prov.obs[s]) line(depth + 1, "RAW OBSERVATION " + s + " (" + prov.obs[s] + ")", "obs"); });
    }
    n.p.forEach(function(p){ walk(p, depth + 1, seen); });
  }
  if (sel){
    sel.addEventListener("change", function(){
      tree.textContent = "";
      if (sel.value) walk(sel.value, 0, {});
    });
  }
})();
"""
