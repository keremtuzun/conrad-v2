"""One markdown row of live progress for the current Kerem-trainer run (reads ~/p4logs)."""

import datetime
import json
import os
from pathlib import Path

os.chdir(os.path.expanduser("~/conrad-v2"))
L = Path("~/p4logs").expanduser()
rid = (L / "run_id").read_text().strip()
rows = [json.loads(line) for line in Path(f"artifacts/runs/{rid}/metrics.jsonl").read_text().splitlines()]
r = rows[-1]
watch = [json.loads(line) for line in (L / f"watchdog-{rid}.jsonl").read_text().splitlines()]
w = next((x for x in reversed(watch) if "spent_try_so_far" in x), watch[-1])
last = watch[-1]
tr = datetime.timezone(datetime.timedelta(hours=3))
total = int((L / "run_steps").read_text())
fin = "?"
if w.get("proj_total_h"):
    fin = f"{datetime.datetime.fromtimestamp(w['t'] - w['elapsed_h'] * 3600 + w['proj_total_h'] * 3600, tr):%a %d %b %H:%M}"
log = (L / "orchestrate.log").read_text()
status = (
    "finished"
    if "ALL_DONE" in log
    else ("stopped: " + str(last.get("reason")) if last.get("action") == "TERMINATE" else "training")
)
src = " ".join(
    f"{k.split('/')[2]} {v:.0f}"
    for k, v in r.items()
    if k.startswith("val/imaging_sonar/") and k.count("/") == 3 and k.endswith("effective_rank")
)
print(
    f"| {datetime.datetime.now(tr):%a %d %b %H:%M} | {r['step']:,} ({r['step'] / total * 100:.1f}%) | {r.get('steps_per_s', 0):.2f} | "
    f"{r['val/imaging_sonar/effective_rank']:.1f} / {r['val/imaging_sonar/nn_temporal_hit']:.2f} | "
    f"{r['val/rgb_camera/effective_rank']:.1f} / {r['val/rgb_camera/nn_temporal_hit']:.2f} | {src} | {fin} | "
    f"{w.get('spent_try_so_far', 0):.0f} / {(w.get('proj_cost_try') or 0):.0f} | {status} |"
)
