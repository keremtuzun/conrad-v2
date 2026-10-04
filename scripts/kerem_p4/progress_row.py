"""One markdown row of live progress for the current Kerem-trainer run (reads ~/p4logs)."""
import datetime, json, os
os.chdir(os.path.expanduser("~/conrad-v2"))
L = os.path.expanduser("~/p4logs")
rid = open(f"{L}/run_id").read().strip()
rows = [json.loads(l) for l in open(f"artifacts/runs/{rid}/metrics.jsonl")]
r = rows[-1]
w = json.loads(open(f"{L}/watchdog-{rid}.jsonl").read().splitlines()[-1])
tr = datetime.timezone(datetime.timedelta(hours=3))
total = int(open(f"{L}/run_steps").read())
fin = "?"
if w.get("proj_total_h"):
    fin = f"{datetime.datetime.fromtimestamp(w['t'] - w['elapsed_h'] * 3600 + w['proj_total_h'] * 3600, tr):%a %d %b %H:%M}"
log = open(f"{L}/orchestrate.log").read()
status = "finished" if "ALL_DONE" in log else ("stopped: " + str(w.get("reason")) if w.get("action") == "TERMINATE" else "training")
src = " ".join(f"{k.split('/')[2]} {v:.0f}" for k, v in r.items() if k.startswith("val/imaging_sonar/") and k.count("/") == 3 and k.endswith("effective_rank"))
print(f"| {datetime.datetime.now(tr):%a %d %b %H:%M} | {r['step']:,} ({r['step'] / total * 100:.1f}%) | {r.get('steps_per_s', 0):.2f} | "
      f"{r['val/imaging_sonar/effective_rank']:.1f} / {r['val/imaging_sonar/nn_temporal_hit']:.2f} | "
      f"{r['val/rgb_camera/effective_rank']:.1f} / {r['val/rgb_camera/nn_temporal_hit']:.2f} | {src} | {fin} | "
      f"{w['spent_try_so_far']:.0f} / {(w.get('proj_cost_try') or 0):.0f} | {status} |")
