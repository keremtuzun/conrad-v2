#!/usr/bin/env python3
"""External fail-closed watchdog for a V1.1 P4 run on one L4 (Kerem trainer).

Samples the run every POLL seconds and appends one JSON line to ~/p4logs/watchdog-<run>.jsonl. Hard stops (the trainer
is sent SIGTERM, which seals the run FAILED): GPU count != 1, non-finite loss, variance collapse, semantic collapse
(held-out temporal-neighbour hit < NN_COLLAPSE on 3 validations after step 5000), no new metrics for STALL_S,
checkpoint not advancing, wall clock >= MAX_HOURS, projected runtime > MAX_HOURS, projected cost > MAX_TRY.
Rank below RANK_FLOOR is reported; it only stops the run when RANK_STOP=1.
"""
import json, math, os, signal, subprocess, sys, time
from pathlib import Path

RUN_DIR = Path(sys.argv[1]); TRAINER_PID = int(sys.argv[2]); TOTAL_STEPS = int(sys.argv[3]); T0 = float(sys.argv[4])
E = os.environ.get
HOURLY_TRY = float(E("HOURLY_TRY", "43")); SPENT_BEFORE_TRY = float(E("SPENT_BEFORE_TRY", "0"))
MAX_HOURS = float(E("MAX_HOURS", "72")); MAX_TRY = float(E("MAX_TRY", "8500"))
RANK_FLOOR = float(E("RANK_FLOOR", "75")); RANK_GRACE_STEP = int(E("RANK_GRACE_STEP", "20000")); RANK_STOP = E("RANK_STOP", "0") == "1"
NN_COLLAPSE = float(E("NN_COLLAPSE", "0.15")); STALL_S = float(E("STALL_S", "2700")); PROJ_AFTER_S = float(E("PROJ_AFTER_S", "3600"))
POLL = float(E("POLL", "300"))
LOG = Path.home() / "p4logs" / f"watchdog-{RUN_DIR.name}.jsonl"; LOG.parent.mkdir(parents=True, exist_ok=True)
seen: dict[int, float] = {}


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return {"count": len(out), "rows": out}
    except Exception as exc:  # noqa: BLE001
        return {"count": 0, "error": str(exc)}


def write(rec):
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def stop(reason, rec):
    write({**rec, "action": "TERMINATE", "reason": reason})
    if alive(TRAINER_PID):
        os.kill(TRAINER_PID, signal.SIGTERM)
        for _ in range(60):
            if not alive(TRAINER_PID): break
            time.sleep(1)
        if alive(TRAINER_PID):
            os.kill(TRAINER_PID, signal.SIGKILL)
    write({"t": time.time(), "action": "TRAINER_EXITED"}); sys.exit(0)


while True:
    now = time.time()
    rows = []
    mp = RUN_DIR / "metrics.jsonl"
    if mp.is_file():
        for line in mp.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    for r in rows:
        seen.setdefault(int(r["step"]), now)
    last = rows[-1] if rows else {}
    step = int(last.get("step", 0)); elapsed = now - T0
    timed = sorted(seen.items()); rate = None
    if len(timed) >= 3:
        (s0, t0), (s1, t1) = timed[1], timed[-1]
        if t1 > t0 and s1 > s0: rate = (s1 - s0) / (t1 - t0)
    proj_h = (elapsed + (TOTAL_STEPS - step) / rate) / 3600 if rate else None
    proj_try = SPENT_BEFORE_TRY + proj_h * HOURLY_TRY if proj_h else None
    g = gpu(); ck = RUN_DIR / "checkpoints" / "last.pt"
    post = [r for r in rows if int(r["step"]) >= RANK_GRACE_STEP and r.get("val/representation_effective_rank") is not None]
    nn_rows = [r for r in rows if int(r["step"]) >= 5000 and r.get("val/nn_temporal_hit_min") is not None]
    rec = {"t": now, "elapsed_h": round(elapsed / 3600, 3), "step": step, "total_steps": TOTAL_STEPS, "steps_per_s": rate,
           "proj_total_h": proj_h, "proj_cost_try": proj_try, "spent_try_so_far": SPENT_BEFORE_TRY + elapsed / 3600 * HOURLY_TRY,
           "loss": last.get("train/loss"), "val_rank_min": last.get("val/representation_effective_rank"),
           "nn_min": last.get("val/nn_temporal_hit_min"), "gpu": g, "trainer_alive": alive(TRAINER_PID),
           "rank_below_floor_3_consecutive": len(post) >= 3 and all(float(r["val/representation_effective_rank"]) < RANK_FLOOR for r in post[-3:])}
    write(rec)
    if not alive(TRAINER_PID):
        write({"t": now, "action": "TRAINER_EXITED"}); sys.exit(0)
    if g.get("count") != 1: stop("gpu_count_not_1", rec)
    if last.get("train/loss") is not None and not math.isfinite(float(last["train/loss"])): stop("nan_or_inf_loss", rec)
    if last.get("val/representation_collapse_score") is not None and float(last["val/representation_collapse_score"]) <= 1e-6:
        stop("representation_collapse", rec)
    if len(nn_rows) >= 3 and all(float(r["val/nn_temporal_hit_min"]) < NN_COLLAPSE for r in nn_rows[-3:]): stop("semantic_collapse", rec)
    if RANK_STOP and rec["rank_below_floor_3_consecutive"]: stop("rank_below_floor_3_consecutive_validations", rec)
    if now - (max(seen.values()) if seen else T0) > STALL_S: stop("no_new_metrics_stall", rec)
    if step >= 10000 and ck.is_file() and now - ck.stat().st_mtime > max(STALL_S, 3 * 5000 / (rate or 1)): stop("checkpoint_not_advancing", rec)
    if elapsed / 3600 >= MAX_HOURS - 0.1: stop("wall_clock_limit", rec)
    if elapsed > PROJ_AFTER_S and proj_h is not None:
        if proj_h > MAX_HOURS: stop("projected_runtime_over_limit", rec)
        if proj_try > MAX_TRY: stop("projected_cost_over_ceiling", rec)
    time.sleep(POLL)
