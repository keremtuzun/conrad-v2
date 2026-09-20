"""I6-MULTIDOMAIN-E001: gate I6 surrogate (python kernel) on the declared worlds (configs/eval/i6_multidomain.yaml).

For every world and arm (TURBID, CLEAR) one integrated mission runs with Twin2S + Twin2T + Twin2E; the I6 measurements
(``conrad.evaluation.multidomain``) are taken and the per-criterion decision rule of the config is applied.

Usage:
  python -m uv run python scripts/run_i6_multidomain.py --partition development   # design worlds only
  python -m uv run python scripts/run_i6_multidomain.py --partition final_test    # once, on the declared worlds

Writes artifacts/experiments/I6-MULTIDOMAIN-E001[-DEV]/i6_e001.json. SURROGATE evidence only (ADR-0008).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "eval" / "i6_multidomain.yaml"
CRITERIA = (
    "one mission produces 2S, 2T and 2E beliefs",
    "Model1 reasons across all three via the Belief Bus",
    "children remain authoritative within domains",
)


def load_config() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def worlds(cfg: dict[str, Any], partition: str) -> list[int]:
    from conrad.evaluation.partitions import split

    if partition == "final_test":
        allowed = set(split("unity_gate", "final_test", "final_evaluation").world_seeds)
        seeds = [int(s) for s in cfg["final_worlds"]]
    else:
        allowed = set(split("unity_gate", "development", "design").world_seeds)
        seeds = [int(s) for s in cfg["development_worlds"]]
    if not set(seeds) <= allowed:
        raise SystemExit(f"{partition} worlds {seeds} are not all in the unity_gate {partition} split")
    return seeds


def run_one(job: tuple[int, str, str, str]) -> dict[str, Any]:
    seed, arm, scenario, mission_config = job
    from conrad.evaluation.multidomain import (
        authority_ok,
        beliefs_ok,
        i6_measurements,
        reasoning_ok,
        run_with_truth,
    )
    from conrad.settings import load_settings
    from conrad.sim.mission.run import prepare

    s = load_settings(mission_config)
    s = s.model_copy(update={"run": s.run.model_copy(update={"seed": seed})})
    t0 = time.time()
    session = prepare(scenario, s, runs_root=Path(tempfile.mkdtemp(prefix="i6-")), capture=False)
    try:
        truth = run_with_truth(session)
        m = i6_measurements(session, truth)
    finally:
        session.log.close()
        session.engine.dispose()
    r = m[CRITERIA[1]]
    return {
        "seed": seed,
        "arm": arm,
        "scenario": scenario,
        "wall_s": round(time.time() - t0, 1),
        "measured": m,
        "truth": truth,
        "pass": {
            CRITERIA[0]: beliefs_ok(m[CRITERIA[0]]),
            CRITERIA[1]: reasoning_ok(r, arm),
            CRITERIA[2]: authority_ok(m[CRITERIA[2]]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=("development", "final_test"), required=True)
    args = ap.parse_args()
    cfg = load_config()
    seeds = worlds(cfg, args.partition)
    jobs = [(s, arm, scen, cfg["mission_config"]) for s in seeds for arm, scen in cfg["arms"].items()]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"])) as ex:
        results = list(ex.map(run_one, jobs))
    agree_min = float(cfg["truth_agreement_min"])
    for r in results:
        a = r["measured"][CRITERIA[1]]["gate_truth_agreement"]
        r["pass"][CRITERIA[1]] = bool(r["pass"][CRITERIA[1]] and a is not None and a >= agree_min)
    per_world = {
        str(s): {c: all(r["pass"][c] for r in results if r["seed"] == s) for c in CRITERIA} for s in seeds
    }
    summary = {c: all(per_world[str(s)][c] for s in seeds) for c in CRITERIA}
    out_dir = (
        ROOT
        / "artifacts"
        / "experiments"
        / ("I6-MULTIDOMAIN-E001" if args.partition == "final_test" else "I6-MULTIDOMAIN-E001-DEV")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "experiment_id": cfg["experiment_id"],
        "partition": args.partition,
        "evidence_class": "SURROGATE",
        "execution_path": "python L1 kernel instead of Unity (ADR-0008)",
        "config": str(CONFIG.relative_to(ROOT)),
        "worlds": seeds,
        "arms": cfg["arms"],
        "decision_rule": cfg["decision_rule"],
        "truth_agreement_min": agree_min,
        "per_world": per_world,
        "summary": summary,
        "runs": results,
    }
    path = out_dir / "i6_e001.json"
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"summary": summary, "per_world": per_world}, indent=1))
    print(path)


if __name__ == "__main__":
    main()
