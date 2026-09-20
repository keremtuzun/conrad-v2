"""Verify the gate I7 Unity harness WITHOUT launching the Unity player (another workstream owns it).

Two checks, both read from ``configs/eval/i7_unity.yaml``:

* ``--wiring``   dry run of the FORMAL path: the Unity scenarios are registered in
  ``conrad.sim.mission.unity_run.UNITY_SCENARIOS``, every declared flight resolves to a concrete runtime
  (bandwidth, outages, duration, control period, scheduler policy, shadow arms), the declared worlds are
  unity_gate final_test worlds, the criterion names equal ``conrad.evaluation.gates`` and the recorder plan
  agrees with them. No world is built and nothing is launched.
* ``--kernel``   the SAME harness on the python L1 kernel backend, on DEVELOPMENT seeds. This exercises the
  flight loop, the five arms, the scoring and all four criterion rules end to end. Its numbers are SURROGATE
  and are written outside ``artifacts/gates/`` so they can never be mistaken for gate evidence.

Usage:
  python -m uv run python scripts/check_i7_unity_harness.py --wiring
  python -m uv run python scripts/check_i7_unity_harness.py --kernel [--sweep reduced] [--worlds 5100000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conrad.evaluation.decision_experiments import com_i7_unity as H  # noqa: E402
from conrad.evaluation.gates import GATE_BY_ID  # noqa: E402

OUT = ROOT / "artifacts" / "experiments" / "I7-UNITY-HARNESS-CHECK"


def wiring(sweep_name: str | None) -> dict:
    from record_unity_gate_evidence import PLAN  # the recorder's own plan, not a copy

    cfg = H.load_config()
    report = H.wiring_check(cfg, sweep_name)
    plan = [c for c, _, _ in PLAN["I7"]]
    report["recorder_plan"] = plan
    report["recorder_plan_matches_gates_py"] = tuple(plan) == GATE_BY_ID["I7"].criteria
    report["gates_py_criteria"] = list(GATE_BY_ID["I7"].criteria)
    report["ok"] = bool(
        not report["unity_scenarios_missing"]
        and report["criteria_match_gates_py"]
        and report["recorder_plan_matches_gates_py"]
        and report["flights"] > 0
    )
    return report


def kernel(sweep_name: str | None, worlds: list[int] | None) -> dict:
    cfg = H.load_config()
    sweep = H.sweep_of(cfg, sweep_name)
    seeds = worlds if worlds else H.development_worlds(cfg, H.KERNEL)
    return H.run_gate(
        cfg,
        seeds,
        sweep,
        H.KERNEL,
        OUT / "runs",
        results_path=OUT / f"kernel_{sweep['name']}.json",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiring", action="store_true")
    ap.add_argument("--kernel", action="store_true")
    ap.add_argument("--sweep", default=None, help="full | reduced (default: the config's default_sweep)")
    ap.add_argument("--worlds", type=int, nargs="*", default=None, help="kernel development seeds")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.wiring or not args.kernel:
        rep = wiring(args.sweep)
        (OUT / "wiring.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in rep.items() if k != "resolved"}, indent=1, default=str))
        print(f"resolved flights: {len(rep['resolved'])} -> {OUT / 'wiring.json'}")
        if not rep["ok"]:
            raise SystemExit("I7 Unity wiring check FAILED")
    if args.kernel:
        res = kernel(args.sweep, args.worlds)
        print(
            json.dumps(
                {
                    "backend": res["backend"],
                    "evidence_class": res["evidence_class"],
                    "worlds": res["worlds"],
                    "sweep": res["sweep"]["name"],
                    "flights": res["flights"],
                    "wall_clock_s": res["wall_clock_s"],
                    "criteria": res["criteria"],
                },
                indent=1,
            )
        )
