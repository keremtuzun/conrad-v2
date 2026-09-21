"""Verify the gate I5 Unity harness WITHOUT launching the Unity player, then debug it on DEVELOPMENT worlds.

The I5 Unity module (``tests/unity_live/test_i5_unity.py``) was written but never flown, and a stray pytest
collection of it once launched the player by accident and burned two held-out worlds. This script is the
safe way in. It has three modes and only the last one can start a player:

* ``--wiring``  dry run of the FORMAL declaration: every declared scenario resolves to a Unity runtime, the
  omitted scenarios are named with reasons, the declared worlds belong to the pinned held-out split and are
  not among the worlds given up, the thresholds equal the surrogate's, the criterion names equal
  ``conrad.evaluation.gates`` and the recorder plan agrees with them, and the stored M1-ACTION-E001 final
  artifact still decides the seven matrix criteria. No world is built and nothing is launched.
* ``--kernel``  the SAME flight loop, scoring and verdicts on the python L1 kernel backend, on I5 DEVELOPMENT
  seeds. It exercises ``drive_and_score`` -> ``summarize`` -> ``verdicts`` end to end. Its numbers are
  SURROGATE and are written outside ``artifacts/gates/``, so they can never be read as gate evidence.
* ``--unity``   the same loop through ``prepare_unity`` on the DEVELOPMENT worlds of
  ``configs/eval/partitions_i5_unity.yaml`` (7710100-7710109). THIS LAUNCHES THE PLAYER. It refuses any world
  that is not in that development split, so a debug run can never touch a final world.

Usage:
  python -m uv run python scripts/check_i5_unity_harness.py --wiring
  python -m uv run python scripts/check_i5_unity_harness.py --kernel [--seeds 7500000] [--scenarios ...]
  python -m uv run python scripts/check_i5_unity_harness.py --unity --worlds 7710100 [--scenarios ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conrad.evaluation import partitions as P  # noqa: E402
from conrad.evaluation.decision_experiments.m1_action_integrated import (  # noqa: E402
    BASELINES,
    PRIMARY,
    SPECS,
    drive_and_score,
    summarize,
    verdicts,
)
from conrad.evaluation.gates import GATE_BY_ID  # noqa: E402
from conrad.orchestration.mission_config import runtime_config  # noqa: E402
from conrad.settings import REPO_ROOT, ConradSettings, load_settings  # noqa: E402
from conrad.sim.mission.run import prepare  # noqa: E402
from conrad.sim.mission.unity_run import UNITY_SCENARIOS, player_identity, resolve_unity  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "i5_unity.yaml"
SURROGATE_CONFIG = REPO_ROOT / "configs" / "eval" / "m1_action_e004.yaml"
MISSION_CONFIG = REPO_ROOT / "configs" / "sim" / "mission_default.yaml"
OUT = REPO_ROOT / "artifacts" / "experiments" / "I5-UNITY-HARNESS-CHECK"
THRESHOLDS = ("latency_budget_s", "success_floor", "uir_max", "contact_clearance_m")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _settings(seed: int) -> ConradSettings:
    base = load_settings(MISSION_CONFIG)
    settings = base.model_copy(update={"run": base.run.model_copy(update={"seed": int(seed)})})
    assert isinstance(settings, ConradSettings)
    return settings


# ------------------------------------------------------------------------------------------- wiring
def wiring(cfg: dict[str, Any]) -> dict[str, Any]:
    """Dry run of the formal declaration. Builds no world and launches nothing."""
    from record_unity_gate_evidence import MODULES, PLAN, REPLAY  # the recorder's own plan, not a copy

    scenarios = [str(s) for s in cfg["scenarios"]]
    omitted = set(SPECS) - set(scenarios)
    declared_omissions = set(cfg.get("scenarios_not_on_the_unity_path", {}))
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        final = set(P.split(P.I5_UNITY_DOMAIN, "final_test", "final_evaluation").world_seeds)
    with P.purpose_scope(P.Purpose.DESIGN):
        development = set(P.split(P.I5_UNITY_DOMAIN, "development", "design").world_seeds)
    worlds = [int(w) for w in cfg["final_worlds"]]
    given_up = {int(w) for w in cfg.get("worlds_given_up", {})}

    resolved = []
    for scenario in scenarios:
        _, rt = resolve_unity(scenario, dict(_settings(worlds[0]).sim.get("mission", {})))
        rcfg = runtime_config(rt)
        resolved.append(
            {
                "scenario": scenario,
                "duration_s": rcfg.duration_s,
                "control_period_s": rcfg.control_period_s,
                "outages_s": [list(o) for o in rcfg.link.outages_s],
                "time_budget_s": rcfg.time_budget_s,
                "planner": rcfg.planner,
                "view_execution_enabled": rcfg.view_execution.enabled,
                "protect_active_view": rcfg.view_execution.protect_active_view,
                "belief_map_navigation": rcfg.view_execution.belief_map_navigation,
            }
        )

    surrogate = dict(yaml.safe_load(SURROGATE_CONFIG.read_text(encoding="utf-8")))
    same_thresholds = {k: cfg.get(k) == surrogate.get(k) for k in THRESHOLDS}
    matrix_path = REPO_ROOT / cfg["matrix_artifact"]
    matrix: dict[str, Any] = {}
    if matrix_path.exists():
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        per_class = data["summary"]["egdc_structured"]["per_class"]
        matrix = {
            "partition": data["partition"],
            "purpose": data["purpose"],
            "classes": {c: per_class[c]["recall"] for c in per_class},
            "hard_constraint_violations": data["summary"]["egdc_structured"]["hard_constraint_violations"][
                "total"
            ],
        }

    plan = [c for c, _, _ in PLAN["I5"]]
    report = {
        "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "module": MODULES["I5"],
        "final_worlds": worlds,
        "worlds_given_up": sorted(given_up),
        "development_worlds": sorted(development),
        "worlds_are_held_out_final": set(worlds) <= final,
        "worlds_are_not_given_up": not (set(worlds) & given_up),
        "flights": len(worlds) * len(scenarios) * len(cfg["arms"]),
        "scenarios": scenarios,
        "unity_scenarios_missing": sorted(set(scenarios) - set(UNITY_SCENARIOS)),
        "omitted_scenarios": sorted(omitted),
        "omissions_declared_with_reason": omitted == declared_omissions,
        "thresholds_equal_surrogate": same_thresholds,
        "matrix_artifact": str(matrix_path.relative_to(REPO_ROOT)),
        "matrix_artifact_present": matrix_path.exists(),
        "matrix": matrix,
        "recorder_plan": plan,
        "recorder_plan_matches_gates_py": tuple(plan) == GATE_BY_ID["I5"].criteria,
        "gates_py_criteria": list(GATE_BY_ID["I5"].criteria),
        "replay_nodes_empty_on_purpose": REPLAY["I5"] == [],
        "player": player_identity(),
        "resolved": resolved,
    }
    report["ok"] = bool(
        not report["unity_scenarios_missing"]
        and report["worlds_are_held_out_final"]
        and report["worlds_are_not_given_up"]
        and report["omissions_declared_with_reason"]
        and all(same_thresholds.values())
        and report["matrix_artifact_present"]
        and report["recorder_plan_matches_gates_py"]
    )
    return report


# ------------------------------------------------------------------------------------------- flights
def _fly(backend: str, seed: int, scenario: str, arm: str, cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    run_id = f"I5-CHECK-{backend}-{scenario}-s{seed}-{arm}"
    t0 = time.time()
    session: Any
    if backend == "unity":
        from conrad.sim.mission.unity_run import prepare_unity

        session = prepare_unity(scenario, MISSION_CONFIG, run_id=run_id, runs_root=root, seed=seed)
    else:
        session = prepare(scenario, _settings(seed), run_id=run_id, runs_root=root, capture=False)
    try:
        row = drive_and_score(session, seed, scenario, arm, cfg)
        out = session.finish()
    except BaseException:
        abort = getattr(session, "abort", None)
        if abort is not None:
            abort()
        raise
    row["wall_s"] = round(time.time() - t0, 1)
    row["run_dir"] = str(Path(out["run_dir"] if isinstance(out, dict) else session.run_dir))
    if backend != "unity":
        shutil.rmtree(session.run_dir, ignore_errors=True)
    return row


def flights(backend: str, cfg: dict[str, Any], seeds: list[int], scenarios: list[str]) -> dict[str, Any]:
    """The formal module's own loop (sequential, every arm on every world) on a non-final split."""
    root = REPO_ROOT / "artifacts" / "runs" / "I5-UNITY-HARNESS-CHECK" / backend
    rows = [
        _fly(backend, seed, scenario, arm, cfg, root)
        for seed in seeds
        for scenario in scenarios
        for arm in (PRIMARY, *BASELINES)
    ]
    summary = summarize(rows, scenarios)
    return {
        "backend": backend,
        "evidence_class": "SURROGATE / HARNESS CHECK, never gate evidence",
        "data_status": "SYNTHETIC_ONLY",
        "partition": "development",
        "worlds": seeds,
        "scenarios": scenarios,
        "arms": [PRIMARY, *BASELINES],
        "player": player_identity() if backend == "unity" else None,
        **summary,
        "verdicts": verdicts(summary, cfg, scenarios),
        "per_run": rows,
    }


def _check_development(backend: str, seeds: list[int]) -> None:
    """A harness check never reads a final world."""
    if backend == "unity":
        with P.purpose_scope(P.Purpose.DESIGN):
            allowed = set(P.split(P.I5_UNITY_DOMAIN, "development", "design").world_seeds)
        bad = sorted(set(seeds) - allowed)
        if bad:
            raise SystemExit(
                f"refusing to fly {bad}: the I5 Unity harness check runs only on the development worlds "
                f"of configs/eval/partitions_i5_unity.yaml ({min(allowed)}-{max(allowed)})"
            )
    else:
        with P.purpose_scope(P.Purpose.DESIGN):
            allowed = set(P.split(P.I5_V3_DOMAIN, "development", "design").world_seeds)
        bad = sorted(set(seeds) - allowed)
        if bad:
            raise SystemExit(
                f"refusing to run {bad}: not I5 development seeds ({min(allowed)}-{max(allowed)})"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiring", action="store_true", help="dry run of the formal declaration, no player")
    ap.add_argument("--kernel", action="store_true", help="the same loop on the python L1 kernel")
    ap.add_argument("--unity", action="store_true", help="the same loop on DEVELOPMENT Unity worlds")
    ap.add_argument("--seeds", default="", help="kernel: comma-separated I5 development seeds")
    ap.add_argument("--worlds", default="", help="unity: comma-separated I5 Unity DEVELOPMENT worlds")
    ap.add_argument("--scenarios", default="", help="comma-separated subset of the declared scenarios")
    args = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.wiring:
        report = wiring(cfg)
        (OUT / "wiring.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "resolved"}, indent=1, default=str))
        return 0 if report["ok"] else 1

    scenarios = [s for s in (args.scenarios.split(",") if args.scenarios else cfg["scenarios"]) if s]
    unknown = sorted(set(scenarios) - set(cfg["scenarios"]))
    if unknown:
        raise SystemExit(f"not declared in {CONFIG_PATH.name}: {unknown}")

    if args.kernel:
        seeds = [int(x) for x in (args.seeds.split(",") if args.seeds else ["7500000"])]
        _check_development("kernel", seeds)
        report = flights("kernel", cfg, seeds, scenarios)
        (OUT / "kernel.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(json.dumps(report["verdicts"], indent=1, default=str))
        return 0

    if args.unity:
        worlds = [int(x) for x in (args.worlds.split(",") if args.worlds else ["7710100"])]
        _check_development("unity", worlds)
        report = flights("unity", cfg, worlds, scenarios)
        (OUT / "unity_development.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8"
        )
        print(json.dumps(report["verdicts"], indent=1, default=str))
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
