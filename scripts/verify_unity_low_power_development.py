"""Verify corrected LOW_POWER semantics on one declared Unity development world."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/unity_live"))

import test_i5_spatial_v1_1_unity as unity_i5  # noqa: E402

from conrad.evaluation import partitions as P  # noqa: E402
from conrad.evaluation.decision_experiments.m1_action_integrated import PRIMARY  # noqa: E402
from conrad.sim.mission.unity_run import player_identity  # noqa: E402

SEED = 8701100
SCENARIO = "I5-BATTERY-RESERVE"
PLAYER_SHA256 = "c52be937690a5d3d5be03f4acc3fc06842dd17ce3f53000e34f6daa528ab71e9"
RUNS = ROOT / "artifacts/unity/development/I5_LOW_POWER_SEMANTICS_V2"
OUTPUT = ROOT / "artifacts/verification/unity_low_power_semantics_development.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite development verification {OUTPUT}")
    with P.purpose_scope(P.Purpose.DESIGN):
        split = P.split(P.I5_UNITY_V3_DOMAIN, "development", "design")
    if SEED not in split.world_seeds:
        raise SystemExit(f"seed {SEED} is not in the declared development split")
    identity = player_identity()
    if identity["player_exe_sha256"] != PLAYER_SHA256:
        raise SystemExit(f"player mismatch: {identity['player_exe_sha256']} != {PLAYER_SHA256}")
    cfg = unity_i5._config()
    cfg["player"] = {**cfg["player"], "executable_sha256": PLAYER_SHA256}
    row = unity_i5._fly(SEED, SCENARIO, PRIMARY, cfg, RUNS)
    post_fault = [state for state in row["decision_states"] if float(state["t_s"]) > 30.0]
    batteries = [float(state["battery"]) for state in post_fault]
    report = {
        "verification": "Unity LOW_POWER kernel-semantics development check",
        "evidence_class": "DEVELOPMENT",
        "partition": "development",
        "partition_file": str(P.I5_UNITY_V3_PARTITIONS_PATH.relative_to(ROOT)),
        "partition_digest": split.digest,
        "world": SEED,
        "scenario": SCENARIO,
        "arm": PRIMARY,
        "player": identity,
        "source_row": row["run_dir"],
        "source_row_sha256": _sha256(ROOT / row["run_dir"] / "reports/i5_spatial_v1_1_score.json"),
        "post_fault_battery_first": batteries[0] if batteries else None,
        "post_fault_battery_min": min(batteries) if batteries else None,
        "warrant_reached": row["driving"]["warrant_reached"],
        "chosen_at_onset": row["driving"]["chosen_at_onset"],
        "correct": row["driving"]["correct"],
        "violations_total": row["driving"]["violations_total"],
        "pass_rule": {
            "post_fault_battery_at_most": 0.12,
            "warrant_reached": True,
            "chosen_at_onset": "RETURN_TO_SAFE_STATE",
            "correct": True,
            "violations_total": 0,
        },
        "status": "PASS",
        "limitations": ["single development world", "synthetic Unity L1 approximate physics"],
    }
    checks = (
        batteries
        and batteries[0] <= 0.12
        and row["driving"]["warrant_reached"] is True
        and row["driving"]["chosen_at_onset"] == "RETURN_TO_SAFE_STATE"
        and row["driving"]["correct"] is True
        and row["driving"]["violations_total"] == 0
    )
    if not checks:
        report["status"] = "FAIL"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
