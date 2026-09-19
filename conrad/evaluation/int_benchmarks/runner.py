"""Integrated benchmark runner, run summaries and the I4 acceptance record. EVALUATION PLANE.

``run(config, seeds, out_dir)`` follows the ``conrad eval run --experiment`` dispatch contract. Every number
comes from a stored run bundle; nothing is estimated. The I4 record keeps OPEN thresholds and therefore
evaluates to NOT_EVALUABLE (ch28 Acceptance Records).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from conrad.evaluation.acceptance import AcceptanceRecord, Measurements, evaluate
from conrad.orchestration.evaluation import evaluate_run_dir
from conrad.settings import ConradSettings, load_settings
from conrad.sim.mission.run import run_scenario
from conrad.sim.mission.scenarios import INT_DESCRIPTIONS

DEFAULT_CONFIG = "configs/sim/mission_default.yaml"


def summarize(run_dir: Path) -> dict[str, Any]:
    """The per-run numbers the integration report asks for, read from the bundle."""
    rep = evaluate_run_dir(run_dir)
    rt = json.loads((run_dir / "mission" / "runtime_metrics.json").read_text(encoding="utf-8"))
    before, after = rep["target_before"] or {}, rep["target_after"] or {}
    eb, ea = rep["target_error_before"] or {}, rep["target_error_after"] or {}
    comm = rt["communication"]
    return {
        "run_dir": str(run_dir),
        "scenario_id": rep["scenario_id"],
        "seed": rep["seed"],
        "error_before": {
            "corrosion_depth_m": eb.get("corrosion_depth_m.abs_error"),
            "corrosion_depth_m_prior_mean": eb.get("corrosion_depth_m.prior_mean_abs_error"),
            "crack_length_m": eb.get("crack_length_m.abs_error"),
            "crack_length_m_prior_mean": eb.get("crack_length_m.prior_mean_abs_error"),
        },
        "error_after": {
            "corrosion_depth_m": ea.get("corrosion_depth_m.abs_error"),
            "crack_length_m": ea.get("crack_length_m.abs_error"),
        },
        "status_before": before.get("condition.status"),
        "status_after": after.get("condition.status"),
        "U_before": {k: before.get(k) for k in ("U_O", "U_A", "U_C")},
        "U_after": {k: after.get(k) for k in ("U_O", "U_A", "U_C")},
        "patch_max_visible_fraction": rep["patch_max_visible_fraction"],
        "inspection_goal_t_s": rep["inspection_goal_t_s"],
        "direct_revisions_after_inspection": len(rep["target_direct_revisions_after_inspection"]),
        "decisions": rt["decisions"],
        "decision_action_counts": {
            a: rt["decision_actions"].count(a) for a in sorted({str(x) for x in rt["decision_actions"]})
        },
        "plans": [p["status"] for p in rt["plans"]],
        "uir": rt["uir"],
        "commands_accepted": rt["commands_accepted"],
        "commands_rejected": rt["commands_rejected"],
        "commands_refused_by_safety_supervisor": rt["commands_refused_by_safety_supervisor"],
        "bits_sent": comm["bits_sent"],
        "critical_offers": comm["critical_offers"],
        "critical_delivered": comm["critical_delivered"],
        "critical_alert_latency_s": comm["critical_alert_latency_s"],
        "safety_events": [(round(e["t_s"], 1), e["to"], e["reasons"]) for e in rt["safety_events"]],
        "failed_modules": rt["failed_modules"],
        "collisions": rep["vehicle"]["collisions"],
        "energy_used_j": rep["vehicle"]["energy_used_j"],
    }


def write_i4_acceptance(run_dirs: list[Path], baseline_dirs: list[Path], out: Path) -> dict[str, Any]:
    record = AcceptanceRecord(
        gate_id="I4",
        primary_metric="target_hidden_state_abs_error (corrosion_depth_m, crack_length_m)",
        direction="lower",
        baseline="FLAGSHIP-I4-FIXEDVIEW (A-B1_fixed_inspection)",
    )
    seeds = tuple(sorted({int(summarize(d)["seed"]) for d in run_dirs}))
    measurements = Measurements(
        seeds=seeds,
        baseline=record.baseline,
        baseline_evidence=";".join(str(d) for d in baseline_dirs) or None,
        evidence_artifact=";".join(str(d) for d in run_dirs),
    )
    result = evaluate(record, measurements)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump({"acceptance": json.loads(record.canonical_json())}, sort_keys=True), encoding="utf-8"
    )
    (out.with_suffix(".result.json")).write_text(result.canonical_json(), encoding="utf-8")
    return json.loads(result.canonical_json())


def run(config: dict[str, Any] | None, seeds: list[int], out_dir: Path) -> dict[str, Any]:
    """Dispatch entry: every scenario listed in ``config['scenarios']`` on every seed."""
    base = load_settings(DEFAULT_CONFIG)
    scenarios = list((config or {}).get("scenarios", sorted(INT_DESCRIPTIONS)))
    rows = []
    for seed in seeds:
        settings = base.model_copy(update={"run": base.run.model_copy(update={"seed": int(seed)})})
        assert isinstance(settings, ConradSettings)
        for sid in scenarios:
            outcome = run_scenario(sid, settings)
            rows.append(summarize(Path(outcome["run_dir"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "int_benchmarks.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    return {"rows": len(rows), "result": rows}
