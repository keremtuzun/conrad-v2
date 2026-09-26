"""Experiment dispatch for `conrad eval run`: stable experiment ID -> module with ``run(config, seeds, out_dir)``.

Every executed experiment is appended to the experiment registry, including failed hypotheses.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from conrad.evaluation.registry.experiments import ExperimentRecord, ExperimentRegistry, Outcome, Tier
from conrad.settings import REPO_ROOT

EXPERIMENTS: dict[str, tuple[str, str]] = {
    "CORE-ASSOC-E001": ("conrad.evaluation.core_experiments.assoc_e001", "configs/eval/core_assoc_e001.yaml"),
    "CORE-BUO-E001": ("conrad.evaluation.core_experiments.buo_e001", "configs/eval/core_buo_e001.yaml"),
    "CORE-UNC-E001": ("conrad.evaluation.core_experiments.unc_e001", "configs/eval/core_unc_e001.yaml"),
    "CORE-RBP-E001": ("conrad.evaluation.core_experiments.rbp_e001", "configs/eval/core_rbp_e001.yaml"),
    "CORE-TBD-E001": ("conrad.evaluation.core_experiments.tbd_e001", "configs/eval/core_tbd_e001.yaml"),
    "CORE-TBD-E002": ("conrad.evaluation.core_experiments.tbd_e001", "configs/eval/core_tbd_e002.yaml"),
    "CORE-PERSIST-E001": (
        "conrad.evaluation.core_experiments.persist_e001",
        "configs/eval/core_persist_e001.yaml",
    ),
    "CORE-FULL-E001": ("conrad.evaluation.core_experiments.full_e001", "configs/eval/core_full_e001.yaml"),
    "M1-UIR-E001": ("conrad.evaluation.decision_experiments.m1_uir", "configs/eval/m1_uir_e001.yaml"),
    "M1-ACTION-E001": (
        "conrad.evaluation.decision_experiments.action_matrix",
        "configs/eval/m1_action_e001.yaml",
    ),
    "M1-ACTION-E002": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_e002.yaml",
    ),
    "M1-ACTION-E003": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_e003.yaml",
    ),
    "M1-ACTION-E004": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_e004.yaml",
    ),
    "M1-ACTION-E005": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_e005.yaml",
    ),
    "M1-ACTION-SPATIAL-V1-DEV": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_spatial_v1_development.yaml",
    ),
    "M1-ACTION-SPATIAL-V1-DEV-R2": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_spatial_v1_development_r2.yaml",
    ),
    "M1-ACTION-SPATIAL-V1-DEV-R3": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_spatial_v1_development_r3.yaml",
    ),
    "M1-ACTION-SPATIAL-V1-DEV-R4": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_spatial_v1_development_r4.yaml",
    ),
    "M1-ACTION-SPATIAL-V1-DEV-R5": (
        "conrad.evaluation.decision_experiments.m1_action_integrated",
        "configs/eval/m1_action_spatial_v1_development_r5.yaml",
    ),
    "ACTIVE-MCBR-E001": (
        "conrad.evaluation.decision_experiments.active_mcbr",
        "configs/eval/active_mcbr_e001.yaml",
    ),
    "ACTIVE-MCBR-SEL001": (
        "conrad.evaluation.decision_experiments.active_mcbr_reeval",
        "configs/eval/active_mcbr_sel001.yaml",
    ),
    "ACTIVE-MCBR-E002": (
        "conrad.evaluation.decision_experiments.active_mcbr_reeval",
        "configs/eval/active_mcbr_e002.yaml",
    ),
    "ACTIVE-MCBR-E003": (
        "conrad.evaluation.decision_experiments.active_mcbr_reeval",
        "configs/eval/active_mcbr_e003.yaml",
    ),
    "ACTIVE-MCBR-E004": (
        "conrad.evaluation.decision_experiments.active_mcbr_e004",
        "configs/eval/active_mcbr_e004.yaml",
    ),
    "ACTIVE-MCBR-E005-DESIGN": (
        "conrad.evaluation.decision_experiments.active_mcbr_i4_occluded",
        "configs/eval/active_mcbr_e005_design.yaml",
    ),
    "ACTIVE-MCBR-E005-SEL": (
        "conrad.evaluation.decision_experiments.active_mcbr_i4_occluded",
        "configs/eval/active_mcbr_e005_selection.yaml",
    ),
    "ACTIVE-MCBR-E005": (
        "conrad.evaluation.decision_experiments.active_mcbr_i4_occluded",
        "configs/eval/active_mcbr_e005.yaml",
    ),
    "ACTIVE-MCBR-E005-OOD": (
        "conrad.evaluation.decision_experiments.active_mcbr_i4_occluded",
        "configs/eval/active_mcbr_e005_ood.yaml",
    ),
    # Development-only headroom measurement for the MCBR V4 question; it promotes no gate.
    "ACTIVE-MCBR-E006": (
        "conrad.evaluation.decision_experiments.i4_oracle_headroom",
        "configs/eval/i4_oracle_headroom.yaml",
    ),
    "ACTIVE-MCBR-E006-W": (
        "conrad.evaluation.decision_experiments.i4_oracle_headroom",
        "configs/eval/i4_oracle_headroom_weighted.yaml",
    ),
    # MCBR V4: the view execution rounds (docs/audits/MCBR_V4.md). DEV and DEV2 are development, VAL is the
    # selection round on validation. None of them reads final_test or ood_test.
    "ACTIVE-MCBR-E007-DEV": (
        "conrad.evaluation.decision_experiments.i4_view_execution",
        "configs/eval/i4_v4_development.yaml",
    ),
    "ACTIVE-MCBR-E007-DEV2": (
        "conrad.evaluation.decision_experiments.i4_view_execution",
        "configs/eval/i4_v4_development_r2.yaml",
    ),
    "ACTIVE-MCBR-E007-VAL": (
        "conrad.evaluation.decision_experiments.i4_view_execution",
        "configs/eval/i4_v4_validation.yaml",
    ),
    "COM-BAAC-E001": ("conrad.evaluation.decision_experiments.com_baac", "configs/eval/com_baac_e001.yaml"),
    "COM-I7-E001": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e001.yaml"),
    "COM-I7-E002": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e002.yaml"),
    # Re-run of E001/E002 on the fresh partitions_i7_v2.yaml final seeds after the 2026-09-20 BAAC repair.
    "COM-I7-E003": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e003.yaml"),
    "COM-I7-E004": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e004.yaml"),
    # Third run: fresh partitions_i7_v3.yaml final seeds, with the finding-following outage construction.
    "COM-I7-E005": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e005.yaml"),
    "COM-I7-E006": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e006.yaml"),
    # Fourth cycle: versioned compact critical summaries. Validation must select the candidate before the
    # one-shot E007/E008 final split is opened.
    "COM-I7-V4-VAL-BW": (
        "conrad.evaluation.decision_experiments.com_i7",
        "configs/eval/com_i7_v4_validation_bandwidth.yaml",
    ),
    "COM-I7-V4-VAL-OUTAGE": (
        "conrad.evaluation.decision_experiments.com_i7",
        "configs/eval/com_i7_v4_validation_outage.yaml",
    ),
    "COM-I7-E007": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e007.yaml"),
    "COM-I7-E008": ("conrad.evaluation.decision_experiments.com_i7", "configs/eval/com_i7_e008.yaml"),
    "2S-E001": ("conrad.evaluation.spatial_experiments.e001_coverage", "configs/eval/2s_e001.yaml"),
    "2S-E002": ("conrad.evaluation.spatial_experiments.e002_counterfactual", "configs/eval/2s_e002.yaml"),
    "2S-E003": ("conrad.evaluation.spatial_experiments.e003_pose", "configs/eval/2s_e003.yaml"),
    "2S-E004": ("conrad.evaluation.spatial_experiments.e004_modality", "configs/eval/2s_e004.yaml"),
    "2T-E001": ("conrad.evaluation.structural_experiments.e001_direct", "configs/eval/2t_e001.yaml"),
    "2T-E002": ("conrad.evaluation.structural_experiments.e002_persistent", "configs/eval/2t_e002.yaml"),
    "2T-E003": ("conrad.evaluation.structural_experiments.e003_tcdp", "configs/eval/2t_e003.yaml"),
    "2T-E004": ("conrad.evaluation.structural_experiments.e004_temporal", "configs/eval/2t_e004.yaml"),
    "2T-E001-R2-DEV": (
        "conrad.evaluation.structural_experiments.e001_r2",
        "configs/eval/2t_e001_r2_dev.yaml",
    ),
    "2T-E001-R2": ("conrad.evaluation.structural_experiments.e001_r2", "configs/eval/2t_e001_r2.yaml"),
    "2T-E002-R2-DEV": (
        "conrad.evaluation.structural_experiments.e002_persistent",
        "configs/eval/2t_e002_r2_dev.yaml",
    ),
    "2T-E002-R2": (
        "conrad.evaluation.structural_experiments.e002_persistent",
        "configs/eval/2t_e002_r2.yaml",
    ),
    "2T-E003-R2-DEV": (
        "conrad.evaluation.structural_experiments.e003_r2",
        "configs/eval/2t_e003_r2_dev.yaml",
    ),
    "2T-E003-R2": ("conrad.evaluation.structural_experiments.e003_r2", "configs/eval/2t_e003_r2.yaml"),
    "2T-E004-R2-DEV": (
        "conrad.evaluation.structural_experiments.e004_temporal",
        "configs/eval/2t_e004_r2_dev.yaml",
    ),
    "2T-E004-R2": ("conrad.evaluation.structural_experiments.e004_temporal", "configs/eval/2t_e004_r2.yaml"),
    "2E-E001": ("conrad.evaluation.ecological_experiments.e001_field", "configs/eval/2e_e001.yaml"),
    "2E-E002": ("conrad.evaluation.ecological_experiments.e002_turbidity", "configs/eval/2e_e002.yaml"),
    "2E-E003": ("conrad.evaluation.ecological_experiments.e003_coupling", "configs/eval/2e_e003.yaml"),
    "2E-E001-R2": ("conrad.evaluation.ecological_experiments.e001_field", "configs/eval/2e_e001_r2.yaml"),
    "2E-E002-R2": ("conrad.evaluation.ecological_experiments.e002_turbidity", "configs/eval/2e_e002_r2.yaml"),
    "2E-E003-R2": ("conrad.evaluation.ecological_experiments.e003_coupling", "configs/eval/2e_e003_r2.yaml"),
    "2E-E001-R3": ("conrad.evaluation.ecological_experiments.e001_field", "configs/eval/2e_e001_r3.yaml"),
    "2E-E002-R3": ("conrad.evaluation.ecological_experiments.e002_turbidity", "configs/eval/2e_e002_r3.yaml"),
    "2E-E003-R3": ("conrad.evaluation.ecological_experiments.e003_coupling", "configs/eval/2e_e003_r3.yaml"),
    "NAV-CAL-E001": ("conrad.evaluation.nav_benchmarks.calibration", "configs/sim/nav_calibration.yaml"),
    "DATA-REAL-SMOKE-E001": ("conrad.data.real_smoke", "datasets/experiments/data_real_smoke_e001.yaml"),
    "DATA-REAL-E002": ("conrad.data.real_e002", "datasets/experiments/data_real_e002.yaml"),
    # SYNTHETIC_TOOLING_CHECK of the EXT-HW-04 identification protocol; never yields IDENTIFIED values.
    "ID-REHEARSAL-E001": (
        "conrad.evaluation.identification_rehearsal",
        "configs/eval/id_rehearsal_e001.yaml",
    ),
}
DEFAULT_SEEDS = [2026201, 2026202, 2026203]


def register(experiment_id: str, module: str, config: str) -> None:
    EXPERIMENTS[experiment_id] = (module, config)


def _discover() -> None:
    """Domain experiment packages declare EXPERIMENTS = {id: (module, config)} in their __init__."""
    for pkg in ("spatial_experiments", "structural_experiments", "ecological_experiments", "int_benchmarks"):
        try:
            mod = importlib.import_module(f"conrad.evaluation.{pkg}")
        except ModuleNotFoundError:
            continue
        for eid, spec in getattr(mod, "EXPERIMENTS", {}).items():
            EXPERIMENTS.setdefault(eid, tuple(spec))


def run_experiment(
    experiment_id: str,
    config_path: str | None = None,
    seeds: list[int] | None = None,
    out_root: str | Path | None = None,
) -> dict[str, Any]:
    _discover()
    if experiment_id not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {experiment_id!r}; registered: {sorted(EXPERIMENTS)}")
    module_name, default_cfg = EXPERIMENTS[experiment_id]
    cfg_path = REPO_ROOT / (config_path or default_cfg)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    seed_list = seeds or list(config.get("seeds", DEFAULT_SEEDS))
    out = Path(out_root) if out_root else REPO_ROOT / "artifacts" / "experiments" / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    runner = importlib.import_module(module_name).run
    t0 = time.time()
    result: dict[str, Any] = runner(config, seed_list, out)
    record = {
        "experiment_id": experiment_id,
        "module": module_name,
        "config": str(cfg_path.relative_to(REPO_ROOT)),
        "seeds": seed_list,
        "wall_s": round(time.time() - t0, 1),
        "out_dir": str(out),
    }
    (out / "dispatch_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    registry_record = ExperimentRecord(
        experiment_id=experiment_id,
        mechanism=module_name.rsplit(".", 1)[-1],
        hypothesis=str(config.get("hypothesis", f"see {module_name} docstring")),
        tier=Tier.T1_DEVELOPMENT,
        baselines=tuple(str(b) for b in config.get("baselines", ())),
        seeds=tuple(seed_list),
        split_hash=hashlib.sha256(f"{experiment_id}:{seed_list}".encode()).hexdigest(),
        primary_metric=str(config.get("primary_metric", "see result json")),
        direction=str(config.get("direction", "lower")),
        compute={"device": "cpu"},
        config_digest=hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
        config_ref=record["config"],
        code_commit=_git_commit(),
        git_dirty=_git_dirty(),
        result=_flat_floats(result),
        artifact_paths=(str(out),),
        # Outcome judgement (SUPPORTS/REFUTES) needs a paired effect and an ADR; dispatch never grants it.
        outcome=Outcome.INCONCLUSIVE,
        # The config states its own limitations (e.g. a real-data smoke run); the default is the synthetic case.
        limitations=tuple(
            str(x)
            for x in config.get(
                "limitations", ("synthetic data only", "simulation validity L1 or abstract sandbox")
            )
        ),
        recorded_time_ns=time.time_ns(),
    )
    ExperimentRegistry(REGISTRY_PATH).append(registry_record)
    return {**record, "result": result}


REGISTRY_PATH = REPO_ROOT / "artifacts" / "registry" / "experiments.jsonl"


def _flat_floats(
    obj: Any, prefix: str = "", out: dict[str, float] | None = None, depth: int = 0
) -> dict[str, float]:
    out = {} if out is None else out
    if depth > 3 or len(out) > 200:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat_floats(v, f"{prefix}{k}.", out, depth + 1)
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, int | float):
        out[prefix.rstrip(".")] = float(obj)
    return out


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_commit() -> str:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=no"))
