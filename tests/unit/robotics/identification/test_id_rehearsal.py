"""ID-REHEARSAL-E001 pipeline on a tiny profile: runner -> delivery -> intake -> fit -> held-out validation.

SYNTHETIC_TOOLING_CHECK only. The "true" values are the Python kernel's SYNTHETIC_ONLY inputs.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from conrad.evaluation.dispatch import EXPERIMENTS
from conrad.evaluation.identification_rehearsal import compare, record_delivery, true_values
from conrad.robotics.hardware.characterization import Category
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.hardware.identification.intake import check_delivery
from conrad.robotics.hardware.identification.pipeline import PipelineConfig, run_pipeline
from conrad.robotics.hardware.identification.report import (
    ParameterMapping,
    ReportStatus,
    SyntheticDataError,
    to_characterization_records,
)
from conrad.settings import REPO_ROOT

TINY: dict[str, Any] = {
    "robot_config": "configs/robot/sim_reference.yaml",
    "kernel": {"physics_dt_s": 0.01},
    "start_position_world_m": [0.0, 0.0, -20.0],
    "control_period_s": 0.02,
    "thruster_step_period_s": 0.01,
    "settle_s": 0.0,
    "attitude_hold": None,  # open loop: surge then only uses the horizontal thrusters
    "onboard_fix": {"sigma_m": 0.1, "period_s": 1.0},
    "thrusters_under_test": ["H1", "H2", "H3", "H4"],
    "repetitions": 2,
    "reference_noise": {"velocity_mps": 0.002, "thrust_n": 0.05},
    "profiles": {
        "B_THRUSTER_STEP": {"levels": [0.02, -0.02, 0.07, -0.07, 0.3, -0.3], "on_s": 0.4, "off_s": 0.4},
        "C_SURGE_ACCELERATION": {"fractions": [0.15, -0.15], "on_s": 1.5, "off_s": 1.5},
    },
    "pipeline": {"intake": {"min_identification_repetitions": 1}},
}


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    out = tmp_path_factory.mktemp("id_rehearsal")
    manifest, hw, aborted = record_delivery(TINY, 2026201, out)
    robot = load_robot_config(TINY["robot_config"])
    result = run_pipeline(manifest, robot, PipelineConfig.model_validate(TINY["pipeline"]))
    return manifest, hw, aborted, result


def test_runner_writes_an_honest_synthetic_delivery(rehearsal):
    manifest, _, aborted, _ = rehearsal
    assert aborted == []
    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert doc["synthetic"] is True
    assert doc["source"].startswith("SYNTHETIC_ONLY:")
    assert doc["motion_reference"]["kind"] == "SIMULATOR_TRUTH"
    assert all(s["synthetic"] is True for s in doc["segments"])
    assert set(doc["split_plan"]["validation_repetitions"]).isdisjoint(
        doc["split_plan"]["identification_repetitions"]
    )


def test_intake_rejects_the_default_minimum_on_the_tiny_profile(rehearsal):
    manifest = rehearsal[0]
    assert "REPETITIONS" in check_delivery(manifest).codes()  # 1 identification repetition < 3


def test_pipeline_fits_on_identification_and_scores_held_out(rehearsal):
    _, hw, _, result = rehearsal
    report = result.report
    assert report.status is ReportStatus.SYNTHETIC_TOOLING_CHECK
    assert {"B_thruster_H1", "C_SURGE_ACCELERATION"} <= set(report.fits)
    assert {"B_thruster_H1", "C_SURGE_ACCELERATION"} <= set(report.validations)
    assert all(v.within_envelope is None for v in report.validations.values())  # envelope OPEN
    rows = {r["parameter"]: r for r in compare(result, true_values(hw, TINY))}
    for tid in ("H1", "H2", "H3", "H4"):
        for prm in ("k_fwd", "k_rev", "time_constant_s"):
            assert rows[f"B_thruster_{tid}.{prm}"]["rel_error"] < 0.05


def test_synthetic_rehearsal_never_becomes_identified_values(rehearsal):
    report = rehearsal[3].report
    with pytest.raises(SyntheticDataError):
        to_characterization_records(
            report, "x.json", (ParameterMapping(fit_name="k_fwd", target="t", category=Category.THRUSTERS),)
        )


def test_experiment_is_registered_with_tooling_only_limitations():
    module, cfg = EXPERIMENTS["ID-REHEARSAL-E001"]
    assert module == "conrad.evaluation.identification_rehearsal"
    config = yaml.safe_load((REPO_ROOT / cfg).read_text(encoding="utf-8"))
    text = " ".join(config["limitations"])
    assert "tooling only" in text
    assert "never produces IDENTIFIED RobotConfig values" in text
