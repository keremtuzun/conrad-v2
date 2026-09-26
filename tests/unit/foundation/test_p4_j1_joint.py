from __future__ import annotations

import copy
from pathlib import Path
import time

import pytest
import torch

from conrad.foundation.joint import JointOSFMModel
from conrad.foundation.pretraining.j1_joint import (
    J1JointTrainingBundle,
    PARENT_SPECS,
    build_j1_model,
    load_j1_parents,
    run_j1_joint_smoke,
    synthetic_j1_fixture,
)
from conrad.foundation.pretraining.parents import load_parent_component
from conrad.foundation.pretraining.u1_rgb import parameter_count
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import CheckpointError
from conrad.training.checkpoint_meta import ReasonCode
from conrad.training.run_dir import RunDirectory, RunPurpose


def test_j1_parent_checkpoint_loading_equality() -> None:
    model = build_j1_model()
    before = {k: v.detach().clone() for k, v in model.rgb.state_dict().items()}
    status = load_j1_parents(model)
    assert all(item.loaded for item in status.values())
    assert status["rgb"].parameter_tensors_loaded == len(model.rgb.state_dict())
    changed = any(not torch.equal(before[k], model.rgb.state_dict()[k]) for k in before)
    assert changed
    assert status["fusion"].status == "LOADED_VERIFIED_WITH_CONTEXT_EXTENSION"
    assert status["fusion"].parameter_values_loaded < sum(v.numel() for v in model.fusion.state_dict().values())
    assert status["temporal"].max_abs_diff_after_load == 0.0


def test_j1_parent_loader_rejects_wrong_digest_component_and_shape() -> None:
    model = build_j1_model()
    rgb_spec = PARENT_SPECS["rgb"]
    with pytest.raises(CheckpointError) as wrong_digest:
        load_parent_component(
            name="rgb",
            path=REPO_ROOT / str(rgb_spec["path"]),
            expected_digest="0" * 64,
            expected_component=str(rgb_spec["component"]),
            state_prefix=str(rgb_spec["prefix"]),
            module=model.rgb,
        )
    assert ReasonCode.CONTENT_DIGEST_MISMATCH in wrong_digest.value.codes

    with pytest.raises(CheckpointError) as wrong_component:
        load_parent_component(
            name="rgb",
            path=REPO_ROOT / str(rgb_spec["path"]),
            expected_digest=str(rgb_spec["digest"]),
            expected_component="foundation.osfm.not_rgb",
            state_prefix=str(rgb_spec["prefix"]),
            module=model.rgb,
        )
    assert ReasonCode.COMPONENT_MISMATCH in wrong_component.value.codes

    bad_model = JointOSFMModel()
    del bad_model.rgb.blocks[-1]
    with pytest.raises(CheckpointError) as wrong_shape:
        load_parent_component(
            name="rgb",
            path=REPO_ROOT / str(rgb_spec["path"]),
            expected_digest=str(rgb_spec["digest"]),
            expected_component=str(rgb_spec["component"]),
            state_prefix=str(rgb_spec["prefix"]),
            module=bad_model.rgb,
        )
    assert ReasonCode.MODEL_STATE_MISMATCH in wrong_shape.value.codes


def test_j1_joint_contracts_routing_missingness_temporal_and_step() -> None:
    torch.manual_seed(20260416)
    model = build_j1_model()
    load_j1_parents(model)
    bundle = J1JointTrainingBundle(model)
    policy = bundle.configure_smoke_trainability()
    assert policy["trainable_parameters"] > 0
    assert policy["frozen_parameters"] > 0
    assert any(param.requires_grad for param in bundle.student.rgb.blocks[-1].parameters())
    fixture = synthetic_j1_fixture({"batch_size": 4, "sequence_windows": 10}, 20260416)
    out = bundle(fixture)
    assert list(out.student.fusion.scene_latents.shape) == [40, 64, 384]
    assert list(out.student.temporal.window_repr.shape) == [4, 10, 384]
    assert out.student.context is not None
    assert "context" in out.student.fusion.modality_names
    assert list(out.student.context.pooled.shape) == [40, 384]
    statuses = {result.objective_id: result.status.value for result in out.results}
    assert statuses["j1_cross_modal_consistency"] == "ACTIVE"
    assert statuses["j1_temp"] == "ACTIVE"
    assert statuses["j1_geometry_consistency"] == "ACTIVE"
    assert all(torch.isfinite(result.contribution) for result in out.results)
    natural = out.student.fusion.natural_missing_mask.reshape(4, 10, len(out.student.fusion.modality_names))
    artificial = out.student.fusion.artificial_dropout_mask.reshape(4, 10, len(out.student.fusion.modality_names))
    modality_idx = {name: idx for idx, name in enumerate(out.student.fusion.modality_names)}
    assert natural[1, 0, 2]
    assert artificial[1, 2, 1]
    assert not artificial[1, 0, 2]
    assert not natural[0, 0, modality_idx["context"]]
    assert fixture.model2_direct_evidence[0]["model2_direct_measurements"]
    assert out.student.temporal.gap_reset_mask[1, 3]
    assert out.student.temporal.reset_mask[2, 5]
    assert not out.student.temporal.adjacent_eligible_mask[3].any()
    out.loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in bundle.parameters() if p.requires_grad)
    opt = torch.optim.AdamW((p for p in bundle.parameters() if p.requires_grad), lr=1e-4)
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema = bundle.update_teacher_after_optimizer(1)
    assert 0.0 < ema < 1.0


def test_j1_deterministic_fixture_and_no_truth_leakage(tmp_path: Path) -> None:
    a = synthetic_j1_fixture({"batch_size": 4, "sequence_windows": 10}, 20260416)
    b = synthetic_j1_fixture({"batch_size": 4, "sequence_windows": 10}, 20260416)
    assert torch.equal(a.inputs.rgb, b.inputs.rgb)
    assert torch.equal(a.inputs.timestamps_s, b.inputs.timestamps_s)
    assert not hasattr(a.inputs, "truth_state")


def test_j1_run_saves_reloads_and_reports(tmp_path: Path) -> None:
    run = RunDirectory.create(
        tmp_path,
        "train-osfm_j1_joint_smoke-test",
        resolved_config={"job": "osfm_j1_joint_smoke", "batch_size": 4, "sequence_windows": 10},
        manifests={},
        purpose=RunPurpose.DEVELOPMENT,
        clock_ns=time.time_ns,
    )
    result = run_j1_joint_smoke(run, {"job": "osfm_j1_joint_smoke", "batch_size": 4, "sequence_windows": 10}, 20260416)
    assert result["reload"]["matches"]
    assert result["replay"]["deterministic"]
    assert result["provenance_leakage"]["synthetic_truth_runtime_input"] is False
    assert result["metrics"]["gradient_connected"] == 1.0
    assert result["representation_health"]["finite"] is True
    assert all(status["loaded"] for status in result["parent_checkpoint_ancestry"].values())
    assert result["architecture"]["architecture_revision"] == "OSFM-P4-CONTEXT-R02"
    assert "temperature_c" in result["architecture"]["context_schema"]
    assert result["fixture"]["model2_direct_evidence_example"]["model2_direct_measurements"]
    assert Path(result["checkpoint"]).is_file()
    assert (run.path / "reports" / "j1_joint_smoke_report.json").is_file()
