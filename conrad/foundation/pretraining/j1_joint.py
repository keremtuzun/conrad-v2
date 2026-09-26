"""OS-FM J1 complete joint architecture smoke."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from conrad.foundation.data.pairing import PairEdge, PairGraph, PairRelation
from conrad.foundation.encoders.geometry import GeometryEncoderConfig
from conrad.foundation.encoders.range import RangeEncoderConfig
from conrad.foundation.encoders.rgb import RGBEncoderConfig
from conrad.foundation.encoders.sonar import SonarEncoderConfig
from conrad.foundation.joint import JointOSFMInputs, JointOSFMModel, JointOSFMOutput
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter
from conrad.foundation.pretraining.m1_fusion import _objective_metrics, trainable_parameter_count
from conrad.foundation.pretraining.parents import ParentLoadStatus, load_parent_component
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import parameter_count
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory

MODALITIES = ("rgb", "sonar", "range", "geometry")
PARENT_SPECS = {
    "rgb": {
        "path": "artifacts/runs/train-osfm_u1_rgb_smoke-1790410824000679000/checkpoints/osfm_u1_rgb_smoke.pt",
        "digest": "7909a94586f244b9e7ec6f71cbbf274c047b979112d41f8347b7cb8250cb9fe1",
        "component": "foundation.osfm.u1_rgb",
        "prefix": "student",
    },
    "sonar": {
        "path": "artifacts/runs/train-osfm_u1_sonar_smoke-1790411261663354000/checkpoints/osfm_u1_sonar_smoke.pt",
        "digest": "9e7e68cbdcd63d9816323c2b2925862ecc2e142edd378070211df2f3e70955a0",
        "component": "foundation.osfm.u1_sonar",
        "prefix": "student",
    },
    "range": {
        "path": "artifacts/runs/train-osfm_u1_range_smoke-1790411749668941000/checkpoints/osfm_u1_range_smoke.pt",
        "digest": "c797695601b8ee5875165d9c3c9d7da4059726c3d300b50890cdbedd658e9256",
        "component": "foundation.osfm.u1_range",
        "prefix": "student",
    },
    "geometry": {
        "path": "artifacts/runs/train-osfm_u1_geometry_smoke-1790412183310131000/checkpoints/osfm_u1_geometry_smoke.pt",
        "digest": "cf82dede076700d68ece32be888234dc34ef008f7d18e6cd22adc1ce99cc4583",
        "component": "foundation.osfm.u1_geometry",
        "prefix": "student",
    },
    "fusion": {
        "path": "artifacts/runs/train-osfm_m1_fusion_smoke-1790412769496868000/checkpoints/osfm_m1_fusion_smoke.pt",
        "digest": "10cd12aa2590a2241f470b14266e6d37b810d3c4c29a2c547c47aecf6577f15f",
        "component": "foundation.osfm.m1_fusion",
        "prefix": "fusion",
    },
    "temporal": {
        "path": "artifacts/runs/train-osfm_t1_temporal_smoke-1790413248477123000/checkpoints/osfm_t1_temporal_smoke.pt",
        "digest": "7fb161859e97735b6e938a0a8e9af095bf794a317c182c45582a7c068160d629",
        "component": "foundation.osfm.t1_temporal",
        "prefix": "temporal",
    },
}


@dataclass(frozen=True)
class J1Fixture:
    inputs: JointOSFMInputs
    pair_graphs: tuple[tuple[PairGraph, ...], ...]
    sequence_ids: tuple[str, ...]
    structure: tuple[str, ...]
    metric_capable: torch.Tensor
    geometry_capable: torch.Tensor


@dataclass(frozen=True)
class J1StepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: JointOSFMOutput
    teacher: JointOSFMOutput


def _mask_dict(batch: int, steps: int) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(batch, steps, dtype=torch.bool) for name in MODALITIES}


def synthetic_j1_fixture(job: dict[str, Any], seed: int) -> J1Fixture:
    torch.manual_seed(seed)
    batch = int(job.get("batch_size", 4))
    steps = int(job.get("sequence_windows", 10))
    if batch != 4 or steps != 10:
        raise ValueError("J1 smoke fixture is fixed at batch_size=4 and sequence_windows=10")
    rgb_cfg = RGBEncoderConfig(image_size=int(job.get("rgb_image_size", 28)))
    range_cfg = RangeEncoderConfig(image_size=int(job.get("range_image_size", 32)))
    geom_points = int(job.get("points_per_sample", 64))
    base_rgb = torch.linspace(0, 1, rgb_cfg.image_size * rgb_cfg.image_size).reshape(1, 1, rgb_cfg.image_size, rgb_cfg.image_size)
    base_range = torch.linspace(0, 1, range_cfg.image_size * range_cfg.image_size).reshape(1, 1, range_cfg.image_size, range_cfg.image_size)
    rgb = base_rgb.repeat(batch, steps, 3, 1, 1)
    sonar = base_rgb.repeat(batch, steps, 1, 1, 1)
    range_raster = base_range.repeat(batch, steps, 1, 1, 1)
    time_offsets = torch.arange(steps).view(1, steps, 1, 1, 1) * 0.005
    row_offsets = torch.arange(batch).view(batch, 1, 1, 1, 1) * 0.03
    rgb = (rgb + row_offsets + time_offsets).clamp(0, 1)
    sonar = (sonar * 0.8 + row_offsets + time_offsets).clamp(0, 1)
    range_raster = (range_raster + row_offsets + time_offsets).clamp(0, 1)
    range_validity = torch.ones_like(range_raster, dtype=torch.bool)
    points = torch.linspace(-1, 1, geom_points)
    xyz = torch.stack([points, points.square(), torch.sin(points * 3.14)], dim=-1)
    geometry = xyz.reshape(1, 1, geom_points, 3).repeat(batch, steps, 1, 1) + row_offsets.squeeze(-1)
    geometry_validity = torch.ones(batch, steps, geom_points, dtype=torch.bool)
    natural = _mask_dict(batch, steps)
    artificial = _mask_dict(batch, steps)
    padding = _mask_dict(batch, steps)
    natural["range"][1, :] = True
    artificial["sonar"][1, 2] = True
    natural["rgb"][2, :] = True
    natural["range"][2, :] = True
    natural["geometry"][2, :] = True
    natural["sonar"][3, :] = True
    natural["geometry"][3, :] = True
    timestamps = torch.tensor(
        [
            [idx * 0.5 for idx in range(steps)],
            [0.0, 0.5, 1.0, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25],
            [idx * 0.5 for idx in range(steps)],
            [0.0] + [0.0 for _ in range(steps - 1)],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [[True] * steps, [True] * steps, [True] * steps, [True] + [False] * (steps - 1)],
        dtype=torch.bool,
    )
    boundary = torch.zeros(batch, steps, dtype=torch.bool)
    boundary[2, 5] = True
    pair_rows = []
    for _ in range(steps):
        pair_rows.append(
            (
                PairGraph(("rgb", "sonar", "range", "geometry"), (PairEdge("rgb", "sonar", PairRelation.SAME_WINDOW, 4), PairEdge("rgb", "range", PairRelation.SAME_WINDOW, 7), PairEdge("range", "geometry", PairRelation.SAME_WINDOW, 3))),
                PairGraph(("rgb", "sonar", "geometry"), (PairEdge("rgb", "geometry", PairRelation.SAME_WINDOW, 8),)),
                PairGraph(("sonar",), ()),
                PairGraph(("rgb", "range"), (PairEdge("rgb", "range", PairRelation.UNPAIRED, 0),)),
            )
        )
    return J1Fixture(
        inputs=JointOSFMInputs(rgb, sonar, range_raster, range_validity, geometry, geometry_validity, natural, artificial, padding, timestamps, valid, boundary),
        pair_graphs=tuple(tuple(row) for row in zip(*pair_rows)),
        sequence_ids=("fully-paired-temporal", "natural-missing-plus-dropout-gap", "sonar-only-boundary", "unpaired-single-window"),
        structure=(
            "all modalities present, legitimate rgb/sonar/range/geometry pairs, 10 temporal windows",
            "natural range missing, artificial sonar dropout at one window, 1.25 s temporal gap reset",
            "sonar-only single-modality sequence with explicit boundary reset; cross-modal/geometric objectives not applicable for row",
            "rgb+range unpaired single valid window; temporal objective not applicable",
        ),
        metric_capable=torch.tensor([[True] * steps, [False] * steps, [False] * steps, [True] + [False] * 9]),
        geometry_capable=torch.tensor([[True] * steps, [True] * steps, [False] * steps, [False] * steps]),
    )


def build_j1_model() -> JointOSFMModel:
    return JointOSFMModel(
        rgb=None,
        sonar=None,
        range_encoder=None,
        geometry=None,
        fusion=None,
        temporal=None,
    )


def load_j1_parents(model: JointOSFMModel) -> dict[str, ParentLoadStatus]:
    modules = {
        "rgb": model.rgb,
        "sonar": model.sonar,
        "range": model.range,
        "geometry": model.geometry,
        "fusion": model.fusion,
        "temporal": model.temporal,
    }
    status = {}
    for name, spec in PARENT_SPECS.items():
        status[name] = load_parent_component(
            name=name,
            path=REPO_ROOT / str(spec["path"]),
            expected_digest=str(spec["digest"]),
            expected_component=str(spec["component"]),
            state_prefix=str(spec["prefix"]),
            module=modules[name],
        )
    return status


class J1JointTrainingBundle(nn.Module):
    def __init__(self, model: JointOSFMModel, *, ema_schedule: EMASchedule | None = None) -> None:
        super().__init__()
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        d_f = 384
        self.masked_head = nn.Linear(d_f, d_f)
        self.global_projector = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, d_f))
        self.temp_predictor = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, d_f))
        self.modality_decoders = nn.ModuleDict({m: nn.Linear(d_f, d_f) for m in MODALITIES})
        self.degradation_head = nn.Linear(d_f, 2)
        self.geometry_head = nn.Linear(d_f, d_f)
        self.metric_head = nn.Linear(d_f, d_f)
        self.router = ObjectiveRouter(
            {
                "j1_masked_latent_prediction": 1.0,
                "j1_global_consistency": 0.5,
                "j1_cross_modal_consistency": 0.5,
                "j1_temp": 0.5,
                "j1_missing_modality": 0.5,
                "j1_degradation": 0.25,
                "j1_geometry_consistency": 0.25,
                "j1_metric_reconstruction": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)

    def configure_smoke_trainability(self) -> dict[str, int | str]:
        for param in self.student.parameters():
            param.requires_grad_(False)
        for module in (self.student.fusion, self.student.temporal, self.masked_head, self.global_projector, self.temp_predictor, self.modality_decoders, self.degradation_head, self.geometry_head, self.metric_head):
            for param in module.parameters():
                param.requires_grad_(True)
        for param in self.student.rgb.blocks[-1].parameters():
            param.requires_grad_(True)
        return {
            "policy": "freeze U1 parents except final RGB encoder block; train fusion, temporal, and J1 heads",
            "trainable_parameters": trainable_parameter_count(self),
            "frozen_parameters": parameter_count(self) - trainable_parameter_count(self),
        }

    def _eligible_pair_mask(self, fixture: J1Fixture, output: JointOSFMOutput, pairs: tuple[tuple[str, str], ...]) -> torch.Tensor:
        fused = output.fusion
        batch, steps = fixture.inputs.temporal_valid_mask.shape
        idx = {name: pos for pos, name in enumerate(fused.modality_names)}
        present = fused.modality_presence.reshape(batch, steps, len(fused.modality_names))
        eligible = torch.zeros(batch, steps, dtype=torch.bool, device=fused.global_repr.device)
        for row in range(batch):
            for step in range(steps):
                graph = fixture.pair_graphs[row][step]
                for left, right in pairs:
                    if left not in idx or right not in idx:
                        continue
                    if bool(present[row, step, idx[left]] and present[row, step, idx[right]]) and graph.relation(left, right) is PairRelation.SAME_WINDOW:
                        eligible[row, step] = True
        return eligible

    def forward(self, fixture: J1Fixture) -> J1StepOutput:
        student = self.student(fixture.inputs)
        with torch.no_grad():
            teacher = self.teacher(fixture.inputs)
        batch, steps = fixture.inputs.temporal_valid_mask.shape
        scene = student.fusion.scene_latents
        device = scene.device
        flat_valid = fixture.inputs.temporal_valid_mask.reshape(batch * steps).to(device)
        latent_mask = torch.zeros(scene.shape[:2], dtype=torch.bool, device=device)
        latent_mask[:, ::3] = flat_valid.unsqueeze(1)
        mask_loss = self.router._result(
            "j1_masked_latent_prediction",
            (F.mse_loss(self.masked_head(scene), teacher.fusion.scene_latents.detach(), reduction="none").mean(dim=-1) * latent_mask).sum(),
            latent_mask.float().sum(),
        )
        global_student = student.temporal.window_repr
        global_teacher = teacher.temporal.window_repr.detach()
        valid = student.temporal.temporal_valid_mask
        global_loss = self.router._result(
            "j1_global_consistency",
            (F.mse_loss(self.global_projector(global_student), global_teacher, reduction="none").mean(dim=-1) * valid).sum(),
            valid.float().sum(),
        )
        xm_mask = self._eligible_pair_mask(fixture, student, (("rgb", "sonar"), ("rgb", "range"), ("range", "geometry")))
        xm_loss = self.router._result(
            "j1_cross_modal_consistency",
            (F.mse_loss(global_student, global_teacher, reduction="none").mean(dim=-1) * xm_mask).sum(),
            xm_mask.float().sum(),
        )
        temp_mask = student.temporal.adjacent_eligible_mask
        temp_loss = self.router._result(
            "j1_temp",
            (F.mse_loss(self.temp_predictor(global_student[:, :-1]), global_teacher[:, 1:], reduction="none").mean(dim=-1) * temp_mask[:, 1:]).sum(),
            temp_mask.float().sum(),
        )
        natural = student.fusion.natural_missing_mask.reshape(batch, steps, -1)
        artificial = student.fusion.artificial_dropout_mask.reshape(batch, steps, -1)
        padding = student.fusion.padding_mask.reshape(batch, steps, -1)
        missing_slots = (natural | artificial) & ~padding
        decoded = torch.stack([head(global_student) for head in self.modality_decoders.values()], dim=2)
        missing_loss = self.router._result(
            "j1_missing_modality",
            (F.mse_loss(decoded, global_teacher.unsqueeze(2).expand_as(decoded), reduction="none").mean(dim=-1) * missing_slots).sum(),
            missing_slots.float().sum(),
        )
        degradation_labels = ((natural | artificial).any(dim=2) | student.temporal.reset_mask).long()
        degradation_loss = self.router._result(
            "j1_degradation",
            (F.cross_entropy(self.degradation_head(global_student).flatten(0, 1), degradation_labels.flatten(), reduction="none").view_as(valid) * valid).sum(),
            valid.float().sum(),
        )
        geo_mask = self._eligible_pair_mask(fixture, student, (("range", "geometry"),)) & fixture.geometry_capable.to(device) & valid
        geo_loss = self.router._result(
            "j1_geometry_consistency",
            (F.mse_loss(self.geometry_head(global_student), global_teacher, reduction="none").mean(dim=-1) * geo_mask).sum(),
            geo_mask.float().sum(),
        )
        metric_mask = fixture.metric_capable.to(device) & valid
        metric_loss = self.router._result(
            "j1_metric_reconstruction",
            (F.mse_loss(self.metric_head(global_student), global_teacher, reduction="none").mean(dim=-1) * metric_mask).sum(),
            metric_mask.float().sum(),
        )
        results = (mask_loss, global_loss, xm_loss, temp_loss, missing_loss, degradation_loss, geo_loss, metric_loss)
        return J1StepOutput(self.router.total(results), results, student, teacher)

    @torch.no_grad()
    def update_teacher_after_optimizer(self, step: int) -> float:
        momentum = self.ema_schedule.value(step)
        update_ema_teacher(self.student, self.teacher, momentum)
        self.teacher.eval()
        return momentum


def _coverage(fixture: J1Fixture, out: JointOSFMOutput) -> dict[str, Any]:
    batch, steps = fixture.inputs.temporal_valid_mask.shape
    return {
        "sequence_ids": list(fixture.sequence_ids),
        "structure": list(fixture.structure),
        "modality_presence": out.fusion.modality_presence.reshape(batch, steps, -1).to(torch.int).tolist(),
        "natural_missing": out.fusion.natural_missing_mask.reshape(batch, steps, -1).to(torch.int).tolist(),
        "artificial_dropout": out.fusion.artificial_dropout_mask.reshape(batch, steps, -1).to(torch.int).tolist(),
        "padding": out.fusion.padding_mask.reshape(batch, steps, -1).to(torch.int).tolist(),
        "timestamps_s": fixture.inputs.timestamps_s.tolist(),
        "temporal_valid_mask": fixture.inputs.temporal_valid_mask.to(torch.int).tolist(),
        "boundary_reset_mask": fixture.inputs.boundary_reset_mask.to(torch.int).tolist(),
        "reset_mask": out.temporal.reset_mask.to(torch.int).tolist(),
        "gap_reset_mask": out.temporal.gap_reset_mask.to(torch.int).tolist(),
        "adjacent_eligible_mask": out.temporal.adjacent_eligible_mask.to(torch.int).tolist(),
        "history_lengths": out.temporal.history_lengths.tolist(),
    }


def run_j1_joint_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = build_j1_model()
    parent_status = load_j1_parents(model)
    bundle = J1JointTrainingBundle(model)
    trainable_policy = bundle.configure_smoke_trainability()
    fixture = synthetic_j1_fixture(job, seed)
    optimizer = torch.optim.AdamW((p for p in bundle.parameters() if p.requires_grad), lr=float(job.get("lr", 1e-4)))
    bundle.train()
    out = bundle(fixture)
    out.loss.backward()
    grad_connected = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0.0 for p in bundle.parameters() if p.requires_grad)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema = bundle.update_teacher_after_optimizer(1)
    bundle.eval()
    with torch.no_grad():
        post = bundle(fixture)
    replay_fixture = synthetic_j1_fixture(job, seed)
    replay_same = (
        fixture.sequence_ids == replay_fixture.sequence_ids
        and torch.equal(fixture.inputs.timestamps_s, replay_fixture.inputs.timestamps_s)
        and torch.equal(fixture.inputs.temporal_valid_mask, replay_fixture.inputs.temporal_valid_mask)
        and torch.equal(fixture.inputs.boundary_reset_mask, replay_fixture.inputs.boundary_reset_mask)
    )
    split_plan = {
        "sequence_ids": fixture.sequence_ids,
        "timestamps_s": fixture.inputs.timestamps_s.tolist(),
        "valid": fixture.inputs.temporal_valid_mask.to(torch.int).tolist(),
        "boundary": fixture.inputs.boundary_reset_mask.to(torch.int).tolist(),
        "seed": seed,
    }
    split_hash = split_hash_for_plan(split_plan)
    health = _representation_health(post.student.temporal.window_repr[post.student.temporal.temporal_valid_mask])
    metrics: dict[str, float | str] = {
        "val/osfm_j1_joint_loss": float(post.loss.detach().cpu()),
        "optimizer_steps": 1.0,
        "ema_updates": 1.0,
        "ema_momentum": float(ema),
        "gradient_connected": float(grad_connected),
        "scene_latents_finite": float(torch.isfinite(post.student.fusion.scene_latents).all().detach().cpu()),
        "temporal_repr_finite": float(torch.isfinite(post.student.temporal.window_repr).all().detach().cpu()),
        "valid_windows": float(post.student.temporal.temporal_valid_mask.float().sum().detach().cpu()),
        "gap_reset_events": float(post.student.temporal.gap_reset_mask.float().sum().detach().cpu()),
        "explicit_boundary_resets": float(fixture.inputs.boundary_reset_mask.float().sum().detach().cpu()),
        "temp_eligible_pairs": float(post.student.temporal.adjacent_eligible_mask.float().sum().detach().cpu()),
        "total_parameter_count": float(parameter_count(bundle)),
        "trainable_parameter_count": float(trainable_parameter_count(bundle)),
        "representation_variance_mean": health.variance_mean,
        "representation_collapse_score": health.collapse_score,
        "representation_effective_rank": health.effective_rank,
        "formal_rank_guard_numeric": {"PASS": 1.0, "FAIL": 0.0, "NOT_EVALUABLE": -1.0}[health.formal_rank_guard],
        "memory_allocated_bytes": float(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
    }
    metrics.update({k: v for k, v in _objective_metrics(post.results).items() if isinstance(v, float)})
    manifest_digest = hashlib.sha256(b"SYNTHETIC:osfm-j1-joint-smoke:v1").hexdigest()
    ckpt = run.path / "checkpoints" / "osfm_j1_joint_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.j1_joint",
        config=job,
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        seeds={"torch": seed, "fixture": seed},
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={"cuda_rng_state": [], "sequence_ids": fixture.sequence_ids},
        is_encoder=True,
        representation_pretraining_id="OSFM-J1-JOINT-SMOKE-001",
        selection_metric="val/osfm_j1_joint_loss",
        extra_compatibility={"osfm_stage": "J1-JOINT", "p4_smoke": "true"},
    )
    fresh = J1JointTrainingBundle(build_j1_model())
    fresh.configure_smoke_trainability()
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.j1_joint",
            representation_pretraining_id="OSFM-J1-JOINT-SMOKE-001",
            extra={"osfm_stage": "J1-JOINT", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(fixture)
    reload_max_abs_diff = float((post.student.temporal.window_repr - reloaded.student.temporal.window_repr).abs().max().detach().cpu())
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.j1_joint",
        "experiment_id": "OSFM-J1-JOINT-SMOKE-001",
        "architecture": {
            "d_f": 384,
            "encoders": {
                "rgb": "ViT-S/14 12L 6H width384 dense taps 3/6/9/12",
                "sonar": "independent single-channel ViT-S/14 12L 6H width384 dense taps 3/6/9/12",
                "range": "patch8 width256 6L 4H projection256->384",
                "geometry": "Point-MAE-style grouped width256 6L 4H projection256->384",
            },
            "fusion": {"scene_latents": [64, 384], "cross_attention_blocks": 3, "latent_self_attention_blocks": 6, "heads": 6, "mlp_expansion": 4},
            "temporal": {"memory_tokens": [16, 384], "transformer_blocks": 4, "heads": 6, "mlp_expansion": 4, "max_windows": 10, "gap_reset_s": 1.0},
            "outputs": {"fusion_global": list(post.student.fusion.global_repr.shape), "temporal_window_repr": list(post.student.temporal.window_repr.shape)},
            "parameter_counts": {"total": parameter_count(bundle), **trainable_policy},
        },
        "parent_checkpoint_ancestry": {name: status.as_dict() for name, status in parent_status.items()},
        "trainable_policy": trainable_policy,
        "fixture": _coverage(fixture, post.student),
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {"deterministic": replay_same, "fixture_hash": hashlib.sha256(json.dumps(split_plan, sort_keys=True).encode()).hexdigest()},
        "representation_health": health.__dict__,
        "provenance_leakage": {"synthetic_truth_runtime_input": False, "truth_state_used": False, "persistent_model2_state_used": False, "downstream_state_inference": False},
        "compute": {"device": "cpu", "peak_memory_bytes": int(metrics["memory_allocated_bytes"]), "wall_clock_s": wall_s, "throughput_windows_per_s": int(metrics["valid_windows"]) / wall_s},
    }
    run.log_metrics({"step": 1, **metrics})
    run.log_event(
        "OSFM_J1_JOINT_SMOKE_STEP",
        time.time_ns(),
        {
            "sequence_ids": fixture.sequence_ids,
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": health.formal_rank_guard,
            "parents_loaded": {name: status.loaded for name, status in parent_status.items()},
        },
    )
    run.write_artifact("reports", "j1_joint_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact("reports", "checkpoint_index.json", json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2))
    return result
