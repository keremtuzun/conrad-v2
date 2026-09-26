"""OS-FM T1 temporal smoke training path."""

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

from conrad.foundation.fusion.scene_fusion import SceneFusionConfig, SceneFusionTransformer
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter
from conrad.foundation.pretraining.m1_fusion import (
    _objective_metrics,
    parent_checkpoint_status,
    synthetic_m1_fixture,
    trainable_parameter_count,
)
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import parameter_count
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.foundation.temporal.memory import TemporalMemoryConfig, TemporalMemoryOutput, TemporalMemoryTransformer
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory


@dataclass(frozen=True)
class T1TemporalFixture:
    fused_global: torch.Tensor
    timestamps_s: torch.Tensor
    valid_mask: torch.Tensor
    boundary_reset_mask: torch.Tensor
    sequence_ids: tuple[str, ...]
    structure: tuple[str, ...]
    m1_relation_rows: tuple[tuple[str, ...], ...]
    m1_parent_status: dict[str, dict[str, str | bool | None]]


@dataclass(frozen=True)
class T1StepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: TemporalMemoryOutput
    teacher: TemporalMemoryOutput


def synthetic_t1_fixture(job: dict[str, Any], seed: int, fusion: SceneFusionTransformer | None = None) -> T1TemporalFixture:
    torch.manual_seed(seed)
    steps = int(job.get("sequence_windows", 10))
    if steps != 10:
        raise ValueError("P3 baseline sequence construction is fixed at 10 windows for T1 smoke")
    m1_fixture = synthetic_m1_fixture({"batch_size": 4, **job}, seed)
    fusion_model = fusion or SceneFusionTransformer(SceneFusionConfig())
    fusion_model.eval()
    windows: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(steps):
            shifted_sets = []
            for token_set in m1_fixture.token_sets:
                shifted_sets.append(
                    type(token_set)(
                        token_set.modality,
                        token_set.tokens + (step * 0.001),
                        token_set.valid_token_mask,
                        token_set.naturally_missing,
                        token_set.artificially_dropped,
                        token_set.padded,
                    )
                )
            fused = fusion_model(tuple(shifted_sets)).global_repr
            windows.append(fused)
    stacked = torch.stack(windows, dim=1)
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
        [
            [True] * steps,
            [True] * steps,
            [True] * steps,
            [True] + [False] * (steps - 1),
        ],
        dtype=torch.bool,
    )
    boundary = torch.zeros(4, steps, dtype=torch.bool)
    boundary[2, 5] = True
    return T1TemporalFixture(
        fused_global=stacked,
        timestamps_s=timestamps,
        valid_mask=valid,
        boundary_reset_mask=boundary,
        sequence_ids=("continuous-5s", "gap-split", "capture-boundary", "short-single"),
        structure=(
            "10 valid windows at 500 ms spacing, one bounded ~5 s sequence",
            "10 valid windows with a 1.25 s gap before window 3; temporal state resets there",
            "10 valid windows with an explicit new-capture boundary before window 5",
            "one valid window only; TEMP is NOT_APPLICABLE for this row",
        ),
        m1_relation_rows=tuple(tuple(r.value for r in row) for row in m1_fixture.relation_matrix),
        m1_parent_status=parent_checkpoint_status(),
    )


class T1TemporalTrainingBundle(nn.Module):
    def __init__(self, temporal: TemporalMemoryTransformer, *, ema_schedule: EMASchedule | None = None) -> None:
        super().__init__()
        self.temporal = temporal
        self.teacher = copy.deepcopy(temporal)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        d_f = temporal.config.d_f
        self.masked_head = nn.Linear(d_f, d_f)
        self.global_projector = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, d_f))
        self.temp_predictor = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, d_f))
        self.degradation_head = nn.Linear(d_f, 2)
        self.router = ObjectiveRouter(
            {
                "t1_masked_latent_prediction": 0.5,
                "t1_global_consistency": 0.25,
                "t1_temp": 0.5,
                "t1_degradation": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)

    def forward(self, fixture: T1TemporalFixture) -> T1StepOutput:
        student = self.temporal(
            fixture.fused_global,
            fixture.timestamps_s,
            fixture.valid_mask,
            fixture.boundary_reset_mask,
        )
        with torch.no_grad():
            teacher = self.teacher(
                fixture.fused_global,
                fixture.timestamps_s,
                fixture.valid_mask,
                fixture.boundary_reset_mask,
            )
        valid = student.temporal_valid_mask
        device = fixture.fused_global.device
        deterministic_mask = valid & (torch.arange(valid.shape[1], device=device).unsqueeze(0) % 3 == 1)
        mask_loss = self.router._result(
            "t1_masked_latent_prediction",
            (F.mse_loss(self.masked_head(student.window_repr), teacher.window_repr.detach(), reduction="none").mean(dim=-1) * deterministic_mask).sum(),
            deterministic_mask.float().sum(),
        )
        global_loss = self.router._result(
            "t1_global_consistency",
            (F.mse_loss(self.global_projector(student.window_repr), teacher.window_repr.detach(), reduction="none").mean(dim=-1) * valid).sum(),
            valid.float().sum(),
        )
        temp_mask = student.adjacent_eligible_mask
        predicted_next = self.temp_predictor(student.window_repr[:, :-1])
        target_next = teacher.window_repr[:, 1:].detach()
        temp_loss = self.router._result(
            "t1_temp",
            (
                F.mse_loss(predicted_next, target_next, reduction="none").mean(dim=-1)
                * temp_mask[:, 1:].float()
            ).sum(),
            temp_mask.float().sum(),
        )
        reset_target = (student.reset_mask | ~valid).long()
        degradation_loss = self.router._result(
            "t1_degradation",
            (F.cross_entropy(self.degradation_head(student.window_repr).flatten(0, 1), reset_target.flatten(), reduction="none").view_as(valid) * valid).sum(),
            valid.float().sum(),
        )
        results = (mask_loss, global_loss, temp_loss, degradation_loss)
        return T1StepOutput(self.router.total(results), results, student, teacher)

    @torch.no_grad()
    def update_teacher_after_optimizer(self, step: int) -> float:
        momentum = self.ema_schedule.value(step)
        update_ema_teacher(self.temporal, self.teacher, momentum)
        self.teacher.eval()
        return momentum


def run_t1_temporal_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    fusion = SceneFusionTransformer(SceneFusionConfig())
    for param in fusion.parameters():
        param.requires_grad_(False)
    fixture = synthetic_t1_fixture(job, seed, fusion)
    temporal_config = TemporalMemoryConfig()
    bundle = T1TemporalTrainingBundle(TemporalMemoryTransformer(temporal_config))
    optimizer = torch.optim.AdamW((p for p in bundle.parameters() if p.requires_grad), lr=float(job.get("lr", 2e-4)))
    out = bundle(fixture)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema = bundle.update_teacher_after_optimizer(1)
    bundle.eval()
    with torch.no_grad():
        post = bundle(fixture)
    replay = synthetic_t1_fixture(job, seed, fusion)
    replay_same = (
        fixture.sequence_ids == replay.sequence_ids
        and fixture.structure == replay.structure
        and torch.equal(fixture.timestamps_s, replay.timestamps_s)
        and torch.equal(fixture.valid_mask, replay.valid_mask)
        and torch.equal(fixture.boundary_reset_mask, replay.boundary_reset_mask)
    )
    split_plan = {
        "sequence_ids": fixture.sequence_ids,
        "timestamps_s": fixture.timestamps_s.tolist(),
        "valid_mask": fixture.valid_mask.to(torch.int).tolist(),
        "boundary_reset_mask": fixture.boundary_reset_mask.to(torch.int).tolist(),
        "seed": seed,
    }
    split_hash = split_hash_for_plan(split_plan)
    health = _representation_health(post.student.window_repr[post.student.temporal_valid_mask])
    metrics: dict[str, float | str] = {
        "val/osfm_t1_temporal_loss": float(post.loss.detach().cpu()),
        "optimizer_steps": 1.0,
        "ema_updates": 1.0,
        "ema_momentum": float(ema),
        "valid_windows": float(post.student.temporal_valid_mask.float().sum().detach().cpu()),
        "sequence_count": float(len(fixture.sequence_ids)),
        "gap_reset_events": float(post.student.gap_reset_mask.float().sum().detach().cpu()),
        "explicit_boundary_resets": float(fixture.boundary_reset_mask.float().sum().detach().cpu()),
        "temp_eligible_pairs": float(post.student.adjacent_eligible_mask.float().sum().detach().cpu()),
        "temporal_parameter_count": float(parameter_count(bundle.temporal)),
        "t1_trainable_parameter_count": float(trainable_parameter_count(bundle)),
        "representation_variance_mean": health.variance_mean,
        "representation_collapse_score": health.collapse_score,
        "representation_effective_rank": health.effective_rank,
        "formal_rank_guard_numeric": {"PASS": 1.0, "FAIL": 0.0, "NOT_EVALUABLE": -1.0}[health.formal_rank_guard],
        "memory_allocated_bytes": float(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
    }
    metrics.update({k: v for k, v in _objective_metrics(post.results).items() if isinstance(v, float)})
    manifest_digest = hashlib.sha256(b"SYNTHETIC:osfm-t1-temporal-smoke:v1").hexdigest()
    ckpt = run.path / "checkpoints" / "osfm_t1_temporal_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.t1_temporal",
        config=job,
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        seeds={"torch": seed, "fixture": seed},
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={"cuda_rng_state": [], "amp_scaler": None, "sequence_ids": fixture.sequence_ids},
        is_encoder=True,
        representation_pretraining_id="OSFM-T1-TEMPORAL-SMOKE-001",
        selection_metric="val/osfm_t1_temporal_loss",
        extra_compatibility={"osfm_stage": "T1-TEMPORAL", "p4_smoke": "true"},
    )
    fresh = T1TemporalTrainingBundle(TemporalMemoryTransformer(temporal_config))
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.t1_temporal",
            representation_pretraining_id="OSFM-T1-TEMPORAL-SMOKE-001",
            extra={"osfm_stage": "T1-TEMPORAL", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(fixture)
    reload_max_abs_diff = float((post.student.window_repr - reloaded.student.window_repr).abs().max().detach().cpu())
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.t1_temporal",
        "experiment_id": "OSFM-T1-TEMPORAL-SMOKE-001",
        "architecture": {
            "position": "after_scene_fusion",
            "d_f": temporal_config.d_f,
            "memory_tokens": [temporal_config.memory_tokens, temporal_config.d_f],
            "transformer_blocks": temporal_config.transformer_blocks,
            "heads": temporal_config.heads,
            "mlp_expansion": temporal_config.mlp_ratio,
            "max_windows": temporal_config.max_windows,
            "gap_reset_s": temporal_config.gap_reset_s,
            "outputs": {
                "window_repr": list(post.student.window_repr.shape),
                "memory_states": list(post.student.memory_states.shape),
            },
            "temporal_parameter_count": parameter_count(bundle.temporal),
            "t1_trainable_parameter_count": trainable_parameter_count(bundle),
        },
        "m1_input_path": {
            "actual_m1_fusion_outputs_used": True,
            "parent_u1_checkpoints_loaded": False,
            "parent_checkpoint_ancestry": fixture.m1_parent_status,
            "note": "P4.5 smoke exercises SceneFusionTransformer in-path on synthetic U1 encoder fixture outputs; promoted U1 parent checkpoint loading remains verified-but-not-loaded as in P4.4.",
        },
        "fixture": {
            "sequence_ids": list(fixture.sequence_ids),
            "structure": list(fixture.structure),
            "timestamps_s": fixture.timestamps_s.tolist(),
            "valid_mask": fixture.valid_mask.to(torch.int).tolist(),
            "boundary_reset_mask": fixture.boundary_reset_mask.to(torch.int).tolist(),
            "reset_mask": post.student.reset_mask.to(torch.int).tolist(),
            "gap_reset_mask": post.student.gap_reset_mask.to(torch.int).tolist(),
            "adjacent_eligible_mask": post.student.adjacent_eligible_mask.to(torch.int).tolist(),
            "history_lengths": post.student.history_lengths.tolist(),
            "m1_relation_rows": [list(row) for row in fixture.m1_relation_rows],
        },
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {
            "deterministic": replay_same,
            "fixture_hash": hashlib.sha256(json.dumps(split_plan, sort_keys=True).encode()).hexdigest(),
        },
        "representation_health": health.__dict__,
        "provenance_leakage": {"truth_state_used": False, "persistent_model2_state_used": False, "downstream_state_inference": False},
        "compute": {
            "device": "cpu",
            "peak_memory_bytes": int(metrics["memory_allocated_bytes"]),
            "wall_clock_s": wall_s,
            "throughput_windows_per_s": int(metrics["valid_windows"]) / wall_s,
        },
    }
    run.log_metrics({"step": 1, **metrics})
    run.log_event(
        "OSFM_T1_TEMPORAL_SMOKE_STEP",
        time.time_ns(),
        {
            "sequence_ids": fixture.sequence_ids,
            "reset_events": int(post.student.reset_mask.sum().detach().cpu()),
            "gap_reset_events": int(post.student.gap_reset_mask.sum().detach().cpu()),
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": health.formal_rank_guard,
        },
    )
    run.write_artifact("reports", "t1_temporal_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact("reports", "checkpoint_index.json", json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2))
    return result
