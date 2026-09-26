"""OS-FM U1 grouped-geometry smoke training path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from conrad.foundation.encoders.geometry import GeometryEncoderConfig, GeometryEncoderOutput, GeometryGroupedEncoder
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import RepresentationHealth, parameter_count
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory


@dataclass(frozen=True)
class U1GeometryViews:
    teacher_points: torch.Tensor
    teacher_validity_mask: torch.Tensor
    student_points: torch.Tensor
    student_validity_mask: torch.Tensor
    metric_targets: torch.Tensor
    metric_validity_mask: torch.Tensor
    group_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    corruption_trace: tuple[str, ...]


@dataclass(frozen=True)
class U1GeometryStepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: GeometryEncoderOutput
    teacher: GeometryEncoderOutput
    metric_prediction: torch.Tensor
    degradation_logits: torch.Tensor
    health: RepresentationHealth


def deterministic_group_mask(
    *,
    batch_size: int,
    num_groups: int,
    mask_fraction: float,
    min_visible_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    max_masked = num_groups - max(1, math.ceil(num_groups * min_visible_fraction))
    target = min(max_masked, max(1, round(num_groups * mask_fraction)))
    mask = torch.zeros(batch_size, num_groups, dtype=torch.bool)
    for row in range(batch_size):
        perm = torch.randperm(num_groups, generator=generator)
        mask[row, perm[:target]] = True
    return mask


def _geometry_config_from_job(job: dict[str, Any]) -> GeometryEncoderConfig:
    return GeometryEncoderConfig(
        num_groups=int(job.get("num_groups", job.get("smoke_num_groups", 128))),
        group_size=int(job.get("group_size", job.get("smoke_group_size", 32))),
    )


def synthetic_geometry_views(job: dict[str, Any], seed: int, *, all_invalid: bool = False) -> U1GeometryViews:
    cfg = _geometry_config_from_job(job)
    batch = int(job.get("batch_size", 2))
    points_per_sample = int(job.get("points_per_sample", max(cfg.num_groups * cfg.group_size, 64)))
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, points_per_sample, dtype=torch.float32)
    angle = t * torch.pi * 4.0
    radius = 1.0 + 0.18 * torch.sin(t * torch.pi * 3.0)
    base = torch.stack(
        [
            radius * torch.cos(angle),
            radius * torch.sin(angle),
            0.35 * torch.sin(t * torch.pi * 2.0) + 0.8 * t,
        ],
        dim=-1,
    )
    teacher = base.unsqueeze(0).repeat(batch, 1, 1)
    offsets = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1) * torch.tensor([0.04, -0.03, 0.02])
    teacher = teacher + offsets
    valid = torch.ones(batch, points_per_sample, dtype=torch.bool)
    valid[:, ::7] = False
    valid[:, points_per_sample // 3 : points_per_sample // 3 + max(2, points_per_sample // 16)] = False
    if all_invalid:
        valid = torch.zeros_like(valid)
    teacher_recorded = teacher.masked_fill(~valid.unsqueeze(-1), 0.0)
    ranges = torch.linalg.norm(teacher, dim=-1, keepdim=True).clamp_min(1e-6)
    directions = teacher / ranges
    range_noise = torch.randn((batch, points_per_sample, 1), generator=gen, dtype=teacher.dtype) * 0.015
    student = teacher + directions * range_noise
    dropout = torch.rand((batch, points_per_sample), generator=gen, dtype=teacher.dtype) < 0.08
    density_drop = (torch.arange(points_per_sample).unsqueeze(0) % 11 == 0).expand(batch, -1)
    quantized = torch.round(student / 0.01) * 0.01
    student_valid = valid & ~dropout & ~density_drop
    student_recorded = quantized.masked_fill(~student_valid.unsqueeze(-1), 0.0)
    group_mask = deterministic_group_mask(
        batch_size=batch,
        num_groups=cfg.num_groups,
        mask_fraction=float(job.get("token_mask_fraction", job.get("group_mask_fraction", 0.50))),
        min_visible_fraction=float(job.get("min_visible_fraction", 0.10)),
        generator=gen,
    )
    return U1GeometryViews(
        teacher_points=teacher_recorded,
        teacher_validity_mask=valid,
        student_points=student_recorded,
        student_validity_mask=student_valid,
        metric_targets=teacher_recorded,
        metric_validity_mask=valid,
        group_mask=group_mask,
        sample_ids=tuple(f"synthetic-metric-geometry-{idx}" for idx in range(batch)),
        corruption_trace=(
            "measurement_direction_xyz_noise",
            "validity_mask_aware_point_dropout",
            "density_reduction",
            "xyz_quantization",
        ),
    )


class U1GeometryTrainingBundle(nn.Module):
    def __init__(self, encoder: GeometryGroupedEncoder, *, ema_schedule: EMASchedule | None = None) -> None:
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.mask_predictor = nn.Linear(encoder.config.output_dim, encoder.config.output_dim)
        self.global_projector = nn.Sequential(
            nn.LayerNorm(encoder.config.output_dim),
            nn.Linear(encoder.config.output_dim, encoder.config.output_dim),
        )
        self.degradation_head = nn.Linear(encoder.config.output_dim, 2)
        self.metric_head = nn.Linear(encoder.config.output_dim, encoder.config.group_size * 3)
        self.router = ObjectiveRouter(
            {
                "u1_geometry_masked_latent_prediction": 1.0,
                "u1_geometry_global_consistency": 0.5,
                "u1_geometry_degradation": 0.25,
                "u1_geometry_metric_reconstruction": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)

    def _metric_objective(
        self,
        views: U1GeometryViews,
        student: GeometryEncoderOutput,
    ) -> tuple[ObjectiveResult, torch.Tensor]:
        pred = self.metric_head(student.group_tokens).view(
            student.group_tokens.shape[0], student.group_tokens.shape[1], self.student.config.group_size, 3
        )
        target = views.metric_targets.gather(
            1, student.neighbor_indices.reshape(views.metric_targets.shape[0], -1).unsqueeze(-1).expand(-1, -1, 3)
        ).view_as(pred)
        valid = views.metric_validity_mask.gather(
            1, student.neighbor_indices.reshape(views.metric_targets.shape[0], -1)
        ).view(pred.shape[:-1])
        eligible = valid & student.masked_group_mask.unsqueeze(-1)
        numerator = (F.mse_loss(pred, target.detach(), reduction="none") * eligible.unsqueeze(-1).float()).sum()
        denominator = eligible.float().sum() * 3.0
        return self.router._result("u1_geometry_metric_reconstruction", numerator, denominator), pred

    def forward(self, views: U1GeometryViews) -> U1GeometryStepOutput:
        student = self.student(views.student_points, views.student_validity_mask, views.group_mask)
        with torch.no_grad():
            teacher = self.teacher(
                views.teacher_points,
                views.teacher_validity_mask,
                torch.zeros_like(views.group_mask),
            )
        masked_pred = self.mask_predictor(student.group_tokens)
        valid_masked_groups = views.group_mask & (teacher.group_valid_fraction > 0)
        mask_loss = self.router._result(
            "u1_geometry_masked_latent_prediction",
            (
                F.mse_loss(masked_pred, teacher.group_tokens.detach(), reduction="none").mean(dim=-1)
                * valid_masked_groups
            ).sum(),
            valid_masked_groups.float().sum(),
        )
        global_loss = self.router._result(
            "u1_geometry_global_consistency",
            F.mse_loss(
                self.global_projector(student.modality_repr),
                teacher.modality_repr.detach(),
                reduction="sum",
            )
            / student.modality_repr.shape[-1],
            torch.tensor(float(student.modality_repr.shape[0]), device=student.modality_repr.device),
        )
        logits = self.degradation_head(student.modality_repr)
        labels = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device)
        degradation_loss = self.router._result(
            "u1_geometry_degradation",
            F.cross_entropy(logits, labels, reduction="sum"),
            torch.tensor(float(logits.shape[0]), device=logits.device),
        )
        metric_loss, metric_prediction = self._metric_objective(views, student)
        results = (mask_loss, global_loss, degradation_loss, metric_loss)
        return U1GeometryStepOutput(
            loss=self.router.total(results),
            results=results,
            student=student,
            teacher=teacher,
            metric_prediction=metric_prediction,
            degradation_logits=logits,
            health=_representation_health(student.modality_repr),
        )

    @torch.no_grad()
    def update_teacher_after_optimizer(self, step: int) -> float:
        self.teacher.float()
        momentum = self.ema_schedule.value(step)
        update_ema_teacher(self.student, self.teacher, momentum)
        self.teacher.eval()
        return momentum


def _objective_metrics(results: tuple[ObjectiveResult, ...]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for result in results:
        for key, value in result.as_metrics().items():
            metrics[key.replace("/", "_")] = value
    return metrics


def run_u1_geometry_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = _geometry_config_from_job(job)
    views = synthetic_geometry_views(job, seed)
    bundle = U1GeometryTrainingBundle(GeometryGroupedEncoder(config))
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=float(job.get("lr", 2e-4)))
    bundle.train()
    out = bundle(views)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema = bundle.update_teacher_after_optimizer(1)
    bundle.eval()
    with torch.no_grad():
        post = bundle(views)
    split_hash = split_hash_for_plan(
        {
            "sample_ids": views.sample_ids,
            "group_mask": views.group_mask.to(torch.int).tolist(),
            "metric_validity_mask": views.metric_validity_mask.to(torch.int).tolist(),
            "neighbor_indices": post.student.neighbor_indices.to(torch.int).tolist(),
            "seed": seed,
        }
    )
    manifest_digest = hashlib.sha256(b"SYNTHETIC:osfm-u1-metric-geometry-smoke:v1").hexdigest()
    valid_points = views.metric_validity_mask.float().sum()
    total_points = torch.tensor(float(views.metric_validity_mask.numel()))
    metrics: dict[str, float | str] = {
        "val/osfm_u1_geometry_loss": float(post.loss.detach().cpu()),
        "optimizer_steps": 1.0,
        "ema_updates": 1.0,
        "ema_momentum": float(ema),
        "group_mask_fraction_requested": float(job.get("token_mask_fraction", job.get("group_mask_fraction", 0.50))),
        "group_mask_fraction_actual": float(views.group_mask.float().mean().detach().cpu()),
        "visible_group_fraction_actual": float((~views.group_mask).float().mean().detach().cpu()),
        "validity_coverage": float((valid_points / total_points).detach().cpu()),
        "metric_valid_point_count": float(valid_points.detach().cpu()),
        "representation_finite": float(post.health.finite),
        "representation_variance_mean": post.health.variance_mean,
        "representation_collapse_score": post.health.collapse_score,
        "representation_effective_rank": post.health.effective_rank,
        "formal_rank_guard_numeric": {"PASS": 1.0, "FAIL": 0.0, "NOT_EVALUABLE": -1.0}[
            post.health.formal_rank_guard
        ],
        "parameter_count": float(parameter_count(bundle.student)),
        "memory_allocated_bytes": float(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
    }
    metrics.update({k: v for k, v in _objective_metrics(post.results).items() if isinstance(v, float)})
    ckpt = run.path / "checkpoints" / "osfm_u1_geometry_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.u1_geometry",
        config=job,
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        seeds={"torch": seed, "group_mask": seed},
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={"cuda_rng_state": [], "amp_scaler": None, "sample_ids": views.sample_ids},
        is_encoder=True,
        representation_pretraining_id="OSFM-U1-GEOMETRY-SMOKE-001",
        selection_metric="val/osfm_u1_geometry_loss",
        extra_compatibility={"osfm_stage": "U1-GEOMETRY", "p4_smoke": "true"},
    )
    fresh = U1GeometryTrainingBundle(GeometryGroupedEncoder(config))
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.u1_geometry",
            representation_pretraining_id="OSFM-U1-GEOMETRY-SMOKE-001",
            extra={"osfm_stage": "U1-GEOMETRY", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(views)
    replay_views = synthetic_geometry_views(job, seed)
    reload_max_abs_diff = float(
        (post.student.modality_repr - reloaded.student.modality_repr).abs().max().detach().cpu()
    )
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.u1_geometry",
        "experiment_id": "OSFM-U1-GEOMETRY-SMOKE-001",
        "architecture": {
            "encoder": "GeometryGroupedEncoder",
            "input_semantics": "XYZ-primary metric point-cloud geometry with explicit point validity mask",
            "canonical_num_groups": 128,
            "canonical_group_size": 32,
            "smoke_num_groups": config.num_groups,
            "smoke_group_size": config.group_size,
            "smoke_points_per_sample": int(job.get("points_per_sample", max(config.num_groups * config.group_size, 64))),
            "smoke_reduction": "group/point fixture reduced for CPU smoke only; width/depth/heads/projection and group masking unchanged",
            "grouping_method": config.grouping_method,
            "internal_width": config.width,
            "output_dim": config.output_dim,
            "d_f": config.output_dim,
            "layers": config.depth,
            "heads": config.heads,
            "dense_tap_layers": list(config.dense_tap_layers),
            "dense_tap_contract": "internal taps remain width 256; projected dense taps are exposed at D_F=384",
            "parameter_count": parameter_count(bundle.student),
        },
        "data": {
            "source": "synthetic_metric_point_cloud_engineering_smoke",
            "public_real_dataset_used": False,
            "not_domain_pretraining_evidence": True,
            "metric_target_present": True,
            "metric_objective_eligible": post.results[-1].status.value == "ACTIVE",
            "limitation": "Synthetic metric point-cloud fixture with known valid/invalid points; no public-real geometry evidence is claimed.",
        },
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "views": {
            "teacher": "recorded synthetic XYZ metric point-cloud with explicit invalid/missing points",
            "student_corruptions": list(views.corruption_trace),
        },
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {
            "sample_ids": list(views.sample_ids),
            "group_mask_sha256": hashlib.sha256(
                json.dumps(views.group_mask.to(torch.int).tolist()).encode()
            ).hexdigest(),
            "validity_mask_sha256": hashlib.sha256(
                json.dumps(views.metric_validity_mask.to(torch.int).tolist()).encode()
            ).hexdigest(),
            "deterministic": torch.equal(views.group_mask, replay_views.group_mask)
            and torch.equal(views.teacher_points, replay_views.teacher_points)
            and torch.equal(views.student_points, replay_views.student_points)
            and torch.equal(views.metric_validity_mask, replay_views.metric_validity_mask),
        },
        "representation_health": post.health.__dict__,
        "compute": {
            "device": "cpu",
            "peak_memory_bytes": int(metrics["memory_allocated_bytes"]),
            "wall_clock_s": wall_s,
            "throughput_samples_per_s": len(views.sample_ids) / wall_s,
        },
    }
    run.log_metrics({"step": 1, **metrics})
    run.log_event(
        "OSFM_U1_GEOMETRY_SMOKE_STEP",
        time.time_ns(),
        {
            "sample_ids": views.sample_ids,
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": post.health.formal_rank_guard,
            "student_corruptions": views.corruption_trace,
            "metric_valid_denominator": result["objectives"]["u1_geometry_metric_reconstruction"][
                "u1_geometry_metric_reconstruction/denominator"
            ],
            "grouping_method": config.grouping_method,
        },
    )
    run.write_artifact("reports", "u1_geometry_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact(
        "reports",
        "checkpoint_index.json",
        json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2),
    )
    return result
