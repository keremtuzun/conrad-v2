"""OS-FM U1 RGB smoke training path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from conrad.foundation.encoders.rgb import RGBEncoderConfig, RGBEncoderOutput, RGBViTS14Encoder
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter, ObjectiveStatus
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory


@dataclass(frozen=True)
class U1RGBViews:
    teacher_rgb: torch.Tensor
    student_rgb: torch.Tensor
    token_mask: torch.Tensor
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class RepresentationHealth:
    finite: bool
    variance_mean: float
    collapse_score: float
    effective_rank: float
    formal_rank_guard: str
    formal_rank_reason: str


@dataclass(frozen=True)
class U1RGBStepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: RGBEncoderOutput
    teacher: RGBEncoderOutput
    degradation_logits: torch.Tensor
    health: RepresentationHealth
    rank_diversity_loss: torch.Tensor
    rank_entropy: torch.Tensor


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def rank_diversity_loss(
    representations: torch.Tensor, *, target: float = 64.0
) -> tuple[torch.Tensor, torch.Tensor]:
    centered = representations.float() - representations.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probs = singular / singular.sum().clamp_min(1e-12)
    entropy_rank = torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())
    target_tensor = torch.tensor(float(target), device=representations.device, dtype=entropy_rank.dtype)
    loss = (target_tensor - entropy_rank).clamp_min(0.0) / target_tensor.clamp_min(1.0)
    return loss.to(representations.dtype), entropy_rank.to(representations.dtype)


def contiguous_2d_mask(
    *,
    batch_size: int,
    grid_size: int,
    mask_fraction: float,
    min_visible_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    total = grid_size * grid_size
    max_masked = total - max(1, math.ceil(total * min_visible_fraction))
    target = min(max_masked, max(1, round(total * mask_fraction)))
    mask = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.bool)
    side = max(1, min(grid_size, math.ceil(math.sqrt(target))))
    for row in range(batch_size):
        y = int(torch.randint(0, grid_size - side + 1, (1,), generator=generator))
        x = int(torch.randint(0, grid_size - side + 1, (1,), generator=generator))
        mask[row, y : y + side, x : x + side] = True
        overflow = int(mask[row].sum()) - target
        if overflow > 0:
            coords = mask[row].flatten().nonzero().flatten()
            mask[row].flatten()[coords[-overflow:]] = False
    return mask.flatten(1)


def synthetic_rgb_views(job: dict[str, Any], seed: int) -> U1RGBViews:
    cfg = RGBEncoderConfig(image_size=int(job.get("image_size", 28)))
    batch = int(job.get("batch_size", 2))
    gen = torch.Generator().manual_seed(seed)
    values = torch.linspace(0, 1, cfg.image_size * cfg.image_size, dtype=torch.float32)
    base = values.reshape(1, 1, cfg.image_size, cfg.image_size).repeat(batch, 3, 1, 1)
    offsets = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1) * 0.05
    teacher = (base + offsets).clamp(0, 1)
    noise = torch.randn(teacher.shape, generator=gen, dtype=teacher.dtype) * 0.03
    student = (teacher * 0.92 + 0.04 + noise).clamp(0, 1)
    mask = contiguous_2d_mask(
        batch_size=batch,
        grid_size=cfg.grid_size,
        mask_fraction=float(job.get("token_mask_fraction", 0.60)),
        min_visible_fraction=float(job.get("min_visible_fraction", 0.10)),
        generator=gen,
    )
    return U1RGBViews(
        teacher_rgb=teacher,
        student_rgb=student,
        token_mask=mask,
        sample_ids=tuple(f"synthetic-rgb-{idx}" for idx in range(batch)),
    )


class U1RGBTrainingBundle(nn.Module):
    def __init__(
        self,
        encoder: RGBViTS14Encoder,
        *,
        ema_schedule: EMASchedule | None = None,
        rank_diversity_weight: float = 0.0,
        rank_diversity_target: float = 64.0,
    ) -> None:
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.mask_predictor = nn.Linear(encoder.config.width, encoder.config.width)
        self.global_projector = nn.Sequential(
            nn.LayerNorm(encoder.config.width), nn.Linear(encoder.config.width, encoder.config.width)
        )
        self.degradation_head = nn.Linear(encoder.config.width, 2)
        self.router = ObjectiveRouter(
            {
                "u1_rgb_masked_latent_prediction": 1.0,
                "u1_rgb_global_consistency": 0.5,
                "u1_rgb_degradation": 0.25,
                "u1_rgb_metric": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)
        self.rank_diversity_weight = float(rank_diversity_weight)
        self.rank_diversity_target = float(rank_diversity_target)

    def forward(self, views: U1RGBViews) -> U1RGBStepOutput:
        student = self.student(views.student_rgb, views.token_mask)
        with torch.no_grad():
            teacher = self.teacher(views.teacher_rgb, torch.zeros_like(views.token_mask))
        masked_pred = self.mask_predictor(student.patch_tokens)
        mask_loss = self.router._result(
            "u1_rgb_masked_latent_prediction",
            (F.mse_loss(masked_pred, teacher.patch_tokens.detach(), reduction="none").mean(dim=-1) * views.token_mask).sum(),
            views.token_mask.float().sum(),
        )
        global_loss = self.router._result(
            "u1_rgb_global_consistency",
            F.mse_loss(self.global_projector(student.modality_repr), teacher.modality_repr.detach(), reduction="sum")
            / student.modality_repr.shape[-1],
            torch.tensor(float(student.modality_repr.shape[0]), device=student.modality_repr.device),
        )
        logits = self.degradation_head(student.modality_repr)
        labels = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device)
        degradation_loss = self.router._result(
            "u1_rgb_degradation",
            F.cross_entropy(logits, labels, reduction="sum"),
            torch.tensor(float(logits.shape[0]), device=logits.device),
        )
        metric_loss = self.router._result(
            "u1_rgb_metric",
            student.modality_repr.sum() * 0.0,
            torch.tensor(0.0, device=student.modality_repr.device),
        )
        results = (mask_loss, global_loss, degradation_loss, metric_loss)
        rank_loss, rank_entropy = rank_diversity_loss(
            student.modality_repr, target=self.rank_diversity_target
        )
        total_loss = self.router.total(results)
        if self.rank_diversity_weight > 0:
            total_loss = total_loss + rank_loss * self.rank_diversity_weight
        return U1RGBStepOutput(
            loss=total_loss,
            results=results,
            student=student,
            teacher=teacher,
            degradation_logits=logits,
            health=representation_health(student.modality_repr),
            rank_diversity_loss=rank_loss,
            rank_entropy=rank_entropy,
        )

    @torch.no_grad()
    def update_teacher_after_optimizer(self, step: int) -> float:
        self.teacher.float()
        momentum = self.ema_schedule.value(step)
        update_ema_teacher(self.student, self.teacher, momentum)
        self.teacher.eval()
        return momentum


def representation_health(repr_tensor: torch.Tensor, *, formal_rank_threshold: int = 64) -> RepresentationHealth:
    finite = bool(torch.isfinite(repr_tensor).all().detach().cpu())
    centered = repr_tensor.float() - repr_tensor.float().mean(dim=0, keepdim=True)
    variance = centered.var(dim=0, unbiased=False)
    collapse = float(variance.mean().detach().cpu())
    sample_support = min(repr_tensor.shape[0], repr_tensor.shape[1])
    if sample_support < formal_rank_threshold:
        rank = float(torch.linalg.matrix_rank(centered).detach().cpu())
        return RepresentationHealth(
            finite=finite,
            variance_mean=float(variance.mean().detach().cpu()),
            collapse_score=collapse,
            effective_rank=rank,
            formal_rank_guard="NOT_EVALUABLE",
            formal_rank_reason=f"sample_support={sample_support} < formal_threshold={formal_rank_threshold}",
        )
    singular = torch.linalg.svdvals(centered)
    probs = singular / singular.sum().clamp_min(1e-12)
    entropy_rank = torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())
    rank_value = float(entropy_rank.detach().cpu())
    return RepresentationHealth(
        finite=finite,
        variance_mean=float(variance.mean().detach().cpu()),
        collapse_score=collapse,
        effective_rank=rank_value,
        formal_rank_guard="PASS" if rank_value >= formal_rank_threshold else "FAIL",
        formal_rank_reason=f"threshold={formal_rank_threshold}",
    )


def _objective_metrics(results: tuple[ObjectiveResult, ...]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for result in results:
        for key, value in result.as_metrics().items():
            metrics[key.replace("/", "_")] = value
    return metrics


def run_u1_rgb_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = RGBEncoderConfig(image_size=int(job.get("image_size", 28)))
    views = synthetic_rgb_views(job, seed)
    bundle = U1RGBTrainingBundle(RGBViTS14Encoder(config))
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
        {"sample_ids": views.sample_ids, "token_mask": views.token_mask.to(torch.int).tolist(), "seed": seed}
    )
    metrics: dict[str, float | str] = {
        "val/osfm_u1_rgb_loss": float(post.loss.detach().cpu()),
        "optimizer_steps": 1.0,
        "ema_updates": 1.0,
        "ema_momentum": float(ema),
        "mask_fraction_actual": float(views.token_mask.float().mean().detach().cpu()),
        "visible_fraction_actual": float((~views.token_mask).float().mean().detach().cpu()),
        "representation_finite": float(post.health.finite),
        "representation_variance_mean": post.health.variance_mean,
        "representation_collapse_score": post.health.collapse_score,
        "representation_effective_rank": post.health.effective_rank,
        "formal_rank_guard_numeric": {"PASS": 1.0, "FAIL": 0.0, "NOT_EVALUABLE": -1.0}[post.health.formal_rank_guard],
        "parameter_count": float(parameter_count(bundle.student)),
        "memory_allocated_bytes": float(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
        "rank_diversity_loss": float(post.rank_diversity_loss.detach().cpu()),
        "rank_entropy": float(post.rank_entropy.detach().cpu()),
    }
    metrics.update({k: v for k, v in _objective_metrics(post.results).items() if isinstance(v, float)})
    ckpt = run.path / "checkpoints" / "osfm_u1_rgb_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.u1_rgb",
        config=job,
        manifest_digests=(hashlib.sha256(b"SYNTHETIC:osfm-u1-rgb-smoke:v1").hexdigest(),),
        split_hash=split_hash,
        seeds={"torch": seed, "mask": seed},
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={"cuda_rng_state": [], "amp_scaler": None, "sample_ids": views.sample_ids},
        is_encoder=True,
        representation_pretraining_id="OSFM-U1-RGB-SMOKE-001",
        selection_metric="val/osfm_u1_rgb_loss",
        extra_compatibility={"osfm_stage": "U1-RGB", "p4_smoke": "true"},
    )
    fresh = U1RGBTrainingBundle(RGBViTS14Encoder(config))
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(hashlib.sha256(b"SYNTHETIC:osfm-u1-rgb-smoke:v1").hexdigest(),),
            split_hash=split_hash,
            component="foundation.osfm.u1_rgb",
            representation_pretraining_id="OSFM-U1-RGB-SMOKE-001",
            extra={"osfm_stage": "U1-RGB", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(views)
    reload_max_abs_diff = float(
        (post.student.modality_repr - reloaded.student.modality_repr).abs().max().detach().cpu()
    )
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.u1_rgb",
        "experiment_id": "OSFM-U1-RGB-SMOKE-001",
        "architecture": {
            "encoder": "RGBViTS14Encoder",
            "patch_size": config.patch_size,
            "width": config.width,
            "layers": config.depth,
            "heads": config.heads,
            "dense_tap_layers": list(config.dense_tap_layers),
            "image_size": config.image_size,
            "parameter_count": parameter_count(bundle.student),
        },
        "pretrained_weights": {
            "used": False,
            "source": config.pretrained_source,
            "note": "No exact rights-cleared DINOv2 checkpoint was available locally; deterministic random initialization used for engineering smoke only.",
        },
        "synthetic_data": True,
        "not_domain_pretraining_evidence": True,
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {
            "sample_ids": list(views.sample_ids),
            "token_mask_sha256": hashlib.sha256(json.dumps(views.token_mask.to(torch.int).tolist()).encode()).hexdigest(),
            "deterministic": True,
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
        "OSFM_U1_RGB_SMOKE_STEP",
        time.time_ns(),
        {
            "sample_ids": views.sample_ids,
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": post.health.formal_rank_guard,
        },
    )
    run.write_artifact("reports", "u1_rgb_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact(
        "reports",
        "checkpoint_index.json",
        json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2),
    )
    return result
