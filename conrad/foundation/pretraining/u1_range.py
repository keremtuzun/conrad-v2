"""OS-FM U1 metric range/depth smoke training path."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from conrad.foundation.encoders.range import RangeEncoderConfig, RangeEncoderOutput, RangeViTP8Encoder
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import RepresentationHealth, contiguous_2d_mask, parameter_count
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory


@dataclass(frozen=True)
class U1RangeViews:
    teacher_range: torch.Tensor
    teacher_validity_mask: torch.Tensor
    student_range: torch.Tensor
    student_validity_mask: torch.Tensor
    metric_targets: torch.Tensor
    metric_validity_mask: torch.Tensor
    token_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    corruption_trace: tuple[str, ...]


@dataclass(frozen=True)
class U1RangeStepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: RangeEncoderOutput
    teacher: RangeEncoderOutput
    metric_prediction: torch.Tensor
    degradation_logits: torch.Tensor
    health: RepresentationHealth


def synthetic_range_views(job: dict[str, Any], seed: int, *, all_invalid: bool = False) -> U1RangeViews:
    cfg = RangeEncoderConfig(image_size=int(job.get("image_size", 32)))
    batch = int(job.get("batch_size", 2))
    gen = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, cfg.image_size, dtype=torch.float32).reshape(1, 1, cfg.image_size, 1)
    xx = torch.linspace(0, 1, cfg.image_size, dtype=torch.float32).reshape(1, 1, 1, cfg.image_size)
    sloped_plane = 1.25 + 4.0 * yy + 0.65 * torch.sin(xx * torch.pi * 2.0)
    near_structure = 0.35 * torch.exp(-((xx - 0.68) ** 2 + (yy - 0.42) ** 2) / 0.035)
    teacher = (sloped_plane - near_structure).repeat(batch, 1, 1, 1)
    offsets = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1) * 0.18
    teacher = teacher + offsets
    valid = torch.ones_like(teacher, dtype=torch.bool)
    valid[:, :, : cfg.patch_size, : cfg.patch_size] = False
    valid[:, :, cfg.image_size // 2 :, cfg.image_size - cfg.patch_size :] = False
    if all_invalid:
        valid = torch.zeros_like(valid)
    teacher_recorded = teacher.masked_fill(~valid, 0.0)
    noise = torch.randn(teacher.shape, generator=gen, dtype=teacher.dtype) * 0.045
    dropout = torch.rand(teacher.shape, generator=gen, dtype=teacher.dtype) < 0.045
    quantized = torch.round((teacher + noise).clamp(0, 8) / 0.025) * 0.025
    saturated = quantized.clamp(max=5.25)
    student_valid = valid & ~dropout
    student = saturated.masked_fill(~student_valid, 0.0)
    token_mask = contiguous_2d_mask(
        batch_size=batch,
        grid_size=cfg.grid_size,
        mask_fraction=float(job.get("token_mask_fraction", 0.50)),
        min_visible_fraction=float(job.get("min_visible_fraction", 0.10)),
        generator=gen,
    )
    return U1RangeViews(
        teacher_range=teacher_recorded,
        teacher_validity_mask=valid,
        student_range=student,
        student_validity_mask=student_valid,
        metric_targets=teacher_recorded,
        metric_validity_mask=valid,
        token_mask=token_mask,
        sample_ids=tuple(f"synthetic-metric-range-{idx}" for idx in range(batch)),
        corruption_trace=(
            "bounded_metric_noise",
            "validity_mask_aware_dropout",
            "quantization",
            "saturation",
        ),
    )


class U1RangeTrainingBundle(nn.Module):
    def __init__(self, encoder: RangeViTP8Encoder, *, ema_schedule: EMASchedule | None = None) -> None:
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
        self.metric_head = nn.Linear(
            encoder.config.output_dim,
            encoder.config.patch_size * encoder.config.patch_size,
        )
        self.router = ObjectiveRouter(
            {
                "u1_range_masked_latent_prediction": 1.0,
                "u1_range_global_consistency": 0.5,
                "u1_range_degradation": 0.25,
                "u1_range_metric_reconstruction": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)

    def _metric_objective(
        self,
        views: U1RangeViews,
        student: RangeEncoderOutput,
    ) -> tuple[ObjectiveResult, torch.Tensor]:
        cfg = self.student.config
        pred = self.metric_head(student.patch_tokens)
        target = views.metric_targets.unfold(2, cfg.patch_size, cfg.patch_size).unfold(
            3,
            cfg.patch_size,
            cfg.patch_size,
        )
        target = target.contiguous().view(
            target.shape[0],
            target.shape[1],
            -1,
            cfg.patch_size * cfg.patch_size,
        )
        target = target.squeeze(1)
        valid = views.metric_validity_mask.unfold(2, cfg.patch_size, cfg.patch_size).unfold(
            3,
            cfg.patch_size,
            cfg.patch_size,
        )
        valid = valid.contiguous().view(
            valid.shape[0],
            valid.shape[1],
            -1,
            cfg.patch_size * cfg.patch_size,
        )
        valid = valid.squeeze(1)
        eligible = valid & student.masked_token_mask.unsqueeze(-1)
        numerator = (F.mse_loss(pred, target.detach(), reduction="none") * eligible.float()).sum()
        denominator = eligible.float().sum()
        return self.router._result("u1_range_metric_reconstruction", numerator, denominator), pred

    def forward(self, views: U1RangeViews) -> U1RangeStepOutput:
        student = self.student(views.student_range, views.student_validity_mask, views.token_mask)
        with torch.no_grad():
            teacher = self.teacher(
                views.teacher_range,
                views.teacher_validity_mask,
                torch.zeros_like(views.token_mask),
            )
        masked_pred = self.mask_predictor(student.patch_tokens)
        valid_masked_tokens = views.token_mask & (teacher.patch_valid_fraction > 0)
        mask_loss = self.router._result(
            "u1_range_masked_latent_prediction",
            (
                F.mse_loss(masked_pred, teacher.patch_tokens.detach(), reduction="none").mean(dim=-1)
                * valid_masked_tokens
            ).sum(),
            valid_masked_tokens.float().sum(),
        )
        global_loss = self.router._result(
            "u1_range_global_consistency",
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
            "u1_range_degradation",
            F.cross_entropy(logits, labels, reduction="sum"),
            torch.tensor(float(logits.shape[0]), device=logits.device),
        )
        metric_loss, metric_prediction = self._metric_objective(views, student)
        results = (mask_loss, global_loss, degradation_loss, metric_loss)
        return U1RangeStepOutput(
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


def run_u1_range_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = RangeEncoderConfig(image_size=int(job.get("image_size", 32)))
    views = synthetic_range_views(job, seed)
    bundle = U1RangeTrainingBundle(RangeViTP8Encoder(config))
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
            "token_mask": views.token_mask.to(torch.int).tolist(),
            "metric_validity_mask": views.metric_validity_mask.to(torch.int).tolist(),
            "seed": seed,
        }
    )
    manifest_digest = hashlib.sha256(b"SYNTHETIC:osfm-u1-metric-range-smoke:v1").hexdigest()
    valid_pixels = views.metric_validity_mask.float().sum()
    total_pixels = torch.tensor(float(views.metric_validity_mask.numel()))
    metrics: dict[str, float | str] = {
        "val/osfm_u1_range_loss": float(post.loss.detach().cpu()),
        "optimizer_steps": 1.0,
        "ema_updates": 1.0,
        "ema_momentum": float(ema),
        "mask_fraction_requested": float(job.get("token_mask_fraction", 0.50)),
        "mask_fraction_actual": float(views.token_mask.float().mean().detach().cpu()),
        "visible_fraction_actual": float((~views.token_mask).float().mean().detach().cpu()),
        "validity_coverage": float((valid_pixels / total_pixels).detach().cpu()),
        "metric_valid_pixel_count": float(valid_pixels.detach().cpu()),
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
    ckpt = run.path / "checkpoints" / "osfm_u1_range_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.u1_range",
        config=job,
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        seeds={"torch": seed, "mask": seed},
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={"cuda_rng_state": [], "amp_scaler": None, "sample_ids": views.sample_ids},
        is_encoder=True,
        representation_pretraining_id="OSFM-U1-RANGE-SMOKE-001",
        selection_metric="val/osfm_u1_range_loss",
        extra_compatibility={"osfm_stage": "U1-RANGE", "p4_smoke": "true"},
    )
    fresh = U1RangeTrainingBundle(RangeViTP8Encoder(config))
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.u1_range",
            representation_pretraining_id="OSFM-U1-RANGE-SMOKE-001",
            extra={"osfm_stage": "U1-RANGE", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(views)
    replay_views = synthetic_range_views(job, seed)
    reload_max_abs_diff = float(
        (post.student.modality_repr - reloaded.student.modality_repr).abs().max().detach().cpu()
    )
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.u1_range",
        "experiment_id": "OSFM-U1-RANGE-SMOKE-001",
        "architecture": {
            "encoder": "RangeViTP8Encoder",
            "input_semantics": "metric range/depth-like raster with explicit validity mask",
            "patch_size": config.patch_size,
            "internal_width": config.width,
            "output_dim": config.output_dim,
            "d_f": config.output_dim,
            "layers": config.depth,
            "heads": config.heads,
            "dense_tap_layers": list(config.dense_tap_layers),
            "dense_tap_contract": (
                "internal taps remain width 256; projected dense taps are exposed at D_F=384"
            ),
            "image_size": config.image_size,
            "parameter_count": parameter_count(bundle.student),
        },
        "data": {
            "source": "synthetic_metric_range_depth_engineering_smoke",
            "public_real_dataset_used": False,
            "not_domain_pretraining_evidence": True,
            "metric_target_present": True,
            "metric_objective_eligible": post.results[-1].status.value == "ACTIVE",
            "limitation": (
                "Synthetic metric fixture with known valid/invalid regions; "
                "no public-real range evidence is claimed."
            ),
        },
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "views": {
            "teacher": (
                "minimally transformed recorded metric range/depth observation "
                "with explicit invalid regions"
            ),
            "student_corruptions": list(views.corruption_trace),
        },
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {
            "sample_ids": list(views.sample_ids),
            "token_mask_sha256": hashlib.sha256(
                json.dumps(views.token_mask.to(torch.int).tolist()).encode()
            ).hexdigest(),
            "validity_mask_sha256": hashlib.sha256(
                json.dumps(views.metric_validity_mask.to(torch.int).tolist()).encode()
            ).hexdigest(),
            "deterministic": torch.equal(views.token_mask, replay_views.token_mask)
            and torch.equal(views.teacher_range, replay_views.teacher_range)
            and torch.equal(views.student_range, replay_views.student_range)
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
        "OSFM_U1_RANGE_SMOKE_STEP",
        time.time_ns(),
        {
            "sample_ids": views.sample_ids,
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": post.health.formal_rank_guard,
            "student_corruptions": views.corruption_trace,
            "metric_valid_denominator": result["objectives"]["u1_range_metric_reconstruction"][
                "u1_range_metric_reconstruction/denominator"
            ],
        },
    )
    run.write_artifact("reports", "u1_range_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact(
        "reports",
        "checkpoint_index.json",
        json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2),
    )
    return result
