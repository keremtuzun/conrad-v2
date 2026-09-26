"""OS-FM U1 rendered-sonar smoke training path."""

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

from conrad.foundation.encoders.sonar import SonarEncoderConfig, SonarEncoderOutput, SonarViTS14Encoder
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import RepresentationHealth, contiguous_2d_mask, parameter_count
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory


@dataclass(frozen=True)
class U1SonarViews:
    teacher_sonar: torch.Tensor
    student_sonar: torch.Tensor
    token_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    corruption_trace: tuple[str, ...]
    metric_targets: torch.Tensor | None = None


@dataclass(frozen=True)
class U1SonarStepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: SonarEncoderOutput
    teacher: SonarEncoderOutput
    degradation_logits: torch.Tensor
    health: RepresentationHealth


def synthetic_sonar_views(job: dict[str, Any], seed: int) -> U1SonarViews:
    cfg = SonarEncoderConfig(image_size=int(job.get("image_size", 28)))
    batch = int(job.get("batch_size", 2))
    gen = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, cfg.image_size, dtype=torch.float32).reshape(1, 1, cfg.image_size, 1)
    xx = torch.linspace(0, 1, cfg.image_size, dtype=torch.float32).reshape(1, 1, 1, cfg.image_size)
    fan = torch.exp(-2.5 * yy) * (0.35 + 0.65 * torch.sin((xx * 5.0 + yy * 1.5) * torch.pi).abs())
    range_bands = 0.12 * torch.cos((yy * 9.0) * torch.pi).abs()
    base = (fan + range_bands).repeat(batch, 1, 1, 1)
    offsets = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1) * 0.035
    teacher = (base + offsets).clamp(0, 1)

    speckle = torch.randn(teacher.shape, generator=gen, dtype=teacher.dtype) * 0.035
    gain = 0.90 + torch.rand((batch, 1, 1, 1), generator=gen, dtype=teacher.dtype) * 0.16
    attenuation = torch.linspace(1.0, 0.82, cfg.image_size, dtype=teacher.dtype).reshape(1, 1, cfg.image_size, 1)
    dropout = torch.rand(teacher.shape, generator=gen, dtype=teacher.dtype) < 0.035
    student = (teacher * gain * attenuation + speckle).masked_fill(dropout, 0.0).clamp(0, 1)
    mask = contiguous_2d_mask(
        batch_size=batch,
        grid_size=cfg.grid_size,
        mask_fraction=float(job.get("token_mask_fraction", 0.60)),
        min_visible_fraction=float(job.get("min_visible_fraction", 0.10)),
        generator=gen,
    )
    return U1SonarViews(
        teacher_sonar=teacher,
        student_sonar=student,
        token_mask=mask,
        sample_ids=tuple(f"synthetic-rendered-sonar-{idx}" for idx in range(batch)),
        corruption_trace=("bounded_speckle_like_noise", "gain_variation", "attenuation_like_rolloff", "dropout"),
    )


class U1SonarTrainingBundle(nn.Module):
    def __init__(self, encoder: SonarViTS14Encoder, *, ema_schedule: EMASchedule | None = None) -> None:
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
                "u1_sonar_masked_latent_prediction": 1.0,
                "u1_sonar_global_consistency": 0.5,
                "u1_sonar_degradation": 0.25,
                "u1_sonar_metric": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)

    def forward(self, views: U1SonarViews) -> U1SonarStepOutput:
        student = self.student(views.student_sonar, views.token_mask)
        with torch.no_grad():
            teacher = self.teacher(views.teacher_sonar, torch.zeros_like(views.token_mask))
        masked_pred = self.mask_predictor(student.patch_tokens)
        mask_loss = self.router._result(
            "u1_sonar_masked_latent_prediction",
            (F.mse_loss(masked_pred, teacher.patch_tokens.detach(), reduction="none").mean(dim=-1) * views.token_mask).sum(),
            views.token_mask.float().sum(),
        )
        global_loss = self.router._result(
            "u1_sonar_global_consistency",
            F.mse_loss(self.global_projector(student.modality_repr), teacher.modality_repr.detach(), reduction="sum")
            / student.modality_repr.shape[-1],
            torch.tensor(float(student.modality_repr.shape[0]), device=student.modality_repr.device),
        )
        logits = self.degradation_head(student.modality_repr)
        labels = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device)
        degradation_loss = self.router._result(
            "u1_sonar_degradation",
            F.cross_entropy(logits, labels, reduction="sum"),
            torch.tensor(float(logits.shape[0]), device=logits.device),
        )
        metric_denominator = torch.tensor(0.0, device=student.modality_repr.device)
        metric_numerator = student.modality_repr.sum() * 0.0
        if views.metric_targets is not None:
            metric_denominator = torch.tensor(float(views.metric_targets.shape[0]), device=student.modality_repr.device)
            metric_numerator = F.mse_loss(student.modality_repr[:, : views.metric_targets.shape[1]], views.metric_targets, reduction="sum")
        metric_loss = self.router._result("u1_sonar_metric", metric_numerator, metric_denominator)
        results = (mask_loss, global_loss, degradation_loss, metric_loss)
        return U1SonarStepOutput(
            loss=self.router.total(results),
            results=results,
            student=student,
            teacher=teacher,
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


def run_u1_sonar_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = SonarEncoderConfig(image_size=int(job.get("image_size", 28)))
    views = synthetic_sonar_views(job, seed)
    bundle = U1SonarTrainingBundle(SonarViTS14Encoder(config))
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
    manifest_digest = hashlib.sha256(b"SYNTHETIC:osfm-u1-rendered-sonar-smoke:v1").hexdigest()
    metrics: dict[str, float | str] = {
        "val/osfm_u1_sonar_loss": float(post.loss.detach().cpu()),
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
    }
    metrics.update({k: v for k, v in _objective_metrics(post.results).items() if isinstance(v, float)})
    ckpt = run.path / "checkpoints" / "osfm_u1_sonar_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.u1_sonar",
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
        representation_pretraining_id="OSFM-U1-SONAR-SMOKE-001",
        selection_metric="val/osfm_u1_sonar_loss",
        extra_compatibility={"osfm_stage": "U1-SONAR", "p4_smoke": "true"},
    )
    fresh = U1SonarTrainingBundle(SonarViTS14Encoder(config))
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.u1_sonar",
            representation_pretraining_id="OSFM-U1-SONAR-SMOKE-001",
            extra={"osfm_stage": "U1-SONAR", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(views)
    replay_views = synthetic_sonar_views(job, seed)
    reload_max_abs_diff = float((post.student.modality_repr - reloaded.student.modality_repr).abs().max().detach().cpu())
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.u1_sonar",
        "experiment_id": "OSFM-U1-SONAR-SMOKE-001",
        "architecture": {
            "encoder": "SonarViTS14Encoder",
            "input_semantics": "single-channel rasterized/rendered side-scan sonar imagery",
            "patch_size": config.patch_size,
            "width": config.width,
            "d_f": config.width,
            "layers": config.depth,
            "heads": config.heads,
            "dense_tap_layers": list(config.dense_tap_layers),
            "image_size": config.image_size,
            "parameter_count": parameter_count(bundle.student),
        },
        "initialization": {
            "used_rgb_parent_checkpoint": False,
            "rgb_parent_checkpoint_digest": None,
            "patch_projection_transform": None,
            "method": "deterministic_random_smoke",
            "note": "No robust local RGB parent-checkpoint loading path was used; no pretrained or incompatible tensors were copied.",
        },
        "data": {
            "source": "synthetic_rendered_sonar_engineering_smoke",
            "subpipe_used": False,
            "not_domain_pretraining_evidence": True,
            "limitation": "SubPipe-style sonar is rendered side-scan imagery, not raw acoustic backscatter; this smoke does not claim raw-acoustic modeling.",
        },
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "views": {"teacher": "minimally transformed recorded/synthetic sonar image", "student_corruptions": list(views.corruption_trace)},
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {
            "sample_ids": list(views.sample_ids),
            "token_mask_sha256": hashlib.sha256(json.dumps(views.token_mask.to(torch.int).tolist()).encode()).hexdigest(),
            "deterministic": torch.equal(views.token_mask, replay_views.token_mask)
            and torch.equal(views.teacher_sonar, replay_views.teacher_sonar)
            and torch.equal(views.student_sonar, replay_views.student_sonar),
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
        "OSFM_U1_SONAR_SMOKE_STEP",
        time.time_ns(),
        {
            "sample_ids": views.sample_ids,
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": post.health.formal_rank_guard,
            "student_corruptions": views.corruption_trace,
        },
    )
    run.write_artifact("reports", "u1_sonar_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact(
        "reports",
        "checkpoint_index.json",
        json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2),
    )
    return result
