"""OS-FM U1 rendered-sonar smoke training path."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2
import torch
import torch.nn.functional as F
from torch import nn

from conrad.data.adapters.subpipe import SubPipeAdapter
from conrad.data.manifest import data_root_for, load_manifest, verify_loaded_manifest
from conrad.foundation.encoders.sonar import SonarEncoderConfig, SonarEncoderOutput, SonarViTS14Encoder
from conrad.foundation.data.manifest import CorpusPartition
from conrad.foundation.data.osfm_public_real import (
    SUBPIPE_MANIFEST,
    SUBPIPE_SOURCE_ID,
    _partition_ranges,
    load_subpipe_osfm_evidence,
    validate_subpipe_osfm_evidence,
)
from conrad.foundation.pretraining.dinov2 import verify_dinov2_checkpoint
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import RepresentationHealth, contiguous_2d_mask, parameter_count
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.determinism import inspect_compute
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


@dataclass(frozen=True)
class SubPipeSonarPool:
    """Verified, partition-scoped rendered-sonar tensors decoded once per run."""

    partition: CorpusPartition
    stream: str
    images: torch.Tensor
    sample_ids: tuple[str, ...]

    def views(
        self,
        job: dict[str, Any],
        seed: int,
        *,
        train_stats: tuple[float, float],
        sample_offset: int = 0,
        batch_size: int | None = None,
    ) -> U1SonarViews:
        batch = int(batch_size if batch_size is not None else job.get("batch_size", 2))
        if len(self.sample_ids) < batch:
            raise RuntimeError(
                f"{self.partition.value} has only {len(self.sample_ids)} frames for batch {batch}"
            )
        generator = torch.Generator().manual_seed(
            seed + (0 if self.partition is CorpusPartition.PRETRAIN_REAL else 10_000)
        )
        order = torch.randperm(len(self.sample_ids), generator=generator).tolist()
        indices = [order[(sample_offset + index) % len(order)] for index in range(batch)]
        teacher = self.images[indices].clone()
        mean, std = train_stats
        teacher = ((teacher - mean) / std).clamp(-5.0, 5.0)
        speckle = torch.randn(teacher.shape, generator=generator, dtype=teacher.dtype) * float(
            job.get("noise_std", 0.025)
        )
        gain = 0.96 + torch.rand(
            (teacher.shape[0], 1, 1, 1), generator=generator, dtype=teacher.dtype
        ) * 0.08
        student = (teacher * gain + speckle).clamp(-5.0, 5.0)
        mask = contiguous_2d_mask(
            batch_size=batch,
            grid_size=SonarEncoderConfig(image_size=int(job.get("image_size", 28))).grid_size,
            mask_fraction=float(job.get("token_mask_fraction", 0.60)),
            min_visible_fraction=float(job.get("min_visible_fraction", 0.10)),
            generator=generator,
        )
        return U1SonarViews(
            teacher_sonar=teacher,
            student_sonar=student,
            token_mask=mask,
            sample_ids=tuple(self.sample_ids[index] for index in indices),
            corruption_trace=("bounded_speckle_like_noise", "gain_variation"),
        )

    def evidence(self, train_stats: tuple[float, float], sample_ids: tuple[str, ...]) -> dict[str, Any]:
        mean, std = train_stats
        return {
            "partition": self.partition.value,
            "stream": self.stream,
            "sample_ids": list(sample_ids),
            "source_frame_count": len(self.sample_ids),
            "lineage_unit": f"subpipe/mini_sss/{self.stream}",
            "normalization_fit_partitions": [CorpusPartition.PRETRAIN_REAL.value],
            "normalization_excluded_partitions": [
                CorpusPartition.VALIDATION.value,
                CorpusPartition.FINAL_TEST.value,
                CorpusPartition.OOD_TEST.value,
            ],
            "rendered_sonar_limitation": (
                "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter."
            ),
            "train_stats": {"mean": mean, "std": std},
        }


def _repo_tracked_clean() -> tuple[bool, str]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"git status unavailable: {exc}"
    return not bool(status), status


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object")
    return data


def _configure_cuda_determinism() -> dict[str, Any]:
    """Force deterministic scaled-dot-product attention backends for formal CUDA runs."""
    settings: dict[str, Any] = {"applied": False}
    cuda_backends = getattr(torch.backends, "cuda", None)
    if not torch.cuda.is_available() or cuda_backends is None:
        return settings

    for name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        fn = getattr(cuda_backends, name, None)
        if callable(fn):
            fn(enabled)
            settings[name] = enabled
    settings["applied"] = True
    return settings


def _require_u1_sonar_research_gate(job: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before any P4.8 real-data run starts."""
    if job.get("formal_run_allowed") is not True:
        raise RuntimeError("P4.8 U1 sonar research config is not formally allowed")
    clean, detail = _repo_tracked_clean()
    if bool(job.get("require_clean_git", True)) and not clean:
        raise RuntimeError(f"P4.8 requires clean tracked git state: {detail}")
    compute = inspect_compute()
    min_vram = int(job.get("min_vram_gb", 16))
    if not compute.cuda_available:
        raise RuntimeError("P4.8 requires CUDA; CPU/MPS execution is refused")
    measured_vram = [
        int(torch.cuda.get_device_properties(i).total_memory // (1024**3))
        for i in range(torch.cuda.device_count())
    ]
    if not measured_vram or max(measured_vram) < min_vram:
        raise RuntimeError(f"P4.8 requires >= {min_vram} GB CUDA VRAM; measured {measured_vram}")

    p47 = _load_json(job.get("p47_go_artifact", "artifacts/gates/P4.7D_L4/osfm_readiness.json"))
    if p47.get("status") != "VALIDATED-RUN" or p47.get("decision") != "GO" or p47.get("blockers"):
        raise RuntimeError("P4.8 requires a blocker-free P4.7 GO readiness artifact")

    manifest = load_manifest(SUBPIPE_MANIFEST)
    manifest_problems = verify_loaded_manifest(manifest, data_root_for(manifest.dataset_id))
    if manifest_problems:
        raise RuntimeError(f"SubPipe manifest verification failed: {[str(p) for p in manifest_problems]}")
    evidence = load_subpipe_osfm_evidence(job.get("evidence", "configs/data/osfm/subpipe_p47b_evidence.json"))
    evidence_problems = validate_subpipe_osfm_evidence(evidence)
    if evidence_problems:
        raise RuntimeError(f"SubPipe OSFM evidence verification failed: {evidence_problems}")

    init = Path(job.get("init", "artifacts/external/dinov2/dinov2_vits14.pth"))
    if not init.is_absolute():
        init = REPO_ROOT / init
    init_verification = verify_dinov2_checkpoint(init)
    return {
        "p47_go_artifact": job.get("p47_go_artifact", "artifacts/gates/P4.7D_L4/osfm_readiness.json"),
        "p47_report_digest": p47.get("report_digest"),
        "compute": compute.model_dump(mode="json"),
        "measured_cuda_vram_gb": measured_vram,
        "subpipe_source_id": SUBPIPE_SOURCE_ID,
        "subpipe_evidence_digest": evidence.get("evidence_digest"),
        "dinov2": init_verification,
        "clean_tracked_git": clean,
    }


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


def load_subpipe_sonar_pool(job: dict[str, Any], *, partition: CorpusPartition) -> SubPipeSonarPool:
    """Decode one lineage-safe partition once; no final or OOD access is permitted."""
    if partition not in (CorpusPartition.PRETRAIN_REAL, CorpusPartition.VALIDATION):
        raise ValueError("P4.8 U1 sonar may read only PRETRAIN_REAL or VALIDATION")
    image_size = int(job.get("image_size", 28))
    stream = str(job.get("stream", "sss_lf"))
    if stream not in {"sss_lf", "sss_hf"}:
        raise ValueError("P4.8 U1 sonar stream must be sss_lf or sss_hf")
    manifest = load_manifest(SUBPIPE_MANIFEST)
    images: list[torch.Tensor] = []
    sample_ids: list[str] = []
    with tempfile.TemporaryDirectory(prefix="conrad-p48-u1-sonar-") as tmp:
        adapter = SubPipeAdapter(
            manifest,
            data_root_for(manifest.dataset_id),
            ObjectStore(Path(tmp) / "objects"),
            IdFactory(seed=2026092602),
            UUID("00000000-0000-7000-8000-00000000048a"),
            streams=(stream,),
            manifest_path=SUBPIPE_MANIFEST,
        )
        try:
            refs = adapter.frames(stream)
            ranges = _partition_ranges(len(refs))
            selected = refs[slice(*ranges[partition])]
            for ref in selected:
                obs = adapter.normalize_observation(ref)
                if obs.sensor_context.get("sonar_rendering") != "colormapped PPM, not raw backscatter":
                    raise RuntimeError("SubPipe rendered-sonar limitation missing from observation metadata")
                raw = adapter.load(ref)
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
                resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
                images.append(torch.from_numpy(resized).float().unsqueeze(0) / 255.0)
                sample_ids.append(f"{partition.value}:{stream}:{ref.member}")
        finally:
            adapter.close()
    if not images:
        raise RuntimeError(f"{partition.value} contains no usable rendered-sonar frames")
    return SubPipeSonarPool(
        partition=partition,
        stream=stream,
        images=torch.stack(images, dim=0),
        sample_ids=tuple(sample_ids),
    )


def real_subpipe_sonar_views(
    job: dict[str, Any],
    seed: int,
    *,
    partition: CorpusPartition,
    train_stats: tuple[float, float] | None = None,
    sample_offset: int = 0,
) -> tuple[U1SonarViews, dict[str, Any], tuple[float, float]]:
    pool = load_subpipe_sonar_pool(job, partition=partition)
    if train_stats is None:
        mean = float(pool.images.mean())
        std = float(pool.images.std(unbiased=False).clamp_min(1e-6))
    else:
        mean, std = train_stats
    stats = (mean, std)
    views = pool.views(job, seed, train_stats=stats, sample_offset=sample_offset)
    return views, pool.evidence(stats, views.sample_ids), stats


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


def run_u1_sonar_research(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    gate = _require_u1_sonar_research_gate(job)
    rehearsal_only = bool(job.get("rehearsal_only", True))
    if not rehearsal_only and job.get("full_run_allowed") is not True:
        raise RuntimeError("Full P4.8 U1 sonar run requires full_run_allowed: true")
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    deterministic_backend = _configure_cuda_determinism()
    torch.use_deterministic_algorithms(True, warn_only=bool(job.get("determinism_warn_only", rehearsal_only)))
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    config = SonarEncoderConfig(image_size=int(job.get("image_size", 28)))
    train_pool = load_subpipe_sonar_pool(job, partition=CorpusPartition.PRETRAIN_REAL)
    val_pool = load_subpipe_sonar_pool(job, partition=CorpusPartition.VALIDATION)
    train_stats = (
        float(train_pool.images.mean()),
        float(train_pool.images.std(unbiased=False).clamp_min(1e-6)),
    )
    train_views = train_pool.views(job, seed, train_stats=train_stats)
    validation_support = min(int(job.get("validation_support", 96)), len(val_pool.sample_ids))
    if validation_support < 64:
        raise RuntimeError(
            f"formal P4.8 validation requires >=64 independent samples; found {validation_support}"
        )
    val_views = val_pool.views(job, seed + 1, train_stats=train_stats, batch_size=validation_support)
    train_evidence = train_pool.evidence(train_stats, train_views.sample_ids)
    val_evidence = val_pool.evidence(train_stats, val_views.sample_ids)
    bundle = U1SonarTrainingBundle(
        SonarViTS14Encoder(config),
        ema_schedule=EMASchedule(start=0.996, end=0.9999, total_steps=int(job.get("optimizer_steps", 2))),
    ).to(device)
    train_views = U1SonarViews(
        train_views.teacher_sonar.to(device),
        train_views.student_sonar.to(device),
        train_views.token_mask.to(device),
        train_views.sample_ids,
        train_views.corruption_trace,
    )
    val_views = U1SonarViews(
        val_views.teacher_sonar.to(device),
        val_views.student_sonar.to(device),
        val_views.token_mask.to(device),
        val_views.sample_ids,
        val_views.corruption_trace,
    )
    optimizer = torch.optim.AdamW(
        bundle.parameters(),
        lr=float(job.get("lr", 1e-4)),
        weight_decay=float(job.get("weight_decay", 0.05)),
    )
    steps = int(job.get("optimizer_steps", 2))
    validation_every = int(job.get("validation_every_steps", max(1, steps)))
    last = None
    val = None
    all_train_sample_ids: list[str] = []
    for step in range(1, steps + 1):
        if not rehearsal_only and step > 1:
            train_views = train_pool.views(
                job,
                seed + step,
                train_stats=train_stats,
                sample_offset=(step - 1) * int(job.get("batch_size", 2)),
            )
            step_evidence = train_pool.evidence(train_stats, train_views.sample_ids)
            train_evidence = {
                **train_evidence,
                "sample_ids": list(dict.fromkeys([*train_evidence["sample_ids"], *step_evidence["sample_ids"]])),
            }
            train_views = U1SonarViews(
                train_views.teacher_sonar.to(device),
                train_views.student_sonar.to(device),
                train_views.token_mask.to(device),
                train_views.sample_ids,
                train_views.corruption_trace,
            )
        bundle.train()
        out = bundle(train_views)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.parameters(), float(job.get("grad_clip_norm", 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema = bundle.update_teacher_after_optimizer(step)
        last = out
        all_train_sample_ids.extend(train_views.sample_ids)
        run.log_metrics(
            {
                "step": step,
                "train/osfm_u1_sonar_loss": float(out.loss.detach().cpu()),
                "ema_momentum": float(ema),
            }
        )
        if not rehearsal_only and step % validation_every == 0:
            bundle.eval()
            with torch.no_grad():
                val = bundle(val_views)
            run.log_metrics({"step": step, "val/osfm_u1_sonar_loss": float(val.loss.detach().cpu())})
    bundle.eval()
    with torch.no_grad():
        val = bundle(val_views) if val is None else val
    split_plan = {
        "train_sample_ids": tuple(dict.fromkeys(all_train_sample_ids or train_views.sample_ids)),
        "validation_sample_ids": val_views.sample_ids,
        "lineage": train_evidence["lineage_unit"],
        "partition_policy": "PRETRAIN_REAL trains; VALIDATION selects; FINAL_TEST/OOD_TEST forbidden",
    }
    split_hash = split_hash_for_plan(split_plan)
    manifest_digest = load_manifest(SUBPIPE_MANIFEST).manifest_digest()
    metrics: dict[str, float | str] = {
        "val/osfm_u1_sonar_loss": float(val.loss.detach().cpu()),
        "train/osfm_u1_sonar_loss": float(last.loss.detach().cpu()) if last is not None else float("nan"),
        "optimizer_steps": float(steps),
        "representation_finite": float(val.health.finite),
        "representation_variance_mean": val.health.variance_mean,
        "representation_collapse_score": val.health.collapse_score,
        "representation_effective_rank": val.health.effective_rank,
        "formal_rank_guard_numeric": {"PASS": 1.0, "FAIL": 0.0, "NOT_EVALUABLE": -1.0}[val.health.formal_rank_guard],
        "parameter_count": float(parameter_count(bundle.student)),
        "memory_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
    }
    metrics.update({k: v for k, v in _objective_metrics(val.results).items() if isinstance(v, float)})
    ckpt_name = "osfm_u1_sonar_research_rehearsal.pt" if rehearsal_only else "osfm_u1_sonar_research_full.pt"
    ckpt = run.path / "checkpoints" / ckpt_name
    representation_id = "P4.8A-U1-SONAR-REAL-REHEARSAL" if rehearsal_only else "P4.8-U1-SONAR-RESEARCH"
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
        step=steps,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={
            "cuda_rng_state": [torch.cuda.get_rng_state(device).cpu()],
            "amp_scaler": None,
            "split_plan": split_plan,
        },
        is_encoder=True,
        representation_pretraining_id=representation_id,
        selection_metric="val/osfm_u1_sonar_loss",
        extra_compatibility={"osfm_stage": "U1-SONAR-RESEARCH", "p48_rehearsal": str(rehearsal_only).lower()},
    )
    fresh = U1SonarTrainingBundle(SonarViTS14Encoder(config)).to(device)
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.u1_sonar",
            representation_pretraining_id=representation_id,
            extra={"osfm_stage": "U1-SONAR-RESEARCH", "p48_rehearsal": str(rehearsal_only).lower()},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(val_views)
    reload_max_abs_diff = float((val.student.modality_repr - reloaded.student.modality_repr).abs().max().detach().cpu())
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.u1_sonar",
        "experiment_id": "P4.8A-U1-SONAR-REAL-REHEARSAL" if rehearsal_only else "P4.8-U1-SONAR-RESEARCH",
        "formal_p4_8": not rehearsal_only,
        "rehearsal_only": rehearsal_only,
        "gate": gate,
        "data": {
            "source": SUBPIPE_SOURCE_ID,
            "subpipe_used": True,
            "train": train_evidence,
            "validation": val_evidence,
            "final_test_used": False,
            "ood_test_used": False,
            "limitation": "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter.",
        },
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in val.results},
        "representation_health": val.health.__dict__,
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "compute": {
            "device": str(device),
            "deterministic_backend": deterministic_backend,
            "peak_memory_bytes": int(metrics["memory_allocated_bytes"]),
            "wall_clock_s": wall_s,
            "throughput_samples_per_s": (len(train_views.sample_ids) * steps) / wall_s,
        },
        "next_gate": "P4.8 full run evidence may promote U1 sonar only after checkpoint/reload and held-out validation review."
        if not rehearsal_only
        else "Full P4.8 remains locked until rehearsal evidence is committed and full_run_allowed is set explicitly.",
    }
    run.log_event(
        "P48A_U1_SONAR_REAL_REHEARSAL" if rehearsal_only else "P48_U1_SONAR_RESEARCH",
        time.time_ns(),
        {
            "train_sample_ids": train_views.sample_ids,
            "validation_sample_ids": val_views.sample_ids,
            "objective_status": {r.objective_id: r.status.value for r in val.results},
            "rendered_sonar_limitation": result["data"]["limitation"],
        },
    )
    report_name = "p48a_u1_sonar_rehearsal_report.json" if rehearsal_only else "p48_u1_sonar_research_report.json"
    run.write_artifact("reports", report_name, json.dumps(result, indent=2, default=str))
    run.write_artifact(
        "reports",
        "checkpoint_index.json",
        json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2),
    )
    return result
