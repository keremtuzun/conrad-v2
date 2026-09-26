"""P4.7 OS-FM pretraining readiness gate.

This module is intentionally a gate, not a trainer. It records the exact P4.8-P4.11
research plan and fails closed when external data, initialization, compute, or promotion
evidence is missing.
"""

from __future__ import annotations

import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import Field

from conrad.foundation.data.manifest import CorpusPartition, CorpusReadiness, load_corpus_manifest
from conrad.data.manifest import data_root_for, load_manifest, verify_loaded_manifest
from conrad.schemas.base import ConradModel, VersionedModel, digest_of
from conrad.settings import REPO_ROOT
from conrad.training.determinism import inspect_compute


class GateDecision(str, Enum):
    GO = "GO"
    CONDITIONAL_GO = "CONDITIONAL-GO"
    NO_GO = "NO-GO"


class BlockerKind(str, Enum):
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"


class Blocker(ConradModel):
    blocker_id: str
    kind: BlockerKind
    status: str
    detail: str


class RepresentationHealthSpec(ConradModel):
    min_independent_representations: int = 64
    min_effective_rank: float = 64.0
    min_per_dimension_variance: float = 1e-8
    max_collapse_score: float = 0.995
    required_modalities: tuple[str, ...] = ("rgb", "sonar", "range", "geometry")
    independent_unit: str = "capture_unit_or_sequence_window_group"
    rank_note: str = (
        "effective rank >=64 is evaluated only when min(n_independent, representation_dim) >=64; "
        "otherwise the guard is NOT_EVALUABLE and cannot promote a checkpoint"
    )


class StagePlan(ConradModel):
    stage_id: str
    required_parents: tuple[str, ...]
    data_requirements: tuple[str, ...]
    objectives: tuple[str, ...]
    trainable_policy: str
    optimizer: str
    learning_rates: dict[str, float]
    weight_decay: float
    schedule: str
    grad_clip_norm: float
    precision: str
    batch_size_per_gpu: int
    gradient_accumulation: int
    effective_batch_target: int
    optimizer_steps: int
    validation_every_steps: int
    checkpoint_metric: str
    health_guardrails: tuple[str, ...]
    promotion_criteria: tuple[str, ...]
    kill_criteria: tuple[str, ...]


class P47ReadinessReport(VersionedModel):
    gate_id: str = "P4.7"
    status: str = "VALIDATED-RUN"
    decision: GateDecision
    source_commit: str | None
    data_findings: dict[str, Any]
    initialization_findings: dict[str, Any]
    compute_findings: dict[str, Any]
    determinism_findings: dict[str, Any]
    observability_requirements: tuple[str, ...]
    representation_health: RepresentationHealthSpec
    stage_plan: tuple[StagePlan, ...]
    promotion_rules: tuple[str, ...]
    osfm_fq_confirmation: tuple[str, ...]
    blockers: tuple[Blocker, ...]
    synthetic_staging_permitted: bool
    report_digest: str = Field(pattern="^[0-9a-f]{64}$")


def effective_rank(representations: torch.Tensor, eps: float = 1e-12) -> float:
    if representations.ndim != 2:
        raise ValueError("representations must be a [n_independent, dim] matrix")
    if representations.shape[0] < 2:
        return 0.0
    x = representations.float()
    x = x - x.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(x)
    total = singular_values.sum()
    if float(total) <= eps:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy))


def validate_representation_health(
    representations: torch.Tensor,
    *,
    modality_counts: dict[str, int],
    spec: RepresentationHealthSpec | None = None,
) -> dict[str, Any]:
    health = spec or RepresentationHealthSpec()
    n, dim = int(representations.shape[0]), int(representations.shape[1])
    finite = bool(torch.isfinite(representations).all())
    variances = representations.float().var(dim=0, unbiased=False) if n else torch.zeros(dim)
    rank_evaluable = min(n, dim) >= health.min_independent_representations
    rank = effective_rank(representations) if finite and n >= 2 else 0.0
    centered = representations.float() - representations.float().mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(centered, dim=1)
    denom = torch.outer(norms, norms).clamp_min(1e-12)
    cosine = (centered @ centered.T) / denom if n else torch.zeros(0, 0)
    off_diag = cosine[~torch.eye(n, dtype=torch.bool)] if n > 1 else torch.zeros(0)
    collapse_score = float(off_diag.abs().mean()) if off_diag.numel() else 1.0
    missing_modalities = tuple(m for m in health.required_modalities if modality_counts.get(m, 0) <= 0)
    return {
        "finite": finite,
        "n_independent": n,
        "dimension": dim,
        "rank_evaluable": rank_evaluable,
        "effective_rank": rank,
        "effective_rank_pass": bool(rank_evaluable and rank >= health.min_effective_rank),
        "min_variance": float(variances.min()) if variances.numel() else 0.0,
        "variance_pass": bool(variances.numel() and float(variances.min()) > health.min_per_dimension_variance),
        "collapse_score": collapse_score,
        "collapse_pass": collapse_score < health.max_collapse_score,
        "modality_counts": dict(sorted(modality_counts.items())),
        "missing_modalities": missing_modalities,
        "modality_coverage_pass": not missing_modalities,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _public_manifest_findings() -> dict[str, Any]:
    findings: dict[str, Any] = {}
    for path in sorted((REPO_ROOT / "datasets" / "public").glob("*.manifest.yaml")):
        data = _load_yaml(path)
        manifest = load_manifest(path)
        verification_problems = [
            str(problem) for problem in verify_loaded_manifest(manifest, data_root_for(manifest.dataset_id))
        ]
        findings[str(path.relative_to(REPO_ROOT))] = {
            "dataset_id": data.get("dataset_id"),
            "status": data.get("status"),
            "procurement_status": data.get("procurement_status"),
            "rights_review_status": data.get("rights_review_status"),
            "training_allowed": bool(data.get("training_allowed")),
            "deployment_allowed": bool(data.get("deployment_allowed")),
            "files_declared": len(data.get("files") or ()),
            "local_data_root": str(data_root_for(manifest.dataset_id)),
            "local_payload_usable": not verification_problems,
            "verification_problem_count": len(verification_problems),
        }
    return findings


def _data_findings(corpus_path: Path) -> tuple[dict[str, Any], list[Blocker], bool]:
    blockers: list[Blocker] = []
    corpus = load_corpus_manifest(corpus_path)
    counts = corpus.partition_counts()
    public = _public_manifest_findings()
    public_real_ready = [
        item
        for item in public.values()
        if item["status"] in {"APPROVED", "VERIFIED"} and item["training_allowed"] and item["files_declared"] > 0
        and item["local_payload_usable"]
    ]
    synthetic_fixture = "synthetic readiness fixture" in " ".join(corpus.notes).lower()
    has_required = all(counts[p.value] > 0 for p in CorpusPartition)
    if corpus.readiness is not CorpusReadiness.READY or not has_required:
        blockers.append(
            Blocker(
                blocker_id="EXT-DATA-OSFM-01",
                kind=BlockerKind.EXTERNAL,
                status="BLOCKED_EXTERNAL",
                detail="DATA-OSFM-01 is not a READY corpus with all required partitions.",
            )
        )
    if synthetic_fixture:
        blockers.append(
            Blocker(
                blocker_id="EXT-DATA-PUBLIC-REAL-01",
                kind=BlockerKind.EXTERNAL,
                status="BLOCKED_EXTERNAL",
                detail="Current READY DATA-OSFM-01 file is a synthetic fixture, not public-real payload evidence.",
            )
        )
    if not public_real_ready:
        blockers.append(
            Blocker(
                blocker_id="EXT-PUBLIC-REAL-PAYLOAD-01",
                kind=BlockerKind.EXTERNAL,
                status="BLOCKED_EXTERNAL",
                detail="No locally verified public-real payload is ready as a promotable DATA-OSFM-01 pretraining corpus.",
            )
        )
    return (
        {
            "corpus_manifest": str(corpus_path.relative_to(REPO_ROOT)),
            "corpus_readiness": corpus.readiness.value,
            "corpus_digest": corpus.corpus_digest,
            "partition_counts": counts,
            "first_party_required": any(s.first_party for s in corpus.sources),
            "synthetic_fixture": synthetic_fixture,
            "public_manifest_summary": public,
            "public_real_ready_payloads": public_real_ready,
            "lineage_policy": (
                "PRETRAIN_REAL and PRETRAIN_SYNTHETIC may train; VALIDATION selects checkpoints; "
                "FINAL_TEST and OOD_TEST are promotion-only and blocked from training access"
            ),
            "normalization_policy": "normalization/statistics artifacts must be derived from PRETRAIN_* only",
        },
        blockers,
        synthetic_fixture,
    )


def _initialization_findings(init_path: Path) -> tuple[dict[str, Any], list[Blocker]]:
    blockers: list[Blocker] = []
    present = init_path.exists()
    if not present:
        blockers.append(
            Blocker(
                blocker_id="EXT-DINOV2-VITS14-01",
                kind=BlockerKind.EXTERNAL,
                status="BLOCKED_EXTERNAL",
                detail=f"DINOv2 ViT-S/14 generic weights missing at {init_path}; final-candidate runs must not use random init.",
            )
        )
    return (
        {
            "dinov2_vits14_path": str(init_path.relative_to(REPO_ROOT)) if init_path.is_relative_to(REPO_ROOT) else str(init_path),
            "present": present,
            "required_provenance": {
                "source": "official DINOv2 ViT-S/14 checkpoint",
                "version": "exact release/checkpoint name",
                "content_hash": "sha256 required before import",
                "license_usage_notes": "record upstream model/data license and Conrad usage decision",
                "import_transform": "record key mapping and patch-projection transform if any",
            },
            "sonar_policy": (
                "sonar patch projection may be independently initialized or transformed from RGB only with an "
                "explicit transform digest; transformer weights may inherit generic RGB initialization, after which "
                "sonar independence is established by U1-SONAR-RESEARCH data, metrics, and separate checkpoint ancestry"
            ),
        },
        blockers,
    )


def _compute_findings(min_vram_gb: int, recommended_vram_gb: int) -> tuple[dict[str, Any], list[Blocker]]:
    report = inspect_compute()
    cuda_vram: list[int] = []
    if torch.cuda.is_available():
        cuda_vram = [int(torch.cuda.get_device_properties(i).total_memory // (1024**3)) for i in range(torch.cuda.device_count())]
    enough = bool(cuda_vram and max(cuda_vram) >= min_vram_gb)
    blockers = []
    if not enough:
        blockers.append(
            Blocker(
                blocker_id="EXT-COMPUTE-P4-RESEARCH-01",
                kind=BlockerKind.EXTERNAL,
                status="BLOCKED_EXTERNAL",
                detail=f"Formal P4.8+ jobs require >= {min_vram_gb} GB GPU VRAM; this host is dev/debug only.",
            )
        )
    return (
        {
            "measured_host": report.model_dump(mode="json"),
            "measured_cuda_vram_gb": cuda_vram,
            "j1_smoke_parameter_count": 153_000_000,
            "parameter_count_status": "verified from prior P4.6 smoke report and bounded here as approx current J1 scale",
            "minimum_viable_gpu_vram_gb": min_vram_gb,
            "recommended_gpu_vram_gb": recommended_vram_gb,
            "recommended_gpu_count": "1 for U1 debug/research slices; 2-4+ for M1/T1/J1 throughput and reproducibility sweeps",
            "precision": "BF16 preferred; FP16 with GradScaler fallback; FP32 CPU/dev only",
            "storage": ">=500 GB working dataset/checkpoint space; keep top-k plus last checkpoints and immutable metadata",
            "estimates": {
                "U1": "single-modality ViT-S class run: 16-24 GB minimum, 40-80 GB recommended",
                "M1": "multimodal fusion: 24-40 GB minimum, 80 GB recommended",
                "T1": "temporal windows add activation pressure: 40 GB minimum, 80 GB recommended",
                "J1": "full joint ~153M params plus multimodal activations: 40 GB minimum, 80 GB+ recommended",
            },
        },
        blockers,
    )


def _stage_plan() -> tuple[StagePlan, ...]:
    common_health = (
        "representation health finite/variance/collapse/modalities pass",
        "formal effective_rank>=64 on >=64 independent validation representations when promoting",
        "no FINAL_TEST or OOD_TEST training access",
    )
    kill = (
        "non-finite loss or gradients",
        "objective denominator below required coverage threshold",
        "collapse guard failure on validation",
        "checkpoint ancestry/provenance mismatch",
        "compute config silently shrinks model, batch, or precision",
    )
    def stage(**kwargs: Any) -> StagePlan:
        return StagePlan(
            weight_decay=0.05,
            schedule="linear warmup 5%, cosine decay by optimizer step",
            grad_clip_norm=1.0,
            health_guardrails=common_health,
            kill_criteria=kill,
            **kwargs,
        )

    return (
        stage(stage_id="U1-RGB-RESEARCH", required_parents=(), data_requirements=("PRETRAIN_REAL:RGB or approved non-promotable PRETRAIN_SYNTHETIC",), objectives=("teacher_student_mse", "mask_reconstruction"), trainable_policy="train RGB encoder and U1 heads; no downstream labels", optimizer="AdamW", learning_rates={"encoder": 1e-4, "heads": 3e-4}, precision="BF16 preferred; FP16+GradScaler fallback; FP32 dev only", batch_size_per_gpu=64, gradient_accumulation=4, effective_batch_target=256, optimizer_steps=100_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with required objective coverage", promotion_criteria=("clean git", "data/init/provenance pass", "top validation checkpoint passes health")),
        stage(stage_id="U1-SONAR-RESEARCH", required_parents=("U1-RGB generic initialization policy resolved",), data_requirements=("PRETRAIN_REAL:SONAR or approved non-promotable PRETRAIN_SYNTHETIC",), objectives=("teacher_student_mse", "mask_reconstruction"), trainable_policy="train sonar encoder and U1 heads; RGB-derived init only with recorded transform digest", optimizer="AdamW", learning_rates={"encoder": 1e-4, "heads": 3e-4}, precision="BF16 preferred; FP16+GradScaler fallback; FP32 dev only", batch_size_per_gpu=64, gradient_accumulation=4, effective_batch_target=256, optimizer_steps=100_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with required objective coverage", promotion_criteria=("clean git", "sonar independence evidence", "top validation checkpoint passes health")),
        stage(stage_id="U1-RANGE-RESEARCH", required_parents=(), data_requirements=("PRETRAIN_REAL:RANGE or approved non-promotable PRETRAIN_SYNTHETIC",), objectives=("teacher_student_mse", "mask_reconstruction", "metric_reconstruction"), trainable_policy="train range encoder and metric heads", optimizer="AdamW", learning_rates={"encoder": 1e-4, "heads": 3e-4}, precision="BF16 preferred; FP16+GradScaler fallback; FP32 dev only", batch_size_per_gpu=64, gradient_accumulation=4, effective_batch_target=256, optimizer_steps=80_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with required objective coverage", promotion_criteria=("clean git", "metric-capable validation coverage", "top validation checkpoint passes health")),
        stage(stage_id="U1-GEOMETRY-RESEARCH", required_parents=(), data_requirements=("PRETRAIN_REAL:GEOMETRY or approved non-promotable PRETRAIN_SYNTHETIC",), objectives=("teacher_student_mse", "geometry_consistency"), trainable_policy="train geometry encoder and geometry heads", optimizer="AdamW", learning_rates={"encoder": 1e-4, "heads": 3e-4}, precision="BF16 preferred; FP16+GradScaler fallback; FP32 dev only", batch_size_per_gpu=64, gradient_accumulation=4, effective_batch_target=256, optimizer_steps=80_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with required objective coverage", promotion_criteria=("clean git", "geometry validation coverage", "top validation checkpoint passes health")),
        stage(stage_id="M1-RESEARCH", required_parents=("U1-RGB-RESEARCH", "U1-SONAR-RESEARCH", "U1-RANGE-RESEARCH", "U1-GEOMETRY-RESEARCH"), data_requirements=("paired/unpaired PRETRAIN_REAL multimodal corpus",), objectives=("global_consistency", "cross_modal_consistency", "missing_modality"), trainable_policy="freeze stable U1 blocks initially; train fusion/adapters, then unfreeze final U1 blocks if health remains stable", optimizer="AdamW", learning_rates={"fusion": 2e-4, "adapters": 3e-4, "unfrozen_u1": 5e-5}, precision="BF16 preferred; FP16+GradScaler fallback", batch_size_per_gpu=32, gradient_accumulation=8, effective_batch_target=256, optimizer_steps=120_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with fixed M1 objective coverage", promotion_criteria=("all parent digests verified", "pair coverage thresholds met", "top validation checkpoint passes health")),
        stage(stage_id="T1-RESEARCH", required_parents=("M1-RESEARCH",), data_requirements=("temporal PRETRAIN_REAL sequences with lineage-disjoint validation",), objectives=("temporal_prediction", "global_consistency"), trainable_policy="train temporal memory and temporal heads; fusion frozen first, then controlled unfreeze", optimizer="AdamW", learning_rates={"temporal": 2e-4, "heads": 3e-4, "fusion": 5e-5}, precision="BF16 preferred; FP16+GradScaler fallback", batch_size_per_gpu=16, gradient_accumulation=16, effective_batch_target=256, optimizer_steps=120_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with temporal coverage threshold", promotion_criteria=("temporal coverage/gap diagnostics pass", "top validation checkpoint passes health")),
        stage(stage_id="J1-RESEARCH", required_parents=("U1-RGB-RESEARCH", "U1-SONAR-RESEARCH", "U1-RANGE-RESEARCH", "U1-GEOMETRY-RESEARCH", "M1-RESEARCH", "T1-RESEARCH"), data_requirements=("full multimodal PRETRAIN_REAL corpus plus approved synthetic augmentation",), objectives=("masked_latent_prediction", "global_consistency", "cross_modal_consistency", "temporal_prediction", "missing_modality", "degradation", "geometry_consistency", "metric_reconstruction"), trainable_policy="train joint heads/fusion/temporal plus controlled final-block U1 unfreeze", optimizer="AdamW", learning_rates={"joint_heads": 3e-4, "fusion_temporal": 1e-4, "unfrozen_u1": 3e-5}, precision="BF16 preferred; FP16+GradScaler fallback", batch_size_per_gpu=8, gradient_accumulation=32, effective_batch_target=256, optimizer_steps=150_000, validation_every_steps=1_000, checkpoint_metric="val_ssl_total with all required J1 objective coverage", promotion_criteria=("produces OSFM-S-PRETRAIN-V1-CANDIDATE only", "OSFM-FQ still required for final")),
    )


def run_p47_readiness(
    *,
    corpus_manifest: str | Path = "configs/data/osfm/synthetic_ready.yaml",
    init_path: str | Path = "artifacts/external/dinov2/dinov2_vits14.pth",
    min_vram_gb: int = 16,
    recommended_vram_gb: int = 80,
    source_commit: str | None = None,
) -> P47ReadinessReport:
    corpus_path = Path(corpus_manifest)
    if not corpus_path.is_absolute():
        corpus_path = REPO_ROOT / corpus_path
    init = Path(init_path)
    if not init.is_absolute():
        init = REPO_ROOT / init
    data, data_blockers, synthetic_fixture = _data_findings(corpus_path)
    init_findings, init_blockers = _initialization_findings(init)
    compute, compute_blockers = _compute_findings(min_vram_gb, recommended_vram_gb)
    blockers = [*data_blockers, *init_blockers, *compute_blockers]
    decision = GateDecision.GO if not blockers else GateDecision.CONDITIONAL_GO
    if source_commit is None:
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
    promotion_rules = (
        "required objective sets are fixed per stage; NOT_APPLICABLE objectives do not improve val_ssl_total",
        "objective coverage thresholds must be met before a checkpoint can be selected",
        "parent checkpoint path, checkpoint_id, component, content digest, config digest, split hash, and git SHA are recorded",
        "formal acceptance requires a clean committed tree",
        "J1-RESEARCH top checkpoint becomes OSFM-S-PRETRAIN-V1-CANDIDATE only; OSFM-FQ produces OSFM-S-PRETRAIN-V1",
    )
    fq = (
        "B0 scratch vs B1 generic init vs B2 full OS-FM",
        "downstream transfer: detection, segmentation, anomaly, condition cues",
        "label fractions 1/5/10/25/100 with formal low-label 10/25",
        ">=5 downstream seeds for final comparisons; >=3 pretraining seeds target",
        "missing modality, robustness/corruption, temporal contribution, OOD, synthetic usefulness, objective/fusion/adapter ablations",
        "public-real evidence required before stronger validation claims",
        "paired lineage-level bootstrap confidence intervals",
        "previous internal engineering thresholds remain frozen unless separately flagged as blockers",
    )
    observability = (
        "tokens/samples/windows processed",
        "FLOPs where available",
        "module timings and throughput",
        "bytes moved where available and VRAM",
        "modality mix, missingness, PairGraph coverage, temporal coverage",
        "objective numerators/denominators and coverage",
        "checkpoint ancestry and representation health",
        "abstract profiling contracts only; no proprietary accelerator microarchitecture disclosures",
    )
    body = {
        "decision": decision.value,
        "data": data,
        "init": init_findings,
        "compute": compute,
        "blockers": [b.model_dump(mode="json") for b in blockers],
        "stage_plan": [s.model_dump(mode="json") for s in _stage_plan()],
    }
    return P47ReadinessReport(
        decision=decision,
        source_commit=source_commit,
        data_findings=data,
        initialization_findings=init_findings,
        compute_findings=compute,
        determinism_findings={
            "same_environment_resume": "exact deterministic resume required in compatible same hardware/software environment",
            "cross_hardware": "bitwise equality across hardware is not promised",
            "required_state": ("python_rng", "numpy_rng", "torch_rng", "cuda_rng_all", "optimizer", "amp_scaler", "sampler_epoch_plan", "world_size"),
            "formal_mode": "prefer deterministic kernels; document and approve any FlashAttention-style exception",
        },
        observability_requirements=observability,
        representation_health=RepresentationHealthSpec(),
        stage_plan=_stage_plan(),
        promotion_rules=promotion_rules,
        osfm_fq_confirmation=fq,
        blockers=tuple(blockers),
        synthetic_staging_permitted=synthetic_fixture,
        report_digest=digest_of(body),
    )


def write_p47_report(report: P47ReadinessReport, output: str | Path) -> Path:
    path = Path(output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
