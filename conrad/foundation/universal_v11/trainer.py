"""Reviewed Universal V1.1 10P trainer surface.

This is intentionally narrow: it trains the current UniversalOSFMV11 interface on
verified SubPipe rendered-sonar payloads, records immutable run evidence, and
keeps semantic claims limited to data-backed modalities.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import torch
import torch.nn.functional as F
import yaml

from conrad.data.adapters.subpipe import SubPipeAdapter
from conrad.data.manifest import data_root_for, load_manifest
from conrad.foundation.data.manifest import CorpusPartition
from conrad.foundation.data.osfm_public_real import SUBPIPE_MANIFEST, SUBPIPE_SOURCE_ID, _partition_ranges
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11.core import (
    FamilyEncoderConfig,
    SensorMetadata,
    UniversalModalityInput,
    UniversalOSFMV11,
)
from conrad.foundation.universal_v11.registry import MODALITY_REGISTRY, EncoderFamily, ModalityState
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.determinism import inspect_compute
from conrad.training.run_dir import RunDirectory, RunPurpose, TerminalStatus, environment_snapshot


PAYLOAD_MANIFEST = REPO_ROOT / "artifacts/gates/V1.1/10P_KEREM_HANDOFF/required_payloads.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_required_payloads(manifest_path: str | Path = PAYLOAD_MANIFEST) -> dict[str, Any]:
    """Verify the ignored binary payloads needed before formal V1.1 10P training."""

    path = Path(manifest_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for payload in manifest["payloads"]:
        payload_path = REPO_ROOT / payload["path"]
        present = payload_path.is_file()
        actual = _sha256(payload_path) if present else None
        ok = present and actual == payload["sha256"]
        if not ok:
            blockers.append(f"{payload['id']} missing or hash mismatch")
        results.append(
            {
                "id": payload["id"],
                "path": payload["path"],
                "present": present,
                "expected_sha256": payload["sha256"],
                "actual_sha256": actual,
                "sha256_ok": ok,
            }
        )
    return {
        "gate_id": "OSFM-V11-10P-PAYLOAD-PREFLIGHT",
        "decision": "PASS" if not blockers else "BLOCKED",
        "payloads": results,
        "blockers": blockers,
    }


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _device(require_cuda: bool) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("formal V1.1 10P training requires CUDA unless --allow-cpu-smoke is set")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _subpipe_frame_refs(stream: str, partition: CorpusPartition, frame_stride: int) -> tuple[Any, ...]:
    manifest = load_manifest(SUBPIPE_MANIFEST)
    root = data_root_for(manifest.dataset_id)
    adapter = SubPipeAdapter(
        manifest,
        root,
        ObjectStore(REPO_ROOT / "artifacts/tmp/v11_10p_object_store"),
        IdFactory(seed=2026092912),
        UUID("00000000-0000-7000-8000-000000001011"),
        streams=(stream,),
        frame_stride=frame_stride,
        manifest_path=SUBPIPE_MANIFEST,
    )
    try:
        refs = adapter.frames(stream)[::frame_stride]
    finally:
        adapter.close()
    start, end = _partition_ranges(len(refs))[partition]
    selected = tuple(refs[start:end])
    if not selected:
        raise RuntimeError(f"SubPipe {stream} {partition.value} partition is empty")
    return selected


def _frame_features(adapter: SubPipeAdapter, ref: Any, *, tokens: int = 4, dim: int = 8) -> torch.Tensor:
    image = adapter.load(ref)
    tensor = torch.from_numpy(image).float()
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=2)
    tensor = tensor / 255.0
    flat = tensor.flatten()
    chunks = torch.chunk(flat, tokens * dim)
    values = torch.tensor([float(chunk.mean()) if chunk.numel() else 0.0 for chunk in chunks], dtype=torch.float32)
    return values.reshape(tokens, dim)


class _SubPipeBatcher:
    def __init__(self, *, stream: str, frame_stride: int, batch_size: int, device: torch.device, seed: int) -> None:
        self.manifest = load_manifest(SUBPIPE_MANIFEST)
        self.root = data_root_for(self.manifest.dataset_id)
        self.store = ObjectStore(REPO_ROOT / "artifacts/tmp/v11_10p_object_store")
        self.adapter = SubPipeAdapter(
            self.manifest,
            self.root,
            self.store,
            IdFactory(seed=seed),
            UUID("00000000-0000-7000-8000-000000001012"),
            streams=(stream,),
            frame_stride=frame_stride,
            manifest_path=SUBPIPE_MANIFEST,
        )
        all_refs = self.adapter.frames(stream)
        stride = max(1, frame_stride)
        while True:
            refs = all_refs[::stride]
            ranges = _partition_ranges(len(refs))
            train_start, train_end = ranges[CorpusPartition.PRETRAIN_REAL]
            val_start, val_end = ranges[CorpusPartition.VALIDATION]
            self.train_refs = tuple(refs[train_start:train_end])
            self.val_refs = tuple(refs[val_start:val_end])
            if self.train_refs and self.val_refs:
                break
            if stride == 1:
                break
            stride = max(1, stride // 2)
        if not self.train_refs or not self.val_refs:
            raise RuntimeError("SubPipe PRETRAIN_REAL and VALIDATION partitions must both be non-empty")
        self.stream = stream
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self._cache: dict[str, torch.Tensor] = {}

    def close(self) -> None:
        self.adapter.close()

    def _features(self, ref: Any) -> torch.Tensor:
        if ref.member not in self._cache:
            self._cache[ref.member] = _frame_features(self.adapter, ref)
        return self._cache[ref.member]

    def batch(self, refs: tuple[Any, ...], step: int) -> tuple[torch.Tensor, tuple[str, ...]]:
        gen = torch.Generator().manual_seed(self.seed + step)
        order = torch.randperm(len(refs), generator=gen).tolist()
        picked = tuple(refs[order[idx % len(order)]] for idx in range(self.batch_size))
        payload = torch.stack([self._features(ref) for ref in picked], dim=0).to(self.device)
        sample_ids = tuple(f"{self.stream}:{ref.member}" for ref in picked)
        return payload, sample_ids


def _make_input(name: str, payload: torch.Tensor, step: int) -> UniversalModalityInput:
    return UniversalModalityInput(
        name=name,
        payload=payload,
        state=ModalityState.AVAILABLE,
        timestamp_s=torch.full((payload.shape[0],), float(step), dtype=torch.float32, device=payload.device),
        metadata=SensorMetadata(
            quality=0.9,
            calibrated=False,
            acquisition_time_s=float(step),
            provenance="subpipe-rendered-side-scan-hash-verified",
            units="normalized_image_features",
        ),
    )


def _representation_health(reprs: torch.Tensor, rank_floor: float) -> dict[str, Any]:
    finite = bool(torch.isfinite(reprs).all().detach().cpu())
    collapse = float(reprs.float().std().detach().cpu())
    _, rank = rank_diversity_loss(reprs.float(), target=rank_floor)
    effective_rank = float(rank.detach().cpu())
    return {
        "finite": finite,
        "collapse_score": collapse,
        "effective_rank": effective_rank,
        "rank_floor": rank_floor,
        "rank_pass": finite and collapse > 1.0e-6 and effective_rank >= rank_floor,
    }


def run_v11_10p_training(
    *,
    config_path: str | Path,
    readiness_path: str | Path,
    runs_root: str | Path = "artifacts/runs",
    run_id: str | None = None,
    max_steps_override: int | None = None,
    resume: str | Path | None = None,
    allow_cpu_smoke: bool = False,
) -> dict[str, Any]:
    readiness = json.loads((REPO_ROOT / readiness_path if not Path(readiness_path).is_absolute() else Path(readiness_path)).read_text(encoding="utf-8"))
    if readiness.get("decision") != "READY FOR KEREM":
        raise RuntimeError("V1.1 10P training requires READY FOR KEREM readiness output")
    payload_report = verify_required_payloads()
    if payload_report["decision"] != "PASS":
        raise RuntimeError(f"required payloads are not verified: {payload_report['blockers']}")

    cfg = _load_config(config_path)
    steps = int(max_steps_override or cfg["training_budget"]["total_optimizer_steps"])
    batch_size = int(cfg["training_budget"].get("effective_batch_target", 256))
    if allow_cpu_smoke:
        batch_size = min(batch_size, 2)
    rank_floor = float(cfg["training_budget"].get("rank_floor", 75.0))
    checkpoint_every = int(cfg["training_budget"].get("checkpoint_every_steps", 5000))
    validation_every = int(cfg["training_budget"].get("validation_every_steps", 1000))
    seed = int(cfg.get("seed", 2026092910))
    device = _device(require_cuda=not allow_cpu_smoke)

    run_name = run_id or f"v11-10p-{time.time_ns()}"
    env = environment_snapshot(device=str(device), precision="float32")
    run = RunDirectory.create(
        runs_root,
        run_name,
        resolved_config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke},
        manifests={
            "payloads": payload_report,
            "subpipe_manifest": str(SUBPIPE_MANIFEST.relative_to(REPO_ROOT)),
        },
        purpose=RunPurpose.DEVELOPMENT if allow_cpu_smoke else RunPurpose.ACCEPTANCE,
        clock_ns=time.time_ns,
        environment=env,
    )
    model = UniversalOSFMV11(
        input_dims={"imaging_sonar": 8},
        family_encoder_config=FamilyEncoderConfig(depth=1 if allow_cpu_smoke else 2),
        include_v1_bank=not allow_cpu_smoke,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    start_step = 0
    manifest_digest = load_manifest(SUBPIPE_MANIFEST).manifest_digest()
    split_hash = split_hash_for_plan({"dataset": SUBPIPE_SOURCE_ID, "policy": "contiguous PRETRAIN_REAL/VALIDATION"})
    compatibility = CompatibilityTuple(
        config_digest=config_digest({**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke}),
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        component="foundation.osfm.universal_v11",
        representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
        extra={"trainer": "reviewed_v11_10p_subpipe_v1"},
    )
    if resume:
        loaded = load_checkpoint(resume, expected=compatibility, model=model, map_location=str(device))
        if loaded.optimizer_state is not None:
            optimizer.load_state_dict(loaded.optimizer_state)
        if loaded.scheduler_state is not None:
            scheduler.load_state_dict(loaded.scheduler_state)
        start_step = int(loaded.trainer_state.get("step", loaded.metadata.step))

    batcher = _SubPipeBatcher(stream="sss_lf", frame_stride=16 if not allow_cpu_smoke else 512, batch_size=batch_size, device=device, seed=seed)
    best_rank = -math.inf
    best_path: str | None = None
    last_path: str | None = None
    train_sample_ids: set[str] = set()
    try:
        for step in range(start_step + 1, steps + 1):
            model.train()
            payload, sample_ids = batcher.batch(batcher.train_refs, step)
            train_sample_ids.update(sample_ids)
            student = payload + torch.randn_like(payload) * 0.01
            out_a = model((_make_input("imaging_sonar", payload, step),), reference_time_s=torch.full((payload.shape[0],), float(step), device=device))
            out_b = model((_make_input("imaging_sonar", student, step),), reference_time_s=torch.full((payload.shape[0],), float(step), device=device))
            repr_a = out_a.fusion.global_repr.float()
            repr_b = out_b.fusion.global_repr.float()
            rank_loss, _ = rank_diversity_loss(repr_a, target=rank_floor)
            loss = F.mse_loss(F.normalize(repr_a, dim=-1), F.normalize(repr_b.detach(), dim=-1)) + rank_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % validation_every == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    val_payload, val_ids = batcher.batch(batcher.val_refs, step)
                    val_out = model((_make_input("imaging_sonar", val_payload, step),), reference_time_s=torch.full((val_payload.shape[0],), float(step), device=device))
                    health = _representation_health(val_out.fusion.global_repr, rank_floor)
                metric_record = {
                    "step": step,
                    "train/loss": float(loss.detach().cpu()),
                    "val/representation_effective_rank": health["effective_rank"],
                    "val/representation_collapse_score": health["collapse_score"],
                    "val/rank_pass": float(health["rank_pass"]),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                run.log_metrics(metric_record)
                if health["effective_rank"] > best_rank:
                    best_rank = health["effective_rank"]
                    best_path = str(run.path / "checkpoints" / "best.pt")
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        component="foundation.osfm.universal_v11",
                        config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke},
                        manifest_digests=(manifest_digest,),
                        split_hash=split_hash,
                        seeds={"torch": seed},
                        metrics={k: float(v) for k, v in metric_record.items() if isinstance(v, (int, float))},
                        epoch=0,
                        step=step,
                        created_time_ns=time.time_ns(),
                        device=str(device),
                        is_encoder=True,
                        representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
                        selection_metric="val/representation_effective_rank",
                        extra_compatibility={"trainer": "reviewed_v11_10p_subpipe_v1"},
                        trainer_state={"step": step, "train_sample_ids": sorted(train_sample_ids), "val_sample_ids": list(val_ids)},
                    )
            if step % checkpoint_every == 0 or step == steps:
                last_path = str(run.path / "checkpoints" / "last.pt")
                save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    component="foundation.osfm.universal_v11",
                    config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke},
                    manifest_digests=(manifest_digest,),
                    split_hash=split_hash,
                    seeds={"torch": seed},
                    metrics={"train/loss": float(loss.detach().cpu()), "best_rank": float(best_rank)},
                    epoch=0,
                    step=step,
                    created_time_ns=time.time_ns(),
                    device=str(device),
                    is_encoder=True,
                    representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
                    selection_metric="val/representation_effective_rank",
                    extra_compatibility={"trainer": "reviewed_v11_10p_subpipe_v1"},
                    trainer_state={"step": step, "train_sample_ids": sorted(train_sample_ids)},
                )
        report = {
            "gate_id": "OSFM-V11-10P-FORMAL-TRAINER",
            "status": "VALIDATED-RUN",
            "decision": "CONDITIONAL-GO" if best_rank >= rank_floor else "NO-GO",
            "run_dir": str(run.path),
            "steps_completed": steps,
            "best_rank": best_rank,
            "rank_floor": rank_floor,
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "data_scope": "SubPipe rendered side-scan sonar only",
            "semantic_claim": "data-backed active acoustic training only; 829 registry remains interface coverage",
            "registered_modality_count": len(MODALITY_REGISTRY),
            "family_count": len(EncoderFamily),
            "payloads": payload_report,
        }
        run.write_artifact("reports", "v11_10p_training_report.json", json.dumps(report, indent=2, sort_keys=True))
        run.seal(TerminalStatus.COMPLETED if report["decision"] != "NO-GO" else TerminalStatus.FAILED, time.time_ns())
        return report
    except Exception:
        run.seal(TerminalStatus.FAILED, time.time_ns())
        raise
    finally:
        batcher.close()


__all__ = ["run_v11_10p_training", "verify_required_payloads"]
