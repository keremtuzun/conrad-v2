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


# Kerem variant: the reviewed trainer reduced each frame to 32 row-band means, which carries almost no
# frame-to-frame variation and pinned validation rank near 4. Each frame is now a PATCH_GRID x PATCH_GRID
# grid of real image patches (16 tokens = GenericTokenizer.max_tokens), in the spirit of the P4.8 sonar run.
IMAGE_SIZE = 64
PATCH_GRID = 4
PATCH_DIM = (IMAGE_SIZE // PATCH_GRID) ** 2
TRAINER_VARIANT = "kerem_v11_10p_subpipe_patch_vicreg_v1"
BASE_LR = 2.0e-4
WARMUP_STEPS = 2000
VICREG_WEIGHTS = {"invariance": 25.0, "variance": 25.0, "covariance": 1.0, "rank": 1.0}
TOKEN_MASK_FRACTION = 0.25


def _frame_features(adapter: SubPipeAdapter, ref: Any) -> torch.Tensor:
    import cv2

    image = adapter.load(ref)
    if image.ndim == 3:
        image = image.mean(axis=2)
    resized = cv2.resize(image.astype("float32"), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float() / 255.0
    side = IMAGE_SIZE // PATCH_GRID
    patches = tensor.reshape(PATCH_GRID, side, PATCH_GRID, side).permute(0, 2, 1, 3)
    return patches.reshape(PATCH_GRID * PATCH_GRID, PATCH_DIM).contiguous()


def _augment(payload: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sonar-style view: per-frame gain, speckle noise, and random patch dropout."""
    batch, tokens, _ = payload.shape
    device = payload.device
    gain = 0.8 + 0.4 * torch.rand((batch, 1, 1), generator=generator).to(device)
    speckle = 1.0 + 0.10 * torch.randn(payload.shape, generator=generator).to(device)
    noise = 0.03 * torch.randn(payload.shape, generator=generator).to(device)
    keep = (torch.rand((batch, tokens, 1), generator=generator) >= TOKEN_MASK_FRACTION).to(device)
    return ((payload * gain * speckle + noise).clamp(0.0, 1.0) * keep).float()


def _vicreg_terms(z_a: torch.Tensor, z_b: torch.Tensor, rank_floor: float) -> dict[str, torch.Tensor]:
    z_a = z_a.float()
    z_b = z_b.float()
    invariance = F.mse_loss(z_a, z_b)

    def _var(z: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1.0e-4)).mean()

    def _cov(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        centered = z - z.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(n - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return off.pow(2).sum() / d

    rank_loss, _ = rank_diversity_loss(z_a, target=rank_floor)
    variance = _var(z_a) + _var(z_b)
    covariance = _cov(z_a) + _cov(z_b)
    total = (
        VICREG_WEIGHTS["invariance"] * invariance
        + VICREG_WEIGHTS["variance"] * variance
        + VICREG_WEIGHTS["covariance"] * covariance
        + VICREG_WEIGHTS["rank"] * rank_loss
    )
    return {"total": total, "invariance": invariance, "variance": variance, "covariance": covariance, "rank": rank_loss}


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
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    variant = {
        "trainer_variant": TRAINER_VARIANT,
        "image_size": IMAGE_SIZE,
        "patch_grid": PATCH_GRID,
        "patch_dim": PATCH_DIM,
        "base_lr": BASE_LR,
        "warmup_steps": WARMUP_STEPS,
        "vicreg_weights": VICREG_WEIGHTS,
        "token_mask_fraction": TOKEN_MASK_FRACTION,
        "autocast": "bf16" if use_bf16 else "fp32",
    }
    env = environment_snapshot(device=str(device), precision="bf16-autocast" if use_bf16 else "float32")
    run = RunDirectory.create(
        runs_root,
        run_name,
        resolved_config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke, "kerem_variant": variant},
        manifests={
            "payloads": payload_report,
            "subpipe_manifest": str(SUBPIPE_MANIFEST.relative_to(REPO_ROOT)),
        },
        purpose=RunPurpose.DEVELOPMENT if allow_cpu_smoke else RunPurpose.ACCEPTANCE,
        clock_ns=time.time_ns,
        environment=env,
    )
    model = UniversalOSFMV11(
        input_dims={"imaging_sonar": PATCH_DIM},
        family_encoder_config=FamilyEncoderConfig(depth=1 if allow_cpu_smoke else 2),
        include_v1_bank=not allow_cpu_smoke,
    ).to(device)
    # Anchor import: copy every shape-compatible tensor from the qualified upstream checkpoint.
    upstream_import: dict[str, Any] = {"attempted": False}
    if not resume and not allow_cpu_smoke:
        from conrad.foundation.universal_v11.migration import load_v1_checkpoint_for_v11

        upstream_path = REPO_ROOT / json.loads(PAYLOAD_MANIFEST.read_text(encoding="utf-8"))["payloads"][0]["path"]
        try:
            mig = load_v1_checkpoint_for_v11(model, upstream_path, strict_gate=False)
            upstream_import = {
                "attempted": True,
                "checkpoint_sha256": mig.checkpoint_sha256,
                "source_label": mig.source_label,
                "loaded_tensor_count": len(mig.loaded_keys),
                "skipped_tensor_count": len(mig.skipped_keys),
                "missing_v11_tensor_count": len(mig.missing_v11_keys),
                "loaded_tensors": list(mig.loaded_keys),
            }
        except Exception as exc:
            upstream_import = {"attempted": True, "error": f"{type(exc).__name__}: {exc}", "loaded_tensor_count": 0}
        model.to(device)
        run.write_artifact("reports", "upstream_import.json", json.dumps(upstream_import, indent=2, sort_keys=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < WARMUP_STEPS:
            return (step_idx + 1) / WARMUP_STEPS
        progress = (step_idx - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    start_step = 0
    manifest_digest = load_manifest(SUBPIPE_MANIFEST).manifest_digest()
    split_hash = split_hash_for_plan({"dataset": SUBPIPE_SOURCE_ID, "policy": "contiguous PRETRAIN_REAL/VALIDATION"})
    compatibility = CompatibilityTuple(
        config_digest=config_digest({**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke}),
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        component="foundation.osfm.universal_v11",
        representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
        extra={"trainer": TRAINER_VARIANT},
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
    aug_gen = torch.Generator().manual_seed(seed + 17)

    def _two_view_forward(payload: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        view_a = _augment(payload, aug_gen)
        view_b = _augment(payload, aug_gen)
        both = torch.cat([view_a, view_b], dim=0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                (_make_input("imaging_sonar", both, step),),
                reference_time_s=torch.full((both.shape[0],), float(step), device=device),
            )
        z = out.fusion.global_repr.float()
        return z[: payload.shape[0]], z[payload.shape[0] :]

    t_start = time.time()
    t_last, step_last = t_start, start_step

    def _on_sigterm(signum: int, frame: Any) -> None:  # watchdog stop -> sealed FAILED run, not a torn one
        raise RuntimeError("terminated by SIGTERM (external watchdog)")

    import signal

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        for step in range(start_step + 1, steps + 1):
            model.train()
            payload, sample_ids = batcher.batch(batcher.train_refs, step)
            train_sample_ids.update(sample_ids)
            z_a, z_b = _two_view_forward(payload, step)
            terms = _vicreg_terms(z_a, z_b, rank_floor)
            loss = terms["total"]
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
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        val_out = model((_make_input("imaging_sonar", val_payload, step),), reference_time_s=torch.full((val_payload.shape[0],), float(step), device=device))
                    health = _representation_health(val_out.fusion.global_repr.float(), rank_floor)
                    val_za, val_zb = _two_view_forward(val_payload, step)
                    val_terms = _vicreg_terms(val_za, val_zb, rank_floor)
                now = time.time()
                metric_record = {
                    "step": step,
                    "time_s": now - t_start,
                    "steps_per_s": (step - step_last) / max(now - t_last, 1.0e-9),
                    "train/loss": float(loss.detach().cpu()),
                    "train/invariance": float(terms["invariance"].detach().cpu()),
                    "train/variance": float(terms["variance"].detach().cpu()),
                    "train/covariance": float(terms["covariance"].detach().cpu()),
                    "train/rank_loss": float(terms["rank"].detach().cpu()),
                    "val/loss": float(val_terms["total"].cpu()),
                    "val/representation_effective_rank": health["effective_rank"],
                    "val/representation_collapse_score": health["collapse_score"],
                    "val/rank_pass": float(health["rank_pass"]),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                t_last, step_last = now, step
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
                        extra_compatibility={"trainer": TRAINER_VARIANT},
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
                    extra_compatibility={"trainer": TRAINER_VARIANT},
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
            "kerem_variant": variant,
            "reviewed_trainer": False,
            "upstream_import_loaded_tensor_count": upstream_import.get("loaded_tensor_count"),
            "wall_clock_s": time.time() - t_start,
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
