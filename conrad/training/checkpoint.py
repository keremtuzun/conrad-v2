"""Checkpoint format and fail-closed loading (SS-05, ch36 compatibility tuple).

A checkpoint is two files: ``<name>.pt`` (tensors and plain containers only, always read with
``torch.load(weights_only=True)``) and ``<name>.pt.meta.json`` (identity + sha256 of the payload).
Loading verifies the payload digest, then the compatibility tuple, and only then deserializes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID, SchemaVersionError, check_schema_compatible
from conrad.training.checkpoint_meta import (
    CHECKPOINT_FORMAT,
    CheckpointError,
    CheckpointMetadata,
    CompatibilityTuple,
    IncompatibilityReason,
    LoadedCheckpoint,
    ReasonCode,
    compatibility_reasons,
    config_digest,
    device_info,
    is_validation_metric,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "META_SUFFIX",
    "CheckpointError",
    "CheckpointMetadata",
    "CompatibilityTuple",
    "IncompatibilityReason",
    "LoadedCheckpoint",
    "ReasonCode",
    "compatibility_reasons",
    "config_digest",
    "is_validation_metric",
    "load_checkpoint",
    "meta_path",
    "read_metadata",
    "save_checkpoint",
    "verify_checkpoint_metadata",
]

META_SUFFIX = ".meta.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def meta_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + META_SUFFIX)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    component: str,
    config: Mapping[str, Any],
    manifest_digests: tuple[str, ...],
    split_hash: str,
    seeds: Mapping[str, int],
    metrics: Mapping[str, float],
    epoch: int,
    step: int,
    created_time_ns: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    device: str = "cpu",
    device_fallback: bool = False,
    is_encoder: bool = False,
    representation_pretraining_id: str | None = None,
    selection_metric: str | None = None,
    extra_compatibility: Mapping[str, str] | None = None,
    architecture_id: str = ARCHITECTURE_ID,
    stack_id: str = STACK_ID,
) -> CheckpointMetadata:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "trainer_state": dict(trainer_state or {}),
    }
    tmp = target.with_name(target.name + ".tmp")
    # Prove now that the payload needs no pickle globals; otherwise it could never be loaded safely.
    try:
        torch.save(payload, tmp)
        torch.load(tmp, map_location="cpu", weights_only=True)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise CheckpointError(
            [
                IncompatibilityReason(
                    code=ReasonCode.UNSAFE_PAYLOAD, detail=f"payload is not weights_only loadable: {exc}"
                )
            ]
        ) from exc
    os.replace(tmp, target)
    metadata = CheckpointMetadata(
        checkpoint_id=_sha256_file(target),
        component=component,
        is_encoder=is_encoder,
        representation_pretraining_id=representation_pretraining_id,
        architecture_id=architecture_id,
        stack_id=stack_id,
        config=json.loads(json.dumps(dict(config), default=str)),
        config_digest=config_digest(config),
        manifest_digests=tuple(manifest_digests),
        split_hash=split_hash,
        git_commit=git_commit,
        git_dirty=git_dirty,
        seeds=dict(seeds),
        metrics={k: float(v) for k, v in metrics.items()},
        selection_metric=selection_metric,
        epoch=epoch,
        step=step,
        device_info=device_info(device, device_fallback=device_fallback),
        extra_compatibility=dict(extra_compatibility or {}),
        created_time_ns=created_time_ns,
        has_optimizer=optimizer is not None,
        has_scheduler=scheduler is not None,
    )
    meta_path(target).write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def read_metadata(path: str | Path) -> CheckpointMetadata:
    """Digest-verified metadata. Does not deserialize the tensor payload."""
    target = Path(path)
    if not target.is_file():
        raise CheckpointError([IncompatibilityReason(code=ReasonCode.PAYLOAD_MISSING, detail=str(target))])
    sidecar = meta_path(target)
    if not sidecar.is_file():
        raise CheckpointError([IncompatibilityReason(code=ReasonCode.METADATA_MISSING, detail=str(sidecar))])
    try:
        metadata = CheckpointMetadata.model_validate(json.loads(sidecar.read_text(encoding="utf-8")))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise CheckpointError(
            [IncompatibilityReason(code=ReasonCode.METADATA_INVALID, detail=str(exc))]
        ) from exc
    actual = _sha256_file(target)
    if actual != metadata.checkpoint_id:
        raise CheckpointError(
            [
                IncompatibilityReason(
                    code=ReasonCode.CONTENT_DIGEST_MISMATCH,
                    detail=f"payload sha256 {actual} != recorded checkpoint_id {metadata.checkpoint_id}",
                )
            ]
        )
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    expected: CompatibilityTuple,
    model: torch.nn.Module | None = None,
    map_location: str = "cpu",
) -> LoadedCheckpoint:
    """Verify digest and compatibility first; deserialize only with ``weights_only=True``."""
    metadata = read_metadata(path)
    reasons = compatibility_reasons(metadata, expected)
    if reasons:
        raise CheckpointError(reasons)
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(
            [IncompatibilityReason(code=ReasonCode.UNSAFE_PAYLOAD, detail=str(exc))]
        ) from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            [IncompatibilityReason(code=ReasonCode.FORMAT_UNSUPPORTED, detail="payload format marker")]
        )
    if model is not None:
        own = set(model.state_dict())
        stored = set(payload["model"])
        if own != stored:
            detail = f"missing={sorted(own - stored)[:5]} unexpected={sorted(stored - own)[:5]}"
            raise CheckpointError(
                [IncompatibilityReason(code=ReasonCode.MODEL_STATE_MISMATCH, detail=detail)]
            )
        try:
            model.load_state_dict(payload["model"], strict=True)
        except RuntimeError as exc:
            raise CheckpointError(
                [IncompatibilityReason(code=ReasonCode.MODEL_STATE_MISMATCH, detail=str(exc))]
            ) from exc
    return LoadedCheckpoint(
        metadata=metadata,
        model_state=payload["model"],
        optimizer_state=payload.get("optimizer"),
        scheduler_state=payload.get("scheduler"),
        trainer_state=payload.get("trainer_state") or {},
    )


def verify_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    """Doctor check without a run context: payload digest plus frozen stack identity.

    Raises :class:`CheckpointError` with reason codes. Config / manifest / split identity needs a
    full :class:`CompatibilityTuple` and is checked by :func:`load_checkpoint`.
    """
    metadata = read_metadata(path)
    reasons: list[IncompatibilityReason] = []
    if metadata.format != CHECKPOINT_FORMAT:
        reasons.append(IncompatibilityReason(code=ReasonCode.FORMAT_UNSUPPORTED, detail=metadata.format))
    if metadata.architecture_id != ARCHITECTURE_ID:
        reasons.append(
            IncompatibilityReason(code=ReasonCode.ARCHITECTURE_ID_MISMATCH, detail=metadata.architecture_id)
        )
    if metadata.stack_id != STACK_ID:
        reasons.append(IncompatibilityReason(code=ReasonCode.STACK_ID_MISMATCH, detail=metadata.stack_id))
    try:
        check_schema_compatible(metadata.schema_version)
    except SchemaVersionError as exc:
        reasons.append(IncompatibilityReason(code=ReasonCode.SCHEMA_VERSION_INCOMPATIBLE, detail=str(exc)))
    if config_digest(metadata.config) != metadata.config_digest:
        reasons.append(
            IncompatibilityReason(code=ReasonCode.CONFIG_DIGEST_MISMATCH, detail="stored config hash")
        )
    if reasons:
        raise CheckpointError(reasons)
    return metadata
