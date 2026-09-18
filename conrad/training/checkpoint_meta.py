"""Checkpoint identity: metadata, compatibility tuple and explicit reason codes (SS-05, ch36)."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
from pydantic import Field, model_validator

from conrad.schemas.base import (
    ARCHITECTURE_ID,
    SCHEMA_VERSION,
    STACK_ID,
    ConradModel,
    SchemaVersionError,
    VersionedModel,
    check_schema_compatible,
    digest_of,
)

CHECKPOINT_FORMAT = "conrad-checkpoint-v1"


class ReasonCode(str, Enum):
    PAYLOAD_MISSING = "PAYLOAD_MISSING"
    METADATA_MISSING = "METADATA_MISSING"
    METADATA_INVALID = "METADATA_INVALID"
    CONTENT_DIGEST_MISMATCH = "CONTENT_DIGEST_MISMATCH"
    FORMAT_UNSUPPORTED = "FORMAT_UNSUPPORTED"
    ARCHITECTURE_ID_MISMATCH = "ARCHITECTURE_ID_MISMATCH"
    STACK_ID_MISMATCH = "STACK_ID_MISMATCH"
    SCHEMA_VERSION_INCOMPATIBLE = "SCHEMA_VERSION_INCOMPATIBLE"
    CONFIG_DIGEST_MISMATCH = "CONFIG_DIGEST_MISMATCH"
    MANIFEST_DIGEST_MISMATCH = "MANIFEST_DIGEST_MISMATCH"
    SPLIT_HASH_MISMATCH = "SPLIT_HASH_MISMATCH"
    COMPONENT_MISMATCH = "COMPONENT_MISMATCH"
    REPRESENTATION_PRETRAINING_MISMATCH = "REPRESENTATION_PRETRAINING_MISMATCH"
    EXTRA_ELEMENT_UNKNOWN = "EXTRA_ELEMENT_UNKNOWN"
    EXTRA_ELEMENT_MISMATCH = "EXTRA_ELEMENT_MISMATCH"
    UNSAFE_PAYLOAD = "UNSAFE_PAYLOAD"
    MODEL_STATE_MISMATCH = "MODEL_STATE_MISMATCH"


class IncompatibilityReason(ConradModel):
    code: ReasonCode
    detail: str

    def __str__(self) -> str:
        return f"{self.code.value}: {self.detail}"


class CheckpointError(RuntimeError):
    def __init__(self, reasons: list[IncompatibilityReason]) -> None:
        super().__init__("checkpoint rejected: " + "; ".join(str(r) for r in reasons))
        self.reasons = tuple(reasons)

    @property
    def codes(self) -> tuple[ReasonCode, ...]:
        return tuple(r.code for r in self.reasons)


class CompatibilityTuple(ConradModel):
    """What the caller requires. Every element is compared; an unknown element fails closed."""

    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
    schema_version: str = SCHEMA_VERSION
    config_digest: str
    manifest_digests: tuple[str, ...]
    split_hash: str
    component: str
    representation_pretraining_id: str | None = None
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="further ch36 elements when known: database_migration, twin_adapter_version, "
        "robot_config_digest, hardware_interface_capability_version",
    )


class CheckpointMetadata(VersionedModel):
    format: str = CHECKPOINT_FORMAT
    checkpoint_id: str = Field(pattern="^[0-9a-f]{64}$", description="sha256 of the payload file")
    component: str = Field(min_length=1)
    is_encoder: bool = False
    representation_pretraining_id: str | None = None
    architecture_id: str
    stack_id: str
    config: dict[str, Any]
    config_digest: str
    manifest_digests: tuple[str, ...]
    split_hash: str
    git_commit: str | None
    git_dirty: bool | None
    seeds: dict[str, int]
    metrics: dict[str, float]
    selection_metric: str | None = None
    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    device_info: dict[str, Any]
    extra_compatibility: dict[str, str] = Field(default_factory=dict)
    created_time_ns: int = Field(ge=0)
    has_optimizer: bool
    has_scheduler: bool

    @model_validator(mode="after")
    def _encoder_identity(self) -> CheckpointMetadata:
        if self.is_encoder and not self.representation_pretraining_id:
            raise ValueError("an encoder checkpoint must record its representation_pretraining_id")
        if self.selection_metric is not None and not is_validation_metric(self.selection_metric):
            raise ValueError(
                f"checkpoints are selected on validation metrics only, not {self.selection_metric!r}"
            )
        return self


class LoadedCheckpoint(ConradModel):
    model_config = {**ConradModel.model_config, "arbitrary_types_allowed": True}

    metadata: CheckpointMetadata
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    trainer_state: dict[str, Any]


def is_validation_metric(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(("val_", "val/", "validation_", "validation/"))


def config_digest(config: Mapping[str, Any]) -> str:
    return digest_of(dict(config))


def device_info(device: str | torch.device = "cpu", *, device_fallback: bool = False) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "device": str(device),
        "device_fallback": device_fallback,
        "cuda_available": cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda
        else [],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def compatibility_reasons(
    metadata: CheckpointMetadata, expected: CompatibilityTuple
) -> list[IncompatibilityReason]:
    reasons: list[IncompatibilityReason] = []

    def add(code: ReasonCode, detail: str) -> None:
        reasons.append(IncompatibilityReason(code=code, detail=detail))

    if metadata.format != CHECKPOINT_FORMAT:
        add(ReasonCode.FORMAT_UNSUPPORTED, f"{metadata.format!r}")
    if metadata.architecture_id != expected.architecture_id:
        add(
            ReasonCode.ARCHITECTURE_ID_MISMATCH,
            f"{metadata.architecture_id!r} != {expected.architecture_id!r}",
        )
    if metadata.stack_id != expected.stack_id:
        add(ReasonCode.STACK_ID_MISMATCH, f"{metadata.stack_id!r} != {expected.stack_id!r}")
    try:
        check_schema_compatible(metadata.schema_version, expected.schema_version)
    except SchemaVersionError as exc:
        add(ReasonCode.SCHEMA_VERSION_INCOMPATIBLE, str(exc))
    if metadata.config_digest != expected.config_digest:
        add(ReasonCode.CONFIG_DIGEST_MISMATCH, f"{metadata.config_digest} != {expected.config_digest}")
    elif config_digest(metadata.config) != metadata.config_digest:
        add(ReasonCode.CONFIG_DIGEST_MISMATCH, "stored config does not hash to its recorded config_digest")
    if sorted(metadata.manifest_digests) != sorted(expected.manifest_digests):
        add(
            ReasonCode.MANIFEST_DIGEST_MISMATCH,
            f"{sorted(metadata.manifest_digests)} != {sorted(expected.manifest_digests)}",
        )
    if metadata.split_hash != expected.split_hash:
        add(ReasonCode.SPLIT_HASH_MISMATCH, f"{metadata.split_hash!r} != {expected.split_hash!r}")
    if metadata.component != expected.component:
        add(ReasonCode.COMPONENT_MISMATCH, f"{metadata.component!r} != {expected.component!r}")
    if metadata.representation_pretraining_id != expected.representation_pretraining_id:
        add(
            ReasonCode.REPRESENTATION_PRETRAINING_MISMATCH,
            f"{metadata.representation_pretraining_id!r} != {expected.representation_pretraining_id!r}",
        )
    for key, wanted in sorted(expected.extra.items()):
        found = metadata.extra_compatibility.get(key)
        if found is None:
            add(ReasonCode.EXTRA_ELEMENT_UNKNOWN, f"checkpoint does not record {key!r}")
        elif found != wanted:
            add(ReasonCode.EXTRA_ELEMENT_MISMATCH, f"{key}: {found!r} != {wanted!r}")
    return reasons
