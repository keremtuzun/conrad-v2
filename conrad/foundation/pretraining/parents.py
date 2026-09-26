"""Fail-closed parent checkpoint component loading for OS-FM smokes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from conrad.training.checkpoint import CheckpointError, read_metadata
from conrad.training.checkpoint_meta import IncompatibilityReason, ReasonCode


@dataclass(frozen=True)
class ParentLoadStatus:
    name: str
    path: str
    expected_digest: str
    actual_digest: str | None
    expected_component: str
    actual_component: str | None
    state_prefix: str
    loaded: bool
    parameter_tensors_loaded: int
    parameter_values_loaded: int
    max_abs_diff_after_load: float | None
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, str | bool | int | float | None]:
        return self.__dict__.copy()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _error(reason: ReasonCode, detail: str) -> CheckpointError:
    return CheckpointError([IncompatibilityReason(code=reason, detail=detail)])


def load_parent_component(
    *,
    name: str,
    path: Path,
    expected_digest: str,
    expected_component: str,
    state_prefix: str,
    module: nn.Module,
) -> ParentLoadStatus:
    """Load one prefixed component from a governed checkpoint into ``module``.

    The adapter is intentionally narrow: it verifies the file digest, sidecar metadata
    component, state-key equality after prefix stripping, and tensor shapes before
    loading. It then checks the loaded tensors equal the source tensors.
    """

    if not path.is_file():
        raise _error(ReasonCode.PAYLOAD_MISSING, str(path))
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise _error(
            ReasonCode.CONTENT_DIGEST_MISMATCH,
            f"{name}: {actual_digest} != expected {expected_digest}",
        )
    metadata = read_metadata(path)
    if metadata.component != expected_component:
        raise _error(
            ReasonCode.COMPONENT_MISMATCH,
            f"{name}: {metadata.component} != expected {expected_component}",
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise _error(ReasonCode.FORMAT_UNSUPPORTED, f"{name}: missing model state")
    prefix = state_prefix + "."
    source = {
        key.removeprefix(prefix): value
        for key, value in payload["model"].items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    own = module.state_dict()
    if set(source) != set(own):
        raise _error(
            ReasonCode.MODEL_STATE_MISMATCH,
            f"{name}: missing={sorted(set(own) - set(source))[:5]} unexpected={sorted(set(source) - set(own))[:5]}",
        )
    mismatched = [
        key for key, value in source.items()
        if tuple(value.shape) != tuple(own[key].shape)
    ]
    if mismatched:
        key = mismatched[0]
        raise _error(
            ReasonCode.MODEL_STATE_MISMATCH,
            f"{name}: shape mismatch for {key}: {tuple(source[key].shape)} != {tuple(own[key].shape)}",
        )
    module.load_state_dict(source, strict=True)
    reloaded = module.state_dict()
    max_diff = max(
        float((reloaded[key].detach().cpu() - source[key].detach().cpu()).abs().max())
        for key in source
    )
    if max_diff > 0.0:
        raise _error(ReasonCode.MODEL_STATE_MISMATCH, f"{name}: post-load diff {max_diff}")
    return ParentLoadStatus(
        name=name,
        path=str(path),
        expected_digest=expected_digest,
        actual_digest=actual_digest,
        expected_component=expected_component,
        actual_component=metadata.component,
        state_prefix=state_prefix,
        loaded=True,
        parameter_tensors_loaded=len(source),
        parameter_values_loaded=sum(value.numel() for value in source.values()),
        max_abs_diff_after_load=max_diff,
        status="LOADED_VERIFIED",
    )
