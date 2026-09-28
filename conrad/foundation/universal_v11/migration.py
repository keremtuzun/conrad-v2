"""V1 -> Universal V1.1 checkpoint import helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class MigrationReport:
    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    missing_v11_keys: tuple[str, ...]
    source_label: str | None
    source_status: str | None
    decision: str | None
    checkpoint_sha256: str | None = None

    @property
    def qualified_for_training(self) -> bool:
        return self.source_label == "OSFM-S-PRETRAIN-V1" and self.source_status == "VALIDATED-RUN" and self.decision in {"GO", "PROMOTE"}


def load_v1_checkpoint_for_v11(model: nn.Module, checkpoint_path: str | Path, *, strict_gate: bool = True) -> MigrationReport:
    path = Path(checkpoint_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = torch.load(path, map_location="cpu")
    metadata: dict[str, Any] = checkpoint.get("metadata", {})
    report_gate = (
        metadata.get("label"),
        metadata.get("status"),
        metadata.get("decision"),
    )
    if strict_gate and not (report_gate[0] == "OSFM-S-PRETRAIN-V1" and report_gate[1] == "VALIDATED-RUN" and report_gate[2] in {"GO", "PROMOTE"}):
        raise ValueError("V1.1 formal training import requires final P4.12-qualified V1 checkpoint metadata")
    source = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    target = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    skipped = tuple(sorted(key for key, value in source.items() if key not in compatible or key not in target or target.get(key, value).shape != value.shape))
    missing = tuple(sorted(set(target) - set(compatible)))
    model.load_state_dict(compatible, strict=False)
    return MigrationReport(tuple(sorted(compatible)), skipped, missing, *report_gate, checkpoint_sha256=digest)


def parameter_count(model: nn.Module) -> dict[str, int]:
    counts = {name: sum(param.numel() for param in module.parameters()) for name, module in model.named_children()}
    counts["total"] = sum(param.numel() for param in model.parameters())
    return counts
