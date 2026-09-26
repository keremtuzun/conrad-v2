"""Foundation batch collation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from conrad.foundation.data.capture import FoundationWindow


@dataclass(frozen=True)
class FoundationBatch:
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    natural_missing_mask: torch.Tensor
    artificial_dropout_mask: torch.Tensor
    token_mask: torch.Tensor
    modality_ids: tuple[tuple[str, ...], ...]
    lineage_ids: tuple[str, ...]
    pair_coverage: torch.Tensor


def collate_foundation_windows(
    windows: tuple[FoundationWindow, ...],
    *,
    modalities: tuple[str, ...],
    feature_dim: int = 8,
    artificial_dropout: dict[str, bool] | None = None,
) -> FoundationBatch:
    dropout = artificial_dropout or {}
    batch = len(windows)
    max_tokens = max((len(w.units) for w in windows), default=1)
    tokens = torch.zeros(batch, max_tokens, feature_dim, dtype=torch.float32)
    padding = torch.ones(batch, max_tokens, dtype=torch.bool)
    natural = torch.zeros(batch, len(modalities), dtype=torch.bool)
    artificial = torch.zeros(batch, len(modalities), dtype=torch.bool)
    token_mask = torch.zeros(batch, max_tokens, dtype=torch.bool)
    modality_rows: list[tuple[str, ...]] = []
    pair_cov = torch.zeros(batch, dtype=torch.float32)
    for row, window in enumerate(windows):
        row_modalities: list[str] = []
        for col, unit in enumerate(window.units):
            payload = torch.as_tensor(unit.payload, dtype=torch.float32).flatten()
            tokens[row, col, : min(feature_dim, payload.numel())] = payload[:feature_dim]
            padding[row, col] = False
            row_modalities.append(unit.modality)
        missing = set(window.natural_missing_modalities)
        for idx, modality in enumerate(modalities):
            natural[row, idx] = modality in missing
            artificial[row, idx] = bool(dropout.get(modality, False)) and modality not in missing
        modality_rows.append(tuple(row_modalities))
        pair_cov[row] = float(window.pair_graph.coverage()["pair_coverage"])
    return FoundationBatch(
        tokens=tokens,
        padding_mask=padding,
        natural_missing_mask=natural,
        artificial_dropout_mask=artificial,
        token_mask=token_mask,
        modality_ids=tuple(modality_rows),
        lineage_ids=tuple(w.window_id for w in windows),
        pair_coverage=pair_cov,
    )

