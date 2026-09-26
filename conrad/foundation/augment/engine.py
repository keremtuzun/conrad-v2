"""Teacher/student view engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from conrad.foundation.augment.transforms import TransformRegistry, TransformTrace, ViewRecipe, replay_seed
from conrad.foundation.data.batch import FoundationBatch


@dataclass(frozen=True)
class View:
    batch: FoundationBatch
    traces: tuple[TransformTrace, ...]
    lineage_ids: tuple[str, ...]


class ViewEngine:
    def __init__(self, registry: TransformRegistry, *, seed: int) -> None:
        self.registry = registry
        self.seed = seed

    def make_view(self, batch: FoundationBatch, recipe: ViewRecipe) -> View:
        tokens = batch.tokens.clone()
        traces: list[TransformTrace] = []
        for transform_id in recipe.transforms:
            seed = replay_seed(self.seed, recipe.recipe_id, "|".join(batch.lineage_ids), transform_id)
            tokens, trace = self.registry.apply(transform_id, tokens, seed=seed)
            traces.append(trace)
        artificial = batch.artificial_dropout_mask.clone()
        modality_order = sorted({m for row in batch.modality_ids for m in row})
        for modality in recipe.modality_dropout:
            if modality in modality_order:
                artificial[:, modality_order.index(modality)] = True
        token_mask = batch.token_mask.clone()
        if recipe.token_mask_fraction:
            seed = replay_seed(self.seed, recipe.recipe_id, "|".join(batch.lineage_ids), "token_mask")
            g = torch.Generator(device=tokens.device).manual_seed(seed)
            visible = ~batch.padding_mask
            sampled = torch.rand(token_mask.shape, generator=g, device=tokens.device) < recipe.token_mask_fraction
            token_mask = sampled & visible
            tokens[token_mask] = 0
        viewed = FoundationBatch(
            tokens=tokens,
            padding_mask=batch.padding_mask.clone(),
            natural_missing_mask=batch.natural_missing_mask.clone(),
            artificial_dropout_mask=artificial,
            token_mask=token_mask,
            modality_ids=batch.modality_ids,
            lineage_ids=batch.lineage_ids,
            pair_coverage=batch.pair_coverage.clone(),
        )
        return View(batch=viewed, traces=tuple(traces), lineage_ids=batch.lineage_ids)

    def validate_lineage_stable(self, before: FoundationBatch, after: View) -> None:
        if before.lineage_ids != after.batch.lineage_ids:
            raise ValueError("augmentation changed lineage identity")

