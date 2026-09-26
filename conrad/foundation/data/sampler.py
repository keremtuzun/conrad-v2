"""Deterministic hierarchical sampler and epoch plan."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from conrad.foundation.data.capture import FoundationWindow


@dataclass(frozen=True)
class EpochPlan:
    corpus_digest: str
    sampler_digest: str
    seed: int
    epoch: int
    ordered_window_ids: tuple[str, ...]

    @property
    def plan_digest(self) -> str:
        payload = {
            "corpus_digest": self.corpus_digest,
            "sampler_digest": self.sampler_digest,
            "seed": self.seed,
            "epoch": self.epoch,
            "ordered_window_ids": self.ordered_window_ids,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class HierarchicalSampler:
    seed: int
    batch_size: int
    rank: int = 0
    world_size: int = 1

    @property
    def sampler_digest(self) -> str:
        return hashlib.sha256(
            json.dumps({"batch_size": self.batch_size, "world_size": self.world_size}, sort_keys=True).encode()
        ).hexdigest()

    def plan(self, windows: tuple[FoundationWindow, ...], *, corpus_digest: str, epoch: int) -> EpochPlan:
        ids = sorted(w.window_id for w in windows)
        rng = random.Random(f"{corpus_digest}:{self.seed}:{epoch}:{self.sampler_digest}")
        rng.shuffle(ids)
        logical = tuple(ids)
        # Membership/order is defined before rank slicing so distributed workers can verify the same plan.
        return EpochPlan(corpus_digest, self.sampler_digest, self.seed, epoch, logical)

    def shard(self, plan: EpochPlan) -> tuple[str, ...]:
        return tuple(w for idx, w in enumerate(plan.ordered_window_ids) if idx % self.world_size == self.rank)

