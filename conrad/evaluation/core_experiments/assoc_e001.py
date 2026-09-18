"""CORE-ASSOC-E001: learned pair scorer + NO_MATCH vs nearest-neighbour gating (H-CORE-01 identity).

Beliefs are harness-maintained oracle tracks (truth used ONLY to build tracks and labels); a track's
latent is the mean of its past evidence embeddings (a harness summary, not the runtime BUO).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch

from conrad.core.association import (
    AssociationScorer,
    accept_associations,
    association_loss,
    nearest_neighbour_decision,
    pair_extra_features,
    retrieve_candidates,
)
from conrad.core.config import CoreConfig
from conrad.evaluation.core_experiments.common import core_config, run_experiment, seed_everything
from conrad.evaluation.core_experiments.sandbox_world import SandboxConfig, SandboxWorld
from conrad.schemas.belief import BeliefCell, KnowledgeStatus, Lifecycle
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

EXPERIMENT_ID = "CORE-ASSOC-E001"


def _cell(bid: UUID, track: dict[str, Any], ids: IdFactory) -> BeliefCell:
    n = len(track["embs"])
    pos = np.mean(track["pos"], axis=0)
    return BeliefCell(
        belief_id=bid,
        domain=Domain.SPATIAL,
        entity_type="object",
        lifecycle=Lifecycle.ACTIVE,
        revision=0,
        timestamp=track["ts"],
        state_embedding=tuple(float(v) for v in np.mean(track["embs"], axis=0)),
        knowledge_status=KnowledgeStatus.OBSERVED,
        uncertainty=Uncertainty(aleatoric=0.1, epistemic=0.1, contradiction=0.0, observational=1 / (1 + n)),
        spatial_support=SpatialSupport(
            frame_id="WORLD", center_m=tuple(float(x) for x in pos), position_sigma_m=0.3
        ),
        independent_observation_count=n,
        provenance_root=ids.new(),
        model_version="harness",
    )


def collect(world_cfg: SandboxConfig, seed: int, cfg: CoreConfig) -> list[dict[str, Any]]:
    """One sample per evidence: candidates, features and label index (K = NO_MATCH, -1 = retrieval miss)."""
    world = SandboxWorld(world_cfg, seed)
    ids = IdFactory(seed + 7)
    tracks: dict[int, dict[str, Any]] = {}
    bids: dict[int, UUID] = {}
    samples: list[dict[str, Any]] = []
    for step in world.episode():
        cells = {h: _cell(bids[h], t, ids) for h, t in tracks.items()}
        by_bid = {c.belief_id: h for h, c in cells.items()}
        for ev, _ in step.evidence:
            h = step.truth.source[ev.evidence_id]
            cands = retrieve_candidates(ev, list(cells.values()), cfg.association)
            hidden = [by_bid[c.belief_id] for c in cands]
            if h is None or h not in tracks:
                label = len(cands)
            else:
                label = hidden.index(h) if h in hidden else -1
            samples.append(
                {
                    "ev": ev,
                    "cands": cands,
                    "label": label,
                    "new_entity": h is None or h not in tracks,
                    "z": [list(c.state_embedding) for c in cands],
                    "extra": [pair_extra_features(ev, c, cfg) for c in cands],
                }
            )
        for ev, _ in step.evidence:  # oracle track maintenance AFTER the step's queries
            h = step.truth.source[ev.evidence_id]
            if h is None:
                continue
            if h not in tracks:
                tracks[h], bids[h] = {"embs": [], "pos": [], "ts": ev.timestamp}, ids.new()
            t = tracks[h]
            t["embs"] = (t["embs"] + [list(ev.embedding)])[-8:]
            assert ev.spatial_support is not None
            t["pos"] = (t["pos"] + [list(ev.spatial_support.center_m)])[-8:]
            t["ts"] = ev.timestamp
    return samples


def _tensors(
    samples: Sequence[dict[str, Any]], scorer: AssociationScorer, cfg: CoreConfig
) -> dict[str, torch.Tensor]:
    k = max(1, max(len(s["cands"]) for s in samples))
    b = len(samples)
    e = torch.tensor([list(s["ev"].embedding) for s in samples], dtype=torch.float32)
    z = torch.zeros(b, k, cfg.belief_dim)
    extra = torch.zeros(b, k, scorer.extra_dim)
    mask = torch.zeros(b, k, dtype=torch.bool)
    target = torch.zeros(b, dtype=torch.long)
    for i, s in enumerate(samples):
        n = len(s["cands"])
        if n:
            z[i, :n] = torch.tensor(s["z"])
            extra[i, :n] = torch.stack(s["extra"])
            mask[i, :n] = True
        target[i] = k if s["label"] == n else max(s["label"], 0)
    label_ok = torch.tensor([s["label"] >= 0 for s in samples])
    return {"e": e, "z": z, "extra": extra, "mask": mask, "target": target, "label_ok": label_ok}


def _metrics(pred: Sequence[int], samples: Sequence[dict[str, Any]]) -> dict[str, float]:
    """``pred`` is a candidate index, or -1 for NO_MATCH."""
    correct = wrong_assoc = new_total = new_hit = matched_total = matched_hit = 0
    for p, s in zip(pred, samples, strict=True):
        n = len(s["cands"])
        truth = -1 if s["label"] == n else s["label"]
        if s["new_entity"]:
            new_total += 1
            new_hit += int(p == -1)
        else:
            matched_total += 1
            matched_hit += int(p >= 0 and p == truth)
        correct += int((p == -1 and s["new_entity"]) or (p >= 0 and p == truth and not s["new_entity"]))
        wrong_assoc += int(p >= 0 and (s["new_entity"] or p != truth))
    total = max(len(samples), 1)
    return {
        "accuracy": correct / total,
        "false_association_rate": wrong_assoc / total,
        "new_entity_recall": new_hit / max(new_total, 1),
        "existing_entity_recall": matched_hit / max(matched_total, 1),
        "n_queries": float(len(samples)),
        "n_new_entity_queries": float(new_total),
        "retrieval_miss_rate": sum(s["label"] == -1 for s in samples) / total,
    }


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cfg = core_config(config)
    gen = seed_everything(seed)
    wcfg = SandboxConfig(**{**config.get("sandbox", {}), "embedding_dim": cfg.evidence_dim})
    train = [
        s for i in range(int(config.get("train_episodes", 4))) for s in collect(wcfg, seed * 100 + i, cfg)
    ]
    test = [
        s for i in range(int(config.get("test_episodes", 2))) for s in collect(wcfg, seed * 100 + 50 + i, cfg)
    ]
    scorer = AssociationScorer(cfg)
    opt = torch.optim.AdamW(scorer.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    batch = int(config.get("batch_size", 64))
    losses: list[float] = []
    scorer.train()
    for _ in range(int(config.get("epochs", 10))):
        order = gen.permutation(len(train))
        for start in range(0, len(order), batch):
            t = _tensors([train[j] for j in order[start : start + batch]], scorer, cfg)
            logits, _ = scorer(t["e"], t["z"], t["extra"], t["mask"])
            loss = association_loss(logits, t["mask"], t["target"], t["label_ok"], cfg)
            opt.zero_grad()
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg.grad_clip_norm)
            opt.step()
            losses.append(float(loss.total))
    scorer.eval()
    t = _tensors(test, scorer, cfg)
    with torch.no_grad():
        _, probs = scorer(t["e"], t["z"], t["extra"], t["mask"])
    a = cfg.association
    acc = accept_associations(probs, a.accept_probability, a.accept_margin)
    learned = [int(i) for i in acc.index]
    nn_pred = []
    for s in test:
        d = nearest_neighbour_decision(s["ev"], s["cands"], a)
        nn_pred.append(-1 if d.belief_id is None else [c.belief_id for c in s["cands"]].index(d.belief_id))
    return {
        "learned_scorer": _metrics(learned, test),
        "nearest_neighbour": _metrics(nn_pred, test),
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "n_train_queries": len(train),
    }


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_experiment(
        EXPERIMENT_ID, run_seed, config, seed, out_dir, "learned_scorer", ["nearest_neighbour"]
    )
