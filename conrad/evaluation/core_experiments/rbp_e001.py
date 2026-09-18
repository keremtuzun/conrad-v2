"""CORE-RBP-E001: relational propagation, contamination by misleading edges, and INFERRED provenance.

Each graph group: observed source A; hidden B with a true ATTACHED edge A->B (b = a + noise);
hidden C two hops away (B->C ATTACHED); hidden decoy D linked by a misleading NEAR edge A->D (d is
independent). The child declares the same edge confidence for both types (it cannot tell them apart).
Methods: no propagation (population prior), AnalyticRBP, learned typed RBP, learned untyped RBP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from conrad.core.config import CoreConfig
from conrad.core.rbp import AnalyticRBP, RelationalBeliefPropagation, contamination_loss
from conrad.core.state import AnalyticBeliefState, PropertyEstimate
from conrad.evaluation.core_experiments.common import core_config, run_experiment, seed_everything
from conrad.evaluation.core_experiments.rbp_provenance import provenance_check
from conrad.schemas.belief import KnowledgeStatus, Relationship
from conrad.schemas.ids import IdFactory
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain

EXPERIMENT_ID = "CORE-RBP-E001"
ROLES = ("A", "B", "C", "D")
EDGES = ((0, 1, 0), (1, 2, 0), (0, 3, 1))  # (src role, dst role, type: 0 ATTACHED, 1 NEAR)


def make_graph(rng: np.random.Generator, groups: int) -> dict[str, np.ndarray]:
    vals, obs, role, src, dst, etype = [], [], [], [], [], []
    for g in range(groups):
        a = rng.uniform(0.1, 0.9)
        b = a + rng.normal(scale=0.03)
        c = b + rng.normal(scale=0.03)
        d = rng.uniform(0.1, 0.9)
        base = 4 * g
        vals += [a, b, c, d]
        obs += [1.0, 0.0, 0.0, 0.0]
        role += list(range(4))
        for s, t, k in EDGES:
            src.append(base + s)
            dst.append(base + t)
            etype.append(k)
    m = len(src)
    return {
        "vals": np.asarray(vals, dtype=np.float32),
        "obs": np.asarray(obs, dtype=np.float32),
        "role": np.asarray(role),
        "edge_index": np.asarray([src, dst]),
        "edge_type": np.asarray(etype),
        "edge_numeric": np.concatenate(
            [rng.normal(size=(m, 3)), rng.uniform(0.5, 2.0, (m, 2)), np.full((m, 1), 0.8)], 1
        ).astype(np.float32),
    }


class LearnedHarness(nn.Module):
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(2, cfg.belief_dim)
        self.rbp = RelationalBeliefPropagation(cfg)
        self.readout = nn.Linear(cfg.belief_dim, 1)
        self.cfg = cfg

    def forward(self, g: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        vals, obs = torch.from_numpy(g["vals"]), torch.from_numpy(g["obs"])
        z0 = self.embed(torch.stack([vals * obs, obs], -1))
        u = torch.zeros(len(vals), self.cfg.uncertainty_dim)
        out = self.rbp(
            z0,
            u,
            torch.from_numpy(g["edge_index"]),
            torch.from_numpy(g["edge_type"]),
            torch.from_numpy(g["edge_numeric"]),
        )
        return self.readout(z0).squeeze(-1), self.readout(out.z).squeeze(-1)


def train_learned(cfg: CoreConfig, rng: np.random.Generator, steps: int, groups: int) -> LearnedHarness:
    model = LearnedHarness(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate * 3, weight_decay=cfg.weight_decay)
    for _ in range(steps):
        g = make_graph(rng, groups)
        before, after = model(g)
        truth = torch.from_numpy(g["vals"])
        hidden = torch.from_numpy(g["obs"]) == 0
        state = ((after - truth) ** 2)[hidden].mean() + ((before - truth) ** 2)[~hidden].mean()
        cont, _ = contamination_loss(
            (before - truth).abs().detach(), (after - truth).abs(), torch.from_numpy(g["role"] == 3)
        )
        loss = state + cfg.rbp.contamination_weight * cont
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()
    return model.eval()


def analytic_predict(
    g: dict[str, np.ndarray], cfg: CoreConfig, ids: IdFactory
) -> tuple[np.ndarray, np.ndarray]:
    n = len(g["vals"])
    bids = [ids.new() for _ in range(n)]
    states = {
        bids[i]: AnalyticBeliefState(estimates={"p0": PropertyEstimate(float(g["vals"][i]), 0.03**2)})
        if g["obs"][i]
        else AnalyticBeliefState()
        for i in range(n)
    }
    rels = [
        Relationship(
            relationship_id=ids.new(),
            relation_type="ATTACHED" if k == 0 else "NEAR",
            source_belief_id=bids[s],
            target_belief_id=bids[t],
            source_domain=Domain.TECHNICAL,
            target_domain=Domain.TECHNICAL,
            confidence=0.8,
            uncertainty=unknown_uncertainty(),
            provenance_id=ids.new(),
        )
        for s, t, k in zip(g["edge_index"][0], g["edge_index"][1], g["edge_type"], strict=True)
    ]
    rbp = AnalyticRBP(cfg.rbp)
    for _ in range(cfg.rbp.layers):  # multi-hop: iterate over the current inferred states
        for inf in rbp.propagate(states, rels):
            states[inf.belief_id] = inf.state
    mean = np.full(n, 0.5)
    var = np.full(n, 1 / 12)
    for i, b in enumerate(bids):
        est = states[b].estimates.get("p0")
        if est is not None and est.status is not KnowledgeStatus.OBSERVED:
            mean[i], var[i] = est.mean, est.variance
    return mean, var


def _role_metrics(
    pred: np.ndarray, g: dict[str, np.ndarray], var: np.ndarray | None = None
) -> dict[str, float]:
    err = np.abs(pred - g["vals"])
    r = g["role"]
    out = {
        "rmse_hop1_B": float(np.sqrt(np.mean(err[r == 1] ** 2))),
        "rmse_hop2_C": float(np.sqrt(np.mean(err[r == 2] ** 2))),
        "rmse_decoy_D": float(np.sqrt(np.mean(err[r == 3] ** 2))),
        "contamination_rate_err_gt_0.2_D": float(np.mean(err[r == 3] > 0.2)),
    }
    if var is not None:
        out["confident_contamination_rate_D"] = float(
            np.mean((err[r == 3] > 0.2) & (np.sqrt(var[r == 3]) < 0.1))
        )
    return out


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cfg = core_config(config)
    rng = seed_everything(seed)
    groups, steps = int(config.get("groups", 16)), int(config.get("train_steps", 200))
    typed = train_learned(cfg, rng, steps, groups)
    untyped_cfg = cfg.model_copy(update={"rbp": cfg.rbp.model_copy(update={"typed": False})})
    torch.manual_seed(seed)
    untyped = train_learned(untyped_cfg, rng, steps, groups)
    test = [make_graph(rng, groups) for _ in range(int(config.get("test_graphs", 10)))]
    ids = IdFactory(seed)
    res: dict[str, list[dict[str, float]]] = {
        k: [] for k in ("no_propagation", "analytic_rbp", "learned_typed_rbp", "learned_untyped_rbp")
    }
    for g in test:
        res["no_propagation"].append(
            _role_metrics(np.full(len(g["vals"]), 0.5), g, np.full(len(g["vals"]), 1 / 12))
        )
        m, v = analytic_predict(g, cfg, ids)
        res["analytic_rbp"].append(_role_metrics(m, g, v))
        with torch.no_grad():
            res["learned_typed_rbp"].append(_role_metrics(typed(g)[1].numpy(), g))
            res["learned_untyped_rbp"].append(_role_metrics(untyped(g)[1].numpy(), g))
    out: dict[str, Any] = {
        k: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]} for k, rows in res.items()
    }
    out["provenance"] = provenance_check(cfg, seed)
    return out


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_experiment(
        EXPERIMENT_ID,
        run_seed,
        config,
        seed,
        out_dir,
        "learned_typed_rbp + analytic_rbp",
        ["no_propagation", "learned_untyped_rbp"],
    )
