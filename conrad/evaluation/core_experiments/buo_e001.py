"""CORE-BUO-E001: belief update under support / unreliability / contradiction / redundancy / degradation.

Candidate: AnalyticBUO (runtime default) and the learned BUO (trained here, small). Baselines:
latest-only, simple averaging, reliability-weighted averaging; analytic ablations without the
contradiction mechanism and without independence groups. Scenarios follow H-CORE-02 E-BUO-02.x.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from conrad.core.buo import AnalyticBUO, BeliefUpdateOperator
from conrad.core.config import AnalyticBuoConfig, CoreConfig
from conrad.core.pipeline_latent import quality_vector
from conrad.core.state import unknown_state
from conrad.evaluation.core_experiments.common import core_config, run_experiment, seed_everything
from conrad.evaluation.core_experiments.evidence_factory import EvidenceFactory
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence

EXPERIMENT_ID = "CORE-BUO-E001"
SCENARIOS = (
    "reliable_support",
    "unreliable_support",
    "reliable_contradiction",
    "unreliable_contradiction",
    "repeated_redundant",
    "sensor_degradation",
)
Seq = list[tuple[float, list[Evidence]]]  # (true state after the step, evidence of the step)


def make_sequence(kind: str, rng: np.random.Generator, fac: EvidenceFactory, steps: int) -> Seq:
    x = float(rng.uniform(0.2, 0.8))
    out: Seq = []
    for t in range(steps):
        if kind == "reliable_contradiction" and t == steps // 2:
            x = float(np.clip(x + rng.choice([-0.4, 0.4]), 0, 1))
        rel, sd, reps = 0.9, 0.05, 1
        if kind == "unreliable_support":
            rel, sd = 0.3, 0.3
        if kind == "sensor_degradation":
            frac = t / max(steps - 1, 1)
            rel, sd = 0.9 - 0.7 * frac, 0.05 + 0.3 * frac
        if kind == "repeated_redundant":
            reps = 5
        y = x + rng.normal(scale=sd)
        if kind == "unreliable_contradiction" and t % 4 == 3:
            rel, sd, y = 0.3, 0.3, 1.0 - x
        evs = [
            fac.make(
                float(t) * 10,
                {"p0": float(y)},
                reliability=rel,
                aleatoric=sd**2,
                independence_group=f"seq{t}",
            )[0]
            for _ in range(reps)
        ]
        out.append((x, evs))
    return out


def _mean_var_stats(errs: list[float], vars_: list[float] | None) -> dict[str, float | None]:
    e = np.asarray(errs)
    res: dict[str, float | None] = {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "final_abs_error": float(abs(e[-1])),
    }
    if vars_ is not None:
        v = np.maximum(np.asarray(vars_), 1e-9)
        res["nll"] = float(np.mean(0.5 * (np.log(2 * math.pi * v) + e**2 / v)))
        res["coverage95"] = float(np.mean(np.abs(e) <= 1.96 * np.sqrt(v)))
        res["confident_wrong_rate"] = float(np.mean((np.abs(e) > 0.15) & (np.sqrt(v) < 0.05)))
    return res


def run_analytic(
    seq: Seq, cfg: AnalyticBuoConfig, independence: bool = True
) -> tuple[dict[str, Any], list[float]]:
    buo, state = AnalyticBUO(cfg), unknown_state()
    errs, vars_, uc = [], [], []
    for x, evs in seq:
        if not independence:
            evs = [e.model_copy(update={"independence_group": None}) for e in evs]
        state = buo.update(state, evs).state
        est = state.estimates["p0"]
        errs.append(est.mean - x)
        vars_.append(est.variance)
        uc.append(state.uc)
    return _mean_var_stats(errs, vars_), uc


def run_baseline(seq: Seq, kind: str) -> dict[str, Any]:
    ys: list[float] = []
    ws: list[float] = []
    errs, vars_ = [], []
    for x, evs in seq:
        for e in evs:
            ys.append(e.measurements["p0"])
            ws.append(e.reliability if kind == "weighted" else 1.0)
        if kind == "latest":
            mean, var = ys[-1], evs[-1].aleatoric_uncertainty / max(evs[-1].reliability, 0.05)
        else:
            w = np.asarray(ws)
            mean = float(np.average(ys, weights=w))
            spread = float(np.average((np.asarray(ys) - mean) ** 2, weights=w)) if len(ys) > 1 else 0.25
            var = max(spread, 1e-4) / len(ys)
        errs.append(mean - x)
        vars_.append(var)
    return _mean_var_stats(errs, vars_)


class LearnedBuoHarness(nn.Module):
    """Learned BUO with a scalar input embedding (encoder stand-in) and a state readout."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(1, cfg.evidence_dim)
        self.buo = BeliefUpdateOperator(cfg)
        self.readout = nn.Linear(cfg.belief_dim, 1)

    def forward(self, seq: Seq) -> list[torch.Tensor]:
        c = self.cfg
        z, u, tmp = (
            torch.zeros(1, c.belief_dim),
            torch.zeros(1, c.uncertainty_dim),
            torch.zeros(1, c.temporal_dim),
        )
        preds = []
        for _, evs in seq:
            e = self.embed(torch.tensor([[[ev.measurements["p0"]] for ev in evs]], dtype=torch.float32))
            q = torch.tensor([[quality_vector(ev) for ev in evs]], dtype=torch.float32)
            groups = torch.tensor([[hash_group(ev) for ev in evs]])
            out = self.buo(z, u, tmp, e, q, torch.ones(1, len(evs), dtype=torch.bool), groups)
            z, u = out.z, out.u
            preds.append(self.readout(z)[0, 0])
        return preds


def hash_group(ev: Evidence) -> int:
    import zlib

    return zlib.crc32((ev.independence_group or str(ev.evidence_id)).encode()) % (2**31)


def train_learned(cfg: CoreConfig, train: Sequence[Seq], epochs: int, seed: int) -> LearnedBuoHarness:
    torch.manual_seed(seed)
    model = LearnedBuoHarness(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate * 3, weight_decay=cfg.weight_decay)
    for _ in range(epochs):
        for seq in train:
            preds = torch.stack(model(seq))
            truth = torch.tensor([x for x, _ in seq], dtype=torch.float32)
            loss = ((preds - truth) ** 2).mean() + cfg.buo.innovation_l2 * sum(
                p.pow(2).sum() for p in model.buo.innovation.parameters()
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()
    return model.eval()


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cfg = core_config(config)
    rng = seed_everything(seed)
    fac = EvidenceFactory(IdFactory(seed), cfg.evidence_dim, rng)
    steps, n_seq = int(config.get("steps", 16)), int(config.get("sequences_per_scenario", 6))
    train = [
        make_sequence(k, rng, fac, steps)
        for k in SCENARIOS
        for _ in range(int(config.get("train_per_scenario", 6)))
    ]
    learned = train_learned(cfg, train, int(config.get("epochs", 8)), seed)
    no_contra = cfg.analytic_buo.model_copy(update={"contradiction_sigma": 1e9})
    methods: dict[str, Callable[[Seq], dict[str, Any]]] = {
        "analytic_buo": lambda s: run_analytic(s, cfg.analytic_buo)[0],
        "analytic_no_contradiction": lambda s: run_analytic(s, no_contra)[0],
        "analytic_no_independence": lambda s: run_analytic(s, cfg.analytic_buo, independence=False)[0],
        "latest_only": lambda s: run_baseline(s, "latest"),
        "simple_average": lambda s: run_baseline(s, "average"),
        "reliability_weighted": lambda s: run_baseline(s, "weighted"),
    }
    out: dict[str, Any] = {}
    uc_jump: list[float] = []
    for kind in SCENARIOS:
        seqs = [make_sequence(kind, rng, fac, steps) for _ in range(n_seq)]
        res: dict[str, Any] = {}
        for name, fn in methods.items():
            rows = [fn(s) for s in seqs]
            res[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if rows[0][k] is not None}
        with torch.no_grad():
            errs = [[float(p) - x for p, (x, _) in zip(learned(s), s, strict=True)] for s in seqs]
        res["learned_buo"] = {
            "rmse": float(np.mean([np.sqrt(np.mean(np.square(e))) for e in errs])),
            "final_abs_error": float(np.mean([abs(e[-1]) for e in errs])),
        }
        if kind == "reliable_contradiction":
            for s in seqs:
                uc = run_analytic(s, cfg.analytic_buo)[1]
                uc_jump.append(uc[steps // 2] - uc[steps // 2 - 1])
        out[kind] = res
    out["analytic_uc_increase_at_reliable_contradiction"] = float(np.mean(uc_jump))
    return out


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    baselines = [
        "latest_only",
        "simple_average",
        "reliability_weighted",
        "analytic_no_contradiction",
        "analytic_no_independence",
    ]
    return run_experiment(
        EXPERIMENT_ID, run_seed, config, seed, out_dir, "analytic_buo + learned_buo", baselines
    )
