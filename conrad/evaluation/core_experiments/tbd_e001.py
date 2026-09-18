"""CORE-TBD-E001: temporal prediction over irregular PHYSICAL gaps (H-CORE-04).

Hidden x in R^4: two damped oscillators with different frequencies + process noise; observed at
irregular times. Task: predict x at the next observation time from the current observation and
delta_t. Methods: hold-last, linear extrapolation, AnalyticTBD (hold + fitted process-noise variance),
MLP TBD, GRUCell TBD, and a GRUCell TBD fed the SEQUENCE INDEX (constant step) instead of delta_t.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import expm
from scipy.stats import spearmanr
from torch import nn

from conrad.core.config import CoreConfig
from conrad.core.tbd import TemporalBeliefDynamics, build_tbd
from conrad.evaluation.core_experiments.common import core_config, run_experiment, seed_everything

EXPERIMENT_ID = "CORE-TBD-E001"
D = 4
OBS_SD = 0.05


def dynamics_matrix(w1: float, w2: float, damp: float) -> np.ndarray:
    a = np.zeros((D, D))
    a[0:2, 0:2] = [[-damp, -w1], [w1, -damp]]
    a[2:4, 2:4] = [[-damp, -w2], [w2, -damp]]
    return a


def make_sequences(rng: np.random.Generator, n: int, length: int, q: float) -> list[dict[str, np.ndarray]]:
    a = dynamics_matrix(0.15, 0.05, 0.01)
    out = []
    for _ in range(n):
        dts = np.exp(rng.uniform(np.log(0.5), np.log(20.0), size=length))
        x = rng.normal(size=D)
        xs, ys = [], []
        for dt in dts:
            x = expm(a * dt) @ x + rng.normal(scale=np.sqrt(q * dt), size=D)
            xs.append(x)
            ys.append(x + rng.normal(scale=OBS_SD, size=D))
        out.append({"dt": dts, "x": np.asarray(xs), "y": np.asarray(ys)})
    return out


def pairs(seqs: Sequence[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """(y_prev, dt_prev, y_k, dt_next) -> x_next for k >= 1."""
    rows: dict[str, list[Any]] = {"y_prev": [], "dt_prev": [], "y": [], "dt": [], "x_next": []}
    for s in seqs:
        for k in range(1, len(s["dt"]) - 1):
            rows["y_prev"].append(s["y"][k - 1])
            rows["dt_prev"].append(s["dt"][k])
            rows["y"].append(s["y"][k])
            rows["dt"].append(s["dt"][k + 1])
            rows["x_next"].append(s["x"][k + 1])
    return {k: torch.tensor(np.asarray(v), dtype=torch.float32) for k, v in rows.items()}


class Harness(nn.Module):
    def __init__(self, cfg: CoreConfig, kind: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(D, cfg.belief_dim)
        self.tbd: TemporalBeliefDynamics = build_tbd(cfg, kind)
        self.readout = nn.Linear(cfg.belief_dim, D)
        self.var_head = nn.Linear(cfg.uncertainty_dim, 1)

    def forward(self, y: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = y.shape[0]
        z = self.embed(y)
        pred = self.tbd(
            z, torch.zeros(n, self.cfg.uncertainty_dim), torch.zeros(n, self.cfg.temporal_dim), dt
        )
        var = F.softplus(self.var_head(pred.u)) + 1e-4
        return self.readout(pred.z), var.expand(-1, D)


def train(
    cfg: CoreConfig, kind: str, data: dict[str, torch.Tensor], epochs: int, index_time: bool
) -> Harness:
    model = Harness(cfg, kind)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate * 3, weight_decay=cfg.weight_decay)
    dt = torch.ones_like(data["dt"]) if index_time else data["dt"]
    n = data["y"].shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).split(128):
            mean, var = model(data["y"][idx], dt[idx])
            target = data["x_next"][idx]
            loss = F.mse_loss(mean, target) + 0.5 * F.gaussian_nll_loss(mean, target, var)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()
    return model.eval()


def _metrics(
    mean: np.ndarray, target: np.ndarray, dt: np.ndarray, var: np.ndarray | None
) -> dict[str, float]:
    err2 = ((mean - target) ** 2).mean(-1)
    out = {"rmse": float(np.sqrt(err2.mean()))}
    for name, lo, hi in (("short_lt3s", 0, 3), ("mid_3_10s", 3, 10), ("long_gt10s", 10, 1e9)):
        sel = (dt >= lo) & (dt < hi)
        out[f"rmse_{name}"] = float(np.sqrt(err2[sel].mean())) if sel.any() else float("nan")
    if var is not None:
        v = np.maximum(var, 1e-9)
        out["nll"] = float(np.mean(0.5 * (np.log(2 * np.pi * v) + (mean - target) ** 2 / v)))
        out["coverage95"] = float(np.mean(np.abs(mean - target) <= 1.96 * np.sqrt(v)))
        out["spearman_dt_vs_predicted_var"] = float(spearmanr(dt, v.mean(-1)).statistic)
    return out


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cfg = core_config(config)
    rng = seed_everything(seed)
    q = float(config.get("process_noise_per_s", 1e-3))
    length = int(config.get("length", 24))
    tr = pairs(make_sequences(rng, int(config.get("train_sequences", 60)), length, q))
    te = pairs(make_sequences(rng, int(config.get("test_sequences", 30)), length, q))
    y, dt, tgt = te["y"].numpy(), te["dt"].numpy(), te["x_next"].numpy()
    out: dict[str, Any] = {"hold_last": _metrics(y, tgt, dt, None)}
    rate = (te["y"] - te["y_prev"]).numpy() / te["dt_prev"].numpy()[:, None]
    out["linear_extrapolation"] = _metrics(y + rate * dt[:, None], tgt, dt, None)
    # AnalyticTBD: hold + variance growth with a process-noise rate FITTED on the training split
    resid = ((tr["x_next"] - tr["y"]) ** 2).mean(-1).numpy()
    q_hat = max(float(np.mean(np.maximum(resid - OBS_SD**2, 0) / tr["dt"].numpy())), 1e-9)
    var = (OBS_SD**2 + q_hat * dt)[:, None].repeat(D, 1)
    out["analytic_tbd"] = _metrics(y, tgt, dt, var)
    out["analytic_tbd"]["fitted_process_noise_per_s"] = q_hat
    epochs = int(config.get("epochs", 20))
    for name, kind, index in (
        ("mlp_tbd", "mlp", False),
        ("gru_tbd", "gru", False),
        ("gru_tbd_sequence_index", "gru", True),
    ):
        torch.manual_seed(seed)
        model = train(cfg, kind, tr, epochs, index)
        with torch.no_grad():
            m, v = model(te["y"], torch.ones_like(te["dt"]) if index else te["dt"])
            out[name] = _metrics(m.numpy(), tgt, dt, v.numpy())
            if name == "gru_tbd":
                m2, _ = model(te["y"], te["dt"] * 2)
                out[name]["mean_abs_output_change_when_dt_doubled"] = float((m2 - m).abs().mean())
    return out


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_experiment(
        EXPERIMENT_ID,
        run_seed,
        config,
        seed,
        out_dir,
        "gru_tbd (ch33 candidate)",
        ["hold_last", "linear_extrapolation", "analytic_tbd", "mlp_tbd", "gru_tbd_sequence_index"],
    )
