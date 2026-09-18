"""CORE-UNC-E001: does each uncertainty channel respond to its own cause (ch6 Validation)?

Factors swept with controlled severity: sensor noise/corruption (-> UA), OOD (-> UE), credible
contradiction (-> UC), coverage deficit (-> UO). Candidate: AnalyticBUO four channels. Baseline: a
single confidence scalar (posterior variance) which cannot say WHY it is uncertain. Also reports
Gaussian calibration (NLL, 95% coverage) and a binary ECE/Brier on the event |error| < tolerance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, spearmanr

from conrad.core.buo import AnalyticBUO
from conrad.core.state import AnalyticBeliefState, unknown_state
from conrad.core.uncertainty import brier_score, expected_calibration_error, gaussian_nll, interval_coverage
from conrad.evaluation.core_experiments.common import run_experiment, seed_everything
from conrad.evaluation.core_experiments.evidence_factory import EvidenceFactory
from conrad.schemas.ids import IdFactory

EXPERIMENT_ID = "CORE-UNC-E001"
FACTORS = ("noise", "ood", "contradiction", "coverage_deficit")
CHANNELS = ("UA", "UE", "UC", "UO")


def scenario(
    factor: str, s: float, rng: np.random.Generator, fac: EvidenceFactory
) -> tuple[AnalyticBeliefState, float]:
    buo, state = AnalyticBUO(), unknown_state()
    x = float(rng.uniform(0.3, 0.7))
    n_prior = round(4 * (1 - s)) if factor == "coverage_deficit" else 4
    occl = s if factor == "coverage_deficit" else 0.0
    for t in range(n_prior):
        ev = fac.make(
            float(t),
            {"p0": x + rng.normal(scale=0.05)},
            reliability=0.9,
            aleatoric=0.0025,
            ood_score=0.05,
            occlusion=occl,
        )[0]
        state = buo.update(state, [ev]).state
    rel, sd, ood, y = 0.9, 0.05, 0.05, x + rng.normal(scale=0.05)
    if factor == "noise":
        rel, sd = 0.9 - 0.7 * s, 0.05 + 0.3 * s
        y = x + rng.normal(scale=sd)
    elif factor == "ood":
        ood = s
    elif factor == "contradiction":
        y = x + 0.6 * s
    ev = fac.make(10.0, {"p0": y}, reliability=rel, aleatoric=sd**2, ood_score=ood, occlusion=occl)[0]
    state = buo.update(state, [ev]).state
    return state, x


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    rng = seed_everything(seed)
    fac = EvidenceFactory(IdFactory(seed), 8, rng)
    levels = np.linspace(0, 1, int(config.get("levels", 6)))
    reps = int(config.get("repeats", 10))
    matrix: dict[str, dict[str, float]] = {}
    scalar_resp: dict[str, float] = {}
    for factor in FACTORS:
        sev, chans, var = [], [], []
        for s in levels:
            for _ in range(reps):
                st, _ = scenario(factor, float(s), rng, fac)
                sev.append(float(s))
                chans.append(st.channels)
                var.append(st.estimates["p0"].variance if "p0" in st.estimates else 1.0)
        arr = np.asarray(chans)
        matrix[factor] = {c: _rho(sev, arr[:, i].tolist()) for i, c in enumerate(CHANNELS)}
        scalar_resp[factor] = _rho(sev, var)
    diag = [matrix[f][c] for f, c in zip(FACTORS, CHANNELS, strict=True)]
    off = [abs(matrix[f][c]) for f in FACTORS for c in CHANNELS if CHANNELS.index(c) != FACTORS.index(f)]
    calib = _calibration(
        rng, fac, int(config.get("calibration_sequences", 60)), float(config.get("tolerance", 0.1))
    )
    return {
        "responsiveness_spearman": matrix,
        "diagonal_mean": float(np.mean(diag)),
        "diagonal_min": float(np.min(diag)),
        "off_diagonal_mean_abs": float(np.mean(off)),
        "single_scalar_baseline_spearman": scalar_resp,
        "calibration": calib,
    }


def _rho(x: Sequence[float], y: Sequence[float]) -> float:
    if np.std(y) < 1e-12:
        return 0.0
    return float(spearmanr(x, y).statistic)


def _calibration(rng: np.random.Generator, fac: EvidenceFactory, n: int, tol: float) -> dict[str, Any]:
    means, vars_, truth, avg_means, avg_vars = [], [], [], [], []
    for _ in range(n):
        buo, st, x = AnalyticBUO(), unknown_state(), float(rng.uniform(0.2, 0.8))
        ys: list[float] = []
        for t in range(int(rng.integers(1, 8))):
            bad = rng.random() < 0.3
            rel, sd = (0.3, 0.3) if bad else (0.9, 0.05)
            y = x + rng.normal(scale=sd)
            ys.append(y)
            st = buo.update(st, [fac.make(float(t), {"p0": y}, reliability=rel, aleatoric=sd**2)[0]]).state
        est = st.estimates["p0"]
        means.append(est.mean)
        vars_.append(est.variance)
        truth.append(x)
        avg_means.append(float(np.mean(ys)))
        avg_vars.append(max(float(np.var(ys)), 1e-4) / len(ys) if len(ys) > 1 else 0.25)

    def block(m: list[float], v: list[float]) -> dict[str, float]:
        m_a, v_a, t_a = np.asarray(m), np.maximum(np.asarray(v), 1e-9), np.asarray(truth)
        p_hit = norm.cdf(tol / np.sqrt(v_a)) - norm.cdf(-tol / np.sqrt(v_a))
        hit = (np.abs(m_a - t_a) < tol).astype(float)
        return {
            "rmse": float(np.sqrt(np.mean((m_a - t_a) ** 2))),
            "nll": gaussian_nll(m_a, v_a, t_a),
            "coverage95": interval_coverage(m_a, v_a, t_a),
            "ece_within_tol": expected_calibration_error(p_hit, hit),
            "brier_within_tol": brier_score(p_hit, hit),
        }

    return {"analytic_buo": block(means, vars_), "simple_average": block(avg_means, avg_vars), "n": float(n)}


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_experiment(
        EXPERIMENT_ID,
        run_seed,
        config,
        seed,
        out_dir,
        "analytic_buo four channels",
        ["single_scalar_posterior_variance", "simple_average (calibration)"],
    )
