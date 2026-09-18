"""Four-channel uncertainty helpers and calibration metrics (ch6, ch28 Tensor and Mask Semantics).

The channels are never summed into a scalar confidence here.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.schemas.uncertainty import CalibrationMetadata, Uncertainty, UncertaintyRepresentation

CHANNELS: tuple[str, str, str, str] = ("aleatoric", "epistemic", "contradiction", "observational")


class UncertaintyHeads(nn.Module):
    """h_A(u), h_E(u), h_C(u), h_O(u): softplus scalar projection of each Du/4 partition (ch33 BUO)."""

    def __init__(self, uncertainty_dim: int) -> None:
        super().__init__()
        if uncertainty_dim % 4 != 0:
            raise ValueError("uncertainty_dim must be divisible by 4")
        self.partition_dim = uncertainty_dim // 4
        self.proj = nn.ModuleList(nn.Linear(self.partition_dim, 1) for _ in range(4))

    def partitions(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ua, ue, uc, uo = u.split(self.partition_dim, dim=-1)
        return ua, ue, uc, uo

    def forward(self, u: Tensor) -> Tensor:
        """``u`` [..., Du] -> non-negative channels [..., 4] ordered (UA, UE, UC, UO)."""
        parts = self.partitions(u)
        return torch.cat([F.softplus(p(x)) for p, x in zip(self.proj, parts, strict=True)], dim=-1)


def channels_to_uncertainty(
    channels: Tensor | np.ndarray | tuple[float, float, float, float],
    representation: UncertaintyRepresentation = UncertaintyRepresentation.SCALAR_CHANNELS,
    calibration: CalibrationMetadata | None = None,
) -> Uncertainty:
    values = [
        float(v) for v in (channels.detach().cpu().tolist() if isinstance(channels, Tensor) else channels)
    ]
    if len(values) != 4:
        raise ValueError("uncertainty requires exactly four channels (UA, UE, UC, UO)")
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"uncertainty channels must be finite and non-negative: {values}")
    return Uncertainty(
        aleatoric=values[0],
        epistemic=values[1],
        contradiction=values[2],
        observational=values[3],
        representation_type=representation,
        calibration_metadata=calibration or CalibrationMetadata(),
    )


def uncertainty_to_channels(u: Uncertainty) -> Tensor:
    return torch.tensor(u.as_tuple(), dtype=torch.float32)


def uncalibrated_metadata(method: str) -> CalibrationMetadata:
    return CalibrationMetadata(calibrated=False, method=method)


def calibrated_metadata(method: str, calibration_run_id: str) -> CalibrationMetadata:
    """Only a run that actually executed may be referenced here."""
    if not calibration_run_id:
        raise ValueError("a calibrated claim requires the id of the calibration run")
    return CalibrationMetadata(calibrated=True, method=method, calibration_run_id=calibration_run_id)


@dataclass(frozen=True)
class ReliabilityDiagram:
    bin_edges: tuple[float, ...]
    bin_confidence: tuple[float, ...]
    bin_accuracy: tuple[float, ...]
    bin_count: tuple[int, ...]
    ece: float

    def as_dict(self) -> dict[str, object]:
        return {
            "bin_edges": list(self.bin_edges),
            "bin_confidence": list(self.bin_confidence),
            "bin_accuracy": list(self.bin_accuracy),
            "bin_count": list(self.bin_count),
            "ece": self.ece,
        }


def reliability_diagram(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> ReliabilityDiagram:
    """Binary reliability diagram: predicted probability of the positive event vs. empirical frequency."""
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError("probabilities and outcomes differ in shape")
    if p.size == 0:
        raise ValueError("calibration metrics need at least one sample")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    conf, acc, count = [], [], []
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        count.append(n)
        conf.append(float(p[sel].mean()) if n else 0.0)
        acc.append(float(y[sel].mean()) if n else 0.0)
        if n:
            ece += n / p.size * abs(conf[-1] - acc[-1])
    return ReliabilityDiagram(
        tuple(float(e) for e in edges), tuple(conf), tuple(acc), tuple(count), float(ece)
    )


def expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    return reliability_diagram(probabilities, outcomes, bins).ece


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("calibration metrics need at least one sample")
    return float(np.mean((p - y) ** 2))


def binary_nll(probabilities: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64).ravel(), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("calibration metrics need at least one sample")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gaussian_nll(mean: np.ndarray, variance: np.ndarray, target: np.ndarray, min_var: float = 1e-9) -> float:
    m = np.asarray(mean, dtype=np.float64).ravel()
    v = np.maximum(np.asarray(variance, dtype=np.float64).ravel(), min_var)
    t = np.asarray(target, dtype=np.float64).ravel()
    if m.size == 0:
        raise ValueError("calibration metrics need at least one sample")
    return float(np.mean(0.5 * (np.log(2 * np.pi * v) + (t - m) ** 2 / v)))


def interval_coverage(
    mean: np.ndarray, variance: np.ndarray, target: np.ndarray, z: float = 1.959964
) -> float:
    """Fraction of targets inside the central Gaussian interval (nominal 95% at the default z)."""
    m = np.asarray(mean, dtype=np.float64).ravel()
    s = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64).ravel(), 0.0))
    t = np.asarray(target, dtype=np.float64).ravel()
    if m.size == 0:
        raise ValueError("calibration metrics need at least one sample")
    return float(np.mean(np.abs(t - m) <= z * s))


def heteroscedastic_nll(mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Per-element Gaussian NLL (constant dropped) used for ECMER aleatoric training (ch33)."""
    return 0.5 * (log_var + (target - mean) ** 2 * torch.exp(-log_var))
