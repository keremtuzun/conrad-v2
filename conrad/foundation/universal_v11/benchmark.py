"""Tiny Universal V1.1 pillar benchmarks before expensive cloud training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11.core import FamilyEncoderConfig, UniversalModalityInput, UniversalOSFMV11
from conrad.foundation.universal_v11.registry import EncoderFamily, MODALITY_REGISTRY, ModalityState


@dataclass(frozen=True)
class PillarProbe:
    name: str
    modalities: tuple[str, ...]
    status: str
    effective_rank: float
    finite: bool
    collapse_score: float
    reason: str


_PILLARS: tuple[PillarProbe, ...] = ()


DEFAULT_PILLARS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sonar_inherited", ("imaging_sonar",)),
    ("rgb_path", ("rgb_camera",)),
    ("range_geometry", ("structured_light", "tof_optical")),
    ("context_engineering", ("cad", "inspection_history")),
    ("ultrasonic_ndt", ("paut_scan", "tofd_scan", "tfm_scan")),
    ("chemical_ocean", ("ctd_profile", "methane", "ph_probe")),
    ("biology_sample", ("edna_sequence", "microbial_filter")),
    ("radiological", ("gamma_spectrometer",)),
    ("router_global_temporal", ("rgb_camera", "imaging_sonar", "paut_scan", "ctd_profile")),
)


def _chunks(items: tuple[str, ...], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(items[idx : idx + size] for idx in range(0, len(items), size))


def _all_registry_pillars(chunk_size: int = 32) -> tuple[tuple[str, tuple[str, ...]], ...]:
    pillars: list[tuple[str, tuple[str, ...]]] = []
    for family in EncoderFamily:
        names = tuple(spec.name for spec in MODALITY_REGISTRY if spec.family == family)
        for idx, chunk in enumerate(_chunks(names, chunk_size)):
            pillars.append((f"registry:{family.value}:{idx:02d}", chunk))
    return tuple(pillars)


def _family_pillars() -> tuple[tuple[str, tuple[str, ...]], ...]:
    pillars: list[tuple[str, tuple[str, ...]]] = []
    for family in EncoderFamily:
        names = tuple(spec.name for spec in MODALITY_REGISTRY if spec.family == family)
        if names:
            pillars.append((f"family:{family.value}", names[:8]))
    return tuple(pillars)


def _effective_rank(x: torch.Tensor) -> float:
    _, rank = rank_diversity_loss(x.float(), target=64.0)
    return float(rank.detach().cpu())


def _input(name: str, batch: int, input_dim: int, seed: int, state: ModalityState = ModalityState.AVAILABLE) -> UniversalModalityInput:
    gen = torch.Generator().manual_seed(seed)
    payload = torch.randn(batch, 4, input_dim, generator=gen)
    timestamps = torch.zeros(batch)
    return UniversalModalityInput(name=name, payload=payload, state=state, timestamp_s=timestamps)


def _probe(
    model: UniversalOSFMV11,
    name: str,
    modalities: tuple[str, ...],
    *,
    batch: int,
    input_dim: int,
    rank_floor: float,
    rank_target: float,
    seed: int,
) -> PillarProbe:
    with torch.no_grad():
        out = model(
            tuple(_input(modality, batch, input_dim, seed + idx) for idx, modality in enumerate(modalities)),
            reference_time_s=torch.zeros(batch),
            temporal_timestamps_s=torch.zeros(batch, 1),
            temporal_valid_mask=torch.ones(batch, 1, dtype=torch.bool),
        )
        reprs = out.fusion.global_repr
    finite = bool(torch.isfinite(reprs).all().detach().cpu())
    collapse = float(reprs.float().std().detach().cpu())
    rank = _effective_rank(reprs)
    if not finite:
        return PillarProbe(name, modalities, "NO-GO", rank, finite, collapse, "non-finite representation")
    if collapse <= 1e-6:
        return PillarProbe(name, modalities, "NO-GO", rank, finite, collapse, "collapsed representation")
    if rank < rank_floor:
        return PillarProbe(name, modalities, "NO-GO", rank, finite, collapse, f"rank below floor {rank_floor}")
    if rank < rank_target:
        return PillarProbe(name, modalities, "CONDITIONAL-GO", rank, finite, collapse, f"rank below target {rank_target}")
    return PillarProbe(name, modalities, "GO", rank, finite, collapse, "healthy")


def run_v11_pillar_benchmark(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    batch = int(cfg.get("batch_size", 96))
    input_dim = int(cfg.get("input_dim", 8))
    rank_floor = float(cfg.get("rank_floor", 64.0))
    rank_target = float(cfg.get("rank_target", 80.0))
    seed = int(cfg.get("seed", 20260929))
    use_fast_probe = bool(cfg.get("fast_probe", False))
    registry_scope = str(cfg.get("registry_scope", "pillars"))
    if registry_scope == "all":
        pillars = _all_registry_pillars(int(cfg.get("registry_chunk_size", 32))) + _family_pillars() + DEFAULT_PILLARS
    elif registry_scope == "families":
        pillars = _family_pillars() + DEFAULT_PILLARS
    elif registry_scope == "pillars":
        pillars = DEFAULT_PILLARS
    else:
        raise ValueError("registry_scope must be 'pillars', 'families', or 'all'")
    family_config = FamilyEncoderConfig(depth=1) if use_fast_probe else FamilyEncoderConfig()
    torch.manual_seed(seed)
    model = UniversalOSFMV11(
        input_dims={modality: input_dim for _, modalities in pillars for modality in modalities},
        family_encoder_config=family_config,
        include_v1_bank=not use_fast_probe,
    )
    model.eval()
    probes = tuple(
        _probe(
            model,
            name,
            modalities,
            batch=batch,
            input_dim=input_dim,
            rank_floor=rank_floor,
            rank_target=rank_target,
            seed=seed + idx * 100,
        )
        for idx, (name, modalities) in enumerate(pillars)
    )
    if any(probe.status == "NO-GO" for probe in probes):
        decision = "NO-GO"
    elif any(probe.status == "CONDITIONAL-GO" for probe in probes):
        decision = "CONDITIONAL-GO"
    else:
        decision = "GO"
    return {
        "gate_id": "OSFM-V11-PILLAR-BENCH-001",
        "status": "VALIDATED-RUN",
        "decision": decision,
        "rank_floor": rank_floor,
        "rank_target": rank_target,
        "registry_scope": registry_scope,
        "registry_size": len(MODALITY_REGISTRY),
        "registry_coverage": {
            "covered_modalities": len({modality for _, modalities in pillars for modality in modalities}),
            "total_modalities": len(MODALITY_REGISTRY),
            "coverage_complete": len({modality for _, modalities in pillars for modality in modalities})
            == len(MODALITY_REGISTRY),
        },
        "family_count": len(EncoderFamily),
        "cloud_budget": {
            "gpu_type": "L4",
            "gpu_count": 8,
            "max_hours": float(cfg.get("max_hours", 24.0)),
            "max_budget_tl": float(cfg.get("max_budget_tl", 9000.0)),
            "monitoring_policy": "hourly",
            "terminate_on": (
                "NO-GO pillar",
                "non-finite metrics",
                "rank below floor",
                "collapse",
                "runtime projection above budget",
                "cost projection above budget",
            ),
        },
        "formal_training_launched": False,
        "probes": [probe.__dict__ for probe in probes],
    }


def write_v11_pillar_benchmark(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
