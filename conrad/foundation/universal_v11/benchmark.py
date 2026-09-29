"""Tiny Universal V1.1 pillar benchmarks before expensive cloud training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
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
            "gpu_count": int(cfg.get("gpu_count", 8)),
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


def _representative_modalities() -> tuple[str, ...]:
    names: list[str] = []
    for family in EncoderFamily:
        names.append(next(spec.name for spec in MODALITY_REGISTRY if spec.family == family))
    return tuple(names)


def run_v11_training_microbenchmark(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a bounded synthetic optimizer benchmark; never promotes or launches formal training."""

    cfg = config or {}
    steps = int(cfg.get("optimizer_steps", 1000))
    warmup = int(cfg.get("warmup_discard_steps", min(100, max(0, steps // 10))))
    batch = int(cfg.get("batch_size", 1))
    input_dim = int(cfg.get("input_dim", 8))
    seed = int(cfg.get("seed", 2026092911))
    rank_floor = float(cfg.get("rank_floor", 75.0))
    target_steps = int(cfg.get("target_steps", 1_200_000))
    max_hours = float(cfg.get("max_hours", 72.0))
    max_budget_tl = float(cfg.get("max_budget_tl", 8000.0))
    hourly_cost_tl = float(cfg.get("hourly_cost_tl", 0.0))
    require_cuda = bool(cfg.get("require_cuda", False))
    output_dir = Path(str(cfg.get("output_dir", "artifacts/gates/V1.1/P4_1L4_10P_fallback_benchmark")))
    checkpoint = output_dir / "microbenchmark_checkpoint.pt"

    if require_cuda and not torch.cuda.is_available():
        return {
            "gate_id": "OSFM-V11-TRAINING-MICROBENCH-001",
            "status": "VALIDATED-RUN",
            "decision": "NO-GO",
            "reason": "cuda required but unavailable",
            "formal_training_launched": False,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    family_config = FamilyEncoderConfig(depth=int(cfg.get("family_depth", 1)))
    modalities = _representative_modalities()
    model = UniversalOSFMV11(
        input_dims={name: input_dim for name in modalities},
        family_encoder_config=family_config,
        include_v1_bank=False,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1.0e-4)))
    gen = torch.Generator(device=device).manual_seed(seed)
    timed_seconds = 0.0
    finite = True
    last_repr: torch.Tensor | None = None
    losses: list[float] = []

    for step in range(steps):
        inputs = tuple(
            UniversalModalityInput(
                name=name,
                payload=torch.randn(batch, 4, input_dim, generator=gen, device=device),
                state=ModalityState.AVAILABLE,
                timestamp_s=torch.zeros(batch, device=device),
            )
            for name in modalities
        )
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(
            inputs,
            reference_time_s=torch.zeros(batch, device=device),
            temporal_timestamps_s=torch.zeros(batch, 1, device=device),
            temporal_valid_mask=torch.ones(batch, 1, dtype=torch.bool, device=device),
        )
        reprs = out.fusion.global_repr
        rank_loss, _ = rank_diversity_loss(reprs.float(), target=rank_floor)
        loss = reprs.float().square().mean() + rank_loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            finite = False
            break
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= warmup:
            timed_seconds += elapsed
        last_repr = reprs.detach()
        losses.append(float(loss.detach().cpu()))

    successful_steps = len(losses)
    measured_steps = max(0, successful_steps - warmup)
    steps_per_second = measured_steps / timed_seconds if timed_seconds > 0 else 0.0
    final_rank = _effective_rank(last_repr if last_repr is not None else torch.zeros(batch, D_F, device=device))
    projected_hours = target_steps / steps_per_second / 3600.0 if steps_per_second > 0 else float("inf")
    projected_cost_tl = projected_hours * hourly_cost_tl if hourly_cost_tl > 0 else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": successful_steps}, checkpoint)
    loaded = torch.load(checkpoint, map_location=device)
    reload_matches = set(loaded["model"].keys()) == set(model.state_dict().keys())

    blockers: list[str] = []
    if successful_steps != steps:
        blockers.append("successful_steps_below_requested")
    if not finite:
        blockers.append("non_finite_loss")
    if final_rank < rank_floor:
        blockers.append("rank_below_75")
    if len(modalities) != len(EncoderFamily):
        blockers.append("not_all_15_families_routeable")
    if projected_hours > max_hours:
        blockers.append("projected_runtime_over_72h")
    if hourly_cost_tl > 0 and projected_cost_tl > max_budget_tl:
        blockers.append("projected_cost_over_8000_try")
    if not reload_matches:
        blockers.append("checkpoint_reload_failed")
    decision = "GO" if not blockers else "NO-GO"

    return {
        "gate_id": "OSFM-V11-TRAINING-MICROBENCH-001",
        "status": "VALIDATED-RUN",
        "decision": decision,
        "formal_training_launched": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "successful_steps": successful_steps,
        "warmup_discard_steps": warmup,
        "measured_steps": measured_steps,
        "timed_seconds": timed_seconds,
        "steps_per_second": steps_per_second,
        "target_steps": target_steps,
        "projected_hours": projected_hours,
        "hourly_cost_tl": hourly_cost_tl,
        "projected_cost_tl": projected_cost_tl,
        "rank_floor": rank_floor,
        "final_effective_rank": final_rank,
        "finite": finite,
        "family_count": len(EncoderFamily),
        "routeable_family_count": len(modalities),
        "registry_size": len(MODALITY_REGISTRY),
        "checkpoint": str(checkpoint),
        "checkpoint_reload_matches": reload_matches,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "blockers": blockers,
    }


def write_v11_training_microbenchmark(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
