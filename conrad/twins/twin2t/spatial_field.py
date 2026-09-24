"""Spatially heterogeneous capsule-surface truth for structural-observation-v2.

This is truth-plane code.  Neither the local belief model nor planning imports it.
Patches are explicit and may be smaller than a sensor footprint.  The integral
uses the exact patch and cell boundaries, avoiding sample-grid aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, SurfaceRect, union_area
from conrad.schemas.structural_sensor import StructuralSensorModel, StructuralSensorModelV2
from conrad.schemas.structural_support import CapsuleSurfaceSupport

TRUTH_VERSION = "twin2t-spatial-v1"
EVOLUTION_VERSION = "twin2t-spatial-evolution-v1"


@dataclass(frozen=True)
class LocalStructuralState:
    corrosion_depth_m: float = 0.0
    crack_length_m: float = 0.0
    crack_depth_m: float = 0.0

    def __post_init__(self) -> None:
        if min(self.corrosion_depth_m, self.crack_length_m, self.crack_depth_m) < 0:
            raise ValueError("negative structural defect")


@dataclass(frozen=True)
class LocalEvolutionRate:
    """Optional synthetic local rate; empirical degradation is not implied."""

    corrosion_m_per_s: float = 0.0
    crack_m_per_s: float = 0.0
    crack_depth_m_per_s: float = 0.0

    def __post_init__(self) -> None:
        if min(self.corrosion_m_per_s, self.crack_m_per_s, self.crack_depth_m_per_s) < 0:
            raise ValueError("negative local degradation rate")

    def apply(self, state: LocalStructuralState, duration_s: float) -> LocalStructuralState:
        return LocalStructuralState(
            state.corrosion_depth_m + duration_s * self.corrosion_m_per_s,
            state.crack_length_m + duration_s * self.crack_m_per_s,
            state.crack_depth_m + duration_s * self.crack_depth_m_per_s,
        )


@dataclass(frozen=True)
class SpatialEvolutionV1:
    base_rates: tuple[LocalEvolutionRate, ...]
    patch_rates: tuple[LocalEvolutionRate, ...] = ()
    version: str = EVOLUTION_VERSION


@dataclass(frozen=True)
class TruthPatch:
    rect: SurfaceRect
    state: LocalStructuralState


@dataclass(frozen=True)
class SpatialStructuralTruth:
    grid: CapsuleSurfaceGrid
    base: tuple[LocalStructuralState, ...]
    patches: tuple[TruthPatch, ...] = ()

    def __post_init__(self) -> None:
        if len(self.base) != self.grid.n_cells:
            raise ValueError("one base state per surface cell required")
        domain = SurfaceRect(0.0, self.grid.length_m, 0.0, 6.283185307179586)
        for p in self.patches:
            if p.rect.area <= 0 or not domain.contains(p.rect):
                raise ValueError("patch outside surface or empty")
        for i, p in enumerate(self.patches):
            if any(p.rect.intersection(q.rect).area > 0 for q in self.patches[i + 1 :]):
                raise ValueError("overlapping truth patches are ambiguous")

    def state_at(self, x: float, angle: float) -> LocalStructuralState:
        for patch in self.patches:
            r = patch.rect
            if r.x0 <= x < r.x1 and r.a0 <= angle < r.a1:
                return patch.state
        i = min(int(x / self.grid.length_m * self.grid.axial_cells), self.grid.axial_cells - 1)
        j = min(int(angle / 6.283185307179586 * self.grid.sectors), self.grid.sectors - 1)
        return self.base[i * self.grid.sectors + j]

    def worst_local(self) -> LocalStructuralState:
        states = [p.state for p in self.patches]
        for i, base in enumerate(self.base):
            cell = self.grid.cell(i)
            covered = union_area([cell.intersection(p.rect) for p in self.patches])
            if covered < cell.area - 1e-12:
                states.append(base)
        return LocalStructuralState(
            max(s.corrosion_depth_m for s in states),
            max(s.crack_length_m for s in states),
            max(s.crack_depth_m for s in states),
        )

    def evolve(self, duration_s: float, rates: SpatialEvolutionV1) -> SpatialStructuralTruth:
        """Advance local values independently, retaining the spatial partition."""
        if duration_s < 0 or rates.version != EVOLUTION_VERSION:
            raise ValueError("invalid spatial evolution interval or version")
        if len(rates.base_rates) != len(self.base) or len(rates.patch_rates) != len(self.patches):
            raise ValueError("one evolution rate per base cell and patch required")
        return SpatialStructuralTruth(
            self.grid,
            tuple(
                rate.apply(state, duration_s) for rate, state in zip(rates.base_rates, self.base, strict=True)
            ),
            tuple(
                TruthPatch(patch.rect, rate.apply(patch.state, duration_s))
                for rate, patch in zip(rates.patch_rates, self.patches, strict=True)
            ),
        )

    def measure(self, support: CapsuleSurfaceSupport, sensor: StructuralSensorModel) -> LocalStructuralState:
        """Area mean of only truth within the declared measured support, before noise."""
        if (
            support.sensor_model_version != sensor.version
            or support.sensor_config_digest != sensor.digest
            or support.aggregation_kernel != sensor.aggregation_kernel
        ):
            raise ValueError("sensor configuration mismatch")
        r = self.grid.support_rect(support)
        if (
            r.x1 - r.x0 > sensor.footprint_width_m + 1e-9
            or self.grid.radius_m * (r.a1 - r.a0) > sensor.footprint_height_m + 1e-9
        ):
            raise ValueError("measured support exceeds sensor footprint")
        if isinstance(sensor, StructuralSensorModelV2) and (
            r.x1 - r.x0 > sensor.axial_resolution_m + 1e-9
            or self.grid.radius_m * (r.a1 - r.a0) > sensor.lateral_resolution_m + 1e-9
        ):
            raise ValueError("resolution-cell support exceeds declared resolution")
        xs = {r.x0, r.x1}
        angles = {r.a0, r.a1}
        for i in range(self.grid.axial_cells + 1):
            x = self.grid.length_m * i / self.grid.axial_cells
            if r.x0 < x < r.x1:
                xs.add(x)
        for j in range(self.grid.sectors + 1):
            a = 6.283185307179586 * j / self.grid.sectors
            if r.a0 < a < r.a1:
                angles.add(a)
        for p in self.patches:
            for x in (p.rect.x0, p.rect.x1):
                if r.x0 < x < r.x1:
                    xs.add(x)
            for a in (p.rect.a0, p.rect.a1):
                if r.a0 < a < r.a1:
                    angles.add(a)
        sx, sa = sorted(xs), sorted(angles)
        corrosion = crack = crack_depth = 0.0
        max_corrosion = max_crack = max_crack_depth = 0.0
        for x0, x1 in pairwise(sx):
            for a0, a1 in pairwise(sa):
                state = self.state_at((x0 + x1) / 2, (a0 + a1) / 2)
                area = (x1 - x0) * (a1 - a0)
                corrosion += area * state.corrosion_depth_m
                crack += area * state.crack_length_m
                crack_depth += area * state.crack_depth_m
                if sensor.aggregation_kernel in ("LOCAL_MAX", "RESOLUTION_CELL_SAMPLES"):
                    # The optional idealized maximum-response model detects a patch only when
                    # its smaller physical dimension meets the declared resolution limit.
                    patch = next(
                        (
                            p
                            for p in self.patches
                            if p.rect.contains(
                                SurfaceRect((x0 + x1) / 2, (x0 + x1) / 2, (a0 + a1) / 2, (a0 + a1) / 2)
                            )
                        ),
                        None,
                    )
                    if patch is None:
                        max_corrosion = max(max_corrosion, state.corrosion_depth_m)
                        max_crack = max(max_crack, state.crack_length_m)
                        max_crack_depth = max(max_crack_depth, state.crack_depth_m)
                    else:
                        overlap = patch.rect.intersection(r)
                        sized_rect = patch.rect if isinstance(sensor, StructuralSensorModelV2) else overlap
                        size = min(
                            sized_rect.x1 - sized_rect.x0,
                            self.grid.radius_m * (sized_rect.a1 - sized_rect.a0),
                        )
                        if overlap.area > 0 and size >= sensor.minimum_resolvable_corrosion_m:
                            max_corrosion = max(max_corrosion, state.corrosion_depth_m)
                        if overlap.area > 0 and size >= sensor.minimum_resolvable_crack_m:
                            max_crack = max(max_crack, state.crack_length_m)
                            max_crack_depth = max(max_crack_depth, state.crack_depth_m)
        if sensor.aggregation_kernel in ("LOCAL_MAX", "RESOLUTION_CELL_SAMPLES"):
            return LocalStructuralState(max_corrosion, max_crack, max_crack_depth)
        return LocalStructuralState(corrosion / r.area, crack / r.area, crack_depth / r.area)


def spatial_truth_record(truth: SpatialStructuralTruth, rates: SpatialEvolutionV1) -> dict[str, Any]:
    """Versioned truth-plane serialization for mission construction and replay."""
    if len(rates.base_rates) != len(truth.base) or len(rates.patch_rates) != len(truth.patches):
        raise ValueError("one evolution rate per base cell and patch required")
    return {
        "truth_version": TRUTH_VERSION,
        "evolution_version": rates.version,
        "grid": {
            "length_m": truth.grid.length_m,
            "radius_m": truth.grid.radius_m,
            "axial_cells": truth.grid.axial_cells,
            "sectors": truth.grid.sectors,
        },
        "base": [vars(state) for state in truth.base],
        "patches": [{"rect": vars(patch.rect), "state": vars(patch.state)} for patch in truth.patches],
        "base_rates": [vars(rate) for rate in rates.base_rates],
        "patch_rates": [vars(rate) for rate in rates.patch_rates],
    }


def spatial_truth_from_record(raw: dict[str, Any]) -> tuple[SpatialStructuralTruth, SpatialEvolutionV1]:
    if raw.get("truth_version") != TRUTH_VERSION or raw.get("evolution_version") != EVOLUTION_VERSION:
        raise ValueError("incompatible spatial Twin2T truth/evolution version")
    grid = CapsuleSurfaceGrid(**raw["grid"])
    truth = SpatialStructuralTruth(
        grid,
        tuple(LocalStructuralState(**state) for state in raw["base"]),
        tuple(
            TruthPatch(SurfaceRect(**patch["rect"]), LocalStructuralState(**patch["state"]))
            for patch in raw["patches"]
        ),
    )
    rates = SpatialEvolutionV1(
        tuple(LocalEvolutionRate(**rate) for rate in raw["base_rates"]),
        tuple(LocalEvolutionRate(**rate) for rate in raw["patch_rates"]),
    )
    if len(rates.base_rates) != len(truth.base) or len(rates.patch_rates) != len(truth.patches):
        raise ValueError("spatial truth/evolution length mismatch")
    return truth, rates
