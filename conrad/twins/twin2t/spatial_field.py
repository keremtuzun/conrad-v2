"""Spatially heterogeneous capsule-surface truth for structural-observation-v2.

This is truth-plane code.  Neither the local belief model nor planning imports it.
Patches are explicit and may be smaller than a sensor footprint.  The integral
uses the exact patch and cell boundaries, avoiding sample-grid aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, SurfaceRect, union_area
from conrad.schemas.structural_sensor import StructuralSensorModel
from conrad.schemas.structural_support import CapsuleSurfaceSupport

TRUTH_VERSION = "twin2t-spatial-v1"


@dataclass(frozen=True)
class LocalStructuralState:
    corrosion_depth_m: float = 0.0
    crack_length_m: float = 0.0

    def __post_init__(self) -> None:
        if self.corrosion_depth_m < 0 or self.crack_length_m < 0:
            raise ValueError("negative structural defect")


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
            max(s.corrosion_depth_m for s in states), max(s.crack_length_m for s in states)
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
        corrosion = crack = 0.0
        max_corrosion = max_crack = 0.0
        for x0, x1 in pairwise(sx):
            for a0, a1 in pairwise(sa):
                state = self.state_at((x0 + x1) / 2, (a0 + a1) / 2)
                area = (x1 - x0) * (a1 - a0)
                corrosion += area * state.corrosion_depth_m
                crack += area * state.crack_length_m
                if sensor.aggregation_kernel == "LOCAL_MAX":
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
                    else:
                        overlap = patch.rect.intersection(r)
                        size = min(overlap.x1 - overlap.x0, self.grid.radius_m * (overlap.a1 - overlap.a0))
                        if size >= sensor.minimum_resolvable_corrosion_m:
                            max_corrosion = max(max_corrosion, state.corrosion_depth_m)
                        if size >= sensor.minimum_resolvable_crack_m:
                            max_crack = max(max_crack, state.crack_length_m)
        if sensor.aggregation_kernel == "LOCAL_MAX":
            return LocalStructuralState(max_corrosion, max_crack)
        return LocalStructuralState(corrosion / r.area, crack / r.area)
