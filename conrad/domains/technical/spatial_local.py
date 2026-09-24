"""Opt-in local Model2T working state for structural-observation-v2 evidence.

An area mean across multiple cells cannot identify which cell is degraded, so
it never updates a local estimate or supplies healthy coverage.  Legacy
Model2T messages remain on their historical path during this development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, SurfaceRect, union_area
from conrad.schemas.observation import Evidence, EvidenceValidity
from conrad.schemas.structural_sensor import StructuralSensorModel, StructuralSensorModelV2

MODEL_VERSION = "model2t-local-v1"


class LocalCondition(str, Enum):
    UNKNOWN = "UNKNOWN"
    OBSERVED_INTACT = "OBSERVED_INTACT"
    DEGRADED = "DEGRADED"
    SEVERE = "SEVERE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class LocalThresholds:
    corrosion_degraded_m: float
    corrosion_severe_m: float
    corrosion_failed_m: float
    crack_degraded_m: float
    crack_severe_m: float
    crack_failed_m: float
    # Synthetic crack-depth bands; ENGINEERING_ESTIMATE until physical calibration.
    crack_depth_degraded_m: float = 0.002
    crack_depth_severe_m: float = 0.005
    crack_depth_failed_m: float = 0.010

    def __post_init__(self) -> None:
        for prefix in ("corrosion", "crack", "crack_depth"):
            a, b, c = (getattr(self, f"{prefix}_{level}_m") for level in ("degraded", "severe", "failed"))
            if not 0 < a < b < c:
                raise ValueError(f"invalid {prefix} thresholds")

    def condition(self, corrosion: float, crack: float, crack_depth: float = 0.0) -> LocalCondition:
        for level, result in (
            ("failed", LocalCondition.FAILED),
            ("severe", LocalCondition.SEVERE),
            ("degraded", LocalCondition.DEGRADED),
        ):
            if (
                corrosion >= getattr(self, f"corrosion_{level}_m")
                or crack >= getattr(self, f"crack_{level}_m")
                or crack_depth >= getattr(self, f"crack_depth_{level}_m")
            ):
                return result
        return LocalCondition.OBSERVED_INTACT


@dataclass
class LocalCellBelief:
    corrosion_upper_m: float | None = None
    crack_upper_m: float | None = None
    crack_depth_upper_m: float | None = None
    supports: list[SurfaceRect] = field(default_factory=list)
    evidence_ids: list[UUID] = field(default_factory=list)
    independent_groups: set[str] = field(default_factory=set)
    last_time_ns: int | None = None


class SpatialModel2T:
    """Local evidence ledger, keyed by surveyed design-surface cell index."""

    def __init__(
        self,
        grid: CapsuleSurfaceGrid,
        sensor: StructuralSensorModel,
        thresholds: LocalThresholds,
        registry_id: UUID,
        *,
        required_looks: int = 1,
        require_depth: bool = False,
    ) -> None:
        if required_looks < 1:
            raise ValueError("required_looks must be positive")
        self.grid, self.sensor, self.thresholds = grid, sensor, thresholds
        self.registry_id = registry_id
        self.required_looks = required_looks
        self.require_depth = require_depth
        self.cells = [LocalCellBelief() for _ in range(grid.n_cells)]
        self.seen_evidence: set[UUID] = set()
        self.unresolved: list[UUID] = []

    def ingest(self, evidence: Evidence) -> bool:
        if evidence.evidence_id in self.seen_evidence:
            return False
        self.seen_evidence.add(evidence.evidence_id)
        sup = evidence.structural_support
        if sup is None:
            self.unresolved.append(evidence.evidence_id)
            return False
        if (
            sup.support_version != "structural-observation-v2"
            or sup.sensor_model_version != self.sensor.version
            or sup.sensor_config_digest != self.sensor.digest
            or sup.aggregation_kernel != self.sensor.aggregation_kernel
        ):
            raise ValueError("structural architecture mismatch")
        if evidence.validity is EvidenceValidity.INVALID:
            return False
        if not any(c.registry_entity_id == self.registry_id for c in evidence.entity_candidates):
            self.unresolved.append(evidence.evidence_id)
            return False
        measured = self.grid.support_rect(sup)
        if (
            measured.x1 - measured.x0 > self.sensor.footprint_width_m + 1e-9
            or self.grid.radius_m * (measured.a1 - measured.a0) > self.sensor.footprint_height_m + 1e-9
        ):
            raise ValueError("measured support exceeds sensor footprint")
        if isinstance(self.sensor, StructuralSensorModelV2) and (
            measured.x1 - measured.x0 > self.sensor.axial_resolution_m + 1e-9
            or self.grid.radius_m * (measured.a1 - measured.a0) > self.sensor.lateral_resolution_m + 1e-9
        ):
            raise ValueError("resolution-cell support exceeds declared resolution")
        # UNKNOWN uncertainty is never silently zero.  Known edge uncertainty
        # erodes the guaranteed support and must also stay within one cell.
        if sup.axial_uncertainty_m is None or sup.angular_uncertainty_rad is None:
            self.unresolved.append(evidence.evidence_id)
            return False
        dx, da = sup.axial_uncertainty_m, sup.angular_uncertainty_rad
        sure = SurfaceRect(measured.x0 + dx, measured.x1 - dx, measured.a0 + da, measured.a1 - da)
        if sure.area <= 0:
            self.unresolved.append(evidence.evidence_id)
            return False
        matching = [i for i in range(self.grid.n_cells) if self.grid.cell(i).contains(measured)]
        if len(matching) != 1:
            self.unresolved.append(evidence.evidence_id)
            return False
        if (
            evidence.measurement_units.get("apparent_wall_loss") != "m"
            or evidence.measurement_units.get("crack_indication_length") != "m"
        ):
            self.unresolved.append(evidence.evidence_id)
            return False
        corrosion = evidence.measurements.get("apparent_wall_loss")
        crack = evidence.measurements.get("crack_indication_length")
        if corrosion is None or crack is None or corrosion < 0 or crack < 0:
            self.unresolved.append(evidence.evidence_id)
            return False
        cell = self.cells[matching[0]]
        depth = evidence.measurements.get("crack_indication_depth")
        if self.require_depth and (
            evidence.measurement_units.get("crack_indication_depth") != "m" or depth is None or depth < 0
        ):
            self.unresolved.append(evidence.evidence_id)
            return False
        uncertainty = 3 * self.sensor.noise_sigma_m * (1 + evidence.aleatoric_uncertainty)
        cell.corrosion_upper_m = max(cell.corrosion_upper_m or 0, corrosion + uncertainty)
        cell.crack_upper_m = max(cell.crack_upper_m or 0, crack + uncertainty)
        if depth is not None and depth >= 0:
            cell.crack_depth_upper_m = max(cell.crack_depth_upper_m or 0, depth + uncertainty)
        cell.supports.append(sure)
        cell.evidence_ids.append(evidence.evidence_id)
        cell.independent_groups.add(evidence.independence_group or str(evidence.source_observation_id))
        cell.last_time_ns = evidence.timestamp.time_ns
        return True

    def coverage_fraction(self, index: int) -> float:
        cell = self.grid.cell(index)
        clipped = [cell.intersection(r) for r in self.cells[index].supports]
        return min(1.0, union_area(clipped) / cell.area)

    def cell_condition(self, index: int) -> LocalCondition:
        cell = self.cells[index]
        if cell.corrosion_upper_m is None or cell.crack_upper_m is None:
            return LocalCondition.UNKNOWN
        state = self.thresholds.condition(
            cell.corrosion_upper_m, cell.crack_upper_m, cell.crack_depth_upper_m or 0.0
        )
        if state is not LocalCondition.OBSERVED_INTACT:
            return state
        if self.sensor.aggregation_kernel not in ("LOCAL_MAX", "RESOLUTION_CELL_SAMPLES"):
            return LocalCondition.UNKNOWN
        cell_width = min(
            self.grid.length_m / self.grid.axial_cells,
            self.grid.radius_m * 6.283185307179586 / self.grid.sectors,
        )
        if (
            self.sensor.minimum_resolvable_corrosion_m > cell_width
            or self.sensor.minimum_resolvable_crack_m > cell_width
        ):
            return LocalCondition.UNKNOWN
        if self.coverage_fraction(index) < 1 - 1e-9 or len(cell.independent_groups) < self.required_looks:
            return LocalCondition.UNKNOWN
        if self.require_depth and cell.crack_depth_upper_m is None:
            return LocalCondition.UNKNOWN
        return LocalCondition.OBSERVED_INTACT

    def condition(self) -> LocalCondition:
        worst = LocalCondition.OBSERVED_INTACT
        order = list(LocalCondition)
        for i in range(self.grid.n_cells):
            state = self.cell_condition(i)
            if state not in (LocalCondition.UNKNOWN, LocalCondition.OBSERVED_INTACT) and order.index(
                state
            ) > order.index(worst):
                worst = state
        if worst is not LocalCondition.OBSERVED_INTACT:
            return worst
        # An AREA_MEAN observation cannot rule out a local maximum.  The
        # idealized LOCAL_MAX response remains SYNTHETIC_ONLY until calibrated.
        if self.sensor.aggregation_kernel not in ("LOCAL_MAX", "RESOLUTION_CELL_SAMPLES"):
            return LocalCondition.UNKNOWN
        if all(self.cell_condition(i) is LocalCondition.OBSERVED_INTACT for i in range(self.grid.n_cells)):
            return LocalCondition.OBSERVED_INTACT
        return LocalCondition.UNKNOWN
