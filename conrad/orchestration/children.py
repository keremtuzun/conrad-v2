"""Construction of the Model 2 children from the mission context and runtime config. DEPLOYMENT PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.ecological import EcologicalEncoder, Model2E
from conrad.domains.ecological.config import Model2EConfig
from conrad.domains.spatial.config import spatial_config
from conrad.domains.spatial.model import Model2S
from conrad.domains.spatial.queries import UnknownPolicy
from conrad.domains.technical import Model2T, model2t_config_from_dict, production_propagation_mode
from conrad.domains.technical.spatial_local import LocalThresholds
from conrad.domains.technical.spatial_mission import SpatialMissionModel2T
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.schemas.capsule_surface import CapsuleSurfaceGrid
from conrad.schemas.ids import IdFactory
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.timebase import TimeStamp


@dataclass
class Children:
    m2s: Model2S
    m2t: Model2T
    m2e: Model2E | None
    encoder_e: EcologicalEncoder | None


def eco_grid(ctx: MissionContext) -> dict[str, Any]:
    """Model2E belief grid spanning the mission boundary (deployment knowledge, not truth)."""
    lo = ctx.spec.boundary_min_m or (-10.0, -10.0, -2.0)
    hi = ctx.spec.boundary_max_m or (10.0, 10.0, 5.0)
    shape = (4, 4, 2)
    spacing = [(hi[i] - lo[i] + 2.0) / shape[i] for i in range(3)]
    return {
        "grid": {
            "origin_m": [lo[0] - 1.0, lo[1] - 1.0, lo[2] - 1.0],
            "spacing_m": spacing,
            "shape": list(shape),
        }
    }


def surface_occlusion(m2s: Model2S) -> Callable[[Sequence[tuple[float, float, float]]], list[bool]]:
    """Model2T's ``surface_occlusion`` hook: True where the Model2S belief map says the water-side probe of a
    surface cell is occupied, so the payload could not have looked at that cell. UNKNOWN space never blocks
    (``UnknownPolicy.PERMISSIVE``): only what the map has actually observed removes coverage credit."""

    def blocked(points: Sequence[tuple[float, float, float]]) -> list[bool]:
        if not points:
            return []
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return [not bool(f) for f in m2s.is_free(arr, UnknownPolicy.PERMISSIVE)]

    return blocked


def build_children(
    ctx: MissionContext,
    cfg: MissionRuntimeConfig,
    ids: IdFactory,
    store: ObjectStore,
    repo: Repository,
    run_id: UUID,
    clock: str,
    now_ns: int,
    ecological: bool,
) -> Children:
    geometric = [s for s in ctx.sensors if s.modality in ("DEPTH_RANGE", "SONAR")]
    m2s = Model2S(ids.child("model2s"), store, spatial_config(cfg.model2s), repository=repo, run_id=run_id)
    m2s.initialize({"sensors": geometric, "asset_registry": ctx.model2s_registry(), "run_id": run_id})
    m2t_cfg = model2t_config_from_dict({**cfg.model2t, "clock_domain": clock})
    # ADR-0009: relational propagation is an EXPERIMENTAL arm; NONE unless model2t.tcdp.experimental_enabled.
    mode = production_propagation_mode(cfg.model2t_mode, m2t_cfg)
    m2t: Model2T
    if cfg.model2t_backend == "spatial_v1":
        spatial = cfg.model2t_spatial
        if spatial is None or len(ctx.critical_component_ids) != 1:
            raise ValueError("spatial_v1 requires explicit settings and one critical component")
        target = ctx.component(ctx.critical_component_ids[0])
        if target.shape != "CAPSULE":
            raise ValueError("spatial_v1 currently requires a surveyed capsule target")
        required = {"axial_cells", "sectors", "sensor", "thresholds", "required_looks"}
        allowed = required | {"required_domain"}
        if not required <= set(spatial) or not set(spatial) <= allowed:
            raise ValueError(f"spatial_v1 settings require {sorted(required)} with optional required_domain")
        domain = spatial.get("required_domain") or {}
        if set(domain) - {"axial_fraction", "sectors"}:
            raise ValueError("invalid spatial required_domain keys")
        length = float(np.linalg.norm(np.asarray(target.p1_m) - np.asarray(target.p0_m)))
        grid = CapsuleSurfaceGrid(
            length, target.radius_m, int(spatial["axial_cells"]), int(spatial["sectors"])
        )
        m2t = SpatialMissionModel2T(
            ids.child("model2t"),
            m2t_cfg,
            repository=repo,
            run_id=run_id,
            mode=mode,
            registry_id=target.registry_id,
            grid=grid,
            sensor=StructuralSensorModelV2.model_validate(spatial["sensor"]),
            thresholds=LocalThresholds(**spatial["thresholds"]),
            required_looks=int(spatial["required_looks"]),
            required_axial_fraction=tuple(domain.get("axial_fraction", (0.0, 1.0))),
            required_sectors=None if "sectors" not in domain else tuple(domain["sectors"]),
        )
    else:
        if cfg.model2t_spatial is not None:
            raise ValueError("spatial Model2T settings supplied to legacy backend")
        m2t = Model2T(ids.child("model2t"), m2t_cfg, repository=repo, run_id=run_id, mode=mode)
    m2t.initialize(
        {
            "asset_registry": ctx.asset_registry,
            # surveyed design surfaces (deployment-plane mission context): Model2T surface coverage
            "design_geometry": [c.model_dump(mode="json") for c in ctx.design],
            # Belief-side occlusion test from the Model2S map (never truth): a surface cell whose water-side
            # probe sits in space Model2S has OBSERVED as occupied could not have been seen, so a reading's
            # footprint never credits it (docs/audits/MODEL2T_REPAIR.md iteration 4).
            "surface_occlusion": surface_occlusion(m2s),
            "timestamp": TimeStamp(time_ns=now_ns, clock_domain=clock),
        }
    )
    if not ecological:
        return Children(m2s, m2t, None, None)
    e_cfg = Model2EConfig.model_validate({**eco_grid(ctx), **cfg.model2e})
    m2e = Model2E(e_cfg, ids.child("model2e"), repo)
    m2e.initialize({"asset_registry": ctx.model2e_registry(), "clock_domain": clock, "run_id": run_id})
    return Children(m2s, m2t, m2e, EcologicalEncoder(e_cfg, ids.child("ecmer_e")))
