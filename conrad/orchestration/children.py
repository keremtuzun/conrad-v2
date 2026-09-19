"""Construction of the Model 2 children from the mission context and runtime config. DEPLOYMENT PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from conrad.domains.ecological import EcologicalEncoder, Model2E
from conrad.domains.ecological.config import Model2EConfig
from conrad.domains.spatial.config import spatial_config
from conrad.domains.spatial.model import Model2S
from conrad.domains.technical import Model2T, PropagationMode, model2t_config_from_dict
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.schemas.ids import IdFactory
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
    m2t = Model2T(
        ids.child("model2t"), m2t_cfg, repository=repo, run_id=run_id, mode=PropagationMode(cfg.model2t_mode)
    )
    m2t.initialize(
        {"asset_registry": ctx.asset_registry, "timestamp": TimeStamp(time_ns=now_ns, clock_domain=clock)}
    )
    if not ecological:
        return Children(m2s, m2t, None, None)
    e_cfg = Model2EConfig.model_validate({**eco_grid(ctx), **cfg.model2e})
    m2e = Model2E(e_cfg, ids.child("model2e"), repo)
    m2e.initialize({"asset_registry": ctx.model2e_registry(), "clock_domain": clock, "run_id": run_id})
    return Children(m2s, m2t, m2e, EcologicalEncoder(e_cfg, ids.child("ecmer_e")))
