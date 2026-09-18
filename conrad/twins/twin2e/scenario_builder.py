"""Standalone small shared-world scenario for Twin2E tests and demos. SYNTHETIC_ONLY.

Positions go into ``spatial_state`` (Twin2S owns geometry); Twin2E reads them as shared context.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.world import DomainOwnership, Scenario, ScenarioEvent, WorldEntity
from conrad.twins.twin2e.config import Twin2EConfig
from conrad.twins.twin2e.priors import populate_ecological_state


def build_small_scenario(
    ids: IdFactory,
    seed: int,
    config: Twin2EConfig | None = None,
    n_structures: int = 2,
    n_benthic: int = 1,
    n_mobile: int = 1,
    environment: dict[str, Any] | None = None,
    events: tuple[tuple[float, str, int | None, dict[str, Any]], ...] = (),
    populate: bool = True,
) -> Scenario:
    """Pipeline segments with biofouling, benthic patches and fish groups inside one habitat region.

    ``events`` items are (time_s, event_type, index_into_non_habitat_entities | None, parameters).
    """
    cfg = config or Twin2EConfig()
    lg = cfg.local_grid
    rng = np.random.default_rng(seed)
    t0 = stamp(0.0, cfg.clock_domain)
    habitat = WorldEntity(
        id=ids.new(),
        entity_type="habitat_region",
        reference_frame=lg.frame_id,
        created_at=t0,
        domain_ownership=DomainOwnership(spatial=True, ecological=True),
    )
    lo = np.asarray(lg.origin_m)
    hi = lo + np.asarray(lg.spacing_m) * np.asarray(lg.shape)
    ents: list[WorldEntity] = []
    spatial: dict[str, Any] = {}
    kinds = ["pipeline_segment"] * n_structures + ["coral_patch"] * n_benthic + ["fish_school"] * n_mobile
    for i, etype in enumerate(kinds):
        e = WorldEntity(
            id=ids.new(),
            entity_type=etype,
            parent_id=habitat.id,
            reference_frame=lg.frame_id,
            created_at=t0,
            domain_ownership=DomainOwnership(
                spatial=True, technical=etype == "pipeline_segment", ecological=True
            ),
            metadata={"name": f"{etype}_{i}"},
        )
        frac = (i + 1) / (len(kinds) + 1)
        x = lo[0] + (hi[0] - lo[0]) * (0.2 + 0.6 * frac)
        y = lo[1] + (hi[1] - lo[1]) * float(rng.uniform(0.3, 0.7))
        z = lo[2] + 0.5 * lg.spacing_m[2] if etype != "fish_school" else lo[2] + 3.0 * lg.spacing_m[2]
        spatial[str(e.id)] = {"position_m": [float(x), float(y), float(z)]}
        ents.append(e)
    scen_events = tuple(
        ScenarioEvent(
            event_id=ids.new(),
            time_s=t,
            event_type=etype,
            target_entity_id=None if idx is None else ents[idx].id,
            parameters=params,
        )
        for (t, etype, idx, params) in events
    )
    scenario = Scenario(
        scenario_id=ids.new(),
        scenario_version="twin2e-small-1",
        seed=seed,
        metadata={"family": "twin2e_small", "data_label": "SYNTHETIC_ONLY"},
        world_entities=(habitat, *ents),
        environment=dict(environment or {}),
        spatial_state={"entities": spatial},
        events=scen_events,
    )
    if not populate:
        return scenario
    return populate_ecological_state(scenario, np.random.default_rng(seed + 7919))
