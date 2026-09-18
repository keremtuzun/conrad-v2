"""Procedural scenario generator, difficulty curriculum and sensor-degradation curriculum (ch11).

Scenario ~ P(Theta): topology, component counts, materials, initial degradation, environment, loading
and event schedules are randomized inside the logged prior ranges (physically plausible by construction;
no "physics randomization nonsense").

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.world import DomainOwnership, Scenario, ScenarioEvent, WorldEntity
from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.scenario import populate_structural_state

COVERAGE_LEVELS: tuple[float, ...] = (1.0, 0.5, 0.2, 0.05)
"""ch10/ch11 observation coverage levels."""


@dataclass(frozen=True)
class TierSpec:
    tier: int
    n_pipelines: tuple[int, int]
    n_segments: tuple[int, int]
    fatigue_active: bool
    coverage: float
    sensor_degradation: float
    contradiction: bool
    events: bool
    maintenance: bool
    coupled: bool
    ood_materials: bool


TIERS: dict[int, TierSpec] = {
    1: TierSpec(1, (1, 1), (1, 1), False, 1.0, 0.0, False, False, False, False, False),
    2: TierSpec(2, (1, 1), (3, 5), False, 0.5, 0.1, False, False, False, False, False),
    3: TierSpec(3, (1, 2), (3, 6), True, 0.5, 0.2, False, False, False, False, False),
    4: TierSpec(4, (1, 2), (4, 8), True, 0.2, 0.5, True, False, False, False, False),
    5: TierSpec(5, (2, 3), (4, 8), True, 0.05, 0.8, True, True, True, True, True),
}


def tier_mcde_config(tier: int, base: MCDEConfig | None = None) -> MCDEConfig:
    spec = TIERS[tier]
    base = base or MCDEConfig()
    return replace(
        base,
        mechanism_coupling=spec.coupled,
        topology_coupling=spec.coupled,
        events_enabled=spec.events,
        interventions_enabled=spec.maintenance,
    )


def generate_scenario(rng: np.random.Generator, tier: int, ids: IdFactory, seed: int) -> Scenario:
    """Random hierarchical asset (pipelines, segments, welds, joints, surface regions, supports, aux)."""
    spec = TIERS[tier]
    t0 = stamp(0.0, "sim")
    tech = DomainOwnership(spatial=True, technical=True)
    ents: list[WorldEntity] = []
    struct: dict[str, dict[str, Any]] = {}
    rels: list[dict[str, str]] = []

    def add(etype: str, parent: UUID | None, **attrs: Any) -> UUID:
        eid = ids.new()
        ents.append(
            WorldEntity(
                id=eid,
                entity_type=etype,
                parent_id=parent,
                reference_frame="WORLD",
                created_at=t0,
                domain_ownership=tech,
            )
        )
        struct[str(eid)] = {"component_type": etype, **attrs}
        if parent is not None:
            rels.append({"source": str(eid), "target": str(parent), "type": "PART_OF"})
        return eid

    def rel(a: UUID, b: UUID, t: str) -> None:
        rels.append({"source": str(a), "target": str(b), "type": t})

    no_crack = {} if spec.fatigue_active else {"initial_crack_length_m": 0.0}
    asset = add("ASSET", None)
    segments: list[UUID] = []
    if tier == 1:
        segments.append(add("SEGMENT", asset, material="carbon_steel", **no_crack))
    else:
        for _ in range(int(rng.integers(spec.n_pipelines[0], spec.n_pipelines[1] + 1))):
            pipe = add("PIPELINE", asset)
            n_seg = int(rng.integers(spec.n_segments[0], spec.n_segments[1] + 1))
            segs = [add("SEGMENT", pipe, **no_crack) for _ in range(n_seg)]
            for i, seg in enumerate(segs):
                if rng.random() < 0.5:
                    add("SURFACE_REGION", seg)
                rel(add("WELD", seg, **no_crack), seg, "ATTACHED_TO")
                if i > 0:
                    rel(segs[i - 1], seg, "CONNECTED_TO")
                    rel(add("JOINT", segs[i - 1], **no_crack), seg, "CONNECTED_TO")
                if rng.random() < 0.4:
                    mat = "concrete" if rng.random() < 0.4 else "carbon_steel"
                    sup = add(
                        "SUPPORT",
                        pipe,
                        material=mat,
                        **({"coating": "none"} if mat == "concrete" else {}),
                        **no_crack,
                    )
                    rel(seg, sup, "SUPPORTED_BY")
                elif i > 0 and rng.random() < 0.3:
                    rel(segs[i - 1], seg, "ADJACENT_TO")
            segments.extend(segs)
        if rng.random() < 0.5:
            add("AUXILIARY_COMPONENT", asset)
    if spec.ood_materials:
        for key in list(struct)[: max(1, len(struct) // 5)]:
            if struct[key]["component_type"] in ("SEGMENT", "WELD", "JOINT"):
                struct[key]["material"] = "stainless_steel_316"
    events: list[ScenarioEvent] = []
    if spec.events:
        yr = 365.25 * 86400.0
        for _ in range(int(rng.integers(1, 4))):
            target = segments[int(rng.integers(len(segments)))]
            etype = str(rng.choice(["IMPACT", "LOAD_SPIKE", "COATING_FAILURE"]))
            events.append(
                ScenarioEvent(
                    event_id=ids.new(),
                    time_s=float(rng.uniform(0, 3 * yr)),
                    event_type=etype,
                    target_entity_id=target,
                )
            )
        if spec.maintenance:
            target = segments[int(rng.integers(len(segments)))]
            events.append(
                ScenarioEvent(
                    event_id=ids.new(),
                    time_s=float(rng.uniform(2 * yr, 5 * yr)),
                    event_type=str(rng.choice(["REPAIR", "REPLACEMENT", "COATING_RENEWAL"])),
                    target_entity_id=target,
                )
            )
    scenario = Scenario(
        scenario_id=ids.new(),
        scenario_version=f"twin2t-procedural-tier{tier}",
        seed=seed,
        world_entities=tuple(ents),
        structural_state={"entities": struct, "relationships": rels},
        events=tuple(sorted(events, key=lambda e: e.time_s)),
        metadata={
            "generator": "twin2t.generate_scenario",
            "difficulty_tier": tier,
            "world_family": f"tier{tier}",
        },
    )
    return populate_structural_state(scenario, rng)


@dataclass(frozen=True)
class SensorDegradationCurriculum:
    """O' = Corrupt(O; theta). ``level`` in [0, 1] scales every corruption variable."""

    max_bias_m: float = 5.0e-4
    max_pose_error_m: float = 0.5
    max_temporal_gap_s: float = 30 * 86400.0
    missing_modality_prob: float = 0.2

    def degradation(self, level: float, rng: np.random.Generator) -> dict[str, float]:
        if not 0.0 <= level <= 1.0:
            raise ValueError("curriculum level must be in [0, 1]")
        return {
            "corruption": level,
            "blur": level,
            "turbidity": 0.6 * level,
            "biofouling_cover": 0.5 * level,
            "occlusion": 0.3 * level,
            "sensor_bias_m": self.max_bias_m * level * float(rng.choice([-1.0, 1.0])),
            "pose_error_m": self.max_pose_error_m * level,
            "temporal_gap_s": self.max_temporal_gap_s * level,
            "missing_modality": 1.0 if rng.random() < self.missing_modality_prob * level else 0.0,
        }


def contradiction_pair(
    base: dict[str, float], severity: float = 1.0
) -> tuple[dict[str, float], dict[str, float]]:
    """Controlled contradiction: surface channel biased toward severe vs. an unbiased reading."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    return {**base, "contradiction": severity}, {**base, "contradiction": 0.0}
