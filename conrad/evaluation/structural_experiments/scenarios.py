"""Scenario factories for the 2T experiments (each factory has its own seeded IdFactory)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Scenario, ScenarioEvent
from conrad.twins.twin2t import (
    build_small_scenario,
    coverage_visibility,
    generate_scenario,
    populate_structural_state,
)


def make_scenario(seed: int, spec: Mapping[str, Any]) -> Scenario:
    ids = IdFactory(seed).child("scenario")
    kind = spec.get("kind", "tier")
    if kind == "small":
        return build_small_scenario(seed, n_segments=int(spec.get("n_segments", 4)), ids=ids)
    if kind == "tier":
        return generate_scenario(np.random.default_rng([seed, 3]), int(spec.get("tier", 3)), ids, seed)
    raise ValueError(f"unknown scenario kind {kind!r}")


def unpopulated_small(seed: int, n_segments: int) -> Scenario:
    return build_small_scenario(
        seed, n_segments=n_segments, ids=IdFactory(seed).child("scenario"), populate=False
    )


def populate(scenario: Scenario, seed: int, values: Mapping[UUID, Mapping[str, float]]) -> Scenario:
    """Fill unset design/prior fields after pinning chosen initial states (scenario-level injection)."""
    data = dict(scenario)
    section = dict(scenario.structural_state)
    ents = {k: dict(v) for k, v in section.get("entities", {}).items()}
    for eid, vals in values.items():
        ents.setdefault(str(eid), {}).update(vals)
    section["entities"] = ents
    data["structural_state"] = section
    return populate_structural_state(Scenario(**data), np.random.default_rng([seed, 11]))


def with_events(scenario: Scenario, events: Sequence[ScenarioEvent]) -> Scenario:
    data = dict(scenario)
    data["events"] = tuple(sorted((*scenario.events, *events), key=lambda e: e.time_s))
    return Scenario(**data)


def coverage_schedule(ids: Sequence[UUID], fraction: float, seed: int) -> Any:
    rng = np.random.default_rng([seed, 5])

    def visible(_: int) -> list[UUID]:
        keys = coverage_visibility(list(ids), fraction, rng)
        return [UUID(k.split(":", 1)[1]) for k in keys]

    return visible


def relations(scenario: Scenario) -> list[tuple[UUID, UUID, str]]:
    return [
        (UUID(r["source"]), UUID(r["target"]), str(r["type"]))
        for r in scenario.structural_state.get("relationships", [])
    ]


def entity_types(scenario: Scenario) -> dict[UUID, str]:
    struct = scenario.structural_state.get("entities", {})
    return {
        e.id: str(struct.get(str(e.id), {}).get("component_type", e.entity_type))
        for e in scenario.world_entities
    }
