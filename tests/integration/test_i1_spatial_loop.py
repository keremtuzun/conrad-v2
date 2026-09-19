"""Gate I1 spatial loop: the robot moves through the Twin2S world and Model2S builds a persistent map."""

from __future__ import annotations

import numpy as np
import pytest

from conrad.schemas.belief import KnowledgeStatus, UpdateKind
from conrad.schemas.world import Domain
from tests.acceptance._runs import short_session


@pytest.fixture(scope="module")
def session():
    s = short_session("I1-SPATIAL", 24.0)
    yield s
    s.finish()


def test_map_keeps_unknown_space_next_to_observed_space(session):
    m2s = session.runtime.m2s
    lo, hi = np.array(session.world.t2s.world.bounds_min), np.array(session.world.t2s.world.bounds_max)
    axes = [np.linspace(lo[i], hi[i], 14) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    status = {s.value for s in m2s.occupancy_status(pts)}
    assert "OBSERVED" in status and "UNKNOWN" in status


def test_free_space_claims_agree_with_truth_where_claimed(session):
    lane = np.asarray(session.world.context.transit_lane)
    rng = np.random.default_rng(3)
    pts = np.concatenate([p + rng.normal(0, 1.0, (200, 3)) for p in lane])
    free = session.runtime.m2s.is_free(pts)
    assert free.any()
    wrong = session.world.t2s.occupancy_truth_at(pts[free]).mean()  # truth only through the twin oracle
    assert wrong < 0.05


def test_spatial_beliefs_persist_with_direct_provenance(session):
    revs = [r for r in session.repo.all_revisions() if r.cell.domain is Domain.SPATIAL]
    assert any(r.update_kind is UpdateKind.DIRECT and r.consumed_evidence_ids for r in revs)
    assert any(r.revision > 0 for r in revs)  # the map is revised, not re-created
    assert all(
        r.cell.spatial_support is not None and r.cell.spatial_support.frame_id == "WORLD" for r in revs
    )


def test_spatial_beliefs_carry_only_registry_identities(session):
    registry = {c.registry_id for c in session.world.context.design}
    world_ids = {e.id for e in session.world.scenario.world_entities}
    cells = [r.cell for r in session.repo.all_revisions() if r.cell.domain is Domain.SPATIAL]
    ids = {c.registry_entity_id for c in cells if c.registry_entity_id is not None}
    assert ids and ids <= registry and not ids & world_ids
    assert {c.knowledge_status for c in cells} & {KnowledgeStatus.OBSERVED, KnowledgeStatus.MIXED}


def test_spatial_gate_runs_without_the_ecological_payload(session):
    assert session.runtime.m2e is None
    assert "ecological_survey" not in session.world.hardware.capabilities().environmental_sensors
