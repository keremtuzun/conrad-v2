"""Gate I3 structural inspection: partial views, geometric association, persistent Model2T beliefs."""

from __future__ import annotations

import json

import pytest

from conrad.schemas.belief import UpdateKind
from conrad.schemas.world import Domain
from tests.acceptance._runs import short_session


@pytest.fixture(scope="module")
def session():
    s = short_session("I3-STRUCTURAL", 30.0)
    yield s
    s.finish()


def test_lane_views_associate_structural_observations_correctly(session):
    labels = {x["observation_id"]: x["world_id"] for x in session.world.recorder.series["structural_label"]}
    to_world = {str(r): str(w) for r, w in session.world.mapping.to_world.items()}
    matched = [a for a in session.runtime.perception.structural_log if a["registry_id"] is not None]
    assert matched, "no structural observation was associated"
    wrong = [a for a in matched if to_world[a["registry_id"]] != labels[a["observation_id"]]]
    assert not wrong, f"{len(wrong)} of {len(matched)} associations wrong (ambiguous cases must be NO_MATCH)"


def test_observed_components_get_persistent_direct_beliefs(session):
    revs = [r for r in session.repo.all_revisions() if r.cell.domain is Domain.TECHNICAL]
    direct = [r for r in revs if r.update_kind is UpdateKind.DIRECT]
    assert direct and all(r.consumed_evidence_ids for r in direct)
    assert any(r.cell.claim("corrosion_depth_m").status.value == "OBSERVED" for r in direct)


def test_hidden_target_stays_unknown_from_the_lane(session):
    target = session.world.mapping.to_registry[session.world.target]
    msg = next(m for m in session.runtime.m2t.export_beliefs() if m.world_entity_id == target)
    cond = next(c for c in msg.state_summary if c.name == "condition")
    assert cond.status.value == "UNKNOWN" and msg.uncertainty.observational >= 0.9
    patch = session.world.recorder.series["patch_visibility"]
    assert max(p["visible_fraction"] for p in patch) == 0.0  # truth oracle: the far side is never seen


def test_registry_carries_design_data_only(session):
    comps = session.world.context.asset_registry["components"]
    allowed = {"registry_id", "parent_id", "component_type", "material", "coating", "wall_thickness_m"}
    assert all(set(c) <= allowed for c in comps)
    assert "initial_corrosion_depth_m" not in json.dumps(comps, default=str)
