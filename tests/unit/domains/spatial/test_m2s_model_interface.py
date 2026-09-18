"""Model2Child contract: messages, repository revisions, queries, reset, visibility, context, baselines."""

from __future__ import annotations

import numpy as np
import pytest
from m2s_unit_helpers import depth_image, depth_spec, line, make_obs, new_model, pose

from conrad.domains.spatial.baselines import MaxLikelihoodSurfaceFill, PlainOccupancyGrid
from conrad.domains.spatial.evidence import GeometricEvidenceEncoder
from conrad.domains.spatial.model import Model2S
from conrad.domains.spatial.publisher import RegistryAsset
from conrad.domains.spatial.sensing import SpatialSensingError
from conrad.schemas.belief import (
    Availability,
    BeliefMessage,
    BeliefQuery,
    KnowledgeStatus,
    Lifecycle,
    PropertyClaim,
    UpdateKind,
)
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

NO_REFINE = {"refinement": {"enabled": False}}


def _feed(m, ids, store, spec, t, wall_x=3.1, robot=None):
    robot = robot or pose()
    m.ingest_observations([make_obs(ids, store, spec, robot, depth_image(spec, robot, wall_x), t)])
    return m.update_beliefs(stamp(t, "SIM"))


def test_messages_are_spatial_with_status_level_claims(store):
    ids = IdFactory(21)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    assert m.availability() is Availability.AVAILABLE
    msgs = _feed(m, ids, store, spec, 1.0)
    assert msgs and all(isinstance(x, BeliefMessage) and x.domain is Domain.SPATIAL for x in msgs)
    for x in msgs:
        names = {c.name for c in x.state_summary}
        assert "coverage" in names and x.spatial is not None and x.lifecycle is Lifecycle.ACTIVE
        assert x.world_entity_id is None  # no registry supplied -> no world identity
        for c in x.state_summary:
            assert c.status is KnowledgeStatus.UNKNOWN or c.provenance_id in x.provenance_refs
    assert any(c.name == "occupancy.observed" for x in msgs for c in x.state_summary)
    assert any(c.name == "occupancy.unknown" and c.value is None for x in msgs for c in x.state_summary)


def test_revisions_are_persisted_with_provenance(repo, store):
    ids = IdFactory(22)
    spec = depth_spec(ids)
    run = ids.new()
    m = Model2S(ids, store, repository=repo, run_id=run)
    m.initialize({"sensors": [spec]})
    first = _feed(m, ids, store, spec, 1.0)
    second = _feed(m, ids, store, spec, 2.0)
    heads = {c.belief_id: c for c in repo.heads(run)}
    assert {x.belief_id for x in first} <= set(heads)
    b = second[0].belief_id
    revs = repo.revisions(b)
    assert [r.revision for r in revs] == list(range(len(revs))) and len(revs) >= 2
    assert all(r.update_kind is UpdateKind.DIRECT and r.consumed_evidence_ids for r in revs)
    closure = repo.provenance_closure(revs[-1].provenance_root)
    assert revs[0].provenance_root in closure  # revisions chain their provenance
    ev = repo.contributed_evidence(b)
    assert ev and all(repo.evidence(e) is not None for e in ev)


def test_ingest_evidence_resolves_observation_through_repository(repo, store):
    ids = IdFactory(23)
    spec = depth_spec(ids)
    obs = make_obs(ids, store, spec, pose(), depth_image(spec, pose(), 3.1), 1.0)
    repo.put_observation(obs)
    ev, _ = GeometricEvidenceEncoder(ids, store).encode(obs)
    m = Model2S(ids, store, repository=repo, run_id=ids.new())
    m.initialize({"sensors": [spec]})
    m.ingest([ev, ev])
    assert m.diagnostics["duplicate_evidence"] == 1
    assert m.update_beliefs(stamp(1.0, "SIM"))
    unknown = Model2S(ids, store)
    unknown.initialize({"sensors": []})
    unknown.register_observations([obs])
    unknown.ingest([ev])
    with pytest.raises(SpatialSensingError):
        unknown.update_beliefs(stamp(1.0, "SIM"))


def test_missing_pose_estimate_degrades_availability(store):
    ids = IdFactory(24)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    m.ingest_observations([make_obs(ids, store, spec, None, depth_image(spec, pose(), 3.1), 1.0)])
    assert m.update_beliefs(stamp(1.0, "SIM")) == []
    assert m.availability() is Availability.DEGRADED and len(m.map.base) == 0


def test_query_by_region_entity_and_uncertainty(store):
    ids = IdFactory(25)
    spec = depth_spec(ids)
    asset = RegistryAsset(ids.new(), (3.1, 0.0, 0.0), (0.2, 1.0, 1.0), "wall")
    m = Model2S(ids, store)
    m.initialize({"sensors": [spec], "asset_registry": [asset]})
    msgs = _feed(m, ids, store, spec, 1.0)
    obj = [x for x in msgs if x.world_entity_id == asset.registry_entity_id]
    assert len(obj) == 1
    assert m.query(BeliefQuery(entity_ids=(asset.registry_entity_id,))) == obj
    region = SpatialSupport(frame_id="WORLD", center_m=(3.1, 0.0, 0.0), half_extent_m=(0.1, 0.1, 0.1))
    near = m.query(BeliefQuery(region=region))
    assert near and len(near) < len(m.export_beliefs())
    high_uo = m.query(BeliefQuery(min_observational_uncertainty=0.99))
    assert all(x.uncertainty.observational >= 0.99 for x in high_uo)
    assert m.query(BeliefQuery(domain=Domain.TECHNICAL)) == []


def test_reset_working_memory_keeps_the_persistent_map(store):
    ids = IdFactory(26)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec])
    _feed(m, ids, store, spec, 1.0)
    before = m.export_grid().level(0)
    beliefs = m.export_beliefs()
    assert len(m.export_grid().level(1).indices) > 0
    m.reset_working_memory()
    after = m.export_grid().level(0)
    assert np.array_equal(before.indices, after.indices)
    assert np.array_equal(before.probability, after.probability)
    assert np.array_equal(before.status, after.status)
    assert len(m.export_grid().level(1).indices) == 0
    assert m.export_beliefs() == beliefs
    again = _feed(m, ids, store, spec, 2.0)
    prior = {x.belief_id: x.revision for x in beliefs}
    assert all(x.revision == prior[x.belief_id] + 1 for x in again if x.belief_id in prior)


def test_predicted_visibility_treats_unknown_as_possibly_blocking(store):
    ids = IdFactory(27)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    target = SpatialSupport(frame_id="WORLD", center_m=(2.0, 0.0, 0.0), half_extent_m=(0.2, 0.2, 0.2))
    blank = m.predicted_visibility(pose(), target, spec)
    assert blank.in_fov.all() and blank.mean < 0.05  # all-UNKNOWN path with p=0.5 per cell
    assert m.predicted_visibility(pose(), target, spec, p_unknown_block=0.0).mean == pytest.approx(1.0)
    for k in range(3):
        _feed(m, ids, store, spec, 1.0 + k)
    assert m.predicted_visibility(pose(), target, spec).mean > 0.9  # observed free path
    behind = SpatialSupport(frame_id="WORLD", center_m=(4.5, 0.0, 0.0), half_extent_m=(0.2, 0.2, 0.2))
    assert m.predicted_visibility(pose(), behind, spec).mean < 0.1  # the believed wall blocks


def test_cross_domain_context_annotates_semantics_only(store):
    ids = IdFactory(28)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    _feed(m, ids, store, spec, 1.0)
    pts = np.array([[3.1, 0.1, 0.1]])
    p0 = m.occupancy_probability(pts)
    unc = Uncertainty(aleatoric=0.1, epistemic=0.1, contradiction=0.0, observational=0.1)
    prov = ids.new()
    ctx = BeliefMessage(
        message_id=ids.new(),
        belief_id=ids.new(),
        revision=0,
        domain=Domain.TECHNICAL,
        timestamp=stamp(1.0, "SIM"),
        state_summary=(
            PropertyClaim(
                name="semantic_class",
                value="pipeline_segment",
                status=KnowledgeStatus.OBSERVED,
                uncertainty=unc,
                provenance_id=prov,
            ),
        ),
        state_embedding=(0.0,),
        knowledge_status=KnowledgeStatus.OBSERVED,
        uncertainty=unc,
        provenance_refs=(prov,),
        spatial_support=SpatialSupport(
            frame_id="WORLD", center_m=(3.1, 0.0, 0.0), half_extent_m=(0.2, 0.5, 0.5)
        ),
        lifecycle=Lifecycle.ACTIVE,
        model_version="t",
    )
    m.receive_context([ctx])
    assert np.array_equal(m.occupancy_probability(pts), p0)
    assert "pipeline_segment" in set(m.export_grid().level(0).semantic_class.tolist())
    m.reset_working_memory()
    assert m.context_messages() == []


def test_baselines_share_the_interface_and_show_their_failure_modes(store):
    ids = IdFactory(29)
    spec = depth_spec(ids)
    hidden = line(3.5, 6.0)
    for cls in (PlainOccupancyGrid, MaxLikelihoodSurfaceFill):
        m = new_model(ids, store, [spec], cls=cls)
        msgs = _feed(m, ids, store, spec, 1.0)
        assert msgs and m.domain is Domain.SPATIAL
        front = line(0.5, 2.5)
        assert set(m.occupancy_status(front)) == {KnowledgeStatus.OBSERVED}  # one scan -> OBSERVED
        p = m.occupancy_probability(hidden)
        if cls is MaxLikelihoodSurfaceFill:
            assert (np.maximum(p, 1 - p) >= 0.9).all()  # confident claims where nothing was seen
            assert KnowledgeStatus.UNKNOWN not in m.occupancy_status(hidden)
        else:
            assert set(m.occupancy_status(hidden)) == {KnowledgeStatus.UNKNOWN}
