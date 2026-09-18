"""Gate P0 / Interface Freeze V1 contract tests (ch25 Phase 0 acceptance)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

import conrad.schemas as schemas
from conrad.robotics.hardware.config import load_robot_config, validate_for_lane
from conrad.schemas.base import ARCHITECTURE_ID, SCHEMA_VERSION, STACK_ID, SchemaVersionError, check_schema_compatible
from conrad.schemas.belief import KnowledgeStatus, PropertyClaim
from conrad.schemas.frames import (
    FramedPoint,
    FrameError,
    FrameGraph,
    Pose,
    Transform,
    quat_from_euler,
    transform_point,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceError, ProvenanceRecord, SourceType, validate_provenance_dag
from conrad.schemas.robot import Sourced, SourceKind
from conrad.schemas.timebase import ClockDomainError, TimeStamp, stamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.settings import ExecutionLane
from tests.conftest import make_evidence, make_revision, ts


def test_identity_constants() -> None:
    assert ARCHITECTURE_ID == "conrad_v2_impl_freeze_v0_2"
    assert STACK_ID == "conrad_v2_stack_freeze_v0_2"


def test_round_trip_serialization(ids: IdFactory) -> None:
    ev, prov = make_evidence(ids, ids.new())
    rev, rprov = make_revision(ids, ids.new(), 0, 1.0, (ev.evidence_id,), (prov.record_id,))
    for obj in (ev, prov, rev, rprov, rev.cell):
        again = type(obj).model_validate_json(obj.canonical_json())
        assert again == obj
        assert again.content_digest() == obj.content_digest()
        if "schema_version" in type(obj).model_fields:
            assert json.loads(obj.canonical_json())["schema_version"] == SCHEMA_VERSION


def test_unknown_keys_rejected(ids: IdFactory) -> None:
    ev, _ = make_evidence(ids, ids.new())
    data = json.loads(ev.canonical_json())
    data["hidden_truth_label"] = 1
    with pytest.raises(ValidationError):
        type(ev).model_validate(data)


def test_schema_version_gate() -> None:
    check_schema_compatible("1.4.2")
    with pytest.raises(SchemaVersionError):
        check_schema_compatible("2.0.0")
    with pytest.raises(SchemaVersionError):
        check_schema_compatible("garbage")


def test_uuid_uniqueness_and_determinism() -> None:
    a, b = IdFactory(seed=1), IdFactory(seed=1)
    first = [a.new() for _ in range(2000)]
    assert len(set(first)) == 2000
    assert first == [b.new() for _ in range(2000)]
    assert all(u.version == 7 for u in first)
    live = IdFactory()
    assert len({live.new() for _ in range(2000)}) == 2000


def test_time_semantics() -> None:
    with pytest.raises(ValidationError):
        TimeStamp(time_ns=5, clock_domain="SIM", measurement_start_ns=10, measurement_end_ns=5)
    assert stamp(2.5, "SIM").delta_s(stamp(1.0, "SIM")) == pytest.approx(1.5)
    with pytest.raises(ClockDomainError):
        stamp(2.0, "SIM").delta_s(stamp(1.0, "ROBOT"))


def test_evidence_cannot_predate_observation(ids: IdFactory) -> None:
    ev, _ = make_evidence(ids, ids.new(), t=2.0)
    with pytest.raises(ValidationError):
        ev.model_copy(update={"created_time_ns": 1}).model_validate(
            {**json.loads(ev.canonical_json()), "created_time_ns": 1}
        )


def test_frame_contract() -> None:
    t_ab = Transform(parent_frame="A", child_frame="B", translation_m=(1.0, -2.0, 0.5), rotation_wxyz=quat_from_euler(0.2, -0.4, 1.1))
    t_ba = t_ab.inverse()
    assert (t_ba.parent_frame, t_ba.child_frame) == ("B", "A")
    assert np.allclose(t_ab.rotation @ t_ba.rotation, np.eye(3), atol=1e-9)
    assert np.allclose(t_ab.rotation @ t_ba.translation + t_ab.translation, 0, atol=1e-9)
    # unit-axis probe: 90 deg yaw maps +X to +Y in a right-handed Z-up frame
    yaw = Transform(parent_frame="A", child_frame="B", translation_m=(0, 0, 0), rotation_wxyz=quat_from_euler(0, 0, math.pi / 2))
    out = transform_point(FramedPoint(frame_id="B", xyz_m=(1, 0, 0)), yaw)
    assert out.frame_id == "A" and np.allclose(out.xyz_m, (0, 1, 0), atol=1e-9)
    with pytest.raises(FrameError):
        transform_point(FramedPoint(frame_id="C", xyz_m=(1, 0, 0)), yaw)
    with pytest.raises(ValidationError):
        Transform(parent_frame="A", child_frame="A", translation_m=(0, 0, 0))


def test_frame_graph_round_trip() -> None:
    g = FrameGraph()
    g.set(Transform(parent_frame="WORLD", child_frame="ROBOT", translation_m=(3, 4, -5), rotation_wxyz=quat_from_euler(0, 0, 0.7)))
    g.set(Transform(parent_frame="ROBOT", child_frame="CAM", translation_m=(0.2, 0, 0.1), rotation_wxyz=quat_from_euler(0, 0.3, 0)))
    p = FramedPoint(frame_id="CAM", xyz_m=(1.0, 0.5, 2.0))
    world = transform_point(p, g.lookup("WORLD", "CAM"))
    back = transform_point(world, g.lookup("CAM", "WORLD"))
    assert np.allclose(back.xyz_m, p.xyz_m, atol=1e-9)
    with pytest.raises(FrameError):
        g.lookup("WORLD", "SONAR")


def test_every_spatial_type_requires_a_frame() -> None:
    with pytest.raises(ValidationError):
        Pose(frame_id="", position_m=(0, 0, 0))
    with pytest.raises(ValidationError):
        FramedPoint.model_validate({"xyz_m": (0, 0, 0)})
    assert Pose(frame_id="WORLD", position_m=(0, 0, 0)).position_sigma_m() is None


def test_uncertainty_contract() -> None:
    u = Uncertainty(aleatoric=0.1, epistemic=0.2, contradiction=0.9, observational=0.3)
    assert u.dominant_channel() == "contradiction"
    assert not any(name in type(u).model_fields for name in ("confidence", "total", "score"))
    with pytest.raises(ValidationError):
        Uncertainty(aleatoric=-0.1, epistemic=0, contradiction=0, observational=0)
    with pytest.raises(ValidationError):
        Uncertainty.model_validate({"aleatoric": 0.1})


def test_unknown_is_a_status_not_a_value(ids: IdFactory) -> None:
    u = Uncertainty(aleatoric=0, epistemic=1, contradiction=0, observational=1)
    PropertyClaim(name="occupancy", value=None, status=KnowledgeStatus.UNKNOWN, uncertainty=u)
    with pytest.raises(ValidationError):
        PropertyClaim(name="occupancy", value=1.0, status=KnowledgeStatus.UNKNOWN, uncertainty=u)
    with pytest.raises(ValidationError):
        PropertyClaim(name="occupancy", value=1.0, status=KnowledgeStatus.OBSERVED, uncertainty=u)


def test_provenance_dag(ids: IdFactory) -> None:
    def rec(parents: tuple = ()) -> ProvenanceRecord:
        return ProvenanceRecord(
            record_id=ids.new(), source_type=SourceType.DIRECT_OBSERVATION, source_ids=(ids.new(),),
            operation="x", module="m", model_version="v", timestamp=ts(1.0), parent_records=parents,
        )

    a = rec()
    b = rec((a.record_id,))
    c = rec((a.record_id, b.record_id))
    graph = {r.record_id: r for r in (a, b, c)}
    assert validate_provenance_dag(graph, [c.record_id]) == set(graph)
    with pytest.raises(ProvenanceError):
        validate_provenance_dag({c.record_id: c}, [c.record_id])
    cyc_a = a.model_copy(update={"parent_records": (c.record_id,)})
    with pytest.raises(ProvenanceError):
        validate_provenance_dag({**graph, a.record_id: cyc_a}, [c.record_id])


def test_relational_and_predicted_updates_cannot_claim_evidence(ids: IdFactory) -> None:
    from conrad.schemas.belief import UpdateKind

    with pytest.raises(ValidationError):
        make_revision(ids, ids.new(), 0, 1.0, (ids.new(),), (), kind=UpdateKind.RELATIONAL)
    with pytest.raises(ValidationError):
        make_revision(ids, ids.new(), 0, 1.0, (), (), kind=UpdateKind.DIRECT)


def test_robot_config_validation() -> None:
    sim = load_robot_config("configs/robot/sim_reference.yaml")
    assert validate_for_lane(sim, ExecutionLane.SIMULATION) == []
    assert validate_for_lane(sim, ExecutionLane.PHYSICAL), "synthetic values must never satisfy the physical lane"
    phys = load_robot_config("configs/robot/physical_template.yaml")
    assert phys.open_parameters() and any("thruster layout unknown" in p for p in validate_for_lane(phys, ExecutionLane.PHYSICAL))
    with pytest.raises(ValidationError):
        Sourced[float](value=3.0, units="kg", source=SourceKind.OPEN)
    with pytest.raises(ValidationError):
        Sourced[float](value=3.0, units="kg", source=SourceKind.MEASURED)  # measured without raw-log provenance
    with pytest.raises(ValidationError):
        Sourced[float](value=None, units="kg", source=SourceKind.ENGINEERING_ESTIMATE)


def test_truth_is_not_part_of_the_public_schema_surface() -> None:
    assert "TruthState" not in schemas.__all__ and not hasattr(schemas, "TruthState")
