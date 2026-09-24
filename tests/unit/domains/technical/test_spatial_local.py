"""Adversarial identifiability checks for the opt-in spatial architecture."""

import math
from uuid import UUID

import pytest

from conrad.domains.technical.spatial_local import LocalCondition, LocalThresholds, SpatialModel2T
from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, SurfaceRect
from conrad.schemas.observation import EntityCandidate, Evidence, Modality, Observation
from conrad.schemas.structural_sensor import StructuralSensorModel, StructuralSensorModelV2
from conrad.schemas.structural_support import CapsuleSurfaceSupport, ParameterAuthority
from conrad.schemas.timebase import TimeStamp
from conrad.twins.twin2t.spatial_emission import resolution_cell_supports
from conrad.twins.twin2t.spatial_field import (
    LocalEvolutionRate,
    LocalStructuralState,
    SpatialEvolutionV1,
    SpatialStructuralTruth,
    TruthPatch,
)

GRID = CapsuleSurfaceGrid(length_m=2.0, radius_m=1.0, axial_cells=2, sectors=1)
RID = UUID(int=42)
THRESHOLDS = LocalThresholds(0.002, 0.006, 0.012, 0.003, 0.01, 0.03)


def sensor(kernel="AREA_MEAN"):
    return StructuralSensorModel(
        footprint_width_m=2.0 if kernel == "AREA_MEAN" else 1.0,
        footprint_height_m=2 * math.pi,
        minimum_resolvable_corrosion_m=0.1,
        minimum_resolvable_crack_m=0.1,
        range_min_m=0.3,
        range_max_m=4.0,
        noise_sigma_m=0,
        position_uncertainty_m=0,
        orientation_uncertainty_rad=0,
        footprint_uncertainty_m=0,
        aggregation_kernel=kernel,
        authority=ParameterAuthority.ENGINEERING_ESTIMATE,
    )


def support(rect, model, *, ux=0.0, ua=0.0):
    return CapsuleSurfaceSupport(
        sensor_model_version=model.version,
        sensor_config_digest=model.digest,
        frame_id="CAPSULE_DESIGN",
        axial_start_m=rect.x0,
        axial_end_m=rect.x1,
        angle_start_rad=rect.a0,
        angle_end_rad=rect.a1,
        axial_uncertainty_m=ux,
        angular_uncertainty_rad=ua,
        aggregation_kernel=model.aggregation_kernel,
    )


def evidence(i, value, sup, group=None):
    return Evidence(
        evidence_id=UUID(int=100 + i),
        source_observation_id=UUID(int=200 + i),
        mission_id=UUID(int=1),
        run_id=UUID(int=2),
        trace_id=UUID(int=3),
        modality=Modality.STRUCTURED,
        timestamp=TimeStamp(time_ns=i, clock_domain="sim"),
        created_time_ns=i,
        embedding=(0.0,),
        reliability=1.0,
        aleatoric_uncertainty=0,
        measurements={
            "apparent_wall_loss": value.corrosion_depth_m,
            "crack_indication_length": value.crack_length_m,
        },
        measurement_units={"apparent_wall_loss": "m", "crack_indication_length": "m"},
        structural_support=sup,
        independence_group=group,
        entity_candidates=(EntityCandidate(registry_entity_id=RID, score=1.0),),
        provenance_id=UUID(int=300 + i),
        encoder_version="test-structural-v2",
    )


def test_same_global_mean_is_ambiguous_coarse_but_separable_resolved():
    uniform = SpatialStructuralTruth(GRID, (LocalStructuralState(0.001),) * 2)
    patchy = SpatialStructuralTruth(GRID, (LocalStructuralState(0), LocalStructuralState(0.002)))
    assert uniform.worst_local().corrosion_depth_m < patchy.worst_local().corrosion_depth_m
    coarse = sensor()
    whole = support(SurfaceRect(0, 2, 0, 2 * math.pi), coarse)
    assert uniform.measure(whole, coarse) == patchy.measure(whole, coarse)
    resolved = sensor("LOCAL_MAX")
    first, second = (support(GRID.cell(i), resolved) for i in range(2))
    assert uniform.measure(second, resolved) != patchy.measure(second, resolved)
    a, b = SpatialModel2T(GRID, resolved, THRESHOLDS, RID), SpatialModel2T(GRID, resolved, THRESHOLDS, RID)
    for i, sup in enumerate((first, second)):
        a.ingest(evidence(i, uniform.measure(sup, resolved), sup))
        b.ingest(evidence(i, patchy.measure(sup, resolved), sup))
    assert a.condition() is LocalCondition.OBSERVED_INTACT
    assert b.condition() is LocalCondition.DEGRADED


def test_missed_defect_and_partial_healthy_cannot_be_intact():
    model = sensor("LOCAL_MAX")
    world = SpatialStructuralTruth(GRID, (LocalStructuralState(), LocalStructuralState(0.008)))
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
    sup = support(GRID.cell(0), model)
    assert belief.ingest(evidence(1, world.measure(sup, model), sup))
    assert belief.condition() is LocalCondition.UNKNOWN
    assert belief.coverage_fraction(0) == pytest.approx(1.0)
    assert belief.coverage_fraction(1) == 0


def test_detected_defect_and_healthy_full_coverage():
    model = sensor("LOCAL_MAX")
    defective = SpatialStructuralTruth(GRID, (LocalStructuralState(), LocalStructuralState(0.008)))
    healthy = SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2)
    for world, expected in ((defective, LocalCondition.SEVERE), (healthy, LocalCondition.OBSERVED_INTACT)):
        belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
        for i in range(2):
            sup = support(GRID.cell(i), model)
            assert belief.ingest(evidence(i, world.measure(sup, model), sup))
        assert belief.condition() is expected


def test_coarse_average_and_unknown_support_never_credit_cells():
    mean = sensor()
    belief = SpatialModel2T(GRID, mean, THRESHOLDS, RID)
    whole = support(SurfaceRect(0, 2, 0, 2 * math.pi), mean)
    assert not belief.ingest(evidence(1, LocalStructuralState(0.001), whole))
    assert belief.condition() is LocalCondition.UNKNOWN
    resolved = sensor("LOCAL_MAX")
    belief = SpatialModel2T(GRID, resolved, THRESHOLDS, RID)
    unknown = support(GRID.cell(0), resolved, ux=None)
    assert not belief.ingest(evidence(2, LocalStructuralState(), unknown))
    assert belief.coverage_fraction(0) == 0


def test_subresolution_and_edge_support_do_not_create_false_intact():
    model = sensor("LOCAL_MAX")
    tiny = TruthPatch(SurfaceRect(0.2, 0.24, 1.0, 1.04), LocalStructuralState(0.01))
    truth = SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2, (tiny,))
    first = support(GRID.cell(0), model)
    assert truth.worst_local().corrosion_depth_m == 0.01
    assert truth.measure(first, model).corrosion_depth_m == 0.0
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
    for i in range(2):
        sup = support(GRID.cell(i), model, ux=0.01)
        belief.ingest(evidence(i, truth.measure(sup, model), sup))
    assert belief.condition() is LocalCondition.UNKNOWN
    assert all(belief.coverage_fraction(i) < 1 for i in range(2))


def test_independence_groups_and_replay_deduplication():
    model = sensor("LOCAL_MAX")
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID, required_looks=2)
    for i in range(2):
        sup = support(GRID.cell(i), model)
        ev = evidence(i, LocalStructuralState(), sup, group="same-capture")
        assert belief.ingest(ev)
        assert not belief.ingest(ev)
    assert belief.condition() is LocalCondition.UNKNOWN
    for i in range(2):
        sup = support(GRID.cell(i), model)
        belief.ingest(evidence(10 + i, LocalStructuralState(), sup, group=f"capture-{i}"))
    assert belief.condition() is LocalCondition.OBSERVED_INTACT


def test_partial_overlap_requires_resolved_intersection():
    model = sensor("LOCAL_MAX")
    patch = TruthPatch(SurfaceRect(0.8, 1.2, 1.0, 1.4), LocalStructuralState(0.008))
    truth = SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2, (patch,))
    edge = support(SurfaceRect(0.75, 0.85, 1.0, 1.4), model)
    inside = support(SurfaceRect(0.75, 0.95, 1.0, 1.4), model)
    assert truth.measure(edge, model).corrosion_depth_m == 0.0
    assert truth.measure(inside, model).corrosion_depth_m == 0.008


def test_observation_evidence_support_roundtrip_and_legacy_parse():
    model = sensor("LOCAL_MAX")
    sup = support(GRID.cell(0), model)
    obs = Observation(
        observation_id=UUID(int=8),
        mission_id=UUID(int=1),
        run_id=UUID(int=2),
        trace_id=UUID(int=3),
        sensor_id=UUID(int=4),
        modality=Modality.STRUCTURED,
        timestamp=TimeStamp(time_ns=1, clock_domain="sim"),
        sensor_frame="SENSOR",
        robot_pose_estimate=None,
        inline_values=(0.0, 0.0),
        inline_units="m,m",
        structural_support=sup,
    )
    assert Observation.model_validate_json(obs.canonical_json()) == obs
    old = obs.model_dump(mode="json")
    del old["structural_support"]
    assert Observation.model_validate(old).structural_support is None
    ev = evidence(1, LocalStructuralState(), sup)
    assert Evidence.model_validate_json(ev.canonical_json()) == ev
    old_evidence = ev.model_dump(mode="json")
    del old_evidence["structural_support"]
    assert Evidence.model_validate(old_evidence).structural_support is None


def test_config_mismatch_fails_closed_before_local_credit():
    model = sensor("LOCAL_MAX")
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
    altered = model.model_copy(update={"minimum_resolvable_corrosion_m": 0.2})
    sup = support(GRID.cell(0), altered)
    with pytest.raises(ValueError, match="architecture mismatch"):
        belief.ingest(evidence(1, LocalStructuralState(), sup))
    assert belief.coverage_fraction(0) == 0


def test_unassociated_evidence_cannot_update_a_component():
    model = sensor("LOCAL_MAX")
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
    sup = support(GRID.cell(0), model)
    unassociated = evidence(1, LocalStructuralState(), sup).model_copy(update={"entity_candidates": ()})
    assert not belief.ingest(unassociated)
    assert belief.coverage_fraction(0) == 0


def test_union_coverage_never_double_counts_duplicate_overlap():
    model = sensor("LOCAL_MAX")
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
    half = support(SurfaceRect(0, 0.5, 0, 2 * math.pi), model)
    for i in range(4):
        belief.ingest(evidence(i, LocalStructuralState(), half, group=f"capture-{i}"))
    assert belief.coverage_fraction(0) == pytest.approx(0.5)
    assert belief.condition() is LocalCondition.UNKNOWN


def test_v2_resolution_cells_separate_same_mean_and_reject_coarse_credit():
    model = StructuralSensorModelV2(
        footprint_width_m=2.0,
        footprint_height_m=2 * math.pi,
        axial_resolution_m=1.0,
        lateral_resolution_m=2 * math.pi,
        minimum_resolvable_corrosion_m=0.1,
        minimum_resolvable_crack_m=0.1,
        range_min_m=0.3,
        range_max_m=4.0,
        noise_sigma_m=0,
        position_uncertainty_m=0,
        orientation_uncertainty_rad=0,
        footprint_uncertainty_m=0,
        authority=ParameterAuthority.ENGINEERING_ESTIMATE,
    )
    whole = support(SurfaceRect(0, 2, 0, 2 * math.pi), model)
    cells = resolution_cell_supports(whole, model, GRID.radius_m)
    assert len(cells) == 2
    assert [(s.axial_start_m, s.axial_end_m) for s in cells] == [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match="resolution-cell"):
        SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2).measure(whole, model)
    uniform = SpatialStructuralTruth(GRID, (LocalStructuralState(0.001),) * 2)
    patchy = SpatialStructuralTruth(GRID, (LocalStructuralState(), LocalStructuralState(0.002)))
    coarse = sensor()
    coarse_whole = support(SurfaceRect(0, 2, 0, 2 * math.pi), coarse)
    assert uniform.measure(coarse_whole, coarse) == patchy.measure(coarse_whole, coarse)
    assert [uniform.measure(s, model) for s in cells] != [patchy.measure(s, model) for s in cells]
    belief = SpatialModel2T(GRID, model, THRESHOLDS, RID)
    with pytest.raises(ValueError, match="resolution-cell"):
        belief.ingest(evidence(20, LocalStructuralState(), whole))
    for i, cell_support in enumerate(cells):
        assert belief.ingest(evidence(30 + i, uniform.measure(cell_support, model), cell_support))
    assert belief.condition() is LocalCondition.OBSERVED_INTACT


def test_v2_subresolution_patch_does_not_appear_as_detected():
    model = StructuralSensorModelV2(
        footprint_width_m=1.0,
        footprint_height_m=2 * math.pi,
        axial_resolution_m=1.0,
        lateral_resolution_m=2 * math.pi,
        minimum_resolvable_corrosion_m=0.1,
        minimum_resolvable_crack_m=0.1,
        range_min_m=0.3,
        range_max_m=4.0,
        noise_sigma_m=0,
        authority=ParameterAuthority.ENGINEERING_ESTIMATE,
    )
    tiny = TruthPatch(SurfaceRect(0.2, 0.24, 1.0, 1.04), LocalStructuralState(0.01))
    truth = SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2, (tiny,))
    assert truth.measure(support(GRID.cell(0), model), model).corrosion_depth_m == 0.0

    straddling = TruthPatch(SurfaceRect(0.95, 1.05, 1.0, 1.2), LocalStructuralState(0.01))
    truth = SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2, (straddling,))
    assert truth.measure(support(GRID.cell(0), model), model).corrosion_depth_m == 0.01
    assert truth.measure(support(GRID.cell(1), model), model).corrosion_depth_m == 0.01


def test_optional_local_evolution_keeps_independent_cells_and_patches():
    patch = TruthPatch(SurfaceRect(0.2, 0.4, 1.0, 1.2), LocalStructuralState(0.002))
    initial = SpatialStructuralTruth(GRID, (LocalStructuralState(),) * 2, (patch,))
    rates = SpatialEvolutionV1(
        base_rates=(LocalEvolutionRate(0.001), LocalEvolutionRate()),
        patch_rates=(LocalEvolutionRate(crack_m_per_s=0.002),),
    )
    evolved = initial.evolve(2.0, rates)
    assert evolved.base == (LocalStructuralState(0.002), LocalStructuralState())
    assert evolved.patches[0].state == LocalStructuralState(0.002, 0.004)
    assert initial.base == (LocalStructuralState(),) * 2
    with pytest.raises(ValueError, match="one evolution rate"):
        initial.evolve(1.0, SpatialEvolutionV1(base_rates=()))
