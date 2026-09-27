from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_DNS

import pytest

from conrad.oceansense.alpha import AssetMemory, OceanSenseAlpha
from conrad.oceansense.host import HostCapability, HostCapabilityProfile, HostIntentGateway, InspectionIntent
from conrad.oceansense.inspection import AdaptiveInspectionPlanner, value_of_information
from conrad.oceansense.model2_adapter import MODEL2_DIM, RepresentationAdapter, adapt_evidence_for_model2
from conrad.oceansense.reasoning import CrossDomainReasoner, provenance_aware_uncertainty_update
from conrad.oceansense.task_heads import (
    AnomalyHead,
    ConditionHead,
    DetectionHead,
    DownstreamValidationStatus,
    SegmentationHead,
    TaskHeadInput,
)
from conrad.schemas.belief import BeliefMessage, KnowledgeStatus, Lifecycle, PropertyClaim
from conrad.schemas.decision import (
    InformationNeed,
    ObservationAction,
    ObservationPlan,
    PlanStatus,
    QuestionType,
    ResourceCost,
    UncertaintyType,
)
from conrad.schemas.frames import Pose, SpatialSupport
from conrad.schemas.observation import Evidence, Modality
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain
from conrad.active.planner import PlanResult, PlanningRequest
from conrad.schemas.provenance import ProvenanceRecord, SourceType


def uid(name: str) -> UUID:
    return uuid5(NAMESPACE_DNS, f"oceansense-p5-p10::{name}")


def representation() -> tuple[float, ...]:
    return tuple(((i % 17) - 8) / 100.0 for i in range(384))


def support() -> SpatialSupport:
    return SpatialSupport(frame_id="WORLD", center_m=(1.0, 2.0, -3.0), half_extent_m=(0.5, 0.5, 0.5))


def evidence(with_measurements: bool = True) -> Evidence:
    measurements = {"crack_length_m": 0.012} if with_measurements else {}
    units = {"crack_length_m": "m"} if with_measurements else {}
    return Evidence(
        evidence_id=uid("evidence"),
        source_observation_id=uid("observation"),
        mission_id=uid("mission"),
        run_id=uid("run"),
        trace_id=uid("trace"),
        modality=Modality.SONAR,
        timestamp=stamp(10.0, "fixture"),
        created_time_ns=11_000_000_000,
        embedding=tuple(0.01 for _ in range(256)),
        spatial_support=support(),
        reliability=0.9,
        aleatoric_uncertainty=0.05,
        measurements=measurements,
        measurement_units=units,
        provenance_id=uid("provenance"),
        encoder_version="fixture-direct-evidence",
    )


def task_input() -> TaskHeadInput:
    ev = evidence()
    return TaskHeadInput(
        representation=representation(),
        source_evidence_id=ev.evidence_id,
        source_observation_id=ev.source_observation_id,
        mission_id=ev.mission_id,
        run_id=ev.run_id,
        trace_id=ev.trace_id,
        modality=ev.modality,
        timestamp=ev.timestamp,
        provenance_id=ev.provenance_id,
    )


def plan() -> ObservationPlan:
    action = ObservationAction(
        action_id=uid("action"),
        pose=Pose(frame_id="WORLD", position_m=(1.0, 2.0, -2.5)),
        sensor_id=uid("sensor"),
        target_region=support(),
        duration_s=3.0,
        expected_cost=ResourceCost(time_s=4.0, energy_j=20.0, risk=0.1, travel_m=1.2),
        predicted_visibility=0.8,
        expected_information_gain=0.4,
    )
    return ObservationPlan(
        plan_id=uid("plan"),
        need_id=uid("need"),
        trace_id=uid("trace"),
        status=PlanStatus.PLAN,
        target_beliefs=(uid("belief"),),
        primary_action=action,
        expected_information_gain=0.4,
        targeted_uncertainty=(UncertaintyType.EPISTEMIC,),
        expected_cost=action.expected_cost,
        confidence=0.7,
        provenance=uid("provenance"),
    )


class StubPlanner:
    name = "stub-mcbr"

    def plan(self, request: PlanningRequest) -> PlanResult:
        return PlanResult(
            plan=plan().model_copy(update={"need_id": request.need.need_id}),
            provenance=ProvenanceRecord(
                record_id=uid("plan-provenance"),
                source_type=SourceType.PLAN,
                source_ids=(request.need.need_id,),
                operation="fixture adaptive inspection plan",
                module="tests.unit.oceansense",
                model_version="stub-mcbr",
                timestamp=request.now,
            ),
        )


def belief(name: str, domain: Domain, value: str, u: Uncertainty) -> BeliefMessage:
    prov = uid(f"{name}-provenance")
    claim = PropertyClaim(name="condition", value=value, status=KnowledgeStatus.OBSERVED, uncertainty=u, provenance_id=prov)
    return BeliefMessage(
        message_id=uid(f"{name}-message"),
        belief_id=uid(f"{name}-belief"),
        revision=1,
        domain=domain,
        timestamp=stamp(12.0, "fixture"),
        state_summary=(claim,),
        state_embedding=tuple(0.0 for _ in range(256)),
        knowledge_status=KnowledgeStatus.OBSERVED,
        uncertainty=u,
        evidence_support=(uid(f"{name}-evidence"),),
        provenance_refs=(prov,),
        spatial_support=support(),
        lifecycle=Lifecycle.ACTIVE,
        model_version="fixture-model2",
    )


def test_p5_task_heads_emit_schema_outputs_and_preserve_status() -> None:
    x = task_input()
    outputs = [h.infer(x) for h in (DetectionHead(), SegmentationHead(), AnomalyHead(), ConditionHead())]
    assert {o.head_name for o in outputs} == {
        "p5-detection-head-v0",
        "p5-segmentation-head-v0",
        "p5-anomaly-head-v0",
        "p5-condition-head-v0",
    }
    assert all(o.status is DownstreamValidationStatus.IMPLEMENTED for o in outputs)
    with pytest.raises(ValueError, match="VALIDATED-RUN"):
        DetectionHead().infer(x.model_copy(update={"status": DownstreamValidationStatus.VALIDATED_RUN}))


def test_p6_adapter_is_384_512_256_and_preserves_direct_measurements() -> None:
    ev = evidence()
    adapter = RepresentationAdapter(seed=7)
    bundle = adapt_evidence_for_model2(ev, representation(), adapter)
    assert adapter.w1.shape == (384, 512)
    assert adapter.w2.shape == (512, MODEL2_DIM)
    assert bundle.model2_embedding_dim == 256
    assert bundle.direct_physical_evidence is ev
    assert bundle.direct_physical_evidence.measurements == {"crack_length_m": 0.012}
    assert bundle.learned_evidence.measurements == ev.measurements
    assert bundle.semantic_evidence["modality"] == "SONAR"
    assert bundle.semantic_evidence["measurement_keys"] == ("crack_length_m",)
    assert bundle.semantic_evidence["measurement_units"] == {"crack_length_m": "m"}
    assert bundle.semantic_evidence["provenance_id"] == str(ev.provenance_id)


def test_p6_adapter_rejects_wrong_osfm_dimension_and_is_deterministic() -> None:
    adapter_a = RepresentationAdapter(seed=99)
    adapter_b = RepresentationAdapter(seed=99)
    assert adapter_a.transform(representation()) == adapter_b.transform(representation())
    with pytest.raises(ValueError, match="384D"):
        adapter_a.transform(tuple(0.0 for _ in range(383)))


def test_p7_uncertainty_channels_and_competing_hypotheses() -> None:
    base = Uncertainty(aleatoric=0.1, epistemic=0.8, contradiction=0.0, observational=0.9)
    fresh = Uncertainty(aleatoric=0.2, epistemic=0.3, contradiction=0.4, observational=0.2)
    updated = provenance_aware_uncertainty_update(base, fresh, independent=True)
    assert updated.as_tuple() == pytest.approx((0.2, 0.21, 0.4, 0.14))
    reasoner = CrossDomainReasoner()
    left = belief("technical", Domain.TECHNICAL, "nominal", updated)
    right = belief("spatial", Domain.SPATIAL, "blocked", fresh)
    assert reasoner.contradiction_between(left, right) == 1.0
    hyps = reasoner.build_hypotheses((left, right), (uid("h1"), uid("h2")))
    assert len(hyps.ranked()) == 2


def test_p8_information_need_value_of_information_contract() -> None:
    need = InformationNeed(
        need_id=uid("need"),
        trace_id=uid("trace"),
        target_belief_ids=(uid("belief"),),
        question_type=QuestionType.RESOLVE_CONTRADICTION,
        target_properties=("condition",),
        priority=0.8,
        desired_uncertainty_reduction={"CONTRADICTION": 0.4, "EPISTEMIC": 0.2},
    )
    assert value_of_information(need) == pytest.approx(0.56)


def test_p8_adaptive_inspection_updates_plan_contract_fields() -> None:
    need = InformationNeed(
        need_id=uid("need"),
        trace_id=uid("trace"),
        target_belief_ids=(uid("belief"),),
        question_type=QuestionType.RESOLVE_CONTRADICTION,
        target_properties=("condition",),
        priority=0.8,
        desired_uncertainty_reduction={"CONTRADICTION": 0.4, "EPISTEMIC": 0.2},
    )
    request = PlanningRequest(
        need=need,
        beliefs=(),
        robot_pose=Pose(frame_id="WORLD", position_m=(0.0, 0.0, 0.0)),
        sensors=(),
        is_free=lambda points: [],
        predicted_visibility=lambda pose, region: 0.0,
        navigation_cost=lambda start, end: ResourceCost(time_s=0.0, energy_j=0.0, risk=0.0, travel_m=0.0),
        now=stamp(12.0, "fixture"),
    )
    out = AdaptiveInspectionPlanner(StubPlanner()).plan(request)
    assert out.need_id == need.need_id
    assert out.expected_mission_gain == pytest.approx(0.56)
    assert out.targeted_uncertainty == (UncertaintyType.CONTRADICTION, UncertaintyType.EPISTEMIC)


def test_p9_gateway_preserves_host_boundary_and_degraded_paths() -> None:
    gateway = HostIntentGateway()
    profile = HostCapabilityProfile(
        host_id="rov-partner",
        capabilities=(HostCapability.OPERATOR_ESCALATION,),
        degraded=True,
        degradation_reasons=("NO_REVISIT_AUTONOMY",),
    )
    decision = gateway.from_plan(plan(), profile, uid("intent"))
    assert decision.accepted
    assert decision.intent is not None
    assert decision.intent.requested_capability is HostCapability.OPERATOR_ESCALATION
    assert "HOST_DEGRADED" in decision.reason_codes
    with pytest.raises(ValueError, match="low-level control"):
        InspectionIntent(
            intent_id=uid("bad-intent"),
            trace_id=uid("trace"),
            source_plan_id=uid("plan"),
            requested_capability=HostCapability.REVISIT_REGION,
            low_level_control_included=True,
        )


def test_p9_gateway_refuses_when_host_has_no_supported_fallback() -> None:
    decision = HostIntentGateway().from_plan(
        plan(),
        HostCapabilityProfile(host_id="minimal-host", capabilities=()),
        uid("intent"),
    )
    assert not decision.accepted
    assert decision.intent is None
    assert decision.reason_codes == ("HOST_CAPABILITY_UNAVAILABLE",)


def test_p9_capability_negotiation_filters_without_inventing_host_authority() -> None:
    profile = HostCapabilityProfile(
        host_id="degraded-host",
        capabilities=(HostCapability.STORE_AND_FORWARD, HostCapability.OPERATOR_ESCALATION),
        degraded=True,
        degradation_reasons=("NO_LOCAL_AUTONOMY",),
    )
    available = HostIntentGateway().negotiate(
        (
            HostCapability.REVISIT_REGION,
            HostCapability.OPERATOR_ESCALATION,
            HostCapability.STORE_AND_FORWARD,
        ),
        profile,
    )
    assert available == (HostCapability.OPERATOR_ESCALATION, HostCapability.STORE_AND_FORWARD)


def test_p10_alpha_records_asset_memory_replay_and_degraded_operation(tmp_path: Path) -> None:
    memory = AssetMemory.load(tmp_path / "asset_memory.json")
    alpha = OceanSenseAlpha(memory, RepresentationAdapter(seed=7))
    profile = HostCapabilityProfile(host_id="rov-partner", capabilities=(HostCapability.REVISIT_REGION,))
    result = alpha.run(evidence(), representation(), plan(), profile, uid("intent"))
    assert result.evidence_bundle.model2_embedding_dim == 256
    assert len(result.task_outputs) == 4
    assert result.gateway_decision is not None and result.gateway_decision.accepted
    assert result.coverage.planned
    assert result.coverage.host_accepted is True
    assert result.coverage.observation_count == 1
    assert result.coverage.status == "ACCEPTED_PLAN"
    assert not result.degraded
    loaded = AssetMemory.load(tmp_path / "asset_memory.json")
    assert loaded.records[0]["replay_key"] == result.replay_key
    assert loaded.records[0]["direct_physical_measurements"] == ["crack_length_m"]
    assert loaded.records[0]["coverage"]["status"] == "ACCEPTED_PLAN"


def test_p10_alpha_replay_key_and_outputs_are_deterministic(tmp_path: Path) -> None:
    first = OceanSenseAlpha(AssetMemory.load(tmp_path / "a.json"), RepresentationAdapter(seed=123)).run(
        evidence(), representation()
    )
    second = OceanSenseAlpha(AssetMemory.load(tmp_path / "b.json"), RepresentationAdapter(seed=123)).run(
        evidence(), representation()
    )
    assert first.replay_key == second.replay_key
    assert first.evidence_bundle.learned_evidence.embedding == second.evidence_bundle.learned_evidence.embedding
    assert [o.model_dump(mode="json") for o in first.task_outputs] == [
        o.model_dump(mode="json") for o in second.task_outputs
    ]


def test_p10_alpha_records_adapter_failure_as_degraded_operation(tmp_path: Path) -> None:
    memory = AssetMemory.load(tmp_path / "asset_memory.json")
    result = OceanSenseAlpha(memory, RepresentationAdapter(seed=7)).run(evidence(), tuple(0.0 for _ in range(383)))
    assert result.evidence_bundle is None
    assert result.task_outputs == ()
    assert result.gateway_decision is None
    assert result.degraded
    assert result.failures and result.failures[0].startswith("ADAPTER_FAILURE:")
    loaded = AssetMemory.load(tmp_path / "asset_memory.json")
    assert loaded.records[0]["degraded"] is True
    assert loaded.records[0]["model2_embedding_dim"] is None
    assert loaded.records[0]["coverage"]["status"] == "FAILED_BEFORE_PLANNING"
    assert loaded.records[0]["coverage"]["observation_count"] == 0
    assert loaded.records[0]["direct_physical_measurements"] == ["crack_length_m"]


def test_p5_p10_end_to_end_fixture_chain_preserves_boundaries(tmp_path: Path) -> None:
    ev = evidence()
    rep = representation()
    adapter = RepresentationAdapter(seed=321)

    bundle = adapt_evidence_for_model2(ev, rep, adapter)
    assert bundle.model2_embedding_dim == MODEL2_DIM
    assert bundle.direct_physical_evidence is ev

    head_input = task_input()
    head_outputs = [h.infer(head_input) for h in (DetectionHead(), SegmentationHead(), AnomalyHead(), ConditionHead())]
    assert all(o.status is DownstreamValidationStatus.IMPLEMENTED for o in head_outputs)

    base = Uncertainty(aleatoric=0.1, epistemic=0.7, contradiction=0.2, observational=0.6)
    condition_uncertainty = Uncertainty.model_validate(head_outputs[-1].payload["condition"]["uncertainty"])
    updated = provenance_aware_uncertainty_update(base, condition_uncertainty, independent=True)
    technical = belief("technical-e2e", Domain.TECHNICAL, "requires-review", updated)
    spatial = belief("spatial-e2e", Domain.SPATIAL, "blocked", base)
    hypotheses = CrossDomainReasoner().build_hypotheses((technical, spatial), (uid("e2e-h1"), uid("e2e-h2")))
    assert hypotheses.ranked()[0].score >= hypotheses.ranked()[-1].score

    need = InformationNeed(
        need_id=uid("need-e2e"),
        trace_id=ev.trace_id,
        target_belief_ids=(technical.belief_id, spatial.belief_id),
        question_type=QuestionType.DISCRIMINATE_HYPOTHESES,
        target_properties=("condition",),
        priority=0.9,
        desired_uncertainty_reduction={"CONTRADICTION": 0.3, "EPISTEMIC": 0.2},
    )
    request = PlanningRequest(
        need=need,
        beliefs=(technical, spatial),
        robot_pose=Pose(frame_id="WORLD", position_m=(0.0, 0.0, 0.0)),
        sensors=(),
        is_free=lambda points: [],
        predicted_visibility=lambda pose, region: 0.0,
        navigation_cost=lambda start, end: ResourceCost(time_s=0.0, energy_j=0.0, risk=0.0, travel_m=0.0),
        now=stamp(12.0, "fixture"),
    )
    observation_plan = AdaptiveInspectionPlanner(StubPlanner()).plan(request)
    assert observation_plan.need_id == need.need_id
    assert observation_plan.targeted_uncertainty[:2] == (UncertaintyType.CONTRADICTION, UncertaintyType.EPISTEMIC)

    alpha = OceanSenseAlpha(AssetMemory.load(tmp_path / "asset_memory.json"), adapter)
    result = alpha.run(
        ev,
        rep,
        observation_plan,
        HostCapabilityProfile(host_id="partner-host", capabilities=(HostCapability.REVISIT_REGION,)),
        uid("intent-e2e"),
    )
    assert result.gateway_decision is not None and result.gateway_decision.accepted
    assert result.coverage.status == "ACCEPTED_PLAN"
    assert not result.degraded
    assert result.evidence_bundle.direct_physical_evidence is ev
    assert {o.head_name for o in result.task_outputs} == {o.head_name for o in head_outputs}
    memory = AssetMemory.load(tmp_path / "asset_memory.json")
    assert memory.records[0]["replay_key"] == result.replay_key
    assert memory.records[0]["direct_physical_measurements"] == ["crack_length_m"]
