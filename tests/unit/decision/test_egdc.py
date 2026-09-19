"""EGDC: grounding, UIR, constraints, vocabulary, router, cause -> need, abstention, imperfect upstream."""

import ast
import pathlib

import pytest

from conrad.decision import (
    ACTION_VOCABULARY,
    EGDC,
    ROUTE_TABLE,
    ClaimGraphBuilder,
    ConstraintEngine,
    DecisionConfig,
    DecisionSummary,
    RejectedAction,
    RouteTarget,
    uir_report,
    unsupported_inference_rate,
)
from conrad.evaluation.decision_experiments.fixtures import (
    make_belief,
    make_context,
    make_requirement,
    region,
    unc,
)
from conrad.schemas.belief import Availability
from conrad.schemas.decision import (
    ActionProposal,
    ActionType,
    ClaimType,
    GroundingStatus,
    InformationNeed,
    NavigationGoal,
    QuestionType,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import SourceType
from conrad.schemas.robot import HealthLevel
from conrad.schemas.world import Domain

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _ctx(ids, u=None, **kw):
    b = make_belief(ids, uncertainty=u or unc(), **kw.pop("belief_kw", {}))
    req = make_requirement(ids, belief_ids=[b.belief_id], **kw.pop("req_kw", {}))
    return b, make_context(ids, [b], [req], **kw)


def test_grounding_invariant_and_provenance_chain():
    ids = IdFactory(1)
    b, ctx = _ctx(ids)
    out = EGDC(ids).decide(ctx)
    world = [c for c in out.record.claims if c.claim_type is ClaimType.BELIEF_CLAIM]
    assert world
    for c in world:
        if c.grounding is GroundingStatus.GROUNDED:
            assert c.source_belief_ids and c.evidence_refs  # Decision -> Claim -> Belief -> Evidence
    assert out.provenance.source_type is SourceType.DECISION
    assert set(out.provenance.parent_records) == set(b.provenance_refs)
    assert out.record.provenance_id == out.provenance.record_id


def test_observed_belief_without_evidence_is_unsupported_and_never_relied_on():
    ids = IdFactory(2)
    _, ctx = _ctx(ids, belief_kw={"n_evidence": 0})
    out = EGDC(ids).decide(ctx)
    assert out.record.unsupported_claim_ids
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is not ActionType.CONTINUE_MISSION
    assert unsupported_inference_rate([out.record]) == 0.0


def test_uir_counts_relied_unsupported_claims():
    ids = IdFactory(3)
    _, ctx = _ctx(ids, belief_kw={"n_evidence": 0})
    record = EGDC(ids).decide(ctx).record
    bad = record.unsupported_claim_ids[0]
    forced = record.model_copy(
        update={
            "chosen": ActionProposal(
                action_id=ids.new(), action_type=ActionType.CONTINUE_MISSION, supporting_claims=(bad,)
            )
        }
    )
    rep = uir_report([forced])
    assert rep.relied_world_claims == 1 and rep.relied_unsupported_claims == 1
    assert rep.unsupported_inference_rate == 1.0
    assert uir_report([]).unsupported_inference_rate == 0.0


@pytest.mark.parametrize(
    ("u", "question"),
    [
        (unc(ua=0.9), QuestionType.IMPROVE_MEASUREMENT),
        (unc(uo=0.9), QuestionType.EXTEND_COVERAGE),
        (unc(uc=0.9), QuestionType.DISCRIMINATE_HYPOTHESES),
        (unc(ue=0.9), QuestionType.CONFIRM_CONDITION),
    ],
)
def test_uncertainty_cause_maps_to_need(u, question):
    ids = IdFactory(4)
    _, ctx = _ctx(ids, u)
    out = EGDC(ids).decide(ctx)
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is ActionType.REQUEST_INFORMATION
    assert out.routed is not None
    need = out.routed.payload
    assert isinstance(need, InformationNeed) and need.question_type is question
    if question is QuestionType.CONFIRM_CONDITION:
        assert need.constraints["require_alternate_modality"] is True


def test_contradiction_with_conflicting_evidence_resolves_contradiction():
    ids = IdFactory(5)
    _, ctx = _ctx(ids, unc(uc=0.9), belief_kw={"n_conflicts": 2})
    routed = EGDC(ids).decide(ctx).routed
    assert routed is not None
    need = routed.payload
    assert isinstance(need, InformationNeed) and need.question_type is QuestionType.RESOLVE_CONTRADICTION


def test_ood_without_alternate_modality_escalates():
    ids = IdFactory(6)
    _, ctx = _ctx(ids, unc(ue=0.9), modalities=("RGB",), notes={"modalities_used": ["RGB"]})
    out = EGDC(ids).decide(ctx)
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is ActionType.ESCALATE_TO_OPERATOR
    assert out.routed is not None
    assert out.routed.target is RouteTarget.OPERATOR and out.record.abstained


def test_low_consequence_uncertainty_does_not_trigger_sensing():
    ids = IdFactory(7)
    _, ctx = _ctx(ids, unc(uo=0.9), req_kw={"consequence": 0.1})
    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None
    assert chosen.action_type is ActionType.CONTINUE_MISSION


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"belief_kw": {"time_s": 10.0}}, ActionType.QUERY_BELIEF),  # stale
        ({"availability": {Domain.TECHNICAL: Availability.UNAVAILABLE}}, ActionType.REVISIT_REGION),
    ],
)
def test_imperfect_upstream(kw, expected):
    ids = IdFactory(8)
    _, ctx = _ctx(ids, **kw)
    out = EGDC(ids).decide(ctx)
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is expected
    assert out.record.chosen.action_type is not ActionType.CONTINUE_MISSION


def test_missing_belief_and_wrong_association():
    ids = IdFactory(9)
    req = make_requirement(ids, belief_ids=[ids.new()])
    out = EGDC(ids).decide(make_context(ids, [], [req]))
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is ActionType.QUERY_BELIEF
    other = make_belief(ids, world_entity_id=ids.new())
    req2 = make_requirement(ids, belief_ids=[other.belief_id], entity_ids=[ids.new()])
    out2 = EGDC(ids).decide(make_context(ids, [other], [req2]))
    assert out2.record.chosen is not None
    assert out2.record.chosen.action_type is not ActionType.CONTINUE_MISSION


def test_cross_domain_disagreement_requests_coverage():
    ids = IdFactory(10)
    t = make_belief(ids)
    s = make_belief(ids, domain=Domain.SPATIAL, coverage=0.1, properties={"occupied": True})
    req = make_requirement(ids, belief_ids=[t.belief_id], context_domains=[Domain.SPATIAL])
    out = EGDC(ids).decide(make_context(ids, [t, s], [req]))
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is ActionType.REQUEST_INFORMATION
    assert out.routed is not None
    need = out.routed.payload
    assert isinstance(need, InformationNeed) and need.question_type is QuestionType.EXTEND_COVERAGE


def test_justified_abstention_when_nothing_is_permitted():
    ids = IdFactory(11)
    _, ctx = _ctx(
        ids, unc(uo=0.9), health=HealthLevel.FAULT, motion_permitted=False, operator_reachable=False
    )
    out = EGDC(ids).decide(ctx)
    assert out.record.abstained
    assert out.record.chosen is not None
    assert out.record.chosen.action_type in (
        ActionType.WAIT,
        ActionType.ESCALATE_TO_OPERATOR,
        ActionType.RETURN_TO_SAFE_STATE,
        ActionType.ABORT_MISSION,
    )
    assert all(r.reason_codes for r in out.rejected)


def test_repeated_attempts_lead_to_escalation_or_replan():
    ids = IdFactory(12)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    hist = [
        DecisionSummary(
            decision_id=ids.new(),
            time_ns=0,
            action_type=ActionType.REQUEST_INFORMATION,
            target_belief_ids=(b.belief_id,),
        )
        for _ in range(3)
    ]
    out = EGDC(ids).decide(make_context(ids, [b], [req], previous=hist))
    assert out.record.chosen is not None
    assert out.record.chosen.action_type is not ActionType.REQUEST_INFORMATION


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"battery": 0.05}, "BATTERY_BELOW_RESERVE"),
        ({"motion_permitted": False}, "MOTION_NOT_PERMITTED"),
        ({"health": HealthLevel.FAULT}, "SYSTEM_HEALTH_FAULT"),
        ({"robot_state_age_s": 10.0}, "STALE_ROBOT_STATE"),
        ({"boundary": ((0.0, 0.0, -1.0), (2.0, 2.0, 1.0))}, "OUTSIDE_MISSION_BOUNDARY"),
    ],
)
def test_constraint_engine_rejections(kw, code):
    ids = IdFactory(13)
    cfg = DecisionConfig()
    b, ctx = _ctx(ids, **kw)
    graph = ClaimGraphBuilder(ids, cfg).build(ctx)
    action = ActionProposal(
        action_id=ids.new(),
        action_type=ActionType.REVISIT_REGION,
        target_belief_ids=(b.belief_id,),
        target_region=region(),
    )
    verdict = ConstraintEngine(cfg).check(action, graph, ctx)
    assert not verdict.accepted and code in verdict.reason_codes


def test_constraint_engine_argument_validation_and_unsupported_fact():
    ids = IdFactory(14)
    cfg = DecisionConfig()
    _, ctx = _ctx(ids, belief_kw={"n_evidence": 0})
    graph = ClaimGraphBuilder(ids, cfg).build(ctx)
    eng = ConstraintEngine(cfg)
    bad_wait = ActionProposal(action_id=ids.new(), action_type=ActionType.WAIT, parameters={"duration_s": -1})
    assert "INVALID_ARGUMENT:duration_s" in eng.check(bad_wait, graph, ctx).reason_codes
    no_target = ActionProposal(
        action_id=ids.new(),
        action_type=ActionType.REQUEST_INFORMATION,
        parameters={"question_type": "NOPE", "cause": "X"},
    )
    codes = eng.check(no_target, graph, ctx).reason_codes
    assert "MISSING_TARGET" in codes and "INVALID_ARGUMENT:question_type" in codes
    relies = ActionProposal(
        action_id=ids.new(),
        action_type=ActionType.CONTINUE_MISSION,
        supporting_claims=graph.unsupported_ids(),
    )
    assert "UNSUPPORTED_CLAIM_AS_FACT" in eng.check(relies, graph, ctx).reason_codes


def test_vocabulary_is_closed():
    assert frozenset(ActionType) == ACTION_VOCABULARY
    assert set(ROUTE_TABLE) == set(ActionType)
    ids = IdFactory(15)
    for u in (unc(), unc(uo=0.9), unc(ua=0.9), unc(uc=0.9), unc(ue=0.9)):
        _, ctx = _ctx(ids, u)
        for a in EGDC(ids).decide(ctx).record.candidates:
            assert a.action_type in ACTION_VOCABULARY


def test_router_mapping_and_rejected_record():
    ids = IdFactory(16)
    _, ctx = _ctx(ids, unc(uo=0.9))
    egdc = EGDC(ids)
    out = egdc.decide(ctx)
    assert out.routed is not None
    assert out.routed.target is RouteTarget.MCBR and isinstance(out.routed.payload, InformationNeed)
    revisit = next(a for a in out.record.candidates if a.action_type is ActionType.CONTINUE_MISSION)
    graph = egdc.last_graph
    assert graph is not None
    verdict = egdc.constraints.check(revisit, graph, ctx, egdc.estimator.estimate(revisit, graph, ctx))
    assert "RISK_LIMIT_EXCEEDED" in verdict.reason_codes  # continuing on an open critical gap
    routed = egdc.router.route(revisit, verdict, ctx)
    assert isinstance(routed.payload, RejectedAction) and routed.target is None
    _, ctx2 = _ctx(ids, availability={Domain.TECHNICAL: Availability.UNAVAILABLE})
    out2 = EGDC(ids).decide(ctx2)
    assert out2.routed is not None
    assert isinstance(out2.routed.payload, NavigationGoal)


def test_decision_package_never_imports_truth_or_sends_commands():
    forbidden = ("conrad.twins", "conrad.schemas.truth", "conrad.sim", "conrad.evaluation")
    for pkg in ("decision", "active", "communication"):
        for path in (ROOT / "conrad" / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    assert not n.startswith(forbidden), f"{path} imports {n}"
    for path in [ROOT / "conrad" / "orchestration" / "belief_bus.py"]:
        text = path.read_text(encoding="utf-8")
        assert "conrad.twins" not in text and "schemas.truth" not in text and "evaluation" not in text


def test_deterministic_replay():
    def run(seed):
        ids = IdFactory(seed)
        _, ctx = _ctx(ids, unc(uc=0.9))
        return EGDC(ids).decide(ctx).record.canonical_json()

    assert run(99) == run(99)
