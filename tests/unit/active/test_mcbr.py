"""MCBR: filter-before-rank, stop statuses, cause-specific gain, baselines, learned ranker, no motion."""

import pathlib
import re

import numpy as np
import torch

from conrad.active import MCBRConfig, MCBRPlanner, PlanningRequest, PriorView, SensorOption, make_planners
from conrad.active.learned import MCBRBatch, MCBRRankerConfig, build_ranker, mcbr_loss, train_ranker
from conrad.evaluation.decision_experiments.fixtures import make_belief, region, unc
from conrad.schemas.decision import InformationNeed, PlanStatus, QuestionType, ResourceCost
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _sensors(ids):
    return (
        SensorOption(
            sensor_id=ids.new(),
            modality="RGB",
            min_range_m=0.5,
            max_range_m=4.0,
            discrimination={"default": 0.1},
            measurement_quality=0.6,
        ),
        SensorOption(
            sensor_id=ids.new(),
            modality="SONAR",
            min_range_m=1.0,
            max_range_m=8.0,
            discrimination={"default": 0.7},
            measurement_quality=0.8,
            power_w=15.0,
        ),
    )


def _request(ids, u, question, *, free=None, vis=0.8, cost=None, views=(), constraints=None, beliefs=None):
    b = make_belief(ids, uncertainty=u, support=region((0.0, 0.0, 0.0)))
    need = InformationNeed(
        need_id=ids.new(),
        trace_id=ids.new(),
        target_belief_ids=(b.belief_id,),
        question_type=question,
        target_properties=("condition",),
        priority=0.9,
        constraints=constraints or {},
    )
    return PlanningRequest(
        need=need,
        beliefs=[b] if beliefs is None else beliefs,
        robot_pose=Pose(frame_id=WORLD, position_m=(0.0, -5.0, 0.0)),
        sensors=_sensors(ids),
        is_free=free or (lambda pts: np.ones(len(pts), dtype=bool)),
        predicted_visibility=vis if callable(vis) else (lambda pose, reg: vis),
        navigation_cost=cost
        or (lambda a, c: ResourceCost(time_s=5.0, energy_j=50.0, risk=0.05, travel_m=2.0)),
        now=stamp(1.0, "SIM"),
        prior_views=tuple(views),
        rng=np.random.default_rng(0),
    )


def test_candidates_bounded_and_plan_has_provenance():
    ids = IdFactory(1)
    r = MCBRPlanner(ids).plan(_request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE))
    assert r.plan.status is PlanStatus.PLAN
    assert len(r.table) <= 128
    assert r.provenance.source_type is SourceType.PLAN and r.plan.provenance == r.provenance.record_id
    assert r.plan.primary_action.predicted_visibility > 0


def test_feasibility_filter_runs_before_ranking():
    ids = IdFactory(2)
    scored = []

    def scorer(gap, c, req):
        scored.append(c.action.pose.position_m)
        return 0.0

    free = lambda pts: pts[:, 0] > 0  # noqa: E731  only half the ring is free in the belief map
    r = MCBRPlanner(ids, scorer=scorer, value_gate=False).plan(
        _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE, free=free)
    )
    assert r.plan.rejected and all(
        "POSE_NOT_FREE" in c.reason_codes for c in r.plan.rejected if c.action.pose.position_m[0] <= 0
    )
    assert scored and all(p[0] > 0 for p in scored)  # rejected candidates were never ranked


def test_no_feasible_observation_status():
    ids = IdFactory(3)
    r = MCBRPlanner(ids).plan(
        _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE, cost=lambda a, c: None)
    )
    assert r.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION and r.plan.primary_action is None
    assert all("UNREACHABLE_IN_BELIEF_MAP" in c.reason_codes for c in r.plan.rejected)
    missing = MCBRPlanner(ids).plan(_request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE, beliefs=[]))
    assert missing.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION


def test_need_satisfied_and_not_worth_cost():
    ids = IdFactory(4)
    done = MCBRPlanner(ids).plan(_request(ids, unc(uo=0.1), QuestionType.EXTEND_COVERAGE))
    assert done.plan.status is PlanStatus.NEED_SATISFIED
    pricey = lambda a, c: ResourceCost(time_s=500.0, energy_j=5000.0, risk=0.4, travel_m=300.0)  # noqa: E731
    r = MCBRPlanner(ids).plan(_request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE, cost=pricey))
    assert r.plan.status is PlanStatus.NOT_WORTH_COST


def test_contradiction_prefers_discriminating_sonar():
    ids = IdFactory(5)
    r = MCBRPlanner(ids).plan(_request(ids, unc(uc=0.9), QuestionType.DISCRIMINATE_HYPOTHESES))
    assert r.plan.primary_action.sensor_configuration["modality"] == "SONAR"
    assert r.plan.primary_action.hypothesis_discrimination > 0


def test_epistemic_gain_requires_new_modality():
    ids = IdFactory(6)
    views = [PriorView(position_m=(0.0, 3.0, 0.0), modality="RGB")]
    r = MCBRPlanner(ids).plan(
        _request(
            ids, unc(ue=0.9), QuestionType.CONFIRM_CONDITION, views=views, constraints={"cause": "EPISTEMIC"}
        )
    )
    rows = [x for x in r.table if x["feasible"]]
    assert all(x["gain"]["epistemic"] == 0.0 for x in rows if x["modality"] == "RGB")
    assert any(x["gain"]["epistemic"] > 0.0 for x in rows if x["modality"] == "SONAR")
    constrained = MCBRPlanner(ids).plan(
        _request(
            ids,
            unc(ue=0.9),
            QuestionType.CONFIRM_CONDITION,
            views=views,
            constraints={"require_alternate_modality": True, "alternate_modalities": ["SONAR"]},
        )
    )
    assert all(
        "MODALITY_NOT_ALTERNATE" in c.reason_codes
        for c in constrained.plan.rejected
        if c.action.sensor_configuration["modality"] == "RGB"
    )


def test_redundant_view_has_lower_coverage_gain():
    ids = IdFactory(7)
    planner = MCBRPlanner(ids)
    r = planner.plan(_request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE))
    best = r.plan.primary_action.pose.position_m
    again = planner.plan(
        _request(
            ids,
            unc(uo=0.9),
            QuestionType.EXTEND_COVERAGE,
            views=[PriorView(position_m=best, modality="SONAR")],
        )
    )
    assert again.plan.primary_action.pose.position_m != best


def test_all_baselines_share_interface():
    ids = IdFactory(8)
    planners = make_planners(ids, MCBRConfig(n_azimuth=6))
    assert len(planners) == 10
    for p in planners.values():
        r = p.plan(_request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE))
        assert r.plan.status in (PlanStatus.PLAN, PlanStatus.NOT_WORTH_COST)


def test_mcbr_never_emits_motion_commands():
    pattern = re.compile(r"AllocatedCommand|WrenchCommand|\.send\(|command_gateway|RobotHardwareInterface")
    for path in (ROOT / "conrad" / "active").rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_learned_ranker_forward_backward_and_training():
    cfg = MCBRRankerConfig(
        belief_dim=8,
        uncertainty_dim=4,
        hypothesis_dim=4,
        viewpoint_dim=4,
        observability_dim=4,
        cost_dim=4,
        mission_dim=4,
        context_dim=8,
        model_dim=16,
        head_hidden=8,
        num_blocks=1,
        num_heads=2,
    )
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False
    batch = MCBRBatch(
        candidates=torch.randn(2, 5, cfg.candidate_dim, generator=g),
        candidate_mask=mask,
        context=torch.randn(2, 3, cfg.context_dim, generator=g),
        context_mask=torch.ones(2, 3, dtype=torch.bool),
        oracle_value=torch.randn(2, 5, generator=g),
        visible=(torch.rand(2, 5, generator=g) > 0.5).float(),
    )
    model = build_ranker(cfg)
    out = model(batch)
    assert out.score.shape == (2, 5) and torch.isinf(out.score[1, 3:]).all()
    loss, _ = mcbr_loss(out, batch, cfg)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    losses = train_ranker(build_ranker(cfg), [batch], cfg, epochs=15)
    assert losses[-1] < losses[0]
