"""MCBR: filter-before-rank, stop statuses, cause-specific gain, baselines, learned ranker, no motion."""

import math
import pathlib
import re
from dataclasses import replace

import numpy as np
import torch

from conrad.active import MCBRConfig, MCBRPlanner, PlanningRequest, PriorView, SensorOption, make_planners
from conrad.active.learned import MCBRBatch, MCBRRankerConfig, build_ranker, mcbr_loss, train_ranker
from conrad.active.surface_predictive import QuantityChannel, SurfaceCellPredictive
from conrad.evaluation.decision_experiments.fixtures import make_belief, region, unc
from conrad.orchestration.deliberation import view_pose
from conrad.schemas.decision import InformationNeed, PlanStatus, QuestionType, ResourceCost
from conrad.schemas.frames import WORLD, Pose, quat_from_euler, quat_to_matrix
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
    assert r.plan.primary_action is not None
    assert r.plan.primary_action.predicted_visibility > 0


def test_spatial_predictive_candidate_regions_expand_views_without_exceeding_cap():
    ids = IdFactory(401)
    request = _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE)
    predictive = SurfaceCellPredictive(
        cell_prior=np.array([1.0]),
        channels=(QuantityChannel("corrosion_depth_m", 0.002, 1e-5, 0.001),),
        cell_weights=lambda pose, sensor: np.array([1.0]),
        candidate_regions=(region((0.0, 0.0, 0.0)), region((20.0, 0.0, 0.0))),
    )
    result = MCBRPlanner(ids).plan(replace(request, predictive=predictive))
    positions = [row["position_m"][0] for row in result.table]
    assert len(result.table) <= MCBRConfig().max_candidates
    assert any(x < 10.0 for x in positions)
    assert any(x > 10.0 for x in positions)
    far_result = MCBRPlanner(
        IdFactory(402),
        scorer=lambda gap, candidate, req: candidate.action.pose.position_m[0],
        value_gate=False,
    ).plan(replace(request, predictive=predictive))
    assert far_result.plan.primary_action is not None
    assert far_result.plan.primary_action.target_region.center_m == (20.0, 0.0, 0.0)

    pitched = replace(predictive, candidate_elevations_rad=(1.0,))
    pitched_result = MCBRPlanner(
        IdFactory(403),
        scorer=lambda gap, candidate, req: candidate.action.pose.position_m[2],
        value_gate=False,
    ).plan(replace(request, predictive=pitched))
    assert pitched_result.plan.primary_action is not None
    action = pitched_result.plan.primary_action
    mount = (0.0, 0.15, 0.0)
    vehicle = view_pose(pitched_result.plan, math.pi / 2, preserve_pitch=True, mount_position_m=mount)
    body_rotation = quat_to_matrix(vehicle.orientation_wxyz)
    sensor_origin = np.asarray(vehicle.position_m) + body_rotation @ np.asarray(mount)
    sensor_forward = body_rotation @ quat_to_matrix(quat_from_euler(0.0, 0.0, math.pi / 2))[:, 0]
    expected = np.asarray(action.target_region.center_m) - sensor_origin
    expected /= np.linalg.norm(expected)
    assert np.allclose(sensor_origin, action.pose.position_m)
    assert float(sensor_forward @ expected) > 1 - 1e-12

    prior = PriorView(
        position_m=action.pose.position_m,
        modality="STRUCTURED",
        orientation_wxyz=action.pose.orientation_wxyz,
    )
    next_result = MCBRPlanner(
        IdFactory(404),
        scorer=lambda gap, candidate, req: candidate.action.pose.position_m[2],
        value_gate=False,
    ).plan(replace(request, predictive=pitched, prior_views=(prior,)))
    assert any("DUPLICATE_SPATIAL_VIEW" in rejected.reason_codes for rejected in next_result.plan.rejected)
    assert next_result.plan.primary_action is not None
    assert next_result.plan.primary_action.pose.position_m != action.pose.position_m


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


def test_low_raw_uncertainty_does_not_close_unanswered_calibration_requirement():
    ids = IdFactory(41)
    request = _request(ids, unc(ue=0.2), QuestionType.CONFIRM_CONDITION)
    belief = request.beliefs[0]
    need = request.need.model_copy(
        update={
            "constraints": {"cause": "EPISTEMIC", "calibration_check": True},
            "minimum_belief_revisions": {belief.belief_id: belief.revision + 1},
            "minimum_independent_observation_counts": {
                belief.belief_id: belief.independent_observation_count + 1
            },
        }
    )
    request = replace(request, need=need)
    unresolved = MCBRPlanner(ids, value_gate=False).plan(request)
    assert unresolved.plan.status is PlanStatus.PLAN

    context_only = belief.model_copy(update={"revision": belief.revision + 1})
    still_open = MCBRPlanner(ids, value_gate=False).plan(replace(request, beliefs=[context_only]))
    assert still_open.plan.status is PlanStatus.PLAN

    answered = belief.model_copy(
        update={
            "revision": belief.revision + 1,
            "independent_observation_count": belief.independent_observation_count + 1,
        }
    )
    closed = MCBRPlanner(ids, value_gate=False).plan(replace(request, beliefs=[answered]))
    assert closed.plan.status is PlanStatus.NEED_SATISFIED


def test_unanswered_calibration_requirement_with_no_feasible_view_terminates_explicitly():
    ids = IdFactory(42)
    request = _request(
        ids,
        unc(ue=0.2),
        QuestionType.CONFIRM_CONDITION,
        cost=lambda a, c: None,
    )
    belief = request.beliefs[0]
    request = replace(
        request,
        need=request.need.model_copy(
            update={
                "constraints": {"cause": "EPISTEMIC", "calibration_check": True},
                "minimum_belief_revisions": {belief.belief_id: belief.revision + 1},
                "minimum_independent_observation_counts": {
                    belief.belief_id: belief.independent_observation_count + 1
                },
            }
        ),
    )
    result = MCBRPlanner(ids).plan(request)
    assert result.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION
    assert result.plan.primary_action is None


def test_calibration_uses_valid_current_pose_and_shortest_completion_time():
    ids = IdFactory(43)
    request = _request(
        ids,
        unc(ue=0.2),
        QuestionType.CONFIRM_CONDITION,
        cost=lambda a, c: ResourceCost(
            time_s=0.0 if a.position_m == c.position_m else 12.0,
            energy_j=0.0 if a.position_m == c.position_m else 120.0,
            risk=0.01,
            travel_m=0.0 if a.position_m == c.position_m else 5.0,
        ),
    )
    belief = request.beliefs[0]
    # The target surface extends 0.5 m and the robot is 5 m from its centre: the 4.5 m surface standoff is
    # valid for SONAR (1..8 m), but not RGB (0.5..4 m).
    request = replace(
        request,
        need=request.need.model_copy(
            update={
                "constraints": {"cause": "EPISTEMIC", "calibration_check": True},
                "minimum_belief_revisions": {belief.belief_id: belief.revision + 1},
                "minimum_independent_observation_counts": {
                    belief.belief_id: belief.independent_observation_count + 1
                },
            }
        ),
    )
    result = MCBRPlanner(ids).plan(request)
    action = result.plan.primary_action
    assert result.plan.status is PlanStatus.PLAN and action is not None
    assert action.pose.position_m == request.robot_pose.position_m
    assert action.sensor_configuration["modality"] == "SONAR"
    assert action.pose.orientation_wxyz == request.robot_pose.orientation_wxyz
    assert action.sensor_configuration["calibration_inline"] is True
    assert action.expected_cost.time_s == 2.0


def test_non_calibration_request_does_not_add_or_prefer_station_keep_candidate():
    ids = IdFactory(44)
    request = _request(
        ids,
        unc(ue=0.9),
        QuestionType.CONFIRM_CONDITION,
        cost=lambda a, c: ResourceCost(time_s=0.0, energy_j=0.0, risk=0.0, travel_m=0.0),
    )
    result = MCBRPlanner(ids, value_gate=False).plan(request)
    assert result.plan.primary_action is not None
    assert result.plan.primary_action.pose.position_m != request.robot_pose.position_m
    assert not any(row["position_m"] == list(request.robot_pose.position_m) for row in result.table)


def test_contradiction_prefers_discriminating_sonar():
    ids = IdFactory(5)
    r = MCBRPlanner(ids).plan(_request(ids, unc(uc=0.9), QuestionType.DISCRIMINATE_HYPOTHESES))
    assert r.plan.primary_action is not None
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
    assert r.plan.primary_action is not None
    best = r.plan.primary_action.pose.position_m
    again = planner.plan(
        _request(
            ids,
            unc(uo=0.9),
            QuestionType.EXTEND_COVERAGE,
            views=[PriorView(position_m=best, modality="SONAR")],
        )
    )
    assert again.plan.primary_action is not None
    assert again.plan.primary_action.pose.position_m != best


def test_all_baselines_share_interface():
    ids = IdFactory(8)
    planners = make_planners(ids, MCBRConfig(n_azimuth=6))
    assert len(planners) == 14  # A-B0..A-B10 (no A-B8) + A-B5b, A-B6b, A-B11, A-B12
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


def _order(name, request, ids):
    """Ranked candidate poses (action ids are minted per planner, so they are not comparable)."""
    result = make_planners(ids)[name].plan(request)
    return [tuple(round(x, 6) for x in row["position_m"]) for row in result.table if row["feasible"]]


def test_a_b5_entropy_nbv_is_not_independent_of_coverage_when_uncertainty_is_per_need():
    """A-B5 multiplies a candidate-INDEPENDENT uncertainty level, so it ranks exactly as A-B2 coverage."""
    ids = IdFactory(31)
    request = _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE)
    assert _order("A-B5_entropy_nbv", request, IdFactory(41)) == _order(
        "A-B2_coverage", request, IdFactory(42)
    )


def test_a_b5b_entropy_nbv_uses_the_candidates_own_predicted_entropy_reduction():
    """A-B5b keeps the NBV novelty shape but takes uncertainty from the candidate, so it is independent."""
    from conrad.active.predictive import PredictedOutcome, ScalarBelief

    ids = IdFactory(32)
    # one view already taken, so the novelty term actually varies between candidates
    request = _request(
        ids,
        unc(uo=0.9),
        QuestionType.EXTEND_COVERAGE,
        views=[PriorView(position_m=(4.0, 0.0, 0.0), modality="RGB")],
    )

    class _Predictive:
        scalars = (ScalarBelief(key="q", mean=0.5, var=1.0),)
        hypotheses = ()
        epistemic = 0.0
        used_modalities: frozenset[str] = frozenset()

        def predict(self, pose, sensor):
            # a candidate-dependent measurement quality: the closer to +X, the sharper the reading
            std = 1.0 / (1.0 + max(0.0, float(pose.position_m[0])))
            return PredictedOutcome(noise_std={"q": std})

    request.predictive = _Predictive()
    assert _order("A-B5b_entropy_nbv_predictive", request, IdFactory(43)) != _order(
        "A-B2_coverage", request, IdFactory(44)
    )
    assert _order("A-B5b_entropy_nbv_predictive", request, IdFactory(45)) != _order(
        "A-B6b_bayes_eig", request, IdFactory(46)
    )
