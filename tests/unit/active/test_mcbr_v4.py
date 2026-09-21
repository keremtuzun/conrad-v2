"""MCBR V4: execution-aware feasibility, sequence value, frozen configs and the no-truth boundary."""

import ast
import pathlib

import numpy as np
import pytest

from conrad.active import MCBRConfig, MCBRPlanner, PlanningRequest, SensorOption
from conrad.active.candidates import R_VISIBILITY
from conrad.active.gap import AbandonedView, build_knowledge_gaps
from conrad.active.production import FrozenConfigError, load_frozen, planner_digest
from conrad.active.surface_predictive import QuantityChannel, SurfaceCellPredictive, gaussian_info
from conrad.active.v4 import (
    R_APPROACH_ABANDONED,
    R_TIME_BUDGET,
    ExecutionFilter,
    V4Config,
    v4_planner,
)
from conrad.evaluation.decision_experiments.fixtures import make_belief, region, unc
from conrad.schemas.decision import InformationNeed, PlanStatus, QuestionType, ResourceCost
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _sensor(ids):
    return SensorOption(
        sensor_id=ids.new(),
        modality="RGB",
        min_range_m=0.5,
        max_range_m=4.0,
        discrimination={"default": 0.2},
        measurement_quality=0.7,
    )


def _request(ids, **kw):
    b = make_belief(ids, uncertainty=unc(uo=0.9), support=region((0.0, 0.0, 0.0)))
    need = InformationNeed(
        need_id=ids.new(),
        trace_id=ids.new(),
        target_belief_ids=(b.belief_id,),
        question_type=QuestionType.EXTEND_COVERAGE,
        target_properties=("condition",),
        priority=0.9,
    )
    base = {
        "need": need,
        "beliefs": [b],
        "robot_pose": Pose(frame_id=WORLD, position_m=(0.0, -5.0, 0.0)),
        "sensors": (_sensor(ids),),
        "is_free": lambda pts: np.ones(len(pts), dtype=bool),
        "predicted_visibility": lambda pose, reg: 0.8,
        "navigation_cost": lambda a, c: ResourceCost(time_s=5.0, energy_j=50.0, risk=0.05, travel_m=2.0),
        "now": __import__("conrad.schemas.timebase", fromlist=["stamp"]).stamp(1.0, "SIM"),
        "rng": np.random.default_rng(0),
    }
    base.update(kw)
    return PlanningRequest(**base)


# ------------------------------------------------------------------ sequence value is not a naive sum
def _model(n_cells=16):
    channel = QuantityChannel(
        quantity="crack_length_m",
        unread_mean=0.05,
        unread_var=0.01,
        look_std_unread=0.02,
        detection_probability=1.0,
    )
    prior = np.full(n_cells, 1.0 / n_cells)
    return SurfaceCellPredictive(
        cell_prior=prior, channels=(channel,), cell_weights=lambda pose, sensor: np.zeros(n_cells)
    )


def test_sequence_value_of_overlapping_views_is_below_the_naive_sum():
    """Two views of the same surface do not add: the defect is ONE worst case over the whole surface.

    The mission-relevant quantity is a worst case, so the value of a sequence is driven by the UNION of
    the cells the views resolve. A planner that adds per-view values double counts the overlap.
    """
    m = _model(16)
    a = np.zeros(16)
    a[0:8] = 1.0
    b = np.zeros(16)
    b[4:12] = 1.0  # overlaps a on cells 4..7
    union = np.maximum(a, b)
    assert m.phi(union) < m.phi(a) + m.phi(b) - 1e-9
    ch = m.channels[0]
    info = gaussian_info(ch.unread_var, ch.look_std_unread)
    assert m.phi(union) * info < (m.phi(a) + m.phi(b)) * info - 1e-9


def test_sequence_value_of_disjoint_views_is_the_sum():
    m = _model(16)
    a = np.zeros(16)
    a[0:4] = 1.0
    b = np.zeros(16)
    b[8:12] = 1.0
    assert m.phi(np.maximum(a, b)) == pytest.approx(m.phi(a) + m.phi(b), abs=1e-12)


# ------------------------------------------------------------------ abandoned views are re-planned against
def _abandoned_at(position, reason="ABANDONED_ROUTE_BLOCKED", closest=4.0):
    return AbandonedView(
        position_m=position, aim_point_m=(0.0, 0.0, 0.0), reason=reason, closest_approach_m=closest
    )


def test_unreachable_view_is_refused_with_a_visible_reason_and_a_different_view_is_planned():
    ids = IdFactory(11)
    cfg = MCBRConfig(n_azimuth=8, elevations_rad=(0.0,), standoff_fractions=(0.5,))
    planner = v4_planner(ids, cfg, V4Config())
    first = planner.plan(_request(ids))
    assert first.plan.status is PlanStatus.PLAN and first.plan.primary_action is not None
    chosen = first.plan.primary_action.pose.position_m

    ids2 = IdFactory(11)
    planner2 = v4_planner(ids2, cfg, V4Config())
    again = planner2.plan(_request(ids2, abandoned_views=(_abandoned_at(chosen),)))
    assert again.plan.status is PlanStatus.PLAN and again.plan.primary_action is not None
    # the view that was never reached is refused, with its reason, and a different one is planned
    refused = [r for r in again.plan.rejected if R_APPROACH_ABANDONED in r.reason_codes]
    assert refused, "the abandoned view must be refused with an explicit reason code"
    assert again.plan.primary_action.pose.position_m != chosen
    rows = {tuple(r["position_m"]): r for r in again.table}
    assert rows[tuple(chosen)]["feasible"] is False


def test_a_view_that_was_reached_is_not_treated_as_an_unreachable_approach():
    """A view abandoned ON THE SPOT was reached; its pose is not the reason it failed."""
    f = ExecutionFilter(V4Config())
    ids = IdFactory(12)
    req = _request(ids, abandoned_views=(_abandoned_at((1.0, 0.0, 0.0), closest=0.05),))
    raw = MCBRPlanner(ids, MCBRConfig()).generator.generate(
        req.need.constraints.get("target_region") or req.beliefs[0].spatial_support, req.sensors
    )[0]
    near = type(raw)(
        Pose(frame_id=WORLD, position_m=(1.0, 0.0, 0.0)),
        raw.sensor,
        raw.configuration,
        raw.standoff_m,
        raw.azimuth_rad,
        raw.elevation_rad,
        raw.index,
    )
    gap = build_knowledge_gaps(req.need, req.beliefs, MCBRConfig())[0]
    assert f(near, 0.8, ResourceCost(time_s=1.0, energy_j=1.0, risk=0.0, travel_m=1.0), req, gap) == ()
    # the same candidate IS refused when the abandonment ended far from the commanded pose
    far = _request(ids, abandoned_views=(_abandoned_at((1.0, 0.0, 0.0), closest=4.0),))
    reasons = f(near, 0.8, ResourceCost(time_s=1.0, energy_j=1.0, risk=0.0, travel_m=1.0), far, gap)
    assert reasons == (R_APPROACH_ABANDONED,)


# ------------------------------------------------------------------ filter before rank
def test_extra_filter_runs_before_ranking_and_cannot_rescue_a_refused_candidate():
    ids = IdFactory(13)
    cfg = MCBRConfig(n_azimuth=6, elevations_rad=(0.0,), standoff_fractions=(0.5,), min_visibility=0.9)
    planner = v4_planner(ids, cfg, V4Config())
    r = planner.plan(_request(ids, predicted_visibility=lambda pose, reg: 0.1))
    assert r.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION
    assert r.plan.primary_action is None
    assert all(R_VISIBILITY in row["reason_codes"] for row in r.table)
    assert all(row["feasible"] is False for row in r.table)


def test_budget_filter_refuses_a_view_that_does_not_fit_the_remaining_mission_time():
    ids = IdFactory(14)
    cfg = MCBRConfig(n_azimuth=6, elevations_rad=(0.0,), standoff_fractions=(0.5,))
    planner = v4_planner(ids, cfg, V4Config())
    r = planner.plan(_request(ids, time_remaining_s=1.0))
    assert r.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION
    assert any(R_TIME_BUDGET in row["reason_codes"] for row in r.table)
    # with the budget undeclared the same request plans normally: V4 adds a test, it does not remove one
    ids2 = IdFactory(14)
    assert v4_planner(ids2, cfg, V4Config()).plan(_request(ids2)).plan.status is PlanStatus.PLAN


def test_v4_planner_emits_no_motion_command():
    ids = IdFactory(15)
    r = v4_planner(ids, MCBRConfig(), V4Config()).plan(_request(ids))
    dumped = r.plan.model_dump(mode="json")
    for banned in ("thruster", "wrench", "force_n", "torque_nm", "command_id"):
        assert banned not in str(dumped)


# ------------------------------------------------------------------ no truth reaches the runtime planner
FORBIDDEN = ("conrad.twins", "conrad.sim", "conrad.evaluation", "conrad.schemas.truth")
V4_MODULES = (
    ROOT / "conrad" / "active" / "v4.py",
    ROOT / "conrad" / "orchestration" / "view_execution.py",
)


@pytest.mark.parametrize("path", V4_MODULES, ids=lambda p: p.name)
def test_v4_modules_import_no_truth_side_package(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    assert not [n for n in names if any(n.startswith(f) for f in FORBIDDEN)]


def test_abandoned_view_carries_only_belief_side_fields():
    fields = set(AbandonedView.model_fields)
    assert fields == {"position_m", "aim_point_m", "reason", "closest_approach_m", "plan_id"}


# ------------------------------------------------------------------ frozen configs
def test_v3_frozen_config_still_loads():
    raw = load_frozen(ROOT / "configs" / "active" / "mcbr_frozen_v3.yaml")
    assert raw["planner"]["selected"] == "V-bayes_eig_ratio"
    assert planner_digest(raw["planner"]) == raw["config_digest"]


V4_FROZEN = ROOT / "configs" / "active" / "mcbr_frozen_v4.yaml"


@pytest.mark.skipif(not V4_FROZEN.exists(), reason="no V4 planner has been frozen yet")
def test_v4_frozen_config_loads_and_fails_closed_when_edited(tmp_path):
    raw = load_frozen(V4_FROZEN)
    assert planner_digest(raw["planner"]) == raw["config_digest"]
    edited = V4_FROZEN.read_text(encoding="utf-8").replace("cost_weight: 1.0", "cost_weight: 2.0")
    assert edited != V4_FROZEN.read_text(encoding="utf-8"), "the edit must actually change the file"
    copy = tmp_path / "mcbr_frozen_v4.yaml"
    copy.write_text(edited, encoding="utf-8")
    with pytest.raises(FrozenConfigError):
        load_frozen(copy)


# ------------------------------------------------------------------ empty plans are reported
def test_empty_plan_is_reported_with_its_rejection_reasons():
    """A plan with no primary action used to return silently; it is the candidate-supply signal."""
    from uuid import uuid4

    from conrad.orchestration.deliberation import AdoptedPlan
    from conrad.orchestration.routing import empty_plan_row

    ids = IdFactory(21)
    cfg = MCBRConfig(n_azimuth=6, elevations_rad=(0.0,), standoff_fractions=(0.5,), min_visibility=0.9)
    req = _request(ids, predicted_visibility=lambda pose, reg: 0.1)
    result = v4_planner(ids, cfg, V4Config()).plan(req)
    assert result.plan.status is PlanStatus.NO_FEASIBLE_OBSERVATION
    row = empty_plan_row(req.need, AdoptedPlan(result.plan, uuid4(), uuid4(), result.table), 12.5)
    assert row["status"] == "NO_FEASIBLE_OBSERVATION"
    assert row["feasible"] == 0 and row["candidates"] == len(result.table)
    assert row["rejection_reasons"][R_VISIBILITY] == len(result.table)
    assert row["t_s"] == 12.5


def test_empty_plan_row_reports_an_unavailable_planner_module_too():
    from conrad.orchestration.routing import empty_plan_row

    ids = IdFactory(22)
    row = empty_plan_row(_request(ids).need, None, 1.0)
    assert row["status"] == "MODULE_UNAVAILABLE"
    assert row["candidates"] == 0 and row["rejection_reasons"] == {}
