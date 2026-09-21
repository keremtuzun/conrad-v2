"""The view execution ledger: a view is credited only when it was flown, and abandonment is visible."""

import math
from uuid import uuid4

import pytest

from conrad.orchestration.view_execution import (
    ABANDONED_ROUTE_BLOCKED,
    FLOWN,
    ViewCommand,
    ViewLedger,
    summarize,
)

AIM = (0.0, 0.0, 0.0)


def _command(position=(2.0, 0.0, 0.0), yaw=math.pi, dwell=2.0, pos_tol=0.35, ang_tol=0.3):
    return ViewCommand(
        plan_id=uuid4(),
        action_id=uuid4(),
        need_id=uuid4(),
        position_m=position,
        yaw_rad=yaw,
        aim_point_m=AIM,
        position_tolerance_m=pos_tol,
        orientation_tolerance_rad=ang_tol,
        dwell_s=dwell,
    )


def _ledger(start=(9.0, 0.0, 0.0)):
    led = ViewLedger()
    led.open(_command(), uuid4(), "INSPECT", 0, start)
    return led


def test_a_view_that_is_never_reached_is_abandoned_with_a_visible_reason():
    led = _ledger()
    for i in range(1, 11):  # drifts towards the view but never arrives
        led.update(int(i * 1e9), (9.0 - 0.2 * i, 0.0, 0.0), math.pi, 1.0)
    rec = led.close(int(11e9), ABANDONED_ROUTE_BLOCKED, "EXECUTING")
    assert rec is not None
    assert rec.outcome == ABANDONED_ROUTE_BLOCKED and not rec.flown and not rec.reached
    assert rec.closest_approach_m == pytest.approx(5.0, abs=1e-6)
    abandoned = led.abandoned()
    assert len(abandoned) == 1
    assert abandoned[0].reason == ABANDONED_ROUTE_BLOCKED
    assert abandoned[0].position_m == (2.0, 0.0, 0.0)
    assert summarize(led.records)["fraction_flown"] == 0.0


def test_an_observation_is_credited_only_inside_the_declared_pose_and_aim_tolerance():
    """At the commanded position but pointing the wrong way is not a flown view."""
    led = _ledger(start=(2.0, 0.0, 0.0))
    for i in range(1, 11):
        led.update(int(i * 1e9), (2.0, 0.0, 0.0), 0.0, 1.0)  # inside pose tolerance, aimed 180 deg away
    rec = led.records[0] if led.records else led.open_record
    assert rec is not None
    assert rec.reached is True  # the POSE tolerance was met
    assert rec.aimed is False  # the AIM tolerance was not
    assert rec.time_within_tolerance_s == 0.0
    assert rec.time_within_position_tolerance_s == pytest.approx(10.0)
    assert rec.min_boresight_error_in_position_rad == pytest.approx(math.pi, abs=1e-6)


def test_a_view_flown_to_the_declared_pose_and_aim_accumulates_dwell():
    led = _ledger(start=(2.0, 0.0, 0.0))
    for i in range(1, 6):
        led.update(int(i * 1e9), (2.0, 0.0, 0.0), math.pi, 1.0)
    rec = led.open_record
    assert rec is not None and rec.aimed and rec.reached
    assert rec.time_within_tolerance_s == pytest.approx(5.0)
    assert rec.time_within_tolerance_s >= rec.dwell_s
    led.close(int(6e9), FLOWN, "COMPLETE")
    assert summarize(led.records)["fraction_flown"] == 1.0
    assert led.abandoned() == ()


def test_pose_inside_tolerance_but_outside_aim_is_not_counted_as_flown_by_the_summary():
    led = _ledger(start=(2.0, 0.0, 0.0))
    for i in range(1, 4):
        led.update(int(i * 1e9), (2.0, 0.0, 0.0), 0.0, 1.0)
    led.close(int(4e9), "ARRIVED_DWELL_SHORT", "COMPLETE")
    s = summarize(led.records)
    assert s["flown"] == 0
    assert s["reached_tolerance"] == 1
    assert s["reached_pose_and_aim_tolerance"] == 0


def test_ledger_with_no_open_record_closes_to_nothing():
    led = ViewLedger()
    assert led.close(0, FLOWN) is None
    assert summarize(led.records)["accepted_views"] == 0
    assert summarize(led.records)["fraction_flown"] is None
