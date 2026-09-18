"""Shared helpers for the Unity / HIL / hardware-stub tests (test-only)."""

from __future__ import annotations

from uuid import UUID

from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import AllocatedCommand, RobotConfig, SafetyAuthorization

ALLOCATOR = "conrad.robotics.allocation"


def sim_robot_config() -> RobotConfig:
    return load_robot_config("configs/robot/sim_reference.yaml")


def make_command(
    ids: IdFactory,
    config: RobotConfig,
    mission_id: UUID,
    run_id: UUID,
    now_ns: int,
    values: dict[str, float] | None = None,
    clock_domain: str = "SIM",
    authorized: bool = True,
    digest: str | None = None,
) -> AllocatedCommand:
    command_id = ids.new()
    thrusters = dict.fromkeys((t.thruster_id for t in config.thrusters), 0.0)
    thrusters.update(values or {})
    auth = (
        SafetyAuthorization(
            authorization_id=ids.new(),
            command_id=command_id,
            supervisor_version="test",
            issued_time_ns=now_ns,
            safety_state="NORMAL",
        )
        if authorized
        else None
    )
    return AllocatedCommand(
        command_id=command_id,
        mission_id=mission_id,
        run_id=run_id,
        trace_id=ids.new(),
        belief_snapshot_id=None,
        robot_config_digest=digest or config.content_digest(),
        clock_domain=clock_domain,
        issued_time_ns=now_ns,
        deadline_ns=now_ns + 200_000_000,
        thruster_commands=thrusters,
        source_wrench_id=ids.new(),
        provenance_root=ids.new(),
        producer=ALLOCATOR,
        safety_authorization=auth,
    )
