"""CC-08, SS-07, SS-10: the Command Gateway fails closed on every invalid actuator path."""

from __future__ import annotations

import pytest

from conrad.persistence.repository import Repository
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.hardware.interface import HardwareUnavailableError, PhysicalRobotHardware
from conrad.runtime.command_gateway import ALLOCATOR_PRODUCER, CommandGateway, GatewayReason
from conrad.schemas.robot import AllocatedCommand, SafetyAuthorization
from conrad.settings import CommandMode, ConradSettings, ExecutionLane, RuntimeSettings, load_settings
from tests.fixtures.fake_system import FakeFullSystem


@pytest.fixture
def system(repo: Repository) -> FakeFullSystem:
    return FakeFullSystem(repo, seed=3)


def good(system: FakeFullSystem, **over) -> AllocatedCommand:
    now = system.clock.ns()
    cid = system.ids.new()
    fields = {
        "command_id": cid,
        "mission_id": system.mission_id,
        "run_id": system.run_id,
        "trace_id": system.ids.new(),
        "belief_snapshot_id": None,
        "robot_config_digest": system.config.content_digest(),
        "clock_domain": "SIM",
        "issued_time_ns": now,
        "deadline_ns": now + 250_000_000,
        "thruster_commands": {t.thruster_id: 0.1 for t in system.config.thrusters},
        "source_wrench_id": system.ids.new(),
        "provenance_root": system.ids.new(),
        "producer": ALLOCATOR_PRODUCER,
        "safety_authorization": SafetyAuthorization(
            authorization_id=system.ids.new(),
            command_id=cid,
            supervisor_version="t",
            issued_time_ns=now,
            safety_state="NORMAL",
        ),
    }
    fields.update(over)
    return AllocatedCommand(**fields)


def test_valid_command_is_accepted_once(system: FakeFullSystem) -> None:
    cmd = good(system)
    assert system.gateway.submit(cmd).accepted
    replay = system.gateway.submit(cmd)
    assert not replay.accepted and GatewayReason.DUPLICATE in replay.reason_codes
    assert len(system.robot.received) == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda s: {"robot_config_digest": "0" * 64}, GatewayReason.CONFIG_DIGEST),
        (lambda s: {"mission_id": s.ids.new()}, GatewayReason.UNKNOWN_MISSION),
        (lambda s: {"run_id": s.ids.new()}, GatewayReason.UNKNOWN_RUN),
        (lambda s: {"clock_domain": "WALL"}, GatewayReason.CLOCK_DOMAIN),
        (lambda s: {"safety_authorization": None}, GatewayReason.NO_AUTH),
        (lambda s: {"producer": "conrad.decision.egdc"}, GatewayReason.PRODUCER),
        (lambda s: {"thruster_commands": {"H1": 0.1}}, GatewayReason.ACTUATOR_COUNT),
        (
            lambda s: {"thruster_commands": {t.thruster_id: 1.7 for t in s.config.thrusters}},
            GatewayReason.ENVELOPE,
        ),
        (
            lambda s: {"thruster_commands": {t.thruster_id: float("nan") for t in s.config.thrusters}},
            GatewayReason.ENVELOPE,
        ),
        (lambda s: {"schema_version": "9.0.0"}, GatewayReason.SCHEMA),
    ],
)
def test_invalid_commands_rejected(system: FakeFullSystem, mutation, reason: str) -> None:
    ack = system.gateway.submit(good(system, **mutation(system)))
    assert not ack.accepted and reason in ack.reason_codes
    assert system.robot.received == []


def test_expired_command_rejected(system: FakeFullSystem) -> None:
    cmd = good(system)
    system.clock.t += 1.0
    ack = system.gateway.submit(cmd)
    assert not ack.accepted and GatewayReason.EXPIRED in ack.reason_codes and system.robot.received == []


def test_safety_state_forbidding_motion_and_mismatched_authorization(system: FakeFullSystem) -> None:
    cmd = good(system)
    assert cmd.safety_authorization is not None
    stop = cmd.model_copy(
        update={
            "safety_authorization": cmd.safety_authorization.model_copy(
                update={"safety_state": "EMERGENCY_STOP"}
            )
        }
    )
    assert GatewayReason.AUTH_STATE in system.gateway.submit(stop).reason_codes
    other = cmd.model_copy(
        update={
            "safety_authorization": cmd.safety_authorization.model_copy(
                update={"command_id": system.ids.new()}
            )
        }
    )
    assert GatewayReason.AUTH_MISMATCH in system.gateway.submit(other).reason_codes


def test_raw_and_untyped_payloads_rejected(system: FakeFullSystem) -> None:
    for raw in ({"pwm": [1500] * 8}, [0.5] * 8, b"\x01\x02", good(system).model_dump()):
        ack = system.gateway.submit(raw)
        assert not ack.accepted and ack.reason_codes == (GatewayReason.NOT_TYPED,)
    assert system.robot.received == []


def test_unknown_peer_rejected(system: FakeFullSystem) -> None:
    ack = system.gateway.submit(good(system), peer="203.0.113.9")
    assert not ack.accepted and GatewayReason.UNKNOWN_PEER in ack.reason_codes


def test_stale_state_rejected(system: FakeFullSystem) -> None:
    gw = CommandGateway(
        system.robot,
        system.config,
        RuntimeSettings(command_mode=CommandMode.SIMULATED),
        ExecutionLane.SIMULATION,
        system.mission_id,
        system.run_id,
        state_age_s=lambda: 5.0,
    )
    assert GatewayReason.STALE_STATE in gw.submit(good(system)).reason_codes
    gw_unknown = CommandGateway(
        system.robot,
        system.config,
        RuntimeSettings(command_mode=CommandMode.SIMULATED),
        ExecutionLane.SIMULATION,
        system.mission_id,
        system.run_id,
        state_age_s=lambda: None,
    )
    assert GatewayReason.STALE_STATE in gw_unknown.submit(good(system)).reason_codes


def test_command_mode_defaults_to_disabled(system: FakeFullSystem) -> None:
    assert (
        RuntimeSettings().command_mode is CommandMode.DISABLED and RuntimeSettings().hardware_enable is False
    )
    gw = CommandGateway(
        system.robot, system.config, RuntimeSettings(), ExecutionLane.DEV, system.mission_id, system.run_id
    )
    assert GatewayReason.MODE_DISABLED in gw.submit(good(system)).reason_codes


def test_ss10_physical_hardware_is_unreachable_without_every_prerequisite(system: FakeFullSystem) -> None:
    physical = PhysicalRobotHardware()
    with pytest.raises(HardwareUnavailableError):
        physical.send(good(system))
    with pytest.raises(ValueError):  # simulation lane cannot select hardware command mode
        ConradSettings.model_validate(
            {"run": {"lane": "simulation"}, "runtime": {"command_mode": "hardware"}}
        )
    with pytest.raises(ValueError):
        RuntimeSettings(bind_address="0.0.0.0")
    example = load_settings("configs/runtime/physical_example.yaml")
    assert example.runtime.hardware_enable is False and example.runtime.hil_evidence_ref is None
    template = load_robot_config(example.runtime.robot_config)
    assert template.ungrounded_parameters() and not template.thrusters


def test_explicit_zero_command_accepted_in_stop_states_but_motion_refused(system: FakeFullSystem) -> None:
    for state in ("EMERGENCY_STOP", "RECOVER", "RETURN"):
        cmd = good(system, thruster_commands={t.thruster_id: 0.0 for t in system.config.thrusters})
        assert cmd.safety_authorization is not None
        stop = cmd.model_copy(
            update={
                "safety_authorization": cmd.safety_authorization.model_copy(update={"safety_state": state})
            }
        )
        assert system.gateway.submit(stop).accepted, state
        move = good(system)
        assert move.safety_authorization is not None
        moving = move.model_copy(
            update={
                "safety_authorization": move.safety_authorization.model_copy(update={"safety_state": state})
            }
        )
        assert GatewayReason.AUTH_STATE in system.gateway.submit(moving).reason_codes
