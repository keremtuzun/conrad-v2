"""The manoeuvre runner refuses physical control by default and commands only through the gateway."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from conrad.robotics.hardware.identification.protocol import MotionReferenceKind
from conrad.robotics.hardware.identification.runner import (
    HardwareControlRefusedError,
    IdentificationSafetyEnvelope,
    ManoeuvreRunner,
    ReferenceSource,
    RunnerConfig,
    StateSource,
    check_hardware_permission,
)
from conrad.robotics.hardware.interface import PhysicalRobotHardware, RobotHardwareInterface
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig

RUNNER = (
    Path(__file__).resolve().parents[4] / "conrad" / "robotics" / "hardware" / "identification" / "runner.py"
)


class SpyPhysical(PhysicalRobotHardware):
    """Physical adapter double that records every call (and never moves anything)."""

    adapter_name = "spy_physical"

    def __init__(self):
        self.calls = []

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and not name.startswith("_") and name not in ("calls",):
            object.__getattribute__(self, "calls").append(name)
        return attr


class _Ref:
    name = "tracking"
    kind = MotionReferenceKind.EXTERNAL_TRACKING


ENVELOPE = IdentificationSafetyEnvelope(
    max_abs_command=0.2,
    max_segment_duration_s=30.0,
    max_speed_mps=0.3,
    max_depth_m=2.0,
    max_abs_roll_pitch_rad=0.3,
    set_by="test fixture",
)


def _make(hw: RobotHardwareInterface, config: RunnerConfig, reference: Any = None) -> ManoeuvreRunner:
    """Build a runner from deliberate stand-ins.

    Every test here asserts the runner refuses *before* it touches the collaborators, so the
    gateway/config/ids/state/ids arguments are never dereferenced; they are cast to the declared
    types to keep the refusal the only thing under test.
    """
    return ManoeuvreRunner(
        hw,
        cast(CommandGateway, None),
        cast(RobotConfig, None),
        cast(IdFactory, None),
        lambda s: None,
        cast(ReferenceSource, reference or _Ref()),
        cast(StateSource, None),
        cast(UUID, None),
        cast(UUID, None),
        config,
    )


def test_hardware_control_is_off_by_default():
    cfg = RunnerConfig()
    assert cfg.hardware_control_enabled is False
    assert cfg.operator_acknowledged is False
    assert cfg.safety_envelope is None


@pytest.mark.parametrize(
    ("config", "missing"),
    [
        (RunnerConfig(), "hardware_control_enabled is false"),
        (RunnerConfig(hardware_control_enabled=True, operator_acknowledged=True), "no safety envelope"),
        (RunnerConfig(hardware_control_enabled=True, safety_envelope=ENVELOPE), "operator acknowledgement"),
        (RunnerConfig(operator_acknowledged=True, safety_envelope=ENVELOPE), "hardware_control_enabled"),
    ],
)
def test_physical_runner_refuses_before_touching_hardware(config, missing):
    hw = SpyPhysical()
    with pytest.raises(HardwareControlRefusedError, match=missing):
        _make(hw, config)
    assert "send" not in hw.calls
    assert "set_thruster_commands" not in hw.calls
    assert hw.calls == []  # no method of the physical adapter was even called


def test_physical_runner_refuses_simulator_truth_reference():
    cfg = RunnerConfig(hardware_control_enabled=True, operator_acknowledged=True, safety_envelope=ENVELOPE)
    check_hardware_permission(SpyPhysical(), cfg)  # all three flags: the permission gate itself opens

    class SimRef(_Ref):
        kind = MotionReferenceKind.SIMULATOR_TRUTH

    with pytest.raises(HardwareControlRefusedError, match="simulator truth"):
        _make(SpyPhysical(), cfg, SimRef())


def test_runner_requires_the_real_command_gateway():
    cfg = RunnerConfig(hardware_control_enabled=True, operator_acknowledged=True, safety_envelope=ENVELOPE)
    with pytest.raises(TypeError, match="command_gateway"):
        _make(SpyPhysical(), cfg)


def test_envelope_has_no_default_limits():
    with pytest.raises(ValueError):
        IdentificationSafetyEnvelope()  # type: ignore[call-arg]


def test_runner_source_never_calls_send():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = [
        n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "send" not in calls and "set_thruster_commands" not in calls
    assert "submit" in calls
