"""RobotConfig loading and lane validation. Missing physical facts stay OPEN (INV-ARCH-08)."""

from __future__ import annotations

from pathlib import Path

import yaml

from conrad.schemas.robot import RobotConfig
from conrad.settings import REPO_ROOT, ExecutionLane


class RobotConfigError(ValueError):
    pass


def load_robot_config(path: str | Path) -> RobotConfig:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise RobotConfigError(f"RobotConfig not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    try:
        return RobotConfig.model_validate(data)
    except Exception as exc:
        raise RobotConfigError(f"invalid RobotConfig {p}: {exc}") from exc


def validate_for_lane(config: RobotConfig, lane: ExecutionLane) -> list[str]:
    """Problems that forbid using ``config`` in ``lane``. Empty list means permitted."""
    problems: list[str] = []
    if lane in (ExecutionLane.SIMULATION, ExecutionLane.DEV, ExecutionLane.HIL):
        problems += [
            f"OPEN parameter needed by simulation: {name}"
            for name in config.open_parameters()
            if not name.startswith("safety.")
        ]
    if lane is ExecutionLane.PHYSICAL:
        problems += [f"not physically grounded: {name}" for name in config.ungrounded_parameters()]
    if not config.thrusters:
        problems.append("thruster layout unknown; allocation impossible")
    return problems
