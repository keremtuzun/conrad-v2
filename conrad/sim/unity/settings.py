"""Typed view of ``settings.sim["unity"]`` loaded from ``configs/sim/unity_*.yaml``.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from conrad.adapters.unity.conversion import UnityBridgeConfig
from conrad.schemas.base import ConradModel
from conrad.settings import ConradSettings, load_settings


class UnitySimSettings(ConradModel):
    bridge: UnityBridgeConfig = UnityBridgeConfig()
    truth_endpoint: str | None = None
    physics_dt_ns: int = Field(default=5_000_000, gt=0)
    player_path: str | None = Field(default=None, description="built Unity player; machine-local")
    robot_config_export: str = "artifacts/unity/robot_config.json"


def unity_settings(settings: ConradSettings) -> UnitySimSettings:
    raw = settings.sim.get("unity")
    if raw is None:
        raise KeyError("settings.sim has no 'unity' section; use configs/sim/unity_*.yaml")
    return UnitySimSettings.model_validate(raw)


def load_unity_settings(config_path: str | Path) -> tuple[ConradSettings, UnitySimSettings]:
    settings = load_settings(config_path)
    return settings, unity_settings(settings)
