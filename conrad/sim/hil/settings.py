"""Load ``HilConfig`` from ``configs/runtime/hil_*.yaml`` (section ``sim.hil``).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from pathlib import Path

from conrad.settings import ConradSettings, ExecutionLane, load_settings
from conrad.sim.hil.harness import HilConfig


class HilSettingsError(ValueError):
    pass


def hil_config(settings: ConradSettings) -> HilConfig:
    if settings.run.lane is not ExecutionLane.HIL:
        raise HilSettingsError(f"HIL configs must run in the hil lane, got {settings.run.lane.value}")
    raw = settings.sim.get("hil")
    if raw is None:
        raise HilSettingsError("settings.sim has no 'hil' section")
    return HilConfig.model_validate(raw)


def load_hil_config(path: str | Path) -> tuple[ConradSettings, HilConfig]:
    settings = load_settings(path)
    return settings, hil_config(settings)
