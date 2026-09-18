"""Export a RobotConfig to the JSON document the Unity ``RobotConfigLoader`` reads.

* Refuses OPEN parameters (the C# loader refuses them too; both sides fail closed).
* Values stay in the Conrad BODY convention (+X forward, +Y left, +Z up, SI). The C# side converts
  with the same axis map as ``conrad.adapters.unity.frames`` so there is one documented conversion.
* Embeds ``robot_config_digest`` = ``RobotConfig.content_digest()``. Unity echoes it in the
  handshake; the Python client refuses to talk to a simulator that loaded anything else.
* Each value carries its ``source`` so Unity can report its simulation validity level honestly.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from conrad.adapters.unity.frames import UNITY_FRAME_CONVENTION
from conrad.schemas.base import VersionedModel
from conrad.schemas.robot import OpenParameterError, RobotConfig, Sourced

UNITY_ROBOT_CONFIG_FORMAT = "conrad.unity.robot_config.v1"
BODY_CONVENTION = "CONRAD_BODY_RH_X_FORWARD_Y_LEFT_Z_UP"


def _sourced(s: Sourced[Any]) -> dict[str, Any]:
    value = s.value
    if isinstance(value, tuple):
        value = list(value)
    return {"value": value, "units": s.units, "source": s.source.value, "sigma": s.uncertainty_1sigma}


def _walk(obj: Any) -> Any:
    if isinstance(obj, Sourced):
        return _sourced(obj)
    if isinstance(obj, VersionedModel):
        return {
            name: _walk(getattr(obj, name)) for name in type(obj).model_fields if name != "schema_version"
        }
    if isinstance(obj, tuple):
        return [_walk(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, UUID):
        return str(obj)
    return obj


def robot_config_to_unity(config: RobotConfig, *, allow_open_safety: bool = True) -> dict[str, Any]:
    """Build the Unity document. OPEN physics parameters raise :class:`OpenParameterError`."""
    open_params = [p for p in config.open_parameters() if not (allow_open_safety and p.startswith("safety."))]
    if open_params:
        raise OpenParameterError(f"Unity cannot simulate OPEN parameters: {open_params}")
    if not config.thrusters:
        raise OpenParameterError("thruster layout is empty; Unity cannot allocate thrust")
    doc: dict[str, Any] = {
        "format": UNITY_ROBOT_CONFIG_FORMAT,
        "robot_config_digest": config.content_digest(),
        "body_convention": BODY_CONVENTION,
        "wire_convention": UNITY_FRAME_CONVENTION,
        "schema_version": config.schema_version,
    }
    doc.update(_walk(config))
    return doc


def write_unity_robot_config(config: RobotConfig, path: str | Path) -> str:
    """Write the document; returns the RobotConfig digest Unity must echo."""
    doc = robot_config_to_unity(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return str(doc["robot_config_digest"])
