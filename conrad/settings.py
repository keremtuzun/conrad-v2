"""Typed configuration loading and immutable run snapshot (ch34 Configuration).

Composition is explicit: base -> mode -> experiment -> local override. Unknown keys fail.
A local override may only contain machine paths / device selectors.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID, ConradModel

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_OVERRIDE_ALLOWED_KEYS = frozenset({"paths", "device"})


class RunMode(str, Enum):
    SIM_TRAIN = "sim_train"
    SIM = "sim"
    EVAL = "eval"
    REPLAY = "replay"
    RUNTIME = "runtime"
    HIL = "hil"


class CommandMode(str, Enum):
    DISABLED = "disabled"
    SIMULATED = "simulated"
    HARDWARE = "hardware"


class ExecutionLane(str, Enum):
    DEV = "dev"
    SIMULATION = "simulation"
    HIL = "hil"
    PHYSICAL = "physical"


class RunSettings(ConradModel):
    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
    seed: int = 2026201
    mode: RunMode = RunMode.SIM
    lane: ExecutionLane = ExecutionLane.SIMULATION
    parent_run_id: str | None = None
    require_clean_git: bool = False

    @model_validator(mode="after")
    def _identity(self) -> RunSettings:
        if self.architecture_id != ARCHITECTURE_ID:
            raise ValueError(f"architecture_id must be {ARCHITECTURE_ID}")
        if self.stack_id != STACK_ID:
            raise ValueError(f"stack_id must be {STACK_ID}")
        return self


class DataSettings(ConradModel):
    manifest_ids: tuple[str, ...] = ()
    split_hash: str | None = None


class PathSettings(ConradModel):
    artifacts_dir: str = "artifacts"
    database: str = "artifacts/conrad.sqlite"
    object_store: str = "artifacts/objects"
    runs_dir: str = "artifacts/runs"


class RuntimeSettings(ConradModel):
    persistence_mode: str = "sqlite_wal_single_writer"
    command_mode: CommandMode = CommandMode.DISABLED
    hardware_enable: bool = Field(default=False, description="machine-local physical enable; default false")
    hil_evidence_ref: str | None = None
    robot_config: str = "configs/robot/sim_reference.yaml"
    allowed_peers: tuple[str, ...] = ("127.0.0.1",)
    bind_address: str = "127.0.0.1"
    min_free_disk_bytes: int = 256 * 1024 * 1024
    late_evidence_policy: str = "EXPLICIT_LATE"  # EXPLICIT_LATE | REJECT  (ADR-0003)
    required_adapters: tuple[str, ...] = ("sim_kernel",)
    required_secrets: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _bind_private(self) -> RuntimeSettings:
        if self.bind_address in ("0.0.0.0", "::"):
            raise ValueError("binding a control endpoint to all interfaces is forbidden (ch34 Network)")
        return self


class ConradSettings(ConradModel):
    run: RunSettings = RunSettings()
    data: DataSettings = DataSettings()
    paths: PathSettings = PathSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    device: str = "cpu"
    sim: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)
    eval: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _lane_authority(self) -> ConradSettings:
        mode, lane = self.runtime.command_mode, self.run.lane
        if lane is ExecutionLane.DEV and mode is not CommandMode.DISABLED:
            raise ValueError("dev lane cannot send commands")
        if lane is ExecutionLane.SIMULATION and mode is CommandMode.HARDWARE:
            raise ValueError("simulation lane can command only simulated adapters")
        if (
            mode is CommandMode.HARDWARE
            and lane is not ExecutionLane.PHYSICAL
            and lane is not ExecutionLane.HIL
        ):
            raise ValueError("command_mode hardware requires the hil or physical lane")
        return self

    def config_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def load_settings(config_path: str | Path, local_override: str | Path | None = None) -> ConradSettings:
    """Load ``config_path``; its optional ``extends: [..]`` list is merged first, in order."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    merged: dict[str, Any] = {}
    chain: list[Path] = []

    def collect(p: Path, depth: int = 0) -> None:
        if depth > 8:
            raise ConfigError("config extends chain too deep")
        data = _read_yaml(p)
        for parent in data.get("extends", []) or []:
            collect((REPO_ROOT / parent) if not Path(parent).is_absolute() else Path(parent), depth + 1)
        chain.append(p)

    collect(path)
    for p in chain:
        data = _read_yaml(p)
        data.pop("extends", None)
        merged = _deep_merge(merged, data)
    override_path = Path(local_override) if local_override else REPO_ROOT / "configs" / "local.yaml"
    if override_path.exists():
        local = _read_yaml(override_path)
        illegal = set(local) - LOCAL_OVERRIDE_ALLOWED_KEYS
        if illegal:
            raise ConfigError(
                f"local override may only contain {sorted(LOCAL_OVERRIDE_ALLOWED_KEYS)}; found {sorted(illegal)}"
            )
        merged = _deep_merge(merged, local)
    try:
        return ConradSettings.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"invalid configuration {path}: {exc}") from exc


_SECRET_MARKERS = ("password", "secret", "token", "apikey", "api_key", "private_key", "credential")


def redact(obj: Any) -> Any:
    """Remove secret-shaped values before a snapshot is written."""
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if any(m in str(k).lower() for m in _SECRET_MARKERS) else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list | tuple):
        return [redact(v) for v in obj]
    return obj


def snapshot_yaml(settings: ConradSettings) -> str:
    return yaml.safe_dump(redact(json.loads(settings.canonical_json())), sort_keys=True)
