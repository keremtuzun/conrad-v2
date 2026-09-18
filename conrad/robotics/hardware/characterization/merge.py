"""Merge a characterization bundle into a NEW, versioned RobotConfig plus a change report.

The input RobotConfig is never mutated. Every changed parameter is listed with its old and new value,
source and units. Rules:

* units are converted to the target parameter's declared units (``units.convert_value``);
* body-frame vectors must be stated in the ROBOT frame;
* MEASURED requires a raw-log digest (enforced by the record) and replaces anything;
* downgrading a MEASURED/IDENTIFIED parameter to a weaker source requires ``allow_downgrade``;
* a thruster absent from the base config is created only when all of its fields are supplied.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field
from pydantic_core import to_jsonable_python

from conrad.robotics.hardware.characterization.records import CharacterizationBundle, CharacterizationRecord
from conrad.robotics.hardware.characterization.units import convert_sigma, convert_value
from conrad.schemas.base import ConradModel
from conrad.schemas.frames import ROBOT
from conrad.schemas.robot import PHYSICALLY_GROUNDED, RobotConfig, Sourced, SourceKind

_PATH = re.compile(r"^(?P<head>[a-z_][a-z_0-9]*)(\[(?P<key>[^\]]+)\])?(\.(?P<field>[a-z_0-9]+))?$")
_KEYED = {"thrusters": "thruster_id", "sensors": "sensor_name", "communications": "link_name"}
_SINGLE = ("battery", "compute", "safety")
THRUSTER_FIELD_UNITS = {
    "position_body_m": "m",
    "direction_body": "unit",
    "max_forward_thrust_n": "N",
    "max_reverse_thrust_n": "N",
    "deadzone_command": "unitless",
    "time_constant_s": "s",
    "latency_s": "s",
    "thrust_coefficient": "N",
}
_BODY_VECTORS = {
    "center_of_mass_body_m",
    "center_of_buoyancy_body_m",
    "position_body_m",
    "direction_body",
    "mount_pose_body",
}
_RANK = {
    SourceKind.OPEN: 0,
    SourceKind.SYNTHETIC_ONLY: 1,
    SourceKind.ENGINEERING_ESTIMATE: 2,
    SourceKind.LITERATURE_PRIOR: 3,
    SourceKind.IDENTIFIED: 4,
    SourceKind.MEASURED: 5,
}


class MergeError(ValueError):
    pass


class ParameterChange(ConradModel):
    target: str
    record_id: str
    old_value: Any
    old_source: str
    new_value: Any
    new_source: str
    units: str
    uncertainty_1sigma: float | None
    created: bool = False


class ChangeReport(ConradModel):
    bundle_id: str
    base_config_name: str
    base_config_version: str
    base_digest: str
    new_config_version: str
    new_digest: str
    changes: tuple[ParameterChange, ...]
    unchanged_targets: tuple[str, ...] = ()
    non_config_records: tuple[str, ...] = Field(default=(), description="health.* records kept as evidence")
    remaining_open: tuple[str, ...]
    remaining_ungrounded: tuple[str, ...]


def _sourced(rec: CharacterizationRecord, old: Sourced[Any]) -> Sourced[Any]:
    field = rec.target.rsplit(".", 1)[-1].split("[")[0]
    if field in _BODY_VECTORS and rec.source is not SourceKind.OPEN and rec.frame_id != ROBOT:
        raise MergeError(
            f"{rec.record_id}: {rec.target} must be stated in frame {ROBOT!r}, got {rec.frame_id!r}"
        )
    value: Any = rec.value
    if isinstance(value, int | float | tuple):
        value = convert_value(value, rec.units, old.units)
        if isinstance(old.value, tuple) and not isinstance(value, tuple):
            raise MergeError(f"{rec.record_id}: {rec.target} expects a vector")
        if isinstance(old.value, tuple) and isinstance(value, tuple) and len(value) != len(old.value):
            raise MergeError(f"{rec.record_id}: {rec.target} expects {len(old.value)} components")
    elif value is not None and rec.units != old.units:
        raise MergeError(f"{rec.record_id}: non-numeric value needs units {old.units!r}")
    vrange = None
    if rec.valid_range is not None:
        lo, hi = (convert_value(v, rec.units, old.units) for v in rec.valid_range)
        assert isinstance(lo, float) and isinstance(hi, float)
        vrange = (lo, hi)
    return Sourced[Any](
        value=value,
        units=old.units,
        source=rec.source,
        frame_id=rec.frame_id,
        measured_at=rec.timestamp,
        uncertainty_1sigma=convert_sigma(rec.uncertainty_1sigma, rec.units, old.units),
        valid_range=vrange,
        provenance=rec.provenance_string() if rec.source is not SourceKind.OPEN else rec.provenance,
    )


def _template(units: str) -> Sourced[Any]:
    return Sourced[Any](value=None, units=units, source=SourceKind.OPEN)


def merge_characterization(
    base: RobotConfig,
    bundle: CharacterizationBundle,
    new_version: str,
    *,
    allow_downgrade: bool = False,
) -> tuple[RobotConfig, ChangeReport]:
    if new_version == base.config_version:
        raise MergeError("new_version must differ from the base config_version")
    data: dict[str, Any] = {name: getattr(base, name) for name in type(base).model_fields}
    keyed: dict[str, dict[str, dict[str, Any]]] = {
        head: {getattr(m, key): dict(m) for m in data[head]} for head, key in _KEYED.items()
    }
    singles: dict[str, dict[str, Any]] = {name: dict(data[name]) for name in _SINGLE}
    changes: list[ParameterChange] = []
    unchanged: list[str] = []
    health: list[str] = []
    new_thrusters: set[str] = set()
    for rec in bundle.records:
        if rec.target.startswith("health."):
            health.append(rec.record_id)
            continue
        m = _PATH.match(rec.target)
        if m is None:
            raise MergeError(f"{rec.record_id}: unparseable target {rec.target!r}")
        head, key, field = m["head"], m["key"], m["field"]
        if head in _KEYED:
            if key is None or field is None:
                raise MergeError(f"{rec.record_id}: {head} targets need [key].field")
            items = keyed[head]
            if key not in items:
                if head != "thrusters":
                    raise MergeError(f"{rec.record_id}: unknown {head[:-1]} {key!r} in the base config")
                items[key] = {
                    "thruster_id": key,
                    **{f: _template(u) for f, u in THRUSTER_FIELD_UNITS.items()},
                }
                new_thrusters.add(key)
            container = items[key]
        elif head in _SINGLE:
            if field is None or key is not None:
                raise MergeError(f"{rec.record_id}: {head} targets need .field")
            container = singles[head]
        else:
            if key is not None or field is not None or head not in data:
                raise MergeError(f"{rec.record_id}: unknown target {rec.target!r}")
            container, field = data, head
        old = container.get(field)
        if not isinstance(old, Sourced):
            raise MergeError(f"{rec.record_id}: {rec.target} is not a sourced physical parameter")
        new = _sourced(rec, old)
        if (
            not allow_downgrade
            and old.source in PHYSICALLY_GROUNDED
            and _RANK[new.source] < _RANK[old.source]
        ):
            raise MergeError(
                f"{rec.record_id}: would downgrade {rec.target} from {old.source.value} to {new.source.value}"
            )
        if new.value == old.value and new.source is old.source:
            unchanged.append(rec.target)
        changes.append(
            ParameterChange(
                target=rec.target,
                record_id=rec.record_id,
                old_value=old.value,
                old_source=old.source.value,
                new_value=new.value,
                new_source=new.source.value,
                units=new.units,
                uncertainty_1sigma=new.uncertainty_1sigma,
                created=key in new_thrusters,
            )
        )
        container[field] = new
    for tid in new_thrusters:
        missing = [f for f, v in keyed["thrusters"][tid].items() if isinstance(v, Sourced) and v.is_open]
        if missing:
            raise MergeError(f"new thruster {tid!r} is incomplete; missing {missing}")
    for head in _KEYED:
        data[head] = list(keyed[head].values())
    data.update(singles)
    data["config_version"] = new_version
    try:
        # Round-trip through plain JSON so every Sourced[...] is re-validated against its declared type.
        merged = RobotConfig.model_validate(to_jsonable_python(data))
    except Exception as exc:
        raise MergeError(f"merged RobotConfig is invalid: {exc}") from exc
    report = ChangeReport(
        bundle_id=bundle.bundle_id,
        base_config_name=base.config_name,
        base_config_version=base.config_version,
        base_digest=base.content_digest(),
        new_config_version=new_version,
        new_digest=merged.content_digest(),
        changes=tuple(changes),
        unchanged_targets=tuple(unchanged),
        non_config_records=tuple(health),
        remaining_open=tuple(merged.open_parameters()),
        remaining_ungrounded=tuple(merged.ungrounded_parameters()),
    )
    return merged, report
