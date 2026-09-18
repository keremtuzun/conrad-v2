"""Read characterization handoffs from YAML or CSV and verify every raw log they cite.

YAML: ``{bundle_id, robot_config_name, records: [<CharacterizationRecord fields>]}``.
CSV header (one record per row; vectors as ``a;b;c``; empty cell = null)::

    record_id,category,target,value,units,frame_id,time_ns,clock_domain,uncertainty_1sigma,
    range_min,range_max,source,method,provenance,raw_log,operator

Raw logs are resolved relative to the handoff file, hashed here, and the digest is compared with any
digest the handoff states. A MEASURED row whose raw log is missing is rejected.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml

from conrad.robotics.hardware.characterization.records import CharacterizationBundle, CharacterizationRecord

CSV_COLUMNS = (
    "record_id",
    "category",
    "target",
    "value",
    "units",
    "frame_id",
    "time_ns",
    "clock_domain",
    "uncertainty_1sigma",
    "range_min",
    "range_max",
    "source",
    "method",
    "provenance",
    "raw_log",
    "operator",
)


class IngestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attach_digest(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    log = raw.get("raw_log")
    if not log:
        return raw
    path = (root / log).resolve()
    if not path.is_file():
        raise IngestError(f"record {raw.get('record_id')}: raw log {log!r} not found under {root}")
    digest = sha256_file(path)
    stated = raw.get("raw_log_digest")
    if stated and stated != digest:
        raise IngestError(f"record {raw.get('record_id')}: raw log digest mismatch ({stated} != {digest})")
    return {**raw, "raw_log_digest": digest}


def _record(raw: dict[str, Any], root: Path) -> CharacterizationRecord:
    try:
        return CharacterizationRecord.model_validate(_attach_digest(raw, root))
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"invalid record {raw.get('record_id')!r}: {exc}") from exc


def load_yaml_bundle(path: str | Path) -> CharacterizationBundle:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "records" not in data:
        raise IngestError(f"{p}: expected a mapping with a 'records' list")
    root = p.parent
    records = tuple(_record(dict(r), root) for r in data["records"])
    return CharacterizationBundle(
        bundle_id=str(data.get("bundle_id") or p.stem),
        robot_config_name=str(data.get("robot_config_name", "")),
        root=str(root),
        records=records,
    )


def _cell(row: dict[str, str], key: str) -> str | None:
    v = (row.get(key) or "").strip()
    return v or None


def _parse_value(text: str | None) -> float | tuple[float, ...] | str | None:
    if text is None:
        return None
    if ";" in text:
        return tuple(float(x) for x in text.split(";"))
    try:
        return float(text)
    except ValueError:
        return text


def csv_row_to_raw(row: dict[str, str]) -> dict[str, Any]:
    time_ns, domain = _cell(row, "time_ns"), _cell(row, "clock_domain")
    rmin, rmax = _cell(row, "range_min"), _cell(row, "range_max")
    sigma = _cell(row, "uncertainty_1sigma")
    return {
        "record_id": _cell(row, "record_id"),
        "category": _cell(row, "category"),
        "target": _cell(row, "target"),
        "value": _parse_value(_cell(row, "value")),
        "units": _cell(row, "units"),
        "frame_id": _cell(row, "frame_id"),
        "timestamp": None if time_ns is None else {"time_ns": int(time_ns), "clock_domain": domain or "UTC"},
        "uncertainty_1sigma": None if sigma is None else float(sigma),
        "valid_range": None if rmin is None or rmax is None else (float(rmin), float(rmax)),
        "source": _cell(row, "source"),
        "method": _cell(row, "method"),
        "provenance": _cell(row, "provenance"),
        "raw_log": _cell(row, "raw_log"),
        "operator": _cell(row, "operator"),
    }


def load_csv_bundle(
    path: str | Path, bundle_id: str | None = None, robot_config_name: str = ""
) -> CharacterizationBundle:
    p = Path(path)
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise IngestError(f"{p}: missing CSV columns {sorted(missing)}")
        records = tuple(_record(csv_row_to_raw(row), p.parent) for row in reader)
    return CharacterizationBundle(
        bundle_id=bundle_id or p.stem,
        robot_config_name=robot_config_name,
        root=str(p.parent),
        records=records,
    )


def load_bundle(path: str | Path) -> CharacterizationBundle:
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_yaml_bundle(path)
    if suffix == ".csv":
        return load_csv_bundle(path)
    raise IngestError(f"unsupported characterization file type {suffix!r} (use .yaml or .csv)")
