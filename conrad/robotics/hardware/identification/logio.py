"""Load identification logs from a manifest of CSV files (the hardware owner's delivery format).

Manifest (YAML)::

    split_id: pool-2026-10-a
    segments:
      - trajectory_id: C-surge-01
        kind: C_SURGE_ACCELERATION          # ExperimentKind value
        split: identification               # identification | validation
        file: logs/C-surge-01.csv           # relative to the manifest
        units: {force: N, velocity: m/s}    # one entry per CSV column except time_s
        metadata: {water_density_kgm3: 998.2}
        synthetic: false

CSV: header row, first column ``time_s`` (strictly increasing seconds), one column per signal.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from conrad.robotics.hardware.identification.dataset import ExperimentKind, IdentificationDataset, LogSegment


class LogFormatError(ValueError):
    pass


def load_csv_segment(
    path: str | Path,
    trajectory_id: str,
    kind: ExperimentKind,
    units: dict[str, str],
    *,
    synthetic: bool,
    metadata: dict[str, float] | None = None,
) -> LogSegment:
    p = Path(path)
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header or header[0] != "time_s":
            raise LogFormatError(f"{p}: first column must be time_s")
        rows = [[float(x) for x in row] for row in reader if row]
    if not rows:
        raise LogFormatError(f"{p}: no samples")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != len(header):
        raise LogFormatError(f"{p}: ragged rows")
    names = header[1:]
    missing = set(names) - set(units)
    if missing:
        raise LogFormatError(f"{p}: columns without declared units {sorted(missing)}")
    return LogSegment(
        trajectory_id,
        kind,
        data[:, 0],
        {n: data[:, i + 1] for i, n in enumerate(names)},
        {n: units[n] for n in names},
        synthetic=synthetic,
        source=f"csv:{p.name}",
        metadata=metadata,
    )


def load_manifest(path: str | Path) -> IdentificationDataset:
    p = Path(path)
    doc: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "segments" not in doc:
        raise LogFormatError(f"{p}: expected split_id and segments")
    ident: list[LogSegment] = []
    valid: list[LogSegment] = []
    for entry in doc["segments"]:
        if "synthetic" not in entry:
            raise LogFormatError(f"{entry.get('trajectory_id')}: 'synthetic' must be stated explicitly")
        seg = load_csv_segment(
            p.parent / entry["file"],
            str(entry["trajectory_id"]),
            ExperimentKind(entry["kind"]),
            dict(entry["units"]),
            synthetic=bool(entry["synthetic"]),
            metadata={k: float(v) for k, v in (entry.get("metadata") or {}).items()},
        )
        split = entry.get("split")
        if split == "identification":
            ident.append(seg)
        elif split == "validation":
            valid.append(seg)
        else:
            raise LogFormatError(f"{seg.trajectory_id}: split must be identification or validation")
    return IdentificationDataset(str(doc.get("split_id") or p.stem), ident, valid)
