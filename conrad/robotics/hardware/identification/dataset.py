"""Identification logs and the identification / validation split (ch20 "Use different trajectories").

``D_identify != D_validate``: a dataset refuses overlapping trajectories or byte-identical logs, and
validation helpers raise :class:`SplitLeakageError` if handed an identification segment.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import Enum

import numpy as np

from conrad.schemas.base import digest_of


class ExperimentKind(str, Enum):
    """ch22 identification experiment sequence A..F."""

    STATIC_TRIM = "A_STATIC_TRIM"
    THRUSTER_STEP = "B_THRUSTER_STEP"
    SURGE_ACCELERATION = "C_SURGE_ACCELERATION"
    HEAVE = "D_HEAVE"
    SWAY = "D_SWAY"
    YAW_ROTATION = "E_YAW_ROTATION"
    STATION_KEEPING = "F_STATION_KEEPING"


class SplitLeakageError(ValueError):
    """Identification data was used for validation (or overlaps it)."""


class LogSegment:
    """One recorded trajectory. Arrays are copied and made read-only; the digest covers every byte."""

    def __init__(
        self,
        trajectory_id: str,
        kind: ExperimentKind,
        time_s: np.ndarray,
        signals: dict[str, np.ndarray],
        units: dict[str, str],
        *,
        synthetic: bool,
        source: str,
        metadata: dict[str, float] | None = None,
    ) -> None:
        t = np.array(time_s, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError(f"{trajectory_id}: time_s must be 1-D and strictly increasing")
        missing = set(signals) - set(units)
        if missing:
            raise ValueError(f"{trajectory_id}: signals without units {sorted(missing)}")
        self.trajectory_id = trajectory_id
        self.kind = kind
        self.synthetic = synthetic
        self.source = source
        self.metadata = dict(metadata or {})
        t.setflags(write=False)
        self.time_s = t
        self.signals: dict[str, np.ndarray] = {}
        for name, arr in signals.items():
            a = np.array(arr, dtype=np.float64)
            if a.shape[0] != t.size:
                raise ValueError(
                    f"{trajectory_id}: signal {name} has {a.shape[0]} samples, time has {t.size}"
                )
            a.setflags(write=False)
            self.signals[name] = a
        self.units = dict(units)
        h = hashlib.sha256(t.tobytes())
        for name in sorted(self.signals):
            h.update(name.encode())
            h.update(self.signals[name].tobytes())
        self.digest = h.hexdigest()

    def signal(self, name: str, units: str) -> np.ndarray:
        if name not in self.signals:
            raise KeyError(f"{self.trajectory_id}: missing signal {name!r}")
        if self.units[name] != units:
            raise ValueError(f"{self.trajectory_id}: {name} is in {self.units[name]!r}, expected {units!r}")
        return self.signals[name]


class IdentificationDataset:
    """Disjoint identification and validation segments with a split lineage record."""

    def __init__(
        self, split_id: str, identification: Iterable[LogSegment], validation: Iterable[LogSegment]
    ) -> None:
        self.split_id = split_id
        self.identification = tuple(identification)
        self.validation = tuple(validation)
        if not self.identification:
            raise ValueError("identification set is empty")
        ids_i = {s.trajectory_id for s in self.identification}
        ids_v = {s.trajectory_id for s in self.validation}
        dig_i = {s.digest for s in self.identification}
        dig_v = {s.digest for s in self.validation}
        if ids_i & ids_v:
            raise SplitLeakageError(f"trajectories in both splits: {sorted(ids_i & ids_v)}")
        if dig_i & dig_v:
            raise SplitLeakageError("a validation log is byte-identical to an identification log")
        if len(ids_i) != len(self.identification) or len(ids_v) != len(self.validation):
            raise ValueError("duplicate trajectory_id within a split")
        self._ident_digests = frozenset(dig_i)

    @property
    def synthetic(self) -> bool:
        return all(s.synthetic for s in (*self.identification, *self.validation))

    @property
    def any_synthetic(self) -> bool:
        return any(s.synthetic for s in (*self.identification, *self.validation))

    def lineage(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "identification": sorted((s.trajectory_id, s.digest) for s in self.identification),
            "validation": sorted((s.trajectory_id, s.digest) for s in self.validation),
            "synthetic": self.synthetic,
            "lineage_digest": digest_of(
                [sorted(s.digest for s in self.identification), sorted(s.digest for s in self.validation)]
            ),
        }

    def by_kind(self, kind: ExperimentKind, *, split: str = "identification") -> tuple[LogSegment, ...]:
        pool = self.identification if split == "identification" else self.validation
        return tuple(s for s in pool if s.kind is kind)

    def assert_validation_only(self, segments: Iterable[LogSegment]) -> None:
        for s in segments:
            if s.digest in self._ident_digests:
                raise SplitLeakageError(f"{s.trajectory_id} is identification data; it cannot validate")
            if all(s.digest != v.digest for v in self.validation):
                raise SplitLeakageError(f"{s.trajectory_id} is not part of this dataset's validation split")
