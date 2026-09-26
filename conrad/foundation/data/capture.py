"""Foundation capture units and fixed-window construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from conrad.data.adapters.base import SequenceSample
from conrad.foundation.data.pairing import PairEdge, PairGraph, PairRelation

WINDOW_MS = 500
STRIDE_MS = 500
MAX_CONTEXT_WINDOWS = 10
GAP_TERMINATION_MS = 1000


@dataclass(frozen=True)
class FoundationCaptureUnit:
    unit_id: str
    sequence_id: str
    timestamp_ms: int
    modality: str
    payload: np.ndarray
    lineage: dict[str, str]
    calibration_ref: str | None
    natural_missing: bool = False


@dataclass(frozen=True)
class FoundationWindow:
    window_id: str
    sequence_id: str
    start_ms: int
    duration_ms: int
    stride_ms: int
    units: tuple[FoundationCaptureUnit, ...]
    natural_missing_modalities: tuple[str, ...]
    pair_graph: PairGraph
    context_index: int

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted({u.modality for u in self.units}))


def _payload_from_observation(obs: object) -> np.ndarray:
    values = getattr(obs, "inline_values", None)
    if values is None:
        ref = getattr(obs, "payload_ref", None)
        shape = getattr(ref, "shape", ()) or (1,)
        return np.zeros(tuple(int(x) for x in shape), dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def sequence_to_capture_units(sample: SequenceSample) -> tuple[FoundationCaptureUnit, ...]:
    out: list[FoundationCaptureUnit] = []
    for obs in sample.observations:
        modality = getattr(obs.modality, "value", str(obs.modality))
        out.append(
            FoundationCaptureUnit(
                unit_id=str(obs.observation_id),
                sequence_id=sample.sequence_id,
                timestamp_ms=int(obs.timestamp.time_ns // 1_000_000),
                modality=modality,
                payload=_payload_from_observation(obs),
                lineage={
                    "dataset_id": sample.dataset_id,
                    "manifest_digest": sample.manifest_digest,
                    "sequence_id": sample.sequence_id,
                },
                calibration_ref=obs.calibration_ref,
                natural_missing=False,
            )
        )
    return tuple(out)


def build_windows(
    units: Sequence[FoundationCaptureUnit],
    *,
    expected_modalities: Sequence[str],
    window_ms: int = WINDOW_MS,
    stride_ms: int = STRIDE_MS,
) -> tuple[FoundationWindow, ...]:
    if window_ms != WINDOW_MS or stride_ms != STRIDE_MS:
        raise ValueError("P3 frozen design requires 500 ms windows with 500 ms stride")
    if not units:
        return ()
    ordered = sorted(units, key=lambda u: (u.sequence_id, u.timestamp_ms, u.modality))
    windows: list[FoundationWindow] = []
    segment_start = ordered[0].timestamp_ms
    previous = ordered[0].timestamp_ms
    context = 0
    for unit in ordered:
        if unit.timestamp_ms - previous > GAP_TERMINATION_MS:
            segment_start = unit.timestamp_ms
            context = 0
        previous = unit.timestamp_ms
        slot = (unit.timestamp_ms - segment_start) // stride_ms
        start = segment_start + slot * stride_ms
        if slot >= MAX_CONTEXT_WINDOWS:
            segment_start = unit.timestamp_ms
            start = segment_start
            context = 0
        bucket = [u for u in ordered if u.sequence_id == unit.sequence_id and start <= u.timestamp_ms < start + window_ms]
        node_ids = tuple(u.unit_id for u in bucket)
        edges = []
        for i, left in enumerate(bucket):
            for right in bucket[i + 1 :]:
                relation = PairRelation.SAME_WINDOW if left.modality != right.modality else PairRelation.TEMPORAL_NEIGHBOR
                edges.append(PairEdge(left.unit_id, right.unit_id, relation, abs(right.timestamp_ms - left.timestamp_ms)))
        present = {u.modality for u in bucket}
        missing = tuple(sorted(set(expected_modalities) - present))
        window = FoundationWindow(
            window_id=f"{unit.sequence_id}:{start}",
            sequence_id=unit.sequence_id,
            start_ms=start,
            duration_ms=window_ms,
            stride_ms=stride_ms,
            units=tuple(bucket),
            natural_missing_modalities=missing,
            pair_graph=PairGraph(nodes=node_ids, edges=tuple(edges)),
            context_index=context,
        )
        if window.window_id not in {w.window_id for w in windows}:
            windows.append(window)
            context += 1
    return tuple(windows)

