"""Adapter from twin-generated samples to the shared dataset interface (ch26 Dataset pipeline).

Twin observations are already contract-valid ``Observation`` objects. Supervision stays in the
separate :class:`PartialTruth` channel. TRUTH PLANE: training/evaluation only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Protocol

import numpy as np

from conrad.data.adapters.base import (
    DatasetAdapter,
    DatasetAudit,
    DatasetContractError,
    PartialTruth,
    SequenceSample,
)
from conrad.data.manifest import DatasetManifest, LineageKeys, SourceCategory, SplitUnit
from conrad.data.splits import SplitAssignment
from conrad.schemas.observation import Observation
from conrad.schemas.truth import TRUTH_MARKER, SupervisionLabel


class TwinSampleLike(Protocol):
    @property
    def observation(self) -> Observation: ...

    @property
    def supervision(self) -> SupervisionLabel | None: ...


def parse_lineage(text: str) -> LineageKeys:
    """``"world_family=A;seed=3;trajectory=t1"`` -> LineageKeys. A bare string is a world family."""
    if "=" not in text:
        return LineageKeys(world_family=text)
    fields: dict[str, str] = {}
    valid = {u.value for u in SplitUnit}
    for part in text.split(";"):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        if key.strip() not in valid:
            raise DatasetContractError(f"unknown lineage key {key!r} in {text!r}")
        fields[key.strip()] = value.strip()
    return LineageKeys(**fields)


class SyntheticTwinAdapter(DatasetAdapter):
    adapter_name = "synthetic_twin"
    adapter_version = "1.0.0"

    def __init__(
        self,
        manifest: DatasetManifest,
        samples: Iterable[TwinSampleLike],
        *,
        lineage_parser: Callable[[str], LineageKeys] = parse_lineage,
        split_assignment: SplitAssignment | None = None,
    ) -> None:
        super().__init__(manifest)
        if manifest.source_category not in (SourceCategory.TWIN_SYNTHETIC, SourceCategory.UNITY_SYNTHETIC):
            raise DatasetContractError("SyntheticTwinAdapter requires a synthetic source_category manifest")
        self._samples = samples
        self._lineage_parser = lineage_parser
        self.split_assignment = split_assignment

    def inspect(self) -> DatasetAudit:
        return DatasetAudit(
            dataset_id=self.manifest.dataset_id,
            manifest_digest=self.manifest.manifest_digest(),
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            usable=True,
            sequence_count=None,
            modalities=tuple(m.value for m in self.manifest.modalities),
            label_names=self.manifest.labels,
            notes=("streamed twin samples; sequence count is unknown until iterated", "SYNTHETIC_ONLY"),
        )

    def normalize_observation(self, raw: Any) -> Observation:
        if not isinstance(raw, Observation):
            raise DatasetContractError("twin samples must already carry contract Observation objects")
        if TRUTH_MARKER in raw.sensor_context:
            raise DatasetContractError(f"observation {raw.observation_id} carries truth-plane content")
        return raw

    def map_labels(self, raw_labels: Any) -> PartialTruth | None:
        labels: Sequence[SupervisionLabel | None] = raw_labels
        if all(label is None for label in labels):
            return None
        names = sorted({k for label in labels if label is not None for k in label.targets})
        targets: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        for name in names:
            rows: list[np.ndarray | None] = []
            available: list[bool] = []
            for label in labels:
                value = None if label is None else label.targets.get(name)
                present = value is not None and (label is None or label.target_masks.get(name, True))
                try:
                    rows.append(np.asarray(value, dtype=np.float64) if present else None)
                except (TypeError, ValueError):
                    rows = []
                    break
                available.append(bool(present))
            shapes = {r.shape for r in rows if r is not None}
            if not rows or len(shapes) != 1:
                skipped.append(name)
                continue
            shape = shapes.pop()
            targets[name] = np.stack([r if r is not None else np.full(shape, np.nan) for r in rows])
            masks[name] = np.asarray(available, dtype=np.bool_)
        return PartialTruth(
            targets=targets,
            availability_masks=masks,
            label_source_refs=dict.fromkeys(
                targets, f"twin:{self.manifest.dataset_id}@{self.manifest.version}"
            ),
            quality_metadata={"ground_truth_quality": "SYNTHETIC_EXACT", "non_tensor_targets": skipped},
        )

    def iter_sequences(self, split: str | None = None) -> Iterator[SequenceSample]:
        if split is not None and self.split_assignment is None:
            raise DatasetContractError("a split was requested but no SplitAssignment was supplied")
        order: list[tuple[str, str]] = []
        grouped: dict[tuple[str, str], list[TwinSampleLike]] = {}
        for sample in self._samples:
            lineage_text = sample.supervision.lineage if sample.supervision is not None else ""
            if not lineage_text:
                raise DatasetContractError(
                    f"twin observation {sample.observation.observation_id} has no split lineage"
                )
            key = (str(sample.observation.mission_id), lineage_text)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(sample)
        for key in order:
            mission, lineage_text = key
            sequence_id = f"{mission}|{lineage_text}"
            assignment = self.split_assignment
            if split is not None and assignment is not None and assignment.split_of(sequence_id) != split:
                continue
            members = sorted(grouped[key], key=lambda s: s.observation.timestamp.time_ns)
            lineage = self._lineage_parser(lineage_text)
            if lineage.sequence is None:
                lineage = lineage.model_copy(update={"sequence": sequence_id})
            yield SequenceSample(
                dataset_id=self.manifest.dataset_id,
                manifest_digest=self.manifest.manifest_digest(),
                sequence_id=sequence_id,
                lineage=lineage,
                observations=tuple(self.normalize_observation(s.observation) for s in members),
                supervision=self.map_labels([s.supervision for s in members]),
                split=split,
            )
