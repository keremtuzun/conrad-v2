"""DATA-OSFM-01 corpus manifest and readiness checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel, digest_of

DATA_OSFM_ID = "DATA-OSFM-01"


class CorpusPartition(str, Enum):
    PRETRAIN_REAL = "PRETRAIN_REAL"
    PRETRAIN_SYNTHETIC = "PRETRAIN_SYNTHETIC"
    VALIDATION = "VALIDATION"
    FINAL_TEST = "FINAL_TEST"
    OOD_TEST = "OOD_TEST"


class CorpusReadiness(str, Enum):
    READY = "READY"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
    INVALID = "INVALID"


class CorpusSource(VersionedModel):
    source_id: str = Field(min_length=1)
    manifest_digest: str = Field(pattern="^[0-9a-f]{64}$")
    source_digest: str = Field(pattern="^[0-9a-f]{64}$")
    partition: CorpusPartition
    first_party: bool = False


class CorpusManifest(VersionedModel):
    corpus_id: str = DATA_OSFM_ID
    readiness: CorpusReadiness
    sources: tuple[CorpusSource, ...]
    required_partitions: tuple[CorpusPartition, ...] = tuple(CorpusPartition)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_first_party_and_required_partitions(self) -> CorpusManifest:
        if any(s.first_party for s in self.sources):
            raise ValueError("DATA-OSFM-01 corpus must not depend on first-party data")
        present = {s.partition for s in self.sources}
        missing = set(self.required_partitions) - present
        if self.readiness is CorpusReadiness.READY and missing:
            raise ValueError(f"READY corpus is missing partitions: {sorted(p.value for p in missing)}")
        return self

    @property
    def corpus_digest(self) -> str:
        return digest_of(self.model_dump(mode="json"))

    def partition_counts(self) -> dict[str, int]:
        out = {p.value: 0 for p in CorpusPartition}
        for source in self.sources:
            out[source.partition.value] += 1
        return out


def digest_mapping(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(data), sort_keys=True, default=str).encode()).hexdigest()


def digest_path(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return CorpusManifest.model_validate(data)


def verify_corpus_manifest(manifest: CorpusManifest) -> list[str]:
    problems: list[str] = []
    present = {s.partition for s in manifest.sources}
    for partition in manifest.required_partitions:
        if partition not in present:
            problems.append(f"missing partition {partition.value}")
    if any(s.first_party for s in manifest.sources):
        problems.append("first-party source declared")
    if manifest.readiness is CorpusReadiness.READY and problems:
        problems.append("readiness READY contradicts missing/invalid corpus requirements")
    return problems


def synthetic_manifest_for_tests(partitions: Sequence[CorpusPartition] = tuple(CorpusPartition)) -> CorpusManifest:
    sources = []
    for idx, partition in enumerate(partitions):
        payload = {"partition": partition.value, "idx": idx, "kind": "synthetic-smoke"}
        digest = digest_mapping(payload)
        sources.append(
            CorpusSource(
                source_id=f"synthetic-{partition.value.lower()}",
                manifest_digest=digest,
                source_digest=digest,
                partition=partition,
                first_party=False,
            )
        )
    return CorpusManifest(readiness=CorpusReadiness.READY, sources=tuple(sources))

