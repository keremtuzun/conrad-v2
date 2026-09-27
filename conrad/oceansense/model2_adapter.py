"""P6 OS-FM representation adapter and dual-path evidence rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.observation import Evidence

OSFM_DIM = 384
ADAPTER_HIDDEN_DIM = 512
MODEL2_DIM = 256


class RepresentationAdapter:
    """Deterministic 384 -> 512 -> 256 adapter used at the OS-FM/Model2 boundary."""

    version = "osfm-representation-adapter-384-512-256-v0"

    def __init__(self, seed: int = 20260927) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 1.0 / np.sqrt(OSFM_DIM), size=(OSFM_DIM, ADAPTER_HIDDEN_DIM))
        self.b1 = np.zeros(ADAPTER_HIDDEN_DIM, dtype=np.float64)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(ADAPTER_HIDDEN_DIM), size=(ADAPTER_HIDDEN_DIM, MODEL2_DIM))
        self.b2 = np.zeros(MODEL2_DIM, dtype=np.float64)

    def transform(self, representation: tuple[float, ...]) -> tuple[float, ...]:
        if len(representation) != OSFM_DIM:
            raise ValueError(f"OS-FM representation must be {OSFM_DIM}D")
        x = np.asarray(representation, dtype=np.float64)
        hidden = np.tanh(x @ self.w1 + self.b1)
        out = hidden @ self.w2 + self.b2
        return tuple(float(v) for v in out)


@dataclass(frozen=True)
class Model2EvidenceBundle:
    """Learned, semantic and direct evidence prepared for Model2 ingestion."""

    learned_evidence: Evidence
    semantic_evidence: dict[str, Any]
    direct_physical_evidence: Evidence | None
    osfm_context_provenance_id: UUID
    adapter_version: str

    @property
    def model2_embedding_dim(self) -> int:
        return len(self.learned_evidence.embedding)


def adapt_evidence_for_model2(evidence: Evidence, osfm_representation: tuple[float, ...], adapter: RepresentationAdapter) -> Model2EvidenceBundle:
    """Return learned 256D evidence while preserving direct physical measurements separately.

    Measurements remain on the original evidence object. The learned evidence carries the
    adapted representation and context provenance, but it does not replace calibrated
    physical values.
    """

    learned = evidence.model_copy(
        update={
            "embedding": adapter.transform(osfm_representation),
            "encoder_version": adapter.version,
        }
    )
    direct = evidence if evidence.measurements else None
    semantic = {
        "source_evidence_id": str(evidence.evidence_id),
        "source_observation_id": str(evidence.source_observation_id),
        "modality": evidence.modality.value,
        "spatial_support": None if evidence.spatial_support is None else evidence.spatial_support.model_dump(mode="json"),
        "reliability": evidence.reliability,
        "measurement_keys": tuple(sorted(evidence.measurements)),
        "measurement_units": dict(sorted(evidence.measurement_units.items())),
        "provenance_id": str(evidence.provenance_id),
        "encoder_version": evidence.encoder_version,
    }
    return Model2EvidenceBundle(
        learned_evidence=learned,
        semantic_evidence=semantic,
        direct_physical_evidence=direct,
        osfm_context_provenance_id=evidence.provenance_id,
        adapter_version=adapter.version,
    )
