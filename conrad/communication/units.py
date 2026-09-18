"""InformationUnit builder with progressive fidelity F0..F4 (ch16 'Progressive fidelity', ch19).

Each fidelity level is an INCREMENT on top of the previous one. ``size_bits`` is measured, not
assumed: 8 x the UTF-8 length of the canonical JSON of the cumulative payload, plus 8 x the byte length
of referenced evidence objects at F3 (compressed) and F4 (raw). Evidence bytes are referenced by ID
(``EvidenceSizes``, typically ``PayloadRef.byte_length``) so raw evidence stays discoverable onboard.

    F0 critical alert (critical units only)   F1 structured semantic deltas
    F2 + evidence summary                     F3 + compressed supporting evidence
    F4 + full raw evidence

implementation_status: EXPERIMENTAL_CANDIDATE (payload layout) / FROZEN_CONTRACT (InformationUnit)
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from conrad.communication.config import BAACConfig
from conrad.communication.delta import View, compute_deltas
from conrad.schemas.base import ConradModel
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.comms import Fidelity, FidelityOption, InformationType, InformationUnit
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp

EvidenceSizes = Mapping[UUID, int]


def payload_bits(payload: Any) -> int:
    return 8 * len(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


class UnitContent(ConradModel):
    """Incremental payload per fidelity plus the evidence bytes each increment references."""

    unit: InformationUnit
    increments: dict[int, dict[str, Any]]
    evidence_bits: dict[int, int]
    uncertainty: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    critical: bool = False
    new_revision: int | None = None

    def option(self, fidelity: int) -> FidelityOption | None:
        for o in self.unit.fidelity_levels:
            if int(o.fidelity) == fidelity:
                return o
        return None

    @property
    def levels(self) -> list[int]:
        return sorted(int(o.fidelity) for o in self.unit.fidelity_levels)


def _alert(m: BeliefMessage, delta_types: Sequence[str]) -> dict[str, Any]:
    return {
        "kind": "alert",
        "belief_id": str(m.belief_id),
        "revision": m.revision,
        "domain": m.domain.value,
        "delta_types": list(delta_types),
        "summary": m.change_summary or "critical belief change",
    }


class UnitBuilder:
    def __init__(self, id_factory: IdFactory, config: BAACConfig) -> None:
        self._ids = id_factory
        self.config = config

    def belief_unit(
        self,
        message: BeliefMessage,
        known: View | None,
        mission_value: float,
        now: TimeStamp,
        evidence_sizes: EvidenceSizes | None = None,
        deadline_ns: int | None = None,
        created_time_ns: int | None = None,
    ) -> tuple[UnitContent, ProvenanceRecord] | None:
        """None when the receiver already holds this revision (zero novelty; nothing to send)."""
        deltas = compute_deltas(known, message)
        if not deltas:
            return None
        cfg = self.config
        critical = mission_value >= cfg.critical_value
        sizes = evidence_sizes or {}
        ev_ids = list(message.evidence_support)
        raw_bytes = [sizes.get(e, cfg.default_raw_evidence_bytes) for e in ev_ids]
        increments: dict[int, dict[str, Any]] = {}
        evidence_bits: dict[int, int] = {}
        if critical:
            increments[0] = _alert(message, [d.delta_type.value for d in deltas])
        increments[1] = {"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in deltas]}
        emb = message.state_embedding[: cfg.embedding_summary_dims]
        increments[2] = {
            "kind": "evidence_summary",
            "belief_id": str(message.belief_id),
            "evidence_ids": [str(e) for e in ev_ids],
            "conflict_ids": [str(e) for e in message.evidence_conflicts],
            "embedding_q8": [max(-127, min(127, round(v * 127))) for v in emb],
        }
        if ev_ids:
            increments[3] = {
                "kind": "evidence_compressed",
                "evidence": [{"id": str(e), "codec": "lossy"} for e in ev_ids],
            }
            evidence_bits[3] = sum(8 * max(1, int(b * cfg.compressed_evidence_fraction)) for b in raw_bytes)
            increments[4] = {
                "kind": "evidence_raw",
                "evidence": [{"id": str(e), "codec": "raw"} for e in ev_ids],
            }
            evidence_bits[4] = sum(8 * b for b in raw_bytes)
        options = []
        cumulative = 0
        for level in sorted(increments):
            cumulative += payload_bits(increments[level]) + evidence_bits.get(level, 0)
            options.append(
                FidelityOption(
                    fidelity=Fidelity(level),
                    size_bits=cumulative,
                    information_retained=cfg.information_retained[level],
                )
            )
        provenance = ProvenanceRecord(
            record_id=self._ids.new(),
            source_type=SourceType.PLAN,
            source_ids=(message.message_id, message.belief_id),
            operation="baac.build_unit",
            module="conrad.communication.units",
            model_version=cfg.model_version,
            timestamp=now,
            parent_records=message.provenance_refs,
        )
        if deadline_ns is None and cfg.default_deadline_s is not None:
            deadline_ns = now.time_ns + int(cfg.default_deadline_s * 1e9)
        unit = InformationUnit(
            unit_id=self._ids.new(),
            trace_id=message.message_id,
            content_type=InformationType.ALERT if critical else InformationType.BELIEF_DELTA,
            belief_ids=(message.belief_id,),
            evidence_ids=message.evidence_support,
            semantic_delta=deltas[0],
            priority=min(1.0, mission_value),
            mission_value=mission_value,
            created_time_ns=now.time_ns if created_time_ns is None else created_time_ns,
            deadline_ns=deadline_ns,
            fidelity_levels=tuple(options),
            provenance_id=provenance.record_id,
        )
        content = UnitContent(
            unit=unit,
            increments=increments,
            evidence_bits=evidence_bits,
            uncertainty=message.uncertainty.as_tuple(),
            critical=critical,
            new_revision=message.revision,
        )
        return content, provenance

    def evidence_unit(
        self,
        evidence_ids: Sequence[UUID],
        mission_value: float,
        now: TimeStamp,
        evidence_sizes: EvidenceSizes,
    ) -> UnitContent:
        """Evidence-on-demand: the receiver asked for these exact IDs (raw evidence stays retrievable)."""
        payload = {"kind": "evidence_raw", "evidence": [{"id": str(e), "codec": "raw"} for e in evidence_ids]}
        bits = payload_bits(payload) + sum(
            8 * evidence_sizes.get(e, self.config.default_raw_evidence_bytes) for e in evidence_ids
        )
        unit = InformationUnit(
            unit_id=self._ids.new(),
            trace_id=self._ids.new(),
            content_type=InformationType.EVIDENCE,
            evidence_ids=tuple(evidence_ids),
            priority=min(1.0, mission_value),
            mission_value=mission_value,
            created_time_ns=now.time_ns,
            fidelity_levels=(
                FidelityOption(fidelity=Fidelity.F4_RAW_EVIDENCE, size_bits=bits, information_retained=1.0),
            ),
            provenance_id=self._ids.new(),
        )
        return UnitContent(
            unit=unit, increments={4: payload}, evidence_bits={4: bits - payload_bits(payload)}
        )
