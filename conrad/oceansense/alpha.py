"""P10 OceanSense Alpha integration harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from conrad.oceansense.host import GatewayDecision, HostCapabilityProfile, HostIntentGateway
from conrad.oceansense.model2_adapter import Model2EvidenceBundle, RepresentationAdapter, adapt_evidence_for_model2
from conrad.oceansense.task_heads import (
    AnomalyHead,
    ConditionHead,
    DetectionHead,
    SegmentationHead,
    TaskHeadInput,
    TaskHeadOutput,
)
from conrad.schemas.decision import ObservationPlan
from conrad.schemas.observation import Evidence


@dataclass
class AssetMemory:
    path: Path
    records: list[dict[str, object]] = field(default_factory=list)

    def append(self, record: dict[str, object]) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AssetMemory:
        if not path.exists():
            return cls(path)
        return cls(path, json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class AlphaIntegrationResult:
    evidence_bundle: Model2EvidenceBundle | None
    task_outputs: tuple[TaskHeadOutput, ...]
    gateway_decision: GatewayDecision | None
    replay_key: str
    degraded: bool
    failures: tuple[str, ...] = ()


class OceanSenseAlpha:
    """Fixture-safe integration layer through P5-P10."""

    version = "oceansense-alpha-p5-p10-v0"

    def __init__(self, asset_memory: AssetMemory, adapter: RepresentationAdapter | None = None) -> None:
        self.asset_memory = asset_memory
        self.adapter = adapter or RepresentationAdapter()
        self.heads = (DetectionHead(), SegmentationHead(), AnomalyHead(), ConditionHead())
        self.gateway = HostIntentGateway()

    def run(
        self,
        evidence: Evidence,
        osfm_representation: tuple[float, ...],
        plan: ObservationPlan | None = None,
        host_profile: HostCapabilityProfile | None = None,
        intent_id: UUID | None = None,
    ) -> AlphaIntegrationResult:
        failures: list[str] = []
        try:
            bundle = adapt_evidence_for_model2(evidence, osfm_representation, self.adapter)
        except ValueError as exc:
            replay_key = f"{evidence.run_id}:{evidence.evidence_id}:{self.adapter.version}:failed"
            failures.append(f"ADAPTER_FAILURE:{exc}")
            self.asset_memory.append(
                {
                    "replay_key": replay_key,
                    "evidence_id": str(evidence.evidence_id),
                    "model2_embedding_dim": None,
                    "direct_physical_measurements": sorted(evidence.measurements),
                    "task_heads": [],
                    "gateway_accepted": None,
                    "degraded": True,
                    "failures": failures,
                }
            )
            return AlphaIntegrationResult(
                evidence_bundle=None,
                task_outputs=(),
                gateway_decision=None,
                replay_key=replay_key,
                degraded=True,
                failures=tuple(failures),
            )
        task_input = TaskHeadInput(
            representation=osfm_representation,
            source_evidence_id=evidence.evidence_id,
            source_observation_id=evidence.source_observation_id,
            mission_id=evidence.mission_id,
            run_id=evidence.run_id,
            trace_id=evidence.trace_id,
            modality=evidence.modality,
            timestamp=evidence.timestamp,
            provenance_id=evidence.provenance_id,
        )
        outputs = tuple(head.infer(task_input) for head in self.heads)
        gateway_decision = None
        if plan is not None and host_profile is not None and intent_id is not None:
            gateway_decision = self.gateway.from_plan(plan, host_profile, intent_id)
            if not gateway_decision.accepted:
                failures.extend(gateway_decision.reason_codes)
        replay_key = f"{evidence.run_id}:{evidence.evidence_id}:{bundle.adapter_version}"
        degraded = bool(host_profile and host_profile.degraded) or bool(failures)
        self.asset_memory.append(
            {
                "replay_key": replay_key,
                "evidence_id": str(evidence.evidence_id),
                "model2_embedding_dim": bundle.model2_embedding_dim,
                "direct_physical_measurements": sorted(evidence.measurements),
                "task_heads": [o.head_name for o in outputs],
                "gateway_accepted": None if gateway_decision is None else gateway_decision.accepted,
                "degraded": degraded,
                "failures": failures,
            }
        )
        return AlphaIntegrationResult(
            evidence_bundle=bundle,
            task_outputs=outputs,
            gateway_decision=gateway_decision,
            replay_key=replay_key,
            degraded=degraded,
            failures=tuple(failures),
        )
