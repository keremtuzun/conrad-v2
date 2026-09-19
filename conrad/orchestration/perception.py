"""Observation -> Evidence -> Model2S / Model2T / Model2E. DEPLOYMENT PLANE.

Routing is by modality and the observation's declared ``sensor_context["kind"]``, never by adapter type.
Every stage emits OBSERVATION_RECEIVED / EVIDENCE_CREATED / BELIEF_COMMITTED with the observation's
trace_id; the children commit their own revisions and provenance to the Repository.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from conrad.domains.ecological import EcologicalEncoder, Model2E
from conrad.domains.spatial.model import Model2S
from conrad.domains.technical import Model2T, structured_evidence
from conrad.orchestration.association import StructuralAssociator, registry_of
from conrad.orchestration.mission_context import MissionContext
from conrad.orchestration.services import ModuleRunner, RuntimeServices
from conrad.persistence.object_store import ObjectStoreError
from conrad.robotics.estimation.interface import POSITION_FIX_KIND
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.events import EventType, Severity
from conrad.schemas.observation import Evidence, Modality, Observation
from conrad.schemas.timebase import TimeStamp

MODULE = "conrad.orchestration.perception"
GEOMETRIC = (Modality.DEPTH_RANGE, Modality.POINT_CLOUD, Modality.SONAR)
ECO_KINDS = ("field_sample", "ecological_survey")


@dataclass
class PerceptionStats:
    received: int = 0
    duplicates: int = 0
    late: int = 0
    corrupt: int = 0
    fixes_accepted: int = 0
    fixes_rejected: int = 0
    structural_matched: int = 0
    structural_no_match: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)


class Perception:
    def __init__(
        self,
        s: RuntimeServices,
        runner: ModuleRunner,
        ctx: MissionContext,
        m2s: Model2S | None,
        m2t: Model2T,
        m2e: Model2E | None,
        encoder_e: EcologicalEncoder | None,
        associator: StructuralAssociator,
        position_fix: Callable[[Observation], bool],
        late_policy: str,
        clock_domain: str,
    ) -> None:
        self.s, self.runner, self.ctx = s, runner, ctx
        self.clock_domain = clock_domain
        self.m2s, self.m2t, self.m2e, self.encoder_e = m2s, m2t, m2e, encoder_e
        self.associator = associator
        self.position_fix = position_fix
        self.late_policy = late_policy
        self.ids = s.ids.child("perception")
        self.stats = PerceptionStats()
        self._seen: set[UUID] = set()
        self._last_ns: dict[UUID, int] = {}
        self.structural_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ admission
    def _admit(self, obs: Observation) -> bool:
        if obs.observation_id in self._seen:
            self.stats.duplicates += 1
            self.s.emit(
                EventType.EVIDENCE_DUPLICATE_DROPPED,
                MODULE,
                obs.trace_id,
                {"observation_id": str(obs.observation_id), "reason": "DUPLICATE_OBSERVATION_ID"},
            )
            return False
        problem = self._corruption(obs)
        if problem is not None:
            self.stats.corrupt += 1
            self.s.emit(
                EventType.FAULT_DETECTED,
                MODULE,
                obs.trace_id,
                {"observation_id": str(obs.observation_id), "reason": problem},
                severity=Severity.WARNING,
            )
            return False
        self._seen.add(obs.observation_id)
        last = self._last_ns.get(obs.sensor_id)
        if last is not None and obs.timestamp.time_ns < last:
            self.stats.late += 1
            reject = self.late_policy == "REJECT"
            self.s.emit(
                EventType.LATE_EVIDENCE_HANDLED,
                MODULE,
                obs.trace_id,
                {
                    "observation_id": str(obs.observation_id),
                    "policy": self.late_policy,
                    "lag_s": (last - obs.timestamp.time_ns) / 1e9,
                    "action": "REJECTED" if reject else "ACCEPTED_LATE",
                },
            )
            if reject:
                return False
        self._last_ns[obs.sensor_id] = max(last or 0, obs.timestamp.time_ns)
        return True

    def _corruption(self, obs: Observation) -> str | None:
        if obs.inline_values is not None and not all(math.isfinite(v) for v in obs.inline_values):
            return "NON_FINITE_INLINE_VALUES"
        if obs.payload_ref is not None:
            try:
                self.s.store.get_bytes(obs.payload_ref)
            except ObjectStoreError as exc:
                return f"PAYLOAD_UNVERIFIABLE: {type(exc).__name__}"
        if obs.timestamp.clock_domain != self.clock_domain:
            return "WRONG_CLOCK_DOMAIN"
        return None

    # ------------------------------------------------------------------ entry point
    def process(self, observations: Sequence[Observation], now: TimeStamp) -> list[BeliefMessage]:
        geometric: list[Observation] = []
        structural: list[Observation] = []
        ecological: list[Observation] = []
        for obs in observations:
            if not self._admit(obs):
                continue
            self.stats.received += 1
            kind = str(obs.sensor_context.get("kind", obs.modality.value))
            self.stats.by_kind[kind] = self.stats.by_kind.get(kind, 0) + 1
            self.s.repo.put_observation(obs)
            self.s.emit(
                EventType.OBSERVATION_RECEIVED,
                MODULE,
                obs.trace_id,
                {
                    "observation_id": str(obs.observation_id),
                    "modality": obs.modality.value,
                    "kind": kind,
                    "sensor_id": str(obs.sensor_id),
                },
                measurement_time_ns=obs.timestamp.time_ns,
            )
            if kind == POSITION_FIX_KIND:
                ok = self.position_fix(obs)
                self.stats.fixes_accepted += int(ok)
                self.stats.fixes_rejected += int(not ok)
            elif obs.modality in GEOMETRIC:
                geometric.append(obs)
            elif kind in ECO_KINDS:
                ecological.append(obs)
            elif obs.modality is Modality.STRUCTURED and "twin2t_fidelity" in obs.sensor_context:
                structural.append(obs)
        out: list[BeliefMessage] = []
        out += self._spatial(geometric, now)
        out += self._structural(structural, now)
        out += self._ecological(ecological, now)
        return out

    def _evidence_events(
        self, evs: Sequence[Evidence], extra: dict[UUID, dict[str, Any]] | None = None
    ) -> None:
        for ev in evs:
            payload = {
                "evidence_id": str(ev.evidence_id),
                "observation_id": str(ev.source_observation_id),
                "modality": ev.modality.value,
                "provenance_id": str(ev.provenance_id),
            }
            payload.update((extra or {}).get(ev.evidence_id, {}))
            self.s.emit(
                EventType.EVIDENCE_CREATED,
                MODULE,
                ev.trace_id,
                payload,
                measurement_time_ns=ev.timestamp.time_ns,
                causation=(ev.source_observation_id,),
            )

    def _committed(self, msgs: Sequence[BeliefMessage], trace: UUID, module: str) -> list[BeliefMessage]:
        for m in msgs:
            self.s.emit(
                EventType.BELIEF_COMMITTED,
                module,
                trace,
                {
                    "belief_id": str(m.belief_id),
                    "revision": m.revision,
                    "domain": m.domain.value,
                    "status": m.knowledge_status.value,
                    "evidence": [str(e) for e in m.evidence_support],
                    "provenance": [str(p) for p in m.provenance_refs],
                },
                measurement_time_ns=m.timestamp.time_ns,
            )
        return list(msgs)

    # ------------------------------------------------------------------ domains
    def _spatial(self, obs: list[Observation], now: TimeStamp) -> list[BeliefMessage]:
        m2s = self.m2s
        if m2s is None or not self.runner.available("model2s"):
            return []
        if not obs:
            self.runner.heartbeat_idle("model2s")
            return []
        trace = obs[0].trace_id

        def run() -> tuple[list[Evidence], list[BeliefMessage]]:
            evs = m2s.ingest_observations(obs)
            return evs, m2s.update_beliefs(now)

        res = self.runner.call("model2s", run, trace)
        if res is None:
            return []
        self._evidence_events(res[0])
        return self._committed(res[1], trace, "conrad.domains.spatial")

    def _structural(self, obs: list[Observation], now: TimeStamp) -> list[BeliefMessage]:
        if not self.runner.available("model2t"):
            return []
        if not obs:
            self.runner.heartbeat_idle("model2t")
            return []
        trace = obs[0].trace_id
        matched: list[Evidence] = []
        extra: dict[UUID, dict[str, Any]] = {}
        all_evs: list[Evidence] = []
        for o in obs:
            ev, prov = structured_evidence(o, self.ids, None, independence_group=f"obs:{o.observation_id}")
            ev, decision = self.associator.associate(o, ev)
            self.s.repo.put_provenance(self.s.run_id, [prov])
            self.s.repo.put_evidence(ev)
            rid = registry_of(ev)
            extra[ev.evidence_id] = {
                "association": "NO_MATCH" if rid is None else "MATCH",
                "registry_id": None if rid is None else str(rid),
                "candidates": 0 if decision is None else len(decision.candidate_ids),
            }
            self.structural_log.append(
                {
                    "t_s": ev.timestamp.time_ns / 1e9,
                    "evidence_id": str(ev.evidence_id),
                    "observation_id": str(o.observation_id),
                    "registry_id": extra[ev.evidence_id]["registry_id"],
                    "sensor_id": str(o.sensor_id),
                    "measurements": ev.measurements,
                }
            )
            all_evs.append(ev)
            if rid is not None:
                matched.append(ev)
        self.stats.structural_matched += len(matched)
        self.stats.structural_no_match += len(all_evs) - len(matched)
        self._evidence_events(all_evs, extra)

        def run() -> list[BeliefMessage]:
            self.m2t.ingest(matched)
            return self.m2t.update_beliefs(now)

        msgs = self.runner.call("model2t", run, trace)
        return [] if msgs is None else self._committed(msgs, trace, "conrad.domains.technical")

    def _ecological(self, obs: list[Observation], now: TimeStamp) -> list[BeliefMessage]:
        m2e, enc = self.m2e, self.encoder_e
        if m2e is None or enc is None or not self.runner.available("model2e"):
            return []
        if not obs:
            self.runner.heartbeat_idle("model2e")
            return []
        trace = obs[0].trace_id

        def run() -> tuple[list[Evidence], list[BeliefMessage]]:
            evs = []
            for o in obs:
                ev, prov = enc.encode(o, max(now.time_ns, o.timestamp.time_ns))
                self.s.repo.put_provenance(self.s.run_id, [prov])
                evs.append(ev)
            m2e.ingest(evs)
            return evs, m2e.update_beliefs(now)

        res = self.runner.call("model2e", run, trace)
        if res is None:
            return []
        self._evidence_events(res[0])
        return self._committed(res[1], trace, "conrad.domains.ecological")
