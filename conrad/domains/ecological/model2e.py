"""Model2E(Model2Child): ecological/environmental belief child with analytic CEFD (ch12, ch33).

Persistent state = entity + field beliefs and their ledger heads (mirrored in the repository).
Working memory = pending evidence, received cross-domain context, per-call scratch.
Any internal exception sets availability UNAVAILABLE and is re-raised, never swallowed.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.base import Model2Child
from conrad.domains.ecological import context as obs_context
from conrad.domains.ecological import readout, views
from conrad.domains.ecological.cefd_analytic import AnalyticCEFD
from conrad.domains.ecological.config import Model2EConfig
from conrad.domains.ecological.ecmer_e import evidence_kind
from conrad.domains.ecological.entity_belief import EntityBelief, EntityBeliefStore, parse_registry
from conrad.domains.ecological.field_belief import FieldBeliefGrid
from conrad.domains.ecological.ledger import BeliefLedger
from conrad.persistence.repository import Repository
from conrad.schemas.belief import Availability, BeliefMessage, BeliefQuery
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, EvidenceValidity
from conrad.schemas.timebase import ClockDomainError, TimeStamp
from conrad.schemas.world import Domain


class Model2E(Model2Child):
    domain = Domain.ECOLOGICAL

    def __init__(self, config: Model2EConfig, ids: IdFactory, repo: Repository | None = None) -> None:
        self.cfg = config
        self.model_version = config.model_version
        self.ids = ids
        self.repo = repo
        self.cefd = AnalyticCEFD(config)
        self._availability = Availability.UNAVAILABLE  # until initialize()
        self.failure: str | None = None
        self.pending: list[Evidence] = []
        self.context: dict[UUID, BeliefMessage] = {}
        self.stats: dict[str, int] = {}

    # ------------------------------------------------------------------ guard
    @contextmanager
    def _guard(self, op: str) -> Iterator[None]:
        if self._availability is Availability.UNAVAILABLE and op != "initialize":
            raise RuntimeError(f"Model2E is UNAVAILABLE ({self.failure}); call initialize() to restart")
        try:
            yield
        except Exception as exc:
            self._availability = Availability.UNAVAILABLE
            self.failure = f"{op}: {type(exc).__name__}: {exc}"
            raise

    def _count(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    # ------------------------------------------------------------------ Model2Child
    def initialize(self, context: dict[str, Any]) -> None:
        with self._guard("initialize"):
            self.clock_domain = str(context.get("clock_domain", self.cfg.clock_domain))
            run_id = context.get("run_id") or self.ids.new()
            self.ledger = BeliefLedger(self.ids, self.repo, UUID(str(run_id)), self.model_version)
            self.fields = FieldBeliefGrid(self.cfg)
            self.entities = EntityBeliefStore(self.cfg.entity)
            self.field_ids: dict[str, UUID] = {}
            self.pending, self.context, self.stats = [], {}, {}
            self._sink_ns: int | None = None
            self.direct_prov: dict[UUID, UUID] = {}
            self.stress_prov: dict[UUID, UUID] = {}
            t0 = context.get("start_time") or TimeStamp(time_ns=0, clock_domain=self.clock_domain)
            if self.cfg.switches.fields:
                for name in self.fields.fields:
                    self.field_ids[name] = self.ids.new()
                    views.create_field(self, name, t0)
            if self.cfg.switches.entities:
                for asset in parse_registry(context.get("asset_registry", ()), self.cfg.entity):
                    b = self.entities.add(
                        self.ids.new(), asset.eco_class, asset.position_m, asset.frame_id, asset
                    )
                    views.create_entity(self, b, t0)
            self._availability = Availability.AVAILABLE
            self.failure = None

    def ingest(self, evidence: Sequence[Evidence]) -> None:
        with self._guard("ingest"):
            for ev in evidence:
                if ev.timestamp.clock_domain != self.clock_domain:
                    raise ClockDomainError(
                        f"evidence clock {ev.timestamp.clock_domain!r} != {self.clock_domain!r}"
                    )
                self.pending.append(ev)

    def update_beliefs(self, now: TimeStamp) -> list[BeliefMessage]:
        with self._guard("update_beliefs"):
            ready = [e for e in self.pending if e.timestamp.time_ns <= now.time_ns]
            self.pending = [e for e in self.pending if e.timestamp.time_ns > now.time_ns]  # future: wait
            ready.sort(key=lambda e: (e.timestamp.time_ns, e.timestamp.sequence_index, str(e.evidence_id)))
            touched_f: dict[str, list[Evidence]] = {}
            touched_e: dict[UUID, list[Evidence]] = {}
            created: list[EntityBelief] = []
            for ev in ready:  # each observation at ITS OWN timestamp
                self._apply(ev, touched_f, touched_e, created)
            sink = self._entity_to_field()
            stress = self._field_to_entity()
            out: list[BeliefMessage] = []
            for b in created:
                views.create_entity(self, b, ready[0].timestamp)
            for name in self.field_ids:
                if name in touched_f or name in sink:
                    out.append(views.commit_field(self, name, touched_f.get(name, []), name in sink))
            for bid, b in self.entities.beliefs.items():
                if bid in touched_e or bid in stress:
                    out.append(views.commit_entity(self, b, touched_e.get(bid, []), stress.get(bid)))
            return out

    def predict(self, delta_t_s: float, now: TimeStamp) -> list[BeliefMessage]:
        with self._guard("predict"):
            if not (delta_t_s >= 0 and np.isfinite(delta_t_s)):
                raise ValueError(f"delta_t_s must be finite and >= 0, got {delta_t_s}")
            return readout.predict_all(self, now.time_ns + round(delta_t_s * 1e9), now)

    def receive_context(self, messages: Sequence[BeliefMessage]) -> None:
        with self._guard("receive_context"):
            for m in messages:
                if m.domain is Domain.ECOLOGICAL:
                    continue  # own beliefs are never re-ingested as context
                self.context[m.message_id] = m
                self._count("context_messages")

    def query(self, query: BeliefQuery) -> list[BeliefMessage]:
        with self._guard("query"):
            return readout.run_query(self, query)

    def export_beliefs(self) -> list[BeliefMessage]:
        with self._guard("export_beliefs"):
            return [h.message for h in self.ledger.heads.values() if h.message is not None]

    def reset_working_memory(self) -> None:
        with self._guard("reset_working_memory"):
            self.pending, self.context = [], {}

    def availability(self) -> Availability:
        return self._availability

    def observability_context(self, now: TimeStamp) -> list[BeliefMessage]:
        """2E belief-derived CROSS_DOMAIN_CONTEXT for 2S/2T: biofouling cover and turbidity effects."""
        with self._guard("observability_context"):
            return obs_context.observability_context(self, now)

    # ------------------------------------------------------------------ internals
    def gate_inflation(self) -> float:
        """2S localisation uncertainty (received as context) widens association gates."""
        uo = [m.uncertainty.observational for m in self.context.values() if m.domain is Domain.SPATIAL]
        return 1.0 + (max(uo) if uo else 0.0)

    def _apply(
        self,
        ev: Evidence,
        tf: dict[str, list[Evidence]],
        te: dict[UUID, list[Evidence]],
        created: list[EntityBelief],
    ) -> None:
        kind = evidence_kind(ev, self.cfg)
        if ev.validity is EvidenceValidity.INVALID or ev.spatial_support is None or kind == "OTHER":
            self._count("rejected_evidence")
            return
        p = np.asarray(ev.spatial_support.center_m, dtype=np.float64)
        if kind == "FIELD":
            if not self.cfg.switches.fields:
                self._count("ignored_field_evidence")
                return
            for key, value in ev.measurements.items():
                name, _, comp = key.partition(".")
                spec = self.cfg.fields[name]
                var = spec.sensor_sd**2 / max(ev.reliability, 0.05)
                upd = self.fields.update_point(name, int(comp or 0), p, value, var, ev.timestamp.time_ns)
                self._count("late_evidence", int(upd.late))
                tf.setdefault(name, []).append(ev)
            return
        if not self.cfg.switches.entities:
            self._count("ignored_entity_evidence")
            return
        mkey = next(k for k in ("cover_fraction", "detection", "survey_hit") if k in ev.measurements)
        sigma = ev.spatial_support.position_sigma_m or 0.0
        b, ambiguous = self.entities.associate(
            p, sigma, ev.spatial_support.frame_id, mkey, self.gate_inflation()
        )
        if ambiguous:
            self._count("ambiguous_association")
            return
        if b is None:
            cls = {"cover_fraction": "UNKNOWN_ECOLOGICAL", "detection": "MOBILE_GROUP"}.get(
                mkey, "UNKNOWN_ECOLOGICAL"
            )
            b = self.entities.add(self.ids.new(), cls, p, ev.spatial_support.frame_id, None)
            created.append(b)
        t = ev.timestamp.time_ns
        gate = self.cefd.gate(self.fields, "temperature", b.position_m) if self.cfg.switches.fields else 0.0
        infl = self.cefd.process_inflation(b, gate)
        if mkey == "cover_fraction":
            noise = self.cefd.survey_noise(
                self.fields if self.cfg.switches.fields else None, b, ev.sensor_context.range_m
            )
            late = self.entities.update_cover(b, ev.measurements[mkey], noise.variance, t, infl)
        elif mkey == "detection":
            late = self.entities.update_detection(b, t)
        else:
            b.n_hits += 1
            late = False
        self._count("late_evidence", int(late))
        te.setdefault(b.belief_id, []).append(ev)

    def _entity_to_field(self) -> set[str]:
        if not self.cefd.entity_to_field or "turbidity" not in self.fields.fields:
            return set()
        fb = self.fields.fields["turbidity"]
        if fb.time_ns is None:
            return set()
        dt = 0.0 if self._sink_ns is None else (fb.time_ns - self._sink_ns) / 1e9
        self._sink_ns = fb.time_ns
        pos, rates = self.cefd.filtration(list(self.entities.beliefs.values()), self.fields)
        if dt <= 0 or not pos:
            return set()
        self.fields.apply_sink("turbidity", pos, rates, dt)
        return {"turbidity"}

    def _field_to_entity(self) -> dict[UUID, float]:
        changed: dict[UUID, float] = {}
        if not self.cfg.switches.fields:
            return changed
        for bid, b in self.entities.beliefs.items():
            res = self.cefd.stress(self.fields, b)
            if res is None:
                continue
            p, gate = res
            old = b.stress_p
            b.stress_p = p
            b.stress_time_ns = self.fields.fields["temperature"].time_ns
            b.stress_drift_per_day = self.cefd.stress_drift(b, gate)
            if old is None or abs(p - old) > self.cfg.material_change_fraction:
                changed[bid] = gate
        return changed
