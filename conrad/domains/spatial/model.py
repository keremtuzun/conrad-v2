"""Model2S: UAHSM spatial belief child (ch14, ch33). BELIEF PLANE.

Consumes only Evidence / Observation objects (payloads through the ObjectStore), each observation's
ESTIMATED robot pose with its covariance, and the robot's sensor configuration (``SensorSpec`` from
mission context). It never sees truth, true poses or simulator entity IDs.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.base import Model2Child
from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.evidence import GEOMETRIC_MODALITIES, GeometricEvidenceEncoder, capture_group
from conrad.domains.spatial.export import GridExport, export_map
from conrad.domains.spatial.hierarchy import HierarchicalMap, PointState
from conrad.domains.spatial.integrate import MassFn
from conrad.domains.spatial.keys import FloatArr, encode, region_indices
from conrad.domains.spatial.measurements import depth_mass_fn, point_cloud_mass_fn, sonar_mass_fn
from conrad.domains.spatial.publisher import RegionChange, RegionPublisher, RegistryAsset, blocks_of
from conrad.domains.spatial.queries import SpatialQueries
from conrad.domains.spatial.relational import relational_fill
from conrad.domains.spatial.sensing import SpatialSensingError, sensor_frame
from conrad.domains.spatial.temporal import predict_level
from conrad.domains.spatial.view import filter_messages
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.schemas.belief import Availability, BeliefMessage, BeliefQuery, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, EvidenceValidity, Modality, Observation, SensorHealth
from conrad.schemas.timebase import NS_PER_S, ClockDomainError, TimeStamp
from conrad.schemas.world import Domain, SensorSpec

_MASS_FNS = {
    Modality.DEPTH_RANGE: depth_mass_fn,
    Modality.POINT_CLOUD: point_cloud_mass_fn,
    Modality.SONAR: sonar_mass_fn,
}


class Model2S(SpatialQueries, Model2Child):
    domain = Domain.SPATIAL

    def __init__(
        self,
        ids: IdFactory,
        store: ObjectStore,
        cfg: SpatialConfig | None = None,
        repository: Repository | None = None,
        run_id: UUID | None = None,
    ) -> None:
        self.cfg = cfg or SpatialConfig()
        self.model_version = self.cfg.model_version
        self.ids, self.store, self.repository = ids, store, repository
        self.map = HierarchicalMap(self.cfg)
        self.publisher = RegionPublisher(self.cfg, ids, repository, run_id)
        self.sensors: dict[UUID, SensorSpec] = {}
        self.registry: list[RegistryAsset] = []
        self.diagnostics: Counter[str] = Counter()
        self._initialized = False
        self._clock: str | None = None
        self._consumed: set[UUID] = set()
        self._last_direct_ns: int | None = None
        self._last_now_ns: int | None = None
        self._encoder: GeometricEvidenceEncoder | None = None
        self._reset_working()

    # ------------------------------------------------------------------ lifecycle
    def _reset_working(self) -> None:
        self._pending: dict[UUID, Evidence] = {}
        self._observations: dict[UUID, Observation] = {}
        self._context: list[BeliefMessage] = []
        self._last_rejected = 0
        self.map.clear_local()

    def initialize(self, context: dict[str, Any]) -> None:
        """``sensors``: the robot's SensorSpecs (required); ``asset_registry``: known assets; ``run_id``."""
        if "sensors" not in context:
            raise ValueError("Model2S needs the robot sensor configuration in context['sensors']")
        self.sensors = {s.sensor_id: s for s in context["sensors"]}
        self.registry = [
            a if isinstance(a, RegistryAsset) else RegistryAsset(**a)
            for a in context.get("asset_registry", ())
        ]
        if context.get("run_id") is not None:
            self.publisher.run_id = context["run_id"]
        self._initialized = True

    def reset_working_memory(self) -> None:
        """Drops pending evidence, cached observations, context and the LOCAL refined level; the regional
        map, belief identities, revision heads and the evidence archive survive (CC-09)."""
        self._reset_working()

    # ------------------------------------------------------------------ input
    def _check_clock(self, ts: TimeStamp) -> None:
        if self._clock is None:
            self._clock = ts.clock_domain
        elif ts.clock_domain != self._clock:
            raise ClockDomainError(f"Model2S runs in {self._clock!r}; got {ts.clock_domain!r}")

    def register_observations(self, observations: Sequence[Observation]) -> None:
        for o in observations:
            self._observations[o.observation_id] = o
            if self.repository is not None:
                self.repository.put_observation(o)

    def ingest(self, evidence: Sequence[Evidence]) -> None:
        for ev in evidence:
            if ev.evidence_id in self._consumed or ev.evidence_id in self._pending:
                self.diagnostics["duplicate_evidence"] += 1
                continue
            self._check_clock(ev.timestamp)
            self._pending[ev.evidence_id] = ev
            if self.repository is not None:
                self.repository.put_evidence(ev)

    def ingest_observations(self, observations: Sequence[Observation]) -> list[Evidence]:
        """Convenience path: register observations, encode them with the geometric encoder, ingest."""
        self.register_observations(observations)
        if self._encoder is None:
            self._encoder = GeometricEvidenceEncoder(self.ids, self.store)
        out = []
        for o in observations:
            ev, rec = self._encoder.encode(o)
            if self.repository is not None and self.publisher.run_id is not None:
                self.repository.put_provenance(self.publisher.run_id, [rec])
            out.append(ev)
        self.ingest(out)
        return out

    def _observation(self, ev: Evidence) -> Observation:
        obs = self._observations.get(ev.source_observation_id)
        if obs is None and self.repository is not None:
            obs = self.repository.observation(ev.source_observation_id)
        if obs is None:
            raise SpatialSensingError(f"evidence {ev.evidence_id}: source observation is not available")
        return obs

    def _mass_fn(
        self, ev: Evidence, obs: Observation, depth_captures: set[str]
    ) -> tuple[MassFn, FloatArr] | None:
        if ev.validity is EvidenceValidity.INVALID or obs.sensor_health is SensorHealth.FAULT:
            self.diagnostics["invalid"] += 1
            return None
        if obs.modality not in GEOMETRIC_MODALITIES:
            self.diagnostics["non_geometric"] += 1
            return None
        if obs.sensor_id not in self.sensors:
            raise SpatialSensingError(f"sensor {obs.sensor_id} is not in the robot sensor configuration")
        if obs.robot_pose_estimate is None or obs.payload_ref is None:
            self.diagnostics["rejected_no_pose_or_payload"] += 1
            self._last_rejected += 1
            return None
        if (
            obs.modality is Modality.POINT_CLOUD
            and self.cfg.sensor.dedupe_point_cloud_with_depth
            and capture_group(obs) in depth_captures
        ):
            self.diagnostics["duplicate_capture"] += 1
            return None
        oc = self.cfg.occupancy
        q = ev.reliability if oc.use_evidence_reliability else 1.0
        if ev.validity is EvidenceValidity.DEGRADED or obs.sensor_health is SensorHealth.DEGRADED:
            q *= oc.degraded_health_weight
        if q <= 0:
            self.diagnostics["zero_quality"] += 1
            return None
        spec = self.sensors[obs.sensor_id]
        frame = sensor_frame(obs.robot_pose_estimate, spec, self.cfg)
        arr = self.store.get_array(obs.payload_ref)
        return _MASS_FNS[obs.modality](frame, arr, spec, self.cfg, q), frame.robot_position

    # ------------------------------------------------------------------ belief update
    def update_beliefs(self, now: TimeStamp) -> list[BeliefMessage]:
        if not self._initialized:
            raise RuntimeError("Model2S.update_beliefs before initialize()")
        self._check_clock(now)
        self._last_rejected = 0
        pending = sorted(self._pending.values(), key=lambda e: (e.timestamp.time_ns, str(e.evidence_id)))
        resolved = [(ev, self._observation(ev)) for ev in pending]
        depth_caps = {capture_group(o) for _, o in resolved if o.modality is Modality.DEPTH_RANGE}
        block_ev: dict[int, list[UUID]] = {}
        block_t: dict[int, int] = {}
        focus: FloatArr | None = None
        for ev, obs in resolved:
            got = self._mass_fn(ev, obs, depth_caps)
            if got is not None:
                touched = self.map.integrate(got[0], ev.evidence_id, obs.timestamp.time_ns)
                focus = got[1]
                self.diagnostics["integrated"] += 1
                self._last_direct_ns = max(self._last_direct_ns or 0, obs.timestamp.time_ns)
                for b in blocks_of(self.map, touched).tolist():
                    block_ev.setdefault(b, []).append(ev.evidence_id)
                    block_t[b] = max(block_t.get(b, 0), obs.timestamp.time_ns)
            self._consumed.add(ev.evidence_id)
        self._pending.clear()
        rel_codes, refs = relational_fill(self.map.base, self.cfg, self.map.base_rows)
        self.map.refine(focus if focus is not None else self.map.focus)
        self._last_now_ns = now.time_ns
        return self.publisher.publish_changes(
            self.map, block_ev, block_t, (rel_codes, refs), self.registry, now, self.availability()
        )

    def predict(self, delta_t_s: float, now: TimeStamp) -> list[BeliefMessage]:
        """Dynamic cells decay toward ignorance (PREDICTED); static cells persist unchanged and age."""
        self._check_clock(now)
        codes = [predict_level(lvl, delta_t_s, self.cfg) for lvl in self.map.levels]
        self._last_now_ns = now.time_ns
        base = codes[0]
        avail = self.availability()
        out = []
        for b in blocks_of(self.map, base).tolist():
            ch = RegionChange(
                UpdateKind.PREDICTED, (), now.time_ns, f"dynamic cells predicted over {delta_t_s:.3f} s"
            )
            out.append(self.publisher.publish_block(self.map, b, ch, now, avail))
        return out

    def receive_context(self, messages: Sequence[BeliefMessage]) -> None:
        """Cross-domain context only annotates semantics (claim ``semantic_class``); occupancy is untouched."""
        for m in messages:
            if m.domain is Domain.SPATIAL:
                self.diagnostics["context_same_domain_ignored"] += 1
                continue
            self._context.append(m)
            label: str | None = None
            for c in m.state_summary:
                if c.name == "semantic_class" and isinstance(c.value, str):
                    label = c.value
            if (
                label is None
                or m.spatial_support is None
                or m.spatial_support.frame_id != self.cfg.grid.frame_id
            ):
                continue
            sup = m.spatial_support
            idx = region_indices(np.asarray(sup.center_m), np.asarray(sup.half_extent_m), self.map.base.res)
            rows = self.map.base.rows(encode(idx))
            for r in rows[rows >= 0].tolist():
                self.map.base.semantic[r] = label
            self.diagnostics["context_semantic_cells"] += int((rows >= 0).sum())

    # ------------------------------------------------------------------ output
    def query(self, query: BeliefQuery) -> list[BeliefMessage]:
        return filter_messages(list(self.publisher.latest.values()), query, self.cfg.grid.frame_id)

    def export_beliefs(self) -> list[BeliefMessage]:
        return sorted(self.publisher.latest.values(), key=lambda m: str(m.belief_id))

    def availability(self) -> Availability:
        if not self._initialized:
            return Availability.UNAVAILABLE
        last, now = self._last_direct_ns, self._last_now_ns
        if last is not None and now is not None and (now - last) / NS_PER_S > self.cfg.temporal.stale_after_s:
            return Availability.STALE
        return Availability.DEGRADED if self._last_rejected else Availability.AVAILABLE

    def occupancy_state(self, points: FloatArr) -> PointState:
        return self.map.point_state(points)

    def export_grid(self, now_ns: int | None = None) -> GridExport:
        return export_map(self.map, now_ns if now_ns is not None else (self._last_now_ns or 0))

    def context_messages(self) -> list[BeliefMessage]:
        return list(self._context)
