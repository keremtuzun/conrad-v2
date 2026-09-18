"""Twin2T: structural/degradation truth twin over the MCDE (ch10, ch11, ch33 Twin2T freeze).

Per tick: corrosion -> fatigue -> coupling -> event/intervention (MCDE), then observation masking at
sensing time. Truth leaves only through ``get_truth`` / ``structural_truth`` / ``export_*`` and
``TwinSample.supervision``; Observations carry sensor-shaped values only.

There is deliberately NO accessor for the hidden mechanism parameters or the transition function: the
truth/model separation of ch10 forbids handing F_Twin to Model 2T. Parameters appear only in the audit
export (``export_domain_state()["prior_log"]``), which is truth-plane data.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any
from uuid import UUID

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.truth import TruthState
from conrad.schemas.world import Domain, Scenario, ScenarioEvent
from conrad.twins.base import SensingContext, Twin, TwinSample
from conrad.twins.twin2t.assembly import Assembly, assemble
from conrad.twins.twin2t.baselines import GeneratorKind, make_engine
from conrad.twins.twin2t.config import Twin2TConfig
from conrad.twins.twin2t.coupling import export_coupling_graph
from conrad.twins.twin2t.emission import emit_sample
from conrad.twins.twin2t.events import parse_event_type
from conrad.twins.twin2t.mcde import MCDE, MCDEStepResult, StructuralEvent
from conrad.twins.twin2t.observation import (
    FidelityLevel,
    ObservationModel,
    SensingQuality,
    SensorLevelRenderer,
    surface_appearance_of,
    visibility_of,
)
from conrad.twins.twin2t.state import STATE_DIMENSIONS, ComponentRuntime, status_of

TWIN2T_VERSION = "twin2t-0.1.0"


class Twin2TNotInitialized(RuntimeError):
    pass


class Twin2T(Twin):
    domain = Domain.TECHNICAL
    twin_version = TWIN2T_VERSION

    def __init__(self, ids: IdFactory, store: ObjectStore, config: Twin2TConfig | None = None) -> None:
        super().__init__(ids, store)
        self.config = config or Twin2TConfig()
        self._source: Scenario | None = None
        self._asm: Assembly | None = None
        self._mcde: MCDE | None = None
        self._obs: ObservationModel | None = None
        self._obs_rng: np.random.Generator | None = None
        self._seed = 0
        self._pending: list[tuple[float, StructuralEvent]] = []
        self._seen_events: set[UUID] = set()
        self._last: MCDEStepResult | None = None
        self._renderers: dict[Modality, SensorLevelRenderer] = {}
        self.truth_sequence: list[dict[str, Any]] = []
        self.observation_masks: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, scenario: Scenario) -> None:
        self._source = scenario
        self._build(scenario.seed)

    def reset(self, seed: int) -> None:
        if self._source is None:
            raise Twin2TNotInitialized("reset before initialize")
        self._build(seed)

    def _build(self, seed: int) -> None:
        assert self._source is not None
        self._seed = seed
        rng = np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF, 0x2]))
        self._asm = assemble(self._source, rng, prior_overrides=self.config.prior_overrides)
        kind = GeneratorKind(self.config.generator)
        self._mcde = make_engine(kind, self._asm.graph, self._asm.runtimes, self.config.mcde, seed)
        self._obs = ObservationModel(self.config.observation, seed)
        self._obs_rng = np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF, 0x0B5]))
        self._pending, self._seen_events, self._last = [], set(), None
        self.truth_sequence, self.observation_masks, self.event_log = [], [], []
        for ev in self._source.events:
            se = self._convert(ev)
            if se is not None:
                self._pending.append((ev.time_s, se))
        self._pending.sort(key=lambda p: (p[0], str(p[1].event_id)))
        self._record_truth()

    def _require(self) -> tuple[Assembly, MCDE]:
        if self._asm is None or self._mcde is None:
            raise Twin2TNotInitialized("Twin2T.initialize(scenario) has not been called")
        return self._asm, self._mcde

    def _convert(self, ev: ScenarioEvent) -> StructuralEvent | None:
        etype = parse_event_type(ev.event_type)
        if etype is None:
            self.event_log.append({"event_id": str(ev.event_id), "ignored": "not a structural event type"})
            return None
        return StructuralEvent(etype, ev.target_entity_id, dict(ev.parameters), ev.event_id)

    # ------------------------------------------------------------------ dynamics
    def step(self, dt_s: float, events: Sequence[ScenarioEvent] = ()) -> None:
        self.advance(dt_s, events)

    def advance(self, dt_s: float, events: Sequence[ScenarioEvent] = ()) -> MCDEStepResult:
        """MCDE step API (ch10). Scheduled scenario events due by t+dt fire in this tick."""
        _, mcde = self._require()
        horizon = mcde.time_s + dt_s
        due = [e for t, e in self._pending if t <= horizon]
        self._pending = [(t, e) for t, e in self._pending if t > horizon]
        for ev in events:
            se = self._convert(ev)
            if se is not None:
                due.append(se)
        batch: list[StructuralEvent] = []
        for se in due:
            if se.event_id is not None:
                if se.event_id in self._seen_events:
                    continue
                self._seen_events.add(se.event_id)
            batch.append(se)
        self._last = mcde.step(dt_s, batch)
        for eid, rec in self._last.records.items():
            for e in rec.events:
                self.event_log.append({"time_s": mcde.time_s, "entity_id": str(eid), **e})
        self.event_log.extend({"time_s": mcde.time_s, **i} for i in self._last.ignored_events)
        self._record_truth()
        return self._last

    def _record_truth(self) -> None:
        _, mcde = self._require()
        arr, mask = mcde.truth_arrays()
        self.truth_sequence.append(
            {
                "time_s": mcde.time_s,
                "entity_ids": [str(e) for e in mcde.order],
                "state": arr.tolist(),
                "mask": mask.tolist(),
            }
        )

    # ------------------------------------------------------------------ truth
    def _runtime(self, entity_id: UUID) -> ComponentRuntime:
        asm, _ = self._require()
        if entity_id not in asm.runtimes:
            raise KeyError(f"{entity_id} has no Twin2T degradation state")
        return asm.runtimes[entity_id]

    def _component_truth(self, rt: ComponentRuntime) -> dict[str, Any]:
        s = rt.state
        rec = None if self._last is None else self._last.records.get(rt.component.entity_id)
        return {
            **{d: getattr(s, d) for d in STATE_DIMENSIONS},
            "validity": dict(s.validity),
            "status": status_of(s, rt.effective_wall_m).value,
            "component_type": rt.component.component_type.name,
            "material": None if rt.component.material is None else rt.component.material.name,
            "wall_thickness_m": rt.params.wall_thickness_m,
            "effective_wall_thickness_m": rt.effective_wall_m,
            "environment": asdict(rt.environment),
            "loading": asdict(rt.loading),
            "mechanism_state": {}
            if rec is None
            else {
                "corrosion_rate_m_per_yr": rec.corrosion_rate_m_per_yr,
                "effective_stress_range_pa": rec.effective_stress_range_pa,
                "delta_k_mpa_sqrt_m": rec.delta_k,
                "delta_k_threshold_mpa_sqrt_m": rec.delta_k_threshold,
                "crack_grew": rec.crack_grew,
                "coupling": dict(rec.coupling),
            },
        }

    def get_truth(self, timestamp: TimeStamp) -> list[TruthState]:
        asm, mcde = self._require()
        out = []
        for eid in mcde.order:
            rt = asm.runtimes[eid]
            rels = tuple(
                sorted(
                    {r.target if r.source == eid else r.source for r in asm.graph.relationships_of(eid)},
                    key=str,
                )
            )
            evs = [] if self._last is None else list(self._last.records[eid].events)
            out.append(
                TruthState(
                    scenario_id=asm.scenario.scenario_id,
                    world_entity_id=eid,
                    domain=Domain.TECHNICAL,
                    timestamp=timestamp,
                    state=self._component_truth(rt),
                    relationships=rels,
                    event_context={"events": evs},
                    generator_version=self.config.generator_version,
                )
            )
        return out

    def structural_truth(self, timestamp: TimeStamp) -> dict[str, Any]:
        """ch10 StructuralTruthState(timestamp, component_states, mechanism_states, relationships, events,
        environmental_context). Training/evaluation only."""
        asm, mcde = self._require()
        comps = {str(e): self._component_truth(asm.runtimes[e]) for e in mcde.order}
        return {
            "timestamp": timestamp.model_dump(mode="json"),
            "component_states": comps,
            "mechanism_states": {k: v.pop("mechanism_state") for k, v in comps.items()},
            "relationships": asm.graph.export()["relationships"],
            "events": []
            if self._last is None
            else [{"entity_id": str(k), **e} for k, r in self._last.records.items() for e in r.events],
            "environmental_context": {k: v["environment"] for k, v in comps.items()},
        }

    def surface_appearance(self, entity_id: UUID) -> dict[str, Any]:
        """Appearance API for a renderer (Twin2S / Unity): rust coverage, pitting scale, crack length."""
        return surface_appearance_of(self._runtime(entity_id))

    def truth_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self._require()[1].truth_arrays()

    @property
    def time_s(self) -> float:
        return self._require()[1].time_s

    @property
    def component_ids(self) -> tuple[UUID, ...]:
        """Truth-plane ids of components that carry degradation state (for scenario tooling only)."""
        return self._require()[1].order

    def export_domain_state(self) -> dict[str, Any]:
        asm, mcde = self._require()
        return {
            "twin_version": self.twin_version,
            "generator_version": self.config.generator_version,
            "scenario_id": str(asm.scenario.scenario_id),
            "seed": self._seed,
            "time_s": mcde.time_s,
            "component_graph": asm.graph.export(),
            "coupling_graph": export_coupling_graph(),
            "mcde_config": {
                k: (v.value if hasattr(v, "value") else v) for k, v in asdict(self.config.mcde).items()
            },
            "prior_log": asm.prior_log,
            "state_dimensions": list(STATE_DIMENSIONS),
            "truth_sequence": self.truth_sequence,
            "observation_masks": self.observation_masks,
            "events": [e for e in self.event_log if not e.get("intervention")],
            "interventions": [e for e in self.event_log if e.get("intervention")],
        }

    # ------------------------------------------------------------------ observation
    def register_sensor_renderer(self, modality: Modality, renderer: SensorLevelRenderer) -> None:
        self._renderers[modality] = renderer

    def generate_observation(self, ctx: SensingContext) -> list[TwinSample]:
        asm, mcde = self._require()
        assert self._obs is not None and self._obs_rng is not None
        q = SensingQuality.from_degradation(ctx.degradation)
        fidelity = FidelityLevel(
            str(ctx.sensor.parameters.get("t2t_fidelity", self.config.observation.default_fidelity))
        )
        thr = self.config.observation.visibility_threshold
        family = str(asm.scenario.metadata.get("world_family", "twin2t"))
        lineage = f"{asm.scenario.scenario_id}/{family}/{self._seed}"
        mask: dict[str, bool] = {}
        samples: list[TwinSample] = []
        for eid in mcde.order:
            vis = visibility_of(ctx.degradation, eid, q.occlusion)
            observed = vis > thr and not q.missing_modality
            mask[str(eid)] = observed
            if observed:
                samples.append(
                    emit_sample(
                        ids=self.ids,
                        store=self.store,
                        model=self._obs,
                        rng=self._obs_rng,
                        renderers=self._renderers,
                        rt=asm.runtimes[eid],
                        ctx=ctx,
                        quality=q,
                        fidelity=fidelity,
                        visibility=vis,
                        lineage=lineage,
                    )
                )
        self.observation_masks.append(
            {"time_s": mcde.time_s, "sensor_id": str(ctx.sensor.sensor_id), "observed": mask}
        )
        return samples
