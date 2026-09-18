"""Twin2E: ecological/environmental truth, S = (H, Z, Phi, R, Theta) via MEIFE. TRUTH PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.base import digest_of
from conrad.schemas.frames import FramedPoint
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import NS_PER_S, ClockDomainError, TimeStamp
from conrad.schemas.truth import TruthState
from conrad.schemas.world import Domain, Scenario, ScenarioEvent
from conrad.twins.base import SensingContext, Twin, TwinSample
from conrad.twins.twin2e.config import Twin2EConfig
from conrad.twins.twin2e.ecology import SESSILE, EcologyEngine, EntityState
from conrad.twins.twin2e.events import apply_event
from conrad.twins.twin2e.fields import FIELD_UNITS, FieldEngine, FieldParams
from conrad.twins.twin2e.observation import ObservationGenerator
from conrad.twins.twin2e.priors import (
    ENTITY_PRIORS,
    ENVIRONMENT_PRIORS,
    KIND_BIOFOULING,
    KIND_HABITAT,
    PARAMETER_PRIORS,
    Twin2EScenarioError,
    env_setting,
    populate_ecological_state,
)
from conrad.twins.twin2e.residual import LearnedResidualHook, ResidualFn

_STREAMS = {"prior": 1, "current": 2, "temperature": 3, "turbidity": 4, "observation": 5, "features": 6}


def _words(eid: UUID) -> tuple[int, ...]:
    return tuple((eid.int >> (32 * i)) & 0xFFFFFFFF for i in range(4))


class Twin2ENotInitializedError(RuntimeError):
    pass


class Twin2E(Twin):
    domain = Domain.ECOLOGICAL
    twin_version = "twin2e-meife-0.1.0"

    def __init__(
        self,
        ids: IdFactory,
        store: ObjectStore,
        config: Twin2EConfig | None = None,
        residual_fn: ResidualFn | None = None,
    ) -> None:
        super().__init__(ids, store)
        self.cfg = config or Twin2EConfig()
        self.residual = LearnedResidualHook(self.cfg.learned_residual, residual_fn)
        self._scenario: Scenario | None = None
        self._source_scenario: Scenario | None = None

    # ------------------------------------------------------------------ lifecycle
    def _rng(self, stream: str, extra: tuple[int, ...] = ()) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, _STREAMS.get(stream, 99), *extra]))

    def initialize(self, scenario: Scenario, seed: int | None = None) -> None:
        self._source_scenario = scenario
        self.seed = scenario.seed if seed is None else seed
        if self._needs_priors(scenario):
            if not self.cfg.allow_prior_sampling:
                raise Twin2EScenarioError("scenario lacks Twin2E keys and prior sampling is disabled")
            scenario = populate_ecological_state(scenario, self._rng("prior"))
        self._scenario = scenario
        env, par = scenario.environment, scenario.ecological_state["parameters"]
        self.fields = FieldEngine(
            self.cfg,
            FieldParams(
                surface_temperature_c=float(env_setting(env, "temperature")),
                lapse_c_per_m=float(par["temperature_lapse_c_per_m"]),
                turbidity_bg_ntu=float(env_setting(env, "turbidity")),
                current_base_m_s=np.asarray(env_setting(env, "current"), dtype=np.float64),
                surface_light_w_m2=float(env_setting(env, "light")),
                kd_water_per_m=float(par["kd_water_per_m"]),
                k_turbidity=float(par["k_turbidity_per_m_per_ntu"]),
                kh_m2_s=float(par["diffusivity_h_m2_s"]),
                kv_m2_s=float(par["diffusivity_v_m2_s"]),
                resuspension_critical_m_s=float(par["resuspension_critical_speed_m_s"]),
                resuspension_rate=float(par["resuspension_rate"]),
                settling_time_s=float(par["settling_time_s"]),
                temperature_relaxation_s=float(par["temperature_relaxation_time_s"]),
                ou_sigma_m_s=float(par["current_fluctuation_sigma_m_s"]),
                ou_time_s=float(par["current_fluctuation_time_s"]),
                filtration_m3_s_per_m2=float(par["filtration_m3_s_per_m2"]),
                start_time_of_day_h=float(par["start_time_of_day_h"]),
                surface_z_m=float(par["surface_z_m"]),
            ),
        )
        self.entities = self._build_entities(scenario)
        self.ecology = EcologyEngine(self.cfg.ecology, self.cfg.switches)
        self.field_rngs = {k: self._rng(k) for k in ("current", "temperature", "turbidity")}
        self.entity_rngs = {eid: self._rng("entity", _words(eid)) for eid in self.entities}
        self.observer = ObservationGenerator(self, self._rng("observation"), self._rng("features"))
        self.t_s = 0.0
        self.step_index = 0
        self.applied_events: set[UUID] = set()
        self.event_log: list[dict[str, Any]] = []
        for e in self.entities.values():
            e.frozen_env = self.local_env(e)
        EcologyEngine.aggregate(self.entities)

    def _needs_priors(self, s: Scenario) -> bool:
        eco = s.ecological_state
        if any(k not in s.environment for k in ENVIRONMENT_PRIORS):
            return True
        if any(k not in eco.get("parameters", {}) for k in PARAMETER_PRIORS):
            return True
        entries = eco.get("entities", {})
        for e in s.world_entities:
            if not e.domain_ownership.ecological:
                continue
            entry = entries.get(str(e.id))
            if entry is None or "entity_kind" not in entry:
                return True
            if any(k not in entry for k in ENTITY_PRIORS.get(entry["entity_kind"], {})):
                return True
        return False

    def _build_entities(self, s: Scenario) -> dict[UUID, EntityState]:
        entries = s.ecological_state["entities"]
        owned = {e.id: e for e in s.world_entities if e.domain_ownership.ecological}
        for key in entries:
            if UUID(str(key)) not in owned:
                raise Twin2EScenarioError(f"ecological state for non-ecological or unknown entity {key}")
        out: dict[UUID, EntityState] = {}
        for eid, we in sorted(owned.items(), key=lambda kv: str(kv[0])):
            entry = entries[str(eid)]
            kind = entry["entity_kind"]
            pos = entry.get("position_m")
            st = EntityState(
                entity_id=eid,
                kind=kind,
                parent_id=we.parent_id,
                position_m=np.zeros(3) if pos is None else np.asarray(pos, dtype=np.float64),
                radius_m=float(entry["radius_m"]),
                params=dict(entry),
                cover=float(entry.get("initial_cover", 0.0)),
                condition=float(entry.get("initial_condition", 1.0)),
            )
            if pos is None and kind != KIND_HABITAT:
                raise Twin2EScenarioError(f"entity {eid} has no position (no Twin2S context, no sampler)")
            if kind == "MOBILE_GROUP":
                st.presence = bool(
                    self._rng("entity", (7, *_words(eid))).uniform() < entry["initial_presence_probability"]
                )
            out[eid] = st
        for h in out.values():
            if h.kind == KIND_HABITAT and h.params.get("position_m") is None:
                kids = [k.position_m for k in out.values() if k.parent_id == h.entity_id]
                if not kids:
                    raise Twin2EScenarioError(f"habitat {h.entity_id} has neither position nor members")
                h.position_m = np.mean(kids, axis=0)
        for e in out.values():
            if not self.fields.regional_grid.contains(e.position_m):
                raise Twin2EScenarioError(f"entity {e.entity_id} lies outside the regional field grid")
        return out

    def _require(self) -> Scenario:
        if self._scenario is None:
            raise Twin2ENotInitializedError("Twin2E.initialize(scenario) has not been called")
        return self._scenario

    def reset(self, seed: int) -> None:
        if self._source_scenario is None:
            raise Twin2ENotInitializedError("reset() before initialize()")
        self.initialize(self._source_scenario, seed=seed)

    # ------------------------------------------------------------------ dynamics
    def step(self, dt_s: float, events: Sequence[ScenarioEvent] = ()) -> None:
        scen = self._require()
        if not (dt_s > 0 and math.isfinite(dt_s)):
            raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
        t_end = self.t_s + dt_s
        due = [e for e in scen.events if e.time_s <= t_end] + list(events)
        for ev in sorted(due, key=lambda e: (e.time_s, str(e.event_id))):
            if ev.event_id in self.applied_events:
                continue
            self.applied_events.add(ev.event_id)
            self.event_log.append(apply_event(ev, self.t_s, self.fields, self.entities, self.cfg.switches))
        sw = self.cfg.switches
        if sw.field_dynamics:
            sources = [(e.position_m, e.covered_area_m2) for e in self.entities.values() if e.kind in SESSILE]
            self.fields.step(
                dt_s, t_end, self.field_rngs, sources, sw.stochasticity, sw.multi_scale, sw.entity_to_field
            )
            if self.residual.active:
                for st in (self.fields.regional, self.fields.local):
                    st.temperature = self.residual.apply("temperature", st.temperature, dt_s)
                    st.turbidity = np.clip(self.residual.apply("turbidity", st.turbidity, dt_s), 0.0, None)
        env = {eid: self.local_env(e) for eid, e in self.entities.items()}
        self.ecology.step(self.entities, env, dt_s, self.entity_rngs)
        self.t_s = t_end
        self.step_index += 1

    # ------------------------------------------------------------------ truth
    def now(self) -> TimeStamp:
        return TimeStamp(
            time_ns=round(self.t_s * NS_PER_S),
            clock_domain=self.cfg.clock_domain,
            sequence_index=self.step_index,
        )

    def check_time(self, ts: TimeStamp) -> None:
        if ts.clock_domain != self.cfg.clock_domain:
            raise ClockDomainError(f"Twin2E runs in {self.cfg.clock_domain!r}, got {ts.clock_domain!r}")
        if abs(ts.time_ns - round(self.t_s * NS_PER_S)) > 1_000_000:
            raise ValueError(f"Twin2E is at t={self.t_s}s; cannot report state at {ts.seconds}s")

    def local_env(self, e: EntityState) -> dict[str, float]:
        p = e.position_m[None]
        cur = self.fields.sample("current", p)[0]
        return {
            "temperature": float(self.fields.sample("temperature", p)[0]),
            "turbidity": float(self.fields.sample("turbidity", p)[0]),
            "light": float(self.fields.sample("light", p)[0]),
            "current_speed": float(np.linalg.norm(cur)),
        }

    def get_truth(self, timestamp: TimeStamp) -> list[TruthState]:
        scen = self._require()
        self.check_time(timestamp)
        out = []
        for eid, e in sorted(self.entities.items(), key=lambda kv: str(kv[0])):
            state = e.as_truth()
            state["exposed_to"] = self.local_env(e)
            state["exposed_to_units"] = {
                **{k: FIELD_UNITS[k] for k in ("temperature", "turbidity", "light")},
                "current_speed": "m s-1",
            }
            rel = tuple(k.entity_id for k in self.entities.values() if k.parent_id == eid)
            if e.parent_id is not None:
                rel = (e.parent_id, *rel)
            out.append(
                TruthState(
                    scenario_id=scen.scenario_id,
                    world_entity_id=eid,
                    domain=Domain.ECOLOGICAL,
                    timestamp=timestamp,
                    state=state,
                    relationships=rel,
                    event_context={"events_applied": len(self.event_log)},
                    generator_version=self.twin_version,
                )
            )
        return out

    def surface_cover(self, entity_id: UUID) -> float:
        self._require()
        if entity_id not in self.entities:
            raise KeyError(f"entity {entity_id} has no ecological state in Twin2E")
        e = self.entities[entity_id]
        return 0.0 if e.kind == "MOBILE_GROUP" else float(e.cover)

    def observability_modifiers(self, position: FramedPoint) -> dict[str, Any]:
        """Truth-plane hook for Twin2S/Twin2T/renderer. Biofouling affects OBSERVABILITY only."""
        self._require()
        self.fields.local_grid.check_frame(position.frame_id)
        p = position.array()
        turb = float(self.fields.sample("turbidity", p[None])[0])
        o = self.cfg.observation
        fouling, over = 0.0, None
        for e in self.entities.values():
            if (
                e.kind == KIND_BIOFOULING
                and np.linalg.norm(e.position_m - p) <= e.radius_m
                and e.cover >= fouling
            ):
                fouling, over = e.cover, e.entity_id
        return {
            "frame_id": position.frame_id,
            "turbidity_ntu": turb,
            "beam_attenuation_per_m": o.beam_attenuation_clear_per_m
            + o.beam_attenuation_per_m_per_ntu * turb,
            "ambient_light_w_m2": float(self.fields.sample("light", p[None])[0]),
            "biofouling_cover": fouling,
            "biofouling_entity_id": over,
            "surface_occlusion_fraction": fouling,
            "biofouling_implies_corrosion": False,
            "units": {"turbidity_ntu": "NTU", "beam_attenuation_per_m": "m-1", "ambient_light_w_m2": "W m-2"},
        }

    def generate_observation(self, ctx: SensingContext) -> list[TwinSample]:
        self._require()
        self.check_time(ctx.timestamp)
        return self.observer.generate(ctx)

    def export_domain_state(self) -> dict[str, Any]:
        scen = self._require()
        fields: dict[str, Any] = {}
        for st in (self.fields.regional, self.fields.local):
            for name, units in FIELD_UNITS.items():
                ref = self.store.put_array(st.get(name))
                fields[f"{st.grid.name}/{name}"] = {
                    "values_ref": ref.model_dump(mode="json"),
                    "units": units,
                    "grid": st.grid.geometry(),
                    "timestamp_ns": self.now().time_ns,
                }
        return {
            "domain": Domain.ECOLOGICAL.value,
            "twin_version": self.twin_version,
            "scenario_id": str(scen.scenario_id),
            "seed": self.seed,
            "time_s": self.t_s,
            "step_index": self.step_index,
            "Z": {str(k): v.as_truth() for k, v in sorted(self.entities.items(), key=lambda kv: str(kv[0]))},
            "Phi": fields,
            "event_log": list(self.event_log),
            "prior_log": list(scen.ecological_state.get("prior_log", [])),
            "switches": self.cfg.switches.model_dump(),
            "config_digest": digest_of(self.cfg.model_dump(mode="json")),
            "learned_residual_active": self.residual.active,
            "last_cfl": {k: v.cfl_number for k, v in self.fields.last_reports.items()},
        }
