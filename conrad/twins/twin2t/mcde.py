"""Mechanism-Coupled Degradation Engine (ch10 MCDE computational architecture, MCDE step API; ch33).

Tick order (ch33 Twin2T freeze): corrosion -> fatigue -> coupling -> event/intervention; observation
masking happens afterwards in the observation layer. Each component owns an independent random stream
derived from (seed, entity UUID), so one component's state can never change another's noise.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE (H-T2T-01)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.coupling import (
    shared_environment_targets,
    update_mechanism_coupling,
    update_topology_coupling,
)
from conrad.twins.twin2t.events import (
    GLOBAL_OK,
    INTERVENTIONS,
    StructuralEventError,
    StructuralEventType,
    apply_event,
)
from conrad.twins.twin2t.mechanisms import coating_step, corrosion_step, fatigue_step
from conrad.twins.twin2t.state import STATE_DIMENSIONS, ComponentRuntime
from conrad.twins.twin2t.topology import StructuralGraph


@dataclass(frozen=True)
class StructuralEvent:
    event_type: StructuralEventType
    target: UUID | None
    parameters: dict[str, Any] = field(default_factory=dict)
    event_id: UUID | None = None


@dataclass
class ComponentStepRecord:
    entity_id: UUID
    before: tuple[float, ...]
    after: tuple[float, ...]
    mask: tuple[bool, ...]
    corrosion_rate_m_per_yr: float = 0.0
    effective_stress_range_pa: float = 0.0
    delta_k: float = 0.0
    delta_k_threshold: float = 0.0
    crack_grew: bool = False
    coupling: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def intervened(self) -> bool:
        return any(e["intervention"] for e in self.events)

    @property
    def event_applied(self) -> bool:
        return bool(self.events)


@dataclass
class MCDEStepResult:
    time_s: float
    dt_s: float
    records: dict[UUID, ComponentStepRecord]
    ignored_events: list[dict[str, Any]]


def component_stream(seed: int, entity_id: UUID) -> np.random.Generator:
    words = [(entity_id.int >> (32 * k)) & 0xFFFFFFFF for k in range(4)]
    return np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF, *words]))


class MCDE:
    def __init__(
        self,
        graph: StructuralGraph,
        runtimes: dict[UUID, ComponentRuntime],
        config: MCDEConfig,
        seed: int,
    ) -> None:
        for eid in runtimes:
            if eid not in graph:
                raise ValueError(f"runtime for component {eid} absent from the structural graph")
        self.graph = graph
        self.runtimes = runtimes
        self.config = config
        self.seed = seed
        self.time_s = 0.0
        self._rngs = {eid: component_stream(seed, eid) for eid in runtimes}

    @property
    def order(self) -> tuple[UUID, ...]:
        return tuple(i for i in self.graph.component_ids if i in self.runtimes)

    def _exposed_fraction(self, rt: ComponentRuntime) -> float:
        if not rt.params.coated or not rt.state.validity.get("coating_breakdown_fraction", False):
            return 1.0
        if self.config.mechanism_coupling:
            return rt.state.coating_breakdown_fraction
        return rt.initial_state.coating_breakdown_fraction  # uncoupled: corrosion ignores coating evolution

    def _loading(self, rt: ComponentRuntime) -> tuple[float, float]:
        cfg = self.config
        if cfg.loading_conditioning:
            dsig, f = rt.loading.stress_range_pa, rt.loading.load_cycles_per_s
        else:
            dsig, f = cfg.reference_stress_range_pa, cfg.reference_cycles_per_s
        if cfg.mechanism_coupling:
            dsig *= rt.stress_concentration
        if cfg.topology_coupling:
            dsig *= rt.load_multiplier
        return dsig, f

    def step(self, dt_s: float, events: Sequence[StructuralEvent] = ()) -> MCDEStepResult:
        if dt_s < 0:
            raise ValueError("dt_s must be >= 0")
        cfg = self.config
        recs: dict[UUID, ComponentStepRecord] = {}
        for eid in self.order:
            rt = self.runtimes[eid]
            recs[eid] = ComponentStepRecord(eid, rt.state.vector(), rt.state.vector(), rt.state.mask())
        # 1. corrosion (incl. coating ageing, which feeds corrosion exposure)
        for eid in self.order:
            rt, rng = self.runtimes[eid], self._rngs[eid]
            rt.state = coating_step(rt.state, rt.params, dt_s)
            res = corrosion_step(
                rt.state,
                rt.params,
                rt.environment,
                self._exposed_fraction(rt),
                dt_s,
                rng,
                cfg.corrosion_model,
                material_factor=rt.component.material.corrosion_rate_factor if rt.component.material else 0.0,
                stochastic=cfg.stochastic,
                environment_conditioned=cfg.environment_conditioning,
            )
            rt.state = res.state
            recs[eid].corrosion_rate_m_per_yr = res.rate_m_per_yr
        # 2. fatigue
        for eid in self.order:
            rt, rng = self.runtimes[eid], self._rngs[eid]
            dsig, f = self._loading(rt)
            fr = fatigue_step(
                rt.state,
                rt.params,
                dsig,
                f,
                dt_s,
                rng,
                stochastic=cfg.stochastic,
                max_crack_length_m=cfg.max_crack_length_m,
            )
            rt.state = fr.state
            r = recs[eid]
            r.effective_stress_range_pa, r.delta_k, r.delta_k_threshold, r.crack_grew = (
                dsig,
                fr.delta_k,
                fr.threshold,
                fr.grew,
            )
        # 3. coupling (sets conditions for the next tick)
        if cfg.mechanism_coupling:
            for eid in self.order:
                recs[eid].coupling.update(update_mechanism_coupling(self.runtimes[eid], cfg))
        if cfg.topology_coupling:
            for eid, mult in update_topology_coupling(self.graph, self.runtimes, cfg).items():
                recs[eid].coupling["load_multiplier"] = mult
        # 4. events / interventions
        ignored = self._apply_events(events, recs)
        self.time_s += dt_s
        for eid in self.order:
            recs[eid].after = self.runtimes[eid].state.vector()
        return MCDEStepResult(self.time_s, dt_s, recs, ignored)

    def _apply_events(
        self, events: Sequence[StructuralEvent], recs: dict[UUID, ComponentStepRecord]
    ) -> list[dict[str, Any]]:
        cfg = self.config
        ignored: list[dict[str, Any]] = []
        for ev in events:
            is_iv = ev.event_type in INTERVENTIONS
            if ev.event_type is not StructuralEventType.INSPECTION and (
                (is_iv and not cfg.interventions_enabled) or (not is_iv and not cfg.events_enabled)
            ):
                ignored.append({"event_type": ev.event_type.value, "reason": "disabled_by_ablation"})
                continue
            if ev.target is None:
                if ev.event_type not in GLOBAL_OK:
                    raise StructuralEventError(f"{ev.event_type.value} requires a target component")
                targets = list(self.order)
            else:
                if ev.target not in self.graph:
                    raise StructuralEventError(f"event targets unknown component {ev.target}")
                targets = [ev.target]
                if (
                    ev.event_type is StructuralEventType.ENVIRONMENT_CHANGE
                    and cfg.topology_coupling
                    and ev.parameters.get("scope", "shared") == "shared"
                ):
                    targets += [
                        t for t in shared_environment_targets(self.graph, ev.target) if t not in targets
                    ]
            applied = False
            for t in targets:
                if t in self.runtimes:
                    rec = apply_event(self.runtimes[t], ev.event_type, ev.parameters)
                    rec["event_id"] = None if ev.event_id is None else str(ev.event_id)
                    recs[t].events.append(rec)
                    applied = True
            if not applied:
                ignored.append(
                    {"event_type": ev.event_type.value, "reason": "target_has_no_degradation_state"}
                )
        return ignored

    def truth_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """S^true (N x D_T) and validity mask M (N x D_T), rows in ``order``."""
        rows = [self.runtimes[e].state.vector() for e in self.order]
        masks = [self.runtimes[e].state.mask() for e in self.order]
        n, d = len(rows), len(STATE_DIMENSIONS)
        return np.asarray(rows, dtype=np.float64).reshape(n, d), np.asarray(masks, dtype=bool).reshape(n, d)
