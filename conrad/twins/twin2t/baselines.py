"""H-T2T-01 baseline generators (spec ch11 Twin 2T baselines, ch31 H-T2T-01).

MCDE-based ladder (same code, switches only):
  ENGINEERING               mechanisms only, deterministic, no coupling
  ENGINEERING_STOCHASTIC    + stochastic residual
  ENGINEERING_TOPOLOGY      + topology (load-path, shared environment) coupling, deterministic
  INDEPENDENT_STOCHASTIC    independent-mechanism twin: stochastic, no mechanism or topology coupling
  MCDE_NO_COUPLING          stochastic + topology, no mechanism coupling
  FULL_MCDE                 everything on
  CONFIGURED                MCDE with the switches exactly as given in MCDEConfig
Non-mechanistic baselines (same state layout and masks):
  RANDOM_WALK               each valid dimension performs a bounded Gaussian random walk
  RANDOM_STATIC_DEFECTS     random defects drawn once and held constant

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from uuid import UUID

import numpy as np

from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.mcde import MCDE, ComponentStepRecord, MCDEStepResult, StructuralEvent
from conrad.twins.twin2t.mechanisms import SECONDS_PER_YEAR
from conrad.twins.twin2t.state import STATE_DIMENSIONS, ComponentRuntime
from conrad.twins.twin2t.topology import StructuralGraph


class GeneratorKind(str, Enum):
    CONFIGURED = "CONFIGURED"
    """Use ``MCDEConfig`` switches exactly as configured (ablation studies)."""
    ENGINEERING = "ENGINEERING"
    ENGINEERING_STOCHASTIC = "ENGINEERING_STOCHASTIC"
    ENGINEERING_TOPOLOGY = "ENGINEERING_TOPOLOGY"
    INDEPENDENT_STOCHASTIC = "INDEPENDENT_STOCHASTIC"
    MCDE_NO_COUPLING = "MCDE_NO_COUPLING"
    FULL_MCDE = "FULL_MCDE"
    RANDOM_WALK = "RANDOM_WALK"
    RANDOM_STATIC_DEFECTS = "RANDOM_STATIC_DEFECTS"


_LADDER: dict[GeneratorKind, dict[str, bool]] = {
    GeneratorKind.ENGINEERING: {"mechanism_coupling": False, "topology_coupling": False, "stochastic": False},
    GeneratorKind.ENGINEERING_STOCHASTIC: {
        "mechanism_coupling": False,
        "topology_coupling": False,
        "stochastic": True,
    },
    GeneratorKind.ENGINEERING_TOPOLOGY: {
        "mechanism_coupling": False,
        "topology_coupling": True,
        "stochastic": False,
    },
    GeneratorKind.INDEPENDENT_STOCHASTIC: {
        "mechanism_coupling": False,
        "topology_coupling": False,
        "stochastic": True,
    },
    GeneratorKind.MCDE_NO_COUPLING: {
        "mechanism_coupling": False,
        "topology_coupling": True,
        "stochastic": True,
    },
    GeneratorKind.FULL_MCDE: {"mechanism_coupling": True, "topology_coupling": True, "stochastic": True},
}

RANDOM_WALK_SIGMA_PER_SQRT_YR: dict[str, float] = {
    "corrosion_depth_m": 2.0e-4,
    "corrosion_area_fraction": 0.1,
    "coating_breakdown_fraction": 0.05,
    "crack_length_m": 2.0e-3,
    "crack_depth_m": 5.0e-4,
}
"""SYNTHETIC_ONLY scales for the random-walk baseline (no physical meaning by construction)."""


def mcde_config_for(kind: GeneratorKind, base: MCDEConfig | None = None) -> MCDEConfig:
    base = base or MCDEConfig()
    if kind not in _LADDER:
        return base
    sw = _LADDER[kind]
    return replace(
        base,
        mechanism_coupling=sw["mechanism_coupling"],
        topology_coupling=sw["topology_coupling"],
        stochastic=sw["stochastic"],
    )


def _upper(rt: ComponentRuntime, dim: str) -> float:
    return (
        rt.effective_wall_m
        if dim in ("corrosion_depth_m", "crack_depth_m")
        else (1.0 if dim.endswith("fraction") else 0.2)
    )


class RandomWalkEngine(MCDE):
    """Bounded random walk per valid dimension; events/interventions still apply (same event layer)."""

    def step(self, dt_s: float, events: Sequence[StructuralEvent] = ()) -> MCDEStepResult:
        if dt_s < 0:
            raise ValueError("dt_s must be >= 0")
        recs: dict[UUID, ComponentStepRecord] = {}
        root = math.sqrt(dt_s / SECONDS_PER_YEAR)
        for eid in self.order:
            rt, rng = self.runtimes[eid], self._rngs[eid]
            before = rt.state.vector()
            changes = {}
            for dim in STATE_DIMENSIONS:
                step = float(rng.normal(0.0, RANDOM_WALK_SIGMA_PER_SQRT_YR[dim] * root))
                if rt.state.validity.get(dim, False):
                    changes[dim] = float(np.clip(getattr(rt.state, dim) + step, 0.0, _upper(rt, dim)))
            rt.state = rt.state.with_(**changes)
            recs[eid] = ComponentStepRecord(eid, before, before, rt.state.mask())
        ignored = self._apply_events(events, recs)
        self.time_s += dt_s
        for eid in self.order:
            recs[eid].after = self.runtimes[eid].state.vector()
        return MCDEStepResult(self.time_s, dt_s, recs, ignored)


class RandomStaticDefectEngine(MCDE):
    """Random defects drawn once at construction and held constant (only events change them)."""

    def __init__(
        self, graph: StructuralGraph, runtimes: dict[UUID, ComponentRuntime], config: MCDEConfig, seed: int
    ) -> None:
        super().__init__(graph, runtimes, config, seed)
        for eid in self.order:
            rt, rng = self.runtimes[eid], self._rngs[eid]
            draws = {d: float(rng.uniform(0.0, 0.5 * _upper(rt, d))) for d in STATE_DIMENSIONS}
            rt.state = rt.state.with_(**{d: v for d, v in draws.items() if rt.state.validity.get(d, False)})
            rt.initial_state = rt.state

    def step(self, dt_s: float, events: Sequence[StructuralEvent] = ()) -> MCDEStepResult:
        if dt_s < 0:
            raise ValueError("dt_s must be >= 0")
        recs = {
            eid: ComponentStepRecord(eid, rt.state.vector(), rt.state.vector(), rt.state.mask())
            for eid, rt in ((e, self.runtimes[e]) for e in self.order)
        }
        ignored = self._apply_events(events, recs)
        self.time_s += dt_s
        for eid in self.order:
            recs[eid].after = self.runtimes[eid].state.vector()
        return MCDEStepResult(self.time_s, dt_s, recs, ignored)


def make_engine(
    kind: GeneratorKind,
    graph: StructuralGraph,
    runtimes: dict[UUID, ComponentRuntime],
    config: MCDEConfig,
    seed: int,
) -> MCDE:
    if kind is GeneratorKind.RANDOM_WALK:
        return RandomWalkEngine(graph, runtimes, config, seed)
    if kind is GeneratorKind.RANDOM_STATIC_DEFECTS:
        return RandomStaticDefectEngine(graph, runtimes, config, seed)
    return MCDE(graph, runtimes, mcde_config_for(kind, config), seed)
