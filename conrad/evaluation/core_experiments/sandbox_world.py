"""Abstract synthetic hidden-state world (TRUTH PLANE, SYNTHETIC_ONLY). Never imported by conrad.core.

Hidden entities carry a position, an appearance vector and scalar properties evolving in physical
time. The generator emits incomplete / noisy / redundant / contradictory / OOD evidence, clutter
(no-match), temporal gaps, births, disappearances and near-coincident pairs (merge/split cases).
Truth is returned in a separate :class:`Supervision` channel keyed by evidence_id; hidden IDs never
appear inside Evidence objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from conrad.evaluation.core_experiments.evidence_factory import EvidenceFactory
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, EvidenceValidity
from conrad.schemas.provenance import ProvenanceRecord

PROPS = ("p0", "p1")


class SandboxConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_entities: int = Field(default=8, gt=0)
    area_m: float = 20.0
    steps: int = 30
    dt_range_s: tuple[float, float] = (5.0, 60.0)
    observe_prob: float = 0.6
    embedding_dim: int = 16
    appearance_noise: float = 0.3
    position_noise_m: float = 0.3
    measurement_noise: float = 0.05
    corrupt_prob: float = 0.1
    ood_prob: float = 0.05
    redundant_prob: float = 0.1
    outlier_prob: float = 0.05
    clutter_rate: float = 0.3
    birth_prob: float = 0.05
    death_prob: float = 0.01
    reappear_gap_steps: int = 0
    near_pair_fraction: float = 0.2
    change_prob: float = 0.03
    rate_scale: float = 0.002
    process_noise_per_s: float = 1e-5
    coupled_pairs: int = 0


@dataclass
class HiddenEntity:
    hidden_id: int
    position: np.ndarray
    appearance: np.ndarray
    props: dict[str, float]
    rates: dict[str, float]
    alive: bool = True
    hidden_until_step: int = -1


@dataclass
class Supervision:
    """Truth channel. ``source`` maps evidence_id -> hidden id (None = clutter)."""

    time_s: float
    states: dict[int, dict[str, float]] = field(default_factory=dict)
    positions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    source: dict[object, int | None] = field(default_factory=dict)
    flags: dict[object, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class SandboxStep:
    time_s: float
    evidence: list[tuple[Evidence, list[ProvenanceRecord]]]
    truth: Supervision


class SandboxWorld:
    def __init__(self, cfg: SandboxConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ids = IdFactory(seed)
        self.factory = EvidenceFactory(self.ids, cfg.embedding_dim, np.random.default_rng(seed + 1))
        self.entities: list[HiddenEntity] = []
        self.time_s = 0.0
        self.step_index = 0
        for _ in range(cfg.n_entities):
            self._spawn()
        n_near = int(cfg.near_pair_fraction * cfg.n_entities / 2)
        for i in range(n_near):  # near-coincident pairs: hard association / merge-split cases
            a, b = self.entities[2 * i], self.entities[2 * i + 1]
            b.position = a.position + self.rng.normal(scale=0.6, size=3)

    def _spawn(self) -> HiddenEntity:
        c = self.cfg
        ent = HiddenEntity(
            hidden_id=len(self.entities),
            position=self.rng.uniform(0, c.area_m, size=3),
            appearance=self.rng.normal(size=c.embedding_dim),
            props={p: float(self.rng.uniform(0.2, 0.8)) for p in PROPS},
            rates={p: float(self.rng.normal(scale=c.rate_scale)) for p in PROPS},
        )
        self.entities.append(ent)
        return ent

    def _evolve(self, dt: float) -> None:
        c = self.cfg
        for ent in self.entities:
            for p in PROPS:
                noise = self.rng.normal(scale=np.sqrt(c.process_noise_per_s * dt))
                ent.props[p] = float(np.clip(ent.props[p] + ent.rates[p] * dt + noise, 0.0, 1.0))
            if self.rng.random() < c.change_prob:  # abrupt change: credible contradiction scenario
                p = PROPS[int(self.rng.integers(len(PROPS)))]
                ent.props[p] = float(np.clip(ent.props[p] + self.rng.choice([-0.4, 0.4]), 0.0, 1.0))
        for i in range(c.coupled_pairs):  # hidden coupling: entity 2i+1 shares p1 with entity 2i
            self.entities[2 * i + 1].props["p1"] = self.entities[2 * i].props["p1"]
        if self.rng.random() < c.birth_prob:
            self._spawn()
        for ent in self.entities:
            if ent.alive and self.rng.random() < c.death_prob:
                ent.alive = False
            if c.reappear_gap_steps and ent.alive and self.rng.random() < 0.05:
                ent.hidden_until_step = self.step_index + c.reappear_gap_steps

    def _emit(
        self, ent: HiddenEntity, sup: Supervision, group: str | None = None
    ) -> list[tuple[Evidence, list[ProvenanceRecord]]]:
        c = self.cfg
        flags: list[str] = []
        reliability, noise, ood = 0.9, c.measurement_noise, 0.05
        if self.rng.random() < c.corrupt_prob:
            reliability, noise = 0.3, 0.3
            flags.append("corrupted")
        if self.rng.random() < c.ood_prob:
            ood = 0.9
            flags.append("ood")
        meas = {p: float(v + self.rng.normal(scale=noise)) for p, v in ent.props.items()}
        if self.rng.random() < c.outlier_prob:  # confidently wrong sensor: reliable-looking contradiction
            p = PROPS[0]
            meas[p] = float(1.0 - ent.props[p])
            flags.append("outlier")
        pos = ent.position + self.rng.normal(scale=c.position_noise_m, size=3)
        emb = ent.appearance + self.rng.normal(scale=c.appearance_noise, size=c.embedding_dim)
        if "ood" in flags:
            emb = emb + self.rng.normal(scale=3.0, size=c.embedding_dim)
        ev, rec = self.factory.make(
            self.time_s,
            meas,
            center_m=(float(pos[0]), float(pos[1]), float(pos[2])),
            reliability=reliability,
            aleatoric=noise**2,
            ood_score=ood,
            independence_group=group,
            validity=EvidenceValidity.DEGRADED if "corrupted" in flags else EvidenceValidity.VALID,
            embedding=emb,
            position_sigma_m=c.position_noise_m,
        )
        sup.source[ev.evidence_id] = ent.hidden_id
        sup.flags[ev.evidence_id] = tuple(flags)
        out = [(ev, [rec])]
        if self.rng.random() < c.redundant_prob:  # near-identical repeat from the SAME source
            g = group or f"obs-{ev.evidence_id}"
            ev = ev.model_copy(update={"independence_group": g})
            out[0] = (ev, [rec])
            dup, rec2 = self.factory.make(
                self.time_s,
                meas,
                center_m=ev.spatial_support.center_m if ev.spatial_support else None,
                reliability=reliability,
                aleatoric=noise**2,
                ood_score=ood,
                independence_group=g,
                embedding=np.asarray(ev.embedding),
                position_sigma_m=c.position_noise_m,
            )
            sup.source[dup.evidence_id] = ent.hidden_id
            sup.flags[dup.evidence_id] = (*flags, "redundant")
            out.append((dup, [rec2]))
        return out

    def step(self) -> SandboxStep:
        c = self.cfg
        dt = float(self.rng.uniform(*c.dt_range_s)) if self.step_index else 0.0
        self.time_s += dt
        if dt:
            self._evolve(dt)
        sup = Supervision(time_s=self.time_s)
        evidence: list[tuple[Evidence, list[ProvenanceRecord]]] = []
        for ent in self.entities:
            if not ent.alive:
                continue
            sup.states[ent.hidden_id] = dict(ent.props)
            sup.positions[ent.hidden_id] = (
                float(ent.position[0]),
                float(ent.position[1]),
                float(ent.position[2]),
            )
            if self.step_index < ent.hidden_until_step or self.rng.random() > c.observe_prob:
                continue
            evidence += self._emit(ent, sup)
        for _ in range(int(self.rng.poisson(c.clutter_rate))):
            pos = self.rng.uniform(0, c.area_m, size=3)
            ev, rec = self.factory.make(
                self.time_s,
                {p: float(self.rng.uniform()) for p in PROPS},
                center_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                reliability=0.5,
                aleatoric=0.05,
                embedding=self.rng.normal(size=c.embedding_dim),
                position_sigma_m=c.position_noise_m,
            )
            sup.source[ev.evidence_id] = None
            sup.flags[ev.evidence_id] = ("clutter",)
            evidence.append((ev, [rec]))
        self.step_index += 1
        return SandboxStep(self.time_s, evidence, sup)

    def coupled(self) -> list[tuple[int, int]]:
        return [(2 * i, 2 * i + 1) for i in range(self.cfg.coupled_pairs)]

    def episode(self) -> list[SandboxStep]:
        return [self.step() for _ in range(self.cfg.steps)]
