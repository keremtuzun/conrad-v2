"""Runs Model2Core over a sandbox episode and records what the belief plane did (truth kept aside).

Truth is used here ONLY for scoring and, when enabled, to play the role of a child that supplies
known topology (relationships between already-existing beliefs) as mission context.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from conrad.core.config import CoreConfig
from conrad.core.pipeline import Model2Core
from conrad.core.state import AnalyticBeliefState
from conrad.core.tbd_analytic import PropertyDynamics
from conrad.evaluation.core_experiments.common import temp_repository
from conrad.evaluation.core_experiments.evidence_factory import CLOCK
from conrad.evaluation.core_experiments.sandbox_world import PROPS, SandboxStep
from conrad.schemas.belief import Relationship
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain


@dataclass
class RunLog:
    assignment: dict[UUID, UUID] = field(default_factory=dict)  # evidence_id -> belief_id
    evidence_step: dict[UUID, int] = field(default_factory=dict)
    source: dict[UUID, int | None] = field(default_factory=dict)
    per_step_estimates: list[dict[UUID, AnalyticBeliefState]] = field(default_factory=list)
    duplicates_rejected: int = 0
    duplicates_sent: int = 0
    relationships_added: int = 0
    merges: list[tuple[UUID, UUID, UUID]] = field(default_factory=list)
    lineage: dict[UUID, UUID] = field(default_factory=dict)  # merged parent -> child
    relations: list[Relationship] = field(default_factory=list)


def majority(log: RunLog) -> dict[UUID, tuple[int | None, float]]:
    """belief -> (majority hidden source, purity). Clutter counts as source None."""
    by_belief: dict[UUID, Counter[int | None]] = {}
    for eid, bid in log.assignment.items():
        by_belief.setdefault(root(log, bid), Counter())[log.source[eid]] += 1
    out: dict[UUID, tuple[int | None, float]] = {}
    for bid, cnt in by_belief.items():
        src, n = cnt.most_common(1)[0]
        out[bid] = (src, n / sum(cnt.values()))
    return out


def root(log: RunLog, bid: UUID) -> UUID:
    while bid in log.lineage:
        bid = log.lineage[bid]
    return bid


def run_pipeline(
    cfg: CoreConfig,
    steps: list[SandboxStep],
    tmp: str,
    seed: int,
    *,
    process_noise_per_s: float,
    use_rbp: bool = True,
    coupled: list[tuple[int, int]] | None = None,
    redeliver_prob: float = 0.0,
    reset_every: int = 0,
    merge_distance_m: float = 0.0,
) -> tuple[Model2Core, RunLog]:
    repo = temp_repository(tmp)
    ids = IdFactory(seed + 99)
    dyn = {p: PropertyDynamics(0.0, process_noise_per_s) for p in PROPS}
    run_id = ids.new()
    core = Model2Core(cfg, repo, run_id, ids, Domain.SPATIAL, "object", dynamics=dyn, use_rbp=use_rbp)
    log = RunLog()
    previous: list[Any] = []
    for k, step in enumerate(steps):
        items = [(ev, recs) for ev, recs in step.evidence]
        replay = [p for p in previous if (k * 7919 + p[0].evidence_id.int) % 1000 < redeliver_prob * 1000]
        log.duplicates_sent += len(replay)
        now = stamp(step.time_s, CLOCK)
        for ev, _ in items:
            log.source[ev.evidence_id] = step.truth.source[ev.evidence_id]
            log.evidence_step[ev.evidence_id] = k
        res = core.forward_step([*items], now, relationships=_relations(core, log, coupled, ids))
        if replay:
            dup_res = core.forward_step(replay, now, sweep=False)
            log.duplicates_rejected += len(dup_res.duplicates)
        created = iter(res.created)
        for d in res.decisions:
            log.assignment[d.evidence_id] = d.belief_id if d.belief_id is not None else next(created)
        if merge_distance_m > 0:
            _merge_pass(core, log, merge_distance_m, now)
        log.per_step_estimates.append(
            {
                c.belief_id: core.pmbl.node(c.belief_id).predicted or core.pmbl.node(c.belief_id).corrected
                for c in core.beliefs()
            }
        )
        if reset_every and (k + 1) % reset_every == 0:
            core.reset_working_memory()
            core.add_relationships(log.relations)
        previous = items
    return core, log


def _relations(
    core: Model2Core, log: RunLog, coupled: list[tuple[int, int]] | None, ids: IdFactory
) -> list[Relationship]:
    """Child-supplied topology: once both coupled entities have beliefs, declare edges both ways."""
    if not coupled:
        return []
    maj = majority(log)
    live = {c.belief_id for c in core.beliefs()}
    by_src: dict[int, UUID] = {}
    for bid, (src, _) in maj.items():
        if src is not None and bid in live:
            by_src.setdefault(src, bid)
    known = {(r.source_belief_id, r.target_belief_id) for r in core.pmbl.graph.relationships.values()}
    out = []
    for a, b in coupled:
        if a in by_src and b in by_src and by_src[a] != by_src[b]:
            for s, t in ((by_src[a], by_src[b]), (by_src[b], by_src[a])):
                if (s, t) in known:
                    continue
                rel = Relationship(
                    relationship_id=ids.new(),
                    relation_type="ATTACHED",
                    source_belief_id=s,
                    target_belief_id=t,
                    source_domain=Domain.SPATIAL,
                    target_domain=Domain.SPATIAL,
                    confidence=0.9,
                    uncertainty=unknown_uncertainty(),
                    provenance_id=ids.new(),
                    attributes={"relation_variance": 1e-4},
                )
                out.append(rel)
                log.relations.append(rel)
    log.relationships_added += len(out)
    return out


def _merge_pass(core: Model2Core, log: RunLog, dist_m: float, now: Any) -> None:
    """Evaluation-side merge policy (PMBL merge rules are child-specific): close + consistent beliefs."""
    cells = [c for c in core.beliefs() if c.spatial_support is not None]
    done: set[UUID] = set()
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if a.belief_id in done or b.belief_id in done:
                continue
            assert a.spatial_support is not None and b.spatial_support is not None
            if math.dist(a.spatial_support.center_m, b.spatial_support.center_m) > dist_m:
                continue
            ea, eb = (
                core.pmbl.node(a.belief_id).corrected.estimates,
                core.pmbl.node(b.belief_id).corrected.estimates,
            )
            consistent = all(
                abs(ea[p].mean - eb[p].mean) <= 3 * math.sqrt(ea[p].variance + eb[p].variance)
                for p in PROPS
                if p in ea and p in eb
            )
            if not consistent:
                continue
            child = core.pmbl.merge(a.belief_id, b.belief_id, now)
            log.merges.append((a.belief_id, b.belief_id, child.belief_id))
            log.lineage[a.belief_id] = log.lineage[b.belief_id] = child.belief_id
            done |= {a.belief_id, b.belief_id}
