"""Counterfactual world pairs W_A != W_B with O(W_A) ~= O(W_B) for a given observation set (T2S-CF-01).

The key fixture for "Model2S must not confidently invent hidden geometry". Every pair is VERIFIED by
rendering all scheduled observations in both worlds with identical noise streams and reporting the
maximum observation difference; a pair that fails verification is never returned.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Scenario, WorldEntity
from conrad.twins.twin2s.ocpwe import ObservationSchedule, ScheduledView, run_schedule
from conrad.twins.twin2s.sdf import Box, Capsule, Ellipsoid, PolyCapsule, Primitive, _t3
from conrad.twins.twin2s.terrain import HeightfieldTerrain
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2s.world import SpatialEntity

_ID_SALT = 0x5EEDCF
KINDS = ("bend_tail", "remove_tail", "hidden_object")


@dataclass(frozen=True)
class DistinguishabilityReport:
    max_abs_difference: dict[str, float]  # per modality, over measured payloads / inline values
    n_observations: int
    identical_digests: int

    @property
    def max_difference(self) -> float:
        return max(self.max_abs_difference.values(), default=0.0)


@dataclass(frozen=True)
class CounterfactualPair:
    kind: str
    world_a: Scenario
    world_b: Scenario
    changed_entity_ids: tuple[UUID, ...]
    changed_region_min_m: tuple[float, float, float]
    changed_region_max_m: tuple[float, float, float]
    schedule: ObservationSchedule
    report: DistinguishabilityReport


def _twin(scenario: Scenario, store: ObjectStore, noise_seed: int) -> Twin2S:
    tw = Twin2S(IdFactory(noise_seed ^ _ID_SALT), store)  # never reuse the scenario IdFactory stream
    tw.initialize(scenario)
    tw.reset(noise_seed)
    return tw


def verify_indistinguishable(
    world_a: Scenario,
    world_b: Scenario,
    schedule: ObservationSchedule,
    noise_seed: int = 0,
    store: ObjectStore | None = None,
) -> DistinguishabilityReport:
    """Render the schedule in both worlds with the same noise streams; return max |O_A - O_B| per modality.

    NaN (no return / dropout) must match NaN exactly; a NaN mismatch counts as an infinite difference.
    """
    store = store or ObjectStore(tempfile.mkdtemp(prefix="twin2s-cf-"))
    ids = IdFactory(noise_seed)
    mission, run, trace = ids.new(), ids.new(), ids.new()
    sa = run_schedule(_twin(world_a, store, noise_seed), schedule, mission, run, trace)
    sb = run_schedule(_twin(world_b, store, noise_seed), schedule, mission, run, trace)
    if len(sa) != len(sb):
        return DistinguishabilityReport({"count": math.inf}, len(sa), 0)
    diff: dict[str, float] = {}
    same = 0
    for a, b in zip(sa, sb, strict=True):
        oa, ob = a.observation, b.observation
        key = oa.modality.value
        if oa.payload_ref is not None and ob.payload_ref is not None:
            same += int(oa.payload_ref.digest == ob.payload_ref.digest)
            xa = store.get_array(oa.payload_ref).astype(np.float64)
            xb = store.get_array(ob.payload_ref).astype(np.float64)
            if xa.shape != xb.shape or np.any(np.isnan(xa) != np.isnan(xb)):
                d = math.inf
            else:
                ok = ~np.isnan(xa)
                d = float(np.max(np.abs(xa[ok] - xb[ok]), initial=0.0))
        else:
            d = float(
                np.max(np.abs(np.subtract(oa.inline_values or (), ob.inline_values or ())), initial=0.0)
            )
        diff[key] = max(diff.get(key, 0.0), d)
    return DistinguishabilityReport(diff, len(sa), same)


def _replace_geometry(
    scenario: Scenario,
    updates: dict[UUID, SpatialEntity | None],
    added: list[tuple[WorldEntity, SpatialEntity]],
) -> Scenario:
    """New scenario sharing all identities; only the listed entities' spatial geometry differs."""
    ents = dict(scenario.spatial_state["entities"])
    world_entities = list(scenario.world_entities)
    for eid, ent in updates.items():
        if ent is None:
            ents[str(eid)] = {**ents[str(eid)], "active": False}
        else:
            ents[str(eid)] = ent.to_dict()
            world_entities = [
                w.model_copy(
                    update={
                        "entity_type": ent.semantic_class,
                        "geometry_ref": f"twin2s:primitive:{ent.primitive.kind}",
                    }
                )
                if w.id == eid
                else w
                for w in world_entities
            ]
    for we, se in added:
        world_entities.append(we)
        ents[str(we.id)] = se.to_dict()
    spatial = {**scenario.spatial_state, "entities": ents}
    meta = {**scenario.metadata, "counterfactual_of": str(scenario.scenario_id)}
    return scenario.model_copy(
        update={"world_entities": tuple(world_entities), "spatial_state": spatial, "metadata": meta}
    )


def _tail(tw: Twin2S) -> SpatialEntity:
    groups = tw.scenario.spatial_state.get("groups", {})
    order = [UUID(u) for u in groups.get("segments", []) + groups.get("bends", [])]
    pieces = [e for e in tw.world.entities if e.entity_id in order]
    if not pieces:
        raise ValueError("scenario has no pipeline to modify")
    # the tail is the piece whose far end is farthest from the pipeline start
    start = np.asarray(_endpoints(pieces[0].primitive)[0])
    return max(pieces, key=lambda e: float(np.linalg.norm(np.asarray(_endpoints(e.primitive)[1]) - start)))


def _endpoints(p: Primitive) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if isinstance(p, Capsule):
        return p.a, p.b
    if isinstance(p, PolyCapsule):
        return p.points[0], p.points[-1]
    raise ValueError(f"not a pipeline piece: {p.kind}")


def _variant(
    kind: str, tw: Twin2S, ids: IdFactory, rng: np.random.Generator
) -> tuple[Scenario, tuple[UUID, ...], list[Primitive]]:
    if kind in ("bend_tail", "remove_tail"):
        tail = _tail(tw)
        a, b = (np.asarray(x) for x in _endpoints(tail.primitive))
        r = float(tail.primitive.radius)  # type: ignore[attr-defined]
        if kind == "remove_tail":
            return (
                _replace_geometry(tw.scenario, {tail.entity_id: None}, []),
                (tail.entity_id,),
                [tail.primitive],
            )
        d = b - a
        length, hdg = float(np.linalg.norm(d[:2])), math.atan2(d[1], d[0])
        turn = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.2))
        step = length / 8.0  # a true circular arc of the same length, tangent to the original heading
        heads = hdg + turn * (np.arange(8) + 0.5) / 8.0
        steps = step * np.stack([np.cos(heads), np.sin(heads), np.zeros(8)], axis=1)
        pts = [a, *(a + np.cumsum(steps, axis=0))]
        prim = PolyCapsule(tuple(_t3(q) for q in pts), r)
        new = SpatialEntity(tail.entity_id, "pipeline_bend", prim, tail.material_id)
        return (
            _replace_geometry(tw.scenario, {tail.entity_id: new}, []),
            (tail.entity_id,),
            [tail.primitive, prim],
        )
    if kind == "hidden_object":
        lo, hi = np.asarray(tw.world.bounds_min), np.asarray(tw.world.bounds_max)
        xy = rng.uniform(lo[:2] + 1.0, hi[:2] - 1.0)
        floor = next(e.primitive for e in tw.world.entities if isinstance(e.primitive, HeightfieldTerrain))
        ground = float(floor.height(xy[None, :])[0])
        he = rng.uniform(0.2, 0.5, 3)
        prim_b: Primitive = (
            Box((float(xy[0]), float(xy[1]), ground + float(he[2])), _t3(he))
            if rng.random() < 0.5
            else Ellipsoid((float(xy[0]), float(xy[1]), ground + 0.3 * float(he[2])), _t3(he))
        )
        we = WorldEntity(
            id=ids.new(),
            entity_type="debris",
            reference_frame="WORLD",
            created_at=tw.scenario.world_entities[0].created_at,
            geometry_ref=f"twin2s:primitive:{prim_b.kind}",
        )
        return (
            _replace_geometry(
                tw.scenario, {}, [(we, SpatialEntity(we.id, "debris", prim_b, "debris_generic"))]
            ),
            (we.id,),
            [prim_b],
        )
    raise ValueError(f"unknown counterfactual kind {kind!r}")


def make_counterfactual_pair(
    world_a: Scenario,
    schedule: ObservationSchedule,
    ids: IdFactory,
    rng: np.random.Generator,
    kind: str = "bend_tail",
    tolerance: float = 1e-3,
    max_attempts: int = 12,
    noise_seed: int = 0,
) -> CounterfactualPair:
    """Search for W_B differing from W_A only where ``schedule`` never looks; verify and return, else raise."""
    tw = _twin(world_a, ObjectStore(tempfile.mkdtemp(prefix="twin2s-cf-")), noise_seed)
    for _ in range(max_attempts):
        world_b, changed, prims = _variant(kind, tw, ids, rng)
        rep = verify_indistinguishable(world_a, world_b, schedule, noise_seed)
        if rep.max_difference <= tolerance:
            lo = np.min([p.bounds()[0] for p in prims], axis=0)
            hi = np.max([p.bounds()[1] for p in prims], axis=0)
            return CounterfactualPair(kind, world_a, world_b, changed, _t3(lo), _t3(hi), schedule, rep)
    raise RuntimeError(f"no {kind} counterfactual hidden from this schedule after {max_attempts} attempts")


def discriminating_views(
    pair: CounterfactualPair, candidates: list[ScheduledView], noise_seed: int = 0, threshold: float = 0.05
) -> list[tuple[ScheduledView, float]]:
    """Views a* whose observations separate W_A from W_B (difference above ``threshold``), best first."""
    out = []
    for v in candidates:
        single = ObservationSchedule((v,), pair.schedule.conditions, {})
        d = verify_indistinguishable(pair.world_a, pair.world_b, single, noise_seed).max_difference
        if d > threshold:
            out.append((v, d))
    return sorted(out, key=lambda x: -x[1])
