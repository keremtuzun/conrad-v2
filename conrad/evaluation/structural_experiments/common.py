"""Shared harness for the 2T experiments. TRUTH PLANE ALLOWED (evaluation only).

Truth flows only into metrics. The belief side receives (a) the asset registry built from DESIGN fields of
the scenario (type, material, coating, nominal wall, relationships), and (b) Evidence built from the twin's
sensor-shaped Observations. Association of an observation to its registry component is taken from the
twin's supervision label: an evaluation-only ASSOCIATION ORACLE (association is not under test here).
"""

from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch
import yaml

from conrad.domains.technical import CORROSION_DEPTH, CRACK_LENGTH, AssetRegistry, structured_evidence
from conrad.domains.technical.baselines import StructuralEstimator
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.timebase import stamp
from conrad.schemas.world import Scenario, SensorSpec
from conrad.twins.base import SensingContext
from conrad.twins.twin2t import SensorDegradationCurriculum, Twin2T

DAY = 86400.0
YEAR = 365.25 * DAY
QUANTITIES = (CORROSION_DEPTH, CRACK_LENGTH)
TRUTH_COLUMN = {CORROSION_DEPTH: 0, CRACK_LENGTH: 3}
DESIGN_KEYS = ("component_type", "material", "coating", "wall_thickness_m")
CLOCK = "sim"


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def registry_context(scenario: Scenario) -> dict[str, Any]:
    """Asset registry from DESIGN data only; sampled initial degradation / mechanism parameters excluded."""
    struct = scenario.structural_state.get("entities", {})
    comps = []
    for e in scenario.world_entities:
        if not e.domain_ownership.technical:
            continue
        entry = struct.get(str(e.id), {})
        item: dict[str, Any] = {"registry_id": e.id, "parent_id": e.parent_id}
        item["component_type"] = entry.get("component_type", e.entity_type)
        for k in DESIGN_KEYS[1:]:
            if k in entry:
                item[k] = entry[k]
        comps.append(item)
    rels = [dict(r) for r in scenario.structural_state.get("relationships", [])]
    return {"asset_registry": {"components": comps, "relationships": rels}, "timestamp": stamp(0.0, CLOCK)}


@dataclass
class World:
    seed: int
    scenario: Scenario
    twin: Twin2T
    registry: AssetRegistry
    context: dict[str, Any]
    ev_ids: IdFactory
    rng: np.random.Generator
    sensor: SensorSpec = field(init=False)

    def __post_init__(self) -> None:
        f = IdFactory(seed=self.seed).child("sensing")
        pose = Pose(frame_id="WORLD", position_m=(0.0, 0.0, -10.0))
        self.sensor = SensorSpec(
            sensor_id=f.new(),
            modality="STRUCTURED",
            frame_id="sensor",
            mount_pose=pose,
            rate_hz=1.0,
            parameters={"t2t_fidelity": "T0"},
        )
        self._ctx_ids = (f.new(), f.new(), f.new())

    @property
    def component_ids(self) -> tuple[UUID, ...]:
        return self.twin.component_ids

    def observe(self, visible: Sequence[UUID], degradation_level: float = 0.0) -> list[Evidence]:
        if not visible:
            return []
        deg: dict[str, float] = {f"visibility:{e}": 1.0 for e in visible}
        if degradation_level > 0.0:
            extra = SensorDegradationCurriculum(missing_modality_prob=0.0).degradation(
                degradation_level, self.rng
            )
            deg.update(
                {k: v for k, v in extra.items() if k not in ("pose_error_m", "temporal_gap_s", "blur")}
            )
        pose = Pose(frame_id="WORLD", position_m=(0.0, 0.0, -10.0))
        ctx = SensingContext(
            mission_id=self._ctx_ids[0],
            run_id=self._ctx_ids[1],
            trace_id=self._ctx_ids[2],
            sensor=self.sensor,
            true_pose=pose,
            estimated_pose=None,
            timestamp=stamp(self.twin.time_s, CLOCK),
            degradation=deg,
        )
        out = []
        for s in self.twin.generate_observation(ctx):
            assert s.supervision is not None
            ev, _ = structured_evidence(s.observation, self.ev_ids, s.supervision.true_world_entity_id)
            out.append(ev)
        return out

    def truth(self) -> dict[UUID, dict[str, float | None]]:
        arr, mask = self.twin.truth_arrays()
        return {
            eid: {q: float(arr[i, c]) if mask[i, c] else None for q, c in TRUTH_COLUMN.items()}
            for i, eid in enumerate(self.twin.component_ids)
        }


def make_world(seed: int, scenario: Scenario, store_dir: Path) -> World:
    twin = Twin2T(IdFactory(seed).child("twin2t"), ObjectStore(store_dir / f"obj_{seed}"))
    twin.initialize(scenario)
    ctx = registry_context(scenario)
    return World(
        seed=seed,
        scenario=scenario,
        twin=twin,
        registry=AssetRegistry.from_context(ctx),
        context=ctx,
        ev_ids=IdFactory(seed).child("evidence"),
        rng=np.random.default_rng([seed, 17]),
    )


Estimates = dict[str, dict[UUID, dict[str, tuple[float, float] | None]]]


def experiment_id(config: Mapping[str, Any], default: str) -> str:
    return str(dict(config.get("experiment", {})).get("id", default))


def _local_seeds(spec: Any) -> set[int]:
    if isinstance(spec, Mapping):
        lo, hi = spec.get("range", [0, 0])
        return {int(s) for s in spec.get("explicit", [])} | set(range(int(lo), int(hi)))
    return {int(s) for s in spec}


def _checked_local(config: Mapping[str, Any], seeds: list[int], part: str, purpose: str) -> list[int]:
    """Config-local 2T partitions (iteration 3): ``partitions: {name: [seeds] | {range, explicit}}``.

    ``configs/eval/partitions.yaml`` is digest-pinned, so a fresh held-out split for 2T lives in the configs.
    A partition named ``final*`` is read with FINAL_TEST access; ``spent*`` lists are never readable. Every
    non-spent local partition must be disjoint from the others and from every pinned mission/abstract seed,
    except that development/validation may reuse the pinned mission development/validation seeds."""
    from conrad.evaluation import partitions

    local = {str(k): _local_seeds(v) for k, v in dict(config["partitions"]).items()}
    if part not in local or part.startswith("spent"):
        raise partitions.PartitionAccessError(f"partition {part!r} is not a readable local 2T partition")
    names = [k for k in local if not k.startswith("spent")]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if local[a] & local[b]:
                raise partitions.PartitionAccessError(f"local partitions {a} and {b} overlap")
        spent = set().union(*(v for k, v in local.items() if k.startswith("spent")))
        if a.startswith("final") and local[a] & spent:
            raise partitions.PartitionAccessError(f"local partition {a} reuses spent seeds")
    access = partitions.Partition.FINAL_TEST if part.startswith("final") else partitions.Partition(part)
    partitions.check_access(access, purpose)
    domain = str(config.get("partition_domain", "mission"))
    for s in local[part]:
        pinned = partitions.partition_of(domain, s) or partitions.partition_of("abstract", s)
        if access is partitions.Partition.FINAL_TEST and pinned is not None:
            raise partitions.PartitionAccessError(f"final seed {s} is already in the pinned {pinned} split")
        if access is not partitions.Partition.FINAL_TEST and pinned not in (None, access):
            raise partitions.PartitionAccessError(f"seed {s} is pinned as {pinned}, not {access}")
    wrong = [s for s in seeds if s not in local[part]]
    if wrong:
        raise partitions.PartitionAccessError(f"seeds {wrong} are not in local partition {part}")
    return seeds


def checked_seeds(config: Mapping[str, Any], seeds: Sequence[int]) -> list[int]:
    """Enforce the declared partition: every seed must belong to it and the purpose must allow reading it.

    Configs without ``partition`` are legacy (2026201-3 are DEVELOPMENT seeds, not held-out). Configs with a
    ``partitions`` mapping use config-local 2T partitions (``_checked_local``)."""
    from conrad.evaluation import partitions

    seeds = list(seeds)
    part = config.get("partition")
    if part is None:
        return seeds
    if config.get("partitions"):
        return _checked_local(config, seeds, str(part), str(config.get("purpose", "design")))
    purpose = str(config.get("purpose", "design"))
    partitions.check_access(part, purpose)
    domain = str(config.get("partition_domain", "mission"))
    wrong = [s for s in seeds if partitions.partition_of(domain, s) != partitions.Partition(part)]
    if wrong:
        raise partitions.PartitionAccessError(f"seeds {wrong} are not in {domain}:{part}")
    return seeds


def paired_bootstrap(
    diffs: Sequence[float | None], n_boot: int = 10000, seed: int = 0, level: float = 0.95
) -> dict[str, float] | None:
    """Mean paired difference with a percentile bootstrap CI over the paired units (seeds)."""
    x = np.asarray([d for d in diffs if d is not None and math.isfinite(d)], dtype=np.float64)
    if x.size < 2:
        return None
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    a = (1.0 - level) / 2.0
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(boots, a)),
        "ci_high": float(np.quantile(boots, 1.0 - a)),
        "n": float(x.size),
    }


def run_episode(
    world: World,
    estimators: Sequence[StructuralEstimator],
    *,
    steps: int,
    dt_s: float,
    visible_at: Callable[[int], Sequence[UUID]],
    degradation_level: float = 0.0,
    probe: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    """Advance truth, observe, update every estimator; returns one record per step."""
    t0 = stamp(0.0, CLOCK)
    for estimator in estimators:
        estimator.reset(world.registry, t0)
    records = []
    for k in range(1, steps + 1):
        world.twin.advance(dt_s)
        now = stamp(world.twin.time_s, CLOCK)
        visible = list(visible_at(k))
        evs = world.observe(visible, degradation_level)
        seen = {c.registry_entity_id for e in evs for c in e.entity_candidates}
        est: Estimates = {}
        latent: Estimates = {}
        for m in estimators:
            m.step(now, evs)
            est[m.name] = {rid: {q: m.estimate(rid, q) for q in QUANTITIES} for rid in world.component_ids}
            lat = getattr(m, "latent", None)
            if lat is not None:
                latent[m.name] = {rid: {q: lat(rid, q) for q in QUANTITIES} for rid in world.component_ids}
        records.append(
            {
                "step": k,
                "time_s": world.twin.time_s,
                "observed": seen,
                "truth": world.truth(),
                "est": est,
                "latent": latent,
                "probe": None if probe is None else probe(),
            }
        )
    return records


def gaussian_scores(mean_m: float, var_m2: float, truth_m: float) -> tuple[float, float, float]:
    """(abs error mm, NLL in mm units, inside 95% interval)."""
    m, v, x = mean_m * 1e3, max(var_m2 * 1e6, 1e-8), truth_m * 1e3
    nll = 0.5 * math.log(2 * math.pi * v) + (x - m) ** 2 / (2 * v)
    return abs(x - m), nll, float(abs(x - m) <= 1.96 * math.sqrt(v))


def summarize(values: Mapping[str, list[float]]) -> dict[str, float | None]:
    """Mean per key, plus the median for absolute-error keys (crack errors are heavy-tailed)."""
    out: dict[str, float | None] = {k: (float(np.mean(v)) if v else None) for k, v in values.items()}
    for k, v in values.items():
        if k.endswith("mae_mm") and v:
            out[k.replace("mae_mm", "median_ae_mm")] = float(np.median(v))
    return out


def aggregate(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    flat: dict[int, dict[str, float]] = {}

    def walk(prefix: str, value: Any, out: dict[str, float]) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v, out)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[prefix] = float(value)

    for s, m in per_seed.items():
        flat[s] = {}
        walk("", m, flat[s])
    keys = sorted({k for f in flat.values() for k in f})
    res: dict[str, dict[str, float | None]] = {}
    for key in keys:
        vals = np.asarray([f[key] for f in flat.values() if key in f])
        res[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "n": float(vals.size),
        }
    return res


def write_result(
    experiment_id: str,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    per_seed: Mapping[int, Any],
    out_dir: str | Path,
    started: float,
    verdicts: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "experiment_id": experiment_id,
        "data": "SYNTHETIC_ONLY: Twin2T (MCDE) scenarios, T0 abstract observations",
        "seeds": list(seeds),
        "config": dict(config),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "summary": aggregate(per_seed),
        "verdicts": dict(verdicts),
        "wall_time_s": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "claim_note": "Numbers are what this run produced on synthetic data; no real-world claim.",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{experiment_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
