"""Structural (2T) lineage: what reaches Model2T is sensor-shaped only (docs/audits/STRUCTURAL_LINEAGE_AUDIT.md).

1. The serialized STRUCTURED Observations and their Evidence in a real FLAGSHIP-I4 run carry no world-entity
   UUID, no hidden Twin2T value (exactly or within a tiny tolerance) and no truth-derived sensor_context key.
2. Model2T's output is a function of the captured sensor payloads only: re-running the deployment-side
   pipeline (structured_evidence -> StructuralAssociator -> Model2T) on the same captured Observations while
   the truth world's hidden state and registry mapping are randomized gives bit-identical beliefs. A positive
   control (changing the payloads) shows the comparison has power.
3. At the Twin2T level, hidden fields a T0 sensor cannot see (crack depth, wall thickness) do not change the
   emitted payload; the sensed crack length does.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conrad.domains.technical import Model2T, PropagationMode, model2t_config_from_dict, structured_evidence
from conrad.orchestration.association import StructuralAssociator, registry_of
from conrad.orchestration.mission_config import runtime_config
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, Observation
from conrad.schemas.timebase import TimeStamp, stamp
from conrad.schemas.world import SensorSpec
from conrad.settings import load_settings
from conrad.sim.mission.options import world_options
from conrad.sim.mission.scenarios import resolve
from conrad.sim.mission.world import MissionWorld
from conrad.twins.base import SensingContext
from conrad.twins.twin2t import Twin2T, build_small_scenario
from conrad.twins.twin2t.state import STATE_DIMENSIONS
from tests.acceptance._runs import SMALL, flagship

SENSOR_CONTEXT_ALLOWED = {
    "twin2t_fidelity",
    "measurements",
    "units",
    "measured_range_m",
    "measured_bearing_rad",
    "measured_elevation_rad",
    "range_sigma_m",
    "angle_sigma_rad",
}
REL_TOL, ABS_TOL = 1e-9, 1e-12


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def run():
    out = flagship()
    d = Path(out["run_dir"])
    truth = json.loads((d / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    con = sqlite3.connect(str(d / "conrad.sqlite"))
    try:
        obs = [
            Observation.model_validate_json(p)
            for (p,) in con.execute("SELECT payload_json FROM observations")
        ]
        evs = [Evidence.model_validate_json(p) for (p,) in con.execute("SELECT payload_json FROM evidence")]
    finally:
        con.close()
    structural = [o for o in obs if "twin2t_fidelity" in o.sensor_context]
    ids = {o.observation_id for o in structural}
    return {
        "dir": d,
        "truth": truth,
        "structural": sorted(structural, key=lambda o: (o.timestamp.time_ns, o.observation_id.int)),
        "evidence": [e for e in evs if e.source_observation_id in ids],
    }


@pytest.fixture
def world(tmp_path):
    """The truth world of the flagship run, rebuilt deterministically (seed + scenario options)."""
    settings = load_settings(SMALL)
    world_raw, runtime_raw = resolve("FLAGSHIP-I4", dict(settings.sim.get("mission", {})))
    run_uuid = IdFactory(settings.run.seed).child("run").new()
    w = MissionWorld.build(
        settings.run.seed, "FLAGSHIP-I4", world_options(world_raw), run_uuid, tmp_path / "obj"
    )
    return w, runtime_config(runtime_raw)


def _hidden_values(run, world_obj) -> list[float]:
    vals: list[float] = []
    for state in run["truth"]["target_states"].values():
        vals += [float(v) for k, v in state.items() if k != "t_s" and isinstance(v, (int, float))]
    vals += [float(v) for v in run["truth"]["meta"]["defect"].values() if isinstance(v, (int, float))]
    arr, mask = world_obj.t2t.truth_arrays()
    vals += [float(x) for x in arr[mask]]
    return [v for v in vals if v != 0.0]


def _numbers(obj, out):
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out)
    return out


# ---------------------------------------------------------------------------------------------- 1. payloads
def test_structural_payloads_carry_no_world_id_no_hidden_value_no_truth_key(run, world):
    w, _ = world
    meta = run["truth"]["meta"]
    world_ids = (
        set(meta["world_entity_ids"]) | set(meta["registry_to_world"].values()) | {meta["scenario_uuid"]}
    )
    assert str(w.target) == meta["target_world_id"]  # the rebuilt world is the run's world
    hidden = np.asarray(_hidden_values(run, w))
    assert len(run["structural"]) > 10 and len(run["evidence"]) == len(run["structural"])
    for o in run["structural"]:
        assert set(o.sensor_context) <= SENSOR_CONTEXT_ALLOWED, set(o.sensor_context) - SENSOR_CONTEXT_ALLOWED
        for v in o.inline_values or ():
            assert not np.any(np.isclose(hidden, v, rtol=REL_TOL, atol=ABS_TOL)), (
                f"hidden value {v} in payload"
            )
    for item in [*run["structural"], *run["evidence"]]:
        blob = item.model_dump_json()
        assert not any(wid in blob for wid in world_ids)
        assert "visibility" not in blob and "component_type" not in blob and "supervision" not in blob
        for v in _numbers(json.loads(blob), []):
            if v != 0.0:
                assert not np.any(np.isclose(hidden, v, rtol=REL_TOL, atol=ABS_TOL)), f"hidden value {v}"
    for e in run["evidence"]:
        qc = e.sensor_context.model_dump(exclude={"schema_version", "sensor_health"})
        assert all(v is None for v in qc.values()), qc  # no truth-derived quality context is fabricated
        rid = registry_of(e)
        assert rid is None or str(rid) in meta["registry_to_world"]  # hints are registry IDs, never world IDs


# ---------------------------------------------------------------------------------------------- 2. Model2T
def _pipeline(world_obj, rcfg, observations: list[Observation]) -> dict[str, tuple]:
    """The deployment-side structural path of conrad.orchestration.perception, run on captured payloads."""
    clock = observations[0].timestamp.clock_domain
    ids = IdFactory(424242)
    m2t = Model2T(
        ids.child("model2t"),
        model2t_config_from_dict({**rcfg.model2t, "clock_domain": clock}),
        mode=PropagationMode(rcfg.model2t_mode),
    )
    m2t.initialize(
        {
            "asset_registry": world_obj.context.asset_registry,
            "timestamp": TimeStamp(time_ns=0, clock_domain=clock),
        }
    )
    assoc = StructuralAssociator(world_obj.context, rcfg.association)
    by_t: dict[int, list[Observation]] = {}
    for o in observations:
        by_t.setdefault(o.timestamp.time_ns, []).append(o)
    for t_ns in sorted(by_t):
        matched = []
        for o in by_t[t_ns]:
            ev, _ = structured_evidence(o, ids, None, independence_group=f"obs:{o.observation_id}")
            ev, _ = assoc.associate(o, ev)
            if registry_of(ev) is not None:
                matched.append(ev)
        m2t.ingest(matched)
        m2t.update_beliefs(TimeStamp(time_ns=t_ns, clock_domain=clock))
    return {
        str(rid): (
            b.uc,
            b.ua,
            tuple((q, e.level, e.level_var, e.status.value) for q, e in sorted(b.estimates.items())),
        )
        for rid, b in sorted(m2t.beliefs.items(), key=lambda kv: str(kv[0]))
    }


def _randomize_truth(world_obj, seed: int) -> None:
    """Scramble everything a sensor must not reach: hidden state, hidden parameters, the registry mapping."""
    rng = np.random.default_rng(seed)
    asm, _ = world_obj.t2t._require()
    for rt in asm.runtimes.values():
        rt.state = replace(
            rt.state,
            corrosion_depth_m=float(rng.uniform(0, 0.012)),
            corrosion_area_fraction=float(rng.uniform(0, 1)),
            coating_breakdown_fraction=float(rng.uniform(0, 1)),
            crack_length_m=float(rng.uniform(0, 0.5)),
            crack_depth_m=float(rng.uniform(0, 0.01)),
        )
    wids = list(world_obj.mapping.to_world.values())
    rng.shuffle(wids)
    world_obj.mapping.to_world.update(dict(zip(list(world_obj.mapping.to_world), wids, strict=True)))
    world_obj.recorder.target_states.clear()
    world_obj.recorder.meta["defect"] = {"corrosion_depth_m": 0.0, "crack_length_m": 0.0}


def test_model2t_output_is_invariant_to_inaccessible_truth(run, world):
    w, rcfg = world
    captured = run["structural"]
    target = run["truth"]["meta"]["target_registry_id"]
    before = _pipeline(w, rcfg, captured)
    assert target in before and any(st == "OBSERVED" for _, _, _, st in before[target][2])
    for seed in (1, 2):
        _randomize_truth(w, seed)
        assert _pipeline(w, rcfg, captured) == before
    # positive control: the same pipeline does respond to the sensor payload itself
    scaled = [
        o.model_copy(update={"inline_values": tuple(v * 1.5 for v in o.inline_values or ())})
        for o in captured
    ]
    assert _pipeline(w, rcfg, scaled)[target] != before[target]


# ---------------------------------------------------------------------------------------------- 3. Twin2T
def _ctx(deg):
    ids = IdFactory(seed=77)
    pose = Pose(frame_id="WORLD", position_m=(0.0, 0.0, -10.0))
    sensor = SensorSpec(
        sensor_id=ids.new(),
        modality="STRUCTURED",
        frame_id="sensor",
        mount_pose=pose,
        rate_hz=1.0,
        parameters={"t2t_fidelity": "T0"},
    )
    return SensingContext(
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor=sensor,
        true_pose=pose,
        estimated_pose=None,
        timestamp=stamp(10.0, "sim"),
        degradation=deg,
    )


def _twin(root: Path, name: str) -> Twin2T:
    """A fresh, identically seeded twin (twins hold a store lock, so they are rebuilt, not deep-copied)."""
    twin = Twin2T(IdFactory(seed=31), ObjectStore(root / name))
    twin.initialize(build_small_scenario(31))
    twin.step(400 * 86400.0)
    for rt in twin._require()[0].runtimes.values():  # every crack-capable component gets a sizeable crack
        rt.state = replace(rt.state, crack_length_m=0.06 if rt.state.validity.get("crack_length_m") else 0.0)
    return twin


def test_t0_payload_ignores_hidden_fields_a_sensor_cannot_see(tmp_path):
    def payload(t):
        deg = {f"visibility:{e}": 0.8 for e in t.component_ids}
        return [tuple(s.observation.inline_values) for s in t.generate_observation(_ctx(deg))]

    base = payload(_twin(tmp_path, "a"))
    assert len(base) >= 3 and any(v[2] > 0.01 for v in base)
    hidden = _twin(tmp_path, "b")
    for rt in hidden._require()[0].runtimes.values():
        rt.state = replace(rt.state, crack_depth_m=rt.state.crack_depth_m * 3.0 + 1e-3)
        rt.params = replace(rt.params, wall_thickness_m=rt.params.wall_thickness_m * 1.7)
    assert payload(hidden) == base
    sensed = _twin(tmp_path, "c")
    for rt in sensed._require()[0].runtimes.values():
        rt.state = replace(rt.state, crack_length_m=rt.state.crack_length_m * 2.0)
    assert payload(sensed) != base  # positive control
    assert {"crack_depth_m", "crack_length_m"} <= set(STATE_DIMENSIONS)
