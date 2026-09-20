"""Sensor-characterised Model2T measurement model (docs/audits/MODEL2T_REPAIR.md). No Twin2T involvement."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from m2t_helpers import claim, evidence, model

from conrad.domains.technical import CORROSION_DEPTH, CRACK_LENGTH, Model2TConfig, PropagationMode
from conrad.domains.technical.config import SensorCharacteristics
from conrad.domains.technical.evidence import structured_evidence
from conrad.domains.technical.measurement import Reading, grid_update, log_likelihood
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation, SensorHealth
from conrad.schemas.timebase import stamp

SC = SensorCharacteristics()


def _feed(m, r, ev, readings, *, sensor="S", target="seg_a", t0=10.0, wall=None):
    for i, z in enumerate(readings):
        w = None if wall is None else wall[i]
        m.ingest([evidence(ev, r[target], t0 + i, crack=z, wall=w, group=f"sensor:{sensor}|obs:{i}")])
        m.update_beliefs(stamp(t0 + i, "sim"))
    return m.beliefs[r[target]]


def test_missed_detection_never_drags_a_large_crack_to_zero():
    m, r, ev = model(mode=PropagationMode.NONE)
    b = _feed(m, r, ev, [0.050, 0.047, 0.055, 0.052, 0.049])
    before = b.estimates[CRACK_LENGTH].level
    b = _feed(m, r, ev, [0.0008], t0=20.0)
    after = b.estimates[CRACK_LENGTH]
    assert after.level > 0.8 * before and after.level > 0.03
    assert after.status is KnowledgeStatus.OBSERVED
    assert b.uc < 0.35  # a miss is weak negative evidence, not a credible contradiction


def test_non_detection_only_leaves_the_crack_unknown_but_lowers_large_crack_odds():
    m, r, ev = model(mode=PropagationMode.NONE)
    b = _feed(m, r, ev, [0.0004, 0.0, 0.0009, 0.0002])
    est = b.estimates[CRACK_LENGTH]
    assert est.status is KnowledgeStatus.UNKNOWN and est.direct_lineage
    msg = m.export_beliefs()
    seg = next(x for x in msg if x.world_entity_id == r["seg_a"])
    assert claim(seg, CRACK_LENGTH).status is KnowledgeStatus.UNKNOWN
    grid = np.array([0.001, 0.005, 0.02, 0.08])
    ll = log_likelihood(Reading(CRACK_LENGTH, 0.0005, 0.9, 0.1), grid, SC)
    assert np.all(np.diff(ll) < 0)  # the bigger the crack, the less likely a miss
    assert ll[-1] > np.log(1e-3)  # ...but even an 80 mm crack can be missed (partial view)


def test_relative_noise_does_not_turn_scatter_on_a_large_crack_into_contradiction():
    m, r, ev = model(mode=PropagationMode.NONE)
    rng = np.random.default_rng(3)
    readings = [0.06 * float(np.exp(0.25 * rng.standard_normal())) for _ in range(12)]
    b = _feed(m, r, ev, readings)
    assert b.uc < 0.2 and not b.conflicts
    est = b.estimates[CRACK_LENGTH]
    assert 0.045 < est.level < 0.11


def test_credible_disagreement_beyond_relative_noise_raises_uc():
    m, r, ev = model(mode=PropagationMode.NONE)
    b = _feed(m, r, ev, [0.004] * 5)
    uc0 = b.uc
    b = _feed(m, r, ev, [0.080], t0=30.0)
    assert b.uc > uc0 and b.conflicts
    assert b.estimates[CRACK_LENGTH].level > 0.02  # the change is followed, not smoothed away


def test_same_sensor_bias_floor_is_not_averaged_away():
    def sd(sensors: list[str]) -> float:
        m, r, ev = model(mode=PropagationMode.NONE)
        for i, s in enumerate(sensors):
            m.ingest([evidence(ev, r["seg_a"], 10.0 + i, wall=4e-3, group=f"sensor:{s}|obs:{i}")])
            m.update_beliefs(stamp(10.0 + i, "sim"))
        e = m.beliefs[r["seg_a"]].estimates[CORROSION_DEPTH]
        return float(e.sd / e.level)

    one_sensor = sd(["A"] * 30)
    two_sensors = sd(["A", "B"] * 15)
    assert one_sensor >= SC.wall_systematic_rel_sigma * 0.99
    assert two_sensors < one_sensor


def test_partial_view_reading_is_a_lower_bound_not_a_contradiction():
    post = grid_update(0.08, 0.004**2, Reading(CRACK_LENGTH, 0.035, 0.9, 0.1), SC, inflate_on_surprise=True)
    assert not post.surprise and post.mean > 0.06
    full = replace(SC, full_view_prob=1.0)
    strict = grid_update(
        0.08, 0.004**2, Reading(CRACK_LENGTH, 0.035, 0.9, 0.1), full, inflate_on_surprise=True
    )
    assert strict.mean < post.mean


def test_legacy_absolute_model_is_selectable():
    cfg = Model2TConfig()
    legacy = replace(cfg, direct=replace(cfg.direct, measurement_model="ABSOLUTE_GAUSSIAN"))
    m, r, ev = model(config=legacy, mode=PropagationMode.NONE)
    b = _feed(m, r, ev, [0.0008])
    assert b.estimates[CRACK_LENGTH].status is KnowledgeStatus.OBSERVED  # the old behaviour


def test_adapter_groups_readings_by_sensor():
    ids = IdFactory(seed=5)
    sensor = ids.new()
    obs = Observation(
        observation_id=ids.new(),
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor_id=sensor,
        modality=Modality.STRUCTURED,
        timestamp=stamp(1.0, "sim"),
        sensor_frame="sensor",
        robot_pose_estimate=None,
        inline_values=(0.001, 0.1, 0.0),
        inline_units="m,1,m",
        sensor_health=SensorHealth.OK,
        sensor_context={
            "measurements": ["apparent_wall_loss", "surface_anomaly_score", "crack_indication_length"],
            "units": ["m", "1", "m"],
        },
    )
    ev, _ = structured_evidence(obs, ids, None, independence_group="obs:x")
    assert ev.independence_group == f"sensor:{sensor}|obs:x"
    assert ev.sensor_context.range_m is None  # nothing is copied into quality context


def test_a_miss_on_another_region_says_nothing_about_the_defect():
    from conrad.schemas.frames import SpatialSupport

    def at(e, x):
        sup = SpatialSupport(frame_id="WORLD", center_m=(x, 0.0, 0.0), position_sigma_m=0.05)
        return e.model_copy(update={"spatial_support": sup})

    def final(far_x: float) -> tuple[float, float]:
        m, r, ev = model(mode=PropagationMode.NONE)
        for i in range(4):
            e = evidence(ev, r["seg_a"], 10.0 + i, crack=0.06, wall=6e-3, group=f"sensor:S|obs:{i}")
            m.ingest([at(e, 0.0)])
            m.update_beliefs(stamp(10.0 + i, "sim"))
        for i in range(4):
            e = evidence(ev, r["seg_a"], 20.0 + i, crack=0.0, wall=5e-4, group=f"sensor:S|obs:n{i}")
            m.ingest([at(e, far_x)])
            m.update_beliefs(stamp(20.0 + i, "sim"))
        b = m.beliefs[r["seg_a"]]
        return b.estimates[CRACK_LENGTH].level, b.estimates[CORROSION_DEPTH].level

    same, other = final(0.0), final(1.0)
    assert other[0] > same[0] and other[1] > same[1]
    assert other[0] > 0.05 and other[1] > 5e-3  # the far-region readings do not drag the defect down
