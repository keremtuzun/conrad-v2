from __future__ import annotations

import numpy as np
import pytest

from conrad.schemas.frames import FramedPoint, FrameError
from conrad.schemas.observation import Modality
from conrad.twins.twin2e import Twin2EObservationLevelError
from conrad.twins.twin2e.fields import FIELD_UNITS
from conrad.twins.twin2e.priors import KIND_BIOFOULING


def _sessile(tw):
    return [e for e in tw.entities.values() if e.kind == KIND_BIOFOULING]


def test_environmental_point_sensor(make_twin, ctx_for):
    tw = make_twin()
    tw.step(60.0)
    pos = (0.0, 0.0, -10.0)
    samples = tw.generate_observation(ctx_for(tw, "ENVIRONMENTAL", pos))
    assert [s.observation.sensor_context["channel"] for s in samples] == list(FIELD_UNITS)
    for s in samples:
        o = s.observation
        ch = o.sensor_context["channel"]
        assert o.modality is Modality.ENVIRONMENTAL
        assert o.inline_units == FIELD_UNITS[ch]
        assert len(o.inline_values) == (3 if ch == "current" else 1)
        assert s.supervision is not None and s.supervision.observation_id == o.observation_id
        assert s.supervision.true_world_entity_id is None
        assert o.robot_pose_estimate is None  # the true pose never leaks into the observation


def test_sensor_field_subset_and_unknown_modality(make_twin, ctx_for):
    tw = make_twin()
    only = tw.generate_observation(ctx_for(tw, "ENVIRONMENTAL", (0, 0, -5), {"fields": ["temperature"]}))
    assert len(only) == 1
    assert tw.generate_observation(ctx_for(tw, "SONAR", (0, 0, -5))) == []


def test_survey_observation_carries_no_truth(make_twin, ctx_for):
    tw = make_twin(seed=11)
    target = _sessile(tw)[0]
    pos = tuple(target.position_m + np.array([0.0, 0.0, 1.5]))
    samples = []
    for _ in range(10):
        samples += tw.generate_observation(ctx_for(tw, "RGB", pos, {"footprint_radius_m": 3.0}))
    assert samples, "a close, clear-water survey should detect something"
    truth_ids = {str(k) for k in tw.entities}
    for s in samples:
        js = s.observation.canonical_json()
        assert not any(t in js for t in truth_ids)
        assert "true" not in js.lower()
        assert s.observation.modality is Modality.STRUCTURED
        assert s.observation.sensor_context["source_modality"] == "RGB"
        assert str(s.supervision.true_world_entity_id) in truth_ids
    hit = [s for s in samples if s.supervision.true_world_entity_id == target.entity_id]
    assert hit and all(0.0 <= s.observation.inline_values[0] <= 1.0 for s in hit)
    assert all(s.supervision.targets["cover"] == pytest.approx(target.cover) for s in hit)


def test_turbidity_reduces_detection(make_twin, ctx_for):
    def detections(extra_ntu):
        tw = make_twin(seed=12)
        tw.fields.local.turbidity += extra_ntu
        tw.fields.regional.turbidity += extra_ntu
        e = _sessile(tw)[0]
        pos = tuple(e.position_m + np.array([0.0, 0.0, 2.0]))
        n = 0
        for _ in range(60):
            n += sum(
                1
                for s in tw.generate_observation(ctx_for(tw, "RGB", pos, {"footprint_radius_m": 2.5}))
                if s.supervision.true_world_entity_id == e.entity_id
            )
        return n

    assert detections(40.0) < detections(0.0)


def test_feature_level_and_e2_refusal(make_twin, ctx_for):
    tw = make_twin(seed=13)
    e = _sessile(tw)[0]
    pos = tuple(e.position_m + np.array([0.0, 0.0, 1.0]))
    feats = []
    for _ in range(5):
        feats += tw.generate_observation(ctx_for(tw, "RGB", pos, {"level": "E1_FEATURE"}))
    assert feats and len(feats[0].observation.inline_values) == tw.cfg.observation.feature_dim + 3
    with pytest.raises(Twin2EObservationLevelError):
        tw.generate_observation(ctx_for(tw, "RGB", pos, {"level": "E2_SENSOR"}))


def test_each_stream_keeps_its_own_timestamp(make_twin, ctx_for):
    tw = make_twin()
    tw.step(10.0)
    a = tw.generate_observation(ctx_for(tw, "ENVIRONMENTAL", (0, 0, -5), {"fields": ["temperature"]}))
    tw.step(7.0)
    b = tw.generate_observation(ctx_for(tw, "ENVIRONMENTAL", (5, 5, -5), {"fields": ["turbidity"]}))
    assert a[0].observation.timestamp.time_ns == 10 * 10**9
    assert b[0].observation.timestamp.time_ns == 17 * 10**9
    with pytest.raises(ValueError):  # the twin cannot report a time it is not at
        tw.generate_observation(ctx_for(tw, "ENVIRONMENTAL", (0, 0, -5), ts=a[0].observation.timestamp))


def test_observability_modifiers_and_surface_cover(make_twin):
    tw = make_twin()
    e = _sessile(tw)[0]
    mods = tw.observability_modifiers(FramedPoint(frame_id="WORLD", xyz_m=tuple(e.position_m)))
    assert mods["biofouling_implies_corrosion"] is False
    assert mods["biofouling_entity_id"] == e.entity_id or mods["biofouling_cover"] >= e.cover
    assert mods["beam_attenuation_per_m"] > 0 and mods["turbidity_ntu"] >= 0
    assert mods["units"]["turbidity_ntu"] == "NTU"
    far = tw.observability_modifiers(FramedPoint(frame_id="WORLD", xyz_m=(-300.0, -300.0, -10.0)))
    assert far["biofouling_cover"] == 0.0
    assert tw.surface_cover(e.entity_id) == pytest.approx(e.cover)
    with pytest.raises(FrameError):
        tw.observability_modifiers(FramedPoint(frame_id="ROBOT", xyz_m=(0, 0, 0)))
    with pytest.raises(KeyError):
        tw.surface_cover(tw.ids.new())


def test_turbidity_raises_beam_attenuation(make_twin):
    tw = make_twin()
    p = FramedPoint(frame_id="WORLD", xyz_m=(0.0, 0.0, -10.0))
    before = tw.observability_modifiers(p)["beam_attenuation_per_m"]
    tw.fields.local.turbidity += 10.0
    assert tw.observability_modifiers(p)["beam_attenuation_per_m"] > before
