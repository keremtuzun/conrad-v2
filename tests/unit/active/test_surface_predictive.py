"""Belief-side surface-cell predictive model of MCBR in the mission (I4 repair)."""

from __future__ import annotations

import math
import uuid

import numpy as np
import pytest

from conrad.active.candidates import SensorOption
from conrad.active.predictive import expected_values, predicted_coverage, scalar_entropy_reduction
from conrad.active.surface_predictive import (
    QuantityChannel,
    SurfaceCellPredictive,
    SurfacePredictiveConfig,
    information_equivalent_std,
)
from conrad.domains.technical.config import CoverageConfig, SensorCharacteristics
from conrad.domains.technical.coverage import geometry_from_context
from conrad.orchestration.mission_predictive import capsule_cells, detection_probability, look_rel_sd
from conrad.schemas.frames import WORLD, Pose

SENSOR = SensorOption(sensor_id=uuid.UUID(int=7), modality="STRUCTURAL", min_range_m=0.5, max_range_m=4.0)
RID = uuid.UUID(int=11)


def _geometry():
    ctx = {
        "design_geometry": [
            {
                "registry_id": str(RID),
                "shape": "CAPSULE",
                "p0_m": [-1, 0, 0],
                "p1_m": [1, 0, 0],
                "radius_m": 0.2,
            }
        ]
    }
    return geometry_from_context(ctx, CoverageConfig())[RID]


@pytest.mark.parametrize("var,info", [(1.0, 0.3), (4e-6, 1.2), (2.5e-7, 0.01)])
def test_information_equivalent_std_reproduces_the_expected_entropy_reduction(var, info):
    std = information_equivalent_std(var, info)
    assert scalar_entropy_reduction(var, std) == pytest.approx(info, rel=1e-6)
    assert information_equivalent_std(var, 0.0) > 1e3 * math.sqrt(var)


def test_capsule_cells_are_indexed_like_model2t_coverage():
    g = _geometry()
    pts, nrm = capsule_cells(g)
    assert len(pts) == g.n_cells
    for i, p in enumerate(pts):
        assert g.cell_of(p) == i
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)


def _model(prior: np.ndarray, weights: np.ndarray, read: bool = False) -> SurfaceCellPredictive:
    ch = QuantityChannel(
        quantity="crack_length_m",
        unread_mean=2e-3,
        unread_var=1e-5,
        look_std_unread=2e-3,
        detection_probability=0.5,
        **(
            {
                "read_mean": 0.05,
                "read_var": 4e-4,
                "look_std_read": 0.01,
                "read_var_floor": 1e-5,
                "locus_cell": 0,
            }
            if read
            else {}
        ),
    )
    return SurfaceCellPredictive(cell_prior=prior, channels=(ch,), cell_weights=lambda pose, s: weights)


def test_eig_is_the_erasure_probability_times_the_gaussian_information():
    pose = Pose(frame_id=WORLD, position_m=(0.0, 2.0, 0.0))
    prior = np.array([0.0, 0.5, 0.5])
    blind = _model(prior, np.array([1.0, 0.0, 0.0]))  # sees only an already-covered cell
    half = _model(prior, np.array([0.0, 1.0, 0.0]))
    full = _model(prior, np.array([0.0, 1.0, 1.0]))
    e = [expected_values(m, m.predict(pose, SENSOR))["entropy"] for m in (blind, half, full)]
    assert e[0] == pytest.approx(0.0, abs=1e-9)
    info = 0.5 * math.log1p(1e-5 / 4e-6)
    assert e[1] == pytest.approx(0.5 * 0.5 * info, rel=1e-6)
    assert e[2] == pytest.approx(2 * e[1], rel=1e-6)
    # every scalar is always reported: the shared feasibility filter is unchanged for every planner
    assert predicted_coverage(blind, pose, SENSOR) == 1.0


def test_read_channel_is_capped_by_the_persistent_bias_floor():
    pose = Pose(frame_id=WORLD, position_m=(0.0, 2.0, 0.0))
    m = _model(np.array([0.0, 1.0]), np.array([1.0, 0.0]), read=True)
    info = m.expected_information(pose, SENSOR)
    assert info["crack_length_m:read"] == pytest.approx(
        min(0.5 * math.log1p(4e-4 / 1e-4), 0.5 * math.log(4e-4 / 1e-5)), rel=1e-9
    )


def test_datasheet_terms_come_from_model2t_sensor_characteristics():
    sc = SensorCharacteristics()
    ind, sys_ = look_rel_sd("crack_length_m", 0.05, 0.0, 1.0, sc)
    assert sys_ == sc.crack_systematic_rel_sigma and ind > sc.crack_rel_sigma  # partial-view scatter adds
    assert detection_probability(0.08, 1e-6, 0.0, sc) > 0.9 > detection_probability(1e-3, 1e-8, 0.0, sc)
    assert detection_probability(0.02, 1e-6, 0.5, sc) < detection_probability(0.02, 1e-6, 0.0, sc)


def test_config_defaults_are_bounded():
    c = SurfacePredictiveConfig()
    assert 0.0 <= c.footprint_discount <= 1.0 and c.enabled


def test_worst_band_probability_follows_the_model2t_condition_rule():
    from types import SimpleNamespace

    from conrad.domains.technical.config import ConditionConfig
    from conrad.orchestration.mission_predictive import worst_band_probability

    b = SimpleNamespace(condition_cfg=ConditionConfig(), spec=SimpleNamespace(wall_thickness_m=None))
    band = ConditionConfig().bands[2] * ConditionConfig().crack_critical_m
    assert worst_band_probability("crack_length_m", band, 1e-6, b) == pytest.approx(0.5)
    assert worst_band_probability("crack_length_m", 3e-3, 1e-6, b) < 1e-6
    assert worst_band_probability("corrosion_depth_m", 0.014, 1e-8, b) > 0.99


def test_track_views_credit_only_cells_facing_past_positions():
    from types import SimpleNamespace

    from conrad.orchestration.mission_predictive import MissionPredictive

    g = _geometry()
    pts, nrm = capsule_cells(g)
    sensor = SimpleNamespace(
        sensor_id=uuid.UUID(int=3), modality="STRUCTURAL", parameters={"max_range_m": 4.0}
    )
    mp = MissionPredictive(
        m2t=None,
        m2s=None,
        boresight_sensor=sensor,
        unknown_block_probability=0.5,
        cfg=SurfacePredictiveConfig(track_discount=0.8),
        track=lambda: [[t, x, -2.0, 0.0] for t, x in enumerate(np.linspace(-1.0, 1.0, 9))],
    )
    w = mp._track_views(pts, nrm, pts + 0.15 * nrm)
    facing = nrm[:, 1] < -0.1
    assert (w[facing] > 0).all() and (w[nrm[:, 1] > 0.1] == 0).all()
    empty = MissionPredictive(
        None, None, sensor, 0.5, SurfacePredictiveConfig(), track=lambda: [[0, 50, 50, 0]]
    )
    assert not empty._track_views(pts, nrm, pts).any()
