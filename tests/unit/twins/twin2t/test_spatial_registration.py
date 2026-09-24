"""Hard-bounded synthetic survey registration keeps credited support inside truth."""

from __future__ import annotations

import math

import numpy as np

from conrad.schemas.capsule_surface import surface_coordinates, surface_point
from conrad.sim.mission.registry import _noisy
from conrad.sim.mission.spatial_registration import capsule_registration_uncertainty


def test_survey_endpoint_error_is_hard_bounded_and_coordinate_bound_holds():
    rng = np.random.default_rng(2026201)
    truth_a = np.array((0.0, 0.0, -3.0))
    truth_b = np.array((4.0, 0.5, -3.0))
    radius = 0.35
    bound = 0.002
    length = float(np.linalg.norm(truth_b - truth_a))
    for _ in range(100):
        design_a = _noisy(truth_a, 0.001, rng, bound)
        design_b = _noisy(truth_b, 0.001, rng, bound)
        assert np.linalg.norm(np.asarray(design_a) - truth_a) <= bound + 1e-15
        assert np.linalg.norm(np.asarray(design_b) - truth_b) <= bound + 1e-15
        certificate = capsule_registration_uncertainty(design_a, design_b, radius, bound)
        assert certificate is not None
        dx, da = certificate
        for x in (0.0, 0.2 * length, 0.5 * length, 0.9 * length, length):
            for angle in (0.0, 0.4, math.pi, 4.2, 2 * math.pi - 0.01):
                point = surface_point(truth_a, truth_b, radius, x, angle)
                mapped_x, mapped_angle, _ = surface_coordinates(
                    np.asarray(design_a), np.asarray(design_b), point
                )
                angle_gap = abs((mapped_angle - angle + math.pi) % (2 * math.pi) - math.pi)
                assert abs(mapped_x - x) <= dx
                assert angle_gap <= da


def test_registration_refuses_geometries_without_a_useful_bound():
    assert capsule_registration_uncertainty((0, 0, 0), (0, 0, 4), 0.35, 0.002) is None
    assert capsule_registration_uncertainty((0, 0, 0), (4, 0, 0), 0.35, 0.1) is None
