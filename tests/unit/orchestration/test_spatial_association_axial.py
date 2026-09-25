"""Axial segment association must abstain near joints and overlapping axes."""

from uuid import UUID

import numpy as np

from conrad.orchestration.association import unique_axial_segment
from conrad.orchestration.mission_context import DesignComponent


def segment(identity: int, x0: float, x1: float, survey_sigma_m: float = 0.0) -> DesignComponent:
    return DesignComponent(
        registry_id=UUID(int=identity),
        component_type="SEGMENT",
        shape="CAPSULE",
        p0_m=(x0, 0.0, 0.0),
        p1_m=(x1, 0.0, 0.0),
        radius_m=0.3,
        survey_sigma_m=survey_sigma_m,
    )


def test_exact_adjacent_segments_resolve_only_with_axial_separation():
    left, target, right = segment(1, -4.0, 0.0), segment(2, 0.0, 4.0), segment(3, 4.0, 8.0)
    candidates = [left, target, right]
    assert unique_axial_segment(np.array([0.28, 0.3, 0.0]), 0.05, candidates, 3.0) == target.registry_id
    assert unique_axial_segment(np.array([3.72, 0.3, 0.0]), 0.05, candidates, 3.0) == target.registry_id
    assert unique_axial_segment(np.array([0.04, 0.3, 0.0]), 0.05, candidates, 3.0) is None
    assert unique_axial_segment(np.array([4.0, 0.3, 0.0]), 0.05, candidates, 3.0) is None


def test_overlapping_or_uncertain_designs_remain_unassociated():
    target, overlapping = segment(2, 0.0, 4.0), segment(4, 0.1, 4.1)
    point = np.array([0.28, 0.3, 0.0])
    assert unique_axial_segment(point, 0.05, [target, overlapping], 3.0) is None
    assert unique_axial_segment(point, 0.05, [target, segment(5, -4.0, 0.0, 0.001)], 3.0) is None
