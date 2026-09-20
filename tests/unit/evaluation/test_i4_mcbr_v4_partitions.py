"""The MCBR V4 partition is pinned, fresh and purpose-guarded, and the view oracle is evaluation only."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from conrad.active.config import MCBRConfig
from conrad.evaluation import partitions as P
from conrad.evaluation.oracle.i4_view_oracle import ARMS, ORACLE_1, ORACLE_2, TruthView, oracle_planner
from conrad.schemas.ids import IdFactory

REPO = Path(__file__).resolve().parents[3]
SPENT = (*range(8000200, 8000220), *range(8001000, 8001060))


def test_the_v4_partition_file_is_digest_pinned() -> None:
    loaded = P.load_i4_mcbr_v4()
    assert loaded["digest"] == P.I4_MCBR_V4_PARTITIONS_SHA256


def test_every_v4_seed_is_fresh_and_the_splits_are_disjoint() -> None:
    seeds = P.load_i4_mcbr_v4()["raw"]["world_seeds"]
    by_part = {k: set(P._seeds(v)) for k, v in seeds.items()}
    assert set(by_part) == {"development", "validation", "final_test", "ood_test"}
    everything: set[int] = set()
    for part in by_part.values():
        assert not (part & everything)
        everything |= part
    assert not (everything & set(SPENT)), "a V4 seed collides with a spent gate I4 world"


def test_design_may_read_development_but_never_the_held_out_splits() -> None:
    with P.purpose_scope(P.Purpose.DESIGN):
        split = P.split(P.I4_MCBR_V4_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN)
        assert len(split.world_seeds) == 40
        for held in (P.Partition.FINAL_TEST, P.Partition.OOD_TEST):
            with pytest.raises(P.PartitionAccessError):
                P.split(P.I4_MCBR_V4_DOMAIN, held, P.Purpose.DESIGN)


def test_partition_of_reports_the_v4_domain() -> None:
    assert P.partition_of(P.I4_MCBR_V4_DOMAIN, 8002000) is P.Partition.DEVELOPMENT
    assert P.partition_of(P.I4_MCBR_V4_DOMAIN, 8002200) is P.Partition.FINAL_TEST
    assert P.partition_of(P.I4_MCBR_V4_DOMAIN, 8001000) is None


def _truth() -> TruthView:
    """Four cells on a ring; only cell 0 carries the defect, and a slab hides it from +x."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    normals = points.copy()
    patch = np.array([True, False, False, False])
    return TruthView(1, points, normals, patch, (), 10.0)


def test_the_one_step_oracle_prefers_the_pose_that_sees_the_defect() -> None:
    truth = _truth()
    assert truth.patch_fraction(truth.visible(np.array([5.0, 0.0, 0.0]))) == 1.0
    assert truth.patch_fraction(truth.visible(np.array([-5.0, 0.0, 0.0]))) == 0.0


def test_incidence_weighting_discounts_a_grazing_look_but_binary_visibility_does_not() -> None:
    """The binary oracle over-credits a grazing view, which is why it lost to coverage-only."""
    truth = _truth()
    head_on = np.array([5.0, 0.0, 0.0])
    grazing = np.array([1.4, 5.0, 0.0])  # in front of the defect cell, but at about 85 degrees of incidence
    assert truth.patch_fraction(truth.visible(grazing)) == truth.patch_fraction(truth.visible(head_on))
    assert truth.patch_fraction(truth.weights(grazing)) < 0.5 * truth.patch_fraction(truth.weights(head_on))


def test_every_oracle_arm_builds_a_planner_that_never_refuses_a_feasible_view() -> None:
    for name, (steps, cost_mode, weighted) in ARMS.items():
        planner = oracle_planner(IdFactory(7), MCBRConfig(), _truth(), steps, cost_mode, weighted, name)
        assert planner.name == name
        assert planner.value_gate is False
    assert ARMS[ORACLE_1] == (1, "none", False)
    assert ARMS[ORACLE_2] == (2, "none", False)


def test_no_runtime_package_imports_the_view_oracle() -> None:
    """The deployment planes may not reach the oracle; tests/leakage forbids conrad.evaluation outright."""
    offenders = []
    for package in ("active", "orchestration", "decision", "domains", "runtime"):
        base = REPO / "conrad" / package
        for path in sorted(base.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any("i4_view_oracle" in n for n in names):
                    offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders
