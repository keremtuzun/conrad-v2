"""The pinned NAV noise-seed partition of gate I2 (configs/eval/partitions_nav.yaml)."""

import pytest

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.action_matrix import SEEDS as I5_SEEDS


def test_nav_partition_is_pinned_and_disjoint():
    loaded = P.load_nav()
    assert loaded["digest"] == P.NAV_PARTITIONS_SHA256
    final = P.split(P.NAV_DOMAIN, "final_test", "final_evaluation")
    dev = P.split(P.NAV_DOMAIN, "development", "tuning")
    assert final.world_seeds == tuple(range(7400001, 7400007))
    assert final.families == ("NAV-001", "NAV-002", "NAV-003", "NAV-004", "NAV-005", "NAV-006")
    assert not set(final.world_seeds) & set(dev.world_seeds)
    assert not set(final.world_seeds) & {s for seeds in I5_SEEDS.values() for s in seeds}


def test_former_i2_seeds_are_contaminated_development():
    dev = set(P.split(P.NAV_DOMAIN, "development", "design").world_seeds)
    raw = P.load_nav()["raw"]
    bad = {int(s) for c in raw["contaminated"] for s in c["seeds"]}
    assert bad == set(range(7300001, 7300007)) and bad <= dev
    assert all(c["label"] == P.CONTAMINATED_LABEL for c in raw["contaminated"])


def test_tuning_cannot_read_nav_final_seeds():
    with pytest.raises(P.PartitionAccessError):
        P.split(P.NAV_DOMAIN, "final_test", "tuning")
    with P.purpose_scope("design"), pytest.raises(P.PartitionAccessError):
        P.split(P.NAV_DOMAIN, "final_test", "final_evaluation")
