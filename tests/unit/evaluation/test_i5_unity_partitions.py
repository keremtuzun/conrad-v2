from pathlib import Path

import pytest

from conrad.evaluation import partitions as P


def test_v2_formal_worlds_are_digest_pinned_and_disjoint_from_v1():
    loaded = P.load_i5_unity_v2()
    assert loaded["digest"] == P.I5_UNITY_V2_PARTITIONS_SHA256
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        v1 = set(P.split(P.I5_UNITY_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds)
        v2 = set(
            P.split(P.I5_UNITY_V2_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
        )
    assert v1.isdisjoint(v2)
    assert {7710300, 7710301} <= v2


def test_v2_partition_refuses_digest_drift(tmp_path: Path):
    changed = tmp_path / "partitions_i5_unity_v2.yaml"
    changed.write_text(P.I5_UNITY_V2_PARTITIONS_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    # Canonical YAML is insensitive to a trailing blank line, so mutate a value as well.
    changed.write_text(changed.read_text(encoding="utf-8").replace("7710310", "7710311"), encoding="utf-8")
    with pytest.raises(P.PartitionIntegrityError, match="changed after freezing"):
        P.load_i5_unity_v2(changed)


def test_v3_spatial_formal_worlds_are_digest_pinned_and_fresh():
    loaded = P.load_i5_unity_v3()
    assert loaded["digest"] == P.I5_UNITY_V3_PARTITIONS_SHA256
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        v2 = set(
            P.split(P.I5_UNITY_V2_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
        )
        v3 = set(
            P.split(P.I5_UNITY_V3_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
        )
    assert v2.isdisjoint(v3)
    assert tuple(sorted(v3)) == tuple(range(8701000, 8701010))


def test_v3_partition_refuses_digest_drift(tmp_path: Path):
    changed = tmp_path / "partitions_i5_unity_v3.yaml"
    changed.write_text(P.I5_UNITY_V3_PARTITIONS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    changed.write_text(changed.read_text(encoding="utf-8").replace("8701010", "8701011"), encoding="utf-8")
    with pytest.raises(P.PartitionIntegrityError, match="changed after freezing"):
        P.load_i5_unity_v3(changed)
