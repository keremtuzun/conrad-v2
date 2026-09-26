"""The Spatial V1.1 remediation split is digest-pinned and held-out worlds stay sealed."""

from pathlib import Path

import pytest

from conrad.evaluation import partitions as P


def test_i5_v10_splits_are_fresh_disjoint_and_access_controlled():
    loaded = P.load_i5_v10()
    assert loaded["digest"] == P.I5_V10_PARTITIONS_SHA256
    with P.purpose_scope(P.Purpose.DESIGN):
        development = P.split(P.I5_V10_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
        with pytest.raises(P.PartitionAccessError):
            P.split(P.I5_V10_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
    validation = P.split(P.I5_V10_DOMAIN, P.Partition.VALIDATION, P.Purpose.SELECTION).world_seeds
    final = P.split(P.I5_V10_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    v9 = P.load_i5_v9()["raw"]["world_seeds"]
    historical = {seed for spec in v9.values() for seed in P._seeds(spec)}
    assert (len(development), len(validation), len(final)) == (10, 10, 40)
    assert not (
        set(development) & set(validation) or set(development) & set(final) or set(validation) & set(final)
    )
    assert not (set(development) | set(validation) | set(final)) & historical


def test_i5_v10_modified_file_is_refused(tmp_path: Path):
    changed = tmp_path / "partitions_i5_v10.yaml"
    changed.write_text(
        P.I5_V10_PARTITIONS_PATH.read_text(encoding="utf-8").replace("8700010", "8700011"),
        encoding="utf-8",
    )
    with pytest.raises(P.PartitionIntegrityError):
        P.load_i5_v10(changed)
