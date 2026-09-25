"""The corrected Spatial V1 I5 split is digest-pinned and keeps held-out worlds sealed."""

from pathlib import Path

import pytest

from conrad.evaluation import partitions as P


def test_i5_v9_splits_are_fresh_disjoint_and_access_controlled():
    loaded = P.load_i5_v9()
    assert loaded["digest"] == P.I5_V9_PARTITIONS_SHA256
    with P.purpose_scope(P.Purpose.DESIGN):
        development = P.split(P.I5_V9_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
        with pytest.raises(P.PartitionAccessError):
            P.split(P.I5_V9_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
    validation = P.split(P.I5_V9_DOMAIN, P.Partition.VALIDATION, P.Purpose.SELECTION).world_seeds
    final = P.split(P.I5_V9_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    v8 = P.load_i5_v8()["raw"]["world_seeds"]
    historical = {seed for spec in v8.values() for seed in P._seeds(spec)}
    assert (len(development), len(validation), len(final)) == (10, 10, 40)
    assert not (
        set(development) & set(validation)
        or set(development) & set(final)
        or set(validation) & set(final)
    )
    assert not (set(development) | set(validation) | set(final)) & historical


def test_i5_v9_modified_file_is_refused(tmp_path: Path):
    changed = tmp_path / "partitions_i5_v9.yaml"
    changed.write_text(
        P.I5_V9_PARTITIONS_PATH.read_text(encoding="utf-8").replace("8600010", "8600011"),
        encoding="utf-8",
    )
    with pytest.raises(P.PartitionIntegrityError):
        P.load_i5_v9(changed)
