"""The I5 repair split stays sealed and digest-pinned after development access."""

from pathlib import Path

import pytest

from conrad.evaluation import partitions as P


def test_i5_v7_splits_are_disjoint_and_final_is_sealed_during_design():
    loaded = P.load_i5_v7()
    assert loaded["digest"] == P.I5_V7_PARTITIONS_SHA256
    with P.purpose_scope(P.Purpose.DESIGN):
        dev = P.split(P.I5_V7_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
        with pytest.raises(P.PartitionAccessError):
            P.split(P.I5_V7_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
    validation = P.split(P.I5_V7_DOMAIN, P.Partition.VALIDATION, P.Purpose.SELECTION).world_seeds
    final = P.split(P.I5_V7_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    assert len(dev) == len(validation) == len(final) == 10
    assert not (set(dev) & set(validation) or set(dev) & set(final) or set(validation) & set(final))


def test_i5_v7_modified_file_is_refused(tmp_path: Path):
    changed = tmp_path / "partitions_i5_v7.yaml"
    changed.write_text(
        P.I5_V7_PARTITIONS_PATH.read_text(encoding="utf-8").replace("8300010", "8300011"),
        encoding="utf-8",
    )
    with pytest.raises(P.PartitionIntegrityError):
        P.load_i5_v7(changed)
