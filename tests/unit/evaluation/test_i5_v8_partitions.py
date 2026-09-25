"""The Spatial V1 I5 split is digest-pinned and keeps final worlds sealed."""

from pathlib import Path

import pytest

from conrad.evaluation import partitions as P


def test_i5_v8_splits_are_disjoint_and_access_controlled():
    loaded = P.load_i5_v8()
    assert loaded["digest"] == P.I5_V8_PARTITIONS_SHA256
    assert "I5-NOMINAL-READABLE" not in loaded["raw"]["scenarios"]
    with P.purpose_scope(P.Purpose.DESIGN):
        dev = P.split(P.I5_V8_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
        with pytest.raises(P.PartitionAccessError):
            P.split(P.I5_V8_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
    validation = P.split(P.I5_V8_DOMAIN, P.Partition.VALIDATION, P.Purpose.SELECTION).world_seeds
    final = P.split(P.I5_V8_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    assert (len(dev), len(validation), len(final)) == (10, 10, 40)
    assert not (set(dev) & set(validation) or set(dev) & set(final) or set(validation) & set(final))


def test_i5_v8_modified_file_is_refused(tmp_path: Path):
    changed = tmp_path / "partitions_i5_v8.yaml"
    changed.write_text(
        P.I5_V8_PARTITIONS_PATH.read_text(encoding="utf-8").replace("8500010", "8500011"),
        encoding="utf-8",
    )
    with pytest.raises(P.PartitionIntegrityError):
        P.load_i5_v8(changed)
