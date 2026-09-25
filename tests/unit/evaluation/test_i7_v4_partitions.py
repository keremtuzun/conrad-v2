from __future__ import annotations

import pytest

from conrad.evaluation.partitions import (
    I7_V4_DOMAIN,
    I7_V4_PARTITIONS_SHA256,
    Partition,
    PartitionAccessError,
    Purpose,
    load_i7_v4,
    partition_of,
    split,
)


def test_i7_v4_digest_and_splits_are_frozen() -> None:
    loaded = load_i7_v4()
    assert loaded["digest"] == I7_V4_PARTITIONS_SHA256
    assert split(I7_V4_DOMAIN, Partition.VALIDATION, Purpose.SELECTION).world_seeds == tuple(
        range(8_400_000, 8_400_005)
    )
    assert split(I7_V4_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds == tuple(
        range(8_400_100, 8_400_105)
    )


def test_i7_v4_partition_lookup_and_access_are_fail_closed() -> None:
    assert partition_of(I7_V4_DOMAIN, 8_400_000) is Partition.VALIDATION
    assert partition_of(I7_V4_DOMAIN, 8_400_100) is Partition.FINAL_TEST
    assert partition_of(I7_V4_DOMAIN, 8_400_099) is None
    with pytest.raises(PartitionAccessError):
        split(I7_V4_DOMAIN, Partition.FINAL_TEST, Purpose.SELECTION)
