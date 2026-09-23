"""The new I4 cycle cannot consume spent or undeclared held-out worlds."""

from pathlib import Path

import pytest

from conrad.evaluation import partitions as p


def test_new_i4_energy_partition_is_pinned_and_disjoint() -> None:
    loaded = p.load_i4_energy_v1()
    assert loaded["digest"] == p.I4_ENERGY_V1_PARTITIONS_SHA256
    stages = {
        stage: set(p.split(p.I4_ENERGY_V1_DOMAIN, stage, purpose).world_seeds)
        for stage, purpose in (
            (p.Partition.DEVELOPMENT, p.Purpose.DESIGN),
            (p.Partition.VALIDATION, p.Purpose.SELECTION),
            (p.Partition.FINAL_TEST, p.Purpose.FINAL_EVALUATION),
            (p.Partition.OOD_TEST, p.Purpose.FINAL_EVALUATION),
        )
    }
    assert len(set().union(*stages.values())) == sum(map(len, stages.values()))
    assert not set().union(*stages.values()) & set(range(8002200, 8002320))


def test_new_i4_energy_partition_refuses_edit(tmp_path: Path) -> None:
    changed = tmp_path / "partitions_i4_energy_v1.yaml"
    changed.write_text(p.I4_ENERGY_V1_PARTITIONS_PATH.read_text().replace("8100000", "8100001", 1))
    with pytest.raises(p.PartitionIntegrityError, match="changed after freezing"):
        p.load_i4_energy_v1(changed)


def test_design_scope_cannot_open_new_final() -> None:
    with p.purpose_scope(p.Purpose.DESIGN), pytest.raises(p.PartitionAccessError):
        p.split(p.I4_ENERGY_V1_DOMAIN, p.Partition.FINAL_TEST, p.Purpose.FINAL_EVALUATION)
