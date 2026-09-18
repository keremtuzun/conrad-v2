from __future__ import annotations

import pytest

from conrad.data.manifest import LineageKeys, SplitUnit
from conrad.data.splits import (
    TEST_OOD,
    LineageError,
    LineageLeakageError,
    SampleRef,
    SplitSpec,
    assert_no_lineage_leakage,
    build_splits,
    find_lineage_leakage,
)


def s(sid: str, **lineage) -> SampleRef:
    return SampleRef(sample_id=sid, lineage=LineageKeys(**lineage))


def test_frames_of_one_sequence_stay_together():
    samples = [s(f"q{q}_f{f}", sequence=f"q{q}") for q in range(6) for f in range(5)]
    result = build_splits(samples, SplitSpec(units=(SplitUnit.SEQUENCE,)))
    for q in range(6):
        assert len({result.split_of(f"q{q}_f{f}") for f in range(5)}) == 1


def test_samples_without_lineage_are_refused():
    with pytest.raises(LineageError, match="frame-level"):
        build_splits([s("a"), s("b")], SplitSpec(units=(SplitUnit.SEQUENCE,)))


def test_shared_site_links_sequences():
    samples = [
        s("a", sequence="q1", site="X"),
        s("b", sequence="q2", site="X"),
        s("c", sequence="q3", site="Y"),
    ]
    result = build_splits(samples, SplitSpec(units=(SplitUnit.SEQUENCE, SplitUnit.SITE)))
    assert result.split_of("a") == result.split_of("b")
    assert result.group_count == 2


def test_leakage_detection_sequence_derivative_and_duplicate():
    a = s("a", sequence="q1")
    crop = SampleRef(sample_id="a_crop", derived_from="a")
    with pytest.raises(LineageLeakageError):
        assert_no_lineage_leakage({"train": [a], "test": [s("b", sequence="q1")]})
    findings = find_lineage_leakage({"train": [a], "test": [crop]})
    assert any("derivative" in f for f in findings) and any("sequence" in f for f in findings)
    dup = find_lineage_leakage(
        {
            "train": [SampleRef(sample_id="x", lineage=LineageKeys(sequence="1"), content_sha256="d" * 64)],
            "test": [SampleRef(sample_id="y", lineage=LineageKeys(sequence="2"), content_sha256="d" * 64)],
        }
    )
    assert any("identical content" in f for f in dup)


def test_derivative_inherits_source_lineage():
    samples = [
        s("a", sequence="q1"),
        SampleRef(sample_id="a_aug", derived_from="a", lineage=LineageKeys(sequence="other")),
        s("b", sequence="q2"),
        s("c", sequence="q3"),
    ]
    result = build_splits(samples, SplitSpec(units=(SplitUnit.SEQUENCE,)))
    assert result.split_of("a") == result.split_of("a_aug")


def test_ood_holds_out_structural_family():
    samples = [s(f"{fam}{i}", world_family=fam, seed=str(i)) for fam in "ABCD" for i in range(3)]
    spec = SplitSpec(units=(SplitUnit.WORLD_FAMILY,), ood_unit=SplitUnit.WORLD_FAMILY, ood_holdout=("D",))
    result = build_splits(samples, spec)
    assert set(result.splits[TEST_OOD]) == {"D0", "D1", "D2"}
    with pytest.raises(ValueError):
        SplitSpec(units=(SplitUnit.SEED,), ood_unit=SplitUnit.SEED, ood_holdout=("1",))
    with pytest.raises(LineageError):
        build_splits(samples, spec.model_copy(update={"ood_holdout": ("Z",)}))


def test_split_hash_is_deterministic_and_seed_sensitive():
    samples = [s(f"f{i}", sequence=f"q{i}") for i in range(20)]
    a = build_splits(samples, SplitSpec(units=(SplitUnit.SEQUENCE,), seed=1))
    b = build_splits(list(reversed(samples)), SplitSpec(units=(SplitUnit.SEQUENCE,), seed=1))
    c = build_splits(samples, SplitSpec(units=(SplitUnit.SEQUENCE,), seed=2))
    assert a.split_hash == b.split_hash
    assert a.split_hash != c.split_hash
