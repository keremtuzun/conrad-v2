from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.data.manifest import LineageKeys, SplitUnit
from conrad.data.splits import SampleRef, SplitSpec, build_splits, find_lineage_leakage


@st.composite
def sample_sets(draw):
    n = draw(st.integers(min_value=2, max_value=40))
    samples: list[SampleRef] = []
    for i in range(n):
        if samples and draw(st.booleans()) and draw(st.integers(0, 3)) == 0:
            parent = draw(st.sampled_from(samples)).sample_id
            samples.append(SampleRef(sample_id=f"s{i}", derived_from=parent))
            continue
        samples.append(
            SampleRef(
                sample_id=f"s{i}",
                lineage=LineageKeys(
                    sequence=f"q{draw(st.integers(0, 8))}",
                    site=draw(st.sampled_from([None, "A", "B", "C", "D"])),
                ),
            )
        )
    return samples


@settings(max_examples=60, deadline=None)
@given(sample_sets(), st.integers(0, 1000))
def test_built_splits_never_leak_and_cover_every_sample(samples, seed):
    spec = SplitSpec(units=(SplitUnit.SEQUENCE, SplitUnit.SITE), seed=seed)
    result = build_splits(samples, spec)
    assigned = [sid for ids in result.splits.values() for sid in ids]
    assert sorted(assigned) == sorted(x.sample_id for x in samples)
    by_id = {x.sample_id: x for x in samples}
    assert (
        find_lineage_leakage({k: [by_id[i] for i in v] for k, v in result.splits.items()}, spec.units) == []
    )
    assert build_splits(list(reversed(samples)), spec).split_hash == result.split_hash


@settings(max_examples=40, deadline=None)
@given(sample_sets())
def test_moving_one_lineage_member_is_always_detected(samples):
    spec = SplitSpec(units=(SplitUnit.SEQUENCE, SplitUnit.SITE))
    result = build_splits(samples, spec)
    by_id = {x.sample_id: x for x in samples}
    groups: dict[str, list[str]] = {}
    for name, split_ids in result.splits.items():
        for sid in split_ids:
            groups.setdefault(name, []).append(sid)
    for name, ids in groups.items():
        if len(ids) < 2:
            continue
        # Move one member of a multi-member lineage group into a fresh split.
        seq = {i: by_id[i].lineage.sequence for i in ids}
        for sid in ids:
            partners = [j for j in ids if j != sid and seq[j] is not None and seq[j] == seq[sid]]
            if partners:
                splits = {name: [by_id[j] for j in ids if j != sid], "moved": [by_id[sid]]}
                assert find_lineage_leakage(splits, spec.units)
                return
