"""Lineage-aware split builder and leakage validator (ch24, ch26 Data splits, ch36 Data provenance).

Samples are never split at frame level when they share a lineage. The unit of assignment is a
lineage group: the connected component of samples that share a value on any chosen split unit.
Augmented, cropped or re-rendered derivatives inherit the lineage of their source sample.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from pydantic import Field, model_validator

from conrad.data.manifest import DatasetManifest, LineageKeys, SplitUnit
from conrad.schemas.base import ConradModel, digest_of

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TEST_OOD = "test_ood"


class LineageError(ValueError):
    """A sample cannot be placed without breaking lineage rules."""


class LineageLeakageError(AssertionError):
    """Two partitions share a lineage, a derivative chain or duplicated content."""

    def __init__(self, findings: Sequence[str]) -> None:
        super().__init__("lineage leakage: " + "; ".join(findings[:10]))
        self.findings = tuple(findings)


class SampleRef(ConradModel):
    sample_id: str = Field(min_length=1)
    lineage: LineageKeys = LineageKeys()
    derived_from: str | None = Field(default=None, description="sample_id of the source sample")
    content_sha256: str | None = None


class SplitSpec(ConradModel):
    units: tuple[SplitUnit, ...] = Field(min_length=1)
    fractions: dict[str, float] = Field(default_factory=lambda: {TRAIN: 0.7, VALIDATION: 0.15, TEST: 0.15})
    seed: int = 0
    ood_unit: SplitUnit | None = None
    ood_holdout: tuple[str, ...] = ()
    ood_split_name: str = TEST_OOD

    @model_validator(mode="after")
    def _check(self) -> SplitSpec:
        if not self.fractions:
            raise ValueError("fractions is empty")
        if any(v < 0 for v in self.fractions.values()):
            raise ValueError("negative split fraction")
        total = sum(self.fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions sum to {total}, expected 1.0")
        if self.ood_split_name in self.fractions:
            raise ValueError("the OOD split is filled by structural holdout, not by a fraction")
        if bool(self.ood_holdout) != (self.ood_unit is not None):
            raise ValueError("ood_unit and ood_holdout must be given together")
        if self.ood_unit is SplitUnit.SEED:
            raise ValueError("OOD must hold out a structural family; a seed alone is not structural")
        return self


class SplitAssignment(ConradModel):
    spec: SplitSpec
    splits: dict[str, tuple[str, ...]]
    group_count: int
    split_hash: str

    def split_of(self, sample_id: str) -> str:
        for name, ids in self.splits.items():
            if sample_id in ids:
                return name
        raise KeyError(sample_id)


def resolve_lineages(samples: Sequence[SampleRef]) -> dict[str, LineageKeys]:
    """Root lineage for every sample; derivatives inherit and cannot override."""
    by_id: dict[str, SampleRef] = {}
    for s in samples:
        if s.sample_id in by_id:
            raise LineageError(f"duplicate sample_id {s.sample_id!r}")
        by_id[s.sample_id] = s
    resolved: dict[str, LineageKeys] = {}
    for s in samples:
        seen: set[str] = set()
        current = s
        while current.derived_from is not None:
            if current.sample_id in seen:
                raise LineageError(f"derived_from cycle through {current.sample_id!r}")
            seen.add(current.sample_id)
            parent = by_id.get(current.derived_from)
            if parent is None:
                raise LineageError(
                    f"{current.sample_id!r} derives from unknown sample {current.derived_from!r}"
                )
            current = parent
        resolved[s.sample_id] = current.lineage
    return resolved


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def lineage_groups(samples: Sequence[SampleRef], units: Sequence[SplitUnit]) -> list[tuple[str, ...]]:
    """Connected components over shared lineage values, derivative links and identical content."""
    resolved = resolve_lineages(samples)
    uf = _UnionFind()
    for s in samples:
        node = f"s:{s.sample_id}"
        uf.find(node)
        keys = [(u, v) for u in units if (v := resolved[s.sample_id].value(u)) is not None]
        if not keys:
            raise LineageError(
                f"sample {s.sample_id!r} has no lineage key for units {[u.value for u in units]}; "
                "a frame-level random split is not permitted"
            )
        for unit, value in keys:
            uf.union(node, f"k:{unit.value}={value}")
        if s.derived_from is not None:
            uf.union(node, f"s:{s.derived_from}")
        if s.content_sha256 is not None:
            uf.union(node, f"d:{s.content_sha256}")
    groups: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        groups[uf.find(f"s:{s.sample_id}")].append(s.sample_id)
    return sorted(tuple(sorted(g)) for g in groups.values())


def compute_split_hash(spec: SplitSpec, splits: Mapping[str, Sequence[str]]) -> str:
    return digest_of(
        {"spec": spec.model_dump(mode="json"), "splits": {k: sorted(v) for k, v in sorted(splits.items())}}
    )


def build_splits(samples: Sequence[SampleRef], spec: SplitSpec) -> SplitAssignment:
    """Deterministic group-level assignment. Same samples + same spec -> same split_hash."""
    resolved = resolve_lineages(samples)
    groups = lineage_groups(samples, spec.units)
    holdout = set(spec.ood_holdout)
    ood_groups: list[tuple[str, ...]] = []
    regular: list[tuple[str, ...]] = []
    for group in groups:
        is_ood = spec.ood_unit is not None and any(
            resolved[sid].value(spec.ood_unit) in holdout for sid in group
        )
        (ood_groups if is_ood else regular).append(group)
    if spec.ood_unit is not None:
        present = {resolved[s.sample_id].value(spec.ood_unit) for s in samples}
        absent = sorted(holdout - {p for p in present if p is not None})
        if absent:
            raise LineageError(f"OOD holdout families not present in the data: {absent}")

    def order_key(group: tuple[str, ...]) -> str:
        return hashlib.sha256(f"{spec.seed}|{group[0]}".encode()).hexdigest()

    regular.sort(key=order_key)
    names = sorted(spec.fractions, key=lambda n: (-spec.fractions[n], n))
    total = sum(len(g) for g in regular)
    assigned: dict[str, list[str]] = {name: [] for name in spec.fractions}
    for group in regular:
        # Largest remaining deficit relative to the target count; ties resolved by name order.
        deficits = [(spec.fractions[n] * total - len(assigned[n]), n) for n in names if spec.fractions[n] > 0]
        best = max(deficits, key=lambda d: (d[0], -names.index(d[1])))[1]
        assigned[best].extend(group)
    splits: dict[str, tuple[str, ...]] = {n: tuple(sorted(ids)) for n, ids in assigned.items()}
    if spec.ood_unit is not None:
        splits[spec.ood_split_name] = tuple(sorted(sid for g in ood_groups for sid in g))
    result = SplitAssignment(
        spec=spec, splits=splits, group_count=len(groups), split_hash=compute_split_hash(spec, splits)
    )
    by_id = {s.sample_id: s for s in samples}
    assert_no_lineage_leakage({n: [by_id[i] for i in ids] for n, ids in splits.items()}, spec.units)
    return result


def find_lineage_leakage(
    splits: Mapping[str, Sequence[SampleRef]], units: Sequence[SplitUnit] | None = None
) -> list[str]:
    """Return every leakage finding. ``units=None`` checks every lineage unit (strictest)."""
    check_units = tuple(units) if units is not None else tuple(SplitUnit)
    all_samples = [s for members in splits.values() for s in members]
    findings: list[str] = []
    owner: dict[str, str] = {}
    for name, members in splits.items():
        for s in members:
            if s.sample_id in owner and owner[s.sample_id] != name:
                findings.append(f"sample {s.sample_id!r} is in both {owner[s.sample_id]!r} and {name!r}")
            owner.setdefault(s.sample_id, name)
    unique = list({s.sample_id: s for s in all_samples}.values())
    try:
        resolved = resolve_lineages(unique)
    except LineageError as exc:
        return [*findings, f"unresolvable derivation: {exc}"]
    key_owner: dict[tuple[str, str], str] = {}
    digest_owner: dict[str, str] = {}
    for name, members in splits.items():
        for s in members:
            for unit in check_units:
                value = resolved[s.sample_id].value(unit)
                if value is None:
                    continue
                first = key_owner.setdefault((unit.value, value), name)
                if first != name:
                    findings.append(f"{unit.value}={value!r} appears in both {first!r} and {name!r}")
            if s.derived_from is not None and owner.get(s.derived_from, name) != name:
                findings.append(
                    f"derivative {s.sample_id!r} ({name!r}) is separated from its source "
                    f"{s.derived_from!r} ({owner[s.derived_from]!r})"
                )
            if s.content_sha256 is not None:
                first = digest_owner.setdefault(s.content_sha256, name)
                if first != name:
                    findings.append(
                        f"identical content {s.content_sha256[:12]} in both {first!r} and {name!r}"
                    )
    return sorted(set(findings))


def assert_no_lineage_leakage(
    splits: Mapping[str, Sequence[SampleRef]], units: Sequence[SplitUnit] | None = None
) -> None:
    findings = find_lineage_leakage(splits, units)
    if findings:
        raise LineageLeakageError(findings)


def samples_from_manifest(manifest: DatasetManifest, streams: Iterable[str] | None = None) -> list[SampleRef]:
    """One SampleRef per manifest file; derivation and content identity are carried over."""
    wanted = set(streams) if streams is not None else None
    return [
        SampleRef(sample_id=f.path, lineage=f.lineage, derived_from=f.derived_from, content_sha256=f.sha256)
        for f in manifest.files
        if wanted is None or f.stream in wanted
    ]
