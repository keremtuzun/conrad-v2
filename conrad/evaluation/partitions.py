"""Immutable evaluation partitions (development / validation / final test / OOD test). EVALUATION PLANE.

The partition file ``configs/eval/partitions.yaml`` is pinned by the SHA-256 of its canonical content
(``PARTITIONS_SHA256``): a changed file is refused, so a split cannot drift after results are seen.

Every seed request names a *purpose*. Design and tuning code paths may only touch DEVELOPMENT and VALIDATION;
selection only VALIDATION; final evaluation only FINAL_TEST and OOD_TEST. A code path can also mark itself
with ``purpose_scope("design")``: any request for final-test or OOD seeds inside that scope raises
``PartitionAccessError`` even if the caller claims another purpose.

implementation_status: FROZEN_CONTRACT (split v1)
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from conrad.settings import REPO_ROOT

PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions.yaml"
# Digest of the canonical JSON of partitions.yaml v1 (recorded 2026-09-19). Changing the file breaks loading.
PARTITIONS_SHA256 = "074ae11310252217315b2ac3324fd914d1964a3e34b76b57a65f0ca3a3acb904"
CONTAMINATED_LABEL = "DEVELOPMENT / CONTAMINATED_FOR_FINAL_EVALUATION"
# NAV benchmark noise seeds (gate I2): a separate versioned file, because adding a domain to partitions.yaml would
# change the v1 digest pinned by the I4 artifacts. Pinned on 2026-09-19 before any run on its final_test seeds.
NAV_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_nav.yaml"
NAV_PARTITIONS_SHA256 = "4e1886400c78969118e5b8326f77a43133b9ce6ef499e964f096601637abad90"
NAV_DOMAIN = "nav"
# I5 integrated-mission seeds (M1-ACTION-E002): a separate versioned file for the same reason as NAV. Pinned on
# 2026-09-19 before any I5 mission ran on its final_test seeds.
I5_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i5.yaml"
I5_PARTITIONS_SHA256 = "a6342e86ccaa9c391d351a181064048cc1479a5af177bdd70912ac096d4f67c8"
I5_DOMAIN = "i5_mission"
# Formal Unity integration gates: the mission final_test worlds of partitions.yaml are SPENT (2T-R2, MCBR-E003),
# so I1/I3 moved to fresh worlds. Pinned on 2026-09-19 before any run on its final_test seeds.
UNITY_GATES_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_unity_gates.yaml"
UNITY_GATES_PARTITIONS_SHA256 = "ad753291d5825dace90f5757790efd096a5f867d51abe51de6545dceff12f88c"
UNITY_GATES_DOMAIN = "unity_gate"
# I5 integrated-mission seeds, version 2 (M1-ACTION-E003): the v1 final_test seeds are SPENT (M1-ACTION-E002).
# Development equals the v1 development split; final_test is fresh. Pinned on 2026-09-19 before any run on it.
I5_V2_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i5_v2.yaml"
I5_V2_PARTITIONS_SHA256 = "762eb144d3d25fd577affb5226791452d82a0c426e79013c6009a41bd73cba16"
I5_V2_DOMAIN = "i5_mission_v2"


class Partition(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    FINAL_TEST = "final_test"
    OOD_TEST = "ood_test"


class Purpose(StrEnum):
    DESIGN = "design"
    TUNING = "tuning"
    SELECTION = "selection"
    FINAL_EVALUATION = "final_evaluation"


ALLOWED: dict[Purpose, frozenset[Partition]] = {
    Purpose.DESIGN: frozenset({Partition.DEVELOPMENT, Partition.VALIDATION}),
    Purpose.TUNING: frozenset({Partition.DEVELOPMENT, Partition.VALIDATION}),
    Purpose.SELECTION: frozenset({Partition.VALIDATION}),
    Purpose.FINAL_EVALUATION: frozenset({Partition.FINAL_TEST, Partition.OOD_TEST}),
}
HELD_OUT = frozenset({Partition.FINAL_TEST, Partition.OOD_TEST})
_SCOPE: contextvars.ContextVar[tuple[Purpose, ...]] = contextvars.ContextVar("partition_scope", default=())


class PartitionAccessError(PermissionError):
    """A design/tuning/selection code path asked for held-out data (or final evaluation for design data)."""


class PartitionIntegrityError(RuntimeError):
    """The partition file changed after it was frozen, or its partitions overlap."""


def canonical_digest(path: Path = PARTITIONS_PATH) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Split:
    domain: str
    partition: Partition
    world_seeds: tuple[int, ...]
    families: tuple[str, ...]
    replicates_per_world: int
    digest: str


@contextlib.contextmanager
def purpose_scope(purpose: str | Purpose) -> Iterator[None]:
    """Mark the enclosed code path (e.g. 'design'); held-out requests inside it raise."""
    token = _SCOPE.set((*_SCOPE.get(), Purpose(purpose)))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def check_access(partition: str | Partition, purpose: str | Purpose) -> None:
    part, purp = Partition(partition), Purpose(purpose)
    scopes = _SCOPE.get()
    for p in (*scopes, purp):
        if part not in ALLOWED[p]:
            raise PartitionAccessError(
                f"purpose {p.value!r} may not read partition {part.value!r} "
                f"(allowed: {sorted(x.value for x in ALLOWED[p])}; active scopes: {[s.value for s in scopes]})"
            )


def _seeds(spec: dict[str, Any]) -> tuple[int, ...]:
    lo, hi = spec.get("range", [0, 0])
    return tuple(int(s) for s in spec.get("explicit", [])) + tuple(range(int(lo), int(hi)))


def load(path: Path = PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    digest = canonical_digest(path)
    if verify_digest and digest != PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate(raw)
    return {"raw": raw, "digest": digest}


def validate(raw: dict[str, Any]) -> None:
    """Seeds are pairwise disjoint across partitions and domains; OOD families are held out."""
    seen: dict[int, str] = {}
    for domain in ("abstract", "mission"):
        fams = raw[domain]["families"]
        in_dist = {f for p in Partition if p is not Partition.OOD_TEST for f in fams[p.value]}
        if in_dist & set(fams[Partition.OOD_TEST.value]):
            raise PartitionIntegrityError(f"{domain}: OOD families overlap in-distribution families")
        for p in Partition:
            for s in _seeds(raw[domain]["world_seeds"][p.value]):
                key = f"{domain}:{p.value}"
                if s in seen and seen[s].split(":")[1] != p.value:
                    raise PartitionIntegrityError(f"seed {s} is in both {seen[s]} and {key}")
                seen[s] = key
    for c in raw.get("contaminated", []):
        owner = seen.get(int(c["seed"]))
        if owner is None or not owner.endswith(Partition.DEVELOPMENT.value):
            raise PartitionIntegrityError(f"contaminated seed {c['seed']} must be in development only")


def load_nav(path: Path = NAV_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The NAV noise-seed partition; its seeds must be disjoint from every seed of ``partitions.yaml``."""
    digest = canonical_digest(path)
    if verify_digest and digest != NAV_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {NAV_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_nav(raw, load()["raw"])
    return {"raw": raw, "digest": digest}


def validate_nav(raw: dict[str, Any], main: dict[str, Any]) -> None:
    seeds = raw["noise_seeds"]
    dev, final = set(_seeds(seeds["development"])), _seeds(seeds["final_test"])
    if set(final) & dev:
        raise PartitionIntegrityError("nav: final_test and development seeds overlap")
    if len(final) != len(raw["benchmarks"]):
        raise PartitionIntegrityError("nav: one final_test noise seed per benchmark is required")
    other = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    if (dev | set(final)) & other:
        raise PartitionIntegrityError("nav seeds collide with partitions.yaml seeds")
    for c in raw.get("contaminated", []):
        if not {int(s) for s in c["seeds"]} <= dev:
            raise PartitionIntegrityError(f"contaminated nav seeds {c['seeds']} must be in development only")


def _nav_split(part: Partition) -> Split:
    loaded = load_nav()
    raw = loaded["raw"]
    if part.value not in raw["noise_seeds"]:
        raise KeyError(f"nav partition has no {part.value!r} split")
    return Split(
        domain=NAV_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["noise_seeds"][part.value]),
        families=tuple(raw["benchmarks"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i5(path: Path = I5_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The I5 integrated-mission partition; disjoint from partitions.yaml, the NAV file and reserved ranges."""
    digest = canonical_digest(path)
    if verify_digest and digest != I5_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I5_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i5(raw, load()["raw"], load_nav()["raw"])
    return {"raw": raw, "digest": digest}


def validate_i5(raw: dict[str, Any], main: dict[str, Any], nav: dict[str, Any]) -> None:
    seeds = raw["world_seeds"]
    dev, final = set(_seeds(seeds["development"])), set(_seeds(seeds["final_test"]))
    if dev & final:
        raise PartitionIntegrityError("i5: final_test and development seeds overlap")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    if (dev | final) & taken:
        raise PartitionIntegrityError("i5 seeds collide with seeds used by another partition or experiment")


def _i5_split(part: Partition) -> Split:
    loaded = load_i5()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i5 partition has no {part.value!r} split")
    return Split(
        domain=I5_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["scenarios"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_unity_gates(
    path: Path = UNITY_GATES_PARTITIONS_PATH, *, verify_digest: bool = True
) -> dict[str, Any]:
    """The formal Unity gate worlds; disjoint from every other partition file and the reserved ranges."""
    digest = canonical_digest(path)
    if verify_digest and digest != UNITY_GATES_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {UNITY_GATES_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_unity_gates(raw, load()["raw"], load_nav()["raw"], load_i5()["raw"])
    return {"raw": raw, "digest": digest}


def validate_unity_gates(
    raw: dict[str, Any], main: dict[str, Any], nav: dict[str, Any], i5: dict[str, Any]
) -> None:
    seeds = raw["world_seeds"]
    dev, final = set(_seeds(seeds["development"])), set(_seeds(seeds["final_test"]))
    if dev & final:
        raise PartitionIntegrityError("unity_gate: final_test and development seeds overlap")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in i5["world_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    if (dev | final) & taken:
        raise PartitionIntegrityError(
            "unity_gate seeds collide with seeds used by another partition or experiment"
        )
    if not set(raw.get("roles", {}).values()) <= final:
        raise PartitionIntegrityError("unity_gate roles must name final_test seeds")


def unity_gate_seed(gate: str) -> int:
    """The held-out world of one formal Unity gate (final evaluation only)."""
    check_access(Partition.FINAL_TEST, Purpose.FINAL_EVALUATION)
    return int(load_unity_gates()["raw"]["roles"][gate])


def _unity_gates_split(part: Partition) -> Split:
    loaded = load_unity_gates()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"unity_gate partition has no {part.value!r} split")
    return Split(
        domain=UNITY_GATES_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["families"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i5_v2(path: Path = I5_V2_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The I5 v2 partition; disjoint from every other partition file and the reserved ranges (see validate)."""
    digest = canonical_digest(path)
    if verify_digest and digest != I5_V2_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I5_V2_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i5_v2(raw, load()["raw"], load_nav()["raw"], load_i5()["raw"], load_unity_gates()["raw"])
    return {"raw": raw, "digest": digest}


def validate_i5_v2(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    i5: dict[str, Any],
    unity: dict[str, Any],
) -> None:
    """final_test must be disjoint from every seed anywhere; development may only equal v1 development."""
    seeds = raw["world_seeds"]
    dev, final = set(_seeds(seeds["development"])), set(_seeds(seeds["final_test"]))
    if dev & final:
        raise PartitionIntegrityError("i5_v2: final_test and development seeds overlap")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in unity["world_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    v1_dev = set(_seeds(i5["world_seeds"]["development"]))
    v1_other = {s for p, spec in i5["world_seeds"].items() if p != "development" for s in _seeds(spec)}
    if final & (taken | v1_dev | v1_other):
        raise PartitionIntegrityError("i5_v2 final_test seeds collide with seeds used by another partition")
    if dev & (taken | v1_other) or not dev <= v1_dev:
        raise PartitionIntegrityError("i5_v2 development seeds must be v1 development seeds and nothing else")


def _i5_v2_split(part: Partition) -> Split:
    loaded = load_i5_v2()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i5_v2 partition has no {part.value!r} split")
    return Split(
        domain=I5_V2_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["scenarios"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def split(domain: str, partition: str | Partition, purpose: str | Purpose) -> Split:
    """The only sanctioned way to obtain evaluation seeds. Raises on a forbidden (purpose, partition)."""
    part = Partition(partition)
    check_access(part, purpose)
    if domain == NAV_DOMAIN:
        return _nav_split(part)
    if domain == I5_DOMAIN:
        return _i5_split(part)
    if domain == UNITY_GATES_DOMAIN:
        return _unity_gates_split(part)
    if domain == I5_V2_DOMAIN:
        return _i5_v2_split(part)
    loaded = load()
    raw = loaded["raw"]
    if domain not in ("abstract", "mission"):
        raise KeyError(f"unknown partition domain {domain!r}")
    return Split(
        domain=domain,
        partition=part,
        world_seeds=_seeds(raw[domain]["world_seeds"][part.value]),
        families=tuple(raw[domain]["families"][part.value]),
        replicates_per_world=int(raw["replicates_per_world"]),
        digest=loaded["digest"],
    )


def partition_of(domain: str, seed: int) -> Partition | None:
    if domain == I5_DOMAIN:
        i5 = load_i5()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in i5.items() if int(seed) in _seeds(spec)), None)
    if domain == I5_V2_DOMAIN:
        v2 = load_i5_v2()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in v2.items() if int(seed) in _seeds(spec)), None)
    raw = load()["raw"]
    for p in Partition:
        if int(seed) in _seeds(raw[domain]["world_seeds"][p.value]):
            return p
    return None


def contamination_label(seed: int) -> str | None:
    raw = load()["raw"]
    return next((str(c["label"]) for c in raw.get("contaminated", []) if int(c["seed"]) == int(seed)), None)
