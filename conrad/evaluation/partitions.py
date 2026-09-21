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
# I5 integrated-mission seeds, version 3 (M1-ACTION-E004): the v2 final_test seeds are SPENT (M1-ACTION-E003).
# Development equals the v1/v2 development split; final_test is fresh. Pinned on 2026-09-20 before any run on it.
I5_V3_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i5_v3.yaml"
I5_V3_PARTITIONS_SHA256 = "ac7e5bf025f57a1472c6b61e2d76b4c9b0c97eca1112f4991138943679364495"
I5_V3_DOMAIN = "i5_mission_v3"
# I5 integrated-mission seeds, version 4 (M1-ACTION-E005): the v3 final_test seeds are SPENT (M1-ACTION-E004),
# and E004 itself ran before commit 9280b5b turned the frozen MCBR V4 view execution protocol on for the
# PRODUCTION planner, so its numbers no longer describe the code on disk. Development equals the v1/v2/v3
# development split; final_test is fresh. Pinned on 2026-09-21 before any run on it.
I5_V4_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i5_v4.yaml"
I5_V4_PARTITIONS_SHA256 = "73454d92fd55cd482d21cafafe284b8d9036a5fb65740cb18465fab2bea56c77"
I5_V4_DOMAIN = "i5_mission_v4"
# Held-out worlds of the FORMAL Unity gate I5 integrated missions. The unity_gate final_test split is fully
# allocated and every I5 python-kernel final split is spent by its surrogate, so formal I5 gets its own worlds.
# Pinned on 2026-09-20 before any run on them.
I5_UNITY_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i5_unity.yaml"
I5_UNITY_PARTITIONS_SHA256 = "1102ccab29a654c4aec35266267c81353c6fd64bf5d1d8f824d4976e9744dd21"
I5_UNITY_DOMAIN = "i5_unity"
# Gate I4 world family ACTIVE_INSPECTION_OCCLUDED_V1 (docs/audits/I4_WORLD_FAMILY.md): the unity_gate
# final_test seeds 7800002-7800013 are SPENT (ACTIVE-MCBR-E004 and the formal I4-UNITY run), so the new family
# gets fresh worlds in the free 8 000 000 block. Pinned on 2026-09-20 before any world of the family was built.
I4_OCCLUDED_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i4_occluded.yaml"
I4_OCCLUDED_PARTITIONS_SHA256 = "d929beb67c051ea1116e33ebb8e51cd638f5894ef54989a0a3f2ec3a9221b557"
I4_OCCLUDED_DOMAIN = "i4_occluded"
# Pre-registered REPLICATION of the two I4 criteria that formal run 1 could not resolve (coverage-only and
# information per time/energy). Fresh final worlds; run 1's seeds 8000200-8000219 are SPENT. Pinned on
# 2026-09-20 before any flight on it. Same family, planner, budgets, metric and decision rule as run 1.
I4_OCCLUDED_V2_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i4_occluded_v2.yaml"
I4_OCCLUDED_V2_PARTITIONS_SHA256 = "ad4f97312ea19eaeef47995b2d80299c4b9cd49a59b448c365bb545c5f363fb6"
I4_OCCLUDED_V2_DOMAIN = "i4_occluded_v2"
# MCBR V4 investigation on the same world family. Gate I4 is decided and both of its final splits
# (8000200-8000219 and 8001000-8001059) are SPENT, so a V4 mechanism needs entirely fresh development,
# validation and final worlds. Pinned on 2026-09-20 before any run on any of its seeds; the diagnostic phase
# that declared it read the development split only.
I4_MCBR_V4_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i4_mcbr_v4.yaml"
I4_MCBR_V4_PARTITIONS_SHA256 = "17490d3dbcc85673ebe624e6f04357f42b87a2316d431d2982af58e40e864457"
I4_MCBR_V4_DOMAIN = "i4_mcbr_v4"
# Gate I7 surrogate missions, version 2 (COM-I7-E003/E004): the v1 final seeds 5300000-5300004, and the whole
# mission final_test range they come from, are SPENT by COM-I7-E001/E002, and the 2026-09-20 BAAC scheduler
# repair made that evidence stale. Pinned on 2026-09-20 before any run on its final_test seeds.
I7_V2_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i7_v2.yaml"
I7_V2_PARTITIONS_SHA256 = "030fb08270ff71315d28a67fe52fee1b326f53ecd668d41dbe3cb5b77cf151c7"
I7_V2_DOMAIN = "i7_mission_v2"
# Gate I7 surrogate missions, version 3 (COM-I7-E005/E006): the v2 final seeds 5500000-5500004 are SPENT by
# COM-I7-E003/E004, and the I7-OUTAGE-CRITICAL outage construction changed afterwards (the fixed window could
# miss the finding it stresses). Pinned on 2026-09-20 before any run on its final_test seeds.
I7_V3_PARTITIONS_PATH = REPO_ROOT / "configs" / "eval" / "partitions_i7_v3.yaml"
I7_V3_PARTITIONS_SHA256 = "4ac2982c26ea64aba6e6d950bedde8b2de0c11991975bc3376a5768d1b80b3a9"
I7_V3_DOMAIN = "i7_mission_v3"


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


def load_i5_v3(path: Path = I5_V3_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The I5 v3 partition; disjoint from every other partition file and the reserved ranges (see validate)."""
    digest = canonical_digest(path)
    if verify_digest and digest != I5_V3_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I5_V3_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i5_v3(
        raw,
        load()["raw"],
        load_nav()["raw"],
        load_i5()["raw"],
        load_i5_v2()["raw"],
        load_unity_gates()["raw"],
    )
    return {"raw": raw, "digest": digest}


def validate_i5_v3(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    i5: dict[str, Any],
    i5_v2: dict[str, Any],
    unity: dict[str, Any],
) -> None:
    """final_test must be disjoint from every seed anywhere; development may only equal v1/v2 development."""
    seeds = raw["world_seeds"]
    dev, final = set(_seeds(seeds["development"])), set(_seeds(seeds["final_test"]))
    if dev & final:
        raise PartitionIntegrityError("i5_v3: final_test and development seeds overlap")
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
    older_dev = set(_seeds(i5["world_seeds"]["development"])) & set(
        _seeds(i5_v2["world_seeds"]["development"])
    )
    older_final = {
        s
        for older in (i5, i5_v2)
        for p, spec in older["world_seeds"].items()
        if p != "development"
        for s in _seeds(spec)
    }
    if final & (taken | older_dev | older_final):
        raise PartitionIntegrityError("i5_v3 final_test seeds collide with seeds used by another partition")
    if dev & (taken | older_final) or not dev <= older_dev:
        raise PartitionIntegrityError(
            "i5_v3 development seeds must be v1/v2 development seeds and nothing else"
        )


def load_i5_v4(path: Path = I5_V4_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The I5 v4 partition; disjoint from every other partition file and the reserved ranges (see validate)."""
    digest = canonical_digest(path)
    if verify_digest and digest != I5_V4_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I5_V4_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i5_v4(
        raw,
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"]],
        load_i5_unity()["raw"],
        load_unity_gates()["raw"],
    )
    return {"raw": raw, "digest": digest}


def validate_i5_v4(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    older: list[dict[str, Any]],
    i5_unity: dict[str, Any],
    unity: dict[str, Any],
) -> None:
    """final_test must be disjoint from every seed anywhere; development may only equal v1/v2/v3 development."""
    seeds = raw["world_seeds"]
    dev, final = set(_seeds(seeds["development"])), set(_seeds(seeds["final_test"]))
    if dev & final:
        raise PartitionIntegrityError("i5_v4: final_test and development seeds overlap")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in unity["world_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in i5_unity["world_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    older_dev = set.intersection(*(set(_seeds(f["world_seeds"]["development"])) for f in older))
    older_final = {
        s for f in older for p, spec in f["world_seeds"].items() if p != "development" for s in _seeds(spec)
    }
    if final & (taken | older_dev | older_final):
        raise PartitionIntegrityError("i5_v4 final_test seeds collide with seeds used by another partition")
    if dev & (taken | older_final) or not dev <= older_dev:
        raise PartitionIntegrityError(
            "i5_v4 development seeds must be v1/v2/v3 development seeds and nothing else"
        )


def _i5_v4_split(part: Partition) -> Split:
    loaded = load_i5_v4()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i5_v4 partition has no {part.value!r} split")
    return Split(
        domain=I5_V4_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["scenarios"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def _i5_v3_split(part: Partition) -> Split:
    loaded = load_i5_v3()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i5_v3 partition has no {part.value!r} split")
    return Split(
        domain=I5_V3_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["scenarios"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i5_unity(path: Path = I5_UNITY_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The formal Unity gate I5 worlds; disjoint from every other partition file and reserved range."""
    digest = canonical_digest(path)
    if verify_digest and digest != I5_UNITY_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I5_UNITY_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i5_unity(
        raw,
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"]],
        load_unity_gates()["raw"],
    )
    return {"raw": raw, "digest": digest}


def validate_i5_unity(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    i5_files: list[dict[str, Any]],
    unity: dict[str, Any],
) -> None:
    """Every Unity I5 world must be fresh: no other partition and no reserved range may contain it."""
    seeds = raw["world_seeds"]
    dev, final = set(_seeds(seeds["development"])), set(_seeds(seeds["final_test"]))
    if dev & final:
        raise PartitionIntegrityError("i5_unity: final_test and development seeds overlap")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in unity["world_seeds"].values() for s in _seeds(part)}
    taken |= {s for f in i5_files for spec in f["world_seeds"].values() for s in _seeds(spec)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    if (dev | final) & taken:
        raise PartitionIntegrityError("i5_unity seeds collide with seeds used by another partition")


def _i5_unity_split(part: Partition) -> Split:
    loaded = load_i5_unity()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i5_unity partition has no {part.value!r} split")
    return Split(
        domain=I5_UNITY_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["families"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i4_occluded(
    path: Path = I4_OCCLUDED_PARTITIONS_PATH, *, verify_digest: bool = True
) -> dict[str, Any]:
    """The gate I4 occluded-family partition; disjoint from every other partition file and reserved range."""
    digest = canonical_digest(path)
    if verify_digest and digest != I4_OCCLUDED_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I4_OCCLUDED_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i4_occluded(
        raw,
        load()["raw"],
        load_nav()["raw"],
        load_i5()["raw"],
        load_unity_gates()["raw"],
        load_i5_v2()["raw"],
    )
    return {"raw": raw, "digest": digest}


def validate_i4_occluded(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    i5: dict[str, Any],
    unity: dict[str, Any],
    i5_v2: dict[str, Any],
) -> None:
    """Every split is pairwise disjoint, and no seed collides with another partition file or reserved range."""
    seeds = raw["world_seeds"]
    by_part = {p: set(_seeds(seeds[p.value])) for p in Partition if p.value in seeds}
    for a in by_part:
        for b in by_part:
            if a is not b and by_part[a] & by_part[b]:
                raise PartitionIntegrityError(f"i4_occluded: {a.value} and {b.value} seeds overlap")
    if set(raw["families"]) & set(raw["ood_families"]):
        raise PartitionIntegrityError("i4_occluded: OOD families overlap in-distribution families")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in i5["world_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in i5_v2["world_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in unity["world_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    mine = set().union(*by_part.values()) if by_part else set()
    if mine & taken:
        raise PartitionIntegrityError(
            "i4_occluded seeds collide with seeds used by another partition or experiment"
        )


def _i4_occluded_split(part: Partition) -> Split:
    loaded = load_i4_occluded()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i4_occluded partition has no {part.value!r} split")
    fams = raw["ood_families"] if part is Partition.OOD_TEST else raw["families"]
    return Split(
        domain=I4_OCCLUDED_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(fams),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i7_v2(path: Path = I7_V2_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The gate I7 surrogate partition v2; disjoint from every other partition file and reserved range."""
    digest = canonical_digest(path)
    if verify_digest and digest != I7_V2_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I7_V2_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i7_v2(
        raw,
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"], load_i5_unity()["raw"]],
        load_unity_gates()["raw"],
        load_i4_occluded()["raw"],
    )
    return {"raw": raw, "digest": digest}


def validate_i7_v2(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    i5_files: list[dict[str, Any]],
    unity: dict[str, Any],
    i4_occluded: dict[str, Any],
) -> None:
    """Every I7 v2 seed must be fresh: no other partition and no reserved range may contain it."""
    by_part = {p: set(_seeds(spec)) for p, spec in raw["world_seeds"].items()}
    for a in by_part:
        for b in by_part:
            if a != b and by_part[a] & by_part[b]:
                raise PartitionIntegrityError(f"i7_v2: {a} and {b} seeds overlap")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for f in i5_files for spec in f["world_seeds"].values() for s in _seeds(spec)}
    taken |= {s for part in unity["world_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in i4_occluded["world_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    mine = set().union(*by_part.values()) if by_part else set()
    if mine & taken:
        raise PartitionIntegrityError("i7_v2 seeds collide with seeds used by another partition")


def _i7_v2_split(part: Partition) -> Split:
    loaded = load_i7_v2()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i7_v2 partition has no {part.value!r} split")
    return Split(
        domain=I7_V2_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["scenarios"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i4_occluded_v2(
    path: Path = I4_OCCLUDED_V2_PARTITIONS_PATH, *, verify_digest: bool = True
) -> dict[str, Any]:
    """The I4 replication partition; disjoint from every other partition file and reserved range."""
    digest = canonical_digest(path)
    if verify_digest and digest != I4_OCCLUDED_V2_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I4_OCCLUDED_V2_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i4_occluded_v2(raw, load_i4_occluded()["raw"])
    return {"raw": raw, "digest": digest}


def validate_i4_occluded_v2(raw: dict[str, Any], v1: dict[str, Any]) -> None:
    """Replication seeds are fresh: disjoint from every v1 split and from every reserved range."""
    if raw.get("replicates_digest") != I4_OCCLUDED_PARTITIONS_SHA256:
        raise PartitionIntegrityError("i4_occluded_v2 must name the v1 partition digest it replicates")
    mine = set(_seeds(raw["world_seeds"]["final_test"]))
    if not mine:
        raise PartitionIntegrityError("i4_occluded_v2 has no final_test seeds")
    taken = {s for spec in v1["world_seeds"].values() for s in _seeds(spec)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    if mine & taken:
        raise PartitionIntegrityError(
            "i4_occluded_v2 seeds collide with seeds used by another partition or experiment"
        )


def _i4_occluded_v2_split(part: Partition) -> Split:
    loaded = load_i4_occluded_v2()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i4_occluded_v2 partition has no {part.value!r} split")
    return Split(
        domain=I4_OCCLUDED_V2_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(raw["families"]),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i4_mcbr_v4(path: Path = I4_MCBR_V4_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The MCBR V4 partition; every seed is fresh and disjoint from every other file and reserved range."""
    digest = canonical_digest(path)
    if verify_digest and digest != I4_MCBR_V4_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I4_MCBR_V4_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i4_mcbr_v4(
        raw,
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"], load_i5_unity()["raw"]],
        load_unity_gates()["raw"],
        [load_i4_occluded()["raw"], load_i4_occluded_v2()["raw"]],
        [load_i7_v2()["raw"], load_i7_v3()["raw"]],
    )
    return {"raw": raw, "digest": digest}


def validate_i4_mcbr_v4(
    raw: dict[str, Any],
    main: dict[str, Any],
    nav: dict[str, Any],
    i5_files: list[dict[str, Any]],
    unity: dict[str, Any],
    i4_files: list[dict[str, Any]],
    i7_files: list[dict[str, Any]],
) -> None:
    """Splits are pairwise disjoint, families do not overlap the OOD families, and no seed is reused."""
    seeds = raw["world_seeds"]
    by_part = {p: set(_seeds(seeds[p.value])) for p in Partition if p.value in seeds}
    missing = [p.value for p in Partition if p.value not in seeds]
    if missing:
        raise PartitionIntegrityError(f"i4_mcbr_v4 is missing splits: {missing}")
    for a in by_part:
        for b in by_part:
            if a is not b and by_part[a] & by_part[b]:
                raise PartitionIntegrityError(f"i4_mcbr_v4: {a.value} and {b.value} seeds overlap")
    if set(raw["families"]) & set(raw["ood_families"]):
        raise PartitionIntegrityError("i4_mcbr_v4: OOD families overlap in-distribution families")
    taken = {
        s
        for domain in ("abstract", "mission")
        for p in Partition
        for s in _seeds(main[domain]["world_seeds"][p.value])
    }
    taken |= {s for part in nav["noise_seeds"].values() for s in _seeds(part)}
    taken |= {s for part in unity["world_seeds"].values() for s in _seeds(part)}
    for other in (*i5_files, *i4_files, *i7_files):
        taken |= {s for part in other["world_seeds"].values() for s in _seeds(part)}
    for r in raw.get("reserved_elsewhere", []):
        taken |= set(_seeds(r))
    mine = set().union(*by_part.values()) if by_part else set()
    if mine & taken:
        raise PartitionIntegrityError(
            "i4_mcbr_v4 seeds collide with seeds used by another partition or experiment"
        )


def _i4_mcbr_v4_split(part: Partition) -> Split:
    loaded = load_i4_mcbr_v4()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i4_mcbr_v4 partition has no {part.value!r} split")
    fams = raw["ood_families"] if part is Partition.OOD_TEST else raw["families"]
    return Split(
        domain=I4_MCBR_V4_DOMAIN,
        partition=part,
        world_seeds=_seeds(raw["world_seeds"][part.value]),
        families=tuple(fams),
        replicates_per_world=1,
        digest=loaded["digest"],
    )


def load_i7_v3(path: Path = I7_V3_PARTITIONS_PATH, *, verify_digest: bool = True) -> dict[str, Any]:
    """The gate I7 surrogate partition v3; disjoint from every other partition file and reserved range."""
    digest = canonical_digest(path)
    if verify_digest and digest != I7_V3_PARTITIONS_SHA256:
        raise PartitionIntegrityError(
            f"{path} changed after freezing: digest {digest} != pinned {I7_V3_PARTITIONS_SHA256}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_i7_v2(
        raw,
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"], load_i5_unity()["raw"]],
        load_unity_gates()["raw"],
        load_i4_occluded()["raw"],
    )
    v2 = {s for spec in load_i7_v2()["raw"]["world_seeds"].values() for s in _seeds(spec)}
    mine = {s for spec in raw["world_seeds"].values() for s in _seeds(spec)}
    if mine & v2:
        raise PartitionIntegrityError("i7_v3 seeds collide with the SPENT i7_v2 seeds")
    return {"raw": raw, "digest": digest}


def _i7_v3_split(part: Partition) -> Split:
    loaded = load_i7_v3()
    raw = loaded["raw"]
    if part.value not in raw["world_seeds"]:
        raise KeyError(f"i7_v3 partition has no {part.value!r} split")
    return Split(
        domain=I7_V3_DOMAIN,
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
    if domain == I5_V3_DOMAIN:
        return _i5_v3_split(part)
    if domain == I5_V4_DOMAIN:
        return _i5_v4_split(part)
    if domain == I5_UNITY_DOMAIN:
        return _i5_unity_split(part)
    if domain == I4_OCCLUDED_DOMAIN:
        return _i4_occluded_split(part)
    if domain == I4_OCCLUDED_V2_DOMAIN:
        return _i4_occluded_v2_split(part)
    if domain == I4_MCBR_V4_DOMAIN:
        return _i4_mcbr_v4_split(part)
    if domain == I7_V2_DOMAIN:
        return _i7_v2_split(part)
    if domain == I7_V3_DOMAIN:
        return _i7_v3_split(part)
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
    if domain == I4_OCCLUDED_DOMAIN:
        ws = load_i4_occluded()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in ws.items() if int(seed) in _seeds(spec)), None)
    if domain == I4_MCBR_V4_DOMAIN:
        v4 = load_i4_mcbr_v4()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in v4.items() if int(seed) in _seeds(spec)), None)
    if domain == I5_DOMAIN:
        i5 = load_i5()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in i5.items() if int(seed) in _seeds(spec)), None)
    if domain == I5_V2_DOMAIN:
        v2 = load_i5_v2()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in v2.items() if int(seed) in _seeds(spec)), None)
    if domain == I5_V3_DOMAIN:
        v3 = load_i5_v3()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in v3.items() if int(seed) in _seeds(spec)), None)
    if domain == I5_V4_DOMAIN:
        v4 = load_i5_v4()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in v4.items() if int(seed) in _seeds(spec)), None)
    if domain == I7_V2_DOMAIN:
        i7 = load_i7_v2()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in i7.items() if int(seed) in _seeds(spec)), None)
    if domain == I7_V3_DOMAIN:
        i7v3 = load_i7_v3()["raw"]["world_seeds"]
        return next((Partition(p) for p, spec in i7v3.items() if int(seed) in _seeds(spec)), None)
    raw = load()["raw"]
    for p in Partition:
        if int(seed) in _seeds(raw[domain]["world_seeds"][p.value]):
            return p
    return None


def contamination_label(seed: int) -> str | None:
    raw = load()["raw"]
    return next((str(c["label"]) for c in raw.get("contaminated", []) if int(c["seed"]) == int(seed)), None)
