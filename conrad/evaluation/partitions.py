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


def split(domain: str, partition: str | Partition, purpose: str | Purpose) -> Split:
    """The only sanctioned way to obtain evaluation seeds. Raises on a forbidden (purpose, partition)."""
    part = Partition(partition)
    check_access(part, purpose)
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
    raw = load()["raw"]
    for p in Partition:
        if int(seed) in _seeds(raw[domain]["world_seeds"][p.value]):
            return p
    return None


def contamination_label(seed: int) -> str | None:
    raw = load()["raw"]
    return next((str(c["label"]) for c in raw.get("contaminated", []) if int(c["seed"]) == int(seed)), None)
