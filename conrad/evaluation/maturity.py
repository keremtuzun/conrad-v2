"""D0-D8 component maturity ladder (ch25 Definition of Done). "Do not call D1 done."

A subsystem may claim level Dn only when every level D0..Dn has at least one evidence link
whose artifact exists (and, when a digest is recorded, still hashes to it).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path

import yaml
from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.settings import REPO_ROOT


class MaturityLevel(str, Enum):
    D0 = "D0"  # Specified
    D1 = "D1"  # Implemented
    D2 = "D2"  # Unit validated
    D3 = "D3"  # Baseline validated
    D4 = "D4"  # Integrated
    D5 = "D5"  # Synthetic validated
    D6 = "D6"  # Real-data validated
    D7 = "D7"  # Physical validated
    D8 = "D8"  # Research validated

    @property
    def rank(self) -> int:
        return int(self.value[1:])


LEVEL_MEANING = {
    "D0": "Specified",
    "D1": "Implemented",
    "D2": "Unit validated",
    "D3": "Baseline validated",
    "D4": "Integrated",
    "D5": "Synthetic validated",
    "D6": "Real-data validated",
    "D7": "Physical validated",
    "D8": "Research validated",
}


class EvidenceLink(ConradModel):
    level: MaturityLevel
    artifact: str = Field(
        min_length=1, description="repo-relative path (spec section, test file, report, run dir)"
    )
    sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    description: str = Field(min_length=1)


class SubsystemMaturity(ConradModel):
    subsystem: str = Field(min_length=1)
    claimed_level: MaturityLevel
    evidence: tuple[EvidenceLink, ...] = ()


class MaturityFile(ConradModel):
    subsystems: tuple[SubsystemMaturity, ...]


def _link_problem(link: EvidenceLink, root: Path) -> str | None:
    target = root / link.artifact
    if not target.exists():
        return f"{link.level.value}: evidence artifact missing: {link.artifact}"
    if link.sha256 is not None:
        if not target.is_file():
            return f"{link.level.value}: digest recorded for a directory: {link.artifact}"
        if hashlib.sha256(target.read_bytes()).hexdigest() != link.sha256:
            return f"{link.level.value}: evidence artifact changed since it was linked: {link.artifact}"
    return None


def validate_subsystem(entry: SubsystemMaturity, root: Path = REPO_ROOT) -> list[str]:
    problems: list[str] = []
    for link in entry.evidence:
        if link.level.rank > entry.claimed_level.rank:
            continue
        problem = _link_problem(link, root)
        if problem:
            problems.append(f"{entry.subsystem} {problem}")
    for rank in range(entry.claimed_level.rank + 1):
        level = MaturityLevel(f"D{rank}")
        valid = [e for e in entry.evidence if e.level is level and _link_problem(e, root) is None]
        if not valid:
            problems.append(
                f"{entry.subsystem}: claims {entry.claimed_level.value} but has no valid {level.value} evidence"
            )
    return problems


def supported_level(entry: SubsystemMaturity, root: Path = REPO_ROOT) -> MaturityLevel | None:
    """Highest level whose whole ladder below it is evidenced; None when not even D0 is."""
    best: MaturityLevel | None = None
    for rank in range(9):
        level = MaturityLevel(f"D{rank}")
        if not any(e.level is level and _link_problem(e, root) is None for e in entry.evidence):
            break
        best = level
    return best


def load_maturity_file(path: str | Path, root: Path = REPO_ROOT) -> MaturityFile:
    """Load and validate; raises ValueError listing every unsupported claim."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tracking = MaturityFile.model_validate(data)
    names = [s.subsystem for s in tracking.subsystems]
    if len(set(names)) != len(names):
        raise ValueError("duplicate subsystem in maturity file")
    problems = [p for s in tracking.subsystems for p in validate_subsystem(s, root)]
    if problems:
        raise ValueError("maturity claims without evidence: " + "; ".join(problems))
    return tracking


def dump_maturity_file(tracking: MaturityFile, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(tracking.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
