"""Release lanes and release manifest (ch34 CI/release, ch36 Security/release, prompt s116).

dev: noncomparable development. candidate: reproducible sim/HIL evidence from a clean commit.
physical: a candidate that has passed the required I-gate with physical/HIL evidence. A checkpoint
is never promoted merely because it is newest.
"""

from __future__ import annotations

import hashlib
import subprocess
from enum import Enum
from pathlib import Path

from pydantic import Field

from conrad.persistence import db
from conrad.schemas.base import ARCHITECTURE_ID, SCHEMA_VERSION, STACK_ID, ConradModel
from conrad.settings import REPO_ROOT


class ReleaseLane(str, Enum):
    DEV = "dev"
    CANDIDATE = "candidate"
    PHYSICAL = "physical"


class GateEvidence(ConradModel):
    gate_id: str
    status: str  # PASS | FAIL | NOT_EVALUABLE | BLOCKED_EXTERNAL
    evidence_artifact: str | None = None


class ReleaseManifest(ConradModel):
    lane: ReleaseLane
    git_sha: str
    git_dirty: bool
    lock_digest: str
    migration_revision: str
    schema_versions: dict[str, str]
    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
    unity_adapter_version: str | None
    robot_config_digest: str
    checkpoint_ids: tuple[str, ...] = ()
    gates: tuple[GateEvidence, ...] = ()
    problems: tuple[str, ...] = Field(default=(), description="why the requested lane was refused")

    @property
    def releasable(self) -> bool:
        return not self.problems


PHYSICAL_REQUIRED_GATES = ("I8", "I9")


def git_state(root: Path = REPO_ROOT) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return sha, dirty


def build_release_manifest(
    lane: ReleaseLane,
    robot_config_digest: str,
    gates: list[GateEvidence],
    checkpoint_ids: list[str] | None = None,
    unity_adapter_version: str | None = None,
    git: tuple[str, bool] | None = None,
) -> ReleaseManifest:
    sha, dirty = git or git_state()
    problems: list[str] = []
    if lane is not ReleaseLane.DEV and dirty:
        problems.append("candidate/physical releases require a clean tracked tree")
    if lane is ReleaseLane.CANDIDATE and not any(g.status == "PASS" and g.evidence_artifact for g in gates):
        problems.append("candidate release needs at least one passed gate with an evidence artifact")
    if lane is ReleaseLane.PHYSICAL:
        for gate_id in PHYSICAL_REQUIRED_GATES:
            ev = next((g for g in gates if g.gate_id == gate_id), None)
            if ev is None or ev.status != "PASS" or not ev.evidence_artifact:
                problems.append(f"physical release requires gate {gate_id} PASS with physical/HIL evidence")
    return ReleaseManifest(
        lane=lane,
        git_sha=sha,
        git_dirty=dirty,
        lock_digest=hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest(),
        migration_revision=db.head_revision(),
        schema_versions={"public": SCHEMA_VERSION},
        unity_adapter_version=unity_adapter_version,
        robot_config_digest=robot_config_digest,
        checkpoint_ids=tuple(checkpoint_ids or ()),
        gates=tuple(gates),
        problems=tuple(problems),
    )
