"""Append-only experiment registry (ch26 Experiment discipline, ch27 Registry rules).

Every entry records code + config + data + seed + checkpoint + metrics. Entries are hash-chained
JSONL lines; failed and inconclusive experiments stay recorded. Nothing is ever rewritten.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from conrad.evaluation.claims import ClaimLevel
from conrad.evaluation.metrics.bootstrap import PairedComparison
from conrad.schemas.base import ConradModel, digest_of

EXPERIMENT_ID_PATTERN = (
    r"^(CORE|2S|2T|2E|M1|ACTIVE|COM|NAV|SYS|XFER|X|ECMER|DATA|ID)-[A-Z0-9]+(-[A-Z0-9.]+)*$"
)
GENESIS = "0" * 64


class Outcome(str, Enum):
    PLANNED = "PLANNED"
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAILED_RUN = "FAILED_RUN"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class Tier(str, Enum):
    T0_SMOKE = "T0_SMOKE"
    T1_DEVELOPMENT = "T1_DEVELOPMENT"
    T2_BENCHMARK = "T2_BENCHMARK"
    T3_RESEARCH = "T3_RESEARCH"
    T4_FINAL = "T4_FINAL"


class ExperimentRecord(ConradModel):
    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    hypothesis_id: str | None = Field(default=None, pattern=r"^H-[A-Z0-9]+-[0-9]{2}$")
    mechanism: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    tier: Tier
    baselines: tuple[str, ...] = ()
    ablations: tuple[str, ...] = ()
    manifest_digests: tuple[str, ...] = ()
    split_hash: str | None = None
    seeds: tuple[int, ...] = ()
    primary_metric: str = Field(min_length=1)
    direction: str = Field(pattern="^(lower|higher)$")
    compute: dict[str, str] = Field(default_factory=dict)
    config_digest: str | None = None
    config_ref: str | None = None
    code_commit: str | None = None
    git_dirty: bool | None = None
    checkpoint_ids: tuple[str, ...] = ()
    result: dict[str, float] = Field(default_factory=dict)
    effect: PairedComparison | None = None
    artifact_paths: tuple[str, ...] = ()
    outcome: Outcome = Outcome.PLANNED
    evidence_level: ClaimLevel = ClaimLevel.NONE
    limitations: tuple[str, ...] = ()
    recorded_time_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def _discipline(self) -> ExperimentRecord:
        if self.outcome is Outcome.PLANNED:
            return self
        missing = [
            name
            for name, value in (
                ("code_commit", self.code_commit),
                ("config_digest", self.config_digest),
                ("seeds", self.seeds),
                ("split_hash", self.split_hash),
            )
            if not value
        ]
        if missing and self.outcome is not Outcome.FAILED_RUN:
            raise ValueError(f"{self.experiment_id}: executed experiment lacks {missing}")
        if self.outcome is Outcome.SUPPORTS:
            if self.git_dirty is not False:
                raise ValueError(f"{self.experiment_id}: SUPPORTS needs clean tracked source")
            if self.effect is None or not self.effect.repeatable_benefit:
                raise ValueError(
                    f"{self.experiment_id}: SUPPORTS needs a paired effect whose CI excludes zero"
                )
            if not self.baselines:
                raise ValueError(f"{self.experiment_id}: SUPPORTS needs at least one baseline")
        elif self.evidence_level.rank > ClaimLevel.IMPLEMENTED.rank:
            raise ValueError(f"{self.experiment_id}: only a SUPPORTS outcome can carry validation evidence")
        return self


class RegistryIntegrityError(RuntimeError):
    pass


class ExperimentRegistry:
    """JSONL file; each line = {"prev": digest, "record": {...}, "digest": digest}."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _lines(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    def verify(self) -> list[str]:
        problems: list[str] = []
        prev = GENESIS
        for number, line in enumerate(self._lines(), start=1):
            if line.get("prev") != prev:
                problems.append(f"line {number}: chain broken")
            expected = digest_of({"prev": line.get("prev"), "record": line.get("record")})
            if line.get("digest") != expected:
                problems.append(f"line {number}: content digest mismatch (edited entry)")
            prev = str(line.get("digest"))
        return problems

    def append(self, record: ExperimentRecord) -> str:
        problems = self.verify()
        if problems:
            raise RegistryIntegrityError("; ".join(problems))
        lines = self._lines()
        prev = str(lines[-1]["digest"]) if lines else GENESIS
        body = record.model_dump(mode="json")
        digest = digest_of({"prev": prev, "record": body})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"prev": prev, "record": body, "digest": digest}, sort_keys=True) + "\n")
        return digest

    def records(self) -> list[ExperimentRecord]:
        return [ExperimentRecord.model_validate(line["record"]) for line in self._lines()]

    def history(self, experiment_id: str) -> list[ExperimentRecord]:
        return [r for r in self.records() if r.experiment_id == experiment_id]

    def latest(self, experiment_id: str) -> ExperimentRecord | None:
        found = self.history(experiment_id)
        return found[-1] if found else None

    def query(
        self,
        *,
        hypothesis_id: str | None = None,
        outcome: Outcome | None = None,
        prefix: str | None = None,
        tier: Tier | None = None,
    ) -> list[ExperimentRecord]:
        out = self.records()
        if hypothesis_id is not None:
            out = [r for r in out if r.hypothesis_id == hypothesis_id]
        if outcome is not None:
            out = [r for r in out if r.outcome is outcome]
        if prefix is not None:
            out = [r for r in out if r.experiment_id.startswith(prefix)]
        if tier is not None:
            out = [r for r in out if r.tier is tier]
        return out
