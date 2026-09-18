"""Readiness ladder R0..R6 and the autonomy activation ladder as an evidence-gated state machine.

ch22 sim-to-real ladder: R0 pure simulation -> R1 SIL -> R2 HIL -> R3 dry bench -> R4 controlled water
-> R5 controlled representative environment -> R6 target operating envelope.
Phase 15 activation order: manual -> hold -> waypoint -> station keep -> pipe follow -> sensing -> 2S
-> 2T -> active -> Model1 -> BAAC.

Rules enforced here:
* a gate can pass only when every earlier gate on the same ladder has passed (no skipping);
* every pass names an evidence artifact that exists; its sha256 is recorded and re-verified on load;
* an autonomy stage additionally needs a minimum readiness level (configurable mapping);
* the ledger is an append-only JSON file; revocation is an entry, never a deletion.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import Field

from conrad.robotics.hardware.characterization.ingest import sha256_file
from conrad.schemas.base import ConradModel


class ReadinessLevel(str, Enum):
    R0_PURE_SIMULATION = "R0_PURE_SIMULATION"
    R1_SOFTWARE_IN_THE_LOOP = "R1_SOFTWARE_IN_THE_LOOP"
    R2_HARDWARE_IN_THE_LOOP = "R2_HARDWARE_IN_THE_LOOP"
    R3_DRY_BENCH = "R3_DRY_BENCH"
    R4_CONTROLLED_WATER = "R4_CONTROLLED_WATER"
    R5_REPRESENTATIVE_ENVIRONMENT = "R5_REPRESENTATIVE_ENVIRONMENT"
    R6_TARGET_ENVELOPE = "R6_TARGET_ENVELOPE"


class AutonomyStage(str, Enum):
    MANUAL = "MANUAL"
    ATTITUDE_DEPTH_HOLD = "ATTITUDE_DEPTH_HOLD"
    WAYPOINT = "WAYPOINT"
    STATION_KEEPING = "STATION_KEEPING"
    PIPELINE_FOLLOWING = "PIPELINE_FOLLOWING"
    SENSING = "SENSING"
    MODEL_2S = "MODEL_2S"
    MODEL_2T = "MODEL_2T"
    ACTIVE_INSPECTION = "ACTIVE_INSPECTION"
    MODEL_1 = "MODEL_1"
    BAAC = "BAAC"


READINESS_ORDER = tuple(ReadinessLevel)
AUTONOMY_ORDER = tuple(AutonomyStage)

# ASSUMPTION (not in the spec): physical autonomy stages need controlled-water readiness; manual
# operation needs a passed dry bench. Configurable through ``GateLedger(min_readiness=...)``.
DEFAULT_MIN_READINESS: dict[AutonomyStage, ReadinessLevel] = dict.fromkeys(
    AutonomyStage, ReadinessLevel.R4_CONTROLLED_WATER
) | {AutonomyStage.MANUAL: ReadinessLevel.R3_DRY_BENCH}


class GateError(ValueError):
    pass


class EntryKind(str, Enum):
    PASS = "PASS"
    REVOKE = "REVOKE"


class LedgerEntry(ConradModel):
    sequence: int = Field(ge=0)
    kind: EntryKind
    ladder: str = Field(pattern="^(readiness|autonomy)$")
    gate: str
    evidence_ref: str | None
    evidence_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    approved_by: str = Field(min_length=1)
    time_ns: int = Field(ge=0, description="injected time; never read from the wall clock here")
    note: str = ""


class GateLedger:
    """Persisted JSON ledger. ``root`` resolves relative evidence paths."""

    def __init__(
        self,
        path: str | Path,
        root: str | Path | None = None,
        min_readiness: dict[AutonomyStage, ReadinessLevel] | None = None,
    ) -> None:
        self._path = Path(path)
        self._root = Path(root) if root is not None else self._path.parent
        self._min = dict(DEFAULT_MIN_READINESS if min_readiness is None else min_readiness)
        self._entries: list[LedgerEntry] = []
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [LedgerEntry.model_validate(e) for e in raw["entries"]]
            self.verify_evidence()

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def _passed(self, ladder: str) -> set[str]:
        state: set[str] = set()
        for e in self._entries:
            if e.ladder != ladder:
                continue
            if e.kind is EntryKind.PASS:
                state.add(e.gate)
            else:
                state.discard(e.gate)
        return state

    def highest_readiness(self) -> ReadinessLevel | None:
        passed = self._passed("readiness")
        best = None
        for level in READINESS_ORDER:
            if level.value not in passed:
                break
            best = level
        return best

    def active_autonomy(self) -> AutonomyStage | None:
        """Highest stage whose chain is passed AND whose readiness requirement still holds."""
        passed = self._passed("autonomy")
        have = self.highest_readiness()
        rank = -1 if have is None else READINESS_ORDER.index(have)
        best = None
        for stage in AUTONOMY_ORDER:
            if stage.value not in passed or READINESS_ORDER.index(self._min[stage]) > rank:
                break
            best = stage
        return best

    def _evidence(self, ref: str) -> str:
        p = Path(ref) if Path(ref).is_absolute() else self._root / ref
        if not p.is_file():
            raise GateError(f"evidence artifact {ref!r} does not exist")
        return sha256_file(p)

    def _append(self, entry: LedgerEntry) -> LedgerEntry:
        self._entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        doc = {
            "ledger_version": "gate_ledger.v1",
            "entries": [e.model_dump(mode="json") for e in self._entries],
        }
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        return entry

    def pass_readiness(
        self, level: ReadinessLevel, evidence_ref: str, approved_by: str, time_ns: int
    ) -> LedgerEntry:
        idx = READINESS_ORDER.index(level)
        passed = self._passed("readiness")
        missing = [lv.value for lv in READINESS_ORDER[:idx] if lv.value not in passed]
        if missing:
            raise GateError(f"cannot pass {level.value}: prerequisites not passed {missing}")
        return self._append(
            LedgerEntry(
                sequence=len(self._entries),
                kind=EntryKind.PASS,
                ladder="readiness",
                gate=level.value,
                evidence_ref=evidence_ref,
                evidence_sha256=self._evidence(evidence_ref),
                approved_by=approved_by,
                time_ns=time_ns,
            )
        )

    def activate_autonomy(
        self, stage: AutonomyStage, evidence_ref: str, approved_by: str, time_ns: int
    ) -> LedgerEntry:
        idx = AUTONOMY_ORDER.index(stage)
        passed = self._passed("autonomy")
        missing = [s.value for s in AUTONOMY_ORDER[:idx] if s.value not in passed]
        if missing:
            raise GateError(f"cannot activate {stage.value}: earlier stages not passed {missing}")
        required = self._min[stage]
        have = self.highest_readiness()
        if have is None or READINESS_ORDER.index(have) < READINESS_ORDER.index(required):
            raise GateError(
                f"{stage.value} needs readiness {required.value}; current {None if have is None else have.value}"
            )
        return self._append(
            LedgerEntry(
                sequence=len(self._entries),
                kind=EntryKind.PASS,
                ladder="autonomy",
                gate=stage.value,
                evidence_ref=evidence_ref,
                evidence_sha256=self._evidence(evidence_ref),
                approved_by=approved_by,
                time_ns=time_ns,
            )
        )

    def revoke(self, ladder: str, gate: str, approved_by: str, time_ns: int, note: str) -> LedgerEntry:
        """Revoking a gate also revokes every later gate on that ladder (they depended on it)."""
        order = [g.value for g in (READINESS_ORDER if ladder == "readiness" else AUTONOMY_ORDER)]
        if gate not in order:
            raise GateError(f"unknown {ladder} gate {gate!r}")
        passed = self._passed(ladder)
        last = None
        for g in order[order.index(gate) :]:
            if g in passed:
                last = self._append(
                    LedgerEntry(
                        sequence=len(self._entries),
                        kind=EntryKind.REVOKE,
                        ladder=ladder,
                        gate=g,
                        evidence_ref=None,
                        approved_by=approved_by,
                        time_ns=time_ns,
                        note=note,
                    )
                )
        if last is None:
            raise GateError(f"{ladder} gate {gate} is not passed; nothing to revoke")
        return last

    def verify_evidence(self) -> None:
        """Every PASS entry's artifact must still exist with the recorded digest."""
        for e in self._entries:
            if (
                e.kind is EntryKind.PASS
                and e.evidence_ref is not None
                and self._evidence(e.evidence_ref) != e.evidence_sha256
            ):
                raise GateError(f"evidence for {e.ladder}:{e.gate} changed since it was recorded")
