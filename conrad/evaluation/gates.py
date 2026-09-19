"""Formal gate registry (spec ch25): criteria, dependencies, and evidence-derived status.

Rules enforced here:
- A gate's official status is DERIVED from evidence records under ``artifacts/gates/<gate>/``; nobody types PASS.
- Only FORMAL evidence (the exact source path, e.g. through Unity for I1-I4) can make a gate PASS.
  SURROGATE evidence (e.g. the Python L1 kernel instead of Unity) is reported separately and never promotes.
- A gate cannot PASS while any upstream gate is not PASS (dependency order, ch25 master graph).
- Every criterion must be PASS for the gate to PASS; an OPEN numerical threshold makes it NOT_EVALUABLE.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.settings import REPO_ROOT

GATES_DIR = REPO_ROOT / "artifacts" / "gates"


class EvidenceClass(str, Enum):
    FORMAL = "FORMAL"
    SURROGATE = "SURROGATE"


class CriterionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NOT_RUN = "NOT_RUN"


class BlockerKind(str, Enum):
    PHYSICAL_EXTERNAL_INPUT = "PHYSICAL_EXTERNAL_INPUT"
    HUMAN_INTERFACE_AUTHORITY = "HUMAN_INTERFACE_AUTHORITY"
    EXECUTION_ENVIRONMENT = "EXECUTION_ENVIRONMENT"
    COMPUTE = "COMPUTE"
    DATA_LICENSE_OR_ACCESS = "DATA_LICENSE_OR_ACCESS"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NOT_RUN = "NOT_RUN"
    BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"


class GateDefinition(ConradModel):
    gate_id: str
    phase: str
    source: str
    criteria: tuple[str, ...]
    requires: tuple[str, ...] = ()
    formal_path: str
    external_blocker: BlockerKind | None = Field(
        default=None, description="set only when formal evidence can never be produced by software alone"
    )


class CriterionResult(ConradModel):
    criterion: str
    status: CriterionStatus
    measured: str = ""


class GateEvidence(ConradModel):
    gate_id: str
    evidence_class: EvidenceClass
    execution_path: str
    git_commit: str
    criteria: tuple[CriterionResult, ...]
    artifacts: tuple[str, ...] = ()
    notes: str = ""


GATES: tuple[GateDefinition, ...] = (
    GateDefinition(
        gate_id="P0",
        phase="0",
        source="ch25 Phase 0 acceptance / Gate P0",
        criteria=(
            "serialization/deserialization",
            "schema version",
            "timestamps",
            "coordinate frames",
            "UUID uniqueness",
            "provenance graph validity",
            "uncertainty contract",
            "RobotConfig validation",
        ),
        formal_path="contract tests on the public schemas",
    ),
    GateDefinition(
        gate_id="I0",
        phase="1",
        source="ch25 Integration Gate I0",
        requires=("P0",),
        criteria=(
            "complete architectural loop executes",
            "timestamps",
            "IDs",
            "frames",
            "provenance",
            "uncertainty",
            "logging",
            "configs",
        ),
        formal_path="fake full system (TEST_FIXTURE)",
    ),
    GateDefinition(
        gate_id="C1",
        phase="2",
        source="ch25 Core Gate C1",
        requires=("I0",),
        criteria=(
            "persistent identity",
            "evidence association",
            "no-match/new entity",
            "credible contradiction handling",
            "decomposed uncertainty",
            "relational inference",
            "temporal prediction/correction",
            "provenance",
            "long sequence persistence",
        ),
        formal_path="Core sandbox experiments + Core contract tests",
    ),
    GateDefinition(
        gate_id="2S-FIRST",
        phase="3",
        source="ch25 2S first gate",
        requires=("C1",),
        criteria=(
            "observed != inferred != unknown",
            "counterfactual worlds produce uncertainty, not confident hallucination",
        ),
        formal_path="Twin2S/OCPWE counterfactual worlds -> Model2S (abstract observations)",
    ),
    GateDefinition(
        gate_id="U0",
        phase="4",
        source="ch25 Unity Gate U0",
        criteria=(
            "stable physics",
            "known command moves robot correctly",
            "sensor frames correct",
            "timestamps correct",
            "reset/replay works",
            "RobotHardwareInterface works",
        ),
        formal_path="compiled Unity V2 player driven through UnityRobotHardware",
    ),
    GateDefinition(
        gate_id="I1",
        phase="5",
        source="ch25 Integration Gate I1",
        requires=("U0", "2S-FIRST"),
        criteria=(
            "robot moves through synthetic world",
            "persistent map",
            "geometry",
            "occupancy",
            "coverage",
            "observed/inferred/unknown",
            "uncertainty",
            "provenance",
            "no Twin truth leakage",
        ),
        formal_path="Twin2S -> Unity -> sensors -> ECMER -> Model2S",
    ),
    GateDefinition(
        gate_id="I2",
        phase="6",
        source="ch25 Integration Gate I2",
        requires=("I1",),
        criteria=(
            "reach waypoint",
            "avoid obstacle",
            "follow pipeline",
            "station keep",
            "uses estimated state",
            "NAV-001..NAV-006",
            "faults reach defined safe states",  # ch26 Phase 6 pass criterion
        ),
        formal_path="estimator -> planner -> trajectory -> controller -> allocator -> RHI on Unity",
    ),
    GateDefinition(
        gate_id="2T",
        phase="7",
        source="ch25 2T Gate (functional part)",
        requires=("C1",),
        criteria=("direct inference works before TCDP gets credit",),
        formal_path="Twin2T -> Model2T experiments",
    ),
    GateDefinition(
        gate_id="2T-TCDP",
        phase="7",
        source="ch25 2T Gate (research part)",
        requires=("2T",),
        criteria=(
            "TCDP improves hidden-state reconstruction vs generic/no propagation without excessive contamination",
        ),
        formal_path="2T-E003 on held-out scenarios",
    ),
    GateDefinition(
        gate_id="I3",
        phase="7",
        source="ch25 Integration Gate I3",
        requires=("I2", "2T"),
        criteria=(
            "Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T",
            "robot sees only partial infrastructure",
            "persistent technical belief",
        ),
        formal_path="Twin2S + Twin2T -> Unity -> ECMER -> Model2S + Model2T",
    ),
    GateDefinition(
        gate_id="I4",
        phase="8",
        source="ch25 Integration Gate I4",
        requires=("I3",),
        criteria=(
            "critical structure partly hidden -> uncertain -> MCBR view -> navigation -> new evidence -> belief improves",
            "beats fixed views on actual hidden-state reconstruction",
            "beats random views",
            "beats coverage-only",
            "beats simple views on information/time/energy",
        ),
        formal_path="closed active inspection through Unity on held-out final-test worlds",
    ),
    GateDefinition(
        gate_id="I5",
        phase="9",
        source="ch25 Integration Gate I5",
        requires=("I4",),
        criteria=(
            "continue",
            "request evidence",
            "replan",
            "change sensing",
            "return",
            "escalate",
            "hard constraints inviolable",
            "actions exercised correctly inside integrated missions",
            # ch26 Phase 9 pass criterion
            "traceable decisions with low measured UIR",
            "competitive mission outcomes vs decision baselines",
        ),
        formal_path="Model1 action-matrix suite on imperfect beliefs + integrated mission",
    ),
    GateDefinition(
        gate_id="2E",
        phase="10",
        source="ch25 2E Gate (functional part)",
        requires=("C1",),
        criteria=("entity model works", "field model works", "persistent inference works"),
        formal_path="Twin2E -> Model2E experiments (uncoupled production path)",
    ),
    GateDefinition(
        gate_id="2E-CEFD",
        phase="10",
        source="ch25 2E Gate (research part)",
        requires=("2E",),
        criteria=("CEFD beats uncoupled baselines without unsupported ecological claims",),
        formal_path="2E-E003 on held-out worlds",
    ),
    GateDefinition(
        gate_id="I6",
        phase="10",
        source="ch25 Integration Gate I6",
        requires=("I5", "2E"),
        criteria=(
            "one mission produces 2S, 2T and 2E beliefs",
            "Model1 reasons across all three via the Belief Bus",
            "children remain authoritative within domains",
        ),
        formal_path="multi-domain mission through Unity",
    ),
    GateDefinition(
        gate_id="I7",
        phase="11",
        source="ch25 Integration Gate I7",
        requires=("I6",),
        criteria=(
            "full mission under constrained bandwidth",
            "full mission under outages",
            "BAAC retains more mission-relevant information than raw/FIFO/fixed-priority",
            "critical latency and sync error compared against baselines",  # ch26 Phase 11 pass criterion
        ),
        formal_path="integrated missions over the channel simulator with baselines",
    ),
    GateDefinition(
        gate_id="I8",
        phase="14",
        source="ch25 Integration Gate I8",
        requires=("I7",),
        criteria=(
            "real compute",
            "real software",
            "simulated vehicle",
            "full mission",
            "compute deadlines",
            "memory limits",
            "frame rates",
            "fault recovery",
            "communication constraints",
        ),
        formal_path="Unity -> real onboard computer -> full stack -> Unity",
        external_blocker=BlockerKind.PHYSICAL_EXTERNAL_INPUT,
    ),
    GateDefinition(
        gate_id="I9",
        phase="15",
        source="ch25 Integration Gate I9",
        requires=("I8",),
        criteria=("architectural loop on the physical robot",),
        formal_path="physical robot",
        external_blocker=BlockerKind.PHYSICAL_EXTERNAL_INPUT,
    ),
)
GATE_BY_ID = {g.gate_id: g for g in GATES}


def _load(gate_id: str, root: Path) -> list[GateEvidence]:
    folder = root / gate_id
    if not folder.exists():
        return []
    return [
        GateEvidence.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(folder.glob("evidence*.json"))
    ]


def _criteria_status(defn: GateDefinition, ev: GateEvidence) -> GateStatus:
    by_name = {c.criterion: c.status for c in ev.criteria}
    statuses = [by_name.get(c, CriterionStatus.NOT_RUN) for c in defn.criteria]
    if any(s is CriterionStatus.FAIL for s in statuses):
        return GateStatus.FAIL
    if any(s is CriterionStatus.NOT_RUN for s in statuses):
        return GateStatus.NOT_RUN
    if any(s is CriterionStatus.NOT_EVALUABLE for s in statuses):
        return GateStatus.NOT_EVALUABLE
    return GateStatus.PASS


class GateReport(ConradModel):
    gate_id: str
    official_status: GateStatus
    formal_status: GateStatus
    surrogate_status: GateStatus
    blocking_upstream: tuple[str, ...] = ()
    evidence_files: tuple[str, ...] = ()


def evaluate_gates(root: Path = GATES_DIR) -> dict[str, GateReport]:
    reports: dict[str, GateReport] = {}
    for defn in GATES:  # GATES is in dependency order
        evidence = _load(defn.gate_id, root)
        formal = [e for e in evidence if e.evidence_class is EvidenceClass.FORMAL]
        surrogate = [e for e in evidence if e.evidence_class is EvidenceClass.SURROGATE]
        formal_status = _criteria_status(defn, formal[-1]) if formal else GateStatus.NOT_RUN
        surrogate_status = _criteria_status(defn, surrogate[-1]) if surrogate else GateStatus.NOT_RUN
        upstream = tuple(r for r in defn.requires if reports[r].official_status is not GateStatus.PASS)
        if defn.external_blocker is not None and not formal:
            official = GateStatus.BLOCKED_EXTERNAL
        elif formal_status is GateStatus.FAIL:
            official = GateStatus.FAIL
        elif upstream:
            official = GateStatus.BLOCKED_UPSTREAM
        else:
            official = formal_status
        reports[defn.gate_id] = GateReport(
            gate_id=defn.gate_id,
            official_status=official,
            formal_status=formal_status,
            surrogate_status=surrogate_status,
            blocking_upstream=upstream,
            evidence_files=tuple(
                str(p.relative_to(root.parent.parent)) if p.is_relative_to(root.parent.parent) else str(p)
                for p in sorted((root / defn.gate_id).glob("evidence*.json"))
            )
            if (root / defn.gate_id).exists()
            else (),
        )
    return reports


def write_evidence(evidence: GateEvidence, root: Path = GATES_DIR) -> Path:
    folder = root / evidence.gate_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"evidence_{evidence.evidence_class.value.lower()}.json"
    path.write_text(json.dumps(evidence.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
