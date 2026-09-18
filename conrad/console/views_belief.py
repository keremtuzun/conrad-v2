"""Belief-plane views: status timelines, uncertainty channels, provenance DAG, estimated pose track.

Everything here is read from the run database and event log. Truth never enters these views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select

from conrad.persistence import db
from conrad.persistence.repository import Repository
from conrad.schemas.events import EventType, RuntimeEvent
from conrad.schemas.observation import Observation
from conrad.schemas.provenance import ProvenanceError, SourceType

DOMAIN_LABEL = {"SPATIAL": "2S spatial", "TECHNICAL": "2T technical", "ECOLOGICAL": "2E ecological"}
CHANNELS = (("U_A", "aleatoric"), ("U_E", "epistemic"), ("U_C", "contradiction"), ("U_O", "observational"))


@dataclass(frozen=True)
class ClaimPoint:
    status: str
    value: str
    units: str
    uncertainty: tuple[float, float, float, float]
    provenance_id: str | None


@dataclass(frozen=True)
class RevisionRow:
    revision: int
    time_ns: int
    seq: int | None
    update_kind: str
    knowledge_status: str
    lifecycle: str
    late: bool
    uncertainty: tuple[float, float, float, float]
    provenance_root: str
    claims: dict[str, ClaimPoint]


@dataclass
class BeliefView:
    belief_id: str
    domain: str
    entity_type: str
    registry_entity_id: str | None
    revisions: list[RevisionRow] = field(default_factory=list)

    @property
    def claim_names(self) -> list[str]:
        return sorted({n for r in self.revisions for n in r.claims})


@dataclass(frozen=True)
class ProvNode:
    record_id: str
    source_type: str
    operation: str
    module: str
    source_ids: tuple[str, ...]
    parents: tuple[str, ...]
    subject_id: str | None


@dataclass
class ProvenanceView:
    nodes: dict[str, ProvNode] = field(default_factory=dict)
    roots: list[tuple[str, str]] = field(default_factory=list)  # (label, record_id)
    observations: dict[str, str] = field(default_factory=dict)  # observation_id -> short description
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PosePoint:
    time_ns: int
    frame: str
    position: tuple[float, float, float]


def _value(v: object) -> str:
    return "none" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def belief_views(repo: Repository | None, events: list[RuntimeEvent]) -> list[BeliefView]:
    if repo is None:
        return []
    seq_of = {
        (str(e.payload.get("belief_id")), int(e.payload.get("revision", -1))): e.sequence
        for e in events
        if e.event_type is EventType.BELIEF_COMMITTED
    }
    views: dict[str, BeliefView] = {}
    for rev in repo.all_revisions():
        cell = rev.cell
        bid = str(rev.belief_id)
        view = views.setdefault(
            bid,
            BeliefView(
                bid,
                DOMAIN_LABEL.get(cell.domain.value, cell.domain.value),
                cell.entity_type,
                str(cell.registry_entity_id) if cell.registry_entity_id else None,
            ),
        )
        claims = {
            c.name: ClaimPoint(
                c.status.value,
                _value(c.value),
                c.units or "",
                c.uncertainty.as_tuple(),
                str(c.provenance_id) if c.provenance_id else None,
            )
            for c in cell.claims
        }
        view.revisions.append(
            RevisionRow(
                rev.revision,
                rev.measurement_time_ns,
                seq_of.get((bid, rev.revision)),
                rev.update_kind.value,
                cell.knowledge_status.value,
                cell.lifecycle.value,
                rev.late,
                cell.uncertainty.as_tuple(),
                str(rev.provenance_root),
                claims,
            )
        )
    for v in views.values():
        v.revisions.sort(key=lambda r: r.revision)
    return list(views.values())


def _command_roots(events: list[RuntimeEvent]) -> list[tuple[str, str]]:
    out = []
    for e in events:
        root = e.payload.get("provenance_root") if e.event_type is EventType.COMMAND_SENT else None
        if root:
            out.append((f"command {e.payload.get('command_id')} (seq {e.sequence})", str(root)))
    return out


def provenance_view(
    repo: Repository | None, beliefs: list[BeliefView], events: list[RuntimeEvent]
) -> ProvenanceView:
    view = ProvenanceView()
    if repo is None:
        return view
    roots: list[tuple[str, str]] = []
    for b in beliefs:
        for r in b.revisions:
            roots.append((f"belief {b.belief_id[:8]} rev {r.revision}", r.provenance_root))
            roots += [
                (f"claim {name} @ {b.belief_id[:8]} rev {r.revision}", c.provenance_id)
                for name, c in r.claims.items()
                if c.provenance_id
            ]
    for e in events:
        pid = e.payload.get("provenance_id")
        if pid and e.event_type in (EventType.DECISION_MADE, EventType.PLAN_PROPOSED):
            roots.append((f"{e.event_type.value.lower()} (seq {e.sequence})", str(pid)))
    roots += _command_roots(events)
    for label, rid in roots:
        try:
            _absorb(repo, view, UUID(rid))
            view.roots.append((label, rid))
        except (ProvenanceError, ValueError) as exc:
            view.errors.append(f"{label}: {exc}")
    for node in view.nodes.values():
        if node.source_type == SourceType.DIRECT_OBSERVATION.value:
            for sid in node.source_ids:
                obs = repo.observation(UUID(sid))
                if obs is not None:
                    view.observations[sid] = f"{obs.modality.value} sensor {str(obs.sensor_id)[:8]}"
    return view


def _absorb(repo: Repository, view: ProvenanceView, root: UUID) -> None:
    """Add ``root``'s closure. Parents already present carry their full closure, so they are skipped."""
    if str(root) in view.nodes:
        return
    rec = repo.provenance_record(root)
    if rec is None:
        raise ProvenanceError(f"dangling provenance reference {root}")
    closure = {root: rec}
    for parent in rec.parent_records:
        if str(parent) not in view.nodes:
            closure.update(repo.provenance_closure(parent))
    for rid, r in closure.items():
        view.nodes[str(rid)] = ProvNode(
            str(rid),
            r.source_type.value,
            r.operation,
            r.module,
            tuple(str(s) for s in r.source_ids),
            tuple(str(p) for p in r.parent_records),
            str(r.subject_id) if r.subject_id else None,
        )


def chain_lines(view: ProvenanceView, root: str, max_lines: int = 200) -> list[tuple[int, str]]:
    """Depth-indented walk from ``root`` down to raw observations (shared parents shown once)."""
    lines: list[tuple[int, str]] = []
    seen: set[str] = set()

    def walk(rid: str, depth: int) -> None:
        if len(lines) >= max_lines:
            return
        node = view.nodes.get(rid)
        if node is None:
            lines.append((depth, f"MISSING record {rid}"))
            return
        if rid in seen:
            lines.append((depth, f"(already shown) {node.source_type} {rid}"))
            return
        seen.add(rid)
        lines.append((depth, f"{node.source_type} {node.operation} [{node.module}] record {rid}"))
        if node.source_type == SourceType.DIRECT_OBSERVATION.value:
            for sid in node.source_ids:
                desc = view.observations.get(sid)
                if desc:
                    lines.append((depth + 1, f"RAW OBSERVATION {sid} ({desc})"))
        for parent in node.parents:
            walk(parent, depth + 1)

    walk(root, 0)
    return lines


def pose_track(repo: Repository | None) -> list[PosePoint]:
    """Estimated robot pose, as stamped on each stored observation (never truth)."""
    if repo is None:
        return []
    stmt = select(db.observations.c.payload_json).order_by(db.observations.c.measurement_time_ns)
    with repo.engine.connect() as conn:
        rows = [r[0] for r in conn.execute(stmt)]
    out = []
    for raw in rows:
        obs = Observation.model_validate_json(raw)
        pose = obs.robot_pose_estimate
        if pose is not None:
            x, y, z = pose.position_m
            out.append(PosePoint(obs.timestamp.time_ns, pose.frame_id, (float(x), float(y), float(z))))
    return out
