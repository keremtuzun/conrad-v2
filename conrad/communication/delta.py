"""Semantic delta encoding between the receiver-known view and the current BeliefMessage (ch16, ch19).

A belief *view* is the plain-data subset of a BeliefMessage that a receiver reconstructs. Deltas carry
revision IDs and changed fields; ``apply_deltas(view(old), compute_deltas(view(old), new)) == view(new)``.
Raw evidence is never inlined; it stays discoverable through the evidence IDs in the view.

implementation_status: FROZEN_CONTRACT (delta types) / deterministic
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from conrad.schemas.belief import BeliefMessage, Lifecycle
from conrad.schemas.comms import DeltaType, SemanticDelta

View = dict[str, Any]

RETIRED_LIFECYCLES = frozenset({Lifecycle.RETIRED.value, Lifecycle.REJECTED.value, Lifecycle.MERGED.value})
_ALWAYS = ("revision", "timestamp_ns", "provenance_refs", "model_version")
_FIELDS: dict[DeltaType, tuple[str, ...]] = {
    DeltaType.STATE_CHANGED: ("state", "knowledge_status", "lifecycle", "world_entity_id"),
    DeltaType.UNCERTAINTY_CHANGED: ("uncertainty",),
    DeltaType.CONTRADICTION_ADDED: ("evidence_conflicts",),
    DeltaType.EVIDENCE_ADDED: ("evidence_support",),
    DeltaType.RELATIONSHIP_CHANGED: ("relationships",),
}


class ResyncRequired(ValueError):
    """The receiver does not hold the delta's base revision; it must request a resync."""

    def __init__(self, belief_id: str, have: int | None, base: int | None) -> None:
        super().__init__(f"belief {belief_id}: receiver has revision {have}, delta needs base {base}")
        self.belief_id = belief_id
        self.have = have
        self.base = base


def belief_view(m: BeliefMessage) -> View:
    return {
        "belief_id": str(m.belief_id),
        "domain": m.domain.value,
        "revision": m.revision,
        "timestamp_ns": m.timestamp.time_ns,
        "world_entity_id": None if m.world_entity_id is None else str(m.world_entity_id),
        "knowledge_status": m.knowledge_status.value,
        "lifecycle": m.lifecycle.value,
        "state": {
            c.name: {
                "value": c.value,
                "units": c.units,
                "status": c.status.value,
                "uncertainty": list(c.uncertainty.as_tuple()),
            }
            for c in m.state_summary
        },
        "uncertainty": list(m.uncertainty.as_tuple()),
        "evidence_support": [str(e) for e in m.evidence_support],
        "evidence_conflicts": [str(e) for e in m.evidence_conflicts],
        "relationships": [str(r) for r in m.relationships],
        "provenance_refs": [str(p) for p in m.provenance_refs],
        "model_version": m.model_version,
    }


def compute_deltas(known: View | None, current: BeliefMessage) -> list[SemanticDelta]:
    new = belief_view(current)
    if known is None:
        return [
            SemanticDelta(
                delta_type=DeltaType.NEW_BELIEF,
                belief_id=current.belief_id,
                base_revision=None,
                new_revision=current.revision,
                changed_fields=new,
            )
        ]
    if int(known["revision"]) >= current.revision:
        return []
    base = int(known["revision"])
    out: list[SemanticDelta] = []
    header = {k: new[k] for k in _ALWAYS}
    if new["lifecycle"] in RETIRED_LIFECYCLES and known["lifecycle"] not in RETIRED_LIFECYCLES:
        out.append(_delta(DeltaType.BELIEF_RETIRED, current, base, {**header, "lifecycle": new["lifecycle"]}))
    for dtype, fields in _FIELDS.items():
        changed = {f: new[f] for f in fields if new[f] != known.get(f)}
        if dtype is DeltaType.STATE_CHANGED and out:
            changed.pop("lifecycle", None)
        if changed:
            out.append(_delta(dtype, current, base, {**header, **changed}))
    if not out:  # revision advanced with no semantic change: carry the header only
        out.append(_delta(DeltaType.STATE_CHANGED, current, base, header))
    return out


def _delta(dtype: DeltaType, m: BeliefMessage, base: int, fields: View) -> SemanticDelta:
    return SemanticDelta(
        delta_type=dtype,
        belief_id=m.belief_id,
        base_revision=base,
        new_revision=m.revision,
        changed_fields=fields,
    )


def apply_deltas(view: View | None, deltas: Sequence[SemanticDelta]) -> View:
    """Apply one revision step atomically. Raises ResyncRequired if the base revision is not held."""
    if not deltas:
        if view is None:
            raise ValueError("no deltas and no base view")
        return dict(view)
    first = deltas[0]
    if any(d.belief_id != first.belief_id or d.new_revision != first.new_revision for d in deltas):
        raise ValueError("deltas of one step must share belief and new revision")
    have = None if view is None else int(view["revision"])
    if first.delta_type is DeltaType.NEW_BELIEF:
        if view is not None and have is not None and have >= first.new_revision:
            return dict(view)  # stale re-send: keep the fresher receiver state
        return dict(first.changed_fields)
    if view is None or have != first.base_revision:
        raise ResyncRequired(str(first.belief_id), have, first.base_revision)
    out = dict(view)
    for d in deltas:
        out.update(d.changed_fields)
    return out
