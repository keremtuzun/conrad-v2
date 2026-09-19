"""Receiver knowledge K_R(t) (sender-side approximation) and receiver reconciliation (ch16, ch19).

* ``ReceiverKnowledge`` lives on the SENDER. It only advances on acknowledged delivery, so an
  unacknowledged unit keeps its novelty and is retried.
* ``ReceiverStore`` lives on the RECEIVER. It applies deltas, never lets an older revision overwrite
  a newer one, and asks for a resync when a delta's base revision is missing. Evidence IDs stay
  listed so the receiver can request exact raw evidence later (evidence-on-demand).

implementation_status: FROZEN_CONTRACT (semantics) / deterministic
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from conrad.communication.delta import ResyncRequired, View, apply_deltas
from conrad.schemas.base import ConradModel
from conrad.schemas.comms import SemanticDelta


class ResyncRequest(ConradModel):
    belief_id: UUID
    have_revision: int | None
    needed_base: int | None


class ReceiverKnowledge:
    def __init__(self) -> None:
        self.views: dict[UUID, View] = {}
        self.alerted: dict[UUID, int] = {}
        self.evidence: set[str] = set()

    def known(self, belief_id: UUID) -> View | None:
        return self.views.get(belief_id)

    def known_revision(self, belief_id: UUID) -> int | None:
        v = self.views.get(belief_id)
        return None if v is None else int(v["revision"])

    def acknowledge(self, increment: dict[str, Any]) -> None:
        """Mirror what the receiver now holds after an ACKed increment."""
        kind = increment.get("kind")
        if kind == "alert":
            bid = UUID(str(increment["belief_id"]))
            self.alerted[bid] = max(self.alerted.get(bid, -1), int(increment["revision"]))
        elif kind == "deltas":
            deltas = [SemanticDelta.model_validate(d) for d in increment["deltas"]]
            if deltas:
                bid = deltas[0].belief_id
                try:
                    self.views[bid] = apply_deltas(self.views.get(bid), deltas)
                except ResyncRequired:
                    self.views.pop(bid, None)  # receiver will ask; next unit carries full state
        elif kind in ("evidence_compressed", "evidence_raw"):
            self.evidence.update(str(e["id"]) for e in increment.get("evidence", []))

    def forget(self, belief_id: UUID) -> None:
        """Resync requested: stop assuming anything about this belief at the receiver."""
        self.views.pop(belief_id, None)


class ReceiverStore:
    def __init__(self) -> None:
        self.views: dict[UUID, View] = {}
        self.alerts: dict[UUID, dict[str, Any]] = {}
        self.evidence_available: dict[str, str] = {}  # evidence id -> best codec received
        self.known_evidence_ids: set[str] = set()
        self.resync_requests: list[ResyncRequest] = []
        self.stale_ignored = 0
        # belief -> revisions whose deltas were APPLIED, in order (one entry = one belief contribution)
        self.applied: dict[UUID, list[int]] = {}

    def receive(self, increment: dict[str, Any]) -> ResyncRequest | None:
        kind = increment.get("kind")
        if kind == "alert":
            bid = UUID(str(increment["belief_id"]))
            prev = self.alerts.get(bid)
            if prev is None or int(prev["revision"]) <= int(increment["revision"]):
                self.alerts[bid] = dict(increment)
            return None
        if kind == "deltas":
            return self._apply([SemanticDelta.model_validate(d) for d in increment["deltas"]])
        if kind == "evidence_summary":
            self.known_evidence_ids.update(str(e) for e in increment.get("evidence_ids", []))
            return None
        if kind in ("evidence_compressed", "evidence_raw"):
            codec = "raw" if kind == "evidence_raw" else "lossy"
            for e in increment.get("evidence", []):
                if self.evidence_available.get(str(e["id"])) != "raw":
                    self.evidence_available[str(e["id"])] = codec
            return None
        raise ValueError(f"unknown increment kind {kind!r}")

    def _apply(self, deltas: list[SemanticDelta]) -> ResyncRequest | None:
        if not deltas:
            return None
        bid = deltas[0].belief_id
        current = self.views.get(bid)
        if current is not None and int(current["revision"]) >= deltas[0].new_revision:
            self.stale_ignored += 1
            return None
        try:
            view = apply_deltas(current, deltas)
        except ResyncRequired as exc:
            req = ResyncRequest(belief_id=bid, have_revision=exc.have, needed_base=exc.base)
            self.resync_requests.append(req)
            return req
        self.views[bid] = view
        self.applied.setdefault(bid, []).append(int(view["revision"]))
        self.known_evidence_ids.update(view.get("evidence_support", []))
        return None

    def duplicate_contributions(self) -> int:
        """Revisions applied more than once for the same belief (must be 0: retransmissions never double-count)."""
        return sum(len(revs) - len(set(revs)) for revs in self.applied.values())

    def revision(self, belief_id: UUID) -> int | None:
        v = self.views.get(belief_id)
        return None if v is None else int(v["revision"])

    def evidence_to_request(self, belief_id: UUID) -> list[UUID]:
        """IDs of supporting evidence the receiver knows exists but has not received in raw form."""
        v = self.views.get(belief_id)
        if v is None:
            return []
        return [UUID(e) for e in v.get("evidence_support", []) if self.evidence_available.get(e) != "raw"]
