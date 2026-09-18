"""Communication oracle for COM-BAAC-E001 (evaluation only).

Scores what the RECEIVER actually holds against the sender's latest beliefs:
V_comm ~ receiver performance with the delivered information (ch16 'BAAC oracle').
"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from conrad.communication.receiver import ReceiverStore
from conrad.schemas.belief import BeliefMessage

STALE_CREDIT = 0.5


def belief_score(store: ReceiverStore, latest: BeliefMessage, retained: tuple[float, ...]) -> float:
    """Fraction of mission information about ``latest`` available at the receiver (0..1)."""
    bid = latest.belief_id
    view_rev = store.revision(bid)
    alert = store.alerts.get(bid)
    alert_rev = -1 if alert is None else int(alert["revision"])
    score = 0.0
    if view_rev is not None:
        ev = [str(e) for e in latest.evidence_support]
        level = retained[1]
        if ev and all(e in store.known_evidence_ids for e in ev):
            level = retained[2]
        if ev and all(store.evidence_available.get(e) in ("lossy", "raw") for e in ev):
            level = retained[3]
        if ev and all(store.evidence_available.get(e) == "raw" for e in ev):
            level = retained[4]
        score = level if view_rev >= latest.revision else STALE_CREDIT * level
    if alert_rev >= latest.revision:
        score = max(score, retained[0])
    return score


def mission_information_retained(
    store: ReceiverStore,
    latest: Mapping[UUID, BeliefMessage],
    values: Mapping[UUID, float],
    retained: tuple[float, ...],
) -> float:
    total = sum(values[b] for b in latest)
    if total <= 0:
        return 0.0
    return sum(values[b] * belief_score(store, m, retained) for b, m in latest.items()) / total


def sync_error(store: ReceiverStore, latest: Mapping[UUID, BeliefMessage]) -> float:
    """Fraction of beliefs whose receiver revision differs from the sender's latest."""
    if not latest:
        return 0.0
    return sum(store.revision(b) != m.revision for b, m in latest.items()) / len(latest)
