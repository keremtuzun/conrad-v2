"""Versioned critical summary keeps bounded, honest semantic information."""

from __future__ import annotations

import pytest

from conrad.communication import BAACConfig, ReceiverStore, UnitBuilder
from conrad.communication.units import ALERT_FRAME, SUMMARY_FRAME, critical_summary_frame, increment_bits
from conrad.evaluation.decision_experiments.fixtures import make_belief
from conrad.evaluation.oracle.comm_oracle import belief_score
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp


def test_compact_summary_requires_its_own_model_version() -> None:
    with pytest.raises(ValueError, match="baac-critical-summary-v1"):
        BAACConfig(compact_critical_summary=True)


def test_critical_summary_is_measured_and_retains_stale_condition_information() -> None:
    ids = IdFactory(2026)
    config = BAACConfig(model_version="baac-critical-summary-v1", compact_critical_summary=True)
    first = make_belief(ids, revision=7, properties={"condition": "SEVERE"})
    made = UnitBuilder(ids, config).belief_unit(first, None, 0.95, stamp(0.0, "SIM"))
    assert made is not None
    summary = made[0].increments[0]
    assert summary == {
        "kind": "critical_summary",
        "belief_id": str(first.belief_id),
        "revision": 7,
        "domain": "TECHNICAL",
        "condition": "SEVERE",
        "status": "OBSERVED",
    }
    assert len(critical_summary_frame(summary)) == SUMMARY_FRAME.size == ALERT_FRAME.size == 23
    assert increment_bits(summary) == made[0].unit.fidelity_levels[0].size_bits == 184

    receiver = ReceiverStore()
    receiver.receive(summary)
    retained = config.information_retained
    assert belief_score(receiver, first, retained) == retained[0]
    newer = make_belief(ids, belief_id=first.belief_id, revision=8, properties={"condition": "SEVERE"})
    assert belief_score(receiver, newer, retained) == 0.5 * retained[0]
    assert receiver.revision(first.belief_id) is None  # summary is not a full belief view


def test_historical_bare_alert_stays_current_only() -> None:
    ids = IdFactory(2027)
    old = make_belief(ids, revision=7, properties={"condition": "SEVERE"})
    made = UnitBuilder(ids, BAACConfig()).belief_unit(old, None, 0.95, stamp(0.0, "SIM"))
    assert made is not None
    assert made[0].increments[0]["kind"] == "alert"
    receiver = ReceiverStore()
    receiver.receive(made[0].increments[0])
    newer = make_belief(ids, belief_id=old.belief_id, revision=8, properties={"condition": "SEVERE"})
    assert belief_score(receiver, newer, BAACConfig().information_retained) == 0.0
