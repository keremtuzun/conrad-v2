"""Gate I6 multi-domain: 2S/2T/2E on one world, typed context, and a 2E crash does not stop 2S/2T."""

from __future__ import annotations

import pytest

from conrad.schemas.events import EventType
from conrad.schemas.world import Domain
from tests.acceptance._runs import short_session


@pytest.fixture(scope="module")
def i6():
    s = short_session("I6-MULTIDOMAIN", 24.0)
    yield s
    s.finish()


def test_three_domains_publish_on_one_bus(i6):
    domains = {r.cell.domain for r in i6.repo.all_revisions()}
    assert domains == {Domain.SPATIAL, Domain.TECHNICAL, Domain.ECOLOGICAL}
    assert not i6.runtime.bus.rejected  # no child wrote another domain's belief


def test_ecological_context_reaches_2t_only_as_context(i6):
    deliveries = i6.runtime.bus.context_deliveries
    eco_to_t = [
        d for d in deliveries if d.source_domain is Domain.ECOLOGICAL and d.target_domain is Domain.TECHNICAL
    ]
    assert eco_to_t and all(d.source_type.value == "CROSS_DOMAIN_CONTEXT" for d in eco_to_t)
    ctx = [
        r
        for r in i6.repo.all_revisions()
        if r.cell.domain is Domain.TECHNICAL and r.update_kind.value == "CONTEXT"
    ]
    assert ctx and all(not r.consumed_evidence_ids for r in ctx)


def test_model2e_crash_leaves_2s_and_2t_running():
    s = short_session("I6-MULTIDOMAIN", 24.0, runtime={"module_faults": {"model2e_crash_at_s": 6.0}})
    try:
        assert "model2e" in s.runtime.runner.failed
        crash = [
            e
            for e in s.log.events
            if e.event_type is EventType.FAULT_DETECTED and e.payload.get("module") == "model2e"
        ]
        assert crash
        t_crash = crash[0].envelope.measurement_time_ns
        later = [r for r in s.repo.all_revisions() if r.measurement_time_ns > t_crash]
        assert {Domain.SPATIAL, Domain.TECHNICAL} <= {r.cell.domain for r in later}
        assert not [r for r in later if r.cell.domain is Domain.ECOLOGICAL]
        assert s.runtime.supervisor.state.value in ("RUNNING", "DEGRADED")
        assert s.runtime.bus.availability(s.world.hardware.now_ns())["ECOLOGICAL"].value == "UNAVAILABLE"
    finally:
        s.finish()
