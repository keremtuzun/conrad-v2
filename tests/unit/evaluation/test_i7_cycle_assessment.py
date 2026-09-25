from __future__ import annotations

from copy import deepcopy
from typing import Any

from conrad.evaluation.decision_experiments.com_i7 import POLICIES, assess_cycle


def _arm(retained: float) -> dict[str, Any]:
    return {
        "mission_information_retained": retained,
        "duplicate_contributions": 0,
        "critical_alert_latency_s": 1.0,
        "critical_delta_latency_s": 2.0,
        "sync_critical_all_equal": True,
        "coalesced": 1,
        "resync_requests": 0,
        "critical": [],
        "sync_critical": [],
    }


def _row(seed: int, level: float, *, outage: bool) -> dict[str, Any]:
    arms = {policy: _arm(0.2 if policy == "baac" else 0.1) for policy in POLICIES}
    row: dict[str, Any] = {
        "run_id": f"test-s{seed}-bw{level:g}-shadow",
        "seed": seed,
        "bandwidth_factor": level,
        "offers": 1,
        "critical_offers": [["belief", 4, 10.0]],
        "arms": arms,
        "reconnection": {},
    }
    if outage:
        baac = arms["baac"]
        baac["critical"] = [
            {
                "belief_id": "belief",
                "revision": 4,
                "link_down_at_finding": True,
            }
        ]
        baac["sync_critical"] = [{"belief_id": "belief", "receiver_revision": 4}]
        row["reconnection"] = {"baac": [{"critical_delivered_first": True}]}
    return row


def _artifact(scenario: str, levels: set[float]) -> dict[str, Any]:
    return {
        "partition": "validation",
        "partition_domain": "i7_mission_v4",
        "seeds": [1],
        "scenario": scenario,
        "per_run": [_row(1, level, outage=scenario == "I7-OUTAGE-CRITICAL") for level in levels],
    }


def test_i7_cycle_assessment_passes_complete_strict_pair() -> None:
    bandwidth = _artifact("I7-BANDWIDTH", {1.0, 0.5, 0.1, 0.01, 0.001, 0.0})
    outage = _artifact("I7-OUTAGE-CRITICAL", {1.0, 0.1})
    result = assess_cycle(
        bandwidth,
        outage,
        expected_partition="validation",
        expected_domain="i7_mission_v4",
        expected_seeds=[1],
    )
    assert result["ok"]
    assert all(result["criteria"].values())


def test_i7_cycle_assessment_rejects_one_retention_tie() -> None:
    bandwidth = _artifact("I7-BANDWIDTH", {1.0, 0.5, 0.1, 0.01, 0.001, 0.0})
    outage = _artifact("I7-OUTAGE-CRITICAL", {1.0, 0.1})
    broken = deepcopy(bandwidth)
    row = next(r for r in broken["per_run"] if r["bandwidth_factor"] == 0.001)
    row["arms"]["raw"]["mission_information_retained"] = 0.2
    result = assess_cycle(
        broken,
        outage,
        expected_partition="validation",
        expected_domain="i7_mission_v4",
        expected_seeds=[1],
    )
    assert not result["ok"]
    assert not result["criteria"]["strict_retention"]
