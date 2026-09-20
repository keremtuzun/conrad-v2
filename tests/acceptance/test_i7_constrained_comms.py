"""Gate I7 (ch25): full mission under bandwidth, then outages; BAAC must retain more mission-relevant
information than raw / FIFO / fixed-priority. Reads the stored FINAL artifacts of COM-I7-E003/E004 (python-kernel
SURROGATE missions); a missing artifact FAILS the test, it is never skipped.

E003/E004 replaced E001/E002 on 2026-09-20: the BAAC scheduler repair (bounded ch19 Level 0/1 pre-emption plus
the receiver-relative increment value) changed BAAC's behaviour, so the E001/E002 numbers on the now SPENT
mission final_test seeds 5300000-5300004 are stale. E003/E004 run on the freshly declared, digest-pinned
final_test split of configs/eval/partitions_i7_v2.yaml.
"""

import json
from pathlib import Path

import pytest

from conrad.evaluation.partitions import Partition, partition_of

ROOT = Path(__file__).resolve().parents[2]
E001 = ROOT / "artifacts" / "experiments" / "COM-I7-E003" / "com_i7_e003.json"
E002 = ROOT / "artifacts" / "experiments" / "COM-I7-E004" / "com_i7_e004.json"
SPEC_SET = ("raw", "fifo", "fixed_priority")
POLICIES = ("baac", "raw", "fifo", "fixed_priority", "value_per_bit")


def _load(path):
    if not path.exists():
        pytest.fail(f"I7 artifact missing: {path} (run COM-I7-E001/E002 first)")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def e001():
    return _load(E001)


@pytest.fixture(scope="module")
def e002():
    return _load(E002)


def _shadow(d):
    return [r for r in d["per_run"] if r["run_id"].endswith("-shadow")]


def test_artifacts_are_final_split_surrogate(e001, e002):
    for d in (e001, e002):
        assert d["partition"] == "final_test"
        assert d["evidence_class"].startswith("SURROGATE")
        domain = d.get("partition_domain", "mission")
        assert d["seeds"] and all(partition_of(domain, s) is Partition.FINAL_TEST for s in d["seeds"])


def test_full_mission_under_constrained_bandwidth(e001):
    levels = e001["config"]["bandwidth_levels"]
    assert {1.0, 0.5, 0.1, 0.01, 0.001} <= set(levels)
    runs = _shadow(e001)
    assert len(runs) == len(levels) * len(e001["seeds"])
    full = next(r for r in runs if r["bandwidth_factor"] == 1.0)
    ref = full["link"]["bandwidth_bps"]  # the declared mission link
    for r in runs:
        assert set(r["arms"]) == set(POLICIES)
        assert r["link"]["bandwidth_bps"] == pytest.approx(ref * r["bandwidth_factor"])
        assert r["critical_offers"], f"{r['run_id']}: no critical finding in the mission"
        # every arm saw the identical mission offer stream (shadow arms)
        assert r["offers"] > 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GATE I7 criterion 2 = FAIL on COM-I7-E004 (2026-09-20, seeds 5500000-5500004). The scenario drops "
        "the link over a fixed window [6 s, 60 s) and relies on the lane pass reaching the lane-side defect "
        "inside it. It does on 4 of the 5 worlds (findings at 15.1, 55.1, 16.1 and 17.1 s), but on seed "
        "5500002 the patch first becomes visible at 66.25 s and the finding is made at 68.1 s, after the link "
        "is already back. The outage path is therefore not exercised at all on that world. This is world "
        "geometry against a fixed outage window, not a communication fault; widening the window after seeing "
        "which seed missed would be tuning the scenario to the result, so it is left failing and recorded as "
        "an open declaration question in docs/audits/I7_COMMUNICATIONS.md."
    ),
)
def test_outage_finding_is_created_while_the_link_is_down(e002):
    for r in _shadow(e002):
        (a, b), *_ = r["link"]["outages_s"]
        crit = r["arms"]["baac"]["critical"]
        assert crit, f"{r['run_id']}: no critical finding"
        assert a <= crit[0]["finding_t_s"] < b and crit[0]["link_down_at_finding"]
        # the lane-side patch is not visible before the link drops (truth-side visibility oracle, when recorded)
        vis = r["patch_first_visible_t_s"]
        assert vis is None or vis >= a


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GATE I7 criterion 2 = FAIL on COM-I7-E004 (2026-09-20, seeds 5500000-5500004), same cause as "
        "test_outage_finding_is_created_while_the_link_is_down: on seed 5500002 no critical finding is made "
        "during the outage, so there is nothing to deliver first after reconnection and "
        "findings_during_outage is empty. On the other 4 seeds BAAC delivered the F0 alert and the F1 delta "
        "ahead of every routine delta at both 100 % and 10 %, with critical delta latencies of 98.4, 51.3, "
        "60.1, 11.4, 97.2, 50.4, 98.3 and 49.3 s and 0 duplicate contributions."
    ),
)
def test_critical_delta_is_delivered_first_after_reconnection(e002):
    for r in _shadow(e002):
        rec = r["reconnection"]["baac"][0]
        assert rec["findings_during_outage"]
        assert rec["critical_delivered_first"], f"{r['run_id']}: {rec['first_receipts_after_reconnect'][:4]}"
        assert r["arms"]["baac"]["critical_delta_latency_s"] is not None


def test_no_duplicate_contribution_at_the_receiver(e001, e002):
    for d in (e001, e002):
        for r in _shadow(d):
            for p, arm in r["arms"].items():
                assert arm["duplicate_contributions"] == 0, (r["run_id"], p)
            for s in r["arms"]["baac"]["sync_critical"]:
                assert s["duplicates"] == 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GATE I7 criterion 2 = FAIL on COM-I7-E004 (2026-09-20, seeds 5500000-5500004). At 100 % the "
        "receiver ends at the sender's latest critical revision on 4 of 5 seeds (27/27, 361/361, 28/28, "
        "29/29). On seed 5500001 it ends at 348 of 376: that belief is revised 376 times in 240 s, so its "
        "last offers are younger than the link latency. The receiver does hold revision 46, the revision the "
        "finding was made at during the outage, which is what the Unity harness rule requires; this "
        "surrogate test still demands exact end-of-mission equality and is left failing rather than "
        "relaxed after the fact."
    ),
)
def test_receiver_synchronised_for_critical_beliefs_after_reconnection(e002):
    for r in _shadow(e002):
        if r["bandwidth_factor"] < 1.0:
            continue
        sync = r["arms"]["baac"]["sync_critical"]
        assert sync and all(s["equal"] for s in sync), (r["run_id"], sync)


def test_stale_and_redundant_units_are_coalesced_or_dropped(e002):
    for r in _shadow(e002):
        arm = r["arms"]["baac"]
        assert arm["coalesced"] + sum(arm["drop_reasons"].values()) > 0


def _check_strict(d):
    bad = []
    for r in _shadow(d):
        if r["bandwidth_factor"] <= 0:
            continue
        base = r["arms"]["baac"]["mission_information_retained"]
        for p in SPEC_SET:
            other = r["arms"][p]["mission_information_retained"]
            if not base > other:
                bad.append((r["run_id"], p, base, other))
    return bad


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GATE I7 criterion 3 = FAIL on COM-I7-E003 (2026-09-20, seeds 5500000-5500004). BAAC is strictly "
        "above raw, FIFO and fixed priority on every seed at 100 % (0.626 vs 0.231/0.231/0.163), 50 % "
        "(0.589 vs 0.231/0.231/0.111) and 10 % (0.261 vs 0.231/0.231/0.012, worst seed +0.028). It is NOT "
        "strictly above at 1 % and 0.1 % on seeds 5500001 and 5500002, where every policy scores exactly "
        "0.000: those worlds revise the critical belief 376 and 361 times, 1 % of the link carries 2,880 "
        "bits over the mission while the cheapest F1 delta measures about 5,000, and comm_oracle credits an "
        "F0 alert only while it is not stale. BAAC did deliver the 184-bit alert on both seeds (1.0 s at "
        "1 %, 107.0 and 94.0 s at 0.1 %) where raw, FIFO and fixed priority delivered nothing, but the "
        "criterion is scored on retention and on retention it is a tie. The 10 % regime that the 2026-09-20 "
        "BAAC scheduler repair fixed is a clean win on all 5 seeds. Strict: if this starts passing, update "
        "the gate record."
    ),
)
def test_baac_retains_more_than_raw_fifo_fixed_priority_every_nonzero_level(e001, e002):
    bad = _check_strict(e001) + _check_strict(e002)
    assert not bad, bad


def test_value_per_bit_comparison_is_reported(e001):
    for cond, row in e001["comparisons"].items():
        assert "value_per_bit" in row, cond
