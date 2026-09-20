"""The finding-following outage window (gate I7 criterion 2, 2026-09-20).

A fixed outage window can miss the event it exists to stress: on gate I7 final seed 5500002 the critical
finding was made at 68.1 s against a [6 s, 60 s) window, so that world never exercised the outage path.
The window now starts where it is declared and stays DOWN until the first critical finding has had its
declared hold, capped so a mission that never makes a finding still reconnects.
"""

import pytest

from conrad.communication import ChannelSim, LinkProfile

FIXED = LinkProfile(name="a", bandwidth_bps=1200.0, outages_s=((6.0, 60.0),))
FOLLOWS = FIXED.model_copy(
    update={
        "outage_follows_critical_finding": True,
        "outage_hold_after_finding_s": 45.0,
        "outage_max_s": 120.0,
    }
)


def test_a_fixed_window_can_miss_a_late_finding():
    """The defect this construction repairs, pinned so it cannot come back unnoticed."""
    ch = ChannelSim([FIXED], seed=0)
    assert ch.bandwidth("a", 68.1) > 0.0  # the link is already back when the finding is made


def test_the_window_stays_down_until_the_finding_has_had_its_hold():
    ch = ChannelSim([FOLLOWS], seed=0)
    assert ch.bandwidth("a", 5.9) > 0.0
    assert ch.bandwidth("a", 6.0) == 0.0
    assert ch.bandwidth("a", 68.1) == 0.0, "a late finding must still happen during the outage"
    ch.hold_outage_until(68.1 + 45.0)
    assert ch.bandwidth("a", 68.1) == 0.0
    assert ch.bandwidth("a", 113.0) == 0.0
    assert ch.bandwidth("a", 113.1) > 0.0
    assert ch.outage_windows("a") == ((6.0, pytest.approx(113.1)),)


def test_an_early_finding_reproduces_the_original_window():
    """The 45 s hold is the original declaration's own interval, so worlds that worked are unchanged."""
    ch = ChannelSim([FOLLOWS], seed=0)
    ch.hold_outage_until(15.1 + 45.0)
    assert ch.bandwidth("a", 60.0) == 0.0
    assert ch.bandwidth("a", 60.1) > 0.0
    assert ch.outage_windows("a")[0][1] == pytest.approx(60.1)


def test_the_hold_is_armed_once_so_every_arm_sees_the_same_link():
    ch = ChannelSim([FOLLOWS], seed=0)
    ch.hold_outage_until(60.1)
    ch.hold_outage_until(200.0)
    assert ch.outage_windows("a")[0][1] == pytest.approx(60.1)


def test_a_mission_that_never_makes_a_finding_still_reconnects():
    ch = ChannelSim([FOLLOWS], seed=0)
    assert ch.bandwidth("a", 125.9) == 0.0
    assert ch.bandwidth("a", 126.0) > 0.0  # start + outage_max_s
    assert ch.outage_windows("a") == ((6.0, 126.0),)


def test_a_profile_without_the_mode_is_untouched():
    ch = ChannelSim([FIXED], seed=0)
    ch.hold_outage_until(200.0)
    assert ch.outage_windows("a") == ((6.0, 60.0),)
    assert ch.bandwidth("a", 60.0) > 0.0
