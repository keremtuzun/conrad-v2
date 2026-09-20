"""Seeded channel simulator (ch16 'Channel simulator', ch19 'Communications simulator').

Bandwidth, latency, packet loss, bit errors, dropout/outage windows, bandwidth schedules, energy per
bit and range. Every number is a SYNTHETIC_ONLY configuration value until calibrated against
physical communication tests. Packet-level ARQ: each packet is retried up to ``max_retries``; a unit
is delivered only if every packet arrives. Randomness comes from one seeded numpy Generator.

implementation_status: EXPERIMENTAL_CANDIDATE (SYNTHETIC_ONLY physics)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.schemas.comms import LinkState, LinkStatus
from conrad.schemas.timebase import stamp


class LinkProfile(ConradModel):
    name: str
    bandwidth_bps: float = Field(ge=0)
    latency_s: float = Field(default=1.0, ge=0)
    packet_loss: float = Field(default=0.0, ge=0, le=1)
    bit_error_rate: float = Field(default=0.0, ge=0, le=1)
    energy_per_bit_j: float = Field(default=1e-6, ge=0)
    packet_bits: int = Field(default=2048, gt=0)
    max_retries: int = Field(default=3, ge=0)
    max_range_m: float | None = Field(default=None, gt=0)
    outages_s: tuple[tuple[float, float], ...] = ()
    outage_follows_critical_finding: bool = Field(
        default=False,
        description="the outage window's END follows the mission's first critical finding instead of being "
        "fixed: each window starts at its declared start and stays DOWN until the finding has been made "
        "plus ``outage_hold_after_finding_s``, capped at ``outage_max_s`` after the start. A fixed window "
        "can miss the event it is meant to stress (gate I7 seed 5500002, finding at 68.1 s against a "
        "[6 s, 60 s) window). The declared END of ``outages_s`` is unused in this mode.",
    )
    outage_hold_after_finding_s: float = Field(
        default=45.0, ge=0, description="link stays down this long after the first critical finding"
    )
    outage_max_s: float = Field(
        default=120.0,
        gt=0,
        description="hard cap on one window, so a mission that never makes a finding still reconnects",
    )
    bandwidth_schedule: tuple[tuple[float, float], ...] = Field(
        default=(), description="(start_s, factor) steps applied to bandwidth_bps"
    )
    degraded_below_fraction: float = Field(default=0.5, ge=0, le=1)
    source: str = "SYNTHETIC_ONLY"


class ChannelResult(ConradModel):
    delivered: bool
    bits_used: int = Field(ge=0)
    energy_j: float = Field(ge=0)
    delivered_time_s: float | None = None
    packets: int
    retransmissions: int


class ChannelSim:
    def __init__(self, profiles: Sequence[LinkProfile], seed: int, clock_domain: str = "SIM") -> None:
        self.profiles = {p.name: p for p in profiles}
        self.rng = np.random.default_rng(seed)
        self.clock_domain = clock_domain
        self.range_m: float | None = None
        self.outage_hold_until_s: float | None = None  # armed by the runtime at the first critical finding

    def hold_outage_until(self, t_s: float) -> None:
        """Arm a finding-following outage (``LinkProfile.outage_follows_critical_finding``).

        Called once, at the first critical finding, with the same value for every arm of the I7 harness, so
        every policy still sees the identical link.
        """
        self.outage_hold_until_s = t_s if self.outage_hold_until_s is None else self.outage_hold_until_s

    def outage_windows(self, name: str) -> tuple[tuple[float, float], ...]:
        """The effective outage windows, after applying a finding-following end."""
        p = self.profiles[name]
        if not p.outage_follows_critical_finding:
            return p.outages_s
        hold = self.outage_hold_until_s
        return tuple(
            (a, min(a + p.outage_max_s, a + p.outage_max_s if hold is None else hold)) for a, _ in p.outages_s
        )

    def bandwidth(self, name: str, t_s: float) -> float:
        p = self.profiles[name]
        if any(a <= t_s < b for a, b in self.outage_windows(name)):
            return 0.0
        if p.max_range_m is not None and self.range_m is not None and self.range_m > p.max_range_m:
            return 0.0
        factor = 1.0
        for start, f in sorted(p.bandwidth_schedule):
            if t_s >= start:
                factor = f
        return p.bandwidth_bps * factor

    def packet_failure(self, name: str) -> float:
        p = self.profiles[name]
        return 1.0 - (1.0 - p.packet_loss) * (1.0 - p.bit_error_rate) ** p.packet_bits

    def expected_bits(self, name: str, bits: int) -> int:
        """Budget reservation: nominal bits inflated by the expected ARQ retransmissions."""
        pf = min(self.packet_failure(name), 0.99)
        return math.ceil(bits / (1.0 - pf))

    def link_state(self, name: str, t_s: float) -> LinkState:
        p = self.profiles[name]
        bw = self.bandwidth(name, t_s)
        if bw <= 0:
            status = LinkStatus.DOWN
        elif bw < p.degraded_below_fraction * p.bandwidth_bps:
            status = LinkStatus.DEGRADED
        else:
            status = LinkStatus.UP
        return LinkState(
            link_name=name,
            timestamp=stamp(t_s, self.clock_domain),
            status=status,
            bandwidth_bps=bw,
            latency_s=p.latency_s,
            packet_loss=p.packet_loss,
            bit_error_rate=p.bit_error_rate,
            energy_per_bit_j=p.energy_per_bit_j,
            range_m=self.range_m,
        )

    def link_states(self, t_s: float) -> list[LinkState]:
        return [self.link_state(n, t_s) for n in sorted(self.profiles)]

    def transmit(self, name: str, bits: int, t_s: float) -> ChannelResult:
        p = self.profiles[name]
        bw = self.bandwidth(name, t_s)
        packets = max(1, math.ceil(bits / p.packet_bits))
        if bw <= 0:
            return ChannelResult(
                delivered=False, bits_used=0, energy_j=0.0, packets=packets, retransmissions=0
            )
        pf = self.packet_failure(name)
        used = 0
        retx = 0
        delivered = True
        for i in range(packets):
            size = p.packet_bits if i < packets - 1 else bits - p.packet_bits * (packets - 1)
            for attempt in range(p.max_retries + 1):
                used += size
                if self.rng.random() >= pf:
                    break
                if attempt == p.max_retries:
                    delivered = False
                retx += 1
            if not delivered:
                break
        return ChannelResult(
            delivered=delivered,
            bits_used=used,
            energy_j=used * p.energy_per_bit_j,
            delivered_time_s=t_s + p.latency_s + used / bw if delivered else None,
            packets=packets,
            retransmissions=retx,
        )
