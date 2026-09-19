"""BAAC: Belief-Aware Adaptive Communication (spec ch16, ch19, ch33).

DECISION PLANE. Model 1 decides communication intent; BAAC decides representation, fidelity, timing
and link. It never determines mission goals and never commands hardware.
"""

from conrad.communication.baac import BAACSender
from conrad.communication.channel import ChannelResult, ChannelSim, LinkProfile
from conrad.communication.config import BAACConfig
from conrad.communication.delta import ResyncRequired, apply_deltas, belief_view, compute_deltas
from conrad.communication.queue import DropRecord, PersistentQueue, QueueEntry
from conrad.communication.receiver import ReceiverKnowledge, ReceiverStore, ResyncRequest
from conrad.communication.scheduler import (
    ALL_POLICIES,
    BAAC_POLICY,
    BASELINE_POLICIES,
    ScheduledIncrement,
    SchedulingPolicy,
    confidence_adjustment,
    novelty,
    policy_by_name,
    schedule,
)
from conrad.communication.units import UnitBuilder, UnitContent, alert_frame, increment_bits, payload_bits

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch16 BAAC", "ch19", "ch33 BAAC exact implementation"],
    "configuration_keys": ["BAACConfig.*", "LinkProfile.*", "SchedulingPolicy.*", "BAACHeadsConfig.*"],
    "assumptions": [
        "information_retained per fidelity level is an ENGINEERING_ESTIMATE",
        "F3/F4 sizes use referenced evidence byte lengths (PayloadRef.byte_length or configured default)",
        "link-layer ARQ success counts as receiver acknowledgement",
        "increments larger than one step are fragmented; a chunk lost after ARQ retries is resent",
        "channel numbers are SYNTHETIC_ONLY until calibrated against physical links",
    ],
    "baselines": [
        "C-B0 send all",
        "C-B1 FIFO",
        "C-B2 fixed priority",
        "C-B3 fixed compression",
        "C-B4 value per bit",
        "C-B10 BAAC",
    ],
    "open": ["C-B5..C-B9 (semantic compression, learned scheduler, ablations) not implemented"],
    "acceptance_tests": ["tests/unit/communication", "tests/property/communication", "COM-BAAC-E001"],
    "claim_status": "EVALUATED",
}

__all__ = [
    "ALL_POLICIES",
    "BAAC_POLICY",
    "BASELINE_POLICIES",
    "IMPLEMENTATION_METADATA",
    "BAACConfig",
    "BAACSender",
    "ChannelResult",
    "ChannelSim",
    "DropRecord",
    "LinkProfile",
    "PersistentQueue",
    "QueueEntry",
    "ReceiverKnowledge",
    "ReceiverStore",
    "ResyncRequest",
    "ResyncRequired",
    "ScheduledIncrement",
    "SchedulingPolicy",
    "UnitBuilder",
    "UnitContent",
    "alert_frame",
    "apply_deltas",
    "belief_view",
    "compute_deltas",
    "confidence_adjustment",
    "increment_bits",
    "novelty",
    "payload_bits",
    "policy_by_name",
    "schedule",
]
