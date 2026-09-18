# BAAC: bandwidth-aware communication

`conrad/communication` sends belief changes over constrained links. The sender keeps a model of what the receiver
knows and sends semantic deltas at progressive fidelity.

## Sender pipeline (`baac.py`, `BAACSender`)

1. `offer(BeliefMessage, mission_value, now, evidence_sizes)` builds an `InformationUnit` with deltas against the
   receiver model (`UnitBuilder`) and puts it in the `PersistentQueue`.
2. `step(now_s, dt_s, sink)`: on link recovery, `reevaluate` drops expired non-critical units and rebuilds deltas;
   `schedule` picks increments; `ChannelSim.transmit` sends them; ACKed increments update `ReceiverKnowledge`.
3. `offer_evidence`, `handle_resync`, `state() -> CommunicationState`.

## Units, fidelity and deltas

- Fidelity levels (`conrad/schemas/comms.py`): F0_CRITICAL_ALERT (critical units only), F1_STRUCTURED_BELIEF,
  F2_EVIDENCE_SUMMARY, F3_COMPRESSED_EVIDENCE, F4_RAW_EVIDENCE. Sizes are measured from canonical JSON length plus
  evidence bytes. `information_retained` defaults (0.3, 0.7, 0.85, 0.95, 1.0) are ENGINEERING_ESTIMATE.
- Delta types (`delta.py`): NEW_BELIEF, STATE_CHANGED, UNCERTAINTY_CHANGED, CONTRADICTION_ADDED, EVIDENCE_ADDED,
  RELATIONSHIP_CHANGED, BELIEF_RETIRED. A missing base revision raises `ResyncRequired`.
- Receiver (`receiver.py`): `ReceiverKnowledge` (sender side, advances only on ACK) and `ReceiverStore` (receiver
  side, applies deltas, requests resync).

## Scheduling and critical alerts

- `scheduler.py`: deterministic constrained greedy on `mission_value x retained gain x novelty x
  confidence_adjustment / remaining_bits`, within per-link bandwidth x dt (with expected ARQ overhead), energy,
  deadlines and dependencies. Large increments are fragmented.
- A unit is critical when `mission_value >= 0.8`. Critical units use the lowest-latency link and are never dropped
  on overload or expired on deadline; the queue drops the lowest-value non-critical entry first.
- Policies: `C-B10_baac` (BAAC); baselines C-B0 send all, C-B1 FIFO, C-B2 fixed priority, C-B3 fixed compression,
  C-B4 value per bit. C-B5..C-B9 are not implemented.

## Channel (`channel.py`)

`LinkProfile` (bandwidth, latency, packet loss, bit error rate, energy per bit, ARQ retries, outages, bandwidth
schedule; `source="SYNTHETIC_ONLY"`). `ChannelSim(profiles, seed)` is seeded and packet-level.

## Tests and experiment

```
uv run pytest tests/unit/communication tests/property/communication -q
uv run conrad eval run --experiment COM-BAAC-E001
```

COM-BAAC-E001 (synthetic 20 kbps acoustic link): retained mission information 0.866 / 0.749 / 0.389 / 0.136 /
0.048 / 0 at 100 / 50 / 10 / 1 / 0.1 / 0 % bandwidth. Critical alerts were delivered at 1.0 at every non-zero
budget (latency 1.1 to 74 s), while send-all, FIFO and fixed priority delivered 0 to 0.06 and no alerts. At 1 %
the value-per-bit baseline won (0.166 vs 0.136).

## Limits and OPEN items

- Scheduler, learned value estimator, codec/fidelity predictor and channel predictor are OPEN.
- Channel numbers are SYNTHETIC_ONLY until calibrated against physical links.
- The learned value heads have only a smoke test ([MODEL_CARDS/baac_value_heads.md](MODEL_CARDS/baac_value_heads.md)).
