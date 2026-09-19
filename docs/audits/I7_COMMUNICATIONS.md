# Gate I7: constrained communications (audit, 2026-09-19)

Spec I7 (ch25): "Run full mission under bandwidth. Then outages. BAAC must retain more mission-relevant
information than raw/FIFO/fixed-priority approaches."

**Evidence class: SURROGATE.** Every number below comes from integrated missions on the python L1 kernel
(`conrad.sim.mission`), not Unity. The surrogate evidence is in `artifacts/gates/I7/evidence_surrogate.json`
and cannot promote the gate. The formal run through Unity still has to come from the integrator. All link
numbers are SYNTHETIC_ONLY.

## Why the earlier I7 runs failed

`I7-COMMS-OUTAGE` and `INT-010` put the defect on the far (+Y) side. In those runs MCBR returned
NOT_WORTH_COST, the target stayed UNKNOWN, and no critical finding was made while the link was down. So the
critical-finding-during-outage path was never exercised.

## What changed

| Area | Change |
|---|---|
| `conrad/communication/config.py` | `BAACConfig.scheduler_policy`: C-B10_baac (default) or baseline C-B0..C-B4. `BAACSender` resolves it via `policy_by_name`. |
| `conrad/communication/scheduler.py` | `critical_first` (BAAC only): a critical unit's F0 alert and F1 belief delta pre-empt other traffic (ch19 Level 0/1). Chunks are packet-granular, and sub-packet capacity carries over (`carry_bits`). `ALL_POLICIES`, `policy_by_name`. |
| `conrad/communication/baac.py` | Carries capacity only while data is queued on an UP link, capped at one packet's reservation. Delivery time is not charged twice for carried capacity. Fresher revisions of a belief whose increment is mid-fragment are deferred until the increment finishes. Without this, superseding threw away delivered fragments and a belief revised faster than one F1 could cross the link never arrived (seen on dev seed 5100000 at 10 %). `coalesced` counter. |
| `conrad/communication/units.py` | F0 alert is a packed 23-byte frame (`ALERT_FRAME`: UUID, revision, domain, delta bitmask) = 184 bits. F1..F4 sizes are measured as DEFLATE(canonical JSON) (`increment_bits`). |
| `conrad/communication/receiver.py` | `ReceiverStore.applied` logs every applied revision per belief. `duplicate_contributions()` counts double-applied revisions. |
| `conrad/orchestration/comms.py` | `ShoreLink` owns a primary arm plus optional `baac.shadow_arms`. Each shadow arm has its own sender, channel and receiver, gets the identical offer stream at identical times over the same link profile and channel seed, and never feeds back. `harness_report()` provides per-arm traces. |
| `conrad/sim/mission/scenarios.py` | `I7-BANDWIDTH` and `I7-OUTAGE-CRITICAL` (see below). |
| `conrad/evaluation/decision_experiments/com_i7.py` | COM-I7-E001 / E002 runner and scorer. |
| `conrad/evaluation/dispatch.py` | Entries `COM-I7-E001`, `COM-I7-E002`. |
| `configs/eval/com_i7_e001.yaml`, `com_i7_e002.yaml` | FINAL configs. `com_i7_dev.yaml` and `com_i7_devbw.yaml` are the development design runs. |
| `tests/unit/communication/test_i7_harness.py` | Policy selection, pre-emption after reconnection, FIFO head-of-line, slow-link carry, no double count. |
| `tests/acceptance/test_i7_constrained_comms.py` | Gate acceptance from stored FINAL artifacts. Fails if they are missing. |
| `scripts/record_gate_evidence.py` | `SURROGATE_PLAN["I7"]` with the three `gates.py` criteria names. |

## Harness design

- **Same mission for every policy.** BAAC is the primary arm and drives the mission. Raw (C-B0), FIFO (C-B1),
  fixed priority (C-B2) and value-per-bit (C-B4) are shadow arms on the same `ShoreLink`. They see the same
  offers, the same link profile and the same channel RNG seed.
- **Closed-loop check.** E002 also runs each baseline as the primary at 100 %.
- **Reference link.** The declared mission link, unchanged: 1200 bps, 1 s latency, 5 % packet loss, BER 1e-6,
  1024-bit packets. Levels are 100 / 50 / 10 / 1 / 0.1 % of it, plus 0 %.
- **Mission.** 240 s (the default is 120 s). At 0.1 % (1.2 bps) a 120 s mission carries 144 bits, less than
  one 184-bit alert.
- **Deterministic critical finding.** Both I7 scenarios move the defect to the lane side (`defect.side=near`),
  so the lane pass makes the finding, not MCBR. `I7-OUTAGE-CRITICAL` drops the link at 6 s and restores it at
  60 s. The robot reaches its view of the defect at about 14-20 s. The test checks that the finding is created
  inside the outage.
- **Metric.** Mission-relevant information retained comes from the existing oracle
  (`comm_oracle.mission_information_retained`). It is scored against the latest revision and mission value of
  every belief Model 1 asked to report (`ShoreLink.intent_latest`).
- **Deadline success.** Deadlines are 30 s (critical) and 120 s (routine) of *available-link* time after the
  offer. These are ENGINEERING_ESTIMATE values, not from the spec.
- **Seeds.** Design used development seed 5100000 only. Reporting uses final_test seeds 5300000-5300004, checked
  through `conrad.evaluation.partitions`.

## COM-I7-E001: bandwidth sweep (FINAL, 5 seeds, means)

Retained = mission-relevant information at the receiver at mission end (0..1).

| bw | policy | retained | critical retained | bits | retained/Mbit | alert latency s | delta latency s | queue max / end | backlog max Mbit | deadline crit / routine | crit sync |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 % | **BAAC** | **0.628** | 0.85 | 287,640 | 2.18 | 1.06 | 5.41 | 14.4 / 11.4 | 27.7 | 1.00 / 0.71 | 5/5 |
| | raw | 0.227 | 0.35 | 287,862 | 0.79 | never | never | 158 / 158 | 196 | 0 / 0.07 | 0/5 |
| | FIFO | 0.227 | 0.35 | 287,848 | 0.79 | never | never | 157 / 157 | 196 | 0 / 0.07 | 0/5 |
| | fixed priority | 0.159 | 0.28 | 287,854 | 0.55 | never | never | 159 / 159 | 196 | 0 / 0.05 | 0/5 |
| | value-per-bit | 0.558 | 0.55 | 287,606 | 1.94 | 1.15 | 29.2 | 157 / 157 | 189 | 1.00 / 0.63 | 2/5 |
| 50 % | **BAAC** | **0.589** | 0.85 | 144,574 | 4.08 | 2.38 | 12.7 | 14.4 / 11.4 | 27.7 | 1.00 / 0.58 | 5/5 |
| | raw | 0.227 | 0.35 | 143,401 | 1.59 | never | never | 158 / 158 | 196 | 0 / 0.07 | 0/5 |
| | FIFO | 0.227 | 0.35 | 143,331 | 1.59 | never | never | 158 / 158 | 196 | 0 / 0.07 | 0/5 |
| | fixed priority | 0.055 | 0.00 | 145,069 | 0.38 | never | never | 162 / 162 | 196 | 0 / 0.02 | 0/5 |
| | value-per-bit | 0.501 | 0.55 | 145,582 | 3.45 | 1.00 | 46.5 | 159 / 159 | 195 | 1.00 / 0.39 | 2/5 |
| 10 % | **BAAC** | **0.330** | 0.85 | 28,468 | 11.6 | 1.31 | 49.8 | 14.4 / 12.6 | 26.0 | 0.96 / 0.08 | 5/5 |
| | raw | 0.227 | 0.35 | 28,142 | 8.08 | never | never | 158 / 158 | 196 | 0 / 0.03 | 0/5 |
| | FIFO | 0.227 | 0.35 | 28,026 | 8.12 | never | never | 158 / 158 | 196 | 0 / 0.03 | 0/5 |
| | fixed priority | 0.000 | 0.00 | 28,276 | 0.00 | never | never | 163 / 163 | 196 | 0 / 0 | 0/5 |
| | value-per-bit | 0.239 | 0.35 | 28,620 | 8.35 | 1.31 | never | 163 / 163 | 196 | 1.00 / 0.03 | 0/5 |
| 1 % | **BAAC** | **0.060** | 0.30 (alert) | 2,424 | 24.9 | 1.40 | never | 14.4 / 14.4 | 27.7 | 1.00 / 0 | 0/5 |
| | raw / FIFO / fixed | 0.000 | 0.00 | 2,049 | 0 | never | never | 163 / 163 | 196 | 0 / 0 | 0/5 |
| | value-per-bit | 0.036 | 0.18 | 2,590 | 12.7 | 1.40 | never | 163 / 163 | 196 | 1.00 / 0 | 0/5 |
| 0.1 % | **BAAC** | **0.060** | 0.30 (alert) | 184 | 329 | 145.2 | never | 14.4 / 14.4 | 27.7 | 0.29 / 0 | 0/5 |
| | raw / FIFO / fixed | 0.000 | 0.00 | 0 | n/a | never | never | 163 / 163 | 196 | 0 / 0 | 0/5 |
| | value-per-bit | 0.000 | 0.00 | 184 | 0 | 145.2 | never | 163 / 163 | 196 | 0 / 0 | 0/5 |
| 0 % | all | 0.000 | 0.00 | 0 | n/a | never | never | BAAC 14.4, others 163 | | n/a | 0/5 |

"Crit sync" is the number of seeds (out of 5) where the receiver revision equals the sender revision for the
critical belief at mission end. Duplicate contributions were 0 in every arm, run and level.

**BAAC minus policy retained (mean / worst seed), per non-zero level:**

| level | raw | FIFO | fixed priority | value-per-bit |
|---|---|---|---|---|
| 100 % | +0.401 / +0.392 | +0.401 / +0.392 | +0.469 / +0.436 | +0.070 / **0.000** |
| 50 % | +0.362 / +0.347 | +0.362 / +0.347 | +0.534 / +0.524 | +0.088 / +0.048 |
| 10 % | +0.103 / +0.038 | +0.103 / +0.038 | +0.330 / +0.256 | +0.091 / +0.038 |
| 1 % | +0.060 / +0.058 | +0.060 / +0.058 | +0.060 / +0.058 | +0.025 / **0.000** |
| 0.1 % | +0.060 / +0.058 | +0.060 / +0.058 | +0.060 / +0.058 | +0.060 / +0.058 |

BAAC is strictly above raw, FIFO and fixed priority on every seed at every non-zero level: 25/25 seed-level
pairs per baseline.

**Value-per-bit ties BAAC on 5 of 25 seed-levels.** It never beats BAAC.
- 100 %: seeds 5300003 and 5300004 (0.629 = 0.629).
- 1 %: seeds 5300000, 5300001 and 5300002 (0.058-0.062 for both).

At 1 % both policies deliver only the 184-bit alert, so their retention is equal. Value-per-bit is outside the
spec's comparison set, but these ties are part of the result.

## COM-I7-E002: critical finding during an outage (FINAL, 5 seeds)

Scenario: link down from 6 to 60 s. The critical finding is made at 14.1-20.1 s (per seed: 16.1, 14.1, 20.1,
18.1, 16.1 s), always while the link is down (`link_down_at_finding=True`, 10/10 runs). MCBR is not involved.

**Reconnection trace, seed 5300000 at 100 % (BAAC primary):**

1. 6.0 s: the link drops. Routine 2S/2T/2E updates keep arriving.
2. 16.1 s: the critical finding is made (belief `...05f841`, revision 24, condition FAILED observed on a
   mission-critical component) and queued.
3. By 60 s BAAC's queue holds 15 units, 16.9 Mbit backlog, 1 of them critical. BAAC merged 147 re-offers of the
   same beliefs into those units; 0 units were dropped. The baseline queues hold about 160 units and 180-196 Mbit.
4. 60 s: the link returns. `reevaluate()` rebuilds the deltas against what the receiver already holds.
5. 61.3 s: the F0 alert arrives (rev 24). A second alert copy arrives at 66.4 s. The F1 transmission of the rebuilt unit carries all of its increments
   up to F1, including its F0. It changes no belief revision, and the alert store keeps the latest revision.
6. 66.4 s: the critical F1 delta arrives (rev 24).
7. 72.0 s: the first routine delta arrives (`...62cf3f` rev 13), after the critical delta.

At mission end the receiver holds revision 24 = the sender's revision 24, with 0 duplicate contributions and
0 resync requests.

| | 100 % BAAC | 10 % BAAC | 100 % value-per-bit | 100 % raw | 100 % FIFO | 100 % fixed |
|---|---|---|---|---|---|---|
| retained (mean) | **0.628** | **0.328** | 0.509 | 0.227 | 0.155 | 0.039 |
| critical delivered first after reconnect | 5/5 | 5/5 | alert yes, delta no | no | no | no |
| alert latency from finding, s | 44.4 | 45.7 | 44.4 | never | never | never |
| critical delta latency from finding, s | 49.5 (46.2-52.6) | 97.6 (95.0-104.0) | 114.7 (2 of 5 seeds) | never | never | never |
| critical sync at end | 5/5 | 5/5 | 2/5 | 0/5 | 0/5 | 0/5 |
| queue max (units) | 14.4 | 14.4 | 153.8 | 158 | 160 | 162 |
| duplicate contributions | 0 | 0 | 0 | 0 | 0 | 0 |

The latency floor is the outage itself. The finding happens 40-46 s before the link returns, and the alert
arrives about 1.3 s after reconnection.

**BAAC minus policy retained, E002:**
- 100 %: raw +0.401, FIFO +0.473, fixed priority +0.589, value-per-bit +0.119.
- 10 %: raw +0.186, FIFO +0.234, fixed priority +0.328, value-per-bit +0.189.
- The worst seed is still positive in every case.

**Closed loop.** Each baseline was also run as the mission primary at 100 %, 20 runs. BAAC-primary retained
0.628; the baseline primaries retained:
- raw: 0.218-0.233
- FIFO: 0.038-0.143
- fixed priority: 0.038-0.040
- value-per-bit: 0.459-0.580

In 12 of these 20 runs the offer stream was byte-identical to the BAAC-primary run. In the other 8 (seeds
5300002 and 5300003) it differed, because the receiver state feeds routing's `pending_report_belief_ids`, which
changes later operator offers. The critical finding time was the same in every run. This is why the headline
comparison uses shadow arms.

## Honest failures and limits

- **Surrogate only.** These are python-kernel missions, not Unity. The official I7 status stays whatever
  `gates.py` derives from formal evidence plus upstream gates (I6).
- **Not in the stored retention.** At 1 % and 0.1 % BAAC delivers only the 184-bit alert. No F1 delta reaches the
  receiver, and critical sync is 0/5. At 0.1 % the alert takes 145 s (critical deadline met in only 29 % of
  evaluable offers). At 1 % the routine deadline success is 0 for every policy.
- **Raw evidence is never sent.** Evidence byte sizes are the configured default (200 kB per evidence item,
  ENGINEERING_ESTIMATE), because `ShoreLink` does not look up `PayloadRef.byte_length`. That is why raw, FIFO and
  fixed priority, which send full fidelity, deliver at most a few partial units.
- **The `patch_first_visible_t_s` oracle** recorded a value for only 2 of 5 E002 seeds (15.25 s and 14.25 s,
  both after the 6 s link drop). For the other 3 the finding time (18-20 s) is the evidence.
- **Design choices made in this pass**, all fixed on development seed 5100000 before the FINAL runs:
  compact F0 frame, DEFLATE sizing, critical pre-emption, carried capacity, deferring fresher revisions while a
  fragment is in progress. They change BAAC's behaviour relative to COM-BAAC-E001. **COM-BAAC-E001 was not
  re-run.**
- **Unit tests.** `test_sizes_are_measured_from_payloads_and_monotonic` now checks F0 against
  `increment_bits` / `ALERT_FRAME`.
- **Deadlines** are ENGINEERING_ESTIMATE (30 s / 120 s of available link).

## Reproduce

```
python -m uv run conrad eval run --experiment COM-I7-E001   # 30 missions, workers from config
python -m uv run conrad eval run --experiment COM-I7-E002   # 30 missions (10 shadow + 20 closed-loop)
python -m uv run pytest tests/acceptance/test_i7_constrained_comms.py -q
python -m uv run python scripts/record_gate_evidence.py --surrogate I7
```

The FINAL artifacts were produced by calling `com_i7.run` directly (wall time about 6,820 s each with 8 workers
on a shared machine), so `conrad eval run` was not used and no experiment-registry record was appended.
`dispatch_record.json` notes this. Artifacts are in `artifacts/experiments/COM-I7-E00{1,2}/`, with per-run
traces in `traces/`. The E002 run bundles are kept in `artifacts/runs/COM-I7-E002/`.
