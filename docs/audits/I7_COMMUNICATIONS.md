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

## Formal Unity path (prepared 2026-09-20, NOT YET RUN)

I7 is BLOCKED_UPSTREAM behind I6, which is behind I5 and I4. The formal run was therefore deliberately not
started. What follows is the harness that records formal I7 as a single command once the chain opens, and the
verification that was possible without launching the player (another workstream owns it).

### What is new

| File | Purpose |
|---|---|
| `configs/eval/i7_unity.yaml` | I7-UNITY-E001 declaration: worlds, arms, both sweeps, the reduction and its reason, deadlines, and the per-criterion decision rule. Written before any run on a final world. |
| `conrad/sim/mission/unity_run.py` | `UNITY_SCENARIOS["I7-UNITY-BANDWIDTH"]` and `["I7-UNITY-OUTAGE"]`: the kernel I7 scenarios at a 0.1 s Unity control period. Nothing else changes; the world, the 240 s mission and the declared link stay the surrogate's. |
| `conrad/evaluation/decision_experiments/com_i7_unity.py` | The flight driver and the four criterion rules. It reuses the surrogate's scorer (`com_i7._score_arm`, `_reconnection_trace`) unchanged, so a Unity flight and a kernel flight are scored by the same code. `backend="unity"` is the formal path; `backend="kernel"` is the verification path and is refused on the final worlds. |
| `tests/unity_live/test_i7_unity.py` | The formal test module: one module-scoped fixture flies the declared sweep sequentially, four criterion tests, plus replay, leakage, the sweep declaration and the results file. |
| `scripts/record_unity_gate_evidence.py` | `MODULES["I7"]`, `REPLAY["I7"]`, `PLAN["I7"]` and the I7 note. Criterion names are exactly `gates.py`; the recorder refuses the gate otherwise. |
| `scripts/check_i7_unity_harness.py` | The two checks that do not need a player: `--wiring` (dry run of the formal path) and `--kernel` (the same harness on the python kernel, development seeds). |
| `tests/unit/communication/test_i7_receiver_machinery.py` | 23 tests of the receiver-side machinery the criteria are computed from. |
| `tests/property/communication/test_receiver_properties.py` | 5 property tests of the same machinery over arbitrary increment streams. |

### Worlds

Unity gate worlds come from the digest-pinned `configs/eval/partitions_unity_gates.yaml` (`final_test`
7800000 to 7800019). 7800000 is I1, 7800001 is I3, 7800002 to 7800013 are I4, 7800014 to 7800016 are I6 and
7800017 is the flagship. **I7 takes 7800018 and 7800019, the last two unspent seeds of that split.** No
partition file is edited and no new digest is pinned.

Two consequences, both stated rather than hidden:

1. **Two worlds, not the surrogate's five seeds.** This is the first declared reduction. The surrogate's
   "strictly more on every seed" claim rests on 25 seed-level pairs per baseline; the full Unity sweep gives
   14 per baseline (5 non-zero bandwidth levels plus 2 outage levels, times 2 worlds) and the reduced sweep 8
   (3 plus 1, times 2 worlds).
2. **The split is now exhausted.** Checked on 2026-09-20 against the concurrent I5 Unity workstream: it
   declared its own file (`configs/eval/partitions_i5_unity.yaml`, worlds 7710000 and 7710001), so 7800018 and
   7800019 are free for I7. If that ever changes, I7 must declare a new digest-pinned partition file (the
   `partitions_i4_occluded.yaml` pattern) rather than reuse a spent seed. Re-check
   `configs/eval/i7_unity.yaml` against the other gate configs before running.

### The two sweeps

One flight measures all five policies: BAAC is the primary arm and drives the mission, and raw/send-all,
FIFO, fixed priority and value-per-bit are shadow arms of the same `ShoreLink` over the identical offer
stream, link profile and channel seed. **No baseline is ever dropped from a flight**, in either sweep.

| | full | reduced |
|---|---|---|
| bandwidth levels | 100, 50, 10, 1, 0.1, 0 % | 100, 10, 1, 0 % |
| outage scenario | 100 %, 10 % | 100 % |
| closed loop (each baseline as primary) | at 100 % | none |
| arms per flight | 5 | 5 |
| flights per world | 12 | 5 |
| flights total (2 worlds) | 24 | 10 |

**Declared reason for the reduction** (also in the config, and recorded in the evidence): a Unity flight is
the same python mission plus a lock-step player round trip per control step, so the full sweep costs about
2.3x the reduced one (about 3 hours against about 1 hour and a quarter). The reduced sweep keeps every arm and every qualitative regime, and drops only levels
the surrogate showed to be bracketed by a kept level:

- **50 %** lies between the kept 100 % and 10 %, and no policy changes rank there (surrogate E001 table).
- **0.1 %** reproduces the 1 % regime: BAAC delivers only the 184-bit F0 alert and retains 0.060 at both. The
  two differ only in alert latency (1.40 s against 145.2 s), which the kept levels already measure.
- **Closed loop** is a robustness check of the shadow-arm design, not a gate criterion, and the surrogate
  already ran all 20 of its runs.

A reduced sweep is evidence only when it is recorded as reduced: `test_declared_sweep_is_recorded` writes the
sweep name, `reduced: true`, the kept levels and the dropped levels into `unity_measured.json`, the recorder
copies them into the evidence note, and that test is attached to every criterion.

### Estimated Unity cost

All four numbers below are measured on this machine; the Unity flight estimate is derived from them and is an
ENGINEERING_ESTIMATE, not a measurement of an I7 Unity flight.

| Measurement | Value |
|---|---|
| Gate I4 Unity flights (measured) | 48 sequential flights, 1000 control steps each, 83.1 s per flight (`artifacts/unity/gate_runs/I4` bundle times; the recorded pytest wall clock was 4,096.85 s) |
| Kernel twin of that I4 flight (measured) | 58.7 s, so Unity adds about 24 s per 1000 control steps, about **0.024 s per control step** |
| Kernel I7 flight, 2400 steps, five arms (measured) | 284 s at 100 %, 216 to 241 s at lower levels; 10 flights of the reduced sweep took 2,840 s, a mean of **284 s** |
| Derived I7 Unity flight | 284 s + 2400 x 0.024 s = about **340 s**, call it **6 to 8 min** with the world build and player start |

| Sweep | Flights | Plus the replay re-flight | Estimated wall clock |
|---|---|---|---|
| full | 24 | 25 | **2.5 to 3.3 h** |
| reduced | 10 | 11 | **65 to 90 min** |

For comparison, the same reduced sweep on the python kernel measured 2,840 s (47 min) for its 10 flights.

Run it sequentially. Only one Unity player at a time (HANDOFF section 1, rule 8).

### The exact command

```
python -m uv run python scripts/record_unity_gate_evidence.py I7
```

That runs `tests/unity_live/test_i7_unity.py` against the built player and writes
`artifacts/gates/I7/{evidence_formal.json, unity_measured.json, unity_i7_results.json, unity_pytest.xml,
unity_pytest.txt}`. For the declared fallback:

```
CONRAD_I7_SWEEP=reduced python -m uv run python scripts/record_unity_gate_evidence.py I7
```

Before running, rebuild the player if the C# changed. The module skips itself (which the recorder reads as
NOT_RUN, never as PASS) until the built `ConradUnityV2.dll` contains the `optics_grid` literal, so a stale
player is never launched on a held-out world. The I7 mission runs the full 2S + 2T + 2E stack, as the
surrogate and the flagship do.

Even with a formal PASS, I7 stays BLOCKED_UPSTREAM until I6, I5 and I4 pass.

### What each criterion decides

Criterion names are exactly `conrad.evaluation.gates` and the recorder checks this. A criterion passes iff it
passes on **both** final worlds. Every criterion additionally needs the replay, the leakage scan, the sweep
declaration and the results file to pass.

1. **full mission under constrained bandwidth.** At every declared level the mission ran a full mission (the
   world clock reached the declared duration **and** no deployment module ended in a failed state), all five
   arms produced a full measurement row, at every non-zero level at least one increment reached BAAC's
   receiver, the 0 % floor case delivered zero bits and nothing at all in every arm, and no arm recorded a
   duplicate belief contribution at any level. This criterion is about the mission running and delivering
   under the constraint. Superiority is criterion 3 and is not checked here.
2. **full mission under outages.** The outage flight ran a full mission; the mission-critical structural
   finding was created while the link was
   down; at least one critical unit was held in the store-and-forward queue for the whole remaining outage;
   after reconnection BAAC delivered that belief's F0 alert and F1 delta ahead of every routine delta; at
   mission end the receiver holds at least the critical revision that was found during the outage; BAAC
   coalesced at least one superseded re-offer; duplicate contributions and resync requests are both zero.
3. **BAAC retains more mission-relevant information than raw/FIFO/fixed-priority.** At every non-zero level
   and in the outage scenario, BAAC's `mission_information_retained` is strictly greater than raw, FIFO and
   fixed priority on every world. Value-per-bit is reported and never required; it is outside the ch25 set.
4. **critical latency and sync error compared against baselines** (ch26 Phase 11). Every arm has a critical
   alert latency, a critical delta latency, a receiver sync error over everything Model 1 asked to report, a
   critical-belief sync flag and deadline success, all on the same offer stream. The criterion is the
   comparison being complete, and every case where BAAC is worse than a baseline is listed in the evidence.

The surrogate reported `sync_critical` only. The Unity harness adds `comm_oracle.sync_error` over every
reported belief, which is what ch26 asks for by name.

The closed-loop flights of the full sweep (each baseline run as the mission primary) are flown and stored in
`unity_i7_results.json`, but they decide no criterion, exactly as in the surrogate. They exist to show that
the shadow-arm comparison is not an artefact of BAAC driving the mission.

### What has and has not been exercised

**Exercised.**

- The whole harness end to end on the python-kernel backend, reduced sweep, mission development seeds
  5100000 and 5100001, 10 flights, 2,840 s of mission wall clock:
  `python -m uv run python scripts/check_i7_unity_harness.py --kernel --sweep reduced`. Result in
  `artifacts/experiments/I7-UNITY-HARNESS-CHECK/kernel_reduced.json` (SURROGATE, written outside
  `artifacts/gates/` so it can never be mistaken for gate evidence). All ten flights ran the full 240 s with
  no failed deployment module. Criteria 1, 2 and 4 pass on both worlds. **Criterion 3 fails on development
  world 5100001**, which is the finding below.
- The wiring of the formal path, by `python -m uv run python scripts/check_i7_unity_harness.py --wiring`:
  both Unity scenarios are registered, all 24 declared flights resolve to concrete runtimes with the right
  bandwidth (1200 / 600 / 120 / 12 / 1.2 / 0 bps), the right outage window `[6 s, 60 s)`, 240 s duration, a
  0.1 s control period, the right primary policy and the four shadow arms; the declared worlds are
  `unity_gate` final_test worlds; and the criterion names match both `gates.py` and the recorder's `PLAN`.
  Result in `artifacts/experiments/I7-UNITY-HARNESS-CHECK/wiring.json`. No world is built and nothing is
  launched.
- The receiver-side machinery, by 23 unit tests and 5 property tests (below).

### The kernel verification changed two criterion rules, and found one real risk

Both rule changes were made on development seeds, before any final world had ever been built. Criterion 3,
which is the spec's actual claim, was not touched.

1. **Criterion 1 now checks delivery, not score.** It first read "BAAC retained more than zero mission
   information at every non-zero level". On development world 5100001 at 1 % the F0 alert arrives at 1.0 s and
   then scores zero, because `comm_oracle.belief_score` credits an alert only while `alert_revision >= the
   sender's latest revision` and that belief keeps being revised. Whether the alert is still current at
   mission end is a property of the world, not of the mission running under the constraint, so the rule now
   requires at least one increment to reach BAAC's receiver.
2. **Criterion 2 now requires the found revision, not end-of-mission equality.** It first required
   `receiver revision == sender latest revision` for the critical belief at mission end. On development world
   5100001 that belief is revised 383 times: BAAC's receiver ends at revision 355 against a sender latest of
   383, while raw ends at 0 and FIFO, fixed priority and value-per-bit hold nothing at all. Demanding exact
   equality there fails BAAC for being 28 revisions behind on a belief no baseline delivered. The rule now
   requires the receiver to hold at least the revision that was FOUND during the outage (which it does), and
   the end-of-mission sync error is compared against the baselines under criterion 4, which is where ch26
   Phase 11 puts it.

**The risk: BAAC loses to raw and FIFO on development world 5100001 at 10 % bandwidth.** This is a result, not
a harness bug, and nothing was tuned to remove it.

| world 5100001, 10 % | BAAC | raw | FIFO | fixed priority | value-per-bit |
|---|---|---|---|---|---|
| retained | **0.080** | 0.222 | 0.222 | 0.028 | 0.222 |
| beliefs at the receiver | 2 | 5 | 5 | 1 | 5 |
| bits sent | 26,844 | 28,230 | 29,251 | 28,659 | 26,929 |
| coalesced re-offers | 163 | 0 | 0 | 0 | 0 |
| queue depth max | 18 | 182 | 182 | 186 | 187 |

That world has a few extremely active beliefs (the critical one is revised 383 times in 240 s). BAAC's
novelty and value-per-bit ranking keeps superseding its own queued units with fresher revisions of those few
beliefs, so it spends its 27 kbit tracking them and lands 2 distinct beliefs at the receiver. Dumb FIFO at
full fidelity lands 5. The oracle averages over all 18 beliefs Model 1 asked to report, so FIFO scores higher.
At 1 % on the same world every policy scores 0.000, so BAAC is not strictly greater there either. Criterion 3
therefore fails on that world at 10 % and at 1 %, and passes at 100 % and in the outage scenario.

Consequences the integrator should weigh before spending the two held-out worlds:

- The formal I7 run can fail. The surrogate's five final seeds (5300000 to 5300004) all had BAAC strictly
  above raw, FIFO and fixed priority at every non-zero level; development world 5100001 shows that this is not
  a property of the mechanism on every world.
- Two candidate root causes, both open: BAAC's coalescing may be too aggressive when a belief is revised
  faster than the link can carry it, and the oracle's `STALE_CREDIT` is applied to a stale view (half credit)
  but not to a stale alert (no credit), which penalises exactly the F0-only regime BAAC is designed for.
- **Neither was changed.** Changing either after seeing this would be tuning the metric to the result. They
  are listed here so the decision is made deliberately, on development evidence, before the final run.

**Not exercised.**

- **No Unity flight of any kind was run.** Every number under "Estimated Unity cost" for the Unity path is
  derived from gate I4 flights, not measured on an I7 flight.
- **No final world was touched.** 7800018 and 7800019 have never been built.
- Replay of an I7 Unity bundle, the leakage scan on an I7 Unity bundle, and the `optics_grid` player guard
  have never run against a real I7 bundle. They are the same calls gates I4 and I6 already use.
- `scripts/record_unity_gate_evidence.py I7` has not been executed, because executing it launches the player.
  Its plan was checked against `gates.py` in code instead.

### Re-verification of the receiver-side machinery

`tests/unit/communication/test_i7_receiver_machinery.py` (23 tests) and
`tests/property/communication/test_receiver_properties.py` (5 property tests) cover, in the order of the
criteria that depend on them: receiver knowledge tracking, semantic deltas, the durable queue,
acknowledgements and reconciliation, blackout recovery, stale-item re-evaluation, fidelity control,
mission-critical priority, and that duplicate or retransmitted packets never double-count belief evidence.
They all pass. Three things found on the way, none of which changes a surrogate number:

1. **F2 adds nothing to the oracle score once F1 has been applied.** `ReceiverStore._apply` records the
   evidence IDs listed in the view it just applied, and `comm_oracle.belief_score` promotes a belief to the F2
   level as soon as every evidence ID is known. Every NEW_BELIEF F1 delta carries `evidence_support`, so for
   those beliefs F2 is worth zero. `mission_information_retained` therefore distinguishes four levels, not
   five: 0.30 (alert), 0.85 (F1 or F2), 0.95 (F3), 1.0 (F4). F3 and F4 still score, because they need the
   evidence bytes. This is a coarseness of the scoring model, not of the wire format, it applies identically
   to every arm, and it inverts no comparison. Pinned by
   `test_f2_adds_no_oracle_score_once_f1_carried_the_evidence_ids`.
2. **The sender's model of the receiver can regress on a stale acknowledgement.** If `ReceiverKnowledge`
   acknowledges an increment it has already applied, `apply_deltas` raises `ResyncRequired`, the view is
   dropped, and a later stale NEW_BELIEF increment re-seats it at the older revision. The direction is safe:
   the sender then under-claims and resends, it never over-claims and suppresses a needed send. The property
   test `test_sender_model_is_never_ahead_of_the_receiver` pins the safe direction. It is reachable only for
   a baseline that keeps several revisions of one belief queued and may deliver them out of order
   (C-B4 value-per-bit); BAAC coalesces, and the strict FIFO and fixed-priority orders deliver in
   creation order. The cost is wasted bits on that baseline, never a wrong belief.
3. **The mission runs the queue in memory.** `PersistentQueue` writes every mutation to disk atomically when
   it is given a path, but `BAACSender` is constructed without one in `ShoreLink`, so a mission exercises
   store-and-forward without exercising the on-disk part. The disk path is covered directly instead:
   `test_persistent_queue_survives_a_restart_with_partial_fidelity_and_drops` and the property test
   `test_durable_queue_round_trips_through_disk` reload a queue from its file and check entries, fragment
   progress, attempt counts, drop records and queued bits.

#### One environment trap, not a defect

The first kernel verification run was started from the session scratchpad
(`C:\Users\Kerem\AppData\Local\Temp\claude\...\scratchpad\...`) and its bandwidth flight lost its sensing
module at about 8 s with `[WinError 3] The system cannot find the path specified: ...\objects\`. That is the
Windows 260-character path limit hitting the sharded object store, not a fault in the communication stack:
the outage flight in the same batch, whose run directory name is four characters shorter, ran all 2400 steps
normally. Run the harness from a short root; `scripts/check_i7_unity_harness.py` writes under
`artifacts/experiments/I7-UNITY-HARNESS-CHECK/` inside the repository for this reason. A mission that dies
this way would not silently pass: the `ran_full_mission` and `arms_measured` checks of criterion 1 are there
for it, and the retention criterion failed on that crippled run because every policy tied at 0.700.

### Honest limits of the prepared path

- **Two worlds.** See "Worlds" above.
- **The Unity control period is 0.1 s, the surrogate's was 0.05 s.** Unity requires a whole number of physics
  steps, and every other Unity gate uses 0.1 s. The mission length (240 s), the link, the scenario overrides
  and the scoring are unchanged, but the numbers are not expected to match the surrogate seed for seed.
- **Raw evidence is still never sent.** `ShoreLink` does not look up `PayloadRef.byte_length`, so evidence
  byte sizes stay at the configured 200 kB default (ENGINEERING_ESTIMATE). This is the same limitation the
  surrogate has, carried into the formal path unchanged, and it is why raw, FIFO and fixed priority deliver so
  little.
- **Deadlines stay ENGINEERING_ESTIMATE** (30 s critical, 120 s routine, in seconds of available link).
- **All link numbers stay SYNTHETIC_ONLY.**
- **The gate can fail.** Criterion 3 fails on development world 5100001 at 10 % and at 1 %. See the risk
  section above. A failed research mechanism is an acceptable result; do not spend the two held-out worlds
  expecting a pass.
