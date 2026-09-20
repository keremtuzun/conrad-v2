# Gate I7: constrained communications (audit, 2026-09-19)

Spec I7 (ch25): "Run full mission under bandwidth. Then outages. BAAC must retain more mission-relevant
information than raw/FIFO/fixed-priority approaches."

**Evidence class: SURROGATE.** Every number below comes from integrated missions on the python L1 kernel
(`conrad.sim.mission`), not Unity. The surrogate evidence is in `artifacts/gates/I7/evidence_surrogate.json`
and cannot promote the gate. The formal run through Unity still has to come from the integrator. All link
numbers are SYNTHETIC_ONLY.

> **Superseded sections.** On 2026-09-20 a BAAC scheduling defect found on development world 5100001 was
> diagnosed and repaired (see "The BAAC scheduling repair" at the end of this file). The repair changes BAAC's
> behaviour, so the COM-I7-E001 and COM-I7-E002 tables below are the PRE-REPAIR record on the now SPENT seeds
> 5300000-5300004 and are no longer the gate's surrogate evidence. The current surrogate evidence is
> COM-I7-E003 / COM-I7-E004 on the freshly declared final seeds 5500000-5500004. The baseline arms are
> unchanged by the repair; only the BAAC rows below are stale.

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
  expecting a pass. **Repaired on 2026-09-20; see the next section. The 10 % failure is fixed on merit, the
  1 % one is an all-policy tie at 0.000 and is still there.**

## The BAAC scheduling repair (2026-09-20)

The risk recorded above was real, and it was a defect in the mechanism, not an artefact of the measurement.
It was diagnosed and fixed on the mission DEVELOPMENT seeds 5100000 and 5100001 only. No final world and no
Unity player was touched, criterion 3 (the spec's own claim) was not changed, no threshold, test or baseline
was weakened, and the communication oracle was not touched.

### What the link was actually being spent on

One instrumented development flight (world 5100001, `I7-BANDWIDTH`, 10 % of the declared link, all five arms)
accounts for every transmitted bit by belief, and by whether that belief was already at the receiver when the
bits were sent. Artifact: `artifacts/experiments/I7-REPAIR-DIAG/diag-base-s5100001-bw0.1.json`.

| world 5100001, 10 %, pre-repair | BAAC | raw | FIFO | fixed priority | value-per-bit |
|---|---|---|---|---|---|
| bits sent | 26,844 | 28,230 | 29,251 | 28,659 | 26,929 |
| beliefs any bits were spent on | 2 | 6 | 6 | 3 | 5 |
| beliefs at the receiver | 2 | 5 | 5 | 1 | 5 |
| bits on the ONE critical belief | 20,538 (76.5 %) | 5,296 | 6,320 | 21,514 (75.1 %) | 6,728 |
| bits re-sending a belief the receiver already held | 16,521 (61.5 %) | 0 | 0 | 0 | 4,254 (15.8 %) |
| F0 alert frames sent | 6 (1,104 bits) | 0 | 0 | 0 | 13 (2,392 bits) |
| coalesced re-offers | 163 | 0 | 0 | 0 | 0 |

The mission makes 187 offers over 240 s and reports 18 beliefs. The critical belief is offered 16 times (it is
revised 383 times; `reoffer_interval_s` is 15 s). Pre-repair BAAC put 76.5 % of a 27 kbit link into that one
belief, landing 3 of its deltas and 6 alert frames and ending at receiver revision 245 of 383. raw and FIFO
spend about 5.3 kbit per belief and land 5 distinct beliefs each. The oracle averages over all 18 reported
beliefs, so 5 stale-but-distinct beliefs beat 2 well-tracked ones.

Its delta was not cheap either: because that belief changes in almost every field, a revision delta measures
3.5 to 6 kbit, about the size of a first delivery. The traffic was near-full-size re-deliveries of one belief,
not cheap incremental updates.

Two mechanisms produced it.

1. **Pre-emption was unbounded.** `policy.critical_first` puts a critical unit's F0/F1 first in the ordering
   key lexicographically, so it beats everything else whatever the value per bit. Every 15 s the critical
   belief was re-offered, a fresh delta was built against the receiver model, and it pre-empted every
   never-sent belief again. ch19 gives Level 0/1 pre-emption so the FINDING gets through; it does not ask for
   one belief to own the link for the rest of the mission.
2. **The value model ignored what the receiver already held.** The term was
   `mission_value * information_retained[level]`, the unit's ABSOLUTE information content, identical for a
   re-offer of a belief the receiver already holds and for the first delivery of one it has never seen.

### Which cause dominates (measured, not assumed)

Four full missions on the failing cell, one per combination, everything else identical. Artifact:
`artifacts/experiments/I7-REPAIR-DIAG/sweep_abl.json`.

| world 5100001, bandwidth 10 % | BAAC retained | beliefs at the receiver |
|---|---|---|
| A, neither (the shipped behaviour) | 0.080 | 2 |
| B, bounded pre-emption only | 0.222 | 4 |
| C, receiver-relative value only | 0.080 | 2 |
| D, both (the repair) | **0.250** | 5 |

Unbounded pre-emption dominates: alone it recovers 0.142 of the 0.170 gap. The value model alone changes
nothing at all, because pre-emption outranks it; what it adds is the last belief, which turns a tie with raw
and FIFO (0.222 = 0.222, not "strictly more") into a strict win. Both are needed and neither is sufficient.
raw 0.222, FIFO 0.222 and fixed priority 0.028 are identical in all four runs: no baseline moves.
At 1 % on the same world all four combinations score 0.000, and so does every baseline.

### What changed

| File | Change |
|---|---|
| `conrad/communication/config.py` | `preempt_until_delivered` (default true), `receiver_relative_value` (default true), `stale_view_credit` (default 0.5, ENGINEERING_ESTIMATE). All three are ablation switches, so the table above is reproducible. |
| `conrad/communication/scheduler.py` | `_critical_undelivered()`: a critical unit pre-empts only while the receiver does not yet hold that belief (no alert of it at F0, no view of it at F1). Afterwards it competes on value per bit, keeping the high mission value the finding gave it. `receiver_credit()`: the information the receiver already holds about the unit's belief, so an increment is scored by what the receiver GAINS. Both are applied only when `policy.use_novelty and policy.use_receiver_knowledge`, which is BAAC only. |
| `conrad/communication/__init__.py` | Exports `receiver_credit`; the two modelling assumptions are listed in `IMPLEMENTATION_METADATA`. |
| `tests/unit/communication/test_i7_scheduler_repair.py` | 13 new tests: the pre-repair choice with both switches off, each switch alone reproducing the measured ablation, the repaired choice, pre-emption still absolute until the finding lands, the first-alert exemption, and that no baseline plan moves when either switch is toggled. |

The stale-view credit is 0.5 of the F1 level. It is the same modelling assumption the communication oracle
makes for a stale view, and that is stated rather than hidden: it is a property of "an older revision of a
belief is not ignorance of it", not a number fitted to the score. Nothing in `conrad/evaluation/oracle/` was
changed, and the scheduler does not import it (`tests/leakage` still passes).

### One regression the first attempt caused, and its fix

The first repaired sweep improved retention but pushed BAAC's critical alert latency from about 1.0 s to
12.5 s at 100 % on both development worlds: the stale-view discount was also being applied to the F0 alert, so
whenever the belief had been reported as routine before it became critical, the alert scored below its cost
and the finding only arrived with the whole F1 delta. ch19 Level 0 is an emergency control message whose
value is timeliness, not information content, so the FIRST alert about a belief is now exempt from the
discount and every later alert of the same belief is discounted like any other increment. Measured effect,
BAAC critical alert latency, pre-repair / first attempt / shipped:

| cell | pre-repair | first attempt | shipped |
|---|---|---|---|
| 5100000 bandwidth 100 % | 1.03 s | 12.51 s | 1.03 s |
| 5100001 bandwidth 50 % | 1.10 s | 12.42 s | 1.05 s |
| 5100001 bandwidth 100 % | 1.15 s | 10.34 s | 5.00 s |

Retention is bit-identical between the two attempts at every cell, so the exemption costs nothing there. The
residual is the last row: 1.15 s to 5.00 s on one cell at full bandwidth. It is inside the 30 s critical
deadline, raw, FIFO and fixed priority never deliver that belief at all, and value-per-bit reaches 1.00 s. It
is listed under criterion 4 below rather than tuned away.

### The development sweep, before and after

All five arms, both development worlds, kernel backend, 240 s missions at a 0.1 s control period, one flight
per cell. "Verdict" is criterion 3 as written: BAAC STRICTLY greater than raw, FIFO and fixed priority.
Artifacts: `sweep_before-extra.json`, `sweep_after2.json` and the earlier
`artifacts/experiments/I7-UNITY-HARNESS-CHECK/kernel_reduced.json` for the cells it already covered.

| world | case | BAAC before | BAAC after | raw | FIFO | fixed | value-per-bit | verdict before -> after |
|---|---|---|---|---|---|---|---|---|
| 5100000 | bandwidth 100 % | 0.660 | 0.660 | 0.179 | 0.179 | 0.100 | 0.577 | WIN -> WIN |
| 5100000 | bandwidth 50 % | 0.629 | 0.660 | 0.179 | 0.179 | 0.026 | 0.561 | WIN -> WIN |
| 5100000 | bandwidth 10 % | 0.220 | 0.232 | 0.179 | 0.179 | 0.000 | 0.179 | WIN -> WIN |
| 5100000 | bandwidth 1 % | 0.041 | 0.041 | 0.000 | 0.000 | 0.000 | 0.041 | WIN -> WIN |
| 5100000 | bandwidth 0.1 % | 0.041 | 0.041 | 0.000 | 0.000 | 0.000 | 0.000 | WIN -> WIN |
| 5100000 | bandwidth 0 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | floor, not judged |
| 5100000 | outage 100 % | 0.660 | 0.660 | 0.179 | 0.158 | 0.026 | 0.554 | WIN -> WIN |
| 5100000 | outage 10 % | 0.220 | 0.220 | 0.126 | 0.158 | 0.000 | 0.198 | WIN -> WIN |
| 5100001 | bandwidth 100 % | 0.537 | 0.572 | 0.222 | 0.222 | 0.222 | 0.537 | WIN -> WIN |
| 5100001 | bandwidth 50 % | 0.486 | 0.423 | 0.222 | 0.222 | 0.222 | 0.520 | WIN -> WIN |
| 5100001 | bandwidth 10 % | **0.080** | **0.250** | 0.222 | 0.222 | 0.028 | 0.222 | **LOSS -> WIN** |
| 5100001 | bandwidth 1 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | TIE -> TIE |
| 5100001 | bandwidth 0.1 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | TIE -> TIE |
| 5100001 | bandwidth 0 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | floor, not judged |
| 5100001 | outage 100 % | 0.537 | 0.572 | 0.222 | 0.028 | 0.028 | 0.486 | WIN -> WIN |
| 5100001 | outage 10 % | **0.051** | **0.165** | 0.142 | 0.114 | 0.000 | 0.171 | **LOSS -> WIN** |

Beliefs at BAAC's receiver on world 5100001 went from 2 to 5 at bandwidth 10 %, from 1 to 3 at outage 10 %,
from 11 to 13 at 100 %, and from 8 to 10 at 50 %.

**Two losses, not one.** The outage scenario at 10 % had never been run on a development world before this
pass (the earlier kernel check only flew the outage at 100 %), and the same defect was losing there too:
BAAC 0.051 against raw 0.142 and FIFO 0.114. Both losses are now wins.

**Levels that are still not a strict win, stated plainly.**

- **World 5100001 at 1 % and at 0.1 %: every policy scores exactly 0.000.** BAAC is not worse, but it is not
  strictly greater either, so criterion 3 as written does not pass there. This is arithmetic, not scheduling:
  1 % of the link carries 2,880 bits over the mission and 0.1 % carries 288, while the cheapest F1 delta in
  that world measures about 5,000 bits. The only thing that fits is the 184-bit F0 alert, and the oracle
  credits an alert only while `alert_revision >= the sender's latest revision`; that world's critical belief
  is revised 383 times, so any alert is stale by mission end. No scheduling policy can score above zero there.
  Making it score would mean changing the oracle or adding a fidelity level below F1 after seeing the result,
  which is tuning the measurement. It is left failing.
- **Value-per-bit still beats BAAC on retention at 50 % on world 5100001** (0.520 against 0.423), and the
  repair widened that gap from 0.034 to 0.097. Value-per-bit is outside the ch25 comparison set, it is
  reported and never required, but it is a real cost of the repair on that cell.

### The outage path was re-run and did not regress

Both outage flights per world (100 % and 10 %), repaired BAAC, from `sweep_after2.json`:

| | 5100000 100 % | 5100000 10 % | 5100001 100 % | 5100001 10 % |
|---|---|---|---|---|
| ran a full mission, no failed module | yes | yes | yes | yes |
| finding created while the link was down | yes | yes | yes | yes |
| a critical unit held in the queue for the whole remaining outage | yes | yes | yes | yes |
| F0 alert and F1 delta delivered ahead of every routine delta | yes | yes | yes | yes |
| receiver holds at least the revision found during the outage | 24 of 24 | 24 of 24 | 355 of 54 | 54 of 54 |
| coalesced re-offers | 159 | 160 | 168 | 167 |
| duplicate contributions | 0 | 0 | 0 | 0 |
| resync requests | 0 | 0 | 0 | 0 |

The critical alert and delta latencies are unchanged by the repair in every outage cell (46.31 / 51.25,
47.53 / 98.35, 4.15 / 9.33 and 5.53 / 57.15 s). One end-of-mission number did move: on world 5100001 at
outage 10 % the receiver's critical revision at mission end is 54 instead of 245, because BAAC no longer
spends the scarce link re-tracking that one belief. It still holds the revision the finding was made at,
which is what criterion 2 requires, and the end-of-mission sync error is compared under criterion 4, where
ch26 Phase 11 puts it. On the same cell BAAC's overall receiver sync error improved from 1.000 to 0.889 and
its retention tripled.

### Criterion 4: every case where BAAC is worse than a baseline

Measured on the repaired development sweep. **BAAC is never worse than raw, FIFO or fixed priority** on
critical alert latency, critical delta latency, receiver sync error or critical sync, at any level on either
world. Against value-per-bit, which is outside the ch25 set:

| cell | BAAC is worse on |
|---|---|
| 5100000 bandwidth 50 % | alert latency 1.31 s against 1.00 s |
| 5100000 outage 100 % | alert latency 46.31 s against 46.15 s |
| 5100001 bandwidth 100 % | alert latency 5.00 s against 1.00 s |
| 5100001 bandwidth 50 % | alert latency 1.05 s against 1.00 s; sync error 0.778 against 0.611 |
| 5100001 outage 10 % | sync error 0.889 against 0.833 |

### The two harness findings, re-checked in this code

Both were reported OPEN by the earlier pass and both stay OPEN. Neither was changed, and the repair does not
depend on either.

- **F2 adds nothing to the oracle score once F1 carries the evidence IDs.** Still true, and now bounded:
  in the instrumented 10 % flight BAAC never reaches F2 at all (100 % of its bits are F0 and F1), so the
  coarseness costs nothing in the regime where criterion 3 was failing. It only bites at high bandwidth,
  where BAAC pays roughly 1 kbit per belief for an increment the oracle already credited from F1. The wire
  format genuinely carries more at F2 (the evidence IDs and a quantised embedding); it is the scoring model
  that is coarse. Making the scheduler skip F2 would be tuning to the oracle, so it is not done.
- **The sender's receiver-model can regress on a stale acknowledgement.** Still reachable only for
  value-per-bit (C-B4), which can hold several revisions of one belief and deliver them out of order. BAAC
  coalesces to one unit per belief, and the repair does not change that: bounded pre-emption changes WHEN a
  unit is sent, never how many units of one belief are queued. Duplicate contributions and resync requests
  were 0 in every arm of all 16 repaired development flights, and the property test
  `test_sender_model_is_never_ahead_of_the_receiver` still pins the safe direction.

### Fresh final seeds for the surrogate

The repair changes BAAC's behaviour, so the COM-I7-E001 / E002 evidence on mission final_test seeds
5300000-5300004 is stale, and those seeds (and the whole 5300000-5300059 range) are SPENT. A new
digest-pinned partition file was declared BEFORE the re-run:

- `configs/eval/partitions_i7_v2.yaml`, domain `i7_mission_v2`, final_test **5500000-5500004**, pinned in
  `conrad.evaluation.partitions` as `I7_V2_PARTITIONS_SHA256`. It carries no development split: the I7
  development runs keep using the `mission` development split.
- Collision check before choosing it: every 5/6/7/8-million seed literal under `configs/`, `conrad/`,
  `scripts/`, `tests/` and `docs/` was listed; the used bands are 5000000, 5100000-5400040, 6100000-6700060,
  6900000-7000000, 7100000-7100010, 7300000-7300010, 7400001-7410021, 7500000-7500010, 7600000-7600010,
  7700000-7700010, 7710000-7710002, 7800000-7820000, 7900000-7900010 and 8000000-8000313. Nothing anywhere
  used 55xxxxx. The file's loader also checks disjointness against every other partition file on load.
- `configs/eval/com_i7_e003.yaml` and `com_i7_e004.yaml` re-run the E001 and E002 designs unchanged on those
  seeds; `com_i7.py` takes a `partition_domain` key so the seed check uses the new domain;
  `conrad/evaluation/dispatch.py` registers `COM-I7-E003` and `COM-I7-E004`;
  `tests/acceptance/test_i7_constrained_comms.py` and the `SURROGATE_PLAN["I7"]` entries of
  `scripts/record_gate_evidence.py` read the E003/E004 artifacts. E001/E002 stay on disk as the spent
  pre-repair record.

```
python -m uv run conrad eval run --experiment COM-I7-E003
python -m uv run conrad eval run --experiment COM-I7-E004
python -m uv run python scripts/record_gate_evidence.py --surrogate I7
```

### COM-I7-E003: bandwidth sweep, FINAL, seeds 5500000-5500004, repaired BAAC

Means over the 5 fresh final seeds. Shadow arms, so every policy saw the identical offer stream and link.

| bw | BAAC | raw | FIFO | fixed priority | value-per-bit | BAAC strictly more than the ch25 set on EVERY seed |
|---|---|---|---|---|---|---|
| 100 % | **0.626** | 0.231 | 0.231 | 0.163 | 0.570 | yes (worst seed +0.350) |
| 50 % | **0.589** | 0.231 | 0.231 | 0.111 | 0.545 | yes (worst seed +0.254) |
| 10 % | **0.261** | 0.231 | 0.231 | 0.012 | 0.208 | yes (worst seed +0.028) |
| 1 % | 0.027 | 0.000 | 0.000 | 0.000 | 0.027 | **no** |
| 0.1 % | 0.027 | 0.000 | 0.000 | 0.000 | 0.000 | **no** |
| 0 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | floor, not judged |

The 10 % regime, which is where the defect lived, is a win on all five seeds. The 1 % and 0.1 % rows are the
failure: on 2 of the 5 seeds every policy scores exactly 0.000, so "strictly greater" does not hold.

| seed | critical belief revised | 1 % BAAC / raw | 0.1 % BAAC / raw |
|---|---|---|---|
| 5500000 | 27 times | 0.046 / 0.000 | 0.046 / 0.000 |
| 5500001 | 376 times | **0.000 / 0.000** | **0.000 / 0.000** |
| 5500002 | 361 times | **0.000 / 0.000** | **0.000 / 0.000** |
| 5500003 | 28 times | 0.046 / 0.000 | 0.046 / 0.000 |
| 5500004 | 29 times | 0.044 / 0.000 | 0.044 / 0.000 |

This is the 5100001 regime, and it is arithmetic rather than scheduling: 1 % of the declared link carries
2,880 bits over the 240 s mission and 0.1 % carries 288, while the cheapest F1 delta in those worlds measures
about 5,000 bits. Only the 184-bit F0 alert fits, and `comm_oracle.belief_score` credits an alert only while
`alert_revision >= the sender's latest revision`.

**The repair is not the cause.** On development world 5100001 the 1 % cell scores 0.000 for every policy both
before and after the repair (TIE -> TIE in the sweep table above), and on 5100000 it scores 0.041 both before
and after. What changed is the draw: the old final seeds 5300000-5300004 happened to contain five
slow-revising critical beliefs, and the fresh draw contains two fast-revising ones.

**What BAAC actually did on those two seeds, which the score does not show.** BAAC delivered the critical F0
alert and raw, FIFO and fixed priority delivered nothing at all:

| seed | level | BAAC alert arrival | raw / FIFO / fixed | BAAC bits | baseline bits |
|---|---|---|---|---|---|
| 5500001 | 1 % | yes, 1.0 s after the offer | never | 2,392 | 2,049 (an F1 fragment that never completed) |
| 5500001 | 0.1 % | yes, 107.0 s | never | 184 | 0 |
| 5500002 | 1 % | yes, 1.0 s | never | 2,576 | 2,049 (never completed) |
| 5500002 | 0.1 % | yes, 94.0 s | never | 184 | 0 |

The alert carried revision 46 on 5500001 and 63 on 5500002; those beliefs reach 376 and 361 by mission end,
so the alert is stale and scores 0.000. Value-per-bit matches BAAC exactly on these four cells. This is a real
behavioural difference that `mission_information_retained` does not capture. It is recorded as a limitation of
the measurement, **not** as a claimed win: the criterion is scored on retention, and on retention it is a tie.

Value-per-bit also beats BAAC on one seed at 50 % (worst-seed difference -0.060). It is outside the ch25
comparison set, reported and never required.

### COM-I7-E004: critical finding during an outage, FINAL, seeds 5500000-5500004, repaired BAAC

30 missions: 10 shadow flights (100 % and 10 %) plus 20 closed-loop flights with each baseline as the primary.

| | 100 % BAAC | 10 % BAAC | 100 % raw | 100 % FIFO | 100 % fixed | 100 % value-per-bit |
|---|---|---|---|---|---|---|
| retained (mean of 5 seeds) | **0.626** | **0.217** | 0.231 | 0.130 | 0.042 | 0.541 |
| 10 % baselines | | | 0.151 | 0.154 | 0.000 | 0.205 |

BAAC is strictly above raw, FIFO and fixed priority in the outage scenario at both levels on every seed.
Duplicate contributions were 0 in every arm of all 30 missions, and coalescing ran in all of them (106 to 159
superseded re-offers per flight).

Where criterion 2 fails:

| seed | patch first visible | finding made at | inside the outage [6 s, 60 s)? | critical revision at the receiver, 100 % |
|---|---|---|---|---|
| 5500000 | 14.25 s | 15.1 s | yes | 27 of 27 |
| 5500001 | 53.25 s | 55.1 s | yes | **348 of 376** |
| 5500002 | 66.25 s | 68.1 s | **no** | 361 of 361 |
| 5500003 | 15.25 s | 16.1 s | yes | 28 of 28 |
| 5500004 | 16.25 s | 17.1 s | yes | 29 of 29 |

1. **Seed 5500002 never exercises the outage path.** The outage window is fixed at [6 s, 60 s) by the
   `I7-OUTAGE-CRITICAL` scenario, and on that world the lane pass only reaches a view of the lane-side defect
   at 66.25 s, so the finding is made 8 s after the link is already back. This is world geometry against a
   fixed window, not a communication fault: on the other 4 seeds the finding is made during the outage, is
   held in the queue for the rest of it, and its F0 alert and F1 delta arrive ahead of every routine delta at
   both bandwidth levels (8 of 8 flights).
2. **Seed 5500001 ends 348 of 376 at 100 %.** That belief is revised 376 times in 240 s, so its last offers
   are younger than the link latency. The receiver does hold revision 46, the revision the finding was made
   at during the outage, which is what the Unity harness rule (criterion 2, revised on development seeds
   before any final world was built) requires. The surrogate test still demands exact end-of-mission
   equality, so it fails here. The two rules disagree, and that disagreement is now recorded rather than
   silently resolved.

### OPEN, and decisions to make BEFORE any further run

The rule implemented in `tests/acceptance/test_i7_constrained_comms.py` and in
`com_i7_unity.retention_measure` requires BAAC to be **strictly greater than raw, FIFO and fixed priority on
every seed at every non-zero level**. The ch25/ch26 wording is "BAAC must retain more mission-relevant
information than raw/FIFO/fixed-priority approaches", which does not itself say how to aggregate over seeds
and levels, nor what to do with a level where the link cannot carry a single update from any policy and every
arm ties at 0.000.

That gap is now load-bearing: it is the whole difference between the recorded FAIL and a PASS.

**It is not resolved here, and it must not be resolved by this pass.** Re-deriving the aggregation after
seeing which seeds tie would be manufacturing a pass, which is exactly what ADR-0008 and the workstream rules
forbid. The result stands as FAIL. If the aggregation is to change, the sequence is: decide the rule, write it
into `configs/eval/i7_unity.yaml` and the acceptance test as a declaration, and only then run, on seeds that
have not been used. Candidate readings someone will have to choose between, listed so the decision is explicit
and not smuggled in:

1. Keep the current rule. I7 criterion 3 is FAIL and stays FAIL until the mechanism can deliver an update at
   1 % on a world whose critical belief is revised hundreds of times, which at 2,880 bits it cannot.
2. Judge only levels where at least one policy delivers a scored update, and record the all-zero levels as
   NOT EVALUABLE with the bits that were carried. This changes what "every non-zero level" means.
3. Judge the mean over seeds per level rather than every seed, with the per-seed spread reported. At 1 % and
   0.1 % the mean is +0.027 for BAAC against 0.000 for all three baselines.

Options 2 and 3 both turn the current FAIL into a PASS, which is precisely why neither may be adopted now.

**Adaptive fidelity was considered as a way to win this on merit, and does not work.** The obvious mechanism
answer is a genuinely lower-fidelity belief increment that fits the 2,880-bit budget: the identity plus a
quantised point estimate plus a coarse uncertainty band, carrying enough for the receiver to hold a usable
revision. BAAC would then deliver a real update where raw, FIFO and fixed priority deliver nothing. It was
investigated and not built, because the binding constraint is the scoring model, not the wire format:

1. `Fidelity` is a frozen schema enum F0..F4 (`conrad/schemas/comms.py`), which this workstream may not edit,
   so no level below F1 can be added there.
2. `comm_oracle.belief_score` credits ANY view at `retained[1]`, whatever built it. A coarse update stored as
   a real belief view would therefore score exactly like a full 5,000-bit F1 delta: 0.7 current, 0.35 stale.
   That is an overclaim, it would inflate BAAC at every level and not just the failing ones, and it would
   flip criterion 3 by paying a roughly 400-bit point estimate the price of a full delta.
3. Storing it alongside the alert instead scores 0.000 on exactly the failing cells. `belief_score` credits
   an alert only while `alert_revision >= the sender's latest revision`, and **on seed 5500001 BAAC ends at
   revision 348 of 376 even at 100 % bandwidth, 24 times the 1 % budget** (measured, COM-I7-E003 and E004).
   If the receiver cannot be current on that belief with the whole link, a 400-bit update on a 12 bps link
   certainly cannot: at 1 % it takes about 33 s to serialise, so the last one that completes is tens of
   revisions behind. The same holds on development world 5100001, where BAAC ends at 355 of 383 at 100 %.

So the three possible landings are: overclaim, no effect, or add a scoring rule that credits the new level,
and only the third produces a pass. Its credit value would be the thing that decides the gate, chosen after
seeing the failure, and it is the same `STALE_CREDIT` asymmetry the 2026-09-20 pass already refused to change
for that reason. Nothing under `conrad/evaluation/oracle/` was touched.

This is a finding about the criterion meeting a belief that is revised 376 times in 240 s, not a defect that
can be coded around. The currency requirement (an alert counts only while it is not stale) and the mission
value of a fast-revising belief are in genuine tension at 1 % of the link, and resolving it is question 1
above, not an implementation task.

Two more declarations need the same treatment, for criterion 2, and for the same reason:

4. **The outage window was fixed and the finding time is not. REPAIRED, see the next section.** The question
   is closed: the window now follows the finding, which was a construction defect rather than a threshold.
5. **The surrogate and the Unity harness disagree about end-of-mission sync.** The Unity rule requires the
   receiver to hold at least the revision found during the outage; the surrogate test requires exact
   equality with the sender's newest revision. The Unity rule was declared on development seeds before any
   final world was built and is the better one, but adopting it for the surrogate now, after seeing that
   5500001 fails the stricter form, would again be changing the rule to fit the outcome.

Of the five, only number 4 is decided, and it is decided as a construction repair rather than a
reinterpretation (next section). The sequence for the rest is the same: decide, declare in
`configs/eval/i7_unity.yaml` and the acceptance test, then run on unused seeds.

## The three surrogate runs, side by side

Three runs, three disjoint final ranges, each run once after a freeze. Nothing was re-run on a spent range.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| experiments | COM-I7-E001 / E002 | COM-I7-E003 / E004 | COM-I7-E005 / E006 |
| final seeds | 5300000-5300004 (SPENT) | 5500000-5500004 (SPENT) | 5600000-5600004 (SPENT) |
| partition file | `partitions.yaml` mission | `partitions_i7_v2.yaml` | `partitions_i7_v3.yaml` |
| BAAC | pre-repair | repaired | repaired, unchanged from run 2 |
| outage construction | fixed [6 s, 60 s) | fixed [6 s, 60 s) | follows the finding |
| surrogate criteria passed | 4 of 4 | 2 of 4 | **3 of 4** |

Bandwidth sweep, mean retained over the 5 seeds of each run:

| level | run 1 BAAC | run 2 BAAC | run 3 BAAC | run 3 raw | run 3 FIFO | run 3 fixed | run 3 value-per-bit |
|---|---|---|---|---|---|---|---|
| 100 % | 0.628 | 0.626 | 0.634 | 0.217 | 0.217 | 0.155 | 0.577 |
| 50 % | 0.589 | 0.589 | 0.590 | 0.217 | 0.217 | 0.106 | 0.529 |
| 10 % | 0.330 | 0.261 | 0.251 | 0.217 | 0.217 | 0.011 | 0.211 |
| 1 % | 0.060 | 0.027 | 0.026 | 0.000 | 0.000 | 0.000 | 0.026 |
| 0.1 % | 0.060 | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0 % | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The three runs are on different worlds, so the BAAC columns are not a controlled comparison of the repair;
the controlled comparison is the development sweep above, where the same worlds were run before and after.
What the three runs do show is that the repaired BAAC clears raw, FIFO and fixed priority at 100 %, 50 % and
10 % on every seed of every run, and that the 1 % and 0.1 % tie is a property of the world draw: run 1 drew
five slow-revising critical beliefs and scored 0.060, runs 2 and 3 each drew two fast-revising ones.

**Criterion 2 is won on run 3, and it is won by the construction repair.** All 10 shadow flights made the
critical finding while the link was down (findings at 15.1, 16.1, 17.1, 54.1 and 75.1 s, the last two of
which the old fixed window would have missed or nearly missed), held a critical unit for the whole remaining
outage, delivered the F0 alert and the F1 delta ahead of every routine delta, and ended with the receiver at
the sender's latest critical revision at 100 % on all five seeds. Duplicate contributions and resync requests
were 0 in every arm of all 30 missions. Criterion 2 numbers are NOT comparable with runs 1 and 2, because the
scenario changed; that change was declared before run 3 and the defect and the seed it hit are named.

**Criterion 3 still fails on run 3, in three places, and the third one is new:**

1. **1 % on seeds 5600001 and 5600002** (critical belief revised 353 and 381 times): every policy 0.000.
   BAAC delivered the 184-bit alert at 1.0 s; raw, FIFO and fixed priority delivered nothing.
2. **0.1 % on the same seeds**: every policy 0.000. BAAC's alert arrived at 87.0 s and 108.0 s; the
   baselines sent 0 bits.
3. **Outage at 10 % on seed 5600001: BAAC 0.108 against FIFO 0.114.** This is a real loss of 0.006, not a
   tie, and it is a cost of the outage repair: that world's finding comes at 75.1 s, so the finding-following
   window runs [6 s, 120.1 s) and leaves about 120 s of a 120 bps link for 19 reported beliefs. The old fixed
   window would have scored that seed higher by not stressing it for as long. Reporting it rather than
   choosing the construction that flatters the number.

## The outage construction repair (2026-09-20)

**A stressor that can miss the event it exists to test is a construction defect.** `I7-OUTAGE-CRITICAL`
dropped the link over a FIXED `[6 s, 60 s)` window and relied on the lane pass reaching the lane-side defect
inside it. The finding time is a property of the world, not of the scenario: across the ten final seeds used
so far it ranges from 15.1 s to 68.1 s. On COM-I7-E004 seed 5500002 the patch first became visible at
66.25 s and the finding was made at 68.1 s, with the link already back for 8 s, so that world never exercised
the outage path at all. The criterion then failed for a world that was never actually stressed.

**The new construction**, declared in `conrad/sim/mission/scenarios.py` BEFORE the run that uses it:

- the outage opens at its declared start, 6 s, which is still before the robot can reach any view of the
  defect on any world seen so far;
- it stays DOWN until 45 s after the mission's first critical finding;
- capped at 120 s of outage, so a mission that never makes a finding still reconnects and the mission is
  never silently starved.

The 45 s hold is not a new number: `[6, 60)` left 44.9 s of outage after the 15.1 s finding on the worlds
where the fixed window did work, so the construction keeps the original declaration's own interval and only
stops the window from expiring before the event arrives. A world whose finding comes at 15.1 s reconnects at
60.1 s, which is the old behaviour to within 0.1 s; a world whose finding comes at 68.1 s now reconnects at
113.1 s instead of never having been stressed.

| File | Change |
|---|---|
| `conrad/communication/channel.py` | `LinkProfile.outage_follows_critical_finding`, `outage_hold_after_finding_s`, `outage_max_s`; `ChannelSim.hold_outage_until()` and `ChannelSim.outage_windows()`, which resolves the effective window. Armed once and idempotent, so a second call cannot move it. |
| `conrad/orchestration/mission_config.py` | The same three fields on `LinkConfig`. |
| `conrad/orchestration/comms.py` | `ShoreLink._arm_outage_hold()` arms the hold at the mission's FIRST critical offer with the identical value on every arm, so all five policies still share one link; `ShoreLink.effective_outages_s` exposes what actually happened. |
| `conrad/evaluation/decision_experiments/com_i7.py`, `com_i7_unity.py` | The scorer and the reconnection trace read the EFFECTIVE window, not the declared one. |
| `tests/unit/communication/test_finding_following_outage.py` | 6 tests, one of which pins the old defect (a fixed window is already back up at 68.1 s). |

Verified on the development worlds before the final run (4 flights, `sweep_outagefix.json`): the finding is
made during the outage on both worlds at both levels (15.1 s and 57.1 s), a critical unit is held for the
whole remaining outage, the F0 alert and F1 delta arrive ahead of every routine delta, duplicate
contributions and resync requests are 0, and BAAC stays strictly above raw, FIFO and fixed priority in all
four cells (0.660, 0.220, 0.572, 0.165 against a best baseline of 0.179, 0.158, 0.222, 0.114).

**Criterion 2 numbers before and after this change are not comparable.** On a world whose finding is late the
outage is now materially longer, which makes the scenario harder for every arm, and on world 5100001 at 10 %
the baselines drop (raw 0.142 to 0.085, FIFO 0.114 to 0.114) while BAAC holds at 0.165. The change was
declared before the run, and the defect and the seed it hit are named in the scenario file, in
`configs/eval/com_i7_e006.yaml` and here.

### What this does and does not mean for the gate

The formal path is unchanged and was not run: no Unity player was launched, `configs/eval/i7_unity.yaml`
still declares worlds 7800018 and 7800019, and `scripts/check_i7_unity_harness.py --wiring` still resolves all
24 declared flights. I7 stays BLOCKED_UPSTREAM behind I6, I5 and I4 whatever the surrogate says.

The repair is real and the development evidence for it is clean: criterion 3 now passes on both development
worlds at every level where any policy scores above zero, where before it lost outright at bandwidth 10 % and
at outage 10 % on world 5100001. The surrogate verdict is a different matter. **On the fresh final seeds
criterion 3 records FAIL**, because 2 of the 5 seeds tie every policy at 0.000 at 1 % and 0.1 %. The two
statements are both true and neither cancels the other: a mechanism defect was fixed on merit, and the gate
criterion as currently declared is not met on the final draw.

The recorded surrogate status is (`artifacts/gates/I7/evidence_surrogate.json`, re-recorded 2026-09-20
against COM-I7-E005 / E006 on seeds 5600000-5600004):

| criterion | surrogate status | why |
|---|---|---|
| full mission under constrained bandwidth | PASS | all 30 missions ran, every arm measured, 0 duplicate contributions |
| full mission under outages | **PASS** | all 10 shadow flights make the finding during the outage, hold it, deliver it first, and end in sync at 100 % |
| BAAC retains more mission-relevant information than raw/FIFO/fixed-priority | **FAIL** | all-policy 0.000 ties at 1 % and 0.1 %, plus a 0.006 loss to FIFO in the outage at 10 % on seed 5600001 |
| critical latency and sync error compared against baselines | PASS | every arm measured at every level; every case where BAAC is worse is listed in the evidence |

The record went 4 of 4 (run 1, pre-repair) to 2 of 4 (run 2) to **3 of 4** (run 3). One acceptance test is a
STRICT xfail whose reason quotes the measured failure, so both a silent improvement and a silent regression
break the suite: `test_baac_retains_more_than_raw_fifo_fixed_priority_every_nonzero_level`. The three
criterion 2 tests were strict xfails against run 2 and are now ordinary passing tests again; one of them,
`test_outage_finding_is_created_while_the_link_is_down`, had to stop comparing the finding time against the
DECLARED outage end, because that end no longer exists. It now checks the finding time against the declared
START and against `link_down_at_finding`, which is measured from the channel at the finding time and is
strictly stronger than the interval test it replaced.

`scripts/record_gate_evidence.py` was reading a strict xfail as NOT_RUN, because pytest reports it in the
junit XML as `<skipped type="pytest.xfail">`. That hid a failing criterion behind "no evidence", which is the
opposite of what HANDOFF section 1 rule 3 asks for. It now reads `pytest.xfail` as FAIL and keeps real skips
as NOT_RUN, and the artifact check can no longer upgrade a failed test node, only add its measured numbers.
No other gate's recorded criteria point at an xfailed node, so only I7's record changes.
