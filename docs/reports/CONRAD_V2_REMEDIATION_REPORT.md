# Conrad V2: strict remediation, falsification and completion pass

> **2026-09-26 completion addendum:** This document preserves the 2026-09-20
> remediation record below. The current artifact-derived closure report is
> `docs/reports/SOFTWARE_COMPLETION_2026_09_26.md`, and the current generated
> gate table is `docs/reports/CONRAD_V2_GATE_TABLE.md`. Later negative results
> remain negative: I4 is FAIL, Spatial V1.1 I5 formal is 7/10 FAIL, I6 formal
> evidence is 3/3 PASS but blocked upstream, and I7 is surrogate 3/4 FAIL with
> formal NOT_RUN.

Date: 2026-09-20. Repository: `C:\Users\Kerem\Kerem\conrad-v2`, branch `main`, no remote.
Start of this pass: `28871b4`. Detail: `docs/audits/CONRAD_V2_REMEDIATION_AUDIT.md` and the per-topic audits
in `docs/audits/`.

## A. New repository

- Same repository as the first build, continued: 29 further commits on `main`.
- **Digital Twin (legacy) was not modified.** Checked again at the end of this pass: HEAD `6bf01ee`,
  `git status --porcelain` returns 0 lines.
- Nothing was imported or copied from the legacy repository.

## B. Legacy salvage

Unchanged from the first build: no legacy subsystem was ported. The salvage ledger and its classifications
(3 REIMPLEMENT_FROM_CONCEPT, 7 REFERENCE_ONLY, 15 OBSOLETE, 6 REJECT_ARCHITECTURAL_CONFLICT) stand, and a
contract test now checks the ledger rather than leaving it as prose.

## C. Legacy independence

**Re-executed on 2026-09-21 at the end of this pass: PASS** (`artifacts/migration/legacy_independence_2026-09-21.json`).
The legacy repository was renamed away, Conrad V2 was clean-cloned into a temporary directory with no caches
and no virtualenv, and every step ran with the legacy repo unreachable:

| Step | Result |
|---|---|
| `uv sync --all-groups --locked` | OK, 41.0 s |
| `ruff format --check` | OK, 709 files |
| `ruff check` | OK |
| `mypy conrad tests` | OK, 0 issues in 621 source files |
| tests (quick profile) | 162 passed |
| `conrad db migrate`, `conrad doctor` | OK |
| simulation smoke, replay smoke, core training smoke | OK; replay REPRODUCED |

The rename is restored in a `finally` block. Verified afterwards: the legacy repository is at HEAD `6bf01ee`
with 0 modified files, exactly as it was before this pass began.

## D. Requirements coverage

Statuses are now DERIVED from files, retained experiment artifacts and formal gate status; hand-typed
INTEGRATED / EVALUATED / VALIDATED values are ignored. Coarse rows were split (REQ-SCHEMA-001 into a-g,
REQ-GATE-001 into a-d), and each row may name the gate that decides it.

| Status | Count |
|---|---|
| TESTED | 71 |
| INTEGRATED (formal gate PASS) | 9 |
| EVALUATED (experiment artifact retained) | 8 |
| VALIDATED | 0 |
| BLOCKED_EXTERNAL | 2 |
| BLOCKED_OPEN_DECISION | 4 |
| MISSING / PARTIAL / IMPLEMENTED-only | 0 |

VALIDATED stays 0: there is no physical evidence, and the only real-data run is a smoke-level experiment.

## E. Phase status (0 to 16)

| Phase | Status | Basis |
|---|---|---|
| 0 Contracts | COMPLETE (software); owner review OPEN | P0 formal PASS |
| 1 Fake full system | COMPLETE | I0 formal PASS |
| 2 Model2 core | COMPLETE as software; no research win | C1 formal PASS; learned association and RBP lose and are off |
| 3 Domain pairs | COMPLETE | 2S-FIRST formal PASS |
| 4 Unity V2 | COMPLETE | U0 formal PASS on the built player (previously reported BLOCKED_EXTERNAL) |
| 5 Spatial | COMPLETE | I1 formal PASS through Unity |
| 6 Navigation and control | COMPLETE | I2 formal PASS incl. fault safe states |
| 7 Structural | COMPLETE for I3; TCDP research gate FAILED | 2T and I3 formal PASS; 2T-TCDP FAIL |
| 8 Active | INCOMPLETE | I4 formal FAIL |
| 9 Model1 | INCOMPLETE | action matrix PASS; integrated missions FAIL |
| 10 Ecological | 2E COMPLETE; CEFD killed; I6 not formally integrated | 2E PASS, 2E-CEFD FAIL, I6 surrogate PASS |
| 11 BAAC | INCOMPLETE | I7 surrogate FAIL (3 of 4); criterion 3 remains strict-fail and formal was not run |
| 12 Hardware characterization | BLOCKED_EXTERNAL | software harness complete and rehearsed |
| 13 Parameter identification | BLOCKED_EXTERNAL | protocol, intake, runner and a synthetic rehearsal exist |
| 14 HIL | BLOCKED_EXTERNAL | host harness only |
| 15 Physical robot | BLOCKED_EXTERNAL | |
| 16 Research validation | NOT_EVALUABLE for D8 | failures recorded and mechanisms killed |

## F. Gates

Official status is derived by `conrad gates status` from `artifacts/gates/<gate>/evidence_*.json` and
dependency order. FORMAL means the gate's own execution path (Unity for I1-I7). SURROGATE means the Python
L1 kernel; it never promotes a gate (ADR-0008).

| Gate | Official | Formal | Surrogate |
|---|---|---|---|
| P0, I0, C1, 2S-FIRST, U0 | PASS | PASS | |
| I1 | PASS | PASS (9 of 9, world 7800000, re-recorded after the safety fix) | PASS |
| I2 | PASS | PASS (7 of 7, NAV seeds 7400001-7400006, incl. 5 fault cases) | |
| 2T | PASS | PASS (FINAL-4, 6700000-6700059) | |
| 2T-TCDP | FAIL | FAIL, killed | |
| I3 | PASS | PASS (3 of 3, world 7800001, re-recorded after Model2T R4) | |
| I4 | FAIL | FAIL (4 of 5, 30 worlds 8002200-8002229, 150 Unity flights) | FAIL |
| I5 | BLOCKED_UPSTREAM | action matrix 7 of 7 PASS; mission criteria NOT_RUN | FAIL |
| 2E | PASS | PASS (FINAL-3, 6600000-6600019) | |
| 2E-CEFD | FAIL | FAIL, killed | |
| I6 | BLOCKED_UPSTREAM | PASS (3 of 3, worlds 7800014-7800016) | PASS |
| I7 | BLOCKED_UPSTREAM | not run | FAIL (3 of 4, seeds 5600000-5600004) |
| I8, I9 | BLOCKED_EXTERNAL | | |

I7's 2026-09-23 final software-gate verdict is **I7 FORMAL = FAIL**, fail-closed before Unity because the
strict surrogate-development prerequisite for criterion 3 was not met. The evidence distinction remains
important: formal Unity evidence is NOT_RUN and no `evidence_formal.json` exists; official status remains
BLOCKED_UPSTREAM. See `docs/audits/I7_CRITERION3_FAILURE_ANALYSIS.md`.

**I4, the gate that received the most work.** Three formal runs across two world families, the first of which
rested on a generator defect that left the hidden defect visible from the lane. Final run: 4 of 5 criteria
PASS, including "beats coverage-only" at +0.1345 [+0.0234, +0.2535], which both earlier runs failed. The gate
fails on the information criterion, which requires six pairings: five pass and information per kJ against
random views is +0.0377 [-0.00021, +0.0749], missing zero by 0.00021. The rule was fixed before the run, a CI
spanning zero is a tie, and a tie is a FAIL.

The substantive finding stands: the production system reads the defect in 16 of 30 worlds against a systematic
sweep's 11, using fewer views (1.77 against 2.60), less redundancy (0.97 against 2.17), less travel and less
energy. A control arm running the same frozen planner with the execution repair disabled scores 0.3220 against
0.4291, and the ranking code is byte-identical between them, so the gain is the execution repair rather than
the planner or the family. The repair itself: the routing layer had no busy guard on the mission-executive
branch and detoured only transit goals, so the approach to any view that could resolve the defect was
cancelled. V3 accepted 103 views and flew 48.

## G. Maturity (D0 to D8)

The ladder is ordered, and D4 means the subsystem sits on a gate that formally passed through Unity.

| Level | Subsystems |
|---|---|
| D5 synthetic validated | Model2T (direct, no propagation), Model2E |
| D4 integrated | schemas/contracts, Twin2S, Twin2T, Unity V2, Model2S/UAHSM, ECMER spatial path, PMBL/BUO/uncertainty, state estimation, navigation/control/allocation/safety |
| D3 baseline validated | MCBR, Model1/EGDC, BAAC, TCDP (killed), CEFD (killed), learned association (not used), full system |
| D2 unit validated | RBP (off), HIL |

Nothing is at D6 or above: the only real-data result is a smoke-level experiment.

## H. Tests and checks

Commands run on the working tree at the commit named in each gate's evidence:

```
python -m uv run ruff format --check .
python -m uv run ruff check .
python -m uv run mypy conrad tests
python -m uv run pytest -q --ignore=tests/unity_live
```

- Format and lint: clean.
- mypy: 0 errors across 596 source files. It had regressed to 75 errors, all in tests added during this pass;
  they were fixed as real typing defects (narrowing asserts, correct annotations, documented casts), with no
  ignores added and no test weakened. One Unity assertion was strengthened in the process.
- pytest: **1373 passed, 1 skipped, 4 xfailed** in 15 min 17 s. The Unity-live modules are excluded from that
  command because they launch the player; they are run by the gate recorders instead.
- The 4 strict xfails are the recorded gate failures, each quoting its measured reason: two for I4, one for
  the I5 integrated-actions criterion and one for I7 criterion 3. The Unity recorder reads an xfail as FAIL,
  and `scripts/record_gate_evidence.py` was fixed during this pass to do the same: it had been reading a
  strict xfail as NOT_RUN, so a failing criterion could appear as absent evidence.
- The 1 skip is the Unity-bundle falsification, which reports itself inconclusive when the source has changed
  since the bundle was recorded rather than claiming a pass.

Test kinds are separated: smoke, integration, gate acceptance. A test that encodes a currently failing gate is
a strict xfail, and the Unity recorder now reads an xfail as FAIL rather than as a skip.

## I. Training

Unchanged: only CPU smoke and experiment-scale runs. The temporal reconciliation
(`docs/audits/TEMPORAL_RESULTS_RECONCILIATION.md`) showed the GRU beats hold-last at a matched budget
(0.364 vs 0.751 at q=0.02); the earlier "GRU loses" figure was undertraining. The runtime default stays
analytic, because no runtime-regime comparison exists.

## J. Experiments

Every experiment in this pass ran once on a digest-pinned, previously unused final partition, declared before
the run. Spent partitions: MCBR seed 2026201, mission final 5300000-5300059, 2E FINAL and FINAL-2, the former
I2 NAV seeds 7300001-7300006 and the first I5 final seeds.

Wins: Model2T direct inference (crack +5.81 mm [4.93, 6.72] over latest-observation at level 0, coverage
0.950), Model2E fields and entities (all calibrated on FINAL-3), Model1 action matrix (1400/1400, UIR 0).
BAAC is not a gate win: the current I7 surrogate record is 3 of 4 and criterion 3 fails its strict
every-seed/every-nonzero-level rule. The 2026-09-23 development repair removes the genuine outage loss to
FIFO without mid/high-bandwidth regression, but leaves all-policy zero-retention ties at 1 % and 0.1 %.

Failures kept: TCDP, CEFD, RBP, learned association, MCBR against fixed views.

## K. Real-data status

- **Procurement audited per dataset** (`docs/audits/PUBLIC_DATA_PROCUREMENT.md`): UVVID, SubPipe, SeaClear and
  Underwater Caves are APPROVED_WITH_RESTRICTIONS; AQUALOC, BenthicNet, FathomNet, Seaview and SUIM need human
  rights review; UW-SLAM does not exist under that name.
- **Downloaded with the user's explicit approval:** UVVID sample (10.8 MB) and SubPipeMini2.zip (4.95 GB,
  CC BY 4.0, MD5 verified against Zenodo).
- **DATA-REAL-SMOKE-E001 (UVVID):** engineered quality features move as expected under corruption on 100% of
  48 real frames; the ECMER quality head is untrained and its reliability barely moves, so no reliability claim.
- **DATA-REAL-E002 (SubPipe):** a linear probe on untrained ECMER evidence detects pipeline presence in
  side-scan sonar with balanced accuracy 0.81-0.84 on held-out time blocks (majority baseline 0.50). One
  mission, so this is not a generalisation claim. Model2S was not run: the dataset publishes no ground-truth
  pose or sensor mounting geometry, and inventing them is forbidden.
- **No D6.** RealOnly vs Twin+Real comparison is still NOT_EVALUABLE.

## L. Integrated mission and falsification

- **The previous flagship does not reproduce.** See section Q.
- **Replay was repaired:** it rebuilt run options from current defaults instead of the bundle's stored options.
- **Bundles now capture the hardware-interface stream,** so a run can be replayed from observations alone.
- **Truth-removal falsification: PASS.** With `truth/` moved away, a Python-kernel flagship bundle and a Unity
  I1 bundle replay bit-exactly, and no truth path is opened. Controls: corrupting truth changes nothing;
  altering one depth reading or dropping one structural observation changes the output; an edited recording
  with stale digests is refused.
- **New Unity flagship with outage, contradiction and localization degradation:** `FLAGSHIP-UNITY` on held-out world 7800017, 150 s, production defaults, all three stressors declared before
the run (`docs/audits/FLAGSHIP_UNITY.md`).

- **Outage:** link down 6 to 70 s. The critical finding was made at 14.3 s while the link was down, queued,
  and delivered after reconnection: alert at 71.3 s, belief update at 77.0 s. Receiver in sync with the
  sender, 0 duplicates, 0 resync requests.
- **Contradiction:** two credible disagreeing structural readings drove U_C from 0.350 to 1.000. The estimate
  swung and the variance stayed wide; the belief was not averaged away.
- **Localization degradation:** fix outage at 90 s, DEGRADED at +12.0 s, LOCALIZATION_LOST and HOLD at
  +23.6 s.
- **Structural result under the realistic sensor model:** corrosion 13.587 mm against 6.003 mm truth
  (error 7.584 mm), crack 70.61 mm against 80.00 mm (error 9.397 mm). No inspection goal was ever accepted:
  all four MCBR plans reported the need satisfied and three revisit goals were refused by the planner.
- **Replay:** CC-10 on the integrated run, 13 of 13 items identical (505 observations, 266 evidence items,
  1346 revisions, 75 decisions, the MCBR candidate tables, 1500 commands, BAAC deltas, receiver state,
  terminal status), Unity trajectory difference 0.0.
- **Truth-removal falsification: PASS.** 1500 ticks, 0 divergences, no truth path opened and no truth module
  imported.
- **A safety defect it exposed, since repaired:** with the fix lost, the configured safe-hold action was
  station keeping against a dead-reckoned estimate. The vehicle chased its own drift at 0.207 m/s, its true
  position left the mission boundary at 122.6 s and a collision-envelope violation followed. A HOLD now never
  closes a control loop on an estimate declared untrustworthy, and the boundary is judged on the 3-sigma
  worst case. Re-measured on the same condition: zero authorized thrust for 371 steps, true speed 0.045 m/s,
  0 of 1500 steps outside the boundary.

## M. Safety

- Gateway rejections, safe-hold behaviour and the adversarial suite stand as before.
- **New in this pass:** the I2 fault criterion. Five injected faults (loss of localization, thruster failure,
  stale state, low battery, leak) reach their declared safety state in Unity, and the vehicle actually stops
  where a stop is required. Dev runs exposed a real gap first: the supervisor changed state but the vehicle
  kept executing its last command.
- **Estimator honesty:** a regression introduced by the I2 hover fix (loss declared at 36.3 s) was found and
  fixed (29.94 s, max error 3.25 m against a 4 m bound).

## N. External blockers

| ID | Class | Needed |
|---|---|---|
| EXT-HW-01 | PHYSICAL | vehicle driver |
| EXT-HW-02 | PHYSICAL | measured characterization |
| EXT-HW-03 | HUMAN_AUTHORITY | frame contract acceptance |
| EXT-HW-04 | PHYSICAL | identification logs (software side now complete: protocol, intake, runner, rehearsal) |
| EXT-HW-05 | HUMAN_AUTHORITY | safety envelope |
| EXT-HIL-01 | EXECUTION_ENVIRONMENT | target onboard computer |
| EXT-COMPUTE-01 | COMPUTE | research GPU |
| EXT-DATA-01 | DATA_LICENCE, per dataset | see section K |
| P0 Interface V1 | HUMAN_AUTHORITY | owner review |
| EXT-UNITY-01 | REMOVED | it was software-owned; Unity 6000.5.9f1 is compiled and running |

## O. Research claims

No mechanism is validated on real data or physically, so no mechanism exceeds "candidate implemented and
evaluated in synthetic experiments". Model2T direct inference and Model2E reach D5 on held-out synthetic
worlds against predeclared thresholds. TCDP, CEFD and RBP are killed for production. MCBR is retained as
EXPERIMENTAL with a frozen production configuration and a failed integration gate.

## P. Verdict

The remediation program completed the software implementation and validation
work to its declared stopping rules. Spatial V1.1 A-J is PASS for synthetic
software, while I4, I5, and I7 remain honest negative research results. No
physical, HIL, or deployment claim is made. See the 2026-09-26 completion
report for the final decision table, clean-clone proof, exact newer evidence,
and physical-only remainder.

## Q. Corrections to the previous report

| Item | Previous status | New status | Reason | Source criterion | Evidence |
|---|---|---|---|---|---|
| U0 | BLOCKED_EXTERNAL ("Unity never compiled") | PASS | Unity was software-owned, not external. The project was upgraded to 6000.5.9f1, NetMQ removed, a dependency-free TCP bridge added, and a headless player built | ch26 Phase 4 | `U0_EVIDENCE.md`, 23 live tests |
| I1 | PASS (Python kernel) | PASS (FORMAL, Unity) | The previous PASS came from a surrogate simulator, so it could not be formal | ch25 I1 | `artifacts/gates/I1`, world 7800000 |
| I2 | PASS ("NAV-001 to NAV-008") | PASS (FORMAL, Unity), after a FAIL | First formal run FAILED: NAV-005 station RMS 0.1534 m > 0.15 m. Cause: EKF velocity prior kept position sigma above the fix noise at hover. Also the registry was missing the ch26 fault criterion | ch25 I2 + ch26 Phase 6 | `artifacts/gates/I2`, NAV seeds 7400001-7400006 |
| I3 | PASS (Python kernel) | PASS (FORMAL, Unity), after a FAIL | First formal run FAILED: a never-observed component carried an INFERRED condition, because the mission runtime ran Model2T with TCDP | ch25 I3 | `artifacts/gates/I3`, world 7800001; ADR-0009 |
| I4 | NOT_EVALUABLE ("thresholds OPEN") | FAIL | The criterion is explicit ("beat simple views under matched time/energy"), so it is evaluable. MCBR beats random and coverage-only but not fixed views | ch25 I4, ch26 Phase 8 | `artifacts/gates/I4`, 48 Unity flights; `MCBR_REEVALUATION.md` |
| I5 | PARTIAL | Action matrix PASS; integrated missions FAIL | "PARTIAL" is not a status. The action matrix is formal and passes; the integrated-mission criterion fails | ch25 I5, ch26 Phase 9 | `I5_ACTION_MATRIX.md`, M1-ACTION-E001/E003 |
| I6 | PASS (functional) | BLOCKED_UPSTREAM, surrogate PASS | The previous PASS had no Belief Bus cross-domain reasoning evidence and no formal path; it also cannot outrank a failed I4/I5 | ch25 I6 | `I6_MULTI_DOMAIN.md` |
| I7 | FAIL ("outage path never exercised") | BLOCKED_UPSTREAM, surrogate PASS on 4 of 4 | The outage path now runs in every seed and the critical finding is delivered after reconnection. Still not formal | ch25 I7, ch26 Phase 11 | `I7_COMMUNICATIONS.md`, COM-I7-E001/E002 |
| C1 | "PASS for analytic operators only" | PASS, meaning software completeness | C1 is a software-completeness gate. It never implied a research win, and the learned candidates still lose | ch26 Phase 2 | `artifacts/gates/C1` |
| 2T (functional) | not separated | PASS after two failed iterations | The first FINAL run FAILED (crack lost to latest-observation, coverage outside band). Fixed by a sensor-datasheet likelihood, missed-detection handling, a regime-switching crack filter and coverage geometry | ch25 2T | `MODEL2T_REPAIR.md`, FINAL-3 |
| 2T-TCDP | implied by "Phase 7 COMPLETE" | FAIL, killed for production | TCDP makes corrosion worse than no propagation (-0.110 mm [-0.116, -0.103]) | ch25 2T research part | ADR-0009 |
| 2E | "Phase 10 COMPLETE" | PASS after two failed iterations | First runs FAILED (entity coverage 0.849 < 0.85; then a 1-sensor calibration floor). Fixed by station filters, change detection, an observability/coupling split and a refit depth-trend prior | ch25 2E | `MODEL2E_REPAIR.md`, FINAL-3 |
| 2E-CEFD | "about zero benefit" (prose) | FAIL, killed for production | Now measured: coupling is significantly WORSE (-0.0011 [-0.0023, -0.00003]) and makes stress claims on healthy entities in 30 of 80 worlds | ch25 2E research part | ADR-0007 |
| Flagship crack 78.5 -> 0.24 mm | headline result | WITHDRAWN | It came from commit `4478cd98` and does not reproduce. Two causes: replay rebuilt run options from current defaults, and the code has genuinely drifted. The number also used an optimistic structural sensor model | ch25 I4 / flagship | `FLAGSHIP_REPLAY_AND_FALSIFICATION.md` |
| Structural sensor model | truth-accurate sizing plus Gaussian noise | replaced with a realistic model | No truth LEAK was found, but sizing was optimistic: it now has visibility-dependent detection, a 20% per-sensor bias, 25% scatter and wall loss | leakage rules | `STRUCTURAL_LINEAGE_AUDIT.md` |
| MCBR seed 2026201 | reported as evidence with a caveat | CONTAMINATED, never reused | The visibility formula was tuned after inspecting that seed | evaluation discipline | `MCBR_REEVALUATION.md` |
| Estimator | "overconfident: 5.8 m error with sigma < 1.5 m" | repaired and calibrated | 18-state EKF with a velocity-mismatch state, adaptive IMU noise and NIS consistency. Held-out: missed danger 1.71 s -> 0 s, normalized error 0.82 -> 0.27, NAV-007 max error 5.82 -> 3.09 m | ch26 Phase 6 | `ESTIMATOR_CALIBRATION.md` |
| Requirements ledger | 75 TESTED, 1 INTEGRATED, 0 EVALUATED (hand-set) | derived: 71 TESTED, 9 INTEGRATED, 8 EVALUATED | EVALUATED now requires a retained artifact and INTEGRATED requires a formal gate PASS | evidence discipline | `scripts/build_requirements_ledger.py` |
| Unity blocker EXT-UNITY-01 | external blocker | removed, was software-owned | Misclassification: no external input was needed | blocker classification | section N |
| Real data | "no rights-cleared datasets" | 4 datasets cleared, 2 downloaded with approval, 2 real-data experiments run | The audit was per dataset rather than blanket | ch26 Phase 16 / data rules | `PUBLIC_DATA_PROCUREMENT.md` |
| Maturity table | Twin2S and schemas at D5; BAAC and full system at D4 | corrected, see section G | D4 now requires a formally passed integration gate; D5 requires held-out synthetic validation against predeclared thresholds | Definition of Done ladder | section G |
