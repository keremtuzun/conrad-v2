# Conrad V2 remediation audit (2026-09-19)

This audit compares every claim in the first build's final report (ending commit `28871b4`, verdict
CONRAD_V2_SOFTWARE_PARTIALLY_COMPLETE) against the exact source criterion in
`docs/specification/source/ready_new_agent_v2.txt`. Where needed, it re-runs the evidence.

Official gate status comes from `artifacts/gates/<gate>/evidence_formal.json` and the dependency order in
`conrad/evaluation/gates.py` (`conrad gates status`). Surrogate evidence (the Python L1 kernel instead of Unity,
ADR-0008) is reported next to the gate, but it never promotes it.

## Rules applied

1. **Exact criteria.** A formal PASS needs the exact ch25 / ch26 pass criterion, on the formal execution path, on
   held-out data that was never used to design or tune the mechanism.
2. **A green test suite is not a passed gate.** Smoke, integration and gate-acceptance tests are separate.
   A test that encodes a gate currently failing is a strict xfail, and the Unity recorder reads an xfail as FAIL.
3. **A surrogate simulator is not Unity.** A Python-kernel result is `surrogate_evidence` only.
4. **A spent seed is not held-out evidence.** Every result seen before a repair makes its seeds
   DEVELOPMENT / CONTAMINATED_FOR_FINAL_EVALUATION. Each repair got a new digest-pinned final partition,
   declared before the run and run once:

   | Partition | Seeds | Status |
   |---|---|---|
   | MCBR `2026201` | | contaminated |
   | Mission final_test | 5300000-5300059 | spent by 2T-R2 and MCBR-E003 |
   | 2E FINAL | 6300000-6300011 | spent |
   | 2E FINAL-2 | 6400000-6400019 | |
   | 2T FINAL-3 | 6500000-6500059 | |
   | Former I2 NAV seeds | 7300001-7300006 | contaminated |
   | NAV final | 7400001-7400006 | `partitions_nav.yaml` |
   | I5 final | 7600000-7600009 | `partitions_i5.yaml` |
   | Unity gate worlds | 7800000-7800019 | `partitions_unity_gates.yaml` |
5. **Keep failures.** Failed research mechanisms are kept as recorded failures and switched off in production.
   Their thresholds are not relaxed.

## Gate registry corrections

The first build's registry left out three pass criteria that ch26 states. They are now in `gates.py`:

| Gate | Added criterion | ch26 source |
|---|---|---|
| I2 | faults reach defined safe states | Phase 6 |
| I5 | traceable decisions with low measured UIR; competitive mission outcomes vs decision baselines | Phase 9 |
| I7 | critical latency and sync error compared against baselines | Phase 11 |

## Gates: source criterion vs artifact

| Gate | Source criterion (ch25 / ch26) | Formal artifact | Official status |
|---|---|---|---|
| P0 | contract suite passes; owners review Interface V1 | `artifacts/gates/P0/evidence_formal.json`: 8 of 8 criteria, pytest nodes | PASS (software half). The Interface V1 owner review is a human action (Burak), not software |
| I0 | every boundary carries valid IDs/time/frames/provenance; run terminates and replays | `I0/evidence_formal.json` | PASS |
| C1 | all listed core behaviours and provenance work; comparisons recorded; no research win assumed | `C1/evidence_formal.json` (CORE-* experiments, PMBL keys) | PASS as *software completeness*. It does not mean the learned candidates win: learned association 0.669 vs NN 0.955; RBP killed (ADR-0006); the GRU wins only at a matched budget, and the runtime stays analytic (`TEMPORAL_RESULTS_RECONCILIATION.md`) |
| 2S-FIRST | observed/inferred/unknown distinct; ambiguity does not cause unjustified confidence | `2S-FIRST/evidence_formal.json` | PASS |
| U0 | correct force/motion directions, stable integration, correct frames/time, replay within tolerance | `U0/evidence_formal.json`, `docs/audits/U0_EVIDENCE.md` (Unity 6000.5.9f1 player, 23 live tests) | PASS |
| I1 | geometry, coverage, knowledge status, uncertainty, provenance exported without truth leakage | `I1/evidence_formal.json`, Unity player, fresh world 7800000 | PASS (9 of 9) |
| I2 | movement within configured tolerances using estimates; faults reach defined safe states | `I2/evidence_formal.json`, Unity, NAV seeds 7400001-7400006 | PASS: 7 of 7 criteria; NAV-005 0.072 m against a 0.15 m engineering-estimate bound; 5 of 5 fault cases reach the declared state and the vehicle stops |
| 2T | direct inference works before TCDP gets credit | `2T/evidence_formal.json` (2T-E00x-R3, FINAL-3) | PASS |
| 2T-TCDP | improved hidden-state error with controlled contamination | `2T-TCDP/evidence_formal.json` | FAIL: TCDP worsens corrosion by -0.110 mm [-0.116, -0.103]. Off in production (ADR-0009) |
| I3 | partial inspection maintains structural beliefs | `I3/evidence_formal.json`, Unity, world 7800001 | PASS (3 of 3); the earlier FAIL on 5300058 was fixed by ADR-0009 |
| I4 | viewpoint execution acquires evidence and improves measured belief; beats simple views under matched time/energy | `I4/evidence_formal.json`, Unity, 12 declared worlds 7800002-7800013, 48 sequential flights | FAIL: 3 of 5 criteria pass. Beats random +0.249 [+0.074, +0.453] and coverage-only +0.386 [+0.017, +1.009]; does NOT beat fixed views +0.066 [-0.043, +0.236], and loses to fixed on information per kJ. Surrogate E004 also FAIL |
| I5 | traceable constrained decisions, low measured UIR, competitive outcomes; invalid proposals cannot execute | formal: action matrix 7 of 7 PASS (1400/1400, UIR 0); mission criteria NOT_RUN formally; surrogate M1-ACTION-E003 | BLOCKED_UPSTREAM (I4). Surrogate: traceability/UIR PASS (3910/3910 traceable, UIR 0, 0 violations), competitive outcomes PASS (55/80 vs rule/FSM 54/80, naive 0/80), integrated actions FAIL (both nominal scenarios 0/10) |
| 2E | entity, field and persistent inference work | `2E/evidence_formal.json` (R4, FINAL-3 seeds 6600000-6600019) | PASS: entity, field and persistence. The 1-sensor over-caution was a depth-trend prior about 1.3x too wide; refit on DEV worlds to 0.078 degC/m, at a cost of +0.027 degC RMSE at 2 sensors |
| 2E-CEFD | coupling benefit without unsupported ecological inference | `2E-CEFD/evidence_formal.json` (R4) | FAIL: benefit -0.0011 [-0.0023, -0.00003] against uncoupled, so coupling is significantly WORSE; confident stress claims on healthy entities in 30 of 80 worlds. Off in production (ADR-0007) |
| I6 | three domain beliefs coexist on one world | surrogate I6-MULTIDOMAIN-E001, 3 declared worlds 7800014-7800016; the formal Unity harness is written and ready | BLOCKED_UPSTREAM (I5). Surrogate PASS on 3 of 3: every decision cites 2S+2T+2E, 0 bus rejections, every foreign-write probe rejected |
| I7 | useful receiver state persists across bandwidth sweep and outages; compare critical latency and sync error | surrogate COM-I7-E001/E002 | BLOCKED_UPSTREAM; surrogate PASS on 4 of 4 criteria. Value-per-bit beats BAAC on critical alert latency at 50% (1.00 s vs 2.38 s) |
| I8 | complete mission within real compute budgets with required fault recovery | none | BLOCKED_EXTERNAL (target onboard computer) |
| I9 | full loop runs physically in the stated envelope | none | BLOCKED_EXTERNAL (physical vehicle) |

### Unity I1/I3 re-run

**Worlds.** Both gates re-ran through the Unity player on fresh worlds from `configs/eval/partitions_unity_gates.yaml`,
pinned and committed (`4485dab`) before the run.

**I1** (world 7800000): **PASS, 9 of 9 criteria.**
- Movement: 15.9 m true path in 60 s, 600 commands accepted, 0 collisions.
- Map: 275 direct spatial revisions.
- Geometry: 206 occupied cells, median distance to the Twin2S surface 0.101 m, 100% within 0.5 m.
- Occupancy: 78 cells claimed free, 0% of them wrong.
- Knowledge status: 913 cells observed, 12 inferred, 3655 unknown; 500 of 500 points below the seabed are UNKNOWN.
- Uncertainty: mean U_O is 0.05 for observed cells and 0.85 for inferred or unknown.
- Provenance: 25 of 25 evidence items re-derive from the raw Unity frames.
- Leakage: 2230 runtime texts scanned, 0 violations.
- Replay: exact.

**I3** (world 7800001): **PASS, 3 of 3 criteria.**
- Association: 25 of 25 structural readings associated, 0 wrong.
- Partial view: 3 of 6 components observed; the far-side defect patch is never visible (maximum visible fraction 0.0).
- Beliefs: 25 direct Model2T revisions, all with evidence, maximum revision 8.
- Unseen components: every never-observed component is UNKNOWN with U_O 1.0 (or has no belief).
- Leakage: 0 violations.
- Replay: exact.

**The earlier I3 FAIL, on world 5300058.** A never-seen component carried an INFERRED condition. The cause was TCDP
running in the mission runtime. It was fixed by ADR-0009 (no relational propagation in production), and the gate
passed again on a world that was not used to diagnose the problem.

## Phases 0-16: source pass criterion vs artifact

| Phase | ch26 pass criterion | Evidence | Status |
|---|---|---|---|
| 0 Repository and contracts | P0 | P0 formal PASS; Interface V1 owner review not performed | COMPLETE (software); owner review OPEN (human) |
| 1 Fake full system | I0 | I0 formal PASS | COMPLETE |
| 2 Model2 core sandbox | C1 | C1 formal PASS; comparisons recorded; learned association and RBP lose and are off | COMPLETE as software; no research win |
| 3 Domain pairs begin | 2S first gate | 2S-FIRST formal PASS | COMPLETE |
| 4 Unity V2.0 | U0 | U0 formal PASS on the built player | COMPLETE (previous report: BLOCKED_EXTERNAL) |
| 5 Spatial intelligence and robot | I1 | Unity I1 formal PASS | COMPLETE |
| 6 Navigation and control | I2 | Unity I2 formal PASS incl. fault safe states | COMPLETE |
| 7 Structural intelligence | I3; TCDP research gate | 2T PASS; I3 formal PASS; 2T-TCDP FAIL | COMPLETE for I3; TCDP research gate FAILED and TCDP is off in production |
| 8 Active intelligence | I4 | formal Unity FAIL | INCOMPLETE. The predictive model now exists and the closed loop works (MCBR planned in 7 of 12 worlds; all 7 produced new target evidence; target OBSERVED in 12 of 12), but fixed views are unbeaten |
| 9 Model1 decision autonomy | I5 | action matrix formal PASS; integrated missions surrogate FAIL | INCOMPLETE. Over-escalation fell from 409 to 74 per 10 nominal missions; the remaining blocker is Model2T surface coverage (0.6-0.7), not Model 1 |
| 10 Ecological | I6; CEFD | 2E PASS; CEFD FAIL; I6 surrogate PASS, formal blocked by I5 | 2E COMPLETE; CEFD killed for production; I6 not formally integrated |
| 11 BAAC | I7 | surrogate PASS; formal blocked | INCOMPLETE on the formal path |
| 12 Hardware characterization | parameter evidence recorded; unresolved stays OPEN | software harness tested; no measurements | BLOCKED_EXTERNAL (physical) |
| 13 Parameter identification | held-out errors in envelope | software harness tested; no logs | BLOCKED_EXTERNAL (physical) |
| 14 HIL | I8 | host harness only | BLOCKED_EXTERNAL (execution environment) |
| 15 Controlled physical robot | I9 | none | BLOCKED_EXTERNAL (physical) |
| 16 Research validation | D8 only where evidence supports | no mechanism reaches D8 | NOT_EVALUABLE for D8; failures recorded |

## D-levels (Definition of Done ladder, ch "Definition of Done": D0 specified ... D8 research validated)

The level is the highest one whose evidence exists. The ladder is ordered: a subsystem cannot hold D5
(synthetic validated) without D4 (integrated), and D4 here means the subsystem is on the path of a gate that
formally PASSED through Unity.

| Subsystem | Previous | Corrected | Reason |
|---|---|---|---|
| Schemas / contracts | D5 | D4 | P0 and I0 PASS and on the formal I1/I2 path; no predeclared held-out validation applies to a contract |
| Twin2S | D5 | D4 | on the formal I1 path; no twin-validity experiment against a predeclared envelope |
| Unity V2 | D2 | D4 | U0, I1, I2 formal on the built player |
| Model2S / UAHSM | D4 | D4 | I1 formal |
| ECMER (spatial path) | D3 | D4 | on the formal I1 path; quality head untrained (DATA-REAL-SMOKE-E001) |
| PMBL, BUO, uncertainty | D4 | D4 | on the formal I1 path |
| State estimation | D3 | D4 | I2 formal; recalibrated (NAV-CAL-E001); the held-out calibration table predates the I2 hover change |
| Navigation / control / allocation / safety | D4 | D4 | I2 formal incl. faults. The previous D4 had no Unity evidence and no fault criterion |
| Twin2T | D4 | D4 | I3 formal; sensor model made realistic (`STRUCTURAL_LINEAGE_AUDIT.md`) |
| Model2T (direct, no propagation) | D4 | D5 | 2T PASS on held-out FINAL-3 against the gate thresholds, and I3 formal PASS. D5 covers corrosion and crack under the realistic synthetic sensor model only |
| TCDP | D3 | D3, killed for production | 2T-TCDP FAIL |
| MCBR | D3 | D3 | I4 formal FAIL. It beats random and coverage-only views in Unity, but not fixed views |
| Model1 / EGDC | D3 | D3 | action matrix formal PASS; integrated missions FAIL |
| Twin2E / Model2E | D3 | D5 | 2E PASS on held-out FINAL-3 against predeclared thresholds, synthetic worlds only. I6 is not formally integrated |
| CEFD | D3 | D3, killed for production | 2E-CEFD FAIL |
| BAAC | D4 | D3 | I7 surrogate only; no formal integration |
| RBP | D2 | D2, off | ADR-0006 |
| Learned association | D3 | D3, not used | loses to NN |
| HIL | D2 | D2 | host only |
| Full system | D4 | D3 | formal Unity integration reaches I3; I4-I7 are not formally integrated |

No subsystem is at D6 or above. The only real-data run is a smoke check (DATA-REAL-SMOKE-E001, UVVID), which
is not a validation.

## Flagship and replay

- **Previous flagship: not reproducible.** The previous report's flagship (`FLAGSHIP-I4-s2026201-3fc605f1`, crack
  78.5 -> 0.24 mm) came from commit `4478cd98`. It does not reproduce on the current code.
- **Replay bug.** Replay rebuilt the run options from the current defaults, not from the options stored in the
  bundle. Fixed.
- **Genuine code drift.** With the stored options, the run still diverges at the first thruster command, because
  the code changed.
- **Old crack figure.** The 0.24 mm figure also used the old optimistic structural sensor model: sizing was
  truth-accurate apart from Gaussian noise (`STRUCTURAL_LINEAGE_AUDIT.md`, VALID_BUT_OPTIMISTIC_SENSOR_MODEL).
- **Fresh flagship.** On current code (Python kernel) the fresh flagship bundle gives 26.76 mm crack error, and its
  replay is exact.
- **Truth-removal falsification: PASS on two fresh bundles.**
  - Bundles: one flagship (Python kernel) and one I1-UNITY.
  - With `truth/` moved away, events, 18 SQLite tables, decisions and commands replay bit-exactly, and no truth
    path is opened.
  - Positive controls: altering one depth reading or dropping one structural observation changes the output.
  - Old bundles could not be replayed from observations, because they did not capture the hardware-interface
    stream. New bundles do. See `FLAGSHIP_REPLAY_AND_FALSIFICATION.md`.

## Blockers, reclassified

| Blocker | Class | Note |
|---|---|---|
| EXT-HW-01..05 (driver, characterization, frame contract, identification logs, safety envelope) | PHYSICAL_EXTERNAL_INPUT / HUMAN_AUTHORITY (frame and safety contract acceptance) | Burak-owned |
| EXT-UNITY-01 | REMOVED | Unity 6000.5.9f1 compiled and running; this was software-owned, not external |
| EXT-HIL-01 | EXECUTION_ENVIRONMENT | target onboard computer |
| EXT-COMPUTE-01 | COMPUTE | research GPU for full-scale training |
| EXT-DATA-01 | split per dataset (`PUBLIC_DATA_PROCUREMENT.md`) | UVVID, SubPipe, SeaClear and Caves (NC-SA) APPROVED_WITH_RESTRICTIONS; AQUALOC, BenthicNet, FathomNet, Seaview and SUIM NEEDS_HUMAN_RIGHTS_REVIEW; UW-SLAM UNAVAILABLE (a human must confirm which dataset was meant) |
| P0 Interface V1 owner review | HUMAN_AUTHORITY | |

Everything else that stands between the current state and I3-I7 is **software-owned**:
- the I4 predictive model;
- EGDC over-escalation;
- the 2E temperature calibration;
- Unity support for the ecological twin, needed for formal I6;
- formal I7 through Unity.

## Failed research results kept

| Mechanism | Result | Decision |
|---|---|---|
| Learned association | 0.669 vs NN 0.955 | not used |
| RBP | 58% decoys confidently wrong | off (ADR-0006) |
| TCDP | worse than no propagation | off (ADR-0009) |
| CEFD | significantly worse than uncoupled; unsupported stress claims | off (ADR-0007) |
| MCBR in integrated missions | beats random and coverage-only in Unity, loses to fixed views | production planner frozen (v2); I4 FAIL |
| GRU temporal | wins at matched budget; not compared in the runtime regime | analytic runtime |
| ECMER quality head | untrained; reliability flat under real corruption | no reliability claim |

## What each open gate would take to pass

Honest routes only. None of these is "lower the threshold", re-run a spent seed, or report a surrogate as formal.

### I4 (FAIL): MCBR does not beat fixed views
The closed loop works: MCBR planned in 7 of 12 Unity worlds, all 7 produced new target evidence, and the
target ended OBSERVED in 12 of 12. It beats random (+0.249) and coverage-only (+0.386) but not fixed
(+0.066 [-0.043, +0.236]). Two causes, two honest routes:

1. **The world family does not exercise the gate premise.** The ch25 criterion begins "critical structure
   partly hidden". In `straight_pipeline` the nominal fixed route already includes an oblique far-side look,
   so nothing is hidden from it and 5 of 12 worlds tied outright. The fix is a world family where the defect
   is genuinely not visible from the nominal route (occlusion by supports, a defect under a clamp, a larger
   structure). This is legitimate ONLY if the family is declared and pinned before the run, the same design is
   used, and the old result is reported next to the new one. Choosing a family after seeing which one MCBR wins
   would be exactly the contamination this pass removed. Cost: about 3 to 5 h.
2. **Model2T's partial-view sizing limits the payoff.** The outcome hinges on a chosen view moving the crack
   estimate into the worst band. Better use of a single oblique look (locus refinement, per-cell rather than
   per-component sizing) raises the value of any view, including the fixed ones, so it may not change the
   comparison. Cost: about 4 to 6 h, uncertain.

**It may also be correct to kill it.** Phase 16 says "simplify mechanisms whose complexity adds no measurable
value". If MCBR cannot beat a fixed route on a family where the fixed route is adequate, the honest outcome is
that a fixed inspection route is the production default and MCBR stays EXPERIMENTAL.

### I5 (surrogate FAIL, blocked by I4)
The decision layer is no longer the blocker: UIR 0, every decision traceable, outcomes equal to the rule
baseline, over-escalation down from 409 to 74. Both nominal scenarios still score 0 of 10 because the
"continue" warrant needs 80% Model2T surface coverage and the mission reaches 0.6 to 0.7, and because a
pristine surface reads severity 0.061 against a 0.05 band. Route: more readings per view or per-cell coverage
credit in Model2T, and recalibrating the intact band against the realistic sensor model. Cost: about 2 to 4 h.
Then the formal run needs Unity missions (the harness exists) and I4 above it.

### I6 (surrogate PASS, blocked by I5)
Nothing is known to be wrong. The Unity Twin2E path and the formal harness are written; the player needs one
rebuild, then one command. Cost: about 1 h after I5 passes.

### I7 (surrogate PASS, blocked by I6)
The criteria all pass in the Python kernel. Formal evidence needs the same missions through Unity, which is
slower: about 4 to 6 h at full sweep, or about 2 h on a reduced sweep (3 bandwidth levels plus the outage),
which must be declared as reduced.

### 2T-TCDP and 2E-CEFD (FAIL)
Both mechanisms are measurably worse than not using them. Passing means a genuinely better mechanism, not a
threshold change. The recorded decision is to keep them OFF in production and treat the gates as failed
research, which the spec allows.

### I8, I9
Not software-passable: they need the target onboard computer and the physical vehicle.
