# I5 decision autonomy: the M1-ACTION-E001 action matrix (2026-09-19)

Spec ch25 Integration Gate I5 (L11835-11845) says Model1 must correctly use continue, request evidence, replan,
change sensing, return and escalate while hard constraints stay inviolable, and it names UIR as the measure.
Before this suite, the only evidence was UIR = 0 in single integrated missions, which says nothing about whether
the right action is chosen. M1-ACTION-E001 tests action choice directly, condition by condition.

Everything below is SYNTHETIC_ONLY belief-level fixtures. No perception, no Unity, no twin truth reaches EGDC.

## What was built

| item | path |
|---|---|
| scenario generator, expected-action table (spec line refs in the docstring), metrics, violation audit, invalid-proposal injection | `conrad/evaluation/decision_experiments/action_matrix.py` |
| config (final split, 10 seeds x 10 scenarios per condition) | `configs/eval/m1_action_e001.yaml` |
| dispatch entry | `conrad/evaluation/dispatch.py` (`M1-ACTION-E001`) |
| final artifact / dev artifact | `artifacts/experiments/M1-ACTION-E001/m1_action_e001.json`, `m1_action_e001_development.json` |
| acceptance test (reads the stored final artifact and fails if it is missing) | `tests/acceptance/test_i5_action_matrix.py` |
| unit tests for the new semantics | `tests/unit/decision/test_action_semantics.py` |
| gate evidence | `scripts/record_gate_evidence.py` (`PLAN["I5"]`), `artifacts/gates/I5/evidence_formal.json` |

Conditions: correct, missing, stale, credible contradiction (fresh and after the attempts are used up), wrong
association, high U_E (with and without an alternate modality), high U_O, high U_A (first time and after one
repeat), cross-domain disagreement, uncalibrated source, blocked route, resource (battery and time budget),
unhealthy (fault, leak, motion not permitted), link down with a critical finding, and invalid proposals injected
straight into the ConstraintEngine (21 kinds). Each scenario has an expected action set. A choice is correct
only if it is in that set. REQUEST_INFORMATION is scored with its question type, so a request for the wrong kind
of evidence counts as wrong.

Splits: `configs/eval/partitions.yaml` is digest-pinned and has no decision-matrix domain, so the module keeps
local disjoint seed lists: development 7100000-7100009, final test 7300000-7300009. Access goes through
`partitions.check_access`, so a tuning code path cannot read final seeds (unit-tested). All policy fixes were
made against development scenarios only.

## Final-split results (1400 scenarios per arm, 100 per condition, plus 210 injected invalid proposals)

| metric | EGDC (structured) | naive baseline |
|---|---|---|
| correct (chosen action in the expected set) | 1400 / 1400 = 1.000 | 130 / 1400 = 0.093 |
| UIR | 0.000 (0 of 632 relied world claims unsupported) | 0.058 (200 of 3436) |
| hard-constraint violations, audit of chosen actions | 0 | 300 (200 unsupported-as-fact, 100 stale support) |
| injected invalid proposals accepted | 0 / 210 | 30 / 210 (grounding checks off in that arm) |
| safe rate / useful rate / safe and useful | 1.000 / 1.000 / 1.000 | 0.286 / 0.093 / 0.093 |
| abstention rate | 0.161 | 0.104 |

Per-class canonical recall for EGDC (the I5 criteria). The floor is 0.9, an ENGINEERING_ESTIMATE, because the
spec leaves the bound OPEN.

| class | canonical scenarios | recall | times chosen | precision |
|---|---|---|---|---|
| continue | 100 | 1.00 | 100 | 1.00 |
| request evidence | 770 | 1.00 | 770 | 1.00 |
| replan | 100 | 1.00 | 100 | 1.00 |
| change sensing | 50 | 1.00 | 50 | 1.00 |
| return | 170 | 1.00 | 170 | 1.00 |
| escalate | 80 | 1.00 | 98 | 1.00 |
| report (store-and-forward) | 100 | 1.00 | 100 | 1.00 |
| hold (motion not permitted) | 30 | 1.00 | 12 | 1.00 |

EGDC confusion matrix (rows are the expected class, columns the chosen class; empty cells are 0). The only
cross-class entry is 18 ESCALATE choices under "motion not permitted", which are in that row's expected set
(WAIT, ESCALATE, QUERY_BELIEF, CHANGE_SENSOR_MODE).

| expected \ chosen | continue | req. evidence | replan | change sensing | return | escalate | report | hold |
|---|---|---|---|---|---|---|---|---|
| continue | 100 | | | | | | | |
| request evidence | | 770 | | | | | | |
| replan | | | 100 | | | | | |
| change sensing | | | | 50 | | | | |
| return | | | | | 170 | | | |
| escalate | | | | | | 80 | | |
| report | | | | | | | 100 | |
| hold | | | | | | 18 | | 12 |

The naive baseline continues in every belief-fault condition (correct only on CORRECT) and escalates under
resource, health and link faults (correct only in the 30 motion-not-permitted scenarios). Per-action recall is
also in the artifact. REVISIT_REGION and ABORT_MISSION are never chosen: EGDC always prefers the cheaper QUERY_BELIEF
or RETURN_TO_SAFE_STATE when both are acceptable.

The development split gives the same picture: EGDC 1400/1400, UIR 0, 0 violations; naive 0.093.

## Baseline before the fixes (development seeds, same generator)

EGDC as it was scored 0.736 correct, with 30 constraint findings on a 4-scenario-per-condition probe:

- **replan 0/40.** REPLAN was only proposed after information attempts ran out, so a blocked route was never replanned (the robot continued).
- **change sensing 0/20.** After one repeat failed to reduce U_A, it escalated. CHANGE_SENSOR_MODE was discounted by the same repeat count as REQUEST_INFORMATION.
- **time-budget return 0/20.** Nothing looked at `ResourceState.time_remaining_s`.
- **store-and-forward 0/40.** A pending critical finding was worth less than continuing.
- **uncalibrated source 9/40.** Mostly ESCALATE. This is the "ESCALATE dominates integrated runs" defect from `docs/research/EXPERIMENT_RESULTS_2026-09-18.md`. The uncalibrated-source floor pushes U_E over its threshold, and with one modality that read as OOD with no autonomous path.

## Fixes (all deterministic, all in `conrad/decision`, constraints only added)

1. **Replan on a blocked route** (`route.py`, `claims.py`, `actions.py`). A fresh, evidence-backed, well-observed SPATIAL belief with `occupied=True` on a leg of `MissionState.notes["planned_route"]` becomes a GROUNDED claim with a BLOCKS edge to CONTINUE. REPLAN(reason=ROUTE_BLOCKED) relies on that claim. Continuing gets risk `blocked_route_risk`. Obstacles that are uncertain, lack evidence or are stale do not trigger a replan (unit-tested).
2. **Change sensing after a repeat** (`consequence.py`). CHANGE_SENSOR_MODE is valued at `resolvability * (1 - decay^repeats) * decay^switches`. The order is now repeat or improve the measurement first, then change modality, as in ch16 L6899-6904.
3. **Time reserve** (`config.py`, `constraints.py`, `consequence.py`). New hard constraint `TIME_BELOW_RESERVE` (`ConstraintConfig.time_reserve_s`, default 120 s, an ENGINEERING_ESTIMATE). It is active only when `time_remaining_s` is known. It also triggers the retreat value. The engine version is now `constraint-engine-0.3`.
4. **Report a finding** (`consequence.py`). TRANSMIT or STORE_AND_FORWARD with intent REPORT_FINDING is worth its requirement's consequence times `report_value`. The requirement is matched by belief ID or registry identity. The value decays after a report of that belief has already been issued, so the mission resumes (unit-tested).
5. **Over-escalation** (`graph.py`, `claims.py`, `actions.py`, `consequence.py`):
   - A U_E excess caused only by the uncalibrated floor is flagged `calibration_only_epistemic`. It gets an autonomous CONFIRM_CONDITION request in any modality (resolvability key `CALIBRATION_CHECK`, 0.8), and ESCALATE follows only once attempts are exhausted. Genuine OOD without an alternate modality still escalates (unit-tested).
   - Operator value is multiplied by `operator_value_autonomous_factor` (0.4) while every open item still has an autonomous path.
   - CONTINUE's mission term counts each open consequential requirement as -consequence, not 0 (ch17 L7681-7695).

No constraint was removed or relaxed. The ConstraintEngine stays outside utility and is never injectable.

Regression checks:

- M1-UIR-E001 re-run into a temporary folder reproduces the stored summary exactly: EGDC UIR 0, safe 0.909; naive UIR 0.188, safe 0.506.
- `tests/unit/decision`, `tests/property/decision`, `tests/leakage` and the I5 acceptance tests pass: 82 passed.
- The integrated-mission acceptance tests pass: `test_flagship_mission.py` and `test_golden_suite.py`, 19 passed.

## Gate evidence

`scripts/record_gate_evidence.py I5` wrote `artifacts/gates/I5/evidence_formal.json`. All seven criteria are PASS:
continue, request evidence, replan, change sensing, return, escalate, hard constraints inviolable. Each is backed
by an acceptance-test node plus a check on the stored final artifact. `evaluate_gates()` reports I5
`formal_status=PASS` and `official_status=BLOCKED_UPSTREAM` (blocked by I4), which is the correct state.

## Limitations (read before citing these numbers)

- **The evidence covers only half of the I5 formal path.** The path is "action-matrix suite on imperfect beliefs + integrated mission". This evidence covers the action matrix only, yet the script records it as FORMAL. How often each action is used inside integrated missions was not measured here, including whether escalation now stops dominating them.
- **Same authors, same templates.** The expected sets, the generator and the policy fixes come from the same author. Final seeds differ from development seeds, but the scenario templates are shared, so 1.000 on final shows the fixes generalise across seeds and magnitudes, not across unseen scenario structure.
- **Design readings, not spec statements:**
  - UNCALIBRATED should confirm, not escalate. Miscalibration is listed as an injected fault (ch16 L6825) but is not an escalation trigger in ch17 L7968-7979.
  - With high U_A, repeat first and change modality afterwards.
  - With high U_E and an alternate modality, use the alternate evidence rather than the operator.
  - Motion not permitted excludes RETURN from the useful set.
- **The naive baseline is deliberately weak** (M1-UIR-E001's act-on-claims arm). Stronger ch16 baselines (M1-B0 FSM, B2 rule system, B3 utility controller) are not run yet.
- **Unmeasured thresholds.** The 0.9 recall floor, `time_reserve_s`, `blocked_route_risk`, `report_value`, `operator_value_autonomous_factor` and the `CALIBRATION_CHECK` resolvability are ENGINEERING_ESTIMATE values, not measurements.
- **Final sample size changed once.** The final split was first run at 6 scenarios per condition per seed (EGDC 840/840 correct, 0 violations). It was re-run at 10 so that every canonical class has at least 50 scenarios. The policy did not change between the two runs.

## Integrated missions (M1-ACTION-E002, 2026-09-19)

The matrix above tests action choice on belief fixtures. M1-ACTION-E002 tests the same actions inside full
integrated missions: sensing, Model2S/2T/2E, the Belief Bus, EGDC, MCBR, navigation, safety and BAAC, on the python
L1 kernel. **This is SURROGATE evidence (not Unity).** It is recorded in `artifacts/gates/I5/evidence_surrogate.json`
and cannot promote the formal gate.

### The planned-route producer (REPLAN was unreachable in missions)

`route.py` reads the route from `MissionState.notes["planned_route"]`, but nothing in a mission ever wrote it, so
REPLAN could not fire. The runtime now publishes it every decision cycle:

1. `MissionExecutive.planned_route()` (`conrad/orchestration/executive.py`) takes the navigation stack's own
   trajectory for the active motion goal, starts it at the EKF estimate and cuts the next 6 m into 1 m corridor legs
   (WORLD boxes, half width 0.45 m). Station keeping has no route. No truth is read.
2. `Deliberation.route_context()` (`conrad/orchestration/deliberation.py`) asks Model2S, at cell level, whether a leg
   holds OBSERVED occupied cells that the surveyed registry design and the charted seabed do not explain. Model2S
   publishes 2.8 m map blocks, far coarser than a corridor, so block overlap alone would flag the seabed or the pipe.
   The result goes into `notes["route_leg_occupancy"]` (leg index to the SPATIAL block beliefs holding those cells),
   and those block messages join the decision snapshot.
3. `route.py` accepts this second belief form: a Model2S block (claim `occupancy.observed`, status OBSERVED) blocks a
   leg only when that leg lists it. The freshness, evidence, conflict and U_O checks are unchanged, and the boolean
   `occupied` form used by the matrix is untouched (unit tests in `tests/unit/decision/test_route_leg_occupancy.py`).
4. `DecisionRouting` sends REPLAN(ROUTE_BLOCKED) to `MissionExecutive.replan_detour()`, which re-routes the transit over
   the blocked legs (climb, pass, descend). The navigation stack, safety supervisor and command gateway still check
   every command.

The runtime also fills `ResourceState.time_remaining_s` from a mission time budget (`MissionRuntimeConfig.time_budget_s`,
default the MissionSpec's 1800 s), so the time reserve can be exercised in a mission.

### Suite

| item | path |
|---|---|
| experiment | `conrad/evaluation/decision_experiments/m1_action_integrated.py`, dispatch ID `M1-ACTION-E002` |
| config (final seeds, thresholds fixed before the final run) | `configs/eval/m1_action_e002.yaml` |
| seed partition (new versioned file, digest-pinned) | `configs/eval/partitions_i5.yaml`, `conrad/evaluation/partitions.py` (`I5_DOMAIN`) |
| scenarios (appended only) | `conrad/sim/mission/scenarios.py`: `I5-NOMINAL`, `I5-CRITICAL-FINDING`, `I5-UNCERTAIN-BELIEF`, `I5-ROUTE-BLOCKED`, `I5-BATTERY-RESERVE`, `I5-TIME-RESERVE`, `I5-COMMS-OUTAGE` |
| unregistered lane obstacle (truth side) | `conrad/sim/mission/obstacles.py`, `MissionWorldOptions.lane_obstacle`, one call in `run.prepare` (world.py untouched) |
| final artifact | `artifacts/experiments/M1-ACTION-E002/m1_action_e002.json` |
| acceptance test | `tests/acceptance/test_i5_integrated_missions.py` |

Each scenario makes one action class warranted. The warrant onset is the first decision whose own DecisionContext
shows it, because Model 1 can only act on the belief it has:

| scenario | class | onset (first decision where ...) | expected |
|---|---|---|---|
| I5-NOMINAL (zero-size defect) | continue | critical component OBSERVED INTACT, no report pending | CONTINUE_MISSION |
| I5-CRITICAL-FINDING (lane-side defect) | escalate / operator alert | critical component OBSERVED DEGRADED or worse, link up | ESCALATE_TO_OPERATOR or TRANSMIT_INFORMATION(REPORT_FINDING) |
| I5-UNCERTAIN-BELIEF (hidden far-side defect) | request evidence | critical condition not OBSERVED | QUERY_BELIEF, REQUEST_INFORMATION, REVISIT_REGION |
| I5-ROUTE-BLOCKED (1.0 x 1.4 x 1.0 m box on the lane) | replan | Model2S shows OBSERVED occupied cells on a planned leg | REPLAN(ROUTE_BLOCKED) |
| I5-BATTERY-RESERVE (LOW_POWER fault at 30 s) | return | battery below the 0.2 reserve | RETURN_TO_SAFE_STATE or ABORT_MISSION |
| I5-TIME-RESERVE (150 s budget) | return | time remaining below the 120 s reserve | RETURN_TO_SAFE_STATE or ABORT_MISSION |
| I5-COMMS-OUTAGE (link down 6 to 80 s, lane-side defect) | store-and-forward | a finding is OBSERVED while the link is down | STORE_AND_FORWARD(REPORT_FINDING) |

A mission counts as correct when the warrant arose, the expected action came within 4 s of onset (two decision
cycles), no scenario-forbidden action came in between (for example CONTINUE on a blocked route, TRANSMIT on a down
link, anything but retreat below a reserve), the independent M1-ACTION-E001 audit found no hard-constraint violation
in any decision, and (nominal only) nothing escalated, returned or aborted. UIR uses `conrad.decision.uir` on the same
basis as E001. A decision is traceable when every supporting claim is in its record, every relied world claim names
a belief that is in the snapshot with provenance, and the record matches its snapshot and provenance IDs.

Arms, on the same seeds:

- `egdc_structured` drives the mission.
- `rule_fsm` (new, evaluation only): a fixed-precedence rule system from the ch16 M1-B0/B2 family. Retreat below a
  reserve, then report a pending finding, then replan a blocked route, then request information while a requirement
  is open, then continue. It never escalates. The ConstraintEngine still checks it.
- `naive_act_on_claims`: the E001 baseline, unchanged.

Each baseline drives its own closed-loop mission, for mission outcomes. Both also rank EGDC's own decision contexts as
open-loop shadow arms (same basis as E001).

Seeds: `partitions_i5.yaml` (version `partitions-i5-missions-2026-09-19-v1`), development 7500000-7500009, final
7600000-7600009. On load it checks that these are disjoint from `partitions.yaml`, `partitions_nav.yaml` and the
ranges it lists as used elsewhere (6300000-6300011, 6400000-6400019, 7100000-7100009, 7300000-7300009). The final
range was declared before any I5 mission ran on it. Only I5 development seeds 7500000-7500002 and mission development
seeds 5100000-5100001 (`partitions.yaml`, used for the first single-mission probes) were used while building. Final
ran once: 210 missions, 2484 s wall.

Thresholds were fixed before the final run and are ENGINEERING_ESTIMATE values, because ch25/ch26 leave the I5 bounds
OPEN: success floor 0.9 per scenario, UIR at most 0.0, latency budget 4 s, contact clearance 0.3 m. "Competitive" means
EGDC's pooled task success is at least each baseline's, its safety events (transitions into a safety state worse than
DEGRADED) are at most each baseline's, and it has no more hard-constraint violations.

### Development runs

- 7500000, built with an earlier harness (the nominal onset did not yet require an empty report queue, and every
  DEGRADED episode counted as a safety event). EGDC was correct in 6/7 scenarios, each at 0 s latency. Nominal failed
  with 22 over-escalations.
- 7500001 and 7500002, with the final harness. EGDC: nominal 0/2 (54 over-escalations), critical finding 0/2 and
  store-and-forward 0/2 (the warrant never arose: EGDC never observed the defect), uncertain belief 2/2, route
  blocked 2/2 (latency at most 2 s), battery 2/2, time 2/2. Task success: EGDC 6/14, rule_fsm 9/14, naive 2/14.

The harness changes made after these runs were the nominal onset (no pending report), the safety-event definition
(DEGRADED episodes reported separately) and adding `correct_given_warrant` to the summary. Nothing in `conrad/decision`
was tuned.

### Final results (7600000-7600009, 10 missions per scenario per arm)

EGDC, closed loop:

| scenario | class | warrant reached | expected action within 4 s | correct | latency mean / max (s) | forbidden after onset | violations | over-escalations | ESCALATE decisions |
|---|---|---|---|---|---|---|---|---|---|
| I5-NOMINAL | continue | 0/10 | 0/10 | 0/10 | - | 0 | 0 | 409 | 409 |
| I5-CRITICAL-FINDING | escalate | 9/10 | 9/10 | 9/10 | 0.0 / 0.0 | 0 | 0 | 0 | 235 |
| I5-UNCERTAIN-BELIEF | request evidence | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 220 |
| I5-ROUTE-BLOCKED | replan | 10/10 | 10/10 | 10/10 | 0.4 / 2.0 | 0 | 0 | 0 | 234 |
| I5-BATTERY-RESERVE | return | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 76 |
| I5-TIME-RESERVE | return | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 81 |
| I5-COMMS-OUTAGE | store-and-forward | 9/10 | 9/10 | 9/10 | 0.0 / 0.0 | 0 | 0 | 0 | 151 |
| all | | 58/70 | 58/70 | 58/70 | 0.1 / 2.0 | 0 | 0 | 409 | 1406 |

Whenever the warrant arose, EGDC issued the expected action within the budget (correct given warrant = 1.0 in all six
scenarios where it arose). The misses are one critical-finding and one store-and-forward mission where the finding was
never observed, and all ten nominal missions.

Correct missions per arm:

| arm | NOMINAL | CRITICAL-FINDING | UNCERTAIN-BELIEF | ROUTE-BLOCKED | BATTERY | TIME | COMMS-OUTAGE | all |
|---|---|---|---|---|---|---|---|---|
| egdc_structured, closed loop | 0/10 | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 58/70 |
| rule_fsm, closed loop | 0/10 | 10/10 | 10/10 | 8/10 | 10/10 | 10/10 | 9/10 | 57/70 |
| naive_act_on_claims, closed loop | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/70 |
| rule_fsm, shadow on EGDC contexts | 0/10 | 9/10 | 10/10 | 8/10 | 10/10 | 10/10 | 9/10 | 56/70 |
| naive_act_on_claims, shadow on EGDC contexts | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/70 |

Traceability, UIR and hard constraints:

| arm | decisions | traceable | UIR (relied unsupported / relied world claims) | hard-constraint violations |
|---|---|---|---|---|
| egdc_structured | 3310 | 3310 | 0.0 (0/23) | 0 |
| rule_fsm | 3310 | 3310 | 0.0 (0/23) | 0 |
| naive_act_on_claims | 4200 | 4200 | 0.074 (16322/220711) | 82112 (16322 unsupported-as-fact, 65790 stale support) |
| naive_act_on_claims, shadow | 3310 | 3310 | 0.072 (16224/224782) | 49802 |

Mission outcomes (closed loop, 70 missions per arm):

| arm | task success | NOMINAL | CRITICAL | UNCERTAIN | ROUTE | BATTERY | TIME | COMMS | safety events | findings delivered | ESCALATE per mission | energy J (mean, without the battery scenario) | path m (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| egdc_structured | 57/70 | 0/10 | 9/10 | 10/10 | 8/10 | 10/10 | 10/10 | 10/10 | 11 | 49 | 20.1 | 17253 | 23.1 |
| rule_fsm | 57/70 | 0/10 | 10/10 | 10/10 | 7/10 | 10/10 | 10/10 | 10/10 | 13 | 49 | 0.0 | 18322 | 24.6 |
| naive_act_on_claims | 15/70 | 0/10 | 5/10 | 5/10 | 0/10 | 0/10 | 0/10 | 5/10 | 20 | 30 | 13.8 | 15302 | 20.1 |

Task success is scenario-level. NOMINAL: inspected INTACT with no escalation or safe hold. CRITICAL and COMMS: the
finding reached the shore receiver. UNCERTAIN: the critical condition is OBSERVED at the end. ROUTE: passed the
obstacle with more than 0.3 m clearance. BATTERY and TIME: retreat within budget and safe hold entered. The two EGDC
route misses cleared the box by 0.30 m and 0.24 m. The robot in contact settles at 0.23 m, so the 0.30 m case failed
only on the threshold.

### Gate status

`scripts/record_gate_evidence.py` now lists all ten I5 criteria. These include the ch26 Phase 9 criteria
"traceable decisions with low measured UIR" and "competitive mission outcomes vs decision baselines".

- FORMAL (`evidence_formal.json`): the seven matrix criteria PASS. The three mission criteria are NOT_RUN because the
  formal path is integrated missions through Unity and none exists.
- SURROGATE (`evidence_surrogate.json`): the seven matrix criteria PASS. "Actions exercised correctly inside integrated
  missions" FAILS. "Traceable decisions with low measured UIR" PASSES. "Competitive mission outcomes" PASSES.
- `conrad gates status`: I5 official BLOCKED_UPSTREAM (by I4), formal NOT_RUN, surrogate FAIL. **I5 is not passed.**
- `tests/acceptance/test_i5_integrated_missions.py` encodes the failed claim as a strict xfail with the measured
  reason.

### Findings

1. **Over-escalation in the nominal mission is the blocker.** In 10 nominal missions EGDC chose ESCALATE_TO_OPERATOR
   409 times, about 41 of 60 decisions per mission. rule_fsm chose it 0 times. A diagnosis on development seed 7500000
   shows the mechanism. At 2 s and 4 s EGDC chooses REQUEST_INFORMATION. Routing defers both requests while the lane
   survey runs (`DEFERRED_UNTIL_LANE_SURVEY_COMPLETE`), so neither is executed, but both count as attempts in the
   decision history. At 6 s the repeat-decayed information value of a third request (0.18) scores -0.020. ESCALATE,
   with its autonomous-path discount, scores -0.004. ESCALATE wins by 0.016. Two fixes to evaluate on development
   seeds: count only executed information requests as attempts, or stop the scalar utility from ranking ESCALATE
   above an available autonomous path. Neither was made here.
2. **The nominal continue warrant never arose, in any arm.** With the defect set to zero size, no arm ever got an
   OBSERVED condition for the critical component. rule_fsm asked for information in all 60 decisions of every nominal
   mission. So "continue" was not tested inside a mission. The scenario needs a readable intact surface, for example
   the `REGION_READINGS` path in `world.py`, which is off. That was out of scope here because world.py belongs to
   another workstream.
3. **REPLAN now works in missions.** The warrant arose in 10/10 missions and REPLAN(ROUTE_BLOCKED) followed within
   2 s in 10/10. At onset EGDC chose REPLAN in 8 missions. In 2 it first transmitted a pending finding, then replanned
   one cycle later. The detour executed in every mission. rule_fsm, which ranks reporting first, took 22 s in 2
   missions. The naive arm never replanned and came within 0.23 m of the box (contact) in all 10.
4. **REPLAN in a mission with no obstacle.** One nominal mission (seed 7600004) had 3 REPLAN(ROUTE_BLOCKED) decisions
   and no scripted obstacle. Clutter rocks are unregistered, so this may be a real rock near an inspection path or map
   noise. The cause was not established.
5. **UIR has a small base.** EGDC relied on only 23 world claims in 3310 decisions. UIR = 0 is correct but thinly
   exercised, because most mission actions are information or report actions that rely on no world claim.
6. **Warrants depend on sensing.** In the critical-finding and store-and-forward missions on the final seeds, EGDC
   first had the finding at 16 s (from the lane pass) on 5 seeds and at 64 s (after an inspection) on 4. On seed
   7600000 it never had it. On development seeds 7500001 and 7500002 the lane pass never saw the defect, and EGDC,
   busy escalating, never inspected in time.

### Limitations

- SURROGATE only: python L1 kernel, not Unity. It cannot promote I5.
- Other workstreams were editing `conrad/domains/technical/*`, `conrad/sim/mission/world.py`, `sensing.py` and
  `run.py` (replay capture) during the run. The final used the code on disk at 17:58 on 2026-09-19. Replay capture
  was switched off in this harness (`prepare(..., capture=False)`) because the new capture path failed in
  `finish()`.
- The onsets, expected sets, task-success definitions and the rule_fsm baseline come from the same author as the
  harness. The expected sets follow the E001 table. The mission-level definitions are design readings.
- Energy in the battery scenario is dominated by the LOW_POWER fault, which sets the energy used, so it is left out
  of the energy mean.
