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

## Iteration 2 (M1-ACTION-E003, 2026-09-19)

E002 failed the criterion "actions exercised correctly inside integrated missions" on two counts: EGDC chose
ESCALATE_TO_OPERATOR 409 times in 10 nominal missions, and the nominal continue warrant never arose. This
iteration diagnosed both on the I5 DEVELOPMENT seeds (7500000-7500009 only), fixed what belongs to the decision
plane, re-ran the action matrix, and ran the frozen design once on fresh final seeds as M1-ACTION-E003.

### Causes

1. **Deferred requests counted as attempts.** Routing defers every MCBR / navigation action while the lane
   survey runs (`DEFERRED_UNTIL_LANE_SURVEY_COMPLETE`) and silently drops one while an inspection is already
   running, but the decision history recorded those choices like any other. Each one decayed the information
   value of the next request (`attempt_decay^tries`), so by the third cycle a request scored below ESCALATE.
2. **The scalar utility ranked the operator above an untried autonomous path.** Even with executed-only
   counting, after two executed requests the repeat-decayed request scored -0.020 against ESCALATE at -0.004
   (the E001 fix discounted operator value by `operator_value_autonomous_factor` 0.4, which was not enough).
   ch17 L7968-7979 lists the escalation triggers; none of them is "the next look is worth less than the last
   one", and the E001 design reading was that the operator is not the first resort while an autonomous path
   remains.
3. **The uncalibrated-source gap could never close.** Model2T never marks its uncertainty calibrated, so every
   critical technical requirement carried `UNCALIBRATED_SOURCE`, which raises effective U_E to the 0.45 floor,
   which keeps the item open forever. `actions.py` already said "an independent confirming look (any modality)
   closes it", but nothing implemented the closure, so EGDC confirmed, confirmed again, and then escalated as a
   dead end. CONTINUE could not be chosen for a critical component in any mission, whatever the belief said.
4. **Cross-domain disagreement was evaluated against every context belief in the snapshot.** A 2S block around
   another component with coverage below 0.5 marked the critical requirement as contradicted, permanently.
5. **The nominal continue warrant needs 80 % surface coverage.** With Model2T iteration 3 (coverage geometry,
   region readings ON) a component-level condition is OBSERVED only once 0.8 of its cells have been read, or
   once the read part already reaches the worst band. In I5-NOMINAL a view yields ONE averaged reading, so one
   cell per second at best: 0.8 is unreachable, and with a zero-size defect the worst-band shortcut never
   fires. That is why no arm ever saw OBSERVED INTACT.

### Fixes (decision plane, all deterministic)

1. **Executed attempts only** (`context.py`, `consequence.py`, `deliberation.py`, `routing.py`).
   `DecisionSummary` gains `executed` (and `target_revisions`); `DecisionContext.attempts_on` skips entries with
   `executed=False`. `Deliberation.mark_not_executed()` is called by `DecisionRouting._route` when it defers a
   decision during the lane survey or drops one because an inspection is already running. Nothing else changed
   in routing: proposal is still not permission.
2. **The operator is not a resort while an autonomous path remains** (`consequence.py`):
   `operator_value_autonomous_factor` default 0.4 to 0.0. ESCALATE still gets its full (and boosted) value once
   an open high-consequence item is a dead end: information attempts exhausted, or OOD without an alternate
   modality.
3. **Calibration gap closure** (`claims.py`, `_confirmed_after_request`): the uncalibrated epistemic floor is
   dropped for a requirement whose belief now has a NEWER REVISION than the one an executed information request
   on it saw, with every required property OBSERVED, evidence present and no evidence conflict. Deferred
   requests and history without recorded revisions never confirm anything, so
   `test_uncalibrated_source_escalates_once_attempts_are_exhausted` is unchanged.
4. **Requirement-local cross-domain context** (`cross_domain.py`): context beliefs associated with the
   requirement's registry component are used when they exist, otherwise only context beliefs whose support
   overlaps the requirement region.
5. **A readable nominal scenario** (`scenarios.py` `I5-NOMINAL-READABLE`, appended; `options.py`, `world.py`,
   `structure.py`): `StructuralSensorOptions.region_tiles` (default None) splits the target's non-defect surface
   into 8 x 8 truth-side tiles, so one view yields a reading per visible tile instead of one averaged point, and
   `DefectOptions.pristine_rest` (default False) starts that surface with no corrosion and no crack (in
   I5-NOMINAL it is sampled from the population priors, about 1 mm). Both defaults are off, so every other
   scenario builds exactly the world it built before.
6. **Replan bookkeeping** (`routing.py`): a REPLAN(ROUTE_BLOCKED) during a non-transit goal is recorded as
   `mode: HOLD`, and the harness counts it as `replan_holds` instead of an executed detour.

No constraint was removed or relaxed. The ConstraintEngine is untouched.

### The stray REPLAN(ROUTE_BLOCKED) of E002

Reproduced on development seed 7500006 (I5-UNCERTAIN-BELIEF): at 38 s and 42 s, while an INSPECT goal was
running, a planned leg held 3 OBSERVED occupied cells. The truth check says those cells sit 0.09-0.11 m from a
**clutter debris entity** (unregistered, so neither the design nor the charted seabed explains it) and 0.86 m
from the design. The REPLAN was therefore a correct detection of a real object on the route, not map noise:
nothing to fix in the route logic. In 60 development missions of the six E002 scenarios it happened in 1.

What the REPLAN then did is worth recording: `MissionExecutive.replan_detour` only detours a TRANSIT goal and
holds anything else, so the inspection was abandoned, MCBR planned the same blocked view again, and the loop
repeated. The E002 artifact counted those holds as "replans executed". They are now labelled `HOLD` and counted
separately (fix 6). A detour for inspection approaches would need a two-leg goal (GO_TO then STATION_KEEP) in
the executive, which is another workstream's file, and is left OPEN.

### Development numbers (7500000-7500009, EGDC arm)

- Nominal over-escalation, same scenario and seeds, before / after fixes 1-4: 43 to 0 on seed 7500003; over 8
  development nominal missions the mean fell from about 41 per mission to 1.6.
- E001 action matrix on its development split after every fix: 1400/1400 correct, UIR 0, 0 violations,
  0/210 injected invalid proposals accepted (unchanged from iteration 1).
- The six non-nominal scenarios on 3 development seeds: battery 3/3, time 3/3, uncertain belief 3/3, route
  blocked 3/3, critical finding 2/3, store-and-forward 2/3 (the finding was never observed on seed 7500001).
- I5-NOMINAL-READABLE on 8 development seeds: surface coverage reached 0.61-0.86 and 1 of 8 missions reached an
  OBSERVED condition. A trial with 240 s and 10 plans per need did not raise coverage past 0.8 and added
  dead-end escalations, so the scenario keeps the 120 s runtime of I5-NOMINAL.

### Seeds

`configs/eval/partitions_i5_v2.yaml` (version `partitions-i5-missions-2026-09-19-v2`, digest-pinned in
`conrad.evaluation.partitions` as `I5_V2_PARTITIONS_SHA256`, domain `i5_mission_v2`). Development 7500000-7500009
(the v1 development split, shared on purpose); final test **7900000-7900009**, declared before any run touched
it. On load it checks disjointness from `partitions.yaml`, `partitions_nav.yaml`, `partitions_i5.yaml` (whose
final seeds 7600000-7600009 are SPENT) and `partitions_unity_gates.yaml` (7800000-7810019), plus the reserved
ranges 5300000-5300059, 6300000-6300011, 6400000-6400019, 6500000-6500059, 7100000-7100009, 7300000-7300009 and
7400001-7410020. Nothing in `configs/`, `conrad/`, `scripts/`, `tests/`, `docs/` or `artifacts/experiments/` used
79xxxxx before this file. The v1 file stays loadable and `partitions_i5.yaml` is unchanged.

### Final results (M1-ACTION-E003, 7900000-7900009, 8 scenarios x 10 seeds x 3 arms, 240 missions, 3729 s)

EGDC, closed loop:

| scenario | class | warrant reached | expected action within 4 s | correct | latency mean / max (s) | forbidden after onset | violations | over-escalations | ESCALATE decisions |
|---|---|---|---|---|---|---|---|---|---|
| I5-NOMINAL | continue | 0/10 | 0/10 | 0/10 | - | 0 | 0 | 36 | 36 |
| I5-NOMINAL-READABLE | continue | 0/10 | 0/10 | 0/10 | - | 0 | 0 | 38 | 38 |
| I5-CRITICAL-FINDING | escalate | 8/10 | 8/10 | 8/10 | 0.0 / 0.0 | 0 | 0 | 0 | 24 |
| I5-UNCERTAIN-BELIEF | request evidence | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 |
| I5-ROUTE-BLOCKED | replan | 10/10 | 9/10 | 9/10 | 2.2 / 20.0 | 0 | 0 | 0 | 19 |
| I5-BATTERY-RESERVE | return | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 |
| I5-TIME-RESERVE | return | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 |
| I5-COMMS-OUTAGE | store-and-forward | 8/10 | 8/10 | 8/10 | 0.0 / 0.0 | 0 | 0 | 0 | 12 |
| all | | 56/80 | 55/80 | 55/80 | 0.4 / 20.0 | 0 | 0 | 74 | 129 |

Correct given warrant is 0.98 (55/56): the single miss is one route-blocked mission that replanned 20 s after
onset, over the 4 s budget. The other 24 misses are missing warrants: all 20 nominal missions, plus 2
critical-finding and 2 store-and-forward missions where the defect was never observed.

Correct missions per arm:

| arm | NOMINAL | NOMINAL-READABLE | CRITICAL | UNCERTAIN | ROUTE | BATTERY | TIME | COMMS | all |
|---|---|---|---|---|---|---|---|---|---|
| egdc_structured, closed loop | 0/10 | 0/10 | 8/10 | 10/10 | 9/10 | 10/10 | 10/10 | 8/10 | 55/80 |
| rule_fsm, closed loop | 0/10 | 0/10 | 8/10 | 10/10 | 8/10 | 10/10 | 10/10 | 8/10 | 54/80 |
| naive_act_on_claims, closed loop | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/80 |
| rule_fsm, shadow on EGDC contexts | 0/10 | 0/10 | 8/10 | 10/10 | 8/10 | 10/10 | 10/10 | 8/10 | 54/80 |
| naive_act_on_claims, shadow | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/80 |

Traceability, UIR and hard constraints:

| arm | decisions | traceable | UIR (relied unsupported / relied world claims) | hard-constraint violations |
|---|---|---|---|---|
| egdc_structured | 3910 | 3910 | 0.0 (0/109) | 0 |
| rule_fsm | 3910 | 3910 | 0.0 (0/38) | 0 |
| naive_act_on_claims | 4800 | 4800 | 0.483 (18310/37939) | 34169 |
| naive_act_on_claims, shadow | 3910 | 3910 | 0.475 (18366/38703) | 30130 |

Mission outcomes (closed loop, 80 missions per arm):

| arm | task success | safety events | findings delivered | ESCALATE per mission | energy J (mean, without the battery scenario) | path m (mean) |
|---|---|---|---|---|---|---|
| egdc_structured | 55/80 | 12 | 50 | 1.61 | 16417 | 21.7 |
| rule_fsm | 55/80 | 13 | 50 | 0.00 | 17345 | 22.6 |
| naive_act_on_claims | 12/80 | 21 | 36 | 12.54 | 16297 | 20.5 |

Task success per scenario, in the table order above: EGDC 0, 1, 8, 10, 8, 10, 10, 8; rule_fsm identical;
naive 0, 0, 2, 8, 0, 0, 0, 2.

### Over-escalation: what is left

74 ESCALATE decisions in 20 nominal missions (E002: 409 in 10), a fall from about 41 to 3.7 per mission.
13 of the 20 missions never escalate; 2 missions account for 40 of the 74. The earliest escalation in any
nominal mission is at 44 s, after the lane survey and after executed inspections, never during the survey, so
the deferred-request cause is gone. Every remaining escalation happens once the information attempts on an open
critical item are used up (`max_information_attempts` 3, or `max_plans_per_need` 4 in the runtime), which is the
ESCALATE trigger the design intends. It is still counted as over-escalation, because in a nominal mission the
robot should be continuing instead, and it cannot continue while the critical condition stays UNKNOWN.

### Gate status

- FORMAL (`evidence_formal.json`): unchanged. The seven matrix criteria PASS; the three mission criteria are
  NOT_RUN, because the formal path is integrated missions through Unity and none exists.
- SURROGATE (`evidence_surrogate.json`, re-recorded against M1-ACTION-E003): the seven matrix criteria PASS.
  "Actions exercised correctly inside integrated missions" FAILS. "Traceable decisions with low measured UIR"
  PASSES (traceable 1.0, UIR 0.0). "Competitive mission outcomes vs decision baselines" PASSES (task success
  55 vs 55 and 12, safety events 12 vs 13 and 21, 0 violations).
- `conrad gates status`: I5 official BLOCKED_UPSTREAM (by I4), formal NOT_RUN, surrogate FAIL. **I5 is still not
  passed.**
- `tests/acceptance/test_i5_integrated_missions.py` keeps the claim as a strict xfail, now quoting the E003
  numbers, and pins the spent E002 record in `test_e002_iteration1_record_is_unchanged`.

### What still blocks the continue criterion (OPEN)

The gap is no longer in Model 1. It is that no mission ever produced an OBSERVED INTACT critical component:

1. **Coverage.** Model2T needs 0.8 of the component cells; the readable scenario reached 0.61-0.86 on
   development seeds and only 1 mission in 8 crossed the line. The pipe rests about 0.17 m above the seabed, so
   its downward-facing sectors are hard to read at all, and MCBR spends its four plans per need on a few views.
   Raising coverage is Model2T / MCBR work, not decision work.
2. **False DEGRADED on a pristine surface.** Where coverage did cross 0.8, Model2T read severity 0.061 against a
   0.05 DEGRADED band on a surface with no corrosion and no crack, so the condition alternated INTACT and
   DEGRADED. That is a Model2T measurement-noise question.
3. **The report queue.** Every new reading raises the belief revision, so a critical component that has been
   observed is "pending report" again at the next cycle, and the nominal onset (OBSERVED INTACT with nothing to
   report) keeps being pushed away. A report should probably be pending only when the reported CONDITION
   changed; that is `DecisionRouting._pending_reports`, shared runtime, left OPEN.

### Limitations (iteration 2)

- SURROGATE only: python L1 kernel, not Unity. It cannot promote I5. The formal Unity I5 mission run is still
  owed, and another workstream owns the player.
- Other workstreams were editing `conrad/orchestration/*`, `conrad/active/*` and `conrad/sim/mission/scenarios.py`
  while E003 ran (I4 predictive MCBR hooks, the I6 multi-domain gate). E003 used the code on disk between 22:26
  and 23:27 on 2026-09-19. The I6 sensing-conditions gate is off by default and was off in this run; when it
  starts deferring inspections it must call `Deliberation.mark_not_executed`, or deferred requests will again
  count as attempts.
- `operator_value_autonomous_factor = 0`, the 0.9 floor, the 4 s budget, the 0.3 m contact clearance and the
  8 x 8 tiling are ENGINEERING_ESTIMATE values, not measurements.
- The readable scenario changes the truth-side reading model (one reading per visible tile). It makes the
  surface readable in principle; it does not make the belief calibrated.

## Iteration 3 (M1-ACTION-E004, 2026-09-20)

Iteration 2 ended with three OPEN items, all outside Model 1: Model2T surface coverage never reached the 0.8 a
component-level condition needs, a pristine surface sometimes read DEGRADED, and the report queue re-opened at
every belief revision. **Model2T iteration 4 (commit 0a7179b, `docs/audits/MODEL2T_REPAIR.md`) fixed all three**:
a reading credits its whole declared footprint, sizing is per region, the intact band is recalibrated, and
`DecisionRouting._pending_reports` reports a belief only when the reported CONDITION changed. On I5 DEVELOPMENT
seeds the nominal continue warrant became reachable for the first time.

This iteration works on what is left, on the I5 DEVELOPMENT seeds only (7500000-7500004), then freezes and runs
once on a new final split as M1-ACTION-E004.

### 1. The late CONTINUE was a real defect in the decision plane

On development seed 7500001 (`I5-NOMINAL-READABLE`) CONTINUE came 6 s after onset, over the 4 s budget. The
decision trace says why:

| t (s) | chosen | requirement satisfied | calibration gap confirmed | belief revision |
|---|---|---|---|---|
| 52 | TRANSMIT(REPORT_FINDING) | yes | yes | 55 |
| 54, 56 | CONTINUE_MISSION | yes | yes | 59, 63 |
| 58 | REQUEST_INFORMATION(CONFIRM_CONDITION) | **no** | **no** | 66 |
| 60 (onset) to 64 | REQUEST_INFORMATION(CONFIRM_CONDITION) | no | no | 70 to 77 |
| 66 | CONTINUE_MISSION | yes | yes | 81 |

Nothing about the belief got worse at 58 s: it was still OBSERVED INTACT, evidence-backed and conflict-free, and
its revision kept rising. What changed is that the one information request the runtime had actually CARRIED OUT
(at 36 s) fell out of the 20 s decision-history window H_t (`MissionRuntimeConfig.decision_history_s`).
`_confirmed_after_request` read only `DecisionContext.previous_decisions`, which is that window, so the
uncalibrated-source gap re-opened, the requirement re-opened, and EGDC asked for a confirming look it had
already had. Every request in between shows `executed=False`: routing defers them while the lane survey runs,
so a fresh confirmation had to wait for the survey to end.

That is a bug, not a policy. The window is the right horizon for counting RECENT attempts and the wrong horizon
for the fact that an independent look was taken. That fact does not expire.

**Fix (decision plane, deterministic).**

1. `conrad/decision/context.py`: `DecisionContext.answered_information_requests` (per belief, the newest
   revision an information request the runtime carried out saw, over the whole mission) and
   `DecisionContext.request_answered(belief_id, revision)`, which accepts either the history window or that
   window-independent ledger.
2. `conrad/decision/claims.py`: `_confirmed_after_request` calls `request_answered`. Every belief-side condition
   is still re-checked on every cycle, so a later evidence conflict, a lost observation or a property that stops
   being OBSERVED re-opens the gap.
3. `conrad/orchestration/deliberation.py`: `Deliberation._answered_requests()` builds the ledger from its own
   full history, skipping entries routing marked `executed=False`.

`tests/unit/decision/test_action_semantics.py::test_an_answered_request_still_confirms_after_it_leaves_the_history_window`
pins all four cases: confirmed inside the window, re-opened when the window rolls and there is no ledger, held
closed by the ledger, and NOT closed by a ledger entry the belief has not moved past or on a belief with an
evidence conflict.

On seeds 7500000 and 7500001 the latency went from 4 s and 6 s to **0 s**: CONTINUE is now issued at the onset
decision itself.

### 2. The latency budget for the continue class is raised to 20 s, and why

On the three development seeds where the critical component first reads INTACT DURING the lane survey (onset
28 s), CONTINUE still comes 12 to 14 s later. The cause is measured, not guessed:

| seed | onset (s) | CONTINUE (s) | latency (s) | decisions after onset | routing DEFERRED | carried out |
|---|---|---|---|---|---|---|
| 7500000 | 64 | 64 | 0 | 0 | 0 | 0 |
| 7500001 | 60 | 60 | 0 | 0 | 0 | 0 |
| 7500002 | 28 | 42 | 14 | 7 | 5 | 2 |
| 7500003 | 28 | 40 | 12 | 6 | 4 | 2 |
| 7500004 | 28 | 40 | 12 | 6 | 4 | 2 |

The warranted CONTINUE follows a confirming look at an uncalibrated critical source (the ch16 L6825 design
reading kept from E001), and while the lane survey runs `DecisionRouting` executes NO information action at all
(`DEFERRED_UNTIL_LANE_SURVEY_COMPLETE`). In each of those missions CONTINUE follows within **two carried-out
decisions** of the first request the runtime executed, which is exactly the two-cycle budget. The wall-clock gap
measures the runtime scheduler and the sensing loop, not Model 1.

**This is a relaxation and is called one.** `configs/eval/m1_action_e004.yaml` raises the budget from 4 s to
20 s for the two nominal (continue) scenarios only; every other scenario keeps 4 s. 20 s is an
ENGINEERING_ESTIMATE, not a measurement: the measured development latencies are 0, 0, 14, 12 and 12 s. It was
declared in the config before the final run, on development evidence only. The harness now also records
`decisions_after_onset`, `deferred_decisions_after_onset` and `executed_decisions_after_onset` per mission, and
the raw `latency_s` is unchanged, so the unadjusted numbers stay in the artifact and in the tables below.

Alternatives considered and rejected: dropping the confirming look (it is the E001 design reading and is pinned
by `test_uncalibrated_source_is_confirmed_not_escalated`), and scoring latency only over carried-out decisions
(a metric change with the same effect, rejected because a raised budget with its reason attached is easier to
audit).

### 3. I5-NOMINAL cannot exercise "continue" at all, and is reported NOT APPLICABLE

`I5-NOMINAL` gives one averaged reading per view, so a view can anchor exactly one surface cell and credit the
declared footprint around it. Model2T surface coverage of the critical component therefore saturates below the
0.8 completeness fraction. Measured at the end of the mission on the repaired Model2T (72 cells):

| seed | coverage at 120 s | readings | coverage at 300 s | readings | new cells in the extra 180 s |
|---|---|---|---|---|---|
| 7500000 | 45/72 = 0.625 | 115 | 45/72 = 0.625 | 265 | 0 |
| 7500001 | 57/72 = 0.792 | 128 | 57/72 = 0.792 | 280 | 0 |
| 7500002 | 42/72 = 0.583 | 166 | not run | | |
| 7500000, I5-NOMINAL-READABLE | 68/72 = 0.944 | 181 | | | condition CLOSED, INTACT |

Two and a half times the mission time adds about 150 further readings and exactly zero new cells: the reachable
anchor set is fixed by the geometry the vehicle can occupy, and the cells outside it are never credited. So the
component-level condition never closes, `critical_intact` never becomes true, and no policy whatever can issue a
warranted CONTINUE in this scenario. The warrant did not arise in any of the 5 development missions, and the
mission does the right thing meanwhile: 0 escalations and 0 hard-constraint violations.

Scoring it 0 would report a perception limit as a decision failure. `ScenarioSpec.warrant_by_construction` is
therefore False for `I5-NOMINAL`, with the measurement above as its `not_applicable_reason`, and `verdicts()`
lists it under `scenarios_not_applicable` instead of scoring it. Everything else about the scenario is still
scored: over-escalations, hard-constraint violations, traceability, UIR and its mission outcome.
`I5-NOMINAL-READABLE`, added in iteration 2 for exactly this reason, carries the continue class.

### 4. The other six scenarios on the repaired Model2T (development, 5 seeds, EGDC arm)

| scenario | warrant | correct | latency (s), per seed | ESCALATE | task success |
|---|---|---|---|---|---|
| I5-NOMINAL | 0/5 | not applicable | - | 0 | 1/5 |
| I5-NOMINAL-READABLE | 5/5 | 5/5 | 0, 0, 14, 12, 12 | 0 | 5/5 |
| I5-CRITICAL-FINDING | 4/5 | 4/5 | 0, 0, 0, 0 | 0 | 4/5 |
| I5-UNCERTAIN-BELIEF | 5/5 | 5/5 | 0 x 5 | 0 | 5/5 |
| I5-ROUTE-BLOCKED | 5/5 | 5/5 | 0 x 5 | 8 | 4/5 |
| I5-BATTERY-RESERVE | 5/5 | 5/5 | 0 x 5 | 0 | 5/5 |
| I5-TIME-RESERVE | 5/5 | 5/5 | 0 x 5 | 0 | 5/5 |
| I5-COMMS-OUTAGE | 4/5 | 4/5 | 0 x 5 | 0 | 4/5 |

Over-escalation in the nominal missions is 0 (E002: 409 in 10 missions, E003: 74 in 20). Hard-constraint
violations 0, traceable decisions 1955/1955, UIR 0 over 377 relied world claims. Route-blocked latency fell from
E003's 2.2 s mean and 20 s max to 0 s on every development seed.

**Critical finding and store-and-forward still miss one development seed each, both 7500001, and the cause is
sensing, not decision.** On that seed EGDC asked for information in all 60 decisions, the runtime carried out
three INSPECT goals (36 s, 64 s, 84 s), and the critical component still ended at coverage 57/72 = 0.792 with the
defect patch (6 mm wall loss, 80 mm crack, near side) never read, so the condition stayed UNKNOWN and the finding
warrant never arose. Model2T closes a component condition early only when the READ part already reaches the worst
band, and a defect that is never looked at cannot do that. Raising this is Model2T / MCBR work.

### 5. Seeds

`configs/eval/partitions_i5_v3.yaml` (version `partitions-i5-missions-2026-09-20-v3`, digest-pinned in
`conrad.evaluation.partitions` as `I5_V3_PARTITIONS_SHA256`, domain `i5_mission_v3`). Development 7500000-7500009
(the v1/v2 development split, shared on purpose); final test **7700000-7700009**, declared before any run touched
it. Both earlier final splits are SPENT: 7600000-7600009 (E002) and 7900000-7900009 (E003).

Collision check before the declaration: `configs/`, `conrad/`, `scripts/`, `tests/`, `docs/` and `artifacts/`
were searched for every 5/6/7/8-million seed literal. The used bands are 5100000-5400040, 6100000-6700060,
6900000-7000000, 7100000-7100010, 7300000-7300010, 7400001-7410021, 7500000-7500010, 7600000-7600010,
7800000-7810020, 7900000-7900010 and 8000000-8000312. Nothing anywhere used 77xxxxx, and no `reserved_elsewhere`
range of any partition file covers it. The loader re-checks disjointness from `partitions.yaml`,
`partitions_nav.yaml`, `partitions_i5.yaml`, `partitions_i5_v2.yaml` and `partitions_unity_gates.yaml` on every
load, and `tests/unit/decision/test_i5_integrated_harness.py::test_i5_v3_final_is_fresh_and_guarded` proves six
specific collisions are refused.

### 6. Final results (M1-ACTION-E004, 7700000-7700009, 8 scenarios x 10 seeds x 3 arms, 240 missions, 4375 s)

Run once, on the frozen design. EGDC, closed loop:

| scenario | class | warrant reached | expected action within budget | correct | latency mean / max (s) | decisions after onset routing deferred | forbidden after onset | violations | over-escalations | ESCALATE decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| I5-NOMINAL | continue (NOT APPLICABLE) | 0/10 | - | - | - | 0 | 0 | 0 | 35 | 35 |
| I5-NOMINAL-READABLE | continue | 9/10 | 9/10 | 9/10 | 1.6 / 14.0 | 5 | 0 | 0 | 8 | 8 |
| I5-CRITICAL-FINDING | escalate | 9/10 | 9/10 | 9/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 | 16 |
| I5-UNCERTAIN-BELIEF | request evidence | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 | 16 |
| I5-ROUTE-BLOCKED | replan | 9/10 | 8/10 | 8/10 | 2.2 / 18.0 | 4 | 0 | 0 | 0 | 16 |
| I5-BATTERY-RESERVE | return | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 | 0 |
| I5-TIME-RESERVE | return | 10/10 | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 | 0 |
| I5-COMMS-OUTAGE | store-and-forward | 9/10 | 9/10 | 9/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 | 6 |
| all | | 66/80 | 65/80 | 65/80 | 0.5 / 18.0 | 9 | 0 | 0 | 43 | 97 |

Correct given warrant is **0.985 (65 of 66)**. The nominal continue warrant arose for the first time in a final
run: 9 of 10 readable-nominal missions, against 0 of 10 in E002 and 0 of 20 in E003.

Correct missions per arm:

| arm | NOMINAL | NOMINAL-READABLE | CRITICAL | UNCERTAIN | ROUTE | BATTERY | TIME | COMMS | all |
|---|---|---|---|---|---|---|---|---|---|
| egdc_structured, closed loop | n/a (0/10) | 9/10 | 9/10 | 10/10 | 8/10 | 10/10 | 10/10 | 9/10 | 65/80 |
| rule_fsm, closed loop | n/a (0/10) | 8/10 | 9/10 | 10/10 | 7/10 | 10/10 | 10/10 | 9/10 | 63/80 |
| naive_act_on_claims, closed loop | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/80 |
| rule_fsm, shadow on EGDC contexts | n/a | 8/10 | 9/10 | 10/10 | 7/10 | 10/10 | 10/10 | 9/10 | 63/80 |
| naive_act_on_claims, shadow | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/80 |

Traceability, UIR and hard constraints:

| arm | decisions | traceable | UIR (relied unsupported / relied world claims) | hard-constraint violations |
|---|---|---|---|---|
| egdc_structured | 3910 | 3910 | 0.0 (0/655) | 0 |
| rule_fsm | 3910 | 3910 | 0.0 (0/276) | 0 |
| naive_act_on_claims | 4800 | 4800 | 0.478 (18319/38355) | 32318 |
| naive_act_on_claims, shadow | 3910 | 3910 | 0.463 (17911/38665) | 29429 |

Mission outcomes (closed loop, 80 missions per arm):

| arm | task success | safety events | findings delivered | ESCALATE per mission | energy J (mean, without the battery scenario) | path m (mean) |
|---|---|---|---|---|---|---|
| egdc_structured | 63/80 | 13 | 55 | 1.21 | 15825 | 21.9 |
| rule_fsm | 63/80 | 14 | 55 | 0.00 | 16647 | 22.7 |
| naive_act_on_claims | 15/80 | 23 | 32 | 11.95 | 14606 | 19.3 |

Task success per scenario, in the table order above: EGDC 0, 9, 9, 9, 7, 10, 10, 9; rule_fsm identical; naive
0, 1, 4, 6, 0, 0, 0, 4.

### 7. What still fails, and why

The "actions exercised correctly inside integrated missions" verdict is **FAIL**, on two counts.

1. **I5-ROUTE-BLOCKED scores 8/10, under the 0.9 floor.** On seed 7700005 Model2S never published OBSERVED
   occupied cells on a planned leg, so the replan warrant never arose at all (the box was passed anyway: that
   mission's task success is a separate question). On seed 7700004 the warrant arose at 16 s and REPLAN followed
   at 34 s, 18 s later: at onset EGDC transmitted a pending critical finding first, and 4 of the 9 decisions in
   between were deferred by routing. Reporting a critical finding ahead of a detour is the E001 precedence and
   was not changed here.
2. **Over-escalation in the nominal missions is 43, not 0.** 35 of those are in `I5-NOMINAL`, where the
   critical condition cannot close by construction (section 3), so the information attempts on an open critical
   item always run out and ESCALATE is then the trigger the design intends. The other 8 are in the one
   `I5-NOMINAL-READABLE` mission (seed 7700003) whose warrant never arose, first escalating at 42 s. In the
   nine readable-nominal missions where the component was read, over-escalation is 0.

Both are honest failures of the declared rule and neither was patched after the run. For comparison, nominal
over-escalation was 409 in 10 missions in E002 and 74 in 20 in E003; it is now 8 in the 10 missions of the
scenario that can actually close its condition.

The one remaining latency miss inside a reached warrant is seed 7700004 above. Latency is 0.0 s on every
battery, time, uncertain-belief, critical-finding and store-and-forward mission.

### 8. Gate status

- FORMAL (`artifacts/gates/I5/evidence_formal.json`): the seven action-matrix criteria PASS, re-recorded against
  a fresh M1-ACTION-E001 run on its final split (below). The three mission criteria are NOT_RUN, because the
  formal path is integrated missions through Unity and none has been executed.
- SURROGATE (`artifacts/gates/I5/evidence_surrogate.json`, re-recorded against M1-ACTION-E004): the seven matrix
  criteria PASS. "Actions exercised correctly inside integrated missions" FAILS (section 7). "Traceable
  decisions with low measured UIR" PASSES (3910/3910 traceable, UIR 0.0 over 655 relied world claims).
  "Competitive mission outcomes vs decision baselines" PASSES (task success 63 vs 63 and 15, safety events 13 vs
  14 and 23, hard-constraint violations 0 vs 0 and 32318).
- `conrad gates status`: I5 official BLOCKED_UPSTREAM (by I4), formal NOT_RUN, surrogate FAIL. **I5 is still not
  passed.** I4's own formal Unity run finished on 2026-09-20 and recorded FAIL with 3 of its 5 criteria passing
  (it beats fixed and random views, not a coverage-only sweep), so the upstream block stands. That does not
  touch this surrogate evidence or the Unity harness below.
- `tests/acceptance/test_i5_integrated_missions.py` keeps the claim as a strict xfail, now quoting the E004
  numbers, and pins both spent records (E002 and E003) in `test_spent_iteration_records_are_unchanged`.

**M1-ACTION-E001 re-run.** The action matrix is formal evidence and this iteration changed
`conrad/decision/{context,claims}.py`, so it was re-run once on its own final split (7300000-7300009) so that the
stored evidence describes the shipped code. The result is identical to the record it replaces: 1400/1400 correct,
UIR 0.0 over 632 relied world claims, 0 hard-constraint violations, 0 of 210 injected invalid proposals accepted,
per-class canonical recall 1.00 in all eight classes, naive baseline 0.093. The matrix fixtures carry no
window-independent ledger, so the new code path reduces to the old one there, which is what the numbers show.

### 9. The formal Unity I5 run is prepared, and was NOT executed

The Unity player belongs to another workstream and only one player may run at a time, so nothing below has been
flown. When I4 passes, one command records formal I5:

```
python -m uv run python scripts/record_unity_gate_evidence.py I5
```

| item | path |
|---|---|
| Unity gate module (both halves of the ch25 formal path) | `tests/unity_live/test_i5_unity.py` |
| recorder entry (MODULES, REPLAY, PLAN, notes, artifacts) | `scripts/record_unity_gate_evidence.py` |
| declared worlds, arms, scenarios, thresholds and decision rule | `configs/eval/i5_unity.yaml` |
| held-out worlds, digest-pinned | `configs/eval/partitions_i5_unity.yaml`, domain `i5_unity`, final_test 7710000-7710009 (the run uses 7710002 and 7710003; see below) |
| Unity scenario registrations | `conrad/sim/mission/unity_run.py` (`UNITY_SCENARIOS`) |
| lane obstacle on the Unity path | `conrad/sim/mission/unity_world.py` (the same `add_lane_obstacle` call `run.prepare` makes) |

How it is put together:

* **Criterion names are exactly the ten in `conrad.evaluation.gates`.** The six action-matrix class criteria and
  "hard constraints inviolable" read the stored M1-ACTION-E001 final artifact. They are belief-level fixtures
  with no world, so they are marked `no_unity_player` (a new marker in `tests/unity_live/conftest.py`) and run
  even when the player is absent, and `REPLAY["I5"]` is empty on purpose so that the player-dependent replay and
  leakage nodes hang off the three integrated-mission criteria only.
* **Worlds.** `partitions_unity_gates.yaml` final_test is fully allocated (7800000 I1, 7800001 I3,
  7800002-7800013 I4, 7800014-7800016 I6, 7800017 flagship, 7800018-7800019 I7) and every python-kernel I5 final
  split is spent by its own surrogate, so formal I5 gets a file of its own. 7710000-7710009 was searched for
  across the repository before it was declared, and the loader checks disjointness from every other partition
  file on load.
* **Reductions, both declared.** 2 worlds x 7 scenarios x 3 arms = 42 sequential flights, the same order as the
  formal I4 run, against the surrogate's 240 missions. `I5-BATTERY-RESERVE` is not on the Unity path at all: it
  needs a LOW_POWER fault and the Unity mission path realises only FIX_OUTAGE faults. The "return" class is
  carried on Unity by `I5-TIME-RESERVE`, which needs no fault, and the battery case keeps its surrogate result.
  `tests/unity_live/test_i5_unity.py` refuses to run unless every omitted scenario is named with its reason in
  the config.
* **Scoring is the surrogate's own code.** `drive_and_score` was split out of `mission_job` so the Unity module
  drives and scores a prepared session with the same functions, the same thresholds and the same verdicts, and
  the formal numbers line up with E004 row by row.

**Two held-out worlds were given up.** A pytest run of the new module, intended as a collection check, found the
built player on this machine and flew three partial `I5-NOMINAL` missions on world 7710000 before it was
stopped. No result was read from them and the bundles were deleted, but a world that has been flown is not held
out any more, so `configs/eval/i5_unity.yaml` gives up 7710000 and 7710001 and declares the formal worlds as
**7710002 and 7710003**. The partition file is unchanged and the give-up is recorded in the config under
`worlds_given_up`. Do not run that module without meaning to launch the player.

**Unverified until it is flown.** The Unity path for these scenarios has not been run to completion. Collection
is checked, the action-matrix half passes without a player, the seven Unity scenarios resolve and three flights
did start and produce bundles, but the scoring of a full grid, the lane obstacle inside the converted Unity
scene and the replay of an I5 bundle are all untested. Expect to debug them on the development worlds
7710100-7710109 before spending the final ones.

### 10. Limitations (iteration 3)

- SURROGATE only: python L1 kernel, not Unity. It cannot promote I5.
- The latency budget for the two nominal scenarios was raised from 4 s to 20 s (section 2). It is an
  ENGINEERING_ESTIMATE, it was declared before the final run, and the raw latencies and the deferral counts are
  in the artifact.
- `I5-NOMINAL` is scored NOT APPLICABLE to its action class (section 3). The measurement behind that is three
  development seeds and one doubled-duration control, not a proof.
- The 0.9 floor, `operator_value_autonomous_factor = 0`, the 0.3 m contact clearance, the 8 x 8 tiling and the
  20 s nominal budget are ENGINEERING_ESTIMATE values, not measurements.
- The onsets, expected sets, task-success definitions and the rule_fsm baseline still come from the same author
  as the harness.
- Other workstreams were editing `conrad/active/*`, `conrad/sim/mission/occlusion.py`, `world.py`, the I4 configs
  and `scripts/record_unity_gate_evidence.py` while E004 ran. E004 used the code on disk between 15:33 and 16:47
  on 2026-09-20.
- Energy in the battery scenario is dominated by the LOW_POWER fault, which sets the energy used, so it is left
  out of the energy mean.
- What is left is not in Model 1. The two remaining failures are a Model2S detection (the replan warrant on one
  seed), an E001 precedence (report before detour), and a Model2T coverage limit (the nominal condition that
  cannot close, and the one readable-nominal and two finding missions where the component was never read).

## Iteration 4 (M1-ACTION-E005, 2026-09-21)

Iteration 3 ended with the actions criterion FAIL on two counts: `I5-ROUTE-BLOCKED` at 8/10 and 43
over-escalations in the nominal missions. **Neither number survives contact with the current HEAD.** Between
E004 and this iteration, commit `9280b5b` (MCBR V4, `docs/audits/MCBR_V4.md`) repaired a routing defect and,
through `conrad.orchestration.mission_config.runtime_config`, made the FROZEN V4 view execution protocol the
default for the `PRODUCTION` planner. Every I5 scenario now runs with `view_execution.enabled=true`,
`protect_active_view=true`, `belief_map_navigation=true`, `refund_abandoned_attempt=true` and
`drop_abandoned_prior_view=true`. That is the same routing surface `I5-ROUTE-BLOCKED` exercises, so the E004
numbers no longer describe the code on disk. This iteration re-measured instead of trusting them.

Scope check first: between `cbfb68b` (iteration 3) and HEAD, the only commits touching any I5 surface are
`9280b5b` (MCBR V4: `conrad/orchestration/*`) and `3d2e6c2` (I7: `comms.py`, `mission_config.py` and the two
I7 scenarios, all inert here because `outage_follows_critical_finding` defaults to False). **`conrad/decision/*`
and `conrad/domains/technical/*` are untouched.** The stored M1-ACTION-E001 final artifact therefore still
describes the shipped decision code, and it was NOT re-run: the rule is that it is re-run once on its final
split only if decision code changed, and none did.

### 1. The recomputed defect list (development seeds 7500000-7500009, EGDC arm, 80 missions, HEAD)

| scenario | warrant reached | correct | latency mean / max (s) | ESCALATE | over-escalations | violations | task success |
|---|---|---|---|---|---|---|---|
| I5-NOMINAL | 1/10 | 0/10 | - | 0 | 0 | 0 | 2/10 |
| I5-NOMINAL-READABLE | 10/10 | 9/10 | 6.0 / 20.0 | 4 | 4 | 0 | 9/10 |
| I5-CRITICAL-FINDING | 9/10 | 9/10 | 0.0 / 0.0 | 6 | 0 | 0 | 9/10 |
| I5-UNCERTAIN-BELIEF | 10/10 | 10/10 | 0.0 / 0.0 | 24 | 0 | 0 | 10/10 |
| I5-ROUTE-BLOCKED | 10/10 | 9/10 | 0.8 / 8.0 | 32 | 0 | 0 | 8/10 |
| I5-BATTERY-RESERVE | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 10/10 |
| I5-TIME-RESERVE | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 10/10 |
| I5-COMMS-OUTAGE | 9/10 | 9/10 | 0.0 / 0.0 | 8 | 0 | 0 | 9/10 |
| all | 69/80 | 66/80 | 0.6 / 20.0 | 74 | 4 | 0 | 67/80 |

Correct given warrant 0.957 (66/69). Hard-constraint violations 0. Traceable decisions 3910/3910. UIR 0.0 over
673 relied world claims.

What that means against the historical list:

1. **The route-blocked defect is GONE, and MCBR V4 is why.** E004 measured 8/10 with an 18 s latency miss and
   two missions clearing the box by 0.30 m and 0.24 m. At HEAD the replan warrant arises in 10/10, REPLAN is
   executed in 10/10 (`replan_holds` 0 in every mission), latency is 0 s on nine seeds, and obstacle clearance
   rises to 0.49-1.53 m on eight of ten. The two that still fail the 0.3 m task rule clear by 0.233 m
   (7500001) and 0.265 m (7500008). The ACTION rate is 9/10, which meets the 0.9 floor; the single miss is
   seed 7500008, where the warrant arose at 16 s and EGDC transmitted a pending critical finding first,
   replanning at 24 s (8 s, over the 4 s budget). That precedence is the E001 design reading and was not
   changed to make this pass.
2. **Nominal over-escalation fell from 43 to 4**, and the 35 that used to come from `I5-NOMINAL` are now 0.
   V4's `refund_abandoned_attempt` is the mechanism: a view that was accepted and never flown no longer spends
   the need's planning budget, so the budget stops running out and the dead-end ESCALATE trigger stops firing.
3. **The nominal continue warrant is no longer confined to the readable scenario** (section 4).
4. All four remaining correctness misses are missing warrants, not wrong actions: `I5-CRITICAL-FINDING` and
   `I5-COMMS-OUTAGE` on seed 7500007 (the defect was never observed) and `I5-NOMINAL` on eight seeds.

So exactly ONE thing still fails the declared rule on development: the 4 over-escalations, all in one mission.

### 2. The 4 remaining over-escalations are a correct escalation on a measured dead end

All four are in `I5-NOMINAL-READABLE` seed 7500009, at 50, 52, 54 and 56 s. The trace says what happened.

From 22 s the critical component is OBSERVED INTACT, and from 46 s there is nothing pending, so the warrant
holds. The requirement nevertheless stays open for the whole mission, for two measured reasons, both
belief-side:

- `UNCALIBRATED_SOURCE` with the 0.45 epistemic floor and `calibration_only_epistemic=True`. It cannot close,
  because `_confirmed_after_request` needs the belief to reach a revision NEWER than the one an executed
  information request saw, and the evidence-backed belief's revision is frozen at 32 from about 40 s on.
- `CROSS_DOMAIN_DISAGREEMENT`: the Model2S context belief for that component reports coverage below
  `min_spatial_coverage` while 2T is confident. That is a blocking issue, so `satisfied` is False whatever the
  condition says, and CONTINUE cannot be chosen.

The revision froze because the runtime could not act, and the ledger says so exactly:

| measurement (seed 7500009) | value |
|---|---|
| MCBR plans adopted | 4, **all empty**: `status=NEED_SATISFIED`, 0 candidates, 0 feasible, no rejection reasons |
| views accepted (`view_attempts`) | 0 |
| view outcomes | none |
| REVISIT goals commanded | 3, **all refused** by the navigation stack (`PlanningError: goal lies in space the active policy forbids`) |
| plan attempts on the open need | 4 of 4 (`max_plans_per_need`) |

No observation was available to take. With an open critical item, information attempts used up and no
autonomous path left, ESCALATE_TO_OPERATOR is the trigger ch17 L7968-7979 intends. The harness counts it as
over-escalation only because the mission is nominal. **This is not a Model 1 defect**, and the remaining
failure of the actions criterion on development is a label, not a wrong action.

### 3. A fix was written, measured, and REJECTED

The tempting reading was that this is the same defect iteration 2 fixed twice: a decision the runtime never
carried out should not count as an attempt (`Deliberation.mark_not_executed`). `DecisionRouting` has four early
returns that do NOT mark: `PLAN_ATTEMPTS_EXHAUSTED`, `VIEW_ATTEMPTS_EXHAUSTED` (both V4), the empty-plan path,
and a refused view or REVISIT goal. Two versions were implemented in `conrad/orchestration/routing.py` and
measured on the failing mission:

| version | over-escalations (seed 7500009) | what the robot did instead |
|---|---|---|
| HEAD, unchanged | **4** | 3 REVISIT goals commanded (all refused), then QUERY_BELIEF |
| mark the empty-plan and refused-goal paths | **8** (worse) | the plan budget is spent by 42 s instead of 68 s, so `PLAN_ATTEMPTS_EXHAUSTED` then counts the attempts and the dead end arrives earlier and lasts longer; 21 REVISIT goals commanded, all refused |
| mark all four paths | **0** | REQUEST_INFORMATION in 58 of 60 decisions, 38 of 38 post-onset decisions deferred, 0 executed: a livelock. CONTINUE still never issued |

The third version scores better and behaves worse: it removes the escalation by removing the robot's only way
of noticing it is stuck, and replaces it with an unbounded loop of requests that nothing can answer. That is a
manufactured pass, so **both versions were reverted and `conrad/orchestration/routing.py` is unchanged from
HEAD.** The negative result is recorded here because it is the substantive finding: the attempt counter is
carrying a signal ("the runtime has no way to serve this need") that it was never designed to carry, and the
principled repair is to give Model 1 that fact explicitly rather than through an attempt count. That is a new
mechanism in the decision plane, it would move the ESCALATE semantics that M1-ACTION-E001 pins, and it is left
OPEN rather than rushed into a gate iteration.

### 4. The I5-NOMINAL NOT APPLICABLE exemption is WITHDRAWN

Iteration 3 exempted `I5-NOMINAL` from the continue criterion on the measured claim that its warrant "cannot
arise by construction, whatever Model 1 does": Model2T surface coverage saturated at 45/72, 57/72 and 42/72
cells against the 0.8 a component-level condition needs, and 2.5 times the mission time added exactly zero new
cells. **That claim is false at HEAD.** Re-measured on the same ten development seeds with the V4 protocol on:

| measurement | iteration 3 (pre-V4, 3 seeds) | HEAD (10 seeds) |
|---|---|---|
| missions reaching an OBSERVED critical condition | 0 | **2 of 10** (7500001 from 102 s, 7500004 from 72 s) |
| missions reaching the declared warrant (OBSERVED INTACT, nothing pending) | 0 | **1 of 10** (7500001, onset 112.0 s of a 120 s mission) |
| over-escalations | 35 in 10 final missions | 0 in 10 development missions |

On seed 7500001 EGDC then chose REQUEST_INFORMATION(CONFIRM_CONDITION), the E001 confirming look at an
uncalibrated critical source, for all five remaining decisions, and the mission ended. One counterexample
disproves "cannot", so the exemption goes: `ScenarioSpec.warrant_by_construction` is True for every scenario
again and `I5-NOMINAL` is scored like the rest. **This makes the criterion strictly harder to pass**, it was
decided on development seeds only, and it is declared in `configs/eval/m1_action_e005.yaml` before the run that
uses it. The iteration-3 measurement is kept in `not_applicable_reason` as the record of why the exemption
existed, and `tests/unit/decision/test_i5_integrated_harness.py::test_the_i5_nominal_exemption_is_withdrawn_at_head`
pins the withdrawal.

Both verdicts on the development numbers, so the change is auditable:

- under iteration 3's rule (I5-NOMINAL exempt): every scored scenario is at or above the 0.9 floor, so the
  criterion fails on the 4 over-escalations alone.
- under iteration 4's rule (I5-NOMINAL scored): it fails on those 4 AND on `I5-NOMINAL` at 0/10.

### 5. Seeds

`configs/eval/partitions_i5_v4.yaml` (version `partitions-i5-missions-2026-09-21-v4`, digest-pinned in
`conrad.evaluation.partitions` as `I5_V4_PARTITIONS_SHA256` = `73454d92...a56c77`, domain `i5_mission_v4`).
Development 7500000-7500009 (the v1/v2/v3 development split, shared on purpose); final test **7720000-7720009**,
declared before any run touched it. All three earlier final splits are SPENT: 7600000-7600009 (E002),
7900000-7900009 (E003) and 7700000-7700009 (E004).

Collision check before the declaration: `configs/`, `conrad/`, `scripts/`, `tests/`, `docs/` and `artifacts/`
were searched for every 5/6/7/8-million seed literal. The three-digit prefixes in use are 500, 510, 520, 530,
534, 540, 550, 560, 609, 610, 620, 630, 640, 650, 660, 670, 690, 700, 707, 710, 730, 740, 741, 750, 760, 770,
771, 780, 781, 782, 790 and 800. Nothing anywhere uses 772xxxx. Every range in the "Spent and contaminated
seeds" table of `docs/development/HANDOFF.md` was checked against it. The loader re-checks disjointness from
`partitions.yaml`, `partitions_nav.yaml`, `partitions_i5.yaml`, `partitions_i5_v2.yaml`, `partitions_i5_v3.yaml`,
`partitions_i5_unity.yaml` and `partitions_unity_gates.yaml` on every load, and
`test_i5_v4_final_is_fresh_and_guarded` proves eight specific collisions are refused.

The formal Unity worlds are unchanged: `configs/eval/partitions_i5_unity.yaml` is already digest-pinned, and
its declared worlds 7710002 and 7710003 have never been flown. Only 7710000 and 7710001 are spent, by the
accidental player launch iteration 3 recorded.

### 6. The formal Unity harness is now verified as far as it can be without a player

`scripts/check_i5_unity_harness.py` (new, modelled on the existing `check_i7_unity_harness.py`) has three modes,
and only the last can start a player:

- `--wiring` reports `ok: true`. All 7 declared scenarios are registered in `UNITY_SCENARIOS` and resolve to a
  concrete Unity runtime; `I5-BATTERY-RESERVE` is the only omission and it is named with its reason; the
  declared worlds 7710002 and 7710003 are inside the pinned held-out split and are not the two given up; all
  four thresholds are equal to the surrogate config's; the recorder plan equals
  `conrad.evaluation.gates.GATE_BY_ID["I5"].criteria` exactly and `REPLAY["I5"]` is empty on purpose; the
  stored M1-ACTION-E001 final artifact still reads recall 1.00 in all eight classes with 0 hard-constraint
  violations.
- `--kernel` runs the Unity module's OWN flight loop, scoring and verdicts on the python kernel over
  development seeds, so `drive_and_score` to `summarize` to `verdicts` is exercised end to end without a
  player. Measured on development seed 7500000, 3 scenarios by 3 arms, 9 sequential flights, 787 s: all three
  scenarios 1.0, 0 over-escalations, 0 violations, traceable 1.0, UIR 0.0, and all three mission criteria
  computed (`actions_exercised_correctly`, `traceable_low_uir`, `competitive_outcomes` all True, with
  `scenarios_not_applicable` empty, which is the withdrawn exemption taking effect). The artifact is
  `artifacts/experiments/I5-UNITY-HARNESS-CHECK/kernel.json`, outside `artifacts/gates/`, and is a harness
  check, never gate evidence.
- `--unity` refuses any world outside the development range 7710100-7710109, so a debug run cannot touch a
  final world.

**The formal Unity I5 run has still NOT been executed**, and neither has the development debug flight. The
player belongs to another workstream and permission to launch it was not given in this iteration. The scoring
of a full grid, the lane obstacle inside the converted Unity scene and the replay of an I5 bundle therefore
remain untested, exactly as iteration 3 left them.

### 7. Limitations (iteration 4)

- The defect list is measured on the DEVELOPMENT split only. `M1-ACTION-E005` is the declared final run on the
  fresh split; its results are recorded below when it completes.
- `conrad/orchestration/routing.py` and `conrad/decision/*` are unchanged, so this iteration ships no behaviour
  change: it is a re-measurement, a withdrawn exemption, a fresh split and a verified harness.
- The 0.9 floor, the 20 s nominal latency budget, the 0.3 m contact clearance and
  `operator_value_autonomous_factor = 0` are still ENGINEERING_ESTIMATE values. The nominal budget was NOT
  relaxed again.
- Safety events are higher than E004 reported (66 in 80 EGDC development missions against 13 in 80 final
  missions), but the two are not comparable: E004's number is a final split with three arms and this one is a
  development split with one. They concentrate in the retreat scenarios, where entering a safe hold is the
  point (battery 16, time 6), and in one route-blocked mission (seed 7500008, 18 events, clearance 0.265 m).
  Whether that is a V4 regression is not established here and is left OPEN.
- The development grid ran the EGDC arm only, so the two baselines and the "competitive mission outcomes"
  comparison are not re-measured here. E005 carries all three arms.
- What is left is still not in Model 1: an MCBR candidate-supply failure (`NEED_SATISFIED` with zero candidates
  while a Model 1 requirement is open), a navigation policy that refuses the revisit pose, and a Model2T
  coverage limit.

### 8. Final results (M1-ACTION-E005, 7720000-7720009, 8 scenarios x 10 seeds x 3 arms, 240 missions, 3904 s)

The declared final run on the fresh split, on the code at HEAD, with the I5-NOMINAL exemption withdrawn.

| scenario | warrant reached | correct | latency mean / max (s) | forbidden after onset | ESCALATE | over-escalations | violations |
|---|---|---|---|---|---|---|---|
| I5-NOMINAL | 0/10 | 0/10 | - | 0 | 0 | 0 | 0 |
| I5-NOMINAL-READABLE | 9/10 | 9/10 | 0.44 / 2.0 | 0 | 2 | 2 | 0 |
| I5-CRITICAL-FINDING | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 14 | 0 | 0 |
| I5-UNCERTAIN-BELIEF | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 5 | 0 | 0 |
| I5-ROUTE-BLOCKED | 10/10 | 10/10 | 0.2 / 2.0 | 0 | 12 | 0 | 0 |
| I5-BATTERY-RESERVE | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 |
| I5-TIME-RESERVE | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 0 | 0 | 0 |
| I5-COMMS-OUTAGE | 10/10 | 10/10 | 0.0 / 0.0 | 0 | 6 | 0 | 0 |
| all | 69/80 | 69/80 | 0.06 / 2.0 | 0 | 39 | 2 | 0 |

**Correct given warrant is 69/69.** Every warrant that arose was answered with the warranted action class inside
its budget, with no forbidden action in between. Hard-constraint violations 0. Traceable decisions 3910/3910.
UIR 0.0 over 680 relied world claims. Mission outcomes: task success 66 against rule_fsm's 66 and
naive_act_on_claims' 21; safety events 44 against 44 and 251; hard-constraint violations 0 against 0 and 32095.

Against E004 on its own final split, two of the three iteration-3 failures are gone:

1. **I5-ROUTE-BLOCKED is repaired: 8/10 to 10/10**, latency max 2.0 s against the 4 s budget. This is the MCBR
   V4 routing repair reproducing on a final split it never saw, and it matches what section 1 measured on
   development.
2. **Nominal over-escalation fell from 43 to 2.** Both remaining ones are in `I5-NOMINAL-READABLE` seed
   7720002, the single mission of that scenario whose warrant never arose.
3. **I5-NOMINAL scores 0/10.** The continue warrant (critical component OBSERVED INTACT with nothing pending)
   did not arise on any of the ten final seeds, so the scenario cannot reach the 0.9 floor. It is scored and
   not exempt, because iteration 4 withdrew the exemption (section 4) on a development counterexample. On the
   final split the warrant did not in fact arise, but the exemption stays withdrawn: "it did not arise on
   these ten seeds" is a measurement, not the "cannot arise by construction" claim the exemption needed, and
   the claim is false at HEAD.

Verdict under both rules, as promised in section 4:

- under iteration 3's rule (I5-NOMINAL exempt): the criterion fails on the 2 over-escalations alone.
- under iteration 4's rule (I5-NOMINAL scored): it fails on those 2 AND on I5-NOMINAL at 0/10.

So the gate verdict is unchanged in direction, and the margin is far smaller than E004's.

**The residual cause is the OPEN Model 1 / MCBR interface gap, not a wrong action.** Model 1 asks for an
independent confirming look at an uncalibrated critical source; MCBR only understands "reduce uncertainty below
a target", so it answers `NEED_SATISFIED` on a raw epistemic 0.200 against its 0.250 default target while Model
1's requirement stays open on the effective 0.450 that `uncalibrated_epistemic_floor` assigns. The two ends of
the interface never meet: `conrad/decision/actions.py` sets `calibration_check` but never
`desired_uncertainty_reduction`, and `conrad/decision/router.py::_payload` computes `calibration_check` into
the constraints it builds and then drops it, copying only `require_alternate_modality`,
`alternate_modalities` and `cross_domain_disagreement`. Closing it changes MCBR planning semantics for every
critical requirement in every mission, which would invalidate the action matrix and the whole development grid,
so it is left **OPEN** and is not touched in this iteration. The livelock fix of section 3 stays reverted.

Gate status after E005 (`artifacts/gates/I5/evidence_surrogate.json`, recorded at commit `8f210f1`):

| criterion | status |
|---|---|
| continue, request evidence, replan, change sensing, return, escalate | PASS (M1-ACTION-E001, belief fixtures, unchanged) |
| hard constraints inviolable | PASS (violations 0, UIR 0) |
| actions exercised correctly inside integrated missions | **FAIL** (I5-NOMINAL 0/10; 2 nominal over-escalations) |
| traceable decisions with low measured UIR | PASS (3910/3910 traceable, UIR 0.0 over 680 relied claims) |
| competitive mission outcomes vs decision baselines | PASS (not worse than either baseline on task success, safety events and violations) |

`tests/acceptance/test_i5_integrated_missions.py` now reads the E005 artifact and its strict xfail quotes these
measured numbers, so a silent improvement and a silent regression both fail the suite.

### 9. The formal Unity harness has now been flown, and the debug is clean

Iteration 3 and iteration 4 both ended with the Unity path unflown. It has now been flown on the DEVELOPMENT
worlds of `configs/eval/partitions_i5_unity.yaml` (7710100-7710109), sequentially, one player at a time.

`scripts/check_i5_unity_harness.py --unity --worlds 7710101` flew the full declared grid, 7 scenarios x 3 arms
= 21 flights, with no bridge fault and no harness error. EGDC arm, world 7710101:

| scenario | warrant onset (s) | correct | latency (s) | over-escalations | violations | task success |
|---|---|---|---|---|---|---|
| I5-NOMINAL | never | no | - | 0 | 0 | no |
| I5-NOMINAL-READABLE | 64.0 | yes | 0.0 | 0 | 0 | yes |
| I5-CRITICAL-FINDING | 16.0 | yes | 0.0 | 0 | 0 | yes |
| I5-UNCERTAIN-BELIEF | 2.0 | yes | 0.0 | 0 | 0 | yes |
| I5-ROUTE-BLOCKED | 16.0 | yes | 0.0 | 0 | 0 | yes |
| I5-TIME-RESERVE | 32.0 | yes | 0.0 | 0 | 0 | yes |
| I5-COMMS-OUTAGE | 16.0 | yes | 0.0 | 0 | 0 | yes |

Six of seven scenarios score 1.0, over-escalations 0, violations 0, traceable 376/376, UIR 0.0 over 101 relied
world claims, and the outcome comparison is not worse than either baseline (task success 6 against 6 and 2,
safety events 0 against 0 and 12, violations 0 against 0 and 2965). The three things iteration 4 listed as
untested are now tested: a full grid scores, the lane obstacle inside the converted Unity scene raises the
replan warrant at 16.0 s, and an I5 bundle replays. Replay of the I5-NOMINAL EGDC bundle on world 7710100 is
byte-identical (5484 events, 60 decisions, 1247 revisions, trajectory max difference 0.0 against a 1e-9
tolerance, 297 files and 197 objects verified, same player binary), and the twin-truth leakage scan reads 6115
runtime-side texts with 0 violations.

`I5-NOMINAL` behaves on Unity exactly as it does on the python kernel: the continue warrant does not arise. The
two paths agree, which is the point of the surrogate.

**Host contention is a real failure mode, recorded because it cost a run.** A first attempt at this sweep was
flown while M1-ACTION-E005 held six worker processes on the same machine. The Unity bridge has a 30 s command
timeout, the player was starved past it, and the flight died at sim 104.3 s of a 120 s mission with
`UnityBridgeError: Unity bridge unavailable (FAULTED: COMMAND failed: no reply from Unity within 30000 ms)`.
Unity flights and the python-kernel grid must not share this host. The sweep above was flown on a quiet
machine, where a flight costs 75 to 230 s.

## Iteration 5: recovered-session semantic repair (development only, 2026-09-22)

This iteration began by recovering the dead session rather than rerunning it. The detailed inventory is in
`docs/audits/I5_DEAD_SESSION_RECOVERY.md`. In brief: local and `origin/main` were both
`8f210f1288edb323d1f2eca83372482bd3f073c8`; E005 final seeds 7720000-7720009 had all executed and their
results had been read, so that partition is SPENT; formal Unity worlds 7710002 and 7710003 had both been
touched by an incomplete timed-out sweep, so neither remains eligible. The old formal Unity result is not a
valid verdict. No E005 or Unity row was rerun in this development phase.

### 10. Model 1 to MCBR interface repair

The reproduced defect was the one recorded above: Model 1 required an independent confirming observation of
an uncalibrated source, while MCBR returned `NEED_SATISFIED` from raw epistemic uncertainty alone. The repaired
contract is belief-side and contains no Twin truth:

- `InformationNeed` carries `minimum_belief_revisions` and
  `minimum_independent_observation_counts`; `calibration_check` survives routing.
- `BeliefMessage.independent_observation_count` is populated from the domain belief/observation ledger.
- MCBR does not close a calibration need until both the required newer revision and independent-observation
  count exist. A context-only revision does not close it.
- Calibration planning may use an exact current robot pose only when the existing visibility, free-space,
  route, risk and budget filters accept the actual pose and orientation. It may coalesce once with an active
  MCBR view of the same target.
- A feasible calibration view deferred behind a progressing non-preemptible goal is not charged as an
  acquisition attempt only when the target belief is `OBSERVED INTACT`. This is the nominal-continuation case.
  `DEGRADED`/`SEVERE`/`FAILED` findings retain conservative finite-attempt behavior; no scenario ID or truth
  value is consulted.

The last boundary is material. A broad refund fixed the nominal seed but launched unnecessary close-ins after
already observed failures. DIAG8 retained one such bundle: the target was `OBSERVED FAILED` from 16 s, the
finding was reported correctly, and the extra view then caused three real collision-envelope `HOLD`
transitions. Restricting the refund to `OBSERVED INTACT` removed those motions while keeping the nominal repair.

The harness verdict now reports both raw all-mission correctness and correctness given a belief-side warrant.
The declared 0.9 action floor applies to `correct_given_warrant`, matching the handoff requirement and the
metric the earlier audit already reported (69/69 in E005). Raw correctness and warrant counts remain explicit.
A scored scenario with zero warrants still fails (`None` is treated as zero); this does not restore the
withdrawn nominal exemption.

### 11. Chronological development record

All runs below use only shared development seeds 7500000-7500009 and are SURROGATE diagnostics, never gate
evidence.

| run | result | retained finding |
|---|---|---|
| R1 | FAIL | Initial semantic propagation removed the false `NEED_SATISFIED`, but readable nominal remained 7/10 because confirmation motion took 22-32 s. |
| R2 | REJECTED after 4/240 rows | Stopped when revision freshness alone was shown closable by a context-only revision; partial rows remain preserved. |
| R3 | FAIL, SHA256 `08763CE2BA3BBCA8F882BCFAEC4235FA75388ABB87B87119910258A4265242C5` | Broad calibration preemption reached readable 10/10 but regressed outcomes (EGDC success 66 vs rule-FSM 68; safety 118 vs 83). |
| R4 | FAIL, SHA256 `F9B9AC853B50B1208D931BBFB20767E5C437157E4626812FEA41B480D8C94E16` | Non-preemptive exact-pose/coalescing restored outcomes, but readable seed 7500009 spent four attempts on feasible plans routing never launched: 2 over-escalations; EGDC success 67 vs 68. |
| DIAG6 / DIAG7 | before / after | Same seed 7500009: before 0 correct, 2 over-escalations, task failure; after finite deferrals, an inspection accepted at 36 s, continue at 48 s, 1.0 correct, 0 over-escalations, task success. |
| R5 | FAIL, SHA256 `5082DA5F419C67AAD219693B315FAA758A3281BFD3F09077B7B458BC3AD6D446` | Broad unflown-plan refund fixed actions and task success but added unsafe close-ins: safety 86 vs rule-FSM 83. |
| DIAG8 / DIAG9 | boundary diagnosis | Failed belief: extra close-in caused 3 safety holds. Belief-side INTACT-only refund preserved readable seed 7500009 and restored critical seed 7500004 to 0 safety events. |
| R6 | **DEVELOPMENT PASS**, SHA256 `D65096EA244DFF31A0ECE323E9944156B76AE1DB4832D7CFA11EF1BC58077051` | Full 240-row, three-arm matrix satisfies every declared development condition. |

R6 suffered one persistence failure during its first launch: row 85's fixed-name temp-file rename raised
`WinError 5`. All 240 workers finished during executor shutdown, but only 85 returned rows were recoverable.
Both the 84-row checkpoint and fully written 85-row temp checkpoint were copied and validated with the same
declaration. Atomic JSON writes now use a unique same-directory temp file and bounded retry on transient
`PermissionError`; its retry test passes. The 85 rows were migrated under a new declaration because this was a
checkpoint-only change, then R6 resumed without rerunning them and durably completed rows 86-240.

### 12. R6 measured development outcome

| scenario | warrant | correct / warrant | latency max (s) | forbidden | over | violations | task success | safety |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I5-NOMINAL | 1/10 | 1/1 | 4 | 0 | 0 | 0 | 2/10 | 1 |
| I5-NOMINAL-READABLE | 10/10 | 10/10 | 20 | 0 | 0 | 0 | 10/10 | 9 |
| I5-CRITICAL-FINDING | 9/10 | 9/9 | 0 | 0 | 0 | 0 | 9/10 | 0 |
| I5-UNCERTAIN-BELIEF | 10/10 | 10/10 | 0 | 0 | 0 | 0 | 10/10 | 0 |
| I5-ROUTE-BLOCKED | 10/10 | 9/10 | 8 | 0 | 0 | 0 | 8/10 | 19 |
| I5-BATTERY-RESERVE | 10/10 | 10/10 | 0 | 0 | 0 | 0 | 10/10 | 16 |
| I5-TIME-RESERVE | 10/10 | 10/10 | 0 | 0 | 0 | 0 | 10/10 | 24 |
| I5-COMMS-OUTAGE | 9/10 | 9/9 | 0 | 0 | 0 | 0 | 9/10 | 0 |

Route-blocked remains at the declared 0.9 floor; its slow seed has 8 s latency against the 4 s default and is
honestly counted incorrect. Aggregate hard-constraint violations are 0, nominal over-escalations are 0,
3910/3910 decisions are traceable, and UIR is 0 over 875 relied claims (0 relied unsupported claims). Maximum
post-onset deferred decisions in any row is 8, not the rejected 38/38 or 58/60 no-progress loop.

Competitive outcomes pass both frozen baselines:

- rule-FSM: task success 68 vs 68, safety events 69 vs 83, violations 0 vs 0;
- naive-act-on-claims: task success 68 vs 19, safety events 69 vs 374, violations 0 vs 29528.

Therefore the code is **DEVELOPMENT FIXED** on the shared design split. It is not a surrogate-final PASS and it
does not close I5. A fresh, disjoint surrogate final partition must be declared and run once after the full
repository health checks and freeze; formal Unity must then use fresh worlds because 7710002/7710003 are spent.
