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
