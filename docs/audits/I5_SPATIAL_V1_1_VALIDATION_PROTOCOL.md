# I5 Spatial V1.1 v10 validation protocol

Status: **PASS; FINAL TEST STILL SEALED**

The separately frozen Spatial V1.1 architecture permits validation but does
not pass it. This protocol opens all ten and only the v10 validation seeds
8700100..8700109. Final seeds 8700200..8700239 remain sealed.

The experiment is `M1-ACTION-SPATIAL-V1-1-VAL`, using the same seven
scenarios, three arms, mission durations, speed, sensor, truth, support,
Model2T, warrant definitions, latency budgets, action floor, safety metrics,
and competitiveness comparison as R6 development. Validation changes no model
or threshold. It measures independent warrant incidence and rejects a
candidate that cannot exercise its declared action classes.

The validation configuration file SHA-256 is
`aaa32e4a772c3f61aed0c628af715302bb434d2146ff1c3597019ea653a30aa1`.
Its parsed, sorted compact-JSON SHA-256 is
`799ea102be4724df23b9c8fa893c4ee8de3801d78833db25570b657a799ee006`.

## Frozen selection rule

Validation passes only if all conditions below hold:

1. at least 6 of 10 primary `I5-NOMINAL` missions reach the legitimate intact
   warrant;
2. every non-nominal scenario reaches its warrant at least once and has
   correct-given-warrant rate at least 0.9;
3. false intact on a covered, resolvable defect is zero;
4. primary decision violations and nominal over-escalations are both zero;
5. decision traceability is 1.0 and UIR is 0.0; and
6. pooled primary task success is at least, safety entries at most, and
   decision violations at most each frozen baseline.

All 210 seed/scenario/arm missions must be present exactly once. Missing,
duplicate, interrupted, or undeclared rows fail validation. Safety outliers
remain in the pooled result and are inspected, not deleted.

If and only if validation passes, its measured nominal-warrant incidence may
feed a separately committed final-sample power calculation. Before any final
world is opened, that declaration must freeze the final prefix, sample size,
minimum nominal-warrant count, all per-scenario and safety thresholds, player
scope, and one-shot stop rule. This validation artifact is synthetic Python L1
selection evidence, not final, formal Unity, physical, ROS, HIL, or deployment
evidence.

## Result recorded 2026-09-26

The complete declared grid finished with 210/210 unique rows: ten seeds, seven
scenarios, and three arms, with no missing or duplicate mission. The result was
produced from source commit
`650140b6de312156df88194dc9913b4b17594d19` with an empty tracked
`conrad`/`configs`/`scripts` source diff. Artifact:

`artifacts/experiments/M1-ACTION-SPATIAL-V1-1-VAL/m1_action_spatial_v1_1_val_validation.json`

SHA-256:
`a3c27c0edc277f79dd29d7b9b71b6aba65337ff8bdc583d2375ce98a5b6687e1`.

The machine-readable verifier returned PASS for every frozen predicate. Its
SHA-256 is
`54145f8d61eb27f9543f6f3cc2a54e0d56e5b03ddfceec1412deff6ff845821b`.
Measured results were:

- primary nominal warrants: 10/10;
- non-nominal warrants: 10/10 in every scenario except route-blocked 9/10;
- non-nominal correct-given-warrant: 1.0 in every scenario;
- false intact on covered, resolvable defects: 0;
- primary decision violations: 0;
- nominal over-escalations: 0;
- traceability: 1.0;
- UIR: 0.0; and
- pooled primary task success/safety/violations: 57/88/0, equal to rule FSM
  at 57/88/0 and better than naive at 20/719/41621.

The missing route-blocked warrant is seed 8700102. EGDC and rule FSM have the
same trajectory and result there: no warrant, 0.23 m minimum clearance, 74
safety entries, and a negative task outcome. Both arms also have identical
pooled safety totals. Ten battery entries are the expected one-event reserve
transition. Four other one-event cases at seed 8700107 are mirrored by rule
FSM. All entries remain in the pooled result; none was removed or relabelled.

Validation therefore passes the frozen selection rule. Final seeds
8700200..8700239 remain unopened pending a separately committed power and
one-shot final declaration.
