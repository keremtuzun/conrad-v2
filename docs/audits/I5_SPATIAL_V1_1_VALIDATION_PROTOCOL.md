# I5 Spatial V1.1 v10 validation protocol

Status: **DECLARED / SEALED BEFORE EXECUTION**

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
