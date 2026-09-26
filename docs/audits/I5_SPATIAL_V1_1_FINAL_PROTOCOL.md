# I5 Spatial V1.1 v10 one-shot surrogate final protocol

Status: **DECLARED / SEALED BEFORE EXECUTION**

Validation passed 10/10 nominal warrants and every frozen selection predicate.
The exact validation result SHA-256 is
`a3c27c0edc277f79dd29d7b9b71b6aba65337ff8bdc583d2375ce98a5b6687e1`.
No v10 final seed was opened before this declaration.

The final configuration file SHA-256 is
`a98ec719f901b79574d59e9511eeac4ee5779038716ddcaa949271c2bfe14bbf`.
Its parsed, sorted compact-JSON SHA-256 is
`ad663fdcf827814f4b6d3947728c92634400cce5881f2152f54628427bab7451`.

## Power and fixed final prefix

For the all-success validation result, the one-sided 95% exact
Clopper-Pearson lower bound on nominal-warrant probability is
`0.7411344491069477`. The final design tests the null incidence `p = 0.5`
with one-sided alpha 0.05 and requires at least 0.8 power at that conservative
validation bound.

The null `p = 0.5` is the fail-closed boundary below the already-declared
majority warrant-reachability requirement (R6 and validation both required at
least 6/10); it was not selected from any final observation.

Searching the available final prefix prospectively, the smallest qualifying
design is 28 worlds with at least 19 nominal warrants:

- false-positive probability at `p = 0.5`: `0.04357927665114403`;
- power at `p = 0.7411344491069477`: `0.8352075352835021`; and
- power artifact SHA-256:
  `cecf8a6593194654100089410f51816cf1cf50c0f7c7abc86292fa474ef2195c`.

The fixed prefix is 8700200..8700227. The unused suffix 8700228..8700239 is
not a reserve for post-result supplementation and must remain unopened after
this one-shot run.

## Frozen final rule

Run exactly 28 seeds x 7 scenarios x 3 arms = 588 unique missions under the
unchanged validation contract. Final passes only if all conditions hold:

1. primary nominal warrants are at least 19/28;
2. every scenario reaches at least one warrant and its
   correct-given-warrant rate is at least 0.9;
3. false intact on covered, resolvable defects is zero;
4. primary decision violations and nominal over-escalations are zero;
5. traceability is 1.0 and UIR is 0.0; and
6. pooled primary task success is at least, safety entries at most, and
   decision violations at most each frozen baseline.

All rows must be present exactly once. Any missing, duplicate, interrupted, or
failed predicate is a final FAIL. There is no threshold change, seed addition,
failed-world rerun, or use of the unused suffix after observation.

This is one-shot synthetic Python L1 surrogate-final evidence. A PASS may
authorize a separately declared formal Unity final on fresh reserved worlds
and the already-frozen native player hash. It is not itself formal Unity,
physical, ROS, HIL, or deployment evidence. A FAIL remains a FAIL and does not
authorize formal Unity.
