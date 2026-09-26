# I5 Spatial V1.1 v10 one-shot surrogate final protocol

Status: **SURROGATE FINAL PASS**

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

## Recorded result

The one-shot run completed all 588 declared missions without opening the
barred suffix. The result SHA-256 is
`0dbc96d699a698ed95011330692533312094d941b5e3fb9c9bf25a965e10e3d1`.
The durable checkpoint SHA-256 is
`ade1a1b2ecc818fd4a2fdcfd29d6e351499c73374b2068390f8732ed202ffd24`.
The committed compact checkpoint manifest SHA-256 is
`181ab792565f4226de57b78ba51c8144349fd66600b53c40886c909aff5d260e`;
it preserves the declaration, source identity, row count, canonical row hash,
and full-checkpoint hash without committing the 34 MB duplicate checkpoint.

The checkpoint declaration pins source commit
`d65ce7bcc22f313656ecebf0cb94d2fcc560ad0d` and an empty tracked diff over
`conrad/`, `configs/`, and `scripts/` (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

The independent raw-row verifier passed every fail-closed check. Its artifact
SHA-256 is
`39d5252fd8313da6ece3fd057e55077e3b6c3b94ec8e29645e1779822faec203`,
and the verifier source SHA-256 is
`0cd776e7f552a220b09344d65dbdbd23f6f95aec91bbe4bb2d74e1ec27f67fd2`.

Measured primary results:

- exact grid: 588/588 unique rows;
- nominal warrants: 25/28 (threshold 19/28);
- every non-nominal scenario: 28/28 warrants;
- correct given warrant: 1.0 except route-blocked at 27/28 =
  0.9642857142857143;
- false intact, primary decision violations, and nominal over-escalations: 0;
- traceability: 1.0; UIR: 0.0; and
- pooled task-success / safety-entry / violation counts: EGDC 162/72/0,
  rule FSM 162/74/0, naive 54/1520/118654.

### Negative and outlier audit

The PASS does not erase negative rows. Nominal seeds 8700200, 8700207, and
8700223 never reached an intact warrant and ended with unknown or mixed
knowledge; they were not counted as correct. Route-blocked seed 8700207 reached
its warrant at 20 s, chose `WAIT`, issued the expected replan at 26 s (6 s,
outside the 4 s budget), passed the obstacle with only 0.2511672213 m minimum
clearance, accumulated seven safety-state entries, and failed task success.
The rule FSM had the same miss on that world. Route-blocked seed 8700218 issued
the correct replan but still failed task success with six safety-state entries.
These outcomes remain in the immutable result and are why route correctness is
27/28 rather than a perfect result.

All uncertain-belief missions correctly requested more evidence, but the
mission outcome definition does not mark that scenario as task success because
the critical condition remains unknown. This is preserved as a limitation, not
relabelled as success.

This PASS authorizes only a separately preregistered formal Unity V1.1 run on
fresh worlds. It does not change the historical I5 formal result (9/10 FAIL),
and it makes no claim about physical, ROS, HIL, or deployment readiness.
