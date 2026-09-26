# I5 Spatial V1 post-R5 software remediation

Status: **V1.1 FROZEN; V10 VALIDATION NOT YET OPENED**
Date: 2026-09-26  
Historical R5 verdict: **FAIL (immutable)**

## Scope and integrity boundary

The immutable R5 result remains
`artifacts/experiments/M1-ACTION-SPATIAL-V1-DEV-R5/m1_action_spatial_v1_dev_r5_development.json`
with SHA-256
`4b799f7b1d77e4852c2d8064fe99784a1c76bc362377a563c8e6008ac43147a4`.
It recorded two legitimate nominal warrants out of five and 47 EGDC safety
entries versus 44 for the rule baseline. Nothing in this candidate relabels
that result.

All diagnostics in this document reuse already-spent v9 development worlds
8600005..8600009. They are mechanism checks, not selection, validation, gate,
or final evidence. The v9 validation and final partitions remain unopened.

## Diagnosed defects

Three independent software defects were retained in the R5 bundles.

1. A valid 1 Hz absolute-position-fix sequence could remain outside the EKF
   innovation gate after manoeuvre-model divergence. Rejections then prevented
   the correction required to bring the prediction back into agreement. The
   candidate permits reacquisition only after three consecutive fixes from the
   same source form a max-speed-reachable track from both the last accepted fix
   and the preceding rejected fix. A repeated physically impossible jump stays
   rejected. Reacquisition inflates position covariance in the innovation
   direction before the ordinary correction; it does not snap the state or
   weaken the configured innovation gate.
2. Active-view feasibility reserved only the configured uncertainty margin,
   while navigation can legally complete within an additional position
   tolerance. Candidate and endpoint checks now reserve both quantities.
   Intermediate route points reserve uncertainty only. A stationary generic
   hold does not treat its completion tolerance as commanded drift; a full
   attitude hold remains subject to the endpoint reserve.
3. The conservative capsule visibility certificate stopped after two recursive
   subdivisions. A retained flown view had all 231 dense truth-side samples
   visible, but the certificate rejected its 0.2 m resolution cell at depths
   zero through four and proved it at depth five. The candidate therefore uses
   depth five and versions the replay-sensitive contract as
   `capsule-sdf-recursive-v2`. The historical v1 identifier and evidence remain
   unchanged.

## Focused verification

The corrected estimator, navigation/boundary contract, visibility certificate,
spatial predictive model, and mission integration passed 170 focused tests
(82 navigation/estimation/visibility tests and 88 MCBR/spatial mission tests).
Ruff format and lint checks passed on every changed source and test file.

## Spent-world mechanism check at the unchanged R5 contract

The diagnostic retained R5's declared `cruise_speed_fraction: 0.4`, sensor,
truth, support, Model2T, warrant, thresholds, durations, attempt bounds, arms,
and scoring. Results were:

| Seed | EGDC nominal warrant/task | Rule nominal warrant/task | EGDC safety entries | Rule safety entries |
|---:|:---:|:---:|---:|---:|
| 8600005 | PASS | FAIL | 0 | 0 |
| 8600006 | PASS | PASS | 0 | 0 |
| 8600007 | FAIL | FAIL | 0 | 0 |
| 8600008 | PASS | PASS | 0 | 0 |
| 8600009 | FAIL | FAIL | 0 | 0 |

The primary arm therefore reaches three of five legitimate nominal warrants,
the old prospective floor, while every checked nominal world is safety-clean.
The two negative primary worlds remain negative. On the two retained
communication-outage worlds (8600006 and 8600007), both arms retained task
success with zero safety entries and zero supervisor refusals.

A separate 0.8-speed stress diagnostic stayed negative on nominal seeds
8600005 and 8600009; only 8600006 completed. That stress result is retained as
a negative diagnostic and is not used to justify advancement.

## Advancement boundary

This mechanism is viable on spent development worlds, but it changes the
replay-visible visibility-certificate contract. Before any new I5 validation or
final world is read it requires:

1. a fresh partition declaration;
2. versioned Spatial V1 visibility-certificate revalidation, including kernel
   and Unity semantic parity and replay mismatch refusal;
3. a complete fresh development matrix under a preregistered rule; and
4. a commit and remote push of the candidate and declarations.

No new validation or final partition has been opened.

## Fresh V1.1 R6 development result

The V1.1 candidate, fresh v10 partition, and R6 advancement rule were committed
and pushed before any v10 seed was opened. R6 then executed the complete
preregistered development matrix: ten seeds, seven scenarios, and three arms
(210 unique missions, with no missing or duplicate rows). Only development
seeds 8700000..8700009 were used. Validation seeds 8700100..8700109 and final
seeds 8700200..8700239 remain unopened.

Artifact:
`artifacts/experiments/M1-ACTION-SPATIAL-V1-1-DEV-R6/m1_action_spatial_v1_1_dev_r6_development.json`
SHA-256:
`7c3b7c2549aa3a1b426b71431e223a5e37d74e2fecff4f260b78b07fb7efea3e`

The measured result passes every preregistered R6 advancement predicate:

- primary nominal warrants: 10/10 (minimum 6/10);
- non-nominal correct-given-warrant: 60/60 = 1.0;
- false intact on a covered, resolvable defect: 0;
- primary decision violations: 0;
- nominal over-escalations: 0;
- traceability: 5110/5110 = 1.0;
- unsupported inference rate: 0/1913 = 0.0; and
- pooled primary outcomes: 59 task successes, 18 safety entries, and 0
  violations, versus 59/24/0 for rule FSM and 20/490/41875 for the naive arm.

The preregistered safety-outlier inspection found ten entries in the
battery-reserve scenario and eight in two route-blocked worlds. The battery
entry is the expected transition into safe hold and is identical across the
primary and rule arms. In route-blocked seeds 8700000 and 8700009, the primary
and rule arms have identical actions, trajectories, task results, and safety
counts (2 and 6 respectively). Seed 8700007 misses the 0.30 m obstacle-clearance
criterion in both arms and remains a negative task outcome. These results are
retained in the pooled 59/70 task-success count; no outlier was removed or
relabelled.

R6 is synthetic Python L1 development evidence only. It does not qualify
Spatial V1.1, reopen the immutable Spatial V1 freeze, or authorize validation.
Versioned controlled-support, identifiability, truth-boundary, replay-mismatch,
and Unity semantic-parity checks must pass before a validation declaration can
be frozen and any v10 validation seed opened.

## Architecture revalidation completion

The controlled V1.1 matrix passed 28/28, the focused contract/boundary suite
passed 102/102, and the separately preregistered native-macOS Unity supplement
passed 16/16 on fresh Unity seeds 2003 and 2004. The latter is bounded to Unity
6000.5.9f1 and player executable SHA-256
`7e42f64199b043204d62737d06aabc6b6ceb1f3c1c699f158a612b938ea87ac6`.
The Windows-pinned protocol remains NOT RUN and was not relabelled.

Spatial V1.1 is therefore frozen in
`../architecture/SPATIAL_V1_1_FREEZE.md`. Validation seeds 8700100..8700109
and final seeds 8700200..8700239 remain unopened. The next permitted step is a
separately committed prospective v10 validation declaration; this architecture
freeze is not an I5 validation or final pass.
