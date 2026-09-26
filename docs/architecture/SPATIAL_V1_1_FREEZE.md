# Spatial Structural Architecture V1.1 freeze

Status: **SOFTWARE VALIDATED ON PINNED macOS UNITY PLAYER; SYNTHETIC_ONLY**
Frozen: 2026-09-26

This freezes the separately versioned Spatial V1.1 software contract. It does
not modify, overwrite, or relabel the historical Spatial V1 freeze at
`SPATIAL_V1_FREEZE.md`. The V1 implementation and its Windows-player evidence
remain immutable.

## Pinned contract

The architecture contract is the candidate implemented at source commit
`942b5799394a5ae555d7d1e7880032c0bb10dc9e`. It retains every identifier and
digest in the Spatial V1 freeze except for the visibility certificate:

| Element | Frozen V1.1 identifier |
|---|---|
| Mission truth | `twin2t-spatial-v1` |
| Local evolution | `twin2t-spatial-evolution-v1` |
| Structural sensor | `structural-sensor-v2` |
| Detectability | `spatial-synthetic-detectability-v2` |
| Visibility certificate | `capsule-sdf-recursive-v2`, maximum recursive depth 5 |
| Support schema | `structural-observation-v2` |
| Registration | `exact-bounded-or-unregistered-v2` |
| Association | `spatial-structural-association-axial-v3` |
| Mission Model2T | `model2t-spatial-v1` |
| Native Unity player executable SHA-256 | `7e42f64199b043204d62737d06aabc6b6ceb1f3c1c699f158a612b938ea87ac6` |
| Unity editor | `6000.5.9f1 (b57deb96f08d)` |

The only architecture-level model change from V1 is the deeper conservative
capsule visibility proof. Unresolved subregions still deny healthy coverage.
Sensor parameters, detectability, support, registration, association,
required domain, Model2T, and condition thresholds are unchanged. The adjacent
EKF reacquisition and navigation endpoint fixes are runtime-contract repairs,
not changes to the structural truth or belief model.

## Acceptance evidence

The candidate and its v10 development rule were declared and pushed before any
v10 world was opened. The R6 development matrix then passed its full
preregistered rule across 210/210 unique missions at result commit
`9549c23cadda98f1921bb6edfa0eb3d5a6c7594d`:

- 10/10 nominal warrants;
- 60/60 non-nominal correct-given-warrant;
- zero false intact on covered, resolvable defects;
- zero primary decision violations and zero nominal over-escalations;
- 5110/5110 traceability and 0/1913 unsupported inferences; and
- pooled primary success/safety/violations 59/18/0 versus rule 59/24/0.

The separately declared controlled architecture matrix passed 28/28 on fresh
seeds 2001 and 2002 at result commit
`137ee0e8323ba9377a91c18b22530346305dfed9`. Its retained SHA-256 values are:

| File | SHA-256 |
|---|---|
| `protocol.json` | `e7bceb2ed001c20e82f94d7ba1871f3d08d53d1c5ddba605f9367c8e8f1c69b6` |
| `results.jsonl` | `2c4d97edf5d420ad77f464186c2e47448673fa7d683f28e0d7a8d38b09f83510` |
| `summary.json` | `2adafbd1ce532998304e2649a9ef5cd8a3c26966239109d0f44c50d1b61ce7f8` |

An additional 102/102 focused contract and boundary tests passed, covering
exact replay and edited-identity refusal, static and runtime truth isolation,
structural lineage, adversarial local identifiability, visibility geometry,
EKF reacquisition, and navigation endpoint margins. The broad remote-clone
regression at `9dff6fc764cd52e0bc83409bc411e2165ebb1c80` passed Ruff, mypy,
the secret scan, and pytest with 1497 passed, 15 documented skips, 125 opt-in
Unity tests deselected, and 3 immutable expected failures.

The original V1.1 Windows-pinned Unity protocol remains **NOT RUN** and its
seeds 2001 and 2002 were never opened by Unity. After the license blocker was
cleared, a separate native-macOS supplement was declared and pushed at
`7704f6a0e3bbfb6581038beb0db3b87c35377b36`, before its fresh Unity seeds were
opened. It passed 16/16 at result commit
`3c9adec495eacb2b50b4bbbab9b37038bad5dda7` on seeds 2003 and 2004. Its
retained SHA-256 values are:

| File | SHA-256 |
|---|---|
| `protocol.json` | `2ca446bf5045049cc9bb8cbac8b2b9a01b7989eb8ffd10aa352ea56c167b2e3f` |
| `results.jsonl` | `d606993f9bb50814fbf8090dba6fcd894c5f2a46f82d3bd9bcc2a92d6eb0c393` |
| `summary.json` | `45fd75debe10fa8d950fc88edb2bb3d2f6174696aced5ed37a6873683392a23b` |

All 16 declared clear/occluded, deterministic/noisy, and exact/bounded cases
passed with the frozen semantic tolerances and a single player at a time.

## Scope and advancement boundary

This freeze is synthetic software evidence for the exact macOS player hash on
the declared arm64 host. It is not execution of the historical Windows player
and does not establish physical sensor performance, vehicle dynamics fidelity,
ROS integration, HIL, endurance, underwater communication quality, or
deployment authorization.

The v10 validation seeds 8700100..8700109 and final seeds
8700200..8700239 remain unopened at freeze time. This freeze permits a separate
prospective v10 validation declaration; it does not itself pass I5 validation
or final. Any change to the visibility algorithm or its replay identity, or to
the frozen sensor, support, registration, association, Model2T, required-domain,
or condition contracts, requires a new version and independent evidence.
