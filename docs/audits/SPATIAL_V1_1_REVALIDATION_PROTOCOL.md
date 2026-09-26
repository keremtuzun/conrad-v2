# Spatial Structural Architecture V1.1 revalidation protocol

Status: **CONTROLLED/CONTRACT PASS; UNITY PARITY BLOCKED**

This protocol revalidates the separately versioned Spatial V1.1 candidate
declared in `../architecture/SPATIAL_V1_1_CANDIDATE.md`. It does not modify the
immutable Spatial V1 freeze or any historical result.

The candidate delta is limited to visibility certificate
`capsule-sdf-recursive-v2` with maximum recursive depth five. Sensor,
detectability, support, registration, association, Model2T, required-domain,
condition, and parity thresholds remain unchanged.

## Fresh partition

Seeds 2001 and 2002 are reserved for this protocol. Before this declaration,
they were not used by a committed Conrad spatial protocol, configuration, test,
or audit. Both matrices must run once without selecting cases or changing the
acceptance rule after results are observed.

## Controlled mission-path matrix

`scripts/validate_spatial_v1_1_case_matrix.py` inherits the frozen 14-case
matrix and runs it for both fresh seeds: 28 cases total. All 28 must pass.
Covered resolvable corrosion, crack, multiple-defect, edge-defect, and local
same-mean cases must never produce `INTACT`. Partial, outside-support, and
occluded-defect controls must remain unqualified. Healthy, subthreshold,
subresolution, heterogeneous-healthy, and uniform-same-mean cases must retain
their declared expected conditions without changing thresholds.

## Kernel/Unity semantic parity

`scripts/validate_spatial_v1_1_unity_parity.py` inherits the frozen eight-case
cross product for each seed: clear/occluded, deterministic/noisy, and
exact/bounded survey, for 16 cases total. It pins Unity player SHA-256
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`.
All 16 must pass with exactly matched supports and inline measurements, matched
association, credited-cell coverage within absolute tolerance `1e-12`, matched
cell and component condition, zero kernel-visible/Unity-hidden rows, positive
bounded coverage, and an actually blocked view in occluded cases. Only one
Unity player may run at a time.

## Contract and boundary checks

The current integration, leakage, and replay suites must demonstrate that:

- the exact v2 visibility identity replays;
- edited visibility, support, Model2T, and sensor identities are refused;
- mission truth remains absent from runtime decision and belief records;
- covered resolvable defects produce zero false intact; and
- estimator reacquisition and endpoint-boundary regressions remain passing.

Spatial V1.1 may be frozen only if both prospective matrices and all contract
and boundary checks pass, and the already-preregistered v10 R6 development
matrix passes. A missing Unity player, a skipped Unity case, or any failure is
not a pass. No v10 validation or final seed may be opened until the complete
result is committed and pushed and a separate validation declaration is frozen.

## Result recorded 2026-09-26

The controlled mission-path matrix completed 28/28 without failures or
interruptions. Its retained files are:

| File | SHA-256 |
|---|---|
| `protocol.json` | `e7bceb2ed001c20e82f94d7ba1871f3d08d53d1c5ddba605f9367c8e8f1c69b6` |
| `results.jsonl` | `2c4d97edf5d420ad77f464186c2e47448673fa7d683f28e0d7a8d38b09f83510` |
| `summary.json` | `2adafbd1ce532998304e2649a9ef5cd8a3c26966239109d0f44c50d1b61ce7f8` |

An additional 102/102 focused tests passed across mission belief/truth
integration, exact replay identity and edited-identity refusal, static and
dynamic truth-boundary enforcement, structural lineage, adversarial local
identifiability, visibility geometry, EKF fix reacquisition, and navigation
endpoint margins.

Unity parity preflight stopped with `Unity player unavailable` before creating
the output directory or opening either reserved Unity world. No
`ConradSim.exe` exists in the repository, Git history, GitHub releases,
Actions artifacts, or another indexed Conrad checkout.

Unity Hub 3.21.3 and the exact Apple-silicon editor 6000.5.9f1 changeset
`b57deb96f08d` were then installed. The batch project-compile preflight stopped
before project import with exit code 198 and `No valid Unity Editor license
found`. The Mac was locked when Unity Hub was opened for the required account
sign-in. Therefore the 16-case Unity parity matrix is **NOT RUN**, Spatial V1.1
remains unfrozen, and v10 validation/final remain sealed. The controlled and
contract passes do not substitute for this missing software execution gate.
