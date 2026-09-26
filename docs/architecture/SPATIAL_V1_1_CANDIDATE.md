# Spatial Structural Architecture V1.1 candidate

Status: **PREREGISTERED CANDIDATE; NOT FROZEN**  
Declared: 2026-09-26, before opening the v10 partition

Spatial Structural Architecture V1 remains frozen at source commit
`8c79b7127dc1424fc093e9c0814e41ca64e66b00` with visibility certificate
`capsule-sdf-recursive-v1`. This document declares a separately versioned
candidate; it does not modify or supersede that historical freeze.

## Candidate delta

The candidate changes one architecture-level replay contract:

| Element | Frozen V1 | V1.1 candidate |
|---|---|---|
| Visibility certificate | `capsule-sdf-recursive-v1`, maximum recursive depth 2 | `capsule-sdf-recursive-v2`, maximum recursive depth 5 |

The deeper bound is a correctness repair for a retained small-radius flown
view that the depth-two proof rejected despite complete dense truth-side
visibility. It remains conservative: unresolved subregions are still denied
healthy coverage. Sensor parameters, support schema, registration,
association, Model2T, detectability, required domain, and condition thresholds
are unchanged.

The same candidate also repairs mission-runtime contracts outside the spatial
model identity:

- EKF absolute-fix reacquisition requires a three-fix, same-source,
  max-speed-reachable measurement track anchored to the last accepted fix;
- active-view and navigation endpoint checks reserve both the configured
  uncertainty margin and accepted endpoint error; and
- intermediate route points and generic stationary holds retain their narrower
  semantically correct margins.

## Acceptance rule

Before V1.1 can be frozen, the existing Spatial A--J criteria must be
reconfirmed for the changed contract. At minimum:

1. the controlled support/identifiability matrix remains fully passing;
2. covered resolvable defects produce zero false intact;
3. matched kernel/Unity support, measurement, credited cells, and condition
   remain within the already-declared semantic tolerances;
4. replay accepts the exact v2 contract and refuses edited visibility,
   support, Model2T, or sensor identities;
5. static/runtime truth-boundary checks remain at zero violations; and
6. the full v10 development rule in
   `configs/eval/m1_action_spatial_v1_1_development_r6.yaml` passes.

Failure of any item keeps V1 frozen and V1.1 unfrozen. Validation and final
worlds remain sealed until the candidate and declarations are committed and
pushed, then development passes without changing this rule.

The canonical SHA-256 digest of the parsed v10 partition declaration is
`359b7de4fd0172458ba960881402be05f0551a2dcd49aebeeaf827a0ec18e365`.
The canonical digest of the parsed R6 development declaration is
`3c5e8f0adb6357dc07c602e739a4f9fc3a3dd81c44be98d4890b919eaf4df536`.
