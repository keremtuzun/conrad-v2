# Spatial structural architecture development audit

Base `ae41e467c30afc2be2efa090e3f9a90dce09641d` from
`origin/kerem/i4-i5-full-closure`. The dirty main checkout and the legacy
repository at `6bf01ee` were left untouched. This is a new architectural
workstream. Historical I4 FORMAL FAIL 4/5, I5 FORMAL FAIL 9/10, and I7
SURROGATE FAIL 3/4 / FORMAL NOT_RUN are unchanged.

## Trace of the historical path

`MissionWorld.build` creates one patch Twin2T component and one shared
rest-region component. `MissionSensorSuite._structural` collects visible
surface points, reads the target's T0 state through `Twin2T`, and supplies a
noisy range/bearing/elevation at their mean point. ECMER creates Evidence;
`StructuralAssociator` projects that point and associates it to a registry
surface. `direct.region_of` chooses one Model2T cell and `credit_view`
expands it through the declared 50 degree / 1 m footprint. The resulting
component condition depends on credited coverage, not a measured structural
footprint. The closure documents why that is insufficient for local-defect
identifiability.

## Development slice and acceptance state

| Criterion | State | Evidence / remaining work |
|---|---|---|
| A. Truth expressiveness | DEVELOPMENT PARTIAL | Opt-in static capsule field and local patches. Mission Twin2T dynamics and local crack depth pending. |
| B. Observation locality | DEVELOPMENT PARTIAL | Exact synthetic support integral; pose/visibility/occlusion driven footprint pending. |
| C. Support provenance | DEVELOPMENT PARTIAL | Versioned Observation → ECMER Evidence support and config digest tested. Pose-driven mission producer and complete audit chain pending. |
| D. Local belief update | DEVELOPMENT PARTIAL | Opt-in `SpatialModel2T` refuses multi-cell averaged attribution; production Model2T path pending. |
| E. Conservative status | DEVELOPMENT PARTIAL | Unknown edges/partial coverage refuse intact. Physical detectability model and thresholds pending. |
| F. Identifiability | DEVELOPMENT PARTIAL | Synthetic same-mean pair and resolved local observations tested; mission-path test pending. |
| G. Detectability | DEVELOPMENT PARTIAL | Synthetic minimum-size boundary and qualified claim; empirical calibration unknown. |
| H. Kernel/Unity parity | NOT_RUN | No matched development-world Unity comparison. |
| I. Truth boundary | PARTIAL | Opt-in belief module imports only schemas and Evidence; full runtime leak audit pending. |
| J. Replay | PARTIAL | Digest mismatch fails local ingestion; deterministic mission replay and version migration pending. |

**SPATIAL STRUCTURAL ARCHITECTURE = FAIL (not accepted).** This is an
implementation milestone, not an architecture freeze. No validation/final
or formal world has been opened. The `LOCAL_MAX` development sensor is an
idealized engineering estimate and cannot authorize a physical intact claim.

Development verification at this milestone: `python -m uv run pytest
tests/unit/domains/technical tests/unit/core/test_core_ecmer.py
tests/unit/twins/twin2t/test_t2t_observation.py -q` passed 96 tests, including
the new same-mean, missed-defect, detected-defect, healthy coverage, edge,
resolution, independent-look and compatibility tests. These are synthetic
unit results. They do not measure a calibrated sensor or Unity parity.
An additional opt-in test now exercises spatial truth → synthetic Observation
→ ECMER Evidence → local belief without placing a truth ID in the Observation.
The emission adapter requires the measured footprint and range/bearing to be
supplied; it does not yet calculate them from a mission pose and visibility.
The local belief additionally requires a registry association before crediting
the support. With these tests included, the same targeted command passes
98/98 tests. `ruff check` passes on the new and touched Python files.

## Impact and gate state

I4's structural observations, MCBR value calculation, and energy accounting
would change if the v2 path is activated. Historical I4 evidence stays valid
for its historical architecture, but a new runtime would require a separate
impact measurement and cycle. I7 message size and revision frequency could
also change. The required before/after development communication audit has
not yet run, so **I7 impact = UNDETERMINED**; old surrogate evidence cannot
be declared representative of a new active architecture.

I5 is **NOT REOPENED**. A fresh I5 surrogate or Unity formal run is prohibited
until acceptance A–J pass, architecture validation freezes the contract, and
fresh partitions are declared. I7 formal remains **NOT_RUN** and must wait for
a refreshed 4/4 surrogate pass if impact proves material.
