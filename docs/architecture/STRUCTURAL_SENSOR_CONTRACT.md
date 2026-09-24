# Structural sensor contract, development version

Version: `structural-observation-v2` / `structural-sensor-v1`. This is a
synthetic architecture contract, **not a calibrated physical sensor**. The
existing T0 and historical gate paths retain their old meaning. A later
sensor specification or measurement campaign must replace the estimates
below before physical validation claims. A formal Unity I5 cycle can still
produce software/simulation evidence after architecture and surrogate gates
pass.

`structural-sensor-v2` is an additional synthetic contract. It partitions an
already visible and clipped surface rectangle into axial and lateral resolution
cells. Each cell returns a peak only for local features meeting the declared
minimum resolvable size. From detectability contract
`spatial-synthetic-detectability-v2`, a crack also needs both its local length
and depth to meet separate, explicit positive thresholds. Patch footprint size
is a separate test; it does not stand in for crack length or depth. The cell
geometry and detectability values are
`ENGINEERING_ESTIMATE`; the peak response is an idealized model hypothesis.
For v2 only, a resolvable patch that straddles a cell boundary produces a
peak in each cell with nonzero overlap. This is deliberately conservative
against false intact labels and is an uncalibrated response hypothesis.
The mission response first uses the pose-derived support and continuous
visibility certificate. Range and orientation dependence are deterministic
step responses at the declared working-range and field-of-view/incidence
limits. The response uses only truth under the visible support, then applies
the declared synthetic measurement noise. These are software hypotheses,
not measured detection curves.

## Parameter authority

| Parameter | Units | Current value | Source | Status |
|---|---|---|---|---|
| Horizontal and vertical FOV | deg | 100, 80 | `StructuralSensorOptions` defaults | ENGINEERING_ESTIMATE |
| Working range and standoff | m | 0.3 to 4.0 | `StructuralSensorOptions` defaults | ENGINEERING_ESTIMATE |
| Range noise | m, one sigma | 0.05 | `StructuralSensorOptions` default | ENGINEERING_ESTIMATE |
| Angle noise and orientation uncertainty | rad, one sigma | 0.02 | `StructuralSensorOptions` default | ENGINEERING_ESTIMATE |
| Legacy credit half-angle | deg | 50 | `CoverageConfig` default, belief only | ENGINEERING_ESTIMATE |
| Legacy axial credit | m | 1.0 | `CoverageConfig` default, belief only | ENGINEERING_ESTIMATE |
| Physical footprint width and height | m | unavailable | No payload measurement | UNKNOWN |
| Synthetic footprint width and height | m | explicit per configuration, no default | `StructuralSensorModel` | ENGINEERING_ESTIMATE |
| Physical axial and lateral resolution | m | unavailable | No payload measurement | UNKNOWN |
| Synthetic axial and lateral resolution | m | explicit per configuration, no default | `StructuralSensorModelV2` | ENGINEERING_ESTIMATE |
| Physical minimum corrosion patch | m | unavailable | No payload measurement | UNKNOWN |
| Synthetic minimum corrosion patch | m | explicit per configuration, no default | `StructuralSensorModel` | ENGINEERING_ESTIMATE |
| Physical minimum crack length and depth | m | unavailable | No payload measurement | UNKNOWN |
| Synthetic minimum crack patch size | m | explicit per configuration, no default | `StructuralSensorModel.minimum_resolvable_crack_m` | ENGINEERING_ESTIMATE |
| Synthetic minimum crack length and depth | m | explicit per configuration, no default | `StructuralSensorModelV2.minimum_detectable_crack_length_m`, `minimum_detectable_crack_depth_m` | ENGINEERING_ESTIMATE |
| Physical structural measurement noise | m, one sigma | unavailable | No payload measurement | UNKNOWN |
| Synthetic structural measurement noise | m, one sigma | explicit per configuration, no default | `StructuralSensorModel` | ENGINEERING_ESTIMATE |
| Pose position uncertainty | m | absent unless pose covariance supplied | `Pose.covariance_6x6` | UNKNOWN |
| Support edge uncertainty | m axial, rad angular | absent unless geometry supplies bounds | `CapsuleSurfaceSupport` | UNKNOWN |
| Synthetic support representation | m axial, rad circumferential | capsule rectangle | `CapsuleSurfaceSupport` | SPECIFIED |
| Synthetic aggregation rule | enum | `AREA_MEAN`, `LOCAL_MAX`, `RESOLUTION_CELL_SAMPLES` | sensor model version | SPECIFIED |
| Physical aggregation rule | response function | unavailable | No payload datasheet | UNKNOWN |

For the v2 synthetic model, `axial_resolution_m`, `lateral_resolution_m`,
`minimum_detectable_crack_length_m`, and `minimum_detectable_crack_depth_m`
are required configuration values in metres. There is no default. Its
`RESOLUTION_CELL_SAMPLES` mode is not a physical measurement claim. An
`AREA_MEAN` response remains ambiguous for same-global-mean surfaces;
`LOCAL_MAX` remains a diagnostic upper bound. All three names are versioned
in the support and sensor configuration digest.

`OBSERVED_INTACT` in the local model means that, under the declared synthetic
peak-response and noise assumptions, no defect above all applicable declared
detectability limits was found over every required surface cell. Crack length
and depth must both clear their limits to count as detectable. It does not rule
out smaller defects or establish a physical detection probability. Unknown
support edges, association failure, insufficient coverage, or a coarse area
mean prohibit that status.

Sources reviewed include `conrad/sim/mission/options.py`, `conrad/domains/technical/config.py`,
`conrad/twins/twin2t/config.py`, `docs/audits/STRUCTURAL_LINEAGE_AUDIT.md`,
`docs/audits/MODEL2T_REPAIR.md`, and the I4/I5 closure. The architectural
source of actual support dimensions remains open. Existing nominal FOV is a
view limit; it does not establish the spatial resolution of the T0 structural
scalar.

## V2 measurement and evidence meaning

`CapsuleSurfaceSupport` names the region **contributing to one measurement**
using axial metres and angular radians on the surveyed capsule design surface.
It has no belief-cell IDs. A seam-crossing support must be split. The sensor
configuration digest, model version, edge uncertainty and clipping flag are
part of the support. `Observation.structural_support` is copied into
`Evidence.structural_support` by ECMER; `SpatialSupport` continues to locate
the measured point for association. An associated registry identity is a
separate inference result, never a Twin truth ID.

The synthetic `AREA_MEAN` kernel integrates only truth inside support. It
cannot rule out local maxima even under broad full coverage. The optional
`LOCAL_MAX` kernel is an *idealized hypothesis*, not a claim that the current
payload has this response. It reports the maximum intersected local state
only for patches meeting the explicitly declared minimum resolvable size.
Smaller patches remain outside the intact claim. Values and measured support
must originate from the same sensor configuration.

`SpatialModel2T` accepts only compatible v2 evidence. An unknown support edge,
an average spanning several belief cells, or an uncertain edge crossing a
cell boundary cannot supply healthy cell coverage. Evidence must also carry
an inference-side association to the component's registry identity. Coverage is computed from
the union of guaranteed support inside each belief cell. A qualified
`OBSERVED_INTACT` means no defect above the **declared detectability scale**
was found in the covered domain under the idealized local-response model; it
does not assert microscopic intactness. A physical version requires measured
response, noise, pose, and support uncertainty before that meaning can be
promoted.

Old bundles with no `structural_support` still parse under the optional field.
They are treated as legacy and cannot be silently interpreted as v2 support.
The replay version/digest check is implemented at v2 local ingestion; a
mission-wide replay migration is still required.
