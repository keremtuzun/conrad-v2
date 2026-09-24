# Structural sensor contract, development version

Version: `structural-observation-v2` / `structural-sensor-v1`. This is a
synthetic architecture contract, **not a calibrated physical sensor**. The
existing T0 and historical gate paths retain their old meaning. A later
sensor specification or measurement campaign must replace the estimates
below before physical or formal I5 claims.

## Parameter authority

| Parameter | Meaning and units | Repository source | Status |
|---|---|---|---|
| FOV, horizontal / vertical | Angular view limits, deg | `StructuralSensorOptions.hfov_deg=100`, `vfov_deg=80` | ENGINEERING_ESTIMATE |
| Operating range | Standoff interval, m | `StructuralSensorOptions.min_range_m=0.3`, `max_range_m=4.0` | ENGINEERING_ESTIMATE |
| Range noise | One-sigma range, m | `StructuralSensorOptions.range_sigma_m=0.05` | ENGINEERING_ESTIMATE |
| Angle noise | One-sigma direction, rad | `StructuralSensorOptions.angle_sigma_rad=0.02` | ENGINEERING_ESTIMATE |
| Declared old credit half-angle | Belief credit only, deg | `CoverageConfig.footprint_half_angle_deg=50` | ENGINEERING_ESTIMATE; **not measured support** |
| Declared old axial credit | Belief credit only, m | `CoverageConfig.footprint_axial_m=1` | ENGINEERING_ESTIMATE; **not measured support** |
| Footprint width / height | Actual surface patch measured, m | No calibrated value found; required explicit `StructuralSensorModel` parameters | UNKNOWN physically; ENGINEERING_ESTIMATE in synthetic tests |
| Lateral / axial resolution | Smallest resolvable position separation, m | No calibrated value found | UNKNOWN |
| Minimum corrosion-patch size | Smallest physical patch producing a local response, m | No calibrated value found; required explicit synthetic parameter | UNKNOWN physically; ENGINEERING_ESTIMATE in tests |
| Minimum crack-patch size | Same for crack, m | No calibrated value found; required explicit synthetic parameter | UNKNOWN physically; ENGINEERING_ESTIMATE in tests |
| Support shape | Capsule surface axial-angular rectangle | Development design in `CapsuleSurfaceSupport` | ENGINEERING_ESTIMATE |
| Aggregation kernel | `AREA_MEAN` or idealized `LOCAL_MAX` | Development design in `StructuralSensorModel` | ENGINEERING_ESTIMATE; intended physical sensor response UNKNOWN |
| Structural-value noise | One-sigma depth/length error, m | Required explicit synthetic parameter; old T0 has separate noise settings | UNKNOWN physically; ENGINEERING_ESTIMATE in tests |
| Pose uncertainty | Position / orientation covariance | `Pose.covariance_6x6` may supply it, but may be absent | MEASURED only when a specific pose estimate supplies covariance; otherwise UNKNOWN |
| Support uncertainty | Axial / angular edge bounds | `CapsuleSurfaceSupport` optional fields | UNKNOWN until supplied by measurement geometry |
| Sensor datasheet | Calibrated modality, response, occlusion, and detectability | No authoritative payload document found in repository search | UNKNOWN |

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
cell boundary cannot supply healthy cell coverage. Coverage is computed from
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
