# Spatial Twin2T V1, development design

`SpatialStructuralTruth` is an opt-in truth-plane capsule surface field. It
uses axial bins and circumferential sectors aligned with the existing
`SurfaceGeometry` representation. Each cell has local corrosion depth and
crack length; non-overlapping rectangular patches can override smaller
areas, including patches below the declared sensor resolution. The field can
represent healthy and degraded uniform surfaces, separated defects, edge
defects, and same-global-mean adversarial surfaces. `worst_local()` is
computed from truth before any measurement. The sensor integrates only over
the supplied support. Exact patch and cell boundaries are used in the
synthetic integral so small patches are not accidentally erased by sampling.
`emit_spatial_observation` is the opt-in adapter that turns a supplied measured
footprint into a structured Observation with the sensor version and digest.
It records a capture independence group and measured range/bearing/elevation,
but it has no world-entity or belief-cell ID. The footprint producer remains
to be implemented from pose, field of view, visibility and occlusion.

The new field does **not** yet replace the historical rest-region
`ComponentRuntime` in `MissionWorld.build`. Historical worlds, including
`I5-NOMINAL-READABLE` with `pristine_rest=True`, retain their historical
semantics. That separation is necessary until truth evolution, mission
sensor placement, Unity geometry, and replay are integrated and checked.

The representation currently models local corrosion depth and crack length.
An optional `SpatialEvolutionV1` applies separate nonnegative synthetic rates
to every base cell and patch without collapsing them into a global scalar.
It does not claim a physical degradation law and is not called by the mission
Twin2T scheduler. Local crack depth, coating, mission-integrated dynamics,
occlusion clipping, actual pose-driven support construction, and
component threshold harmonization remain open. A versioned mission world
family and digest-pinned development/validation/final/OOD partitions have
not yet been declared. None of the old partitions is used by this module.

The 2026-09-24 continuation adds a geometry-only capsule support producer and
a synthetic resolution-cell response mode. Neither is wired into the mission
Twin2T / Model2T path. The producer uses finite visibility probes and should
not be treated as a continuous occlusion guarantee. Its output remains
development evidence until mission integration and Unity parity are checked.
