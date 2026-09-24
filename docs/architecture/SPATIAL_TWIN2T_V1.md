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

The new field does **not** yet replace the historical rest-region
`ComponentRuntime` in `MissionWorld.build`. Historical worlds, including
`I5-NOMINAL-READABLE` with `pristine_rest=True`, retain their historical
semantics. That separation is necessary until truth evolution, mission
sensor placement, Unity geometry, and replay are integrated and checked.

The representation currently models local corrosion depth and crack length
for static development counterexamples. Local crack depth, coating, dynamic
mechanisms, occlusion clipping, actual pose-driven support construction, and
component threshold harmonization remain open. A versioned mission world
family and digest-pinned development/validation/final/OOD partitions have
not yet been declared. None of the old partitions is used by this module.
