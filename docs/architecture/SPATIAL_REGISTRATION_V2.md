# Spatial capsule survey registration, synthetic contract

`survey_sigma_m` is a Gaussian scale and cannot certify a maximum error.
With nonzero sigma and no hard endpoint bound, spatial Observations use
`CAPSULE_UNREGISTERED` and unknown axial/angular registration uncertainty.
Spatial Model2T associates those Evidence records but grants no healthy
coverage. An exact survey (`survey_sigma_m: 0`) retains exact design-frame
support, subject to the separately declared sensor and pose uncertainties.

For development worlds that declare `survey_endpoint_bound_m`, registry
generation clips each synthetic endpoint error vector to that Euclidean
radius. The bound and sigma appear in the persisted mission configuration;
the bound also appears on each `DesignComponent`. The sensor still measures
truth inside its visible physical footprint, then reports the footprint's
nominal axial/angular coordinates with additional conservative registration
uncertainty. The belief erodes the nominal rectangle before crediting any
coverage. Supports extending beyond the surveyed length are unresolved,
never silently clipped. A certificate that cannot safely bound the geometry
also remains unregistered. This is an engineering estimate and a synthetic
registration mode, not a claim about an actual survey.

For design length `L`, radius `r`, and hard endpoint error `e`, the bound
routine first requires `L > 2e`. Its direction difference bound is
`B = 4e/(L-2e)`. It bounds axial error by
`e + (L+2e+r)B`, then bounds the radial projection, normalized radial
direction, and capsule basis differences to obtain an angular error. It
refuses axes near the vertical basis-branch boundary or errors large enough
to make the angular bound ambiguous. The routine uses only design geometry
and the declared bound. A randomized property test checks endpoint clipping
and both coordinate bounds against physical surface points, including the
angular seam. The replay contract pins this policy as
`exact-bounded-or-unregistered-v2`.

A finite bound is necessary but does not imply full inspectability. Eroded
supports may leave gaps, and a mission only earns qualified `OBSERVED_INTACT`
when the declared required surface is fully certified with sufficient
independent looks. Physical survey registration error and actual support
localization still require measurement before a real-world intact claim.
