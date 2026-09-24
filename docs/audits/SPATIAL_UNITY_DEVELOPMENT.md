# Spatial Unity development parity

This is development evidence for the opt-in spatial mission path. It does not
replace historical I4, I5, or I7 gate evidence.

- Unity Editor: 6000.5.9f1 (project version).
- Windows player build: succeeded with zero errors and zero warnings.
- Player SHA-256: `36C5C9F13481406382A8E9EF8FC0EA7CDF055C43BB12FC8FD545B07C199CD277`.
- Live test: two matched sensor viewpoints produced the same spatial support
  and measurements through kernel and Unity mission sensing.
- A fixed development occluder hid one of the two views completely. Kernel
  and Unity still agreed on support and measurements for both views (clear
  counts `4, 4`; occluded counts `4, 0`).
- For each matched view, both paths also agreed on registry association,
  credited local coverage, every local cell condition, and the component
  condition after ECMER Evidence and spatial Model2T ingestion.
- Live 4-second Unity mission: replay equality passed against that player.
- The Unity truth endpoint supplies collider line of sight; only the structural
  sensor receives the answer. The mission requires both the kernel visibility
  certificate and Unity line of sight before crediting support.
- The spatial runtime smoke test scans events, mission artifacts, and persisted
  runtime rows for world identifiers and truth-only keys.
- A follow-up matched run included four views per case, adding two pitched,
  off-centre viewpoints at different axial positions and standoffs. All five
  opt-in live tests passed for clear/occluded and deterministic/noisy sensor
  cases against the same player SHA. Support, values, association, local
  coverage, cell status, and component status still matched.

The matched views and short mission do not establish parity across the full
occlusion, geometry, defect, and noise regimes. Broader prospective parity and mission
identifiability evaluations remain required before a Spatial V1 freeze.

Qualified healthy support currently requires an exact surveyed design axis.
For nonzero Gaussian `survey_sigma_m`, the mission runs but labels truth-frame
surface support `CAPSULE_UNREGISTERED` with unknown registration uncertainty;
Spatial Model2T refuses coverage credit. This avoids a false healthy claim
without treating an unbounded Gaussian standard deviation as a hard spatial
bound. A useful finite-error registration model remains software work before
Spatial V1 can be accepted for noisy-axis missions.
