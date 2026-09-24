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

The matched views and short mission do not establish parity across the full
occlusion, geometry, defect, and noise regimes. Broader prospective parity and mission
identifiability evaluations remain required before a Spatial V1 freeze.

The current spatial support coordinates assume an exact surveyed design axis.
Mission launch rejects nonzero `survey_sigma_m` until a conservative
truth-to-design support transform is implemented and validated. This is a
software boundary, not a physical calibration dependency.
