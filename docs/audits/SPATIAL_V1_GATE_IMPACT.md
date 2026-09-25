# Spatial V1 impact on I4 and I7, development audit

This is a post-freeze DEVELOPMENT comparison at source commit
`c1d678c22277498feb9d9aeb731082ac5e31db4f`. The script
`scripts/audit_spatial_gate_impact.py` used one `GOLDEN-SMOKE` seed
(`2026201`), 120 simulated seconds, the same geometry and link settings,
and the production planner. The legacy truth control has a 0.008 m local
corrosion patch and pristine rest; Spatial V1 has one 0.008 m local cell.
The truth representations are different, so this pair diagnoses runtime
impact but is not an I4 or I7 gate comparison. Both resulting manifests
passed bundle verification (443 legacy and 420 spatial files).

| Development observable | Legacy | Spatial V1 |
|---|---:|---:|
| Structural observations (all) | 125 scalar | 36 scalar plus 179 spatial |
| Target structural observations | 40 scalar | 179 spatial; 176 associated Evidence |
| Final target direct coverage | 0.6875 | 0.5794 |
| Final target condition | unresolved | observed `SEVERE` |
| MCBR plans | 7 | 4 |
| Accepted / flown views | 7 / 4 | 4 / 3 |
| Battery energy, J | 28,071.7 | 32,960.9 |
| Estimated travel, m | 44.21 | 52.29 |
| Technical belief revisions | 426 | 286 |
| All belief revision JSON bytes | 6,308,171 | 5,408,583 |
| BAAC offers | 134 | 130 |
| F1-equivalent offered bits, built units | 641,768 | 606,648 |
| Bits sent | 142,224 | 140,197 |
| Mean / peak queued bits | 28,279,869 / 38,771,952 | 31,185,416 / 47,161,936 |
| Coalesced units / removed queued bits | 105 / 155,209,232 | 102 / 177,036,616 |
| Receiver beliefs | 18 | 20 |
| Mean / max sender-to-receiver revision lag among delivered beliefs | 1.94 / 20 | 0.25 / 5 |
| Critical finding offer-log entries | 0 | 5 (one first critical finding) |
| Duplicate receiver contributions | 0 | 0 |

The F1-equivalent total counts every built unit before scheduler selection,
including superseded builds. It is not the amount sent. The queue figures
come from one-second sender traces. The final queue sizes are 38,761,368
bits and 30,334,720 bits, respectively. The spatial run generated a
critical offer absent in the legacy run. This is a material change in the
observation, belief, decision, and communication paths, although a single
world cannot establish a gate improvement.

**I4 impact: MATERIAL.** The existing I4 scorer reads legacy component
numeric claims and the legacy `truth_target_end`. Spatial V1's component
claim reports conservative condition and coverage, with a spatial local
state in its embedding; the legacy truth summary does not describe the
Spatial V1 local map. Consequently its `target_error_before/after` numeric
errors are null for this pair. Feeding the bundles into the old scorer would
report zero information gain for both and would be invalid as a Spatial V1
info/time or info/kJ comparison. No I4 validation or final world has been
opened, and no new formal I4 cycle is justified by this audit. Historical
I4 FORMAL FAIL 4/5 remains unchanged. A spatial-aware, predeclared common
hidden-state reconstruction metric and matched planner-arm development
comparison are needed before an I4 selection or validation decision.

**I7 impact: MATERIAL.** Offer count alone changed little, but local
evidence changed critical finding incidence, queue peak, receiver population,
and the serialized belief stream. The old I7 surrogate remains a valid
legacy record, not a Spatial V1 verdict. The known 1% and 0.1% capacity
problem is still a software-side issue. Fresh I7 development must check
the unchanged 100/50/10% and outage regimes as well as the ultra-low
bandwidth regimes before any surrogate final partition is declared.

The full machine-readable audit, including sender and receiver revision
maps and queue traces, is in the ignored local development artifact
`artifacts/software_completion/spatial_impact_local_120s_link.json` and
the two `SPATIAL-IMPACT-*-120S-LINKDEV` bundles. No formal seed was used.
