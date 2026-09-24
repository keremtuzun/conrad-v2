# Spatial Structural Architecture V1 acceptance ledger

This ledger is for the new `kerem/software-completion` branch. It does not
change the historical I4 FORMAL FAIL 4/5, I5 FORMAL FAIL 9/10, or I7
SURROGATE FAIL 3/4 / FORMAL NOT_RUN records. The architecture is **NOT FROZEN**.
All evidence below is synthetic software evidence with engineering-estimate
sensor parameters. No physical calibration or deployment claim follows.

| Criterion | Development evidence | Acceptance state |
|---|---|---|
| A Truth expressiveness | Versioned mission spatial truth, independent local evolution, healthy, heterogeneous, corrosion, crack, separated, edge, occluded and subresolution cases | DEVELOPMENT VERIFIED; prospective validation pending |
| B Observation locality | Pose-derived resolution supports, visibility certificate, measured local response, full mission Observation path | DEVELOPMENT VERIFIED; prospective validation pending |
| C Support provenance | Versioned support and sensor digest through Observation, ECMER Evidence, association; exact, bounded and unregistered survey modes | DEVELOPMENT VERIFIED; prospective validation pending |
| D Local belief update | Spatial mission Model2T updates only associated supported cells; partial and uncertain support fail closed | DEVELOPMENT VERIFIED; prospective validation pending |
| E Conservative condition | Required-domain complete coverage and independent looks for qualified intact; severe local finding overrides incomplete coverage | DEVELOPMENT VERIFIED; prospective validation pending |
| F Identifiability | Controlled mission sensor/Evidence/Model2T sweep separates same-mean worlds; one closed-loop healthy and two defect development missions | PARTIAL: independent closed-loop case matrix pending |
| G Synthetic detectability | Detectability v2: independent explicit crack length/depth thresholds plus patch size; unit boundary and mission-path negative controls passed | DEVELOPMENT VERIFIED; prospective validation pending |
| H Kernel/Unity parity | Rebuilt player SHA pinned; 9 opt-in live tests across clear/occluded, exact/bounded, deterministic/noisy cases | DEVELOPMENT VERIFIED; prospective parity partition pending |
| I Truth boundary | Static import guards and runtime artifact scan passed on spatial mission development path | DEVELOPMENT VERIFIED; current-HEAD sweep pending |
| J Replay/versioning | Full registration-v2 exact-survey 480 s replay (2,402 Observation, 1,922 Evidence, 8,001 belief payloads equal); bounded 80 s exact payload replay; mismatch mutation test. These predate detectability-v2 | PARTIAL: new-contract full replay pending |

The ordinary clean-checkout suite passed at `68de752` (1,452 passed,
14 skipped, 121 deselected, 3 historical xfailed). A later clean checkout at
`c5bfd5b` produced 1,457 passes and one ordinary regression: the new
spatial-only `survey_endpoint_bound_m: null` appeared in a frozen legacy I4
world snapshot comparison. The historical snapshot and digest were preserved;
the test now explicitly verifies that legacy's bound is null and excludes
that spatial-only field from the legacy comparison. Its focused rerun passed;
a final broad clean-checkout rerun at the completed HEAD remains required. The historical
gitignored I5 artifact is an explicit skip in the normal clean checkout,
never regenerated from spent final seeds.

Architecture A-J cannot be marked PASS from this development ledger. Before
freeze, a declared prospective validation partition must test the case
matrix and matched Unity semantics without changing criteria after seeing
results. The full healthy closed-loop run and replay at the current
registration contract completed in development under the earlier detectability
contract. The declared 480-second same-mean development pair completed: the
uniform world reached qualified `INTACT`, and the local world had 386 `SEVERE`
revisions and zero `INTACT` revisions. Performance evidence includes 80-second
untraced kernel timing with actual MCBR plans; Unity step overhead still needs
a measured development comparison. I4/I7 impact, I5 reopening, and I6
freshness remain ordered after architecture freeze.
