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
| H Kernel/Unity parity | Pinned player; 9/9 opt-in live development tests at current association/detectability-v2 across clear/occluded, exact/bounded, deterministic/noisy cases | DEVELOPMENT VERIFIED; cycle-2 parity partition pending |
| I Truth boundary | Static import guards and runtime artifact scan passed on spatial mission development path | DEVELOPMENT VERIFIED; current-HEAD sweep pending |
| J Replay/versioning | Full association-v2/detectability-v2 480 s replay (1,822 Observation, 1,155 Evidence, 7,801 belief payloads equal), plus explicit old-contract rejection | DEVELOPMENT VERIFIED; cycle-2 validation pending |

The first prospectively declared controlled-view validation on seeds 1401
and 1402 is **FAIL (6/28)**. All 22 failures stopped at the unchanged
full-required-coverage assertion. Healthy coverage was 0.9884 and 0.9778,
respectively. Read-only artifact inspection found emitted supports near a
segment boundary with no registry association; the local condition correctly
remained `UNKNOWN`. Those seeds are spent and will not be rerun. Architecture
A-J remains **NOT FROZEN**. A unique-axial-segment association rule passed
the full 28/28 case matrix on fresh development seeds 407/408, with
near-joint/overlap/uncertain-survey abstention checks. Current-association
closed-loop replay passed exactly. Unity parity and the separately
predeclared cycle-2 validation are still required.

The ordinary clean-checkout suite passed at `68de752` (1,452 passed,
14 skipped, 121 deselected, 3 historical xfailed). A later clean checkout at
`c5bfd5b` produced 1,457 passes and one ordinary regression: the new
spatial-only `survey_endpoint_bound_m: null` appeared in a frozen legacy I4
world snapshot comparison. The historical snapshot and digest were preserved;
the test now explicitly verifies that legacy's bound is null and excludes
that spatial-only field from the legacy comparison. Its focused rerun passed;
a later clean checkout at `6b082e7` passed 1,454 tests, with 14 documented
skips, 139 slow tests deselected, and 3 historical xfails. The full suite at
`2611851` passed 1,468 tests, with 14 documented skips, 125 opt-in Unity
tests deselected, and 3 historical xfails. That full run preceded the
axial-association repair. The clean checkout at `af53152` passed the full
suite with 1,471 passes, 14 documented skips, 125 opt-in Unity tests
deselected, and 3 historical xfails. A final full suite at the frozen HEAD
remains required.
The historical
gitignored I5 artifact is an explicit skip in the normal clean checkout,
never regenerated from spent final seeds.

Architecture A-J cannot be marked PASS from this development ledger. Before
freeze, the separately declared cycle-2 partition must test the unchanged
case matrix and matched Unity semantics. The healthy closed-loop run and
replay at the current association/detectability/registration contract
completed in development with exact full replay. The declared 480-second
same-mean development pair under the earlier association contract completed: the
uniform world reached qualified `INTACT`, and the local world had 386 `SEVERE`
revisions and zero `INTACT` revisions. Performance evidence includes 80-second
untraced kernel timing with actual MCBR plans; a 40-step matched development
probe measured Unity step cost and startup separately. I4/I7 impact, I5 reopening, and I6
freshness remain ordered after architecture freeze.
