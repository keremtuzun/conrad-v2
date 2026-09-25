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
| F Identifiability | Controlled mission sensor/Evidence/Model2T sweep separates same-mean worlds; one closed-loop healthy and two defect development missions; current fixture development matrix 28/28 | DEVELOPMENT VERIFIED; prospective validation pending |
| G Synthetic detectability | Detectability v2: independent explicit crack length/depth thresholds plus patch size; unit boundary and mission-path negative controls passed | DEVELOPMENT VERIFIED; prospective validation pending |
| H Kernel/Unity parity | Pinned player; 16/16 fresh development cases at association-v3 across clear/occluded, exact/bounded, deterministic/noisy cases | DEVELOPMENT VERIFIED; prospective validation pending |
| I Truth boundary | Static boundary tests 22/22; association-v3 bundle scan of 31,785 runtime records against 18 world IDs found zero leaks | DEVELOPMENT VERIFIED; prospective validation pending |
| J Replay/versioning | Association-v3/detectability-v2 480 s replay: 1,703 files, 478 objects, 2,174 Observation, 1,694 Evidence, 8,166 belief payloads equal; old-contract rejection | DEVELOPMENT VERIFIED; prospective validation pending |

The first prospectively declared controlled-view validation on seeds 1401
and 1402 is **FAIL (6/28)**. All 22 failures stopped at the unchanged
full-required-coverage assertion. Healthy coverage was 0.9884 and 0.9778,
respectively. Read-only artifact inspection found emitted supports near a
segment boundary with no registry association; the local condition correctly
remained `UNKNOWN`. Those seeds are spent and will not be rerun. Architecture
A-J remains **NOT FROZEN**. A unique-axial-segment association rule passed
the full 28/28 case matrix on fresh development seeds 407/408, with
near-joint/overlap/uncertain-survey abstention checks. The association-v2
closed-loop replay passed exactly before cycle 2 was opened.

Cycle 2 is now **FAIL: controlled-view 17/28, Unity parity 6/16**.
Seed 1501 passed all 14 controlled-view cases. On seed 1502 the three
partial/outside/occluded controls passed; 11 full-sweep cases stopped at
required coverage 0.7924. All emitted spatial supports were associated, but
the support-family geometry physically hid strips of the required surface.
Unity parity's exact kernel/Unity comparisons passed before its failed
fixture assertions: four seed-1501 occluded cases did not hide a fixed test
view, and seed 1502's clear/bounded cases lacked the predeclared support or
positive bounded coverage. These partitions are immutable and the seeds
will not be rerun. The inspectable no-support/no-clutter fixture and
association-v3 then passed the unchanged 14-case matrix on development
seeds 418/419 (**28/28**). The strengthened Unity parity fixture, including
a deliberately blocked ray and positive bounded-survey coverage, passed
development seeds 426/427 (**16/16**). Earlier Unity development seeds 420
and 425 exposed the fixture and bounded-association gaps and remain recorded
as failures.

Cycle 3 is **FAIL: controlled-view 28/28, Unity parity 12/16**. The four
seed-1702 clear-view parity cases stopped at the predeclared requirement
that every clear test view emit support: support counts were `[0, 0, 4, 4]`.
Kernel and Unity agreed on the zero supports, so this is a test-view
feasibility failure, not evidence of a backend mismatch. Seeds 1701/1702
are spent and will not be rerun. Clear-view poses with shorter standoff
are being checked on fresh development seeds before any new declaration.

Cycle 4 is **FAIL: controlled-view 16/28, Unity parity 12/16**. Seed 1802
passed the partial and outside-support controls, but its healthy and ten
other full-sweep cases lacked full required support. The occluded negative
control also lacked its required clear-world detectable observation. Four
bounded-survey Unity cases on seed 1802 earned zero credited coverage.
Read-only inspection of the healthy case found 240 supported observations,
31 unassociated observations, and final target direct support about 0.43;
the emitted wide angular cells covered only narrow axial strips around the
four fixed sweep positions. No required threshold is lowered. Seeds
1801/1802 are spent and will not be rerun.
The cycle-5 fixture has passed fresh development: complete controlled-view
matrix 28/28 on 565/566, 20/20 extra healthy full sweeps on 567–586,
20/20 bounded clear Unity on 587–606, 20/20 bounded occluded Unity on
535–554, full Unity parity 16/16 on 607/608, focused integration 19/19,
and opt-in live Unity tests 9/9. Cycle 5 remains unopened at this entry;
architecture A–J is still **NOT FROZEN**.

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
deselected, and 3 historical xfails. The association-v3 preflight full suite
passed 1,473 tests with 14 explicit skips, 125 opt-in Unity cases deselected,
and 3 historical xfails. A final full suite at the frozen HEAD remains
required.
The historical
gitignored I5 artifact is an explicit skip in the normal clean checkout,
never regenerated from spent final seeds.

Architecture A-J cannot be marked PASS from this development ledger. Before
freeze, a fresh, prospectively declared partition must test the unchanged
case matrix and matched Unity semantics. A healthy closed-loop run and
replay at the current association-v3 contract completed in development with
exact full replay. The declared 480-second
same-mean development pair under the earlier association contract completed: the
uniform world reached qualified `INTACT`, and the local world had 386 `SEVERE`
revisions and zero `INTACT` revisions. Performance evidence includes 80-second
untraced kernel timing with actual MCBR plans; a 40-step matched development
probe measured Unity step cost and startup separately. I4/I7 impact, I5 reopening, and I6
freshness remain ordered after architecture freeze.
