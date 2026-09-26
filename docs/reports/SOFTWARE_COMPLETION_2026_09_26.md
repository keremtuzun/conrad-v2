# Conrad V2 software completion record

Date: 2026-09-26

Branch: `kerem/software-completion`

Evidence baseline: `071acc72a443b9127462533ff1d71a9dd92eb653`

## Outcome

All software architecture, mission integration, Unity, Model2T, MCBR,
communication, replay, truth-boundary, evaluation-harness, and repository
quality work authorized by the completion program has been executed to its
scientifically valid stopping point. This does not mean every research gate is
green. Negative gate results remain negative and no spent held-out evidence was
reused.

Spatial Structural Architecture V1.1 is software validated and frozen for
synthetic use. Physical calibration is still required before any physical
detection, vehicle-performance, communication-performance, endurance, safety,
or deployment claim.

## Spatial architecture A-J

| Criterion | Status | Evidence |
|---|---|---|
| A Truth expressiveness | PASS | versioned heterogeneous local truth and independent local evolution |
| B Observation locality | PASS | pose-derived resolution support and conservative visibility |
| C Support provenance | PASS | versioned support/detectability digests through Observation and Evidence |
| D Local belief | PASS | supported-cell-only SpatialModel2T updates and partial-support controls |
| E Conservative status | PASS | qualified intact requires complete supported domain; false intact remains zero |
| F Identifiability | PASS | full-mission healthy/defect and same-mean adversarial cases |
| G Synthetic detectability | PASS | software-validated detectability v2; physical thresholds remain estimates |
| H Kernel/Unity parity | PASS | declared native-macOS supplement passed 16/16 on the pinned player |
| I Truth boundary | PASS | static checks plus runtime bundle scans found zero deployment-side leaks |
| J Replay/versioning | PASS | deterministic mission replay and fail-closed identity/config mismatch checks |

The frozen identifiers, digests, tolerances, and detailed evidence are in
`docs/architecture/SPATIAL_V1_1_FREEZE.md` and
`docs/audits/SPATIAL_V1_ACCEPTANCE.md`.

## Gate and impact decisions

| Area | Status | Evidence | Remaining blocker | Requires hardware |
|---|---|---|---|---|
| Spatial architecture | PASS, synthetic | A-J above; V1.1 freeze | physical parameter calibration only | yes |
| Kernel/Unity parity | PASS, synthetic | native macOS 16/16 supplement | physical sensor/vehicle equivalence | yes |
| Replay | PASS | exact mission replay and mismatch refusal | none in software | no |
| Truth boundary | PASS | static and runtime scans, including 42-bundle I5 supplement | none in software | no |
| MCBR Spatial V1 | IMPLEMENTED; I4 negative | belief-side spatial prediction and development cycles | mechanism did not earn a new final cycle | no |
| I4 impact | MATERIAL; no new qualifying candidate | impact audit and energy-repair development screen | research result remains negative | no |
| I5 | FORMAL FAIL for Spatial V1.1, 7/10 | 42-flight Unity result on worlds 8701000-8701001 | failed result; same-cycle rerun forbidden | no |
| I6 freshness | FORMAL PASS 3/3; official blocked upstream | current evidence and freshness audit | I5 dependency | no |
| I7 | surrogate FAIL 3/4; formal NOT_RUN | compact-summary development/validation and preserved surrogate | strict ultra-low-bandwidth criterion not cleared | no |
| Physical calibration | PENDING | `docs/hardware/PENDING_PHYSICAL_CALIBRATION.md` | measured constants and real-world trials | yes |

### I4

The immutable historical I4 record began at 4/5 FAIL. Spatial V1 changed the
observation, belief, decision, and communication streams materially, so the
legacy scorer was not reused as a Spatial result. A separate energy-efficiency
cycle diagnosed a real short-leg execution defect and tested repaired
cost/filter variants on development worlds. No candidate showed material,
safe headroom, so the cycle stopped before validation or final data. No new I4
formal claim was earned. A later separately versioned formal record also
remains negative; the generated gate table reports the latest formal record.

### I5

Historical I5 formal 9/10 FAIL remains immutable. Spatial V1.1 then completed:

- R6 development: 210/210 unique missions and the preregistered rule passed;
- controlled architecture matrix: 28/28 passed;
- v10 validation: passed;
- powered surrogate final: 588/588 rows, PASS;
- Unity formal: exact 42-flight grid, one player at a time, 7/10 FAIL.

The seven matrix criteria passed: continue, request evidence, replan, change
sensing, return, escalate, and hard constraints. The three integrated criteria
are formally FAIL because their required support set included the failed action
criterion and a leakage test node that errored before recording. The underlying
trace metric was 1.0, UIR was 0, and competitive outcomes passed, but they are
not relabelled.

The action failure was a real Unity/kernel `LOW_POWER` semantic mismatch:
battery-reserve reached 0/2 warrants. The exact result and 42 bundles were
frozen. A read-only supplemental scan found zero leaks but did not change the
formal node. The software defect was then corrected with a distinct Unity
`LOW_POWER` event, rebuilt at player SHA-256
`c52be937690a5d3d5be03f4acc3fc06842dd17ce3f53000e34f6daa528ab71e9`,
and verified on development world 8701100: first post-fault battery 0.119952,
warrant reached, correct return, zero violations. The formal cycle was not
rerun.

### I6

Spatial V1.1 did not invalidate the three I6 cross-domain criteria. The formal
record remains 3/3 PASS and the freshness audit documents applicability. Its
official registry state is blocked upstream by I5, not a missing software test.

### I7

Historical I7 surrogate 3/4 FAIL and formal NOT_RUN remain immutable. Spatial
V1 has material communication effects. The versioned compact critical-summary
mechanism was implemented and screened, but fresh held-out validation still
failed the unchanged 3/4 rule at 1% and 0.1% bandwidth. Per the declared stop
rule, no surrogate final or formal Unity partition was opened.

## Repository verification

Integrated checkout:

- Ruff format: PASS, 827 files;
- Ruff lint: PASS;
- mypy: PASS, 659 source files;
- secret scan: PASS, 0 findings;
- pytest: 1519 passed, 15 explicit skips, 139 opt-in tests deselected, 3 strict
  historical xfails, 0 failures, 0 setup errors.

Fresh remote clone at exact pushed SHA:

- Ruff format, lint, mypy, and secret scan: PASS;
- pytest: the same 1519 passed, 15 skips, 139 deselected, 3 strict xfails;
- no cached experiment bundles or local artifacts were present.

The skips are explicit for unavailable large public datasets, an unavailable
captured I1 bundle, the historical gitignored I5 artifact that must not be
regenerated from spent seeds, and an environment-only uv discovery check.

## Remaining physical-only work

- real structural sensor footprint and field-of-view calibration;
- real crack/corrosion detectability and resolution;
- real noise versus range, turbidity, and orientation;
- real sensor latency and power draw;
- real thruster power, drag, buoyancy, and current response;
- real underwater communication throughput, BER, latency, loss, and outages;
- measured pose/support registration bounds;
- target-computer HIL and pool/open-water validation; and
- owner approval of deployment safety and interface contracts.

No Unity, schema, mission integration, Model2T, MCBR, replay, truth-boundary,
or ordinary repository-test task is being deferred as a hardware problem.

## Major milestones

| Commit | Milestone |
|---|---|
| `a51a6b9` | repaired clean-checkout regression baseline |
| `c1d678c` | froze Spatial V1 after architecture validation |
| `799ef6b` | preserved failed I7 compact-summary validation |
| `942b579` | repaired I5 localization and visibility contracts |
| `9549c23` | recorded Spatial V1.1 R6 development PASS |
| `137ee0e` | recorded controlled V1.1 architecture revalidation |
| `e1db32e` | reconciled I6 evidence freshness |
| `3c9adec` | recorded native macOS Unity parity PASS |
| `a144f55` | froze Spatial V1.1 software architecture |
| `d938c27` | recorded v10 validation PASS |
| `2c443ef` | recorded powered surrogate final PASS |
| `f45982f` | froze the 7/10 Unity formal FAIL and leakage supplement |
| `b4a31f0` | aligned Unity `LOW_POWER` semantics with the kernel |
| `e74f560` | recorded post-formal development verification |
| `071acc7` | restored repository-wide quality checks |

The authoritative generated status table is
`docs/reports/CONRAD_V2_GATE_TABLE.md`; requirements status is in
`docs/specification/CONRAD_V2_REQUIREMENTS_LEDGER.md`.
