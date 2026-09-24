# I5 nominal CONTINUE warrant: development forensics

Date: 2026-09-24. Historical `I5 FORMAL = FAIL` (9/10) is immutable. The old Unity final worlds
`7710300` and `7710301` are spent. All measurements here use the new digest-pinned v7 **development**
partition (`bda2a696da495ce23403cb54c31aa31bfacc48a0fd425be332a57b24046be153`). No v7
validation or final world has been built.

## Contract and paths

`I5-NOMINAL` requires a critical technical component with `KnowledgeStatus.OBSERVED`, value `INTACT`,
and no pending report at the same decision. `m1_action_integrated.decision_state` computes this from the
snapshot and mission notes. It does not use truth. The action-matrix CONTINUE recall is already 100/100.

The synthetic structural sensor takes one Twin2T scalar reading per visible `SurfaceTarget`, places its
measured range and bearing in the observation, and association attaches a measured surface point to the
evidence. Model2T credits the declared normal-cone and axial footprint around that point. With design
geometry, its component condition stays `UNKNOWN` until coverage is at least 0.8, unless the observed
part alone proves the worst condition band. `I5-NOMINAL` has one averaged reading for the non-defect
surface; `I5-NOMINAL-READABLE` splits that surface into 8 x 8 tiles and also forces its truth-side rest
region pristine. These are two differences, so READABLE is a diagnostic control, not a substitute for
ordinary NOMINAL.

The trace script `scripts/trace_i5_nominal_warrant.py` records, per structural batch, measured support,
target association, newly credited cell IDs, cumulative coverage, condition openness and uncertainty.
Each decision records condition/status, pending report, active information needs, plan count and chosen
action. The full per-time-step data are in `artifacts/experiments/I5-NOMINAL-FORENSICS/`. Truth-side
initial target/rest values are labeled evaluation-only and never passed to the runtime.

| Development mission | Target readings | Final unique coverage | Last new cell | OBSERVED INTACT with no pending report |
|---|---:|---:|---:|---:|
| 8300000 kernel NOMINAL | 25 | 50/72 = 0.694 | 67.25 s | no |
| 8300000 Unity NOMINAL | 15 | 47/72 = 0.653 | 59.30 s | no |
| 8300000 kernel READABLE | 556 | 66/72 = 0.917 | 93.25 s | yes, 64 s |
| 8300000 Unity READABLE | 579 | 72/72 = 1.000 | 59.30 s | no |
| 8300001 kernel NOMINAL | 72 | 38/64 = 0.594 | 55.25 s | no |
| 8300001 Unity NOMINAL | 72 | 45/64 = 0.703 | 56.30 s | no |

The 8300000 kernel and Unity runs first differ in trajectory and target acquisition: the first target
reading arrives at 13.25 s in kernel versus 14.30 s in Unity, at different measured surface points.
Their first credited footprints differ (10 versus 14 cells), then both stop below 0.8. Unity did not
discard target observations wholesale: association and Model2T cell updates are present. The first
back-end divergence is physical pose/visibility, not the warrant evaluator. Unity used player SHA256
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`.

On 8300001, the open need is `EXTEND_COVERAGE`. The kernel accepted a view at 36 s and flew it by
64 s. It then accepted another plan at the **exact same commanded position**. The second goal stayed
active until mission end. Unity independently shows the same two positions and the same 36/64/120 s
sequence. The second kernel candidate table gives that repeated position observational gain 0,
redundancy 1, yet the production ratio ranker scores it 0.346, above the feasible distinct views.
The predictor's broad candidate visibility still values it. Repeating a scalar reading at the same pose
cannot extend its spatial support. The final component condition remains `UNKNOWN`, with no pending
report. This is blocker **D** (repeat support) plus **J** (mission ends), with **B** (uncovered route-side
surface) as the resulting state. It does not justify lowering the 0.8 requirement.

READABLE yields hundreds of spatially distinct readings instead of tens of averaged readings, making
coverage close much earlier. Yet on Unity seed 8300000 its condition was `OBSERVED INTACT` at 54 s
while a report was pending, then `DEGRADED` at 56 s; the exact CONTINUE warrant never occurred. The
READABLE control therefore also demonstrates a condition/report timing blocker. Its rest-region truth
is separately forced pristine, and its tile readings share that truth-side region. Copying those
scenario overrides to NOMINAL would be an invalid gate repair.

## Development candidate and stop

The tested candidate refused the exact same prior position and modality for an `EXTEND_COVERAGE`
question. A first implementation on the separate V4 planner path passed its unit test but did not
affect the I5 production planner; this was a wiring mistake, not evidence of improvement. A second
implementation attached the same filter before the frozen production ranker. It kept the ranker
weights and thresholds unchanged and passed the targeted planner test. On 8300001, it rejected the
repeated view with an explicit `COVERAGE_VIEW_REPEATED` reason and flew four distinct views instead
of one completed view plus an identical goal abandoned at mission end. Nevertheless, final coverage
rose only from 38/64 to 39/64, with no warrant. On 8300000, final coverage stayed 50/72, also with
no warrant. The candidate failed the development requirement of materially increasing legitimate
warrant reachability. Its production and V4 code changes and unit tests were removed. The separate
candidate traces are retained as development evidence.

The remaining mismatch is between MCBR's cellwise predicted visibility and the ordinary sensor's
actual **single averaged scalar** for the whole visible non-defect surface. Different predicted views
can still produce nearly identical measured support. A scalar average cannot certify each visible
cell free of a local defect. Crediting all visible cells from that scalar would inflate belief-side
coverage without spatially resolved evidence. The READABLE per-tile payload is a synthetic scenario
override with shared region truth, not independent authority that the ordinary payload can resolve
those tiles. Thus neither expanding Model2T's credited footprint nor copying READABLE's tile behavior
is currently justified. Without a justified measurement contract or a candidate that survives these
development probes, the cycle stops before validation. No v7 validation/final seed is touched.
