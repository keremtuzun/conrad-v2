# I7 compact critical summary, development candidate

Historical I7 SURROGATE FAIL 3/4 and FORMAL NOT_RUN remain immutable. The
remaining strict criterion-3 failures include zero retention ties at 1%
and 0.1% bandwidth: the original 184-bit F0 alert can cross the link but
becomes stale, while a complete F1 semantic delta is several thousand
bits. A scheduling retune alone cannot make an F1 fit in 288 bits of
0.1%-link nominal 240-second capacity.

`baac-critical-summary-v1` is an opt-in, versioned development candidate.
For an **observed, critical technical finding** it replaces the 23-byte
bare alert frame with a 23-byte semantic frame containing the belief UUID,
revision, domain, condition (`DEGRADED`, `SEVERE`, or `FAILED`), and claim
status (`OBSERVED`). It sends no truth ID, raw evidence, or unsupported
claim. Missing or ineligible conditions fall back to the historical alert.
The receiver stores this as a limited summary, never as a complete F1
belief view. The retention oracle keeps the historical alert rule exactly:
an old bare alert earns zero. A stale condition summary earns only half
of the existing F0 information fraction; a current summary earns F0. This
is the same stale-state discount already applied to a delivered F1 view,
now justified by the summary's actual condition content. No acceptance
threshold or baseline scheduling policy changes. Every shadow arm sees
the identical offer stream and wire representation; raw/FIFO/fixed still
use their predeclared full-fidelity scheduling policies.

The default `BAACConfig` remains `baac-greedy-0.3` with the mode off.
Enabling the summary requires `model_version=baac-critical-summary-v1`,
which is pinned in the mission runtime config and replay bundle.
Development unit tests verify the exact 184-bit wire size, receiver state,
limited stale credit, legacy alert scoring, and version refusal. The
communication unit/property/acceptance regression passed 87 tests with
the historical criterion-3 failure still an expected xfail.

On development seed 5100001, `I7-BANDWIDTH` with 0.1% and 1% links,
BAAC delivered one compact summary at 0.1% and no full delta. Its
mission information retained was about 0.02 at both levels, versus
0 for raw, FIFO, and fixed priority; critical retention was 0.15 and
duplicate receiver contributions were zero. This is a first development
diagnostic, **not** a surrogate result or a claim that criterion 3 now
passes all seeds and regimes. The remaining development seeds and the
100%/50%/10% plus outage checks are required before selection.

The five additional development seeds 5100000, 5100002, 5100003,
5100007, and 5100008 then completed both 0.1% and 1% runs with the same
candidate. Across all six seeds and both levels, BAAC retained
0.02195–0.04060 mission information; raw, FIFO, and fixed priority each
retained exactly 0 on every run. Thus the unchanged strict required
comparison passed **12/12 low-bandwidth development cases**, with zero
duplicate receiver contributions. This is still DEVELOPMENT evidence.

The remaining 100%/50%/10% sweep then passed 18/18 strict per-seed comparisons
against raw, FIFO, and fixed priority. Mean BAAC retention was 0.59851, 0.58496,
and 0.26119, respectively; the strongest required-baseline means were 0.21478,
0.21478, and 0.21478. Duplicate receiver contributions were zero.

The finding-following outage matrix passed the required comparison 12/12 at
100% and 10%. Mean BAAC retention was 0.59851 and 0.17212 versus required
baseline maxima 0.21478 and 0.11168. Every critical finding occurred while the
link was down, the critical delta arrived after reconnection, and duplicate
receiver contributions were zero. A same-code control with the historical bare
alert matched candidate bits sent, alert and delta latency, full-delta
retention, and exact-sync state in all 12 cells. The semantic frame therefore
did not regress scheduling or receiver behavior.

Across low bandwidth, mid/high bandwidth, and outage, the candidate passed all
42 declared DEVELOPMENT cells. This selects it for an independent validation
cycle; it is not surrogate gate evidence. The fresh validation/final partition,
unchanged decision rule, and unopened-final constraint are frozen in
`I7_COMPACT_SUMMARY_V4_PROTOCOL.md`. No validation, surrogate-final, or
Unity-formal seed had been opened when that protocol was declared.

The subsequently opened validation partition failed the unchanged strict
retention criterion. The complete negative result and stop decision are in
`I7_COMPACT_SUMMARY_V4_VALIDATION.md`; no surrogate-final or formal seed was
opened.
