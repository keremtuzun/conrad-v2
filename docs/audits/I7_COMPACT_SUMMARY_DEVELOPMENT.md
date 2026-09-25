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

No new I7 validation, surrogate-final, or Unity-formal partition has been
declared or opened. The original criterion-3 development candidate's
completion-feasibility and receiver-relative value behavior remain active.
