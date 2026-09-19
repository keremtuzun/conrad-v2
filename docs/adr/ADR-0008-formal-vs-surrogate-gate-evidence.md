# ADR-0008 Formal gate evidence vs surrogate evidence

- Date: 2026-09-19
- Packages: conrad.evaluation.gates, scripts/record_gate_evidence.py, artifacts/gates/
- Source: ch25 gates and dependency graph, Phase 4-8 text ("Only then connect Unity"; I1 path "Twin2S -> Unity -> ECMER -> Model2S").
- Status: ACCEPTED

## Problem
The previous report marked I1/I2/I3 PASS using the Python L1 simulation kernel while U0 (Unity) had never compiled. The specification's I1 and I3 criteria name Unity explicitly, and the downstream gates depend on them.

## Decision
- A gate's official status is computed by `conrad.evaluation.gates.evaluate_gates()` from evidence files in `artifacts/gates/<gate>/`. It is never typed by hand.
- Evidence is `FORMAL` (the exact source execution path) or `SURROGATE` (e.g. the Python kernel standing in for Unity). Only FORMAL evidence can produce PASS. SURROGATE results are reported alongside the gate and never promote it.
- A gate cannot PASS while any upstream gate is not PASS. It is then BLOCKED_UPSTREAM, unless its own formal evidence FAILs, in which case it is FAIL.
- Every listed criterion must be PASS. A criterion with an OPEN numerical threshold is NOT_EVALUABLE, which never becomes PASS.
- 2T and 2E are split into functional and research gates. A failed research mechanism (TCDP, CEFD) blocks its own research claim, not functional integration.
- External blockers are typed (`BlockerKind`): PHYSICAL_EXTERNAL_INPUT, HUMAN_INTERFACE_AUTHORITY, EXECUTION_ENVIRONMENT, COMPUTE, DATA_LICENSE_OR_ACCESS. Unity compilation is software-owned, not an external blocker.

## Approval
Kerem: PENDING REVIEW
