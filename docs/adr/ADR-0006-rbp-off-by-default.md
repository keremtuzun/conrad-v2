# ADR-0006 Relational propagation is off by default in the Core runtime

- Date: 2026-09-19
- Packages: conrad.core.pipeline (Model2Core), conrad.core.rbp, conrad.domains.technical (TCDP).
- Source: ch9 RBP, ch35 Priority 5 kill protocol, ch36 KILL_CANDIDATE conditions.
- Status: ACCEPTED (applies the kill rule; it does not select a final architecture).

## Evidence
CORE-RBP-E001 (seeds 2026201-2026203, abstract sandbox), from docs/research/EXPERIMENT_RESULTS_2026-09-18.md:
- AnalyticRBP helps on true edges: RMSE 0.031 against 0.231 with no propagation.
- Misleading edges make 0.58 of the decoy beliefs confidently wrong, against 0 with no propagation.
- The learned RBP does worse still.
- CORE-FULL-E001 overall RMSE: 0.113 without RBP, 0.118 with it.

## Decision
`Model2Core(use_rbp=False)` is now the default. The experiments and tests that study RBP enable it explicitly.
Generic relational propagation matches the ch36 automatic KILL_CANDIDATE condition "worse calibration or
unsupported-confidence behaviour". The implementation stays in the repository as a research record.

Model2T's mechanism-constrained TCDP is a separate candidate and stays enabled in Model2T: its contamination was
0.04, against 0.75 for generic propagation (2T-E003). Its benefit is small (about 2-3%), so it remains
EXPERIMENTAL_CANDIDATE.

## Revisit when
A registered experiment shows edge-trust gating that removes decoy contamination without losing the true-edge benefit.

## Approval
Kerem: PENDING REVIEW
