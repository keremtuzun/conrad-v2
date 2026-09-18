# ADR-0004 Analytic operators are the default runtime until learned candidates are evaluated

- Date: 2026-09-18
- Packages: conrad.core (BUO, association, TBD, RBP), conrad.decision, conrad.active, conrad.communication. Source: ch30 authority ledger, ch33, ch35 kill protocol, ch36 promotion policy.
- Status: ACCEPTED as an interim runtime policy; not an architecture selection.

## Problem
ch33 freezes layer-level learned candidates, but ch36 forbids promoting any learned module without matched baseline/ablation/OOD/calibration evidence across the declared seeds. No research-scale compute or rights-cleared real data is available in this environment.
## Decision
Every mechanism ships two interchangeable implementations behind one interface: the ch33 learned EXPERIMENTAL_CANDIDATE (trained and smoke-evaluated on CPU) and a deterministic analytic operator. The integrated runtime uses the analytic operators; a learned module becomes the runtime default only through a later ADR citing a registered experiment. This is the experimental kill rule applied prospectively, not a claim that the analytic operator is the Conrad architecture.
## Consequence
Integrated-mission results are attributable to the architecture and contracts, not to learned-module quality. claim_status of learned modules stays IMPLEMENTED.
## Approval
Kerem: PENDING REVIEW
