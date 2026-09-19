# ADR-0007 Model2E production path is uncoupled; CEFD is KILLED_PENDING_REDESIGN

- Date: 2026-09-19
- Packages: conrad.domains.ecological
- Source: ch12 CEFD, ch25 2E Gate ("CEFD must beat uncoupled baselines without producing unsupported ecological claims"), ch36 kill conditions.
- Status: ACCEPTED

## Evidence
2E-E003, development seeds 2026201-3:
- Coupling benefit on cover is about 0 (between -0.003 and +0.003).
- In the confounded world, CEFD made confident thermal-stress claims on 100% of healthy entities.
- The research part of the 2E gate therefore fails.

## Decision
- `CefdSwitches()` now defaults to `field_to_entity=False, entity_to_field=False`, and `model_version` is `model2e-uncoupled-analytic-0.2.0`.
- The coupled variant is reachable only through `baseline_config("cefd")` in experiments.
- `tests/contract/test_runtime_defaults.py` enforces this.
- The gate registry splits the 2E gate into `2E` (functional: entity, field and persistent inference) and `2E-CEFD` (research). I6 depends on the functional part only.
- Note: CEFD's turbidity-aware survey noise (field-to-entity observability) was its one measured benefit (2E-E002). It is a candidate to split out as a sensing-quality context, separate from ecological causal coupling, and to re-evaluate under the partition protocol.

## Revisit when
CEFD beats uncoupled on the FINAL partition with a paired CI above 0 and zero confident unsupported stress claims on healthy entities.

## Approval
Kerem: PENDING REVIEW
