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

## Addendum 2026-09-19: observability context is split out and ON (Model2E iteration 2)

The single `field_to_entity` switch mixed two different things. It is now two switches:

- `observability_context` (ON in production): the turbidity field belief sets the survey measurement
  variance through beam attenuation, and that reaches UA/UO. This is **sensing quality**: how well the camera
  can see through the water. It adds no process noise, makes no stress or damage claim, and has no term that
  shifts cover. It only decides how much weight a survey reading gets.
- `ecological_coupling` (OFF in production): the killed part. It covers the temperature belief, the
  thermal-stress likelihood and the stress gate's cover process-noise inflation. `entity_to_field` (the
  filtration sink) also stays OFF.

Why this is not a reversal. Without the observability context, the production entity belief was
over-confident in turbid water: at 25 NTU, 2E-E002-R2 coverage was 0.849 and z² 2.10. Nothing ecological is
inferred from the field. The unsupported-claim failure in 2E-E003 came entirely from the stress path, and
that path is still OFF. `tests/contract/test_runtime_defaults.py` asserts
`ecological_coupling is False`, `entity_to_field is False` and `observability_context is True`, and that the
switch set is exhaustive. The model version is `model2e-uncoupled-obsctx-analytic-0.4.0`. Evidence:
`docs/audits/MODEL2E_REPAIR.md`, Iteration 2.

Addendum (iteration 3, 2026-09-19). The switches are unchanged. Only the temperature depth-trend prior was
recalibrated (empirical Bayes on DEV). The model version is now `model2e-uncoupled-obsctx-analytic-0.5.0`.
Evidence: `docs/audits/MODEL2E_REPAIR.md`, Iteration 3.

## Addendum 2026-09-20: one redesign attempt. Unsupported claims fixed, benefit still absent, decision unchanged

The ecological coupling was redesigned once and measured on DEVELOPMENT (8 seeds) and VALIDATION (20 seeds)
only. The thermal-stress claim is no longer `P(water above a point threshold)`. It is a posterior: an
exposure prior that carries the onset temperature's own uncertainty (sharp only where the registry declares
`stress_threshold_c`), times a Bayes factor computed from the entity's own cover readings for "this cover is
declining", with the decline rate marginalised out. The coupling now also moves the cover mean, as a
posterior-weighted fractional loss, instead of only inflating process noise.

Result. Confident stress claims on entities whose true condition stayed at or above 0.9 fall from 30 of 80
worlds to **0 of 80** on VALIDATION, including every HIGH_TEMPERATURE_STABLE_ECOLOGY confounder world, where
the old coupling claimed stress on 100 % of entities. UEI stays 0. The benefit half still fails: paired
cover-RMSE benefit over `production` is -0.00035 [-0.00056, -0.00015] on DEV and -0.00033 [-0.00044,
-0.00021] on VALIDATION (significantly worse), and over `uncoupled` +0.00020 [-0.00090, +0.00130] on
VALIDATION (CI contains 0).

No final split was read or spent, and the recorded 2E-CEFD FORMAL evidence is unchanged (FAIL, iteration-3
FINAL-3). The switches are unchanged: `ecological_coupling` False, `entity_to_field` False,
`observability_context` True, enforced by `tests/contract/test_runtime_defaults.py`. The redesigned
mechanism is kept as the experimental `cefd` arm because it is strictly safer than the one it replaces.
Evidence: `docs/audits/MODEL2E_REPAIR.md`, iteration 4.

The "revisit when" condition above is unchanged, with one addition. In the 2E-E003 world set, thermal stress
and ecological change are decorrelated by design, and in the one world where stress is real the entity
filter is already survey-noise limited, so no field-to-entity term can show a cover benefit there. A retry
needs worlds where stress and ecological change genuinely co-occur alongside the confounders, or an entity
model whose predictive error between surveys is large enough for a causal drift to matter.
