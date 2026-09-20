# ADR-0009 Model2T production path does not propagate condition; TCDP is an EXPERIMENTAL arm

- Date: 2026-09-19
- Packages: conrad.domains.technical, conrad.orchestration (children.py)
- Source: ch10 Model 2T (TCDP, anti-contamination), ch25 2T Gate (research part: "TCDP improves hidden-state
  reconstruction vs generic/no propagation without excessive contamination"), ch36 kill conditions.
- Status: ACCEPTED

## Evidence
- 2T-E003-R2 on the FINAL partition (60 seeds, `docs/audits/MODEL2T_REPAIR.md`): corrosion RB_TCDP
  -0.104 [-0.110, -0.097] mm. TCDP is worse than no propagation, and its CI lies entirely below 0. Crack
  RB_TCDP -0.005 [-0.022, 0.013]. Gate 2T-TCDP = FAIL.
- Gate I3 on Unity (`docs/audits/UNITY_INTEGRATION_I1_I3.md`): a never-observed component carried an
  INFERRED condition with U_O 1.0 instead of UNKNOWN. Cause: the mission runtime built Model2T with
  `model2t_mode = "TCDP"` (default in `conrad/orchestration/mission_config.py`), so after a neighbour was read,
  `StructuralBeliefEngine.propagate` wrote an INFERRED corrosion estimate into the unseen component, and the
  derived condition claim took that status. No other Model2T operator sets INFERRED: the population prior and
  temporal prediction leave a never-read quantity UNKNOWN.

## Decision
- `TCDPConfig.mode` defaults to `NONE`. `TCDPConfig.experimental_enabled` (default False) is the explicit
  opt-in for the mission runtime.
- The mission runtime resolves its mode through `production_propagation_mode(cfg.model2t_mode, m2t_cfg)`
  (`conrad/orchestration/children.py`). It returns `NONE` unless `model2t.tcdp.experimental_enabled` is true,
  whatever `model2t_mode` says. The `model2t_mode` field in `mission_config.py` still defaults to `"TCDP"`.
  That file belongs to another workstream and was not edited. The resolver makes that default inert.
- TCDP and GENERIC stay reachable as EXPERIMENTAL arms: pass `mode=` explicitly to `Model2T` or
  `StructuralBeliefEngine` (2T-E003 experiments and unit tests do this), or set the opt-in in a mission config.
- `tests/contract/test_runtime_defaults.py` enforces this. It checks the config default, the resolver for
  every requested mode, that `children.py` builds the mode only through the resolver, and that a default
  Model2T leaves a never-observed neighbour of a corroded component UNKNOWN (no INFERRED, U_O >= 0.9).

## Consequences
- A never-observed component reports UNKNOWN with U_O 1.0 on the production path (surrogate check in the
  contract test and `tests/integration/test_i3_structural.py`; the Unity I3 test was not re-run here).
- There is no relational benefit in production. That benefit was negative on the FINAL partition anyway.
- The TCDP-only claims in the golden suite still run: `test_golden_suite.py::gs04` passes `mode=TCDP`
  explicitly.

## Revisit when
TCDP beats no propagation on a fresh held-out split with a paired CI above 0 for at least one quantity, below
0 for none, and contamination below GENERIC. The E003 contamination rule should first treat identical zero
contamination as "not higher", declared before that run.

## Approval
Kerem: PENDING REVIEW

## Addendum 2026-09-20: one redesign attempt, still no benefit, decision unchanged

The message model was redesigned once and measured on DEVELOPMENT seeds only (10 seeds, 5100000-5100009).
`MessageModel.MEASURED_EXPOSURE` replaced the linear Gaussian conditional with a shared-exposure mixture
whose edge correlation is measured on the pairs whose two ends both carry direct evidence. It lost on both
halves of the criterion: corrosion RB -0.170 [-0.214, -0.127] mm against -0.113 [-0.126, -0.101] for the
iteration-3 message, and relational contamination 0.429 against 0.307 (GENERIC 0.664). The reason is
measurable: an episode has 0 to 4 edges with both ends directly observed, so the correlation is not
identifiable and the shrunk value ends up lowest in the one episode kind where the correlation is real.

No final split was read or spent. The production default stays `TCDPConfig.mode = NONE`, the recorded
2T-TCDP FORMAL evidence stays FAIL, and `tests/contract/test_runtime_defaults.py` is unchanged.
`message_model` defaults to `GAUSSIAN_CONDITIONAL`, so the experimental TCDP arm and the GENERIC baseline
behave exactly as in iteration 3. Evidence: `docs/audits/MODEL2T_REPAIR.md`, iteration 4.

The "revisit when" condition above is unchanged, with one addition: propagation from neighbour levels alone
is not enough in these worlds. A retry needs exposure structure the belief plane can actually read (shared
coating system, splash zone or flow regime in the asset registry or mission context), or worlds where enough
doubly observed pairs exist for the edge correlation to be identifiable.
