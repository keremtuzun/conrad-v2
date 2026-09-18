# MCBR: active perception

`conrad/active` implements the MCBR planner. It turns an `InformationNeed` from EGDC into an `ObservationPlan`. It
never commands motion; navigation executes the plan.

## Pipeline

`MCBRPlanner.plan(PlanningRequest) -> PlanResult(plan, provenance, table)` (`planner.py`):

1. `build_knowledge_gaps` (`gap.py`) builds a `KnowledgeGap` per targeted belief (uncertainty, coverage, question,
   causes, hypotheses, prior views). No gap gives NO_FEASIBLE_OBSERVATION; a satisfied need gives NEED_SATISFIED.
2. `ViewpointGenerator.generate` (`candidates.py`) samples sensor x configuration x standoff (0.25/0.5/0.8) x
   elevation (0, 0.5) x azimuth (`n_azimuth`, default 12), capped at `max_candidates=128` by deterministic
   thinning.
3. `FeasibilityFilter` runs before ranking and is shared by every baseline: POSE_NOT_FREE,
   UNREACHABLE_IN_BELIEF_MAP, VISIBILITY_BELOW_MIN, RISK_LIMIT_EXCEEDED, ENERGY_BUDGET_EXCEEDED, DEADLINE_EXCEEDED,
   OUTSIDE_MISSION_BOUNDARY, MODALITY_NOT_ALTERNATE, SENSOR_UNAVAILABLE.
4. `InformationGainEstimator` (`eig.py`) scores one gain term per uncertainty cause plus discrimination and
   redundancy. `mission_value = priority x sum(w_c x gain_c)`; off-target channels get weight 0.25.
5. Score = `mission_value - cost`. With the value gate on, NOT_WORTH_COST is returned when the best candidate's
   value is below its cost or its gain is below `min_expected_gain` (0.02).

World knowledge enters only through injected callables on `PlanningRequest`: `is_free`, `predicted_visibility`,
`navigation_cost`. The candidate table and rejected candidates are returned with the plan.

`ObservationPlan` fields: plan_id, need_id, trace_id, status (PLAN / NEED_SATISFIED / NO_FEASIBLE_OBSERVATION /
NOT_WORTH_COST), target_beliefs, primary_action, alternatives, rejected, expected information and mission gain,
targeted_uncertainty, expected_cost, confidence, provenance.

## Baselines

`make_planners` (`baselines.py`): A-B0 random, A-B1 fixed inspection, A-B2 coverage, A-B3 frontier, A-B4 geometric
NBV, A-B5 entropy NBV, A-B6 standard EIG, A-B7 uncertainty NBV, A-B9 MCBR without mission conditioning, A-B10 full
MCBR. A-B8 (RL active perception) is not implemented.

## Configuration

`MCBRConfig` (`config.py`), `model_version="mcbr-analytic-0.2"`. Per-cause gain terms and sensor quality numbers
are ENGINEERING_ESTIMATE.

## Tests and experiment

```
uv run pytest tests/unit/active -q
uv run conrad eval run --experiment ACTIVE-MCBR-E001
```

ACTIVE-MCBR-E001 (`conrad/evaluation/oracle/occlusion_world.py`, synthetic): mission error reduction full MCBR
0.359, ranked 8th of 11, behind geometric NBV 0.487 and standard EIG 0.481. MCBR had the fewest redundant
observations (0.36) and the least travel. Verdict: failed hypothesis, KILL_CANDIDATE for value ranking; the
stop-rule cost weights are over-strict in the OOD scenario.

## Limits and OPEN items

- Information-value estimator, hypothesis representation and learned vs analytic ranker are OPEN.
- The learned ranker has only a smoke test ([MODEL_CARDS/mcbr_learned_ranker.md](MODEL_CARDS/mcbr_learned_ranker.md)).
