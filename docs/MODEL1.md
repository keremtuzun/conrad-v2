# Model 1: EGDC decision layer

`conrad/decision` implements Evidence-Grounded Decision and Control (EGDC). It reads a `BeliefSnapshot`, never
truth, and produces a `DecisionRecord` whose world claims trace back to beliefs and evidence.

## Pipeline

`EGDC.decide(ctx: DecisionContext) -> DecisionOutcome` (`egdc.py`):

1. `ClaimGraphBuilder.build` (`claims.py`) builds the Decision Claim Graph.
2. `CandidateActionGenerator.generate` (`actions.py`) proposes actions from the uncertainty causes
   (A -> IMPROVE_MEASUREMENT, O -> EXTEND_COVERAGE, C -> RESOLVE_CONTRADICTION or DISCRIMINATE_HYPOTHESES,
   E -> CONFIRM_CONDITION with an alternate modality, or escalation).
3. `ConsequenceEstimator.estimate` (`consequence.py`) gives a `ConsequenceVector` per candidate.
4. The policy ranks the candidates. Default: `StructuredReasoningPolicy`. Alternatives: `NaiveActOnClaimsPolicy`
   (baseline), `LearnedEGDCPolicy` (ranks only).
5. In rank order, `ConstraintEngine.check` (deterministic, never optimised) and `ExecutionRouter.route` decide. The
   first accepted action wins; each rejection is kept with its reason codes.
6. The result is a `DecisionRecord` plus a `ProvenanceRecord` (`SourceType.DECISION`), registered with
   `OutcomeMonitor`.

`BeliefQueryEngine` (`query_engine.py`) queries each domain per `MissionRequirement` and merges the replies into
one snapshot. `cross_domain_disagreement` adds a CONTRADICTS edge when 2T and 2S disagree.

## Vocabulary and routing

- `ActionType` (12): CONTINUE_MISSION, QUERY_BELIEF, REQUEST_INFORMATION, REPLAN, CHANGE_SENSOR_MODE,
  REVISIT_REGION, WAIT, TRANSMIT_INFORMATION, STORE_AND_FORWARD, ESCALATE_TO_OPERATOR, RETURN_TO_SAFE_STATE,
  ABORT_MISSION.
- Router (`router.py`): REQUEST_INFORMATION -> MCBR (`InformationNeed`); REVISIT_REGION -> NAVIGATION
  (`NavigationGoal`); TRANSMIT / STORE_AND_FORWARD -> BAAC; ESCALATE -> OPERATOR; QUERY_BELIEF -> BELIEF_BUS;
  CHANGE_SENSOR_MODE -> SENSOR_MANAGER; RETURN_TO_SAFE_STATE / ABORT -> SAFETY_SUPERVISOR;
  CONTINUE / REPLAN / WAIT -> MISSION_EXECUTIVE.
- Constraint reason codes (`constraints.py`) include UNSUPPORTED_CLAIM_AS_FACT, STALE_BELIEF, STALE_ROBOT_STATE,
  POSE_UNCERTAINTY_EXCEEDED, LEAK_DETECTED, BATTERY_BELOW_RESERVE, RISK_LIMIT_EXCEEDED, MOTION_NOT_PERMITTED,
  OUTSIDE_MISSION_BOUNDARY, LINK_DOWN, DOMAIN_UNAVAILABLE.

## Grounding invariant and UIR

- A BELIEF_CLAIM cannot be GROUNDED without `source_belief_ids` (schema validator in `conrad/schemas/decision.py`).
- An OBSERVED belief with no evidence support is UNSUPPORTED.
- UIR (`uir.py`) = relied-upon UNSUPPORTED world claims / relied-upon world claims, where "relied upon" means
  listed in the chosen action's supporting claims.

## Configuration

`DecisionConfig` (`config.py`): `model_version="egdc-structured-0.2"`, `enforce_grounding=True`,
`max_claim_nodes=128`, `consequence_matters_above=0.3`, `max_information_attempts=3`, uncertainty thresholds 0.4.
The metadata marks thresholds, action costs and resolvability priors as ENGINEERING_ESTIMATE simulation defaults.

## Tests and experiment

```
uv run pytest tests/unit/decision tests/property/decision -q
uv run conrad eval run --experiment M1-UIR-E001
```

M1-UIR-E001 (synthetic fault fixtures, `conrad/evaluation/oracle/decision_oracle.py`): UIR 0.000 vs naive 0.188;
safe rate under faults 0.900 vs 0.457; under MISCALIBRATED_SILENT the EGDC safe rate is 0.0. Silent
miscalibration cannot be detected from belief data alone.

## Limits and OPEN items

- Reasoning engine, ranking/value model and learning objective are OPEN (ch32).
- The learned scorer is untrained beyond a smoke test ([MODEL_CARDS/egdc_learned_scorer.md](MODEL_CARDS/egdc_learned_scorer.md)).
- `MISSION_NOT_ACTIVE` is defined in `constraints.py` but not used by `check`.
