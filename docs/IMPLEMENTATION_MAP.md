# Implementation map (spec ch30 / ch32 mandatory handoff)

One record per package from the ch32 audit. Each record carries the fourteen ch32 columns. They are shown
transposed (one field per line) because the full table is too wide to read. Values were filled from the code on
disk on 2026-09-18.

Conventions used below:

- **implementation_status** is the `IMPLEMENTATION_METADATA["implementation_status"]` of the package, plus its
  `claim_status`.
- **train_val_test_ood_manifest**: no package uses a dataset manifest or the `conrad.data.splits` lineage split
  yet. Experiments build train and test data from a synthetic generator with disjoint seeds. Every row therefore
  reads `OPEN`, with the seed separation noted.
- **experiment_ids** refer to [research/EXPERIMENT_RESULTS_2026-09-18.md](research/EXPERIMENT_RESULTS_2026-09-18.md).
  Every registry outcome there is INCONCLUSIVE.
- A final architecture is selected for no package (spec ch32 verdict). REQ-OPEN-004 tracks that.

## Summary

| package_id | implementation_status / claim_status | experiment_ids | ADR_ids |
|---|---|---|---|
| ECMER | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | none | ADR-0004 |
| Association | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | CORE-ASSOC-E001 | ADR-0004 |
| PBA/PBG | FROZEN_CONTRACT (schemas, repository) / IMPLEMENTED | CORE-PERSIST-E001 | ADR-0001, ADR-0003 |
| BUO | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | CORE-BUO-E001, CORE-UNC-E001 | ADR-0004 |
| RBP | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | CORE-RBP-E001, CORE-FULL-E001 | ADR-0004 |
| TBD | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | CORE-TBD-E001, training smoke | ADR-0004 |
| PMBL | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | CORE-PERSIST-E001, CORE-FULL-E001 | ADR-0001, ADR-0003 |
| Model2T/TCDP | EXPERIMENTAL_CANDIDATE / EVALUATED | 2T-E001..E004 | none |
| Twin2T/MCDE | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | used by 2T-E001..E004 | none |
| Model2E/CEFD | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | 2E-E001..E003 | none |
| Twin2E/MEIFE | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | used by 2E-E001..E003 | none |
| Model2S/UAHSM | EXPERIMENTAL_CANDIDATE / EVALUATED | 2S-E001..E004 | ADR-0002 |
| Twin2S/OCPWE | EXPERIMENTAL_CANDIDATE / IMPLEMENTED | used by 2S-E001..E004 | ADR-0002 |
| Model1/EGDC | EXPERIMENTAL_CANDIDATE / EVALUATED | M1-UIR-E001 | ADR-0004 |
| MCBR | EXPERIMENTAL_CANDIDATE / EVALUATED | ACTIVE-MCBR-E001 | ADR-0004 |
| BAAC | EXPERIMENTAL_CANDIDATE / EVALUATED | COM-BAAC-E001 | ADR-0004 |
| Navigation/state estimation | EXPERIMENTAL_CANDIDATE / IMPLEMENTED (nav benchmarks package: EVALUATED) | NAV-001..NAV-008 | ADR-0002, ADR-0005 |
| Unity V2/RHI | RHI: FROZEN_CONTRACT; Unity adapter: EXPERIMENTAL_CANDIDATE / IMPLEMENTED | none | ADR-0002, ADR-0005 |

`EVALUATED` means an experiment was executed and stored. It does not mean the mechanism was retained or
validated.

## ECMER

- **package_id:** ECMER (`conrad/core/ecmer`)
- **source_chapters:** ch3, ch4, ch33 ECMER exact implementation, ch33 SSL pretraining
- **input_schema:** `Sequence[conrad.schemas.observation.Observation]` (modalities rgb, sonar, points from
  POINT_CLOUD/DEPTH_RANGE, scalar)
- **output_schema:** `list[EncodedEvidence]`, each an `Evidence` plus a `ProvenanceRecord`
  (`SourceType.DIRECT_OBSERVATION`); `EcmerEncoder.encode` in `service.py`
- **truth_access_policy:** deployment plane; no truth imports (enforced by `tests/leakage`)
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED
- **candidate_config_path:** `model2_core.ecmer.*` in `configs/train/core_default.yaml` (full) and
  `configs/train/core_tiny.yaml` (tests); classes in `conrad/core/config.py`
- **baseline_config_paths:** `conrad/core/ecmer/baselines.py` (`build_fusion_baseline`: `single_<modality>`,
  `concat`, `early`, `late`, `cross_attention`); ablation switches in `EcmerConfig`
- **losses_and_masks:** `ReprObjective` / `ReprLoss` (view contrast, temporal, cross-modal, masked
  reconstruction; symmetric masked InfoNCE) in `ssl.py`; stages E1..E4 in `training.py`; `heteroscedastic_nll`;
  masks with `True` = valid, missing modalities contribute masked tokens only
- **train_val_test_ood_manifest:** OPEN (no real or synthetic ECMER corpus is registered; DATA-CONRAD-SSL-01 real
  partition is empty, EXT-DATA-01)
- **unit_and_leakage_tests:** `tests/unit/core/test_core_ecmer.py`, `tests/property/core/test_core_mask_permutation.py`,
  `tests/leakage/test_static_boundaries.py`
- **experiment_ids:** none
- **open_decisions:** per-modality backbone, fusion operator, embedding dimension, quality-head architecture
  (ch32). The package metadata names `tests/unit/core/test_ecmer.py`, which does not exist; the real file is
  `test_core_ecmer.py`.
- **ADR_ids:** ADR-0004

## Association

- **package_id:** Association (`conrad/core/association.py`, `conrad/core/association_scorer.py`)
- **source_chapters:** ch3, ch33
- **input_schema:** `Evidence` plus `Sequence[BeliefCell]` candidates, `Domain`, entity type
- **output_schema:** `AssociationDecision` (local dataclass: `belief_id` or `None` for NO_MATCH, probability,
  margin, candidate IDs, method)
- **truth_access_policy:** deployment plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; runtime default is
  `nearest_neighbour_decision` (scorer is `None` in `AssociationEngine`)
- **candidate_config_path:** `model2_core.association.*` (`AssociationConfig` in `conrad/core/config.py`)
- **baseline_config_paths:** `nearest_neighbour_decision` with `baseline_gate_distance_m`; `configs/eval/core_assoc_e001.yaml`
- **losses_and_masks:** `association_loss` = masked cross entropy over K+1 (last column NO_MATCH) + focal loss on
  NO_MATCH rows + ranking hinge; `mask[B,K]` for candidates, `label_mask` for supervised rows
- **train_val_test_ood_manifest:** OPEN (sandbox generator; train episodes seeded `seed*100+i`, test `seed*100+50+i`)
- **unit_and_leakage_tests:** `tests/unit/core/test_core_association.py`, `tests/unit/core/test_core_learned_modules.py`,
  `tests/leakage/test_static_boundaries.py`
- **experiment_ids:** CORE-ASSOC-E001 (nearest neighbour wins; learned scorer KILL_CANDIDATE)
- **open_decisions:** scoring function, matching algorithm, thresholds; ch33 pair-feature width (code 1109, spec 853)
- **ADR_ids:** ADR-0004

## PBA/PBG (persistent belief architecture and graph)

- **package_id:** PBA/PBG (`conrad/schemas/belief.py`, `conrad/core/belief_graph.py`, `conrad/core/state.py`,
  `conrad/persistence/repository.py`)
- **source_chapters:** ch2, ch3, ch28, ch30 fidelity register, ch34
- **input_schema:** `BeliefUpdate` (revision, provenance, relationships, lineage) into `Repository.commit_update`;
  `BeliefQuery`
- **output_schema:** `BeliefCell`, `BeliefRevision`, `BeliefMessage`, `BeliefSnapshot`, `CommitResult`;
  `BeliefGraphBatch` tensors (Z, U, T, H, `edge_index[2,M]`)
- **truth_access_policy:** deployment plane; no truth imports
- **implementation_status:** schemas and repository FROZEN_CONTRACT; `belief_graph.py` part of `conrad.core`
  (EXPERIMENTAL_CANDIDATE, IMPLEMENTED)
- **candidate_config_path:** `configs/base/default.yaml` (`runtime.persistence_mode`, `runtime.late_evidence_policy`)
- **baseline_config_paths:** none
- **losses_and_masks:** none (not learned)
- **train_val_test_ood_manifest:** OPEN (not applicable to a non-learned store)
- **unit_and_leakage_tests:** `tests/contract/test_cc_persistence.py`, `tests/contract/test_p0_contracts.py`,
  `tests/contract/core/test_core_contracts.py`, `tests/replay/test_bundle_backup_restore.py`
- **experiment_ids:** CORE-PERSIST-E001
- **open_decisions:** retrieval index, scalable graph execution, bounded late-evidence rewind (ADR-0003)
- **ADR_ids:** ADR-0001, ADR-0003

## BUO

- **package_id:** BUO (`conrad/core/buo.py`, `conrad/core/buo_analytic.py`, `conrad/core/uncertainty.py`)
- **source_chapters:** ch5, ch6, ch33
- **input_schema:** analytic: `AnalyticBeliefState` + `Sequence[Evidence]` (uses `Evidence.measurements`);
  learned: `z, u, temporal, e[B,K,De], quality[B,K,4], mask, group_ids`
- **output_schema:** analytic: `AnalyticUpdateResult`; learned: `BuoOutput` (z, u, channels, trust, gate,
  innovation, used_mask); channels map to `conrad.schemas.uncertainty.Uncertainty`
- **truth_access_policy:** deployment plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; runtime default `AnalyticBUO`
  (constructed in `PMBLBase`)
- **candidate_config_path:** `model2_core.buo.*` (learned), `model2_core.analytic_buo.*` in `conrad/core/config.py`
- **baseline_config_paths:** `configs/eval/core_buo_e001.yaml` (latest_only, simple_average, reliability_weighted,
  analytic_no_contradiction, analytic_no_independence); `configs/eval/core_unc_e001.yaml`
- **losses_and_masks:** experiment loss MSE + innovation L2 (`buo_e001.py`); masks from `reject_duplicates` and
  `collapse_independence_groups`; `used_mask`
- **train_val_test_ood_manifest:** OPEN (abstract sandbox generator, seed-separated)
- **unit_and_leakage_tests:** `tests/unit/core/test_core_buo_analytic.py`, `tests/unit/core/test_core_learned_modules.py`,
  `tests/property/core/test_core_mask_permutation.py`
- **experiment_ids:** CORE-BUO-E001 (analytic retained; learned BUO KILL_CANDIDATE), CORE-UNC-E001
- **open_decisions:** trust/update family, uncertainty estimator, aggregation, thresholds; a noise-to-U_O
  cross-talk defect recorded in CORE-UNC-E001
- **ADR_ids:** ADR-0004

## RBP

- **package_id:** RBP (`conrad/core/rbp.py`, `conrad/core/rbp_analytic.py`)
- **source_chapters:** ch9, ch33
- **input_schema:** analytic: `Mapping[UUID, AnalyticBeliefState]` + `Sequence[Relationship]`; learned: z, u,
  `edge_index`, `edge_type`, `edge_numeric`, `node_mask`, `edge_mask`
- **output_schema:** analytic: `list[RelationalInference]` (INFERRED only, never overwrites OBSERVED); learned:
  `RbpOutput` (z only)
- **truth_access_policy:** deployment plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; `AnalyticRBP` in `Model2Core`
- **candidate_config_path:** `model2_core.rbp.*` (`typed`, `layers`, `adaptive_stop`)
- **baseline_config_paths:** `configs/eval/core_rbp_e001.yaml` (no_propagation, analytic_rbp, learned_untyped_rbp)
- **losses_and_masks:** state loss + `contamination_weight` x `contamination_loss` (masked relu(after - before) on
  misleading targets); segment softmax over `edge_mask`
- **train_val_test_ood_manifest:** OPEN (abstract sandbox generator, seed-separated)
- **unit_and_leakage_tests:** `tests/unit/core/test_core_analytic_tbd_rbp.py`, `tests/unit/core/test_core_learned_modules.py`
- **experiment_ids:** CORE-RBP-E001 (contamination failure; learned RBP KILL_CANDIDATE), CORE-FULL-E001
  (removing RBP slightly better)
- **open_decisions:** message network, aggregation, propagation depth and stopping policy. The results record
  says RBP should be off by default in the runtime; whether the integrated runtime does so is not final (see
  ARCHITECTURE.md, integrated mission).
- **ADR_ids:** ADR-0004

## TBD

- **package_id:** TBD (`conrad/core/tbd.py`, `conrad/core/tbd_analytic.py`)
- **source_chapters:** ch7, ch33
- **input_schema:** z, u, temporal, `delta_t_s` (physical seconds; negative or non-finite rejected), context
- **output_schema:** learned: `PredictedBelief`; analytic: `AnalyticPrediction`
- **truth_access_policy:** deployment plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; runtime default `AnalyticTBD`
- **candidate_config_path:** `configs/train/core_smoke.yaml` (job `core_tbd_smoke`), `configs/train/core_full.yaml`
  (refused without CUDA), `model2_core.tbd.*`
- **baseline_config_paths:** `configs/eval/core_tbd_e001.yaml` (hold_last, linear_extrapolation, analytic_tbd,
  mlp_tbd, gru_tbd_sequence_index)
- **losses_and_masks:** MSE + 0.5 x Gaussian NLL (`entrypoints._core_tbd_job`, `tbd_e001.train`)
- **train_val_test_ood_manifest:** OPEN (sandbox generator; train seed `seed*10+1`, val seed `seed*10+2`)
- **unit_and_leakage_tests:** `tests/unit/core/test_core_analytic_tbd_rbp.py`, `tests/unit/core/test_core_learned_modules.py`,
  `tests/unit/training`
- **experiment_ids:** CORE-TBD-E001 (GRU beats simple baselines), training smoke (smoke model worse than hold-last)
- **open_decisions:** GRU/LSTM/Transformer/SSM/neural-ODE selection, retrieval/compression policy
- **ADR_ids:** ADR-0004

## PMBL

- **package_id:** PMBL (`conrad/core/pmbl.py`, `pmbl_core.py`, `pmbl_store.py`, `lifecycle.py`)
- **source_chapters:** ch8, ch28, ch34
- **input_schema:** `Evidence` + `ProvenanceRecord`s (`archive_evidence`), direct/prediction/relational updates,
  `BeliefQuery`
- **output_schema:** `DirectUpdateOutcome`, committed `BeliefRevision`s via `PersistentBeliefStore`, query results
- **truth_access_policy:** deployment plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE (within `conrad.core`), claim_status IMPLEMENTED; no learned
  variant
- **candidate_config_path:** `model2_core.pmbl.*` (`PmblConfig`)
- **baseline_config_paths:** `configs/eval/core_persist_e001.yaml` (ablation_no_candidate_stage, latest_only_baseline)
- **losses_and_masks:** none (not learned)
- **train_val_test_ood_manifest:** OPEN (sandbox generator, seed-separated)
- **unit_and_leakage_tests:** `tests/unit/core/test_core_pmbl.py`, `tests/contract/test_cc_persistence.py`
- **experiment_ids:** CORE-PERSIST-E001, CORE-FULL-E001
- **open_decisions:** late-evidence rewind (ADR-0003); `PmblConfig.late_evidence_policy` uses the names
  `REVISE_LATE`/`REJECT` while `runtime.late_evidence_policy` uses `EXPLICIT_LATE`/`REJECT`
- **ADR_ids:** ADR-0001, ADR-0003

## Model2T/TCDP

- **package_id:** Model2T/TCDP (`conrad/domains/technical`)
- **source_chapters:** ch10, ch33 Model 2T exact implementation, ch2 Model2Child interface
- **input_schema:** `Sequence[Evidence]` via `Model2Child.ingest`; asset registry from mission context;
  `receive_context(Sequence[BeliefMessage])`
- **output_schema:** `list[BeliefMessage]` with `PropertyClaim`s (corrosion, crack, condition), revisions and
  provenance through the `Repository`
- **truth_access_policy:** deployment plane; no truth imports (`tests/unit/domains/technical/test_m2t_static.py`)
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status EVALUATED; runtime path analytic
  (`model_version "model2t-analytic-0.1.0"`, `PropagationMode` NONE/TCDP/GENERIC)
- **candidate_config_path:** `Model2TConfig` / `LearnedTCDPConfig` in `conrad/domains/technical/config.py`; no YAML
  (experiments build `Model2TConfig()` in code)
- **baseline_config_paths:** `configs/eval/2t_e001.yaml`..`2t_e004.yaml`; `baselines.py` (LatestObservation B1,
  SingleFrame B0, GRUTemporal B2, INDEPENDENT_COMPONENT, GENERIC_RELATIONAL, NO_RATE_PRIOR ablation)
- **losses_and_masks:** learned TCDP only: `tcdp_loss` = masked heteroscedastic Gaussian NLL
  (`target_mask[N,2]`) + contamination penalty on `misleading` nodes; `observed` nodes never updated
- **train_val_test_ood_manifest:** OPEN (Twin2T SYNTHETIC_ONLY scenarios, seeded per episode)
- **unit_and_leakage_tests:** `tests/unit/domains/technical/*`, `tests/property/domains/technical/test_m2t_properties.py`
- **experiment_ids:** 2T-E001 (corrosion wins; crack estimation failed hypothesis), 2T-E002, 2T-E003, 2T-E004
- **open_decisions:** exact neural blocks, factor dimensions, message/update heads, qualified structural data
  manifest; learned TCDP not evaluated; 2T experiments use the twin supervision label as association oracle
- **ADR_ids:** none

## Twin2T/MCDE

- **package_id:** Twin2T/MCDE (`conrad/twins/twin2t`)
- **source_chapters:** ch10, ch11, ch31 H-T2T-01, ch33
- **input_schema:** `Scenario`, `step(dt_s, events)`, `SensingContext`
- **output_schema:** `TwinSample(observation, supervision)`, `TruthState`, `MCDEStepResult`
- **truth_access_policy:** truth plane; supervision on a separate channel, read only by training/evaluation
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED
- **candidate_config_path:** `configs/sim/twin2t_default.yaml`
- **baseline_config_paths:** `configs/sim/twin2t_independent_baseline.yaml`, `configs/sim/twin2t_random_walk_baseline.yaml`;
  `GeneratorKind` enum (9 kinds)
- **losses_and_masks:** none (mechanistic; no learned residual); observation masks recorded per component
- **train_val_test_ood_manifest:** OPEN (generator itself; per-component streams from (seed, entity UUID))
- **unit_and_leakage_tests:** `tests/unit/twins/twin2t/*`, `tests/property/twins/twin2t/test_t2t_bounds_property.py`,
  `tests/simulation/twin2t/test_t2t_episodes.py`
- **experiment_ids:** none of its own; drives 2T-E001..E004
- **open_decisions:** target-material priors, stochastic calibration, learned residual
- **ADR_ids:** none

## Model2E/CEFD

- **package_id:** Model2E/CEFD (`conrad/domains/ecological`, learned candidate in `learned/`)
- **source_chapters:** ch12, ch33 Model 2E exact implementation, ch2
- **input_schema:** `Sequence[Evidence]` (ECMER-E encoder), cross-domain context messages
- **output_schema:** `list[BeliefMessage]` for entity and field beliefs; `ecological_damage` always UNKNOWN;
  `thermal_stress_likelihood` INFERRED only
- **truth_access_policy:** deployment plane; no truth imports (`test_no_truth_plane_imports_in_belief_package`)
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; runtime `AnalyticCEFD`
- **candidate_config_path:** `model2e:` block in `configs/eval/2e_e001.yaml`..`2e_e003.yaml`; `LearnedCEFDConfig`
  in `conrad/domains/ecological/learned/config.py`
- **baseline_config_paths:** `baselines.py` `BASELINES` (cefd, entity_only, field_only, uncoupled, static_field)
- **losses_and_masks:** learned only: `cefd_loss` = masked Gaussian NLL (entity, field) + masked gradient loss +
  BCE of the gate against a `coupled` label; `entity_target_mask`, `field_target_mask`
- **train_val_test_ood_manifest:** OPEN (Twin2E `twin2e_test_small.yaml`, seed-separated)
- **unit_and_leakage_tests:** `tests/unit/domains/ecological/*`, `tests/property/domains/ecological/test_eco_belief_properties.py`
- **experiment_ids:** 2E-E001 (static baseline wins turbidity), 2E-E002, 2E-E003 (D5 gate not met; coupling
  KILL_CANDIDATE)
- **open_decisions:** field representation, entity encoder, coupling operator, temporal backbone, paired data;
  learned CEFD not evaluated
- **ADR_ids:** none

## Twin2E/MEIFE

- **package_id:** Twin2E/MEIFE (`conrad/twins/twin2e`)
- **source_chapters:** ch12, ch13, ch33, H-T2E-01
- **input_schema:** `Scenario`, `step`, `SensingContext`
- **output_schema:** `TwinSample` at E0_ABSTRACT and E1_FEATURE (E2 refused), `TruthState`
- **truth_access_policy:** truth plane
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED
- **candidate_config_path:** `configs/sim/twin2e_default.yaml`
- **baseline_config_paths:** `configs/sim/twin2e_test_small.yaml`; `BASELINE_SWITCHES` and `ABLATIONS` in `baselines.py`
- **losses_and_masks:** none (learned residual disabled; raises if enabled without calibration data)
- **train_val_test_ood_manifest:** OPEN (generator; priors are logged ENGINEERING_ESTIMATE/SYNTHETIC_ONLY samples)
- **unit_and_leakage_tests:** `tests/unit/twins/twin2e/*`, `tests/property/twins/twin2e/test_twin2e_properties.py`,
  `tests/simulation/twin2e/test_twin2e_trajectories.py`
- **experiment_ids:** none of its own; drives 2E-E001..E003
- **open_decisions:** field solver or surrogate, stochastic entity dynamics, target-environment priors
- **ADR_ids:** none

## Model2S/UAHSM

- **package_id:** Model2S/UAHSM (`conrad/domains/spatial`)
- **source_chapters:** ch14, ch33 Model 2S exact implementation, ch2, ch28
- **input_schema:** `Sequence[Observation]` (`ingest_observations` encodes to `Evidence` via
  `GeometricEvidenceEncoder`); `SensorSpec` from mission context
- **output_schema:** `BeliefMessage`s, `PointState` from `occupancy_state`, `GridExport`
- **truth_access_policy:** deployment plane; no truth imports (`tests/unit/domains/spatial/test_m2s_leakage.py`)
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status EVALUATED; learned cell update and graph
  attention OPEN_BLOCKED (not implemented)
- **candidate_config_path:** `model:` block in `configs/eval/2s_e001.yaml`..`2s_e004.yaml` (`SpatialConfig`)
- **baseline_config_paths:** `PlainOccupancyGrid`, `MaxLikelihoodSurfaceFill`, `ignore_pose_config` ablation
  (names `plain_grid`, `ml_fill`, `uahsm_no_pose_cov` in the same configs)
- **losses_and_masks:** none (analytic)
- **train_val_test_ood_manifest:** OPEN (Twin2S OCPWE worlds, seeded)
- **unit_and_leakage_tests:** `tests/unit/domains/spatial/*`, `tests/property/domains/spatial/test_m2s_properties.py`,
  `tests/simulation/spatial/test_m2s_twin2s_loop.py`
- **experiment_ids:** 2S-E001..E004
- **open_decisions:** spatial representation, learned update/retrieval backbone; sensor quality not inferred from
  signal (2S-E004 defect)
- **ADR_ids:** ADR-0002

## Twin2S/OCPWE

- **package_id:** Twin2S/OCPWE (`conrad/twins/twin2s`, shared scenario in `conrad/sim/scenarios`)
- **source_chapters:** ch14, ch15, ch33, ch2
- **input_schema:** `Scenario` (from `build_pipeline_inspection_scenario`), `SensingContext`
- **output_schema:** `TwinSample` for DEPTH_RANGE, RGB, SONAR, POINT_CLOUD, PRESSURE_DEPTH, IMU; octree export;
  visibility oracle
- **truth_access_policy:** truth plane
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; geometry contract, octree export,
  twin interface and truth/observability separation FROZEN_CONTRACT
- **candidate_config_path:** `configs/sim/twin2s_default.yaml`
- **baseline_config_paths:** OCPWE `STYLES` (coverage, redundant, poor, information_seeking); other baselines
  listed in metadata only
- **losses_and_masks:** none
- **train_val_test_ood_manifest:** OPEN (procedural generator, seeded)
- **unit_and_leakage_tests:** `tests/unit/twins/twin2s/*`, `tests/property/twins/twin2s/test_twin2s_properties.py`,
  `tests/simulation/twin2s/*`
- **experiment_ids:** none of its own; drives 2S-E001..E004
- **open_decisions:** target-domain assets, sensor priors, realism calibration; `conrad.sim.scenarios` metadata
  names `tests/simulation/twin2s/test_shared_scenario.py`, which does not exist
- **ADR_ids:** ADR-0002

## Model1/EGDC

- **package_id:** Model1/EGDC (`conrad/decision`)
- **source_chapters:** ch16, ch17, ch28, ch33 Model 1 EGDC exact implementation
- **input_schema:** `DecisionContext` (mission, requirements, `BeliefSnapshot`, robot/resource/link state, health)
- **output_schema:** `DecisionOutcome` (`DecisionRecord`, `ProvenanceRecord` with `SourceType.DECISION`,
  `RoutedAction`, rejected actions)
- **truth_access_policy:** decision plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status EVALUATED; default `StructuredReasoningPolicy`
- **candidate_config_path:** `DecisionConfig` in `conrad/decision/config.py`; `EGDCScorerConfig` in `conrad/decision/learned.py`
- **baseline_config_paths:** `configs/eval/m1_uir_e001.yaml` (`naive_act_on_claims`)
- **losses_and_masks:** learned scorer only: `egdc_loss` = action CE + support BCE + masked outcome MSE + UIR
  penalty; `node_mask`, `unsupported_mask`, `action_mask`
- **train_val_test_ood_manifest:** OPEN (synthetic fault fixtures, `oracle/decision_oracle.py`)
- **unit_and_leakage_tests:** `tests/unit/decision/test_egdc.py`, `tests/unit/decision/test_egdc_learned.py`,
  `tests/property/decision/test_egdc_properties.py`
- **experiment_ids:** M1-UIR-E001
- **open_decisions:** reasoning engine, ranking/value model, learning objective; silent miscalibration is not
  detectable from belief data (M1-UIR-E001)
- **ADR_ids:** ADR-0004

## MCBR

- **package_id:** MCBR (`conrad/active`)
- **source_chapters:** ch16, ch18, ch33 MCBR exact implementation
- **input_schema:** `PlanningRequest` (an `InformationNeed`, beliefs, prior views, injected `is_free`,
  `predicted_visibility`, `navigation_cost`)
- **output_schema:** `PlanResult(plan: ObservationPlan, provenance, table)`
- **truth_access_policy:** decision plane; no truth imports; world knowledge only through injected callables
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status EVALUATED; `model_version "mcbr-analytic-0.2"`
- **candidate_config_path:** `MCBRConfig` in `conrad/active/config.py`; `MCBRRankerConfig` in `conrad/active/learned.py`
- **baseline_config_paths:** `configs/eval/active_mcbr_e001.yaml`; `make_planners` (A-B0..A-B7, A-B9, A-B10)
- **losses_and_masks:** learned ranker only: `mcbr_loss` = Huber value + pairwise logistic rank + Brier
  visibility; `candidate_mask`, `context_mask`
- **train_val_test_ood_manifest:** OPEN (`oracle/occlusion_world.py`, seeded)
- **unit_and_leakage_tests:** `tests/unit/active/test_mcbr.py`
- **experiment_ids:** ACTIVE-MCBR-E001 (failed hypothesis; value ranking KILL_CANDIDATE)
- **open_decisions:** information-value estimator, hypothesis representation, learned vs analytic ranker; A-B8 RL
  baseline not implemented
- **ADR_ids:** ADR-0004

## BAAC

- **package_id:** BAAC (`conrad/communication`)
- **source_chapters:** ch16, ch19, ch33 BAAC exact implementation
- **input_schema:** `BeliefMessage` + mission value (`BAACSender.offer`), `LinkProfile`s
- **output_schema:** `InformationUnit`s, `Transmission`s, `CommunicationState`; receiver-side `ReceiverStore`
- **truth_access_policy:** decision plane; no truth imports
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status EVALUATED; runtime policy `C-B10_baac`
- **candidate_config_path:** `BAACConfig` in `conrad/communication/config.py`; `BAACHeadsConfig` in `learned.py`
- **baseline_config_paths:** `configs/eval/com_baac_e001.yaml`; `BASELINE_POLICIES` (C-B0..C-B4)
- **losses_and_masks:** learned heads only: `baac_loss` = Huber value + BCE novelty + BCE deadline + MSE info
  loss; `unit_mask`, `receiver_mask`
- **train_val_test_ood_manifest:** OPEN (synthetic channel and units, seeded)
- **unit_and_leakage_tests:** `tests/unit/communication/test_baac.py`, `tests/property/communication/test_baac_properties.py`
- **experiment_ids:** COM-BAAC-E001
- **open_decisions:** scheduler, learned value estimator, codec/fidelity predictor, channel predictor; C-B5..C-B9
  not implemented; channel numbers SYNTHETIC_ONLY
- **ADR_ids:** ADR-0004

## Navigation/state estimation

- **package_id:** Navigation/state estimation (`conrad/robotics/{estimation,navigation,trajectory,control,allocation,safety}`)
- **source_chapters:** ch20
- **input_schema:** RHI getters through a read-only wrapper, `NavigationGoal`, position fixes (`Observation` with
  `sensor_context kind=POSITION_FIX`)
- **output_schema:** `StepResult` with an `AllocatedCommand` carrying a `SafetyAuthorization` (submitted to the gateway
  by the caller)
- **truth_access_policy:** execution plane; no truth imports; benchmarks read truth only through `TruthAccess`
- **implementation_status:** EXPERIMENTAL_CANDIDATE, claim_status IMPLEMENTED; `nav_benchmarks` EVALUATED
- **candidate_config_path:** `configs/sim/nav_kernel.yaml`, `configs/robot/sim_reference.yaml`
- **baseline_config_paths:** `configs/sim/nav_benchmarks.yaml`
- **losses_and_masks:** none (EKF, A*, PID, BVLS allocation, rule supervisor)
- **train_val_test_ood_manifest:** OPEN (not learned)
- **unit_and_leakage_tests:** `tests/unit/robotics/*`, `tests/property/robotics`, `tests/simulation/nav/test_nav_benchmark_suite.py`
- **experiment_ids:** NAV-001..NAV-008
- **open_decisions:** EKF overconfident under unmodelled IMU noise (NAV-007); MPC and learned residual controllers
  are interfaces only (OPEN_BLOCKED); control metadata names `configs/sim/nav_stack.yaml`, which does not exist
- **ADR_ids:** ADR-0002, ADR-0005

## Unity V2/RHI

- **package_id:** Unity V2/RHI (`conrad/robotics/hardware/interface.py`, `conrad/sim/kernel`, `conrad/adapters/unity`,
  `conrad/sim/unity`, `unity/ConradUnityV2`)
- **source_chapters:** ch20, ch21, ch28, ch34
- **input_schema:** `AllocatedCommand` (via the gateway only)
- **output_schema:** `ImuSample`, `DepthSample`, camera/sonar `Observation`s, `ThrusterState`, `SystemHealth`,
  `CommandAck`, `RobotCapabilities`
- **truth_access_policy:** adapters never expose truth through the RHI; kernel truth only via `TruthAccess`;
  Unity truth via the truth-plane client `conrad/sim/unity/truth.py`
- **implementation_status:** interface FROZEN_CONTRACT; kernel and Unity adapter EXPERIMENTAL_CANDIDATE,
  IMPLEMENTED; kernel validity L1; `PhysicalRobotHardware` raises (EXT-HW-01)
- **candidate_config_path:** `configs/sim/unity_lockstep.yaml`, `configs/sim/unity_realtime.yaml`, `configs/sim/nav_kernel.yaml`
- **baseline_config_paths:** `conrad.sim.kernel` is the baseline for Unity
- **losses_and_masks:** none
- **train_val_test_ood_manifest:** OPEN (not learned)
- **unit_and_leakage_tests:** `tests/contract/unity/*`, `tests/unit/adapters/unity/*`, `tests/unit/sim_kernel/*`,
  `tests/hardware_stub/*`, `tests/leakage/test_static_boundaries.py::test_only_the_gateway_calls_hardware_send`
- **experiment_ids:** none
- **open_decisions:** Unity execution and kernel/Unity cross-validation (EXT-UNITY-01); physical driver
  (EXT-HW-01); frame contract (EXT-HW-03)
- **ADR_ids:** ADR-0002, ADR-0005
