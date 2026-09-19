# Executed experiments: results record, 2026-09-18

Every run below was executed on CPU (no GPU on this host) with seeds 2026201/2026202/2026203 unless noted. All data
is **synthetic** (abstract Core sandbox or Twins at simulation validity L1). Nothing here is real-data or physical
validation. Raw JSON is in `artifacts/experiments/**` (gitignored; regenerate with `conrad eval run --experiment <ID>`
or each package's `run_all`). Where a baseline wins, the baseline is reported as winning. Registry outcome for every
row: **INCONCLUSIVE** until an ADR evaluates a paired effect under ch35/36 rules.

## Model 2 Core (abstract sandbox)

| ID | Candidate | Result | Verdict |
|---|---|---|---|
| CORE-ASSOC-E001 | learned pair scorer + NO_MATCH | accuracy 0.669±0.182 vs **nearest-neighbour 0.955±0.016**; false association 0.0008 vs 0.045 (the learned scorer is over-conservative) | **Baseline wins → KILL_CANDIDATE (learned association)** |
| CORE-BUO-E001 | AnalyticBUO / learned BUO | reliable support RMSE 0.021 (= averaging); unreliable contradiction 0.020 vs 0.030; degradation 0.035 vs 0.042; **true change: latest-only 0.049 beats 0.103**. Without the contradiction mechanism: confident-wrong 0.47 vs 0. Learned BUO is worse than analytic everywhere (0.076-0.184) | Analytic operator retained; learned BUO → KILL_CANDIDATE |
| CORE-UNC-E001 | 4-channel uncertainty | diagonal Spearman 1.0/1.0/1.0/0.976; noise→UO cross-talk 1.0; calibration: 95% coverage 0.963, ECE 0.022 vs averaging 0.87 / 0.156 | Decomposition behaves as specified; one cross-talk defect recorded |
| CORE-RBP-E001 | AnalyticRBP / learned typed RBP | true edges RMSE 0.031 vs 0.231 (no propagation); **misleading edges: 0.58 of decoys confidently wrong vs 0**; learned RBP is worse (~0.20) with contamination ~0.47; typing gives no advantage | **Contamination failure → RBP off by default in the runtime; KILL_CANDIDATE (learned RBP)** |
| CORE-TBD-E001 | GRU TBD | RMSE GRU 0.140, MLP 0.230, GRU with sequence index 0.298, hold-last 0.409, linear 0.685; Δt→variance Spearman 0.78 | Learned GRU beats the simple baselines (the only learned Core win) |
| CORE-PERSIST-E001 | PMBL | purity 1.0, re-identification 1.0, provenance integrity 1.0, duplicate rejection 1.0, CC-09 reset 1.0; state RMSE equal to latest-only (0.147) | Identity/lineage behaviour correct; no accuracy benefit |
| CORE-FULL-E001 | full Core | RMSE full 0.118 vs no-RBP 0.113 vs latest-only 0.149; no uncertainty growth: coverage 0.584 | Full beats latest-only; removing RBP is slightly better overall |

## Domains

| ID | Result | Verdict |
|---|---|---|
| 2S-E001 coverage | hidden unsupported-confidence at 5% coverage: UAHSM **0.000**, plain grid 0.005, ML fill 0.905; observed IoU 0.501 vs 0.182; plain grid wins Brier/ECE and overall IoU | UAHSM meets its safety purpose; calibration is weaker |
| 2S-E002 counterfactual | confident-claim rate in differing cells: UAHSM 0.0, ML fill 1.0, plain grid 0.0 (tie); after a discriminating view, UO 1.0→0.01-0.03 | No invented hidden geometry |
| 2S-E003 pose σ | at σ=0.15 m confident-claim error 0.024 vs ablation 0.365 vs plain 0.413; UE rises with σ | Pose propagation works; the ablation has slightly better IoU |
| 2S-E004 modality | sonar-only: UAHSM never claims occupied; degraded sonar does not raise UA (sensor reports healthy) | Defect recorded: quality context is not inferred from signal |
| 2T-E001 direct | corrosion error 0.075 mm vs latest 0.156 (clean), 0.200 vs 0.555 (degraded); **cracks: mean error 1.43 vs latest 0.84** (median 0.53 vs 0.77); crack NLL 163 (overconfident on runaway cracks) | Corrosion wins; crack estimation → failed hypothesis |
| 2T-E002 partial coverage | hidden corrosion: wins at 50%/20% coverage, mixed at 5%; never-observed components are 100% UNKNOWN without TCDP; GRU is the worst arm | Persistence helps corrosion |
| 2T-E003 TCDP | benefit +0.022 mm (coupled), −0.014 (misleading); contamination TCDP 0.04 vs generic propagation 0.75 | Benefit tiny (2-3%); contamination control strong |
| 2T-E004 temporal | +0.045 mm over hold-last, but the zero-rate ablation still gives +0.039 | Most of the gain is smoothing, not the rate prior |
| 2E-E001 fields | static-field baseline wins turbidity (0.05-0.08 vs 0.17-0.21 NTU); all models are over-cautious (z² ≪ 1) | Baseline wins turbidity |
| 2E-E002 turbidity | CEFD UA/UO track turbidity (UO 0.04→0.82); cover RMSE 0.187 vs 0.206 at 25 NTU; uncoupled wins in clear water | Honest uncertainty; accuracy mixed |
| 2E-E003 coupling | coupling benefit ≈0 (−0.003..+0.003); **confounded world: confident thermal-stress claims on 100% of healthy entities** (correctly labelled INFERRED) | **D5 gate not met → CEFD coupling KILL_CANDIDATE** |

## Decision / active / communication

| ID | Result | Verdict |
|---|---|---|
| M1-UIR-E001 | UIR 0.000 vs naive 0.188; safe rate under faults 0.900 vs 0.457; MISCALIBRATED_SILENT: EGDC safe rate **0.0** | Grounding works; silent miscalibration is undetectable from belief data |
| ACTIVE-MCBR-E001 | mission error reduction: geometric NBV 0.487, standard EIG 0.481, MCBR w/o mission 0.477, MCBR no-stop 0.465, **full MCBR 0.359 (8th/11)**; fewest redundant observations (0.36) and least travel | **Failed hypothesis: MCBR does not beat standard baselines on hidden-state error → KILL_CANDIDATE for value ranking**; stop-rule cost weights are over-strict in OOD |
| COM-BAAC-E001 | retained 0.866/0.749/0.389/0.136/0.048/0 at 100/50/10/1/0.1/0%; critical alerts 1.0 delivered at every non-zero budget (latency 1.1-74 s); send-all/FIFO/fixed-priority deliver 0-0.06 and no alerts; **value-per-bit wins at 1%** (0.166 vs 0.136) | BAAC wins at most budgets; not at 1% |

## Navigation (sim kernel L1, SYNTHETIC_ONLY RobotConfig; seeds 1 and 2)

NAV-001..NAV-008 all pass their scenario checks with 0 collisions and 0 gateway rejections. Final error 0.04-0.23 m.
NAV-007 (pose degradation) passes the final check, but during the fix outage the true error reached 5.8 m while the
EKF's own σ was still below the 1.5 m hold limit, so **the estimator is overconfident under unmodelled IMU noise**.

## Training smoke

`conrad train run --config configs/train/core_smoke.yaml` (6 epochs, 102 steps, CPU): best val RMSE 1.034 vs
hold-last 0.758. The smoke-scale model is worse than the trivial baseline, as expected at this scale. Checkpoint
reload reproduces the metric exactly. `configs/train/core_full.yaml` is refused: BLOCKED_EXTERNAL (EXT-COMPUTE-01).

## Integrated missions (2026-09-19; sim kernel L1, SYNTHETIC_ONLY RobotConfig; 120 s missions)

Runs are in `artifacts/runs/<ID>-s<seed>-<hash>/`. Errors are against the evaluation-only truth record; "before" is the
error while the target component was still UNKNOWN (prior mean). Unless a row says otherwise: U_O went from 1.00 to
~0.30, UIR 0.0, 2400 commands accepted and 0 rejected by the gateway, 0 collisions, critical alert delivered once at
~2.4 s latency on a 1200 bps link.

| Run | Target state | Corrosion error before→after (mm) | Crack error (mm) | MCBR |
|---|---|---|---|---|
| FLAGSHIP-I4 s2026201 | UNKNOWN→OBSERVED | 5.50→0.006 | 78.5→0.24 | 1 plan, then NOT_WORTH_COST |
| FIXED-VIEW s2026201 | UNKNOWN→OBSERVED | 5.50→0.043 | 78.5→0.26 | (fixed pattern, 3 plans) |
| FLAGSHIP-I4 s2026203 | UNKNOWN→OBSERVED | 5.50→0.079 | 78.5→0.32 | 1 plan |
| FIXED-VIEW s2026203 | UNKNOWN→OBSERVED | 5.50→0.07 | 78.5→0.27 | 3 plans |
| FLAGSHIP-I4 s2026202 | already OBSERVED from the lane (the scenario's hidden-side premise did not hold) | 0.032 | 0.18 | NOT_WORTH_COST ×4 |

**Verdict.** The architecture closes the full loop on 2 of 3 seeds: UNKNOWN → InformationNeed → ObservationPlan →
navigation through the gateway → new STRUCTURED evidence → DIRECT revision → grounded decision → BAAC transmission,
with a provenance trace from the last command down to the raw observation and no truth leakage (dynamic leakage test).
**MCBR shows no hidden-state-error advantage over the fixed-view baseline on any seed**; it uses 1 plan where the
baseline uses 3. The I4 acceptance record is NOT_EVALUABLE because its thresholds are OPEN. Disclosure: the
cosine-incidence weighting of MCBR predicted visibility was added after inspecting seed 2026201.

**Known failures.**
- **I7-COMMS-OUTAGE / INT-010:** MCBR returned NOT_WORTH_COST (value 0.245 < cost 0.255), so no finding occurred and the
  "critical finding during outage" path was **not exercised**.
- **INT-004:** under sensing degradation, NOT_WORTH_COST left the target UNKNOWN.
- **INT-005:** the critical alert was not delivered at 100 bps with 15% loss.
- **INT-006:** the transit goal was rejected as GOAL_OUTSIDE_ENVELOPE (the lane leaves the mission boundary).
- **INT-003:** credible contradiction drove U_C to 1.0, but the biased sensor pulled the corrosion error to 3.49 mm.
- **EGDC:** decisions are dominated by ESCALATE_TO_OPERATOR because Model2T uncertainty is uncalibrated.

**Passing.**
- INT-001: the defect was visible from the lane.
- INT-002: same as the flagship.
- INT-007
- INT-008: thruster failure, still resolved.
- INT-009: position-fix outage, resolved.
- I6-MULTIDOMAIN: turbidity context raised U_A to 0.88; the target resolved.

## Note appended 2026-09-19: MCBR evidence contamination (nothing above is deleted or edited)

Seed **2026201** is **DEVELOPMENT / CONTAMINATED_FOR_FINAL_EVALUATION** for every MCBR metric: the cosine-incidence
weighting of MCBR predicted visibility (`conrad/orchestration/deliberation.py`) was added after inspecting that seed,
and 2026201 was also used as an evaluation seed. The following rows above are therefore development evidence only
and cannot support any final claim or gate I4:

- ACTIVE-MCBR-E001 (seeds 2026201/2026202/2026203): development evidence. 2026202 and 2026203 were inspected in the
  same write-up, so the whole E001 row is treated as development.
- Integrated missions, FLAGSHIP-I4 / FIXED-VIEW s2026201 (and the s2026202/s2026203 replicates, which were inspected
  together with it): development evidence.
- The previous I4 evidence is **FAIL**: MCBR showed no advantage over the fixed-view baseline, and the spec requires
  I4 to beat simple views under matched budgets (the margin is OPEN, the direction is not).

The re-evaluation with immutable partitions (`configs/eval/partitions.yaml`, `conrad/evaluation/partitions.py`),
validation-based selection and a frozen production planner is in `docs/audits/MCBR_REEVALUATION.md`
(ACTIVE-MCBR-SEL001, ACTIVE-MCBR-E002, ACTIVE-MCBR-E003).

## Correction appended 2026-09-19: structural crack and corrosion numbers (nothing above is deleted or edited)

The integrated-mission crack errors above (78.5→0.24 mm on FLAGSHIP-I4 s2026201; 0.26 / 0.32 / 0.27 / 0.18 mm on the
other rows) and the corrosion errors (0.006 to 0.079 mm) are **VALID_BUT_OPTIMISTIC_SENSOR_MODEL**. See
`docs/audits/STRUCTURAL_LINEAGE_AUDIT.md`. No truth leak reaches Model2T. The numbers came from a Twin2T T0 sensor
that added about 1 mm of iid noise to the true crack length. It reported the full 80 mm crack even when only a third
of the defect patch was in view, and 57 independent looks averaged the error away. The metric itself compared
against the right truth quantity. "Before" (78.5 mm) is the prior mean, not a sensor baseline; the
latest-observation error was 1.0 mm.

The Twin2T sensor model is now REALISTIC by default: partial-view sizing, a log-logistic POD, and multiplicative
sizing error with a persistent per-sensor bias. Re-run on the current tree, crack / corrosion error:
FLAGSHIP-I4 s2026201 24.4 / 0.33 mm, FIXED-VIEW s2026201 44.0 / 0.079 mm, FLAGSHIP-I4 s2026203 17.3 / 1.75 mm,
FIXED-VIEW s2026203 35.4 / 0.049 mm, FLAGSHIP-I4 s2026202 79.2 / 0.13 mm. 2T-E001 is unaffected by any leak. Its
crack failure comes from run-away cracks at the 1 m cap (model 7.6 mm vs latest 1.7 mm there); on near-static cracks
the model already beat latest-observation (0.81 vs 1.34 mm). The flagship crack is exactly that static case.
