# I4 energy-efficiency repair cycle

Status: **STOPPED AFTER DEVELOPMENT**. No V5 selection, freeze, validation, or new formal evaluation exists. The official I4 formal verdict remains **FAIL (4/5)**. This is a new cycle authorized after the immutable V4 failure; its results cannot rewrite that result.

The old V4 audit closed that experiment with "no V5" under its own stopping rule. This document records the user's later, separate authorization for a new mechanism cycle. It does not extend the old final split or reinterpret its CI.

## Chronology and integrity boundary

1. Started from `origin/kerem/i7-final` at `b6f7cb203aad0ed9aee4b8ea677ece329f5b008a` in isolated branch `kerem/i4-energy-repair`. The named main worktree had unrelated modified and untracked I5/Model1 artifacts, which were preserved. The legacy Conrad repository was inspected read-only at `6bf01ee`, with zero modified files.
2. Read `artifacts/gates/I4/evidence_formal.json`, the V4 Unity result and protocol, `MCBR_V4.md`, `MCBR_REEVALUATION.md`, `I4_WORLD_FAMILY.md`, and the frozen V4 configuration. V4's hidden-state improvement over random was +0.155620 [+0.001862, +0.308009], while info/kJ over random was +0.037653 [-0.000208, +0.074893]. The latter fails the unchanged paired-CI rule.
3. Wrote [I4_ENERGY_EFFICIENCY_FAILURE_ANALYSIS.md](I4_ENERGY_EFFICIENCY_FAILURE_ANALYSIS.md) before mechanism changes. Historical E007 development and validation summaries lack per-view energy and predicted-versus-realized cost. They also predate current shared runtime changes; a current-code development repeat does not reproduce the old per-world row. Those summaries are historical context, not current calibration data.
4. Added development-only tracing in `scripts/diagnose_i4_energy.py`, which records per-tick battery-meter deltas, estimated and truth-side motion, executive phase, and the view ledger, then joins accepted views to planner costs. `scripts/summarize_i4_energy.py` separates completed-view error from censored abandonments and requires all 40 development seeds before a complete analysis. The diagnostic samples truth separately for after-run distance analysis; it never supplies truth to the planner. Neither script changes the gate metric or formal evidence.
5. Re-executed all 40 historical development seeds as current-code Python diagnostics. The first batch command returned a serialization error when its worker tried to send a local `defaultdict` result to the parent; all 40 reports had already been written and the complete-report check and summarizer passed. The return value was converted to an ordinary dict; later new-development batches exited zero. Sixteen historical-development views ended at mission clock after consuming 31,802.5 J during their open intervals. Each such view was a 1-5 cm translation goal whose navigation trajectory was 20,917-95,077 s long because the two-point acceleration profile had zero speed at both ends. This is a concrete cost and execution failure, independent of the old formal result.
6. Declared `configs/eval/partitions_i4_energy_v1.yaml` with development 8100000-8100039, validation 8100100-8100139, reserved final pool 8100200-8100319, and OOD 8100400-8100419. Canonical SHA-256 `f608b278eee3200f51f6b695f3adda0b18152c7d965b02b57c5bc1b6884dec1f` is pinned in `conrad/evaluation/partitions.py`. The loader checks disjointness against every other partition YAML and applies the existing purpose guard. Only development worlds were run. Final n was never declared because no candidate qualified.
7. Added an opt-in short-leg trajectory correction. When enabled and the route resamples to only two points, it inserts a midpoint so the existing acceleration model computes a finite triangular profile. The default remains off; the V4 incumbent and historical frozen configuration retain their original behavior. A targeted unit test passes. This is a development candidate, not a selected repair.
8. Ran that candidate on all 40 new development worlds. It reduced `ABANDONED_MISSION_END` from 11 views to 3 and open-view energy in that category from 23,042.4 J to 1,916.1 J. The defect read count stayed 20/40; mean mission energy increased from 16,736.6 J to 16,927.2 J, and mean information/kJ moved only from 0.021475 to 0.021559. The paired difference was +0.000084 with a 4000-resample development bootstrap CI [0.000000, +0.000252]. Only one world increased hidden-state improvement at all (+0.0526), whereas one unchanged-information world consumed an extra 4,817.5 J. The mechanism repairs navigation completion but has not established meaningful energy-efficiency headroom. No candidate selected.
9. Fitted the declared two-offset cost correction on completed views in development worlds 8100000-8100019, then checked worlds 8100020-8100039. The offsets are +2.8 s and +1,600 J. Check-set time MAE fell from 4.32 to 2.27 s, and energy MAE from 3,280.5 to 2,596.7 J. Calibration improved prediction but still had substantial residual energy error.
10. Completed the ten-world paired screen of V4, short-leg, time, energy, horizon and combined variants plus unchanged fixed, random and coverage arms. Time-only and horizon-only produced identical outcomes to V4. Energy-only increased mean mission energy by 889.6 J, yielded no information/kJ gain and caused one collision where V4 had none. Combined increased mean energy by 885.1 J, caused the same collision, and its +0.000336 information/kJ gain exactly matched the short-leg-only arm. That tiny gain arose from one world and was not a new energy mechanism. The mandatory safety and material-effect requirements therefore rejected the combined candidate. The full 40-world short-leg result had already failed the material-effect requirement.
11. Stopped before validation. No candidate advanced to V5, no final sample size or player was declared, and no Unity final flight was authorized by the protocol. All validation, final and OOD worlds remain unread.

The candidate code is opt-in: `TrajectoryGenerator.generate` inserts a midpoint for a two-point route only when `TrajectoryConfig.short_leg_fix` is enabled; `MissionRuntime` passes the `trajectory_short_leg_fix` setting to navigation; `I4CostCalibration.estimate` computes nonnegative finite offsets from belief-side path length; and `Deliberation.navigation_cost` supplies that corrected `ResourceCost` to the existing V4 filter and ranker only when enabled. `DecisionRouting` and the V4 ranking rule were not changed. No second path planner, early-stop state, baseline change or energy-accounting exception was added.

## Predeclared design and selection boundary

The new mechanism will be considered only if development measurements show a systematic, actionable cost bias. Candidate comparisons must include the V4 incumbent, calibrated route time, calibrated energy, completion filtering, and combined calibration plus filtering. A distinct energy-score variant is only warranted after calibration shows residual benefit. Selection is hierarchical: preserve closed loop, hidden-state reconstruction advantage against fixed, random and coverage, truth separation, and safety; then compare info/kJ, no-value energy, mission-end abandonment, cost errors and info/time. Use new disjoint development and validation partitions; do not read new final or OOD worlds during design. The source ch25 I4 wording and X-E04 call for the full adaptive system against simple-view systems under matched resources, so the formal comparison will use full production stacks, explicitly reporting that execution protocols differ.

After the one-world short-leg smoke check and before inspecting the complete corrected-candidate results, the cost fit was limited to completed views on new development seeds 8100000-8100019, with 8100020-8100039 serving as a development-only calibration check. `scripts/fit_i4_energy_cost.py` takes the median residual of full-view elapsed time and metered energy relative to the original V4 expected cost, rounding upward to 0.1 s and 100 J. Path-length slopes remain V4's existing values. Censored views do not enter the fit. This is a deliberately small model; both before- and after-calibration errors will be published on the 20-world check. No validation or final seed may enter this fit.

The development ablation set is fixed before the complete corrected batch: V4 incumbent; V4 with only the short-leg trajectory correction; V4 with the fitted time offset only; V4 with the fitted energy offset only; V4 with the mission-duration horizon and a time reserve equal to the fitted time offset; and V4 with short-leg correction, both fitted offsets and the mission-duration horizon (the calibrated time supplies its margin). First screen the four cost/filter variants on development seeds 8100000-8100009, then run the complete 40-world development split for any mechanism that preserves the incumbent defect-read count and shows an information/kJ gain in the screen. The one-world short-leg smoke check is disclosed above and does not select a candidate. Random, fixed and coverage retain their old arm execution settings on the new paired development worlds.

## Development decision

### Cost and completion evidence

| New development trace | Completed / accepted views | Total-view time bias / MAE | Estimated travel bias / MAE | True travel bias / MAE | Metered energy bias / MAE |
|---|---:|---:|---:|---:|---:|
| V4, 40 worlds | 59 / 70 | +4.91 / 4.91 s | +2.70 / 2.73 m | +0.95 / 1.06 m | +4,017 / 4,017 J |
| Short-leg, 40 worlds | 81 / 84 | +4.16 / 4.16 s | +2.32 / 2.34 m | +0.87 / 0.95 m | +3,094 / 3,094 J |

Bias is realized minus predicted. The completed-view populations differ between arms; these rows do not estimate a paired correction to the same views. The short-leg fix did not itself change the planner's cost estimator. For the same 37 completed views in the held-back development calibration check, the +2.8 s / +1,600 J offsets reduced time MAE 4.32 to 2.27 s and energy MAE 3,281 to 2,597 J. Travel prediction was unchanged by the offset model. Of the 11 V4 mission-end views, ten had unfinishable short-leg trajectories; the short-leg arm had three mission-end views in total. Full error quantiles, RMSE and underprediction frequencies are in the two `development_summary_new_*.json` artifacts.

All numbers below are **Python-kernel surrogate** means on development seeds 8100000-8100009. The baseline arms retained the prior full-stack execution protocol. The complete per-world rows and paired bootstrap screen are in `artifacts/experiments/I4-ENERGY-ABLATIONS/development_screen_summary.json`. No validation or formal result follows from this table.

| Arm | Defect reads / 10 | Info/kJ | Mean energy J | Redundant obs./world | Mission-end abandoned/world | Collisions / 10 |
|---|---:|---:|---:|---:|---:|---:|
| V4 incumbent | 6 | 0.027490 | 17,105 | 1.2 | 0.3 | 0 |
| Short-leg correction | 6 | 0.027825 | 17,129 | 1.4 | 0.1 | 0 |
| Time correction | 6 | 0.027490 | 17,105 | 1.2 | 0.3 | 0 |
| Energy correction | 6 | 0.027490 | 17,995 | 1.1 | 0.4 | 1 |
| Horizon filter | 6 | 0.027490 | 17,105 | 1.2 | 0.3 | 0 |
| Combined | 6 | 0.027825 | 17,990 | 1.4 | 0.2 | 1 |
| Fixed | 2 | 0.008622 | 17,045 | 3.4 | 0.0 | 0 |
| Random | 4 | 0.018286 | 17,895 | 2.1 | 0.3 | 0 |
| Coverage | 4 | 0.015948 | 18,437 | 3.1 | 0.1 | 2 |

On the full 40-world development split, V4 and short-leg both read 20 defects. Short-leg added 22 flown views and 0.3 redundant observations per world, raised mean energy by 190.6 J, and gained only +0.000084 information/kJ paired [0.000000, +0.000252]. The no-value diagnostic `wasted_energy_j` remains **unidentified**: an inspection approach without a direct target revision may still yield mission-relevant spatial evidence. We report the measured energy of abandoned view intervals and post-read intervals without declaring all of it waste.

## Formal protocol status

The new partition file is digest-pinned, but only its development split was used. Validation selection, prospective paired-variance estimate, minimum meaningful effect, fixed final sample size, final-world prefix, frozen V5 planner digest, formal Unity player SHA, and formal config SHA were **not declared** because development did not qualify a candidate. The intended comparator set was fixed, random and coverage under the same 100 s mission and unchanged energy meter. The authoritative paired-world, 95% percentile-bootstrap rule with 4000 resamples and RNG seed 20260919 was not applied to a new final set. A Unity player was not launched, so this cycle has no formal replay or truth-removal result and no new formal comparator CIs. The old I4 formal evidence and verdict remain authoritative.

The optional short-leg correction and calibration fields are not selected production settings. They default off. The shared source files are nevertheless changed in this research branch; enabling them in a future mission could change I5/I6/I7 behavior, so none of those gate records is refreshed or reinterpreted here. The frozen V4 configuration and its loader pointer remain unchanged.

## Integrity and stopped stages

| Stage | State |
|---|---|
| Energy decomposition | New 40-world V4 and short-leg traces complete |
| Predicted versus realized cost | New 40-world V4 and short-leg traces complete |
| Cost calibration | Fitted on 20 development worlds, checked on 20; no validation used |
| Mechanism candidates | Rejected: no material efficiency gain; energy/combined safety regression in screen |
| Development ablations | V4 and short-leg on 40 worlds; four cost/filter variants on ten worlds |
| New validation | 8100100-8100139, unread |
| V5 freeze | None |
| New final and OOD | Reserved, exact final n undeclared, all unread |
| Formal Unity result | NOT_RUN for this cycle; historical I4 remains FAIL |

The old 8002200-8002229 formal worlds were not reused. The old unread 8002230-8002259 and 8002300-8002319 remain unused. No held-out tuning, sample extension, baseline weakening, gate-metric change, or formal rescore occurred. No formal replay or truth-removal claim is made because this cycle had no frozen final candidate. Development code reads only belief-side inputs at runtime; truth-side diagnostic reads are isolated from the planner. Historical I4 stays **FORMAL FAIL, 4/5**, I5 stays **FORMAL FAIL, 9/10**, and I7 stays **surrogate FAIL 3/4; formal NOT_RUN**. Stop after I4; I5 repair is a separate cycle.

**I4 FORMAL = FAIL** (historical V4 verdict; new cycle formal evaluation NOT_RUN).
