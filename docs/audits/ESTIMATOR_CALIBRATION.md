# Estimator calibration audit (NAV-CAL-E001)

Label: SYNTHETIC_ONLY. L1 Python kernel, simulated vehicle `configs/robot/sim_reference.yaml`. Nothing
here says anything about a physical vehicle or a real sea state.

## Defect

NAV-007 (fix outage 8 to 50 s, IMU noise x5, 0.15 m/s cross current), seed 1, legacy EST-B0 EKF:
true position error reached 5.82 m while the filter's RMS 1-sigma was still about 1.5 m.
LOCALIZATION_LOST fired only at t = 39.1 s, when the true error was already about 4.4 m.

Root cause, from per-state traces: the error was not driven by IMU noise. The thrust/drag velocity prior
predicts water-relative velocity, and the water current made it biased by a constant amount. The filter
treated that bias as white noise. It re-applied the prior every tick, which pulled the velocity estimate
onto the biased value while the covariance shrank. Position error grew linearly in t, but the covariance
grew only like sqrt(t). The 5x IMU noise adds per-sample noise of about 0.05 m/s^2, and the configured
0.03 m/s^2/sqrt(Hz) process noise already covers that.

## Fix (EST-B1, `conrad/robotics/estimation/ekf.py`)

- **Prior-mismatch state.** A random-walk WORLD velocity offset `c` (current plus drag-model error) is
  added as states 16 to 18. The prior residual is `v_model - R^T (v - c)`. While fixes arrive, `c` is
  observable. During an outage its estimate is held, and its variance grows with
  `mismatch_walk_mps_per_sqrt_s`, so position variance grows like t^3. A first Gauss-Markov version
  (mean reversion to zero) was rejected on dev seeds because it decayed the learned current to zero
  during the outage.
- **Process noise.** The configured accel/gyro noise densities and the bias random walks are kept. The
  white part of the prior error stays at 0.25/0.25/0.4 m/s. A dev trial at 0.05 m/s looked better on the
  straight transits, but a unit test with a turning vehicle showed it was overconfident, so it was
  dropped.
- **IMU noise inferred from data.** An EWMA of squared first differences of consecutive fresh IMU samples
  (accel and gyro) is added to the process noise. The estimator has no field for the injected noise
  level. The test checks that the estimate rises by more than 4x under a 5x fault.
- **AHRS covariance matching.** `R_hat = mean(nu^2) - mean(diag P_att)`, and the larger of `R_hat` and the
  configured R is used.
- **Consistency monitoring.**
  - Position fixes are chi-square gated (3 dof, p = 0.001, 16.27). Depth is now gated too (1 dof,
    p = 0.001, 10.83).
  - An EWMA of NIS/dof over about 10 fixes is kept. Above `nis_ratio_limit = 2.0` the filter reports
    DEGRADED with `ESTIMATOR_INCONSISTENT`, and the mismatch process noise is scaled by that ratio
    (capped at 25).
- **LOCALIZATION_LOST** uses only thresholds that were already configured: RMS sigma above
  `RobotConfig.safety.max_pose_sigma_m` (1.5 m), 5 consecutive gated fixes, or a non-finite covariance.
- `EkfConfig.legacy_baseline()` reproduces EST-B0 exactly. NAV-007 seed 1 gives max error 5.8243 m, the
  same as before the change.

Every new threshold and noise value is a `SYNTHETIC_ONLY` configured default and is labelled in the field
descriptions. The chi-square gates are statistical constants.

## Safety consumption

- **Existing behaviour, verified.**
  - sigma above 0.5 x max: DEGRADED, speed x0.5.
  - sigma above max, or estimator FAULT: HOLD with the configured `safe_hold_action`.
  - HOLD longer than the relocalization timeout: RETURN.
- **Gap fixed.** Estimator DEGRADED (inconsistency or gated fixes/depth) was ignored whenever sigma looked
  small. It now maps to DEGRADED with `ESTIMATOR_DEGRADED` plus the estimator's own reason codes.
- **Tests** (`tests/unit/robotics/test_nav_estimator_calibration.py`):
  - A blind NavigationStack goes NORMAL, then DEGRADED (speed scaled), then HOLD. With STATION_KEEP the
    hold target is fixed. With ZERO_THRUST every command is all zero.
  - A degraded estimator covariance, fed through `Observation.robot_pose_estimate`, gives Model2S a larger
    hit footprint and a lower maximum occupancy probability.

## Experiment design

- Closed-loop NavigationStack transit, fixes at 1 Hz with sigma 0.1 m. The outage starts at 20 s.
- The IMU noise fault is injected at the outage start.
- Per-seed current: uniform magnitude up to 0.25 m/s, uniform direction.
- One gust: up to 0.15 m/s, 10 s long, starting at a uniform time inside the outage.
- Grid: outage 5/15/30/60 s x IMU noise 1x/3x/5x.
- Metrics are sampled every 0.5 s over the outage window. Containment and NEES_h use the horizontal axes.
- Danger threshold: `SafetyConfig.sigma_k x max_pose_sigma_m` = 2 x 1.5 = 3.0 m. That is the
  collision-envelope allowance at the LOCALIZATION_LOST threshold.
- Missed danger: time during which the true error is above 3.0 m and LOCALIZATION_LOST has not been
  declared.

Seed partitions (`configs/sim/nav_calibration.yaml`):

| Partition | Seeds | Use |
|---|---|---|
| dev | 101, 102, 103, 104, 105 | all design and tuning (mismatch model form, walk rate, prior white noise) |
| final (held-out) | 9001 to 9010 | run once, after the code was frozen, for the numbers below |

NAV-007 seed 1 was also used to diagnose the defect.

## Held-out results (10 seeds per cell)

Artifacts: `artifacts/experiments/NAV-CAL-E001/summary.json` and `cases.json`. Nominal containment is
68.3 / 95.4 / 99.7 %.

| Outage | IMU | Variant | NEES_h/dof | 1s / 2s / 3s % | max e/sqrt(trP) | mean / max err m | LOST at s (share) | danger reached | missed danger mean / max s |
|---|---|---|---|---|---|---|---|---|---|
| 5 s | 5x | before | 0.44 | 86 / 100 / 100 | 1.27 | 0.19 / 0.56 | never | 0 % | 0 / 0 |
| 5 s | 5x | after | 0.47 | 80 / 100 / 100 | 1.27 | 0.21 / 0.59 | never | 0 % | 0 / 0 |
| 15 s | 5x | before | 0.51 | 85 / 100 / 100 | 1.39 | 0.59 / 2.21 | never | 0 % | 0 / 0 |
| 15 s | 5x | after | 0.37 | 93 / 100 / 100 | 1.27 | 0.53 / 1.69 | never | 0 % | 0 / 0 |
| 30 s | 1x | before | 0.84 | 70 / 98 / 100 | 2.00 | 1.34 / 4.98 | never | 40 % | 3.35 / 12.0 |
| 30 s | 1x | after | 0.17 | 97 / 100 / 100 | 1.28 | 0.61 / 2.02 | 18.6 (100 %) | 0 % | 0 / 0 |
| 30 s | 5x | before | 0.81 | 73 / 98 / 100 | 2.07 | 1.31 / 5.16 | never | 30 % | 2.70 / 11.0 |
| 30 s | 5x | after | 0.28 | 96 / 100 / 100 | 1.27 | 0.90 / 2.53 | 18.6 (100 %) | 0 % | 0 / 0 |
| 60 s | 1x | before | 1.58 | 58 / 89 / 97 | 2.99 | 2.80 / 11.04 | 32.1 (100 %) | 70 % | 4.10 / 14.0 |
| 60 s | 1x | after | 0.10 | 99 / 100 / 100 | 1.28 | 0.94 / 2.95 | 18.6 (100 %) | 0 % | 0 / 0 |
| 60 s | 5x | before | 1.53 | 59 / 90 / 97 | 3.11 | 2.73 / 11.45 | 32.1 (100 %) | 70 % | 3.45 / 14.5 |
| 60 s | 5x | after | 0.20 | 98 / 100 / 100 | 1.27 | 1.55 / 5.35 | 18.6 (100 %) | 50 % | 0 / 0 |
| all 12 cells | | before | 0.82 | 75 / 97 / 99 | 3.11 | 1.21 / 11.45 | 32.1 (25 %) | 27 % | 1.71 / 14.5 |
| all 12 cells | | after | 0.27 | 93 / 100 / 100 | 1.28 | 0.65 / 5.35 | 18.6 (50 %) | 6 % | 0 / 0 |

The 3x cells lie between the 1x and 5x cells. They are in `summary.json`.

### What the numbers show

- **Before.** The pooled averages look acceptable, but they hide the tail. In long outages the error runs
  up to 3.1 x sqrt(trace P). The true error passes 3 m in 40 to 70 % of the 30 and 60 s runs before
  LOST is declared, for up to 14.5 s.
- **After.** Missed danger is 0 s in all 120 held-out runs. The worst error-to-sigma ratio is 1.28. LOST
  fires at 18.6 s into the outage, and HOLD follows in the same tick.
  - The 60 s / 5x cells still reach 3 m of true error in 5 of 10 seeds. That happens after LOST, while the
    vehicle is holding station on a drifting estimate. The supervisor's relocalization timeout (60 s) then
    escalates to RETURN.
- **Mean point error falls** from 1.21 to 0.65 m, because `c` learns the current before the outage.

## Remaining limitations

- **Conservative, not calibrated.** After the fix, NEES_h/dof is 0.10 to 0.47, well below 1, and 1-sigma
  containment is 80 to 99 % against a nominal 68 %. The filter overstates uncertainty, which is the safe
  direction. Reaching nominal would mean trusting the sim's thrust model, which is exact apart from the
  current. Dev trials showed that this breaks as soon as the vehicle turns.
- **Growth during an outage does not depend on the data.** Without fixes nothing new can be learned, so
  time to LOST (18.6 s) is the same for every seed and noise level. The NIS-driven noise scaling only acts
  once fixes return.
- **The IMU adaptation is correct but has almost no effect here.** At 5x the measured white noise
  (about 0.05 m/s^2 per sample) is far below the configured accel density. The first-difference estimator
  also includes vehicle dynamics, which makes it conservative.
- **Samples are correlated in time**, so containment rates have fewer effective samples than the counts
  suggest. With 10 seeds per cell, tail estimates are coarse.
- **Envelope convention.** LOCALIZATION_LOST uses the RMS per-axis sigma, which is diluted by the
  well-observed depth axis. The 3 m danger threshold is an evaluation choice tied to
  `sigma_k x max_pose_sigma_m`.
- **Simulation scope.** The results cover one simulated vehicle, one current and gust model, and fixes
  from a synthetic USBL-like sensor.
- **Physical values are open.** The walk rate, white prior noise and NIS limits are open items until
  hardware and sea-state characterization exist.
