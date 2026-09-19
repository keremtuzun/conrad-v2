# Identification log protocol (EXT-HW-04)

Protocol version: `identification-protocol.v1`. Audience: the hardware owner (Burak), who runs the
physical sessions. Software owner: Kerem (runner, intake, fitting, validation).

## Summary

**Definition.** Time-synchronized command, sensor, power and independently measured vehicle-motion
records from repeated controlled maneuvers (static trim, thruster steps, surge, sway, heave, yaw and
station keeping), split into identification and held-out validation runs, for estimating and
validating the robot's physical parameters.

In plain terms: we drive the vehicle through a fixed set of simple, repeatable moves. For each move
we record what we commanded, what the robot's own sensors saw, how much power it drew, and how the
vehicle really moved according to a measurement system that is **not** the robot's own estimator.
Every move is repeated. Before any fitting, whole repetitions are set aside for validation. We fit
the physics model on the rest and check it on the ones set aside.

What you deliver: one YAML manifest plus one CSV per repetition (format below). Before anything is
fitted, run the intake checker on it:

```
python -m uv run python -m conrad.robotics.hardware.identification \
    --manifest <delivery>/manifest.yaml --intake-only --out artifacts/identification/intake.json
```

The checker lists every problem with a code and rejects the delivery if anything is missing.

Labels in this document:

* `ENGINEERING_ESTIMATE`: a starting value chosen before any vehicle data exists. You may tighten it.
  It is never a vehicle measurement.
* `OPEN`: nobody has decided it yet. It stays open until the named owners set it.

No vehicle number (mass, thrust, drag, speed, depth) appears in this protocol. Where a level must be
chosen for the vehicle, it is expressed as a fraction of what the thrusters can deliver, and it is
capped by the session's safety envelope.

## Safety preconditions (every session)

1. Hardware control is **OFF by default**. The runner refuses a physical adapter unless all three
   are true in its config: `hardware_control_enabled: true`, a complete `safety_envelope`, and
   `operator_acknowledged: true`. The command gateway independently requires
   `command_mode: hardware`, `hardware_enable: true`, the HIL or PHYSICAL lane and an HIL evidence
   reference. CI never controls hardware.
2. The session safety envelope is declared before the session and recorded with it. Every field is
   `OPEN` until the hardware owner sets it: maximum normalised thruster command, maximum segment
   duration, maximum speed, maximum depth, maximum roll/pitch, and who set them (`set_by`). The runner
   aborts a repetition as soon as the reference shows a limit exceeded, and sends an explicit
   all-zero command.
3. Physical E-stop reachable by the operator for the whole session; tether and pool/tank clearance
   checked for the full profile length (the surge/sway/heave distances depend on the vehicle and are
   `OPEN` until the first low-level run).
4. Leak detection, battery state and thruster health are live. A safety supervisor state other than
   NORMAL or DEGRADED aborts the repetition. An aborted repetition is recorded in the manifest's
   `aborted` list and is **not** delivered as data.
5. Thruster steps (B) are done with the vehicle or the thruster fixed to a load cell, never free
   swimming.
6. Start every new manoeuvre kind at the lowest level of its profile and step up only after the lower
   level has been inspected.

## Channels (all kinds)

All channels of one repetition share one `time_s` column on one common clock. Column names and units
are exact; the intake checker compares them as strings. Minimum rates are `ENGINEERING_ESTIMATE`.
"Rate" means how often the value really updates: a sensor value repeated between updates does not
count as a new sample.

| Group | Columns | Units | Min rate | Source |
|---|---|---|---|---|
| Thruster command | `cmd_<thruster_id>` for every thruster | `unitless` (normalised, -1..1) | 20 Hz (the control rate) | the command the gateway accepted, stamped when sent |
| Thruster telemetry (if available) | `rpm_<thruster_id>`, `thr_current_<thruster_id>_a` | `rpm`, `A` | 20 Hz | ESC telemetry |
| Measured thrust (B only) | `thrust_n` | `N` | 100 Hz | load cell on the thruster under test |
| Independent pose | `ref_x_m`, `ref_y_m`, `ref_z_m` (WORLD, Z up) | `m` | 10 Hz | DVL / USBL / external tracking |
| Independent orientation | `ref_roll_rad`, `ref_pitch_rad`, `ref_yaw_rad` (ZYX) | `rad` | 10 Hz | same |
| Independent velocity | `ref_u_mps`, `ref_v_mps`, `ref_w_mps` (BODY) | `m/s` | 10 Hz | same |
| Independent rates | `ref_p_rps`, `ref_q_rps`, `ref_r_rps` (BODY) | `rad/s` | 10 Hz | same |
| IMU | `imu_ax_mps2`, `imu_ay_mps2`, `imu_az_mps2`; `imu_gx_rps`, `imu_gy_rps`, `imu_gz_rps` | `m/s^2`; `rad/s` | 50 Hz | onboard IMU |
| Pressure | `pressure_depth_m` | `m` | 5 Hz | onboard pressure sensor |
| Power | `battery_voltage_v`, `battery_current_a` | `V`, `A` | 1 Hz | battery monitor |
| Water current (F only) | `current_x_mps`, `current_y_mps` (WORLD); `current_z_mps` optional | `m/s` | 1 Hz | current meter / ADCP |
| Commanded wrench (optional) | `wrench_cmd_fx_n` ... `wrench_cmd_mz_nm` | `N`, `N*m` | 20 Hz | runner |

The independent reference must not be the robot's own state estimator, and it must not feed its
measurement into the robot's estimator during identification runs. Depth comes from `ref_z_m`; the
onboard `pressure_depth_m` is recorded as a sensor under test.

## Environment and configuration metadata

Per repetition (`metadata`, numbers): `water_density_kgm3` and `water_temperature_c`, measured at the
session. Per delivery (`configuration`, text): `ballast`, `payload`, `tether` (write `none` when there
is none). Any change of ballast, payload or tether starts a new delivery.

## Clock synchronization

One clock domain for every column (`clock.domain`), the method used to put every device on it
(`clock.synchronization`, e.g. PTP, a shared trigger pulse, or post-hoc alignment against a
recorded sync event), and the measured worst-case offset between any two devices
(`clock.max_offset_s`). The offset must be at most 5 ms (`ENGINEERING_ESTIMATE`; the thruster
latency being identified is expected to be of that order or larger, and the real value is `OPEN`).
`time_s` must be strictly increasing, with no gap longer than 5 times the median step.

## Manoeuvres

Minimum repetitions for every group (kind, variant, and thruster for B): **3 identification plus 1
held-out validation** (`ENGINEERING_ESTIMATE`). More is better. The default profiles below live in
`conrad/robotics/hardware/identification/protocol.py` (`DEFAULT_PROFILES`); every level is further
capped by the envelope's maximum command.

### A. Static trim and buoyancy (`A_STATIC_TRIM`)

* **Purpose / identifies:** displaced volume (buoyancy) and the centre-of-buoyancy to centre-of-mass
  offset (`z_bg_m`, `x_bg_m`), i.e. restoring behaviour.
* **Variant `SCALE`:** vehicle hung fully submerged from a scale, motionless. Log `net_weight_n` [N]
  (positive down) at 1 Hz or faster for at least 60 s (`ENGINEERING_ESTIMATE`), plus
  `pressure_depth_m`. Mass comes from the Phase 12 measurement, not from this log.
* **Variant `TILT_WEIGHTS`:** known weights at known lever arms apply pitch moments; log
  `applied_pitch_moment_nm` [N*m] held constant per plateau, with `ref_roll_rad`, `ref_pitch_rad`,
  IMU and pressure. At least 5 moment levels including zero and both signs.
* **Variant `TILT_THRUSTERS`:** pitch moments applied by the thrusters through the runner (needs the
  B models first). Default profile: pitch-moment plateaus at +-1 %, +-2 %, +-3 % of the pitch
  capability, 20 s on, 10 s off. Stop increasing when |pitch| reaches the envelope's attitude limit.
* **Settling:** only the settled tail of each plateau is used (default last 25 %). The settling time
  of the real vehicle is `OPEN`; lengthen the plateau if pitch is still oscillating at its end.

### B. Thruster steps on a load cell (`B_THRUSTER_STEP`)

* **Purpose / identifies:** per thruster forward and reverse thrust coefficients (asymmetry), first
  order time constant, pure delay (latency) and deadzone.
* **Command profile:** one thruster at a time; zero plateau, then level, then zero, for every level
  in the ladder +-0.02, +-0.04, +-0.06, +-0.08, +-0.1, +-0.2, +-0.4, +-0.6, +-0.8, **both signs**; 2 s
  on, 2 s off (`ENGINEERING_ESTIMATE`, at least 10 time constants each once B has been measured).
  The small levels bracket the deadzone; the zero plateaus give the load-cell noise floor.
  The deadzone is bracketed between two neighbouring levels, so its uncertainty is the ladder spacing
  there. Add finer levels once the first run shows where it is: small, noisy commands (station keeping,
  attitude hold) sit near the deadzone, and a coarse bracket biases the forces derived from them.
* **Required:** `cmd_<id>` for every thruster, `thrust_n` from the load cell at 100 Hz or faster,
  battery voltage and current; `thruster_id` on the segment; `thrust_measurement.device` in the
  manifest. RPM and current per thruster if the ESC reports them.
* **Every thruster that is used in C, D, E, F or A-tilt must have B logs.** The body forces used to
  fit those manoeuvres are computed from the logged commands through these identified thruster
  models and the RobotConfig thruster geometry.

### Common rules for A-tilt, C, D, E and F

* Every wrench profile starts and ends with a zero-command plateau (off, on, off, ..., off). The
  leading one gives the thruster models an at-rest initial condition.
* The rotational axes that a manoeuvre does not excite are held by the robot's own attitude
  controller, using the robot's own attitude estimate, during the run. The logged commands include
  that hold. Without it, a single-axis run drifts into turning and pitching (the kernel rehearsal
  showed pitch reaching 0.7 rad in an open-loop surge run), and the one-axis model no longer applies.
  Hold gains are controller tuning and are `OPEN` for the vehicle.
* Choose levels so the excited axis stays slow compared with the log rate. A response time constant
  shorter than a few samples biases the fit.

### C. Surge acceleration and coast-down (`C_SURGE_ACCELERATION`)

* **Purpose / identifies:** surge added mass, linear and quadratic surge drag.
* **Command profile:** surge-force plateaus at +-20 %, +-40 %, +-60 % of the surge capability,
  4 s on then 8 s coast at zero command, alternating direction (`ENGINEERING_ESTIMATE`; shorten to
  fit the facility, the distance is `OPEN`).
* **Required:** command, full independent reference, IMU, pressure, power.

### D. Sway (`D_SWAY`) and heave (`D_HEAVE`)

* **Purpose / identifies:** added mass, linear and quadratic drag on the lateral and vertical axes;
  heave also identifies the net buoyancy (bias), from which the displaced volume follows with the
  measured mass and water density.
* **Command profile:** as C on the sway or heave axis (up and down for heave).
* **Required:** as C.

### E. Yaw rotation (`E_YAW_ROTATION`)

* **Purpose / identifies:** yaw added inertia, linear and quadratic yaw damping. The rigid yaw
  inertia comes from Phase 12.
* **Command profile:** yaw-moment plateaus at +-20 %, +-40 %, +-60 % of the yaw capability, 3 s on,
  6 s off, both directions.
* **Required:** as C.

### F. Station keeping against a measured current (`F_STATION_KEEPING`)

* **Purpose / identifies:** the current the vehicle is holding against, from the hold force and the
  C surge drag; validated against the current meter. It tests the drag model under a disturbance.
* **Command profile:** the vehicle's station-keeping controller holds position with the current on
  the vehicle's surge axis, 120 s per repetition (`ENGINEERING_ESTIMATE`); the settled second half is
  used.
* **Required:** as C plus `current_x_mps`, `current_y_mps` and `current_measurement.device`. A
  station-keeping run without a measured current is rejected.

## Identification versus held-out validation

* Split by **whole repetitions**. A repetition is one execution of one profile; all files of one
  repetition go to the same split.
* The split is **assigned before any fitting** and written into the manifest's `split_plan`
  (`assigned_before_fitting: true`, the list of validation repetition ids and the list of
  identification repetition ids). Every segment's `split` must match the plan.
* Identification and validation data are **never mixed**: the fitting code only sees the
  identification split, and the validation code refuses identification segments.
* Suggested rule: decide the held-out repetition number per group before the session (for example
  "the last repetition of each group"), write it into the plan, and do not change it afterwards.

## Validation envelope

The acceptance limits on the held-out errors (per fit: RMSE of thrust, velocity, yaw rate, pitch
moment and hold force, in the units of that signal) are **OPEN**. They must be declared by the
owners (Kerem and Burak) **before** the held-out runs are scored, and passed to the pipeline as
`envelope_rmse`. Until then, validation reports the errors with `within_envelope: null`, and no
parameter can be promoted (Phase 13 pass criterion: held-out errors meet a pre-declared envelope;
only a validated envelope earns L4 twin status).

## Repeatability

The intake checker reports, per group, the root-mean-square excursion of the primary response
channel (thrust, reference velocity, yaw rate, pitch or position) for every repetition, its mean,
standard deviation and coefficient of variation, and flags repetitions with a robust z-score above
3.5 (`ENGINEERING_ESTIMATE`). Flags are warnings; the owners decide whether a flagged repetition is
re-run. There is no pass/fail threshold on the coefficient of variation yet (`OPEN`).

## Delivery format

The CSV and manifest extend the format in `conrad/robotics/hardware/identification/logio.py`.

```
<delivery>/
  manifest.yaml
  logs/<trajectory_id>.csv
```

CSV: header row; first column `time_s` (seconds on the common clock, strictly increasing); one column
per channel; numbers only.

```yaml
protocol_version: identification-protocol.v1
split_id: pool-2026-10-a
synthetic: false                     # true ONLY for simulator logs
source: PHYSICAL:<adapter or rig name>
thruster_ids: [H1, H2, ...]          # every thruster in the RobotConfig
clock: {domain: PTP-rig, synchronization: "PTP on all loggers", max_offset_s: 0.002}
motion_reference: {kind: EXTERNAL_TRACKING, device: "<make/model/serial>", independent_of_robot_estimator: true}
thrust_measurement: {device: "<load cell make/model/serial, calibration id>"}
current_measurement: {device: "<current meter make/model/serial>"}
configuration: {ballast: "...", payload: "...", tether: "..."}
split_plan:
  assigned_before_fitting: true
  identification_repetitions: [B-H1-r1, B-H1-r2, B-H1-r3, ...]
  validation_repetitions: [B-H1-r4, ...]
segments:
  - trajectory_id: B-H1-r1
    kind: B_THRUSTER_STEP            # A_STATIC_TRIM | B_THRUSTER_STEP | C_SURGE_ACCELERATION | D_SWAY | D_HEAVE | E_YAW_ROTATION | F_STATION_KEEPING
    variant: STANDARD                # A only: SCALE | TILT_WEIGHTS | TILT_THRUSTERS
    repetition_id: B-H1-r1
    thruster_id: H1                  # B only
    split: identification
    file: logs/B-H1-r1.csv
    units: {cmd_H1: unitless, thrust_n: N, battery_voltage_v: V, ...}
    metadata: {water_density_kgm3: <measured>, water_temperature_c: <measured>}
    synthetic: false
    clock_domain: PTP-rig
```

`synthetic` must be stated on the manifest and on every segment, and must be honest: a simulator
source, a `SIM` clock or a `SIMULATOR_TRUTH` reference with `synthetic: false` is rejected, and
synthetic logs must name a `SYNTHETIC_ONLY` source. Synthetic logs can only ever produce a
`SYNTHETIC_TOOLING_CHECK` report, never IDENTIFIED RobotConfig values.

## After delivery

1. Intake: `--intake-only` (above). Fix every `FAIL` and re-deliver.
2. Identification and held-out validation on the protocol logs:

   ```
   python -m uv run python -m conrad.robotics.hardware.identification --manifest <delivery>/manifest.yaml \
       --protocol --robot-config <measured RobotConfig> --out artifacts/identification/report.json
   ```

   Rigid mass and inertia are read from the RobotConfig; if they are still OPEN the run stops.
3. Only a report with status `IDENTIFIED_AND_VALIDATED` against a declared envelope may be turned into
   IDENTIFIED characterization records.

The software side of this protocol was rehearsed on the Python simulation kernel as experiment
`ID-REHEARSAL-E001` (SYNTHETIC_TOOLING_CHECK); see `docs/PHYSICAL_INTEGRATION.md`.
