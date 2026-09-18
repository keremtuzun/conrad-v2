# Physical integration: what the hardware owner hands over, and what runs afterwards

Hardware owner: Burak. Software owner: Kerem. Spec: ch22 "Ownership Classification and Handoff". Every physical
number stays `OPEN` in `configs/robot/physical_template.yaml` until it arrives with provenance. The software
never invents, defaults or back-fills a physical value.

## 1. Characterization handoff (mass properties, thrusters, sensors, power, comms, health)

Deliver one folder:

```
handoff/2026-10/
  handoff.yaml            (or handoff.csv, same fields)
  raw/                    every raw log a MEASURED value is derived from (any text/binary format)
```

`handoff.yaml`:

```yaml
bundle_id: burak-2026-10
robot_config_name: physical
records:
  - record_id: m1                      # unique inside the bundle
    category: mass                     # geometry|mass|inertia|com|cob|displaced_volume|hydrodynamics|thrusters|
                                       # sensors|communications|power|compute|safety|health
    target: mass_kg                    # RobotConfig path, see below
    value: 11480                       # scalar, list for vectors, string for enums
    units: g                           # converted exactly to the target's SI units, or refused
    frame_id: null                     # REQUIRED as ROBOT for body vectors (CoM, CoB, thruster position/direction, mounts)
    timestamp: {time_ns: 1791000000000000000, clock_domain: UTC}   # REQUIRED for MEASURED
    uncertainty_1sigma: 5              # same units as value
    valid_range: [11400, 11560]        # optional
    source: MEASURED                   # MEASURED|IDENTIFIED|LITERATURE_PRIOR|ENGINEERING_ESTIMATE|SYNTHETIC_ONLY|OPEN
    method: "hanging scale, 3 readings"
    provenance: "lab notebook p.14"    # REQUIRED (citation) for LITERATURE_PRIOR
    raw_log: raw/scale_2026-10-02.csv  # REQUIRED for MEASURED; sha256 computed at ingestion
    operator: burak
```

Target paths: `mass_kg`, `displaced_volume_m3`, `center_of_mass_body_m`, `center_of_buoyancy_body_m`,
`inertia_diag_kgm2`, `dimensions_m`, `linear_drag`, `quadratic_drag`, `added_mass_diag`,
`thrusters[<id>].<field>`, `sensors[<name>].<field>`, `communications[<link>].<field>`,
`battery.<field>`, `compute.<field>`, `safety.<field>`, and `health.<device>` (kept as evidence, not config).

Thrusters are created only when **all eight** fields are delivered for an id: `position_body_m`, `direction_body`,
`max_forward_thrust_n`, `max_reverse_thrust_n`, `deadzone_command`, `time_constant_s`, `latency_s`,
`thrust_coefficient` (k in T = k u|u|). The template's thruster layout is intentionally empty.

CSV alternative: header `record_id,category,target,value,units,frame_id,time_ns,clock_domain,uncertainty_1sigma,range_min,range_max,source,method,provenance,raw_log,operator`,
with vectors written as `a;b;c`.

Units accepted with exact conversion: kg g lb, m cm mm in, m^3 L mL cm^3, kg*m^2 g*mm^2 g*cm^2, N kgf lbf,
s ms us, Hz kHz, m/s cm/s knot, bps bit/s kbps Mbps, J Wh kWh, V mV, A mA, W, C K, rad deg,
unitless fraction %, B MB GB MiB GiB. `mAh`/`Ah` are refused: give J or Wh.

**Command afterwards** (writes a new versioned RobotConfig plus a change report and never overwrites):

```
python -m uv run python -m conrad.robotics.hardware.characterization --base configs/robot/physical_template.yaml --handoff handoff/2026-10/handoff.yaml --version 0.3.0 --out configs/robot/physical_v0_3_0.yaml --report artifacts/characterization/change_0_3_0.json
```

The report lists every change (old/new value and source), what is still `OPEN`, and what is not yet
MEASURED/IDENTIFIED. The physical lane requires the latter list to be empty.

## 2. Identification logs (experiments A..F, ch22)

Deliver `logs/manifest.yaml` plus one CSV per trajectory. The first CSV column is `time_s`, strictly
increasing. **Identification and validation trajectories must be different runs**: the tooling refuses
overlapping ids or byte-identical logs, and validating on identification data raises.

```yaml
split_id: pool-2026-10-a
segments:
  - {trajectory_id: B-T1-steps-1, kind: B_THRUSTER_STEP, split: identification, file: B-T1-steps-1.csv,
     units: {command: unitless, thrust_n: N}, synthetic: false}
  - {trajectory_id: C-surge-01, kind: C_SURGE_ACCELERATION, split: identification, file: C-surge-01.csv,
     units: {force: N, velocity: m/s}, synthetic: false}
  - {trajectory_id: C-surge-07, kind: C_SURGE_ACCELERATION, split: validation, file: C-surge-07.csv,
     units: {force: N, velocity: m/s}, synthetic: false}
  - {trajectory_id: A-static-1, kind: A_STATIC_TRIM, split: identification, file: A-static-1.csv,
     units: {net_weight_n: N}, metadata: {water_density_kgm3: 998.2}, synthetic: false}
```

| Kind | Columns (SI) | Protocol |
|---|---|---|
| `A_STATIC_TRIM` | `net_weight_n` (submerged scale, +down) and `metadata.water_density_kgm3`; or `applied_pitch_moment_nm`, `pitch_rad` | still water; tilt tests with known moments |
| `B_THRUSTER_STEP` | `command`, `thrust_n` | alternate zero and level plateaus; forward **and** reverse levels; include levels near the deadzone |
| `C_SURGE_ACCELERATION`, `D_SWAY`, `D_HEAVE` | `force` (N, along the axis), `velocity` (m/s) | force plateaus with coast phases |
| `E_YAW_ROTATION` | `force` (N*m), `velocity` (rad/s) | torque plateaus with coast phases |
| `F_STATION_KEEPING` | `hold_force_n` | steady hold in a current |

`synthetic` must be stated for every segment. A report built from any synthetic segment is
`SYNTHETIC_TOOLING_CHECK` and can never become IDENTIFIED parameters.

**Command afterwards:**

```
python -m uv run python -m conrad.robotics.hardware.identification --manifest logs/manifest.yaml --mass-kg <MEASURED mass> --yaw-inertia <MEASURED Izz> --envelope-rmse <agreed limit> --out artifacts/identification/report_2026-10.json
```

Each fitted parameter comes with a 1-sigma from the Jacobian covariance and a 95 % interval. The deadzone is
bracketed between step levels rather than fitted, because a hard deadzone is not smooth. Promotion to
RobotConfig goes through `to_characterization_records(report, path, mappings)` followed by the step 1 merge
with `source: IDENTIFIED`. It is refused for synthetic or failed-validation reports.

## 3. Readiness and autonomy gates

`conrad.robotics.hardware.characterization.GateLedger("artifacts/gates/ledger.json")` is a persisted,
append-only ledger:

- **Readiness ladder:** R0 pure simulation, R1 SIL, R2 HIL, R3 dry bench, R4 controlled water, R5 representative
  environment, R6 target envelope.
- **Autonomy activation ladder:** manual, attitude/depth hold, waypoint, station keeping, pipeline following,
  sensing, 2S, 2T, active inspection, Model 1, BAAC.

No gate can skip its predecessor. Every pass names an evidence artifact (report, video, log), whose sha256 is
recorded and re-verified whenever the ledger loads. Autonomy stages also need readiness R4, and manual needs R3.
That mapping is an assumption, configurable through `min_readiness`. Revoking a gate revokes everything after
it.

## 4. Status today

| Item | Status |
|---|---|
| characterization ingestion, merge, unit conversion, gate ledger | implemented, tested on labelled synthetic fixtures |
| identification tooling A..F with uncertainty and split enforcement | implemented, recovers known parameters from synthetic logs |
| real characterization data | **BLOCKED_EXTERNAL** (hardware owner) |
| real identification | **BLOCKED_EXTERNAL** (hardware owner, controlled-water access) |
| target-HIL on the onboard computer | **BLOCKED_EXTERNAL** (onboard computer) |
| Unity execution | **BLOCKED_EXTERNAL** (`EXT-UNITY-01`) |
