# Digital Twin legacy repository audit

Status: READ-ONLY archaeological audit. No code, asset, dataset or weight was copied from the legacy
repository into Conrad V2. The legacy repository was not modified, formatted, installed or executed.

| Field | Value |
|---|---|
| Audit date | 2026-09-18 |
| Legacy repository | `C:\Users\Kerem\OneDrive\Documents\ChatGPT\conrad` |
| HEAD commit | `6bf01ee6ef189145fbc57b65aaed2de46ccea45b` |
| Branch | `codex/rov-digital-twin` |
| HEAD subject | `feat: freeze Model 2 S1 learned baseline evaluation` (2026-08-27) |
| History | 47 commits, single author (Kerem Tuzun), first commit 2026-08-21 |
| Tracked files | 425 |
| Second worktree (NOT audited) | `C:\Users\Kerem\.codex\worktrees\8ada\conrad` at `69d7f2e`, branch `codex/model2-s1-temporal-gru` |
| `git status --porcelain` before audit | 0 lines |
| Distribution name | `rov-digital-twin` 0.1.0 (setuptools, `requires-python >=3.10`) |
| Importable packages | `rov_dt` (21 files) and `oceansense` (48 files), about 8.4k lines of Python under `src/` |
| Licence file | MIT, "Copyright (c) 2026 ROV Digital Twin Team" |

Note on naming: the brief calls the legacy package `rov_digital_twin`. No such import package exists.
The distribution is `rov-digital-twin`; the import packages are `rov_dt` and `oceansense`.

## 1. Method

Read-only tools only (file reads, glob, grep, `git log`, `git rev-parse`, `git status`, `git ls-files`,
`git worktree list`, `git grep`). No legacy test was run and the package was not installed, so every
statement about test quality below is from reading the test sources, not from executing them.

Every subsystem is judged against the eight salvage conditions. Default posture is DO NOT COPY.

1. Useful to V2.
2. No frozen-boundary violation (Twins = truth, Model 2 = belief, Model 1 = decision, Twin1 scrapped).
3. Movable without a runtime dependency on the legacy repo.
4. Coverable by V2 tests.
5. Interfaces convertible to V2 versioned contracts.
6. No hidden truth leakage.
7. Does not resurrect obsolete architecture.
8. Licensing and provenance allow reuse.

The V2 clean-room team has already implemented frames math, persistence, the command gateway,
`RobotConfig`, 6-DOF dynamics, PID and allocation, and an EKF from the specification. A legacy item is
therefore recommended for porting only if it carries validated value beyond a from-spec reimplementation.

## 2. Legacy architecture overview

The legacy README describes "four pillars". Their meaning differs from V2, which matters for every
mapping below.

| Legacy term | What it actually is | V2 meaning of the same word |
|---|---|---|
| Model 1 | Underwater image classifier/detector (EfficientNet-B0, optional Ultralytics YOLO) plus a rule ladder. No approved weights exist; freeze is documented as blocked. | Decision world. Completely different thing. |
| Twin 1 / Navigation digital twin | Unity 6 project under `unity/` (rigidbody ROV, hydrodynamics, sensors, ML-Agents PPO, UDP/ROS bridge) plus a pure-Python kinematic replay `oceansense/navigation_twin.py`. | Does not exist. Scrapped. |
| Failure Twin | Two unrelated things: a seeded 2D PIL image-pair generator (`oceansense/failure_twin.py`) and a graph degradation simulator "Failure Twin v0 / Twin 2" (`oceansense/model2/simulator.py`). | Closest V2 concept is `conrad.twins.twin2s`. |
| Model 2 | A hand-weighted "structural-temporal reasoner" (`model2_reasoning.py`) plus dataset release tooling and two trivial baselines. No trained network on this branch. | Belief world. |

Data flow as built:

```text
Unity (truth pose, truth depth) --hand-built JSON over UDP--> ros2 bridge (std_msgs/String)
   --> rov_dt telemetry_contract --> softmax weak-point classifier --> SafetyDecisionAgent
   --> intent_gateway (string allow-list) --> /rov/high_level_command --> Unity MissionController
Unity ML-Agents policy --> ROVRLAgent.OnActionReceived --> Vehicle.SetThrusterCommand (direct)
image --> oceansense.perception --> Classification(confidence: float) --> DecisionAgent thresholds
```

Cross-cutting observations:

- **Single scalar confidence everywhere.** `rov_dt/decision.py` (`confidence_gate = 0.72`),
  `oceansense/schemas.py` (`Classification.confidence`, `Detection.confidence`,
  `InspectionDomain.confidence`), `oceansense/decision.py` (thresholds 0.50 / 0.65 / 0.70 / 0.75),
  `oceansense/mission_decision.py` (`min(confidence_values)`, gate 0.60),
  `oceansense/model2/schemas.py` (`InspectionObservation.confidence`). Incompatible with the V2
  4-channel `Uncertainty`.
- **No provenance DAG.** Provenance is a free string (`PhysicsResidual.provenance = "derived"`,
  `field_status` map in telemetry). Nothing links an output to its inputs.
- **Messages are dataclasses or hand-formatted JSON.** A `schema_version` string exists on telemetry,
  but there is no envelope, no clock domain, float-second timestamps, and no frame names on any
  spatial value. `ros2` transports everything as `std_msgs/String`.
- **No database schema exists.** `git grep` for sqlite, sqlalchemy, postgres and `CREATE TABLE` returns
  nothing. Persistence is ad hoc JSON, CSV and `.npy`.
- **No control or estimation math exists.** `git grep` finds no PID, Kalman/EKF, allocation matrix or
  pseudo-inverse anywhere. The only "controller" is a dot-product heuristic in
  `ROVRLAgent.ComputeGuidanceCommands`. There is no frames or geometry library; the only geometry
  helper is a 3-tuple Euclidean `_distance` in `navigation_twin.py`.
- **Physical constants are invented.** No file under `src/`, `unity/Assets/ROVDigitalTwin/Scripts`,
  `scripts/`, `config/`, `configs/` or `ros2/` cites Fossen, SNAME, a BlueROV or T200 datasheet, a DOI
  or a measurement. `config/domain_randomization.yaml` says so itself: "These are simulator
  hypotheses, not measured ocean constants." `vehicle_profiles/` holds only a README; no identified
  vehicle profile was ever produced.

## 3. Subsystem findings: Unity project (legacy "Twin 1")

### 3.1 Unity Twin 1 project shell

- **Legacy path:** `unity/` (ProjectSettings, Packages, `Assets/ROVDigitalTwin/Scenes/OceanSenseDemo.unity`,
  `Prefabs/OceanSenseROV.prefab`, `Editor/CompleteProjectBuilder.cs`, `MissionController.cs`,
  `DutyManager.cs`, `DutyDefinition.cs`, `ROVVehicle.cs`, `FollowCamera.cs`).
- **Purpose:** The robot/navigation twin. `docs/TWIN1_STATUS_REPORT.md` names it "Twin 1" outright; the
  PlayMode suite is `Twin1RuntimePlayModeTests.cs`.
- **V2 subsystem:** None. Twin1 is scrapped. The nearest V2 seam is `conrad.sim.unity` /
  `conrad.adapters.unity`, which are adapters behind V2 contracts, not a twin.
- **Compatibility:** Conflicts. One Unity scene is the truth world, the robot controller, the sensor model
  and the decision client at once. `MonoBehaviour` components reach each other by direct reference.
- **Contamination:** `com.unity.ml-agents` 4.1.0; `com.unity.robotics.ros-tcp-connector` pulled from a git
  URL at v0.7.0; `UnityEngine.Random` (process-global, unseeded) in the scene builder and sensors;
  `Physics.gravity` global.
- **Tests:** 9 PlayMode smoke tests (scene has objects, values stay finite, emergency stop zeros thrusters,
  UDP rejects raw actuation). Smoke level only; none checks physics against data.
- **Licence/provenance:** Own work, MIT. Packages are Apache-2.0 (ML-Agents, ROS-TCP-Connector) and the
  Unity Companion Licence; none is vendored.
- **Classification:** `REJECT_ARCHITECTURAL_CONFLICT`.
- **Reason:** Fails conditions 2 and 7 by definition. Twin1 does not exist in V2.

### 3.2 `Hydrodynamics6Dof.cs` (hydrodynamics calculation)

- **Purpose:** Buoyancy, linear plus quadratic drag, added mass and restoring torque on a Unity `Rigidbody`.
- **V2 subsystem:** `conrad.sim.kernel` 6-DOF dynamics (already implemented from the spec).
- **What was seen:** Defaults `FluidDensity = 1025`, `DisplacedVolume = 0.02`, `ViscousLinearDrag = (2, 3, 4)`,
  `LinearDrag = (18, 22, 28)` (despite the name, this is the quadratic coefficient), `AngularDrag = (4, 4, 6)`,
  `AddedMass = (5, 8, 10)`, `RestoringTorque = 25`. No source, no units, no fit. Added mass is applied
  as a force from a finite-differenced velocity (`(localVelocity - previousLocalVelocity) / dt`), which is
  numerically fragile and is not a mass-matrix formulation. Buoyancy uses a linear `SubmergedFraction`
  ramp. The same class injects a non-physical `ApplyAttitudeSafetyEnvelope` recovery torque
  (`SafetyRecoveryTorque = 180`) and exposes `PolicyCommandAuthority01`, so control logic lives inside
  the physics model.
- **Contamination:** Unity `Rigidbody`, `Physics.gravity`, `Time.fixedDeltaTime`, a direct `WaterCurrentField` reference.
- **Tests:** `HydrodynamicsTests.cs` only checks that quadratic drag opposes motion and is zero at rest.
- **Classification:** `OBSOLETE`.
- **Reason:** The coefficients are made up, not sourced or validated (fails condition 1, and V2 rule 10
  if they were presented as facts). The formulation is weaker than the V2 from-spec 6-DOF model, and
  this is a Twin1 component (fails 7). Nothing here has validated value.

### 3.3 `Thruster.cs` (thruster model)

- **Purpose:** Maps commands to a thrust curve with dead zone, exponent, reverse ratio, voltage, density,
  thermal derating and fouling factors.
- **V2 subsystem:** `conrad.robotics.allocation` / `conrad.sim.kernel` actuator model.
- **What was seen:** `MaxForceNewtons = 50`, `ThrustCurveExponent = 1.65`, `ReverseThrustRatio = 0.82`,
  `CommandDeadZone = 0.06`, `ResponseTimeSeconds = 0.14`, `SupplyVoltageV = 48`, fouling floor `0.55`,
  voltage clamp `0.45..1.1`. No datasheet or bench reference for any value.
- **Tests:** One EditMode test checks the dead zone, reverse asymmetry and the legacy linear fallback.
  It tests the shape of a made-up curve, not agreement with a real thruster.
- **Classification:** `OBSOLETE`.
- **Reason:** Nominal constants presented as a model with no source. A V2 thruster curve must come from a
  datasheet or be labelled `ENGINEERING_ESTIMATE`; nothing here helps with that.

### 3.4 `WaterCurrentField.cs`, `UnderwaterEnvironment.cs`, `DomainRandomization.cs`, `FaultInjectionController.cs`

- **Purpose:** Current plus wave orbital velocity field; turbidity, visibility and acoustic-quality
  heuristics; difficulty profiles; fault injection.
- **V2 subsystem:** `conrad.sim.scenarios`, `conrad.twins.twin2e`.
- **What was seen:** `WaveComponent` uses the textbook deep-water relations `k = omega^2 / g`, decay
  `exp(-k d)` and orbital speed `0.5 H omega`. That part is standard Airy linear wave theory, not legacy
  IP. Everything else is made up: Perlin turbulence, `OpticalVisibilityMeters = 32 / (1 + 0.75 NTU + 5 c)`,
  `AcousticQuality01 = 1 - 0.012 NTU - 0.002 SSC - 0.12 c`, base current `(0.12, 0, 0.04)`. The profile YAML
  labels itself as hypotheses.
- **Contamination:** `Mathf.PerlinNoise`, `RenderSettings` globals, `Time.time`.
- **Tests:** `WaveOrbitalVelocityDecaysWithDepth` and `ContaminationReducesVisibilityAndAcousticQuality`
  (monotonicity only).
- **Classification:** `OBSOLETE`.
- **Reason:** A Twin1 component. If V2 needs wave orbital velocity, implement it from a cited
  oceanography reference with named frames and SI units, not from this file.

### 3.5 Unity sensor models

- **Legacy path:** `DepthSensor.cs`, `ImuSensor.cs`, `DvlSensor.cs`, `ForwardSonarSensor.cs`,
  `SimulatedPowerSensor.cs`, `SensorNoise.cs`, `ROVCameraCapture.cs`, `SyntheticCaptureController.cs`.
- **Purpose:** Noisy sensor emulation and synthetic frame capture with a provenance sidecar.
- **V2 subsystem:** `conrad.sim.kernel` sensor models, `conrad.schemas.observation`.
- **What was seen:** `SensorNoise.Gaussian` is a Box-Muller transform over `UnityEngine.Random.value`, a
  process-global unseeded generator. That breaks V2's injected-generator rule. DVL quality is
  `1 - 0.35 * distance / max`. Sensors read `body.linearVelocity` and `transform.position` directly.
  Timestamps are `Time.time` floats. No frame is named.
- **Tests:** PlayMode checks that values stay finite and that capture writes a PNG and a sidecar.
- **Classification:** `OBSOLETE`.
- **Reason:** Global RNG state, made-up noise figures, no frames and no ns timebase. The one good idea
  (label synthetic captures) is already a V2 rule.

### 3.6 `ROVRLAgent.cs`, ONNX policies and PPO configs

- **Legacy path:** `Scripts/ROVRLAgent.cs`, `Models/OceanSenseROV_Bootstrap.onnx` (96 KB),
  `Models/OceanSenseROV_OpenSea_Experimental.onnx` (340 KB), `config/unity_ppo*.yaml`,
  `configs/rov_ppo.yaml`, `artifacts/training/*.json`, `docs/rl_policy_model_card.md`.
- **Purpose:** ML-Agents residual PPO waypoint policy.
- **V2 subsystem:** None allowed. The closest is `conrad.robotics.navigation`, which must sit behind the gateway.
- **What was seen:** `OnActionReceived` computes `guidance + residual` and calls
  `Vehicle.SetThrusterCommand(index, command)` directly. `CollectObservations` feeds the policy
  `DutyManager.TargetPosition - transform.position`, `body.linearVelocity`, `body.angularVelocity` and
  `transform.rotation`, which are simulator truth, alongside sensor values. Reward logging uses the
  `Academy.Instance.StatsRecorder` process singleton. The legacy README itself calls the ONNX files a
  "legacy-dynamics baseline" that is no longer valid for the current simulator.
- **Tests:** `AgentContractSizesRemainStable` and a PlayMode check that the behaviour is `HeuristicOnly`.
- **Licence/provenance:** Weights self-trained on the made-up simulator. The rights are clear, but the weights have no value.
- **Classification:** `REJECT_ARCHITECTURAL_CONFLICT`.
- **Reason:** A direct neural-output-to-actuator path (V2 rule 8) that also takes truth as an
  inference feature (rules 2 and 3). Fails conditions 2, 6 and 7.

### 3.7 `TelemetryUdpBridge.cs`, `OceanSenseApiClient.cs`, `OceanSenseDashboard.cs`, `VehicleProfileLoader.cs`

- **Purpose:** Sends telemetry out and receives high-level intents over UDP; calls the Python API over
  HTTP; draws an IMGUI HUD; applies vehicle profiles.
- **V2 subsystem:** `conrad.adapters.unity`, `conrad.schemas.envelope`.
- **What was seen:** Telemetry is one hand-built `FormattableString` JSON literal with
  `"schema_version":"1.0.0"`. It ships `position_m` and `velocity_mps` straight from `transform.position`
  and the rigidbody. It also computes `imu_depth_disagreement_m` as `|Depth.DepthMeters - trueDepth|`,
  where `trueDepth` is the truth transform depth. That truth-derived value reaches the downstream
  classifier as if it were a sensor feature. Inbound command filtering is substring matching
  (`lower.Contains("thruster")`, `"motor"`, `"pwm"`, `"force"`, `"voltage"`) plus an intent allow-list.
- **Tests:** `Twin1UdpLifecycleRejectsRawActuationAndRebindsCleanly`.
- **Classification:** `REJECT_ARCHITECTURAL_CONFLICT`.
- **Reason:** Hidden truth leakage in the wire format (fails 6), no envelope or provenance (fails 5).
  String matching is not a command gateway.

### 3.8 Unity art assets

- **Legacy path:** `Materials/{Pipeline,RobotMetal,SafetyYellow,Sand,Water}.mat`, the prefab and the scene.
- **What was seen:** `CompleteProjectBuilder.cs` builds everything from `GameObject.CreatePrimitive`
  (cubes, cylinders, spheres, capsules). There is no mesh, texture, FBX/OBJ, shader or audio asset. The
  five materials are flat colours.
- **Licence/provenance:** Own work, clean.
- **Classification:** `OBSOLETE`.
- **Reason:** The provenance is fine but there is no art to salvage; the primitives can be rebuilt in minutes.

### 3.9 Unity tests

- **Legacy path:** `Tests/Editor/HydrodynamicsTests.cs` (8 tests), `Tests/PlayMode/Twin1RuntimePlayModeTests.cs` (9 tests).
- **Classification:** `OBSOLETE`.
- **Reason:** Tied to Twin1 components, and they check the shapes of made-up curves. No fixture data.

## 4. Subsystem findings: ROS 2 and the `rov_dt` package

### 4.1 ROS 2 bridge

- **Legacy path:** `ros2/rov_dt_bridge/rov_dt_bridge/{diagnostic_node,intent_gateway_node,unity_udp_bridge}.py`, `src/rov_dt/ros_support.py`.
- **Purpose:** Relays Unity UDP telemetry to `/rov/telemetry_json`, runs the weak-point classifier and
  publishes `/rov/diagnostic_decision`, then maps decisions to `/rov/high_level_command`.
- **V2 subsystem:** `conrad.communication` / `conrad.adapters` (ROS must stay out of core packages).
- **Contamination:** `rclpy`, `std_msgs.msg.String` for every topic (untyped JSON inside a string),
  raw `socket` UDP, direct imports of `rov_dt.decision`, `rov_dt.model` and `rov_dt.telemetry_contract`.
  `ros_support.resolve_model_path` reads the `ROV_DT_MODEL_PATH` environment variable.
- **Tests:** None for the nodes. `ros_support` is exercised indirectly.
- **Classification:** `REJECT_ARCHITECTURAL_CONFLICT`.
- **Reason:** Unversioned `String` messages carrying truth-derived Unity telemetry into a decision
  path. Fails 5 and 6, and V2 rule 9 forbids ROS in core.

### 4.2 Telemetry contract and canonical real-data record

- **Legacy path:** `src/rov_dt/telemetry_contract.py` (`SCHEMA_VERSION = "1.0.0"`), `src/rov_dt/real_data.py`
  (`SCHEMA_VERSION = "2.0.0"`, `DATA_SOURCES = {"simulated","pool","lake","sheltered_water","open_sea"}`),
  `src/rov_dt/schema.py`, `config/telemetry_schema_v1.json`, `config/telemetry_schema_v2.json`.
- **Purpose:** Validates Unity JSON; defines a per-field `field_status` of
  `simulated | measured | derived | unavailable`; allow-lists high-level intents.
- **V2 subsystem:** `conrad.schemas.observation`, `conrad.schemas.envelope`, `conrad.data.simreal_ledger`.
- **Compatibility:** Partial at the concept level only. The per-field status is the same idea as V2
  `SourceKind` and knowledge status, but the legacy record has float-second timestamps, no frame names,
  no clock domain, no envelope and flat `position_m` / `velocity_mps` fields that come from truth.
- **Tests:** `tests/test_telemetry_contract.py` (8 tests), `tests/integration/test_real_data_loaders.py` (3 tests).
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** V2 contracts are already frozen and are strictly richer. The only takeaway is a check
  that V2 keeps a per-field measured/simulated/derived/unavailable distinction; no code is needed.

### 4.3 Telemetry weak-point classifier (legacy "diagnostic model")

- **Legacy path:** `src/rov_dt/{model,dataset,training,ensemble,cli}.py`, `models/weakpoint_v2.json`,
  `artifacts/weakpoint_model.json`, `artifacts/training/weakpoint_v2_metrics.json`, `scripts/train_telemetry_ensemble.py`.
- **Purpose:** A pure-Python softmax classifier over 14 features (`depth_m` ... `temperature_c`) with 5 labels
  (`buoyancy_imbalance`, `hydrodynamic_drag`, `nominal`, `sensor_drift`, `thruster_degradation`).
- **What was seen:** `dataset.generate_dataset(rows=4000, seed=42)` draws every sample from `random.Random`
  with made-up fault signatures, for example `current += rng.uniform(5.0, 13.0)`,
  `speed *= rng.uniform(0.45, 0.78)`, `disagreement = rng.uniform(0.65, 2.8)`. The classifier learns these
  hand-written rules back, so any accuracy it reports is circular. The committed weights were trained
  on that synthetic data.
- **V2 subsystem:** None. V2 fault reasoning belongs to Model 2 belief, with 4-channel uncertainty.
- **Tests:** `tests/test_pipeline.py` (3), `tests/unit/test_ensemble_runtime_vision.py` (4). They test
  that the pipeline runs, not that it is right.
- **Classification:** `OBSOLETE`.
- **Reason:** Synthetic labels encode the generator's own rules, and the output is a single probability.
  The weights have no transferable value (fails condition 1).

### 4.4 `rov_dt` decision stack

- **Legacy path:** `src/rov_dt/{decision,advisor,uncertainty,runtime_monitor,health_monitor,intent_gateway}.py`.
- **Purpose:** `SafetyDecisionAgent` (`confidence_gate = 0.72`) turns classifier output into actions such as
  `abort_and_surface`; `intent_gateway.decision_to_simulation_intent` maps actions to intents;
  `runtime_monitor.authorize_mode` gates SHADOW versus AUTONOMOUS_HIGH_LEVEL.
- **V2 subsystem:** `conrad.decision` (Model 1) and `conrad.runtime.command_gateway`, both already built from the spec.
- **What was seen:** The decision is thresholds on one scalar. `uncertainty.assess_uncertainty` defaults
  `maximum_probability_threshold = 0.55`, `entropy_threshold = 0.72`, `ood_threshold = 4.0`,
  `disagreement_threshold = 0.02` with no calibration source. `intent_gateway._contains_raw_actuator_field`
  rejects any key that contains the substrings `thruster`, `motor`, `pwm`, `force` or `voltage`. That is
  a denylist, not a typed gateway.
- **Tests:** `tests/safety/test_no_raw_actuator_authority.py` (2 tests), `tests/unit/test_reliability_core.py` (6 tests).
- **Classification:** `REJECT_ARCHITECTURAL_CONFLICT`.
- **Reason:** This is the old Model 1 decision logic: single scalar confidence (fails V2 rule 5) and a
  denylist gateway parallel to the frozen V2 gateway (fails 2 and 7). Its intent is already covered by
  V2 rule 8 and the V2 gateway.

### 4.5 Calibration, reliability and data-quality utilities

- **Legacy path:** `src/rov_dt/calibration.py` (`apply_temperature`, `negative_log_likelihood`, `brier_score`,
  `TemperatureScaler`), `src/rov_dt/reliability.py`, `src/rov_dt/data_quality.py` (`FieldQuality`,
  `DataQualityMonitor`), `src/oceansense/evaluation.py` (`classification_metrics`).
- **Purpose:** Textbook calibration metrics and per-field staleness/range checks.
- **V2 subsystem:** `conrad.evaluation`, `conrad.core` calibration metadata.
- **Contamination:** None serious. Pure Python lists, no global state.
- **Tests:** `test_reliability_core.py` covers some of it.
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** NLL, Brier score and temperature scaling are standard formulas that V2 should write
  against numpy/torch in a few lines, with V2 tests. Copying pure-Python list code brings no validated
  value (fails condition 1 on marginal value).

### 4.6 Physics residuals

- **Legacy path:** `src/rov_dt/physics_residuals.py` (`PhysicsResidual`, `compute_physics_residuals`), `src/rov_dt/temporal_model.py`.
- **Purpose:** Observed-minus-expected residuals (acceleration, velocity, thrust response, pressure depth,
  DVL versus IMU, electrical load), each with a `valid` flag and a `sensor_sources` tuple, and no
  substitution of missing measurements.
- **What was seen:** The pressure-to-depth relation `(p - 101.325) * 1000 / (rho * 9.80665)` is standard
  hydrostatics. The "expected" models are placeholders: `thruster_acceleration_gain_mps2` default `1.0`,
  `command_velocity_gain_mps` default `1.2`, `current_per_command_a` default `35.0`, and the normalisation
  scales (`0.5`, `0.2`, `0.25`, `0.3`, `5.0`) are all without a source.
- **V2 subsystem:** Model 2 evidence features (`conrad.domains` / `conrad.core`), health monitoring in `conrad.robotics.safety`.
- **Tests:** None directly. It feeds `temporal_model.py`, which is only exercised through other tests.
- **Classification:** `REIMPLEMENT_FROM_CONCEPT`.
- **Reason:** The pattern is worth keeping: residuals with explicit validity, sensor lineage and no
  imputation. It maps onto V2 provenance and the UO channel. The numbers are not worth keeping. Rebuild
  it on V2 `RobotConfig` and frames with sourced or `ENGINEERING_ESTIMATE`-labelled gains.

## 5. Subsystem findings: the `oceansense` package

### 5.1 Legacy Model 1: image perception and rule-based decision

- **Legacy path:** `src/oceansense/{perception,schemas,decision,mission_decision,scoring,anomaly,taxonomy,vision_uncertainty,underwater_augmentation,model1_baseline_v2}.py`,
  `scripts/train_{classifier,domain_classifier,condition_classifier,detector}.py`, `scripts/evaluate_multidomain.py`,
  `configs/model1_baseline_v2.yaml`, `config/labels.yaml`, `outputs/model1_audit/*`.
- **Purpose:** EfficientNet-B0 domain and condition classifiers, an optional Ultralytics detector, and a
  threshold ladder that maps labels to `continue_survey`, `request_human_review` and so on.
- **V2 subsystem:** None under the same name. Legacy "Model 1" is perception, while V2 Model 1 is
  decision. Its perception role would map to V2 evidence producers under `conrad.domains`.
- **What was seen:** `Classification.confidence`, `Detection.confidence` and `InspectionDomain.confidence`
  are scalar floats. `decision.DecisionAgent` uses literal thresholds 0.50, 0.65, 0.70 and 0.75, and turns
  the scalar into `"high"`, `"medium"` or `"low"` strings. `perception.PerceptionService` multiplies
  confidence by `0.8` when context disagrees. `docs/MODEL1_FREEZE_REPORT.md` and
  `outputs/model1_audit/evaluation_blocker.json` record that no approved checkpoint exists.
- **Contamination:** `torch`, `torchvision`, `ultralytics` (AGPL-3.0, if installed), `PIL`.
- **Tests:** `tests/test_oceansense.py` (8), `tests/unit/test_model1_baseline_v2_gate.py` (3), fixture classifiers only.
- **Classification:** `REJECT_ARCHITECTURAL_CONFLICT`.
- **Reason:** This is the old Model 1. It uses single-scalar confidence semantics, the name clashes with
  the V2 decision world, and no weights exist. The optional Ultralytics dependency is AGPL-3.0, which is
  another reason not to bring its integration code across.

### 5.2 OceanSense API, RAG explainer and knowledge base

- **Legacy path:** `src/oceansense/{api,rag}.py`, `src/oceansense/knowledge_base/*.md`, `knowledge/rov_operations.md`,
  `scripts/run_api.py`, `configs/llm_lora.yaml`, `scripts/train_specialized_llm.py`.
- **Purpose:** FastAPI endpoints around the perception and decision agents; a template "grounded
  explainer"; LoRA fine-tuning entry point for a domain LLM.
- **V2 subsystem:** None in the frozen boundaries.
- **Tests:** `tests/test_oceansense_api.py` (4).
- **Classification:** `OBSOLETE`.
- **Reason:** A service wrapper around the rejected Model 1, plus an LLM track that has no counterpart in V2.
  The knowledge-base notes are short (5 to 9 lines each) and unsourced.

### 5.3 Navigation twin (Python)

- **Legacy path:** `src/oceansense/{navigation_twin,navigation_contracts,digital_twin_demo}.py`,
  `configs/navigation_twin/demo_mission.json`, `scripts/run_navigation_twin_test.py`, `scripts/run_digital_twin_demo.py`,
  `scripts/replay_mission.py`, `docs/navigation_twin_log.md`.
- **Purpose:** `simulate_navigation` moves a point at `commanded_speed_mps` straight toward `target_xyz`,
  adds a constant current and emits `RobotState`, `SensorFrame` and `MissionEvent` dataclasses. Roll and
  pitch are hard-coded to `0.0` and battery drains linearly.
- **V2 subsystem:** None. It is Twin1's Python stand-in. Navigation in V2 is `conrad.robotics.navigation` behind the gateway.
- **Contamination:** `RobotState` carries true pose; `InspectionTarget.distance_to_target` is computed
  from true positions; float-second timestamps; no frames.
- **Tests:** Covered by `tests/unit/test_master_execution_guide.py` (12 tests across several demo items).
- **Classification:** `OBSOLETE`.
- **Reason:** A kinematic toy for Twin1. It fails conditions 2 and 7, and V2's 6-DOF kernel already replaces it.

### 5.4 Failure twin, image-pair generator

- **Legacy path:** `src/oceansense/failure_twin.py`, `config/failure_twin_mvp.json`, `configs/failure_twin/demo_scenario.json`,
  `scripts/run_failure_twin_batch.py`, `docs/failure_twin_spec.md`.
- **Purpose:** Draws a 2D structure with PIL, paints a defect and a `ground_truth_mask`, applies a "water"
  tint and blur, and writes metadata. `STRUCTURES`, `DEFECTS`, `SEVERITIES` are module-level sets.
- **V2 subsystem:** Would be a synthetic visual source for `conrad.twins.twin2s` if anything.
- **Tests:** Part of `test_master_execution_guide.py`.
- **Classification:** `OBSOLETE`.
- **Reason:** The legacy README itself calls it a "visual fixture for interface demos". Flat 2D cartoons
  are no use as V2 twin truth, and the split rule `_split_for(scenario_id)` is a hash that V2 already has
  in `conrad.data.splits`.

### 5.5 Failure Twin v0 / "Twin 2" graph degradation simulator

- **Legacy path:** `src/oceansense/model2/{simulator,schemas,dataset,visualization}.py`, `configs/model2/twin_v0.yaml`,
  `scripts/generate_model2_dataset.py`, `scripts/visualize_model2_scenario.py`, `docs/MODEL2_DATA_AND_TWIN2_REQUIREMENTS.md`.
- **Purpose:** `generate_structure` builds a small networkx graph; `evolve_states` updates 4 hidden defect
  channels by `intrinsic_degradation * override + environment_effect_weight * environment_level +
  neighbor_coupling * neighbor_mean + noise`, clipped; channel 5 is a weighted sum. `_simulate_observations`
  masks by `observation_coverage` and adds noise, false negatives (`*= 0.25`) and false positives
  (`uniform(0.35, 0.75)`).
- **V2 subsystem:** `conrad.twins.twin2s` (structural twin) and `conrad.sim.scenarios`.
- **Compatibility:** This is the closest thing in the repo to V2 thinking. It keeps truth in `states.npy`
  and the manifest literally says `"forbidden as model input"`. It uses seeded `np.random.default_rng`.
  But it still breaks V2 in three places. First, the observation carries a scalar
  `confidence = 1 - |severity - true_severity| + noise`. That is a direct function of truth, so the
  "belief-side" field leaks truth. Second, `timestamp` is an integer step index, not ns time with a
  physical `delta_t` (V2 rule 7). Third, observations use the simulator's `component_id` as the entity
  identity, which V2 rule 3 forbids.
- **Tests:** `tests/unit/test_model2_failure_twin_v0.py` (7). They check shapes, determinism and that
  `observations.json` has no state fields. They do not detect the confidence leak.
- **Licence/provenance:** Own work. The degradation coefficients in `twin_v0.yaml` are synthetic by
  design, and the legacy docs say so.
- **Classification:** `REIMPLEMENT_FROM_CONCEPT`.
- **Reason:** The concept (hidden per-component degradation on a structural graph with neighbour
  coupling, partial and noisy masked observations, false-positive and false-negative injection) is
  useful to V2 twin2s. The code is not: it fails condition 6 (the truth-derived confidence), condition 5
  (step-index time, simulator IDs, scalar confidence) and would need V2's frames, clocks, IdFactory and
  4-channel uncertainty. Rebuild from the concept and add a V2 test that checks no observation field is
  a function of truth.

### 5.6 Legacy Model 2 reasoner and D0/S1 release machinery

- **Legacy path:** `src/oceansense/model2_reasoning.py` (`StructuralTemporalReasoner`, `run_ablation`),
  `src/oceansense/model2/{baselines,baseline_config,evaluation,d0_release,s1_release,release_validator}.py`,
  `configs/model2/*.json`, `scripts/{run_model2_experiment,run_model2_d0_baselines,build_model2_d0_release,build_model2_s1_release,validate_model2_release}.py`,
  `data/model2/{d0_debug,s1_synthetic}/*`, `reports/model2/d0_baselines/*`, `docs/MODEL2_*.md`.
- **Purpose:** A hand-weighted reasoner that outputs one confidence
  (`min(1, (1 - mean_uncertainty) * (0.55 + 0.25 * persistence + viewpoint_confidence))`, unknown if
  below `0.45`), plus two baselines (`last_observation`, `simple_heuristic` carry-forward) and a
  release gate that hashes files, checks splits and checks `REQUIRED_NO_LEAKAGE_RULES`.
- **V2 subsystem:** `conrad.core` / `conrad.domains` (belief) and `conrad.evaluation`, `conrad.data.manifest`.
- **What was seen:** No trained network on this branch (the GRU work is on the unaudited second worktree
  branch `codex/model2-s1-temporal-gru`). The committed `.npy` datasets are generated from the 5.5 simulator,
  so they carry the same truth-derived confidence field.
- **Tests:** Strongest in the repo: `test_model2_release_validator.py` (12), `test_model2_baseline_config.py` (9),
  `test_model2_d0_baselines.py` (9), `test_model2_s1_release.py` (8). They test release hygiene
  (checksums, paths escaping the release directory, split usage, required artifacts), not model quality.
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** The reasoner is a scalar-confidence heuristic (fails V2 rule 5). The datasets inherit the
  leak described in 5.5, so they must not be used as V2 training or evaluation data. The release-gate
  checklist (hash every file, refuse paths outside the release, pin split usage, list the no-leakage
  rules) is a good checklist for V2 `conrad.data.manifest`, which already exists. Use the legacy test
  names as a reminder of cases to cover, and write the V2 tests from scratch.

### 5.7 Experiment and run manifests, dataset governance

- **Legacy path:** `src/oceansense/{experiment,governance,data}.py`, `experiments/run_manifest.example.json`,
  `scripts/{build_dataset_manifest,audit_dataset_licenses,detect_near_duplicates,split_image_dataset,download_approved_assets}.py`.
- **Purpose:** `RunManifest`, `PredictionRecord`, `DatasetManifestRecord`; `license_gate` with
  `ALLOWED_LICENSES = {"public domain","cc0","cc0-1.0","cc by 4.0","cc-by-4.0"}`; `sha256_file`.
- **V2 subsystem:** `conrad.persistence`, `conrad.data.manifest`, provenance DAG.
- **Tests:** `tests/test_governance_and_evaluation.py` (3).
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** V2 persistence and provenance are frozen and more complete. The licence allow-list idea
  is worth one line in V2 data policy; the code is not.

### 5.8 SeaClear review and labelling pipeline

- **Legacy path:** `src/oceansense/{seaclear,seaclear_review,seaclear_reviewer_packages,seaclear_submission_intake}.py`,
  `scripts/{build_seaclear_*,compare_seaclear_reviewer_submissions}.py`, `data/model1_baseline_v2/**`, `docs/SEACLEAR_*.md`.
- **Purpose:** Hash-inventories the SeaClear dataset and runs a two-reviewer labelling and adjudication workflow.
- **V2 subsystem:** Only `conrad.data` if V2 ever uses real imagery.
- **Tests:** 13 tests across the four `test_seaclear_*.py` files.
- **Licence/provenance:** SeaClear Marine Debris Dataset v1, 4TU.ResearchData, DOI
  `10.4121/4f1dff25-e157-4399-a5d4-478055461689.v1`, CC BY 4.0 (attribution required to Duras, Ilioudi,
  Wolf, Palunko, De Schutter). The images are not in git (`.gitignore` excludes `dataset/raw/` and the
  others); only manifests and queues are tracked. The legacy notes say it is debris data and not suitable
  for structural defects. SubPipe and CleanCam are recorded as "License scope unresolved" and were never acquired.
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** Tied to the rejected legacy Model 1 label schema. If V2 wants SeaClear, fetch it again
  from the DOI with its own manifest and attribution. The licence evidence notes are a useful starting
  point for that.

## 6. Subsystem findings: scripts, tools, configs, datasets, docs, CI

### 6.1 Vehicle system identification

- **Legacy path:** `scripts/identify_vehicle_parameters.py`, `vehicle_profiles/README.md`.
- **Purpose:** `identify()` fits `accel = a * command - b * v - c * v|v|` by one-step least squares
  (a hand-written `_solve_3x3` on normal equations, at least 12 rows) and fits battery internal resistance
  by linear regression. Added mass, buoyancy, centre of buoyancy and thruster lag stay `None` with
  `validity = false`, and the output says "Null parameters were not identifiable ... and were not invented."
- **V2 subsystem:** `conrad.robotics` `RobotConfig` parameter provenance, `conrad.data`.
- **What was seen:** It has never been run on real data. `vehicle_profiles/` holds only the README.
  The fit is 1-DOF longitudinal only, uses float-second timestamps and differentiates a speed magnitude.
  It uses `datetime.now` inside the output (non-deterministic).
- **Tests:** None.
- **Classification:** `REIMPLEMENT_FROM_CONCEPT`.
- **Reason:** The rule it encodes (identified parameters carry a validity flag; unidentifiable ones stay
  null, never hand-filled) matches V2 rule 10 and `SourceKind.OPEN`. The maths is a 20-line least-squares
  fit that V2 should write with `numpy.linalg.lstsq`, per axis, with residual statistics, against V2
  `RobotConfig`. No validated output exists to preserve.

### 6.2 Other scripts

- **Legacy path:** the remaining 37 `scripts/*.py` plus `scripts/run_unity_validation.ps1`.
- **Purpose:** CLIs for the subsystems above (training, evaluation, dataset preparation, report generation,
  Unity validation, frame extraction, annotation conversion).
- **Classification:** `OBSOLETE`.
- **Reason:** Thin entry points into rejected or obsolete modules. V2 has its own `conrad.cli`.

### 6.3 Tools

- **Legacy path:** `tools/build_project_doc.py`, `tools/render_pdf_pages.py`.
- **Purpose:** Build and render the Turkish project outline document (`docs/ROV_Digital_Twin_Project_Outline.docx`).
- **Classification:** `OBSOLETE`.
- **Reason:** Document tooling for a superseded project outline.

### 6.4 Configuration files

- **Legacy path:** `config/*` (15 files) and `configs/**` (9 files). Note the two parallel
  directories, `config/` and `configs/`, with overlapping roles.
- **Purpose:** PPO hyperparameters, domain-randomisation ranges, label taxonomy, telemetry JSON Schemas,
  Model 2 twin and release configs, LoRA config, `config/agent_rules.yaml`.
- **Contamination:** Values are loaded ad hoc with `yaml.safe_load` / `json.load` into dicts and then
  `from_mapping` classmethods. There is no single typed settings model and no versioning of the config
  itself beyond a few `schema_version` keys.
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** Config parsing concept offers nothing beyond V2 `conrad.settings` and Pydantic contracts.
  The numbers inside are self-described simulator hypotheses (see section 2). None may be imported as
  V2 defaults.

### 6.5 Datasets and model weights

- **Legacy path:** `dataset/` (README, `sources.yaml`, header-only manifests, `licenses/noaa_ocean_exploration_2026-08-23.txt`,
  `*.example.*` files), `data/model2/{d0_debug,s1_synthetic}/*.npy|json`, `data/model1_baseline_v2/**`,
  `models/weakpoint_v2.json`, `artifacts/*.json`, `unity/.../Models/*.onnx`, `outputs/**`, `reports/**`.
- **What was seen:** `dataset/manifests/approved_assets.csv`, `raw_assets.csv` and `rejected_assets.csv`
  each contain only the header row, so no approved real asset was ever recorded. The only real
  dataset (SeaClear) is referenced by manifest, not stored. Every tracked numeric dataset and every weight
  file is synthetic or trained on synthetic data. `dataset/licenses/README.md` flags TrashCan and SUIM
  as academic-research-only and DeepFish and CoralNet as unclear; none of them was acquired.
- **Classification:** `OBSOLETE`.
- **Reason:** Nothing real to salvage. The Model 2 `.npy` files carry the truth-derived confidence leak
  from 5.5. The weights are tied to made-up dynamics or circular synthetic labels.

### 6.6 Documentation

- **Legacy path:** `docs/` (63 files, including `claim_boundaries.md`, `MODEL2_ARCHITECTURE_AUDIT.md`,
  `TWIN1_STATUS_REPORT.md`, `failure_twin_spec.md`, the `MODEL1_*` series, `hil_and_field_validation_plan.md`),
  `README.md` (English plus a Turkish section).
- **Purpose:** Status reports, claim boundaries, dataset permission logs, validation plans.
- **Classification:** `REFERENCE_ONLY`.
- **Reason:** Useful as history and as a record of which claims were never supported (for example the
  README's own note that the PPO ONNX success figures no longer apply to the current simulator). They
  describe the obsolete four-pillar architecture, so they must not be treated as V2 specification.

### 6.7 CI and repository tests overall

- **Legacy path:** `.github/workflows/ci.yml`, `tests/**` (22 files, 116 `def test_` functions by count).
- **What was seen:** CI installs the package, runs ruff, `compileall`, `pytest -q`,
  `scripts/validate_unity_project.py` and `rovdt demo`. No mypy, no coverage gate, no Unity test run in CI.
  Test weight is concentrated on release hygiene (Model 2 release validator and configs, 38 tests) and the
  SeaClear review workflow (13 tests). Physics, sensors and control have no Python tests; their Unity
  tests check monotonicity only. No test checks for truth leakage across the belief boundary.
- **Classification:** `OBSOLETE`.
- **Reason:** No legacy test targets a V2 contract. The cases worth remembering (path escape from a release
  directory, split-usage pinning, checksum mismatch) are listed in section 5.6 for V2 authors to write fresh.

### 6.8 Untracked local state

`.tools/`, `.pytest_cache/`, `.ruff_cache/`, `tmp/`, `results/` exist on disk but are ignored or
untracked. They were not audited and nothing from them is recommended.

## 7. Explicit checklist from the brief

| Item looked for | Found? | Where | Classification |
|---|---|---|---|
| Old Model 1 | Yes (image perception plus threshold decision) | `oceansense/perception.py`, `decision.py`, `mission_decision.py` | REJECT_ARCHITECTURAL_CONFLICT |
| Twin1 | Yes, named as such | `unity/`, `docs/TWIN1_STATUS_REPORT.md` | REJECT_ARCHITECTURAL_CONFLICT (shell), OBSOLETE (components) |
| Navigation twin | Yes | `oceansense/navigation_twin.py` plus Unity | OBSOLETE |
| Failure twin | Yes, two of them | `oceansense/failure_twin.py`; `oceansense/model2/simulator.py` | OBSOLETE; REIMPLEMENT_FROM_CONCEPT |
| Global-state assumptions | Yes | `UnityEngine.Random`, `Academy.Instance`, `RenderSettings`, `Physics.gravity`, `ROV_DT_MODEL_PATH` env var | n/a |
| Direct simulator truth access | Yes | `ROVRLAgent.CollectObservations`, `TelemetryUdpBridge` (`trueDepth`, `position_m`), Model 2 observation `confidence` | REJECT |
| Database schemas | None exist | `git grep` for sqlite, sqlalchemy, postgres, `CREATE TABLE` is empty | n/a |
| Single scalar confidence | Yes, pervasive | see section 2 | REJECT |
| Direct model-to-actuator path | Yes | `ROVRLAgent.OnActionReceived` to `Vehicle.SetThrusterCommand` | REJECT_ARCHITECTURAL_CONFLICT |
| Unversioned messages | Yes | `std_msgs/String` topics, dataclass events, hand-built JSON | REJECT |
| Fake/nominal constants shown as physics | Yes | `Hydrodynamics6Dof.cs`, `Thruster.cs`, `UnderwaterEnvironment.cs`, `physics_residuals.py` defaults | OBSOLETE |
| Monolithic control flow | Yes | Unity scene wiring; `digital_twin_demo.py` runs sim, fixture, placeholder model and decision in one function | OBSOLETE |
| Maths utilities | Only textbook calibration metrics | `rov_dt/calibration.py` | REFERENCE_ONLY |
| Geometry helpers | Only a 3-tuple `_distance` | `navigation_twin.py` | OBSOLETE (V2 frames math exists) |
| Hydrodynamics with sourced coefficients | No; every coefficient is unsourced | `Hydrodynamics6Dof.cs` | OBSOLETE |
| Sensor-processing utilities | Only noise injection over global RNG | Unity sensor scripts | OBSOLETE |
| Unity art assets | None beyond primitives and five flat materials | `unity/Assets/ROVDigitalTwin/Materials` | OBSOLETE |
| Test fixtures | Synthetic `.npy` releases with a truth leak | `data/model2/*` | OBSOLETE |
| Control maths (PID, allocation) | None exist | `git grep` for PID, Kalman, allocation, pinv is empty | n/a |
| Config parsing | Ad hoc dict loading | `config/`, `configs/`, `from_mapping` | REFERENCE_ONLY |
| Visualisation | Matplotlib scenario plots, IMGUI HUD | `model2/visualization.py`, `OceanSenseDashboard.cs` | OBSOLETE |

## 8. Classification summary

| Classification | Count | Items |
|---|---|---|
| REUSE_WITHOUT_SEMANTIC_CHANGE | 0 | none |
| PORT_AND_ADAPT | 0 | none |
| REIMPLEMENT_FROM_CONCEPT | 3 | physics residuals with validity (4.6), graph degradation twin concept (5.5), system identification with null-if-unidentifiable (6.1) |
| REFERENCE_ONLY | 7 | telemetry field status (4.2), calibration utilities (4.5), Model 2 release gate (5.6), run/governance manifests (5.7), SeaClear pipeline (5.8), configs (6.4), docs (6.6) |
| OBSOLETE | 15 | Unity hydrodynamics (3.2), thruster (3.3), environment (3.4), sensors (3.5), art (3.8), Unity tests (3.9), weak-point classifier (4.3), API/RAG/LLM (5.2), navigation twin (5.3), image failure twin (5.4), scripts (6.2), tools (6.3), datasets and weights (6.5), CI and tests (6.7), visualisation (inside 5.5 and 3.7, recorded separately in the ledger) |
| REJECT_ARCHITECTURAL_CONFLICT | 6 | Twin1 shell (3.1), RL agent and ONNX (3.6), UDP bridge (3.7), ROS 2 bridge (4.1), `rov_dt` decision stack (4.4), legacy Model 1 (5.1) |

Total: 31 ledger entries.

## 9. Conclusion

Nothing in the legacy repository qualifies for `REUSE_WITHOUT_SEMANTIC_CHANGE` or `PORT_AND_ADAPT`.

- There is no hydrodynamics with sourced or validated coefficients. Every drag, added-mass, buoyancy and
  thrust value is unsourced, and the one "identified vehicle profile" mechanism was never run on real data.
- There are no Unity art assets worth keeping. The scene is built from primitives at editor time.
- There is no control or estimation maths (no PID, allocation, EKF, frames library) to compare against
  V2's from-spec implementations.
- The best-tested code (Model 2 release validation) protects datasets that themselves carry a
  truth-derived confidence field, so the tests passing never caught the leak.

The three `REIMPLEMENT_FROM_CONCEPT` items are ideas, not code. They should be rebuilt in V2 from the
specification, with V2 tests written first, and without opening the legacy files during implementation.
The machine-readable version of this audit is `artifacts/migration/digital-twin-salvage-ledger.json`.
