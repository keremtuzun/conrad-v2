# Architecture

Conrad V2 separates the **truth**, **belief** and **decision** worlds, plus an **execution** plane that moves the
robot. This document describes how the code in `conrad/` implements that split. Package-level detail is in
[IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md).

## Planes and ownership

| Plane | Packages | Owns | May import truth |
|---|---|---|---|
| Truth | `conrad.twins`, `conrad.sim`, `conrad.schemas.truth` | World entities, hidden state, supervision labels, vehicle truth (`TruthAccess`) | yes |
| Belief (Model 2) | `conrad.core`, `conrad.domains`, `conrad.orchestration` (Belief Bus) | Evidence, belief cells, revisions, four-channel uncertainty, knowledge status | no |
| Decision (Model 1) | `conrad.decision`, `conrad.active`, `conrad.communication` | Claim graphs, DecisionRecords, ObservationPlans, transmissions | no |
| Execution | `conrad.robotics`, `conrad.runtime` | State estimate, goals, trajectories, wrenches, allocated commands, safety state | no, except `runtime` may build the simulated adapter |
| Evaluation / training | `conrad.evaluation`, `conrad.training`, `conrad.data` | Experiments, oracles, checkpoints, manifests | yes |

Rules that the code enforces:

- Belief IDs are minted by the belief plane from an injected `IdFactory`. A simulator world-entity ID is never a
  belief ID or an inference feature.
- Uncertainty is always `conrad.schemas.uncertainty.Uncertainty` with four channels (aleatoric, epistemic,
  contradiction, observational). It has no total or confidence scalar.
- Knowledge status is `OBSERVED`, `INFERRED`, `PREDICTED`, `UNKNOWN` (plus `MIXED`), in
  `conrad.schemas.belief.KnowledgeStatus`.
- Every spatial value names a frame and every time value is integer nanoseconds in a clock domain
  (`conrad.schemas.frames`, `conrad.schemas.timebase`).
- Neural outputs never command hardware. MCBR returns an `ObservationPlan`; navigation moves the robot; only the
  command gateway calls `RobotHardwareInterface.send`.

## Data flow

```
 Twin2S / Twin2T / Twin2E   sim kernel (6-DOF)        TRUTH (never crosses the RHI)
          |                        |
          +---- sensor rendering --+--> RobotHardwareInterface: get_imu/get_depth/get_camera/get_sonar/...
                                              |
                                              v
 Observation (conrad.schemas.observation)   estimated pose only, frame + clock stamped
      |
      v
 Evidence  <- ECMER (conrad.core.ecmer) or a domain encoder (spatial GeometricEvidenceEncoder, ECMER-E)
      |
      v
 Association (conrad.core.association)      candidate gating, NO_MATCH allowed
      |
      v
 Belief update: BUO (direct) | TBD (temporal, delta_t in seconds) | RBP / TCDP (relational) | PMBL (persist)
 Domain children: Model2T (technical), Model2E (ecological), Model2S (spatial)
      |  BeliefMessage + BeliefRevision + ProvenanceRecord -> Repository (SQLite)
      v
 Belief Bus (conrad.orchestration.belief_bus)   versioned, per-domain ownership, BeliefSnapshot
      |
      v
 EGDC (conrad.decision)   claim graph -> candidates -> ConstraintEngine -> router -> DecisionRecord
      |                                   |                              |
      | REQUEST_INFORMATION               | REVISIT_REGION               | TRANSMIT / STORE_AND_FORWARD
      v                                   v                              v
 MCBR (conrad.active) -> ObservationPlan -> NavigationGoal          BAAC (conrad.communication)
                                          |                              -> ChannelSim -> ReceiverStore
                                          v
 Navigation (conrad.robotics.navigation.NavigationStack)
   EKF estimate -> goal/trajectory -> local planner -> cascaded PID -> ThrusterAllocator
   -> SafetySupervisor.assess / authorize (SafetyAuthorization on the AllocatedCommand)
      |
      v
 CommandGateway (conrad.runtime.command_gateway)   validates, logs COMMAND_SENT / COMMAND_REJECTED
      |
      v
 RobotHardwareInterface.send  (SimRobotHardware | UnityRobotHardware | PhysicalRobotHardware)
```

The full loop is exercised end to end by the Gate I0 fixture test
`tests/integration/test_i0_fake_full_system.py` (belief resolves from UNKNOWN to OBSERVED, commands pass through
the gateway, the causal trace reaches a raw observation, and replay is deterministic). That test uses the
`tests/fixtures/fake_system.py` fixture, not the integrated mission.

`NavigationStack.step` (`conrad/robotics/navigation/stack.py`) reads hardware through a read-only wrapper
(`_ReadOnlyHardware`) and returns a `StepResult`. The caller submits `StepResult.command` to the gateway.

## Truth-leakage enforcement

`tests/leakage/test_static_boundaries.py` parses the AST of every module, so a leak fails CI even when no test
runs the code path:

| Test | Rule |
|---|---|
| `test_deployment_planes_never_import_truth` | `core`, `domains`, `decision`, `active`, `communication`, `robotics`, `runtime`, `orchestration` never import `conrad.twins`, `conrad.schemas.truth`, `conrad.evaluation`, `conrad.training` or `conrad.sim`. Only `runtime` may import `conrad.sim.kernel` and `conrad.sim.scenarios`. |
| `test_deployment_planes_never_name_truth_types` | The same packages never reference `TruthState`, `SupervisionLabel`, `TruthAccess` or `get_truth`. |
| `test_no_vendor_sdk_outside_adapters` | No ROS, Unity, MAVLink, sonar/camera SDK or `zmq` import in schemas, core, domains, twins, decision, active, communication, robotics, runtime. |
| `test_only_the_gateway_calls_hardware_send` | Outside `runtime/command_gateway.py`, `robotics/hardware/interface.py`, adapters and `sim/kernel`, no call to `send` / `set_thruster_commands` on a hardware-like object. |
| `test_no_legacy_repository_reference_anywhere` | No reference to the legacy repository in code, configs or `pyproject.toml`. |
| `test_no_obsolete_architecture_identifiers` | No `Twin1` or `Model2A/B/C` names. |

Domain packages add their own static checks (`tests/unit/domains/technical/test_m2t_static.py`,
`tests/unit/domains/spatial/test_m2s_leakage.py`,
`tests/unit/domains/ecological/test_eco_encoder_learned.py::test_no_truth_plane_imports_in_belief_package`).

Twins keep supervision on a separate channel: `Twin.generate_observation` returns `TwinSample(observation,
supervision)` (`conrad/twins/base.py`), and only training and evaluation code reads `supervision`.

## IdFactory namespaces

`conrad.schemas.ids.IdFactory` mints UUIDv7 values. With a seed it is deterministic (logical millisecond counter
plus a seeded PRNG). `IdFactory.child(namespace)` returns an independent stream salted with
`sha256("<seed>:<namespace>")`, and nested children join names with `/`. Two components that share a run seed
therefore never mint the same ID. Without a seed it uses wall clock and OS entropy.

Namespaces used in the code today:

| Namespace | Where |
|---|---|
| `scenario`, `twin2s`, `twin2t`, `twin2e`, `sensors`, `registry`, `sensing` | `conrad/sim/mission/world.py` (integrated mission, in progress) |
| `scenario`, `events`, `run`, `sensing`, `twin2t`, `evidence` | `conrad/evaluation/structural_experiments` |
| `twin2e`, `twin2e-run`, `model2e/<variant>`, `episode`, `ecmer-e`, `twin2e-confounders` | `conrad/evaluation/ecological_experiments` |
| `estimator/<name>`, `model2t` (constructor `namespace=`) | `conrad/domains/technical/baselines.py` |

Deterministic code never calls `uuid.uuid4()` or reads wall-clock time (WORKSTREAM_BRIEF rule).

## Analytic-default policy (ADR-0004)

Each Model 2 Core mechanism has a learned `EXPERIMENTAL_CANDIDATE` and a deterministic analytic operator behind
one interface. `Model2Core` (`conrad/core/pipeline.py`) uses `AssociationEngine` with no scorer (nearest
neighbour), `AnalyticBUO`, `AnalyticTBD` and `AnalyticRBP`. Learned modules only move latent tensors through
`LatentModules` (`conrad/core/pipeline_latent.py`), which default to `None`. The domain children follow the same
rule: `Model2T` uses analytic TCDP (`learned_tcdp.py` is "not on the Model2T runtime path"), `Model2E` uses
`AnalyticCEFD`, `Model2S` has no learned heads, EGDC defaults to `StructuredReasoningPolicy`, and MCBR and BAAC run
their analytic scoring. A learned module becomes a runtime default only through a later ADR that cites a
registered experiment. ADR-0004 status: ACCEPTED as an interim runtime policy, approval PENDING REVIEW.

## Python kernel and Unity (ADR-0005)

- `conrad.sim.kernel.SimRobotHardware` is a headless, deterministic Fossen-form 6-DOF RK4 kernel behind the same
  `RobotHardwareInterface`. CI, replay, nav benchmarks and host HIL use it. Its validity level is L1.
- `unity/ConradUnityV2` is the C# simulator, delivered as source. It reaches the stack only through
  `conrad.adapters.unity.UnityRobotHardware` over ZeroMQ. Unity execution and kernel/Unity cross-validation are
  BLOCKED_EXTERNAL (EXT-UNITY-01). See [UNITY.md](UNITY.md).
- Neither simulator owns world truth; the twins do.

## Persistence (ADR-0001, ADR-0003)

- SQLite in WAL mode with a single writer (`conrad/persistence/db.py`): `journal_mode=WAL`, `synchronous=FULL`,
  `foreign_keys=ON`, every transaction opened with `BEGIN IMMEDIATE`. The schema is created only by Alembic
  (`conrad db migrate`); `check_migration_state` refuses an unmigrated, behind-head or newer-than-code database.
- `conrad/persistence/repository.py` commits a belief update, its evidence contributions and its provenance in one
  transaction. It rejects duplicate evidence (`DuplicateEvidenceError`), wrong predecessors
  (`PredecessorMismatchError`) and illegal lifecycle transitions (`LifecycleError`). A replayed
  (message_id, producer_version) returns `DUPLICATE_MESSAGE`.
- Late evidence (ADR-0003): a DIRECT update older than the head that is not flagged `late` raises
  `StaleUpdateError`. `runtime.late_evidence_policy` is `EXPLICIT_LATE` (default) or `REJECT`. Bounded rewind is
  OPEN.
- Raw sensor data and large blobs go to the content-addressed `ObjectStore` (`<root>/sha256/<digest>`), which
  re-hashes on every read.

See [REPLAY.md](REPLAY.md) and [PROVENANCE.md](PROVENANCE.md).

## Runtime state machine

`conrad/runtime/supervisor.py`, `RuntimeSupervisor`:

```
BOOT -> SELF_TEST -> CONFIGURED -> CONNECTING -> READY -> RUNNING
RUNNING <-> DEGRADED
RUNNING | DEGRADED -> SAFE_HOLD -> STOPPING -> STOPPED      (SAFE_HOLD -> RUNNING on resume)
READY -> STOPPING
any state -> FAILED
```

- A failed self-test or adapter connection moves to FAILED.
- `tick()` moves to SAFE_HOLD on any critical module failure from the `HealthRegistry`, without a new command.
- Operator actions are `start`, `pause`, `safe_hold`, `resume`, `stop`. Each is logged as an `OPERATOR_ACTION`
  event. `resume` is refused while a critical module is unhealthy.
- `start(hardware=True)` raises `HardwareGateError` unless command_mode is `hardware`, the local hardware enable is
  true, the lane is `hil` or `physical`, a HIL evidence reference is set, and the adapter, safety supervisor and
  command gateway are all marked healthy.

## Command gateway reason codes

`conrad/runtime/command_gateway.py`, `CommandGateway.validate` returns every reason that applies. Any reason
rejects the command and emits `COMMAND_REJECTED`.

| Code | Condition |
|---|---|
| `NOT_ALLOCATED_COMMAND` | The object is not an `AllocatedCommand` (raw PWM, untyped payload, model output) |
| `INCOMPATIBLE_SCHEMA` | Schema major version differs |
| `COMMAND_MODE_DISABLED` | `runtime.command_mode` is `disabled` |
| `HARDWARE_MODE_NOT_ENABLED` | Physical adapter without command_mode `hardware` and `hardware_enable` |
| `HARDWARE_PREREQUISITES_MISSING` | Physical lane without a HIL evidence reference, or with ungrounded RobotConfig parameters |
| `ADAPTER_NOT_PERMITTED_IN_LANE` | Physical adapter outside the hil/physical lane, or command_mode `hardware` with a simulated adapter |
| `UNKNOWN_MISSION` / `UNKNOWN_RUN` | mission_id or run_id does not match the gateway's |
| `UNKNOWN_PEER` | Peer not in `runtime.allowed_peers` |
| `WRONG_CLOCK_DOMAIN` | Command clock domain differs from the adapter's |
| `EXPIRED_DEADLINE` / `ISSUED_IN_FUTURE` | Deadline already passed, or issue time after now |
| `WRONG_ROBOT_CONFIG_DIGEST` | Command or adapter RobotConfig digest differs from the gateway's |
| `MISSING_SAFETY_AUTHORIZATION` / `SAFETY_AUTHORIZATION_MISMATCH` | No authorization, or it names another command |
| `SAFETY_STATE_FORBIDS_MOTION` | Safety state not NORMAL/DEGRADED/HOLD. In EMERGENCY_STOP, RECOVER and RETURN only an all-zero command passes. |
| `NOT_PRODUCED_BY_ALLOCATOR` | `producer` is not `conrad.robotics.allocation` |
| `INVALID_ACTUATOR_SET` | Thruster IDs differ from the RobotConfig, or the layout is empty |
| `COMMAND_OUT_OF_ENVELOPE` | A thruster value is non-finite or outside [-1, 1] |
| `STALE_STATE` | State age unknown or beyond `safety.state_stale_after_s` |
| `INVALID_HEALTH_STATE` | Adapter health FAULT and the command is non-zero |
| `DUPLICATE_COMMAND_ID` | Command ID already accepted |

RETURN and RECOVER motion stays refused until the hardware safety contract defines those behaviours (EXT-HW-05).
Tests: `tests/contract/test_command_gateway.py`.

## Configuration

`conrad/settings.py` loads YAML with an `extends:` chain (base, then mode, then experiment), then an optional
`configs/local.yaml` that may only contain `paths` and `device`. Unknown keys fail. `ConradSettings` refuses
command_mode `hardware` outside the hil/physical lanes, any command mode other than `disabled` in the dev lane,
and a `bind_address` of `0.0.0.0` or `::`.

## Integrated mission (in progress at time of writing)

The integrated mission described in [development/INTEGRATION_BRIEF.md](development/INTEGRATION_BRIEF.md) is being
written concurrently. On disk when this document was written:

- Truth side, `conrad/sim/mission/` (no `__init__.py` yet): `world.py` (`MissionWorld`: one shared scenario,
  three twins, the kernel, the payload suite), `hardware.py` (`MissionHardware(SimRobotHardware)` with
  `get_payload_observations()`), `sensing.py` (`MissionSensorSuite`), `registry.py` (mission context with fresh
  registry IDs), `structure.py`, `options.py`, `truth.py` (`MissionTruthRecorder`, writes `truth/truth_record.json`).
- Deployment side, `conrad/orchestration/`: `belief_bus.py`, `mission_context.py`, `mission_config.py`,
  `association.py` (projects STRUCTURED observations through the estimated pose; NO_MATCH allowed),
  `perception.py`, `deliberation.py` (EGDC and MCBR over the Belief Bus), `comms.py` (BAAC over `ChannelSim`),
  `services.py` (`ModuleRunner`), `routing.py` (cadenced EGDC decision cycle and routing), `executive.py`
  (goals, control tick, gateway submission with COMMAND provenance), `artifacts.py` (writes `mission/` in the
  run directory), `mission.py` (`MissionRuntime`: one tick = sense, encode/associate, Model2S/2T/2E, Belief Bus,
  cadenced EGDC, MCBR, NavigationStack, safety, CommandGateway, BAAC), `evaluation.py` (`evaluate_run`: reads the
  run's SQLite, `mission/` and `truth/truth_record.json` from disk; imports no truth-plane code).
- Not yet on disk: `conrad/evaluation/int_benchmarks`, and the `conrad sim`, `conrad replay` and
  `conrad runtime` subcommands (the command groups exist but have no commands).

No integrated-mission result exists yet, and the I4 acceptance thresholds are OPEN.
