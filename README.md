# Conrad V2

Persistent-belief autonomy stack for underwater inspection robots, built as separate truth, belief and decision
planes.

**Status (spec ch30):** Conrad V2 concept and contract specification with configurable research implementations.

This repository holds the contracts, candidate implementations, baselines and evaluation harnesses. It does not
hold a validated system. Every experiment so far ran on synthetic data on CPU (see "What is and is not validated"
below).

## The plane rule

- **Twins are the truth world.** `conrad/twins` (Twin2T, Twin2E, Twin2S) and `conrad/sim` own world truth.
  Twin1 does not exist.
- **Model 2 is the belief world.** `conrad/core` and `conrad/domains` turn observations into evidence, and
  evidence into persistent beliefs with a four-channel uncertainty (U_A, U_E, U_C, U_O).
- **Model 1 is the decision world.** `conrad/decision` (EGDC), `conrad/active` (MCBR) and
  `conrad/communication` (BAAC) act on belief only.
- **Execution is a fourth plane.** `conrad/robotics` and `conrad/runtime` move the robot. Only
  `conrad.runtime.command_gateway` may call `RobotHardwareInterface.send`.

Deployment packages never import truth. `tests/leakage/test_static_boundaries.py` enforces this on every module's
AST.

```
 TRUTH PLANE (sim / eval / training only)          BELIEF PLANE (Model 2)
 conrad/twins, conrad/sim                          conrad/core, conrad/domains
 Twin2T  Twin2E  Twin2S  sim kernel                Observation -> ECMER -> Evidence
        |                                           -> Association -> BUO / TBD / RBP / PMBL
        | Observation only (never truth)            -> Model2T / Model2E / Model2S
        v                                           -> Belief Bus (BeliefMessage, BeliefSnapshot)
 +--------------------------------+                                 |
 | RobotHardwareInterface (RHI)   |                                 v
 | sim kernel | Unity | physical  |                DECISION PLANE (Model 1)
 +--------------------------------+                conrad/decision  EGDC -> DecisionRecord
        ^                                           conrad/active    MCBR -> ObservationPlan
        | AllocatedCommand (gateway only)           conrad/communication BAAC -> Transmission
        |                                                           |
 EXECUTION PLANE                                                    v
 conrad/robotics, conrad/runtime  <----- NavigationGoal ------------+
 Estimation -> Navigation -> Trajectory -> Control -> Allocation -> Safety -> CommandGateway -> RHI.send
```

## Quick start

On this Windows host `uv` is not on PATH, so every command below was run as `python -m uv ...`
(for example `python -m uv run pytest -q`).

```
uv sync --all-groups
uv run ruff check .
uv run mypy conrad tests
uv run pytest -q
uv run conrad db migrate
uv run conrad doctor
uv run conrad eval list
uv run conrad train run --config configs/train/core_smoke.yaml
```

The ordinary `pytest -q` suite excludes live Unity tests. Run a selected
development player test with `pytest --run-unity-live
tests/unity_live/test_spatial_unity_parity.py`. Historical formal Unity gate
tests require both `--run-unity-live` and `--run-formal-unity-gates` plus an
explicit test path; they launch player missions and write gate measurements.

- `conrad doctor` checks Python 3.11, the lock file, identity, directories, disk, object store, migrations, the
  frame contract, the RobotConfig, the command mode, adapters, checkpoints, dataset manifests and required secret
  names. It exits nonzero on any FAIL. Before `conrad db migrate` it reports a WARN for the missing runtime
  database.
- `conrad eval list` prints the experiments registered with `conrad eval run` (see [docs/EVALUATION.md](docs/EVALUATION.md)).
- `conrad train run --config configs/train/core_smoke.yaml` trains the smoke-scale GRU TBD on CPU and writes a run
  directory under `artifacts/runs/`.

More detail: [docs/SETUP.md](docs/SETUP.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Repository layout

| Path | Contents |
|---|---|
| `conrad/schemas` | Frozen public contracts (Observation, Evidence, BeliefMessage, Uncertainty, provenance, robot, decision, comms). `truth.py` is truth-plane only. |
| `conrad/core` | Model 2 Core: ECMER, association, BUO, uncertainty, TBD, RBP, PMBL, belief graph, pipeline |
| `conrad/domains` | Model 2 children: `technical` (Model2T, TCDP), `ecological` (Model2E, CEFD), `spatial` (Model2S, UAHSM) |
| `conrad/twins` | Truth twins: `twin2t` (MCDE), `twin2e` (MEIFE), `twin2s` (OCPWE) |
| `conrad/decision` | Model 1 / EGDC: claim graph, constraints, router, UIR, learned scorer candidate |
| `conrad/active` | MCBR planner, baselines, learned ranker candidate |
| `conrad/communication` | BAAC sender/receiver, scheduler, channel simulator, learned value heads candidate |
| `conrad/robotics` | Estimation, navigation, trajectory, control, allocation, safety, hardware interface, characterization, identification |
| `conrad/runtime` | Command gateway, runtime supervisor, doctor, event log, health, release lanes |
| `conrad/orchestration` | Belief Bus, mission context and the integrated-mission deployment side (in progress) |
| `conrad/sim` | Python 6-DOF kernel, HIL harness, Unity tooling, shared scenario, integrated-mission truth side (in progress) |
| `conrad/adapters/unity` | Unity V2 bridge adapter (ZeroMQ) |
| `conrad/persistence` | SQLite WAL repository, Alembic migrations, object store, run bundles |
| `conrad/training` | Trainer, curriculum, checkpoints, run directories, `conrad train run` jobs |
| `conrad/evaluation` | Experiment dispatch and registry, metrics, oracles, per-domain experiments, nav benchmarks |
| `conrad/data` | Dataset manifests, lineage splits, SSL corpus tooling, sim/real ledger |
| `conrad/console` | Read-only operator and replay console |
| `conrad/cli` | The `conrad` Typer CLI |
| `configs/` | `base`, `runtime`, `robot`, `sim`, `train`, `eval` YAML |
| `tests/` | `unit`, `property`, `contract`, `leakage`, `integration`, `replay`, `simulation`, `hardware_stub`, `fixtures` |
| `scripts/` | Shell wrappers, `secret_scan.py`, `legacy_independence_check.py`, `build_requirements_ledger.py` |
| `unity/ConradUnityV2` | Unity C# project, delivered as source (not executed here) |
| `datasets/` | Public dataset manifests (templates) and `simreal_ledger.yaml` |
| `docs/` | Specification, ADRs, research results and the documents below |

## Documents

| Document | Subject |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Planes, data flow, ownership, truth-leakage enforcement, runtime state machine, gateway reason codes |
| [docs/IMPLEMENTATION_MAP.md](docs/IMPLEMENTATION_MAP.md) | Spec ch32 handoff: one row per package |
| [docs/SETUP.md](docs/SETUP.md) | Install and first run |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Rules, tooling, tests, CI |
| [docs/MODEL2.md](docs/MODEL2.md) | Model 2 Core and the three domain children |
| [docs/TWINS.md](docs/TWINS.md) | Twin2T, Twin2E, Twin2S and the shared scenario |
| [docs/MODEL1.md](docs/MODEL1.md) | EGDC decision layer |
| [docs/MCBR.md](docs/MCBR.md) | Active perception planner |
| [docs/BAAC.md](docs/BAAC.md) | Bandwidth-aware communication |
| [docs/TRAINING.md](docs/TRAINING.md) | Training jobs, checkpoints, compute gating |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Experiments, registry, metrics |
| [docs/DATASETS.md](docs/DATASETS.md) | Data tooling (see also [docs/DATASET_REGISTER.md](docs/DATASET_REGISTER.md)) |
| [docs/RHI.md](docs/RHI.md) | Robot hardware interface, adapters, robotics stack |
| [docs/REPLAY.md](docs/REPLAY.md) | Event log, run bundles, deterministic replay |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | Provenance DAG and causal trace |
| [docs/SECURITY.md](docs/SECURITY.md) | Network binding, peers, secrets, command authority |
| [docs/RELEASE.md](docs/RELEASE.md) | dev / candidate / physical release lanes |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Known failure messages and fixes |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Operator procedures |
| [docs/MODEL_CARDS/](docs/MODEL_CARDS/README.md) | One card per learned candidate |
| [docs/UNITY.md](docs/UNITY.md) | Unity V2 bridge |
| [docs/HIL.md](docs/HIL.md) | HIL harness |
| [docs/PHYSICAL_INTEGRATION.md](docs/PHYSICAL_INTEGRATION.md) | Hardware handoff, identification, readiness gates |
| [docs/CONSOLE.md](docs/CONSOLE.md) | Operator and replay console |
| [docs/DATASET_REGISTER.md](docs/DATASET_REGISTER.md) | Dataset manifests on disk |
| [docs/adr/](docs/adr) | ADR-0001 to ADR-0005 |
| [docs/research/EXPERIMENT_RESULTS_2026-09-18.md](docs/research/EXPERIMENT_RESULTS_2026-09-18.md) | Executed experiment results |
| [docs/specification/CONRAD_V2_REQUIREMENTS_LEDGER.md](docs/specification/CONRAD_V2_REQUIREMENTS_LEDGER.md) | Generated requirements ledger |
| [docs/migration/DIGITAL_TWIN_SALVAGE_REPORT.md](docs/migration/DIGITAL_TWIN_SALVAGE_REPORT.md) | Legacy audit: nothing was ported |

## What is and is not validated

Every experiment in [the results record](docs/research/EXPERIMENT_RESULTS_2026-09-18.md) ran on CPU on synthetic
data (the abstract Core sandbox, or the twins at simulation validity L1) with three seeds, and every registry
outcome is INCONCLUSIVE until an ADR evaluates a paired effect. Some mechanisms behaved as specified in those runs:
the four-channel uncertainty decomposition, PMBL identity and lineage, UAHSM refusing to invent hidden geometry,
EGDC grounding (UIR 0.000 against 0.188 for the naive baseline), and BAAC delivering critical alerts at every
non-zero budget. Several candidates lost to simple baselines or failed their hypothesis: learned association,
learned BUO, learned RBP, CEFD coupling, Model2T crack estimation, and MCBR value ranking. The GRU TBD is the only
learned Core module that beat its simple baselines. Most other learned candidates (ECMER, learned TCDP, learned
CEFD, the EGDC scorer, the MCBR ranker and the BAAC value heads) have only smoke tests, and ADR-0004 keeps the
analytic operators as the runtime default. Nothing is validated on real data, on the physical robot, in Unity or on
target compute. Those need the external inputs EXT-HW-01..05, EXT-UNITY-01, EXT-HIL-01, EXT-DATA-01 and
EXT-COMPUTE-01.
