# Setup

## Requirements

- Python 3.11 (`requires-python = ">=3.11,<3.12"`; `conrad doctor` fails on any other minor version).
- `uv`. On this Windows host `uv` is not on PATH, so run it as `python -m uv` (for example
  `python -m uv run pytest -q`).
- CPU is enough for everything except `configs/train/core_full.yaml`, which needs a CUDA GPU (EXT-COMPUTE-01).
- Dependencies are pinned in `uv.lock` (pydantic, numpy, scipy, torch 2.4 to 2.6, opencv-python, pyyaml,
  sqlalchemy, alembic, mlflow, typer, structlog, pyzmq). Do not add dependencies.

## First run

```
uv sync --all-groups
uv run conrad db migrate
uv run conrad doctor
```

`scripts/bootstrap.sh` runs the same three commands.

- `conrad db migrate` creates `artifacts/conrad.sqlite` at the Alembic head (`0001_initial`). The schema is never
  created automatically.
- `conrad doctor` prints one line per check and `RESULT: OK` or `RESULT: FAIL` (exit 1). `--json` gives a
  machine-readable report. `--config` selects another runtime config (default `configs/runtime/default.yaml`).

Then check the toolchain and tests:

```
uv run ruff check .
uv run mypy conrad tests
uv run pytest -q
uv run conrad eval list
uv run conrad train run --config configs/train/core_smoke.yaml
```

## Configuration files

| File | Purpose |
|---|---|
| `configs/base/default.yaml` | Identity, seed 2026201, paths, runtime defaults (command_mode `disabled`, bind `127.0.0.1`) |
| `configs/runtime/default.yaml` | Simulation runtime, command_mode `simulated` |
| `configs/runtime/hil_host.yaml` / `hil_target.yaml` | HIL harness (see [HIL.md](HIL.md)) |
| `configs/runtime/physical_example.yaml` | Physical lane template; fails `conrad doctor` by design |
| `configs/robot/sim_reference.yaml` | SYNTHETIC_ONLY vehicle used in simulation |
| `configs/robot/physical_template.yaml` | Every physical parameter OPEN until the hardware owner delivers it |
| `configs/local.yaml` (gitignored, optional) | Machine-local override; only `paths` and `device` are allowed |

Dataset files live outside git at `$CONRAD_DATA_ROOT/<dataset_id>`, or `artifacts/data/<dataset_id>` when the
variable is unset. See [DATASETS.md](DATASETS.md).

Problems: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
