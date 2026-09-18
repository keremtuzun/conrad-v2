"""sim / train / eval / replay / runtime / data commands. Registered on import by ``conrad.cli.app.main``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from conrad.cli.app import data_app, eval_app, train_app


@train_app.command("run")
def train_run(config: str = typer.Option(..., "--config")) -> None:
    """Run a registered training job (writes an immutable run directory and checkpoint)."""
    from conrad.training.entrypoints import ComputeBlockedError, run_training

    try:
        result = run_training(config)
    except ComputeBlockedError as exc:
        typer.echo(f"BLOCKED_EXTERNAL: {exc}")
        raise typer.Exit(3) from exc
    typer.echo(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("compute", "final_history")}, indent=2, default=str
        )
    )


@eval_app.command("run")
def eval_run(
    experiment: str = typer.Option(None, "--experiment", help="experiment ID, e.g. CORE-BUO-E001"),
    run: str = typer.Option(None, "--run", help="run ID to evaluate (integrated missions)"),
    config: str = typer.Option(None, "--config"),
    seeds: str = typer.Option(None, "--seeds", help="comma-separated seeds"),
) -> None:
    """Run a registered experiment, or evaluate a stored mission run against its truth record."""
    if run is not None:
        from conrad.orchestration.evaluation import evaluate_run

        typer.echo(json.dumps(evaluate_run(run, config), indent=2, default=str))
        return
    if experiment is None:
        raise typer.BadParameter("give --experiment or --run")
    from conrad.evaluation.dispatch import run_experiment

    seed_list = [int(s) for s in seeds.split(",")] if seeds else None
    out = run_experiment(experiment, config, seed_list)
    typer.echo(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=2))


@eval_app.command("list")
def eval_list() -> None:
    from conrad.evaluation.dispatch import EXPERIMENTS, _discover

    _discover()
    for eid, (module, cfg) in sorted(EXPERIMENTS.items()):
        typer.echo(f"{eid:<22} {module}  [{cfg}]")


@data_app.command("verify")
def data_verify(
    manifest: str = typer.Option(..., "--manifest", help="manifest id, id@version, or path"),
) -> None:
    """Verify a dataset manifest (files, checksums, licence, declared transformations). Fails closed."""
    from conrad.data.manifest import verify_manifest, verify_manifest_by_id

    path = Path(manifest)
    if path.suffix in (".yaml", ".yml") and path.exists():
        problems = [str(p) for p in verify_manifest(path, path.parent)]
    else:
        problems = verify_manifest_by_id(manifest)
    for p in problems:
        typer.echo(f"PROBLEM: {p}")
    typer.echo("RESULT: " + ("OK" if not problems else f"FAIL ({len(problems)} problems)"))
    raise typer.Exit(0 if not problems else 1)
