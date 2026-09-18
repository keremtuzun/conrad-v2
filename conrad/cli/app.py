"""One coherent Typer CLI (ch34 Required operator commands). The CLI is a contract, not ad hoc scripts."""

from __future__ import annotations

import json

import typer

from conrad.persistence import db as dbmod
from conrad.runtime.doctor import run_doctor
from conrad.settings import load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Conrad V2 operator CLI")
db_app = typer.Typer(no_args_is_help=True, help="Persistence")
sim_app = typer.Typer(no_args_is_help=True, help="Simulation")
train_app = typer.Typer(no_args_is_help=True, help="Training")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation")
replay_app = typer.Typer(no_args_is_help=True, help="Deterministic replay")
runtime_app = typer.Typer(no_args_is_help=True, help="Supervised runtime")
data_app = typer.Typer(no_args_is_help=True, help="Dataset registry")
for name, sub in (
    ("db", db_app),
    ("sim", sim_app),
    ("train", train_app),
    ("eval", eval_app),
    ("replay", replay_app),
    ("runtime", runtime_app),
    ("data", data_app),
):
    app.add_typer(sub, name=name)

DEFAULT_CONFIG = "configs/runtime/default.yaml"


@app.command()
def doctor(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
    as_json: bool = typer.Option(False, "--json", help="machine-readable report on stdout"),
) -> None:
    """Verify material runtime prerequisites. Nonzero exit on any material failure."""
    report = run_doctor(config)
    if as_json:
        typer.echo(json.dumps({**report.model_dump(mode="json"), "ok": report.ok}, indent=2))
    else:
        typer.echo(
            f"conrad doctor  [{report.architecture_id} / {report.stack_id}]  lane={report.lane} command_mode={report.command_mode}"
        )
        for check in report.checks:
            typer.echo(f"  {check.status:<4} {check.name:<18} {check.detail}")
        typer.echo("RESULT: " + ("OK" if report.ok else "FAIL"))
    raise typer.Exit(0 if report.ok else 1)


@db_app.command("migrate")
def db_migrate(config: str = typer.Option(DEFAULT_CONFIG, "--config")) -> None:
    """Apply Alembic migrations to the configured database."""
    settings = load_settings(config)
    target = settings.resolve(settings.paths.database)
    head = dbmod.migrate(target)
    typer.echo(f"database {target} at revision {head}")


def main() -> None:
    from conrad.cli import commands  # noqa: F401  (registers sim/train/eval/replay/runtime/data commands)

    app()
