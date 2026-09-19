"""sim / train / eval / replay / runtime / data commands. Registered on import by ``conrad.cli.app.main``."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import typer

from conrad.cli.app import data_app, eval_app, replay_app, runtime_app, sim_app, train_app


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
    from conrad.data.manifest import (
        ManifestError,
        data_root_for,
        load_manifest,
        verify_manifest,
        verify_manifest_by_id,
    )

    path = Path(manifest)
    if path.suffix in (".yaml", ".yml") and path.exists():
        # data lives outside git (same root as the id form), not next to the manifest yaml
        try:
            root = data_root_for(load_manifest(path).dataset_id)
        except ManifestError:
            root = path.parent  # verify_manifest reports the invalid manifest itself
        problems = [str(p) for p in verify_manifest(path, root)]
    else:
        problems = verify_manifest_by_id(manifest)
    for p in problems:
        typer.echo(f"PROBLEM: {p}")
    typer.echo("RESULT: " + ("OK" if not problems else f"FAIL ({len(problems)} problems)"))
    raise typer.Exit(0 if not problems else 1)


# ---------------------------------------------------------------------- integrated mission (gates I1-I7)
DEFAULT_SIM_CONFIG = "configs/sim/mission_default.yaml"


@sim_app.command("run")
def sim_run(
    scenario: str = typer.Option(..., "--scenario", help="GOLDEN-SMOKE, FLAGSHIP-I4, INT-001 ..."),
    config: str = typer.Option(DEFAULT_SIM_CONFIG, "--config"),
    run_id: str = typer.Option(None, "--run-id"),
) -> None:
    """Run an integrated scenario and write a complete, replayable run bundle."""
    from conrad.sim.mission.run import run_scenario

    out = run_scenario(scenario, config, run_id)
    report = out["report"]
    rt = report.get("runtime", {})
    typer.echo(
        json.dumps(
            {
                "run_id": out["run_id"],
                "run_dir": out["run_dir"],
                "decisions": rt.get("decisions"),
                "commands_accepted": rt.get("commands_accepted"),
                "commands_rejected": rt.get("commands_rejected"),
                "target_after": report.get("target_after"),
            },
            indent=2,
            default=str,
        )
    )


@replay_app.command("run")
def replay_run_cmd(
    run: str = typer.Option(..., "--run", help="run ID under paths.runs_dir, or a run directory"),
    config: str = typer.Option(DEFAULT_SIM_CONFIG, "--config"),
) -> None:
    """Verify bundle digests (fail closed), re-execute from the stored seed/config, compare signatures."""
    from conrad.persistence.replay_store import ReplayIntegrityError
    from conrad.settings import load_settings
    from conrad.sim.mission.replay import replay_run

    run_dir = Path(run)
    if not run_dir.exists():
        s = load_settings(config)
        run_dir = s.resolve(s.paths.runs_dir) / run
    try:
        report = replay_run(run_dir)
    except ReplayIntegrityError as exc:
        typer.echo(json.dumps({"verified": False, "problems": exc.problems}, indent=2))
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2))
    typer.echo("RESULT: " + ("REPRODUCED" if report["equal"] else "MISMATCH"))
    raise typer.Exit(0 if report["equal"] else 1)


@runtime_app.command("start")
def runtime_start(config: str = typer.Option("configs/runtime/default.yaml", "--config")) -> None:
    """Simulation lane: run GOLDEN-SMOKE under the supervisor. HIL/physical lanes are refused by the gates."""
    from conrad.runtime.health import HealthRegistry
    from conrad.runtime.supervisor import RuntimeSupervisor
    from conrad.settings import ExecutionLane, load_settings

    settings = load_settings(config)
    if settings.run.lane in (ExecutionLane.HIL, ExecutionLane.PHYSICAL):
        sup = RuntimeSupervisor(settings, HealthRegistry(lambda: 0), lambda *a, **k: None, UUID(int=0))
        reasons = sup.hardware_gate_reasons()
        typer.echo(
            json.dumps({"refused": True, "lane": settings.run.lane.value, "reasons": reasons}, indent=2)
        )
        raise typer.Exit(3)
    from conrad.sim.mission.run import run_scenario

    out = run_scenario("GOLDEN-SMOKE", settings)
    rt = out["report"]["runtime"]
    typer.echo(
        json.dumps(
            {
                "run_id": out["run_id"],
                "run_dir": out["run_dir"],
                "runtime_state_history": rt["runtime_state_history"],
            },
            indent=2,
            default=str,
        )
    )


@runtime_app.command("safe-hold")
def runtime_safe_hold(
    mission: str = typer.Option(..., "--mission", help="mission UUID"),
    config: str = typer.Option(DEFAULT_SIM_CONFIG, "--config"),
    reason: str = typer.Option("operator request", "--reason"),
) -> None:
    """Record an operator SAFE_HOLD request for the latest run of a mission (append-only, outside digests)."""
    from conrad.settings import load_settings

    s = load_settings(config)
    runs = s.resolve(s.paths.runs_dir)
    matches = []
    for d in sorted(runs.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if runs.exists() else []:
        ev = d / "events.jsonl"
        if ev.exists():
            with ev.open(encoding="utf-8") as handle:
                first = handle.readline()
            if first and json.loads(first)["envelope"]["mission_id"] == mission:
                matches.append(d)
    if not matches:
        typer.echo(f"no run found for mission {mission} under {runs}")
        raise typer.Exit(1)
    run_dir = matches[0]
    inbox = run_dir / "notes" / "operator_requests.jsonl"
    inbox.parent.mkdir(parents=True, exist_ok=True)
    request = {"action": "safe_hold", "mission_id": mission, "operator": "cli", "reason": reason}
    with inbox.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(request) + "\n")
    finished = (run_dir / "bundle_manifest.json").exists()
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "request_file": str(inbox),
                "live_process": not finished,
                "status": "RECORDED_NO_LIVE_PROCESS: the run has finished; the request is recorded for the audit "
                "trail and cannot affect the vehicle"
                if finished
                else "RECORDED: a live MissionRuntime polls this "
                "inbox at its decision cadence and enters SAFE_HOLD through the supervisor",
            },
            indent=2,
        )
    )


gates_app = typer.Typer(no_args_is_help=True, help="Formal gate registry (evidence-derived)")


@gates_app.command("status")
def gates_status(as_json: bool = typer.Option(False, "--json")) -> None:
    """Official gate status derived from artifacts/gates evidence. Surrogate evidence never promotes a gate."""
    from conrad.evaluation.gates import evaluate_gates

    reports = evaluate_gates()
    if as_json:
        typer.echo(json.dumps({k: v.model_dump(mode="json") for k, v in reports.items()}, indent=2))
        return
    for r in reports.values():
        blocked = f" (blocked by {', '.join(r.blocking_upstream)})" if r.blocking_upstream else ""
        typer.echo(
            f"{r.gate_id:<9} official={r.official_status.value:<17} formal={r.formal_status.value:<14} "
            f"surrogate={r.surrogate_status.value}{blocked}"
        )


def _register_gates() -> None:
    from conrad.cli.app import app

    app.add_typer(gates_app, name="gates")


_register_gates()


# ---------------------------------------------------------------------- Unity backend (gates I1-I3, formal path)
# Appended: `conrad sim run` gains --backend {python,unity} and `conrad replay run` dispatches Unity bundles.
sim_app.registered_commands = [c for c in sim_app.registered_commands if c.name != "run"]
replay_app.registered_commands = [c for c in replay_app.registered_commands if c.name != "run"]


@sim_app.command("run")
def sim_run_backend(
    scenario: str = typer.Option(..., "--scenario", help="GOLDEN-SMOKE, FLAGSHIP-I4, I1-UNITY, I2-UNITY-NAV ..."),
    config: str = typer.Option(DEFAULT_SIM_CONFIG, "--config"),
    run_id: str = typer.Option(None, "--run-id"),
    backend: str = typer.Option("python", "--backend", help="python (L1 kernel, SURROGATE) or unity (FORMAL)"),
    seed: int = typer.Option(None, "--seed", help="override run.seed (unity backend)"),
) -> None:
    """Run an integrated scenario on the Python kernel or on the built Unity player and write its bundle."""
    if backend == "python":
        sim_run(scenario, config, run_id)
        return
    if backend != "unity":
        raise typer.BadParameter("--backend must be python or unity")
    from conrad.sim.mission.unity_run import UNITY_SCENARIO_IDS, run_unity_scenario

    if scenario not in UNITY_SCENARIO_IDS:
        raise typer.BadParameter(f"Unity scenarios: {', '.join(UNITY_SCENARIO_IDS)}")
    out = run_unity_scenario(scenario, config, run_id, seed=seed)
    rep = out["report"]
    summary = (
        {k: {"success": v["success"], "checks": v["checks"]} for k, v in rep.items()}
        if scenario == "I2-UNITY-NAV"
        else {"target_after": rep.get("target_after"), "unity": rep.get("unity", {}).get("forwarded_frames")}
    )
    typer.echo(json.dumps({"run_id": out["run_id"], "run_dir": out["run_dir"], **summary}, indent=2, default=str))


@replay_app.command("run")
def replay_run_backend(
    run: str = typer.Option(..., "--run", help="run ID under paths.runs_dir, or a run directory"),
    config: str = typer.Option(DEFAULT_SIM_CONFIG, "--config"),
) -> None:
    """Verify digests, re-execute (Python kernel or Unity, from the bundle's own backend), compare signatures."""
    from conrad.persistence.replay_store import ReplayIntegrityError, load_bundle_manifest
    from conrad.settings import load_settings

    run_dir = Path(run)
    if not run_dir.exists():
        s = load_settings(config)
        run_dir = s.resolve(s.paths.runs_dir) / run
    try:
        backend = load_bundle_manifest(run_dir).replay_inputs.get("backend", "python")
    except ReplayIntegrityError:
        backend = "python"
    if backend != "unity":
        replay_run_cmd(run, config)
        return
    from conrad.sim.mission.unity_run import replay_unity_run

    try:
        report = replay_unity_run(run_dir)
    except ReplayIntegrityError as exc:
        typer.echo(json.dumps({"verified": False, "problems": exc.problems}, indent=2))
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2, default=str))
    typer.echo("RESULT: " + ("REPRODUCED" if report["equal"] else "MISMATCH"))
    raise typer.Exit(0 if report["equal"] else 1)
