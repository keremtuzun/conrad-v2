"""`conrad train run` entrypoints. Every job runs through the governed Trainer, writes an immutable
run directory and a compatibility-stamped checkpoint, then proves the checkpoint reloads.

Registered jobs:
  core_tbd_smoke   CORE-TBD learned GRU temporal dynamics on the abstract sandbox (CPU smoke)

A full-scale config (``scale: full``) is accepted and validated, but running it on a host without
the declared compute raises ComputeBlockedError instead of silently shrinking the job.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from conrad.core.config import CoreConfig, tiny_config
from conrad.evaluation.core_experiments.tbd_e001 import Harness, make_sequences, pairs
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import load_checkpoint
from conrad.training.determinism import inspect_compute
from conrad.training.run_dir import RunDirectory, RunPurpose, TerminalStatus
from conrad.training.trainer import Trainer
from conrad.training.trainer_config import RunIdentity, TrainerConfig


class ComputeBlockedError(RuntimeError):
    """The job is fully configured but the declared compute is not present on this host."""


def _split_hash(parts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _core_tbd_job(job: dict[str, Any], run: RunDirectory, seed: int) -> dict[str, Any]:
    cfg: CoreConfig = tiny_config() if job.get("dims", "tiny") == "tiny" else CoreConfig()
    n_train, n_val, length = (
        int(job["train_sequences"]),
        int(job["val_sequences"]),
        int(job["sequence_length"]),
    )
    # lineage split: train and validation come from disjoint generator seeds, never shared sequences
    train_seed, val_seed = seed * 10 + 1, seed * 10 + 2
    train = pairs(
        make_sequences(np.random.default_rng(train_seed), n_train, length, float(job["process_noise"]))
    )
    val = pairs(make_sequences(np.random.default_rng(val_seed), n_val, length, float(job["process_noise"])))
    split = {
        "generator": "core_tbd_e001.make_sequences",
        "train_seed": train_seed,
        "val_seed": val_seed,
        "n": [n_train, n_val],
    }
    split_hash = _split_hash(split)
    manifest_digest = hashlib.sha256(b"SYNTHETIC:core_sandbox_tbd:v1").hexdigest()
    batch = int(job["batch_size"])

    def batches(epoch: int) -> list[dict[str, torch.Tensor]]:
        g = torch.Generator().manual_seed(seed + epoch)
        idx = torch.randperm(train["y"].shape[0], generator=g)
        return [{k: v[i] for k, v in train.items()} for i in idx.split(batch)]

    def loss_fn(model: torch.nn.Module, b: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, var = model(b["y"], b["dt"])
        return F.mse_loss(mean, b["x_next"]) + 0.5 * F.gaussian_nll_loss(mean, b["x_next"], var)

    def validate(model: torch.nn.Module) -> dict[str, float]:
        with torch.no_grad():
            mean, var = model(val["y"], val["dt"])
            rmse = float(torch.sqrt(F.mse_loss(mean, val["x_next"])))
            nll = float(F.gaussian_nll_loss(mean, val["x_next"], var))
            hold = float(torch.sqrt(F.mse_loss(val["y"], val["x_next"])))
        return {"val_loss": rmse, "val_nll": nll, "val_hold_last_rmse": hold}

    tcfg = TrainerConfig(
        component="core.tbd.gru",
        batch_size=batch,
        effective_batch_size=batch * int(job.get("accumulation", 1)),
        max_epochs=int(job["max_epochs"]),
        early_stopping_patience=int(job.get("patience", 25)),
        seed=seed,
        device="cpu",
    )
    torch.manual_seed(seed)
    model = Harness(cfg, "gru")
    identity = RunIdentity(manifest_digests=(manifest_digest,), split_hash=split_hash)
    trainer = Trainer(
        model,
        tcfg,
        loss_fn=loss_fn,
        train_batches=batches,
        validate=validate,
        identity=identity,
        checkpoint_dir=run.path / "checkpoints",
        clock_ns=time.time_ns,
        resolved_config={"job": job, "core": cfg.model_dump(mode="json")},
    )
    fit = trainer.fit()
    if fit.best_checkpoint is None:
        raise RuntimeError("training produced no checkpoint")
    fresh = Harness(cfg, "gru")
    loaded = load_checkpoint(fit.best_checkpoint, expected=trainer.compatibility(), model=fresh)
    reload_metrics = validate(fresh)
    return {
        "component": tcfg.component,
        "split": split,
        "split_hash": split_hash,
        "epochs_run": fit.epochs_run,
        "optimizer_steps": fit.optimizer_steps,
        "best_epoch": fit.best_epoch,
        "best_val_rmse": fit.best_metric,
        "final_history": [r.model_dump(mode="json") for r in fit.history[-3:]],
        "hold_last_val_rmse": reload_metrics["val_hold_last_rmse"],
        "checkpoint": str(Path(fit.best_checkpoint).relative_to(REPO_ROOT))
        if Path(fit.best_checkpoint).is_relative_to(REPO_ROOT)
        else fit.best_checkpoint,
        "checkpoint_id": loaded.metadata.checkpoint_id,
        "reload_val_rmse": reload_metrics["val_loss"],
        "reload_matches": abs(reload_metrics["val_loss"] - float(fit.best_metric or 0)) < 1e-5,
    }


JOBS: dict[str, Callable[[dict[str, Any], RunDirectory, int], dict[str, Any]]] = {
    "core_tbd_smoke": _core_tbd_job
}


def run_training(config_path: str | Path, runs_root: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    job = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = job.get("job")
    if name not in JOBS:
        raise ValueError(f"unknown training job {name!r}; registered: {sorted(JOBS)}")
    compute = inspect_compute()
    if job.get("scale") == "full" and not compute.cuda_available:
        raise ComputeBlockedError(
            f"{name}: full-scale run requires {job.get('required_compute', 'a CUDA GPU')}; host has none (EXT-COMPUTE-01)"
        )
    seed = int(job.get("seed", 2026201))
    root = Path(runs_root) if runs_root else REPO_ROOT / "artifacts" / "runs"
    run = RunDirectory.create(
        root,
        f"train-{name}-{time.time_ns()}",
        resolved_config=job,
        manifests={},
        purpose=RunPurpose.DEVELOPMENT,
        clock_ns=time.time_ns,
    )
    try:
        result = JOBS[name](job, run, seed)
        result.update(
            {
                "run_id": run.path.name,
                "run_dir": str(run.path),
                "seed": seed,
                "device": "cpu",
                "compute": compute.model_dump(mode="json"),
            }
        )
        (run.path / "reports" / "training_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        run.seal(TerminalStatus.COMPLETED, time.time_ns())
    except Exception:
        run.seal(TerminalStatus.FAILED, time.time_ns())
        raise
    return result
