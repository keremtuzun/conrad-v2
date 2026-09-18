"""Generic deterministic trainer (ch33 Training configuration and implementation gates).

AdamW, lr 3e-4 for new parameters / 1e-4 for in-house pretrained backbones, cosine decay with 5 %
linear warm-up, gradient clipping, accumulation to an effective batch, early stopping, checkpoint
selection on validation metrics only, and a final-only guard for test / OOD evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from conrad.training.checkpoint import (
    CheckpointMetadata,
    CompatibilityTuple,
    config_digest,
    is_validation_metric,
    load_checkpoint,
    save_checkpoint,
)
from conrad.training.determinism import SeedReport, capture_rng_state, restore_rng_state, seed_everything
from conrad.training.trainer_config import (
    EpochRecord,
    FinalOnlyViolation,
    FitResult,
    RunIdentity,
    TrainerConfig,
    build_param_groups,
    warmup_cosine,
)

__all__ = [
    "EpochRecord",
    "FinalOnlyViolation",
    "FitResult",
    "RunIdentity",
    "Trainer",
    "TrainerConfig",
    "build_param_groups",
    "warmup_cosine",
]

Batch = Any
LossFn = Callable[[torch.nn.Module, Batch], torch.Tensor]
BatchProvider = Callable[[int], Sequence[Batch]]
"""``epoch -> batches``. Must be a pure function of the epoch so that a resumed run sees the same data."""
MetricFn = Callable[[torch.nn.Module], Mapping[str, float]]
MetricSink = Callable[[dict[str, Any]], None]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        *,
        loss_fn: LossFn,
        train_batches: BatchProvider,
        validate: MetricFn,
        identity: RunIdentity,
        checkpoint_dir: str | Path,
        clock_ns: Callable[[], int],
        metric_sinks: Sequence[MetricSink] = (),
        resolved_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.seed_report: SeedReport = seed_everything(
            config.seed, deterministic=config.deterministic, intended_device=config.device
        )
        self.device = self.seed_report.device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.train_batches = train_batches
        self.validate = validate
        self.identity = identity
        self.checkpoint_dir = Path(checkpoint_dir)
        self.clock_ns = clock_ns
        self.metric_sinks = tuple(metric_sinks)
        self.resolved_config: dict[str, Any] = dict(resolved_config or config.model_dump(mode="json"))
        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, config), betas=config.betas, weight_decay=config.weight_decay
        )
        steps_per_epoch = math.ceil(len(train_batches(0)) / config.accumulation_steps)
        if steps_per_epoch < 1:
            raise ValueError("the training split is empty")
        self.total_steps = steps_per_epoch * config.max_epochs
        self.warmup_steps = round(config.warmup_fraction * self.total_steps)
        total, warm, floor = self.total_steps, self.warmup_steps, config.min_lr_ratio
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: warmup_cosine(s, total, warm, floor)
        )
        self.epoch = 0
        self.step = 0
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.history: list[EpochRecord] = []
        self.finalized = False
        self.stopped_early = False
        self.last_metadata: CheckpointMetadata | None = None

    # ---- identity -----------------------------------------------------------------------------
    def compatibility(self) -> CompatibilityTuple:
        return CompatibilityTuple(
            config_digest=config_digest(self.resolved_config),
            manifest_digests=self.identity.manifest_digests,
            split_hash=self.identity.split_hash,
            component=self.config.component,
            representation_pretraining_id=self.config.representation_pretraining_id,
            extra=self.identity.extra_compatibility,
        )

    # ---- training -----------------------------------------------------------------------------
    def _is_better(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.best_metric is None:
            return True
        return (
            value < self.best_metric if self.config.metric_direction == "lower" else value > self.best_metric
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        batches = self.train_batches(epoch)
        accumulate = self.config.accumulation_steps
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(batches):
            group_start = (index // accumulate) * accumulate
            group_size = min(accumulate, len(batches) - group_start)
            loss = self.loss_fn(self.model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}, batch {index}")
            (loss / group_size).backward()
            total_loss += float(loss.detach())
            if (index + 1) % accumulate == 0 or index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]], self.config.grad_clip_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
        return total_loss / max(len(batches), 1)

    def fit(self, max_epochs: int | None = None) -> FitResult:
        """Train up to ``max_epochs`` total epochs (default: config). Can be called again to continue."""
        if self.finalized:
            raise FinalOnlyViolation("training is finalized; final test metrics may already have been seen")
        limit = self.config.max_epochs if max_epochs is None else min(max_epochs, self.config.max_epochs)
        best_path: str | None = str(self.checkpoint_dir / "best.pt") if self.best_epoch is not None else None
        last_path: str | None = None
        while self.epoch < limit and not self.stopped_early:
            train_loss = self._train_epoch(self.epoch)
            self.model.eval()
            with torch.no_grad():
                metrics = {k: float(v) for k, v in self.validate(self.model).items()}
            not_validation = sorted(k for k in metrics if not is_validation_metric(k))
            if not_validation:
                raise FinalOnlyViolation(
                    f"validate() returned non-validation metrics during training: {not_validation}"
                )
            if self.config.primary_metric not in metrics:
                raise KeyError(f"validate() did not report primary metric {self.config.primary_metric!r}")
            value = metrics[self.config.primary_metric]
            improved = self._is_better(value)
            self.epoch += 1
            if improved:
                self.best_metric, self.best_epoch = value, self.epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            record = EpochRecord(
                epoch=self.epoch,
                step=self.step,
                train_loss=train_loss,
                lr=float(self.optimizer.param_groups[0]["lr"]),
                metrics=metrics,
                improved=improved,
            )
            self.history.append(record)
            for sink in self.metric_sinks:
                sink(
                    {
                        "epoch": record.epoch,
                        "step": record.step,
                        "train_loss": train_loss,
                        "lr": record.lr,
                        **metrics,
                    }
                )
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.stopped_early = True
            last_path = str(self.save(self.checkpoint_dir / "last.pt", metrics))
            if improved:
                best_path = str(self.save(self.checkpoint_dir / "best.pt", metrics))
        return FitResult(
            epochs_run=self.epoch,
            optimizer_steps=self.step,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            stopped_early=self.stopped_early,
            best_checkpoint=best_path,
            last_checkpoint=last_path,
            history=tuple(self.history),
        )

    # ---- checkpoints --------------------------------------------------------------------------
    def save(self, path: str | Path, metrics: Mapping[str, float]) -> Path:
        target = Path(path)
        self.last_metadata = save_checkpoint(
            target,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            component=self.config.component,
            config=self.resolved_config,
            manifest_digests=self.identity.manifest_digests,
            split_hash=self.identity.split_hash,
            seeds=self.seed_report.seeds(),
            metrics=metrics,
            epoch=self.epoch,
            step=self.step,
            created_time_ns=self.clock_ns(),
            git_commit=self.identity.git_commit,
            git_dirty=self.identity.git_dirty,
            device=self.device,
            device_fallback=self.seed_report.device_fallback,
            is_encoder=self.config.is_encoder,
            representation_pretraining_id=self.config.representation_pretraining_id,
            selection_metric=self.config.primary_metric,
            extra_compatibility=self.identity.extra_compatibility,
            trainer_state={
                "epoch": self.epoch,
                "step": self.step,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "epochs_without_improvement": self.epochs_without_improvement,
                "stopped_early": self.stopped_early,
                "rng": capture_rng_state(),
            },
        )
        return target

    def resume(self, path: str | Path) -> CheckpointMetadata:
        """Restore weights, optimizer, scheduler, counters and RNG. Fails closed on any identity mismatch."""
        if self.finalized:
            raise FinalOnlyViolation("cannot resume a finalized trainer")
        loaded = load_checkpoint(
            path, expected=self.compatibility(), model=self.model, map_location=self.device
        )
        if loaded.optimizer_state is None or loaded.scheduler_state is None:
            raise ValueError("checkpoint has no optimizer/scheduler state; it cannot resume training")
        self.optimizer.load_state_dict(loaded.optimizer_state)
        self.scheduler.load_state_dict(loaded.scheduler_state)
        state = loaded.trainer_state
        self.epoch = int(state["epoch"])
        self.step = int(state["step"])
        self.best_metric = None if state["best_metric"] is None else float(state["best_metric"])
        self.best_epoch = None if state["best_epoch"] is None else int(state["best_epoch"])
        self.epochs_without_improvement = int(state["epochs_without_improvement"])
        self.stopped_early = bool(state["stopped_early"])
        restore_rng_state(state["rng"])
        return loaded.metadata

    # ---- final-only guard ---------------------------------------------------------------------
    def finalize(self) -> None:
        """Freeze training. After this no further optimisation or checkpoint selection is possible."""
        if self.epoch == 0:
            raise FinalOnlyViolation("cannot finalize a trainer that never trained")
        self.finalized = True

    def evaluate_final(self, split: str, metric_fn: MetricFn) -> dict[str, float]:
        """Test / OOD metrics. Raises unless training was finalized first."""
        if not self.finalized:
            raise FinalOnlyViolation(
                f"{split!r} metrics requested before training was finalized; test and OOD are final-only"
            )
        self.model.eval()
        with torch.no_grad():
            metrics = {f"{split}_{k}": float(v) for k, v in metric_fn(self.model).items()}
        for sink in self.metric_sinks:
            sink({"epoch": self.epoch, "step": self.step, "final": True, **metrics})
        return metrics
