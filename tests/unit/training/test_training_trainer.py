from __future__ import annotations

import pytest
import torch

from conrad.training.checkpoint import CheckpointError, ReasonCode
from conrad.training.determinism import inspect_compute, resolve_device
from conrad.training.trainer import (
    FinalOnlyViolation,
    RunIdentity,
    Trainer,
    TrainerConfig,
    build_param_groups,
    warmup_cosine,
)

IDENTITY = RunIdentity(manifest_digests=("m" * 64,), split_hash="s" * 64)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 8)
        self.drop = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.drop(torch.relu(self.encoder(x))))


def batches(epoch: int):
    g = torch.Generator().manual_seed(1000 + epoch)
    out = []
    for _ in range(6):
        x = torch.randn(4, 4, generator=g)
        out.append((x, x.sum(dim=1, keepdim=True)))
    return out


VAL = batches(-1)


def loss_fn(model, batch):
    x, y = batch
    return torch.nn.functional.mse_loss(model(x), y)


def validate(model):
    return {"val_loss": float(sum(loss_fn(model, b) for b in VAL) / len(VAL))}


def make_trainer(tmp_path, name: str, **cfg) -> Trainer:
    torch.manual_seed(0)
    config = TrainerConfig(
        component="tiny", batch_size=1, effective_batch_size=2, max_epochs=6, seed=5, **cfg
    )
    return Trainer(
        Tiny(),
        config,
        loss_fn=loss_fn,
        train_batches=batches,
        validate=validate,
        identity=IDENTITY,
        checkpoint_dir=tmp_path / name,
        clock_ns=lambda: 1,
    )


def test_resume_reproduces_subsequent_losses(tmp_path):
    straight = make_trainer(tmp_path, "a").fit(max_epochs=4)
    first = make_trainer(tmp_path, "b")
    first.fit(max_epochs=2)
    resumed = make_trainer(tmp_path, "c")
    resumed.resume(tmp_path / "b" / "last.pt")
    result = resumed.fit(max_epochs=4)
    expect = [(r.train_loss, r.metrics["val_loss"]) for r in straight.history[2:]]
    got = [(r.train_loss, r.metrics["val_loss"]) for r in result.history]
    assert got == expect
    assert result.optimizer_steps == straight.optimizer_steps == 12  # 6 batches / accumulate 2 * 4 epochs


def test_resume_rejects_other_split(tmp_path):
    make_trainer(tmp_path, "b").fit(max_epochs=1)
    other = make_trainer(tmp_path, "c")
    other.identity = RunIdentity(manifest_digests=IDENTITY.manifest_digests, split_hash="other")
    with pytest.raises(CheckpointError) as info:
        other.resume(tmp_path / "b" / "last.pt")
    assert ReasonCode.SPLIT_HASH_MISMATCH in info.value.codes


def test_final_only_guard(tmp_path):
    trainer = make_trainer(tmp_path, "g")
    with pytest.raises(FinalOnlyViolation):
        trainer.evaluate_final("test", validate)
    with pytest.raises(FinalOnlyViolation):
        trainer.finalize()
    trainer.fit(max_epochs=1)
    trainer.finalize()
    assert "test_val_loss" in trainer.evaluate_final("test", validate)
    with pytest.raises(FinalOnlyViolation):
        trainer.fit()
    with pytest.raises(FinalOnlyViolation):
        trainer.resume(tmp_path / "g" / "last.pt")


def test_selection_only_on_validation(tmp_path):
    with pytest.raises(ValueError):
        TrainerConfig(component="x", primary_metric="test_loss")
    trainer = make_trainer(tmp_path, "v")
    trainer.validate = lambda m: {"val_loss": 1.0, "test_loss": 0.1}
    with pytest.raises(FinalOnlyViolation):
        trainer.fit(max_epochs=1)


def test_early_stopping_and_best_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path, "e", early_stopping_patience=2)
    trainer.validate = lambda m: {"val_loss": 1.0}
    result = trainer.fit()
    assert result.stopped_early and result.epochs_run == 3 and result.best_epoch == 1
    assert (tmp_path / "e" / "best.pt").exists()


def test_schedule_and_param_groups():
    assert warmup_cosine(0, 100, 5, 0.0) == pytest.approx(0.2)
    assert warmup_cosine(4, 100, 5, 0.0) == pytest.approx(1.0)
    assert warmup_cosine(100, 100, 5, 0.0) == pytest.approx(0.0)
    cfg = TrainerConfig(component="x", pretrained_patterns=("encoder.*",))
    groups = {g["group_name"]: g["lr"] for g in build_param_groups(Tiny(), cfg)}
    assert groups == {"new": 3e-4, "pretrained": 1e-4}
    assert TrainerConfig(component="x").accumulation_steps == 4


def test_device_fallback_is_recorded(tmp_path):
    compute = inspect_compute()
    assert compute.compute_tier in ("LOCAL_CPU", "REMOTE_OR_LOCAL_GPU")
    if compute.cuda_available:
        pytest.skip("GPU present; fallback path not exercised")
    assert resolve_device("cuda") == ("cpu", True)
    trainer = make_trainer(tmp_path, "d", device="cuda")
    assert trainer.seed_report.device_fallback and trainer.device == "cpu"
    trainer.fit(max_epochs=1)
    assert trainer.last_metadata is not None and trainer.last_metadata.device_info["device_fallback"] is True
