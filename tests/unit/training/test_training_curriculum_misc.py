from __future__ import annotations

import importlib

import pytest
import torch

from conrad.settings import REPO_ROOT
from conrad.training.curriculum import (
    Curriculum,
    CurriculumError,
    CurriculumRunner,
    CurriculumStage,
    StageStatus,
    apply_stage_freezing,
    load_curriculum,
)
from conrad.training.mlflow_logger import MlflowLogger
from conrad.training.trainer import RunIdentity, Trainer, TrainerConfig
from conrad.training.truncated import state_requires_grad, train_truncated_sequence


class Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.buo = torch.nn.Linear(2, 2)
        self.tbd = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.tbd(self.buo(self.encoder(x)))


def test_base_curriculum_has_c0_to_c9():
    cur = load_curriculum(REPO_ROOT / "configs" / "train" / "base_curriculum.yaml")
    assert [s.stage_id for s in cur.stages] == [f"C{i}" for i in range(10)]
    assert cur.stages[-1].requires_real_data


def test_stage_freezing_and_missing_pattern():
    model = Core()
    report = apply_stage_freezing(
        model, CurriculumStage(stage_id="C2", name="BUO", trainable_patterns=("buo.*",), max_epochs=1)
    )
    assert set(report.trainable) == {"buo.weight", "buo.bias"}
    assert not model.encoder.weight.requires_grad and model.buo.weight.requires_grad
    with pytest.raises(CurriculumError):
        apply_stage_freezing(
            model, CurriculumStage(stage_id="C4", name="RBP", trainable_patterns=("rbp.*",), max_epochs=1)
        )


def test_runner_trains_only_unfrozen_and_blocks_real_stage(tmp_path):
    x = torch.randn(8, 2, generator=torch.Generator().manual_seed(0))
    data = [(x, x.sum(1, keepdim=True))]

    def loss(m, b):
        return torch.nn.functional.mse_loss(m(b[0]), b[1])

    def factory(stage, model):
        cfg = TrainerConfig(
            component=f"core-{stage.stage_id}", batch_size=1, effective_batch_size=1, max_epochs=2
        )
        return Trainer(
            model,
            cfg,
            loss_fn=loss,
            train_batches=lambda e: data,
            validate=lambda m: {"val_loss": 0.0},
            identity=RunIdentity(manifest_digests=(), split_hash="s"),
            checkpoint_dir=tmp_path / stage.stage_id,
            clock_ns=lambda: 0,
        )

    model = Core()
    encoder_before = model.encoder.weight.detach().clone()
    cur = Curriculum(
        curriculum_id="t",
        stages=(
            CurriculumStage(stage_id="C2", name="BUO", trainable_patterns=("buo.*",), max_epochs=2),
            CurriculumStage(
                stage_id="C9", name="real", trainable_patterns=("*",), max_epochs=1, requires_real_data=True
            ),
        ),
    )
    results = CurriculumRunner(cur, factory).run(model)
    assert [r.status for r in results] == [StageStatus.COMPLETED, StageStatus.BLOCKED_EXTERNAL]
    assert torch.equal(model.encoder.weight, encoder_before)


def test_truncated_training_detaches_carried_state():
    cell = torch.nn.GRUCell(2, 3)
    opt = torch.optim.AdamW(cell.parameters(), lr=1e-2)
    seq = [torch.randn(1, 2, generator=torch.Generator().manual_seed(i)) for i in range(10)]

    def step(h, x):
        h2 = cell(x, h)
        return (h2**2).mean(), h2

    result, state = train_truncated_sequence(step, seq, torch.zeros(1, 3), window=4, optimizer=opt)
    assert result.optimizer_steps == 3 and result.steps_processed == 10
    assert not state_requires_grad(state)
    with pytest.raises(ValueError):
        train_truncated_sequence(step, seq, torch.zeros(1, 3), window=0, optimizer=opt)


def test_mlflow_disabled_or_broken_never_fails(tmp_path, monkeypatch):
    off = MlflowLogger(enabled=False, tracking_dir=tmp_path)
    assert off.start("r") is False
    off.log_metrics({"step": 1, "val_loss": 1.0})
    off.end()

    def boom(name):
        raise ImportError("mlflow missing")

    monkeypatch.setattr(importlib, "import_module", boom)
    broken = MlflowLogger(enabled=True, tracking_dir=tmp_path)
    assert broken.start("r") is False and "mlflow missing" in (broken.error or "")
    broken.log_metrics({"step": 1})
    broken.end()
