"""ECMER training stages E1-E4 as callable functions (ch3 ECMER training, ch4 Training ECMER).

Each stage takes a seeded ``batch_fn(step) -> EventBatch`` and returns the per-step loss history.
AdamW with the ch33 learning rate/weight decay and grad clipping come from :class:`CoreConfig`.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.ecmer.fusion import MODALITIES, EcmerModel, EventBatch, sample_modality_dropout
from conrad.core.ecmer.ssl import ReprObjective, corrupt_images, info_nce
from conrad.core.primitives import masked_loss
from conrad.core.uncertainty import heteroscedastic_nll

BatchFn = Callable[[int], EventBatch]


def _optimizer(params: list[nn.Parameter], cfg: CoreConfig, lr: float | None = None) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, lr=lr or cfg.learning_rate, weight_decay=cfg.weight_decay)


def _step(opt: torch.optim.Optimizer, loss: Tensor, params: list[nn.Parameter], cfg: CoreConfig) -> float:
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
    opt.step()
    return float(loss.detach())


def corrupt_batch(batch: EventBatch, severity: Tensor, generator: torch.Generator) -> EventBatch:
    """Apply known-severity degradation to image modalities; other modalities pass unchanged."""
    return replace(
        batch,
        rgb=None if batch.rgb is None else corrupt_images(batch.rgb, severity, generator),
        sonar=None if batch.sonar is None else corrupt_images(batch.sonar, severity, generator),
    )


def _avail(batch: EventBatch) -> Tensor:
    return torch.stack([batch.available(n) for n in MODALITIES], dim=-1)


def stage_e1_representation(
    model: EcmerModel, objective: ReprObjective, batch_fn: BatchFn, steps: int, seed: int
) -> list[float]:
    """Two lightly augmented views + masked-modality reconstruction (L_view + L_masked)."""
    cfg = model.cfg
    gen = torch.Generator().manual_seed(seed)
    params = [*model.parameters(), *objective.parameters()]
    opt = _optimizer(params, cfg)
    model.train()
    history: list[float] = []
    for step in range(steps):
        batch = batch_fn(step)
        sev = torch.rand(batch.batch_size, generator=gen) * 0.3
        a = model(corrupt_batch(batch, sev, gen))
        b = model(corrupt_batch(batch, sev.flip(0), gen))
        avail = _avail(batch)
        drop = sample_modality_dropout(avail, cfg.ecmer.modality_dropout_p, gen)
        masked = model(batch, drop_modalities=drop)
        targets = torch.stack(
            [a.modality_tokens.get(n, a.event.new_zeros(a.event.shape)) for n in MODALITIES], dim=1
        )
        loss = objective(a.event, b.event, event_masked=masked.event, target_tokens=targets, target_mask=drop)
        history.append(_step(opt, loss.total, params, cfg))
    return history


def stage_e2_degradation(model: EcmerModel, batch_fn: BatchFn, steps: int, seed: int) -> list[float]:
    """Invariance of e_k between clean and strongly degraded views (known severity)."""
    cfg = model.cfg
    gen = torch.Generator().manual_seed(seed)
    params = list(model.parameters())
    opt = _optimizer(params, cfg)
    model.train()
    history: list[float] = []
    for step in range(steps):
        batch = batch_fn(step)
        sev = torch.rand(batch.batch_size, generator=gen)
        clean, bad = model(batch), model(corrupt_batch(batch, sev, gen))
        loss, _ = info_nce(clean.event, bad.event, cfg.ecmer.info_nce_temperature)
        history.append(_step(opt, loss, params, cfg))
    return history


def stage_e3_quality(model: EcmerModel, batch_fn: BatchFn, steps: int, seed: int) -> list[float]:
    """Quality heads against CONTROLLED degradation: reliability Brier vs (1 - severity), usability BCE,
    and heteroscedastic NLL of the embedding shift caused by the corruption."""
    cfg = model.cfg
    gen = torch.Generator().manual_seed(seed)
    params = list(model.parameters())
    opt = _optimizer(params, cfg)
    model.train()
    history: list[float] = []
    for step in range(steps):
        batch = batch_fn(step)
        sev = torch.rand(batch.batch_size, generator=gen)
        clean = model(batch)
        bad = model(corrupt_batch(batch, sev, gen))
        imaging = _avail(batch)[:, :2].any(-1)  # severity only defined where an image was corrupted
        brier, _ = masked_loss((bad.reliability - (1 - sev)) ** 2, imaging)
        usable_target = (sev < 0.5).to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(bad.usable_logit, usable_target, reduction="none")
        use, _ = masked_loss(bce, imaging)
        resid = bad.event - clean.event.detach()
        per_row = heteroscedastic_nll(torch.zeros_like(resid), bad.log_var_a.unsqueeze(-1), resid).mean(-1)
        nll, _ = masked_loss(per_row, imaging)
        history.append(_step(opt, brier + use + nll, params, cfg))
    return history


def stage_e4_correspondence(
    model: EcmerModel, objective: ReprObjective, batch_fn: BatchFn, steps: int, seed: int
) -> list[float]:
    """Cross-modal correspondence: RGB and sonar tokens of the SAME event are positives (row-aligned)."""
    cfg = model.cfg
    params = [*model.parameters(), *objective.parameters()]
    opt = _optimizer(params, cfg)
    model.train()
    history: list[float] = []
    torch.manual_seed(seed)
    for step in range(steps):
        batch = batch_fn(step)
        out = model(batch)
        both = batch.available("rgb") & batch.available("sonar")
        if "rgb" not in out.modality_tokens or "sonar" not in out.modality_tokens or not bool(both.any()):
            history.append(0.0)
            continue
        loss = objective(
            out.event,
            out.event,
            modality_a=out.modality_tokens["rgb"],
            modality_b=out.modality_tokens["sonar"],
            cross_mask=both,
        )
        history.append(_step(opt, loss.cross_modal, params, cfg))
    return history
