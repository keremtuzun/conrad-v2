"""Verified DINOv2 ViT-S/14 checkpoint import for the RGB encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from conrad.foundation.encoders.rgb import RGBViTS14Encoder

DINOV2_VITS14_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_VITS14_BYTES = 88_283_115


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dinov2_checkpoint(path: str | Path, *, expected_sha256: str = DINOV2_VITS14_SHA256) -> dict[str, Any]:
    """Verify the exact official artifact before deserializing it."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    size = checkpoint.stat().st_size
    digest = sha256_file(checkpoint)
    if digest != expected_sha256:
        raise ValueError(f"DINOv2 SHA-256 mismatch: expected {expected_sha256}, found {digest}")
    if size != DINOV2_VITS14_BYTES:
        raise ValueError(f"DINOv2 size mismatch: expected {DINOV2_VITS14_BYTES}, found {size}")
    return {"path": str(checkpoint), "sha256": digest, "byte_length": size}


def _load_state_dict(path: Path) -> dict[str, Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) for k in state):
        raise ValueError("DINOv2 checkpoint is not a tensor state dict")
    if not all(isinstance(v, Tensor) for v in state.values()):
        raise ValueError("DINOv2 checkpoint contains non-tensor state")
    return state


def _resize_pos_embed(pos_embed: Tensor, target_tokens: int) -> Tensor:
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1 or pos_embed.shape[2] != 384:
        raise ValueError(f"unexpected DINOv2 positional embedding shape {tuple(pos_embed.shape)}")
    source_grid = int((pos_embed.shape[1] - 1) ** 0.5)
    target_grid = int((target_tokens - 1) ** 0.5)
    if source_grid * source_grid != pos_embed.shape[1] - 1:
        raise ValueError("DINOv2 positional embedding has a non-square patch grid")
    if target_grid * target_grid != target_tokens - 1:
        raise ValueError("RGB encoder positional embedding has a non-square patch grid")
    if source_grid == target_grid:
        return pos_embed
    cls_token, patch_tokens = pos_embed[:, :1], pos_embed[:, 1:]
    patch_tokens = patch_tokens.reshape(1, source_grid, source_grid, 384).permute(0, 3, 1, 2)
    patch_tokens = torch.nn.functional.interpolate(patch_tokens, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
    return torch.cat([cls_token, patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, 384)], dim=1)


def import_dinov2_vits14(path: str | Path, encoder: RGBViTS14Encoder) -> dict[str, Tensor]:
    """Map official DINOv2 names to the frozen Conrad RGB encoder contract.

    DINOv2 ``ls*.gamma`` parameters are intentionally excluded because the local
    encoder uses the existing PyTorch transformer block without LayerScale.
    ``mask_token`` and distilled heads are also intentionally excluded.
    """
    verify_dinov2_checkpoint(path)
    source = _load_state_dict(Path(path))
    target = encoder.state_dict()
    mapped: dict[str, Tensor] = {
        "cls_token": source["cls_token"],
        "pos_embed": _resize_pos_embed(source["pos_embed"], target["pos_embed"].shape[1]),
        "patch_embed.weight": source["patch_embed.proj.weight"],
        "patch_embed.bias": source["patch_embed.proj.bias"],
        "norm.weight": source["norm.weight"],
        "norm.bias": source["norm.bias"],
    }
    for index in range(len(encoder.blocks)):
        source_prefix = f"blocks.{index}"
        target_prefix = source_prefix
        mapped.update(
            {
                f"{target_prefix}.self_attn.in_proj_weight": source[f"{source_prefix}.attn.qkv.weight"],
                f"{target_prefix}.self_attn.in_proj_bias": source[f"{source_prefix}.attn.qkv.bias"],
                f"{target_prefix}.self_attn.out_proj.weight": source[f"{source_prefix}.attn.proj.weight"],
                f"{target_prefix}.self_attn.out_proj.bias": source[f"{source_prefix}.attn.proj.bias"],
                f"{target_prefix}.linear1.weight": source[f"{source_prefix}.mlp.fc1.weight"],
                f"{target_prefix}.linear1.bias": source[f"{source_prefix}.mlp.fc1.bias"],
                f"{target_prefix}.linear2.weight": source[f"{source_prefix}.mlp.fc2.weight"],
                f"{target_prefix}.linear2.bias": source[f"{source_prefix}.mlp.fc2.bias"],
                f"{target_prefix}.norm1.weight": source[f"{source_prefix}.norm1.weight"],
                f"{target_prefix}.norm1.bias": source[f"{source_prefix}.norm1.bias"],
                f"{target_prefix}.norm2.weight": source[f"{source_prefix}.norm2.weight"],
                f"{target_prefix}.norm2.bias": source[f"{source_prefix}.norm2.bias"],
            }
        )
    missing = sorted(set(target) - set(mapped))
    incompatible = sorted(key for key, value in mapped.items() if key in target and tuple(value.shape) != tuple(target[key].shape))
    if missing or incompatible:
        raise ValueError(f"DINOv2 import contract failed: missing={missing}, incompatible={incompatible}")
    return mapped


def write_provenance(path: str | Path, output: str | Path) -> None:
    verification = verify_dinov2_checkpoint(path)
    Path(output).write_text(
        json.dumps(
            {
                "model_id": "dinov2_vits14_pretrain",
                "source_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
                "official_repository": "https://github.com/facebookresearch/dinov2",
                "license": "FAIR Noncommercial Research License",
                "usage_notes": "Noncommercial research use; retain Meta attribution and license terms.",
                "retrieved_at": "2026-09-26",
                "importer": "conrad.foundation.pretraining.dinov2.import_dinov2_vits14",
                "excluded_or_reinitialized": ["mask_token", "blocks.*.ls*.gamma", "distillation heads"],
                **verification,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
