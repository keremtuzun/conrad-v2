"""Reviewed Universal V1.1 10P trainer surface.

This is intentionally narrow: it trains the current UniversalOSFMV11 interface on
verified SubPipe rendered-sonar payloads, records immutable run evidence, and
keeps semantic claims limited to data-backed modalities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from conrad.data.adapters.subpipe import SubPipeAdapter
from conrad.data.manifest import data_root_for, load_manifest
from conrad.foundation.data.manifest import CorpusPartition
from conrad.foundation.data.osfm_public_real import SUBPIPE_MANIFEST, _partition_ranges
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11.core import (
    FamilyEncoderConfig,
    SensorMetadata,
    UniversalModalityInput,
    UniversalOSFMV11,
)
from conrad.foundation.universal_v11.registry import MODALITY_REGISTRY, EncoderFamily, ModalityState
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory, RunPurpose, TerminalStatus, environment_snapshot

PAYLOAD_MANIFEST = REPO_ROOT / "artifacts/gates/V1.1/10P_KEREM_HANDOFF/required_payloads.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_required_payloads(manifest_path: str | Path = PAYLOAD_MANIFEST) -> dict[str, Any]:
    """Verify the ignored binary payloads needed before formal V1.1 10P training."""

    path = Path(manifest_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for payload in manifest["payloads"]:
        payload_path = REPO_ROOT / payload["path"]
        present = payload_path.is_file()
        actual = _sha256(payload_path) if present else None
        ok = present and actual == payload["sha256"]
        if not ok:
            blockers.append(f"{payload['id']} missing or hash mismatch")
        results.append(
            {
                "id": payload["id"],
                "path": payload["path"],
                "present": present,
                "expected_sha256": payload["sha256"],
                "actual_sha256": actual,
                "sha256_ok": ok,
            }
        )
    return {
        "gate_id": "OSFM-V11-10P-PAYLOAD-PREFLIGHT",
        "decision": "PASS" if not blockers else "BLOCKED",
        "payloads": results,
        "blockers": blockers,
    }


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _device(require_cuda: bool) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("formal V1.1 10P training requires CUDA unless --allow-cpu-smoke is set")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _subpipe_frame_refs(stream: str, partition: CorpusPartition, frame_stride: int) -> tuple[Any, ...]:
    manifest = load_manifest(SUBPIPE_MANIFEST)
    root = data_root_for(manifest.dataset_id)
    adapter = SubPipeAdapter(
        manifest,
        root,
        ObjectStore(REPO_ROOT / "artifacts/tmp/v11_10p_object_store"),
        IdFactory(seed=2026092912),
        UUID("00000000-0000-7000-8000-000000001011"),
        streams=(stream,),
        frame_stride=frame_stride,
        manifest_path=SUBPIPE_MANIFEST,
    )
    try:
        refs = adapter.frames(stream)[::frame_stride]
    finally:
        adapter.close()
    start, end = _partition_ranges(len(refs))[partition]
    selected = tuple(refs[start:end])
    if not selected:
        raise RuntimeError(f"SubPipe {stream} {partition.value} partition is empty")
    return selected


# Kerem variant: the reviewed trainer reduced each frame to 32 row-band means, which carries almost no
# frame-to-frame variation and pinned validation rank near 4. Each frame is now a PATCH_GRID x PATCH_GRID
# grid of real image patches (16 tokens = GenericTokenizer.max_tokens), in the spirit of the P4.8 sonar run.
IMAGE_SIZE = 64
PATCH_GRID = 4
PATCH_DIM = (IMAGE_SIZE // PATCH_GRID) ** 2
TRAINER_VARIANT = "kerem_v11_10p_subpipe_full_sonar_camera_cleananchor_v5"
BASE_LR = float(os.environ.get("KEREM_LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("KEREM_WEIGHT_DECAY", "0.05"))
# The project's V1.1 is FamilyEncoderConfig depth 6 (307.8M). The reviewed trainer used depth 2 (201.3M).
FAMILY_DEPTH = int(os.environ.get("KEREM_FAMILY_DEPTH", "6"))
# Top-level modules held frozen (config stage A trains adapters/router/small heads only).
FROZEN_MODULES = tuple(m for m in os.environ.get("KEREM_FREEZE", "").split(",") if m)
WARMUP_STEPS = 2000
VICREG_WEIGHTS = {"invariance": 25.0, "variance": 25.0, "covariance": 1.0, "rank": 1.0}
TOKEN_MASK_FRACTION = 0.25
CLEAN_ANCHOR_VIEW = os.environ.get("KEREM_CLEAN_ANCHOR", "1") == "1"
# Readiness blockers that only say "this is not the reviewed trainer on osfm-universal-v1.1".
# An explicit owner override may launch past exactly these, and the override is recorded in the run.
UNREVIEWED_TRAINER_BLOCKER_IDS = frozenset({2, 4, 8})


def readiness_override_record(readiness: dict[str, Any]) -> dict[str, Any]:
    blockers = [item for item in readiness.get("items", []) if item.get("status") == "BLOCKER"]
    ids = {int(item["id"]) for item in blockers}
    if not ids or not ids <= UNREVIEWED_TRAINER_BLOCKER_IDS:
        raise RuntimeError(f"override only covers unreviewed-trainer blockers {sorted(UNREVIEWED_TRAINER_BLOCKER_IDS)}; got {sorted(ids)}")
    return {
        "readiness_decision": readiness.get("decision"),
        "overridden_blockers": [{"id": b["id"], "title": b["title"], "evidence": b["evidence"]} for b in blockers],
        "authorized_by": "Kerem (repository owner); reviewer Burak unavailable",
        "authorized_on": "2026-09-29",
        "reason": "reviewed trainer benchmarked NO-GO (val rank 4.06 < 75); Kerem variant trainer launched instead",
    }


CACHE_SIZE = 128  # frames are decoded once and kept on the device at this resolution
CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)

# Full SubPipe release (same Zenodo record/licence as SubPipeMini2): chunks 0-4 of side-scan sonar and cameras.
SUBPIPE_FULL_MANIFEST = REPO_ROOT / "datasets/public/subpipe_full.manifest.yaml"
SUBPIPE_FULL_ARCHIVE = REPO_ROOT / "artifacts/data/public.subpipe_full/raw/SubPipe.zip"
SUBPIPE_FULL_CACHE_DIR = REPO_ROOT / "artifacts/tmp/subpipe_full_cache"
_FULL_MEMBER = re.compile(
    r"^SubPipe/DATA/Chunk(\d+)/(Cam0_images|Cam1_images|SSS_LF_images/Image|SSS_HF_images/Image)/(\d+(?:\.\d+)?)\.(jpg|pbm)$"
)
_FOLDER_STREAM = {"Cam0_images": "cam0", "Cam1_images": "cam1", "SSS_LF_images/Image": "sss_lf", "SSS_HF_images/Image": "sss_hf"}
# modality -> channels and per-stream frame stride (cam0 runs at ~30 fps, so every 2nd frame is kept)
MODALITIES: dict[str, dict[str, Any]] = {
    "imaging_sonar": {"channels": 1, "streams": {"sss_lf": 1, "sss_hf": 1}},
    "rgb_camera": {"channels": 3, "streams": {"cam0": 2, "cam1": 1}},
}


def _patch_dim(channels: int) -> int:
    return (IMAGE_SIZE // PATCH_GRID) ** 2 * channels


def verify_subpipe_full() -> dict[str, Any]:
    manifest = yaml.safe_load(SUBPIPE_FULL_MANIFEST.read_text(encoding="utf-8")) if SUBPIPE_FULL_MANIFEST.is_file() else {}
    entry = next((f for f in manifest.get("files", []) if f.get("path") == "raw/SubPipe.zip"), {})
    present = SUBPIPE_FULL_ARCHIVE.is_file()
    actual = _sha256(SUBPIPE_FULL_ARCHIVE) if present else None
    ok = bool(entry) and present and actual == entry.get("sha256")
    return {
        "gate_id": "OSFM-V11-10P-SUBPIPE-FULL-PREFLIGHT",
        "decision": "PASS" if ok else "FAIL",
        "manifest": str(SUBPIPE_FULL_MANIFEST.relative_to(REPO_ROOT)),
        "path": str(SUBPIPE_FULL_ARCHIVE.relative_to(REPO_ROOT)),
        "expected_sha256": entry.get("sha256"),
        "actual_sha256": actual,
    }


def _full_index() -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {stream: [] for stream in _FOLDER_STREAM.values()}
    with zipfile.ZipFile(SUBPIPE_FULL_ARCHIVE) as archive:
        for name in archive.namelist():
            match = _FULL_MEMBER.match(name)
            if match:
                # epoch seconds, with or without a fractional part (the full release has both)
                out[_FOLDER_STREAM[match.group(2)]].append((round(float(match.group(3)) * 1.0e9), name))
    return {stream: sorted(refs) for stream, refs in out.items()}


_WORKER_ZIP: zipfile.ZipFile | None = None


def _decode_member(job: tuple[str, int]) -> np.ndarray:
    import cv2

    global _WORKER_ZIP
    member, channels = job
    if _WORKER_ZIP is None:
        _WORKER_ZIP = zipfile.ZipFile(SUBPIPE_FULL_ARCHIVE)
    raw = np.frombuffer(_WORKER_ZIP.read(member), dtype=np.uint8)  # CRC-32 checked by zipfile
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"{member}: bytes do not decode as an image")
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    image = image[None] if channels == 1 else image.transpose(2, 0, 1)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _patchify(images: torch.Tensor) -> torch.Tensor:
    """[B, C, IMAGE_SIZE, IMAGE_SIZE] -> [B, PATCH_GRID**2, C * patch_side**2]."""
    batch, channels = images.shape[:2]
    side = IMAGE_SIZE // PATCH_GRID
    patches = images.reshape(batch, channels, PATCH_GRID, side, PATCH_GRID, side).permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch, PATCH_GRID * PATCH_GRID, channels * side * side).contiguous()


def _standardize(images: torch.Tensor) -> torch.Tensor:
    """Per-frame brightness/contrast normalisation (sonar gain varies strongly along the mission;
    the held-out block has ~3x lower pixel contrast than the training block)."""
    mean = images.mean(dim=(-3, -2, -1), keepdim=True)
    std = images.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0e-3)
    return (images - mean) / std


def _clean_view(images: torch.Tensor) -> torch.Tensor:
    return _patchify(_standardize(F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="area")))


def _random_resized_crop(images: torch.Tensor, generator: torch.Generator, scale: tuple[float, float]) -> torch.Tensor:
    """Per-sample random-resized crop + horizontal mirror, on the images' device."""
    batch, device = images.shape[0], images.device
    area = torch.empty(batch, device=device).uniform_(*scale, generator=generator)
    log_ratio = torch.empty(batch, device=device).uniform_(math.log(CROP_RATIO[0]), math.log(CROP_RATIO[1]), generator=generator)
    ratio = log_ratio.exp()
    width = (area * ratio).sqrt().clamp(max=1.0)
    height = (area / ratio).sqrt().clamp(max=1.0)
    cx = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * (1 - width)
    cy = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * (1 - height)
    flip = torch.where(torch.rand(batch, device=device, generator=generator) < 0.5, -1.0, 1.0)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = width * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = height
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (batch, images.shape[1], IMAGE_SIZE, IMAGE_SIZE), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _train_view(images: torch.Tensor, generator: torch.Generator, view: dict[str, Any]) -> torch.Tensor:
    """Crop -> photometrics (gain, per-channel colour gain, speckle, additive noise) -> standardise -> patch dropout."""
    crops = _random_resized_crop(images, generator, view["crop_scale"])
    batch, channels, device = crops.shape[0], crops.shape[1], crops.device
    gain = 0.6 + 0.8 * torch.rand((batch, 1, 1, 1), generator=generator, device=device)
    if view.get("channel_gain", 0.0) > 0 and channels > 1:
        gain = gain * (1.0 + view["channel_gain"] * torch.randn((batch, channels, 1, 1), generator=generator, device=device))
    speckle = 1.0 + view["speckle"] * torch.randn(crops.shape, generator=generator, device=device)
    noise = view["noise"] * torch.randn(crops.shape, generator=generator, device=device)
    patches = _patchify(_standardize((crops * gain * speckle + noise).clamp(0.0, 1.0)))
    if view["patch_dropout"] > 0:
        keep = torch.rand((batch, patches.shape[1], 1), generator=generator, device=device) >= view["patch_dropout"]
        patches = patches * keep
    return patches.float()


# Asymmetric views: the rank term acts on the light view, so near-clean frames must spread out.
VIEWS: dict[str, dict[str, dict[str, Any]]] = {
    "imaging_sonar": {
        "light": {"crop_scale": (0.60, 1.0), "speckle": 0.05, "noise": 0.01, "patch_dropout": 0.0},
        "strong": {"crop_scale": (0.30, 1.0), "speckle": 0.10, "noise": 0.03, "patch_dropout": TOKEN_MASK_FRACTION},
    },
    "rgb_camera": {
        "light": {"crop_scale": (0.60, 1.0), "speckle": 0.0, "noise": 0.01, "patch_dropout": 0.0, "channel_gain": 0.05},
        "strong": {"crop_scale": (0.25, 1.0), "speckle": 0.0, "noise": 0.03, "patch_dropout": TOKEN_MASK_FRACTION, "channel_gain": 0.15},
    },
}


def _vicreg_terms(z_a: torch.Tensor, z_b: torch.Tensor, rank_floor: float) -> dict[str, torch.Tensor]:
    z_a = z_a.float()
    z_b = z_b.float()
    invariance = F.mse_loss(z_a, z_b)

    def _var(z: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1.0e-4)).mean()

    def _cov(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        centered = z - z.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(n - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return off.pow(2).sum() / d

    rank_loss, _ = rank_diversity_loss(z_a, target=rank_floor)
    variance = _var(z_a) + _var(z_b)
    covariance = _cov(z_a) + _cov(z_b)
    total = (
        VICREG_WEIGHTS["invariance"] * invariance
        + VICREG_WEIGHTS["variance"] * variance
        + VICREG_WEIGHTS["covariance"] * covariance
        + VICREG_WEIGHTS["rank"] * rank_loss
    )
    return {"total": total, "invariance": invariance, "variance": variance, "covariance": covariance, "rank": rank_loss}


class _SubPipeFullBatcher:
    """Every usable frame of the full SubPipe release, per modality, partitioned per stream by contiguous
    time blocks. Frames are decoded once (cached on disk, keyed by the archive hash) and held on the device;
    validation uses distinct frames only."""

    def __init__(self, *, archive_sha256: str, batch_size: int, device: torch.device, seed: int, smoke: bool = False) -> None:
        index = _full_index()
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.data: dict[str, dict[str, Any]] = {}
        for modality, spec in MODALITIES.items():
            train: list[tuple[str, str]] = []
            val: list[tuple[str, str]] = []
            counts: dict[str, dict[str, int]] = {}
            for stream, stride in spec["streams"].items():
                refs = index[stream][:: stride * (200 if smoke else 1)]
                ranges = _partition_ranges(len(refs))
                t0, t1 = ranges[CorpusPartition.PRETRAIN_REAL]
                v0, v1 = ranges[CorpusPartition.VALIDATION]
                train += [(stream, member) for _, member in refs[t0:t1]]
                val += [(stream, member) for _, member in refs[v0:v1]]
                counts[stream] = {"all": len(refs), "train": t1 - t0, "validation": v1 - v0, "stride": stride}
            if not train or not val:
                raise RuntimeError(f"{modality}: PRETRAIN_REAL and VALIDATION partitions must both be non-empty")
            self.data[modality] = {"channels": spec["channels"], "train": tuple(train), "val": tuple(val), "counts": counts}
        self._load_frames(archive_sha256)

    def _load_frames(self, archive_sha256: str) -> None:
        from multiprocessing import get_context

        for modality, entry in self.data.items():
            members = [member for _, member in entry["train"] + entry["val"]]
            key = hashlib.sha256(("\n".join([archive_sha256, str(CACHE_SIZE), *members])).encode()).hexdigest()[:20]
            cache = SUBPIPE_FULL_CACHE_DIR / f"{modality}_{CACHE_SIZE}_{key}.npy"
            if cache.is_file():
                frames = np.load(cache)
            else:
                with get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
                    frames = np.stack(pool.map(_decode_member, [(m, entry["channels"]) for m in members], chunksize=16))
                SUBPIPE_FULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache, frames)
            entry["frames"] = torch.from_numpy(frames).to(self.device)
            entry["train_index"] = torch.arange(len(entry["train"]), device=self.device)
            entry["val_index"] = torch.arange(len(entry["train"]), len(members), device=self.device)
            entry["cache"] = str(cache.relative_to(REPO_ROOT))

    def coverage(self) -> dict[str, Any]:
        return {
            modality: {
                "streams": entry["counts"],
                "train_frames": len(entry["train"]),
                "validation_frames": len(entry["val"]),
                "frame_cache": entry["cache"],
            }
            for modality, entry in self.data.items()
        }

    def close(self) -> None:
        return None

    def batch(self, modality: str, split: str, step: int, *, distinct: bool = False) -> tuple[torch.Tensor, tuple[str, ...]]:
        entry = self.data[modality]
        refs, index = (entry["train"], entry["train_index"]) if split == "train" else (entry["val"], entry["val_index"])
        gen = torch.Generator().manual_seed(self.seed + step)
        order = torch.randperm(len(refs), generator=gen).tolist()
        size = min(self.batch_size, len(order)) if distinct else self.batch_size
        picks = [order[idx % len(order)] for idx in range(size)]
        images = entry["frames"][index[torch.tensor(picks, device=self.device)]].float() / 255.0
        sample_ids = tuple(f"{refs[i][0]}:{refs[i][1]}" for i in picks)
        return images, sample_ids


def _make_input(name: str, payload: torch.Tensor, step: int) -> UniversalModalityInput:
    return UniversalModalityInput(
        name=name,
        payload=payload,
        state=ModalityState.AVAILABLE,
        timestamp_s=torch.full((payload.shape[0],), float(step), dtype=torch.float32, device=payload.device),
        metadata=SensorMetadata(
            quality=0.9,
            calibrated=False,
            acquisition_time_s=float(step),
            provenance=f"subpipe-full-{name}-hash-verified",
            units="normalized_image_features",
        ),
    )


def _representation_health(reprs: torch.Tensor, rank_floor: float) -> dict[str, Any]:
    finite = bool(torch.isfinite(reprs).all().detach().cpu())
    collapse = float(reprs.float().std().detach().cpu())
    _, rank = rank_diversity_loss(reprs.float(), target=rank_floor)
    effective_rank = float(rank.detach().cpu())
    return {
        "finite": finite,
        "collapse_score": collapse,
        "effective_rank": effective_rank,
        "rank_floor": rank_floor,
        "rank_pass": finite and collapse > 1.0e-6 and effective_rank >= rank_floor,
    }


def run_v11_10p_training(
    *,
    config_path: str | Path,
    readiness_path: str | Path,
    runs_root: str | Path = "artifacts/runs",
    run_id: str | None = None,
    max_steps_override: int | None = None,
    resume: str | Path | None = None,
    allow_cpu_smoke: bool = False,
    readiness_override: bool = False,
) -> dict[str, Any]:
    readiness = json.loads((REPO_ROOT / readiness_path if not Path(readiness_path).is_absolute() else Path(readiness_path)).read_text(encoding="utf-8"))
    override_record: dict[str, Any] | None = None
    if readiness.get("decision") != "READY FOR KEREM":
        if not readiness_override:
            raise RuntimeError("V1.1 10P training requires READY FOR KEREM readiness output")
        override_record = readiness_override_record(readiness)
    payload_report = verify_required_payloads()
    if payload_report["decision"] != "PASS":
        raise RuntimeError(f"required payloads are not verified: {payload_report['blockers']}")
    full_report = verify_subpipe_full()
    if full_report["decision"] != "PASS" and not allow_cpu_smoke:
        raise RuntimeError(f"full SubPipe archive is not hash-verified against its manifest: {full_report}")

    cfg = _load_config(config_path)
    steps = int(max_steps_override or cfg["training_budget"]["total_optimizer_steps"])
    batch_size = int(cfg["training_budget"].get("effective_batch_target", 256))
    if allow_cpu_smoke:
        batch_size = min(batch_size, 2)
    rank_floor = float(cfg["training_budget"].get("rank_floor", 75.0))
    checkpoint_every = int(cfg["training_budget"].get("checkpoint_every_steps", 5000))
    validation_every = int(cfg["training_budget"].get("validation_every_steps", 1000))
    seed = int(cfg.get("seed", 2026092910))
    device = _device(require_cuda=not allow_cpu_smoke)

    run_name = run_id or f"v11-10p-{time.time_ns()}"
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    variant = {
        "trainer_variant": TRAINER_VARIANT,
        "image_size": IMAGE_SIZE,
        "patch_grid": PATCH_GRID,
        "patch_dim": PATCH_DIM,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "family_depth": FAMILY_DEPTH,
        "frozen_modules": list(FROZEN_MODULES),
        "warmup_steps": WARMUP_STEPS,
        "vicreg_weights": VICREG_WEIGHTS,
        "token_mask_fraction": TOKEN_MASK_FRACTION,
        "dataset": "SubPipe full release (Zenodo 12666132 v3.0.1, CC-BY-4.0), chunks 0-4",
        "modalities": json.loads(json.dumps(MODALITIES)),
        "views": json.loads(json.dumps(VIEWS)),
        "step_schedule": "odd steps imaging_sonar, even steps rgb_camera",
        "clean_anchor_view": CLEAN_ANCHOR_VIEW,
        "gate": "every modality's validation rank >= rank_floor (config: every_family_rank_ge)",
        "validation_batch": "distinct_frames",
        "cache_size": CACHE_SIZE,
        "per_frame_standardize": True,
        "crop_ratio": list(CROP_RATIO),
        "mirror_flip": True,
        "autocast": "bf16" if use_bf16 else "fp32",
    }
    env = environment_snapshot(device=str(device), precision="bf16-autocast" if use_bf16 else "float32")
    run = RunDirectory.create(
        runs_root,
        run_name,
        resolved_config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke, "kerem_variant": variant},
        manifests={
            "payloads": payload_report,
            "subpipe_manifest": str(SUBPIPE_MANIFEST.relative_to(REPO_ROOT)),
            "readiness_override": override_record,
            "subpipe_full": full_report,
        },
        purpose=RunPurpose.DEVELOPMENT if allow_cpu_smoke else RunPurpose.ACCEPTANCE,
        clock_ns=time.time_ns,
        environment=env,
    )
    model = UniversalOSFMV11(
        input_dims={modality: _patch_dim(spec["channels"]) for modality, spec in MODALITIES.items()},
        family_encoder_config=FamilyEncoderConfig(depth=1 if allow_cpu_smoke else FAMILY_DEPTH),
        include_v1_bank=not allow_cpu_smoke,
    ).to(device)
    # Anchor import: copy every shape-compatible tensor from the qualified upstream checkpoint.
    upstream_import: dict[str, Any] = {"attempted": False}
    if not resume and not allow_cpu_smoke:
        from conrad.foundation.universal_v11.migration import load_v1_checkpoint_for_v11

        upstream_path = REPO_ROOT / json.loads(PAYLOAD_MANIFEST.read_text(encoding="utf-8"))["payloads"][0]["path"]
        try:
            mig = load_v1_checkpoint_for_v11(model, upstream_path, strict_gate=False)
            upstream_import = {
                "attempted": True,
                "checkpoint_sha256": mig.checkpoint_sha256,
                "source_label": mig.source_label,
                "loaded_tensor_count": len(mig.loaded_keys),
                "skipped_tensor_count": len(mig.skipped_keys),
                "missing_v11_tensor_count": len(mig.missing_v11_keys),
                "loaded_tensors": list(mig.loaded_keys),
            }
        except Exception as exc:
            upstream_import = {"attempted": True, "error": f"{type(exc).__name__}: {exc}", "loaded_tensor_count": 0}
        model.to(device)
        run.write_artifact("reports", "upstream_import.json", json.dumps(upstream_import, indent=2, sort_keys=True))
    for name, param in model.named_parameters():
        if name.split(".")[0] in FROZEN_MODULES:
            param.requires_grad_(False)
    trainable = [param for param in model.parameters() if param.requires_grad]
    variant["trainable_parameters"] = sum(param.numel() for param in trainable)
    variant["total_parameters"] = sum(param.numel() for param in model.parameters())
    run.write_artifact("reports", "trainer_variant.json", json.dumps(variant, indent=2, sort_keys=True))
    optimizer = torch.optim.AdamW(trainable, lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < WARMUP_STEPS:
            return (step_idx + 1) / WARMUP_STEPS
        progress = (step_idx - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    start_step = 0
    manifest_digest = load_manifest(SUBPIPE_MANIFEST).manifest_digest()
    full_manifest_digest = hashlib.sha256(SUBPIPE_FULL_MANIFEST.read_bytes()).hexdigest() if SUBPIPE_FULL_MANIFEST.is_file() else "absent"
    split_hash = split_hash_for_plan({"dataset": "public.subpipe_full", "policy": "per-stream contiguous PRETRAIN_REAL/VALIDATION"})
    compatibility = CompatibilityTuple(
        config_digest=config_digest({**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke}),
        manifest_digests=(manifest_digest, full_manifest_digest),
        split_hash=split_hash,
        component="foundation.osfm.universal_v11",
        representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
        extra={"trainer": TRAINER_VARIANT},
    )
    if resume:
        loaded = load_checkpoint(resume, expected=compatibility, model=model, map_location=str(device))
        if loaded.optimizer_state is not None:
            optimizer.load_state_dict(loaded.optimizer_state)
        if loaded.scheduler_state is not None:
            scheduler.load_state_dict(loaded.scheduler_state)
        start_step = int(loaded.trainer_state.get("step", loaded.metadata.step))

    batcher = _SubPipeFullBatcher(
        archive_sha256=str(full_report["actual_sha256"]), batch_size=batch_size, device=device, seed=seed, smoke=allow_cpu_smoke
    )
    coverage = batcher.coverage()
    run.write_artifact("reports", "data_coverage.json", json.dumps({"dataset": "public.subpipe_full", "modalities": coverage}, indent=2, sort_keys=True))
    best_rank = -math.inf
    best_path: str | None = None
    last_path: str | None = None
    train_sample_ids: set[str] = set()
    aug_gen = torch.Generator(device=device).manual_seed(seed + 17)

    def _two_view_forward(images: torch.Tensor, step: int, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        # View A is the exact clean view used by validation, so the rank/variance terms act on clean frames
        # (augmented views alone reached train rank >= 75 while clean frames stayed near 45).
        view_a = _clean_view(images) if CLEAN_ANCHOR_VIEW else _train_view(images, aug_gen, VIEWS[modality]["light"])
        view_b = _train_view(images, aug_gen, VIEWS[modality]["strong"])
        both = torch.cat([view_a, view_b], dim=0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                (_make_input(modality, both, step),),
                reference_time_s=torch.full((both.shape[0],), float(step), device=device),
            )
        z = out.fusion.global_repr.float()
        return z[: images.shape[0]], z[images.shape[0] :]

    t_start = time.time()
    t_last, step_last = t_start, start_step

    def _on_sigterm(signum: int, frame: Any) -> None:  # watchdog stop -> sealed FAILED run, not a torn one
        raise RuntimeError("terminated by SIGTERM (external watchdog)")

    import signal

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        for step in range(start_step + 1, steps + 1):
            model.train()
            modality = "imaging_sonar" if step % 2 else "rgb_camera"
            images, sample_ids = batcher.batch(modality, "train", step)
            train_sample_ids.update(sample_ids)
            z_a, z_b = _two_view_forward(images, step, modality)
            terms = _vicreg_terms(z_a, z_b, rank_floor)
            loss = terms["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % validation_every == 0 or step == steps:
                model.eval()
                per_modality: dict[str, dict[str, Any]] = {}
                val_ids: tuple[str, ...] = ()
                with torch.no_grad():
                    for val_modality in MODALITIES:
                        val_images, ids = batcher.batch(val_modality, "val", step, distinct=True)
                        val_ids += ids
                        val_payload = _clean_view(val_images)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                            val_out = model(
                                (_make_input(val_modality, val_payload, step),),
                                reference_time_s=torch.full((val_payload.shape[0],), float(step), device=device),
                            )
                        m_health = _representation_health(val_out.fusion.global_repr.float(), rank_floor)
                        val_za, val_zb = _two_view_forward(val_images, step, val_modality)
                        m_health["loss"] = float(_vicreg_terms(val_za, val_zb, rank_floor)["total"].cpu())
                        per_modality[val_modality] = m_health
                # Gate every family: the run's rank is the weakest modality's rank.
                health = {
                    "effective_rank": min(h["effective_rank"] for h in per_modality.values()),
                    "collapse_score": min(h["collapse_score"] for h in per_modality.values()),
                    "rank_pass": all(h["rank_pass"] for h in per_modality.values()),
                }
                now = time.time()
                metric_record = {
                    "step": step,
                    "time_s": now - t_start,
                    "steps_per_s": (step - step_last) / max(now - t_last, 1.0e-9),
                    "train/loss": float(loss.detach().cpu()),
                    "train/invariance": float(terms["invariance"].detach().cpu()),
                    "train/variance": float(terms["variance"].detach().cpu()),
                    "train/covariance": float(terms["covariance"].detach().cpu()),
                    "train/rank_loss": float(terms["rank"].detach().cpu()),
                    "train/modality": modality,
                    "val/loss": sum(h["loss"] for h in per_modality.values()) / len(per_modality),
                    **{f"val/{m}/effective_rank": h["effective_rank"] for m, h in per_modality.items()},
                    **{f"val/{m}/collapse_score": h["collapse_score"] for m, h in per_modality.items()},
                    **{f"val/{m}/loss": h["loss"] for m, h in per_modality.items()},
                    "val/representation_effective_rank": health["effective_rank"],
                    "val/representation_collapse_score": health["collapse_score"],
                    "val/rank_pass": float(health["rank_pass"]),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                t_last, step_last = now, step
                run.log_metrics(metric_record)
                # Never select warm-up weights: an untrained model already scores rank 87-99.
                if step > WARMUP_STEPS and health["effective_rank"] > best_rank:
                    best_rank = health["effective_rank"]
                    best_path = str(run.path / "checkpoints" / "best.pt")
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        component="foundation.osfm.universal_v11",
                        config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke},
                        manifest_digests=(manifest_digest, full_manifest_digest),
                        split_hash=split_hash,
                        seeds={"torch": seed},
                        metrics={k: float(v) for k, v in metric_record.items() if isinstance(v, (int, float))},
                        epoch=0,
                        step=step,
                        created_time_ns=time.time_ns(),
                        device=str(device),
                        is_encoder=True,
                        representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
                        selection_metric="val/representation_effective_rank",
                        extra_compatibility={"trainer": TRAINER_VARIANT},
                        trainer_state={"step": step, "train_sample_ids": sorted(train_sample_ids), "val_sample_ids": list(val_ids)},
                    )
            if step % checkpoint_every == 0 or step == steps:
                last_path = str(run.path / "checkpoints" / "last.pt")
                save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    component="foundation.osfm.universal_v11",
                    config={**cfg, "max_steps_effective": steps, "allow_cpu_smoke": allow_cpu_smoke},
                    manifest_digests=(manifest_digest, full_manifest_digest),
                    split_hash=split_hash,
                    seeds={"torch": seed},
                    metrics={"train/loss": float(loss.detach().cpu()), "best_rank": float(best_rank)},
                    epoch=0,
                    step=step,
                    created_time_ns=time.time_ns(),
                    device=str(device),
                    is_encoder=True,
                    representation_pretraining_id="OSFM-UNIVERSAL-V1.1-10P-CANDIDATE",
                    selection_metric="val/representation_effective_rank",
                    extra_compatibility={"trainer": TRAINER_VARIANT},
                    trainer_state={"step": step, "train_sample_ids": sorted(train_sample_ids)},
                )
        report = {
            "gate_id": "OSFM-V11-10P-FORMAL-TRAINER",
            "status": "VALIDATED-RUN",
            "decision": "CONDITIONAL-GO" if best_rank >= rank_floor else "NO-GO",
            "run_dir": str(run.path),
            "steps_completed": steps,
            "best_rank": best_rank,
            "rank_floor": rank_floor,
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "data_scope": "full SubPipe release: side-scan sonar (sss_lf + sss_hf) and cameras (cam0 + cam1)",
            "semantic_claim": "data-backed imaging_sonar (active acoustic) and rgb_camera (visual image) only; the other 827 registry entries remain interface coverage",
            "data_coverage": coverage,
            "registered_modality_count": len(MODALITY_REGISTRY),
            "family_count": len(EncoderFamily),
            "payloads": payload_report,
            "kerem_variant": variant,
            "reviewed_trainer": False,
            "readiness_override": override_record,
            "upstream_import_loaded_tensor_count": upstream_import.get("loaded_tensor_count"),
            "wall_clock_s": time.time() - t_start,
        }
        run.write_artifact("reports", "v11_10p_training_report.json", json.dumps(report, indent=2, sort_keys=True))
        run.seal(TerminalStatus.COMPLETED if report["decision"] != "NO-GO" else TerminalStatus.FAILED, time.time_ns())
        return report
    except Exception:
        run.seal(TerminalStatus.FAILED, time.time_ns())
        raise
    finally:
        batcher.close()


__all__ = ["run_v11_10p_training", "verify_required_payloads"]
