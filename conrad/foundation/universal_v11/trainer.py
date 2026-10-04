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


def _sha256_full(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SHA_CACHE = REPO_ROOT / "artifacts/tmp/sha256_cache.json"


def _sha256(path: Path) -> str:
    """sha256 of a payload file. A full hash is cached against (size, mtime_ns), so ~90 GB of archives on a slow data
    disk are read once, not at every run start; any change of size or mtime forces a fresh full hash."""
    path = Path(path)
    st = path.stat()
    key = str(path.resolve())
    try:
        cache = json.loads(SHA_CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
        return hit["sha256"]
    digest = _sha256_full(path)
    cache[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
    try:
        SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SHA_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
        tmp.replace(SHA_CACHE)
    except OSError:
        pass
    return digest


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
TRAINER_VARIANT = "kerem_v11_10p_multisource_sonar5_v8"
BASE_LR = float(os.environ.get("KEREM_LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("KEREM_WEIGHT_DECAY", "0.05"))
# The project's V1.1 is FamilyEncoderConfig depth 6 (307.8M). The reviewed trainer used depth 2 (201.3M).
FAMILY_DEPTH = int(os.environ.get("KEREM_FAMILY_DEPTH", "6"))
# Top-level modules held frozen (config stage A trains adapters/router/small heads only).
FROZEN_MODULES = tuple(m for m in os.environ.get("KEREM_FREEZE", "").split(",") if m)
WARMUP_STEPS = 2000
# The explicit rank term was gamed in the first long run (held-out sonar rank 50 -> 154 while temporal
# nearest-neighbour agreement fell 0.61 -> 0.06, i.e. noise). Default is plain VICReg.
VICREG_WEIGHTS = {"invariance": 25.0, "variance": 25.0, "covariance": 1.0, "rank": float(os.environ.get("KEREM_RANK_WEIGHT", "0"))}
# Meaning guard: a checkpoint is only eligible as best when, for every family, the cosine nearest neighbour
# of a held-out frame is one of its +-NN_WINDOW time neighbours at least NN_FLOOR of the time (chance ~0.05).
NN_WINDOW = 3
NN_FLOOR = float(os.environ.get("KEREM_NN_FLOOR", "0.3"))
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


SONAR_ANCHOR = os.environ.get("KEREM_SONAR_ANCHOR", "0") == "1"
ANCHOR_IMAGE_SIZE = 28  # the P4.8 sonar encoder's input size


def _clean_images(images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="area")


def _view_images(images: torch.Tensor, generator: torch.Generator, view: dict[str, Any]) -> torch.Tensor:
    """Crop -> photometrics (gain, per-channel colour gain, speckle, additive noise); stays in [0, 1]."""
    crops = _random_resized_crop(images, generator, view["crop_scale"])
    batch, channels, device = crops.shape[0], crops.shape[1], crops.device
    gain = 0.6 + 0.8 * torch.rand((batch, 1, 1, 1), generator=generator, device=device)
    if view.get("channel_gain", 0.0) > 0 and channels > 1:
        gain = gain * (1.0 + view["channel_gain"] * torch.randn((batch, channels, 1, 1), generator=generator, device=device))
    speckle = 1.0 + view["speckle"] * torch.randn(crops.shape, generator=generator, device=device)
    noise = view["noise"] * torch.randn(crops.shape, generator=generator, device=device)
    return (crops * gain * speckle + noise).clamp(0.0, 1.0)


def _tokens(
    modality: str,
    images: torch.Tensor,
    model: Any,
    *,
    patch_dropout: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """[B, C, IMAGE_SIZE, IMAGE_SIZE] in [0, 1] -> the modality's V1.1 payload tokens.

    Sonar with the anchor: the frozen P4.8 sonar encoder held in V1.1's own V1 compatibility bank turns a
    28x28 frame (its training distribution: raw /255, no standardisation) into [CLS + 4 patch] 384-d tokens.
    Otherwise: 16 standardised pixel patches."""
    if modality == "imaging_sonar" and SONAR_ANCHOR:
        anchor = model.v1.sonar
        anchor.eval()
        with torch.no_grad():
            out = anchor(F.interpolate(images, size=(ANCHOR_IMAGE_SIZE, ANCHOR_IMAGE_SIZE), mode="area"))
        tokens = torch.cat([out.modality_repr[:, None], out.patch_tokens], dim=1).float()
    else:
        tokens = _patchify(_standardize(images))
    if patch_dropout > 0 and generator is not None:
        keep = torch.rand((tokens.shape[0], tokens.shape[1], 1), generator=generator, device=tokens.device) >= patch_dropout
        tokens = tokens * keep
    return tokens.float()


def _clean_view(images: torch.Tensor, modality: str, model: Any) -> torch.Tensor:
    return _tokens(modality, _clean_images(images), model)


def _train_view(images: torch.Tensor, generator: torch.Generator, view: dict[str, Any], modality: str, model: Any) -> torch.Tensor:
    return _tokens(modality, _view_images(images, generator, view), model, patch_dropout=view["patch_dropout"], generator=generator)


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


# Additional camera sources from other sites (both CC-BY-4.0, rights CLEARED, training_allowed in their manifests).
EXTRA_CAMERA = tuple(x for x in os.environ.get("KEREM_EXTRA_CAMERA", "seaclear,uvvid").split(",") if x)
SEACLEAR_MANIFEST = REPO_ROOT / "datasets/public/seaclear.manifest.yaml"
SEACLEAR_DIR = REPO_ROOT / "artifacts/data/public.seaclear"
UVVID_MANIFEST = REPO_ROOT / "datasets/public/uvvid.manifest.yaml"
UVVID_DIR = REPO_ROOT / "artifacts/data/public.uvvid"
UVVID_FRAME_STRIDE = 15  # 2 frames/s from 30 fps GoPro video
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_NON_IMAGE_HINTS = ("mask", "label", "annotation", "segment")


# Additional imaging-sonar sources from other surveys, vehicles and sonars (CC-BY-4.0 or MIT; see their manifests).
EXTRA_SONAR = tuple(x for x in os.environ.get("KEREM_EXTRA_SONAR", "catalunya,china_offshore,aquascan,uatd").split(",") if x)
SONAR_SOURCES: dict[str, dict[str, Any]] = {
    # 434k side-scan patches from Catalan coastal surveys; read from the merged archive, every Nth patch per survey line
    "catalunya": {"manifest": "datasets/public/sss_catalunya.manifest.yaml", "dir": "artifacts/data/public.sss_catalunya",
                  "split_zip": "raw/sss_ssl_dataset_N713_384", "stride": int(os.environ.get("KEREM_CATALUNYA_STRIDE", "6")), "sequential": True},
    # side-scan target chips from four Chinese sea regions (not a time sequence)
    "china_offshore": {"manifest": "datasets/public/china_offshore_sss.manifest.yaml", "dir": "artifacts/data/public.china_offshore_sss",
                       "images": "extracted", "stride": 1, "sequential": False},
    # lakebed side-scan screenshots, ordered by capture timestamp
    "aquascan": {"manifest": "datasets/public/aquascan_1k.manifest.yaml", "dir": "artifacts/data/public.aquascan_1k",
                 "images": "extracted", "stride": 1, "sequential": False},
    # multibeam forward-looking sonar (Tritech Gemini 1200ik), lake and shallow-water sessions
    "uatd": {"manifest": "datasets/public/uatd.manifest.yaml", "dir": "artifacts/data/public.uatd",
             "images": "extracted", "stride": 1, "sequential": True},
}
_CAMERA_SOURCES = {"seaclear": (SEACLEAR_MANIFEST, SEACLEAR_DIR), "uvvid": (UVVID_MANIFEST, UVVID_DIR)}


def _natural_key(text: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def _sonar_group(source: str, ref: str) -> str:
    name = Path(ref).name
    if source == "catalunya":  # survey line, e.g. N9_1_211023112600 from N9_1_211023112600_xtf-CH12_batch-5_ch-1_3_1.tiff
        return name.split("_xtf")[0]
    parts = Path(ref).parts
    if source == "china_offshore":  # sea region: .../images/<region>/...
        return parts[parts.index("images") + 1] if "images" in parts[:-1] else "china_offshore"
    if source == "uatd":  # session archive: .../extracted/<UATD_Training|UATD_Test_1|...>/...
        return parts[parts.index("extracted") + 1] if "extracted" in parts[:-1] else "uatd"
    return source


def _sonar_sort_key(source: str, ref: str) -> list[Any]:
    name = Path(ref).name
    if source == "aquascan":  # "<uuid>-Screenshot_2025-08-10_23.00.36.png": order by the timestamp
        return _natural_key(name.split("-", 1)[-1])
    return _natural_key(name)


def _verify_manifest_files(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    rows = []
    for entry in manifest.get("files", []):
        if not str(entry.get("path", "")).startswith("raw/"):
            continue
        path = root / entry["path"]
        actual = _sha256(path) if path.is_file() else None
        rows.append({"path": entry["path"], "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "ok": actual == entry.get("sha256")})
    return {"manifest": str(manifest_path.relative_to(REPO_ROOT)), "files": rows, "decision": "PASS" if rows and all(r["ok"] for r in rows) else "FAIL"}


def verify_extra_sources() -> dict[str, Any]:
    report: dict[str, Any] = {name: _verify_manifest_files(*_CAMERA_SOURCES[name]) for name in EXTRA_CAMERA}
    for name in EXTRA_SONAR:
        spec = SONAR_SOURCES[name]
        report[name] = _verify_manifest_files(REPO_ROOT / spec["manifest"], REPO_ROOT / spec["dir"])
    report["decision"] = "PASS" if all(r["decision"] == "PASS" for r in report.values()) else "FAIL"
    return report


verify_extra_camera = verify_extra_sources  # backwards-compatible name


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """16-bit / float sonar rasters -> uint8 by robust (0.5..99.5 percentile) scaling; uint8 passes through."""
    if image.dtype == np.uint8:
        return image
    data = image.astype(np.float32)
    lo, hi = np.percentile(data, (0.5, 99.5))
    return np.clip((data - lo) / max(hi - lo, 1.0e-6) * 255.0, 0, 255).astype(np.uint8)


def _finish(image: np.ndarray | None, channels: int, what: str) -> np.ndarray:
    import cv2

    if image is None:
        raise RuntimeError(f"{what}: not a readable image")
    image = _to_uint8(image)
    if channels == 1 and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if channels == 3:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    image = image[None] if channels == 1 else image.transpose(2, 0, 1)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _decode_image_path(job: str | tuple[str, int]) -> np.ndarray:
    import cv2

    path, channels = (job, 3) if isinstance(job, str) else job
    return _finish(cv2.imread(path, cv2.IMREAD_UNCHANGED), channels, path)


class _SplitZip:
    """Read members of a split (spanned) zip archive in place: base.z01, base.z02, ..., base.zip.

    Python's zipfile refuses multi-disk archives; merging this one (52 GB on a 48 MB/s disk) costs about 35 min,
    so the central directory is parsed here and each member is read across the concatenated parts. Stored and
    deflated members are supported; every member's CRC-32 is checked, as zipfile would.
    """

    def __init__(self, base: str) -> None:
        import struct

        stem = Path(base)
        self.parts = [*sorted(stem.parent.glob(stem.name + ".z[0-9][0-9]")), stem.parent / (stem.name + ".zip")]
        self.sizes = [p.stat().st_size for p in self.parts]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self._fh: dict[int, Any] = {}
        tail_len = min(self.sizes[-1], 1 << 20)
        with open(self.parts[-1], "rb") as fh:
            fh.seek(self.sizes[-1] - tail_len)
            tail = fh.read()
        e = tail.rfind(b"PK\x05\x06")
        _, cd_disk, _, n_total, cd_size, cd_off = struct.unpack("<HHHHII", tail[e + 4 : e + 20])
        loc = tail.rfind(b"PK\x06\x07", 0, e)
        if loc >= 0:  # zip64 end of central directory
            z64_disk, z64_off, _ = struct.unpack("<IQI", tail[loc + 4 : loc + 20])
            rec = self._read(self.starts[z64_disk] + z64_off, 56)
            cd_disk, n_total, cd_size, cd_off = struct.unpack("<I", rec[20:24])[0], *struct.unpack("<QQQ", rec[32:56])
        cd = self._read(self.starts[cd_disk] + cd_off, cd_size)
        self.index: dict[str, tuple[int, int, int, int]] = {}
        pos = 0
        for _ in range(n_total):
            (sig, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, disk, _, _, loff) = struct.unpack(
                "<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
            if sig != 0x02014B50:
                raise RuntimeError(f"{base}: bad central directory entry")
            name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
            extra = cd[pos + 46 + nlen : pos + 46 + nlen + xlen]
            x = 0
            while x + 4 <= len(extra):  # zip64 extra: fields present only where the 32/16-bit field is saturated
                hid, hlen = struct.unpack("<HH", extra[x : x + 4])
                if hid == 0x0001:
                    vals, q = extra[x + 4 : x + 4 + hlen], 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[q : q + 8])[0]
                        q += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[q : q + 8])[0]
                        q += 8
                    if loff == 0xFFFFFFFF:
                        loff = struct.unpack("<Q", vals[q : q + 8])[0]
                        q += 8
                    if disk == 0xFFFF:
                        disk = struct.unpack("<I", vals[q : q + 4])[0]
                x += 4 + hlen
            self.index[name] = (self.starts[disk] + loff, csize, method, crc)
            pos += 46 + nlen + xlen + clen

    def _read(self, offset: int, length: int) -> bytes:
        out = bytearray()
        while length > 0:
            i = max(k for k, st in enumerate(self.starts) if st <= offset)
            fh = self._fh.get(i)
            if fh is None:
                fh = self._fh[i] = open(self.parts[i], "rb")  # noqa: SIM115 - kept open for the worker's lifetime
            fh.seek(offset - self.starts[i])
            chunk = fh.read(min(length, self.sizes[i] - (offset - self.starts[i])))
            if not chunk:
                raise RuntimeError("split zip: read past the end")
            out += chunk
            offset += len(chunk)
            length -= len(chunk)
        return bytes(out)

    def namelist(self) -> list[str]:
        return list(self.index)

    def read(self, name: str) -> bytes:
        import struct
        import zlib

        off, csize, method, crc = self.index[name]
        hdr = self._read(off, 30)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"{name}: bad local header")
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        data = self._read(off + 30 + nlen + xlen, csize)
        if method == 8:
            data = zlib.decompress(data, -15)
        elif method != 0:
            raise RuntimeError(f"{name}: unsupported compression method {method}")
        if zlib.crc32(data) & 0xFFFFFFFF != crc:
            raise RuntimeError(f"{name}: CRC-32 mismatch")
        return data


_ZIP_HANDLES: dict[str, Any] = {}


def _decode_zip_image(job: tuple[str, str, int]) -> np.ndarray:
    import cv2

    archive, member, channels = job
    handle = _ZIP_HANDLES.get(archive)
    if handle is None:
        handle = _ZIP_HANDLES[archive] = _SplitZip(archive[6:]) if archive.startswith("split:") else zipfile.ZipFile(archive)
    raw = np.frombuffer(handle.read(member), dtype=np.uint8)  # CRC-32 checked by zipfile
    return _finish(cv2.imdecode(raw, cv2.IMREAD_UNCHANGED), channels, member)


def _decode_video(job: tuple[str, int]) -> np.ndarray:
    import cv2

    path, stride = job
    capture = cv2.VideoCapture(path)
    frames, index = [], 0
    while capture.grab():
        if index % stride == 0:
            ok, image = capture.retrieve()
            if ok:
                image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
                frames.append(image.transpose(2, 0, 1))
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"{path}: no frames decoded")
    return np.ascontiguousarray(np.stack(frames), dtype=np.uint8)


def _split_groups(groups: dict[str, list[str]]) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    train: list[str] = []
    val: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for name, refs in sorted(groups.items()):
        ranges = _partition_ranges(len(refs))
        t0, t1 = ranges[CorpusPartition.PRETRAIN_REAL]
        v0, v1 = ranges[CorpusPartition.VALIDATION]
        train += refs[t0:t1]
        val += refs[v0:v1]
        counts[name] = {"all": len(refs), "train": t1 - t0, "validation": v1 - v0}
    return train, val, counts


def _decode_image_path_safe(job: Any) -> np.ndarray | None:
    try:
        return _decode_image_path(job)
    except Exception:
        return None


def _decode_zip_image_safe(job: Any) -> np.ndarray | None:
    try:
        return _decode_zip_image(job)
    except Exception:
        return None


class _MultiSourceBatcher:
    """Frames per modality and per source, partitioned per stream/site/video by contiguous time blocks.
    Frames are decoded once (disk cache keyed by content hashes) and held on the device. Training batches
    draw equally from every source of a modality; validation uses distinct frames only."""

    def __init__(self, *, archive_sha256: str, extra_camera: dict[str, Any] | None, batch_size: int, device: torch.device, seed: int, smoke: bool = False) -> None:
        extra = extra_camera
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.data: dict[str, dict[str, dict[str, Any]]] = {modality: {} for modality in MODALITIES}
        index = _full_index()
        for modality, spec in MODALITIES.items():
            groups = {stream: [m for _, m in index[stream][:: stride * (200 if smoke else 1)]] for stream, stride in spec["streams"].items()}
            train, val, counts = _split_groups(groups)
            self._add(modality, "subpipe", train, val, counts, key=archive_sha256, loader=("zip", spec["channels"]))
            self.data[modality]["subpipe"]["sequential"] = True
        for name in EXTRA_SONAR if extra is not None else ():
            if name not in extra:
                continue
            spec = SONAR_SOURCES[name]
            root = REPO_ROOT / spec["dir"]
            if "split_zip" in spec:
                archive = "split:" + str(root / spec["split_zip"])
                refs = [n for n in _SplitZip(archive[6:]).namelist() if Path(n).suffix.lower() in _IMAGE_SUFFIXES]
                loader: tuple[Any, ...] = ("zipimg", 1, archive)
            elif "zip" in spec:
                archive = str(root / spec["zip"])
                with zipfile.ZipFile(archive) as handle:
                    refs = [n for n in handle.namelist() if Path(n).suffix.lower() in _IMAGE_SUFFIXES]
                loader = ("zipimg", 1, archive)
            else:
                refs = [
                    str(p) for p in (root / spec["images"]).rglob("*")
                    if p.suffix.lower() in _IMAGE_SUFFIXES and not any(h in str(p).lower() for h in _NON_IMAGE_HINTS)
                ]
                loader = ("image", 1)
            groups: dict[str, list[str]] = {}
            for ref in refs:
                groups.setdefault(_sonar_group(name, ref), []).append(ref)
            groups = {g: sorted(v, key=lambda r: _sonar_sort_key(name, r))[:: spec["stride"] * (50 if smoke else 1)] for g, v in groups.items()}
            groups = {g: v for g, v in groups.items() if len(v) >= 2}
            train, val, counts = _split_groups(groups)
            key = "|".join(r["actual_sha256"] or "" for r in extra[name]["files"]) + f"|stride={spec['stride']}"
            self._add("imaging_sonar", name, train, val, counts, key=key, loader=loader)
            entry = self.data["imaging_sonar"][name]
            entry["sequential"] = spec["sequential"]
            entry["val_groups"] = [_sonar_group(name, r) for r in val]
        if extra is not None and "seaclear" in extra:
            images = [
                p for p in sorted((SEACLEAR_DIR / "extracted").rglob("*"))
                if p.suffix.lower() in _IMAGE_SUFFIXES and not any(h in str(p).lower() for h in _NON_IMAGE_HINTS)
            ]
            site_groups: dict[str, list[str]] = {}
            for path in images:
                site_groups.setdefault(str(path.parent.relative_to(SEACLEAR_DIR)), []).append(str(path))
            train, val, counts = _split_groups(site_groups)
            seaclear_key = "|".join(r["actual_sha256"] for r in extra_camera["seaclear"]["files"])
            self._add("rgb_camera", "seaclear", train, val, counts, key=seaclear_key, loader=("image", 3))
            self.data["rgb_camera"]["seaclear"]["sequential"] = True
        if extra is not None and "uvvid" in extra:
            videos = sorted(str(p) for p in (UVVID_DIR / "raw").glob("ROV_GoPro_*.mp4"))
            uvvid_key = "|".join(r["actual_sha256"] for r in extra_camera["uvvid"]["files"])
            self._add_videos("rgb_camera", "uvvid", videos, key=uvvid_key)
            self.data["rgb_camera"]["uvvid"]["sequential"] = True

    def _cache_path(self, modality: str, source: str, key: str, members: list[str]) -> Path:
        digest = hashlib.sha256("\n".join([key, str(CACHE_SIZE), *members]).encode()).hexdigest()[:20]
        return SUBPIPE_FULL_CACHE_DIR / f"{modality}_{source}_{CACHE_SIZE}_{digest}.npy"

    def _store(self, modality: str, source: str, frames: np.ndarray, train: list[str], val: list[str], counts: dict[str, Any], cache: Path) -> None:
        self.data[modality][source] = {
            "train": tuple(train),
            "val": tuple(val),
            "counts": counts,
            "frames": torch.from_numpy(frames).to(self.device),
            "train_index": torch.arange(len(train), device=self.device),
            "val_index": torch.arange(len(train), len(train) + len(val), device=self.device),
            "cache": str(cache.relative_to(REPO_ROOT)),
        }

    def _add(self, modality: str, source: str, train: list[str], val: list[str], counts: dict[str, Any], *, key: str, loader: tuple[Any, ...]) -> None:
        from multiprocessing import get_context

        if not train or not val:
            raise RuntimeError(f"{modality}/{source}: PRETRAIN_REAL and VALIDATION partitions must both be non-empty")
        members = train + val
        cache = self._cache_path(modality, source, key, members)
        skip_path = cache.with_suffix(".skipped.json")
        if cache.is_file():
            frames = np.load(cache)
            skipped = json.loads(skip_path.read_text()) if skip_path.is_file() else []
        else:
            kind, channels = loader[0], loader[1]
            with get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
                if kind == "zip":  # the verified core archive stays fail-closed
                    decoded = pool.map(_decode_member, [(m, channels) for m in members], chunksize=16)
                elif kind == "zipimg":
                    decoded = pool.map(_decode_zip_image_safe, [(loader[2], m, channels) for m in members], chunksize=64)
                else:
                    decoded = pool.map(_decode_image_path_safe, [(m, channels) for m in members], chunksize=16)
            skipped = [m for m, f in zip(members, decoded, strict=True) if f is None]
            if len(skipped) > max(5, 0.01 * len(members)):
                raise RuntimeError(f"{modality}/{source}: {len(skipped)} of {len(members)} frames unreadable (> max(5, 1 %))")
            frames = np.stack([f for f in decoded if f is not None])
            SUBPIPE_FULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, frames)
            skip_path.write_text(json.dumps(skipped))
        if skipped:
            bad = set(skipped)
            train = [m for m in train if m not in bad]
            val = [m for m in val if m not in bad]
            counts = {**counts, "skipped_unreadable": [Path(m).name for m in skipped]}
        self._store(modality, source, frames, train, val, counts, cache)

    def _add_videos(self, modality: str, source: str, videos: list[str], *, key: str) -> None:
        from multiprocessing import get_context

        if not videos:
            raise RuntimeError(f"{modality}/{source}: no videos found")
        cache = self._cache_path(modality, source, key, [*videos, f"stride={UVVID_FRAME_STRIDE}"])
        meta_path = cache.with_suffix(".json")
        if cache.is_file() and meta_path.is_file():
            frames = np.load(cache)
            meta = json.loads(meta_path.read_text())
            train, val, counts = meta["train"], meta["val"], meta["counts"]
        else:
            with get_context("fork").Pool(min(len(videos), max(1, (os.cpu_count() or 2) - 1))) as pool:
                decoded = pool.map(_decode_video, [(v, UVVID_FRAME_STRIDE) for v in videos])
            groups = {Path(v).name: [f"{Path(v).name}#{i * UVVID_FRAME_STRIDE}" for i in range(len(d))] for v, d in zip(videos, decoded, strict=True)}
            by_id = {ref: frame for v, d in zip(videos, decoded, strict=True) for ref, frame in zip(groups[Path(v).name], d, strict=True)}
            train, val, counts = _split_groups(groups)
            frames = np.stack([by_id[ref] for ref in train + val])
            SUBPIPE_FULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, frames)
            meta_path.write_text(json.dumps({"train": train, "val": val, "counts": counts}))
        if not train or not val:
            raise RuntimeError(f"{modality}/{source}: PRETRAIN_REAL and VALIDATION partitions must both be non-empty")
        self._store(modality, source, frames, train, val, counts, cache)

    def coverage(self) -> dict[str, Any]:
        return {
            modality: {
                source: {
                    "groups": entry["counts"],
                    "train_frames": len(entry["train"]),
                    "validation_frames": len(entry["val"]),
                    "frame_cache": entry["cache"],
                }
                for source, entry in sources.items()
            }
            for modality, sources in self.data.items()
        }

    def heldout_sequences(self, modality: str, max_per_source: int = 1000) -> tuple[torch.Tensor, list[str], list[int]]:
        """Contiguous held-out frames per source with (sequence label, position) for the temporal NN check."""
        images, labels, positions = [], [], []
        for source, entry in self.data[modality].items():
            if not entry.get("sequential", True):
                continue  # target chips / screenshots are not a time sequence; the temporal check does not apply
            refs = entry["val"][:max_per_source]
            images.append(entry["frames"][entry["val_index"][: len(refs)]])
            counters: dict[str, int] = {}
            for ref in refs:
                if "val_groups" in entry:
                    seq = entry["val_groups"][entry["val"].index(ref)] if len(entry["val"]) < 5000 else _sonar_group(source, ref)
                elif source == "subpipe":
                    match = _FULL_MEMBER.match(ref)
                    seq = _FOLDER_STREAM[match.group(2)] if match else "subpipe"
                elif source == "uvvid":
                    seq = ref.split("#")[0]
                else:
                    seq = str(Path(ref).parent)
                label = f"{source}:{seq}"
                labels.append(label)
                positions.append(counters.get(label, 0))
                counters[label] = counters.get(label, 0) + 1
        return torch.cat(images).float() / 255.0, labels, positions

    def sources(self, modality: str) -> tuple[str, ...]:
        return tuple(self.data[modality])

    def batch(self, modality: str, split: str, step: int, *, distinct: bool = False, source: str | None = None) -> tuple[torch.Tensor, tuple[str, ...]]:
        names = (source,) if source is not None else self.sources(modality)
        gen = torch.Generator().manual_seed(self.seed + step)
        images, ids = [], []
        for position, name in enumerate(names):
            entry = self.data[modality][name]
            refs, index = (entry["train"], entry["train_index"]) if split == "train" else (entry["val"], entry["val_index"])
            share = self.batch_size // len(names) + (1 if position < self.batch_size % len(names) else 0)
            order = torch.randperm(len(refs), generator=gen).tolist()
            size = min(share, len(order)) if distinct else share
            picks = [order[k % len(order)] for k in range(size)]
            images.append(entry["frames"][index[torch.tensor(picks, device=self.device)]].float() / 255.0)
            ids += [f"{name}:{refs[i]}" for i in picks]
        return torch.cat(images, dim=0), tuple(ids)

    def close(self) -> None:
        return None


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


def _temporal_nn_hit(z: torch.Tensor, labels: list[str], positions: list[int], window: int = NN_WINDOW) -> float:
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T
    sim.fill_diagonal_(-2.0)
    nearest = sim.argmax(dim=1).tolist()
    hits = sum(1 for i, j in enumerate(nearest) if labels[i] == labels[j] and abs(positions[i] - positions[j]) <= window)
    return hits / max(1, len(nearest))


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
    extra_report = verify_extra_sources() if (EXTRA_CAMERA or EXTRA_SONAR) else None
    if extra_report is not None and extra_report["decision"] != "PASS" and not allow_cpu_smoke:
        raise RuntimeError(f"extra sources are not hash-verified against their manifests: {extra_report}")

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
        "nn_guard": {"window": NN_WINDOW, "floor": NN_FLOOR},
        "sonar_anchor_encoder": "P4.8 teacher (OSFM-S-PRETRAIN-V1) frozen in model.v1.sonar, 28x28 input" if SONAR_ANCHOR else None,
        "extra_camera_sources": list(EXTRA_CAMERA),
        "extra_sonar_sources": {name: {k: v for k, v in SONAR_SOURCES[name].items() if k != "manifest"} for name in EXTRA_SONAR},
        "uvvid_frame_stride": UVVID_FRAME_STRIDE,
        "camera_batch_policy": "equal share per source",
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
            "extra_camera": extra_report,
        },
        purpose=RunPurpose.DEVELOPMENT if allow_cpu_smoke else RunPurpose.ACCEPTANCE,
        clock_ns=time.time_ns,
        environment=env,
    )
    model = UniversalOSFMV11(
        input_dims={
            modality: (384 if modality == "imaging_sonar" and SONAR_ANCHOR else _patch_dim(spec["channels"]))
            for modality, spec in MODALITIES.items()
        },
        family_encoder_config=FamilyEncoderConfig(depth=1 if allow_cpu_smoke else FAMILY_DEPTH),
        include_v1_bank=not allow_cpu_smoke,
    ).to(device)
    # Anchor import (config stage A): the upstream checkpoint is the P4.8 ViT-S sonar encoder, stored under
    # ck["model"] with student./teacher. prefixes. Its teacher weights match V1.1's V1CompatibilityBank.sonar
    # tensor for tensor. (load_v1_checkpoint_for_v11 expects a flat state_dict and so loaded 0 tensors.)
    upstream_import: dict[str, Any] = {"attempted": False}
    if not resume and not allow_cpu_smoke and model.v1 is not None:
        upstream_path = REPO_ROOT / json.loads(PAYLOAD_MANIFEST.read_text(encoding="utf-8"))["payloads"][0]["path"]
        state = torch.load(upstream_path, map_location="cpu", weights_only=False)["model"]
        teacher = {k[len("teacher.") :]: v for k, v in state.items() if k.startswith("teacher.")}
        target = model.v1.sonar.state_dict()
        matched = {k: v for k, v in teacher.items() if k in target and target[k].shape == v.shape}
        if len(matched) != len(target):
            raise RuntimeError(f"P4.8 sonar teacher matches {len(matched)}/{len(target)} V1 bank tensors")
        model.v1.sonar.load_state_dict(matched)
        model.to(device)
        upstream_import = {
            "attempted": True,
            "checkpoint_sha256": _sha256(upstream_path),
            "source": "teacher.* -> model.v1.sonar",
            "loaded_tensor_count": len(matched),
            "target_tensor_count": len(target),
            "used_as_sonar_front_end": SONAR_ANCHOR,
        }
        run.write_artifact("reports", "upstream_import.json", json.dumps(upstream_import, indent=2, sort_keys=True))
    if model.v1 is not None:
        for param in model.v1.parameters():  # the V1 bank is a protected anchor (config: protect_imported_v1_and_selected_sonar_anchor)
            param.requires_grad_(False)
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

    batcher = _MultiSourceBatcher(
        archive_sha256=str(full_report["actual_sha256"]),
        extra_camera=extra_report,
        batch_size=batch_size,
        device=device,
        seed=seed,
        smoke=allow_cpu_smoke,
    )
    coverage = batcher.coverage()
    heldout = {m: batcher.heldout_sequences(m) for m in MODALITIES}
    run.write_artifact("reports", "data_coverage.json", json.dumps({"datasets": ["public.subpipe_full", *(f"public.{x}" for x in EXTRA_CAMERA), *(Path(SONAR_SOURCES[x]["dir"]).name for x in EXTRA_SONAR)], "modalities": coverage}, indent=2, sort_keys=True))
    best_rank = -math.inf
    best_path: str | None = None
    last_path: str | None = None
    train_sample_ids: set[str] = set()
    aug_gen = torch.Generator(device=device).manual_seed(seed + 17)

    def _two_view_forward(images: torch.Tensor, step: int, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        # View A is the exact clean view used by validation, so the rank/variance terms act on clean frames
        # (augmented views alone reached train rank >= 75 while clean frames stayed near 45).
        view_a = _clean_view(images, modality, model) if CLEAN_ANCHOR_VIEW else _train_view(images, aug_gen, VIEWS[modality]["light"], modality, model)
        view_b = _train_view(images, aug_gen, VIEWS[modality]["strong"], modality, model)
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
                        # pooled: equal share of distinct held-out frames from every source of the family
                        val_images, ids = batcher.batch(val_modality, "val", step, distinct=True)
                        val_ids += ids
                        val_payload = _clean_view(val_images, val_modality, model)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                            val_out = model(
                                (_make_input(val_modality, val_payload, step),),
                                reference_time_s=torch.full((val_payload.shape[0],), float(step), device=device),
                            )
                        m_health = _representation_health(val_out.fusion.global_repr.float(), rank_floor)
                        val_za, val_zb = _two_view_forward(val_images, step, val_modality)
                        m_health["loss"] = float(_vicreg_terms(val_za, val_zb, rank_floor)["total"].cpu())
                        m_health["sources"] = {}
                        for src in batcher.sources(val_modality):
                            src_images, _ = batcher.batch(val_modality, "val", step, distinct=True, source=src)
                            src_payload = _clean_view(src_images, val_modality, model)
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                                src_out = model(
                                    (_make_input(val_modality, src_payload, step),),
                                    reference_time_s=torch.full((src_payload.shape[0],), float(step), device=device),
                                )
                            m_health["sources"][src] = _representation_health(src_out.fusion.global_repr.float(), rank_floor)["effective_rank"]
                        seq_images, seq_labels, seq_pos = heldout[val_modality]
                        seq_z = []
                        for start in range(0, seq_images.shape[0], 256):
                            chunk = seq_images[start : start + 256]
                            chunk_payload = _clean_view(chunk, val_modality, model)
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                                chunk_out = model(
                                    (_make_input(val_modality, chunk_payload, step),),
                                    reference_time_s=torch.full((chunk.shape[0],), float(step), device=device),
                                )
                            seq_z.append(chunk_out.fusion.global_repr.float())
                        m_health["nn_temporal_hit"] = _temporal_nn_hit(torch.cat(seq_z), seq_labels, seq_pos)
                        per_modality[val_modality] = m_health
                # Gate every family: the run's rank is the weakest modality's rank.
                health = {
                    "effective_rank": min(h["effective_rank"] for h in per_modality.values()),
                    "collapse_score": min(h["collapse_score"] for h in per_modality.values()),
                    "rank_pass": all(h["rank_pass"] for h in per_modality.values()),
                    "nn_min": min(h["nn_temporal_hit"] for h in per_modality.values()),
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
                    **{f"val/{m}/nn_temporal_hit": h["nn_temporal_hit"] for m, h in per_modality.items()},
                    "val/nn_temporal_hit_min": health["nn_min"],
                    **{f"val/{m}/{src}/effective_rank": r for m, h in per_modality.items() for src, r in h["sources"].items()},
                    "val/representation_effective_rank": health["effective_rank"],
                    "val/representation_collapse_score": health["collapse_score"],
                    "val/rank_pass": float(health["rank_pass"]),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                t_last, step_last = now, step
                run.log_metrics(metric_record)
                # Never select warm-up weights (an untrained model already scores rank 87-99), and never a
                # checkpoint whose held-out neighbourhoods have lost their temporal structure (rank gaming).
                if step > WARMUP_STEPS and health["nn_min"] >= NN_FLOOR and health["effective_rank"] > best_rank:
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
            "data_scope": "full SubPipe release (side-scan sonar sss_lf+sss_hf; cameras cam0+cam1) plus camera frames from SeaClear (multi-site) and UVVID ROV_GoPro_1-8",
            "semantic_claim": "data-backed imaging_sonar (active acoustic) and rgb_camera (visual image) only; the other 827 registry entries remain interface coverage",
            "sonar_anchor_encoder": SONAR_ANCHOR,
            "upstream_import": {k: v for k, v in upstream_import.items() if k != "loaded_tensors"},
            "data_coverage": coverage,
            "registered_modality_count": len(MODALITY_REGISTRY),
            "family_count": len(EncoderFamily),
            "payloads": payload_report,
            "kerem_variant": variant,
            "reviewed_trainer": False,
            "best_selection_rule": f"after warm-up; every family's held-out temporal NN hit >= {NN_FLOOR}; then max weakest-family rank",
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
