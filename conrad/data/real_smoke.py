"""DATA-REAL-SMOKE-E001: first contact of ECMER with real public frames (UVVID, CC BY 4.0).

Hypothesis (engineering check, not a research claim): on real underwater video frames the
engineered ECMER quality features respond in the physically expected direction to controlled
corruptions (Gaussian blur raises ``blur``, additive noise lowers ``snr_db``, darkening lowers
``brightness`` and ``contrast``), and the untrained ECMER encoder turns every real observation into
finite, contract-valid Evidence.

No labels, poses or calibration exist in the sample, so nothing is scored against truth. ECMER
weights are randomly initialised; reliability numbers are reported but support no claim.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2
import numpy as np
import torch

from conrad.core.config import CoreConfig
from conrad.core.ecmer.quality_features import RawQuality, image_quality
from conrad.core.ecmer.service import EcmerEncoder
from conrad.data.adapters.uvvid import UvvidVideoAdapter
from conrad.data.manifest import data_root_for, find_manifest, load_manifest, verify_loaded_manifest
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation

EXPERIMENT_ID = "DATA-REAL-SMOKE-E001"


def _corrupt(image: np.ndarray, kind: str, cfg: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    img = image.astype(np.float64)
    if kind == "blur":
        sigma = float(cfg["blur_sigma_px"])
        out = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif kind == "noise":
        out = img + rng.normal(0.0, float(cfg["noise_sigma_u8"]), size=img.shape)
    elif kind == "darken":
        out = img * float(cfg["darken_factor"])
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return np.asarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def _stats(values: Sequence[float | None]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _quality_table(qs: Sequence[RawQuality]) -> dict[str, dict[str, float]]:
    return {
        "blur": _stats([q.blur for q in qs]),
        "brightness": _stats([q.brightness for q in qs]),
        "contrast": _stats([q.contrast for q in qs]),
        "snr_db": _stats([q.snr_db for q in qs]),
        "saturation_fraction": _stats([q.saturation_fraction for q in qs]),
    }


# expected sign of (corrupted - clean) for each feature under each corruption
EXPECTED: dict[str, dict[str, int]] = {
    "blur": {"blur": +1},
    "noise": {"snr_db": -1},
    "darken": {"brightness": -1, "contrast": -1},
}


def _encode(encoder: EcmerEncoder, observations: Sequence[Observation]) -> dict[str, float]:
    dts: list[float] = []
    for i, o in enumerate(observations):
        prev = observations[i - 1] if i else None
        same_clock = prev is not None and prev.timestamp.clock_domain == o.timestamp.clock_domain
        dts.append((o.timestamp.time_ns - prev.timestamp.time_ns) / 1e9 if same_clock and prev else 0.0)
    encoded = encoder.encode(observations, dts)
    emb = np.asarray([e.evidence.embedding for e in encoded], dtype=np.float64)
    rel = np.asarray([e.evidence.reliability for e in encoded], dtype=np.float64)
    ood = np.asarray([e.evidence.sensor_context.ood_score or 0.0 for e in encoded], dtype=np.float64)
    validity: dict[str, float] = {}
    for e in encoded:
        validity[e.evidence.validity.value] = validity.get(e.evidence.validity.value, 0.0) + 1.0
    return {
        "n_evidence": float(len(encoded)),
        "embedding_dim": float(emb.shape[1]),
        "all_finite": float(bool(np.isfinite(emb).all() and np.isfinite(rel).all())),
        "reliability_mean": float(rel.mean()),
        "ood_score_mean": float(ood.mean()),
        "provenance_direct": float(
            all(e.provenance.source_type.value == "DIRECT_OBSERVATION" for e in encoded)
        ),
        **{f"validity_{k}": v for k, v in sorted(validity.items())},
    }


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: Path) -> dict[str, Any]:
    # decoded frames go to a throw-away store: artifacts/experiments is tracked, frames must not be
    with tempfile.TemporaryDirectory(prefix="conrad-real-smoke-") as tmp:
        return _run(config, seeds, Path(out_dir), Path(tmp))


def _run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: Path, tmp: Path) -> dict[str, Any]:
    manifest_id = str(config["manifest_id"])
    manifest_path = find_manifest(manifest_id)
    manifest = load_manifest(manifest_path)
    data_root = data_root_for(manifest.dataset_id)
    problems = verify_loaded_manifest(manifest, data_root)
    if problems:
        raise RuntimeError(f"{manifest_id} does not verify: {[str(p) for p in problems]}")

    store = ObjectStore(tmp / "object_store")
    ids = IdFactory(seed=int(seeds[0]), namespace=EXPERIMENT_ID)
    adapter = UvvidVideoAdapter(
        manifest,
        data_root,
        store,
        ids,
        UUID(int=int(seeds[0])),
        frame_stride=int(config["frame_stride"]),
        max_frames_per_sequence=int(config["max_frames_per_sequence"]),
        manifest_path=manifest_path,
    )
    audit = adapter.inspect()
    sequences = list(adapter.iter_sequences())

    per_sequence: dict[str, Any] = {}
    all_obs: list[Observation] = []
    clean_images: list[np.ndarray] = []
    for sample in sequences:
        obs = list(sample.observations)
        images = [store.get_array(o.payload_ref) for o in obs if o.payload_ref is not None]
        times = np.asarray([o.timestamp.time_ns for o in obs], dtype=np.int64)
        per_sequence[sample.sequence_id] = {
            "n_frames": len(obs),
            "frame_shape": list(images[0].shape),
            "time_span_s": float((times[-1] - times[0]) / 1e9),
            "median_dt_ms": float(np.median(np.diff(times)) / 1e6) if len(times) > 1 else 0.0,
            "pose_available": any(o.robot_pose_estimate is not None for o in obs),
            "calibration_available": any(o.calibration_ref is not None for o in obs),
            "quality": _quality_table([image_quality(im) for im in images]),
        }
        all_obs += obs
        clean_images += images

    clean_q = [image_quality(im) for im in clean_images]
    response: dict[str, Any] = {}
    encoding: dict[str, Any] = {}
    cfg_corr = config["corruptions"]
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        encoder = EcmerEncoder(
            CoreConfig(), IdFactory(seed=int(seed), namespace=f"{EXPERIMENT_ID}-enc"), loader=store.get_array
        )
        encoding[f"seed_{seed}"] = {"clean": _encode(encoder, all_obs)}
        for kind, features in EXPECTED.items():
            corrupted = [_corrupt(im, kind, cfg_corr, rng) for im in clean_images]
            cq = [image_quality(im) for im in corrupted]
            entry: dict[str, Any] = {}
            for feat, sign in features.items():
                deltas = np.asarray(
                    [getattr(c, feat) - getattr(q, feat) for c, q in zip(cq, clean_q, strict=True)],
                    dtype=np.float64,
                )
                entry[feat] = {
                    "expected_sign": sign,
                    "mean_delta": float(deltas.mean()),
                    "fraction_expected_direction": float((np.sign(deltas) == sign).mean()),
                }
            response.setdefault(kind, {})[f"seed_{seed}"] = entry
            refs = [store.put_array(im) for im in corrupted]
            corrupted_obs = [
                o.model_copy(update={"payload_ref": r, "observation_id": ids.new()})
                for o, r in zip(all_obs, refs, strict=True)
            ]
            encoding[f"seed_{seed}"][kind] = _encode(encoder, corrupted_obs)

    min_fraction = min(
        v["fraction_expected_direction"]
        for per_kind in response.values()
        for s in per_kind.values()
        for v in s.values()
    )
    threshold = float(config["pass_fraction"])
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "dataset": {
            "manifest_id": manifest_id,
            "manifest_digest": manifest.manifest_digest(),
            "files_digest": manifest.files_digest(),
            "license": manifest.license,
            "release_id": manifest.release_id,
            "adapter": f"{audit.adapter_name}@{audit.adapter_version}",
            "verify_problems": 0,
        },
        "n_sequences": len(sequences),
        "n_observations": len(all_obs),
        "sequences": per_sequence,
        "quality_all_frames": _quality_table(clean_q),
        "corruption_response": response,
        "ecmer_encoding_untrained": encoding,
        "min_fraction_expected_direction": min_fraction,
        "pass_fraction": threshold,
        "quality_response_check": "PASS" if min_fraction >= threshold else "FAIL",
        "limitations": [
            "2 short UVVID videos, subsampled; not a benchmark",
            "no labels, pose or calibration: nothing is scored against truth",
            "ECMER weights are random: reliability/OOD values support no claim",
            "corruptions are synthetic perturbations of real frames",
        ],
    }
    (out_dir / f"{EXPERIMENT_ID}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
