"""Observation -> model input tensors. Payloads are resolved through an injected loader (no I/O here).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch

from conrad.core.config import CoreConfig
from conrad.core.ecmer.fusion import EventBatch
from conrad.core.ecmer.quality_features import RawQuality, image_quality, scalar_quality, sonar_quality
from conrad.schemas.observation import Modality, Observation, PayloadRef

PayloadLoader = Callable[[PayloadRef], np.ndarray]

IMAGE_MODALITIES = {Modality.RGB: "rgb", Modality.SONAR: "sonar"}
POINT_MODALITIES = {Modality.POINT_CLOUD, Modality.DEPTH_RANGE}


def stable_index(key: str, size: int) -> int:
    return zlib.crc32(key.encode("utf-8")) % size


def payload_array(obs: Observation, loader: PayloadLoader | None) -> np.ndarray:
    if obs.payload_ref is not None:
        if loader is None:
            raise ValueError(f"observation {obs.observation_id} has a payload_ref but no loader was injected")
        return np.asarray(loader(obs.payload_ref))
    assert obs.inline_values is not None
    return np.asarray(obs.inline_values, dtype=np.float64)


def _image_tensor(arr: np.ndarray, channels: int, size: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
    a = np.nan_to_num(a).astype(np.float32)
    if a.max() > 1.0:
        a = (a / 255.0).astype(np.float32)
    a = np.asarray(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA), dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] != channels:
        a = np.repeat(a.mean(-1, keepdims=True), channels, axis=-1).astype(np.float32)
    return np.moveaxis(a, -1, 0)


def context_vector(obs: Observation, delta_t_s: float) -> np.ndarray:
    """position(3) quat(4) covariance diag summary(6) log1p(delta_t)(1). Unknown pose -> zeros."""
    v = np.zeros(14, dtype=np.float32)
    pose = obs.robot_pose_estimate
    if pose is not None:
        v[0:3] = pose.position_m
        v[3:7] = pose.orientation_wxyz
        if pose.covariance_6x6 is not None:
            c = np.asarray(pose.covariance_6x6).reshape(6, 6)
            v[7:13] = np.sqrt(np.clip(np.diag(c), 0, None))
    v[13] = np.log1p(max(delta_t_s, 0.0))
    return v


def raw_quality_for(obs: Observation, arr: np.ndarray) -> RawQuality:
    if obs.modality is Modality.RGB:
        return image_quality(arr)
    if obs.modality is Modality.SONAR:
        return sonar_quality(arr)
    return scalar_quality(arr)


def build_event_batch(
    observations: Sequence[Observation],
    arrays: Sequence[np.ndarray],
    qualities: Sequence[RawQuality],
    cfg: CoreConfig,
    delta_t_s: Sequence[float] | None = None,
) -> EventBatch:
    """One row per observation; each row carries only its own modality (others are masked)."""
    e = cfg.ecmer
    b = len(observations)
    dts = list(delta_t_s) if delta_t_s is not None else [0.0] * b
    avail = {n: torch.zeros(b, dtype=torch.bool) for n in ("rgb", "sonar", "points", "scalar")}
    rgb = sonar = pts = pts_mask = scal = scal_present = None
    size = e.image_size
    n_points = max(
        [
            len(a.reshape(-1, 3))
            for o, a in zip(observations, arrays, strict=True)
            if o.modality in POINT_MODALITIES and a.size % 3 == 0
        ]
        or [1]
    )
    for i, (obs, arr) in enumerate(zip(observations, arrays, strict=True)):
        if obs.modality in IMAGE_MODALITIES:
            name = IMAGE_MODALITIES[obs.modality]
            ch = e.image_channels_rgb if name == "rgb" else e.image_channels_sonar
            if name == "rgb":
                rgb = torch.zeros(b, ch, size, size) if rgb is None else rgb
                rgb[i] = torch.from_numpy(_image_tensor(arr, ch, size))
            else:
                sonar = torch.zeros(b, ch, size, size) if sonar is None else sonar
                sonar[i] = torch.from_numpy(_image_tensor(arr, ch, size))
            avail[name][i] = True
        elif obs.modality in POINT_MODALITIES and arr.size % 3 == 0 and arr.size > 0:
            p = np.nan_to_num(arr.reshape(-1, 3).astype(np.float32))
            pts = torch.zeros(b, n_points, 3) if pts is None else pts
            pts_mask = torch.zeros(b, n_points, dtype=torch.bool) if pts_mask is None else pts_mask
            pts[i, : len(p)] = torch.from_numpy(p)
            pts_mask[i, : len(p)] = torch.from_numpy(np.isfinite(arr.reshape(-1, 3)).all(-1))
            avail["points"][i] = True
        else:
            flat = arr.reshape(-1).astype(np.float32)[: e.scalar_max_values]
            scal = torch.zeros(b, e.scalar_max_values) if scal is None else scal
            scal_present = (
                torch.zeros(b, e.scalar_max_values, dtype=torch.bool)
                if scal_present is None
                else scal_present
            )
            scal[i, : len(flat)] = torch.from_numpy(np.nan_to_num(flat))
            scal_present[i, : len(flat)] = torch.from_numpy(np.isfinite(flat))
            avail["scalar"][i] = True
    return EventBatch(
        batch_size=b,
        context_raw=torch.from_numpy(
            np.stack([context_vector(o, d) for o, d in zip(observations, dts, strict=True)])
        ),
        sensor_idx=torch.tensor([stable_index(str(o.sensor_id), e.max_sensors) for o in observations]),
        frame_idx=torch.tensor([stable_index(o.sensor_frame, e.max_frames) for o in observations]),
        raw_quality=torch.from_numpy(np.stack([q.vector(e.quality_feature_dim) for q in qualities])),
        rgb=rgb,
        sonar=sonar,
        points=pts,
        points_mask=pts_mask,
        scalars=scal,
        scalar_present=scal_present,
        avail=avail,
    )
