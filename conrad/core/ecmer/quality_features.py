"""Deterministic hand-crafted quality features (ch4 Quality estimator: engineered + learned).

All values are measured from the payload itself. A feature that cannot be measured is returned as
``None`` in :class:`QualityContext` and as 0 with a missingness bit in the feature vector.

implementation_status: EXPERIMENTAL_CANDIDATE (normalisation constants are ENGINEERING_ESTIMATE)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LAPLACIAN_SHARP_VAR = 500.0
"""Laplacian variance (8-bit scale) treated as fully sharp. ENGINEERING_ESTIMATE."""


@dataclass(frozen=True)
class RawQuality:
    blur: float | None = None  # 0 sharp .. 1 fully blurred
    brightness: float | None = None  # mean intensity in [0, 1]
    contrast: float | None = None  # std intensity in [0, 1]
    snr_db: float | None = None
    saturation_fraction: float | None = None
    valid_fraction: float | None = None  # finite, non-missing fraction of the payload

    def vector(self, dim: int) -> np.ndarray:
        """[value, present] pairs, truncated/padded to ``dim``. SNR is squashed to [0, 1]."""
        snr = None if self.snr_db is None else float(np.clip(self.snr_db / 40.0, 0.0, 1.0))
        values = [self.blur, self.brightness, self.contrast, snr]
        flat: list[float] = []
        for v in values:
            flat += [0.0 if v is None else float(v), 0.0 if v is None else 1.0]
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, len(flat))
        out[:n] = flat[:n]
        return out


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
    if arr.ndim == 3:
        arr = arr.mean(axis=-1) if arr.shape[-1] > 1 else arr[..., 0]
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    if arr.max() <= 1.0 + 1e-9:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def image_quality(image: np.ndarray) -> RawQuality:
    """Blur by Laplacian variance, brightness/contrast by intensity moments, SNR by mean/std."""
    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return RawQuality(valid_fraction=0.0)
    gray = _to_gray_u8(image)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    g = gray.astype(np.float64) / 255.0
    mean, std = float(g.mean()), float(g.std())
    # noise estimate: MAD of the Laplacian residual (robust to edges), Immerkaer-style scale
    residual = cv2.Laplacian(gray, cv2.CV_64F) / 255.0
    sigma_n = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826 / np.sqrt(20.0)
    snr = 20.0 * np.log10(max(mean, 1e-6) / max(sigma_n, 1e-6))
    return RawQuality(
        blur=float(np.clip(1.0 - lap_var / LAPLACIAN_SHARP_VAR, 0.0, 1.0)),
        brightness=mean,
        contrast=std,
        snr_db=float(snr),
        saturation_fraction=float(((gray <= 2) | (gray >= 253)).mean()),
        valid_fraction=float(finite.mean()),
    )


def sonar_quality(intensity: np.ndarray, noise_floor_rows: int = 4) -> RawQuality:
    """SNR from the ratio of mean return to the noise floor estimated on the nearest range rows."""
    arr = np.asarray(intensity, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=-1)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return RawQuality(valid_fraction=0.0)
    arr = np.where(finite, arr, 0.0)
    floor = arr[: max(1, min(noise_floor_rows, arr.shape[0]))]
    noise = float(floor.std()) + 1e-9
    signal = float(np.abs(arr).mean())
    peak = float(np.abs(arr).max()) or 1.0
    return RawQuality(
        brightness=float(np.clip(signal / peak, 0.0, 1.0)),
        contrast=float(np.clip(arr.std() / peak, 0.0, 1.0)),
        snr_db=float(20.0 * np.log10(max(signal, 1e-9) / noise)),
        valid_fraction=float(finite.mean()),
    )


def scalar_quality(values: np.ndarray) -> RawQuality:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return RawQuality(valid_fraction=0.0)
    return RawQuality(valid_fraction=float(np.isfinite(arr).mean()))
