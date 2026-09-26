"""Typed sparse context input path for OS-FM.

The context path is representational only: exact physical measurements remain
available to Model2 through direct typed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn


class ContextFamily(str, Enum):
    PLATFORM_NAVIGATION = "platform_navigation"
    ENVIRONMENTAL = "environmental"
    SENSOR_ACQUISITION = "sensor_acquisition"
    ASSET_ENGINEERING = "asset_engineering"
    OPERATIONAL_PROCESS = "operational_process"
    DOCUMENT_KNOWLEDGE = "document_knowledge"
    MISSION_QUERY = "mission_query"
    EXTERNAL_CONTEXT = "external_context"


class ContextAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"

    @property
    def contributes_to_representation(self) -> bool:
        return self is ContextAvailability.AVAILABLE


@dataclass(frozen=True)
class ContextFieldSpec:
    name: str
    family: ContextFamily
    units: str
    width: int = 1
    description: str = ""


DEFAULT_CONTEXT_FIELDS: tuple[ContextFieldSpec, ...] = (
    ContextFieldSpec("pose_xyz_m", ContextFamily.PLATFORM_NAVIGATION, "m", 3),
    ContextFieldSpec("pose_covariance_diag_m2", ContextFamily.PLATFORM_NAVIGATION, "m^2", 3),
    ContextFieldSpec("orientation_rpy_rad", ContextFamily.PLATFORM_NAVIGATION, "rad", 3),
    ContextFieldSpec("angular_velocity_rad_s", ContextFamily.PLATFORM_NAVIGATION, "rad/s", 3),
    ContextFieldSpec("acceleration_m_s2", ContextFamily.PLATFORM_NAVIGATION, "m/s^2", 3),
    ContextFieldSpec("vehicle_depth_m", ContextFamily.PLATFORM_NAVIGATION, "m", 1),
    ContextFieldSpec("pressure_kpa", ContextFamily.PLATFORM_NAVIGATION, "kPa", 1),
    ContextFieldSpec("altitude_m", ContextFamily.PLATFORM_NAVIGATION, "m", 1),
    ContextFieldSpec("velocity_m_s", ContextFamily.PLATFORM_NAVIGATION, "m/s", 3),
    ContextFieldSpec("heading_rad", ContextFamily.PLATFORM_NAVIGATION, "rad", 1),
    ContextFieldSpec("navigation_quality", ContextFamily.PLATFORM_NAVIGATION, "unitless", 1),
    ContextFieldSpec("timing_uncertainty_ms", ContextFamily.PLATFORM_NAVIGATION, "ms", 1),
    ContextFieldSpec("temperature_c", ContextFamily.ENVIRONMENTAL, "degC", 1),
    ContextFieldSpec("salinity_psu", ContextFamily.ENVIRONMENTAL, "PSU", 1),
    ContextFieldSpec("turbidity_ntu", ContextFamily.ENVIRONMENTAL, "NTU", 1),
    ContextFieldSpec("ph", ContextFamily.ENVIRONMENTAL, "pH", 1),
    ContextFieldSpec("dissolved_oxygen_mg_l", ContextFamily.ENVIRONMENTAL, "mg/L", 1),
    ContextFieldSpec("conductivity_ms_cm", ContextFamily.ENVIRONMENTAL, "mS/cm", 1),
    ContextFieldSpec("current_speed_m_s", ContextFamily.ENVIRONMENTAL, "m/s", 1),
    ContextFieldSpec("current_direction_rad", ContextFamily.ENVIRONMENTAL, "rad", 1),
    ContextFieldSpec("camera_exposure_ms", ContextFamily.SENSOR_ACQUISITION, "ms", 1),
    ContextFieldSpec("camera_gain_db", ContextFamily.SENSOR_ACQUISITION, "dB", 1),
    ContextFieldSpec("sonar_frequency_khz", ContextFamily.SENSOR_ACQUISITION, "kHz", 1),
    ContextFieldSpec("sonar_range_m", ContextFamily.SENSOR_ACQUISITION, "m", 1),
    ContextFieldSpec("sonar_gain_db", ContextFamily.SENSOR_ACQUISITION, "dB", 1),
    ContextFieldSpec("calibration_quality", ContextFamily.SENSOR_ACQUISITION, "unitless", 1),
    ContextFieldSpec("sensor_health", ContextFamily.SENSOR_ACQUISITION, "unitless", 1),
    ContextFieldSpec("synchronization_quality", ContextFamily.SENSOR_ACQUISITION, "unitless", 1),
    ContextFieldSpec("sensor_pose_xyz_m", ContextFamily.SENSOR_ACQUISITION, "m", 3),
    ContextFieldSpec("sensor_orientation_rpy_rad", ContextFamily.SENSOR_ACQUISITION, "rad", 3),
)


@dataclass(frozen=True)
class ContextObservation:
    field: str
    value: tuple[float, ...]
    present: bool
    timestamp_ms: int
    units: str
    provenance: str
    uncertainty: tuple[float, ...] | None = None
    source_observation_id: str | None = None
    sensor_lineage: tuple[str, ...] = ()
    availability: ContextAvailability = ContextAvailability.AVAILABLE
    validity: bool = True
    quality: float | None = None
    freshness_ms: int | None = None
    coordinate_frame: str | None = None
    clock_domain: str | None = None
    calibration_ref: str | None = None
    notes: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return self.present and self.validity and self.availability.contributes_to_representation


@dataclass(frozen=True)
class ContextNormalizationStats:
    means: dict[str, tuple[float, ...]]
    stds: dict[str, tuple[float, ...]]
    fit_partitions: tuple[str, ...]


@dataclass(frozen=True)
class ContextBatch:
    raw_values: torch.Tensor
    values: torch.Tensor
    present_mask: torch.Tensor
    uncertainty: torch.Tensor
    uncertainty_mask: torch.Tensor
    timestamps_ms: torch.Tensor
    field_names: tuple[str, ...]
    field_widths: tuple[int, ...]
    units: tuple[str, ...]
    provenance: tuple[tuple[str, ...], ...]
    source_observation_ids: tuple[tuple[str | None, ...], ...]
    availability: tuple[tuple[str, ...], ...]
    validity_mask: torch.Tensor
    quality: torch.Tensor
    quality_mask: torch.Tensor


@dataclass(frozen=True)
class ContextEncoderOutput:
    tokens: torch.Tensor
    pooled: torch.Tensor
    available_mask: torch.Tensor
    field_names: tuple[str, ...]


def context_schema_summary(fields: tuple[ContextFieldSpec, ...] = DEFAULT_CONTEXT_FIELDS) -> dict[str, Any]:
    return {
        field.name: {"family": field.family.value, "units": field.units, "width": field.width}
        for field in fields
    }


def auxiliary_context_contract() -> dict[str, Any]:
    """Stable adapter contract for future non-perception context sources.

    Adapters may emit typed observations that become D_F=384 auxiliary tokens.
    Exact measurements and documents keep their separate provenance-bearing
    evidence paths; this contract is not a shortcut around Model2.
    """

    return {
        "token_dimension": 384,
        "supported_families": tuple(family.value for family in ContextFamily),
        "availability_states": tuple(state.value for state in ContextAvailability),
        "required_observation_fields": (
            "field",
            "value",
            "units",
            "timestamp_ms",
            "provenance",
            "availability",
            "validity",
        ),
        "current_training_families": (
            ContextFamily.PLATFORM_NAVIGATION.value,
            ContextFamily.ENVIRONMENTAL.value,
            ContextFamily.SENSOR_ACQUISITION.value,
        ),
        "future_hook_families": (
            ContextFamily.ASSET_ENGINEERING.value,
            ContextFamily.OPERATIONAL_PROCESS.value,
            ContextFamily.DOCUMENT_KNOWLEDGE.value,
            ContextFamily.MISSION_QUERY.value,
            ContextFamily.EXTERNAL_CONTEXT.value,
        ),
        "model2_dual_path": "exact physical measurements remain provenance-bearing Model2 evidence outside the learned token path",
        "host_boundary": "auxiliary tokens may condition inspection intelligence; host adapters retain execution authority",
    }


def fit_context_normalization(
    observations: tuple[tuple[ContextObservation, ...], ...],
    *,
    fields: tuple[ContextFieldSpec, ...] = DEFAULT_CONTEXT_FIELDS,
    fit_partitions: tuple[str, ...] = ("PRETRAIN_REAL",),
) -> ContextNormalizationStats:
    values: dict[str, list[torch.Tensor]] = {field.name: [] for field in fields}
    widths = {field.name: field.width for field in fields}
    for row in observations:
        for item in row:
            if item.is_available() and item.field in values:
                tensor = torch.tensor(item.value, dtype=torch.float32).flatten()
                if tensor.numel() != widths[item.field]:
                    raise ValueError(f"context field {item.field!r} expected width {widths[item.field]}")
                values[item.field].append(tensor)
    means: dict[str, tuple[float, ...]] = {}
    stds: dict[str, tuple[float, ...]] = {}
    for field in fields:
        if values[field.name]:
            stack = torch.stack(values[field.name])
            std = stack.std(dim=0, unbiased=False).clamp_min(1e-6)
            mean = stack.mean(dim=0)
        else:
            mean = torch.zeros(field.width)
            std = torch.ones(field.width)
        means[field.name] = tuple(float(x) for x in mean.tolist())
        stds[field.name] = tuple(float(x) for x in std.tolist())
    return ContextNormalizationStats(means=means, stds=stds, fit_partitions=fit_partitions)


def collate_context_observations(
    observations: tuple[tuple[ContextObservation, ...], ...],
    *,
    stats: ContextNormalizationStats,
    fields: tuple[ContextFieldSpec, ...] = DEFAULT_CONTEXT_FIELDS,
) -> ContextBatch:
    batch = len(observations)
    max_width = max(field.width for field in fields)
    values = torch.zeros(batch, len(fields), max_width, dtype=torch.float32)
    raw_values = torch.zeros(batch, len(fields), max_width, dtype=torch.float32)
    present = torch.zeros(batch, len(fields), dtype=torch.bool)
    uncertainty = torch.zeros(batch, len(fields), max_width, dtype=torch.float32)
    uncertainty_mask = torch.zeros(batch, len(fields), dtype=torch.bool)
    timestamps = torch.zeros(batch, len(fields), dtype=torch.int64)
    validity = torch.zeros(batch, len(fields), dtype=torch.bool)
    quality = torch.zeros(batch, len(fields), dtype=torch.float32)
    quality_mask = torch.zeros(batch, len(fields), dtype=torch.bool)
    provenance: list[tuple[str, ...]] = []
    source_ids: list[tuple[str | None, ...]] = []
    availability_rows: list[tuple[str, ...]] = []
    field_index = {field.name: idx for idx, field in enumerate(fields)}
    field_width = {field.name: field.width for field in fields}
    field_units = {field.name: field.units for field in fields}
    for row_idx, row in enumerate(observations):
        row_prov: list[str] = ["" for _ in fields]
        row_sources: list[str | None] = [None for _ in fields]
        row_availability: list[str] = [ContextAvailability.MISSING.value for _ in fields]
        seen: set[str] = set()
        for item in row:
            if item.field not in field_index:
                raise ValueError(f"unknown context field {item.field!r}")
            if item.field in seen:
                raise ValueError(f"duplicate context field {item.field!r} in sample")
            seen.add(item.field)
            if item.units != field_units[item.field]:
                raise ValueError(f"context field {item.field!r} uses {item.units!r}, expected {field_units[item.field]!r}")
            idx = field_index[item.field]
            width = field_width[item.field]
            raw = torch.tensor(item.value, dtype=torch.float32).flatten()
            if raw.numel() != width:
                raise ValueError(f"context field {item.field!r} expected width {width}")
            raw_values[row_idx, idx, :width] = raw
            if item.is_available():
                mean = torch.tensor(stats.means[item.field], dtype=torch.float32)
                std = torch.tensor(stats.stds[item.field], dtype=torch.float32)
                values[row_idx, idx, :width] = (raw - mean) / std
                present[row_idx, idx] = True
            validity[row_idx, idx] = item.validity
            if item.quality is not None:
                quality[row_idx, idx] = float(item.quality)
                quality_mask[row_idx, idx] = True
            if item.uncertainty is not None:
                unc = torch.tensor(item.uncertainty, dtype=torch.float32).flatten()
                if unc.numel() != width:
                    raise ValueError(f"context uncertainty for {item.field!r} expected width {width}")
                uncertainty[row_idx, idx, :width] = unc
                uncertainty_mask[row_idx, idx] = True
            timestamps[row_idx, idx] = int(item.timestamp_ms)
            row_prov[idx] = item.provenance
            row_sources[idx] = item.source_observation_id
            row_availability[idx] = item.availability.value
        provenance.append(tuple(row_prov))
        source_ids.append(tuple(row_sources))
        availability_rows.append(tuple(row_availability))
    return ContextBatch(
        raw_values=raw_values,
        values=values,
        present_mask=present,
        uncertainty=uncertainty,
        uncertainty_mask=uncertainty_mask,
        timestamps_ms=timestamps,
        field_names=tuple(field.name for field in fields),
        field_widths=tuple(field.width for field in fields),
        units=tuple(field.units for field in fields),
        provenance=tuple(provenance),
        source_observation_ids=tuple(source_ids),
        availability=tuple(availability_rows),
        validity_mask=validity,
        quality=quality,
        quality_mask=quality_mask,
    )


def direct_model2_evidence(context: ContextBatch) -> tuple[dict[str, Any], ...]:
    """Return exact present measurements for Model2 typed evidence paths."""

    out: list[dict[str, Any]] = []
    for row in range(context.values.shape[0]):
        row_items = []
        for idx, field in enumerate(context.field_names):
            if not bool(context.present_mask[row, idx]):
                continue
            row_items.append(
                {
                    "field": field,
                    "value": tuple(float(x) for x in context.raw_values[row, idx, : context.field_widths[idx]].tolist()),
                    "units": context.units[idx],
                    "timestamp_ms": int(context.timestamps_ms[row, idx]),
                    "provenance": context.provenance[row][idx],
                    "source_observation_id": context.source_observation_ids[row][idx],
                }
            )
        out.append({"model2_direct_measurements": tuple(row_items)})
    return tuple(out)


class ContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        fields: tuple[ContextFieldSpec, ...] = DEFAULT_CONTEXT_FIELDS,
        d_f: int = 384,
        width: int = 256,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if d_f != 384:
            raise ValueError("OS-FM context encoder must project to D_F=384")
        self.fields = fields
        self.d_f = d_f
        self.width = width
        self.max_value_width = max(field.width for field in fields)
        self.input = nn.Linear(self.max_value_width * 2 + 3, width)
        self.field_embeddings = nn.Parameter(torch.zeros(len(fields), width))
        self.family_embeddings = nn.Embedding(len(ContextFamily), width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=depth)
        self.proj = nn.Linear(width, d_f)
        self.pool_norm = nn.LayerNorm(d_f)
        self._family_index = {family: idx for idx, family in enumerate(ContextFamily)}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.field_embeddings, std=0.02)

    def forward(self, context: ContextBatch) -> ContextEncoderOutput:
        values = context.values.float()
        present = context.present_mask.bool()
        uncertainty = context.uncertainty.float()
        uncertainty_flag = context.uncertainty_mask.float().unsqueeze(-1)
        timestamps = context.timestamps_ms.float().unsqueeze(-1) / 1000.0
        present_feature = present.float().unsqueeze(-1)
        x = torch.cat([values, uncertainty, uncertainty_flag, timestamps, present_feature], dim=-1)
        h = self.input(x)
        device = h.device
        field_emb = self.field_embeddings.to(device=device, dtype=h.dtype).unsqueeze(0)
        family_ids = torch.tensor(
            [self._family_index[field.family] for field in self.fields],
            dtype=torch.long,
            device=device,
        )
        h = h + field_emb + self.family_embeddings(family_ids).to(dtype=h.dtype).unsqueeze(0)
        h = self.blocks(h, src_key_padding_mask=~present)
        tokens = self.proj(h)
        masked = tokens * present.unsqueeze(-1)
        denom = present.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = self.pool_norm(masked.sum(dim=1) / denom)
        return ContextEncoderOutput(
            tokens=tokens,
            pooled=pooled,
            available_mask=present,
            field_names=context.field_names,
        )
