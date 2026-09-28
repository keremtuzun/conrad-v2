"""Universal OS-FM V1.1 contracts, tokenizers, routing, fusion, and evidence paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn

from conrad.foundation.context import ContextEncoder
from conrad.foundation.encoders.geometry import GeometryEncoderConfig, GeometryGroupedEncoder
from conrad.foundation.encoders.range import RangeEncoderConfig, RangeViTP8Encoder
from conrad.foundation.encoders.rgb import RGBEncoderConfig, RGBViTS14Encoder
from conrad.foundation.encoders.sonar import SonarEncoderConfig, SonarViTS14Encoder
from conrad.foundation.fusion.scene_fusion import ModalityTokenSet, SceneFusionConfig, SceneFusionOutput, SceneFusionTransformer
from conrad.foundation.temporal.memory import TemporalMemoryOutput, TemporalMemoryTransformer
from conrad.foundation.universal_v11.registry import (
    EncoderFamily,
    MODALITY_REGISTRY,
    ModalityState,
    registry_by_name,
    validate_registry,
)


D_F = 384
SCENE_LATENTS = 64


@dataclass(frozen=True)
class SensorMetadata:
    quality: float = 1.0
    calibrated: bool = True
    acquisition_time_s: float = 0.0
    timing_uncertainty_s: float = 0.0
    calibration_id: str | None = None
    provenance: str | None = None
    units: str | None = None
    physical_range: tuple[float, float] | None = None
    acquisition_settings: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    sensor_pose: tuple[float, ...] | None = None
    platform_pose: tuple[float, ...] | None = None
    spatial_support: str | None = None


@dataclass(frozen=True)
class UniversalModalityInput:
    name: str
    payload: torch.Tensor | None
    state: ModalityState
    timestamp_s: torch.Tensor
    metadata: SensorMetadata = field(default_factory=SensorMetadata)


@dataclass(frozen=True)
class UniversalTokenSet:
    name: str
    family: EncoderFamily
    tokens: torch.Tensor
    valid_token_mask: torch.Tensor
    state: ModalityState
    timestamp_s: torch.Tensor
    metadata: SensorMetadata
    dense_taps: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactEvidenceRecord:
    name: str
    values: Mapping[str, float | int | bool | str]
    unit: str | None = None
    timestamp_s: float | None = None
    provenance: str | None = None
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    spatial_location: tuple[float, ...] | None = None
    sensor: str | None = None
    calibration: str | None = None
    validity: bool = True
    observation_conditions: Mapping[str, float | int | bool | str] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleLabResult:
    name: str
    result: Mapping[str, float | int | bool | str]
    sample_id: str
    method: str
    timestamp_s: float | None = None
    sampling_location: tuple[float, ...] | None = None
    depth_m: float | None = None
    handling_metadata: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    provenance: str | None = None


@dataclass(frozen=True)
class UniversalOSFMOutput:
    token_sets: tuple[UniversalTokenSet, ...]
    family_token_sets: tuple[UniversalTokenSet, ...]
    routed_token_sets: tuple[UniversalTokenSet, ...]
    fusion: SceneFusionOutput
    temporal: TemporalMemoryOutput | None
    exact_evidence: tuple[ExactEvidenceRecord, ...]
    sample_lab_results: tuple[SampleLabResult, ...]
    router_weights: torch.Tensor
    modality_representations: Mapping[str, torch.Tensor]
    family_representations: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class FamilyEncoderConfig:
    input_dim: int = D_F
    width: int = D_F
    depth: int = 6
    heads: int = 6
    mlp_ratio: int = 4
    output_dim: int = D_F


class V1CompatibilityBank(nn.Module):
    """Retains V1 modules so qualified V1 weights remain directly loadable."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb = RGBViTS14Encoder(RGBEncoderConfig())
        self.sonar = SonarViTS14Encoder(SonarEncoderConfig())
        self.range = RangeViTP8Encoder(RangeEncoderConfig())
        self.geometry = GeometryGroupedEncoder(GeometryEncoderConfig())
        self.context = ContextEncoder()


class GenericTokenizer(nn.Module):
    """Small family tokenizer that maps tensors to 384-D tokens without claiming trained semantics."""

    def __init__(self, input_dim: int, max_tokens: int = 16, d_f: int = D_F) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.max_tokens = max_tokens
        self.proj = nn.Linear(input_dim, d_f)
        self.norm = nn.LayerNorm(d_f)

    def forward(self, payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if payload.ndim < 2:
            raise ValueError("payload must have batch dimension and features")
        batch = payload.shape[0]
        flat = payload.reshape(batch, -1, self.input_dim)
        if flat.shape[1] > self.max_tokens:
            flat = flat[:, : self.max_tokens]
        tokens = self.norm(self.proj(flat.float()))
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


class UniversalFamilyEncoder(nn.Module):
    """Reusable family branch that keeps geometry-specific tokenizers outside the encoder."""

    def __init__(self, config: FamilyEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or FamilyEncoderConfig()
        if self.config.output_dim != D_F:
            raise ValueError("Universal V1.1 family encoders must adapt to D_F=384")
        self.input_projection = nn.Linear(self.config.input_dim, self.config.width)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.config.width,
                    nhead=self.config.heads,
                    dim_feedforward=self.config.width * self.config.mlp_ratio,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.config.depth)
            ]
        )
        self.output_projection = nn.Linear(self.config.width, self.config.output_dim)
        self.norm = nn.LayerNorm(self.config.output_dim)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.input_dim:
            raise ValueError("family encoder tokens must be [B, T, 384]")
        key_padding = ~valid_mask.bool()
        x = self.input_projection(tokens)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=key_padding)
        return self.norm(self.output_projection(x)), valid_mask.bool()


class FamilyLocalFusion(nn.Module):
    """Summarizes one or more concrete modalities from the same family before global fusion."""

    def __init__(self, family: EncoderFamily, d_f: int = D_F, summary_tokens: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.family = family
        self.summary_tokens = summary_tokens
        self.latents = nn.Parameter(torch.zeros(1, summary_tokens, d_f))
        self.cross = nn.MultiheadAttention(d_f, 6, dropout=0.0, batch_first=True)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_f,
                    nhead=6,
                    dim_feedforward=d_f * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_f)
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(self, token_sets: tuple[UniversalTokenSet, ...]) -> UniversalTokenSet:
        if not token_sets:
            raise ValueError("family fusion requires at least one token set")
        batch = token_sets[0].tokens.shape[0]
        context = torch.cat([ts.tokens for ts in token_sets], dim=1)
        valid = torch.cat([ts.valid_token_mask for ts in token_sets], dim=1).bool()
        query = self.latents.expand(batch, -1, -1).to(device=context.device, dtype=context.dtype)
        fused, _ = self.cross(query, context, context, key_padding_mask=~valid, need_weights=False)
        for block in self.blocks:
            fused = block(fused)
        family_state = ModalityState.AVAILABLE if any(ts.state == ModalityState.AVAILABLE for ts in token_sets) else token_sets[0].state
        timestamps = torch.stack([ts.timestamp_s for ts in token_sets], dim=0).amin(dim=0)
        return UniversalTokenSet(
            name=f"{self.family.value}_family",
            family=self.family,
            tokens=self.norm(fused),
            valid_token_mask=torch.ones(batch, self.summary_tokens, dtype=torch.bool, device=context.device),
            state=family_state,
            timestamp_s=timestamps,
            metadata=token_sets[0].metadata,
        )


class UniversalAdapter(nn.Module):
    """Adapter contract: family-specific learned output -> [B, T, 384] tokens."""

    def __init__(self, input_dim: int, d_f: int = D_F) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_f)
        self.norm = nn.LayerNorm(d_f)

    def forward(self, outputs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(1)
        if outputs.ndim != 3:
            raise ValueError("adapter output must be [B, T, C] or [B, C]")
        tokens = self.norm(self.proj(outputs.float()))
        if valid_mask is None:
            valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError("valid_mask must match [B, T]")
        return tokens, valid_mask.bool()


class ModalityStateEmbeddings(nn.Module):
    def __init__(self, d_f: int = D_F) -> None:
        super().__init__()
        self.family = nn.Embedding(len(EncoderFamily), d_f)
        self.state = nn.Embedding(len(ModalityState), d_f)
        self.modality_type = nn.Embedding(len(MODALITY_REGISTRY), d_f)

    def forward(self, type_index: int, family: EncoderFamily, state: ModalityState, tokens: torch.Tensor) -> torch.Tensor:
        family_index = list(EncoderFamily).index(family)
        state_index = list(ModalityState).index(state)
        emb = self.family.weight[family_index] + self.state.weight[state_index] + self.modality_type.weight[type_index]
        return tokens + emb.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1)


class SparseModalityRouter(nn.Module):
    def __init__(self, d_f: int = D_F) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, 1))

    def forward(self, token_sets: tuple[UniversalTokenSet, ...]) -> tuple[tuple[UniversalTokenSet, ...], torch.Tensor]:
        if not token_sets:
            raise ValueError("router requires at least one token set")
        pooled = torch.stack(
            [
                (ts.tokens * ts.valid_token_mask.unsqueeze(-1)).sum(dim=1)
                / ts.valid_token_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                for ts in token_sets
            ],
            dim=1,
        )
        logits = self.score(pooled).squeeze(-1)
        allowed = torch.tensor(
            [ts.state.can_route for ts in token_sets],
            dtype=torch.bool,
            device=logits.device,
        )
        if not bool(allowed.any().detach().cpu()):
            raise ValueError("no routeable V1.1 modality/family is available")
        logits = logits.masked_fill(~allowed.unsqueeze(0), -1e9)
        weights = torch.softmax(logits, dim=1)
        routed = []
        for idx, ts in enumerate(token_sets):
            routed.append(
                UniversalTokenSet(
                    ts.name,
                    ts.family,
                    ts.tokens * weights[:, idx].view(-1, 1, 1),
                    ts.valid_token_mask,
                    ts.state,
                    ts.timestamp_s,
                    ts.metadata,
                )
            )
        return tuple(routed), weights


class AsyncTimestampAligner:
    def __init__(self, stale_after_s: float = 1.0, max_skew_s: float = 0.5) -> None:
        self.stale_after_s = stale_after_s
        self.max_skew_s = max_skew_s

    def align(self, token_sets: tuple[UniversalTokenSet, ...], reference_time_s: torch.Tensor) -> tuple[UniversalTokenSet, ...]:
        aligned: list[UniversalTokenSet] = []
        for token_set in token_sets:
            skew = (reference_time_s - token_set.timestamp_s).abs()
            timing_slack = float(token_set.metadata.timing_uncertainty_s)
            state = token_set.state
            if state == ModalityState.AVAILABLE and bool((skew > self.stale_after_s + timing_slack).any().detach().cpu()):
                state = ModalityState.STALE
            valid = token_set.valid_token_mask & (skew <= self.max_skew_s + timing_slack).unsqueeze(1)
            aligned.append(
                UniversalTokenSet(
                    token_set.name,
                    token_set.family,
                    token_set.tokens,
                    valid,
                    state,
                    token_set.timestamp_s,
                    token_set.metadata,
                )
            )
        return tuple(aligned)


class CorrespondenceGraph:
    def __init__(self) -> None:
        self._eligible = {
            (EncoderFamily.VISUAL_IMAGE, EncoderFamily.GEOMETRY_SPATIAL),
            (EncoderFamily.VISUAL_IMAGE, EncoderFamily.ACTIVE_ACOUSTIC),
            (EncoderFamily.ACTIVE_ACOUSTIC, EncoderFamily.GEOMETRY_SPATIAL),
            (EncoderFamily.PASSIVE_ACOUSTIC_VIBRATION, EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES),
            (EncoderFamily.ULTRASONIC_NDT, EncoderFamily.ELECTROMAGNETIC_MAGNETIC),
            (EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, EncoderFamily.BIOLOGICAL_MOLECULAR),
            (EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, EncoderFamily.MICROSCOPIC_PARTICLE),
            (EncoderFamily.GEOPHYSICAL_SEISMIC, EncoderFamily.GEOMETRY_SPATIAL),
            (EncoderFamily.RADIOLOGICAL, EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL),
            (EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT, EncoderFamily.VISUAL_IMAGE),
            (EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT, EncoderFamily.ACTIVE_ACOUSTIC),
        }

    def eligible(self, left: EncoderFamily, right: EncoderFamily) -> bool:
        if left == right:
            return True
        return (left, right) in self._eligible or (right, left) in self._eligible


class UniversalOSFMV11(nn.Module):
    def __init__(
        self,
        input_dims: Mapping[str, int] | None = None,
        *,
        family_encoder_config: FamilyEncoderConfig | None = None,
        include_v1_bank: bool = True,
    ) -> None:
        super().__init__()
        validate_registry()
        self.registry = registry_by_name()
        dims = {spec.name: 8 for spec in MODALITY_REGISTRY}
        if input_dims:
            dims.update(input_dims)
        self.tokenizers = nn.ModuleDict({name: GenericTokenizer(dim) for name, dim in dims.items()})
        self.embeddings = ModalityStateEmbeddings()
        cfg = family_encoder_config or FamilyEncoderConfig()
        self.family_encoders = nn.ModuleDict(
            {family.value: UniversalFamilyEncoder(cfg) for family in EncoderFamily}
        )
        self.family_fusion = nn.ModuleDict(
            {family.value: FamilyLocalFusion(family) for family in EncoderFamily}
        )
        self.router = SparseModalityRouter()
        self.aligner = AsyncTimestampAligner()
        self.fusion = SceneFusionTransformer(
            SceneFusionConfig(modalities=tuple(f"{family.value}_family" for family in EncoderFamily))
        )
        self.temporal = TemporalMemoryTransformer()
        self.v1 = V1CompatibilityBank() if include_v1_bank else None

    def tokenize(self, inputs: tuple[UniversalModalityInput, ...]) -> tuple[UniversalTokenSet, ...]:
        token_sets: list[UniversalTokenSet] = []
        for item in inputs:
            spec = self.registry[item.name]
            if item.payload is None:
                continue
            tokens, mask = self.tokenizers[item.name](item.payload)
            type_index = list(self.registry).index(item.name)
            tokens = self.embeddings(type_index, spec.family, item.state, tokens)
            encoded, mask = self.family_encoders[spec.family.value](tokens, mask)
            token_sets.append(UniversalTokenSet(item.name, spec.family, encoded, mask, item.state, item.timestamp_s, item.metadata))
        return tuple(token_sets)

    def fuse_families(self, token_sets: tuple[UniversalTokenSet, ...]) -> tuple[UniversalTokenSet, ...]:
        grouped: dict[EncoderFamily, list[UniversalTokenSet]] = {family: [] for family in EncoderFamily}
        for token_set in token_sets:
            grouped[token_set.family].append(token_set)
        return tuple(
            self.family_fusion[family.value](tuple(items))
            for family, items in grouped.items()
            if items
        )

    def forward(
        self,
        inputs: tuple[UniversalModalityInput, ...],
        *,
        reference_time_s: torch.Tensor,
        exact_evidence: tuple[ExactEvidenceRecord, ...] = (),
        sample_lab_results: tuple[SampleLabResult, ...] = (),
        temporal_timestamps_s: torch.Tensor | None = None,
        temporal_valid_mask: torch.Tensor | None = None,
    ) -> UniversalOSFMOutput:
        token_sets = self.tokenize(inputs)
        family_sets = self.fuse_families(token_sets)
        aligned = self.aligner.align(family_sets, reference_time_s)
        routed, router_weights = self.router(aligned)
        fusion_sets = tuple(
            ModalityTokenSet(
                ts.name,
                ts.tokens,
                ts.valid_token_mask,
                torch.full(ts.tokens.shape[:1], ts.state == ModalityState.MISSING, dtype=torch.bool, device=ts.tokens.device),
                torch.zeros(ts.tokens.shape[:1], dtype=torch.bool, device=ts.tokens.device),
                torch.zeros(ts.tokens.shape[:1], dtype=torch.bool, device=ts.tokens.device),
            )
            for ts in routed
            if ts.state not in {ModalityState.UNSUPPORTED, ModalityState.FAILED, ModalityState.INVALID}
        )
        fusion = self.fusion(fusion_sets)
        temporal = None
        if temporal_timestamps_s is not None and temporal_valid_mask is not None:
            temporal = self.temporal(fusion.global_repr.unsqueeze(1), temporal_timestamps_s, temporal_valid_mask)
        return UniversalOSFMOutput(
            token_sets=token_sets,
            family_token_sets=family_sets,
            routed_token_sets=routed,
            fusion=fusion,
            temporal=temporal,
            exact_evidence=exact_evidence,
            sample_lab_results=sample_lab_results,
            router_weights=router_weights,
            modality_representations={ts.name: ts.tokens.mean(dim=1) for ts in token_sets},
            family_representations={ts.name: ts.tokens.mean(dim=1) for ts in family_sets},
            diagnostics={
                "registry_size": len(self.registry),
                "family_count": len(EncoderFamily),
                "active_input_count": len(inputs),
                "routed_family_count": len(routed),
            },
        )
