"""Universal OS-FM V1.1 contracts, tokenizers, routing, fusion, and evidence paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn

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
    calibration_id: str | None = None
    provenance: str | None = None


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


@dataclass(frozen=True)
class ExactEvidenceRecord:
    name: str
    values: Mapping[str, float | int | bool | str]
    unit: str | None = None
    timestamp_s: float | None = None
    provenance: str | None = None


@dataclass(frozen=True)
class SampleLabResult:
    name: str
    result: Mapping[str, float | int | bool | str]
    sample_id: str
    method: str
    timestamp_s: float | None = None


@dataclass(frozen=True)
class UniversalOSFMOutput:
    token_sets: tuple[UniversalTokenSet, ...]
    routed_token_sets: tuple[UniversalTokenSet, ...]
    fusion: SceneFusionOutput
    temporal: TemporalMemoryOutput | None
    exact_evidence: tuple[ExactEvidenceRecord, ...]
    sample_lab_results: tuple[SampleLabResult, ...]
    router_weights: torch.Tensor


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
            [ts.state in {ModalityState.AVAILABLE, ModalityState.DEGRADED, ModalityState.STALE} for ts in token_sets],
            dtype=torch.bool,
            device=logits.device,
        )
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
            state = token_set.state
            if state == ModalityState.AVAILABLE and bool((skew > self.stale_after_s).any().detach().cpu()):
                state = ModalityState.STALE
            valid = token_set.valid_token_mask & (skew <= self.max_skew_s).unsqueeze(1)
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
    def __init__(self, input_dims: Mapping[str, int] | None = None) -> None:
        super().__init__()
        validate_registry()
        self.registry = registry_by_name()
        dims = {spec.name: 8 for spec in MODALITY_REGISTRY}
        if input_dims:
            dims.update(input_dims)
        self.tokenizers = nn.ModuleDict({name: GenericTokenizer(dim) for name, dim in dims.items()})
        self.embeddings = ModalityStateEmbeddings()
        self.router = SparseModalityRouter()
        self.aligner = AsyncTimestampAligner()
        self.fusion = SceneFusionTransformer(
            SceneFusionConfig(modalities=tuple(spec.name for spec in MODALITY_REGISTRY))
        )
        self.temporal = TemporalMemoryTransformer()

    def tokenize(self, inputs: tuple[UniversalModalityInput, ...]) -> tuple[UniversalTokenSet, ...]:
        token_sets: list[UniversalTokenSet] = []
        for item in inputs:
            spec = self.registry[item.name]
            if item.payload is None:
                continue
            tokens, mask = self.tokenizers[item.name](item.payload)
            type_index = list(self.registry).index(item.name)
            tokens = self.embeddings(type_index, spec.family, item.state, tokens)
            token_sets.append(UniversalTokenSet(item.name, spec.family, tokens, mask, item.state, item.timestamp_s, item.metadata))
        return tuple(token_sets)

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
        aligned = self.aligner.align(token_sets, reference_time_s)
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
        return UniversalOSFMOutput(token_sets, routed, fusion, temporal, exact_evidence, sample_lab_results, router_weights)
