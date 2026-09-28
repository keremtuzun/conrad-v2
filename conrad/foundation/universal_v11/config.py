"""Declarative configuration for OS-FM Universal V1.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from conrad.foundation.universal_v11.registry import EncoderFamily


@dataclass(frozen=True)
class TokenizerConfig:
    input_dim: int = 8
    max_tokens: int = 16


@dataclass(frozen=True)
class EncoderConfig:
    active: bool = True
    pretrained_weight_path: str | None = None
    frozen: bool = False
    learning_rate_multiplier: float = 1.0
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


@dataclass(frozen=True)
class FamilyFusionConfig:
    enabled: bool = True
    summary_tokens: int = 4
    depth: int = 2


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = True
    token_budget: int | None = None
    route_degraded: bool = True
    route_stale: bool = True
    route_uncalibrated: bool = True
    route_saturated: bool = True


@dataclass(frozen=True)
class TimingConfig:
    stale_after_s: float = 1.0
    max_skew_s: float = 0.5
    allow_out_of_order: bool = False


@dataclass(frozen=True)
class GlobalFusionConfig:
    d_f: int = 384
    scene_latents: int = 64
    cross_attention_blocks: int = 3
    self_attention_blocks: int = 6
    heads: int = 6


@dataclass(frozen=True)
class TemporalConfig:
    d_f: int = 384
    memory_tokens: int = 16
    blocks: int = 4
    max_windows: int = 10
    gap_reset_s: float = 1.0


@dataclass(frozen=True)
class UniversalV11Config:
    active_families: tuple[str, ...] = tuple(family.value for family in EncoderFamily)
    active_modalities: tuple[str, ...] = ()
    family_encoders: dict[str, EncoderConfig] = field(
        default_factory=lambda: {family.value: EncoderConfig() for family in EncoderFamily}
    )
    family_fusion: FamilyFusionConfig = field(default_factory=FamilyFusionConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    modality_dropout: float = 0.0
    use_state_embeddings: bool = True
    timing: TimingConfig = field(default_factory=TimingConfig)
    exact_evidence_enabled: bool = True
    sample_evidence_enabled: bool = True
    global_fusion: GlobalFusionConfig = field(default_factory=GlobalFusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    output_taps: tuple[str, ...] = ("rgb_dense_3_6_9_12", "sonar_dense_3_6_9_12")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> "UniversalV11Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            active_families=tuple(raw.get("active_families", ())),
            active_modalities=tuple(raw.get("active_modalities", ())),
            family_encoders={
                key: EncoderConfig(
                    active=value.get("active", True),
                    pretrained_weight_path=value.get("pretrained_weight_path"),
                    frozen=value.get("frozen", False),
                    learning_rate_multiplier=value.get("learning_rate_multiplier", 1.0),
                    tokenizer=TokenizerConfig(**value.get("tokenizer", {})),
                )
                for key, value in raw.get("family_encoders", {}).items()
            },
            family_fusion=FamilyFusionConfig(**raw.get("family_fusion", {})),
            router=RouterConfig(**raw.get("router", {})),
            modality_dropout=float(raw.get("modality_dropout", 0.0)),
            use_state_embeddings=bool(raw.get("use_state_embeddings", True)),
            timing=TimingConfig(**raw.get("timing", {})),
            exact_evidence_enabled=bool(raw.get("exact_evidence_enabled", True)),
            sample_evidence_enabled=bool(raw.get("sample_evidence_enabled", True)),
            global_fusion=GlobalFusionConfig(**raw.get("global_fusion", {})),
            temporal=TemporalConfig(**raw.get("temporal", {})),
            output_taps=tuple(raw.get("output_taps", ())),
        )
