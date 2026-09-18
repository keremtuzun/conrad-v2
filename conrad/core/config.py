"""Frozen configuration for Model 2 Core. Defaults follow spec ch33; nothing is hard-coded in modules.

implementation_status: EXPERIMENTAL_CANDIDATE
source_sections: [ch3 Universal dimensions, ch33 Shared dimensions and primitives]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from conrad.schemas.base import ARCHITECTURE_ID, ConradModel


class EcmerConfig(ConradModel):
    image_size: int = Field(default=224, gt=0)
    image_channels_rgb: int = 3
    image_channels_sonar: int = 1
    resnet_base_width: int = Field(default=64, gt=0, description="ResNet-18 uses 64 (final 512)")
    patch_grid: int = Field(default=4, gt=0)
    voxel_grid: int = Field(default=32, gt=0)
    voxel_channels: tuple[int, int, int] = (32, 64, 128)
    voxel_groups: int = Field(default=8, gt=0)
    voxel_extent_m: float = Field(default=4.0, gt=0)
    scalar_max_values: int = Field(default=16, gt=0)
    scalar_hidden: int = Field(default=128, gt=0)
    context_hidden: tuple[int, int] = (16, 128)
    max_sensors: int = Field(default=64, gt=0)
    max_frames: int = Field(default=32, gt=0)
    fusion_layers: int = Field(default=4, gt=0)
    quality_feature_dim: int = Field(default=8, gt=0)
    quality_hidden: tuple[int, int] = (256, 128)
    log_var_min: float = -10.0
    log_var_max: float = 5.0
    projection_hidden: int = Field(default=1024, gt=0)
    modality_dropout_p: float = Field(default=0.25, ge=0, le=1)
    info_nce_temperature: float = Field(default=0.1, gt=0)
    usable_threshold: float = Field(default=0.5, ge=0, le=1)
    degraded_reliability: float = Field(default=0.5, ge=0, le=1)
    encoder_version: str = "ecmer-v0.2"
    # ablation switches (ch4 Baselines)
    use_reliability: bool = True
    use_modality_id: bool = True
    use_alignment: bool = True
    use_source_independence: bool = True
    use_disagreement: bool = True
    use_modality_dropout: bool = True


class AssociationConfig(ConradModel):
    max_candidates: int = Field(default=32, gt=0)
    gate_distance_m: float = Field(default=5.0, gt=0)
    gate_sigma_multiplier: float = Field(default=3.0, ge=0)
    gate_max_age_s: float | None = Field(default=None, gt=0)
    relative_pose_dim: int = 12
    time_feature_dim: int = 4
    hidden: tuple[int, int, int] = (512, 256, 128)
    no_match_hidden: int = 128
    accept_probability: float = Field(default=0.60, ge=0, le=1)
    accept_margin: float = Field(default=0.15, ge=0, le=1)
    ranking_margin: float = Field(default=0.20, ge=0)
    focal_gamma: float = Field(default=2.0, ge=0)
    loss_weight_ce: float = 1.0
    loss_weight_focal: float = 1.0
    loss_weight_rank: float = 1.0
    baseline_gate_distance_m: float = Field(default=1.0, gt=0)


class BuoConfig(ConradModel):
    quality_dim: int = 4
    trunk_hidden: int = 512
    trust_hidden: int = 128
    innovation_hidden: int = 512
    gate_hidden: int = 128
    uncertainty_hidden: tuple[int, int] = (256, 128)
    pooling_blocks: int = Field(default=2, gt=0)
    innovation_l2: float = 1e-4
    use_trust: bool = True
    use_innovation_gate: bool = True
    use_contradiction: bool = True
    collapse_trust_innovation: bool = False


class AnalyticBuoConfig(ConradModel):
    """Interpretable, non-learned default runtime operator. All constants are ENGINEERING_ESTIMATE."""

    min_variance: float = Field(default=1e-6, gt=0)
    reliability_floor: float = Field(default=0.05, gt=0, le=1)
    corrupt_reliability: float = Field(default=0.5, ge=0, le=1)
    credible_reliability: float = Field(default=0.6, ge=0, le=1)
    contradiction_sigma: float = Field(default=3.0, gt=0, description="normalised innovation gate")
    contradiction_gain: float = Field(default=0.25, ge=0)
    contradiction_relief: float = Field(default=0.5, ge=0, le=1)
    contradiction_state_weight: float = Field(default=0.5, ge=0, le=1)
    ua_smoothing: float = Field(default=0.5, ge=0, le=1)
    ue_smoothing: float = Field(default=0.5, ge=0, le=1)
    coverage_per_independent_obs: float = Field(default=0.35, gt=0, le=1)
    default_prior_variance: float = Field(default=1.0, gt=0)
    max_contradiction_memory: int = Field(default=16, gt=0)


class RbpConfig(ConradModel):
    layers: int = Field(default=3, gt=0)
    adaptive_stop: bool = False
    adaptive_epsilon: float = 1e-4
    relation_types: int = Field(default=16, gt=0)
    relation_numeric_dim: int = Field(default=6, gt=0, description="geometry(3) distance age confidence")
    relation_hidden: int = 128
    gate_hidden: int = 256
    typed: bool = True
    contamination_weight: float = 1.0
    analytic_gain: float = Field(default=0.25, ge=0, le=1)
    inferred_uo_floor_ratio: float = Field(default=1.0, ge=1.0)


class TbdConfig(ConradModel):
    context_dim: int = 64
    stem_hidden: int = 512
    drift_hidden: int = 128
    mlp_hidden: int = 256
    unroll_length: int = 32
    analytic_process_noise_per_s: float = Field(
        default=1e-4, ge=0, description="ENGINEERING_ESTIMATE variance rate when a child declares no dynamics"
    )
    analytic_uo_time_constant_s: float = Field(default=3600.0, gt=0)


class PmblConfig(ConradModel):
    confirm_independent_observations: int = Field(default=2, gt=0)
    reject_after_s: float = Field(default=600.0, gt=0)
    dormant_after_s: float = Field(default=3600.0, gt=0)
    retire_after_s: float | None = Field(default=None, gt=0)
    late_evidence_policy: str = Field(default="REVISE_LATE", pattern="^(REVISE_LATE|REJECT)$")
    working_set_limit: int = Field(default=1024, gt=0)
    history_decay: float = Field(default=0.9, ge=0, le=1)


class CoreConfig(ConradModel):
    architecture_id: str = ARCHITECTURE_ID
    model_version: str = "model2-core-v0.2"
    evidence_dim: int = Field(default=256, gt=0)
    belief_dim: int = Field(default=256, gt=0)
    uncertainty_dim: int = Field(default=64, gt=0)
    temporal_dim: int = Field(default=64, gt=0)
    history_dim: int = Field(default=128, gt=0)
    relation_dim: int = Field(default=64, gt=0)
    mechanism_dim: int = Field(default=128, gt=0)
    heads: int = Field(default=8, gt=0)
    ffn_hidden: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.10, ge=0, lt=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    finetune_learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    seed_set: tuple[int, ...] = (2026201, 2026202, 2026203)
    ecmer: EcmerConfig = EcmerConfig()
    association: AssociationConfig = AssociationConfig()
    buo: BuoConfig = BuoConfig()
    analytic_buo: AnalyticBuoConfig = AnalyticBuoConfig()
    rbp: RbpConfig = RbpConfig()
    tbd: TbdConfig = TbdConfig()
    pmbl: PmblConfig = PmblConfig()

    @model_validator(mode="after")
    def _consistent(self) -> CoreConfig:
        if self.evidence_dim != self.belief_dim:
            raise ValueError("ch33 pair features (e-z, e*z) require evidence_dim == belief_dim")
        if self.belief_dim % self.heads != 0:
            raise ValueError("belief_dim must be divisible by heads")
        if self.uncertainty_dim % 4 != 0:
            raise ValueError("uncertainty_dim must split into four equal partitions (UA, UE, UC, UO)")
        return self

    @property
    def head_dim(self) -> int:
        return self.belief_dim // self.heads

    @property
    def partition_dim(self) -> int:
        return self.uncertainty_dim // 4

    @property
    def association_pair_dim(self) -> int:
        """4*De + pose(12) + time(4) + Dr + reliability(1) + U summary(4) = 1109 at ch33 defaults.

        ch33 states 853, but its own component list sums to 1109; every listed component is kept
        (spec inconsistency recorded in the report)."""
        a = self.association
        return 4 * self.evidence_dim + a.relative_pose_dim + a.time_feature_dim + self.relation_dim + 1 + 4

    @property
    def buo_input_dim(self) -> int:
        """1156 at ch33 defaults: 4*Dz + Du + quality(4) + Dt."""
        return 4 * self.belief_dim + self.uncertainty_dim + self.buo.quality_dim + self.temporal_dim


def tiny_config(**overrides: Any) -> CoreConfig:
    """Small CPU configuration for tests and smoke experiments."""
    base: dict[str, Any] = {
        "evidence_dim": 16,
        "belief_dim": 16,
        "uncertainty_dim": 8,
        "temporal_dim": 8,
        "history_dim": 8,
        "relation_dim": 8,
        "mechanism_dim": 8,
        "heads": 2,
        "ffn_hidden": 32,
        "dropout": 0.0,
        "ecmer": EcmerConfig(
            image_size=32,
            resnet_base_width=4,
            voxel_grid=8,
            voxel_channels=(4, 8, 8),
            voxel_groups=2,
            scalar_hidden=16,
            context_hidden=(8, 16),
            fusion_layers=1,
            quality_hidden=(16, 8),
            projection_hidden=32,
            max_sensors=8,
            max_frames=4,
        ),
        "association": AssociationConfig(hidden=(32, 16, 8), no_match_hidden=8),
        "buo": BuoConfig(
            trunk_hidden=32, trust_hidden=8, innovation_hidden=32, gate_hidden=8, uncertainty_hidden=(16, 8)
        ),
        "rbp": RbpConfig(relation_hidden=16, gate_hidden=16, relation_types=4),
        "tbd": TbdConfig(context_dim=8, stem_hidden=32, drift_hidden=8, mlp_hidden=32, unroll_length=8),
    }
    base.update(overrides)
    return CoreConfig(**base)


def load_core_config(path: str | Path) -> CoreConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CoreConfig.model_validate(data.get("model2_core", data))
