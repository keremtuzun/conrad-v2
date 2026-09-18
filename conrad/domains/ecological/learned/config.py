"""Learned CEFD configuration. Defaults are the ch33 Model2E freeze; tests pass tiny dimensions.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from pydantic import Field

from conrad.schemas.base import ConradModel


class LearnedCEFDConfig(ConradModel):
    entity_feature_dim: int = Field(default=16, ge=1)
    entity_hidden: int = Field(default=128, ge=1, description="entity_features -> 128 -> 256")
    d_model: int = Field(default=256, ge=1)
    entity_layers: int = Field(default=3, ge=1)
    heads: int = Field(default=8, ge=1)
    relation_types: int = Field(default=4, ge=1, description="typed entity-entity edges (0 = no edge)")
    relation_dim: int = Field(default=64, ge=1)
    ffn_dim: int = Field(default=512, ge=1)
    field_in_channels: int = Field(default=6, ge=1, description="temperature, turbidity, current x3, light")
    field_out_channels: int = Field(default=12, ge=1, description="mean + log-variance per input channel")
    unet_channels: tuple[int, int, int] = (64, 128, 256)
    gn_groups: int = Field(default=8, ge=1)
    uncertainty_in: int = Field(default=4, ge=1, description="U_A, U_E, U_C, U_O of the entity")
    uncertainty_dim: int = Field(default=64, ge=1)
    gate_hidden: int = Field(default=256, ge=1, description="576 -> 256 -> 1 at defaults")
    coupling_rounds: int = Field(default=1, ge=1, description="f->e + e->f pairs per time step")
    distance_scale_cells: float = Field(default=1.0, gt=0)
    entity_heads_out: int = Field(default=4, ge=1, description="cover mu/logvar, condition mu/logvar")
    dropout: float = Field(default=0.10, ge=0, lt=1)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    lambda_entity: float = 1.0
    lambda_field: float = 1.0
    lambda_gradient: float = 0.1
    lambda_coupling: float = 0.5


def tiny_learned_config(**overrides: object) -> LearnedCEFDConfig:
    base: dict[str, object] = {
        "entity_feature_dim": 6,
        "entity_hidden": 8,
        "d_model": 16,
        "entity_layers": 1,
        "heads": 2,
        "relation_types": 2,
        "relation_dim": 4,
        "ffn_dim": 16,
        "field_in_channels": 2,
        "field_out_channels": 4,
        "unet_channels": (4, 8, 8),
        "gn_groups": 2,
        "uncertainty_dim": 4,
        "gate_hidden": 8,
        "dropout": 0.0,
    }
    base.update(overrides)
    return LearnedCEFDConfig.model_validate(base)
