"""Joint OS-FM v0 assembly: U1 encoders -> M1 fusion -> T1 temporal."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from conrad.foundation.encoders.geometry import GeometryEncoderConfig, GeometryGroupedEncoder
from conrad.foundation.encoders.range import RangeEncoderConfig, RangeViTP8Encoder
from conrad.foundation.encoders.rgb import RGBEncoderConfig, RGBViTS14Encoder
from conrad.foundation.encoders.sonar import SonarEncoderConfig, SonarViTS14Encoder
from conrad.foundation.fusion.scene_fusion import ModalityTokenSet, SceneFusionOutput, SceneFusionTransformer
from conrad.foundation.temporal.memory import TemporalMemoryOutput, TemporalMemoryTransformer


@dataclass(frozen=True)
class JointOSFMInputs:
    rgb: torch.Tensor
    sonar: torch.Tensor
    range_raster: torch.Tensor
    range_validity: torch.Tensor
    geometry_points: torch.Tensor
    geometry_validity: torch.Tensor
    natural_missing: dict[str, torch.Tensor]
    artificial_dropout: dict[str, torch.Tensor]
    padding: dict[str, torch.Tensor]
    timestamps_s: torch.Tensor
    temporal_valid_mask: torch.Tensor
    boundary_reset_mask: torch.Tensor


@dataclass(frozen=True)
class JointOSFMOutput:
    modality_sets: tuple[ModalityTokenSet, ...]
    fusion: SceneFusionOutput
    temporal: TemporalMemoryOutput


class JointOSFMModel(nn.Module):
    def __init__(
        self,
        *,
        rgb: RGBViTS14Encoder | None = None,
        sonar: SonarViTS14Encoder | None = None,
        range_encoder: RangeViTP8Encoder | None = None,
        geometry: GeometryGroupedEncoder | None = None,
        fusion: SceneFusionTransformer | None = None,
        temporal: TemporalMemoryTransformer | None = None,
    ) -> None:
        super().__init__()
        self.rgb = rgb or RGBViTS14Encoder(RGBEncoderConfig())
        self.sonar = sonar or SonarViTS14Encoder(SonarEncoderConfig())
        self.range = range_encoder or RangeViTP8Encoder(RangeEncoderConfig())
        self.geometry = geometry or GeometryGroupedEncoder(GeometryEncoderConfig(num_groups=4, group_size=8))
        self.fusion = fusion or SceneFusionTransformer()
        self.temporal = temporal or TemporalMemoryTransformer()

    def _sets_for_windows(self, inputs: JointOSFMInputs) -> tuple[ModalityTokenSet, ...]:
        batch, steps = inputs.temporal_valid_mask.shape
        flat = batch * steps
        rgb_out = self.rgb(inputs.rgb.reshape(flat, *inputs.rgb.shape[2:]))
        sonar_out = self.sonar(inputs.sonar.reshape(flat, *inputs.sonar.shape[2:]))
        range_out = self.range(
            inputs.range_raster.reshape(flat, *inputs.range_raster.shape[2:]),
            inputs.range_validity.reshape(flat, *inputs.range_validity.shape[2:]),
        )
        geo_out = self.geometry(
            inputs.geometry_points.reshape(flat, *inputs.geometry_points.shape[2:]),
            inputs.geometry_validity.reshape(flat, inputs.geometry_validity.shape[-1]),
        )

        def flat_mask(name: str, source: dict[str, torch.Tensor]) -> torch.Tensor:
            return source[name].reshape(flat).to(device=rgb_out.patch_tokens.device, dtype=torch.bool)

        return (
            ModalityTokenSet("rgb", rgb_out.patch_tokens, rgb_out.visible_mask, flat_mask("rgb", inputs.natural_missing), flat_mask("rgb", inputs.artificial_dropout), flat_mask("rgb", inputs.padding)),
            ModalityTokenSet("sonar", sonar_out.patch_tokens, sonar_out.visible_mask, flat_mask("sonar", inputs.natural_missing), flat_mask("sonar", inputs.artificial_dropout), flat_mask("sonar", inputs.padding)),
            ModalityTokenSet("range", range_out.patch_tokens, range_out.visible_mask, flat_mask("range", inputs.natural_missing), flat_mask("range", inputs.artificial_dropout), flat_mask("range", inputs.padding)),
            ModalityTokenSet("geometry", geo_out.group_tokens, geo_out.visible_mask, flat_mask("geometry", inputs.natural_missing), flat_mask("geometry", inputs.artificial_dropout), flat_mask("geometry", inputs.padding)),
        )

    def forward(self, inputs: JointOSFMInputs) -> JointOSFMOutput:
        batch, steps = inputs.temporal_valid_mask.shape
        sets = self._sets_for_windows(inputs)
        fused = self.fusion(sets)
        global_seq = fused.global_repr.reshape(batch, steps, -1)
        temporal = self.temporal(
            global_seq,
            inputs.timestamps_s,
            inputs.temporal_valid_mask,
            inputs.boundary_reset_mask,
        )
        return JointOSFMOutput(sets, fused, temporal)
