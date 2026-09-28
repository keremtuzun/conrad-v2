"""Training observation manifest schema for Universal OS-FM V1.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from conrad.foundation.universal_v11.registry import EncoderFamily, ModalityState


@dataclass(frozen=True)
class UniversalObservationManifest:
    asset_id: str
    scene_id: str
    observation_id: str
    session_id: str
    platform: str
    modality_family: EncoderFamily
    concrete_modality: str
    timestamp_s: float
    timing_uncertainty_s: float | None = None
    location: tuple[float, ...] | None = None
    pose: tuple[float, ...] | None = None
    extrinsics: tuple[float, ...] | None = None
    data_uri: str | None = None
    units: str | None = None
    sampling_frequency_hz: float | None = None
    resolution: str | None = None
    acquisition_settings: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    calibration: str | None = None
    quality: float | None = None
    modality_state: ModalityState = ModalityState.AVAILABLE
    provenance: str | None = None
    source_license: str | None = None
    labels: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    exact_measurements: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    synthetic: bool = False
    simulator_metadata: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    corruption_metadata: Mapping[str, float | int | bool | str] = field(default_factory=dict)
    pairing_ids: tuple[str, ...] = ()
    correspondence_ids: tuple[str, ...] = ()

    def validate_minimum(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if not self.concrete_modality:
            raise ValueError("concrete_modality is required")
        if self.data_uri is None and not self.exact_measurements:
            raise ValueError("manifest requires either data_uri or exact_measurements")
