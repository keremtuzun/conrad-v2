"""Universal OS-FM V1.1 modality registry.

The registry is intentionally broader than the currently trainable encoder set.
Inactive entries are preserved as explicit engineering promises instead of being
silently dropped from the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EncoderFamily(str, Enum):
    VISUAL_IMAGE = "visual_image"
    SPECTRAL_PHOTONIC = "spectral_photonic"
    ACTIVE_ACOUSTIC = "active_acoustic"
    PASSIVE_ACOUSTIC_VIBRATION = "passive_acoustic_vibration"
    ULTRASONIC_NDT = "ultrasonic_ndt"
    ELECTROMAGNETIC_MAGNETIC = "electromagnetic_magnetic"
    GEOMETRY_SPATIAL = "geometry_spatial"
    MECHANICAL_STRUCTURAL_TIME_SERIES = "mechanical_structural_time_series"
    PHYSICAL_OCEANOGRAPHIC_TIME_SERIES = "physical_oceanographic_time_series"
    CHEMICAL_ELECTROCHEMICAL_SPECTRAL = "chemical_electrochemical_spectral"
    BIOLOGICAL_MOLECULAR = "biological_molecular"
    MICROSCOPIC_PARTICLE = "microscopic_particle"
    GEOPHYSICAL_SEISMIC = "geophysical_seismic"
    RADIOLOGICAL = "radiological"
    ENGINEERING_DOCUMENT_CONTEXT = "engineering_document_context"


class ModalityState(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    DEGRADED = "degraded"
    STALE = "stale"
    INVALID = "invalid"
    UNCALIBRATED = "uncalibrated"
    SATURATED = "saturated"


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    family: EncoderFamily
    tokenizer: str
    learned_active: bool
    exact_evidence: bool = False
    sample_or_lab_result: bool = False
    description: str = ""


MODALITY_REGISTRY: tuple[ModalitySpec, ...] = (
    ModalitySpec("rgb_camera", EncoderFamily.VISUAL_IMAGE, "image", True, description="V1 RGB image stream"),
    ModalitySpec("low_light_camera", EncoderFamily.VISUAL_IMAGE, "image", False),
    ModalitySpec("polarization_camera", EncoderFamily.VISUAL_IMAGE, "image", False),
    ModalitySpec("thermal_ir_camera", EncoderFamily.VISUAL_IMAGE, "image", False),
    ModalitySpec("hyperspectral_camera", EncoderFamily.SPECTRAL_PHOTONIC, "spectra", False),
    ModalitySpec("multispectral_camera", EncoderFamily.SPECTRAL_PHOTONIC, "spectra", False),
    ModalitySpec("raman_spectrometer", EncoderFamily.SPECTRAL_PHOTONIC, "spectra", False, exact_evidence=True),
    ModalitySpec("fluorometer", EncoderFamily.SPECTRAL_PHOTONIC, "spectra", False, exact_evidence=True),
    ModalitySpec("imaging_sonar", EncoderFamily.ACTIVE_ACOUSTIC, "sonar", True, description="V1 SonarViT-compatible path"),
    ModalitySpec("sidescan_sonar", EncoderFamily.ACTIVE_ACOUSTIC, "sonar", False),
    ModalitySpec("multibeam_sonar", EncoderFamily.ACTIVE_ACOUSTIC, "sonar", False),
    ModalitySpec("synthetic_aperture_sonar", EncoderFamily.ACTIVE_ACOUSTIC, "sonar", False),
    ModalitySpec("doppler_velocity_log", EncoderFamily.ACTIVE_ACOUSTIC, "time_series", False, exact_evidence=True),
    ModalitySpec("hydrophone_array", EncoderFamily.PASSIVE_ACOUSTIC_VIBRATION, "waveform", False),
    ModalitySpec("ambient_noise_spectrum", EncoderFamily.PASSIVE_ACOUSTIC_VIBRATION, "spectrogram", False),
    ModalitySpec("accelerometer_vibration", EncoderFamily.PASSIVE_ACOUSTIC_VIBRATION, "time_series", False, exact_evidence=True),
    ModalitySpec("paut_scan", EncoderFamily.ULTRASONIC_NDT, "ndt_scan", False),
    ModalitySpec("ut_a_scan", EncoderFamily.ULTRASONIC_NDT, "ndt_scan", False),
    ModalitySpec("tofd_scan", EncoderFamily.ULTRASONIC_NDT, "ndt_scan", False),
    ModalitySpec("tfm_scan", EncoderFamily.ULTRASONIC_NDT, "ndt_scan", False),
    ModalitySpec("magnetometer", EncoderFamily.ELECTROMAGNETIC_MAGNETIC, "em_map", False, exact_evidence=True),
    ModalitySpec("em_ndt_eddy_current", EncoderFamily.ELECTROMAGNETIC_MAGNETIC, "em_map", False),
    ModalitySpec("cathodic_protection_potential", EncoderFamily.ELECTROMAGNETIC_MAGNETIC, "time_series", False, exact_evidence=True),
    ModalitySpec("structured_light_depth", EncoderFamily.GEOMETRY_SPATIAL, "point_cloud", True),
    ModalitySpec("lidar_point_cloud", EncoderFamily.GEOMETRY_SPATIAL, "point_cloud", False),
    ModalitySpec("photogrammetry_mesh", EncoderFamily.GEOMETRY_SPATIAL, "point_cloud", False),
    ModalitySpec("imu_pose", EncoderFamily.GEOMETRY_SPATIAL, "time_series", False, exact_evidence=True),
    ModalitySpec("strain_gauge", EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES, "time_series", False, exact_evidence=True),
    ModalitySpec("pressure_load_cell", EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES, "time_series", False, exact_evidence=True),
    ModalitySpec("fatigue_cycle_counter", EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES, "time_series", False, exact_evidence=True),
    ModalitySpec("ctd_profile", EncoderFamily.PHYSICAL_OCEANOGRAPHIC_TIME_SERIES, "time_series", False, exact_evidence=True),
    ModalitySpec("current_meter_adcp", EncoderFamily.PHYSICAL_OCEANOGRAPHIC_TIME_SERIES, "time_series", False, exact_evidence=True),
    ModalitySpec("turbidity_sensor", EncoderFamily.PHYSICAL_OCEANOGRAPHIC_TIME_SERIES, "time_series", False, exact_evidence=True),
    ModalitySpec("dissolved_oxygen", EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, "time_series", False, exact_evidence=True),
    ModalitySpec("ph_probe", EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, "time_series", False, exact_evidence=True),
    ModalitySpec("orp_probe", EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, "time_series", False, exact_evidence=True),
    ModalitySpec("mass_spectrometry", EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, "spectra", False, exact_evidence=True, sample_or_lab_result=True),
    ModalitySpec("edna_sequence", EncoderFamily.BIOLOGICAL_MOLECULAR, "molecular_sequence", False, sample_or_lab_result=True),
    ModalitySpec("metabarcoding_panel", EncoderFamily.BIOLOGICAL_MOLECULAR, "molecular_sequence", False, sample_or_lab_result=True),
    ModalitySpec("microbiology_assay", EncoderFamily.BIOLOGICAL_MOLECULAR, "molecular_sequence", False, sample_or_lab_result=True),
    ModalitySpec("microscopy_image", EncoderFamily.MICROSCOPIC_PARTICLE, "microscopy", False, sample_or_lab_result=True),
    ModalitySpec("particle_size_distribution", EncoderFamily.MICROSCOPIC_PARTICLE, "particle", False, exact_evidence=True, sample_or_lab_result=True),
    ModalitySpec("microplastic_classifier", EncoderFamily.MICROSCOPIC_PARTICLE, "particle", False, sample_or_lab_result=True),
    ModalitySpec("sub_bottom_profiler", EncoderFamily.GEOPHYSICAL_SEISMIC, "seismic", False),
    ModalitySpec("seismic_reflection", EncoderFamily.GEOPHYSICAL_SEISMIC, "seismic", False),
    ModalitySpec("sediment_core_log", EncoderFamily.GEOPHYSICAL_SEISMIC, "time_series", False, exact_evidence=True),
    ModalitySpec("gamma_spectrometer", EncoderFamily.RADIOLOGICAL, "radiological", False, exact_evidence=True),
    ModalitySpec("dosimeter", EncoderFamily.RADIOLOGICAL, "time_series", False, exact_evidence=True),
    ModalitySpec("rov_operator_note", EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT, "document_context", True),
    ModalitySpec("inspection_work_order", EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT, "document_context", True),
    ModalitySpec("cad_asset_context", EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT, "document_context", True),
    ModalitySpec("maintenance_log", EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT, "document_context", True),
)


def registry_by_name() -> dict[str, ModalitySpec]:
    return {spec.name: spec for spec in MODALITY_REGISTRY}


def inactive_modalities() -> tuple[str, ...]:
    return tuple(spec.name for spec in MODALITY_REGISTRY if not spec.learned_active)


def validate_registry() -> None:
    names = [spec.name for spec in MODALITY_REGISTRY]
    if len(names) != len(set(names)):
        raise ValueError("duplicate V1.1 modality registry names")
    missing = set(EncoderFamily) - {spec.family for spec in MODALITY_REGISTRY}
    if missing:
        raise ValueError(f"missing encoder families: {sorted(item.value for item in missing)}")
